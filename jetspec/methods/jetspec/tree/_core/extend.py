"""Ceiling-raise helpers for gated draft-tree extension."""
from __future__ import annotations

import torch

from .accept import _build_child_maps_cpu
from .ancestor import _build_ancestor_matrix_np, _build_packed_ancestor_matrix_np
from .base import DraftTree
from .fanout_cap_builder import build_with_per_depth_cap


def should_extend(topk_lp, best_path_ranks, *, gap_threshold: float) -> bool:
    """Return True when the heap-best path is top-2-gap confident.

    ``topk_lp`` is a ``(D, K)`` row set of per-depth logprobs. For each depth,
    ``best_path_ranks`` gives the rank used by the heap-best leaf path. Rank-0
    depths use the usual top-1 minus top-2 gap; non-rank-0 depths compare the
    chosen rank against rank 0, which suppresses the chain gate for paths that
    were not locally top-1.
    """
    lp = torch.as_tensor(topk_lp)
    if lp.dim() != 2:
        raise ValueError(f"topk_lp must be (D, K); got {tuple(lp.shape)}")

    if torch.is_tensor(best_path_ranks):
        ranks = best_path_ranks.to(device=lp.device, dtype=torch.long)
    else:
        ranks = torch.as_tensor(list(best_path_ranks), dtype=torch.long, device=lp.device)
    if ranks.numel() == 0:
        return False
    if ranks.numel() != lp.shape[0]:
        raise ValueError(
            f"best_path_ranks length {ranks.numel()} must match topk depth {lp.shape[0]}"
        )
    if lp.shape[1] < 2:
        return False
    if bool(((ranks < 0) | (ranks >= lp.shape[1])).any().item()):
        raise ValueError("best_path_ranks contains a rank outside topk_lp width")

    row = torch.arange(lp.shape[0], device=lp.device)
    chosen_lp = lp[row, ranks]
    best_other_lp = torch.where(ranks == 0, lp[:, 1], lp[:, 0])
    mean_gap = (chosen_lp - best_other_lp).to(dtype=torch.float32).mean()
    return bool(mean_gap.item() > float(gap_threshold))


def splice_extension(
    tree: DraftTree,
    leaf_index,
    ext_logits: torch.Tensor,
    *,
    ext_budget: int,
    tree_width: int,
    mode: str = "chain",
) -> DraftTree:
    """Return a new DraftTree with an extension appended below ``leaf_index``.

    ``mode="chain"`` appends the conditioned rank-1 token for each extension
    depth. ``mode="subtree"`` builds a small top2gap-shaped subtree from the
    conditioned logits, drops its synthetic root, and splices those nodes below
    the selected leaf.
    """
    ext_budget = int(ext_budget)
    if ext_budget < 0:
        raise ValueError("ext_budget must be non-negative")
    if ext_budget == 0:
        return tree

    leaf = _validate_leaf(tree, leaf_index)
    logits = _validate_ext_logits(ext_logits)
    if int(tree_width) < 1:
        raise ValueError("tree_width must be >= 1")

    if mode == "chain":
        return _splice_chain(tree, leaf, logits, ext_budget)
    if mode == "subtree":
        return _splice_subtree(tree, leaf, logits, ext_budget, int(tree_width))
    raise ValueError(f"unknown extension mode {mode!r}; expected 'chain' or 'subtree'")


def _validate_leaf(tree: DraftTree, leaf_index) -> int:
    leaf = int(leaf_index)
    if leaf < 0 or leaf >= int(tree.num_nodes):
        raise IndexError(f"leaf_index {leaf} outside tree with {tree.num_nodes} nodes")
    if bool((tree.parent_indices[: tree.num_nodes] == leaf).any().item()):
        raise ValueError(f"leaf_index {leaf} is not a leaf")
    return leaf


def _validate_ext_logits(ext_logits: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(ext_logits):
        raise TypeError("ext_logits must be a torch.Tensor")
    if ext_logits.dim() != 3 or ext_logits.shape[0] != 1:
        raise ValueError(f"ext_logits must be (1, D, V); got {tuple(ext_logits.shape)}")
    if ext_logits.shape[2] < 1:
        raise ValueError("ext_logits vocabulary dimension must be non-empty")
    return ext_logits


def _splice_chain(
    tree: DraftTree,
    leaf: int,
    ext_logits: torch.Tensor,
    ext_budget: int,
) -> DraftTree:
    if ext_logits.shape[1] < ext_budget:
        raise ValueError(
            f"chain extension needs {ext_budget} logit rows; got {ext_logits.shape[1]}"
        )

    log_probs = torch.log_softmax(ext_logits.squeeze(0)[:ext_budget], dim=-1)
    best_lp, best_tok = torch.max(log_probs, dim=-1)
    base_depth = int(tree.depth[leaf].item())
    old_n = int(tree.num_nodes)

    tokens = [int(tok.item()) for tok in best_tok]
    parents = [leaf] + [old_n + i - 1 for i in range(1, ext_budget)]
    depths = [base_depth + i + 1 for i in range(ext_budget)]

    cum_logprob = None
    if tree.cum_logprob is not None:
        base_lp = float(tree.cum_logprob[leaf].item())
        running = base_lp
        cum_logprob = []
        for lp in best_lp:
            running += float(lp.item())
            cum_logprob.append(running)

    return _append_nodes(tree, tokens, parents, depths, cum_logprob)


def _splice_subtree(
    tree: DraftTree,
    leaf: int,
    ext_logits: torch.Tensor,
    ext_budget: int,
    tree_width: int,
) -> DraftTree:
    log_probs = torch.log_softmax(ext_logits.squeeze(0), dim=-1)
    topk_width = min(tree_width, log_probs.shape[-1])
    topk_lp_t, topk_tok_t = torch.topk(log_probs, topk_width, dim=-1)
    topk_tokens_cpu = topk_tok_t.tolist()
    topk_logprobs_cpu = topk_lp_t.tolist()

    from jetspec.tree.tree_to_chain.fanout_cap.top2gap import Top2GapFanout

    caps = Top2GapFanout().caps_from_topk(topk_logprobs_cpu, topk_width)
    subtree = build_with_per_depth_cap(
        root_token=int(tree.token_ids[leaf].item()),
        topk_tokens_cpu=topk_tokens_cpu,
        topk_logprobs_cpu=topk_logprobs_cpu,
        b_per_depth=caps,
        budget=ext_budget + 1,
        device=tree.token_ids.device,
    )
    spliced_count = int(subtree.num_nodes) - 1
    if spliced_count != ext_budget:
        raise ValueError(
            f"subtree extension produced {spliced_count} nodes, expected {ext_budget}"
        )

    old_n = int(tree.num_nodes)
    base_depth = int(tree.depth[leaf].item())
    tokens: list[int] = []
    parents: list[int] = []
    depths: list[int] = []
    cum_logprob: list[float] | None = [] if tree.cum_logprob is not None else None
    base_lp = float(tree.cum_logprob[leaf].item()) if tree.cum_logprob is not None else 0.0

    for sub_idx in range(1, int(subtree.num_nodes)):
        sub_parent = int(subtree.parent_indices[sub_idx].item())
        tokens.append(int(subtree.token_ids[sub_idx].item()))
        parents.append(leaf if sub_parent == 0 else old_n + sub_parent - 1)
        depths.append(base_depth + int(subtree.depth[sub_idx].item()))
        if cum_logprob is not None:
            cum_logprob.append(base_lp + float(subtree.cum_logprob[sub_idx].item()))

    return _append_nodes(tree, tokens, parents, depths, cum_logprob)


def _append_nodes(
    tree: DraftTree,
    tokens: list[int],
    parents: list[int],
    depths: list[int],
    cum_logprob: list[float] | None,
) -> DraftTree:
    new_num_nodes = int(tree.num_nodes) + len(tokens)
    token_ids = torch.cat(
        [tree.token_ids[: tree.num_nodes], tree.token_ids.new_tensor(tokens)]
    )
    parent_indices = torch.cat(
        [tree.parent_indices[: tree.num_nodes], tree.parent_indices.new_tensor(parents)]
    )
    depth = torch.cat([tree.depth[: tree.num_nodes], tree.depth.new_tensor(depths)])

    new_cum_logprob = None
    if tree.cum_logprob is not None:
        if cum_logprob is None:
            raise ValueError("cum_logprob extension values are required")
        new_cum_logprob = torch.cat(
            [
                tree.cum_logprob[: tree.num_nodes],
                tree.cum_logprob.new_tensor(cum_logprob),
            ]
        )

    token_list = token_ids.tolist()
    parent_list = parent_indices.tolist()
    ancestor_np = _build_ancestor_matrix_np(parent_list, new_num_nodes)
    ancestor_packed_np = _build_packed_ancestor_matrix_np(parent_list, new_num_nodes)

    return DraftTree(
        token_ids=token_ids,
        parent_indices=parent_indices,
        depth=depth,
        num_nodes=new_num_nodes,
        cum_logprob=new_cum_logprob,
        child_maps=_build_child_maps_cpu(token_list, parent_list, new_num_nodes),
        ancestor=torch.from_numpy(ancestor_np).to(tree.token_ids.device, non_blocking=True),
        ancestor_packed=torch.from_numpy(ancestor_packed_np).to(
            tree.token_ids.device,
            non_blocking=True,
        ),
    )
