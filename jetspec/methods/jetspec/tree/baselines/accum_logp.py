"""V0 baseline: cumulative-logprob tree from independent top-k per depth.

Ported from causal_parallel_drafting.model.tree.build_tree_from_topk and
_build_tree_from_topk_cpu (Approach A — independent marginals). Each
depth d shares the same topk_tokens[d] across all parents at that depth;
the verifier treats every path independently.

This is the recovery point for every uncertainty-aware variant in §3 of
the handoff doc — V1-V12 should reduce to accum_logp when their knobs
are at identity (λ=0, K=full, threshold=∞).
"""
from __future__ import annotations

import heapq

import numpy as np
import torch

from jetspec.tree._core.ancestor import _build_ancestor_matrix_np, _build_packed_ancestor_matrix_np
from jetspec.tree._core.accept import _build_child_maps_cpu
from jetspec.tree._core.base import DraftTree, TreeAlgorithm
from jetspec.tree._core.registry import register_tree_algo


def _topk_pair_to_lists(
    topk_tok_t: torch.Tensor,
    topk_lp_t: torch.Tensor,
) -> tuple[list[list[int]], list[list[float]]]:
    """Materialize top-k tokens/logprobs in one host transfer when sourced on GPU."""
    if topk_tok_t.device.type == "cpu" and topk_lp_t.device.type == "cpu":
        return topk_tok_t.tolist(), topk_lp_t.tolist()

    topk_pair_cpu = torch.stack(
        (topk_tok_t.to(dtype=torch.float64), topk_lp_t.to(dtype=torch.float64)),
        dim=-1,
    ).cpu().tolist()
    topk_tokens_cpu = [[int(pair[0]) for pair in row] for row in topk_pair_cpu]
    topk_logprobs_cpu = [[pair[1] for pair in row] for row in topk_pair_cpu]
    return topk_tokens_cpu, topk_logprobs_cpu


@register_tree_algo("accum_logp")
class AccumLogP(TreeAlgorithm):
    """V0 — best-first heap expansion on cumulative log-prob.

    Children at depth d are always the same topk(d) regardless of parent,
    so the cumulative-logprob simplification makes the per-depth top-k tensor
    sufficient input. Tree size is bounded by budget; expansion proceeds
    in descending cumulative-logprob order until budget is reached.
    """

    def build(
        self,
        root_token: int,
        draft_logits: torch.Tensor,  # (1, D, vocab_size)
        block_size: int,
        tree_width: int,
        budget: int,
        device: torch.device,
        **kwargs,
    ) -> DraftTree:
        D_expected = block_size - 1
        if draft_logits.dim() != 3 or draft_logits.shape[0] != 1:
            raise ValueError(
                f"draft_logits must be (1, D, V); got {tuple(draft_logits.shape)}"
            )
        if draft_logits.shape[1] != D_expected:
            raise ValueError(
                f"draft_logits depth {draft_logits.shape[1]} != block_size-1 ({D_expected})"
            )

        # Per-depth top-k extraction (same as causal_parallel_drafting reference).
        log_probs = torch.log_softmax(draft_logits.squeeze(0), dim=-1)  # (D, V)
        topk_lp_t, topk_tok_t = torch.topk(log_probs, tree_width, dim=-1)  # (D, k)

        topk_tokens_cpu, topk_logprobs_cpu = _topk_pair_to_lists(topk_tok_t, topk_lp_t)
        return _build_from_topk(
            root_token=root_token,
            topk_tokens_cpu=topk_tokens_cpu,
            topk_logprobs_cpu=topk_logprobs_cpu,
            budget=budget,
            device=device,
        )

    def caps_from_topk(self, topk_logprobs_cpu, tree_width, **kwargs) -> list[int]:
        """Per-depth fanout cap for the engine `build_from_topk` path: full fanout
        (K at every depth). With cap=[K]*D, build_with_per_depth_cap matches this
        class's own `_build_from_topk` heap (children_to_add = min(K, budget-n))."""
        K = len(topk_logprobs_cpu[0]) if topk_logprobs_cpu else max(tree_width, 1)
        return [K] * len(topk_logprobs_cpu)


def _build_from_topk(
    root_token: int,
    topk_tokens_cpu: list[list[int]],
    topk_logprobs_cpu: list[list[float]],
    budget: int,
    device: torch.device,
) -> DraftTree:
    """Heap-based BFS — separate from the ABC class so other algorithms
    (e.g. V8 EntropyAdjustedScore) can reuse the construction loop with
    a different scoring function."""
    D = len(topk_tokens_cpu)
    k = len(topk_tokens_cpu[0]) if D > 0 else 0

    tokens_list: list[int] = [root_token]
    parents_list: list[int] = [-1]
    depths_list: list[int] = [0]
    cum_lp_list: list[float] = [0.0]
    num_nodes = 1

    # Max-heap keyed by cumulative log-prob (negate for min-heap).
    # Entry: (neg_cum_logprob, insertion_order, node_index)
    counter = 0
    heap: list[tuple[float, int, int]] = [(0.0, counter, 0)]

    while heap and num_nodes < budget:
        neg_cum_lp, _, node_idx = heapq.heappop(heap)
        d = depths_list[node_idx]
        if d >= D:
            continue
        children_to_add = min(k, budget - num_nodes)
        for j in range(children_to_add):
            child_token = topk_tokens_cpu[d][j]
            child_cum_lp = -neg_cum_lp + topk_logprobs_cpu[d][j]
            tokens_list.append(child_token)
            parents_list.append(node_idx)
            depths_list.append(d + 1)
            cum_lp_list.append(child_cum_lp)
            counter += 1
            heapq.heappush(heap, (-child_cum_lp, counter, num_nodes))
            num_nodes += 1

    ancestor_np = _build_ancestor_matrix_np(parents_list, num_nodes)
    ancestor_packed_np = _build_packed_ancestor_matrix_np(parents_list, num_nodes)
    child_maps = _build_child_maps_cpu(tokens_list, parents_list, num_nodes)

    return DraftTree(
        token_ids=torch.tensor(tokens_list, dtype=torch.long, device=device),
        parent_indices=torch.tensor(parents_list, dtype=torch.long, device=device),
        depth=torch.tensor(depths_list, dtype=torch.long, device=device),
        num_nodes=num_nodes,
        cum_logprob=torch.tensor(cum_lp_list, dtype=torch.float32, device=device),
        child_maps=child_maps,
        ancestor=torch.from_numpy(ancestor_np).to(device, non_blocking=True),
        ancestor_packed=torch.from_numpy(ancestor_packed_np).to(device, non_blocking=True),
    )
