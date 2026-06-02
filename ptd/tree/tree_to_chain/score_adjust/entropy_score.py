"""entropy_score — fanout open (b=K), heap key penalized by path entropy.

Leaves fanout fully open (b = K everywhere) and instead changes the heap key
that orders which nodes get expanded under a tight budget:

    s'(π) = Σ_d ( log q_d^{ρ_d} - λ · H_{d|π} )

λ ≥ 0. λ = 0 recovers crossproduct. Large λ deprioritizes branches that
descend through high-entropy positions, so they sit at the heap bottom and
don't expand under tight budget — a chain-like spine emerges automatically
without an explicit cap. Negative λ does the opposite (descend more
aggressively into high-entropy positions); both signs are sweepable.

Caveat: if the drafter hands us top-K-sparse logits, H_d here is
top-K-renormalized entropy, not full-vocab — closer to `entropy_topk`'s signal
than full marginal entropy.

Lineage: sweep id V8 (entropy_adjusted_score).
"""
from __future__ import annotations

import heapq
import math

import torch

from ptd.tree._core.accept import _build_child_maps_cpu
from ptd.tree._core.ancestor import _build_ancestor_matrix_np, _build_packed_ancestor_matrix_np
from ptd.tree._core.base import DraftTree, TreeAlgorithm
from ptd.tree._core.registry import register_tree_algo


@register_tree_algo("entropy_score")
class EntropyAdjustedScore(TreeAlgorithm):
    """Per-node heap key with -λ * entropy adjustment. Fanout cap unchanged."""

    def __init__(self, lambda_: float = 0.5):
        self.lambda_ = float(lambda_)

    def build(
        self,
        root_token: int,
        draft_logits: torch.Tensor,  # (1, D, V)
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

        log_probs = torch.log_softmax(draft_logits.squeeze(0), dim=-1)  # (D, V)
        topk_lp_t, topk_tok_t = torch.topk(log_probs, tree_width, dim=-1)

        # Per-depth marginal entropy. Over the FULL vocab when the drafter
        # provides dense logits; over the top-K-renormalized distribution when
        # it provides sparse (-inf-padded) logits — in that case mask -inf
        # entries before the sum to avoid NaN from 0 * log(0).
        probs = log_probs.exp()
        finite_mask = torch.isfinite(log_probs)
        contrib = torch.where(finite_mask, probs * log_probs, torch.zeros_like(probs))
        H_per_depth = -contrib.sum(dim=-1)  # (D,)

        topk_tokens_cpu = topk_tok_t.tolist()
        topk_logprobs_cpu = topk_lp_t.tolist()
        H_cpu = H_per_depth.tolist()
        return _build_with_entropy_score(
            root_token=int(root_token),
            topk_tokens_cpu=topk_tokens_cpu,
            topk_logprobs_cpu=topk_logprobs_cpu,
            H_per_depth=H_cpu,
            budget=int(budget),
            lambda_=self.lambda_,
            device=device,
        )


def _build_with_entropy_score(
    root_token: int,
    topk_tokens_cpu: list[list[int]],
    topk_logprobs_cpu: list[list[float]],
    H_per_depth: list[float],
    budget: int,
    lambda_: float,
    device: torch.device,
) -> DraftTree:
    """Heap loop identical to crossproduct except for the score:
    heap key uses (cum_logprob - λ * cum_entropy_on_path) instead of
    just cum_logprob. cum_entropy_on_path = Σ_{d on path} H_d.
    """
    D = len(topk_tokens_cpu)
    k = len(topk_tokens_cpu[0]) if D > 0 else 0

    tokens_list: list[int] = [root_token]
    parents_list: list[int] = [-1]
    depths_list: list[int] = [0]
    cum_lp_list: list[float] = [0.0]
    cum_H_list: list[float] = [0.0]  # entropy accumulated on path to this node
    num_nodes = 1

    counter = 0
    # Score is (cum_logprob - λ * cum_H); negate for min-heap.
    heap: list[tuple[float, int, int]] = [(0.0, counter, 0)]

    while heap and num_nodes < budget:
        neg_score, _, node_idx = heapq.heappop(heap)
        d = depths_list[node_idx]
        if d >= D:
            continue
        parent_cum_lp = cum_lp_list[node_idx]
        parent_cum_H = cum_H_list[node_idx]
        # H at the DEPTH of the children we're about to add (d, since
        # we're creating depth-d+1 nodes whose marginal logits come from
        # the parent's outgoing distribution — indexed by d).
        H_at_depth = H_per_depth[d] if d < len(H_per_depth) else 0.0
        children_to_add = min(k, budget - num_nodes)
        for j in range(children_to_add):
            child_token = topk_tokens_cpu[d][j]
            child_cum_lp = parent_cum_lp + topk_logprobs_cpu[d][j]
            child_cum_H = parent_cum_H + H_at_depth
            child_score = child_cum_lp - lambda_ * child_cum_H
            tokens_list.append(child_token)
            parents_list.append(node_idx)
            depths_list.append(d + 1)
            cum_lp_list.append(child_cum_lp)
            cum_H_list.append(child_cum_H)
            counter += 1
            heapq.heappush(heap, (-child_score, counter, num_nodes))
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
