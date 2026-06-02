"""rank_decay — per-node fanout that decays geometrically with sibling rank.

Per-node fanout depends only on the node's ordinal rank among its siblings:

    b_{rank-r child} = max(1, round(K · γ^(r-1)))

γ = 1 recovers crossproduct. γ → 0 collapses to a chain (only the rank-1 spine
fans out). Uses sibling rank only — no probability or entropy.

This is the rigorous NULL control for the uncertainty-aware thesis: if a purely
ordinal, signal-free decay performs comparably to the entropy/gap variants,
then the structural bias toward the spine is what helps, not the uncertainty
signal itself. Ship it as the honest baseline that the signal-driven methods
must beat.

γ ∈ (0, 1].

Lineage: sweep id V10 (rank_aware_fanout).
"""
from __future__ import annotations

import heapq
import math

import torch

from ptd.tree._core.accept import _build_child_maps_cpu
from ptd.tree._core.ancestor import _build_ancestor_matrix_np, _build_packed_ancestor_matrix_np
from ptd.tree._core.base import DraftTree, TreeAlgorithm
from ptd.tree._core.registry import register_tree_algo


@register_tree_algo("rank_decay")
class RankAwareFanout(TreeAlgorithm):
    """Per-node fanout cap = max(1, round(K * γ^(rank-1))) by sibling rank."""

    def __init__(self, gamma: float = 0.5):
        if not 0.0 < gamma <= 1.0:
            raise ValueError(f"gamma must be in (0, 1]; got {gamma}")
        self.gamma = float(gamma)

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
        topk_tokens_cpu = topk_tok_t.tolist()
        topk_logprobs_cpu = topk_lp_t.tolist()

        return _build_with_rank_aware_fanout(
            root_token=int(root_token),
            topk_tokens_cpu=topk_tokens_cpu,
            topk_logprobs_cpu=topk_logprobs_cpu,
            tree_width=tree_width,
            budget=int(budget),
            gamma=self.gamma,
            device=device,
        )


def _fanout_for_sibling_rank(K: int, rank: int, gamma: float) -> int:
    """b_r = max(1, round(K * γ^(r-1))). rank is 1-indexed."""
    return max(1, int(round(K * (gamma ** (rank - 1)))))


def _build_with_rank_aware_fanout(
    root_token: int,
    topk_tokens_cpu: list[list[int]],
    topk_logprobs_cpu: list[list[float]],
    tree_width: int,
    budget: int,
    gamma: float,
    device: torch.device,
) -> DraftTree:
    """Heap loop with per-node fanout cap derived from sibling rank.

    Each node carries a `fanout_cap` attribute (its own b_r) derived from
    its sibling rank at insertion time. When the node is popped, it
    expands up to its own cap (limited by remaining budget), and the
    children get THEIR own caps based on their per-position rank.

    γ=1 → every node gets cap=K → recovers crossproduct.
    γ→0 → rank-1 child gets cap=K, rank-2..K children get cap=1 (chain
          spine).
    """
    D = len(topk_tokens_cpu)
    k = len(topk_tokens_cpu[0]) if D > 0 else 0

    tokens_list: list[int] = [root_token]
    parents_list: list[int] = [-1]
    depths_list: list[int] = [0]
    cum_lp_list: list[float] = [0.0]
    fanout_cap_list: list[int] = [k]  # root expands at full K
    num_nodes = 1

    counter = 0
    heap: list[tuple[float, int, int]] = [(0.0, counter, 0)]

    while heap and num_nodes < budget:
        neg_cum_lp, _, node_idx = heapq.heappop(heap)
        d = depths_list[node_idx]
        if d >= D:
            continue
        cap_here = fanout_cap_list[node_idx]
        children_to_add = min(cap_here, k, budget - num_nodes)
        for j in range(children_to_add):
            child_token = topk_tokens_cpu[d][j]
            child_cum_lp = -neg_cum_lp + topk_logprobs_cpu[d][j]
            child_rank = j + 1  # sibling rank, 1-indexed
            child_fanout_cap = _fanout_for_sibling_rank(k, child_rank, gamma)
            tokens_list.append(child_token)
            parents_list.append(node_idx)
            depths_list.append(d + 1)
            cum_lp_list.append(child_cum_lp)
            fanout_cap_list.append(child_fanout_cap)
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
