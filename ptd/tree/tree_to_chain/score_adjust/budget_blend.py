"""budget_blend — heap key adds a budget-modulated depth reward.

Keeps fanout open and adds a positive depth bonus to the heap key, weighted by
a budget-dependent factor:

    s'(π) = s_tree(π) + λ(B) · depth(π)
    λ(B)  = exp(-B / B_0) · λ_max

s_tree(π) is the cumulative log-prob along the path (negative). depth(π) is the
node's depth. λ_max is the static depth-bonus magnitude; λ(B) is its
budget-dependent effective weight.

- λ_max = 0  → λ(B) = 0 → s' = s_tree → byte-identical to crossproduct.
- λ_max → ∞  → depth dominates → max-heap pops by depth strictly → collapses to
  a near-chain along the rank-1 lineage (insertion-order tiebreak gives rank-1
  first within a depth bucket).
- B small (B ≪ B_0) → exp(-B/B_0) ≈ 1 → strongest chain bias.
- B large (B ≫ B_0) → exp(-B/B_0) ≈ 0 → baseline cross-product behaviour.

The depth term is POSITIVE (rewards depth) so the high-λ limit pops the deepest
paths first — a chain, not BFS. (A summed-rank-1-logprob spine score would sum
negatives, decreasing with depth, and prefer SHORTER paths — the wrong limit.)

λ_max calibration: for depth-16 paths at block_size=16, λ_max · 16 should be
comparable to |s_tree at depth 16|. With per-depth log-probs around -2,
s_tree ≈ -32 nats, so λ_max ≈ 2 is the rough centerpoint.

Lineage: sweep id V13 (budget_blend).
"""
from __future__ import annotations

import heapq
import math

import torch

from ptd.tree._core.accept import _build_child_maps_cpu
from ptd.tree._core.ancestor import _build_ancestor_matrix_np, _build_packed_ancestor_matrix_np
from ptd.tree._core.base import DraftTree, TreeAlgorithm
from ptd.tree._core.registry import register_tree_algo


@register_tree_algo("budget_blend")
class BudgetBlend(TreeAlgorithm):
    """Depth-reward heap key, budget-modulated weight."""

    def __init__(self, B_0: float = 16.0, lambda_max: float = 2.0):
        if B_0 <= 0.0:
            raise ValueError(f"B_0 must be > 0; got {B_0}")
        if lambda_max < 0.0:
            raise ValueError(f"lambda_max must be >= 0; got {lambda_max}")
        self.B_0 = float(B_0)
        self.lambda_max = float(lambda_max)

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

        # Budget-dependent depth-bonus weight computed once per build.
        lambda_B = math.exp(-budget / self.B_0) * self.lambda_max
        return _build_with_depth_reward(
            root_token=int(root_token),
            topk_tokens_cpu=topk_tokens_cpu,
            topk_logprobs_cpu=topk_logprobs_cpu,
            budget=int(budget),
            lambda_B=lambda_B,
            device=device,
        )


def _build_with_depth_reward(
    root_token: int,
    topk_tokens_cpu: list[list[int]],
    topk_logprobs_cpu: list[list[float]],
    budget: int,
    lambda_B: float,
    device: torch.device,
) -> DraftTree:
    """Heap loop identical to crossproduct except for the score:
    heap key adds lambda_B * depth(node) to the cum_logprob. depth is
    counted with the root at 0 so the root's own bonus is zero.

    lambda_B = 0 → identical to crossproduct.
    lambda_B large → max-heap pops by depth strictly; insertion-order
    tiebreak (siblings inserted in rank order) makes the popped path
    follow the rank-1 lineage.
    """
    D = len(topk_tokens_cpu)
    k = len(topk_tokens_cpu[0]) if D > 0 else 0

    tokens_list: list[int] = [root_token]
    parents_list: list[int] = [-1]
    depths_list: list[int] = [0]
    cum_lp_list: list[float] = [0.0]
    num_nodes = 1

    counter = 0
    # Score = cum_lp + lambda_B * depth; negate for min-heap.
    heap: list[tuple[float, int, int]] = [(0.0, counter, 0)]

    while heap and num_nodes < budget:
        neg_score, _, node_idx = heapq.heappop(heap)
        d = depths_list[node_idx]
        if d >= D:
            continue
        parent_cum_lp = cum_lp_list[node_idx]
        children_to_add = min(k, budget - num_nodes)
        for j in range(children_to_add):
            child_token = topk_tokens_cpu[d][j]
            child_cum_lp = parent_cum_lp + topk_logprobs_cpu[d][j]
            child_depth = d + 1
            child_score = child_cum_lp + lambda_B * child_depth
            tokens_list.append(child_token)
            parents_list.append(node_idx)
            depths_list.append(child_depth)
            cum_lp_list.append(child_cum_lp)
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
