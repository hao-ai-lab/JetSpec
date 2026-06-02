"""drift_brake — per-PATH fanout cap from drift off the rank-1 chain.

Unlike the per-depth caps, this decides fanout per node from how far the
node's prefix has drifted below the rank-1 (greedy) chain at the same depth:

    Δ(π) = cum_logprob(π) - cum_logprob(rank-1 chain at same depth)
    b_n  = K  if Δ(π_n) ≥ -δ  else  1

"Fanout the trusted prefix, chain-extend the suspect ones." A path that stayed
close to the rank-1 spine deserves full exploration of its children; a path
that has wandered off should chain-extend (rank-1 only), since spending more
budget on its descendants is probably waste.

This is the finest-grained signal available without re-running the drafter:
each node carries its own cumulative drift, so the decision is personalized to
lineage.

Identity recovery: δ → ∞ → all paths have Δ ≥ -∞ → b = K → crossproduct.

Caveat: the heap already prefers low-drift paths (higher cum_logprob → pop
first), so braking fanout on high-drift paths may double-suppress them — sweep
δ to find the sweet spot.

Lineage: sweep id V9 (cumulative_drift_brake).
"""
from __future__ import annotations

import heapq
import math

import torch

from ptd.tree._core.accept import _build_child_maps_cpu
from ptd.tree._core.ancestor import _build_ancestor_matrix_np, _build_packed_ancestor_matrix_np
from ptd.tree._core.base import DraftTree, TreeAlgorithm
from ptd.tree._core.registry import register_tree_algo


@register_tree_algo("drift_brake")
class CumulativeDriftBrake(TreeAlgorithm):
    """Per-path fanout cap: full K if drift Δ from rank-1 chain ≥ -δ, else 1."""

    def __init__(self, delta: float = math.inf):
        if delta < 0.0:
            raise ValueError(f"delta must be >= 0; got {delta}")
        self.delta = float(delta)

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

        # rank-1 chain cumulative logprob per depth: at depth d (i.e. a node
        # AT depth d), the reference cum_lp is Σ_{i=0}^{d-1} topk_logprobs[i][0].
        # We need this for d in 0..D. Cached in rank1_cum_at_depth[d].
        rank1_cum_at_depth: list[float] = [0.0]
        running = 0.0
        for d in range(len(topk_logprobs_cpu)):
            running += topk_logprobs_cpu[d][0]
            rank1_cum_at_depth.append(running)

        return _build_with_drift_brake(
            root_token=int(root_token),
            topk_tokens_cpu=topk_tokens_cpu,
            topk_logprobs_cpu=topk_logprobs_cpu,
            rank1_cum_at_depth=rank1_cum_at_depth,
            tree_width=tree_width,
            budget=int(budget),
            delta=self.delta,
            device=device,
        )


def _build_with_drift_brake(
    root_token: int,
    topk_tokens_cpu: list[list[int]],
    topk_logprobs_cpu: list[list[float]],
    rank1_cum_at_depth: list[float],
    tree_width: int,
    budget: int,
    delta: float,
    device: torch.device,
) -> DraftTree:
    """Heap loop with per-node fanout cap from path drift.

    For each popped node n at depth d with cum_lp(n) accumulated along
    its lineage:
        Δ(n) = cum_lp(n) - rank1_cum_at_depth[d]
        b_n = tree_width if Δ(n) ≥ -delta else 1
    """
    D = len(topk_tokens_cpu)
    k = tree_width

    tokens_list: list[int] = [root_token]
    parents_list: list[int] = [-1]
    depths_list: list[int] = [0]
    cum_lp_list: list[float] = [0.0]
    num_nodes = 1

    counter = 0
    heap: list[tuple[float, int, int]] = [(0.0, counter, 0)]

    while heap and num_nodes < budget:
        neg_cum_lp, _, node_idx = heapq.heappop(heap)
        d = depths_list[node_idx]
        if d >= D:
            continue
        cum_lp = -neg_cum_lp
        # Drift from rank-1 chain at this depth.
        drift = cum_lp - rank1_cum_at_depth[d]
        b_n = k if drift >= -delta else 1
        children_to_add = min(b_n, k, budget - num_nodes)
        for j in range(children_to_add):
            child_token = topk_tokens_cpu[d][j]
            child_cum_lp = cum_lp + topk_logprobs_cpu[d][j]
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
