"""Shared helper: heap loop with per-depth fanout cap.

Used by every algorithm in tree_to_chain/fanout_cap/ (V1, V2, V3, V5, V6).
Differs from the V0 crossproduct heap loop only in that `children_to_add`
is bounded by a per-depth cap `b_per_depth[d]` instead of the global k.
"""
from __future__ import annotations

import heapq

import torch

from .accept import _build_child_maps_cpu
from .ancestor import _build_ancestor_matrix_np, _build_packed_ancestor_matrix_np
from .base import DraftTree


def build_with_per_depth_cap(
    root_token: int,
    topk_tokens_cpu: list[list[int]],
    topk_logprobs_cpu: list[list[float]],
    b_per_depth: list[int],
    budget: int,
    device: torch.device,
) -> DraftTree:
    """Heap loop where children_to_add uses per-depth cap b_per_depth[d]
    instead of the global k. Identical to crossproduct otherwise.

    Identity recovery: if b_per_depth[d] >= k for all d, output is
    byte-identical to V0 crossproduct.
    """
    D = len(topk_tokens_cpu)
    k = len(topk_tokens_cpu[0]) if D > 0 else 0

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
        b_d = b_per_depth[d] if d < len(b_per_depth) else k
        children_to_add = min(b_d, k, budget - num_nodes)
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
