"""PyTorch reference for tree-masked sparse attention.

Attention pattern during tree-based speculative-decoding verification:
  - Each query node attends to ALL prefix KV positions (DENSE).
  - Each query node attends to tree KV position j ONLY IF j is an
    ancestor of that query node (SPARSE, governed by ancestor matrix).

Tensor shapes:
  query    (B, H, N, D)
  key      (B, H_KV, prefix_len + N, D)
  value    (B, H_KV, prefix_len + N, D)
  ancestor (N, N) bool/uint8
  output   (B, H, N, D)

H != H_KV when using Grouped Query Attention (GQA).
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass

import torch


@dataclass
class DraftTree:
    """Flat BFS representation of a draft-token tree."""

    token_ids: torch.Tensor
    parent_indices: torch.Tensor
    depth: torch.Tensor
    num_nodes: int


def compute_tree_budget(block_size: int, tree_width: int, max_budget: int = 256) -> int:
    """How many nodes in a full tree_width-ary tree of block_size levels."""
    if tree_width <= 1:
        return block_size
    full_tree = (tree_width**block_size - 1) // (tree_width - 1)
    return min(full_tree, max_budget)


def build_tree_from_topk(
    root_token: int,
    topk_tokens: torch.Tensor,
    topk_logprobs: torch.Tensor,
    budget: int,
    device: torch.device,
) -> DraftTree:
    """Greedy tree expansion from independent top-k marginals (Approach A)."""
    D, k = topk_tokens.shape

    tokens_list: list[int] = [root_token]
    parents_list: list[int] = [-1]
    depths_list: list[int] = [0]
    num_nodes = 1

    counter = 0
    heap: list[tuple[float, int, int]] = [(0.0, counter, 0)]

    while heap and num_nodes < budget:
        neg_cum_lp, _, node_idx = heapq.heappop(heap)
        d = depths_list[node_idx]
        if d >= D:
            continue
        for j in range(min(k, budget - num_nodes)):
            tokens_list.append(topk_tokens[d, j].item())
            parents_list.append(node_idx)
            depths_list.append(d + 1)
            counter += 1
            child_cum_lp = -neg_cum_lp + topk_logprobs[d, j].item()
            heapq.heappush(heap, (-child_cum_lp, counter, num_nodes))
            num_nodes += 1

    return DraftTree(
        token_ids=torch.tensor(tokens_list, dtype=torch.long, device=device),
        parent_indices=torch.tensor(parents_list, dtype=torch.long, device=device),
        depth=torch.tensor(depths_list, dtype=torch.long, device=device),
        num_nodes=num_nodes,
    )


def build_ancestor_matrix(tree: DraftTree) -> torch.Tensor:
    """Return an (N, N) bool matrix where ancestor[i, j] is True iff j is on i's path."""
    N = tree.num_nodes
    device = tree.parent_indices.device
    ancestor = torch.eye(N, dtype=torch.bool, device=device)
    for i in range(1, N):
        parent_idx = tree.parent_indices[i].item()
        ancestor[i] = ancestor[i] | ancestor[parent_idx]
    return ancestor


def build_tree_attention_mask(
    tree: DraftTree,
    prefix_len: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Materialise a 4-D additive mask for F.scaled_dot_product_attention."""
    N = tree.num_nodes
    ancestor = build_ancestor_matrix(tree)

    full_len = prefix_len + N
    mask = torch.zeros(1, 1, N, full_len, dtype=dtype, device=device)
    blocked = ~ancestor
    mask[:, :, :, prefix_len:].masked_fill_(
        blocked.unsqueeze(0).unsqueeze(0),
        torch.finfo(dtype).min,
    )
    return mask


def reference_tree_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    ancestor: torch.Tensor,
    prefix_len: int,
    sm_scale: float | None = None,
) -> torch.Tensor:
    """Naive PyTorch tree attention: scores -> mask -> softmax -> output."""
    B, H, N, D = query.shape
    _, H_KV, KV_LEN, _ = key.shape

    if sm_scale is None:
        sm_scale = D**-0.5

    num_kv_groups = H // H_KV
    if num_kv_groups > 1:
        key = key.repeat_interleave(num_kv_groups, dim=1)
        value = value.repeat_interleave(num_kv_groups, dim=1)

    scores = torch.matmul(query.float(), key.float().transpose(-2, -1)) * sm_scale

    q_idx = torch.arange(N, device=query.device)
    kv_idx = torch.arange(KV_LEN, device=query.device)

    is_prefix = kv_idx[None, :] < prefix_len
    in_tree = kv_idx[None, :] >= prefix_len
    tree_kv = (kv_idx[None, :] - prefix_len).clamp(min=0)
    is_ancestor = ancestor[q_idx[:, None], tree_kv].bool()

    attend = is_prefix | (in_tree & is_ancestor)
    scores.masked_fill_(~attend.unsqueeze(0).unsqueeze(0), float("-inf"))

    attn_weights = torch.softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, value.float())
    return output.to(query.dtype)

