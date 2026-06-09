"""Algorithm-agnostic tree acceptance (verification walk).

Ported from causal_parallel_drafting/model/tree.py (tree_accept,
_build_child_maps_cpu). Acceptance is decided by the target's argmax
at each node — walk down child_maps[current_node][target_pred_token]
until no matching child, returning the longest accepted path.
"""
from __future__ import annotations

import torch

from .base import DraftTree


_ACCEPT_STATIC_CACHE: dict[
    tuple[int, tuple[str, int | None], torch.dtype],
    tuple[torch.Tensor, torch.Tensor, torch.Tensor],
] = {}


def _device_key(device: torch.device) -> tuple[str, int | None]:
    device = torch.device(device)
    return device.type, device.index


def _static_accept_tensors(
    num_nodes: int,
    device: torch.device,
    score_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-(N, device, dtype) tensors reused across accept calls."""
    key = (num_nodes, _device_key(device), score_dtype)
    cached = _ACCEPT_STATIC_CACHE.get(key)
    if cached is None:
        node_ids = torch.arange(num_nodes, device=device)
        later_node = node_ids[None, :] > node_ids[:, None]
        neg_ones = torch.empty(num_nodes, dtype=score_dtype, device=device).fill_(-1)
        cached = (node_ids, later_node, neg_ones)
        _ACCEPT_STATIC_CACHE[key] = cached
    return cached


def _build_child_maps_cpu(
    token_ids: list[int],
    parents: list[int],
    num_nodes: int,
) -> list[dict[int, int]]:
    """parent-token -> child-node-index lookup for verified-path follow."""
    child_maps: list[dict[int, int]] = [dict() for _ in range(num_nodes)]
    for child_idx in range(1, num_nodes):
        parent_idx = parents[child_idx]
        if 0 <= parent_idx < num_nodes:
            child_maps[parent_idx][token_ids[child_idx]] = child_idx
    return child_maps


def _sample_greedy(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Local greedy/temperature sampler (avoids cross-package coupling).

    For temperature == 0 returns argmax. For temperature > 0 returns a
    multinomial draw from the temperature-scaled softmax. Matches the
    behavior of causal_parallel_drafting.model.utils.sample for the
    code paths exercised by tree_accept.
    """
    if temperature <= 0.0:
        return logits.argmax(dim=-1)
    probs = torch.softmax(logits / temperature, dim=-1)
    flat = probs.reshape(-1, probs.shape[-1])
    sampled = torch.multinomial(flat, num_samples=1).squeeze(-1)
    return sampled.reshape(probs.shape[:-1])


def gpu_tree_accept(
    tree_tokens: torch.Tensor,
    greedy_targets: torch.Tensor,
    parent_indices: torch.Tensor,
    depths: torch.Tensor,
    max_depth: int | None = None,
) -> tuple[torch.Tensor, int, torch.Tensor]:
    """Vectorized greedy tree acceptance over flat tree tensors.

    Args:
        tree_tokens: (N,) draft token ids, including the root slot.
        greedy_targets: (N,) or (1, N) precomputed target argmax tokens.
        parent_indices: (N,) parent node index per node (-1 for root).
        depths: (N,) depth per node, with root depth 0.
        max_depth: optional upper bound used for path extraction.

    Returns:
        accepted_path: LongTensor (L,) root-inclusive accepted node indices.
        accepted_len: Python int accepted draft-token count; this is the
            single permitted host sync.
        correction: 0-d LongTensor target argmax at the last accepted node.
    """
    device = tree_tokens.device
    num_nodes = tree_tokens.shape[0]
    greedy_targets = greedy_targets.squeeze(0) if greedy_targets.dim() == 2 else greedy_targets

    if num_nodes <= 1:
        return torch.zeros(1, dtype=torch.long, device=device), 0, greedy_targets[0]

    if max_depth is None:
        max_depth = num_nodes - 1

    _, later_node, neg_ones = _static_accept_tensors(num_nodes, device, depths.dtype)
    safe_parents = parent_indices.clamp(min=0, max=num_nodes - 1)
    valid_parent = (parent_indices >= 0) & (parent_indices < num_nodes)

    same_parent = parent_indices[:, None] == parent_indices[None, :]
    same_token = tree_tokens[:, None] == tree_tokens[None, :]
    overwritten = (same_parent & same_token & later_node).any(dim=1)

    match = torch.empty(num_nodes, dtype=torch.bool, device=device)
    match[0] = True
    match[1:] = (
        valid_parent[1:]
        & ~overwritten[1:]
        & (tree_tokens[1:] == greedy_targets[safe_parents[1:]])
    )

    prefix_match = match.clone()
    jump = safe_parents.clone()
    for _ in range(max(1, max_depth.bit_length())):
        prefix_match = prefix_match & prefix_match[jump]
        jump = jump[jump]

    score = torch.where(prefix_match, depths, neg_ones)
    best_node = torch.argmax(score)
    accepted_depth = depths[best_node]
    correction = greedy_targets[best_node]

    path_buf = torch.empty(max_depth + 1, dtype=torch.long, device=device)
    current = best_node.unsqueeze(0)
    for depth in range(max_depth, -1, -1):
        path_buf[depth : depth + 1] = current
        current = safe_parents[current]

    accepted_len = int(accepted_depth.item())
    valid_start = max_depth - accepted_len
    accepted_path = path_buf[valid_start : max_depth + 1].contiguous()
    return accepted_path, accepted_len, correction


def tree_accept(
    tree: DraftTree,
    target_logits: torch.Tensor,  # (1, N, vocab_size)
    temperature: float = 0.0,
) -> tuple[list[int], int, int]:
    """Find the longest accepted root-to-leaf path.

    Returns:
        accepted_path:     node indices of accepted prefix (root … last accepted).
        acceptance_length: number of accepted draft tokens (excludes root).
        correction_token:  target posterior at the last accepted node.
    """
    posterior = _sample_greedy(target_logits, temperature)  # (1, N)
    posterior_tokens = posterior.squeeze(0).tolist()

    if tree.child_maps is None:
        token_ids = tree.token_ids.tolist()
        parent_indices = tree.parent_indices.tolist()
        tree.child_maps = _build_child_maps_cpu(token_ids, parent_indices, tree.num_nodes)

    accepted_path = [0]
    current = 0
    while True:
        next_token = posterior_tokens[current]
        child_idx = tree.child_maps[current].get(next_token)
        if child_idx is None:
            break
        accepted_path.append(child_idx)
        current = child_idx

    acceptance_length = len(accepted_path) - 1
    correction_token = posterior_tokens[current]
    return accepted_path, acceptance_length, correction_token
