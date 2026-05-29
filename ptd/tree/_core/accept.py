"""Algorithm-agnostic tree acceptance (verification walk).

Ported from causal_parallel_drafting/model/tree.py (tree_accept,
_build_child_maps_cpu). Acceptance is decided by the target's argmax
at each node — walk down child_maps[current_node][target_pred_token]
until no matching child, returning the longest accepted path.
"""
from __future__ import annotations

import torch

from .base import DraftTree


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
