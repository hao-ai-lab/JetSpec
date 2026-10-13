"""Algorithm-agnostic ancestor matrix helpers.

Ported from causal_parallel_drafting/model/tree.py (build_ancestor_matrix,
build_packed_ancestor_matrix, _build_ancestor_matrix_np). The Triton
tree-attention kernel consumes a dense uint8 (N, N) matrix; SDPA consumes
the bool form via build_tree_attention_mask.

Trees are tiny (<= a few hundred nodes), so NumPy on the host beats GPU
kernels per call. One H2D copy at the end is the only device transfer.
"""
from __future__ import annotations

import numpy as np
import torch

from .base import DraftTree


def _build_ancestor_matrix_np(parents: list[int], num_nodes: int) -> np.ndarray:
    """Dense bool ancestor matrix in parent-before-child order."""
    anc_np = np.eye(num_nodes, dtype=np.bool_)
    for i in range(1, num_nodes):
        p = parents[i]
        if 0 <= p < num_nodes:
            anc_np[i] |= anc_np[p]
    return anc_np


def _build_packed_ancestor_matrix_np(parents: list[int], num_nodes: int) -> np.ndarray:
    """Dense uint8 ancestor matrix for the Triton tree-attention kernel."""
    return _build_ancestor_matrix_np(parents, num_nodes).astype(np.uint8, copy=False)


def build_ancestor_matrix(tree: DraftTree) -> torch.Tensor:
    """Bool ancestor matrix as a torch tensor on tree.parent_indices.device."""
    if tree.ancestor is not None:
        return tree.ancestor
    N = tree.num_nodes
    device = tree.parent_indices.device
    parents = tree.parent_indices.tolist()
    anc_np = _build_ancestor_matrix_np(parents, N)
    return torch.from_numpy(anc_np).to(device, non_blocking=True)


def build_packed_ancestor_matrix(tree: DraftTree) -> torch.Tensor:
    """uint8 ancestor matrix for the Triton tree-attention path."""
    if tree.ancestor_packed is not None:
        return tree.ancestor_packed
    if tree.ancestor is not None:
        return tree.ancestor.to(dtype=torch.uint8).contiguous()
    N = tree.num_nodes
    device = tree.parent_indices.device
    parents = tree.parent_indices.tolist()
    anc_np = _build_packed_ancestor_matrix_np(parents, N)
    return torch.from_numpy(anc_np).to(device, non_blocking=True)
