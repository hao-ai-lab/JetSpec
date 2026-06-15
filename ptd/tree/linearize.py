"""Linearize draft trees into root-to-leaf causal path segments.

Each path segment is root-inclusive: the root token appears as row 0 in every
segment. Shared tree nodes are intentionally duplicated once per path through
them. `positions` copies the tree node depth, so UNIT-B can add `past_len` before
feeding RoPE.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from ptd.tree._core.base import DraftTree


@dataclass(frozen=True)
class PathPlan:
    """Varlen causal path representation of a `DraftTree`.

    Attributes:
        token_ids: Long tensor, shape `(total_q,)`. Root-inclusive path tokens.
        cu_seqlens: Int tensor, shape `(num_paths + 1,)`. Segment row offsets.
        positions: Long tensor, shape `(total_q,)`. Per-row tree depth.
        node_of_row: Long tensor, shape `(total_q,)`. Source tree node per row.
        leaf_nodes: Long tensor, shape `(num_paths,)`. Leaf node for each segment.
    """

    token_ids: torch.Tensor
    cu_seqlens: torch.Tensor
    positions: torch.Tensor
    node_of_row: torch.Tensor
    leaf_nodes: torch.Tensor


def _root_to_leaf_paths(parent_indices: list[int], num_nodes: int) -> list[list[int]]:
    is_parent = [False] * num_nodes
    for child in range(1, num_nodes):
        parent = parent_indices[child]
        if 0 <= parent < num_nodes:
            is_parent[parent] = True

    paths: list[list[int]] = []
    for node in range(num_nodes):
        if is_parent[node]:
            continue
        path: list[int] = []
        cur = node
        while cur != -1:
            path.append(cur)
            cur = parent_indices[cur] if cur != 0 else -1
        paths.append(list(reversed(path)))
    return paths or [[0]]


def expand_tree_to_paths(tree: DraftTree) -> PathPlan:
    """Enumerate every root-to-leaf path in `tree`.

    Segments are root-inclusive. For a tree with leaves `l_i`, `total_q` is
    `sum(depth[l_i] + 1)`, and `cu_seqlens` partitions the flattened rows into
    one causal segment per leaf.
    """

    device = tree.token_ids.device
    parents = tree.parent_indices.detach().cpu().tolist()
    paths = _root_to_leaf_paths(parents, int(tree.num_nodes))

    cu = [0]
    rows: list[int] = []
    leaf_nodes: list[int] = []
    for path in paths:
        rows.extend(path)
        leaf_nodes.append(path[-1])
        cu.append(len(rows))

    node_of_row = torch.tensor(rows, dtype=torch.long, device=device)
    return PathPlan(
        token_ids=tree.token_ids[node_of_row].to(dtype=torch.long),
        cu_seqlens=torch.tensor(cu, dtype=torch.int32, device=device),
        positions=tree.depth[node_of_row].to(dtype=torch.long),
        node_of_row=node_of_row,
        leaf_nodes=torch.tensor(leaf_nodes, dtype=torch.long, device=device),
    )


def path_accept(
    plan: PathPlan,
    target_greedy: torch.Tensor,
) -> tuple[list[int], int, int]:
    """Accept by longest prefix across path segments.

    `target_greedy` is the per-row argmax token from verifying `plan.token_ids`.
    An edge is accepted only when the child row's token matches the parent row's
    greedy token and that child is the highest-index sibling carrying that token.
    The return value matches `tree_accept`: accepted node path, accepted draft
    token count excluding root, and correction token at the deepest accepted row.
    """

    flat_greedy = target_greedy.reshape(-1)
    if flat_greedy.numel() != plan.token_ids.numel():
        raise ValueError(
            f"target_greedy must have {plan.token_ids.numel()} entries; "
            f"got {flat_greedy.numel()}"
        )

    cu = plan.cu_seqlens.detach().cpu().tolist()
    tokens = plan.token_ids.detach().cpu().tolist()
    nodes = plan.node_of_row.detach().cpu().tolist()
    greedy = flat_greedy.detach().cpu().tolist()

    survivor: dict[tuple[int, int], int] = {}
    for start, end in zip(cu, cu[1:]):
        for row in range(start + 1, end):
            parent_node = nodes[row - 1]
            child_node = nodes[row]
            token = tokens[row]
            key = (parent_node, token)
            if child_node > survivor.get(key, -1):
                survivor[key] = child_node

    best_len = -1
    best_last_node = 0
    best_last_row = 0
    best_start = 0

    for start, end in zip(cu, cu[1:]):
        accepted = 0
        last_node = nodes[start]
        last_row = start
        for row in range(start + 1, end):
            parent_row = row - 1
            parent_node = nodes[parent_row]
            child_node = nodes[row]
            token = tokens[row]
            if token != greedy[parent_row] or survivor.get((parent_node, token)) != child_node:
                break
            accepted += 1
            last_node = child_node
            last_row = row

        if accepted > best_len or (accepted == best_len and last_node > best_last_node):
            best_len = accepted
            best_last_node = last_node
            best_last_row = last_row
            best_start = start

    accepted_path = nodes[best_start : best_start + best_len + 1]
    return accepted_path, best_len, int(greedy[best_last_row])


__all__ = ["PathPlan", "expand_tree_to_paths", "path_accept"]
