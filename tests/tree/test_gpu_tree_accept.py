import random

import pytest
import torch

from jetspec.tree._core.accept import gpu_tree_accept, tree_accept
from jetspec.tree._core.base import DraftTree


def _tree(token_ids, parent_indices):
    parents = torch.tensor(parent_indices, dtype=torch.long)
    depths = torch.zeros(len(parent_indices), dtype=torch.long)
    for idx in range(1, len(parent_indices)):
        depths[idx] = depths[parents[idx]] + 1
    return DraftTree(
        token_ids=torch.tensor(token_ids, dtype=torch.long),
        parent_indices=parents,
        depth=depths,
        num_nodes=len(token_ids),
    )


def _target_logits(greedy_targets):
    greedy_targets = torch.tensor(greedy_targets, dtype=torch.long)
    vocab_size = int(greedy_targets.max().item()) + 1
    logits = torch.full((1, greedy_targets.numel(), vocab_size), -1000.0)
    logits[0, torch.arange(greedy_targets.numel()), greedy_targets] = 1000.0
    return logits


def _assert_matches_oracle(tree, greedy_targets):
    logits = _target_logits(greedy_targets)
    expected_path, expected_len, expected_correction = tree_accept(tree, logits)

    actual_path, actual_len, actual_correction = gpu_tree_accept(
        tree.token_ids,
        torch.tensor(greedy_targets, dtype=torch.long),
        tree.parent_indices,
        tree.depth,
    )

    assert torch.equal(actual_path, torch.tensor(expected_path, dtype=torch.long))
    assert actual_len == expected_len
    assert actual_correction.ndim == 0
    assert int(actual_correction.item()) == expected_correction


@pytest.mark.parametrize(
    ("tree", "greedy_targets"),
    [
        (_tree([0], [-1]), [9]),
        (_tree([0, 11, 12, 13, 14], [-1, 0, 1, 2, 3]), [11, 12, 13, 14, 99]),
        (_tree([0, 21, 22, 31, 32, 33, 34], [-1, 0, 0, 1, 1, 2, 2]), [22, 31, 34, 90, 91, 92, 93]),
        (
            _tree(
                [0, 10, 11, 20, 21, 22, 30, 31, 32, 40, 41, 42, 50, 51, 52],
                [-1, 0, 0, 1, 1, 2, 2, 3, 3, 7, 7, 8, 9, 9, 12],
            ),
            [10, 20, 22, 31, 91, 92, 93, 40, 42, 50, 95, 96, 52, 97, 98],
        ),
        (_tree([0, 5, 6, 7], [-1, 0, 1, 2]), [5, 99, 7, 100]),
        (_tree([0, 61, 62, 63, 64, 65], [-1, 0, 1, 2, 3, 4]), [61, 62, 63, 64, 65, 101]),
        (_tree([0, 7, 7, 8], [-1, 0, 0, 2]), [7, 90, 8, 91]),
        (_tree([0, 3, 4, 5, 5, 6], [-1, 0, 1, 2, 2, 4]), [3, 4, 5, 90, 6, 91]),
        (_tree([0, 10, 20, 20, 30, 40, 50], [-1, 0, 1, 1, 3, 4, 2]), [10, 20, 50, 30, 40, 91, 92]),
    ],
)
def test_gpu_tree_accept_matches_cpu_oracle_on_fixed_trees(tree, greedy_targets):
    _assert_matches_oracle(tree, greedy_targets)


def _child_maps(token_ids, parent_indices):
    maps = [dict() for _ in token_ids]
    for idx in range(1, len(token_ids)):
        parent = parent_indices[idx]
        if 0 <= parent < len(token_ids):
            maps[parent][token_ids[idx]] = idx
    return maps


def _random_tree(rng):
    budget = rng.randint(1, 127)
    max_depth = rng.randint(1, 8)
    max_width = rng.randint(1, 5)
    token_ids = [0]
    parent_indices = [-1]
    child_counts = [0]
    depths = [0]

    while len(token_ids) < budget:
        candidates = [
            idx
            for idx, depth in enumerate(depths)
            if depth < max_depth and child_counts[idx] < max_width
        ]
        if not candidates:
            break
        parent = rng.choice(candidates)
        child_counts[parent] += 1
        parent_indices.append(parent)
        depths.append(depths[parent] + 1)
        child_counts.append(0)
        token_ids.append(rng.randint(1, 19))

    return token_ids, parent_indices


def _missing_token(children, rng):
    used = set(children)
    token = rng.randint(20, 80)
    while token in used:
        token += 1
    return token


def _engineered_targets(token_ids, parent_indices, rng, mode):
    maps = _child_maps(token_ids, parent_indices)
    targets = [rng.randint(1, 80) for _ in token_ids]

    if mode == 0:
        targets[0] = _missing_token(maps[0], rng)
        return targets

    current = 0
    while maps[current]:
        children = list(maps[current].values())
        child = rng.choice(children)
        targets[current] = token_ids[child]
        current = child
        if mode == 1 or (mode == 2 and rng.random() < 0.35):
            break

    targets[current] = _missing_token(maps[current], rng)
    return targets


def test_gpu_tree_accept_matches_cpu_oracle_on_randomized_trees():
    rng = random.Random(20260609)

    for case in range(200):
        token_ids, parent_indices = _random_tree(rng)
        targets = _engineered_targets(token_ids, parent_indices, rng, case % 4)
        _assert_matches_oracle(_tree(token_ids, parent_indices), targets)


def test_gpu_tree_accept_uses_at_most_one_host_sync(monkeypatch):
    tree = _tree([0, 11, 12, 13, 14], [-1, 0, 1, 2, 3])
    greedy_targets = torch.tensor([11, 12, 13, 14, 99], dtype=torch.long)
    counts = {"item": 0, "tolist": 0, "cpu": 0}

    orig_item = torch.Tensor.item
    orig_tolist = torch.Tensor.tolist
    orig_cpu = torch.Tensor.cpu

    def counted_item(self, *args, **kwargs):
        counts["item"] += 1
        return orig_item(self, *args, **kwargs)

    def counted_tolist(self, *args, **kwargs):
        counts["tolist"] += 1
        return orig_tolist(self, *args, **kwargs)

    def counted_cpu(self, *args, **kwargs):
        counts["cpu"] += 1
        return orig_cpu(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "item", counted_item)
    monkeypatch.setattr(torch.Tensor, "tolist", counted_tolist)
    monkeypatch.setattr(torch.Tensor, "cpu", counted_cpu)

    gpu_tree_accept(tree.token_ids, greedy_targets, tree.parent_indices, tree.depth)

    assert sum(counts.values()) <= 1
    assert counts["tolist"] == 0
    assert counts["cpu"] == 0
