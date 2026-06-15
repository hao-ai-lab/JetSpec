import random

import torch

from ptd.tree._core.accept import tree_accept
from ptd.tree._core.base import DraftTree
from ptd.tree.baselines.crossproduct import CrossProduct, _build_from_topk
from ptd.tree.linearize import expand_tree_to_paths, path_accept
from ptd.tree.tree_to_chain.fanout_cap.top2gap import Top2GapFanout


def _make_logits(rng: random.Random, depth: int, vocab_size: int) -> torch.Tensor:
    gen = torch.Generator().manual_seed(rng.randint(0, 2**31 - 1))
    return torch.randn(1, depth, vocab_size, generator=gen)


def _crossproduct_tree(rng: random.Random) -> DraftTree:
    return CrossProduct().build(
        root_token=rng.randint(0, 127),
        draft_logits=_make_logits(rng, depth=15, vocab_size=128),
        block_size=16,
        tree_width=7,
        budget=127,
        device=torch.device("cpu"),
    )


def _top2gap_tree(rng: random.Random) -> DraftTree:
    return Top2GapFanout(beta=1.0, g_0=1.0).build(
        root_token=rng.randint(0, 127),
        draft_logits=_make_logits(rng, depth=15, vocab_size=128),
        block_size=16,
        tree_width=7,
        budget=63,
        device=torch.device("cpu"),
    )


def _dup_sibling_crossproduct_tree(rng: random.Random) -> DraftTree:
    vocab_size = rng.randint(3, 24)
    depth = rng.randint(1, 5)
    width = rng.randint(2, min(vocab_size, 6))
    topk_tokens: list[list[int]] = []
    topk_logprobs: list[list[float]] = []

    for _ in range(depth):
        tokens = rng.sample(range(vocab_size), width)
        dup_token = tokens[0]
        tokens[1] = dup_token
        for idx in rng.sample(range(2, width), rng.randint(0, max(width - 2, 0))):
            tokens[idx] = dup_token
        logprobs = sorted((-abs(rng.gauss(0, 2)) for _ in tokens), reverse=True)
        topk_tokens.append(tokens)
        topk_logprobs.append(logprobs)

    return _build_from_topk(
        root_token=rng.randint(0, vocab_size - 1),
        topk_tokens_cpu=topk_tokens,
        topk_logprobs_cpu=topk_logprobs,
        budget=rng.randint(8, 60),
        device=torch.device("cpu"),
    )


def _has_dup_sibling(tree: DraftTree) -> bool:
    seen = set()
    for parent, token in zip(tree.parent_indices.tolist()[1:], tree.token_ids.tolist()[1:]):
        key = (parent, token)
        if key in seen:
            return True
        seen.add(key)
    return False


def _target_logits(rng: random.Random, tree: DraftTree) -> torch.Tensor:
    tokens = tree.token_ids.tolist()
    parents = tree.parent_indices.tolist()
    vocab_size = int(tree.token_ids.max().item()) + 32
    children: list[list[int]] = [[] for _ in range(tree.num_nodes)]
    for child in range(1, tree.num_nodes):
        parent = parents[child]
        if 0 <= parent < tree.num_nodes:
            children[parent].append(child)

    logits = torch.full((1, tree.num_nodes, vocab_size), -10.0)
    for node in range(tree.num_nodes):
        if children[node] and rng.random() < 0.75:
            target_token = tokens[rng.choice(children[node])]
        else:
            target_token = rng.randint(0, vocab_size - 1)
        logits[0, node] = torch.randn(vocab_size) * 0.01
        logits[0, node, target_token] = 100.0
    return logits


def _per_row_greedy(tree: DraftTree, target_logits: torch.Tensor) -> torch.Tensor:
    plan = expand_tree_to_paths(tree)
    node_greedy = target_logits.squeeze(0).argmax(dim=-1)
    return node_greedy[plan.node_of_row]


def _independent_paths(tree: DraftTree) -> list[list[int]]:
    parents = tree.parent_indices.tolist()
    is_parent = [False] * tree.num_nodes
    for child in range(1, tree.num_nodes):
        parent = parents[child]
        if 0 <= parent < tree.num_nodes:
            is_parent[parent] = True

    paths = []
    for node in range(tree.num_nodes):
        if is_parent[node]:
            continue
        path = []
        cur = node
        while cur != -1:
            path.append(cur)
            cur = parents[cur] if cur != 0 else -1
        paths.append(list(reversed(path)))
    return paths or [[0]]


def _assert_structural_plan(tree: DraftTree) -> None:
    plan = expand_tree_to_paths(tree)
    paths = _independent_paths(tree)
    cu = plan.cu_seqlens.tolist()

    assert cu[0] == 0
    assert all(left <= right for left, right in zip(cu, cu[1:]))
    assert cu[-1] == len(plan.token_ids)
    assert cu[-1] == sum(len(path) for path in paths)
    assert len(cu) == len(paths) + 1

    for path_idx, expected_path in enumerate(paths):
        start, end = cu[path_idx], cu[path_idx + 1]
        nodes = plan.node_of_row[start:end].tolist()
        assert nodes == expected_path
        assert plan.token_ids[start:end].tolist() == tree.token_ids[expected_path].tolist()
        assert plan.positions[start:end].tolist() == list(range(len(expected_path)))
        for parent_node, child_node in zip(nodes, nodes[1:]):
            assert int(tree.parent_indices[child_node].item()) == parent_node


def test_expand_tree_to_paths_structural_real_builders():
    rng = random.Random(20260614)

    for _ in range(200):
        _assert_structural_plan(_crossproduct_tree(rng))

    for _ in range(200):
        _assert_structural_plan(_top2gap_tree(rng))


def test_path_accept_matches_tree_accept_on_real_and_duplicate_trees():
    rng = random.Random(20260615)
    generators = (
        ("crossproduct@127", _crossproduct_tree, 150),
        ("top2gap@63", _top2gap_tree, 150),
        ("dup_sibling_crossproduct", _dup_sibling_crossproduct_tree, 300),
    )
    total = 0
    dup_sibling_trees = 0

    for name, generator, count in generators:
        for _ in range(count):
            tree = generator(rng)
            target_logits = _target_logits(rng, tree)
            expected_path, expected_len, expected_correction = tree_accept(tree, target_logits)

            plan = expand_tree_to_paths(tree)
            actual_path, actual_len, actual_correction = path_accept(
                plan,
                _per_row_greedy(tree, target_logits),
            )

            assert (actual_path, actual_len, actual_correction) == (
                expected_path,
                expected_len,
                expected_correction,
            ), name
            total += 1
            dup_sibling_trees += int(_has_dup_sibling(tree))

    assert total == 600
    assert dup_sibling_trees >= 300


def _tree(token_ids: list[int], parent_indices: list[int]) -> DraftTree:
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


def _strict_logits(greedy_targets: list[int]) -> torch.Tensor:
    greedy = torch.tensor(greedy_targets, dtype=torch.long)
    logits = torch.full((1, greedy.numel(), int(greedy.max().item()) + 1), -1000.0)
    logits[0, torch.arange(greedy.numel()), greedy] = 1000.0
    return logits


def _naive_first_sibling_path_accept(tree: DraftTree, greedy_targets: torch.Tensor):
    paths = _independent_paths(tree)
    tokens = tree.token_ids.tolist()
    greedy = greedy_targets.tolist()
    best_len = -1
    best_last = 0

    for path in paths:
        accepted = 0
        last = path[0]
        for parent, child in zip(path, path[1:]):
            if tokens[child] != greedy[parent]:
                break
            accepted += 1
            last = child
        if accepted > best_len:
            best_len = accepted
            best_last = last

    parents = tree.parent_indices.tolist()
    accepted_path = []
    cur = best_last
    while cur != -1:
        accepted_path.append(cur)
        cur = parents[cur] if cur != 0 else -1
    return list(reversed(accepted_path)), best_len, greedy[best_last]


def test_path_accept_uses_highest_index_duplicate_sibling_tie_break():
    tree = _tree([0, 7, 7, 8], [-1, 0, 0, 1])
    target_logits = _strict_logits([7, 8, 99, 100])
    node_greedy = target_logits.squeeze(0).argmax(dim=-1)
    plan = expand_tree_to_paths(tree)

    expected = tree_accept(tree, target_logits)
    actual = path_accept(plan, node_greedy[plan.node_of_row])
    naive = _naive_first_sibling_path_accept(tree, node_greedy)

    assert expected == ([0, 2], 1, 99)
    assert actual == expected
    assert naive == ([0, 1, 3], 2, 100)
