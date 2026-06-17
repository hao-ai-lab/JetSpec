import torch

from jetflow.tree import build_ancestor_matrix, should_extend, splice_extension, tree_accept
from jetflow.tree._core.base import DraftTree


DEV = torch.device("cpu")


def _tree(token_ids, parent_indices, cum_logprob=True):
    parents = torch.tensor(parent_indices, dtype=torch.long, device=DEV)
    depths = torch.zeros(len(parent_indices), dtype=torch.long, device=DEV)
    for idx in range(1, len(parent_indices)):
        depths[idx] = depths[parents[idx]] + 1
    cum_lp = None
    if cum_logprob:
        cum_lp = torch.arange(len(token_ids), dtype=torch.float32, device=DEV) * -0.1
    return DraftTree(
        token_ids=torch.tensor(token_ids, dtype=torch.long, device=DEV),
        parent_indices=parents,
        depth=depths,
        num_nodes=len(token_ids),
        cum_logprob=cum_lp,
    )


def _logits_with_rank1_tokens(tokens, vocab_size=128):
    logits = torch.full((1, len(tokens), vocab_size), -1000.0, device=DEV)
    for depth, token in enumerate(tokens):
        logits[0, depth, token] = 1000.0
        logits[0, depth, (token + 1) % vocab_size] = 999.0
    return logits


def _target_logits(greedy_targets, vocab_size=128):
    logits = torch.full((1, len(greedy_targets), vocab_size), -1000.0, device=DEV)
    rows = torch.arange(len(greedy_targets), device=DEV)
    logits[0, rows, torch.tensor(greedy_targets, dtype=torch.long, device=DEV)] = 1000.0
    return logits


def test_should_extend_fires_on_mean_best_path_gap():
    topk_lp = torch.tensor(
        [
            [-0.1, -1.6, -2.5],
            [-0.2, -1.2, -3.0],
            [-0.3, -1.8, -4.0],
        ],
        dtype=torch.float32,
    )

    assert should_extend(topk_lp, [0, 0, 0], gap_threshold=1.3)
    assert should_extend(topk_lp, torch.tensor([0, 0, 0]), gap_threshold=1.3)
    assert not should_extend(topk_lp, [0, 0, 0], gap_threshold=1.5)
    assert not should_extend(topk_lp, [0, 1, 0], gap_threshold=0.7)


def test_splice_extension_zero_budget_returns_same_tree():
    tree = _tree([99, 10], [-1, 0])

    assert splice_extension(
        tree,
        leaf_index=1,
        ext_logits=_logits_with_rank1_tokens([30]),
        ext_budget=0,
        tree_width=2,
    ) is tree


def test_splice_extension_chain_preserves_valid_tree_and_ancestors():
    tree = _tree([99, 10, 11, 20, 21], [-1, 0, 0, 1, 1])
    spliced = splice_extension(
        tree,
        leaf_index=3,
        ext_logits=_logits_with_rank1_tokens([30, 40, 50]),
        ext_budget=2,
        tree_width=2,
        mode="chain",
    )

    assert spliced.num_nodes == tree.num_nodes + 2
    assert spliced.token_ids.tolist() == [99, 10, 11, 20, 21, 30, 40]
    assert spliced.parent_indices.tolist() == [-1, 0, 0, 1, 1, 3, 5]
    assert spliced.depth.tolist() == [0, 1, 1, 2, 2, 3, 4]
    assert spliced.token_ids.dtype == tree.token_ids.dtype
    assert spliced.parent_indices.device == tree.parent_indices.device

    for child in range(1, spliced.num_nodes):
        parent = int(spliced.parent_indices[child])
        assert parent < child
        assert int(spliced.depth[child]) == int(spliced.depth[parent]) + 1

    ancestor = build_ancestor_matrix(spliced).bool()
    assert ancestor[6, 0]
    assert ancestor[6, 1]
    assert ancestor[6, 3]
    assert ancestor[6, 5]
    assert not ancestor[6, 4]


def test_splice_extension_subtree_uses_top2gap_heap_shape():
    tree = _tree([99, 10, 11, 20, 21], [-1, 0, 0, 1, 1])
    ext_logits = torch.zeros((1, 4, 16), dtype=torch.float32, device=DEV)
    spliced = splice_extension(
        tree,
        leaf_index=3,
        ext_logits=ext_logits,
        ext_budget=4,
        tree_width=4,
        mode="subtree",
    )

    assert spliced.num_nodes == tree.num_nodes + 4
    direct_children = [
        idx
        for idx in range(tree.num_nodes, spliced.num_nodes)
        if int(spliced.parent_indices[idx]) == 3
    ]
    assert len(direct_children) == 3
    for child in range(1, spliced.num_nodes):
        parent = int(spliced.parent_indices[child])
        assert parent < child
        assert int(spliced.depth[child]) == int(spliced.depth[parent]) + 1
    build_ancestor_matrix(spliced)


def test_splice_extension_accepts_through_splice_with_cpu_oracle():
    tree = _tree([99, 10, 11, 20, 21], [-1, 0, 0, 1, 1])
    spliced = splice_extension(
        tree,
        leaf_index=3,
        ext_logits=_logits_with_rank1_tokens([30, 40]),
        ext_budget=2,
        tree_width=2,
        mode="chain",
    )
    greedy_targets = [10, 20, 90, 30, 91, 40, 77]

    accepted_path, accepted_len, correction = tree_accept(
        spliced,
        _target_logits(greedy_targets),
    )

    assert accepted_path == [0, 1, 3, 5, 6]
    assert accepted_len == 4
    assert correction == 77
