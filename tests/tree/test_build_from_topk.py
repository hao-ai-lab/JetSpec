"""build_from_topk (the engine adapter entry) must be IDENTICAL to build() on
dense logits — when the top-k is extracted from log_softmax(dense_logits), the two
paths share caps_from_topk and the same heap, so the trees must match exactly.

This is what makes the vLLM-fork integration faithful: the fork hands us its
proposer's per-depth top-k, and we reproduce the same tree our HF engine builds
from dense logits. Pure CPU, no model.
"""
import heapq

import torch

from jetflow.tree import get_algorithm, build_from_topk

DEV = torch.device("cpu")
BLOCK, WIDTH, BUDGET = 16, 7, 63          # D = 15 depths, top-7, mid budget
V = 200
PROFILE = {"depth_rank_accept": [[0.9, 0.4, 0.2, 0.08, 0.03, 0.01, 0.0]] * (BLOCK - 1)}

CASES = [
    ("accum_logp", {}, {}),
    ("top2gap_fanout", {"beta": 2.0, "g_0": 1.0}, {}),
    ("depth_rank_histogram", {"tau": 0.05}, {"profile_table": PROFILE}),
]


def _dense_logits(seed):
    torch.manual_seed(seed)
    return torch.randn(1, BLOCK - 1, V)


def _topk_from_dense(dense):
    lp = torch.log_softmax(dense.squeeze(0), dim=-1)
    topk_lp, topk_tok = torch.topk(lp, WIDTH, dim=-1)
    return topk_tok, topk_lp


def _assert_same_tree(a, b, msg):
    assert a.num_nodes == b.num_nodes, f"{msg}: num_nodes {a.num_nodes} != {b.num_nodes}"
    assert torch.equal(a.token_ids, b.token_ids), f"{msg}: token_ids differ"
    assert torch.equal(a.parent_indices, b.parent_indices), f"{msg}: parent_indices differ"
    assert torch.equal(a.depth, b.depth), f"{msg}: depth differ"


def _reference_accum_logp_tree(root_token, topk_tokens_cpu, topk_logprobs_cpu, budget):
    """Independent pre-refactor oracle for the FIFO heap contract."""
    depth_count = len(topk_tokens_cpu)
    width = len(topk_tokens_cpu[0]) if depth_count else 0
    token_ids = [root_token]
    parent_indices = [-1]
    depths = [0]
    num_nodes = 1
    counter = 0
    heap = [(0.0, counter, 0)]

    while heap and num_nodes < budget:
        neg_cum_lp, _, node_idx = heapq.heappop(heap)
        depth = depths[node_idx]
        if depth >= depth_count:
            continue
        children_to_add = min(width, budget - num_nodes)
        for rank in range(children_to_add):
            child_token = topk_tokens_cpu[depth][rank]
            child_cum_lp = -neg_cum_lp + topk_logprobs_cpu[depth][rank]
            token_ids.append(child_token)
            parent_indices.append(node_idx)
            depths.append(depth + 1)
            counter += 1
            heapq.heappush(heap, (-child_cum_lp, counter, num_nodes))
            num_nodes += 1

    return token_ids, parent_indices, depths, num_nodes


def test_build_from_topk_matches_build_dense():
    """For every fork-comparison algo, build_from_topk(topk-from-dense) == build(dense)."""
    for seed, (name, init_kw, build_kw) in enumerate(CASES):
        dense = _dense_logits(seed)
        root = 7
        tree_dense = get_algorithm(name, **init_kw).build(
            root, dense, BLOCK, WIDTH, BUDGET, DEV, **build_kw)
        topk_tok, topk_lp = _topk_from_dense(dense)
        tree_topk = build_from_topk(
            name, root, topk_tok, topk_lp, BUDGET, DEV,
            algo_kwargs=init_kw, tree_width=WIDTH, **build_kw)
        _assert_same_tree(tree_dense, tree_topk, name)


def test_build_from_topk_accepts_list_inputs():
    """Engines may pass python lists (the fork uses .tolist()); same tree as tensors."""
    dense = _dense_logits(0)
    topk_tok, topk_lp = _topk_from_dense(dense)
    t_tensor = build_from_topk("accum_logp", 7, topk_tok, topk_lp, BUDGET, DEV, tree_width=WIDTH)
    t_list = build_from_topk("accum_logp", 7, topk_tok.tolist(), topk_lp.tolist(), BUDGET, DEV, tree_width=WIDTH)
    _assert_same_tree(t_tensor, t_list, "list-vs-tensor")


def test_build_from_topk_rejects_unsupported():
    """An algorithm without caps_from_topk errors clearly (not silently wrong)."""
    import pytest
    dense = _dense_logits(0)
    topk_tok, topk_lp = _topk_from_dense(dense)
    # task_router (semantic_aware) has no caps_from_topk yet -> explicit NotImplementedError
    with pytest.raises(NotImplementedError):
        build_from_topk("task_router", 7, topk_tok, topk_lp, BUDGET, DEV, tree_width=WIDTH)


def test_accum_logp_build_matches_reference_across_random_logits():
    """AccumLogP.build preserves the exact FIFO heap tree over budgets and widths."""
    root = 7
    block_size = 16
    vocab_size = 200

    for seed in range(20):
        torch.manual_seed(20260609 + seed)
        dense = torch.randn(1, block_size - 1, vocab_size)
        log_probs = torch.log_softmax(dense.squeeze(0), dim=-1)

        for budget in (15, 63, 127):
            for width in (2, 7):
                topk_lp, topk_tok = torch.topk(log_probs, width, dim=-1)
                exp_tokens, exp_parents, exp_depths, exp_num_nodes = _reference_accum_logp_tree(
                    root,
                    topk_tok.tolist(),
                    topk_lp.tolist(),
                    budget,
                )
                tree = get_algorithm("accum_logp").build(
                    root,
                    dense,
                    block_size,
                    width,
                    budget,
                    DEV,
                )
                msg = f"seed={seed} budget={budget} width={width}"
                assert tree.num_nodes == exp_num_nodes, msg
                assert torch.equal(tree.token_ids, torch.tensor(exp_tokens, dtype=torch.long)), msg
                assert torch.equal(tree.parent_indices, torch.tensor(exp_parents, dtype=torch.long)), msg
                assert torch.equal(tree.depth, torch.tensor(exp_depths, dtype=torch.long)), msg
