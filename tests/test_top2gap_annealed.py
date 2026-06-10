import torch

from ptd.tree import build_from_topk, get_algorithm, list_algorithms


DEV = torch.device("cpu")
BLOCK, WIDTH, BUDGET = 16, 7, 63
VOCAB = 200


def _dense_logits(seed: int) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(1, BLOCK - 1, VOCAB)


def _topk_from_dense(dense: torch.Tensor):
    log_probs = torch.log_softmax(dense.squeeze(0), dim=-1)
    topk_lp, topk_tok = torch.topk(log_probs, WIDTH, dim=-1)
    return topk_tok, topk_lp


def _assert_same_tree(a, b, msg: str) -> None:
    assert a.num_nodes == b.num_nodes, f"{msg}: num_nodes {a.num_nodes} != {b.num_nodes}"
    assert torch.equal(a.token_ids, b.token_ids), f"{msg}: token_ids differ"
    assert torch.equal(a.parent_indices, b.parent_indices), f"{msg}: parent_indices differ"
    assert torch.equal(a.depth, b.depth), f"{msg}: depth differ"
    assert torch.equal(a.cum_logprob, b.cum_logprob), f"{msg}: cum_logprob differ"


def test_identity_recovery_matches_plain_top2gap():
    dense = _dense_logits(20260610)
    root = 7

    plain = get_algorithm("top2gap_fanout", beta=2.0, g_0=1.0).build(
        root, dense, BLOCK, WIDTH, BUDGET, DEV
    )
    annealed = get_algorithm(
        "top2gap_annealed",
        beta=2.0,
        g_0=1.0,
        chain_until=0,
        tail_from=15,
    ).build(root, dense, BLOCK, WIDTH, BUDGET, DEV)

    _assert_same_tree(plain, annealed, "schedule-off top2gap")


def test_schedule_caps_match_fingerprint_shape():
    uniform = [0.0] * WIDTH
    huge_gap = [0.0] + [-100.0] * (WIDTH - 1)
    topk_logprobs = [list(uniform) for _ in range(BLOCK - 1)]
    topk_logprobs[5] = huge_gap

    caps = get_algorithm("top2gap_annealed").caps_from_topk(topk_logprobs, WIDTH)

    assert caps[:3] == [1, 1, 1]
    assert caps[5] == 1
    assert all(1 <= cap <= WIDTH for cap in caps)
    assert all(cap <= 2 for cap in caps[9:])
    assert caps[9:] == [2] * 6


def test_all_chain_caps_terminate_when_budget_exceeds_depth():
    dense = _dense_logits(11)

    tree = get_algorithm(
        "top2gap_annealed",
        chain_until=BLOCK - 1,
        tail_from=0,
        tail_cap=0,
    ).build(7, dense, BLOCK, WIDTH, 127, DEV)

    assert tree.num_nodes == BLOCK
    assert torch.equal(tree.depth, torch.arange(BLOCK))


def test_build_from_topk_matches_dense_build():
    dense = _dense_logits(13)
    root = 7
    kwargs = {"beta": 2.0, "g_0": 1.0, "chain_until": 3, "tail_from": 9, "tail_cap": 2}

    dense_tree = get_algorithm("top2gap_annealed", **kwargs).build(
        root, dense, BLOCK, WIDTH, BUDGET, DEV
    )
    topk_tok, topk_lp = _topk_from_dense(dense)
    topk_tree = build_from_topk(
        "top2gap_annealed",
        root,
        topk_tok,
        topk_lp,
        BUDGET,
        DEV,
        algo_kwargs=kwargs,
        tree_width=WIDTH,
    )

    _assert_same_tree(dense_tree, topk_tree, "top2gap_annealed build_from_topk")


def test_registry_round_trip():
    assert "top2gap_annealed" in list_algorithms()
    assert get_algorithm("top2gap_annealed").name == "top2gap_annealed"
