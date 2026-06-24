from collections import Counter

import pytest
import torch

from jetspec.tree._core.base import DraftTree

from bench.profiling import compare_conditioned_draft_logits as reseed_probe


def _logits(rows):
    return torch.tensor(rows, dtype=torch.float32)


def _row(vocab_size, ranking):
    row = torch.zeros(vocab_size, dtype=torch.float32)
    for score, token in enumerate(reversed(ranking), start=1):
        row[token] = float(score)
    return row


def _synthetic_tree() -> DraftTree:
    return DraftTree(
        token_ids=torch.tensor([99, 10, 11, 20, 21], dtype=torch.long),
        parent_indices=torch.tensor([-1, 0, 0, 1, 1], dtype=torch.long),
        depth=torch.tensor([0, 1, 1, 2, 2], dtype=torch.long),
        num_nodes=5,
        child_maps=[
            {10: 1, 11: 2},
            {20: 3, 21: 4},
            {},
            {},
            {},
        ],
    )


def test_score_rounds_and_summary_use_correction_depth_and_buckets():
    vocab = 12
    marginal = torch.stack([
        _row(vocab, [0, 1, 2, 3, 4, 5, 6]),
        _row(vocab, [1, 2, 3, 4, 5, 6, 7]),
        _row(vocab, [1, 2, 3, 5, 6, 7, 8, 4]),
        _row(vocab, [3, 4, 5, 6, 7, 8, 9]),
        _row(vocab, [4, 5, 6, 7, 8, 9, 10]),
        _row(vocab, [5, 6, 7, 8, 9, 10, 11]),
        _row(vocab, [6, 7, 8, 9, 10, 11, 0]),
        _row(vocab, [7, 8, 9, 10, 11, 0, 1]),
        _row(vocab, [8, 9, 10, 11, 0, 1, 2]),
        _row(vocab, [9, 10, 11, 0, 1, 2, 3]),
        _row(vocab, [10, 11, 0, 1, 2, 3, 4]),
    ])
    conditioned = marginal.clone()
    conditioned[2] = _row(vocab, [4, 1, 2, 3, 5, 6, 7])
    conditioned[5] = _row(vocab, [5, 6, 7, 8, 9, 10, 3])
    conditioned[10] = _row(vocab, [10, 11, 0, 1, 2, 3, 5, 4])

    scores = [
        reseed_probe.score_round_logits(marginal, conditioned, [10, 20], 4),
        reseed_probe.score_round_logits(marginal, conditioned, [1, 2, 3, 4, 5], 3),
        reseed_probe.score_round_logits(marginal, conditioned, list(range(10)), 4),
    ]
    summary = reseed_probe.summarise_scores(scores, Counter({"unreconstructable": 1}))

    assert scores[0].cond_hit is True
    assert scores[0].marg_topk_hit is False
    assert scores[1].cond_hit is False
    assert scores[1].cond_topk_hit is True
    assert summary["overall"] == {
        "n": 3,
        "cond_top1": pytest.approx(1 / 3),
        "marg_top1": 0.0,
        "cond_top7": pytest.approx(2 / 3),
        "marg_top7": pytest.approx(1 / 3),
    }
    assert summary["by_bucket"]["shallow"]["n"] == 1
    assert summary["by_bucket"]["mid"]["n"] == 1
    assert summary["by_bucket"]["deep"]["n"] == 1
    assert summary["skipped"] == {"unreconstructable": 1}
    assert summary["verdict"] == "P3_BUILD"


def test_l0_round_scores_first_depth_without_feeding_correction():
    marginal = _logits([
        [0.0, 5.0, 1.0],
        [0.0, 0.0, 0.0],
    ])
    conditioned = _logits([
        [0.0, 1.0, 7.0],
        [0.0, 0.0, 0.0],
    ])

    score = reseed_probe.score_round_logits(marginal, conditioned, [], 2)

    assert score.accepted_length == 0
    assert score.cond_hit is True
    assert score.marg_hit is False


def test_conditioned_block_ids_mask_beyond_l_and_never_include_correction():
    class DummyFwd:
        block_size = 5
        mask_token_id = 99
        device = torch.device("cpu")

    context_ids = torch.tensor([[7, 8, 9]], dtype=torch.long)

    out = reseed_probe._conditioned_block_output_ids(DummyFwd(), context_ids, [10, 20])

    assert out.tolist() == [[9, 10, 20, 99, 99]]


def test_reconstruct_rounds_skips_unreconstructable_round(monkeypatch):
    tree = _synthetic_tree()
    monkeypatch.setattr(reseed_probe, "rebuild_recorded_tree", lambda *args, **kwargs: tree)

    rounds, skipped = reseed_probe.reconstruct_rounds(
        records=[{"root_token": 99, "draft_logits": torch.zeros(1, 2, 32)}],
        token_ids=[1, 10, 999, 42],
        accept_lengths=[3],
        block_size=3,
        tree_width=2,
        budget=5,
    )

    assert rounds == []
    assert skipped == Counter({"unreconstructable": 1})


def test_reconstruct_rounds_recovers_l0_correction(monkeypatch):
    tree = _synthetic_tree()
    monkeypatch.setattr(reseed_probe, "rebuild_recorded_tree", lambda *args, **kwargs: tree)

    rounds, skipped = reseed_probe.reconstruct_rounds(
        records=[{"root_token": 99, "draft_logits": torch.zeros(1, 2, 32)}],
        token_ids=[1, 42],
        accept_lengths=[1],
        block_size=3,
        tree_width=2,
        budget=5,
    )

    assert skipped == Counter()
    assert len(rounds) == 1
    assert rounds[0].accepted_tokens == []
    assert rounds[0].accepted_path == [0]
    assert rounds[0].correction_token == 42
