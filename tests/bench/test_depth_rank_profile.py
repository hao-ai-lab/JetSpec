import json

import pytest
import torch

from jetspec.tree import get_algorithm
from jetspec.tree._core.base import DraftTree

from bench.profiling import depth_rank_profile as profile_mod
from bench.profiling.depth_rank_profile import (
    DepthRankProfileCounts,
    accepted_path_from_committed_tokens,
    accumulate_generation_profile,
    accumulate_round_profile,
    build_profile_table,
    fail_if_skip_rate_too_high,
)


DEV = torch.device("cpu")


def _synthetic_tree() -> DraftTree:
    return DraftTree(
        token_ids=torch.tensor([99, 10, 11, 20, 21, 30, 31], dtype=torch.long),
        parent_indices=torch.tensor([-1, 0, 0, 1, 1, 2, 2], dtype=torch.long),
        depth=torch.tensor([0, 1, 1, 2, 2, 2, 2], dtype=torch.long),
        num_nodes=7,
        child_maps=[
            {10: 1, 11: 2},
            {20: 3, 21: 4},
            {30: 5, 31: 6},
            {},
            {},
            {},
            {},
        ],
    )


def test_synthetic_tree_acceptance_profile_counts_rank_nodes_on_path():
    tree = _synthetic_tree()
    counts = DepthRankProfileCounts(depths=2, width=2)

    path = accepted_path_from_committed_tokens(tree, [10, 21])
    assert path == [0, 1, 4]

    accumulate_round_profile(counts, tree, path)
    table = build_profile_table(counts)

    assert table["depth_rank_accept"] == [
        [1.0, 0.0],
        [0.0, 0.5],
    ]
    assert table["meta"]["presence_counts"] == [
        [1, 1],
        [2, 2],
    ]
    assert table["meta"]["accepted_counts"] == [
        [1, 0],
        [0, 1],
    ]


def test_profile_schema_round_trips_into_depth_rank_histogram_caps():
    counts = DepthRankProfileCounts(depths=3, width=3)
    counts.presence = [
        [4, 4, 4],
        [5, 5, 5],
        [6, 6, 6],
    ]
    counts.accepted = [
        [4, 2, 0],
        [5, 0, 0],
        [6, 3, 3],
    ]
    profile = build_profile_table(counts)

    dense = torch.zeros(1, 3, 12)
    tree = get_algorithm("depth_rank_histogram", tau=0.5).build(
        7,
        dense,
        block_size=4,
        tree_width=3,
        budget=40,
        device=DEV,
        profile_table=json.loads(json.dumps(profile)),
    )

    per_depth_nodes = torch.bincount(tree.depth, minlength=4).tolist()
    assert profile["depth_rank_accept"] == [
        [1.0, 0.5, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.5, 0.5],
    ]
    assert per_depth_nodes[:4] == [1, 2, 2, 6]


def test_accepted_path_rejects_non_child_token_sequence():
    tree = _synthetic_tree()
    with pytest.raises(ValueError, match="not a child"):
        accepted_path_from_committed_tokens(tree, [10, 30])


def test_generation_profile_does_not_walk_round_correction(monkeypatch):
    tree = _synthetic_tree()
    monkeypatch.setattr(profile_mod, "rebuild_recorded_tree", lambda *args, **kwargs: tree)
    counts = DepthRankProfileCounts(depths=2, width=2)

    accumulate_generation_profile(
        counts,
        records=[{}],
        token_ids=[1, 10, 999],
        accept_lengths=[2],
        block_size=3,
        tree_width=2,
        budget=7,
    )

    assert counts.rounds_counted == 1
    assert counts.skipped_rounds == 0
    assert counts.accepted == [
        [1, 0],
        [0, 0],
    ]


def test_generation_profile_skips_eos_truncated_final_round(monkeypatch):
    tree = _synthetic_tree()
    monkeypatch.setattr(profile_mod, "rebuild_recorded_tree", lambda *args, **kwargs: tree)
    counts = DepthRankProfileCounts(depths=2, width=2)

    accumulate_generation_profile(
        counts,
        records=[{}],
        token_ids=[1, 10],
        accept_lengths=[3],
        block_size=3,
        tree_width=2,
        budget=7,
    )

    assert counts.rounds_counted == 0
    assert counts.skipped_rounds == 1


def test_generation_profile_skips_max_new_truncated_final_round(monkeypatch):
    tree = _synthetic_tree()
    monkeypatch.setattr(profile_mod, "rebuild_recorded_tree", lambda *args, **kwargs: tree)
    counts = DepthRankProfileCounts(depths=2, width=2)

    accumulate_generation_profile(
        counts,
        records=[{}, {}],
        token_ids=[1, 10, 999, 11],
        accept_lengths=[2, 3],
        block_size=3,
        tree_width=2,
        budget=7,
    )

    assert counts.rounds_counted == 1
    assert counts.skipped_rounds == 1
    assert counts.accepted == [
        [1, 0],
        [0, 0],
    ]


def test_generation_profile_skips_unreconstructable_round(monkeypatch):
    tree = _synthetic_tree()
    monkeypatch.setattr(profile_mod, "rebuild_recorded_tree", lambda *args, **kwargs: tree)
    counts = DepthRankProfileCounts(depths=2, width=2)

    accumulate_generation_profile(
        counts,
        records=[{}],
        token_ids=[1, 10, 999],
        accept_lengths=[3],
        block_size=3,
        tree_width=2,
        budget=7,
    )

    assert counts.rounds_counted == 0
    assert counts.skipped_rounds == 1


def test_skip_rate_gate_exits_above_five_percent(capsys):
    counts = DepthRankProfileCounts(depths=2, width=2)
    counts.rounds_counted = 94
    counts.skipped_rounds = 6

    with pytest.raises(SystemExit, match="> 5%"):
        fail_if_skip_rate_too_high(counts)

    assert "skipped_rounds=6/100 (6.00%)" in capsys.readouterr().out
