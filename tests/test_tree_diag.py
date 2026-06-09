import pytest
import torch

from ptd.draft import RandomTreeDrafter
from tests.test_nano_tree import PROMPT, SP, _tiny_model, _tiny_nano


def test_tree_diag_metrics_formula_and_report_format():
    from bench.tree_diag import format_metrics_report, summarize_tree_diag

    accept_lengths = [1, 2, 4, 4]
    metrics = summarize_tree_diag(
        accept_lengths=accept_lengths,
        tree_nodes_per_depth=[4, 12, 8, 4],
        output_tokens=11,
        num_samples=2,
        block_size=4,
    )

    assert metrics["num_drafts"] == 4
    assert metrics["output_tokens"] == 11
    assert metrics["num_samples"] == 2
    assert metrics["tokens_per_sample"] == pytest.approx(5.5)
    assert metrics["acceptance_length"] == pytest.approx(sum(accept_lengths) / len(accept_lengths))
    assert metrics["acceptance_length_histogram"] == pytest.approx([0.25, 0.25, 0.0, 0.5])
    assert metrics["per_depth_acceptance_rate"] == pytest.approx([0.75, 0.5, 0.5])
    assert metrics["avg_tree_nodes_per_depth"] == pytest.approx([3.0, 2.0, 1.0])
    assert all(
        a >= b
        for a, b in zip(
            metrics["per_depth_acceptance_rate"],
            metrics["per_depth_acceptance_rate"][1:],
        )
    )

    report = format_metrics_report(
        metrics,
        attention_backend="triton_paged_tree_cudagraph",
        block_size=4,
        tree_width=7,
        budget=127,
        algo="crossproduct",
        drafter="eager",
    )
    assert "acceptance_length=2.750000" in report
    assert "per_depth_acceptance_rate=0.750000,0.500000,0.500000" in report
    assert "acceptance_length_histogram=0.250000,0.250000,0.000000,0.500000" in report
    assert "avg_tree_nodes_per_depth=3.00,2.00,1.00" in report


def test_generate_tree_diag_flag_preserves_tokens_and_counts_tree_depths():
    model = _tiny_model(0)
    drafter = RandomTreeDrafter(vocab_size=model.config.vocab_size)

    torch.manual_seed(1)
    plain = _tiny_nano(model).generate_tree(
        PROMPT,
        drafter,
        block_size=4,
        tree_width=2,
        budget=15,
        sampling_params=SP,
        return_stats=True,
    )

    torch.manual_seed(1)
    diag = _tiny_nano(model).generate_tree(
        PROMPT,
        drafter,
        block_size=4,
        tree_width=2,
        budget=15,
        sampling_params=SP,
        return_stats=True,
        tree_diag=True,
    )

    assert "tree_nodes_per_depth" not in plain
    assert diag["token_ids"] == plain["token_ids"]
    assert diag["accept_lengths"] == plain["accept_lengths"]
    assert diag["tree_sizes"] == plain["tree_sizes"]
    assert len(diag["tree_nodes_per_depth"]) == 4
    assert diag["tree_nodes_per_depth"][0] == diag["rounds"]
    assert sum(diag["tree_nodes_per_depth"]) == sum(diag["tree_sizes"])
