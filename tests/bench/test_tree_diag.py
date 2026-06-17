import pytest
import torch

from jetflow.draft import RandomTreeDrafter
from tests.inference_engine.test_jetflow_tree import PROMPT, SP, _tiny_model, _tiny_jetflow


class _DeterministicTreeDrafter:
    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size
        self.calls = []

    def propose_logits(self, context_ids, depth, target_hidden=None, **kwargs):
        call_index = len(self.calls)
        logits = torch.arange(
            depth * self.vocab_size,
            dtype=torch.float32,
            device=context_ids.device,
        ).view(1, depth, self.vocab_size)
        logits = logits + (call_index * 100.0)
        self.calls.append(logits.detach().cpu().clone())
        return logits


def test_tree_diag_metrics_formula_and_report_format():
    from bench.debug.tree_diag import format_metrics_report, summarize_tree_diag

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
        algo="accum_logp",
        drafter="eager",
    )
    assert "acceptance_length=2.750000" in report
    assert "per_depth_acceptance_rate=0.750000,0.500000,0.500000" in report
    assert "acceptance_length_histogram=0.250000,0.250000,0.000000,0.500000" in report
    assert "avg_tree_nodes_per_depth=3.00,2.00,1.00" in report


def test_tree_diag_summary_accepts_deeper_than_block_size_when_max_depth_set():
    from bench.debug.tree_diag import summarize_tree_diag

    metrics = summarize_tree_diag(
        accept_lengths=[1, 5, 6],
        tree_nodes_per_depth=[3, 3, 3, 3, 3, 3],
        output_tokens=12,
        num_samples=1,
        block_size=4,
        max_depth=6,
    )

    assert metrics["acceptance_length_histogram"] == pytest.approx([
        1 / 3,
        0.0,
        0.0,
        0.0,
        1 / 3,
        1 / 3,
    ])
    assert metrics["per_depth_acceptance_rate"] == pytest.approx([
        2 / 3,
        2 / 3,
        2 / 3,
        2 / 3,
        1 / 3,
    ])
    assert metrics["avg_tree_nodes_per_depth"] == pytest.approx([1.0] * 5)


def test_generate_tree_diag_flag_preserves_tokens_and_counts_tree_depths():
    model = _tiny_model(0)
    drafter = RandomTreeDrafter(vocab_size=model.config.vocab_size)

    torch.manual_seed(1)
    plain = _tiny_jetflow(model).generate_tree(
        PROMPT,
        drafter,
        block_size=4,
        tree_width=2,
        budget=15,
        sampling_params=SP,
        return_stats=True,
    )

    torch.manual_seed(1)
    diag = _tiny_jetflow(model).generate_tree(
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


def test_dump_first_rounds_records_drafter_topk_without_changing_metrics():
    from bench.debug.tree_diag import run_tree_diag_measurement

    model = _tiny_model(0)
    tree_kwargs = dict(
        block_size=3,
        tree_width=2,
        budget=7,
        sampling_params=SP,
        return_stats=True,
        tree_diag=True,
    )

    plain_drafter = _DeterministicTreeDrafter(model.config.vocab_size)
    plain_metrics, plain_dump = run_tree_diag_measurement(
        _tiny_jetflow(model),
        [PROMPT],
        plain_drafter,
        tree_kwargs,
        block_size=3,
        tree_width=2,
        dump_first_rounds=0,
    )

    dump_drafter = _DeterministicTreeDrafter(model.config.vocab_size)
    dump_metrics, dump_text = run_tree_diag_measurement(
        _tiny_jetflow(model),
        [PROMPT],
        dump_drafter,
        tree_kwargs,
        block_size=3,
        tree_width=2,
        dump_first_rounds=2,
    )

    assert plain_dump == ""
    assert dump_metrics == plain_metrics
    assert dump_text.count("[ROUND 0] root_token=") == 1
    assert dump_text.count("[ROUND 1] root_token=") == 1
    assert "[ROUND 2]" not in dump_text

    lines = dump_text.strip().splitlines()
    depth_count = dump_drafter.calls[0].shape[1]
    per_round_lines = 1 + depth_count + depth_count
    assert len(lines) == 2 * per_round_lines
    for round_index in range(2):
        base = round_index * per_round_lines
        assert lines[base].startswith(f"[ROUND {round_index}] root_token=")
        assert " accepted_len=" in lines[base]
        logprobs = torch.log_softmax(dump_drafter.calls[round_index], dim=-1)
        topk_lp, topk_tok = torch.topk(logprobs, 2, dim=-1)
        for depth in range(depth_count):
            tok_values = ",".join(str(int(v)) for v in topk_tok[0, depth].tolist())
            lp_values = ",".join(f"{float(v):.6f}" for v in topk_lp[0, depth].tolist())
            assert lines[base + 1 + depth] == (
                f"[ROUND {round_index}] topk_tok[{depth}]={tok_values}"
            )
            assert lines[base + 1 + depth_count + depth] == (
                f"[ROUND {round_index}] topk_lp[{depth}]={lp_values}"
            )
