"""Tau reads use emitted credit, golden token hashes, and paper rule-A cuts."""
from golden_values import golden
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from conftest import tiny_target
from test_hooks import make_llm
from jetspec.core import llm as core
from jetspec.methods.jetspec.draft import TargetEchoTreeDrafter


@pytest.mark.parametrize('seed', [0, 1, 2])
def test_reader_records_match_golden(seed):
    from jetspec.methods.jetspec_plus_plus import reader

    torch.manual_seed(seed)
    ids = torch.randint(0, 41, (20,)).tolist()
    lengths = torch.randint(1, 5, (6,)).tolist()
    assert reader.token_sha256(ids) == golden('value_1')
    actual = reader.prompt_record(seed, ids, lengths, [2] * 6, 0, True)
    expected = golden('value_2')
    assert actual == expected
    assert torch.equal(torch.tensor(actual['accept_lengths']), torch.tensor(expected['accept_lengths']))
    assert reader.summarize([actual]) == golden('value_3')
    assert reader.summarize([]) == golden('value_4')


@pytest.mark.parametrize('seed', [0, 1, 2])
def test_rule_a_reducer_matches_paper(seed, monkeypatch):
    """Port only credit/cuts; engine time attribution depends on served clocks.

    Synthetic positive clocks let the unmodified golden run on CPU. Compare its
    fp32 credit and round outputs, without inventing serving time for HF reads.
    """
    from jetspec.methods.jetspec_plus_plus.reader import reduce_read

    torch.manual_seed(seed)
    pattern = torch.randint(100, 200, (64,)).tolist()
    prefix = torch.randint(0, 40, (64 * (seed + 1),)).tolist()
    repeated = (prefix + pattern * 32)[:2048]
    rows = [{'token_ids': repeated}, {'token_ids': [5, 6, golden('value_5')[0], 7]},
            {'token_ids': torch.randint(0, 40, (128,)).tolist()}]
    # Every synthetic round emits 15, except its actual terminal remainder.
    records = []
    steps = []
    for index, row in enumerate(rows):
        left = len(row['token_ids']) - 1
        lengths = []
        steps.append(1.0)
        while left:
            accepted = min(15, left)
            lengths.append(accepted)
            records.append({'prompt_index': index, 'accepted_len': accepted})
            steps.append(1.0)
            left -= accepted
        row['accept_lengths'] = lengths
    raw = {'outputs': rows, 'metrics': {'target_step_times': steps,
           'jetspec_round_records': records, 'prefill_total_time': 3.0,
           'decode_total_time': len(records)}}
    for loop_cut in [False, True]:
        expected = golden('value_6')
        actual = reduce_read(raw, 'jetspecpp', loop_cut, eos_token_ids=golden('value_5'))
        for key in ['tokens', 'rounds', 'tau', 'cut_prompts', 'cut_onsets',
                    'prompt_tokens', 'prompt_rounds']:
            assert actual[key] == expected[key]
        assert torch.equal(torch.tensor(actual['prompt_tokens'], dtype=torch.float32),
                           torch.tensor(expected['prompt_tokens'], dtype=torch.float32))
    assert actual['cut_onsets'] == {0: len(prefix)}


@pytest.mark.parametrize('seed', [0, 1, 2])
def test_read_passes_runs_without_integrator(seed, monkeypatch):
    """Compare the golden read loop with the same injectable tiny torch drafter.

    The frozen reference's activate() refuses an already imported public jetspec package;
    point it at its unmodified llm.py loaded by filename for this CPU test.
    """
    from jetspec.methods.jetspec_plus_plus.reader import read_passes
    from types import SimpleNamespace


    class TinyDrafter(TargetEchoTreeDrafter):
        def set_prompt_index(self, index):
            self.prompt_index = index

        def reset_cache(self):
            self.round_results = []

        def propose_logits(self, *args, **kwargs):
            self.round_results.append(SimpleNamespace(passes_run=2, converged=False))
            return super().propose_logits(*args, **kwargs)

    target = tiny_target(seed)
    llm = make_llm(core, target)
    seen = []

    def factory(*, llm, config):
        seen.append(str(config.schedule))
        return TinyDrafter(llm.model)

    prompts = [{'index': index, 'token_ids': [index + 1, 2, 3]} for index in range(3)]
    rows = read_passes(llm, prompts, drafter_factory=factory, tokens=64)
    expected_rows = golden('value_7')
    assert seen == ['J1-R1']
    for prompt, row, expected_row in zip(prompts, rows, expected_rows):
        assert torch.equal(torch.tensor(row['token_ids']), torch.tensor(expected_row['output_ids']))
        assert torch.equal(torch.tensor(row['accept_lengths']), torch.tensor(expected_row['accept_lengths']))
        assert {key: value for key, value in row.items() if key != 'token_ids'} == {
            key: value for key, value in expected_row.items() if key != 'output_ids'}
        expected = llm.generate_tree(
            torch.tensor([prompt['token_ids']]), TargetEchoTreeDrafter(target),
            block_size=16, tree_width=1, budget=16, return_stats=True, return_text=False,
            sampling_params=core.SamplingParams(max_new_tokens=64))
        assert row['token_ids'] == expected['token_ids']
        assert row['accept_lengths'] == [16, 16, 16, 15]
        assert sum(row['accept_lengths']) == row['n_tokens'] - 1


def test_read_passes_stops_at_eos_and_handles_prefill_only():
    from jetspec.methods.jetspec_plus_plus.reader import read_passes

    target = tiny_target(3)
    llm = make_llm(core, target)
    factory = lambda **kwargs: TargetEchoTreeDrafter(kwargs['llm'].model)
    prompts = [{'token_ids': [1, 2, 3]}]
    baseline, = read_passes(llm, prompts, drafter_factory=factory, tokens=20)
    llm.eos_token_ids = {baseline['token_ids'][2]}
    stopped, = read_passes(llm, prompts, drafter_factory=factory, tokens=20)
    first_eos = baseline['token_ids'].index(baseline['token_ids'][2])
    assert stopped['token_ids'] == baseline['token_ids'][:first_eos + 1]
    assert sum(stopped['accept_lengths']) == stopped['n_tokens'] - 1
    one, = read_passes(llm, prompts, drafter_factory=factory, tokens=1)
    assert one['n_tokens'] == 1
    assert one['accept_lengths'] == []
    with pytest.raises(ValueError):
        read_passes(llm, prompts, drafter_factory=factory, tokens=0)


def test_cli_writes_read_and_manifest_with_factory(tmp_path):
    from jetspec.methods.jetspec_plus_plus.reader import main

    prompts = tmp_path / 'prompts.json'
    prompts.write_text(json.dumps([{'index': 19, 'token_ids': [1, 2, 3]}]))
    output = tmp_path / 'read.json'
    target = tiny_target(0)
    received = {}

    def factory(**kwargs):
        received.update(kwargs)
        return TargetEchoTreeDrafter(kwargs['llm'].model)

    read = main(['--target', 'tiny-target', '--drafter', 'tiny-drafter',
                 '--adapter', 'adapter.safetensors', '--heads-bundle', 'heads.json',
                 '--prompts-file', str(prompts), '--schedule', 'J3-R3',
                 '--max-new-tokens', '19', '--output', str(output)],
                drafter_factory=factory, llm_factory=lambda **kwargs: make_llm(core, target))
    assert json.loads(output.read_text()) == read
    manifest = json.loads(Path(str(output) + '.manifest.json').read_text())
    assert read['manifest'] == manifest
    assert manifest['schedule'] == 'J3-R3'
    assert manifest['temperature'] == 0
    assert manifest['drafter_revision'] == '43d3aa9d'
    assert manifest['prompts_sha256']
    assert received['drafter_checkpoint'] == 'tiny-drafter'
    assert received['adapter_artifact'] == 'adapter.safetensors'
    assert received['heads_bundle'] == 'heads.json'
    assert read['outputs'][0]['prompt_index'] == 19
    assert read['rule_a']['tokens'] == 18
    assert read['rule_a']['rounds'] == 2


def test_cli_help_and_lazy_imports():
    command = [sys.executable, '-m', 'jetspec.methods.jetspec_plus_plus.reader', '--help']
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    for option in ['--target', '--drafter', '--adapter', '--heads-bundle',
                   '--prompts-file', '--schedule', '--output', '--drafter-factory', '--capture-rounds']:
        assert option in result.stdout
    result = subprocess.run([sys.executable, '-c',
        'import sys; import jetspec; import jetspec.methods.jetspec.tree; '
        'assert "transformers" not in sys.modules; '
        'assert "jetspec.methods.jetspec.draft_head" not in sys.modules'], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
