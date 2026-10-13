"""Read comparisons gate tokens/rounds and reproduce the golden paired bootstrap."""
from golden_values import golden
import copy
import json
import subprocess
import sys

import pytest

def pair():
    candidate = {'outputs': [
        {'prompt_index': 0, 'token_ids': [4, 5, 6, 7], 'accept_lengths': [2, 1], 'passes_run': [2, 2]},
        {'prompt_index': 1, 'token_ids': [8, 9, 10], 'accept_lengths': [2], 'passes_run': [4]}]}
    reference = {'outputs': [{'token_ids': [4, 5, 6, 7]}, {'token_ids': [8, 9, 10]}],
                 'metrics': {'jetspec_round_records': [
                     {'prompt_index': 0, 'accepted_len': 2, 'pass_label': 'J1'},
                     {'prompt_index': 0, 'accepted_len': 1, 'pass_label': 'J1'},
                     {'prompt_index': 1, 'accepted_len': 4, 'pass_label': 'J3'}]}}
    return candidate, reference


def test_token_round_and_tau_tools_match_golden(tmp_path, monkeypatch):
    from tests.jetspec_plus_plus.parity.compare_reads import compare_reads, main
    from tests.jetspec_plus_plus.parity.compare_rounds import compare_rounds
    candidate, reference = pair()
    expected = golden('value_1')
    actual = compare_reads(candidate, reference, mode='tau')
    assert actual['bootstrap'] == expected
    assert actual['tokens']['equal']
    assert actual['tokens']['diverged_share'] == 0
    rounds = compare_rounds(candidate, reference)
    assert rounds['equal']
    assert rounds['fields']['accepted_len']['rounds_compared'] == 3
    assert rounds['fields']['pass_label']['rounds_compared'] == 3
    corrupted = copy.deepcopy(reference)
    corrupted['outputs'][1]['token_ids'][1] = 11
    different = compare_reads(candidate, corrupted)
    assert different['tokens']['prompts'][1]['first_differing_index'] == 1
    assert different['tokens']['diverged_share'] == .5
    corrupted = copy.deepcopy(reference)
    corrupted['metrics']['jetspec_round_records'][1]['pass_label'] = 'J2'
    mismatch = compare_rounds(candidate, corrupted)['fields']['pass_label']['first_difference']
    assert mismatch == {'prompt_index': 0, 'round_index': 1, 'candidate': 'J1', 'reference': 'J2'}
    a, b, output = (tmp_path / name for name in ('a.json', 'b.json', 'report.json'))
    a.write_text(json.dumps(candidate))
    b.write_text(json.dumps(corrupted))
    assert main([str(a), '--reference', str(b), '--output-json', str(output)]) == 0
    corrupted['outputs'][0]['token_ids'].append(12)
    b.write_text(json.dumps(corrupted))
    assert main([str(a), '--reference', str(b), '--output-json', str(output)]) == 1


def test_bootstrap_nonzero_pair_is_exact(monkeypatch):
    from tests.jetspec_plus_plus.parity.compare_reads import compare_reads
    candidate, reference = pair()
    reference['metrics']['jetspec_round_records'] = [
        {'prompt_index': 0, 'accepted_len': 1} for _ in range(3)] + [
        {'prompt_index': 1, 'accepted_len': 1} for _ in range(2)]
    assert compare_reads(candidate, reference, mode='tau')['bootstrap'] == golden('value_2')


def test_comparisons_reject_unpaired_and_incomplete_reads():
    from tests.jetspec_plus_plus.parity.compare_reads import compare_reads
    from tests.jetspec_plus_plus.parity.compare_rounds import compare_rounds
    candidate, reference = pair()
    reference['outputs'].pop()
    with pytest.raises(ValueError, match='prompt'):
        compare_reads(candidate, reference)
    candidate, reference = pair()
    reference['metrics']['jetspec_round_records'].pop()
    with pytest.raises(ValueError, match='cover'):
        compare_rounds(candidate, reference)
    candidate, reference = pair()
    reference['outputs'][0]['token_ids'][0] = 99
    assert not compare_rounds(candidate, reference)['equal']


def test_round_tool_names_first_length_difference_and_exit(tmp_path):
    from tests.jetspec_plus_plus.parity.compare_rounds import main
    candidate, reference = pair()
    reference['metrics']['jetspec_round_records'][0]['accepted_len'] = 1
    reference['metrics']['jetspec_round_records'][1]['accepted_len'] = 2
    a, b, report = (tmp_path / name for name in ('a.json', 'b.json', 'report.json'))
    a.write_text(json.dumps(candidate))
    b.write_text(json.dumps(reference))
    assert main([str(a), '--reference', str(b), '--output-json', str(report)]) == 1
    first = json.loads(report.read_text())['fields']['accepted_len']['first_difference']
    assert first == {'prompt_index': 0, 'round_index': 0, 'candidate': 2, 'reference': 1}


def test_comparisons_check_engine_manifest_and_fp32_tier():
    from tests.jetspec_plus_plus.parity.compare_reads import compare_reads
    from tests.jetspec_plus_plus.parity.compare_rounds import compare_rounds
    candidate, reference = pair()
    candidate['manifest'] = dict(prompts_sha256='file-a', prompt_inputs_sha256='bank-a',
                                 max_new_tokens=4, dtype='fp32', temperature=0)
    reference['run_manifest'] = dict(values=dict(prompts_sha256='bank-b', args={'temp': 0}),
                                     env={'JETSPEC_TARGET_DTYPE': 'fp32'})
    reference['max_new'] = 4
    with pytest.raises(ValueError, match='prompt_inputs_sha256'):
        compare_reads(candidate, reference)
    reference['run_manifest']['values']['prompts_sha256'] = 'bank-a'
    # Same selected inputs may come from JSON banks with different formatting.
    reference['run_manifest']['files'] = [{'role': 'prompts', 'sha256': 'file-b'}]
    assert compare_rounds(candidate, reference)['equal']
    candidate['manifest']['dtype'] = 'bf16'
    reference['run_manifest']['env']['JETSPEC_TARGET_DTYPE'] = 'bf16'
    with pytest.raises(ValueError, match='fp32'):
        compare_rounds(candidate, reference)


@pytest.mark.parametrize('script', ['compare_reads', 'compare_rounds'])
def test_comparison_help(script):
    result = subprocess.run([sys.executable, f'tests/jetspec_plus_plus/parity/{script}.py', '--help'],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
