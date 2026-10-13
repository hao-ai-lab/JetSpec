"""Compare candidate evidence by prompt/round/kind, naming every changed field."""
import json
import subprocess
import sys

import pytest



def tiny_capture(seq=9, prompt=0, r=0, kind='admission'):
    return dict(schema='jetspec.eval_round_candidates/1', seq_id=seq, candidate_round=r,
                record_kind=kind, pass_label='J1', draft_token_ids=[4, 5], p_top1=[0.2, 0.3],
                offers=None, published_keys=[[0, 4]], k_true=0, anchor_true=4,
                verifier=dict(seq_id=seq, prompt_index=prompt, accepted_len=1))


def test_capture_self_diff_and_sequence_normalization():
    from tests.jetspec_plus_plus.parity.diff_round_captures import compare_captures

    one = [tiny_capture(), tiny_capture(r=1, kind='proposal')]
    assert compare_captures(one, one)['equal']
    other = [tiny_capture(seq=42, r=1, kind='proposal'), tiny_capture(seq=42)]
    result = compare_captures(one, other)
    assert result['equal'] and result['records_equal'] == 2


def test_capture_diff_names_all_fields_at_first_differing_round(tmp_path):
    from tests.jetspec_plus_plus.parity.diff_round_captures import compare_captures

    one = [tiny_capture(r=r, kind='proposal') for r in range(3)]
    other = json.loads(json.dumps(one))
    other[1]['p_top1'][1] = 0.9
    other[1]['verifier']['accepted_len'] = 3
    other[2]['pass_label'] = 'J2'
    result = compare_captures(one, other)
    assert not result['equal']
    difference = result['prompts'][0]['first_difference']
    assert difference['round_index'] == 1
    fields = difference['records'][0]['fields']
    assert fields == {'p_top1': {'candidate': [0.2, 0.3], 'reference': [0.2, 0.9]},
                      'verifier.accepted_len': {'candidate': 1, 'reference': 3}}
    a, b, out = (tmp_path / name for name in ('a.jsonl', 'b.jsonl', 'diff.json'))
    for path, rows in ((a, one), (b, other)):
        path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    cmd = [sys.executable, 'tests/jetspec_plus_plus/parity/diff_round_captures.py', str(a),
           '--reference', str(b), '--output-json', str(out)]
    process = subprocess.run(cmd, text=True, capture_output=True)
    assert process.returncode == 1, process.stderr
    assert json.loads(out.read_text()) == result


def test_diff_reports_missing_redraft_and_missing_field_separately():
    from tests.jetspec_plus_plus.parity.diff_round_captures import compare_captures

    proposal = tiny_capture(r=1, kind='proposal')
    redraft = tiny_capture(r=1, kind='redraft')
    redraft.pop('verifier')  # Recover prompt index from the same sequence's proposal.
    other = dict(proposal)
    other.pop('offers')
    result = compare_captures([proposal, redraft], [other])
    differences = result['prompts'][0]['first_difference']['records']
    assert differences[0]['fields']['offers'] == {'candidate': None, 'reference': None,
                                                 'reference_missing': True}
    assert differences[1]['record_kind'] == 'redraft'
    assert differences[1]['fields']['$record']['reference_missing']


def test_diff_refuses_duplicate_or_unaligned_records():
    from tests.jetspec_plus_plus.parity.diff_round_captures import compare_captures

    row = tiny_capture()
    with pytest.raises(ValueError, match='duplicate'):
        compare_captures([row, row], [row])
    row.pop('verifier')
    with pytest.raises(ValueError, match='prompt index'):
        compare_captures([row], [row])
