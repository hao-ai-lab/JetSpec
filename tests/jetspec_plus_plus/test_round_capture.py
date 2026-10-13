"""Candidate captures preserve reference publication and delayed verifier semantics."""
from golden_values import golden
import json
from pathlib import Path

import pytest

from test_round import components
from test_drafter import thresholds
from test_lossless import llm_for
from jetspec.methods.jetspec_plus_plus.drafter import JetSpecPPDrafter
from jetspec.methods.jetspec_plus_plus.reader import main


@pytest.mark.parametrize('schedule', ['J1-R1', 'J3-R3'])
def test_reader_captures_backend_schema_without_changing_tokens(tmp_path, monkeypatch, schedule):
    target, driver, taps = components(0)
    prompts = tmp_path / 'prompts.json'
    prompts.write_text(json.dumps([{'index': 19, 'token_ids': [1, 2, 3, 4, 5, 6]}]))
    output = tmp_path / 'read.json'

    def factory(**kwargs):
        return JetSpecPPDrafter(target=target, head=driver.head, adapter=driver.adapter,
                               decoder=driver.decoder, config=kwargs['config'], thresholds=thresholds())

    argv = ['--target', 'tiny', '--drafter', 'tiny', '--adapter', 'tiny', '--heads-bundle', 'tiny',
            '--prompts-file', str(prompts), '--max-new-tokens', '20', '--output', str(output),
            '--schedule', schedule]
    baseline = main(argv, drafter_factory=factory, llm_factory=lambda **kwargs: llm_for(target))
    actual = main(argv + ['--capture-rounds'], drafter_factory=factory,
                  llm_factory=lambda **kwargs: llm_for(target))
    assert baseline['outputs'] == actual['outputs']
    path = Path(str(output) + '.candidates.jsonl')
    records = [json.loads(line) for line in path.read_text().splitlines()]
    verified = [row for row in records if row['record_kind'] != 'redraft']
    assert [row['candidate_round'] for row in verified] == list(range(actual['summary']['rounds']))
    assert verified[0]['record_kind'] == 'admission'
    assert verified[0]['offers'] is None and verified[0]['p_wire'] is None
    assert verified[0]['published_keys'] == [[k, verified[0]['draft_token_ids'][0]] for k in range(16)]
    assert sum(row['verifier']['accepted_len'] for row in verified) == 19
    assert all(row['verifier']['prompt_index'] == 19 for row in verified)
    redrafts = [row for row in records if row['record_kind'] == 'redraft']
    assert redrafts  # The tiny target supplies deterministic misses.
    for row in redrafts:
        proposal = next(one for one in verified if one['candidate_round'] == row['candidate_round'])
        assert proposal['verifier']['decision'] == 'miss'
        assert proposal['verifier']['rows_published'] == 1
        assert proposal['verifier']['k_set'] == [row['k_true']]
        assert proposal['verifier']['anchor_hit'] and not proposal['verifier']['hit_at_f']
        assert row['published_keys'] == [[row['k_true'], row['anchor_true']]]
        assert row['offers'] is None and row['p_pred'] is None
        assert row['draft_token_ids'][0] == proposal['anchor_true']
        assert row['pass_label'] in ('J0d0', 'J1', 'J2', 'J3')
    for row in verified[1:]:
        previous = verified[row['candidate_round'] - 1]
        assert row['k_true'] == previous['verifier']['accepted_len'] - 1
        assert row['anchor_true'] == previous['verifier']['next_correction']
        assert row['offers'] and len(row['offers']['top_ids'][0]) == 32
    # The golden exporter validates the full public sidecar without adapting its schema.
    shards = Path(str(output) + '.rows')
    shards.mkdir()
    (shards / 'tiny.jsonl').write_text(path.read_text())
    monkeypatch.setenv('JETSPEC_DRAFTS', '15')
    exported = golden('export_contract')
    assert exported['verified_rounds'] == actual['summary']['rounds']
    assert exported['records'] == len(records)
    # Ported from the JetSpec++ reference implementation.
    import math
    for row in records:
        assert row['schema'] == exported['schema']
        assert set(exported['fields']) <= row.keys()
        assert {'p_top1', 'q_shadow', 'shadow_argmax', 'offers', 'published_keys'} <= row.keys()
        width = len(row['p_top1'])
        assert len(row['draft_token_ids']) == width
        for name in ('draft_top16', 'published_draft_top16'):
            top = row[name]
            ids, probabilities = top['token_ids'], top['probabilities']
            assert len(ids) == len(probabilities)
            if name == 'published_draft_top16':
                assert all(len(one) == width and len(one) == len(other)
                           for one, other in zip(ids, probabilities))
                pairs = [(a, b) for one, other in zip(ids, probabilities) for a, b in zip(one, other)]
            else:
                assert len(ids) == width
                pairs = list(zip(ids, probabilities))
            assert pairs
            assert all(len(a) == 16 and len(b) == 16 and
                       all(math.isfinite(p) and 0 <= p <= 1 for p in b) for a, b in pairs)
        for prefix in ('', 'published_'):
            probabilities, pinned = row[prefix+'draft_token_probabilities'], row[prefix+'draft_pinned']
            if prefix:
                assert len(probabilities) == len(row['published_draft_top16']['token_ids'])
            else:
                probabilities, pinned = [probabilities], [pinned]
            assert len(probabilities) == len(pinned)
            assert all(len(values) == width and len(flags) == width and
                       all(math.isfinite(p) and 0 <= p <= 1 for p in values) and
                       all(type(flag) is bool for flag in flags)
                       for values, flags in zip(probabilities, pinned))
        assert row['pass_label'] in ('J0d0', 'J1', 'J2', 'J3')


def test_prefill_only_capture_is_empty_and_ar_capture_is_refused(tmp_path):
    target, driver, _ = components(0)
    prompts = tmp_path / 'prompts.json'
    prompts.write_text(json.dumps([{'token_ids': [1, 2, 3, 4, 5, 6]}]))
    output = tmp_path / 'read.json'

    def factory(**kwargs):
        return JetSpecPPDrafter(target=target, head=driver.head, adapter=driver.adapter,
                               decoder=driver.decoder, config=kwargs['config'], thresholds=thresholds())

    argv = ['--target', 'tiny', '--drafter', 'tiny', '--adapter', 'tiny', '--heads-bundle', 'tiny',
            '--prompts-file', str(prompts), '--max-new-tokens', '1', '--output', str(output),
            '--capture-rounds']
    main(argv, drafter_factory=factory, llm_factory=lambda **kwargs: llm_for(target))
    assert Path(str(output) + '.candidates.jsonl').read_text() == ''
    with pytest.raises(SystemExit) as refusal:
        main(argv + ['--ar'], drafter_factory=factory)
    assert refusal.value.code == 2
