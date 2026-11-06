"""Export decoder candidates and inputs from one offline read."""
from __future__ import annotations

import json
import math
from pathlib import Path


def export_candidates(output_json, metrics):
    """Keep only verified measured rounds; refuse missing or incomplete captures."""
    decisions = {(row['seq_id'], row['candidate_round']): row
                 for row in metrics['jetspec_round_records']}
    seen = set()
    records = 0
    sidecar = Path(str(output_json) + '.candidates.jsonl')
    with sidecar.open('w') as output:
        for source in sorted(Path(str(output_json) + '.rows').glob('*.jsonl')):
            with source.open() as lines:
                for line in lines:
                    row = json.loads(line)
                    key = row['seq_id'], row['candidate_round']
                    if key not in decisions:
                        continue
                    if row.get('schema') != 'jetspec.eval_round_candidates/1':
                        raise ValueError(f'eval capture lacks candidate schema: {key}')
                    for name in ('draft_top16', 'published_draft_top16'):
                        top = row[name]
                        ids, probabilities = top['token_ids'], top['probabilities']
                        if len(ids) != len(probabilities):
                            raise ValueError(f'eval capture has invalid {name} shape: {key}')
                        if name == 'published_draft_top16':
                            if any(len(one) != len(row['p_top1']) or len(one) != len(other)
                                   for one, other in zip(ids, probabilities)):
                                raise ValueError(f'eval capture has invalid {name} slots: {key}')
                            pairs = [(a, b) for one, other in zip(ids, probabilities)
                                     for a, b in zip(one, other)]
                        else:
                            if len(ids) != len(row['p_top1']):
                                raise ValueError(f'eval capture has invalid {name} slots: {key}')
                            pairs = list(zip(ids, probabilities))
                        if not pairs or any(len(a) != 16 or len(b) != 16 or
                                            any(not math.isfinite(p) or not 0 <= p <= 1 for p in b)
                                            for a, b in pairs):
                            raise ValueError(f'eval capture has invalid {name}: {key}')
                    for name in ('p_top1', 'q_shadow', 'shadow_argmax', 'offers', 'published_keys'):
                        if name not in row:
                            raise ValueError(f'eval capture lacks {name}: {key}')
                    for name in ('draft_token_ids', 'draft_token_probabilities', 'draft_pinned',
                                 'published_draft_token_probabilities', 'published_draft_pinned'):
                        if name not in row:
                            raise ValueError(f'eval capture lacks {name}: {key}')
                    width = len(row['p_top1'])
                    if len(row['draft_token_ids']) != width:
                        raise ValueError(f'eval capture has invalid draft_token_ids: {key}')
                    for prefix in ('', 'published_'):
                        probabilities = row[prefix+'draft_token_probabilities']
                        pinned = row[prefix+'draft_pinned']
                        if prefix:
                            if len(probabilities) != len(row['published_draft_top16']['token_ids']):
                                raise ValueError(f'eval capture has invalid published rows: {key}')
                        else:
                            probabilities, pinned = [probabilities], [pinned]
                        if (len(probabilities) != len(pinned) or
                                any(len(values) != width or len(flags) != width or
                                    any(not math.isfinite(p) or not 0 <= p <= 1 for p in values) or
                                    any(type(flag) is not bool for flag in flags)
                                    for values, flags in zip(probabilities, pinned))):
                            raise ValueError(f'eval capture has invalid {prefix}draft_token values: {key}')
                    from jetspec.engines.jetspec_plus_plus.runtime.pass_labels import normalized_label
                    if not normalized_label(row['pass_label']):
                        raise ValueError(f'eval capture lacks derived pass label: {key}')
                    if row['record_kind'] != 'redraft':
                        if key in seen:
                            raise ValueError(f'eval capture duplicates verified round: {key}')
                        seen.add(key)
                        row['verifier'] = dict(decisions[key])
                        # The verifier uses +inf for an unlimited wait budget.
                        # Preserve that sentinel without nonstandard JSON numbers.
                        if row['verifier'].get('wait_ms') == math.inf:
                            row['verifier']['wait_ms'] = 'inf'
                        row.setdefault('k_true', decisions[key]['k'])
                        row.setdefault('anchor_true', decisions[key]['correction'])
                    output.write(json.dumps(row, allow_nan=False) + '\n')
                    records += 1
    missing = decisions.keys() - seen
    if missing:
        raise ValueError(f'eval capture lacks verified rounds: {sorted(missing)[:8]}')
    return dict(schema='jetspec.eval_round_candidates/1', path=str(sidecar),
                verified_rounds=len(seen), records=records,
                fields=['draft_top16', 'published_draft_top16', 'draft_token_ids',
                        'draft_token_probabilities', 'draft_pinned',
                        'published_draft_token_probabilities', 'published_draft_pinned', 'pass_label'])


def export_decoder_inputs(output_json, metrics):
    """Move the decoder-input sidecar beside the read; refuse a verified round without its window."""
    import hashlib
    rows_dir = Path(str(output_json) + '.rows') / 'decoder_inputs'
    data, index = rows_dir / 'decoder_inputs.bin', rows_dir / 'decoder_inputs.jsonl'
    if not data.is_file() or not index.is_file():
        raise ValueError('eval decoder-input capture wrote no sidecar')
    target_data = Path(str(output_json) + '.decoder_inputs.bin')
    target_index = Path(str(output_json) + '.decoder_inputs.jsonl')
    data.replace(target_data)
    index.replace(target_index)
    verified = {(row['seq_id'], row['candidate_round']) for row in metrics['jetspec_round_records']}
    # Only proposals run the decoder; admissions and redrafts carry an exact correction.
    proposals = set()
    with Path(str(output_json) + '.candidates.jsonl').open() as lines:
        for line in lines:
            row = json.loads(line)
            if row['record_kind'] == 'proposal':
                proposals.add((row['seq_id'], row['candidate_round']))
    size = target_data.stat().st_size
    covered, rows, tap_rows = set(), 0, {}
    with target_index.open() as lines:
        for line in lines:
            row = json.loads(line)
            if row.get('schema') != 'jetspec.eval_decoder_inputs/1':
                raise ValueError('eval decoder-input index has an unknown schema')
            for offset, shape in ((row['window_offset'], row['window_shape']),
                                  (row['taps_offset'], row['taps_shape'])):
                if offset + 2 * math.prod(shape) > size:
                    raise ValueError(f'eval decoder-input index points past its data: {row}')
            seq = row['seq_id']
            if row['taps_from'] != tap_rows.get(seq, 0):
                raise ValueError(f'eval decoder-input taps are not contiguous: {row}')
            tap_rows[seq] = row['taps_to']
            covered.add((seq, row['candidate_round']))
            rows += 1
    missing = proposals - covered
    if missing:
        raise ValueError(f'eval decoder-input capture lacks verified proposals: {sorted(missing)[:8]}')
    digest = hashlib.sha256()
    with target_data.open('rb') as source:
        for block in iter(lambda: source.read(1 << 24), b''):
            digest.update(block)
    return dict(schema='jetspec.eval_decoder_inputs/1', path=str(target_data), index=str(target_index),
                rows=rows, verified_rounds=len(verified), verified_proposals=len(proposals),
                bytes=size, sha256=digest.hexdigest())
