"""Diff reference-schema candidate JSONL by prompt index, round and record kind.

Sequence IDs are process-local: verifier.prompt_index (or a top-level
prompt_index) supplies the alignment. Each prompt's first differing round
reports every changed field, including missing publications or redrafts.
Float values are compared exactly; this is diagnostic evidence, not a tolerance gate.
"""
import argparse
import json
from pathlib import Path

SCHEMA = 'jetspec.eval_round_candidates/1'
MISSING = object()


def index_records(records):
    prompts = {}
    for row in records:
        prompt = row.get('prompt_index', row.get('verifier', {}).get('prompt_index'))
        if prompt is not None:
            seq = row['seq_id']
            if seq in prompts and prompts[seq] != prompt:
                raise ValueError(f'sequence {seq} has inconsistent prompt indices')
            prompts[seq] = prompt
    indexed = {}
    for row in records:
        if row.get('schema') != SCHEMA:
            raise ValueError('unknown candidate capture schema')
        seq, r, kind = row['seq_id'], row['candidate_round'], row['record_kind']
        if seq not in prompts:
            raise ValueError(f'sequence {seq} lacks a prompt index for alignment')
        prompt = prompts[seq]
        if type(prompt) is not int or type(r) is not int or r < 0:
            raise ValueError('prompt index and candidate round must be integers; round must be nonnegative')
        if kind not in ('admission', 'proposal', 'redraft'):
            raise ValueError(f'unknown record kind {kind!r}')
        key = prompt, r, kind
        if key in indexed:
            raise ValueError(f'duplicate capture record {key}')
        normalized = dict(row, seq_id=prompt)
        normalized.pop('prompt_index', None)
        if 'verifier' in row:
            normalized['verifier'] = dict(row['verifier'], seq_id=prompt)
        indexed[key] = normalized
    return indexed


def field_differences(candidate, reference, prefix=''):
    if isinstance(candidate, dict) and isinstance(reference, dict):
        fields = {}
        for name in sorted(candidate.keys() | reference.keys()):
            path = f'{prefix}.{name}' if prefix else name
            fields.update(field_differences(candidate.get(name, MISSING), reference.get(name, MISSING), path))
        return fields
    if candidate == reference:
        return {}
    values = dict(candidate=None if candidate is MISSING else candidate,
                  reference=None if reference is MISSING else reference)
    if candidate is MISSING:
        values['candidate_missing'] = True
    if reference is MISSING:
        values['reference_missing'] = True
    return {prefix or '$record': values}


def compare_captures(candidate, reference):
    a, b = index_records(candidate), index_records(reference)
    keys = sorted(a.keys() | b.keys())
    prompts, equal_count = {}, 0
    for prompt, r, kind in keys:
        summary = prompts.setdefault(prompt, dict(prompt_index=prompt, equal=True, first_difference=None))
        fields = field_differences(a.get((prompt, r, kind), MISSING), b.get((prompt, r, kind), MISSING))
        if not fields:
            equal_count += 1
            continue
        summary['equal'] = False
        if summary['first_difference'] is None:
            summary['first_difference'] = dict(round_index=r, records=[])
        if summary['first_difference']['round_index'] == r:
            summary['first_difference']['records'].append(dict(record_kind=kind, fields=fields))
    return dict(schema='jetspec.round_capture_comparison/1', equal=equal_count == len(keys),
                records_compared=len(keys), records_equal=equal_count, prompts=list(prompts.values()),
                alignment='verifier.prompt_index or prompt_index, candidate_round, record_kind; sequence IDs normalized')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('candidate', type=Path)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output-json', type=Path, help='write report here; otherwise print to stdout')
    args = parser.parse_args(argv)
    records = [[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
               for path in (args.candidate, args.reference)]
    result = compare_captures(*records)
    report = json.dumps(result, indent=2, allow_nan=False) + '\n'
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(report, encoding='utf-8')
    else:
        print(report, end='')
    return int(not result['equal'])


if __name__ == '__main__':
    raise SystemExit(main())
