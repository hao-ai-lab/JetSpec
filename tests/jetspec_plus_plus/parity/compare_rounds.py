"""Tier 2a: compare free-running fp32 T0 read tokens and per-round emitted lengths.

Reference metrics.jetspec_round_records[].accepted_len and torch accept_lengths
use acc+1 excluding prefill. Terminal lengths are credited only through the
emitted EOS/cap, as in the reference segment reducer. Pass labels are compared where
both reads record them. Requires identical tokens to establish round alignment;
use eval-j1r1 / eval-j3r3 fp32 qualification reads, never the eager recipe.
"""
import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from tests.jetspec_plus_plus.parity.read_data import paired_outputs, compare_tokens, emitted_rounds, read_identity


def compare_rounds(candidate, reference):
    for raw in (candidate, reference):
        identity = read_identity(raw)
        if identity.get('dtype', 'fp32') != 'fp32' or identity.get('temperature', 0) != 0:
            raise ValueError('tier 2a requires fp32 T0 reads')
    pairs = paired_outputs(candidate, reference)
    tokens = compare_tokens(candidate, reference)
    a, b = emitted_rounds(candidate), emitted_rounds(reference)
    fields = {key: dict(rounds_compared=0, rounds_equal=0, first_difference=None)
              for key in ('accepted_len', 'pass_label')}
    for (index, _, _), one, other in zip(pairs, a, b):
        for r in range(max(len(one), len(other))):
            x = one[r] if r < len(one) else {}
            y = other[r] if r < len(other) else {}
            for key, field in fields.items():
                if key == 'pass_label' and (key not in x or key not in y):
                    continue
                field['rounds_compared'] += 1
                if x.get(key) == y.get(key):
                    field['rounds_equal'] += 1
                elif field['first_difference'] is None:
                    field['first_difference'] = dict(prompt_index=index, round_index=r,
                                                      candidate=x.get(key), reference=y.get(key))
    return dict(schema='jetspec.round_comparison/1', tokens=tokens, fields=fields,
                equal=tokens['equal'] and all(f['rounds_equal'] == f['rounds_compared'] for f in fields.values()),
                limitation='free-running fp32 T0 identity; pass labels only where both reads record them')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('candidate', type=Path)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output-json', type=Path, required=True)
    args = parser.parse_args(argv)
    result = compare_rounds(json.loads(args.candidate.read_text()), json.loads(args.reference.read_text()))
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + '\n')
    return int(not result['equal'])


if __name__ == '__main__':
    raise SystemExit(main())
