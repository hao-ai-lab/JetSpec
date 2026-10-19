"""Compare ordered read tokens (tier 1) or paired rule-A tau (tier 3).

Tau uses the reference paired prompt-cluster bootstrap: 20000 replicates,
seed 9504, difference of ratios of sums and interpolated percentile limits.
Tokens mode exits 1 on any divergence; tau mode reports divergence and the CI.
"""
import argparse
import json
from pathlib import Path
import random
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from tests.jetspec_plus_plus.parity.read_data import compare_tokens, paper_credit


def quantile(values, q):
    ordered = sorted(values)
    at = (len(ordered) - 1) * q
    lo = int(at)
    return ordered[lo] + (ordered[min(lo + 1, len(ordered) - 1)] - ordered[lo]) * (at - lo)


def paired_tau_delta(candidate, reference, *, replicates=20000, seed=9504):
    """Ported from the JetSpec++ reference implementation."""
    et, er = candidate['prompt_tokens'], candidate['prompt_rounds']
    rt, rr = reference['prompt_tokens'], reference['prompt_rounds']
    n = len(et)
    if (n == 0 or any(len(v) != n for v in (er, rt, rr))
            or any(v <= 0 for v in er + rr) or any(v < 0 for v in et + rt)
            or replicates < 2):
        raise ValueError('paired bootstrap requires complete positive-round prompt clusters')
    rng = random.Random(seed)
    samples = []
    for _ in range(replicates):
        a = b = c = d = 0
        for _ in range(n):
            i = rng.randrange(n)
            a += et[i]
            b += er[i]
            c += rt[i]
            d += rr[i]
        samples.append(a / b - c / d)
    ci = [quantile(samples, .025), quantile(samples, .975)]
    return dict(eval_tau=sum(et) / sum(er), reference_tau=sum(rt) / sum(rr),
                delta_tau=sum(et) / sum(er) - sum(rt) / sum(rr), ci95=ci,
                includes_zero=ci[0] <= 0 <= ci[1], seed=seed, replicates=replicates,
                prompts=n, method='paired prompt-cluster bootstrap; difference of ratios of sums; linearly interpolated percentiles',
                limitation='conditional on these two reads; excludes between-run and node variance')


def compare_reads(candidate, reference, *, mode='tokens', replicates=20000, seed=9504):
    if mode not in ('tokens', 'tau'):
        raise ValueError('mode must be tokens or tau')
    result = dict(schema='jetspec.read_comparison/1', mode=mode,
                  tokens=compare_tokens(candidate, reference))
    if mode == 'tau':
        ec, rc = paper_credit(candidate), paper_credit(reference)
        result.update(candidate_rule_a=ec, reference_rule_a=rc,
                      bootstrap=paired_tau_delta(ec, rc, replicates=replicates, seed=seed))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('candidate', type=Path)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--mode', choices=['tokens', 'tau'], default='tokens')
    parser.add_argument('--output-json', type=Path, required=True)
    parser.add_argument('--replicates', type=int, default=20000)
    parser.add_argument('--seed', type=int, default=9504)
    args = parser.parse_args(argv)
    result = compare_reads(json.loads(args.candidate.read_text()), json.loads(args.reference.read_text()),
                           mode=args.mode, replicates=args.replicates, seed=args.seed)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + '\n')
    return int(args.mode == 'tokens' and not result['tokens']['equal'])


if __name__ == '__main__':
    raise SystemExit(main())
