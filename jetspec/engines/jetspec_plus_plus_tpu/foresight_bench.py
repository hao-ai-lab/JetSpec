"""Two-chip foresight JetSpec++ vs the one-chip engine on the same prompts (B = 1).

    python -m jetspec.engines.jetspec_plus_plus_tpu.foresight_bench --schedule J1-R0 --samples 8 \
        --max-tokens 256 --int8-target --output out/fs.json

--int8-target: weight-only int8 for the target's decoder layers (needed on 16 GB v5e chips).
"""
import argparse
import json
import time
from pathlib import Path

import jax
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--schedule', default='J1-R0')
    ap.add_argument('--samples', type=int, default=8)
    ap.add_argument('--max-tokens', type=int, default=256)
    ap.add_argument('--dataset', default='gsm8k')
    ap.add_argument('--int8-target', action='store_true')
    ap.add_argument('--skip-baseline', action='store_true')
    ap.add_argument('--timing', action='store_true')
    ap.add_argument('--rows', type=int, default=8)
    ap.add_argument('--output', required=True)
    args = ap.parse_args()

    from huggingface_hub import snapshot_download
    from . import loader, model
    from .bench import load_rows
    from .engine import TPUJetSpecPP
    from .foresight import TwoChipJetSpecPP

    cpu = jax.devices('cpu')[0]
    d0, d1, d2 = jax.devices()[:3]
    orig = model.load_target

    def host_target(root, dims):
        with jax.default_device(cpu):
            return orig(root, dims)
    loader.load_target = host_target
    weights, dims, _, ckpt = loader.load_weights(draft_dtype='int8', verbose=False)
    tw = weights['target']
    if args.int8_target:
        with jax.default_device(cpu):
            q = jax.jit(model.quantize)
            tw = dict(tw, layers={k: (q(v) if k in ('qkv', 'o', 'gate_up', 'down') else v)
                                  for k, v in tw['layers'].items()})
    thresholds = json.loads((Path(ckpt) / 'calibration' / 'thresholds.json').read_text())
    coef = {k: float(v['value']) for k, v in thresholds['e04_kscore_blend']['pooled'].items()}
    temperature = float(thresholds['e04_rowrule_joint']['pooled']['temperature']['value'])

    w0 = jax.device_put(dict(target=tw), d0)
    w1 = jax.device_put(dict(drafter=weights['drafter'], adapter=weights['adapter'], decoder=weights['decoder'],
                             embed=tw['embed'], norm=tw['norm']), d1)
    rows = load_rows(args.dataset, args.samples)
    result = dict(rows=args.rows, schedule=args.schedule, int8_target=args.int8_target, max_tokens=args.max_tokens,
                  coef=coef, temperature=temperature, prompts=[])

    two = TwoChipJetSpecPP(w0, w1, coef, temperature, dims=dims, schedule=args.schedule, rows=args.rows)
    one = None
    if not args.skip_baseline:
        w2 = jax.device_put(dict(target=tw, drafter=weights['drafter'], adapter=weights['adapter'],
                                 decoder=weights['decoder']), d2)
        one = TPUJetSpecPP(w2, dims=dims, schedule=args.schedule, capacity=2048 + 512)

    # warm both on the first prompt (compiles every span bucket it reaches)
    index, ids = rows[0]
    t = time.perf_counter(); two.generate(ids, args.max_tokens, prompt_index=index)
    print(f'two-chip warm {time.perf_counter() - t:.1f}s', flush=True)
    if one is not None:
        with jax.default_device(d2):
            t = time.perf_counter(); one.generate(ids, args.max_tokens, prompt_index=index)
        print(f'one-chip warm {time.perf_counter() - t:.1f}s', flush=True)

    if args.timing:
        two.timing = True
        two.generate(rows[1][1], args.max_tokens, prompt_index=rows[1][0])
        ms = np.array(two.stage_ms[2:])
        result['stage_ms_median'] = dict(zip(('resolve', 'verify_with_foresight'), np.median(ms, 0).tolist()))
        two.timing, two.stage_ms = 'split', []
        two.generate(rows[1][1], args.max_tokens, prompt_index=rows[1][0])
        ms = np.array(two.stage_ms[2:])
        result['stage_ms_split'] = dict(zip(('resolve', 'verify_alone', 'foresight_alone'), np.median(ms, 0).tolist()))
        print('SPLIT', json.dumps(result['stage_ms_split']), flush=True)
        print('STAGES', json.dumps(result['stage_ms_median']), flush=True)
        two.timing = False
    for index, ids in rows:
        out2, s2 = two.generate(ids, args.max_tokens, prompt_index=index)
        rec = dict(index=index, two=dict(tokens=len(out2), rounds=s2['rounds'], seconds=s2['seconds'],
                                         tau=float(np.mean(s2['accepts'])), hit_rate=float(np.mean(s2['hits']))))
        if one is not None:
            with jax.default_device(d2):
                out1, s1 = one.generate(ids, args.max_tokens, prompt_index=index, stats=True)
            n = min(len(out1), len(out2))
            first_diff = next((i for i in range(n) if out1[i] != out2[i]), None)
            rec['one'] = dict(tokens=len(out1), rounds=s1['rounds'], seconds=s1['decode_seconds'],
                              tau=float(np.mean(s1['accept_lengths'])))
            rec['identical'] = first_diff is None and len(out1) == len(out2)
            rec['first_diff'] = first_diff
        result['prompts'].append(rec)
        print(json.dumps(rec), flush=True)
        Path(args.output).write_text(json.dumps(result, indent=1))

    def agg(side):
        recs = [r[side] for r in result['prompts'] if side in r]
        toks = sum(r['tokens'] - 1 for r in recs)
        secs = sum(r['seconds'] for r in recs)
        rounds = sum(r['rounds'] for r in recs)
        return dict(tok_s=toks / secs, ms_per_round=1e3 * secs / rounds, tau=toks / rounds)
    result['two_summary'] = agg('two')
    result['two_summary']['hit_rate'] = float(np.mean([r['two']['hit_rate'] for r in result['prompts']]))
    if one is not None:
        result['one_summary'] = agg('one')
        result['identical'] = sum(r['identical'] for r in result['prompts'])
    Path(args.output).write_text(json.dumps(result, indent=1))
    print('SUMMARY', json.dumps({k: v for k, v in result.items() if k.endswith('summary') or k == 'identical'}),
          flush=True)


if __name__ == '__main__':
    main()
