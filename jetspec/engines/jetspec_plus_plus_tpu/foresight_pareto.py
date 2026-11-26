"""Two-chip foresight serving vs the one-chip batched server across concurrency (continuous batching).

    python -m jetspec.engines.jetspec_plus_plus_tpu.foresight_pareto --slots 1,4,8,16 --rows 8,4 \
        --requests 32 --max-tokens 256 --int8-target --output out/fs_pareto.json

Two-chip runs on chips 0 (target) and 1 (draft side); the one-chip baseline on chip 2.
"""
import argparse
import json
import time
from pathlib import Path

import jax
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--slots', default='1,4,8,16')
    ap.add_argument('--rows', default='8')
    ap.add_argument('--exact-tail', action='store_true')
    ap.add_argument('--redraft-schedule', default='', help='J/R schedule of a miss redraft (default: --schedule)')
    ap.add_argument('--group', default='0', help='redraft group sizes to sweep (0: whole batch)')
    ap.add_argument('--schedule', default='J1-R0')
    ap.add_argument('--dataset', default='gsm8k')
    ap.add_argument('--requests', type=int, default=32)
    ap.add_argument('--max-tokens', type=int, default=256)
    ap.add_argument('--int8-target', action='store_true')
    ap.add_argument('--skip-baseline', action='store_true')
    ap.add_argument('--output', required=True)
    args = ap.parse_args()

    from . import loader, model
    from .bench import load_rows
    from .foresight_serve import TwoChipServer, summarize
    from .serve import Server, summarize as summarize1

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
    th = json.loads((Path(ckpt) / 'calibration' / 'thresholds.json').read_text())
    coef = {k: float(v['value']) for k, v in th['e04_kscore_blend']['pooled'].items()}
    temperature = float(th['e04_rowrule_joint']['pooled']['temperature']['value'])
    w0 = jax.device_put(dict(target=tw), d0)
    w1 = jax.device_put(dict(drafter=weights['drafter'], adapter=weights['adapter'], decoder=weights['decoder'],
                             embed=tw['embed'], norm=tw['norm']), d1)
    w2 = None
    if not args.skip_baseline:
        w2 = jax.device_put(dict(target=tw, drafter=weights['drafter'], adapter=weights['adapter'],
                                 decoder=weights['decoder']), d2)
    rows = load_rows(args.dataset, args.requests)
    lens = [len(ids) for _, ids in rows]
    capacity = -(-(max(lens) + args.max_tokens + 4 * dims.block) // 128) * 128
    out = dict(schedule=args.schedule, exact_tail=args.exact_tail,
               redraft_schedule=args.redraft_schedule, int8_target=args.int8_target, requests=len(rows),
               max_tokens=args.max_tokens, capacity=capacity, points=[])
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)

    for slots in [int(s) for s in args.slots.split(',')]:
        base_out = None
        if w2 is not None:
            with jax.default_device(d2):
                srv = Server(w2, 'jetspec', slots=slots, capacity=capacity, dims=dims, schedule=args.schedule)
                srv.warmup(lens, args.max_tokens)
                recs, wall = srv.run(rows, args.max_tokens)
            p = dict(engine='one-chip', slots=slots, **summarize1(recs, wall))
            base_out = {r['index']: r['out'] for r in recs}
            out['points'].append(p)
            print(json.dumps(p), flush=True)
            del srv
        for r, g in [(int(x), int(y)) for x in args.rows.split(',') for y in args.group.split(',')]:
            srv = TwoChipServer(w0, w1, coef, temperature, slots=slots, capacity=capacity, dims=dims,
                                schedule=args.schedule, rows=r, group=g, exact_tail=args.exact_tail,
                                redraft_schedule=args.redraft_schedule)
            try:
                t = time.perf_counter()
                srv.warmup(lens, args.max_tokens)
                compile_s = time.perf_counter() - t
                recs, wall, extra = srv.run(rows, args.max_tokens)
            except Exception as error:
                print(f'two-chip B={slots} rows={r} group={g}: failed: {str(error)[:300]}', flush=True)
                out['points'].append(dict(engine='two-chip', slots=slots, rows=r, group=g, error=str(error)[:600]))
                path.write_text(json.dumps(out, indent=1))
                continue
            p = dict(engine='two-chip', slots=slots, rows=r, group=g, compile_seconds=compile_s, **extra,
                     **summarize(recs, wall))
            if base_out is not None:
                mine = {x['index']: x['out'] for x in recs}
                same = [i for i in mine if mine[i] == base_out.get(i)]
                p['identical'] = len(same)
                p['first_diff'] = {i: next((j for j in range(min(len(mine[i]), len(base_out[i])))
                                            if mine[i][j] != base_out[i][j]), None)
                                   for i in mine if i not in same}
            out['points'].append(p)
            print(json.dumps(p), flush=True)
            path.write_text(json.dumps(out, indent=1))
            del srv
        jax.clear_caches()
    path.write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
