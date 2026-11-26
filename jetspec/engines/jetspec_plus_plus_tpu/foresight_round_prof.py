"""Where a two-chip serving round goes, at one batch size: resolve (chip 1, row lookup + redraft of missed slots),
verify (chip 0) and foresight (chip 1), each timed on its own, against the pipelined round.

    python -m jetspec.engines.jetspec_plus_plus_tpu.foresight_round_prof --slots 8 --rows 1 --group 4 --exact-tail

Per round the critical path is resolve -> max(verify, foresight): chip 0 cannot start block t until chip 1 has
resolved it, and chip 1 runs resolve and foresight in order. Slots run in lockstep on GSM8K prompts without
retiring, for --rounds rounds after --warm rounds.
"""
import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import jax
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--slots', type=int, default=8)
    ap.add_argument('--rows', type=int, default=1)
    ap.add_argument('--group', type=int, default=4)
    ap.add_argument('--exact-tail', action='store_true')
    ap.add_argument('--redraft-schedule', default='')
    ap.add_argument('--schedule', default='J1-R0')
    ap.add_argument('--rounds', type=int, default=30)
    ap.add_argument('--warm', type=int, default=3)
    ap.add_argument('--output')
    args = ap.parse_args()

    from . import loader, model
    from .bench import load_rows
    from .foresight_serve import TwoChipServer

    cpu = jax.devices('cpu')[0]
    d0, d1 = jax.devices()[:2]
    host_load = model.load_target

    def load_target_on_host(root, dims):
        with jax.default_device(cpu):
            return host_load(root, dims)
    loader.load_target = load_target_on_host
    weights, dims, _, ckpt = loader.load_weights(draft_dtype='int8', verbose=False)
    tw = weights['target']
    with jax.default_device(cpu):
        q = jax.jit(model.quantize)
        tw = dict(tw, layers={k: (q(v) if k in ('qkv', 'o', 'gate_up', 'down') else v) for k, v in tw['layers'].items()})
    th = json.loads((Path(ckpt) / 'calibration' / 'thresholds.json').read_text())
    coef = {k: float(v['value']) for k, v in th['e04_kscore_blend']['pooled'].items()}
    temp = float(th['e04_rowrule_joint']['pooled']['temperature']['value'])
    w0 = jax.device_put(dict(target=tw), d0)
    w1 = jax.device_put(dict(drafter=weights['drafter'], adapter=weights['adapter'], decoder=weights['decoder'],
                             embed=tw['embed'], norm=tw['norm']), d1)
    rows = load_rows('gsm8k', max(16, args.slots))
    to0 = lambda x: jax.device_put(x, d0)
    to1 = lambda x: jax.device_put(x, d1)
    srv = TwoChipServer(w0, w1, coef, temp, slots=args.slots, capacity=640, dims=dims, rows=args.rows,
                        schedule=args.schedule, group=args.group, exact_tail=args.exact_tail,
                        redraft_schedule=args.redraft_schedule)
    srv.warmup([len(i) for _, i in rows], 256)
    span = 512

    def admit_all():
        c0, s1, v0 = srv.empty_state()
        key = to1(jax.random.PRNGKey(1))
        for s in range(args.slots):
            _, ids = rows[s]
            p = len(ids)
            padded = srv.pad_len(p)
            buf = np.zeros(padded, np.int32)
            buf[:p] = ids
            c0, first, taps, mk, mv = srv._c('prefill', padded)(w0, c0, s, to0(buf), p)
            s1 = srv._c('admit', padded)(w1, s1, s, to1(buf), p, to1(first), to1(taps), to1(mk), to1(mv), key)
        for step in range(args.warm):
            c0, s1, v0, _ = srv._round(c0, s1, v0, step, span)
        return c0, s1, v0

    # 1. each program on its own (synchronised), and the miss count of every round
    c0, s1, v0 = admit_all()
    t_res, t_ver, t_fs, misses, accepted = [], [], [], [], []
    by_miss = defaultdict(list)
    for r in range(args.rounds):
        step = args.warm + r
        v1 = jax.tree.map(to1, v0)
        jax.block_until_ready(v1)
        t = time.perf_counter()
        tokens, block, p1, length, hit, emitted, m = srv._c('resolve', span)(w1, s1, v1, 2 * step)
        jax.block_until_ready(block)
        t1 = time.perf_counter()
        b0, l0 = to0(block), to0(length)
        jax.block_until_ready((b0, l0))
        t2 = time.perf_counter()
        c0, v0 = srv._c('verify', span)(w0, c0, b0, l0)
        jax.block_until_ready(v0)
        t3 = time.perf_counter()
        s1 = srv._c('foresight', span)(w1, s1, tokens, block, p1, length, hit, v1, 2 * step + 1)
        jax.block_until_ready(s1)
        t4 = time.perf_counter()
        m = int(m)
        t_res.append(1e3 * (t1 - t))
        t_ver.append(1e3 * (t3 - t2))
        t_fs.append(1e3 * (t4 - t3))
        misses.append(m)
        by_miss[m].append(1e3 * (t1 - t))
        accepted.append(int(np.sum(np.asarray(emitted) >= 0)))

    # 2. the pipelined round, as serving runs it (host dispatches, reads nothing)
    c0, s1, v0 = admit_all()
    jax.block_until_ready(s1)
    t = time.perf_counter()
    for r in range(args.rounds):
        c0, s1, v0, out = srv._round(c0, s1, v0, args.warm + r, span)
    jax.block_until_ready((s1, v0))
    pipelined = 1e3 * (time.perf_counter() - t) / args.rounds

    med = lambda x: round(float(np.median(x)), 2)
    res = dict(slots=args.slots, rows=args.rows, group=args.group, exact_tail=args.exact_tail, schedule=args.schedule,
               redraft_schedule=args.redraft_schedule,
               rounds=args.rounds,
               resolve_ms=med(t_res), verify_ms=med(t_ver), foresight_ms=med(t_fs),
               resolve_ms_by_misses={k: dict(n=len(v), ms=med(v)) for k, v in sorted(by_miss.items())},
               rounds_with_a_miss=round(float(np.mean([m > 0 for m in misses])), 3),
               slot_miss_rate=round(float(np.mean(misses)) / args.slots, 3),
               mean_misses=round(float(np.mean(misses)), 2),
               tokens_per_round=round(float(np.mean(accepted)), 2),
               critical_path_ms=round(med(t_res) + max(med(t_ver), med(t_fs)), 2),
               pipelined_round_ms=round(pipelined, 2))
    print('ROUND', json.dumps(res), flush=True)
    if args.output:
        Path(args.output).write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
