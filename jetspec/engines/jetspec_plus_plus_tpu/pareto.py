"""Serving Pareto frontier on one TPU chip: per-user tok/s vs total tok/s across concurrency.

    python -m jetspec.engines.jetspec_plus_plus_tpu.pareto --method jetspec --slots 1,2,4,8,16,32 \
        --dataset gsm8k --requests 128 --max-tokens 1024 --output out/pareto-jetspec.json

Run once per method (ar, dflash, jetspec); each run loads only the weights it needs. Every point
uses continuous batching over the same request list. Total tok/s is generated tokens over wall
time, prefills included. Per-user tok/s is each request's tokens over its admission-to-last-token
time; the mean is reported.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--method', choices=['ar', 'dflash', 'jetspec'], required=True)
    parser.add_argument('--slots', default='1,2,4,8,16,32')
    parser.add_argument('--schedule', default='J1-R1', help='JetSpec++ J/R schedule')
    parser.add_argument('--draft-dtype', choices=['int8', 'bf16'], default='int8')
    parser.add_argument('--dataset', default='gsm8k')
    parser.add_argument('--requests', type=int, default=128)
    parser.add_argument('--max-tokens', type=int, default=1024)
    parser.add_argument('--keep-tokens', action='store_true', help='store every output in the JSON')
    parser.add_argument('--output', required=True)
    args = parser.parse_args(argv)

    import jax
    from .bench import load_rows
    from .loader import load_method_weights
    from .serve import Server, summarize

    weights, dims, _, _ = load_method_weights(args.method, draft_dtype=args.draft_dtype)
    rows = load_rows(args.dataset, args.requests)
    lens = [len(ids) for _, ids in rows]
    capacity = -(-(max(lens) + args.max_tokens + 4 * dims.block) // 128) * 128
    label = {'ar': 'AR', 'dflash': f'DFlash ({args.draft_dtype})',
             'jetspec': f'JetSpec++ {args.schedule} ({args.draft_dtype})'}[args.method]
    out = dict(method=args.method, label=label, schedule=args.schedule, draft_dtype=args.draft_dtype,
               dataset=args.dataset, requests=len(rows), max_tokens=args.max_tokens, capacity=capacity,
               points=[])
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    for slots in [int(s) for s in args.slots.split(',')]:
        server = Server(weights, args.method, slots=slots, capacity=capacity, dims=dims, schedule=args.schedule)
        try:
            start = time.perf_counter()
            server.warmup(lens, args.max_tokens)
            compile_s = time.perf_counter() - start
            records, wall = server.run(rows, args.max_tokens)
        except Exception as error:   # most often HBM exhaustion at large B
            print(f'{label} B={slots}: failed: {str(error)[:200]}', flush=True)
            out['points'].append(dict(slots=slots, error=str(error)[:500]))
            path.write_text(json.dumps(out, indent=1))
            break
        point = dict(slots=slots, compile_seconds=compile_s, **summarize(records, wall))
        if args.keep_tokens:
            point['outputs'] = {r['index']: r['out'] for r in records}
        out['points'].append(point)
        path.write_text(json.dumps(out, indent=1))
        print(f'{label} B={slots}: total {point["total_tok_s"]:.1f} tok/s, per-user '
              f'{point["per_user_tok_s_mean"]:.1f} tok/s, tau {point["tau"]}', flush=True)
        del server
        jax.clear_caches()


if __name__ == '__main__':
    main()
