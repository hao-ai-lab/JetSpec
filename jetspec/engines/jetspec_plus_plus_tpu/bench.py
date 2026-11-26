"""JetSpec++ vs greedy AR on one TPU chip: token equality, acceptance length and throughput.

    python -m jetspec.engines.jetspec_plus_plus_tpu.bench --dataset gsm8k --samples 16 \
        --max-tokens 2048 --schedules J1-R1,J1-R0,J0-R0

Throughput counts decode time only (after prefill and the first draft), as the paper does.
Each schedule and the AR baseline are warmed up on one prompt first so compiles are excluded.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

PROMPTS = Path(__file__).resolve().parents[3] / 'bench' / 'jetspec_plus_plus' / 'prompts'
DATASETS = {'gsm8k': 'gsm8k-t0.prompts.json', 'math500': '8b-prompts_math500128.json',
            'humaneval': 'humaneval-t0.prompts.json', 'mbpp': 'mbpp-t0.prompts.json',
            'aime25': 'aime25-t0.prompts.json', 'mt-bench': 'mt-bench-t0.prompts.json',
            'livecodebench': 'lcb-t0.prompts.json'}


def load_rows(dataset, samples):
    data = json.loads((PROMPTS / DATASETS[dataset]).read_text())
    rows = data['prompts'] if isinstance(data, dict) else data
    return [(int(r.get('index', i)), r['token_ids']) for i, r in enumerate(rows[:samples])]


def run(weights, dims, *, dataset='gsm8k', samples=8, max_tokens=512, schedules=('J1-R1',),
        capacity=None, ar=True, warm=True, log=print, draft_weights=None):
    """draft_weights: an optional second weight set (e.g. int8 draft side) for the spec runs."""
    from .engine import TPUJetSpecPP

    rows = load_rows(dataset, samples)
    longest = max(len(ids) for _, ids in rows)
    capacity = capacity or (-(-(longest + max_tokens + 128) // 512) * 512)
    result = dict(dataset=dataset, samples=len(rows), max_tokens=max_tokens, capacity=capacity,
                  schedules={})
    ar_out = {}
    if ar:
        eng = TPUJetSpecPP(weights, dims, capacity=capacity)
        if warm:
            eng.warmup(longest, max_tokens, ar=True)
        tokens = seconds = 0
        for index, ids in rows:
            out, info = eng.generate_ar(ids, max_tokens, stats=True)
            ar_out[index] = out
            tokens += len(out) - 1
            seconds += info['decode_seconds']
        result['ar'] = dict(tokens=tokens, seconds=seconds, tok_s=tokens / seconds)
        log(f'AR: {tokens} tokens in {seconds:.1f}s = {tokens / seconds:.1f} tok/s')
    for schedule in schedules:
        eng = TPUJetSpecPP(draft_weights or weights, dims, schedule=schedule, capacity=capacity)
        if warm:
            eng.warmup(longest, max_tokens)
        tokens = seconds = rounds = 0
        accepts, equal, mismatches = [], 0, []
        for index, ids in rows:
            out, info = eng.generate(ids, max_tokens, prompt_index=index, stats=True)
            tokens += sum(info['accept_lengths'])
            seconds += info['decode_seconds']
            rounds += info['rounds']
            accepts += info['accept_lengths']
            if index in ar_out:
                ref = ar_out[index]
                if out == ref:
                    equal += 1
                else:
                    first = next((k for k, (a, b) in enumerate(zip(out, ref)) if a != b),
                                 min(len(out), len(ref)))
                    mismatches.append(dict(index=index, first_diff=first, len=len(out), ar_len=len(ref)))
        entry = dict(tokens=tokens, seconds=seconds, rounds=rounds, tau=float(np.mean(accepts)),
                     tok_s=tokens / seconds, ms_per_round=seconds / rounds * 1e3,
                     ar_equal=f'{equal}/{len(ar_out)}' if ar_out else None, mismatches=mismatches)
        if ar_out:
            entry['speedup'] = entry['tok_s'] / result['ar']['tok_s']
        result['schedules'][schedule] = entry
        log(f'{schedule}: tau={entry["tau"]:.2f} {entry["ms_per_round"]:.2f} ms/round '
            f'{entry["tok_s"]:.1f} tok/s' + (f' speedup {entry["speedup"]:.2f}x equal {entry["ar_equal"]}'
                                             if ar_out else ''))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset', choices=sorted(DATASETS), default='gsm8k')
    parser.add_argument('--samples', type=int, default=16)
    parser.add_argument('--max-tokens', type=int, default=2048)
    parser.add_argument('--schedules', default='J1-R1')
    parser.add_argument('--target', help='local Qwen3-8B directory (default: pinned hub snapshot)')
    parser.add_argument('--checkpoint', help='local JetSpec++ checkpoint directory')
    parser.add_argument('--no-ar', action='store_true')
    parser.add_argument('--draft-dtype', choices=['int8', 'bf16'], default='int8',
                        help='weight-only int8 for drafter/adapter/decoder copies (lossless: the '
                             'target that verifies stays bf16)')
    parser.add_argument('--output')
    args = parser.parse_args(argv)
    from .loader import load_weights
    weights, dims, _, _ = load_weights(args.target, args.checkpoint, draft_dtype=args.draft_dtype)
    result = run(weights, dims, dataset=args.dataset, samples=args.samples, max_tokens=args.max_tokens,
                 schedules=tuple(args.schedules.split(',')), ar=not args.no_ar)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'schedules'}, indent=1))


if __name__ == '__main__':
    main()
