"""python -m jetspec.engines.jetspec_plus_plus_tpu "PROMPT" [--schedule J1-R1] [--max-tokens 2048]

One prompt on one TPU chip: renders the Qwen3 chat template without thinking, decodes greedily
with JetSpec++ and prints the answer with its throughput.
"""
import argparse
import time


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('prompt')
    parser.add_argument('--schedule', default='J1-R1')
    parser.add_argument('--max-tokens', type=int, default=2048)
    parser.add_argument('--draft-dtype', choices=['int8', 'bf16'], default='int8')
    parser.add_argument('--target')
    parser.add_argument('--checkpoint')
    parser.add_argument('--check-ar', action='store_true', help='also decode with plain greedy AR and compare')
    args = parser.parse_args(argv)

    from transformers import AutoTokenizer
    from .engine import TPUJetSpecPP
    from .loader import load_weights

    weights, dims, target, _ = load_weights(args.target, args.checkpoint, draft_dtype=args.draft_dtype)
    tokenizer = AutoTokenizer.from_pretrained(target)
    text = tokenizer.apply_chat_template([{'role': 'user', 'content': args.prompt}], add_generation_prompt=True,
                                         enable_thinking=False, tokenize=False)
    ids = tokenizer.encode(text, add_special_tokens=False)
    capacity = -(-(len(ids) + args.max_tokens + 128) // 512) * 512
    engine = TPUJetSpecPP(weights, dims, schedule=args.schedule, capacity=capacity)
    start = time.perf_counter()
    engine.warmup(len(ids), args.max_tokens)
    print(f'compiled in {time.perf_counter() - start:.1f}s')
    out, info = engine.generate(ids, args.max_tokens, stats=True)
    print(tokenizer.decode(out, skip_special_tokens=True))
    tokens = sum(info['accept_lengths'])
    print(f'\n[{args.schedule}, {args.draft_dtype} draft] {tokens} tokens, {info["rounds"]} rounds, '
          f'tau {tokens / max(info["rounds"], 1):.2f}, {tokens / info["decode_seconds"]:.1f} tok/s')
    if args.check_ar:
        engine.warmup(len(ids), args.max_tokens, ar=True)
        ar, ar_info = engine.generate_ar(ids, args.max_tokens, stats=True)
        print(f'AR {(len(ar) - 1) / ar_info["decode_seconds"]:.1f} tok/s; identical output: {ar == out}')


if __name__ == '__main__':
    main()
