"""Minimal demo: JetSpec++ greedy generation from local artifacts."""
import argparse
import os
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('model', 'drafter', 'adapter', 'heads-bundle'):
        parser.add_argument('--' + name, default=os.environ.get(
            'JETSPEC_' + name.upper().replace('-', '_')), help='local artifact path')
    parser.add_argument('--prompt', default='The three primary colors are')
    parser.add_argument('--max-tokens', type=int, default=64)
    parser.add_argument('--schedule', default='J1-R1')
    parser.add_argument('--dataset', default='gsm8k', help='heads calibration entry')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default=None)
    parser.add_argument('--dtype', choices=['fp32', 'bf16'], default='fp32')
    args = parser.parse_args(argv)
    for name in ('model', 'drafter', 'adapter', 'heads_bundle'):
        if not getattr(args, name):
            parser.error(f'--{name.replace("_", "-")} or JETSPEC_{name.upper()} is required')
    if args.max_tokens < 1:
        parser.error('--max-tokens must be positive')

    import torch
    from jetspec.core.llm import LLM
    from jetspec.methods.jetspec_plus_plus.drafter import build_drafter
    from jetspec.methods.jetspec_plus_plus.reader import read_ar, read_passes, summarize
    from jetspec.methods.jetspec_plus_plus.round import JetSpecPPConfig

    llm = LLM(args.model, device=args.device or ('cuda' if torch.cuda.is_available() else 'cpu'),
              dtype=torch.float32 if args.dtype == 'fp32' else torch.bfloat16,
              attn_implementation='sdpa')
    config = JetSpecPPConfig.paper_eval(schedule=args.schedule)
    drafter = build_drafter(llm=llm, config=config, drafter_checkpoint=args.drafter,
                            adapter_artifact=args.adapter, heads_bundle=args.heads_bundle,
                            dataset=args.dataset)
    outputs = read_passes(llm, [args.prompt], drafter_factory=lambda **kwargs: drafter,
                          config=config, tokens=args.max_tokens)
    ar = read_ar(llm, [args.prompt], tokens=args.max_tokens)
    equal = outputs[0]['token_ids'] == ar[0]['token_ids']
    mean = summarize(outputs)['tau']
    print(llm.tokenizer.decode(outputs[0]['token_ids'], skip_special_tokens=True))
    print('accepted-length mean:', mean if mean is not None else 0.0)
    print('AR equality:', equal)
    return 0 if equal else 1


if __name__ == '__main__':
    raise SystemExit(main())
