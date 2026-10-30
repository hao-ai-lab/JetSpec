"""Shared inputs and wall-clock accounting for the JetSpec++ torch commands."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import time


def inputs(argv, *, description, samples, max_tokens, output):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--model', '--target', dest='model',
                        default=os.environ.get('JETSPEC_MODEL', 'Qwen/Qwen3-8B'))
    parser.add_argument('--drafter', default=os.environ.get('JETSPEC_DRAFTER'))
    parser.add_argument('--adapter', default=os.environ.get('JETSPEC_ADAPTER'))
    parser.add_argument('--heads-bundle', default=os.environ.get('JETSPEC_HEADS_BUNDLE'))
    parser.add_argument('--prompt-set', choices=['gsm8k', 'math500'], default='gsm8k')
    parser.add_argument('--prompts-file', default=os.environ.get('JETSPEC_PROMPTS_FILE'),
                        help='local JSON prompts (strings, prompt objects or token_ids)')
    parser.add_argument('--samples', type=int, default=samples)
    parser.add_argument('--max-tokens', type=int, default=max_tokens)
    parser.add_argument('--schedule', default='J1-R1', help='paper J/R schedule, e.g. J1-R1 or J3-R3')
    parser.add_argument('--dataset', help='heads calibration entry; defaults to --prompt-set')
    parser.add_argument('--seed', type=int, default=9504)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default=None)
    parser.add_argument('--dtype', choices=['fp32', 'bf16'], default='fp32')
    parser.add_argument('--draft-dtype', choices=['same', 'fp32', 'bf16'], default='same')
    parser.add_argument('--output', default=output, help='result JSON path')
    args = parser.parse_args(argv)
    for name in ('drafter', 'adapter', 'heads_bundle'):
        if not getattr(args, name):
            parser.error(f'--{name.replace("_", "-")} or JETSPEC_{name.upper()} is required')
    if args.samples < 1 or args.max_tokens < 1:
        parser.error('--samples and --max-tokens must be positive')
    if args.prompts_file:
        rows = json.loads(Path(args.prompts_file).read_text(encoding='utf-8'))
        if isinstance(rows, dict):
            rows = rows['prompts']
        if not isinstance(rows, list) or not rows:
            parser.error('--prompts-file must contain a nonempty prompts list')
        rows = rows[:args.samples]
    else:
        rows = None
    return args, rows


def prepare(args, rows, *, llm_factory=None, drafter_factory=None):
    import torch
    from jetspec.core.llm import LLM
    from jetspec.methods.jetspec_plus_plus.drafter import build_drafter
    from jetspec.methods.jetspec_plus_plus.round import JetSpecPPConfig

    args.device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float32 if args.dtype == 'fp32' else torch.bfloat16
    llm = (llm_factory or LLM)(model_name_or_path=args.model, device=args.device,
                               dtype=dtype, attn_implementation='sdpa')
    if rows is None:
        from bench.jetspec.reference.benchmark import build_prompts
        rows = build_prompts(llm.tokenizer, args.prompt_set, args.samples)
    config = JetSpecPPConfig.paper_eval(schedule=args.schedule, seed=args.seed)
    options = {} if args.draft_dtype == 'same' else {
        'draft_dtype': torch.float32 if args.draft_dtype == 'fp32' else torch.bfloat16}
    drafter = (drafter_factory or build_drafter)(
        llm=llm, config=config, drafter_checkpoint=args.drafter,
        adapter_artifact=args.adapter, heads_bundle=args.heads_bundle,
        dataset=args.dataset or args.prompt_set, **options)
    return llm, rows, config, drafter


def timed(llm, fn):
    import torch

    def synchronize():
        if torch.device(llm.device).type == 'cuda':
            torch.cuda.synchronize(llm.device)

    synchronize()
    start = time.perf_counter()
    outputs = fn()
    synchronize()
    return outputs, time.perf_counter() - start


def read_result(args, config, llm, outputs, seconds):
    from jetspec.methods.jetspec_plus_plus.reader import summarize, reduce_read

    summary = dict(summarize(outputs), wall_seconds=seconds)
    summary['tokens_per_second'] = summary['tokens'] / seconds if seconds else 0.0
    return dict(schema='jetspec.jetspec_plus_plus.read/1', outputs=outputs, summary=summary,
                rule_a=reduce_read({'outputs': outputs}, eos_token_ids=llm.eos_token_ids,
                                   max_new_tokens=args.max_tokens),
                manifest=dict(target=args.model, drafter=args.drafter, adapter=args.adapter,
                              heads_bundle=args.heads_bundle, prompt_set=args.prompt_set,
                              prompts_file=args.prompts_file, schedule=args.schedule,
                              dataset=args.dataset or args.prompt_set, config=asdict(config),
                              max_new_tokens=args.max_tokens, temperature=0,
                              dtype=args.dtype, draft_dtype=args.dtype if args.draft_dtype == 'same'
                              else args.draft_dtype, device=args.device,
                              timing='generation only; excludes model and artifact loading'))


def write_result(path, result):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
