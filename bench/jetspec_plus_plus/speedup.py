"""Measure JetSpec++ serving speedup over the graphed AR baseline on the same GPUs.

Both runs use the JetSpec++ serving engine (python -m jetspec.engines.jetspec_plus_plus),
the same prompts, the same token budget and one shared compiler cache. Each prompt stops
at EOS by default: text generated after EOS is easy to draft and inflates the speedup.
AR runs on the first listed GPU; JetSpec++ runs on the pair. The speedup is the ratio of
measured decode tokens per second (the engine's cold first prompt is excluded).

--schedule auto (the default) first reads a few prompts with every validated J/R schedule on
this GPU's kernel family and keeps the fastest. The fast engine's draft deadline still trims
passes round by round inside it.

Without artifact flags, the pinned target and release checkpoint are downloaded from Hugging
Face. Without --prompts-json, the bundled prompts of --dataset are used: the paper's Table 3
prompts for GSM8K and MATH-500, the evaluation exports for the other benchmarks.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / 'jetspec/engines/jetspec_plus_plus'
# Evaluation exports: first-turn Qwen3-8B renders (no thinking) of each benchmark, temperature 0.
BENCHMARKS = {'gsm8k': 'gsm8k-t0', 'math500': 'math500-t0', 'aime25': 'aime25-t0',
              'humaneval': 'humaneval-t0', 'livecodebench': 'lcb-t0', 'mbpp': 'mbpp-t0',
              'mt-bench': 'mt-bench-t0'}


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--model', default='qwen3-8b', help='target model (as in the engine entry)')
    result.add_argument('--schedule', default='auto',
                        help='JetSpec++ J/R schedule; auto measures every validated one on this GPU')
    result.add_argument('--tune-count', type=int, default=4,
                        help='prompts per schedule for --schedule auto (the first is the cold one)')
    result.add_argument('--engine', default='fast', help='fast or reference')
    for role in ('target', 'drafter', 'adapter', 'heads-bundle', 'target-release', 'host-values'):
        result.add_argument('--' + role)
    result.add_argument('--prompts-json', help='prompt rows (default: the bundled prompts of --dataset)')
    result.add_argument('--dataset', default='gsm8k')
    result.add_argument('--count', type=int, default=32)
    result.add_argument('--tokens', type=int, default=2048)
    result.add_argument('--devices', default='0,1')
    result.add_argument('--stop-at-eos', choices=('on', 'off'), default='on',
                        help='end each prompt at EOS (default); off runs to --tokens')
    result.add_argument('--cache-root', default=os.environ.get('JSPP_CACHE_ROOT') or str(
        Path(os.environ.get('XDG_CACHE_HOME') or Path.home() / '.cache') / 'jetspec-plus-plus'),
        help='persistent compiler cache shared by every read (the first read compiles)')
    result.add_argument('--knob', action='append', default=[], metavar='NAME=VALUE',
                        help='set a registered serving knob for the JetSpec++ read (repeatable)')
    result.add_argument('--output-dir', default='bench/outputs/jetspec_plus_plus_speedup')
    return result


def engine_command(args, schedule, output):
    command = [sys.executable, '-m', 'jetspec.engines.jetspec_plus_plus',
               '--prompts-json', args.prompts_json, '--dataset', args.dataset,
               '--count', str(args.count), '--tokens', str(args.tokens),
               '--devices', args.devices, '--output-json', str(output),
               '--stop-at-eos', args.stop_at_eos]
    command += ['--model', args.model]
    if schedule:
        command += ['--schedule', schedule]
    if schedule != 'ar':
        command += ['--engine', args.engine]
    roles = ('target', 'host_values') if schedule == 'ar' else (
        'target', 'drafter', 'adapter', 'heads_bundle', 'target_release', 'host_values')
    for role in roles:
        value = getattr(args, role)
        if value:
            command += ['--' + role.replace('_', '-'), value]
    if args.cache_root:
        command += ['--cache-root', args.cache_root]
    if schedule != 'ar':
        for knob in args.knob:
            command += ['--knob', knob]
    return command


def decode_tok_s(path):
    read = json.loads(Path(path).read_text())
    measured = read['metrics']['serve_read']['measured']
    return measured['decode_tok_s'], read['outputs']


def candidates(args):
    """The kernel family of the visible GPU and its validated JetSpec++ schedules."""
    from jetspec.engines.jetspec_plus_plus import entry
    host = json.loads(Path(args.host_values).read_text()) if args.host_values else {}
    family = entry.kernel_family(host.get('cuda_arch') or entry._device_arch(args.devices), host.get('machine'))
    return family, sorted({s for m, s, e, f in entry.RECIPES
                           if m == args.model and e == args.engine and f == family and s != 'ar'})


def choose_schedule(args, output, environment, run):
    """Read a few prompts with every candidate schedule at the same token cap; keep the fastest."""
    family, schedules = candidates(args)
    if not schedules:
        raise SystemExit(f'no validated {args.engine} schedule for {args.model} on {family}')
    trial = argparse.Namespace(**dict(vars(args), count=min(args.count, args.tune_count)))
    rates, errors = {}, {}
    for schedule in schedules:
        path = output / f'tune-{schedule}.json'
        done = run(engine_command(trial, schedule, path), cwd=ROOT, env=environment)
        if done.returncode == 0:
            rates[schedule] = decode_tok_s(path)[0]
        else:
            errors[schedule] = done.returncode
    if not rates:
        raise SystemExit(f'every candidate schedule failed on {family}: {errors}')
    choice = max(rates, key=rates.get)
    print('TUNE', family, json.dumps(rates), 'failed', json.dumps(errors), '->', choice, flush=True)
    return choice, dict(family=family, tok_s=rates, failed=errors, prompts=trial.count)


def rerender(source, target, destination):
    """Render each first user turn of an 8B export with the target's own chat template."""
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(target)
    data = json.loads(Path(source).read_text())
    rows = []
    for row in data['prompts']:
        user = row['text'].split('<|im_start|>user\n', 1)[1].split('<|im_end|>', 1)[0]
        text = tokenizer.apply_chat_template([{'role': 'user', 'content': user}], tokenize=False,
                                             add_generation_prompt=True, enable_thinking=False)
        rows.append({'index': row['index'], 'text': text,
                     'token_ids': tokenizer(text, add_special_tokens=False).input_ids})
    destination.write_text(json.dumps(dict(data, prompts=rows)))
    return destination


def bundled_prompts(model, dataset, target, directory):
    """The bundled prompts of a dataset, rendered for the model's chat template."""
    size = '27b' if model == 'qwen3.8-27b' else '8b'
    table3 = PACKAGE / 'table3/inputs' / size / f'prompts_{dataset}128.json'
    if table3.is_file():
        return str(table3)
    if dataset not in BENCHMARKS:
        raise SystemExit(f'no bundled prompts for --dataset {dataset} (bundled: {", ".join(BENCHMARKS)}); '
                         'pass --prompts-json')
    path = PACKAGE / 'config/benchmark' / f'{BENCHMARKS[dataset]}.prompts.json'
    if size == '8b':
        return str(path)
    if not target:
        raise SystemExit(f'--dataset {dataset} for {model} renders its prompts with the target tokenizer: pass --target')
    return str(rerender(path, target, Path(directory) / f'prompts_{dataset}.json'))


def main(argv=None, *, run=subprocess.run):
    args = parser().parse_args(argv)
    if not any(getattr(args, role) for role in ('target', 'drafter', 'adapter', 'heads_bundle', 'host_values')):
        from jetspec.engines.jetspec_plus_plus import entry
        target, checkpoint = entry.release_paths(args.model)
        args.target, args.drafter = str(target), str(checkpoint)
        args.adapter, args.heads_bundle = str(checkpoint / 'adapter'), str(checkpoint / 'calibration')
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if args.prompts_json is None:
        args.prompts_json = bundled_prompts(args.model, args.dataset, args.target, output)
        available = len(json.loads(Path(args.prompts_json).read_text())['prompts'])
        if args.count > available:
            print(f'--dataset {args.dataset} bundles {available} prompts; reading all of them', flush=True)
            args.count = available
    environment = dict(os.environ, PYTHONPATH=os.pathsep.join(
        filter(None, [str(ROOT), os.environ.get('PYTHONPATH')])))
    tuning = None
    if args.schedule == 'auto':
        args.schedule, tuning = choose_schedule(args, output, environment, run)
    reads = {}
    for name, schedule in (('ar', 'ar'), ('jetspec_plus_plus', args.schedule)):
        path = output / f'{name}.json'
        run(engine_command(args, schedule, path), cwd=ROOT, env=environment, check=True)
        reads[name] = decode_tok_s(path)
    ar, _ = reads['ar']
    method, outputs = reads['jetspec_plus_plus']
    result = dict(ar_decode_tok_s=ar, jetspec_plus_plus_decode_tok_s=method,
                  speedup=method / ar, prompts=len(outputs), tokens=args.tokens,
                  model=args.model, schedule=args.schedule, engine=args.engine, devices=args.devices, stop_at_eos=args.stop_at_eos,
                  knobs=args.knob, tuning=tuning,
                  baseline='graphed AR, same engine and GPU type, one GPU')
    (output / 'speedup.json').write_text(json.dumps(result, indent=2) + '\n')
    print(f"AR {ar:.1f} tok/s  JetSpec++ {method:.1f} tok/s  speedup {method / ar:.2f}x "
          f"({len(outputs)} prompts, up to {args.tokens} tokens, stop at EOS {args.stop_at_eos})")
    print(f"JSON: {output / 'speedup.json'}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
