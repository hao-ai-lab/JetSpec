"""Measure JetSpec++ serving speedup over the graphed AR baseline on the same GPUs.

Both runs use the JetSpec++ serving engine (python -m jetspec.engines.jetspec_plus_plus),
the same prompts, the same token budget (EOS ignored) and one shared compiler cache.
AR runs on the first listed GPU; JetSpec++ runs on the pair. The speedup is the ratio of
measured decode tokens per second (the engine's cold first prompt is excluded).
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--config', help='JetSpec++ configuration; defaults by device architecture')
    for role in ('target', 'drafter', 'adapter', 'heads-bundle', 'target-release', 'host-values'):
        result.add_argument('--' + role)
    result.add_argument('--prompts-json', required=True)
    result.add_argument('--dataset', default='gsm8k')
    result.add_argument('--count', type=int, default=8)
    result.add_argument('--tokens', type=int, default=256)
    result.add_argument('--devices', default='0,1')
    result.add_argument('--cache-root', default=os.environ.get('JSPP_CACHE_ROOT'))
    result.add_argument('--knob', action='append', default=[], metavar='NAME=VALUE',
                        help='set a registered serving knob for the JetSpec++ read (repeatable)')
    result.add_argument('--output-dir', default='bench/outputs/jetspec_plus_plus_speedup')
    return result


def engine_command(args, config, output):
    command = [sys.executable, '-m', 'jetspec.engines.jetspec_plus_plus',
               '--prompts-json', args.prompts_json, '--dataset', args.dataset,
               '--count', str(args.count), '--tokens', str(args.tokens),
               '--devices', args.devices, '--output-json', str(output)]
    if config:
        command += ['--config', config]
    roles = ('target', 'host_values') if config == 'ar' else (
        'target', 'drafter', 'adapter', 'heads_bundle', 'target_release', 'host_values')
    for role in roles:
        value = getattr(args, role)
        if value:
            command += ['--' + role.replace('_', '-'), value]
    if args.cache_root:
        command += ['--cache-root', args.cache_root]
    if config != 'ar':
        for knob in args.knob:
            command += ['--knob', knob]
    return command


def decode_tok_s(path):
    read = json.loads(Path(path).read_text())
    measured = read['metrics']['serve_read']['measured']
    return measured['decode_tok_s'], read['outputs']


def main(argv=None, *, run=subprocess.run):
    args = parser().parse_args(argv)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ, PYTHONPATH=os.pathsep.join(
        filter(None, [str(ROOT), os.environ.get('PYTHONPATH')])))
    reads = {}
    for name, config in (('ar', 'ar'), ('jetspec_plus_plus', args.config)):
        path = output / f'{name}.json'
        run(engine_command(args, config, path), cwd=ROOT, env=environment, check=True)
        reads[name] = decode_tok_s(path)
    ar, _ = reads['ar']
    method, outputs = reads['jetspec_plus_plus']
    result = dict(ar_decode_tok_s=ar, jetspec_plus_plus_decode_tok_s=method,
                  speedup=method / ar, prompts=len(outputs), tokens=args.tokens,
                  config=args.config, devices=args.devices,
                  knobs=args.knob,
                  baseline='graphed AR, same engine and GPU type, one GPU')
    (output / 'speedup.json').write_text(json.dumps(result, indent=2) + '\n')
    print(f"AR {ar:.1f} tok/s  JetSpec++ {method:.1f} tok/s  speedup {method / ar:.2f}x "
          f"({len(outputs)} prompts x {args.tokens} tokens)")
    print(f"JSON: {output / 'speedup.json'}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
