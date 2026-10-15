"""Resolve local artifacts and run a bundled JetSpec++ serving configuration."""
import argparse
import json
import os
from pathlib import Path
import tempfile


ROOT = Path(__file__).resolve().parent
CONFIGS = {
    'a100-j1-r0': 'jetspecpp-graphed-pair-e6-b16-chain-r4-j1-lean-sm80',
    'b200-j1-r1': 'jetspecpp-graphed-pair-e6v3-b16-chain-r4-j1r1-lean-swap-fast6',
    'reference-j1-r1': 'jetspecpp-reference-pair-e6v3-b16-chain-r4-j1r1-lean-swap-fast6-table3',
    'ar': 'ar-graphed',
    'a100-27b-j1-r1': 'jetspecpp-graphed-pair-27b-base800k-cal28-b16-j1r1-sm80',
    'b200-27b-j1-r1': 'jetspecpp-graphed-pair-27b-drafter6ep-base800k-j1r1-swap-mr-fast',
    'ar-27b': 'ar-hybrid27b-fast-graphed-sm80',
}


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--config', choices=tuple(CONFIGS), help='configuration; defaults by GPU architecture')
    for role in ('target', 'drafter', 'adapter', 'heads-bundle', 'target-release', 'host-values'):
        result.add_argument('--' + role, default=os.environ.get('JSPP_' + role.upper().replace('-', '_')))
    result.add_argument('--prompts-json', required=True)
    result.add_argument('--tokens', type=int, default=64)
    result.add_argument('--placement', choices=('pair', 'one_gpu'))
    result.add_argument('--devices', default='0,1')
    result.add_argument('--output-json', required=True)
    result.add_argument('--dataset', default='gsm8k')
    result.add_argument('--count', type=int, default=2)
    result.add_argument('--cache-root', default=os.environ.get('JSPP_CACHE_ROOT'))
    result.add_argument('--cpu-pin', default='auto')
    result.add_argument('--knob', action='append', default=[], metavar='NAME=VALUE')
    result.add_argument('--stop-at-eos', choices=('on', 'off'), default='off')
    result.add_argument('--resolve-only', action='store_true')
    return result


def _device_arch(devices):
    import torch
    if not torch.cuda.is_available():
        return '8.0'
    first = devices.split(',')[0]
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')
    device = visible.index(first) if first in visible else int(first)
    major, minor = torch.cuda.get_device_capability(device)
    return f'{major}.{minor}'


def resolve(args):
    from .read.entry import resolve_read
    host = json.loads(Path(args.host_values).read_text()) if args.host_values else None
    arch = host.get('cuda_arch') if host else None
    arch = arch or _device_arch(args.devices)
    if args.config is None:
        args.config = 'b200-j1-r1' if int(arch.split('.')[0]) >= 10 else 'reference-j1-r1'
    name = CONFIGS[args.config]
    if args.config == 'ar-27b' and int(arch.split('.')[0]) >= 10:
        name = 'ar-hybrid27b-fast-graphed'
    # Adaptation: artifact paths and byte pins select local files; unread repository labels are neutral.
    rows = json.loads((ROOT / 'read/recipe-index.json').read_text())['rows']
    row = next(row for row in rows if row['launch']['name'] == name)
    autoregressive = row['launch']['method'] == 'ar'
    if args.placement is None:
        args.placement = 'one_gpu' if autoregressive else 'pair'
    if autoregressive:
        args.devices = args.devices.split(',')[0]
    overrides = row['overrides'] + [
        'hardware.placement=' + json.dumps(args.placement),
        'hardware.devices=' + json.dumps(list(range(len(args.devices.split(','))))),
        'benchmark.count=' + str(args.count), 'benchmark.cap=' + str(args.tokens),
        'benchmark.stop_at_eos=' + str(args.stop_at_eos == 'on').lower()]
    record = resolve_read(row['names'], overrides, root=ROOT / 'config',
                          launch=row['launch'], output=args.output_json)
    args._record = record
    recipe = record['recipe']
    host = host if host is not None else {
        'schema': 'jetspec.serve_host_values/1', 'cuda_arch': arch,
        'rendezvous_mode': 'host_port', 'environment': {}, 'artifacts': {}}
    environment = host.setdefault('environment', {})
    if args.target:
        environment['JETSPEC_TARGET_MODEL_PATH'] = args.target
    environment.setdefault('SSD_HF_CACHE', os.environ.get('SSD_HF_CACHE',
        str(Path(args.prompts_json).resolve().parent / 'hf/hub')))
    environment.setdefault('SSD_DATASET_DIR', os.environ.get('SSD_DATASET_DIR',
        str(Path(args.prompts_json).resolve().parent)))
    names = {row['name'] for row in json.loads((ROOT / 'artifacts/manifest.json').read_text())['artifacts']}
    host['artifacts'] = {key: value for key, value in host.get('artifacts', {}).items() if key in names}
    if not autoregressive:
        for field, key in (('drafter', recipe['drafter_artifact']),
                           ('adapter', recipe['adapter_artifact']),
                           ('heads_bundle', recipe['bundle_artifact']),
                           ('target_release', recipe['target_release_artifact'])):
            value = getattr(args, field)
            if value:
                host['artifacts'][key] = value
        if args.drafter:
            host['artifacts'].setdefault(recipe['target_release_artifact'], args.drafter)
    argv = ['--method', recipe['method'], '--regime', recipe['regime'], '--recipe', name,
            '--placement', args.placement, '--devices', args.devices, '--dataset', args.dataset,
            '--prompts-json', args.prompts_json, '--count', str(args.count), '--tokens', str(args.tokens),
            '--temperature', '0', '--sequential-prompts', '--output-json', args.output_json,
            '--cpu-pin', args.cpu_pin, '--anatomy', 'off', '--wire-records', 'off',
            '--stop-at-eos', args.stop_at_eos]
    if args.cache_root:
        argv += ['--cache-root', args.cache_root]
    for knob in args.knob:
        argv += ['--knob', knob]
    if args.resolve_only:
        argv += ['--resolve-only']
    return recipe, argv, host


def main(argv=None):
    args = parser().parse_args(argv)
    recipe, reader_argv, host = resolve(args)
    from .read import legacy
    with tempfile.TemporaryDirectory(prefix='jetspec-pp-host-') as directory:
        path = Path(directory) / 'host.json'
        path.write_text(json.dumps(host))
        launch = legacy._parser().parse_args(reader_argv + ['--host-values', str(path)])
        launch.resolved_recipe = recipe
        launch.config_read = args._record
        return legacy._main(launch)
