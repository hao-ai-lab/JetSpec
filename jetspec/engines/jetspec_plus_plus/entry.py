"""Resolve local artifacts and run a bundled JetSpec++ serving configuration."""
import argparse
import json
import os
from pathlib import Path
import tempfile


ROOT = Path(__file__).resolve().parent
# (model, schedule, engine, kernel family) -> a validated serving composition (a dict of config
# names, as in the release commands) or a named recipe. The kernel family comes from the GPU
# compute capability and, for 10.x, the host CPU architecture, so no entry names a machine.
RECIPES = {
    ('qwen3-8b', 'j1-r0', 'fast', 'sm80'): dict(
        method='jetspecpp-j1r0-b16', model='qwen3-8b-release', engine='fast-sm80',
        hardware='a100-pair', benchmark='table3-gsm8k-first32'),
    ('qwen3-8b', 'j1-r1', 'fast', 'sm100'): dict(
        method='jetspecpp-j1r1-b16-serve', model='qwen3-8b-release', engine='fast-sm100',
        hardware='b200-pair', benchmark='gsm8k-128'),
    ('qwen3-8b', 'j1-r1', 'fast', 'sm100-aarch64'): dict(
        method='jetspecpp-j1r1-b16-floor4', model='qwen3-8b-release', engine='fast-sm100',
        hardware='gb200-pair', benchmark='gsm8k-128'),
    ('qwen3-8b', 'j1-r1', 'reference', 'sm80'): 'jetspecpp-reference-pair-e6v3-b16-chain-r4-j1r1-lean-swap-fast6-table3',
    ('qwen3-8b', 'j1-r1', 'reference', 'sm100'): 'jetspecpp-reference-pair-e6v3-b16-chain-r4-j1r1-lean-swap-fast6-table3',
    ('qwen3-8b', 'j1-r1', 'reference', 'sm100-aarch64'): 'jetspecpp-reference-pair-e6v3-b16-chain-r4-j1r1-lean-swap-fast6-table3',
    ('qwen3-8b', 'ar', 'fast', 'sm80'): 'ar-graphed',
    ('qwen3-8b', 'ar', 'fast', 'sm100'): 'ar-graphed',
    ('qwen3-8b', 'ar', 'fast', 'sm100-aarch64'): 'ar-graphed',
    ('qwen3.8-27b', 'j1-r1', 'fast', 'sm80'): dict(
        method='jetspecpp-j1r1-b16-serve', model='qwen3.8-27b-p2-cal28', engine='fast-sm80',
        hardware='a100-pair', benchmark='gsm8k-128'),
    ('qwen3.8-27b', 'j1-r1', 'fast', 'sm100'): dict(
        method='jetspecpp-j1r1-b16-serve', model='qwen3.8-27b-p2-cal28', engine='fast-sm100',
        hardware='b200-pair', benchmark='gsm8k-128'),
    ('qwen3.8-27b', 'ar', 'fast', 'sm80'): 'ar-hybrid27b-fast-graphed-sm80',
    ('qwen3.8-27b', 'ar', 'fast', 'sm100'): 'ar-hybrid27b-fast-graphed',
}
# The fastest measured schedule when --schedule is not given.
DEFAULT_SCHEDULE = {('qwen3-8b', 'sm80'): 'j1-r0', ('qwen3-8b', 'sm100'): 'j1-r1',
                    ('qwen3-8b', 'sm100-aarch64'): 'j1-r1',
                    ('qwen3.8-27b', 'sm80'): 'j1-r1', ('qwen3.8-27b', 'sm100'): 'j1-r1'}
MODELS = ('qwen3-8b', 'qwen3.8-27b')
SCHEDULES = ('j1-r0', 'j1-r1', 'ar')
ENGINES = ('fast', 'reference')


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--model', choices=MODELS, default='qwen3-8b', help='target model')
    result.add_argument('--schedule', choices=SCHEDULES,
                        help='J/R schedule, or ar for the autoregressive baseline; defaults to the fastest measured one')
    result.add_argument('--engine', choices=ENGINES, default='fast', help='fast (CUDA graphs) or reference verification')
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


def kernel_family(arch, machine=None):
    """The kernel family: sm80 for 8.x, sm100 for 10.x, sm100-aarch64 for 10.x on an Arm host."""
    import platform
    major = int(str(arch).split('.')[0])
    if major < 8:
        raise SystemExit(f'compute capability {arch} is below the supported 8.0')
    if major < 10:
        return 'sm80'
    return 'sm100-aarch64' if (machine or platform.machine()) == 'aarch64' else 'sm100'


def resolve(args):
    from .read.entry import resolve_read
    host = json.loads(Path(args.host_values).read_text()) if args.host_values else None
    arch = host.get('cuda_arch') if host else None
    arch = arch or _device_arch(args.devices)
    family = kernel_family(arch, host.get('machine') if host else None)
    if args.schedule is None:
        args.schedule = DEFAULT_SCHEDULE[(args.model, family)]
    key = (args.model, args.schedule, args.engine, family)
    if key not in RECIPES:
        supported = sorted(f'--model {m} --schedule {s} --engine {e}'
                           for m, s, e, f in RECIPES if f == family)
        raise SystemExit(f'no validated recipe for {key} on compute capability {arch}; '
                         'supported here: ' + '; '.join(supported))
    entry = RECIPES[key]
    name = entry if isinstance(entry, str) else None
    # Adaptation: artifact paths and byte pins select local files; unread repository labels are neutral.
    if name is None:
        row = dict(names=dict(entry), overrides=[], launch=None)
    else:
        rows = json.loads((ROOT / 'read/recipe-index.json').read_text())['rows']
        row = next(row for row in rows if row['launch']['name'] == name)
    autoregressive = args.schedule == 'ar'
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
    name = name or recipe['name']
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
        # Release models take the embedding and LM head from the target, so they name no release artifact.
        release = recipe.get('target_release_artifact')
        for field, key in (('drafter', recipe['drafter_artifact']),
                           ('adapter', recipe['adapter_artifact']),
                           ('heads_bundle', recipe['bundle_artifact']),
                           ('target_release', release)):
            value = getattr(args, field)
            if value and key:
                host['artifacts'][key] = value
        if args.drafter and release:
            host['artifacts'].setdefault(release, args.drafter)
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
