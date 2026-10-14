"""Translate local artifact flags into the copied, pinned serving reader."""
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
}


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--config', help='a100-j1-r0, b200-j1-r1, reference-j1-r1 or ar (graphed autoregressive baseline); defaults by device architecture')
    for role in ('target', 'drafter', 'adapter', 'heads-bundle'):
        result.add_argument('--' + role, default=os.environ.get(
            'JSPP_' + role.upper().replace('-', '_')))
    result.add_argument('--target-release', default=os.environ.get('JSPP_TARGET_RELEASE'))
    result.add_argument('--prompts-json', required=True)
    result.add_argument('--tokens', type=int, default=64)
    result.add_argument('--placement', choices=('pair', 'one_gpu'), default=None,
                        help='pair for JetSpec++, one_gpu for ar (the default for each)')
    result.add_argument('--devices', default='0,1')
    result.add_argument('--output-json', required=True)
    result.add_argument('--host-values', default=os.environ.get('JSPP_HOST_VALUES'))
    result.add_argument('--dataset', default='gsm8k')
    result.add_argument('--count', type=int, default=2)
    result.add_argument('--cache-root', default=os.environ.get('JSPP_CACHE_ROOT'),
                        help='persistent compiler cache root (or JSPP_CACHE_ROOT); reuse for token parity')
    result.add_argument('--cpu-pin', default='auto')
    result.add_argument('--knob', action='append', default=[], metavar='NAME=VALUE',
                        help='set a registered serving knob for this read (repeatable)')
    result.add_argument('--stop-at-eos', choices=('on', 'off'), default='off',
                        help='end each prompt at EOS; off runs every prompt to --tokens')
    result.add_argument('--resolve-only', action='store_true')
    return result


def _device_arch(devices):
    import torch
    if not torch.cuda.is_available():
        return '8.0'
    major, minor = torch.cuda.get_device_capability(int(devices.split(',')[0]))
    return f'{major}.{minor}'


def resolve(args):
    if args.config is None:
        import torch
        major = torch.cuda.get_device_capability(int(args.devices.split(',')[0]))[0] if torch.cuda.is_available() else 8
        args.config = 'b200-j1-r1' if major >= 10 else 'a100-j1-r0'
    name = CONFIGS.get(args.config, args.config)
    if '/' in name or '\\' in name:
        raise ValueError('--config must name a bundled recipe')
    recipe = json.loads((ROOT / 'configs' / (name + '.json')).read_text())
    autoregressive = recipe['method'] == 'ar'
    if args.placement is None:
        args.placement = 'one_gpu' if autoregressive else 'pair'
    if autoregressive:
        args.devices = args.devices.split(',')[0]
    host = json.loads(Path(args.host_values).read_text()) if args.host_values else {
        'schema': 'jetspec.serve_host_values/1',
        'cuda_arch': _device_arch(args.devices) if autoregressive else recipe['cuda_arch'],
        'rendezvous_mode': 'host_port', 'environment': {}, 'artifacts': {}}
    environment = host.setdefault('environment', {})
    artifacts = host.setdefault('artifacts', {})
    if args.target:
        environment['JETSPEC_TARGET_MODEL_PATH'] = args.target
    environment.setdefault('SSD_HF_CACHE', os.environ.get('SSD_HF_CACHE',
        str(Path(args.prompts_json).resolve().parent / 'hf/hub')))
    environment.setdefault('SSD_DATASET_DIR', os.environ.get('SSD_DATASET_DIR',
        str(Path(args.prompts_json).resolve().parent)))
    for field, key in () if autoregressive else (('drafter', recipe['drafter_artifact']),
                       ('adapter', recipe.get('adapter_artifact', 'adapter_head')),
                       ('heads_bundle', recipe['bundle_artifact']),
                       ('target_release', recipe['target_release_artifact'])):
        value = getattr(args, field)
        if value:
            artifacts[key] = value
    if not autoregressive:
        artifacts.setdefault(recipe['target_release_artifact'], args.drafter)
    argv = ['--method', recipe['method'], '--regime', recipe['regime'], '--recipe', name,
            '--placement', args.placement, '--devices', args.devices,
            '--dataset', args.dataset, '--prompts-json', args.prompts_json,
            '--count', str(args.count), '--tokens', str(args.tokens), '--temperature', '0',
            '--sequential-prompts', '--output-json', args.output_json,
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
    _, reader_argv, host = resolve(args)
    from .serving import read
    with tempfile.TemporaryDirectory(prefix='jetspec-pp-host-') as directory:
        path = Path(directory) / 'host.json'
        path.write_text(json.dumps(host))
        return read._main(read._parser().parse_args(reader_argv + ['--host-values', str(path)]))
