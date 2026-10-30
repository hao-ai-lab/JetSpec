"""Resolve local artifacts and run a bundled JetSpec++ serving configuration.

`generate PROMPT` answers one prompt with the pinned release checkpoint, downloading what is missing.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import tempfile


ROOT = Path(__file__).resolve().parent
# The engine's generic draft slot reads this pinned snapshot from SSD_HF_CACHE for every method
# without a draft model of its own (launch/runner.py _generic_draft_path).
GENERIC_DRAFT = ('Qwen/Qwen3-0.6B', 'c1899de289a04d12100db370d81485cdf75e47ca')
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
    ('qwen3.8-27b', 'ar', 'fast', 'sm100-aarch64'): 'ar-hybrid27b-fast-graphed',
    # Further schedules on every family, so speedup.py --schedule auto can measure which one
    # this GPU runs fastest; the release compositions above stay the defaults.
    ('qwen3-8b', 'j1-r1', 'fast', 'sm80'): dict(
        method='jetspecpp-j1r1-b16-serve', model='qwen3-8b-release', engine='fast-sm80',
        hardware='a100-pair', benchmark='gsm8k-128'),
    ('qwen3.8-27b', 'j1-r1', 'fast', 'sm100-aarch64'): dict(
        method='jetspecpp-j1r1-b16-serve', model='qwen3.8-27b-p2-cal28', engine='fast-sm100',
        hardware='gb200-pair', benchmark='gsm8k-128'),
}
# The fastest measured schedule when --schedule is not given.
DEFAULT_SCHEDULE = {('qwen3-8b', 'sm80'): 'j1-r0', ('qwen3-8b', 'sm100'): 'j1-r1',
                    ('qwen3-8b', 'sm100-aarch64'): 'j1-r1',
                    ('qwen3.8-27b', 'sm80'): 'j1-r1', ('qwen3.8-27b', 'sm100'): 'j1-r1',
                    ('qwen3.8-27b', 'sm100-aarch64'): 'j1-r1'}
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
    from .launch.entry import resolve_read
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
        rows = json.loads((ROOT / 'launch/recipe-index.json').read_text())['rows']
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
    if 'SSD_HF_CACHE' not in environment:
        environment['SSD_HF_CACHE'] = os.environ.get('SSD_HF_CACHE') or hub_cache()
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


def hub_cache():
    """The Hugging Face hub cache that `hf download` fills."""
    from huggingface_hub import constants
    return str(constants.HF_HUB_CACHE)


def ensure_generic_draft(cache):
    """Download the pinned generic draft snapshot into `cache` when it is not there yet."""
    from huggingface_hub import snapshot_download
    snapshots = Path(cache) / ('models--' + GENERIC_DRAFT[0].replace('/', '--')) / 'snapshots'
    if not (snapshots / GENERIC_DRAFT[1] / 'config.json').is_file():
        snapshot_download(GENERIC_DRAFT[0], revision=GENERIC_DRAFT[1], cache_dir=cache)


def alias_heads_bundle(host, recipe, directory):
    """Name a published heads bundle after its registration, which is where the reader looks for it.

    A checkpoint keeps its calibration in `calibration/`. A recipe that serves a sibling operating
    point of the bundle (a different bundle_name) still needs that sibling on disk.
    """
    key = recipe.get('bundle_artifact')
    path = host['artifacts'].get(key) if key else None
    if not path or recipe.get('bundle_name') != key or Path(path).name == key:
        return
    alias = Path(directory) / 'bundles' / key
    alias.parent.mkdir(parents=True, exist_ok=True)
    alias.symlink_to(Path(path).resolve(), target_is_directory=True)
    host['artifacts'][key] = str(alias)


def release_paths(model, target=None, checkpoint=None):
    """Local directories of the pinned target and JetSpec++ checkpoint, downloaded when not given."""
    from huggingface_hub import snapshot_download
    name = next(entry['model'] for key, entry in RECIPES.items() if key[0] == model and isinstance(entry, dict))
    config = json.loads((ROOT / 'config/model.json').read_text())[name]
    drafter = next(a for a in config['artifacts'] if a['name'] == config['drafter']['artifact'])
    if target is None:
        target = snapshot_download(config['target']['repository'], revision=config['target']['revision'])
    if checkpoint is None:
        checkpoint = snapshot_download(drafter['repository'], revision=drafter['revision'])
    return Path(target), Path(checkpoint)


def prompt_rows(target, text):
    """One chat-formatted prompt row, in the reader's prompts JSON layout."""
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(target)
    rendered = tokenizer.apply_chat_template([{'role': 'user', 'content': text}], tokenize=False,
                                             add_generation_prompt=True, enable_thinking=False)
    return {'count': 1, 'prompts': [{'index': 0, 'text': rendered,
                                     'token_ids': tokenizer(rendered, add_special_tokens=False).input_ids}]}


def generate(argv):
    """Answer one prompt with the pinned JetSpec++ release, then print the answer and its throughput."""
    result = argparse.ArgumentParser(prog='python -m jetspec.engines.jetspec_plus_plus generate',
                                     description=generate.__doc__)
    result.add_argument('prompt')
    result.add_argument('--model', choices=MODELS, default='qwen3-8b', help='target model')
    result.add_argument('--schedule', choices=('j1-r0', 'j1-r1'), help='defaults to the fastest measured one')
    result.add_argument('--max-tokens', type=int, default=2048)
    result.add_argument('--devices', default='0,1', help='two GPUs, as nvidia-smi numbers them')
    result.add_argument('--dataset', default='gsm8k', help='calibration entry of the heads')
    result.add_argument('--cache-root', default=os.environ.get('JSPP_CACHE_ROOT') or str(
        Path(os.environ.get('XDG_CACHE_HOME') or Path.home() / '.cache') / 'jetspec-plus-plus'),
        help='persistent compiler cache; the first run compiles for several minutes')
    result.add_argument('--target', help='local target directory (default: the pinned Hugging Face snapshot)')
    result.add_argument('--checkpoint', help='local JetSpec++ checkpoint (default: the pinned release)')
    args = result.parse_args(argv)
    target, checkpoint = release_paths(args.model, args.target, args.checkpoint)
    with tempfile.TemporaryDirectory(prefix='jetspec-pp-generate-') as directory:
        prompts, output = Path(directory) / 'prompt.json', Path(directory) / 'read.json'
        prompts.write_text(json.dumps(prompt_rows(target, args.prompt)))
        command = ['--model', args.model, '--target', str(target), '--drafter', str(checkpoint),
                   '--adapter', str(checkpoint / 'adapter'), '--heads-bundle', str(checkpoint / 'calibration'),
                   '--prompts-json', str(prompts), '--count', '1', '--tokens', str(args.max_tokens),
                   '--devices', args.devices, '--dataset', args.dataset, '--output-json', str(output),
                   '--cache-root', args.cache_root, '--stop-at-eos', 'on']
        if args.schedule:
            command += ['--schedule', args.schedule]
        status = main(command)
        if status:
            return status
        read = json.loads(output.read_text())
    answer = read['outputs'][0]
    measured = read['metrics']['serve_read']['measured']
    print(answer['text'])
    # The read's top-level throughput divides the token budget, not the generated tokens.
    print(f"\n[{len(answer['token_ids'])} tokens, decode {measured['decode_tok_s']:.1f} tok/s, "
          'first prompt of a fresh engine included]')
    return 0


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv[:1] == ['generate']:
        return generate(argv[1:])
    args = parser().parse_args(argv)
    recipe, reader_argv, host = resolve(args)
    from .launch import runner
    with tempfile.TemporaryDirectory(prefix='jetspec-pp-host-') as directory:
        alias_heads_bundle(host, recipe, directory)
        if not args.resolve_only:
            ensure_generic_draft(host['environment']['SSD_HF_CACHE'])
        path = Path(directory) / 'host.json'
        path.write_text(json.dumps(host))
        launch = runner._parser().parse_args(reader_argv + ['--host-values', str(path)])
        launch.resolved_recipe = recipe
        launch.config_read = args._record
        return runner._main(launch)
