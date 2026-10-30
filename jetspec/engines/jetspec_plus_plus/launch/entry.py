# Adaptation: package relocation and deferred optional dependencies.
"""CPU resolution and the launch of a manifest-pinned read."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import time

from jetspec.engines.jetspec_plus_plus.draft.method.config import CATEGORIES, canonical_json, config_hash
from jetspec.engines.jetspec_plus_plus.launch_env import environment_name, serving_environment, apply_serving_profile
from .resolution import ROOT, resolve

FAMILIES = {'dspark': 'blockhead', 'fanout-tree': 'ssd', 'domino': 'spec_sync',
            'jetspec-sync': 'jetspec_sync'}


# Launcher metadata retained from the base; config overrides remain authoritative.
_COMPOSED = json.loads((ROOT / 'config/composed.json').read_text())
COMPOSED_BASE_ENVIRONMENTS = _COMPOSED['base_environments']
COMPOSED_RECIPES = _COMPOSED['recipes']


def profile_environment(config):
    """A named profile owns selectors; compatibility leaves own other values."""
    from jetspec.engines.jetspec_plus_plus.profiles import get_profile, legacy_names, legacy_selector_names, TRANSPORT_SWITCHES
    from jetspec.engines.jetspec_plus_plus.runtime.env import REQUIRED
    from jetspec.engines.jetspec_plus_plus.runtime.env import NO_DEFAULT
    from jetspec.engines.jetspec_plus_plus.runtime.env import legacy_registry as _legacy_registry
    REGISTRY = _legacy_registry()
    result = serving_environment(config)
    name = config['engine'].get('profile', 'current')
    if name == 'current':
        return result
    profile = get_profile(name)
    selected = legacy_names(profile.kernels)
    for key in legacy_selector_names():
        if key.removeprefix('JETSPEC_').lower() in TRANSPORT_SWITCHES:
            result[key] = str(int(key.removeprefix('JETSPEC_').lower() in profile.transport))
        else:
            default = ('0' if REGISTRY[key].kind == 'flag'
                       and REGISTRY[key].default not in (REQUIRED, NO_DEFAULT)
                       else REGISTRY[key].default)
            value = selected.get(key, default)
            # Spell only a value that differs from the registry default. An unset selector reads as
            # that default, and some launch contracts require it unset (JETSPEC_CHAIN_GRAPH=seg
            # refuses any JETSPEC_ROWS_FUSED).
            if value in (REQUIRED, NO_DEFAULT) or value == REGISTRY[key].default:
                result.pop(key, None)
            else:
                result[key] = value
    return _without_inert_dependents(result, set(legacy_selector_names()))


def _without_inert_dependents(environment, selectors):
    """Unset each option that a selector left off by the profile makes inert.

    Engine entries carry fast-path options (for example JETSPEC_CTX_KV_DIRECT) that act only with a
    fast kernel. Under a profile that leaves that kernel off, the launcher refuses them as inert, and
    the sealed recipes left them unset. The launcher's own requirement rules name them."""
    from jetspec.engines.jetspec_plus_plus.runtime.env import inert_requirements
    result = dict(environment)
    while True:
        inert = {rule.name for rule in inert_requirements(result)
                 if rule.flag in selectors and rule.name not in selectors}
        if not inert:
            return result
        for name in sorted(inert):
            result.pop(name)


def _recipe_environment(environment):
    """Project canonical compatibility fields through the existing legacy binding."""
    from jetspec.engines.jetspec_plus_plus.launch_env import environment_name
    return {(environment_name(*key.split('.', 1)) if '.' in key else key): value
            for key, value in environment.items()}


def serving_recipe(config, *, root=ROOT):
    """Select artifact/launcher metadata by the resolved model and method."""
    method, model, hardware = config['method'], config['model'], config['hardware']
    family = FAMILIES.get(method['family'], method['family'])
    # These release registrations retain the original serving bytes and metadata.
    release_sources = {
        'jetspecpp_e6_drafter_release': 'jetspecpp_e6_drafter',
        'jetspecpp_e6_adapter_release': 'jetspecpp_e6_adapter',
        'qwen3p8_27b_drafter_main6ep_ep6_release': 'qwen3p8_27b_drafter_main6ep_ep6',
    }
    # qwen3-8b-release and qwen3.8-27b-p2-cal28 scrub provenance only;
    # both composed and named recipes retain their original threshold pins.
    release_thresholds = {
        ('jetspecpp_e6_drafter_release',
         '1fa70faa55a507bb132b46d23b0d340aa400931b428cce81eba5da0fbbe0c964'):
            '1fa70faa55a507bb132b46d23b0d340aa400931b428cce81eba5da0fbbe0c964',
        ('qwen3p8_27b_drafter_main6ep_ep6_release',
         'd08f118243db1a6755fbd64bdf4387bcb32802c78d343431355409d25f501994'):
            'd08f118243db1a6755fbd64bdf4387bcb32802c78d343431355409d25f501994',
    }
    thresholds_sha256 = release_thresholds.get(
        (model.get('drafter', {}).get('artifact'), model.get('thresholds_sha256')),
        model.get('thresholds_sha256'))
    candidates = []
    named = config['engine'].get('recipe')
    if named:
        from .recipes import recipe_index
        rows = sorted(recipe_index(root), key=lambda row: row['path'])
    else:
        rows = [dict(kind='serve', launch={key: value for key, value in recipe.items()
                                         if key not in ('environment', 'unset')},
                     environment=_recipe_environment({**{key: value for key, value in
                                     COMPOSED_BASE_ENVIRONMENTS[recipe['method']].items()
                                     if key not in recipe['unset']}, **recipe['environment']}))
                for recipe in COMPOSED_RECIPES]
    for row in rows:
        if row['kind'] != 'serve':
            continue
        recipe = dict(row['launch'], environment=row['environment'])
        if config['engine'].get('recipe') and recipe['name'] != config['engine']['recipe']:
            continue
        if recipe.get('method') != family:
            continue
        if family == 'jetspecpp':
            if (recipe.get('thresholds_sha256') != thresholds_sha256
                    or recipe.get('drafter_artifact') != release_sources.get(
                        model.get('drafter', {}).get('artifact'), model.get('drafter', {}).get('artifact'))
                    or recipe.get('adapter_artifact') != release_sources.get(
                        model.get('adapter', {}).get('artifact'), model.get('adapter', {}).get('artifact'))
                    or recipe.get('regime') != 'graphed'):
                continue
        candidates.append(recipe)
    if not candidates:
        raise ValueError(f'no legacy serving metadata for {family} and the resolved model')
    # Favor the paper's uncut schedule metadata; the resolved config owns all env keys.
    schedule = method.get('schedule', {})
    tag = f"j{schedule.get('j', 1)}r{schedule.get('r', 1)}"
    candidates.sort(key=lambda recipe: (('reference' in recipe['name']) != (config['engine']['kind'] == 'reference'),
                                        recipe['name'].startswith('eval') != (not config['engine']['cutoffs']),
                                        tag not in recipe['name'],
                                        f"b{method['block_size']}" not in recipe['name'],
                                        recipe.get('cuda_arch') != {'a100': '8.0', 'b200': '10.0',
                                                                    'gb200': '10.0'}[hardware['gpu']],
                                        recipe['name']))
    recipe = deepcopy(candidates[0])
    if family == 'jetspecpp':
        for role in ('drafter', 'adapter'):
            recipe[role + '_artifact'] = model[role]['artifact']
        release_bundles = {
            'jetspecpp_e6_drafter_release': 'jetspecpp_e6_calibration_release',
            'qwen3p8_27b_drafter_main6ep_ep6_release': 'qwen3p8_27b_calibration_release',
        }
        if bundle := release_bundles.get(recipe['drafter_artifact']):
            recipe['bundle_artifact'] = recipe['bundle_name'] = bundle
            recipe['thresholds_sha256'] = model['thresholds_sha256']
    recipe['environment'] = apply_serving_profile(
        {**recipe['environment'], **profile_environment(config)}, config)
    # Current release profiles borrow named metadata, but the method owns their
    # pass cap. Sealed profiles and recipe replays keep their recorded passes.
    release = (config['engine'].get('profile', 'current') == 'current'
               and config['engine'].get('release_finished_requests', False))
    if family == 'jetspecpp' and (not named or release):
        recipe['passes'] = schedule['j'] + 1
    recipe['cuda_arch'] ={'a100': '8.0', 'b200': '10.0', 'gb200': '10.0'}[hardware['gpu']]
    return recipe


def build_plan(config, *, launch=None, root=ROOT, config_root=None, output='read.json',
               sequential_prompts=False, cache_root=None, wire_records=None, anatomy=None):
    """Return the environment and legacy argv without probing host artifacts or GPUs."""
    config_root = Path(config_root or Path(root) / 'config')
    recipe = serving_recipe(config, root=root) if launch is None else deepcopy(launch)
    # Model declarations own new compositions; sealed recipe launches keep their source.
    if (launch is None and recipe['method'] == 'jetspecpp'
            and config['model'].get('target_release', {}).get('slice') is not None):
        recipe['target_release_slice'] = deepcopy(config['model']['target_release']['slice'])
        recipe.pop('target_release_artifact', None)
    if launch is not None:
        recipe['environment'] = profile_environment(config)
    benchmark, hardware = config['benchmark'], config['hardware']
    prompt = next((item for item in benchmark.get('prompt_files', [])
                   if item['temperature'] == benchmark.get('temperature', 0)),
                  next(iter(benchmark.get('prompt_files', [])), None))
    if prompt is None:
        raise ValueError('serving benchmark requires a pinned prompt file')
    argv = ['--method', recipe['method'], '--regime', recipe['regime'],
            '--recipe', recipe['name'], '--placement', hardware['placement'],
            '--devices', ','.join(map(str, hardware['devices'])),
            '--dataset', benchmark['dataset'], '--prompts-json',
            str(config_root / 'prompts' / prompt['file']),
            '--count', str(benchmark['count']), '--tokens', str(benchmark['cap']),
            '--temperature', str(benchmark['temperature']), '--seed', str(benchmark['seed']),
            '--repeats', str(benchmark['repeats']), '--output-json', str(output),
            '--cpu-pin', hardware['cpu_pin']]
    from jetspec.engines.jetspec_plus_plus.run_config import RunConfig
    run_config = RunConfig.from_config(config)
    if sequential_prompts:
        if run_config.concurrency != 1:
            raise ValueError('--sequential-prompts requires benchmark.concurrency=1')
        argv.append('--sequential-prompts')
    if cache_root is not None:
        argv.extend(['--cache-root', str(cache_root)])
    for flag, value in (('--wire-records', wire_records), ('--anatomy', anatomy)):
        if value is not None:
            if value not in ('on', 'off'):
                raise ValueError(f'{flag} must be on or off')
            argv.extend([flag, value])
    if run_config.concurrency > 1 and not run_config.batch_capacity.supports_batch:
        profile = config['engine'].get('profile', 'current')
        raise ValueError(f'engine profile {profile!r} is missing supports_batch=true '
                         f'for benchmark.concurrency={run_config.concurrency}')
    if benchmark['stop_at_eos']:
        argv.extend(['--stop-at-eos', 'on'])
    if hardware.get('mps'):
        argv.extend(['--mps-root', str(Path(output).absolute().parent / 'mps')])
    if run_config.concurrency > 1:
        argv.extend(['--batch', str(run_config.concurrency)])
    return dict(recipe=recipe, environment=recipe['environment'], serve_argv=argv,
                launch_overlay=run_config.batch_overlay())


def resolve_read(names, overrides=(), *, root=ROOT / 'config', launch=None, output='read.json',
                 sequential_prompts=False, cache_root=None, wire_records=None, anatomy=None):
    config = resolve(names, overrides, root=root)
    plan = build_plan(config, launch=launch, root=Path(root).parent, config_root=root, output=output,
                      sequential_prompts=sequential_prompts, cache_root=cache_root,
                      wire_records=wire_records, anatomy=anatomy)
    return dict(schema='jetspec.config_read/1', config=config, canonical_json=canonical_json(config),
                config_sha256=config_hash(config), launched=False, **plan)


def granted_devices(devices, visible):
    """The hardware entry's device indices count from the first card that the host grants.

    A scheduler that grants cards through CUDA_VISIBLE_DEVICES (indices or uuids) shifts them: index 0
    names the first granted card. The launcher reads physical cards, so the indices map here."""
    granted = [item.strip() for item in (visible or '').split(',') if item.strip()]
    if not granted:
        return devices
    indices = [int(item) for item in devices.split(',')]
    if max(indices) >= len(granted):
        raise ValueError(f'hardware devices {devices} exceed the granted cards {visible!r}')
    return ','.join(granted[index] for index in indices)


def write_eval_records(record, *, process_wall_seconds):
    """Write the engine's evaluation records for a read without cutoffs."""
    # Adaptation: retain the public error for incomplete record requests.
    if 'environment' not in record:
        raise ValueError('eval records need the environment of the read')
    from jetspec.engines.jetspec_plus_plus.analysis.eval_records import write_eval_records as writer
    environment = record['environment']
    method = record['config']['method']
    values = dict(
        rows=environment.get(environment_name('method', 'method_settings.eval_rows'), method.get('cache_rows', {}).get('cap', 16)),
        drafting_mode=environment.get(environment_name('engine', 'engine_settings.eval_mode'), 'off'),
        j3_label=environment.get(environment_name('method', 'method_settings.eval_j3_label'), 'J3'),
        draft_delay_ms=environment.get(environment_name('engine', 'engine_settings.eval_draft_delay_ms'), '0'),
        host_fast=environment.get(environment_name('engine', 'engine_settings.host_fast'), '0'),
        refresh_mode=environment.get(environment_name('method', 'method_settings.eval_refresh_mode'), 'full'),
        capture_round_candidates=environment.get(environment_name('engine', 'engine_settings.eval_capture_round_candidates'), '0'))
    output = record['serve_argv'][record['serve_argv'].index('--output-json') + 1]
    return writer(record['config'], recipe_method=record['recipe']['method'],
                               output=output, eval_values=values,
                               process_wall_seconds=process_wall_seconds)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('settings', nargs='+', help='category=entry or category.key=JSON')
    parser.add_argument('--resolve-only', action='store_true')
    parser.add_argument('--output', default='read.json')
    parser.add_argument('--host-values')
    parser.add_argument('--artifacts-dir')
    parser.add_argument('--sequential-prompts', action='store_true',
                        help='serve C=1 prompts sequentially in one loaded process')
    parser.add_argument('--cache-root', help='root for the serving child compiler caches')
    parser.add_argument('--wire-records', choices=('on', 'off'),
                        help='write proposal wire records (serving default: on)')
    parser.add_argument('--anatomy', choices=('on', 'off'),
                        help='write per-read host anatomy (serving default: on)')
    parser.add_argument('--child-wall-seconds', type=float,
                        help='wall limit of the serve child, compile included (default: the launcher\'s)')
    args = parser.parse_args(argv)
    names, overrides = {}, []
    for setting in args.settings:
        key, separator, value = setting.partition('=')
        if not separator:
            parser.error(f'invalid setting {setting!r}')
        if key in CATEGORIES:
            if key in names:
                parser.error(f'duplicate category {key}')
            names[key] = value
        else:
            overrides.append(setting)
    try:
        record = resolve_read(names, overrides, output=args.output,
                              sequential_prompts=args.sequential_prompts, cache_root=args.cache_root,
                              wire_records=args.wire_records, anatomy=args.anatomy)
        from jetspec.engines.jetspec_plus_plus.engine import Engine
        Engine.from_config(record['config'], sealed_record=args.resolve_only)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    if args.resolve_only:
        print(json.dumps(record, sort_keys=True))
        return
    from . import runner
    serve_argv = list(record['serve_argv'])
    devices = serve_argv.index('--devices') + 1
    try:
        serve_argv[devices] = granted_devices(serve_argv[devices], os.environ.get('CUDA_VISIBLE_DEVICES'))
    except ValueError as error:
        parser.error(str(error))
    for flag, value in (('--host-values', args.host_values), ('--artifacts-dir', args.artifacts_dir),
                        ('--child-wall-seconds', args.child_wall_seconds)):
        if value is not None:
            serve_argv.extend([flag, str(value)])
    launch_args = runner._parser().parse_args(serve_argv)
    launch_args.resolved_recipe = record['recipe']
    launch_args.config_read = record
    started = time.perf_counter()
    runner._main(launch_args)
    if not record['config']['engine']['cutoffs']:
        write_eval_records(record, process_wall_seconds=time.perf_counter() - started)
