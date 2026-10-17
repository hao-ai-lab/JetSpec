"""Data-driven compatibility bindings for the legacy serving environment."""
import json
from pathlib import Path

from jetspec.engines.jetspec_plus_plus.draft.method.config import CATEGORIES, flatten
from jetspec.engines.jetspec_plus_plus.draft.method.config_schemas import _field, METHOD_EXECUTION_DEFAULTS
from jetspec.engines.jetspec_plus_plus.profiles import RETIRED_SELECTORS

ROOT = next((parent for parent in Path(__file__).resolve().parents
             if (parent / 'artifacts/manifest.json').is_file()), Path.cwd())
_BINDINGS = json.loads(Path(__file__).with_suffix('.json').read_text())
ENV_FIELDS = _BINDINGS['fields']
# Historical defaults still project to the same launch record. These bindings
# are compatibility inputs only; the profile adapter refuses their nondefaults.
ENV_FIELDS['engine'].update({'JETSPEC_' + name.upper(): 'engine_settings.' + name
                             for name in RETIRED_SELECTORS})
_ATOMIC = ('artifacts', 'thresholds', 'decoder_slice', 'prompt_files',
           'model_execution', 'method_execution',
           'optimization.stage_settings')


def release_recipe(resolved):
    """Bind a qualified model release to its original launcher metadata.

    Only released current fast profiles opt in. Other schedules, reference
    profiles and explicit recipe overrides retain their composition semantics.
    """
    engine, method = resolved['engine'], resolved['method']
    if (engine.get('recipe') or engine.get('kind') != 'fast'
            or not engine.get('cutoffs', True) or engine.get('profile', 'current') != 'current'
            or not engine.get('release_finished_requests') or method.get('family') != 'jetspecpp'):
        return engine.get('recipe')
    architecture = 'sm80' if resolved['hardware']['gpu'] == 'a100' else 'sm100'
    for row in resolved['model'].get('model_execution', {}).get('release_recipes', ()):
        if (row['architecture'] == architecture and row['block_size'] == method['block_size']
                and row['schedule'] == method.get('schedule')):
            return row['recipe']
    return None


def execution_settings(resolved):
    """Lower layer-owned qualifications; explicit engine leaves always win.

    Replay conditions retain their recorded defaults. New fast compositions use
    model capabilities and schedule defaults without consulting checkpoint files.
    """
    engine = resolved['engine']
    explicit = engine.get('engine_settings', {})
    if (engine.get('kind') not in ('fast', 'reference') or 'model' not in resolved
            or not resolved.get('method', {}).get('family')):
        return dict(explicit)
    bundle = any(file['destination']['value'] == 'thresholds.json'
                 for artifact in resolved['model'].get('artifacts', ())
                 for file in artifact.get('files', ()))
    legacy = (not engine.get('cutoffs', True)
              or engine.get('profile', 'current') != 'current' or bundle
              or resolved['method']['family'] != 'jetspecpp')
    result = dict(METHOD_EXECUTION_DEFAULTS)
    for category, field in _BINDINGS['execution_layers'].items():
        layer = resolved[category].get(field, {})
        result.update(layer.get('legacy', layer.get('defaults', {})) if legacy
                      else layer.get('defaults', {}))
        if ((not legacy or engine.get('recipe')) and engine['kind'] == 'fast'
                and engine.get('profile', 'current') == 'current'
                and resolved['method']['family'] == 'jetspecpp'):
            architecture = 'sm80' if resolved['hardware']['gpu'] == 'a100' else 'sm100'
            result.update(layer.get(architecture, {}))
    result.update(explicit)
    capabilities = resolved['model'].get('model_execution', {}).get('capabilities', {})
    if not legacy and capabilities.get('stateful') and result.get('c1_append_fold') == '1':
        raise ValueError('model capabilities do not qualify dense C1 append fold')
    return result


def environment_from_config(resolved):
    from jetspec.engines.jetspec_plus_plus.profiles import RETIRED_SELECTORS, refuse_retired
    retired = {'JETSPEC_' + name.upper(): value
               for name, value in resolved['engine'].get('engine_settings', {}).items()
               if name in RETIRED_SELECTORS}
    refuse_retired(retired)
    result = {environment_name('engine', 'engine_settings.' + name): value
              for name, value in execution_settings(resolved).items() if value is not None}
    for category in CATEGORIES:
        fields = flatten(resolved[category], atomic=_ATOMIC)
        for old, field in ENV_FIELDS[category].items():
            if field in fields and fields[field] is not None:
                result[old] = fields[field]
    result.update(retired)
    return result


def translate_environment(row, *, config_root=ROOT / 'config'):
    from jetspec.engines.jetspec_plus_plus.read.resolution import resolve
    return serving_environment(resolve(row['names'], row['overrides'], root=config_root))


def semantic_overrides(environment, config):
    """Represent recipe values with the structured keys that own launch behavior."""
    result = {}
    # Recipe registration must compare against layer defaults as well as the
    # catalog's explicit engine leaves. Registry defaults alone are insufficient.
    for category, field in _BINDINGS['execution_layers'].items():
        layer = config[category].get(field, {})
        for group, settings in layer.items():
            if group in ('capabilities', 'release_recipes'):
                continue
            for name in settings:
                path = 'engine_settings.' + name
                result['engine.' + path] = environment.get(environment_name('engine', path))
    computed = _BINDINGS['computed']
    numeric = {_field('method', 'block_size'): int, _field('method', 'cache_rows', 'cap'): int,
               _field('method', 'anchor', 'cap'): int, _field('method', 'anchor', 'coverage'): float,
               _field('hardware', 'kv_share'): float}
    for path, (name, conversion) in _BINDINGS['semantic'].items():
        if name not in environment or environment[name] is None:
            continue
        category, field = path.split('.', 1)
        base = flatten(config[category], atomic=_ATOMIC).get(field)
        value = environment[name]
        if conversion == 'bool':
            value = value == '1'
        elif conversion == 'minus_one':
            value = int(value) + 1
        elif path in numeric or type(base) in (int, float):
            value = numeric.get(path, type(base))(value)
        result[path] = value
    if config['method']['family'] == 'jetspecpp':
        tag = environment.get(computed['fixed_config'])
        if tag:
            j, r = tag[1:].split('-R')
            result.update({_field('method', 'schedule', 'j'): int(j), _field('method', 'schedule', 'r'): int(r)})
        if computed['miss_redraft'] in environment and environment[computed['miss_redraft']] is not None:
            result[_field('method', 'miss_redraft')] = environment[computed['miss_redraft']] == '1'
        if computed['deadline'] in environment and environment[computed['deadline']] is not None:
            result[_field('engine', 'deadline')] = environment[computed['deadline']] == '1'
    preserved = set(value[0] for value in _BINDINGS['semantic'].values())
    preserved.update(computed[name] for name in ('eval_rows', 'rows_cap', 'eval_mode'))
    for category, fields in ENV_FIELDS.items():
        for name, field in fields.items():
            if name in preserved and (name not in environment or environment[name] is None
                                       or name in (computed['eval_rows'], computed['rows_cap'], computed['eval_mode'])):
                result[category + '.' + field] = environment.get(name)
    return [key + '=' + json.dumps(value) for key, value in result.items()]


def serving_environment(resolved, *, rank_only=False):
    """Bind semantic keys, then preserve explicit historical compatibility leaves."""
    historical = environment_from_config(resolved)
    declared = {name: fields[field] for category in CATEGORIES
                for fields in (flatten(resolved[category], atomic=_ATOMIC),)
                for name, field in ENV_FIELDS[category].items() if field in fields}
    result = dict(historical)
    for path, (name, conversion) in _BINDINGS['semantic'].items():
        category, field = path.split('.', 1)
        value = flatten(resolved[category], atomic=_ATOMIC).get(field)
        if value is not None:
            value = int(value) if conversion == 'bool' else value - 1 if conversion == 'minus_one' else value
            result[name] = str(value)
    if rank_only:
        return result
    computed = _BINDINGS['computed']
    engine, method = resolved['engine'], resolved['method']
    if method['family'] == 'jetspecpp':
        result[computed['eval_mode']] = historical.get(computed['eval_mode'],
            'pipelined' if not engine['cutoffs'] and engine['deterministic'] else 'off')
        cap = method.get('cache_rows', {}).get('cap')
        if cap is not None:
            result[computed['eval_rows']] = historical.get(computed['eval_rows'], str(cap))
            result[computed['rows_cap']] = historical.get(computed['rows_cap'], str(cap))
        result[computed['deadline']] = str(int(engine['cutoffs'] and engine['deadline']))
        schedule = method.get('schedule')
        if schedule:
            result[computed['fixed_config']] = f"J{schedule['j']}-R{schedule['r']}"
        result[computed['miss_redraft']] = str(int(method['miss_redraft']))
        result[_BINDINGS['semantic'][_field('method', 'chain_miss_redraft')][0]] = str(int(
            method.get('chain_miss_redraft', method['miss_redraft'])))
        if not engine['cutoffs']:
            result[computed['redraft_max_ms']] = historical.get(computed['redraft_max_ms'], 'inf')
        result[computed['verify_reference']] = str(int(engine['kind'] == 'reference'))
        if computed['engine'] in historical:
            result[computed['engine']] = ('reference' if engine['kind'] == 'reference' else
                                          'fast' if engine['cutoffs'] else 'eval')
    # A nullable historical leaf records an intentionally absent recipe value.
    for name, value in declared.items():
        if value is None:
            result.pop(name, None)
    return result


def apply_serving_profile(environment, resolved):
    result = dict(environment)
    engine = resolved['engine']
    if resolved['model']['target']['family'] == 'hybrid':
        result.update(_BINDINGS['profiles']['hybrid'])
    if resolved['engine']['kind'] == 'reference':
        result.update(_BINDINGS['profiles']['reference'])
    if engine.get('profile', 'current') == 'current' and not engine.get('recipe'):
        # Refresh belongs to the selected schedule, including new R0 compositions.
        schedule = resolved['method'].get('schedule')
        if schedule and schedule['r'] == 0:
            from jetspec.engines.jetspec_plus_plus.jetspec.env import inert_requirements
            disabled = {environment_name('engine', 'engine_settings.' + field)
                        for field in ('refresh_swap', 'refresh_inject_inclusive')}
            result.update(dict.fromkeys(disabled, '0'))
            disabled.update(name for name, value in result.items() if value in ('0', 'off'))
            # Remove transitive dependents too (for example swap -> lean -> write).
            while True:
                inert = {rule.name for rule in inert_requirements(result)
                         if rule.flag in disabled}
                if not inert:
                    break
                for name in inert:
                    result.pop(name)
                disabled.update(inert)
    if (engine['kind'] == 'fast' and engine['cutoffs']
            and engine.get('profile', 'current') == 'current' and not engine.get('recipe')
            and resolved['hardware']['gpu'] == 'a100'
            and result.get(environment_name('engine', 'engine_settings.hybrid_verify_triton')) == '1'):
        # This composed route inherits the validated sm80 recipe's qualification.
        # Its backend and reuse must travel together: Triton cannot reuse FI plans.
        result[environment_name('engine', 'engine_settings.attention_backend')] = 'triton'
        result[environment_name('engine', 'engine_settings.verify_plan_reuse')] = '0'
        result[environment_name('engine', 'engine_settings.verify_plan_lean')] = '0'
    return result


def environment_name(category, field):
    return next(name for name, path in ENV_FIELDS[category].items() if path == field)
