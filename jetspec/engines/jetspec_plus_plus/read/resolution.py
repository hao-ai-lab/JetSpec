"""Compose the closed serving and training schemas on CPU."""
from copy import deepcopy
from pathlib import Path

from jetspec.engines.jetspec_plus_plus.draft.method.config import CATEGORIES, Schema, compose, flatten, validate, validate_fields
from jetspec.engines.jetspec_plus_plus.draft.method.config_schemas import METHOD_SCHEMA, MODEL_SCHEMA, method_concurrency_settings
from jetspec.engines.jetspec_plus_plus.draft.method.schema_extensions import training_schemas, require_training_schemas
from jetspec.engines.jetspec_plus_plus.config_schemas import (ENGINE_SCHEMA, HARDWARE_SCHEMA, BENCHMARK_SCHEMA,
                                          BATCH_DEFAULTS, concurrency_settings)

ROOT = next((parent for parent in Path(__file__).resolve().parents
             if (parent / 'artifacts/manifest.json').is_file()), Path.cwd())


def schemas():
    extensions = training_schemas()
    result = {}
    for category, serving in zip(CATEGORIES, (METHOD_SCHEMA, MODEL_SCHEMA, ENGINE_SCHEMA,
                                             HARDWARE_SCHEMA, BENCHMARK_SCHEMA)):
        training = extensions.get(category, Schema({}))
        if set(serving.fields) & set(training.fields):
            raise ValueError(f'duplicate consumer schema keys in {category}')
        result[category] = Schema({**serving.fields, **training.fields}, serving.required,
                                  serving.checks + training.checks, serving.atomic + training.atomic)
    return result


def _concurrency_rows(config, overrides):
    """Apply the smallest bucket >= concurrency; explicit settings keep precedence.

    Above the largest bucket leave the entry unchanged, so its capacity check
    retains its existing error. Tables never enter the resolved identity.
    """
    named = [override.partition('=')[0] for override in overrides]
    concurrency = config['benchmark'].get('concurrency', BATCH_DEFAULTS[('benchmark', 'concurrency')])
    for category, schema, rule in (('method', METHOD_SCHEMA, method_concurrency_settings),
                                   ('engine', ENGINE_SCHEMA, concurrency_settings)):
        if 'concurrency_settings' not in config[category]:
            continue
        table = config[category].pop('concurrency_settings')
        if rule(table) is False:
            raise ValueError(f'{category}.concurrency_settings has invalid type or value')
        validate_fields({'concurrency': concurrency}, BENCHMARK_SCHEMA.fields, (), 'benchmark')
        key = min((int(key) for key in table if int(key) >= concurrency), default=None)
        if key is None:
            continue
        for path, value in flatten(table[str(key)], atomic=schema.atomic).items():
            field = f'{category}.{path}'
            if any(field == name or field.startswith(name + '.') or name.startswith(field + '.')
                   for name in named):
                continue
            *parents, leaf = path.split('.')
            payload = config[category]
            for part in parents:
                payload = payload.setdefault(part, {})
            payload[leaf] = deepcopy(value)
    return config


def resolve(names, overrides=(), *, root=ROOT / 'config'):
    overrides = tuple(overrides)
    config = _concurrency_rows(compose(root, names, overrides), overrides)
    require_training_schemas(config)
    validate(config, schemas())
    from jetspec.engines.jetspec_plus_plus.legacy_env import execution_settings, release_recipe
    recipe = release_recipe(config)
    if recipe is not None:
        config['engine']['recipe'] = recipe
    settings = execution_settings(config)
    if settings:
        config['engine']['engine_settings'] = settings
    for category, field in (('model', 'model_execution'), ('method', 'method_execution')):
        config[category].pop(field, None)
    validate(config, schemas())
    # New optional defaults must not change the identity of existing reads.
    for (category, key), default in BATCH_DEFAULTS.items():
        if config[category].get(key) == default:
            config[category].pop(key, None)
    return config
