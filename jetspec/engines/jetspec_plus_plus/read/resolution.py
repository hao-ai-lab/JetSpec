"""Compose the closed serving and training schemas on CPU."""
from pathlib import Path

from jetspec.engines.jetspec_plus_plus.draft.method.config import CATEGORIES, Schema, compose, validate
from jetspec.engines.jetspec_plus_plus.draft.method.config_schemas import METHOD_SCHEMA, MODEL_SCHEMA
from jetspec.engines.jetspec_plus_plus.draft.method.schema_extensions import training_schemas, require_training_schemas
from jetspec.engines.jetspec_plus_plus.config_schemas import ENGINE_SCHEMA, HARDWARE_SCHEMA, BENCHMARK_SCHEMA, BATCH_DEFAULTS

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


def resolve(names, overrides=(), *, root=ROOT / 'config'):
    config = compose(root, names, overrides)
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
