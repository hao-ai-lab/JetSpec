"""Explicit rank settings, with an adapter for supplied legacy validation maps."""
from collections.abc import Mapping

from dataclasses import replace
from jetspec.engines.jetspec_plus_plus.launch_env import ENV_FIELDS, environment_name
from jetspec.engines.jetspec_plus_plus.run_config import RunConfig, FrozenMapping


def as_run_config(config=None):
    if isinstance(config, RunConfig):
        return config
    if config is None:
        return RunConfig.from_environment({})
    if not isinstance(config, Mapping):
        raise TypeError('draft settings require RunConfig or an explicit legacy mapping')
    known = {name for fields in ENV_FIELDS.values() for name in fields}
    return RunConfig.from_environment({name: value for name, value in config.items() if name in known})


def setting(config, path, default=None):
    """Use the caller's absent-value default while retaining explicit empty strings."""
    if isinstance(config, Mapping):
        return config.get(environment_name(*path.split('.', 1)), default)
    config = as_run_config(config)
    if path not in config.provided_values:
        return default
    value = config.provided_values[path]
    return default if value is None else value


def required_setting(config, path):
    value = setting(config, path)
    if value is None:
        category, field = path.split('.', 1)
        raise KeyError(environment_name(category, field))
    return value


def legacy_values(config):
    """Bind a supplied projection for existing explicit-map validation boundaries."""
    config = as_run_config(config)
    return {name: config.provided_values[category + '.' + path]
            for category, fields in ENV_FIELDS.items() for name, path in fields.items()
            if category + '.' + path in config.provided_values}


def legacy_name(path):
    return environment_name(*path.split('.', 1))


def named_setting(config, name, default=None):
    for category, fields in ENV_FIELDS.items():
        if name in fields:
            return setting(config, category + '.' + fields[name], default)
    raise KeyError(name)


def with_settings(config, values):
    config = as_run_config(config)
    supplied = legacy_values(config)
    supplied.update({legacy_name(path): value for path, value in values.items()})
    updated = RunConfig.from_environment(supplied)
    return replace(updated, config_sha256=config.config_sha256,
                   model=FrozenMapping.from_mapping({**dict(config.model), **dict(updated.model)}),
                   provided_values=FrozenMapping.from_mapping({**dict(config.provided_values),
                                                               **dict(updated.provided_values)}))


def with_method_settings(config, method):
    config = as_run_config(config)
    if method is None:
        return config
    paths = dict(row_floor='method.row_floor', refresh_cap='method.refresh_cap',
                 projected_khat='method.projected_khat', fill_zero='method.cache_rows.fill_zero')
    values = {key: value for key, value in config.provided_values.items() if key not in paths.values()}
    values.update({path: method[key] for key, path in paths.items() if key in method})
    return replace(config, provided_values=FrozenMapping.from_mapping(values))
