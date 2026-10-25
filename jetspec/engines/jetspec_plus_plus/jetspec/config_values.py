"""A lazy view of explicitly supplied rank settings, with absent-value semantics."""
from collections.abc import Mapping
from functools import lru_cache
from types import MappingProxyType


@lru_cache(maxsize=128)
def _flat_values(run_config, provided_values):
    if run_config is None:
        return MappingProxyType({})
    if provided_values is not None:
        return MappingProxyType({key: value for key, value in provided_values.items()
                                 if value is not None})
    return MappingProxyType({f'{category}.{group}.{key}': value
        for category in ('method', 'model', 'engine', 'hardware', 'benchmark')
        for group, values in getattr(run_config, category)._items
        if hasattr(values, '_items')
        for key, value in values._items if value is not None})


class ConfigValues(Mapping):
    def __init__(self, run_config):
        self.run_config = run_config
        self._values = _flat_values(run_config, getattr(run_config, 'provided_values', None))

    def __getitem__(self, path):
        return self._values[path]

    def __iter__(self):
        return iter(self._values)

    def __len__(self):
        return len(self._values)


# One view per live rank record, keyed by identity: the value-keyed cache above hashes the whole
# frozen record (thousands of nested hashes) on every lookup, which per-row serving code pays for
# each setting it reads. Each entry holds its record, so an id is not reused while it is cached.
_VIEWS = {}
_VIEW_LIMIT = 128


def config_values(run_config):
    """Accept a rank record or an explicit legacy launch mapping; never read ambient env."""
    if isinstance(run_config, ConfigValues):
        return run_config
    if isinstance(run_config, Mapping):
        from jetspec.engines.jetspec_plus_plus.run_config import RunConfig
        from jetspec.engines.jetspec_plus_plus.legacy_env import ENV_FIELDS
        bound = {name for fields in ENV_FIELDS.values() for name in fields}
        run_config = RunConfig.from_environment(
            {key: value for key, value in run_config.items() if key in bound})
        return ConfigValues(run_config)
    entry = _VIEWS.get(id(run_config))
    if entry is not None and entry[0] is run_config:
        return entry[1]
    view = ConfigValues(run_config)
    if len(_VIEWS) >= _VIEW_LIMIT:
        del _VIEWS[next(iter(_VIEWS))]
    _VIEWS[id(run_config)] = (run_config, view)
    return view


def config_flag01(key, run_config):
    value = config_values(run_config).get(key, '0')
    if value not in ('0', '1'):
        raise ValueError(f'{key} must be 0 or 1')
    return value == '1'


def launch_config(environment, resolved=None, *, model_entry=None):
    """Project a materialized legacy child plan and retain its resolved launch identity."""
    from dataclasses import replace
    from jetspec.engines.jetspec_plus_plus.run_config import RunConfig
    materialized = dict(environment)
    materialized.setdefault('CUBLAS_WORKSPACE_CONFIG', '')
    from jetspec.engines.jetspec_plus_plus.run_config import legacy_environment
    if model_entry is None and resolved is not None:
        model_entry = resolved['model']
    config = RunConfig.from_environment(legacy_environment({
        key: value for key, value in materialized.items()
        if key not in ('JETSPEC_HOST', 'JETSPEC_SHM_NAME')}), model_entry=model_entry)
    if resolved is not None:
        config = replace(config, config_sha256=RunConfig.from_config(resolved).config_sha256)
    return config
