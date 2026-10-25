"""Immutable, spawn-safe projection of the config leaves read by ranks."""
from collections.abc import Mapping
from dataclasses import dataclass, field
import re

from jetspec.engines.jetspec_plus_plus.draft.method.config import CATEGORIES, config_hash, flatten, validate, validate_fields
from jetspec.engines.jetspec_plus_plus.draft.method.config_schemas import METHOD_SCHEMA, MODEL_SCHEMA
from jetspec.engines.jetspec_plus_plus.config_schemas import (ENGINE_SCHEMA, HARDWARE_SCHEMA, BENCHMARK_SCHEMA,
                                          RANK_CONFIG_FIELDS, BATCH_DEFAULTS, check_batch_capacity)
from jetspec.engines.jetspec_plus_plus.legacy_env import ENV_FIELDS, serving_environment, environment_name


_SCHEMAS = dict(zip(CATEGORIES, (METHOD_SCHEMA, MODEL_SCHEMA, ENGINE_SCHEMA,
                                HARDWARE_SCHEMA, BENCHMARK_SCHEMA)))
# Compatibility fill-only bindings for callers of from_config(launch_environment=...).
# Live children use the complete launch_overlay, whose values take launcher precedence.
LAUNCH_WRITTEN = (
    'CUBLAS_WORKSPACE_CONFIG',
    environment_name('model', 'model_settings.adapter_config'),
    environment_name('model', 'model_settings.adapter_head'),
    environment_name('method', 'method_settings.adapter_refresh'),
    environment_name('engine', 'engine_settings.anatomy_lite'),
    environment_name('model', 'model_settings.blockhead'),
    environment_name('engine', 'engine_settings.c1_compile_pin'),
    environment_name('engine', 'engine_settings.c1_compile_pin_sha'),
    environment_name('engine', 'engine_settings.chain_capture_stamps'),
    environment_name('engine', 'engine_settings.chain_decoder_key_bucket'),
    environment_name('engine', 'engine_settings.chain_glue2_profile'),
    environment_name('engine', 'engine_settings.chain_graph_check'),
    environment_name('engine', 'engine_settings.chain_handoff'),
    environment_name('engine', 'engine_settings.chain_handoff_timing'),
    environment_name('engine', 'engine_settings.chain_kernel_profile'),
    environment_name('engine', 'engine_settings.chain_kernel_profile_every'),
    environment_name('engine', 'engine_settings.chain_miss_redraft'),
    environment_name('engine', 'engine_settings.chain_miss_redraft_max_ms'),
    environment_name('engine', 'engine_settings.chain_miss_redraft_timing'),
    environment_name('engine', 'engine_settings.chain_prepost'),
    environment_name('engine', 'engine_settings.chain_prepost_timing'),
    environment_name('engine', 'engine_settings.chain_tail_trace'),
    environment_name('engine', 'engine_settings.chain_warm_reachable'),
    environment_name('engine', 'engine_settings.chain_warm_read'),
    environment_name('model', 'model_settings.checkpoint_dir'),
    environment_name('benchmark', 'benchmark_settings.dataset'),
    environment_name('engine', 'engine_settings.device_split'),
    environment_name('engine', 'engine_settings.draft_deadline'),
    environment_name('model', 'model_settings.heads_dir'),
    environment_name('hardware', 'hardware_settings.mps_draft_active_thread_percentage'),
    environment_name('hardware', 'hardware_settings.mps_draft_priority'),
    environment_name('engine', 'engine_settings.pins'),
    environment_name('engine', 'engine_settings.roundtrace'),
    environment_name('engine', 'engine_settings.roundtrace_device'),
    environment_name('engine', 'engine_settings.roundtrace_event_pairs'),
    environment_name('engine', 'engine_settings.rows_cap'),
    environment_name('method', 'method_settings.row_select'),
    environment_name('model', 'model_settings.sync_drafter'),
    environment_name('engine', 'engine_settings.sync_graph'),
    environment_name('method', 'method_settings.sync_passes'),
    environment_name('engine', 'engine_settings.sync_static'),
    environment_name('hardware', 'hardware_settings.target_model_path'),
    environment_name('model', 'model_settings.target_release_weights'),
    environment_name('engine', 'engine_settings.verify_host_census'),
    environment_name('engine', 'engine_settings.verify_stage_early'),
    environment_name('benchmark', 'benchmark_settings.warmup_prefix_lengths'),
    'MKL_NUM_THREADS',
    'OMP_NUM_THREADS',
    'OPENBLAS_NUM_THREADS',
    environment_name('engine', 'engine_settings.pred_serve_read_records'),
    environment_name('hardware', 'hardware_settings.cpu_pin_plan'),
    environment_name('hardware', 'hardware_settings.cuda_arch'),
    'SSD_CUTE_DSL_CACHE_ROOT',
    environment_name('hardware', 'hardware_settings.dataset_dir'),
    environment_name('engine', 'engine_settings.eager_verify_causal_n1'),
    environment_name('engine', 'engine_settings.eager_verify_wrapper'),
    environment_name('hardware', 'hardware_settings.hf_cache'),
    environment_name('hardware', 'hardware_settings.rank_cpu_affinity'),
    'TORCHDYNAMO_DISABLE',
    'TORCHINDUCTOR_CACHE_DIR',
    'TORCHINDUCTOR_COMPILE_THREADS',
    'TRITON_CACHE_DIR',
    environment_name('engine', 'engine_settings.w280_record'),
)


@dataclass(frozen=True)
class FrozenMapping(Mapping):
    _items: tuple

    @classmethod
    def from_mapping(cls, values):
        def freeze(value):
            if isinstance(value, Mapping):
                return cls.from_mapping(value)
            if isinstance(value, (list, tuple)):
                return tuple(freeze(child) for child in value)
            return value

        return cls(tuple((key, freeze(value))
                         for key, value in sorted(values.items())))

    def __getitem__(self, key):
        for name, value in object.__getattribute__(self, '_items'):
            if name == key:
                return value
        raise KeyError(key)

    def __iter__(self):
        return (key for key, _ in self._items)

    def __len__(self):
        return len(self._items)

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name) from None


def legacy_environment(environ):
    """Select serving bindings and unknown serving names from a process env.

    Ordinary host variables do not form part of the five-entry config. Unknown
    serving knobs are retained so the strict adapter rejects them.
    """
    from jetspec.engines.jetspec_plus_plus.jetspec.env import SCANNED_PREFIXES
    from jetspec.engines.jetspec_plus_plus.jetspec.env import REGISTRY
    bound = {name for fields in ENV_FIELDS.values() for name in fields}
    return {name: value for name, value in environ.items()
            if name not in REGISTRY and (name in bound or name.startswith(SCANNED_PREFIXES)
                                        or re.match(r'W\d+_', name))}


def launch_overlay(environment):
    """Seal the complete launch binding set, including explicitly absent leaves.

    The launcher has already applied recipe, placement, host and CLI precedence.
    Null entries distinguish an absent option from a dropped handoff key.
    """
    values = legacy_environment(environment)
    bindings = {name: f'{category}.{path}' for category, fields in ENV_FIELDS.items()
                for name, path in fields.items()}
    unknown = values.keys() - bindings.keys()
    if unknown:
        raise ValueError(f'launch_overlay names without a config binding: {sorted(unknown)}')
    values.setdefault('CUBLAS_WORKSPACE_CONFIG', '')
    return {path: values.get(name) for name, path in bindings.items()}


def config_environment(config):
    """Compatibility view for validators and records; never a child process env."""
    return {name: config.provided_values[f'{category}.{path}']
            for category, fields in ENV_FIELDS.items() for name, path in fields.items()
            if f'{category}.{path}' in config.provided_values}


def child_environment(environment):
    """Keep only host/rendezvous values and library-owned process settings."""
    from jetspec.engines.jetspec_plus_plus.jetspec.env import REGISTRY
    process = {'CUBLAS_WORKSPACE_CONFIG', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
               'OPENBLAS_NUM_THREADS', 'TORCHINDUCTOR_CACHE_DIR',
               'TORCHINDUCTOR_COMPILE_THREADS', 'TRITON_CACHE_DIR',
               'TORCHDYNAMO_DISABLE', 'SSD_CUTE_DSL_CACHE_ROOT'}
    serving = legacy_environment(environment)
    return {name: value for name, value in environment.items()
            if name not in serving or name in process or name in REGISTRY}


def _project(resolved, environment=None):
    sections = {}
    for category in CATEGORIES:
        schema = _SCHEMAS[category]
        leaves = flatten(resolved[category], atomic=schema.atomic)
        section = {}
        for path, default in RANK_CONFIG_FIELDS.get(category, {}).items():
            value = leaves.get(path)
            if value is None and environment is not None:
                name = next((name for name, field in ENV_FIELDS[category].items() if field == path), None)
                value = environment.get(name)
            value = default if value is None else value
            if value is None and (category, path) in (
                    ('model', 'thresholds'), ('model', 'thresholds_sha256'),
                    ('hardware', 'hardware_settings.target_model_path')):
                continue
            cursor = section
            *parents, key = path.split('.')
            for parent in parents:
                cursor = cursor.setdefault(parent, {})
            cursor[key] = value
        for (owner, key), default in BATCH_DEFAULTS.items():
            if owner == category and key in leaves and leaves[key] != default:
                section[key] = leaves[key]
        validate_fields(section, schema.fields, (), category, atomic=schema.atomic)
        sections[category] = FrozenMapping.from_mapping(section)
    return sections


@dataclass(frozen=True, slots=True)
class RunConfig:
    method: FrozenMapping
    model: FrozenMapping
    engine: FrozenMapping
    hardware: FrozenMapping
    benchmark: FrozenMapping
    config_sha256: str | None = field(default=None, compare=False)
    provided_values: FrozenMapping = field(default_factory=lambda: FrozenMapping(()), compare=False)

    @property
    def concurrency(self):
        return self.benchmark.get('concurrency', BATCH_DEFAULTS[('benchmark', 'concurrency')])

    @property
    def batch_capacity(self):
        return FrozenMapping.from_mapping({key: self.engine.get(key, default)
            for (owner, key), default in BATCH_DEFAULTS.items() if owner == 'engine'})

    def batch_overlay(self):
        """Typed launch values; defaults are effective without entering config identity."""
        return {_dotted(('benchmark', 'concurrency')): self.concurrency,
                **{_dotted(('engine', key)): list(value) if isinstance(value, tuple) else value
                   for key, value in self.batch_capacity.items()}}

    @classmethod
    def from_overlay(cls, overlay, resolved=None, *, model_entry=None):
        """Read a complete launch overlay without consulting the process environment."""
        bindings = {f'{category}.{path}': name for category, fields in ENV_FIELDS.items()
                    for name, path in fields.items()}
        if not isinstance(overlay, Mapping):
            raise ValueError('missing launch_overlay section')
        unknown = overlay.keys() - bindings.keys() - {_dotted(path) for path in BATCH_DEFAULTS}
        missing = bindings.keys() - overlay.keys()
        if unknown:
            raise ValueError(f'launch_overlay keys without a RunConfig binding: {sorted(unknown)}')
        if missing:
            raise ValueError(f'launch_overlay missing keys: {sorted(missing)}')
        if resolved is not None:
            validate(resolved, _SCHEMAS)
            if model_entry is None:
                model_entry = resolved['model']
        record = cls.from_environment({bindings[path]: value for path, value in overlay.items()
                                       if path in bindings and value is not None}, model_entry=model_entry)
        native = {category: {} for category in CATEGORIES}
        for (category, key), default in BATCH_DEFAULTS.items():
            path = _dotted((category, key))
            expected = resolved[category].get(key, default) if resolved is not None else default
            value = overlay.get(path, default)
            if resolved is not None and value != expected:
                raise ValueError(f'launch_overlay {path} disagrees with resolved config')
            native[category][key] = value
        for category in ('engine', 'benchmark'):
            schema = _SCHEMAS[category]
            validate_fields(native[category], schema.fields, (), category, atomic=schema.atomic)
        check_batch_capacity(native)
        sections = {category: FrozenMapping.from_mapping({**getattr(record, category),
                        **{key: value for key, value in native[category].items()
                           if value != BATCH_DEFAULTS[(category, key)]}})
                    for category in CATEGORIES}
        return cls(**sections, config_sha256=config_hash(resolved) if resolved is not None else None,
                   provided_values=record.provided_values)

    @classmethod
    def from_config(cls, resolved, *, launch_environment=None):
        """``launch_environment`` holds what the launcher derived at launch (cache capacities sized
        from max_model_len, and the ``LAUNCH_WRITTEN`` values). It fills only the leaves the
        config leaves unset, as a legacy launch read them; a value the config sets wins."""
        from jetspec.engines.jetspec_plus_plus.serving.cache_capacity import LAUNCH_DERIVED
        validate(resolved, _SCHEMAS)
        launch_environment = launch_environment or {}
        environment = {name: launch_environment[name] for name in LAUNCH_DERIVED + LAUNCH_WRITTEN
                       if name in launch_environment}
        environment.update(serving_environment(resolved))
        return cls(**_project(resolved, environment), config_sha256=config_hash(resolved),
                   provided_values=FrozenMapping.from_mapping({**{f'{category}.{path}': environment[name]
                       for category in CATEGORIES for name, path in ENV_FIELDS[category].items()
                       if name in environment}, **{f'{category}.{path}': value
                       for category in CATEGORIES
                       for path, value in flatten(resolved[category], atomic=_SCHEMAS[category].atomic).items()
                       if path in ENV_FIELDS[category].values() and value is not None},
                       'method.row_floor': resolved['method'].get('row_floor', resolved['method'].get('cache_rows', {}).get('floor', 0)),
                       'method.refresh_cap': resolved['method'].get('refresh_cap', True),
                       'method.projected_khat': resolved['method'].get('projected_khat', 'predicted'),
                       'method.cache_rows.fill_zero': resolved['method'].get('cache_rows', {}).get('fill_zero', False)}))

    @classmethod
    def from_environment(cls, environ, *, model_entry=None):
        bindings = {name: (category, path) for category, fields in ENV_FIELDS.items()
                    for name, path in fields.items()}
        unknown = set(environ) - bindings.keys()
        if unknown:
            raise ValueError(f'environment names without a config binding: {sorted(unknown)}')
        resolved = {category: {} for category in CATEGORIES}
        if model_entry is not None:
            resolved['model'] = model_entry
        for name, value in environ.items():
            category, path = bindings[name]
            if path not in RANK_CONFIG_FIELDS.get(category, {}):
                continue
            if category == 'engine' and path == 'engine_settings.draft_vocab_k':
                value = int(value)
            cursor = resolved[category]
            *parents, key = path.split('.')
            for parent in parents:
                cursor = cursor.setdefault(parent, {})
            cursor[key] = value
        return cls(**_project(resolved), provided_values=FrozenMapping.from_mapping({
            f'{bindings[name][0]}.{bindings[name][1]}': value
            for name, value in environ.items() if value is not None}))


def _dotted(path):
    """Join a (category, field) pair into its launch-overlay key."""
    return '.'.join(path)


def child_bench_arguments(plan, run_config):
    """Five-entry children derive C from the immutable rank record."""
    arguments = list(plan['bench_arguments'])
    if plan.get('config_read') is not None:
        if '--b' in arguments:
            index = arguments.index('--b')
            arguments[index + 1] = str(run_config.concurrency)
        else:
            arguments.extend(['--b', str(run_config.concurrency)])
    return arguments


def apply_child_concurrency(plan, run_config):
    """Set a five-entry child's C from RunConfig; recipe-only plans keep their arguments."""
    plan['bench_arguments'] = child_bench_arguments(plan, run_config)
    if plan.get('config_read') is not None:
        plan['batch'] = run_config.concurrency


def config_batch_overlay(args):
    """The typed batch overlay of a five-entry read; empty for recipe-only launches."""
    return dict((getattr(args, 'config_read', None) or {}).get('launch_overlay', {}))


def config_concurrency(args, default):
    """C for the parent launcher: from the resolved config when present, else the argv default."""
    record = getattr(args, 'config_read', None)
    return default if record is None else RunConfig.from_config(record['config']).concurrency
