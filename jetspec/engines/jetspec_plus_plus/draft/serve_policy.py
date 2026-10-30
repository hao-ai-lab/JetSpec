"""Unified, default-off serving policy. Read once per path, before graph capture.

``load_policy(environ=None)`` returns an immutable policy; ``section(name)`` is
the small read API used by serving mechanisms. Relative paths are relative to
the repository root, so committed recipes work from both tools/ and serve/.
Disabled sections retain validated parameters but their readers must use
``enabled`` before acting. No policy means no manifest or serving change.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
from jetspec.engines.jetspec_plus_plus.draft.config import setting, as_run_config

SCHEMA = 'jetspec.serve_policy/1'
ROOT = Path(__file__).resolve().parents[3]
SLOT_BUCKETS = ('1', '2-3', '4-7', '8-15')
KHAT_BUCKETS = ('0-2', '3-7', '8-15')
DEFAULTS = {
    'pin_agreed': {'enabled': False},
    'rows_min': {'enabled': False, 'count': 8},
    'after_swap_fill': {'enabled': False, 'mode': 'keep', 'seed': 0},
    'tail_passes': {'enabled': False, 'count': 0},
    'swap_rule': {'enabled': False, 'n': 8, 'floor_factor': 0.5, 'alpha': 0.0,
                  'margin_space': 'prob', 'pin': 1, 'margin': 0.0, 'scan': False,
                  'bias': {s: dict.fromkeys(KHAT_BUCKETS, 0.0) for s in SLOT_BUCKETS}},
}


def _freeze(value):
    return (MappingProxyType({k: _freeze(v) for k, v in value.items()})
            if isinstance(value, dict) else value)


@dataclass(frozen=True)
class ServePolicy:
    path: str | None
    sha256: str | None
    sections: object

    def section(self, name):
        return self.sections[name]


EMPTY = ServePolicy(None, None, _freeze(DEFAULTS))


def _integer(value, low, high=None):
    return type(value) is int and value >= low and (high is None or value <= high)


def _number(value):
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


@lru_cache(maxsize=None)
def _load(path):
    def refuse(message):
        raise ValueError(f'JETSPEC_SERVE_POLICY {path}: {message}')

    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                refuse(f'duplicate key {key!r}')
            result[key] = value
        return result

    def constant(value):
        refuse(f'non-finite number {value}')

    try:
        from jetspec.engines.jetspec_plus_plus.launch.recipes import policy_contents
        raw = policy_contents(path)
        data = json.loads(raw.decode('utf-8'), object_pairs_hook=unique,
                          parse_constant=constant)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        refuse(f'cannot read UTF-8 JSON ({error})')
    if not isinstance(data, dict) or data.get('schema') != SCHEMA:
        refuse(f'must be an object with schema {SCHEMA!r}')
    unknown = set(data) - {'schema', *DEFAULTS}
    if unknown:
        refuse(f'unknown sections {sorted(unknown)}')
    sections = {}
    for name, defaults in DEFAULTS.items():
        supplied = data.get(name, {})
        if not isinstance(supplied, dict):
            refuse(f'{name} must be an object')
        if name == 'rows_min' and 'n' in supplied:
            if 'count' in supplied and supplied['count'] != supplied['n']:
                refuse('rows_min.n conflicts with rows_min.count')
            supplied = {**supplied, 'count': supplied['n']}
            supplied.pop('n')
        supplied = dict(supplied)
        aliases = {'n': 'count'} if name == 'tail_passes' else {}
        if name == 'swap_rule':
            aliases = {'bias_table': 'bias'}
        for alias, canonical in aliases.items():
            if alias in supplied:
                if canonical in supplied:
                    refuse(f'{name}: supply only one of {alias} and {canonical}')
                supplied[canonical] = supplied.pop(alias)
        unknown = set(supplied) - set(defaults)
        if unknown:
            refuse(f'{name}: unknown keys {sorted(unknown)}')
        if name in data and type(supplied.get('enabled')) is not bool:
            refuse(f'{name}.enabled must be a bool')
        sections[name] = {**defaults, **supplied}
    if not _integer(sections['rows_min']['count'], 1, 16):
        refuse('rows_min.count must be an integer in 1..16')
    if not _integer(sections['tail_passes']['count'], 0, 2):
        refuse('tail_passes.count must be an integer in 0..2')
    fill = sections['after_swap_fill']
    if fill['mode'] not in ('keep', 'context', 'mask', 'noise', 'decoder'):
        refuse('after_swap_fill.mode must be keep, context, mask, noise or decoder')
    if not _integer(fill['seed'], 0):
        refuse('after_swap_fill.seed must be a nonnegative integer')
    rule = sections['swap_rule']
    if not _number(rule.get('margin', 0.0)):
        refuse('swap_rule.margin must be a finite number')
    if type(rule.get('scan', False)) is not bool:
        refuse('swap_rule.scan must be a bool')
    if not _integer(rule['n'], 1, 32):
        refuse('swap_rule.n must be an integer in 1..32')
    if not _number(rule['floor_factor']) or not 0 < rule['floor_factor'] <= 1:
        refuse('swap_rule.floor_factor must be a finite number in (0, 1]')
    if not _number(rule['alpha']) or rule['alpha'] < 0:
        refuse('swap_rule.alpha must be a finite number >= 0')
    if rule['margin_space'] not in ('prob', 'log'):
        refuse('swap_rule.margin_space must be prob or log')
    if not _integer(rule['pin'], 0, 1):
        refuse('swap_rule.pin must be the integer 0 or 1')
    bias = rule['bias']
    if not isinstance(bias, dict) or set(bias) != set(SLOT_BUCKETS):
        refuse(f'swap_rule.bias must have exactly slot buckets {SLOT_BUCKETS}')
    for slot, cells in bias.items():
        if not isinstance(cells, dict) or set(cells) != set(KHAT_BUCKETS):
            refuse(f'swap_rule.bias[{slot!r}] must have exactly k-hat buckets {KHAT_BUCKETS}')
        if not all(_number(value) for value in cells.values()):
            refuse(f'swap_rule.bias[{slot!r}] must contain finite numbers')
    return ServePolicy(path, hashlib.sha256(raw).hexdigest(), _freeze(sections))


# The composed paper cells name the retired calibration policy. Its bytes live here unchanged
# (sha256 595245a8...), as a file: the child records the consumed policy file in the run manifest.
CALIBRATION_POLICY = Path(__file__).resolve().parent / 'policies/calibration-theta-rows8.json'


def load_policy(environ=None, *, run_config=None):
    """Parse/cache a policy; no manifest effects, no torch dependency.

    Cache the consumed bytes for the run, including across repeated section
    reads. A new process is required to change a policy at the same path.
    """
    path = setting(run_config if run_config is not None else environ, 'engine.engine_settings.serve_policy')
    if path is None:
        return EMPTY
    if not path:
        raise ValueError('JETSPEC_SERVE_POLICY must name a JSON file')
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    legacy_root = ROOT / 'recipes/serve'
    if not path.is_file() and path.is_relative_to(legacy_root):
        if path.relative_to(legacy_root).as_posix() == 'policies/calibration-theta-rows8.json':
            path = CALIBRATION_POLICY
        else:
            from jetspec.engines.jetspec_plus_plus.launch.recipes import policy_path
            path = policy_path(ROOT, path.relative_to(legacy_root))
    return _load(str(path.resolve()))


def section(name, *, environ=None, run_config=None):
    return load_policy(environ, run_config=run_config).section(name)


def record_policy(run_config=None):
    """Bind the consumed policy to the run manifest once inputs are recorded."""
    policy = load_policy(run_config=run_config)
    if policy.path is not None:
        import jetspec.engines.jetspec_plus_plus.runtime.run_manifest as run_manifest
        entry = run_manifest.record_file('serve_policy', policy.path, run_config=run_config)
        if entry['sha256'] != policy.sha256:
            raise ValueError('JETSPEC_SERVE_POLICY changed after validation')
        run_manifest.record_value('serve_policy_sha256', policy.sha256, run_config=run_config)


def validate_policy(policy, environ, *, jetspec, max_num_seqs):
    """Plan-time regime checks, without changing the supplied environment."""
    enabled = [name for name in DEFAULTS if policy.section(name)['enabled']]
    if not enabled:
        return
    eager_swap = False
    if (setting(environ, 'engine.engine_settings.chain_graph', '0') == '0'
            and setting(environ, 'engine.engine_settings.refresh_swap', '0') == '1'):
        from jetspec.engines.jetspec_plus_plus.draft.method.schedule import parse_schedule
        eager_swap = parse_schedule(setting(environ, 'method.method_settings.chain_fixed_config')).r >= 1
    if (not jetspec or max_num_seqs != 1
            or (setting(environ, 'engine.engine_settings.chain_graph', '0') != 'seg' and not eager_swap)):
        raise ValueError('JETSPEC_SERVE_POLICY enabled sections require N=1 JetSpec segmented chain '
                         'or eager fixed refresh swap')
    if 'tail_passes' in enabled:
        if ((not eager_swap and setting(environ, 'method.method_settings.chain_fixed_config') not in
                ('J1-R1', 'J2-R1', 'J3-R1'))
                or setting(environ, 'method.method_settings.adapter_refresh', 'off') != 'after_j0'):
            config_kind = 'fixed refresh configuration' if eager_swap else 'fixed R1 chain'
            raise ValueError(f'JETSPEC_SERVE_POLICY tail_passes requires a {config_kind} '
                             'and JETSPEC_ADAPTER_REFRESH=after_j0')
    for name in ('pin_agreed', 'after_swap_fill', 'swap_rule'):
        if name in enabled and setting(environ, 'engine.engine_settings.refresh_swap', '0') != '1':
            raise ValueError(f'JETSPEC_SERVE_POLICY {name} requires JETSPEC_REFRESH_SWAP=1')
    if 'rows_min' in enabled:
        try:
            capacity = int(setting(environ, 'engine.engine_settings.rows_cap', '128'))
            cap = int(setting(environ, 'method.method_settings.rows_rule_cap', str(capacity)))
        except ValueError:
            raise ValueError('JETSPEC_SERVE_POLICY rows_min requires integer row capacities') from None
        floor = policy.section('rows_min')['count']
        bucket = 1 << (max(4, cap, floor) - 1).bit_length()
        if bucket > capacity:
            raise ValueError('JETSPEC_SERVE_POLICY rows_min.count exceeds graph capacity '
                             '(JETSPEC_ROWS_CAP)')
        if setting(environ, 'method.method_settings.rowrule') != 'joint':
            raise ValueError('JETSPEC_SERVE_POLICY rows_min requires E04_ROWRULE=joint')
        raw_table = setting(environ, 'engine.engine_settings.chain_tau_table')
        table = json.loads(raw_table) if raw_table is not None else {}
        if policy.section('rows_min')['count'] > 4 and len(table) == 10:
            raise ValueError('JETSPEC_SERVE_POLICY rows_min above 4 refuses cap-indexed tau tables')


def rows_min_count(environ=None, *, run_config=None):
    """Read the row floor before capture; zero retains the existing row body."""
    value = section('rows_min', environ=environ, run_config=run_config)
    configured = value['count'] if value['enabled'] else 0
    return method_settings(environ, run_config=run_config).get('row_floor', configured)


_method_settings = None


def install_method_settings(settings):
    """Install the method choices carried in the serving process's spawn payload."""
    global _method_settings
    previous = _method_settings
    _method_settings = None if settings is None else dict(settings)
    return previous


def method_settings(environ=None, *, resolved=None, run_config=None, engine=None):
    """Method choices, separate from timing cutoffs; bridge old recipes until read lands."""
    if resolved is not None:
        method = resolved['method']
        return dict(row_floor=method.get('row_floor', method.get('cache_rows', {}).get('floor', 0)),
                    refresh_cap=method.get('refresh_cap', True),
                    projected_khat=method.get('projected_khat', 'predicted'),
                    fill_zero=method.get('cache_rows', {}).get('fill_zero', False))
    if _method_settings is not None and environ is None and run_config is None:
        return _method_settings
    from jetspec.engines.jetspec_plus_plus.engine import current_engine
    config = run_config if run_config is not None else environ
    supplied = as_run_config(config).provided_values
    if 'method.refresh_cap' in supplied:
        choices = dict(refresh_cap=supplied['method.refresh_cap'],
                       projected_khat=supplied['method.projected_khat'],
                       fill_zero=supplied['method.cache_rows.fill_zero'])
        if 'method.row_floor' in supplied:
            choices['row_floor'] = supplied['method.row_floor']
        return choices
    engine = engine if engine is not None else current_engine(environ)
    profiling = engine.execution in ('pipelined', 'critical-path', 'on')
    uncut_method = profiling or engine.kind == 'reference' or engine.execution in ('eager', 'uncut')
    settings = dict(refresh_cap=not uncut_method,
                    projected_khat='width-2' if uncut_method else 'predicted',
                    fill_zero=uncut_method)
    if profiling:
        settings['row_floor'] = int(setting(config, 'method.method_settings.eval_rows', '16'))
    return settings


def projected_khat(khat, width, *, run_config=None):
    return (khat.new_full(khat.shape, width - 2)
            if method_settings(run_config=run_config)['projected_khat'] == 'width-2' else khat)


def rows_min_capacity(cap, environ=None, *, run_config=None):
    """Expand the proposal bucket only when a row floor is enabled."""
    floor = rows_min_count(environ, run_config=run_config)
    return 1 << (max(4, cap, floor) - 1).bit_length() if floor else cap
