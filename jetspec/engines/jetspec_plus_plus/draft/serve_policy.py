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
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType

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








def load_policy(environ=None):
    """Parse/cache a policy; no manifest effects, no torch dependency.

    Cache the consumed bytes for the run, including across repeated section
    reads. A new process is required to change a policy at the same path.
    """
    environ = os.environ if environ is None else environ
    path = environ.get('JETSPEC_SERVE_POLICY')
    if path is None:
        return EMPTY
    if not path:
        raise ValueError('JETSPEC_SERVE_POLICY must name a JSON file')
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    return _load(str(path.resolve()))


def section(name, *, environ=None):
    return load_policy(environ).section(name)




def validate_policy(policy, environ, *, jetspec, max_num_seqs):
    """Plan-time regime checks, without changing the supplied environment."""
    enabled = [name for name in DEFAULTS if policy.section(name)['enabled']]
    if not enabled:
        return
    eager_swap = False
    if (environ.get('JETSPEC_CHAIN_GRAPH', '0') == '0'
            and environ.get('JETSPEC_REFRESH_SWAP', '0') == '1'):
        from jetspec.engines.jetspec_plus_plus.draft.eager_schedule import parse_schedule
        eager_swap = parse_schedule(environ.get('JETSPEC_CHAIN_FIXED_CONFIG')).r >= 1
    if (not jetspec or max_num_seqs != 1
            or (environ.get('JETSPEC_CHAIN_GRAPH', '0') != 'seg' and not eager_swap)):
        raise ValueError('JETSPEC_SERVE_POLICY enabled sections require N=1 JetSpec segmented chain '
                         'or eager fixed refresh swap')
    if 'tail_passes' in enabled:
        if ((not eager_swap and environ.get('JETSPEC_CHAIN_FIXED_CONFIG') not in
                ('J1-R1', 'J2-R1', 'J3-R1'))
                or environ.get('JETSPEC_ADAPTER_REFRESH', 'off') != 'after_j0'):
            config_kind = 'fixed refresh configuration' if eager_swap else 'fixed R1 chain'
            raise ValueError(f'JETSPEC_SERVE_POLICY tail_passes requires a {config_kind} '
                             'and JETSPEC_ADAPTER_REFRESH=after_j0')
    for name in ('pin_agreed', 'after_swap_fill', 'swap_rule'):
        if name in enabled and environ.get('JETSPEC_REFRESH_SWAP', '0') != '1':
            raise ValueError(f'JETSPEC_SERVE_POLICY {name} requires JETSPEC_REFRESH_SWAP=1')
    if 'rows_min' in enabled:
        try:
            capacity = int(environ.get('JETSPEC_ROWS_CAP', '128'))
            cap = int(environ.get('JETSPEC_ROWS_RULE_CAP', str(capacity)))
        except ValueError:
            raise ValueError('JETSPEC_SERVE_POLICY rows_min requires integer row capacities') from None
        floor = policy.section('rows_min')['count']
        bucket = 1 << (max(4, cap, floor) - 1).bit_length()
        if bucket > capacity:
            raise ValueError('JETSPEC_SERVE_POLICY rows_min.count exceeds graph capacity '
                             '(JETSPEC_ROWS_CAP)')
        if environ.get('E04_ROWRULE') != 'joint':
            raise ValueError('JETSPEC_SERVE_POLICY rows_min requires E04_ROWRULE=joint')
        raw_table = environ.get('JETSPEC_CHAIN_TAU_TABLE')
        table = json.loads(raw_table) if raw_table is not None else {}
        if policy.section('rows_min')['count'] > 4 and len(table) == 10:
            raise ValueError('JETSPEC_SERVE_POLICY rows_min above 4 refuses cap-indexed tau tables')


def rows_min_count(environ=None):
    """Read the row floor before capture; zero retains the existing row body."""
    from jetspec.engines.jetspec_plus_plus.serving.presets.budget import budget_type
    env = os.environ if environ is None else environ
    value = section('rows_min', environ=environ)
    configured = value['count'] if value['enabled'] else 0
    return budget_type(env).row_floor(configured, env)


def rows_min_capacity(cap, environ=None):
    """Expand the proposal bucket only when a row floor is enabled."""
    floor = rows_min_count(environ)
    return 1 << (max(4, cap, floor) - 1).bit_length() if floor else cap
