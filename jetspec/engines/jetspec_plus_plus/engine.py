"""Immutable execution policy built before ranks are spawned."""
from contextlib import contextmanager
from dataclasses import dataclass
import os
from pathlib import Path
import sys

# Preserve the legacy namespace's rank modules during the read-entry migration.
__path__ = [str(Path(__file__).with_suffix(''))]


@dataclass(frozen=True)
class Engine:
    kind: str = 'fast'
    cutoffs: bool = True
    deadline: bool = False
    profile: str = 'current'
    deterministic: bool = False
    record: str = 'none'
    release_finished_requests: bool = False
    tag_sequence_ids: bool = False
    skip_admission_prewarm: bool = False

    def __post_init__(self):
        if self.kind not in ('fast', 'reference'):
            raise ValueError('engine kind must be fast or reference')

    @classmethod
    def from_config(cls, resolved, *, sealed_record=False):
        from jetspec.engines.jetspec_plus_plus.draft.method.config import validate_fields
        from jetspec.engines.jetspec_plus_plus.config_schemas import ENGINE_SCHEMA, HARDWARE_SCHEMA
        for category, schema in (('engine', ENGINE_SCHEMA), ('hardware', HARDWARE_SCHEMA)):
            validate_fields(resolved[category], schema.fields, schema.required, category,
                            atomic=schema.atomic)
        values = resolved['engine']
        request_policy = {}
        for name in ('release_finished_requests', 'tag_sequence_ids', 'skip_admission_prewarm'):
            if name not in values:
                raise ValueError(f'engine missing {name}')
            request_policy[name] = values[name]
        settings = values.get('engine_settings', {})
        profile = settings.get('eval_mode', 'off')
        if profile == 'off':
            profile = values.get('profile', 'current')
            if values.get('deterministic', False) and profile == 'current' and 'eval_mode' not in settings:
                profile = 'pipelined'
        kind = 'reference' if settings.get('verify_reference') == '1' else values['kind']
        if profile == 'current' and kind == 'fast' and settings.get('torchdynamo_disable') == '1':
            profile = 'eager-timed'
        cutoffs = values.get('cutoffs', True) and kind != 'reference'
        deadline = settings.get('draft_deadline')
        deadline = values.get('deadline', False) if deadline is None else deadline == '1'
        from jetspec.engines.jetspec_plus_plus.profiles import select_config
        profile = select_config(resolved, profile)
        if not sealed_record:
            from jetspec.engines.jetspec_plus_plus.profiles import require_live_execution
            require_live_execution(profile)
        record = values.get('record', 'none')
        if record == 'none' and settings.get('anatomy_lite', '0') != '0':
            record = 'anatomy'
        return cls(kind, cutoffs, cutoffs and deadline, profile,
                   values.get('deterministic', False), record, **request_policy)

    @classmethod
    def from_environment(cls, environ, *, regime=None, profile=None, sealed_record=False):
        """Adapt legacy switches; a launcher may supply its profile before spawn."""
        from jetspec.engines.jetspec_plus_plus.profiles import PROFILES, execution, select_legacy, refuse_retired
        from jetspec.engines.jetspec_plus_plus.legacy_env import environment_name
        refuse_retired(environ)
        def name(field):
            return environment_name('engine', 'engine_settings.' + field)
        named = profile if profile in PROFILES else None
        if profile is not None and named is None:
            environ[name('eval_mode')] = profile
        kind = environ.get(name('engine'))
        mode = execution(named) if named and execution(named) in ('pipelined', 'critical-path', 'on') else environ.get(name('eval_mode'), 'off')
        reference = environ.get(name('verify_reference'), '0') == '1'
        if kind not in ('fast', 'reference', 'eval', 'eager'):
            kind = ('eval' if mode != 'off' else 'reference' if reference else
                    'eager' if regime == 'eager' or environ.get('TORCHDYNAMO_DISABLE') == '1'
                    else 'fast')
        # Legacy fast plus an eval/reference switch still used NullBudget.
        cutoffs = (environ.get(name('engine')) not in ('eager', 'reference', 'eval')
                   and mode == 'off' and not reference)
        profile = (mode if mode != 'off' else
                   ('eager' if not cutoffs else 'eager-timed') if kind == 'eager' else
                   'uncut' if kind == 'eval' else 'current')
        deadline = environ.get(name('draft_deadline'), '0')
        if deadline not in ('0', '1'):
            raise ValueError('JETSPEC_DRAFT_DEADLINE must be 0 or 1')
        capture = environ.get(name('eval_capture_round_candidates'), '0')
        record = ('candidates' if capture != '0' else
                  'anatomy' if environ.get(name('anatomy_lite'), '0') != '0' else 'none')
        profile = named or select_legacy(environ, profile)
        if not sealed_record:
            from jetspec.engines.jetspec_plus_plus.profiles import require_live_execution
            require_live_execution(profile)
        return cls('reference' if kind == 'reference' or reference else 'fast',
                   cutoffs, cutoffs and deadline == '1', profile, mode != 'off', record,
                   release_finished_requests=mode != 'off',
                   tag_sequence_ids=mode != 'off', skip_admission_prewarm=mode != 'off')

    @property
    def execution(self):
        from jetspec.engines.jetspec_plus_plus.profiles import execution
        return execution(self.profile)

    @property
    def budget(self):
        from jetspec.engines.jetspec_plus_plus.cutoffs import DeadlineBudget, NullBudget
        return DeadlineBudget if self.cutoffs else NullBudget

    @property
    def ops(self):
        return ('apply_pins',) if self.kind == 'fast' and not self.execution.startswith('eager') else ()

    @property
    def graphs(self):
        return 'none' if self.execution.startswith('eager') else 'target' if self.kind == 'reference' else 'segments'


    @property
    def verify(self):
        return 'method' if self.execution.startswith('eager') else 'upstream' if self.kind == 'reference' else 'multirow'

    @property
    def transport(self):
        from jetspec.engines.jetspec_plus_plus.serving.placement import Lockstep, Pair
        return Pair if self.graphs == 'segments' and not self.deterministic and self.execution != 'uncut' else Lockstep


EAGER = Engine(cutoffs=False, profile='eager')
REFERENCE = Engine(kind='reference', cutoffs=False)
FAST = Engine()
EVAL = Engine(cutoffs=False, profile='pipelined', deterministic=True,
              release_finished_requests=True, tag_sequence_ids=True, skip_admission_prewarm=True)

_installed = None


def current_engine(environ=None):
    """Ranks use the installed record; only explicit legacy inputs use the adapter."""
    if _installed is not None and (environ is None or environ is os.environ):
        return _installed
    if environ is not None:
        return Engine.from_environment(environ, sealed_record=True)
    return FAST


def install_engine(engine):
    from jetspec.engines.jetspec_plus_plus.profiles import require_live_execution
    require_live_execution(engine.profile)
    global _installed
    from jetspec.engines.jetspec_plus_plus.kernels import install
    from jetspec.engines.jetspec_plus_plus.draft.method.ops import select_engine
    _installed = engine
    install(engine.ops)
    select_engine('fast' if engine.ops else None)
    # A module imported before the install read the default engine; bind its backend again.
    attention = sys.modules.get('jetspec.engines.jetspec_plus_plus.models.attention')
    if attention is not None:
        attention.bind_attention_backend()
    return engine


@contextmanager
def engine_scope(engine):
    from jetspec.engines.jetspec_plus_plus.profiles import require_live_execution
    require_live_execution(engine.profile)
    global _installed
    from jetspec.engines.jetspec_plus_plus.kernels import install
    from jetspec.engines.jetspec_plus_plus.draft.method.ops import use_engine
    previous = _installed
    _installed = engine
    install(engine.ops)
    try:
        with use_engine('fast' if engine.ops else None):
            yield engine
    finally:
        _installed = previous
