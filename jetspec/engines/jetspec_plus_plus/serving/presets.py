"""The optimization order as data. Method components never depend on presets."""
from contextlib import contextmanager
from jetspec.engines.jetspec_plus_plus.engine import (Engine, EAGER, REFERENCE, FAST, EVAL,
                                   engine_scope as record_scope)
PRESETS = dict(eager=EAGER, reference=REFERENCE, eval=EVAL, fast=FAST)


def engine_name(environ, *, regime=None):
    engine = Engine.from_environment(environ, regime=regime, sealed_record=True)
    return ('eval' if engine.deterministic else 'eager' if engine.profile.startswith('eager')
            else engine.kind)


def install_engine(name):
    """Install the record's overrides for legacy callers before capture."""
    from jetspec.engines.jetspec_plus_plus.kernels import install
    from jetspec.engines.jetspec_plus_plus.draft.method.ops import select_engine
    engine = PRESETS[name] if isinstance(name, str) else name
    from jetspec.engines.jetspec_plus_plus.profiles import require_live_execution
    require_live_execution(engine.profile)
    install(engine.ops)
    select_engine('fast' if engine.ops else None)
    return engine


@contextmanager
def engine_scope(name):
    with record_scope(PRESETS[name] if isinstance(name, str) else name) as tier:
        yield tier


def run_method(name, method, state):
    """Run every method step with this tier's overrides and automatic fallback."""
    with engine_scope(name):
        return method.run(state)
