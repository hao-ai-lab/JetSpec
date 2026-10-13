"""The optimization order as data. Method components never depend on presets."""
from contextlib import contextmanager
from dataclasses import dataclass, replace

from jetspec.engines.jetspec_plus_plus.draft.method.ops import use_engine

from jetspec.engines.jetspec_plus_plus.serving.presets.budget import DeadlineBudget; from jetspec.engines.jetspec_plus_plus.serving.presets.budget import NullBudget
from jetspec.engines.jetspec_plus_plus.serving.presets.transport import Lockstep; from jetspec.engines.jetspec_plus_plus.serving.presets.transport import Pair


@dataclass(frozen=True)
class Tier:
    ops: tuple
    graphs: str
    compile: tuple
    verify: str
    budget: type
    transport: type


EAGER = Tier((), 'none', (), 'method', NullBudget, Lockstep)
REFERENCE = Tier((), 'target', (), 'upstream', NullBudget, Lockstep)
FAST = Tier(('apply_pins',), 'segments', ('adapter', 'draft'),
            'multirow', DeadlineBudget, Pair)
EVAL = replace(FAST, budget=NullBudget, transport=Lockstep)
PRESETS = dict(eager=EAGER, reference=REFERENCE, eval=EVAL, fast=FAST)


def engine_name(environ, *, regime=None):
    if environ.get('JETSPEC_ENGINE') in PRESETS:
        return environ['JETSPEC_ENGINE']
    if environ.get('JETSPEC_EVAL_MODE', 'off') != 'off':
        return 'eval'
    if environ.get('JETSPEC_VERIFY_REFERENCE', '0') == '1':
        return 'reference'
    if regime == 'eager' or environ.get('TORCHDYNAMO_DISABLE') == '1':
        return 'eager'
    return 'fast'


@contextmanager
def engine_scope(name):
    from jetspec.engines.jetspec_plus_plus.serving.presets.ops import install
    tier = PRESETS[name]
    install(tier.ops)
    with use_engine(name):
        yield tier


def run_method(name, method, state):
    """Run every method step with this tier's overrides and automatic fallback."""
    with engine_scope(name):
        return method.run(state)
