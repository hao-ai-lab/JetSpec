"""Reference compute ops with engine-specific overrides and automatic fallback."""
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps

_references = {}
_implementations = {}
_engine = ContextVar("jetspec_plus_plus_engine", default=None)


def reference_op(name):
    """Declare a reference; its public callable dispatches through the selected engine."""
    def decorate(fn):
        if name in _references:
            raise ValueError(f"method op already declared: {name}")
        _references[name] = fn

        @wraps(fn)
        def dispatch(*args, **kwargs):
            return _implementations.get((_engine.get(), name), fn)(*args, **kwargs)
        return dispatch
    return decorate


def reference(name):
    """Get the torch oracle, bypassing overrides (for parity checks)."""
    return _references[name]


def register_impl(op_name, fn, *, engine):
    reference(op_name)  # Catch misspelled names at registration.
    if not engine or not callable(fn):
        raise ValueError("an op implementation requires an engine name and callable")
    _implementations[engine, op_name] = fn




@contextmanager
def use_engine(engine):
    """Select overrides for this context; undeclared implementations fall back."""
    token = _engine.set(engine)
    try:
        yield
    finally:
        _engine.reset(token)
