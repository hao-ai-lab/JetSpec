"""Reference compute ops with engine-specific overrides and automatic fallback."""
from contextlib import contextmanager
from functools import wraps

_references = {}
_implementations = {}
_engine = None


def reference_op(name):
    """Declare a reference; its public callable dispatches through the selected engine."""
    def decorate(fn):
        if name in _references:
            raise ValueError(f"method op already declared: {name}")
        _references[name] = fn

        @wraps(fn)
        def dispatch(*args, **kwargs):
            return _implementations.get((_engine, name), fn)(*args, **kwargs)
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


def unregister_impl(op_name, *, engine):
    _implementations.pop((engine, op_name), None)


def select_engine(engine):
    """Select overrides for the whole process, so worker threads see them; return the previous."""
    global _engine
    previous, _engine = _engine, engine
    return previous


@contextmanager
def use_engine(engine):
    """Select overrides for this block, process-wide; undeclared implementations fall back."""
    previous = select_engine(engine)
    try:
        yield
    finally:
        select_engine(previous)
