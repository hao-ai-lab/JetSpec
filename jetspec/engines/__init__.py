"""Lazy serving-engine discovery; importing this package loads no models."""
from importlib import import_module

_ENGINES = {'jetspec': 'jetspec.engines.jetspec.engine'}


def get_engine(name):
    """Return the registered engine class without constructing a model."""
    return import_module(_ENGINES[name]).JetSpecEngine
