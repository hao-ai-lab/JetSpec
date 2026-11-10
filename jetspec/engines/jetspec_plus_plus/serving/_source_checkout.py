"""Resolve comparison routes only when the caller explicitly selects one."""
from importlib import import_module


def baseline_module(name):
    module = 'bench.jetspec_plus_plus.baselines.' + name
    try:
        return import_module(module)
    except ModuleNotFoundError as error:
        if error.name and (error.name == 'bench' or module.startswith(error.name + '.')
                           or error.name == module):
            raise RuntimeError(f'{name} is available from a source checkout') from error
        raise
