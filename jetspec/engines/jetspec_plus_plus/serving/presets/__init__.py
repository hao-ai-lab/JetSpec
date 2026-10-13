"""Four execution presets over the shared JetSpec method."""
from importlib import import_module

__all__ = ['EAGER', 'REFERENCE', 'EVAL', 'FAST', 'PRESETS',
           'engine_scope', 'run_method', 'register_optimization']


