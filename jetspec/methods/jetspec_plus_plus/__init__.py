"""JetSpec++ torch method with lazy model and reader imports."""
import importlib

_LAZY = {
    'JetSpecPPConfig': 'jetspec.methods.jetspec_plus_plus.round',
    'JetSpecPPDrafter': 'jetspec.methods.jetspec_plus_plus.drafter',
}
__all__ = list(_LAZY)


def __getattr__(name):
    if name in _LAZY:
        return getattr(importlib.import_module(_LAZY[name]), name)
    raise AttributeError(name)
