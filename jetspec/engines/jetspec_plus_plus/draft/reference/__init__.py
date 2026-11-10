"""JetSpec++ torch method with lazy model and reader imports."""
import importlib

_LAZY = {
    'JetSpecPPConfig': 'jetspec.engines.jetspec_plus_plus.draft.reference.round',
    'JetSpecPPDrafter': 'jetspec.engines.jetspec_plus_plus.draft.reference.drafter',
}
__all__ = list(_LAZY)


def __getattr__(name):
    if name in _LAZY:
        return getattr(importlib.import_module(_LAZY[name]), name)
    raise AttributeError(name)
