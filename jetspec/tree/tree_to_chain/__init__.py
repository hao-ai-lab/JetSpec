"""Compatibility exports for jetspec.methods.jetspec.tree.tree_to_chain."""
from importlib import import_module as _import_module

_impl = _import_module("jetspec.methods.jetspec.tree.tree_to_chain")
__all__ = getattr(_impl, "__all__", [name for name in vars(_impl) if not name.startswith("_")])


def __getattr__(name):
    return getattr(_impl, name)


def __dir__():
    return dir(_impl)
