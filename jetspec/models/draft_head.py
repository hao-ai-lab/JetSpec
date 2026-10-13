"""Compatibility alias for jetspec.methods.jetspec.draft_head."""
from importlib import import_module as _import_module
import sys as _sys

_sys.modules[__name__] = _import_module("jetspec.methods.jetspec.draft_head")
