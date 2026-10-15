"""Data-only names for reading historical recipe and manifest records."""
import json
from functools import lru_cache
from pathlib import Path

from .config import flatten




def _environment_name(category, field):
    return next(name for name, path in _bindings()['fields'][category].items() if path == field)


