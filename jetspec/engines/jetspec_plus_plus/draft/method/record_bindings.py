"""Data-only names for reading historical recipe and manifest records."""
import json
from functools import lru_cache
from pathlib import Path

from .config import flatten


@lru_cache(maxsize=1)
def _bindings():
    return json.loads(Path(__file__).with_suffix('.json').read_text())


def _environment_name(category, field):
    return next(name for name, path in _bindings()['fields'][category].items() if path == field)


def _config_values(run_config):
    if run_config is None:
        return {}
    provided = getattr(run_config, 'provided_values', None)
    if provided is not None:
        return {key: value for key, value in provided.items() if value is not None}
    return flatten(run_config)
