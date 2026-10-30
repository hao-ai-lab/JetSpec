# Adaptation: package relocation and deferred optional dependencies.
"""The declared store for explicitly requested historical recipes and policies."""
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def recipe_index(root=ROOT):
    return json.loads((ROOT / 'launch/recipe-index.json').read_text())['rows']


def _recipes():
    return json.loads((ROOT / 'config/recipes.json').read_text())


def _stem(name, regime):
    return name if regime is None else f'{name}-{regime}'


def recipe_exists(root, name, *, regime=None):
    return _stem(name, regime) in _recipes()


def recipe_contents(root, name, *, regime=None):
    return json.dumps(_recipes()[_stem(name, regime)])


def policy_path(root, relative):
    return ROOT / 'config' / relative


def policy_contents(path):
    return Path(path).read_bytes()


def serving_lock_bytes(root):
    # No live serving lock exists yet. Keep this legacy probe explicit here.
    return (Path(root) / 'history/retired/serve/uv.lock').read_bytes()
