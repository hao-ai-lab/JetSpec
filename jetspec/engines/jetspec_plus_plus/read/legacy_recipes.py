# Adaptation: package relocation and deferred optional dependencies.
"""The declared store for explicitly requested historical recipes and policies."""
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FROZEN_PAPER_NAMES = frozenset({
    'jetspecpp-eager-1gpu-e6v3-b16-j3r2', 'jetspecpp-eager-1gpu-e6v3-b32-j3r2',
    'jetspecpp-eager-pair-e6v3-b16-j3r2', 'jetspecpp-eager-pair-e6v3-b32-j3r2',
    'jetspecpp-graphed-pair-e6-b16-chain-r4-j1', 'jetspecpp-graphed-pair-e6-b16-chain-r4-j2',
})


def recipe_index(root=ROOT):
    return json.loads((ROOT / 'read/recipe-index.json').read_text())['rows']


def recipe_path(root, name, *, regime=None):
    stem = name if regime is None else f'{name}-{regime}'
    return ROOT / 'configs' / f'{stem}.json'


def recipe_contents(root, name, *, regime=None):
    return recipe_path(root, name, regime=regime).read_text(encoding='utf-8')


def frozen_paper_recipes(root=ROOT):
    path = ROOT / 'configs/frozen-paper-recipes.txt'
    return json.loads(path.read_text(encoding='utf-8'))['recipes']


def policy_path(root, relative):
    return ROOT / 'configs' / relative


def policy_contents(path):
    return Path(path).read_bytes()


def serving_lock_bytes(root):
    # No live serving lock exists yet. Keep this legacy probe explicit here.
    return (Path(root) / 'history/retired/serve/uv.lock').read_bytes()
