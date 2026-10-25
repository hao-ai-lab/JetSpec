"""Method layout preserves public identities, registry state, and draft-head bytes."""
import hashlib
import importlib
from pathlib import Path
import subprocess
import sys
from types import ModuleType

import pytest


TREE_MODULES = [
    '', '_core', '_core.accept', '_core.ancestor', '_core.base',
    '_core.extend', '_core.fanout_cap_builder', '_core.registry', '_core.topk_build',
    'baselines', 'baselines.accum_logp', 'layer_conditional',
    'layer_conditional.path_conditional_refresh', 'profile_guided',
    'profile_guided.depth_rank_histogram', 'profile_guided.online_warmup',
    'profile_guided.template_bandit', 'semantic_aware',
    'semantic_aware.class_histogram', 'semantic_aware.reasoning_router',
    'semantic_aware.task_router', 'tree_to_chain', 'tree_to_chain.fanout_cap',
    'tree_to_chain.fanout_cap.top2gap',
]
PUBLIC_PATHS = [
    ('jetspec.tree' + ('.' + suffix if suffix else ''),
     'jetspec.methods.jetspec.tree' + ('.' + suffix if suffix else ''))
    for suffix in TREE_MODULES
] + [
    ('jetspec.draft', 'jetspec.methods.jetspec.draft'),
    ('jetspec.draft_head_adapter', 'jetspec.methods.jetspec.draft_head_adapter'),
    ('jetspec.models.draft_head', 'jetspec.methods.jetspec.draft_head'),
]


def test_public_tree_identity():
    old = importlib.import_module('jetspec.tree')
    new = importlib.import_module('jetspec.methods.jetspec.tree')
    for name in old.__all__:
        assert getattr(old, name) is getattr(new, name)


@pytest.mark.parametrize('old_path,new_path', PUBLIC_PATHS)
def test_all_public_submodules_reexport_identical_objects(old_path, new_path):
    old = importlib.import_module(old_path)
    new = importlib.import_module(new_path)
    for name, value in vars(new).items():
        if not name.startswith('__') and not isinstance(value, ModuleType):
            assert getattr(old, name) is value, (old_path, name)


@pytest.mark.parametrize('old_first', [True, False])
def test_import_orders_share_one_registry(old_first):
    paths = PUBLIC_PATHS if old_first else [(new, old) for old, new in PUBLIC_PATHS]
    script = f'''
import importlib
for first, second in {paths!r}:
    importlib.import_module(first)
    importlib.import_module(second)
old = importlib.import_module('jetspec.tree._core.registry')
new = importlib.import_module('jetspec.methods.jetspec.tree._core.registry')
assert old._REGISTRY is new._REGISTRY
assert old.list_algorithms() == [
    'accum_logp', 'class_histogram', 'depth_rank_histogram',
    'reasoning_router', 'task_router', 'top2gap_fanout']
assert len(set(new._REGISTRY.values())) == 6
for name in old.list_algorithms():
    assert type(old.get_algorithm(name)) is type(new.get_algorithm(name))
'''
    result = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize('path', ['jetspec', 'jetspec.tree',
                                  'jetspec.methods.jetspec.tree'])
def test_import_stays_model_free(path):
    script = f'''
import sys
import {path}
for name in sys.modules:
    assert not name.startswith(('transformers', 'jetspec.models', 'jetspec.core',
                               'jetspec.inference_engine',
                               'jetspec.methods.jetspec.draft')), name
'''
    result = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_frozen_draft_head_bytes_equal_public_baseline():
    # Byte-identical to 2c7b3fa:jetspec/models/draft_head.py, including the frozen region.
    moved = Path(importlib.import_module('jetspec.methods.jetspec.draft_head').__file__).read_bytes()
    assert hashlib.sha256(moved).hexdigest() == 'cfa7f8d22cf5ce0a28f7208425b979acc57a984a02bba76bee742befe482a008'

