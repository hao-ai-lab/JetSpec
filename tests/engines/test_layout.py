"""Engine discovery stays lazy and legacy imports preserve object identity."""
import subprocess
import sys

import pytest


MODULES = {
    'engine': 'engine',
    'paged_kv_cache': 'serving.paged_kv_cache',
    'attn_metadata': 'serving.attn_metadata',
    'scheduler': 'serving.scheduler',
    'compiled_verify_stack': 'serving.compiled_verify_stack',
    'paged_attn_backend': 'serving.paged_attn_backend',
    'graph_capture': 'verify.graph_capture',
    'paged_tree_attn': 'kernels.paged_tree_attn',
    'paged_tree_attn_op': 'kernels.paged_tree_attn_op',
}


def test_registry_imports_no_models():
    result = subprocess.run([sys.executable, '-c', '''
import sys
import jetspec.engines
assert callable(jetspec.engines.get_engine)
assert not any(name.startswith(('torch', 'transformers', 'jetspec.models',
                                'jetspec.core', 'jetspec.engines.jetspec'))
               for name in sys.modules)
'''], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize('old_first', [True, False])
def test_every_legacy_module_has_one_identity(old_first):
    # The CPU interpreter has no Triton. Stub only its import-time decorators;
    # this gate checks module/object identity and never executes a GPU kernel.
    script = f'''
import importlib
import importlib.util
import sys
from types import ModuleType
for suffix, moved in {MODULES!r}.items():
    if suffix == 'paged_tree_attn' and importlib.util.find_spec('triton') is None:
        triton = ModuleType('triton')
        language = ModuleType('triton.language')
        triton.jit = lambda fn: fn
        language.constexpr = object()
        language.int64 = object()
        language.int32 = object()
        triton.language = language
        sys.modules['triton'] = triton
        sys.modules['triton.language'] = language
    old_path = 'jetspec.inference_engine.' + suffix
    new_path = 'jetspec.engines.jetspec.' + moved
    first, second = (old_path, new_path) if {old_first!r} else (new_path, old_path)
    old = importlib.import_module(first)
    new = importlib.import_module(second)
    assert old is new, suffix
    for name, value in vars(new).items():
        if not name.startswith('__'):
            assert getattr(old, name) is value, (suffix, name)
old = importlib.import_module('jetspec.inference_engine')
new = importlib.import_module('jetspec.engines.jetspec')
for name in old.__all__:
    assert getattr(old, name) is getattr(new, name)
from jetspec.engines import get_engine
assert get_engine('jetspec') is new.JetSpecEngine
try:
    get_engine('missing')
except KeyError:
    pass
else:
    raise AssertionError('unknown engine accepted')
'''
    result = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
