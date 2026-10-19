"""Compatibility exports for the JetSpec serving engine."""
from importlib import import_module
from importlib.abc import Loader, MetaPathFinder
from importlib.util import spec_from_loader
import sys

from jetspec.engines.jetspec import JetSpecEngine, SamplingParams

__all__ = ['JetSpecEngine', 'SamplingParams']

_MODULES = {
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


class _AliasLoader(Loader):
    def __init__(self, path):
        self.path = path

    def create_module(self, spec):
        module = import_module(self.path)
        self.spec = module.__spec__
        return module

    def exec_module(self, module):
        # Keep the canonical spec too, so reload targets the owning module.
        module.__spec__ = self.spec


class _AliasFinder(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        prefix = __name__ + '.'
        if fullname.startswith(prefix):
            moved = _MODULES.get(fullname[len(prefix):])
            if moved is not None:
                loader = _AliasLoader('jetspec.engines.jetspec.' + moved)
                return spec_from_loader(fullname, loader)


# Resolve submodules lazily: importing the package must not require Triton.
sys.meta_path.insert(0, _AliasFinder())
