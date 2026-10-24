"""Chain-graph error audit for the 27B JetSpec++ Table 3 reads.

The Table 3 command puts this directory first on PYTHONPATH with TABLE3_CHAIN_AUDIT=1, as the
paper reads ran the audit. After each chain-graph run it prints the mirrored error word, which the
protocol requires on every round, and it fails the read on a nonzero word.

Adaptation: the paper launcher switched the audit on with W645_GATE_ACTIVE and W645_CHAIN_AUDIT.
The reader's launch contract now rejects W<digits>_ names as unbound serving knobs, so a host
variable outside the serving prefixes switches it on. The printed line keeps its original name.
"""
import importlib.abc
import importlib.machinery
import json
import os
import sys

TARGET = 'jetspec.engines.jetspec_plus_plus.draft.graph'


def patch(module):
    original = module.ChainGraph.run
    def audited(self, key, **kwargs):
        result = original(self, key, **kwargs)
        entry = self.prepare(key)
        if 'err_host' not in entry.buffers:
            raise AssertionError('chain audit needs the mirrored error word')
        error = int(entry.buffers['err_host'][0])
        print('W645_CHAIN_AUDIT ' + json.dumps(dict(error_bits=error)), flush=True)
        if error:
            raise AssertionError('CHAIN_ERR bits are nonzero')
        return result
    module.ChainGraph.run = audited


class Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name != TARGET:
            return None
        spec = importlib.machinery.PathFinder.find_spec(name, path)
        if spec is None or spec.loader is None:
            return spec
        original = spec.loader.exec_module
        def observed(module):
            original(module)
            patch(module)
        spec.loader.exec_module = observed
        return spec


if os.environ.get('TABLE3_CHAIN_AUDIT') == '1':
    sys.meta_path.insert(0, Finder())
