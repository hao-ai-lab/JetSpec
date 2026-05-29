"""Tree-construction algorithms (migrated from spec-infer; engine-agnostic).

Ships the V0 baseline (crossproduct); the V5 fanout-cap variant lands later. Each algorithm
consumes per-depth draft logits and returns a DraftTree; the engine verifies it
with a 4D ancestor mask + tree_accept.
"""
from ptd.tree._core.base import DraftTree, TreeAlgorithm
from ptd.tree._core.registry import get_algorithm, register_tree_algo, list_algorithms
from ptd.tree import baselines  # noqa: F401  (import to register algorithms)

__all__ = ["DraftTree", "TreeAlgorithm", "get_algorithm", "register_tree_algo", "list_algorithms"]
