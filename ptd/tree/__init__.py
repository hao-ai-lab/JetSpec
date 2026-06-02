"""Tree-drafting method layer — engine-agnostic, owner-separable.

This package is the *method*: it turns per-depth draft logits into a verification
tree and selects the accepted path. It is deliberately decoupled from any decode
engine — it imports nothing from `ptd.engine` / `ptd.draft` / `ptd.models`, depends
only on torch + numpy, and is consumed one-way (engine -> tree). Any verify backend
plugs in through the public contract below: this HF engine today, a vLLM / SGLang
integration tomorrow.

Public contract (the only surface an engine should import — never `ptd.tree._core`):

    get_algorithm(name).build(draft_logits, block_size, tree_width, budget, device)
        -> DraftTree                                  # construct the speculative tree
    build_ancestor_matrix(tree) -> (N, N) bool        # ancestor mask the engine folds
                                                      #   into its 4D attention mask
    tree_accept(tree, target_logits, temperature)
        -> (accepted_path, acceptance_length, correction_token)

Ships the V0 baseline (crossproduct); the V5 fanout-cap variant lands later.
"""
from ptd.tree._core.base import DraftTree, TreeAlgorithm
from ptd.tree._core.registry import get_algorithm, register_tree_algo, list_algorithms
from ptd.tree._core.ancestor import build_ancestor_matrix
from ptd.tree._core.accept import tree_accept
from ptd.tree import baselines  # noqa: F401  (import to register algorithms)

__all__ = [
    "DraftTree", "TreeAlgorithm",
    "get_algorithm", "register_tree_algo", "list_algorithms",
    "build_ancestor_matrix", "tree_accept",
]
