"""Tree-drafting method layer — engine-agnostic, owner-separable.

This package is the *method*: it turns per-depth draft logits into a verification
tree and selects the accepted path. It is deliberately decoupled from any decode
engine — it imports nothing from `jetflow.core` / `jetflow.draft` / `jetflow.models`, depends
only on torch + numpy, and is consumed one-way (engine -> tree). Any verify backend
plugs in through the public contract below: this HF engine today, a vLLM / SGLang
integration tomorrow.

Public contract (the only surface an engine should import — never `jetflow.tree._core`):

    get_algorithm(name).build(draft_logits, block_size, tree_width, budget, device)
        -> DraftTree                                  # construct the speculative tree
    build_from_topk(name, root_token, topk_tokens, topk_logprobs, budget, device, ...)
        -> DraftTree                                  # same, from pre-extracted per-depth
                                                      #   top-k (engines that already have it)
    build_ancestor_matrix(tree) -> (N, N) bool        # ancestor mask the engine folds
                                                      #   into its 4D attention mask
    tree_accept(tree, target_logits, temperature)
        -> (accepted_path, acceptance_length, correction_token)

Bundled algorithms, by family:
- `baselines/`      — accum_logp (the full-fanout baseline).
- `tree_to_chain/`  — single-pass uncertainty-aware shaping (top2gap_fanout).
- `semantic_aware/` — prompt-adaptive routing (task_router, reasoning_router,
                      class_histogram).
- `profile_guided/` — offline-profile-driven shaping (depth_rank_histogram).
All recover accum_logp at their identity knob. Importing this package
registers all of them; `list_algorithms()` enumerates the registry.
"""
from jetflow.tree._core.base import DraftTree, TreeAlgorithm
from jetflow.tree._core.registry import get_algorithm, register_tree_algo, list_algorithms
from jetflow.tree._core.ancestor import build_ancestor_matrix
from jetflow.tree._core.accept import gpu_tree_accept, tree_accept
from jetflow.tree._core.extend import should_extend, splice_extension
from jetflow.tree._core.topk_build import build_from_topk
from jetflow.tree import baselines        # noqa: F401  (import to register algorithms)
from jetflow.tree import tree_to_chain    # noqa: F401  (import to register algorithms)
from jetflow.tree import semantic_aware   # noqa: F401  (import to register algorithms)
from jetflow.tree import profile_guided   # noqa: F401  (import to register algorithms)

__all__ = [
    "DraftTree", "TreeAlgorithm",
    "get_algorithm", "register_tree_algo", "list_algorithms",
    "build_ancestor_matrix", "gpu_tree_accept", "tree_accept", "build_from_topk",
    "should_extend", "splice_extension",
]
