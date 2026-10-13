"""semantic_aware — prompt-adaptive tree shaping (route by prompt, not logits alone).

Each algorithm derives the tree shape from a prompt-level signal supplied via
the `prompt_info` kwarg on `build()` (a task label, reasoning mode, or decoded
text). When `prompt_info` is None they fall back to a logit-fingerprint
heuristic over `draft_logits`, so every variant is runnable without the routing
input wired — and recovers accum_logp at its `force_baseline` knob.

Importing this package registers every algorithm via @register_tree_algo.
"""
from . import task_router        # noqa: F401  (task_router)
from . import reasoning_router   # noqa: F401  (reasoning_router)
from . import class_histogram    # noqa: F401  (class_histogram)
