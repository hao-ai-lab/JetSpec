"""Public top-k tree construction — the engine adapter entry point.

Engines whose proposer already produces per-depth top-k (tokens + full-vocab
log-probs) — e.g. the vLLM fork's DFlash proposer — call this instead of
``get_algorithm(name).build(dense_logits, ...)``, avoiding a vocab-wide logits
tensor per step. Each algorithm exposes
``caps_from_topk(topk_logprobs, tree_width, **kwargs) -> per-depth fanout cap``;
the shared heap builder turns (top-k, caps, budget) into a DraftTree.

The top-k log-probs are used AS GIVEN (already full-vocab-normalised by the
proposer), so the cumulative-log-prob heap ordering is faithful — reconstructing
a dense logits tensor and re-running log_softmax would renormalise over the
top-k only and distort accum_logp's cross-depth ordering. Result is identical
to ``build()`` when the top-k came from ``log_softmax(dense_logits)`` (build()
routes through the same ``caps_from_topk``)."""
from __future__ import annotations

import torch

from .base import DraftTree
from .fanout_cap_builder import build_with_per_depth_cap
from .registry import get_algorithm


def _to_rows(x):
    return x.tolist() if torch.is_tensor(x) else [list(r) for r in x]


def build_from_topk(
    name: str,
    root_token: int,
    topk_tokens,                       # (D, K) tensor or list[list[int]]
    topk_logprobs,                     # (D, K) tensor or list[list[float]] — full-vocab log-probs
    budget: int,
    device: torch.device,
    algo_kwargs: dict | None = None,
    tree_width: int | None = None,
    **build_kwargs,                    # forwarded to caps_from_topk (e.g. profile_table=, prompt_info=)
) -> DraftTree:
    """Build a DraftTree for `name` from pre-extracted per-depth top-k.

    `tree_width` defaults to the supplied top-k width K. `algo_kwargs` are the
    algorithm's constructor knobs (e.g. {"beta": 2.0}); `build_kwargs` (e.g.
    profile_table=) are forwarded to caps_from_topk."""
    topk_tokens_cpu = _to_rows(topk_tokens)
    topk_logprobs_cpu = _to_rows(topk_logprobs)
    K = len(topk_logprobs_cpu[0]) if topk_logprobs_cpu else (tree_width or 0)
    algo = get_algorithm(name, **(algo_kwargs or {}))
    caps_fn = getattr(algo, "caps_from_topk", None)
    if caps_fn is None:
        raise NotImplementedError(
            f"tree algorithm {name!r} has no caps_from_topk; build_from_topk supports the "
            f"per-depth-cap algorithms (accum_logp, top2gap_fanout, depth_rank_histogram)."
        )
    b_per_depth = caps_fn(topk_logprobs_cpu, tree_width if tree_width is not None else K, **build_kwargs)
    return build_with_per_depth_cap(
        root_token=int(root_token),
        topk_tokens_cpu=topk_tokens_cpu,
        topk_logprobs_cpu=topk_logprobs_cpu,
        b_per_depth=b_per_depth,
        budget=int(budget),
        device=device,
    )
