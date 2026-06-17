"""online_warmup — learn the per-depth fanout cap online, per request.

STATUS: PLANNED (placeholder, not registered). Lineage: design id B5.

Mechanism: no offline profiler. Accumulate per-depth acceptance over the first
N_warmup spec-steps of a request, then switch the per-depth cap to that
prompt-local histogram for the rest of the request. Short prompts (no warmup
data) fall back to the global cap.

Potential: the niche is long, single-prompt-dominated generations (agentic /
long reasoning traces) where the warmup cost amortizes and per-depth acceptance
is stationary WITHIN a prompt but varies ACROSS prompts. Per-depth granularity
(top2gap's level), trading the free live gap for a request-specific estimate.

Needs: a serving-loop warmup-feedback hook — the engine must thread per-depth
accept counts back into the build call across steps of one request. That's a
vLLM-phase capability (the offline HF loop doesn't carry per-request state).
"""
from __future__ import annotations

import torch

from jetflow.tree._core.base import DraftTree, TreeAlgorithm


class OnlineWarmup(TreeAlgorithm):
    """PLANNED — online per-request warmup of per-depth caps. Not implemented."""

    def build(self, root_token: int, draft_logits: torch.Tensor, block_size: int,
              tree_width: int, budget: int, device: torch.device, **kwargs) -> DraftTree:
        raise NotImplementedError(
            "online_warmup (B5) is planned. Needs a serving-loop warmup-feedback hook "
            "(per-request per-depth accept counts); see jetflow/tree/ROADMAP.md."
        )
