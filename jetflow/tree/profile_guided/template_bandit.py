"""template_bandit — online bandit over a small set of shape templates.

STATUS: PLANNED (placeholder, not registered). Lineage: design id B7.

Mechanism: no profiler. Carry a tiny epsilon-greedy bandit over ~5 fixed shape
templates and learn online which template a given budget/workload prefers, from
the realized accept-length reward.

Potential: zero-calibration self-tuning — at low budget the chain template wins,
at high budget the wide (≈accum_logp) template is fine, and a working bandit
discovers the budget-appropriate endpoint with no offline pass. Primary value is
robustness (auto-picks chain at small B) for deployments that can't profile;
expect it to approach but not beat top2gap_fanout (the template family is coarse
and per-stream, not per-depth).

Needs: a serving-loop reward-feedback hook — the engine must thread
reward = accept_len/budget back to credit the previously chosen arm across
requests. vLLM-phase (per-request reward signal).
"""
from __future__ import annotations

import torch

from jetflow.tree._core.base import DraftTree, TreeAlgorithm


class TemplateBandit(TreeAlgorithm):
    """PLANNED — epsilon-greedy bandit over shape templates. Not implemented."""

    def build(self, root_token: int, draft_logits: torch.Tensor, block_size: int,
              tree_width: int, budget: int, device: torch.device, **kwargs) -> DraftTree:
        raise NotImplementedError(
            "template_bandit (B7) is planned. Needs a serving-loop reward-feedback hook "
            "(per-request accept_len reward); see jetflow/tree/ROADMAP.md."
        )
