"""entropy_soft — per-depth fanout cap, smooth exponential decay on entropy.

Same signal as `entropy_gate` (per-depth marginal entropy) but a smooth
single-knob exponential mapping instead of the hard 3-region cutoff:

    H_d = -Σ_v q_d(v) log q_d(v)
    b_d = max(1, round(K · exp(-α · H_d)))

α ≥ 0. Smoother boundaries, no τ_high / τ_low to tune.

Identity recovery: α → 0 → exp(0) = 1 → b_d = K → crossproduct. The natural
baseline-recovering knob is α = 0.0.

Caveat: H_d's scale depends on whether the drafter gives full-vocab or
top-K-renorm logits, so α calibration shifts when the entropy normalization
changes. `entropy_topk` sidesteps this by always using top-K-renorm entropy
(bounded by log K).

Lineage: sweep id V2 (entropy_gate_marginal_soft).
"""
from __future__ import annotations

import math

import torch

from ptd.tree._core.base import DraftTree, TreeAlgorithm
from ptd.tree._core.fanout_cap_builder import build_with_per_depth_cap
from ptd.tree._core.registry import register_tree_algo


@register_tree_algo("entropy_soft")
class EntropyGateMarginalSoft(TreeAlgorithm):
    """Per-depth fanout cap, smooth exponential decay on marginal entropy."""

    def __init__(self, alpha: float = 0.0):
        if alpha < 0.0:
            raise ValueError(f"alpha must be >= 0; got {alpha}")
        self.alpha = float(alpha)

    def build(
        self,
        root_token: int,
        draft_logits: torch.Tensor,  # (1, D, V)
        block_size: int,
        tree_width: int,
        budget: int,
        device: torch.device,
        **kwargs,
    ) -> DraftTree:
        D_expected = block_size - 1
        if draft_logits.dim() != 3 or draft_logits.shape[0] != 1:
            raise ValueError(
                f"draft_logits must be (1, D, V); got {tuple(draft_logits.shape)}"
            )
        if draft_logits.shape[1] != D_expected:
            raise ValueError(
                f"draft_logits depth {draft_logits.shape[1]} != block_size-1 ({D_expected})"
            )

        log_probs = torch.log_softmax(draft_logits.squeeze(0), dim=-1)  # (D, V)
        topk_lp_t, topk_tok_t = torch.topk(log_probs, tree_width, dim=-1)
        topk_tokens_cpu = topk_tok_t.tolist()
        topk_logprobs_cpu = topk_lp_t.tolist()

        # Per-depth marginal entropy with -inf masking (sparse-logits safe).
        probs = log_probs.exp()
        finite_mask = torch.isfinite(log_probs)
        contrib = torch.where(finite_mask, probs * log_probs, torch.zeros_like(probs))
        H_per_depth = (-contrib.sum(dim=-1)).tolist()  # (D,)

        b_per_depth = [
            _soft_decay_cap(H_d, tree_width, self.alpha) for H_d in H_per_depth
        ]

        return build_with_per_depth_cap(
            root_token=int(root_token),
            topk_tokens_cpu=topk_tokens_cpu,
            topk_logprobs_cpu=topk_logprobs_cpu,
            b_per_depth=b_per_depth,
            budget=int(budget),
            device=device,
        )


def _soft_decay_cap(H_d: float, K: int, alpha: float) -> int:
    """b_d = max(1, round(K · exp(-α · H_d)))."""
    if alpha == 0.0:
        return K
    return max(1, int(round(K * math.exp(-alpha * H_d))))
