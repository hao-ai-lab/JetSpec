"""entropy_topk — per-depth fanout cap, 3-region rule on top-K-renorm entropy.

Like `entropy_gate` but computes entropy over the RENORMALIZED top-K
distribution rather than the full vocab:

    q̃_d(j) = q_d^(j) / Σ_{i≤K} q_d^(i)        (top-K renormalize)
    H̃_d = -Σ_{j≤K} q̃_d(j) log q̃_d(j)        (bounded by log K)
    b_d  = hard 3-region rule on H̃_d          (same as entropy_gate)

Why: `entropy_gate`'s H_d depends on whether the drafter passes full-vocab or
top-K-renorm logits. Pinning entropy to top-K-renorm makes τ_high / τ_low
calibration bounded (max H̃_d = log K ≈ 1.95 for K=7), so a threshold above
log K that silently never fires becomes impossible.

Identity recovery: τ_high = +∞ (any value > log K), τ_low = -∞ → no region fires
→ b_d = K → crossproduct. (τ_high = log K does NOT recover identity: a uniform
top-K has H̃ = log K exactly, and the ≥ high-region test would clamp it to a
chain. The runtime default τ_high = math.inf is the safe identity knob.)

Caveat: top-K renorm discards the tail. A flat-tail distribution (rank-1
dominant + a long tail of near-equal tokens) reads as "decisive" under
top-K-renorm but "uncertain" under full vocab. `entropy_gate` keeps that
nuance; this loses it.

Lineage: sweep id V3 (entropy_gate_per_node).
"""
from __future__ import annotations

import math

import torch

from ptd.tree._core.base import DraftTree, TreeAlgorithm
from ptd.tree._core.fanout_cap_builder import build_with_per_depth_cap
from ptd.tree._core.registry import register_tree_algo


@register_tree_algo("entropy_topk")
class EntropyGatePerNode(TreeAlgorithm):
    """Per-depth fanout cap, hard 3-region rule on top-K-renorm entropy."""

    def __init__(self, tau_high: float = math.inf, tau_low: float = -math.inf):
        self.tau_high = float(tau_high)
        self.tau_low = float(tau_low)
        if self.tau_low > self.tau_high:
            raise ValueError(
                f"tau_low ({self.tau_low}) must be <= tau_high ({self.tau_high})"
            )

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

        # Top-K-renorm entropy: q̃_j = q^(j) / Σ_{i≤K} q^(i), bounded by log K.
        topk_p = topk_lp_t.exp()                              # (D, K)
        topk_p_renorm = topk_p / topk_p.sum(dim=-1, keepdim=True)
        topk_logp_renorm = torch.log(topk_p_renorm.clamp_min(1e-30))
        H_per_depth = (-(topk_p_renorm * topk_logp_renorm).sum(dim=-1)).tolist()

        b_per_depth = [
            _three_region_cap(H_d, tree_width, self.tau_high, self.tau_low)
            for H_d in H_per_depth
        ]

        return build_with_per_depth_cap(
            root_token=int(root_token),
            topk_tokens_cpu=topk_tokens_cpu,
            topk_logprobs_cpu=topk_logprobs_cpu,
            b_per_depth=b_per_depth,
            budget=int(budget),
            device=device,
        )


def _three_region_cap(H_d: float, K: int, tau_high: float, tau_low: float) -> int:
    """Hard 3-region rule, identical to entropy_gate's."""
    if H_d >= tau_high:
        return 1
    if H_d <= tau_low:
        return K
    denom = tau_high - tau_low
    if not math.isfinite(denom) or denom <= 0:
        return K
    return max(1, int(round(K * (tau_high - H_d) / denom)))
