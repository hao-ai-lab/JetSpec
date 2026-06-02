"""entropy_gate — per-depth fanout cap, hard 3-region rule on marginal entropy.

Uses the per-depth marginal entropy of the drafter's distribution as the
fanout signal, mapped through a hard 3-region cap:

    H_d = -Σ_v q_d(v) log q_d(v)                     (per-depth marginal entropy)
    b_d = 1                                           if H_d ≥ τ_high
    b_d = K                                           if H_d ≤ τ_low
    b_d = round(K · (τ_high - H_d) / (τ_high - τ_low))  otherwise

Low entropy = drafter is confident = fanout is well-supported. High entropy =
drafter is unsure = fanout becomes lottery, so chain-extend instead.

Identity recovery: τ_high = +∞, τ_low = -∞ → no region fires → b_d = K
everywhere → byte-identical to crossproduct.

Caveat: marginal entropy is averaged over the depth's whole distribution; a
depth where some branches are confident and others aren't gets a single cap
that hurts the confident branches. `entropy_topk` addresses this by computing
entropy over the renormalized top-K per node. Also: if the drafter hands us
top-K-sparse logits, H_d is top-K-renorm entropy (bounded by log K), not
full-vocab — τ_high / τ_low must be calibrated to whichever input is given.

Lineage: sweep id V1 (entropy_gate_marginal_hard).
"""
from __future__ import annotations

import math

import torch

from ptd.tree._core.base import DraftTree, TreeAlgorithm
from ptd.tree._core.fanout_cap_builder import build_with_per_depth_cap
from ptd.tree._core.registry import register_tree_algo


@register_tree_algo("entropy_gate")
class EntropyGateMarginalHard(TreeAlgorithm):
    """Per-depth fanout cap, hard 3-region rule on marginal entropy."""

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

        # Per-depth marginal entropy with -inf masking (sparse-logits safe).
        probs = log_probs.exp()
        finite_mask = torch.isfinite(log_probs)
        contrib = torch.where(finite_mask, probs * log_probs, torch.zeros_like(probs))
        H_per_depth = (-contrib.sum(dim=-1)).tolist()  # (D,)

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
    """Hard 3-region rule.

    Identity case: if both thresholds are saturated (tau_high=+inf,
    tau_low=-inf), there's no interpolation region and no cap fires —
    return K so the algorithm reduces to crossproduct cleanly.
    """
    if H_d >= tau_high:
        return 1
    if H_d <= tau_low:
        return K
    denom = tau_high - tau_low
    if not math.isfinite(denom) or denom <= 0:
        # No effective interpolation window — degenerate to full fanout.
        return K
    return max(1, int(round(K * (tau_high - H_d) / denom)))
