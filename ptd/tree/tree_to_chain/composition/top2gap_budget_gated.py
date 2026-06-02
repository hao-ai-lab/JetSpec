"""top2gap_budget_gated — the top2gap fanout cap, gated by a budget schedule.

`top2gap_fanout` wins at low budget by collapsing the tree to a near-chain. At
high budget, where crossproduct's branchy shape is correct (there's room to
expand everything), that collapse can overcorrect and hurt accept length. This
gates the collapse by budget so it only fires when budget is tight:

    σ(g_d) = sigmoid(-β · (g_d - g_0))          (top2gap's per-depth sigmoid)
    t(B)   = exp(-B / B_0)                       (budget weight)
    b_d    = max(1, round(K · (1 - t(B) · (1 - σ(g_d)))))

- At low B (t→1):  b_d → round(K · σ(g_d))   ≡ top2gap alone (full collapse on big gap)
- At high B (t→0): b_d → round(K · 1) = K     ≡ crossproduct (no collapse)
- Smooth interpolation between.

Knobs (3): β and g_0 from top2gap; B_0 sets the budget crossover. Defaults
β=2.0, g_0=1.0, B_0=16 give t(16) ≈ 0.37, t(63) ≈ 0.02, t(127) ≈ 4e-4.

Identity recovery: B_0 → 0 (t → 0) recovers crossproduct; B_0 → ∞ (t → 1)
recovers top2gap. The natural baseline-recovering point is B_0 = 0.

Caveat: 3 knobs are harder to tune than top2gap's 2 — hold β/g_0 at top2gap's
winning values and sweep only B_0.

Lineage: sweep id V14 (v5_budget_gated).
"""
from __future__ import annotations

import math

import torch

from ptd.tree._core.base import DraftTree, TreeAlgorithm
from ptd.tree._core.fanout_cap_builder import build_with_per_depth_cap
from ptd.tree._core.registry import register_tree_algo


@register_tree_algo("top2gap_budget_gated")
class V5BudgetGated(TreeAlgorithm):
    """top2gap fanout cap weighted by exp(-B/B_0); collapses at low B, opens at high B."""

    def __init__(self, beta: float = 2.0, g_0: float = 1.0, B_0: float = 16.0):
        self.beta = float(beta)
        self.g_0 = float(g_0)
        self.B_0 = float(B_0)
        if self.B_0 < 0.0:
            raise ValueError(f"B_0 must be >= 0; got {self.B_0}")

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
        if tree_width < 2:
            # No "rank-2" exists; degenerate to chain.
            log_probs = torch.log_softmax(draft_logits.squeeze(0), dim=-1)
            topk_lp_t, topk_tok_t = torch.topk(log_probs, max(tree_width, 1), dim=-1)
            return build_with_per_depth_cap(
                root_token=int(root_token),
                topk_tokens_cpu=topk_tok_t.tolist(),
                topk_logprobs_cpu=topk_lp_t.tolist(),
                b_per_depth=[1] * D_expected,
                budget=int(budget),
                device=device,
            )

        log_probs = torch.log_softmax(draft_logits.squeeze(0), dim=-1)
        topk_lp_t, topk_tok_t = torch.topk(log_probs, tree_width, dim=-1)
        topk_tokens_cpu = topk_tok_t.tolist()
        topk_logprobs_cpu = topk_lp_t.tolist()

        # Budget weight: t = exp(-B / B_0). At B_0 = 0 treat t = 0 (≡ crossproduct).
        if self.B_0 <= 0.0:
            t_B = 0.0
        else:
            t_B = math.exp(-float(budget) / self.B_0)

        gaps = (topk_lp_t[:, 0] - topk_lp_t[:, 1]).tolist()  # (D,)
        b_per_depth = [
            _budget_gated_cap(g_d, tree_width, self.beta, self.g_0, t_B)
            for g_d in gaps
        ]

        return build_with_per_depth_cap(
            root_token=int(root_token),
            topk_tokens_cpu=topk_tokens_cpu,
            topk_logprobs_cpu=topk_logprobs_cpu,
            b_per_depth=b_per_depth,
            budget=int(budget),
            device=device,
        )


def _budget_gated_cap(
    g_d: float, K: int, beta: float, g_0: float, t_B: float
) -> int:
    """b_d = max(1, round(K · (1 - t_B · (1 - σ(g_d)))))."""
    arg = -beta * (g_d - g_0)
    if arg >= 0:
        sigma = 1.0 / (1.0 + math.exp(-arg))
    else:
        e = math.exp(arg)
        sigma = e / (1.0 + e)
    return max(1, int(round(K * (1.0 - t_B * (1.0 - sigma)))))
