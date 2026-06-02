"""prob_mass — per-depth fanout cap from top-K cumulative probability mass.

Uses how much of the distribution the top-K candidates capture as the fanout
signal:

    m_d = Σ_{j=1..K} q_d^{(j)}                       (top-K cumulative mass)
    b_d = max(1, ceil(K · (m_d - m_0) / (1 - m_0)))

If top-K captures most of the mass (m_d ≈ 1) there's no tail to worry about,
so fanout K is well-supported. If m_d is low, top-K covers only a fraction of
the distribution — the verifier may land outside top-K and fanout becomes
lottery.

Differs from the entropy variants in axis: "how much support is the tree
capturing" rather than "how concentrated the distribution is." Highly
correlated with entropy in normal regimes; useful as a control for entropy.

Identity recovery: m_0 → -∞ → b_d → K → crossproduct. The natural
baseline-recovering knob is a large-negative m_0 (m_d ∈ [0, 1]).

Lineage: sweep id V6 (prob_mass_concentration).
"""
from __future__ import annotations

import math

import torch

from ptd.tree._core.base import DraftTree, TreeAlgorithm
from ptd.tree._core.fanout_cap_builder import build_with_per_depth_cap
from ptd.tree._core.registry import register_tree_algo


@register_tree_algo("prob_mass")
class ProbMassConcentration(TreeAlgorithm):
    """Per-depth fanout cap from top-K cumulative probability mass."""

    def __init__(self, m_0: float = 0.5):
        if m_0 >= 1.0:
            raise ValueError(
                f"m_0 must be < 1.0 (denominator (1 - m_0) must be > 0); got {m_0}"
            )
        self.m_0 = float(m_0)

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

        # Top-K cumulative probability mass per depth.
        # Use raw probabilities, not log-probs (mass is in linear space).
        topk_probs = topk_lp_t.exp()  # (D, K)
        m_per_depth = topk_probs.sum(dim=-1).tolist()  # (D,)

        denom = 1.0 - self.m_0
        b_per_depth = [
            max(1, int(math.ceil(tree_width * (m_d - self.m_0) / denom)))
            for m_d in m_per_depth
        ]

        return build_with_per_depth_cap(
            root_token=int(root_token),
            topk_tokens_cpu=topk_tokens_cpu,
            topk_logprobs_cpu=topk_logprobs_cpu,
            b_per_depth=b_per_depth,
            budget=int(budget),
            device=device,
        )
