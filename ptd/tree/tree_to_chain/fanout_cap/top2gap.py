"""top2gap_fanout — per-depth fanout cap from the top-2 logprob gap.

The sweep winner (sweep id: V5_top2_gap_fanout). Replaces entropy with the
rank-1 vs rank-2 logprob gap as the per-depth fanout signal:

    g_d = log q_d^{(1)} - log q_d^{(2)}     (top-2 gap, positive)
    b_d = max(1, round(K · σ(-β · (g_d - g_0))))

Large gap = drafter is decisive = rank-2 is noise = fanout waste.
Small gap = drafter splits = both candidates plausible = fanout makes sense.

Differs from the entropy variants in what signal it uses: entropy responds to
long tails, gap responds to top-vs-runner-up. The verifier picks ONE child per
parent, so the gap is what acceptance actually cares about — which is why this
beats the entropy gates across budgets/datasets.

Identity recovery: β → 0 gives σ(0) = 0.5 → b_d ≈ K/2, NOT baseline. To recover
crossproduct exactly: β > 0 with g_0 very LARGE positive (e.g. β=1, g_0=1e9) →
σ → 1 → b_d → K (byte-identical to crossproduct).

Caveat: insensitive to tail mass; tricked by flat rankings where rank-1 ≈ rank-2
but rank-3..K also matter.
"""
from __future__ import annotations

import math

import torch

from ptd.tree._core.base import DraftTree, TreeAlgorithm
from ptd.tree._core.fanout_cap_builder import build_with_per_depth_cap
from ptd.tree._core.registry import register_tree_algo


@register_tree_algo("top2gap_fanout")
class Top2GapFanout(TreeAlgorithm):
    """Per-depth fanout cap from top-2 logprob gap via sigmoid schedule."""

    def __init__(self, beta: float = 1.0, g_0: float = 1.0):
        self.beta = float(beta)
        self.g_0 = float(g_0)

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

        log_probs = torch.log_softmax(draft_logits.squeeze(0), dim=-1)  # (D, V)
        topk_lp_t, topk_tok_t = torch.topk(log_probs, tree_width, dim=-1)
        topk_tokens_cpu = topk_tok_t.tolist()
        topk_logprobs_cpu = topk_lp_t.tolist()
        b_per_depth = self.caps_from_topk(topk_logprobs_cpu, tree_width)

        return build_with_per_depth_cap(
            root_token=int(root_token),
            topk_tokens_cpu=topk_tokens_cpu,
            topk_logprobs_cpu=topk_logprobs_cpu,
            b_per_depth=b_per_depth,
            budget=int(budget),
            device=device,
        )

    def caps_from_topk(self, topk_logprobs_cpu, tree_width, **kwargs) -> list[int]:
        """Per-depth fanout cap from the top-2 gap g_d = log q^(1) - log q^(2) via
        the sigmoid schedule (the engine build_from_topk path; build() routes here
        too). K<2 -> chain ([1]*D), matching the tree_width<2 branch above."""
        K = len(topk_logprobs_cpu[0]) if topk_logprobs_cpu else 0
        if K < 2:
            return [1] * len(topk_logprobs_cpu)
        gaps = [lp[0] - lp[1] for lp in topk_logprobs_cpu]
        return [_sigmoid_cap(g_d, K, self.beta, self.g_0) for g_d in gaps]


def _sigmoid_cap(g_d: float, K: int, beta: float, g_0: float) -> int:
    """b_d = max(1, round(K · sigmoid(-β · (g_d - g_0)))).

    Large gap → sigmoid argument → -∞ → σ → 0 → b_d → 1 (chain).
    Small gap → sigmoid argument → +∞ → σ → 1 → b_d → K (fanout).
    """
    arg = -beta * (g_d - g_0)
    # numerically-stable sigmoid
    if arg >= 0:
        s = 1.0 / (1.0 + math.exp(-arg))
    else:
        e = math.exp(arg)
        s = e / (1.0 + e)
    return max(1, int(round(K * s)))
