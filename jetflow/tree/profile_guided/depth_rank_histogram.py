"""depth_rank_histogram — per-(depth, rank) acceptance histogram -> fanout cap.

Lineage: design id B2. Like top2gap_fanout it sets a per-depth fanout cap and
feeds the shared heap builder, but it reads the cap from an OFFLINE-collected
acceptance histogram resolved per (depth, rank) rather than from the live top-2
gap. It can express "at depth 3 keep rank-1 and rank-2 but drop rank-3+", which
a single live gap value cannot.

This is the budget-aware generalization the low-budget winners lack: top2gap and
the semantic_aware templates saturate well below a large budget (they leave it on
the table), whereas this keeps every rank whose PROFILED acceptance clears a
threshold, so it spends extra budget exactly where acceptance data says it pays —
expanding with the budget instead of capping at a fixed template size.

profile_table schema (produced by bench/profiling/collect_depth_rank_stats.py):

    {"depth_rank_accept": [[a_00, a_01, ...],   # depth 0: P(accepted child = rank r)
                           [a_10, a_11, ...],   # depth 1
                           ...],                # one row per drafter depth
     "meta": {...}}                             # model / dataset / n (informational)

Cap rule: b_per_depth[d] = #{r : depth_rank_accept[d][r] >= tau}, clamped to
[1, tree_width]. Ranks almost never accepted are dropped (budget saved); the
heap + budget bound then truncate to the most-probable productive paths.

Identity recovery: with no profile_table (None), or tau <= 0, b_per_depth = K at
every depth -> build_with_per_depth_cap returns byte-identical accum_logp.
Lossless for any tree regardless (the verifier only commits its own greedy)."""
from __future__ import annotations

import torch

from jetflow.tree._core.base import DraftTree, TreeAlgorithm
from jetflow.tree._core.fanout_cap_builder import build_with_per_depth_cap
from jetflow.tree._core.registry import register_tree_algo


@register_tree_algo("depth_rank_histogram")
class DepthRankHistogram(TreeAlgorithm):
    """Per-depth fanout cap from an offline per-(depth, rank) acceptance table."""

    def __init__(self, tau: float = 0.02):
        self.tau = float(tau)

    def build(
        self,
        root_token: int,
        draft_logits: torch.Tensor,  # (1, D, V)
        block_size: int,
        tree_width: int,
        budget: int,
        device: torch.device,
        profile_table: dict | None = None,
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

        K = max(tree_width, 1)
        log_probs = torch.log_softmax(draft_logits.squeeze(0), dim=-1)  # (D, V)
        topk_lp_t, topk_tok_t = torch.topk(log_probs, K, dim=-1)
        topk_tokens_cpu = topk_tok_t.tolist()
        topk_logprobs_cpu = topk_lp_t.tolist()

        b_per_depth = self.caps_from_topk(topk_logprobs_cpu, K, profile_table=profile_table)

        return build_with_per_depth_cap(
            root_token=int(root_token),
            topk_tokens_cpu=topk_tokens_cpu,
            topk_logprobs_cpu=topk_logprobs_cpu,
            b_per_depth=b_per_depth,
            budget=int(budget),
            device=device,
        )

    def caps_from_topk(self, topk_logprobs_cpu, tree_width, profile_table=None, **kwargs) -> list[int]:
        """Per-depth cap from the offline profile (engine build_from_topk path;
        build() routes here too). Depends on the profile + #depths, not the logprob
        values — so the dense and topk paths give identical caps."""
        K = len(topk_logprobs_cpu[0]) if topk_logprobs_cpu else max(tree_width, 1)
        return self._caps_from_profile(profile_table, len(topk_logprobs_cpu), K)

    def _caps_from_profile(self, profile_table, D: int, K: int) -> list[int]:
        """Per-depth cap = #ranks whose profiled acceptance >= tau, clamped [1, K].
        No usable profile (or tau <= 0) -> K everywhere (recovers accum_logp)."""
        rows = None
        if profile_table is not None and self.tau > 0.0:
            rows = profile_table.get("depth_rank_accept")
        if not rows:
            return [K] * D
        caps = []
        for d in range(D):
            # missing deep rows -> full fanout (no data to prune on)
            row = rows[d] if d < len(rows) else None
            if not row:
                caps.append(K)
                continue
            keep = sum(1 for r in range(min(K, len(row))) if row[r] >= self.tau)
            caps.append(min(K, max(1, keep)))
        return caps
