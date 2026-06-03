"""depth_rank_histogram — per-(depth, rank) acceptance histogram → fanout cap.

STATUS: PLANNED (placeholder, not registered). Lineage: design id B2.

Mechanism: like top2gap_fanout it sets a per-depth fanout cap, but it reads the
cap from an offline-collected acceptance histogram resolved per (depth, rank)
rather than from the live top-2 gap. It can express "at depth 3 keep rank-1 and
rank-2 but drop rank-3+", which a single live gap value can't.

Potential: the single most promising non-shipped variant — the only one whose
finer-than-per-depth granularity could legitimately beat top2gap_fanout, IF
offline acceptance carries rank structure the live gap misses (open question).
Same win regime as top2gap (middle budget, decisive drafter).

Needs: an offline profiler that runs crossproduct over N calibration prompts and
dumps per-(depth, rank) acceptance counts (a `profile_table`). The profiler is
HF-collectable — this does NOT strictly require vLLM. It also lights up
`class_histogram`'s real (vs template) histogram, so the profiler amortizes.
"""
from __future__ import annotations

import torch

from ptd.tree._core.base import DraftTree, TreeAlgorithm


class DepthRankHistogram(TreeAlgorithm):
    """PLANNED — per-(depth, rank) histogram-driven fanout cap. Not implemented."""

    def build(self, root_token: int, draft_logits: torch.Tensor, block_size: int,
              tree_width: int, budget: int, device: torch.device, **kwargs) -> DraftTree:
        raise NotImplementedError(
            "depth_rank_histogram (B2) is planned. Needs an offline profile_table of "
            "per-(depth, rank) acceptance counts; see ptd/tree/ROADMAP.md."
        )
