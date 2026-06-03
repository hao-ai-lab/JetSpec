"""path_conditional_refresh — depth-by-depth tree build with KV-reuse drafting.

STATUS: PLANNED (placeholder, not registered). Lineage: design id A6.

Mechanism: build the tree depth-by-depth; at each depth transition do ONE
batched, KV-reuse drafter forward that path-conditions the entire frontier at
once (~D forwards total, not N). Deployment-grade, not the research-only
per-node refresh that re-runs the drafter for every node.

Potential: not a standalone winner — raw path-conditioning doesn't beat
crossproduct at our budgets. Its value is as the LOGITS SOURCE for a
composition: feed path-conditional draft logits into top2gap_fanout's per-depth
gate, replacing the marginal/collapsed logits it reads today. The design-space
analysis flagged this as the strongest single combination — it uses PTD's real
causal-head capability instead of a cheap marginal projection.

Needs: a tree-attention mask at DRAFT time + batched-frontier KV reuse — an
engine hook only the vLLM serving phase provides (the HF/SDPA single-stream loop
does not do draft-side tree attention). Genuinely vLLM-required.
"""
from __future__ import annotations

import torch

from ptd.tree._core.base import DraftTree, TreeAlgorithm


class PathConditionalRefresh(TreeAlgorithm):
    """PLANNED — depth-by-depth KV-reuse path-conditional drafting. Not implemented."""

    def build(self, root_token: int, draft_logits: torch.Tensor, block_size: int,
              tree_width: int, budget: int, device: torch.device, **kwargs) -> DraftTree:
        raise NotImplementedError(
            "path_conditional_refresh (A6) is planned. Needs a vLLM draft-side "
            "tree-attention + batched-frontier KV-reuse hook; see ptd/tree/ROADMAP.md."
        )
