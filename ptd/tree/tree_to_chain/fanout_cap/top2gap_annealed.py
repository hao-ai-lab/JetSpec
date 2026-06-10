"""top2gap_annealed — top-2 gap fanout with a depth-calibrated schedule.

The live gate is the shipped top2gap_fanout rule. This variant only anneals the
result by depth:

- early depths chain because fingerprint acceptance is near rank-1 only;
- middle depths use plain V5/top2gap;
- tail depths cap fanout at a small rank budget.

Setting chain_until=0 and tail_from past the available depths recovers plain
top2gap_fanout exactly.
"""
from __future__ import annotations

from ptd.tree._core.registry import register_tree_algo
from ptd.tree.tree_to_chain.fanout_cap.top2gap import Top2GapFanout


@register_tree_algo("top2gap_annealed")
class Top2GapAnnealed(Top2GapFanout):
    """Top-2 gap fanout with depth-local chain and tail clamps."""

    def __init__(
        self,
        beta: float = 2.0,
        g_0: float = 1.0,
        chain_until: int = 3,
        tail_from: int = 9,
        tail_cap: int = 2,
        g0: float | None = None,
    ):
        if g0 is not None:
            g_0 = g0
        super().__init__(beta=beta, g_0=g_0)
        self.chain_until = int(chain_until)
        self.tail_from = int(tail_from)
        self.tail_cap = int(tail_cap)

    def caps_from_topk(self, topk_logprobs_cpu, tree_width, **kwargs) -> list[int]:
        base_caps = super().caps_from_topk(topk_logprobs_cpu, tree_width, **kwargs)
        K = len(topk_logprobs_cpu[0]) if topk_logprobs_cpu else max(int(tree_width), 1)
        tail_cap = max(1, min(K, self.tail_cap))

        caps: list[int] = []
        for depth, cap in enumerate(base_caps):
            if depth < self.chain_until:
                scheduled = 1
            elif depth >= self.tail_from:
                scheduled = min(cap, tail_cap)
            else:
                scheduled = cap
            caps.append(max(1, min(K, int(scheduled))))
        return caps
