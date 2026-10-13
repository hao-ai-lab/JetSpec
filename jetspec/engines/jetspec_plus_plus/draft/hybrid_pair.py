"""Hybrid-only N=1 pair payload; the dense outcome wire stays unchanged."""

import os
import hashlib
from dataclasses import dataclass

import torch

from jetspec.engines.jetspec_plus_plus.wire.chain.contract import OUTCOME_WORDS; from jetspec.engines.jetspec_plus_plus.wire.chain.contract import outcome_message_views




def eager_hybrid_pair(runner):
    """The hybrid pair protocol may execute eagerly without enabling graph knobs."""
    return (getattr(runner, 'enforce_eager', False)
            and getattr(runner, 'hybrid_target', False)
            and os.environ.get('JETSPEC_HYBRID_PAIR', '0') == '1'
            and os.environ.get('JETSPEC_CHAIN_GRAPH', '0') == '0')






@dataclass(frozen=True)
class HybridPairLayout:
    lanes: int
    hidden: int
    key_heads: int
    value_heads: int
    key_dim: int
    value_dim: int
    conv_kernel: int
    kv_heads: int
    attention_dim: int
    dtype: torch.dtype











class HybridOutcomeWire:
    def __init__(self, layout, device):
        self.tensor = torch.empty(layout.outcome_bytes, dtype=torch.uint8, device=device)
        (self.packet, self.taps, self.recurrent, self.conv,
         self.keys, self.values) = hybrid_outcome_views(self.tensor, layout)

    def pack(self, packet, taps, recurrent, conv, keys, values):
        self.packet.copy_(packet)
        self.taps.zero_()
        self.taps[:taps.shape[0]].copy_(taps)
        self.recurrent.copy_(recurrent)
        self.conv.copy_(conv)
        self.keys.copy_(keys)
        self.values.copy_(values)
        return self.tensor




class HybridOutcomeWireRing:
    """Two fixed hybrid envelopes, released after their asynchronous sends."""

    def __init__(self, layout, device):
        self.wires = [HybridOutcomeWire(layout, device) for _ in range(2)]
        self.works = [[], []]
        self.next_slot = 0
        self.pending_slot = None












