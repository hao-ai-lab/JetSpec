"""Device-side chain outcome intake and the published-row mirror.

The ring owns copies of the exact rows placed on the wire. Its entries never
alias live refinement outputs, which may change after an early publication.
"""
from __future__ import annotations

from dataclasses import dataclass
import os
import time

import torch

from jetspec.engines.jetspec_plus_plus.wire.contract import OUTCOME_WORDS
from jetspec.engines.jetspec_plus_plus.wire.contract import FrontInputs
from jetspec.engines.jetspec_plus_plus.wire.contract import Offset
from jetspec.engines.jetspec_plus_plus.wire.contract import OutcomeOffset
from jetspec.engines.jetspec_plus_plus.wire.contract import OutcomePacket
from jetspec.engines.jetspec_plus_plus.wire.contract import TokenBuffers
from jetspec.engines.jetspec_plus_plus.wire.contract import outcome_message_views
from jetspec.engines.jetspec_plus_plus.wire.contract import rng_words
from jetspec.engines.jetspec_plus_plus.wire.contract import HASH_WEIGHTS, hash_last16


CHAIN_ERR_CAPACITY = 1 << 16
CHAIN_ERR_ROW_MISSING = 1 << 17
CHAIN_ERR_DRIFT = 1 << 18
CHAIN_ERR_PACKET = 1 << 19
CHAIN_ERR_PENDING = 1 << 20
CHAIN_ERR_CONTROL = 1 << 21
CHAIN_ERR_AFTER_MISS = 1 << 22
CHAIN_COMMAND_PREFIX = "chain_command_"
CHAIN_RECV_READY_PREFIX = "chain_recv_ready_"











from jetspec.engines.jetspec_plus_plus.draft.state import PublishedRowRing
def fill_front_from_outcome(front: FrontInputs, packet: torch.Tensor,
                            taps: torch.Tensor, tap_store) -> None:
    """Stage the newly verified true taps for both prefixes."""
    if taps.shape != front.taps.shape or taps.device != front.taps.device:
        raise ValueError("outcome taps do not match bound front inputs")
    front.taps.copy_(taps)
    front.decoder_prefix_taps.copy_(tap_store.decoder_prefix(packet))
















def append_outcome(tokens: TokenBuffers, ring: PublishedRowRing,
                   packet: torch.Tensor, err: torch.Tensor,
                   *, hash_weights: torch.Tensor | None = None,
                   max_length: int | None = None,
                   lane_positions: torch.Tensor | None = None) -> torch.Tensor:
    """Append the verified old row; pending comes from the current ring slot."""
    committed = tokens.committed_ids
    row, _, found, _ = ring.select_for_outcome(packet)
    admission = packet[OutcomeOffset.ROUND] == 0
    count = packet[OutcomeOffset.TRUE_K]
    length = packet[OutcomeOffset.L]
    start = length - count
    capacity = committed.numel()
    limit = capacity - 1 if max_length is None else min(capacity - 1, max_length)
    valid_capacity = ((length <= limit) & (start >= 0)
                      & (count >= 0) & (count < row.numel()))
    valid_row = (count == 0) | found
    if lane_positions is None:
        positions = torch.arange(capacity, device=committed.device)
        offsets = positions - start
        # Row lane zero is the already committed query anchor; only draft lanes
        # one through true_k are new IDs before the recovery token.
        accepted = row[(offsets + 1).clamp(0, row.numel() - 1)]
        replacement = torch.where(offsets == count, packet[OutcomeOffset.RECOVERY], accepted)
        write = (~admission) & valid_capacity & valid_row & (offsets >= 0) & (offsets <= count)
        committed.copy_(torch.where(write, replacement, committed))
    else:
        lanes = row.numel()
        if (capacity <= lanes or lane_positions.shape != (lanes,)
                or lane_positions.device != committed.device
                or lane_positions.dtype != torch.int64):
            raise ValueError('lane-sized committed append requires capacity above lanes')
        # At most `lanes` consecutive positions are written. Since capacity is
        # larger, slot 0 or the final slot lies outside that interval and can
        # absorb every inactive lane without changing the committed prefix.
        scratch = torch.where(start > 0, start.new_zeros(()), start.new_full((), capacity - 1))
        write = (~admission) & valid_capacity & valid_row & (lane_positions <= count)
        indices = torch.where(write, start + lane_positions, scratch)
        accepted = row[(lane_positions + 1).clamp_max(lanes - 1)]
        replacement = torch.where(lane_positions == count,
                                  packet[OutcomeOffset.RECOVERY], accepted)
        saved = committed.index_select(0, scratch.reshape(1))[0]
        committed.index_copy_(0, indices, torch.where(write, replacement, saved))
    digest = hash_last16(committed, (length + 1).clamp(0, capacity), hash_weights)
    faults = ((~valid_capacity).to(torch.int64) * CHAIN_ERR_CAPACITY
              | ((count > 0) & ~found).to(torch.int64) * CHAIN_ERR_ROW_MISSING
              | (digest != packet[OutcomeOffset.HASH16]).to(torch.int64) * CHAIN_ERR_DRIFT)
    err[0].bitwise_or_(faults)
    return ~found

