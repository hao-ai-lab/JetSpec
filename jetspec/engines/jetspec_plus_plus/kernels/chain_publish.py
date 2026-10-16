"""Device kernels for draft/publish.py."""
from __future__ import annotations

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None

from jetspec.engines.jetspec_plus_plus.wire.contract import OutcomeOffset

from jetspec.engines.jetspec_plus_plus.wire.contract import CANDIDATE_TRAILER_WORDS

from jetspec.engines.jetspec_plus_plus.wire.triton_offsets import (  # noqa: F401  (read inside the kernels)
    A, AVAILABLE, BLOCKS, CANDIDATE_ROUND, CANDIDATE_SEQ_ID, CMD, EOS_FLAG, HASH16, K, L, RECOVERY,
    RESERVED, ROUND, SEQ_ID, TAP_COUNT, TRUE_K)


if triton is not None:
    @triton.jit
    def _publish_fused(Packet, Final, Keys, Top1, Passes, Published,
                       Wire, Scores, WirePasses, RingRows, RingKeys,
                       RingTop1, RingPasses, RingRounds,
                       ROWS: tl.constexpr, CAP: tl.constexpr,
                       RING: tl.constexpr, LANES: tl.constexpr,
                       WORDS: tl.constexpr, BLOCK_OFF: tl.constexpr,
                       KEY_OFF: tl.constexpr, BLOCK: tl.constexpr,
                       TRAILER_WORDS: tl.constexpr):
        j = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        eos = tl.load(Packet + EOS_FLAG) != 0
        visible = ~eos
        round_id = tl.load(Packet + ROUND) + 1
        slot = round_id % 3
        row = j // LANES
        token = tl.load(Final + j, j < ROWS * LANES, other=-1)
        score = tl.load(Top1 + j, j < ROWS * LANES, other=0.)
        masked_token = tl.where(visible, token, -1)
        masked_score = tl.where(visible, score, 0.)
        tl.store(Final + j, masked_token, j < ROWS * LANES)
        tl.store(Top1 + j, masked_score, j < ROWS * LANES)
        tl.store(Scores + j, tl.where(row < ROWS, masked_score, 0.), j < CAP * LANES)
        tl.store(RingRows + slot * RING * LANES + j,
                 tl.where(visible, token, 0), j < RING * LANES)
        tl.store(RingTop1 + slot * RING * LANES + j,
                 masked_score, j < RING * LANES)

        key = tl.load(Keys + j, j < ROWS * 2, other=-1)
        masked_key = tl.where(visible, key, -1)
        tl.store(Keys + j, masked_key, j < ROWS * 2)
        tl.store(RingKeys + slot * RING * 2 + j, masked_key, j < RING * 2)
        passed = tl.load(Passes + j, j < ROWS, other=0)
        tl.store(WirePasses + j, tl.where(j < ROWS, passed, 0), j < RING)
        tl.store(RingPasses + slot * RING + j,
                 tl.where(visible, passed, 0), j < RING)

        wire_token = tl.load(Final + j - BLOCK_OFF,
                             (j >= BLOCK_OFF) & (j < BLOCK_OFF + ROWS * LANES),
                             other=-1)
        wire_key = tl.load(Keys + j - KEY_OFF,
                           (j >= KEY_OFF) & (j < KEY_OFF + ROWS * 2), other=-1)
        word = tl.full((BLOCK,), -1, tl.int64)
        word = tl.where((j >= BLOCK_OFF) & (j < BLOCK_OFF + CAP * LANES),
                        tl.where(visible, wire_token, -1), word)
        word = tl.where((j >= KEY_OFF) & (j < KEY_OFF + CAP * 2),
                        tl.where(visible, wire_key, -1), word)
        word = tl.where((j >= BLOCKS + CAP * LANES) & (j < BLOCKS + TRAILER_WORDS + CAP * LANES), 0, word)
        word = tl.where(j == AVAILABLE, visible.to(tl.int64), word)
        word = tl.where(j == CANDIDATE_SEQ_ID, tl.load(Packet + SEQ_ID), word)
        word = tl.where(j == CANDIDATE_ROUND, round_id, word)
        tl.store(Wire + j, word, j < WORDS)
        if tl.program_id(0) == 0:
            tl.store(Published, visible)
            tl.store(RingRounds + slot, round_id)

