"""Device kernels for draft/intake.py."""
from __future__ import annotations

from jetspec.engines.jetspec_plus_plus.wire.contract import HASH_WINDOW, HASH_LENGTH_FACTOR

from jetspec.engines.jetspec_plus_plus.wire.triton_offsets import (  # noqa: F401  (read inside the kernels)
    A, AFTER_MISS, C1_COUNT, C1_START, CMD, CORRECTION, DEC_LEN, EOS_FLAG, HASH16, J_ROUND, K, L,
    RECOVERY, RESERVED, RNG_IDX, ROUND, RP_L, RP_TRUE_K, SEQ_ID, SLOT, TAP_COUNT, TAU_IDX, TRUE_K)

from types import SimpleNamespace

try:
    import triton
    import triton.language as tl
except ImportError:  # The pinned Mac CPU test environment has no Triton.
    triton = SimpleNamespace(jit=lambda fn: fn)
    tl = SimpleNamespace(constexpr=object())


@triton.jit
def _select(Keys, Rounds, round_id, wanted_k, wanted_a, fallback_k,
            IS_PENDING: tl.constexpr, ROWS: tl.constexpr):
    r = tl.arange(0, ROWS)
    slot = ((round_id % 3) + 3) % 3
    key_k = tl.load(Keys + slot * ROWS * 2 + r * 2)
    key_a = tl.load(Keys + slot * ROWS * 2 + r * 2 + 1)
    valid = (tl.load(Rounds + slot) == round_id) & (key_k >= 0) & (key_a >= 0)
    exact = valid & (key_k == wanted_k) & (key_a == wanted_a)
    if not IS_PENDING:
        exact = exact & (wanted_k >= 0)
    same_k = valid & (key_k == fallback_k)
    exact_idx = tl.min(tl.where(exact, r, ROWS), 0)
    same_idx = tl.min(tl.where(same_k, r, ROWS), 0)
    valid_idx = tl.min(tl.where(valid, r, ROWS), 0)
    chosen = tl.where(exact_idx < ROWS, exact_idx,
                      tl.where(same_idx < ROWS, same_idx,
                               tl.where(valid_idx < ROWS, valid_idx, 0)))
    return slot, chosen, exact_idx < ROWS, valid_idx < ROWS


@triton.jit
def _metadata(Packet, Rp, Expected, SafeLength, ControlRound, ControlL,
              Committed, Pending, PriorTop1, RingRows, RingTop1, RingPasses,
              RingKeys, RingRounds, HashWeights, RngRows, Err, AdmissionIndex,
              ControlAfterMiss,
              CAPACITY: tl.constexpr, LIMIT: tl.constexpr, LANES: tl.constexpr,
              ROWS: tl.constexpr, RNG_WIDTH: tl.constexpr,
              HAS_CONTROL: tl.constexpr, HAS_RNG: tl.constexpr,
              HAS_PRIOR: tl.constexpr, HAS_AFTER_MISS: tl.constexpr,
              TAU_DEPTH: tl.constexpr, HASH_WINDOW: tl.constexpr,
              HASH_LENGTH_FACTOR: tl.constexpr):
    cmd = tl.load(Packet + CMD)
    round_id = tl.load(Packet + ROUND)
    count = tl.load(Packet + TRUE_K)
    recovery = tl.load(Packet + RECOVERY)
    tap_count = tl.load(Packet + TAP_COUNT)
    incoming_length = tl.load(Packet + L)
    bad_packet = (cmd != 1) | (round_id != tl.load(Expected))
    length = tl.where(bad_packet, tl.load(SafeLength), incoming_length)
    tl.store(Packet + L, length)
    control_bad = tl.full((), False, tl.int1)
    if HAS_CONTROL:
        control_bad = (round_id != tl.load(ControlRound)) | (incoming_length != tl.load(ControlL))

    tl.store(Rp + RP_TRUE_K, count)
    tl.store(Rp + CORRECTION, recovery)
    tl.store(Rp + C1_COUNT, tap_count)
    tl.store(Rp + RP_L, length)
    tl.store(Rp + C1_START, length - tap_count)
    tl.store(Rp + DEC_LEN, length)
    if HAS_RNG:
        rng_row = tl.minimum(tl.maximum(length + 1, 0), CAPACITY)
        idx = tl.arange(0, LANES)
        values = tl.load(RngRows + rng_row * RNG_WIDTH + idx, idx < RNG_WIDTH, other=0)
        tl.store(Rp + RNG_IDX + idx, values, idx < RNG_WIDTH)

    prev_slot, prev_idx, found, _ = _select(
        RingKeys, RingRounds, round_id - 1,
        tl.load(Packet + K), tl.load(Packet + A),
        count, False, ROWS)
    next_slot, next_idx, exact, valid = _select(
        RingKeys, RingRounds, round_id, count, recovery, count, True, ROWS)
    tl.store(AdmissionIndex, next_idx)
    lane = tl.arange(0, LANES)
    prev_row = tl.load(RingRows + (prev_slot * ROWS + prev_idx) * LANES + lane)
    next_row = tl.load(RingRows + (next_slot * ROWS + next_idx) * LANES + lane)
    next_top1 = tl.load(RingTop1 + (next_slot * ROWS + next_idx) * LANES + lane)
    passes = tl.load(RingPasses + next_slot * ROWS + next_idx)
    tl.store(Pending + lane - 1, tl.where(valid, next_row, 0), lane > 0)
    if HAS_PRIOR:
        tl.store(PriorTop1 + lane, tl.where(valid, next_top1, 0))
    tl.store(Rp + AFTER_MISS, tl.where(exact, 0, 1))
    tl.store(Rp + TAU_IDX,
             tl.where(valid, tl.minimum(tl.maximum(passes - 1, 0), TAU_DEPTH), 0))

    start = length - count
    capacity_ok = (length <= LIMIT) & (start >= 0) & (count >= 0) & (count < LANES)
    do_write = (round_id != 0) & capacity_ok & ((count == 0) | found)
    pos = start + lane
    replacement = tl.where(lane == count, recovery,
                           tl.gather(prev_row, tl.minimum(lane + 1, LANES - 1), 0))
    hash_length = tl.minimum(tl.maximum(length + 1, 0), CAPACITY)
    hpos = hash_length - HASH_WINDOW + lane
    selected = tl.load(Committed + tl.minimum(tl.maximum(hpos, 0), CAPACITY - 1))
    offset = hpos - start
    accepted = tl.gather(prev_row, tl.minimum(tl.maximum(offset + 1, 0), LANES - 1), 0)
    projected = tl.where(offset == count, recovery, accepted)
    selected = tl.where(do_write & (offset >= 0) & (offset <= count), projected, selected)
    selected = tl.where((hpos >= 0) & (hpos < hash_length), selected, 0)
    weights = tl.load(HashWeights + lane, lane < HASH_WINDOW, other=0)
    digest = tl.sum((selected + 1) * weights, 0) + hash_length * HASH_LENGTH_FACTOR
    correction = tl.load(Committed + tl.minimum(tl.maximum(length, 0), CAPACITY - 1))
    correction = tl.where(do_write, recovery, correction)
    tl.store(Committed + pos, replacement,
             do_write & (lane <= count) & (pos >= 0) & (pos < CAPACITY))
    anchor = tl.sum(tl.where(lane == 0, next_row, 0), 0)
    faults = tl.where(bad_packet, 524288 | 262144, 0)
    faults |= tl.where(control_bad, 2097152 | 262144, 0)
    faults |= tl.where(~capacity_ok, 65536, 0)
    faults |= tl.where((count > 0) & ~found, 131072, 0)
    faults |= tl.where(digest != tl.load(Packet + HASH16), 262144, 0)
    faults |= tl.where(exact & (anchor != correction),
                       1048576 | 262144, 0)
    if HAS_AFTER_MISS:
        wrong_miss = (round_id != 0) & (tl.load(Packet + EOS_FLAG) == 0) & (
            tl.where(exact, 0, 1) != tl.load(ControlAfterMiss))
        faults |= tl.where(wrong_miss, 4194304 | 262144, 0)
    tl.store(Err, faults)


@triton.jit
def _tap_front(Packet, Taps, Backing, FrontTaps, Prefix, Staged,
               AdmissionHidden, AdmissionIndex,
               CAPACITY: tl.constexpr, HIDDEN: tl.constexpr, LANES: tl.constexpr,
               HAS_FRONT: tl.constexpr, HAS_ADMISSION: tl.constexpr,
               BLOCK: tl.constexpr, TrueTaps=None, HAS_TRUE_TAPS: tl.constexpr = False):
    lane = tl.arange(0, LANES)
    col = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    length = tl.load(Packet + L)
    count = tl.load(Packet + TAP_COUNT)
    round_id = tl.load(Packet + ROUND)
    start = length - count
    append_ok = (length <= CAPACITY) & (start >= 0) & (count >= 0) & (count <= LANES)
    active = append_ok & (lane < count)
    destination = tl.where(active, start + lane, CAPACITY + lane)
    flat = lane[:, None] * (5 * HIDDEN) + col[None, :]
    taps = tl.load(Taps + flat, col[None, :] < 5 * HIDDEN, other=0)
    tl.store(Backing + destination[:, None] * (5 * HIDDEN) + col[None, :],
             taps, col[None, :] < 5 * HIDDEN)
    if HAS_FRONT:
        tl.store(FrontTaps + flat, taps, col[None, :] < 5 * HIDDEN)
        if HAS_ADMISSION:
            chosen = tl.load(AdmissionIndex)
            previous = tl.load(AdmissionHidden +
                               (chosen * LANES + lane[:, None]) * (5 * HIDDEN) +
                               col[None, :], col[None, :] < 5 * HIDDEN, other=0)
            tl.store(Staged + flat, previous,
                     (round_id == 0) & (col[None, :] < 5 * HIDDEN))
        if tl.program_id(0) * BLOCK < HIDDEN:
            h = col
            prefix_pos = length - count + lane
            clipped = tl.minimum(tl.maximum(prefix_pos, 0), CAPACITY - 1)
            old_prefix = tl.load(
                Backing + clipped[:, None] * (5 * HIDDEN) + 4 * HIDDEN + h[None, :],
                h[None, :] < HIDDEN, other=0)
            tap_lane = prefix_pos - start
            new_prefix = tl.load(
                Taps + tap_lane[:, None] * (5 * HIDDEN) + 4 * HIDDEN + h[None, :],
                (tap_lane[:, None] >= 0) & (tap_lane[:, None] < LANES) &
                (h[None, :] < HIDDEN), other=0)
            from_new = append_ok & (prefix_pos >= start) & (prefix_pos < start + count)
            value = tl.where(from_new[:, None], new_prefix, old_prefix)
            valid = (lane < count) & (prefix_pos >= 0) & (prefix_pos < CAPACITY)
            tl.store(Prefix + lane[:, None] * HIDDEN + h[None, :],
                     tl.where(valid[:, None], value, 0), h[None, :] < HIDDEN)
    if HAS_TRUE_TAPS:
        if tl.program_id(0) * BLOCK < HIDDEN:
            recent = tl.arange(0, 4)
            position = tl.maximum(length - 4 + recent, 0)
            offset = position - start
            from_new = append_ok & (offset >= 0) & (offset < count)
            old = tl.load(
                Backing + position[:, None] * (5 * HIDDEN) + 4 * HIDDEN + col[None, :],
                ~from_new[:, None] & (col[None, :] < HIDDEN), other=0)
            incoming = tl.load(
                Taps + offset[:, None] * (5 * HIDDEN) + 4 * HIDDEN + col[None, :],
                from_new[:, None] & (col[None, :] < HIDDEN), other=0)
            tl.store(TrueTaps + recent[:, None] * HIDDEN + col[None, :],
                     tl.where(from_new[:, None], incoming, old), col[None, :] < HIDDEN)
