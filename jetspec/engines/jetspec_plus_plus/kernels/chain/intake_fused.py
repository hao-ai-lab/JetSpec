"""Fused metadata and exact-tap intake for a captured chain round."""
from __future__ import annotations

import os
from types import SimpleNamespace

try:
    from jetspec.engines.jetspec_plus_plus._optional import triton
    from jetspec.engines.jetspec_plus_plus._optional import tl
except ImportError:  # The pinned Mac CPU test environment has no Triton.
    triton = SimpleNamespace(jit=lambda fn: fn)
    tl = SimpleNamespace(constexpr=object())

# Literal field offsets mirror contract.OutcomeOffset and contract.Offset.
# Triton cannot close over those Python enums inside a JIT kernel.
# Error literals mirror CHAIN_ERR_* in intake.py.
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
              TAU_DEPTH: tl.constexpr):
    cmd = tl.load(Packet + 0)
    round_id = tl.load(Packet + 4)
    count = tl.load(Packet + 2)
    recovery = tl.load(Packet + 3)
    tap_count = tl.load(Packet + 5)
    incoming_length = tl.load(Packet + 8)
    bad_packet = (cmd != 1) | (round_id != tl.load(Expected))
    length = tl.where(bad_packet, tl.load(SafeLength), incoming_length)
    tl.store(Packet + 8, length)
    control_bad = tl.full((), False, tl.int1)
    if HAS_CONTROL:
        control_bad = (round_id != tl.load(ControlRound)) | (incoming_length != tl.load(ControlL))

    tl.store(Rp + 4, count)
    tl.store(Rp + 5, recovery)
    tl.store(Rp + 2, tap_count)
    tl.store(Rp + 0, length)
    tl.store(Rp + 1, length - tap_count)
    tl.store(Rp + 3, length)
    if HAS_RNG:
        rng_row = tl.minimum(tl.maximum(length + 1, 0), CAPACITY)
        idx = tl.arange(0, LANES)
        values = tl.load(RngRows + rng_row * RNG_WIDTH + idx, idx < RNG_WIDTH, other=0)
        tl.store(Rp + 32 + idx, values, idx < RNG_WIDTH)

    prev_slot, prev_idx, found, _ = _select(
        RingKeys, RingRounds, round_id - 1,
        tl.load(Packet + 10), tl.load(Packet + 11),
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
    tl.store(Rp + 6, tl.where(exact, 0, 1))
    tl.store(Rp + 7,
             tl.where(valid, tl.minimum(tl.maximum(passes - 1, 0), TAU_DEPTH), 0))

    start = length - count
    capacity_ok = (length <= LIMIT) & (start >= 0) & (count >= 0) & (count < LANES)
    do_write = (round_id != 0) & capacity_ok & ((count == 0) | found)
    pos = start + lane
    replacement = tl.where(lane == count, recovery,
                           tl.gather(prev_row, tl.minimum(lane + 1, LANES - 1), 0))
    hash_length = tl.minimum(tl.maximum(length + 1, 0), CAPACITY)
    hpos = hash_length - 16 + lane
    selected = tl.load(Committed + tl.minimum(tl.maximum(hpos, 0), CAPACITY - 1))
    offset = hpos - start
    accepted = tl.gather(prev_row, tl.minimum(tl.maximum(offset + 1, 0), LANES - 1), 0)
    projected = tl.where(offset == count, recovery, accepted)
    selected = tl.where(do_write & (offset >= 0) & (offset <= count), projected, selected)
    selected = tl.where((hpos >= 0) & (hpos < hash_length), selected, 0)
    weights = tl.load(HashWeights + lane, lane < 16, other=0)
    digest = tl.sum((selected + 1) * weights, 0) + hash_length * 1000003
    correction = tl.load(Committed + tl.minimum(tl.maximum(length, 0), CAPACITY - 1))
    correction = tl.where(do_write, recovery, correction)
    tl.store(Committed + pos, replacement,
             do_write & (lane <= count) & (pos >= 0) & (pos < CAPACITY))
    anchor = tl.sum(tl.where(lane == 0, next_row, 0), 0)
    faults = tl.where(bad_packet, 524288 | 262144, 0)
    faults |= tl.where(control_bad, 2097152 | 262144, 0)
    faults |= tl.where(~capacity_ok, 65536, 0)
    faults |= tl.where((count > 0) & ~found, 131072, 0)
    faults |= tl.where(digest != tl.load(Packet + 9), 262144, 0)
    faults |= tl.where(exact & (anchor != correction),
                       1048576 | 262144, 0)
    if HAS_AFTER_MISS:
        wrong_miss = (round_id != 0) & (tl.load(Packet + 6) == 0) & (
            tl.where(exact, 0, 1) != tl.load(ControlAfterMiss))
        faults |= tl.where(wrong_miss, 4194304 | 262144, 0)
    tl.store(Err, faults)


@triton.jit
def _tap_front(Packet, Taps, Backing, FrontTaps, Prefix, Staged,
               AdmissionHidden, AdmissionIndex,
               CAPACITY: tl.constexpr, HIDDEN: tl.constexpr, LANES: tl.constexpr,
               HAS_FRONT: tl.constexpr, HAS_ADMISSION: tl.constexpr,
               BLOCK: tl.constexpr):
    lane = tl.arange(0, LANES)
    col = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    length = tl.load(Packet + 8)
    count = tl.load(Packet + 5)
    round_id = tl.load(Packet + 4)
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


def fused_intake(buffers: dict, *, max_length: int | None) -> None:
    """Run the two kernels using only static shapes and device pointers."""
    packet = buffers["packet"]
    if not packet.is_cuda:
        raise ValueError("fused intake requires CUDA tensors")
    tokens, ring = buffers["tokens"], buffers["row_ring"]
    lanes = ring.rows.shape[-1]
    from jetspec.engines.jetspec_plus_plus.draft.eval_profile import enabled as eval_enabled
    if lanes not in ((16, 32) if eval_enabled() else (16,)):
        raise ValueError("fused intake supports b16 (b32 in eval)")
    rows = ring.rows.shape[1]
    capacity = tokens.committed_ids.numel()
    limit = capacity - 1 if max_length is None else min(capacity - 1, max_length)
    control = "control_round" in buffers
    rng = "rng_table" in buffers
    prior = "prior_p_top1" in buffers
    taps = "tap_store" in buffers and "taps" in buffers
    front = "front" in buffers and "taps" in buffers
    admission = "admission_hidden" in buffers
    after_miss = ("control_after_miss" in buffers and
                  os.environ.get("JETSPEC_CHAIN_MR_GLUE_LEAN", "0") == "1")
    fallback = packet
    _metadata[(1,)](
        packet, buffers["rp"], buffers["expected_round"], buffers["safe_length"],
        buffers.get("control_round", fallback), buffers.get("control_L", fallback),
        tokens.committed_ids, tokens.pending_draft_ids,
        buffers.get("prior_p_top1", fallback), ring.rows, ring.p_top1,
        ring.passes_run, ring.keys, ring.rounds, buffers["hash_weights"],
        buffers["rng_table"].rows if rng else fallback, buffers["intake_err"],
        buffers["intake_selected"],
        buffers["control_after_miss"] if after_miss else fallback,
        capacity, limit, lanes, rows, lanes - 1,
        control, rng, prior, after_miss, 3 if eval_enabled() else 2, num_warps=4)
    if taps:
        store = buffers["tap_store"]
        bound = buffers.get("front")
        hidden = store.storage.shape[-1]
        _tap_front[(triton.cdiv(5 * hidden, 256),)](
            packet, buffers["taps"], store.backing,
            bound.taps if front else fallback,
            bound.decoder_prefix_taps if front else fallback,
            bound.staged_hidden if front else fallback,
            buffers["admission_hidden"] if admission else fallback,
            buffers["intake_selected"], store.storage.shape[0],
            hidden, lanes, front, admission, 256, num_warps=4)
