"""Device kernels for draft/glue.py."""
from __future__ import annotations

from jetspec.engines.jetspec_plus_plus.wire.contract import Offset

from jetspec.engines.jetspec_plus_plus.wire.triton_offsets import (  # noqa: F401  (read inside the kernels)
    AFTER_MISS, C1_COUNT, C1_START, CORRECTION, DEC_LEN, J_ROUND, RNG_IDX, RP_L, RP_TRUE_K, SLOT,
    TAU_IDX)

try:
    import triton
    import triton.language as tl
except ImportError:  # CPU development hosts do not install Triton.
    triton = None
    tl = None


if triton is not None:
    @triton.jit
    def _window_mask_rope(LENGTH, COS_TABLE, SIN_TABLE, MASK, COS, SIN,
                          ROWS: tl.constexpr, BUCKET: tl.constexpr,
                          DIM: tl.constexpr, NEG: tl.constexpr,
                          MASK_BLOCKS: tl.constexpr, BLOCK: tl.constexpr):
        program = tl.program_id(0)
        offset = tl.arange(0, BLOCK)
        length = tl.load(LENGTH)
        if program < MASK_BLOCKS:
            flat = program * BLOCK + offset
            query = flat // BUCKET
            key = flat % BUCKET
            allowed = (key < length) | ((key >= length) & (key - length <= query))
            tl.store(MASK + flat, tl.where(allowed, 0., NEG), flat < ROWS * BUCKET)
        else:
            flat = (program - MASK_BLOCKS) * BLOCK + offset
            row = flat // DIM
            column = flat % DIM
            source = (length + row) * DIM + column
            cos = tl.load(COS_TABLE + source, flat < ROWS * DIM, other=0)
            sin = tl.load(SIN_TABLE + source, flat < ROWS * DIM, other=0)
            tl.store(COS + flat, cos, flat < ROWS * DIM)
            tl.store(SIN + flat, sin, flat < ROWS * DIM)

    @triton.jit
    def _mask_token_guard(IDS, ACTIVE, ERR, ROWS: tl.constexpr,
                          LANES: tl.constexpr, TOKEN: tl.constexpr,
                          BLOCK: tl.constexpr):
        flat = tl.arange(0, BLOCK)
        row = flat // LANES
        ids = tl.load(IDS + flat, flat < ROWS * LANES, other=0)
        active = tl.load(ACTIVE + row, row < ROWS, other=0)
        found = tl.sum((ids == TOKEN) & active & (flat < ROWS * LANES), 0) > 0
        old = tl.load(ERR)
        tl.store(ERR, old | tl.where(found, 1 << 7, 0).to(tl.int64))

    @triton.jit
    def _j0_prep(COMMITTED, INDICES, ANCHORS, ACTIVE, IDS, FINISHED, PASSES, ERR,
                 ROWS: tl.constexpr, LANES: tl.constexpr,
                 OWNS_ERR: tl.constexpr, BLOCK: tl.constexpr):
        flat = tl.arange(0, BLOCK)
        row = flat // LANES
        lane = flat % LANES
        rng = tl.load(INDICES + lane - 1, (lane > 0) & (lane < LANES), other=0)
        value = tl.load(COMMITTED + rng, (lane > 0) & (flat < ROWS * LANES), other=0)
        anchor = tl.load(ANCHORS + row * 2, (lane == 0) & (row < ROWS), other=0)
        tl.store(IDS + flat, tl.where(lane == 0, tl.maximum(anchor, 0), value),
                 flat < ROWS * LANES)
        tl.store(FINISHED + row, ~tl.load(ACTIVE + row, row < ROWS, other=0),
                 (lane == 0) & (row < ROWS))
        tl.store(PASSES + row, 0, (lane == 0) & (row < ROWS))
        if OWNS_ERR:
            tl.store(ERR, 0)

    @triton.jit
    def _c1_front_prep(RP, COMMITTED, PENDING, IDS, POSITIONS, SLOTS,
                       CAPACITY: tl.constexpr, LANES: tl.constexpr,
                       BLOCK: tl.constexpr):
        lane = tl.arange(0, BLOCK)
        length = tl.load(RP + RP_L)
        start = tl.load(RP + C1_START)
        count = tl.load(RP + C1_COUNT)
        pending = tl.load(PENDING + lane - 1, (lane > 0) & (lane < LANES), other=0)
        current = tl.load(COMMITTED + length)
        tl.store(IDS + lane, tl.where(lane == 0, current, pending), lane < LANES)
        tl.store(POSITIONS + lane, start + lane, lane < LANES)
        tl.store(SLOTS + lane, tl.where(lane < count, start + lane, CAPACITY + lane),
                 lane < LANES)

    @triton.jit
    def _c1_mask(LENGTH, MASK, ROWS: tl.constexpr, BUCKET: tl.constexpr,
                 NEG: tl.constexpr, BLOCK: tl.constexpr):
        flat = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        query = flat // BUCKET
        key = flat % BUCKET
        length = tl.load(LENGTH)
        allowed = (key < length) | ((key >= length) & (key - length < query))
        tl.store(MASK + flat, tl.where(allowed, 0., NEG), flat < ROWS * BUCKET)

    @triton.jit
    def _context_copy_mark(LENGTH, PREFIX, TAIL, EMBEDDING, CONTEXT,
                           P: tl.constexpr, T: tl.constexpr,
                           H: tl.constexpr, BLOCK: tl.constexpr):
        flat = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        position = flat // H
        column = flat % H
        prefix_row = position < P
        valid = flat < (P + T) * H
        length = tl.load(LENGTH)
        prefix = tl.load(PREFIX + flat, valid & prefix_row, other=0).to(tl.float32)
        tail = tl.load(TAIL + flat - P * H, valid & ~prefix_row, other=0).to(tl.float32)
        mark0 = tl.load(EMBEDDING + column, valid & prefix_row, other=0).to(tl.float32)
        mark1 = tl.load(EMBEDDING + H + column, valid & ~prefix_row, other=0).to(tl.float32)
        base = tl.where(prefix_row, tl.where(position < length, prefix, 0.), tail)
        marked = base + tl.where(prefix_row, mark0, mark1)
        tl.store(CONTEXT + flat, marked, valid)

