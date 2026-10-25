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
    def _g27b_refresh_mask(ROW, SLOT, TAIL, LENGTH, START, MASK,
                          TOKENS: tl.constexpr, WIDTH: tl.constexpr, WINDOW: tl.constexpr,
                          NEG: tl.constexpr, B: tl.constexpr):
        flat = tl.program_id(0) * B + tl.arange(0, B)
        query, col = flat // WIDTH, flat % WIDTH
        length, start = tl.load(LENGTH), tl.load(START)
        tail_col, block_col = col - length, col - start
        source = tl.minimum(tl.maximum(block_col, 0), TOKENS - 1)
        row = tl.load(ROW + query, query < TOKENS, 0)
        slot = tl.load(SLOT + query, query < TOKENS, 0)
        tail = tl.load(TAIL + query, query < TOKENS, 0)
        other_row = tl.load(ROW + source)
        other_slot = tl.load(SLOT + source)
        allowed = (col < length) | ((tail_col >= 0) & (tail_col < WINDOW) & (tail_col < tail))
        allowed |= (block_col >= 0) & (block_col < TOKENS) & (other_row == row) & (other_slot < slot)
        tl.store(MASK + flat, tl.where(allowed, 0., NEG), query < TOKENS)

    @triton.jit
    def _g27_attention_inputs(PK, PV, K, V, LENGTH, TAILS, KO, VO, BIAS,
                              PKT: tl.constexpr, PKH: tl.constexpr, PKD: tl.constexpr,
                              PVT: tl.constexpr, PVH: tl.constexpr, PVD: tl.constexpr,
                              KT: tl.constexpr, KH: tl.constexpr, KD: tl.constexpr,
                              VT: tl.constexpr, VH: tl.constexpr, VD: tl.constexpr,
                              PREFIX: tl.constexpr, TOKENS: tl.constexpr,
                              HEADS: tl.constexpr, DIM: tl.constexpr,
                              LANES: tl.constexpr, SLOTS: tl.constexpr,
                              COPIES: tl.constexpr, NEG: tl.constexpr, BLOCK: tl.constexpr):
        program = tl.program_id(0)
        offset = tl.arange(0, BLOCK)
        width: tl.constexpr = PREFIX + TOKENS
        if program < COPIES:
            flat = program * BLOCK + offset
            token, head, col = flat // (HEADS * DIM), flat // DIM % HEADS, flat % DIM
            prefix = token < PREFIX
            valid = token < width
            pk = tl.load(PK + token * PKT + head * PKH + col * PKD, valid & prefix, 0)
            pv = tl.load(PV + token * PVT + head * PVH + col * PVD, valid & prefix, 0)
            k = tl.load(K + (token - PREFIX) * KT + head * KH + col * KD, valid & ~prefix, 0)
            v = tl.load(V + (token - PREFIX) * VT + head * VH + col * VD, valid & ~prefix, 0)
            tl.store(KO + flat, tl.where(prefix, pk, k), valid)
            tl.store(VO + flat, tl.where(prefix, pv, v), valid)
        else:
            flat = (program - COPIES) * BLOCK + offset
            query, column = flat // width, flat % width
            local = column - PREFIX
            length = tl.load(LENGTH)
            if SLOTS > 0:
                qb, kb = query >= LANES, local >= LANES
                qr = tl.maximum(query - LANES, 0) // SLOTS
                kr = tl.maximum(local - LANES, 0) // SLOTS
                qi = tl.where(qb, (query - LANES) % SLOTS, query)
                ki = tl.where(kb, (local - LANES) % SLOTS, local)
                tail = tl.load(TAILS + qr, (query < TOKENS) & qb, 0)
                visible = (~kb & (ki < tl.where(qb, tail, qi + 1))) | (qb & kb & (qr == kr) & (ki <= qi))
            else:
                visible = local <= query
            allowed = tl.where(column < PREFIX, column < length, visible)
            tl.store(BIAS + flat, tl.where(allowed, 0., NEG), query < TOKENS)

    @triton.jit
    def _g27_rotary(Q, K, POS, COS, SIN, QOUT, KOUT,
                    QH: tl.constexpr, KH: tl.constexpr, D: tl.constexpr,
                    ROTARY: tl.constexpr, QT: tl.constexpr, QHEAD: tl.constexpr,
                    QC: tl.constexpr, KT: tl.constexpr, KHEAD: tl.constexpr,
                    KC: tl.constexpr, BLOCK: tl.constexpr,
                    LENGTH=None, TAILS=None, LANES: tl.constexpr = 0, SLOTS: tl.constexpr = 0):
        token, head = tl.program_id(0), tl.program_id(1)
        col = tl.arange(0, BLOCK)
        if SLOTS:
            block = token >= LANES
            row = tl.maximum(token - LANES, 0) // SLOTS
            tail = tl.load(TAILS + row, block, 0)
            position = (tl.load(LENGTH) + tl.where(block, (token - LANES) % SLOTS, token) + tail).to(tl.int64)
        else:
            position = tl.load(POS + token).to(tl.int64)
        other = tl.where(col < ROTARY // 2, col + ROTARY // 2, col - ROTARY // 2)
        if head < QH:
            x = tl.load(Q + token * QT + head * QHEAD + col * QC, col < D, 0)
            shifted = tl.load(Q + token * QT + head * QHEAD + other * QC, col < ROTARY, 0)
        else:
            x = tl.load(K + token * KT + (head - QH) * KHEAD + col * KC, col < D, 0)
            shifted = tl.load(K + token * KT + (head - QH) * KHEAD + other * KC, col < ROTARY, 0)
        dtype: tl.constexpr = Q.dtype.element_ty
        shifted = tl.where(col < ROTARY // 2, -shifted.to(tl.float32), shifted.to(tl.float32))
        cos = tl.load(COS + position * ROTARY + col, col < ROTARY, 0).to(tl.float32)
        sin = tl.load(SIN + position * ROTARY + col, col < ROTARY, 0).to(tl.float32)
        first = (x.to(tl.float32) * cos).to(dtype).to(tl.float32)
        second = (shifted * sin).to(dtype).to(tl.float32)
        value = tl.where(col < ROTARY, first + second, x.to(tl.float32))
        if head < QH:
            tl.store(QOUT + (token * QH + head) * D + col, value, col < D)
        else:
            tl.store(KOUT + (token * KH + head - QH) * D + col, value, col < D)

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
                 OWNS_ERR: tl.constexpr, BLOCK: tl.constexpr,
                 PIN_SLOT=None, PIN_MASK=None, RESET_PINS: tl.constexpr = False,
                 DENSE_PINS: tl.constexpr = False):
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
        if RESET_PINS:
            tl.store(PIN_SLOT + row, -1, (lane == 0) & (row < ROWS))
            if DENSE_PINS:
                tl.store(PIN_MASK + flat, False, flat < ROWS * LANES)

    @triton.jit
    def _c1_front_prep(RP, COMMITTED, PENDING, IDS, POSITIONS, SLOTS,
                       CAPACITY: tl.constexpr, LANES: tl.constexpr,
                       BLOCK: tl.constexpr, WEIGHT=None, EMBEDDINGS=None,
                       QUERY_SLOTS=None, MASK=None, WIDTH: tl.constexpr = 0,
                       BUCKET: tl.constexpr = 0, NEG: tl.constexpr = 0.,
                       FUSED: tl.constexpr = False):
        if FUSED:
            flat = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
            row = flat // WIDTH
            col = flat % WIDTH
            length = tl.load(RP + RP_L)
            start = tl.load(RP + C1_START)
            count = tl.load(RP + C1_COUNT)
            pending = tl.load(PENDING + row - 1, (row > 0) & (row < LANES), other=0)
            current = tl.load(COMMITTED + length)
            token = tl.where(row == 0, current, pending)
            value = tl.load(WEIGHT + token * WIDTH + col, row < LANES, other=0)
            tl.store(EMBEDDINGS + flat, value, row < LANES)
            meta = (col == 0) & (row < LANES)
            tl.store(IDS + row, token, meta)
            tl.store(POSITIONS + row, start + row, meta)
            tl.store(SLOTS + row, tl.where(row < count, start + row, CAPACITY + row), meta)
            tl.store(QUERY_SLOTS + row, length + row, meta)
            if BUCKET > 0:
                query = flat // BUCKET
                key = flat % BUCKET
                tl.store(MASK + flat, tl.where(key < length + query, 0., NEG),
                         flat < LANES * BUCKET)
            return
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

    @triton.jit
    def _context_gather_mark(OLD_LENGTH, LENGTH, PREFIX, TAIL, EMBEDDING, CONTEXT,
                             COS_TABLE, SIN_TABLE, CCOS, CSIN, HIDDEN, COS, SIN,
                             SOURCE, SLOTS, PREVIOUS,
                             P: tl.constexpr, T: tl.constexpr, H: tl.constexpr,
                             B: tl.constexpr, D: tl.constexpr,
                             COS_S0: tl.constexpr, COS_S1: tl.constexpr,
                             SIN_S0: tl.constexpr, SIN_S1: tl.constexpr,
                             TILE: tl.constexpr, ROPE_TILE: tl.constexpr):
        row, tile = tl.program_id(0), tl.program_id(1)
        column = tile * TILE + tl.arange(0, TILE)
        old, length = tl.load(OLD_LENGTH), tl.load(LENGTH)
        appended = old + row
        is_prefix = row < B
        live = is_prefix & (appended < length) & (appended < P)
        source = tl.where(is_prefix, tl.where(live, appended, 0), P + row - B)
        slot = tl.where(is_prefix, tl.where(live, appended, P + T + row), source)
        prefix = tl.load(PREFIX + source * H + column,
                         is_prefix & (source < length) & (column < H), other=0).to(tl.float32)
        tail = tl.load(TAIL + (row - B) * H + column,
                       ~is_prefix & (column < H), other=0).to(tl.float32)
        marker = tl.load(EMBEDDING + tl.where(is_prefix, 0, H) + column,
                         column < H, other=0).to(tl.float32)
        marked = tl.where(is_prefix, prefix, tail) + marker
        tl.store(HIDDEN + row * H + column, marked, column < H)
        tl.store(CONTEXT + source * H + column, marked,
                 (live | ~is_prefix) & (column < H))
        if tile == 0:
            tl.store(SOURCE + row, source)
            tl.store(SLOTS + row, slot)
            if row == 0:
                tl.store(PREVIOUS, old)
            d = tl.arange(0, ROPE_TILE)
            position = tl.where(is_prefix, source, length + row - B)
            cos = tl.load(COS_TABLE + position * COS_S0 + d * COS_S1, d < D, other=0)
            sin = tl.load(SIN_TABLE + position * SIN_S0 + d * SIN_S1, d < D, other=0)
            tl.store(COS + row * D + d, cos, d < D)
            tl.store(SIN + row * D + d, sin, d < D)
            tl.store(CCOS + source * D + d, cos, ~is_prefix & (d < D))
            tl.store(CSIN + source * D + d, sin, ~is_prefix & (d < D))
