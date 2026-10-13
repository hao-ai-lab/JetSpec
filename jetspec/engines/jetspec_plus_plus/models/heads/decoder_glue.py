"""Capture-safe, exact decoder prefix and window staging."""

from __future__ import annotations

import torch

try:
    from jetspec.engines.jetspec_plus_plus._optional import triton
    from jetspec.engines.jetspec_plus_plus._optional import tl
except ImportError:
    triton = None
    tl = None


if triton is not None:
    @triton.jit
    def _prefix(RP, PENDING, COS_TABLE, SIN_TABLE, MASK, POS, SLOTS,
                COS, SIN, REJECTED, LENGTH_OFFSET: tl.constexpr,
                COUNT_OFFSET: tl.constexpr, CORRECTION_OFFSET: tl.constexpr,
                MISS_OFFSET: tl.constexpr, LANES: tl.constexpr,
                BUCKET: tl.constexpr, DIM: tl.constexpr,
                TABLE_ROWS: tl.constexpr, NEG: tl.constexpr,
                WIRE: tl.constexpr, ZERO_MISS: tl.constexpr,
                MASK_BLOCKS: tl.constexpr, ROPE_BLOCKS: tl.constexpr,
                BLOCK: tl.constexpr):
        program = tl.program_id(0)
        x = tl.arange(0, BLOCK)
        length = tl.load(RP + LENGTH_OFFSET)
        count = tl.load(RP + COUNT_OFFSET)
        if program < MASK_BLOCKS:
            flat = program * BLOCK + x
            lane = flat // BUCKET
            key = flat % BUCKET
            position = length - count + lane
            allowed = ((lane < count) & (key <= position) & (key < length)) | \
                      ((lane >= count) & (key == 0))
            tl.store(MASK + flat, tl.where(allowed, 0., NEG), flat < LANES * BUCKET)
        elif program < MASK_BLOCKS + ROPE_BLOCKS:
            flat = (program - MASK_BLOCKS) * BLOCK + x
            lane = flat // DIM
            column = flat % DIM
            position = length - count + lane
            valid = (flat < LANES * DIM) & (position >= 0) & (position < TABLE_ROWS)
            cos = tl.load(COS_TABLE + position * DIM + column, valid, other=0)
            sin = tl.load(SIN_TABLE + position * DIM + column, valid, other=0)
            tl.store(COS + flat, cos, flat < LANES * DIM)
            tl.store(SIN + flat, sin, flat < LANES * DIM)
        else:
            lane = x
            position = length - count + lane
            tl.store(POS + lane, position, lane < LANES)
            tl.store(SLOTS + lane, tl.where(lane < count, position, BUCKET - 1), lane < LANES)
            pending = tl.load(PENDING + tl.where(WIRE, lane, lane - 1),
                              (lane < LANES - 1) if WIRE else ((lane > 0) & (lane < LANES)), other=0)
            if WIRE:
                rejected = tl.where(lane == LANES - 1, -1, pending)
            else:
                rejected = tl.where(lane == 0, tl.load(RP + CORRECTION_OFFSET), pending)
            if ZERO_MISS:
                rejected = tl.where((lane == 0) & (tl.load(RP + MISS_OFFSET) != 0),
                                    -1, rejected)
            tl.store(REJECTED + lane, rejected, lane < LANES)

    @triton.jit
    def _window(LENGTH, STALE, STAGED, COS_TABLE, SIN_TABLE,
                HIDDEN, MASK, SLOTS, COS, SIN,
                ROWS: tl.constexpr, H: tl.constexpr, SOURCE_H: tl.constexpr,
                BUCKET: tl.constexpr, DIM: tl.constexpr,
                TABLE_ROWS: tl.constexpr, NEG: tl.constexpr,
                MASK_BLOCKS: tl.constexpr, HIDDEN_BLOCKS: tl.constexpr,
                ROPE_BLOCKS: tl.constexpr, BLOCK: tl.constexpr):
        program = tl.program_id(0)
        x = tl.arange(0, BLOCK)
        length = tl.load(LENGTH)
        if program < MASK_BLOCKS:
            flat = program * BLOCK + x
            row = flat // BUCKET
            key = flat % BUCKET
            allowed = (key < length) | ((key >= length) & (key - length <= row))
            tl.store(MASK + flat, tl.where(allowed, 0., NEG), flat < ROWS * BUCKET)
        elif program < MASK_BLOCKS + HIDDEN_BLOCKS:
            flat = (program - MASK_BLOCKS) * BLOCK + x
            row = flat // H
            col = flat % H
            stale = tl.load(STALE + flat, (flat < ROWS * H) & (row < 4), other=0)
            current = tl.load(STAGED + (row - 4) * SOURCE_H + SOURCE_H - H + col,
                              (flat < ROWS * H) & (row >= 4), other=0)
            tl.store(HIDDEN + flat, tl.where(row < 4, stale, current), flat < ROWS * H)
        elif program < MASK_BLOCKS + HIDDEN_BLOCKS + ROPE_BLOCKS:
            flat = (program - MASK_BLOCKS - HIDDEN_BLOCKS) * BLOCK + x
            row = flat // DIM
            col = flat % DIM
            position = length + row
            valid = (flat < ROWS * DIM) & (position >= 0) & (position < TABLE_ROWS)
            cos = tl.load(COS_TABLE + position * DIM + col, valid, other=0)
            sin = tl.load(SIN_TABLE + position * DIM + col, valid, other=0)
            tl.store(COS + flat, cos, flat < ROWS * DIM)
            tl.store(SIN + flat, sin, flat < ROWS * DIM)
        else:
            tl.store(SLOTS + x, length + x, x < ROWS)


def prefix_glue(rp, pending, cos_table, sin_table, mask, positions,
                slots, cos, sin, rejected, *, length_offset, count_offset,
                correction_offset, miss_offset, wire, zero_miss):
    """Fill preallocated prefix inputs in one launch on CUDA."""
    lanes, bucket = mask.shape[-2:]
    dim = cos.shape[-1]
    if (mask.shape != (1, 1, lanes, bucket) or positions.shape != (1, lanes)
            or slots.shape != (lanes,) or cos.shape != sin.shape
            or cos.shape != (1, lanes, dim) or rejected.shape != (lanes,)
            or pending.shape != (lanes - 1,) or cos_table.shape != sin_table.shape
            or cos_table.shape[1] != dim):
        raise ValueError('decoder prefix glue shape changed')
    if not rp.is_cuda:
        length, count = rp[length_offset], rp[count_offset]
        lane = torch.arange(lanes, device=rp.device)
        positions.copy_((length - count + lane)[None])
        slots.copy_(torch.where(lane < count, positions[0], bucket - 1))
        key = torch.arange(bucket, device=rp.device)[None]
        allowed = ((lane[:, None] < count) & (key <= positions[0, :, None]) & (key < length)) | \
                  ((lane[:, None] >= count) & (key == 0))
        mask.zero_().masked_fill_(~allowed[None, None], torch.finfo(mask.dtype).min)
        cos.copy_(cos_table.index_select(0, positions[0])[None])
        sin.copy_(sin_table.index_select(0, positions[0])[None])
        rejected.copy_(torch.cat((pending, pending.new_full((1,), -1))) if wire else
                       torch.cat((rp[correction_offset].view(1), pending)))
        if zero_miss:
            rejected[0] = torch.where(rp[miss_offset] != 0, -1, rejected[0])
        return
    if triton is None:
        raise RuntimeError('decoder glue fusion requires Triton')
    block = 256
    mb = triton.cdiv(lanes * bucket, block)
    rb = triton.cdiv(lanes * dim, block)
    _prefix[(mb + rb + 1,)](rp, pending, cos_table, sin_table, mask, positions,
                            slots, cos, sin, rejected, length_offset, count_offset,
                            correction_offset, miss_offset, lanes, bucket, dim,
                            cos_table.shape[0], float(torch.finfo(mask.dtype).min),
                            wire, zero_miss, mb, rb, block)


def window_glue(length, stale, staged, cos_table, sin_table,
                hidden, mask, slots, cos, sin):
    """Fill preallocated window input, mask, positions and rotary rows."""
    rows, hidden_size = hidden.shape[1:]
    bucket = mask.shape[-1]
    dim = cos.shape[-1]
    if (stale.shape != (4, hidden_size) or staged.shape[1] != rows - 4
            or staged.shape[-1] < hidden_size or mask.shape != (1, 1, rows, bucket)
            or slots.shape != (rows,) or cos.shape != sin.shape
            or cos.shape != (1, rows, dim) or cos_table.shape != sin_table.shape
            or cos_table.shape[1] != dim):
        raise ValueError('decoder window glue shape changed')
    if not length.is_cuda:
        hidden[0].copy_(torch.cat((stale, staged[0, :, -hidden_size:]), dim=0))
        lane = torch.arange(rows, device=length.device)
        slots.copy_(length + lane)
        key = torch.arange(bucket, device=length.device)[None]
        allowed = (key < length) | ((key >= length) & (key - length <= lane[:, None]))
        mask.zero_().masked_fill_(~allowed[None, None], torch.finfo(mask.dtype).min)
        cos.copy_(cos_table.index_select(0, slots)[None])
        sin.copy_(sin_table.index_select(0, slots)[None])
        return
    if triton is None:
        raise RuntimeError('decoder glue fusion requires Triton')
    block = 256
    mb = triton.cdiv(rows * bucket, block)
    hb = triton.cdiv(rows * hidden_size, block)
    rb = triton.cdiv(rows * dim, block)
    _window[(mb + hb + rb + 1,)](length, stale, staged, cos_table, sin_table,
                                hidden, mask, slots, cos, sin, rows, hidden_size,
                                staged.shape[-1], bucket, dim, cos_table.shape[0],
                                float(torch.finfo(mask.dtype).min), mb, hb, rb, block)
