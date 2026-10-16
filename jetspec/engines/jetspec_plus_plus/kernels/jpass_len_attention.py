# Adaptation: package relocation and deferred optional dependencies.
"""One-launch J-pass attention over the row head's contiguous KV cache."""

from __future__ import annotations

import torch
from jetspec.engines.jetspec_plus_plus._optional import triton
from jetspec.engines.jetspec_plus_plus._optional import tl


@triton.jit
def _attend(Q, CACHE, MASK, LENGTHS, OUT,
            QS: tl.constexpr, QH: tl.constexpr, QD: tl.constexpr,
            CS: tl.constexpr, CV: tl.constexpr, CH: tl.constexpr, CD: tl.constexpr,
            OS: tl.constexpr, OH: tl.constexpr, OD: tl.constexpr,
            HEADS: tl.constexpr, GROUPS: tl.constexpr, ROWS: tl.constexpr,
            KEYS: tl.constexpr, PREFIX: tl.constexpr, DIM: tl.constexpr,
            SCALE: tl.constexpr, BK: tl.constexpr):
    head = tl.program_id(0)
    tile = tl.program_id(1)
    group = tl.program_id(2)
    row = tile * 16 + tl.arange(0, 16)
    key_lane = tl.arange(0, BK)
    dim = tl.arange(0, DIM)
    kv_head = head // GROUPS
    length = tl.load(LENGTHS + group)
    q = tl.load(Q + (group * ROWS + row[:, None]) * QS + head * QH + dim[None, :] * QD,
                (row[:, None] < ROWS), other=0)
    maximum = tl.full((16,), float('-inf'), tl.float32)
    denominator = tl.full((16,), 0, tl.float32)
    numerator = tl.full((16, DIM), 0, tl.float32)
    for start in range(0, KEYS, BK):
        # Prefix rows past the live length contain padding. The other regions
        # are controlled by the packed mask, including R1 refresh edits.
        if start >= PREFIX or start < length:
            key = start + key_lane
            bit = (group * ROWS + row[:, None]) * KEYS + key[None, :]
            packed = tl.load(MASK + bit // 8,
                             (row[:, None] < ROWS) & (key[None, :] < KEYS), other=0)
            valid = ((packed >> (bit % 8)) & 1) != 0
            if tl.sum(valid.to(tl.int32)) > 0:
                k = tl.load(CACHE + (group * KEYS + key[None, :]) * CS + kv_head * CH
                            + dim[:, None] * CD,
                            key[None, :] < KEYS, other=0)
                v = tl.load(CACHE + (group * KEYS + key[:, None]) * CS + CV + kv_head * CH
                            + dim[None, :] * CD,
                            key[:, None] < KEYS, other=0)
                score = tl.dot(q, k) * SCALE
                score = tl.where(valid, score, float('-inf'))
                block_max = tl.max(score, 1)
                next_max = tl.maximum(maximum, block_max)
                safe_max = tl.where(next_max == float('-inf'), 0., next_max)
                alpha = tl.where(maximum == float('-inf'), 0.,
                                 tl.exp2((maximum - safe_max) * 1.4426950408889634))
                probability = tl.exp2((score - safe_max[:, None]) * 1.4426950408889634)
                denominator = denominator * alpha + tl.sum(probability, 1)
                numerator = numerator * alpha[:, None] + tl.dot(
                    probability.to(v.dtype), v)
                maximum = next_max
    result = numerator / tl.maximum(denominator[:, None], 1.0e-20)
    tl.store(OUT + (group * ROWS + row[:, None]) * OS + head * OH + dim[None, :] * OD,
             result, row[:, None] < ROWS)


@triton.jit
def _blockhead_mask(MASK, LENGTH, KEYS: tl.constexpr, PREFIX: tl.constexpr,
                    ACTIVE: tl.constexpr, BYTES: tl.constexpr, BLOCK: tl.constexpr):
    byte = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    bit = byte[:, None] * 8 + tl.arange(0, 8)[None, :]
    row = bit // KEYS
    key = bit % KEYS
    length = tl.load(LENGTH)
    live = (row < ACTIVE) & ((key < length) | (key >= PREFIX))
    value = tl.sum(tl.where(live, 1 << (bit % 8), 0), axis=1)
    tl.store(MASK + byte, value.to(tl.uint8), byte < BYTES)


def blockhead_mask(length: torch.Tensor, capacity: int, block_size: int,
                   rows: int, output: torch.Tensor) -> None:
    """Pack the live context and every block key for each bidirectional query."""
    keys = capacity + block_size
    if capacity % 64 or rows % 16 or rows < block_size or output.numel() * 8 < rows * keys:
        raise ValueError('unsupported blockhead mask geometry')
    count = output.numel()
    _blockhead_mask[(triton.cdiv(count, 256),)](
        output, length, keys, capacity, block_size, count, 256)


def attend(q: torch.Tensor, cache: torch.Tensor, packed_mask: torch.Tensor,
           lengths: torch.Tensor, *, groups: int, prefix: int, scale: float) -> torch.Tensor:
    """Attend [query, head, dim] to [key, 2, 1, KV head, dim].

    The mask is FlashInfer's live little-endian custom mask buffer. All
    allocation and launch geometry depend only on the prepared bucket.
    """
    rows, heads, dim = q.shape
    keys = cache.shape[0] // groups
    group_rows = rows // groups
    if (rows % groups or group_rows % 16 or heads % cache.shape[3]
            or cache.shape[1:3] != (2, 1) or cache.shape[-1] != dim
            or dim not in (64, 128, 256) or packed_mask.numel() * 8 < rows * keys):
        raise ValueError('unsupported J-pass attention layout')
    output = torch.empty_like(q)
    _attend[(heads, group_rows // 16, groups)](
        q, cache, packed_mask, lengths, output,
        *q.stride(), cache.stride(0), cache.stride(1), cache.stride(3), cache.stride(4),
        *output.stride(), heads, heads // cache.shape[3], group_rows, keys, prefix,
        dim, scale, 64, num_warps=4, enable_fp_fusion=False)
    return output
