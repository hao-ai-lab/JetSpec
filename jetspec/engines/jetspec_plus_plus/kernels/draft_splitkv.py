"""Opt-in BF16 draft split-KV with FP32 state and a fixed-order merge.

Softmax probabilities round to BF16 before the value product. This is the same
function with different rounding, bounded against FP32 SDPA by the GPU tests.
"""
from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # CPU-only hosts: the knob stays off
    triton = None


def enabled(q):
    from jetspec.engines.jetspec_plus_plus.profiles import kernel_enabled
    return (triton is not None and kernel_enabled('draft_fuse_skv', '1') and q.is_cuda
            and q.dtype == torch.bfloat16
            and q.shape[-1] == 128
            and torch.cuda.get_device_capability(q.device) == (10, 0))


if triton is not None:
    @triton.jit
    def _splitkv_partials(Q, K, V, Mask, Length, Count, Packed, P, M, S,
                  QH: tl.constexpr, QR: tl.constexpr, QD: tl.constexpr,
                  KH: tl.constexpr, KR: tl.constexpr, KD: tl.constexpr,
                  VH: tl.constexpr, VR: tl.constexpr, VD: tl.constexpr,
                  MH: tl.constexpr, MR: tl.constexpr, MK: tl.constexpr,
                  HEADS: tl.constexpr, GROUPS: tl.constexpr,
                  ROWS: tl.constexpr, KEYS: tl.constexpr, DIM: tl.constexpr,
                  SCALE: tl.constexpr, MODE: tl.constexpr, PREFIX: tl.constexpr,
                  CHUNK: tl.constexpr):
        head, split, tile = tl.program_id(0), tl.program_id(1), tl.program_id(2)
        row = tile * 16 + tl.arange(0, 16)
        dim = tl.arange(0, DIM)
        lane = tl.arange(0, 64)
        q = tl.load(Q + head * QH + row[:, None] * QR + dim[None, :] * QD,
                    row[:, None] < ROWS, other=0)
        peak = tl.full((16,), -float('inf'), tl.float32)
        denom = tl.full((16,), 0., tl.float32)
        accum = tl.full((16, DIM), 0., tl.float32)
        if MODE != 0:
            length = tl.load(Length)
        if MODE == 2 or MODE == 3:
            count = tl.load(Count)
        for start in range(split * CHUNK, tl.minimum((split + 1) * CHUNK, KEYS), 64):
            key = start + lane
            valid = (key[None, :] < KEYS) & (row[:, None] < ROWS)
            if MODE == 0:
                bias = tl.load(Mask + head * MH + row[:, None] * MR + key[None, :] * MK,
                               valid, other=-float('inf'))
                valid = valid & (bias > -10000.)
            elif MODE == 1:
                valid = valid & (key[None, :] < length + row[:, None])
            elif MODE == 2 or MODE == 3:
                decoder = tl.where(row[:, None] < count,
                                   key[None, :] < length - count + row[:, None] + 1,
                                   key[None, :] == 0)
                if MODE == 3:
                    window = key[None, :] < length + row[:, None] - PREFIX + 1
                    decoder = tl.where(row[:, None] < PREFIX, decoder, window)
                valid = valid & decoder
            else:
                bit = row[:, None] * KEYS + key[None, :]
                byte = tl.load(Packed + bit // 8, valid, other=0)
                valid = valid & (((byte >> (bit % 8)) & 1) != 0)
                # Prefix padding is skipped independently of the packed mask.
                valid = valid & ((key[None, :] >= PREFIX) | (key[None, :] < length))
            if tl.sum(valid.to(tl.int32)) > 0:
                k = tl.load(K + (head // GROUPS) * KH + key[None, :] * KR + dim[:, None] * KD,
                            key[None, :] < KEYS, other=0)
                v = tl.load(V + (head // GROUPS) * VH + key[:, None] * VR + dim[None, :] * VD,
                            key[:, None] < KEYS, other=0)
                score = tl.dot(q, k) * SCALE
                if MODE == 0:
                    score += bias
                score = tl.where(valid, score, -float('inf'))
                new_peak = tl.maximum(peak, tl.max(score, 1))
                safe = tl.where(new_peak == -float('inf'), 0., new_peak)
                alpha = tl.exp2((peak - safe) * 1.4426950408889634)
                probability = tl.exp2((score - safe[:, None]) * 1.4426950408889634)
                accum = accum * alpha[:, None] + tl.dot(probability.to(v.dtype), v)
                denom = denom * alpha + tl.sum(probability, 1)
                peak = new_peak
        offset = (split * HEADS + head) * ROWS + row
        tl.store(P + offset[:, None] * DIM + dim[None, :], accum, row[:, None] < ROWS)
        tl.store(M + offset, peak, row < ROWS)
        tl.store(S + offset, denom, row < ROWS)


def attend(q, k, v, scale, *, mode, mask=None, length=None, count=None,
           prefix=0, packed=None):
    """Attend [1,H,R,D] with a dense, device-length, or packed-bit mask."""
    from jetspec.engines.jetspec_plus_plus.kernels.len_attention import _merge
    modes = {'dense': 0, 'c1': 1, 'decoder': 2, 'merged': 3, 'packed': 4}
    if (mode not in modes or q.ndim != 4 or q.shape[0] != 1 or k.shape != v.shape
            or k.shape[0] != 1 or q.shape[1] % k.shape[1]
            or q.shape[-1] != k.shape[-1] or q.shape[-1] not in (64, 128, 256)):
        raise ValueError('unsupported draft split-KV layout')
    heads, rows, dim = q.shape[1:]
    keys = k.shape[2]
    splits = min(triton.cdiv(keys, 64), max(4, triton.cdiv(keys, 256)))
    chunk = triton.cdiv(triton.cdiv(keys, splits), 64) * 64
    partial = torch.empty((splits, heads, rows, dim), device=q.device, dtype=torch.float32)
    peak = torch.empty((splits, heads, rows), device=q.device, dtype=torch.float32)
    denom = torch.empty_like(peak)
    output = torch.empty_like(q)
    dummy = q
    mask_strides = ((0, 0, 0) if mask is None else
                    (0 if mask.shape[1] == 1 else mask.stride(1),
                     mask.stride(2), mask.stride(3)))
    _splitkv_partials[(heads, splits, triton.cdiv(rows, 16))](
        q, k, v, dummy if mask is None else mask, dummy if length is None else length,
        dummy if count is None else count, dummy if packed is None else packed,
        partial, peak, denom, *q.stride()[1:], *k.stride()[1:], *v.stride()[1:],
        *mask_strides, heads, heads // k.shape[1], rows, keys, dim, scale,
        modes[mode], prefix, chunk, num_warps=4, num_stages=2, enable_fp_fusion=False)
    _merge[(heads, rows)](
        partial, peak, denom, output, *output.stride()[1:],
        heads, rows, dim, splits, triton.next_power_of_2(splits),
        num_warps=4, enable_fp_fusion=False)
    return output
