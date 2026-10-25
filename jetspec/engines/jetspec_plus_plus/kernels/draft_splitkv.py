"""Opt-in BF16 draft split-KV with FP32 state and a fixed-order merge.

Softmax probabilities round to BF16 before the D128 value product. D256 uses
FP32 probabilities and a tf32x3 value product. GPU tests bound this different
accumulation order against FP32 SDPA.
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


def jpass27_enabled(q):
    """Select the long-bucket 27B drafter without changing the D128 policy."""
    from jetspec.engines.jetspec_plus_plus.profiles import kernel_value
    return (kernel_value('draft_fuse_skv27', '0') == '1' and triton is not None
            and q.is_cuda and q.dtype == torch.bfloat16 and q.ndim == 3
            and q.shape[1:] == (32, 128)
            and torch.cuda.get_device_capability(q.device) == (10, 0))


def a256_enabled(q, site):
    """Select hybrid adapter/decoder sites independently of the D128 selector."""
    from jetspec.engines.jetspec_plus_plus.profiles import kernel_value
    return (site in ('adapter', 'decoder') and kernel_value('draft_fuse_a256', '0') in ('1', site)
            and triton is not None and q.is_cuda and q.dtype == torch.bfloat16
            and q.shape[-1] == 256
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
                  CHUNK: tl.constexpr, ROW_TILE: tl.constexpr = 16):
        if DIM == 256:
            # Adjacent row tiles share the same K/V chunk in L2.
            tile, head, split = tl.program_id(0), tl.program_id(1), tl.program_id(2)
        else:
            head, split, tile = tl.program_id(0), tl.program_id(1), tl.program_id(2)
        row = tile * ROW_TILE + tl.arange(0, ROW_TILE)
        dim = tl.arange(0, DIM)
        lane = tl.arange(0, 64)
        q = tl.load(Q + head * QH + row[:, None] * QR + dim[None, :] * QD,
                    row[:, None] < ROWS, other=0)
        peak = tl.full((ROW_TILE,), -float('inf'), tl.float32)
        denom = tl.full((ROW_TILE,), 0., tl.float32)
        accum = tl.full((ROW_TILE, DIM), 0., tl.float32)
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
                if DIM == 256:
                    contribution = tl.dot(probability, v.to(tl.float32), input_precision='tf32x3')
                else:
                    contribution = tl.dot(probability.to(v.dtype), v)
                accum = accum * alpha[:, None] + contribution
                denom = denom * alpha + tl.sum(probability, 1)
                peak = new_peak
        offset = (split * HEADS + head) * ROWS + row
        tl.store(P + offset[:, None] * DIM + dim[None, :], accum, row[:, None] < ROWS)
        tl.store(M + offset, peak, row < ROWS)
        tl.store(S + offset, denom, row < ROWS)


if triton is not None:
    @triton.jit
    def _splitkv_partials27(Q, K, V, Mask, P, M, S,
                  QH: tl.constexpr, QR: tl.constexpr, QD: tl.constexpr,
                  KH: tl.constexpr, KR: tl.constexpr, KD: tl.constexpr,
                  VH: tl.constexpr, VR: tl.constexpr, VD: tl.constexpr,
                  MH: tl.constexpr, MR: tl.constexpr, MK: tl.constexpr,
                  HEADS: tl.constexpr, GROUPS: tl.constexpr,
                  ROWS: tl.constexpr, KEYS: tl.constexpr,
                  SCALE: tl.constexpr, CHUNK: tl.constexpr,
                  ROW_TILE: tl.constexpr = 16, KV_TILE: tl.constexpr = 32):
        # D256 dense-mask specialization; the existing D128 kernel stays intact.
        tile, head, split = tl.program_id(0), tl.program_id(1), tl.program_id(2)
        row = tile * ROW_TILE + tl.arange(0, ROW_TILE)
        dim = tl.arange(0, 256)
        lane = tl.arange(0, KV_TILE)
        q = tl.load(Q + head * QH + row[:, None] * QR + dim[None, :] * QD,
                    row[:, None] < ROWS, other=0)
        peak = tl.full((ROW_TILE,), -float('inf'), tl.float32)
        denom = tl.full((ROW_TILE,), 0., tl.float32)
        accum = tl.full((ROW_TILE, 256), 0., tl.float32)
        for start in range(split * CHUNK, tl.minimum((split + 1) * CHUNK, KEYS), KV_TILE):
            key = start + lane
            valid = (key[None, :] < KEYS) & (row[:, None] < ROWS)
            bias = tl.load(Mask + head * MH + row[:, None] * MR + key[None, :] * MK,
                           valid, other=-float('inf'))
            valid = valid & (bias > -10000.)
            if tl.sum(valid.to(tl.int32)) > 0:
                k = tl.load(K + (head // GROUPS) * KH + key[None, :] * KR + dim[:, None] * KD,
                            key[None, :] < KEYS, other=0)
                v = tl.load(V + (head // GROUPS) * VH + key[:, None] * VR + dim[None, :] * VD,
                            key[:, None] < KEYS, other=0)
                score = tl.dot(q, k) * SCALE
                score += bias
                score = tl.where(valid, score, -float('inf'))
                new_peak = tl.maximum(peak, tl.max(score, 1))
                safe = tl.where(new_peak == -float('inf'), 0., new_peak)
                alpha = tl.exp2((peak - safe) * 1.4426950408889634)
                probability = tl.exp2((score - safe[:, None]) * 1.4426950408889634)
                contribution = tl.dot(probability, v.to(tl.float32), input_precision='tf32x3')
                accum = accum * alpha[:, None] + contribution
                denom = denom * alpha + tl.sum(probability, 1)
                peak = new_peak
        offset = (split * HEADS + head) * ROWS + row
        tl.store(P + offset[:, None] * 256 + dim[None, :], accum, row[:, None] < ROWS)
        tl.store(M + offset, peak, row < ROWS)
        tl.store(S + offset, denom, row < ROWS)


def attention27_launch(rows, keys):
    """Opt in to gated adapter launch buckets; other row captures stay unchanged."""
    from jetspec.engines.jetspec_plus_plus.profiles import kernel_value
    if (kernel_value('draft_fuse_attn27b', '0') != '1' or rows not in (16, 64)
            or not 384 <= keys <= 2652):
        return None
    splits = min((keys + 63) // 64, max(4, (keys + 255) // 256))
    if rows == 16 and keys <= 1024:
        splits = 16
    return splits, 16, 32, 4, 2, 1


def row_tile(dim, rows, splits=4):
    """Use 32-row tiles when they add no padded rows and the grid stays large."""
    from jetspec.engines.jetspec_plus_plus.profiles import kernel_value
    return (32 if dim == 128 and splits >= 4 and (16 < rows <= 32 or 48 < rows <= 64)
            and kernel_value('draft_fuse_attn8a', '0') == '1'
            and kernel_value('draft_fuse_skv27', '0') != '1' else 16)


def merge_warps(dim, splits):
    """Opt in to the D128 one-warp merge; leave hybrid and tiny splits alone.

    The warp layout changes FP32 reduction rounding (class B). The explicit
    27B drafter selector retains its existing four-warp path.
    """
    from jetspec.engines.jetspec_plus_plus.profiles import kernel_value
    return (1 if dim == 128 and splits >= 4
            and kernel_value('draft_fuse_attn8', '0') == '1'
            and kernel_value('draft_fuse_skv27', '0') != '1' else 4)


def ordered_merge(dim, splits):
    """Use the original four-warp reduction tree in one warp for served 8B buckets."""
    from jetspec.engines.jetspec_plus_plus.profiles import kernel_value
    return (dim == 128 and 4 <= splits <= 10
            and kernel_value('draft_fuse_attn8a', '0') == '1'
            and kernel_value('draft_fuse_skv27', '0') != '1')


# Split count, row tile and merge warps for keys 128, 256, ..., 2432.
# Bounded serving production-family choices; the segment gate covers every entry.
_ATTENTION_LAUNCHES = {
    16: (
        (2, 16, 1), (4, 16, 1), (6, 16, 1), (8, 16, 1), (10, 16, 1),
        (12, 16, 1), (14, 16, 1), (16, 16, 1), (18, 16, 1), (20, 16, 1),
        (22, 16, 1), (12, 16, 1), (13, 16, 1), (28, 16, 1), (15, 16, 1),
        (16, 16, 1), (32, 16, 1), (32, 16, 1), (16, 16, 1),
    ),
    17: (
        (2, 16, 4), (4, 32, 1), (6, 16, 1), (8, 32, 1), (10, 32, 1),
        (12, 32, 1), (14, 32, 1), (16, 32, 1), (18, 32, 1), (20, 32, 1),
        (22, 32, 1), (24, 32, 1), (26, 32, 1), (28, 32, 1), (30, 32, 1),
        (32, 32, 1), (9, 32, 1), (9, 32, 1), (10, 32, 1),
    ),
    32: (
        (2, 16, 1), (4, 32, 1), (6, 16, 1), (8, 32, 1), (10, 32, 1),
        (12, 32, 1), (14, 32, 1), (16, 32, 1), (18, 32, 1), (20, 32, 1),
        (22, 32, 1), (24, 32, 1), (26, 32, 1), (28, 32, 1), (30, 32, 1),
        (8, 32, 1), (9, 32, 1), (9, 32, 1), (10, 32, 1),
    ),
    48: (
        (1, 16, 1), (4, 16, 1), (6, 16, 1), (8, 64, 1), (5, 64, 1),
        (6, 64, 1), (7, 64, 1), (8, 64, 1), (9, 64, 1), (5, 64, 1),
        (6, 64, 1), (6, 64, 1), (7, 64, 1), (7, 64, 1), (8, 64, 1),
        (8, 64, 1), (9, 64, 1), (9, 64, 1), (10, 64, 1),
    ),
    60: (
        (2, 16, 4), (4, 16, 1), (6, 32, 1), (8, 64, 1), (5, 64, 1),
        (6, 64, 1), (7, 64, 1), (8, 64, 1), (9, 64, 1), (5, 64, 1),
        (6, 64, 1), (6, 64, 1), (7, 64, 1), (7, 64, 1), (8, 64, 1),
        (8, 64, 1), (9, 64, 1), (9, 64, 1), (10, 64, 1),
    ),
}


def attention_launch(rows, keys):
    """Return the frozen D128 bucket's split, tile, warps, stages and merge warps."""
    splits, tile, merge = _ATTENTION_LAUNCHES[rows][keys // 128 - 1]
    return splits, tile, 4, 2, merge


_ATTENTION_TAIL_KEYS = (304, 400, 432, 528, 560, 912, 1040, 1072, 1552, 2448, 2480)
# Splits, row tile, ordered merge. Measured against attn8a plus attn8b.
# The two longest key counts retain that combined baseline after full-prefix timing.
_ATTENTION_TAIL_LAUNCHES = {
    16: ((5, 16, True), (7, 16, False), (7, 16, True), (9, 16, True),
         (9, 16, False), (15, 16, False), (17, 16, False), (17, 16, False),
         (25, 16, False), (10, 16, True), (10, 16, True)),
    64: ((5, 16, False), (7, 32, False), (7, 32, True), (9, 32, True),
         (9, 32, False), (15, 32, False), (17, 32, False), (17, 32, False),
         (25, 32, False), (10, 32, True), (10, 32, True)),
}


def extended_attention_launch(rows, keys):
    """Unify the unchanged 8B table with segment-gated packed-tail launches."""
    if rows in _ATTENTION_LAUNCHES and keys in range(128, 2433, 128):
        return (*attention_launch(rows, keys), False)
    if rows in _ATTENTION_TAIL_LAUNCHES and keys in _ATTENTION_TAIL_KEYS:
        splits, tile, ordered = _ATTENTION_TAIL_LAUNCHES[rows][_ATTENTION_TAIL_KEYS.index(keys)]
        return splits, tile, 4, 2, 1, ordered
    return None


def bucket_policy(dim, rows, keys):
    """Select measured 8B buckets explicitly; preserve hybrid and 27B launches."""
    from jetspec.engines.jetspec_plus_plus.profiles import kernel_value
    if (dim == 128 and kernel_value('draft_fuse_attn8c', '0') == '1'
            and kernel_value('draft_fuse_skv27', '0') != '1'):
        extended = extended_attention_launch(rows, keys)
        if extended is not None:
            return extended
    if (dim == 128 and rows in (16, 17, 32, 48, 60) and keys in range(128, 2433, 128)
            and kernel_value('draft_fuse_attn8b', '0') == '1'
            and kernel_value('draft_fuse_skv27', '0') != '1'):
        return attention_launch(rows, keys)
    return None


def attend(q, k, v, scale, *, mode, mask=None, length=None, count=None,
           prefix=0, packed=None, out=None, gate=None):
    """Attend [1,H,R,D] with a dense, device-length, or packed-bit mask."""
    from jetspec.engines.jetspec_plus_plus.kernels.len_attention import _merge
    from jetspec.engines.jetspec_plus_plus.profiles import kernel_value
    from jetspec.engines.jetspec_plus_plus.kernels.nvfp4.producers import attention_activation, _arguments
    modes = {'dense': 0, 'c1': 1, 'decoder': 2, 'merged': 3, 'packed': 4}
    if (mode not in modes or q.ndim != 4 or q.shape[0] != 1 or k.shape != v.shape
            or k.shape[0] != 1 or q.shape[1] % k.shape[1]
            or q.shape[-1] != k.shape[-1] or q.shape[-1] not in (64, 128, 256)):
        raise ValueError('unsupported draft split-KV layout')
    heads, rows, dim = q.shape[1:]
    keys = k.shape[2]
    splits = min(triton.cdiv(keys, 64), max(4, triton.cdiv(keys, 256)))
    policy = bucket_policy(dim, rows, keys)
    if policy is not None:
        splits = policy[0]
    launch27 = (attention27_launch(rows, keys) if dim == 256 and gate is not None
                and mode == 'dense' else None)
    if launch27 is not None:
        splits = launch27[0]
    chunk = triton.cdiv(triton.cdiv(keys, splits), 64) * 64
    if launch27 is not None:
        chunk = triton.cdiv(triton.cdiv(keys, splits), launch27[2]) * launch27[2]
    partial = torch.empty((splits, heads, rows, dim), device=q.device, dtype=torch.float32)
    peak = torch.empty((splits, heads, rows), device=q.device, dtype=torch.float32)
    denom = torch.empty_like(peak)
    if out is not None and (out.shape != q.shape or out.dtype != q.dtype or out.device != q.device):
        raise ValueError('draft split-KV output must match query shape, dtype and device')
    output = torch.empty_like(q) if out is None else out
    if gate is not None:
        if (dim != 256 or gate.shape != (1, rows, heads, dim) or
                gate.stride() != (rows * heads * dim * 2, heads * dim * 2, dim * 2, 1)
                or gate.dtype != q.dtype or gate.device != q.device
                or not output.transpose(1, 2).is_contiguous()):
            raise ValueError('gated split-KV requires the hybrid interleaved gate and token-major output')
        from jetspec.engines.jetspec_plus_plus.kernels.nvfp4.producers import allocate_activation
        produced = allocate_activation(output.transpose(1, 2).reshape(rows, heads * dim))
    else:
        produced = attention_activation(output)
    dummy = q
    mask_strides = ((0, 0, 0) if mask is None else
                    (0 if mask.shape[1] == 1 else mask.stride(1),
                     mask.stride(2), mask.stride(3)))
    tile = row_tile(dim, rows, splits)
    if policy is not None:
        tile = policy[1]
    if launch27 is not None:
        tile = launch27[1]
    grid = ((triton.cdiv(rows, tile), heads, splits) if dim == 256 else
            (heads, splits, triton.cdiv(rows, tile)))
    if launch27 is not None:
        _splitkv_partials27[grid](
            q, k, v, mask, partial, peak, denom, *q.stride()[1:], *k.stride()[1:], *v.stride()[1:],
            *mask_strides, heads, heads // k.shape[1], rows, keys, scale, chunk,
            ROW_TILE=tile, KV_TILE=launch27[2], num_warps=launch27[3],
            num_stages=launch27[4], enable_fp_fusion=False)
    else:
        _splitkv_partials[grid](
            q, k, v, dummy if mask is None else mask, dummy if length is None else length,
            dummy if count is None else count, dummy if packed is None else packed,
            partial, peak, denom, *q.stride()[1:], *k.stride()[1:], *v.stride()[1:],
            *mask_strides, heads, heads // k.shape[1], rows, keys, dim, scale,
            modes[mode], prefix, chunk, ROW_TILE=tile,
            num_warps=4 if policy is None else policy[2],
            num_stages=2 if policy is None else policy[3], enable_fp_fusion=False)
    extra = ({'BATCHED': True} if produced is not None and
             kernel_value('draft_fuse_am', '0') == '2' else {})
    if gate is not None:
        extra = {'Gate': gate, 'GATED': True}
    ordered = (policy[5] if policy is not None and len(policy) == 6 else
               policy is None and ordered_merge(dim, splits))
    if ordered:
        extra['ORDERED'] = True
    _merge[(heads, rows)](
        partial, peak, denom, output, *output.stride()[1:],
        heads, rows, dim, splits, triton.next_power_of_2(splits),
        *(_arguments(produced) if produced is not None else ()), MX8=produced is not None,
        num_warps=(policy[4] if policy is not None else
                   launch27[5] if launch27 is not None else
                   1 if ordered else merge_warps(dim, splits)), enable_fp_fusion=False, **extra)
    return output
