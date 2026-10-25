# Adaptation: package relocation and deferred optional dependencies.
"""Device-length GQA attention for the fixed chain query blocks."""

from __future__ import annotations

import torch
from jetspec.engines.jetspec_plus_plus._optional import triton
from jetspec.engines.jetspec_plus_plus._optional import tl
from jetspec.engines.jetspec_plus_plus.jetspec.c1_fused import attention_output
# Adaptation: CUDA helpers are needed only when Triton kernels execute.
try:
    from triton.language.extra.cuda import libdevice
    from jetspec.engines.jetspec_plus_plus.kernels.nvfp4.producers import store_mx8, store_mx8_rows
    from jetspec.engines.jetspec_plus_plus.kernels.nvfp4._w4a8_quant import _launch_dependents
except (ModuleNotFoundError, ImportError):
    libdevice = store_mx8 = store_mx8_rows = _launch_dependents = None


@triton.jit
def _partials(Q, K, V, LENGTH, COUNT, P, M, S,
              QH: tl.constexpr, QR: tl.constexpr, QD: tl.constexpr,
              KH: tl.constexpr, KR: tl.constexpr, KD: tl.constexpr,
              VH: tl.constexpr, VR: tl.constexpr, VD: tl.constexpr,
              HEADS: tl.constexpr, GROUPS: tl.constexpr,
              ROWS: tl.constexpr, KEYS: tl.constexpr, DIM: tl.constexpr,
              SCALE: tl.constexpr, MODE: tl.constexpr,
              SPLITS: tl.constexpr, CHUNK: tl.constexpr,
              BK: tl.constexpr):
    head = tl.program_id(0)
    split = tl.program_id(1)
    kv_head = head // GROUPS
    row = tl.program_id(2) * 16 + tl.arange(0, 16)
    dim = tl.arange(0, DIM)
    key_lane = tl.arange(0, BK)
    length = tl.load(LENGTH)
    if MODE == 1:
        count = tl.load(COUNT)
    q = tl.load(Q + head * QH + row[:, None] * QR + dim[None, :] * QD,
                (row[:, None] < ROWS), other=0)
    maximum = tl.full((16,), float('-inf'), tl.float32)
    denominator = tl.full((16,), 0, tl.float32)
    numerator = tl.full((16, DIM), 0, tl.float32)
    for start in range(split * CHUNK, tl.minimum((split + 1) * CHUNK, KEYS), BK):
        if MODE == 0:
            useful = start < length + ROWS - 1
        else:
            useful = start < length
        if useful:
            key = start + key_lane
            k = tl.load(K + kv_head * KH + key[None, :] * KR + dim[:, None] * KD,
                        (key[None, :] < KEYS), other=0)
            v = tl.load(V + kv_head * VH + key[:, None] * VR + dim[None, :] * VD,
                        (key[:, None] < KEYS), other=0)
            score = tl.dot(q, k) * SCALE
            if MODE == 0:
                valid = key[None, :] < length + row[:, None]
            else:
                valid = tl.where(row[:, None] < count,
                                 key[None, :] < length - count + row[:, None] + 1,
                                 key[None, :] == 0)
            score = tl.where(valid & (key[None, :] < KEYS) &
                             (row[:, None] < ROWS), score, float('-inf'))
            block_max = tl.max(score, 1)
            new_max = tl.maximum(maximum, block_max)
            safe_max = tl.where(new_max == float('-inf'), 0., new_max)
            alpha = tl.where(maximum == float('-inf'), 0.,
                             tl.exp2((maximum - safe_max) * 1.4426950408889634))
            probability = tl.exp2((score - safe_max[:, None]) * 1.4426950408889634)
            denominator = denominator * alpha + tl.sum(probability, 1)
            contribution = tl.dot(probability, v.to(tl.float32),
                                  input_precision="tf32x3")
            numerator = numerator * alpha[:, None] + contribution
            maximum = new_max
    offset = (split * HEADS + head) * ROWS + row
    tl.store(M + offset, maximum, row < ROWS)
    tl.store(S + offset, denominator, row < ROWS)
    tl.store(P + offset[:, None] * DIM + dim[None, :], numerator,
             row[:, None] < ROWS)


@triton.jit
def _merge_stripe(P, M, head, row, dim, safe,
                  HEADS: tl.constexpr, ROWS: tl.constexpr, DIM: tl.constexpr,
                  SPLITS: tl.constexpr, BS: tl.constexpr, STRIPE: tl.constexpr):
    """Keep the register-local accumulation of one original reduction warp."""
    offset = (STRIPE * HEADS + head) * ROWS + row
    maximum = tl.load(M + offset, STRIPE < SPLITS, other=float('-inf'))
    weight = tl.where(maximum == float('-inf'), 0.,
                      tl.exp2((maximum - safe) * 1.4426950408889634))
    value = tl.load(P + offset * DIM + dim, STRIPE < SPLITS, other=0.) * weight
    for split in tl.static_range(STRIPE + 4, BS, 4):
        offset = (split * HEADS + head) * ROWS + row
        maximum = tl.load(M + offset, split < SPLITS, other=float('-inf'))
        weight = tl.where(maximum == float('-inf'), 0.,
                          tl.exp2((maximum - safe) * 1.4426950408889634))
        value += tl.load(P + offset * DIM + dim, split < SPLITS, other=0.) * weight
    return value


@triton.jit
def _merge(P, M, S, OUT, OH: tl.constexpr, OR: tl.constexpr, OD: tl.constexpr,
           HEADS: tl.constexpr, ROWS: tl.constexpr, DIM: tl.constexpr,
           SPLITS: tl.constexpr, BS: tl.constexpr, Q=None, SF=(),
           TILES: tl.constexpr = (), STRIDES: tl.constexpr = (),
           MX8: tl.constexpr = False, BATCHED: tl.constexpr = False,
           Gate=None, GATED: tl.constexpr = False, ORDERED: tl.constexpr = False):
    head = tl.program_id(0)
    row = tl.program_id(1)
    if MX8:
        _launch_dependents(row)
    split = tl.arange(0, BS)
    dim = tl.arange(0, DIM)
    offset = (split * HEADS + head) * ROWS + row
    maximum = tl.load(M + offset, split < SPLITS, other=float('-inf'))
    denominator = tl.load(S + offset, split < SPLITS, other=0.)
    numerator = tl.load(P + offset[:, None] * DIM + dim[None, :],
                        split[:, None] < SPLITS, other=0.)
    common = tl.max(maximum, 0)
    safe = tl.where(common == float('-inf'), 0., common)
    weight = tl.where(maximum == float('-inf'), 0.,
                      tl.exp2((maximum - safe) * 1.4426950408889634))
    total = tl.sum(denominator * weight, 0)
    if ORDERED:
        tl.static_assert(DIM == 128 and SPLITS >= 4 and SPLITS <= 10)
        # Four original warps own split stripes 0/4/8/12, 1/5/9/13, etc.
        # Their final shared-memory butterfly pairs 0+2, 1+3, then combines.
        a = _merge_stripe(P, M, head, row, dim, safe, HEADS, ROWS, DIM, SPLITS, BS, 0)
        b = _merge_stripe(P, M, head, row, dim, safe, HEADS, ROWS, DIM, SPLITS, BS, 1)
        c = _merge_stripe(P, M, head, row, dim, safe, HEADS, ROWS, DIM, SPLITS, BS, 2)
        d = _merge_stripe(P, M, head, row, dim, safe, HEADS, ROWS, DIM, SPLITS, BS, 3)
        output = ((a + c) + (b + d)) / tl.maximum(total, 1.0e-20)
    else:
        output = tl.sum(numerator * weight[:, None], 0) / tl.maximum(total, 1.0e-20)
    if GATED:
        gate = tl.load(Gate + row * HEADS * DIM * 2 + head * DIM * 2 + dim).to(tl.float32)
        sigmoid = (1.0 / (1.0 + libdevice.exp(-gate))).to(tl.bfloat16).to(tl.float32)
        output = output.to(tl.bfloat16).to(tl.float32) * sigmoid
    tl.store(OUT + head * OH + row * OR + dim * OD, output)
    if MX8:
        if BATCHED:
            store_mx8_rows(tl.reshape(output, (1, DIM)), Q, SF,
                           tl.full((1,), 0, tl.int32) + row, head * DIM,
                           HEADS * DIM, DIM, ROWS, TILES, STRIDES,
                           BATCHED=True, OWNED=1)
        else:
            store_mx8(output, Q, SF, row, 0, head * DIM, HEADS * DIM, DIM, TILES, STRIDES)


def attend(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
           length: torch.Tensor, scale: float, *, mode: str,
           count: torch.Tensor | None = None, long: bool = False) -> torch.Tensor:
    """Attend [1,H,R,D] to [1,KV,K,D] with a device scalar length.

    ``c1`` permits key < length + row; ``decoder`` permits the active
    prefix through length - count + row, and key zero for inactive rows.
    """
    from jetspec.engines.jetspec_plus_plus.jetspec.profile import enabled as profile_enabled
    max_rows = 32 if profile_enabled() else 16
    if (q.ndim != 4 or k.shape != v.shape or q.shape[0] != 1 or k.shape[0] != 1
            or not 0 < q.shape[-2] <= max_rows or q.shape[-1] != k.shape[-1]
            or q.shape[1] % k.shape[1] or q.shape[-1] not in (64, 128, 256)
            or mode not in ('c1', 'decoder') or (mode == 'decoder' and count is None)):
        raise ValueError('unsupported chain length attention layout')
    heads, rows, dim = q.shape[1:]
    keys = k.shape[-2]
    from jetspec.engines.jetspec_plus_plus.kernels.draft_splitkv import enabled, attend as split_attend
    if enabled(q):
        return split_attend(q, k, v, scale, mode=mode, length=length, count=count)
    # Bucket size bounds the split geometry; each split reads the live device
    # length and skips blocks outside the exact prefix on replay.
    splits = (min(4, max(1, triton.cdiv(keys, 256))))
    chunk = triton.cdiv(triton.cdiv(keys, splits), 64) * 64
    partial = torch.empty((splits, heads, rows, dim), device=q.device, dtype=torch.float32)
    maximum = torch.empty((splits, heads, rows), device=q.device, dtype=torch.float32)
    denominator = torch.empty_like(maximum)
    output = attention_output(q) if mode == 'c1' else torch.empty_like(q)
    _partials[(heads, splits, triton.cdiv(rows, 16))](
        q, k, v, length, length if count is None else count,
        partial, maximum, denominator,
        q.stride(1), q.stride(2), q.stride(3),
        k.stride(1), k.stride(2), k.stride(3),
        v.stride(1), v.stride(2), v.stride(3),
        heads, heads // k.shape[1], rows, keys, dim, scale,
        0 if mode == 'c1' else 1, splits, chunk, 64,
        num_warps=4, enable_fp_fusion=False)
    _merge[(heads, rows)](partial, maximum, denominator, output,
                          output.stride(1), output.stride(2), output.stride(3),
                          heads, rows, dim, splits, triton.next_power_of_2(splits),
                          num_warps=4, enable_fp_fusion=False)
    return output


@triton.jit
def _merged_partials(Q, K, V, LENGTH, COUNT, P, M, S,
                     QH: tl.constexpr, QR: tl.constexpr, QD: tl.constexpr,
                     KH: tl.constexpr, KR: tl.constexpr, KD: tl.constexpr,
                     VH: tl.constexpr, VR: tl.constexpr, VD: tl.constexpr,
                     HEADS: tl.constexpr, GROUPS: tl.constexpr,
                     ROWS: tl.constexpr, PREFIX: tl.constexpr,
                     KEYS: tl.constexpr, DIM: tl.constexpr,
                     SCALE: tl.constexpr, CHUNK: tl.constexpr, BK: tl.constexpr):
    head = tl.program_id(0)
    split = tl.program_id(1)
    tile = tl.program_id(2)
    kv_head = head // GROUPS
    row = tile * 16 + tl.arange(0, 16)
    dim = tl.arange(0, DIM)
    key_lane = tl.arange(0, BK)
    length = tl.load(LENGTH)
    count = tl.load(COUNT)
    # Window row w attends key <= length + w; the tile's last window row bounds it.
    window_end = tl.minimum((tile + 1) * 16, ROWS) - PREFIX
    limit = length + tl.maximum(window_end, 0)
    q = tl.load(Q + head * QH + row[:, None] * QR + dim[None, :] * QD,
                (row[:, None] < ROWS), other=0)
    maximum = tl.full((16,), float('-inf'), tl.float32)
    denominator = tl.full((16,), 0, tl.float32)
    numerator = tl.full((16, DIM), 0, tl.float32)
    for start in range(split * CHUNK, tl.minimum((split + 1) * CHUNK, KEYS), BK):
        if start < limit:
            key = start + key_lane
            k = tl.load(K + kv_head * KH + key[None, :] * KR + dim[:, None] * KD,
                        (key[None, :] < KEYS), other=0)
            v = tl.load(V + kv_head * VH + key[:, None] * VR + dim[None, :] * VD,
                        (key[:, None] < KEYS), other=0)
            score = tl.dot(q, k) * SCALE
            prefix = tl.where(row[:, None] < count,
                              key[None, :] < length - count + row[:, None] + 1,
                              key[None, :] == 0)
            window = key[None, :] < length + row[:, None] - PREFIX + 1
            valid = tl.where(row[:, None] < PREFIX, prefix, window)
            score = tl.where(valid & (key[None, :] < KEYS) &
                             (row[:, None] < ROWS), score, float('-inf'))
            block_max = tl.max(score, 1)
            new_max = tl.maximum(maximum, block_max)
            safe_max = tl.where(new_max == float('-inf'), 0., new_max)
            alpha = tl.where(maximum == float('-inf'), 0.,
                             tl.exp2((maximum - safe_max) * 1.4426950408889634))
            probability = tl.exp2((score - safe_max[:, None]) * 1.4426950408889634)
            denominator = denominator * alpha + tl.sum(probability, 1)
            contribution = tl.dot(probability, v.to(tl.float32),
                                  input_precision="tf32x3")
            numerator = numerator * alpha[:, None] + contribution
            maximum = new_max
    offset = (split * HEADS + head) * ROWS + row
    tl.store(M + offset, maximum, row < ROWS)
    tl.store(S + offset, denominator, row < ROWS)
    tl.store(P + offset[:, None] * DIM + dim[None, :], numerator,
             row[:, None] < ROWS)


def merged_splits(keys: int) -> int:
    """Split count: at most four up to 2304 keys, then one split per 256 keys."""
    if keys > 2304:
        return triton.cdiv(keys, 256)
    return min(4, max(1, triton.cdiv(keys, 256)))


def attend_merged(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                  length: torch.Tensor, count: torch.Tensor, scale: float, *,
                  prefix: int) -> torch.Tensor:
    """Attend decoder prefix and window rows [1,H,R,D] in one pass; return [1,R,H,D].

    Rows below ``prefix`` use the decoder count rule.  Window row ``w`` (row
    ``prefix + w``) permits key <= length + w, the window mask's rule.
    """
    if (q.ndim != 4 or k.shape != v.shape or q.shape[0] != 1 or k.shape[0] != 1
            or not 0 < q.shape[-2] <= 32 or q.shape[-1] != k.shape[-1]
            or q.shape[1] % k.shape[1] or q.shape[-1] not in (64, 128, 256)
            or not 0 <= prefix <= q.shape[-2]):
        raise ValueError('unsupported merged length attention layout')
    heads, rows, dim = q.shape[1:]
    keys = k.shape[-2]
    from jetspec.engines.jetspec_plus_plus.kernels.draft_splitkv import enabled, attend as split_attend
    if enabled(q):
        return split_attend(q, k, v, scale, mode='merged', length=length,
                            count=count, prefix=prefix).transpose(1, 2)
    splits = merged_splits(keys)
    chunk = triton.cdiv(triton.cdiv(keys, splits), 64) * 64
    partial = torch.empty((splits, heads, rows, dim), device=q.device, dtype=torch.float32)
    maximum = torch.empty((splits, heads, rows), device=q.device, dtype=torch.float32)
    denominator = torch.empty_like(maximum)
    output = q.new_empty((1, rows, heads, dim))
    _merged_partials[(heads, splits, triton.cdiv(rows, 16))](
        q, k, v, length, count, partial, maximum, denominator,
        q.stride(1), q.stride(2), q.stride(3),
        k.stride(1), k.stride(2), k.stride(3),
        v.stride(1), v.stride(2), v.stride(3),
        heads, heads // k.shape[1], rows, prefix, keys, dim, scale, chunk, 64,
        num_warps=4, enable_fp_fusion=False)
    _merge[(heads, rows)](partial, maximum, denominator, output,
                          output.stride(2), output.stride(1), output.stride(3),
                          heads, rows, dim, splits, triton.next_power_of_2(splits),
                          num_warps=4, enable_fp_fusion=False)
    return output
