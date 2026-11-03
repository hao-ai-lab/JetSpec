"""Opt-in SM100 adapter query attention and residual norms (draft_fuse_t1).

One Triton launch replaces the C1 query attention: a dense masked form with the
existing float mask, and a device-length form with the c1 visibility rule. The
residual add + RMSNorm of the query blocks runs as one Triton row kernel. Every
entry point falls back to the existing operation off SM100 BF16, so CPU and FP32
callers keep their operations and operands.
"""
from __future__ import annotations

from functools import partial

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # CPU-only test environments have no Triton package.
    triton = tl = None


ARMS = ('dense', 'len', 'fused_len')
# Knob value 1 keeps the per-bucket winner of the in-unit GB200 A/B (five query
# blocks, mask launches included): dense below P 512, the in-repo length kernel
# from P 512. The fused length form is not split-KV and lost at every bucket.
_LEN_FLOOR = 512


def route(value, prefix_bucket):
    """Attention arm for one static prefix bucket, or None for the unchanged path.

    ``dense`` swaps the dense masked kernel below the device-length floor;
    ``len`` lowers that floor from 2048 to 512 (the in-repo length kernel);
    ``fused_len`` lowers it and runs the length form of this kernel.
    """
    if value in (None, '0'):
        return None
    if value == '1':
        return 'len' if prefix_bucket >= _LEN_FLOOR else 'dense'
    if value not in ARMS:
        raise ValueError(f'draft_fuse_t1={value!r}: expected 0, 1 or one of {ARMS}')
    if prefix_bucket >= 2048:
        return 'fused_len' if value == 'fused_len' else 'len'
    if value == 'dense':
        return 'dense'
    return value if prefix_bucket >= _LEN_FLOOR else 'cudnn'


def enabled(q, k, v):
    return (triton is not None and q.device.type == "cuda" and q.dtype == torch.bfloat16
            and k.device == q.device and v.device == q.device
            and k.dtype == q.dtype and v.dtype == q.dtype
            and q.ndim == 4 and q.shape[0] == 1 and q.shape[-1] == 128
            and q.shape[2] in (16, 64) and k.shape == v.shape
            and k.shape[0] == 1 and k.shape[-1] == 128 and q.shape[1] % k.shape[1] == 0
            and 0 < k.shape[2] <= 2048
            and torch.cuda.get_device_capability(q.device) == (10, 0))


def query(q, k, v, mask, scale):
    """Dense masked attention with the existing [1, heads, rows, keys] float mask."""
    if not enabled(q, k, v):
        from jetspec.engines.jetspec_plus_plus.kernels.attn_sm100 import attend
        return attend(q, k, v, mask, scale)
    return _cuda(q, k, v, scale, mask=mask)


def query_length(q, k, v, length, scale, *, mode):
    """Device-length attention: key < length + row (the len_attention c1 rule)."""
    if mode != "c1" or not enabled(q, k, v):
        from jetspec.engines.jetspec_plus_plus.kernels.len_attention import attend
        return attend(q, k, v, length, scale, mode=mode)
    return _cuda(q, k, v, scale, length=length)


def add_norm(update, residual, weight, eps, *, fallback):
    """Return (normalized, state) with state = bf16(update + residual)."""
    from jetspec.engines.jetspec_plus_plus.profiles import kernel_enabled
    if kernel_enabled('draft_fuse_w1', '1'):
        from jetspec.engines.jetspec_plus_plus.kernels.nvfp4.producers import enabled, add_norm as produce
        if enabled(update):
            return produce(update, residual, weight, eps)
    if (triton is None or update.device.type != "cuda" or update.dtype != torch.bfloat16
            or update.ndim != 2 or tuple(update.shape) not in ((16, 4096), (64, 4096))
            or residual.shape != update.shape or weight.shape != (4096,)
            or any(t.device != update.device or t.dtype != update.dtype or not t.is_contiguous()
                   for t in (update, residual, weight))
            or torch.cuda.get_device_capability(update.device) != (10, 0)):
        return fallback(update, residual, weight, eps)
    from jetspec.engines.jetspec_plus_plus.kernels.c1_fused import _fast_add_rms_norm_kernel

    normalized, state = torch.empty_like(update), torch.empty_like(residual)
    _fast_add_rms_norm_kernel[(update.shape[0],)](
        update, residual, weight, normalized, state, update.shape[0], 4096, eps, 4096,
        num_warps=8, enable_fp_fusion=False)
    return normalized, state


def add_norm_operation(fallback):
    return partial(add_norm, fallback=fallback)


def _cuda(q, k, v, scale, *, mask=None, length=None):
    from jetspec.engines.jetspec_plus_plus.kernels.c1_fused import attention_output
    from jetspec.engines.jetspec_plus_plus.kernels.draft_splitkv import enabled, attend

    if enabled(q):
        return attend(q, k, v, scale, mode='dense' if mask is not None else 'c1',
                      mask=mask, length=length)

    rows, keys = q.shape[2], k.shape[2]
    output = attention_output(q)
    mask_strides = (0, 0, 0) if mask is None else mask.stride()[1:]
    _attention[(q.shape[1], triton.cdiv(rows, 16))](
        q, k, v, q if mask is None else mask, q if length is None else length, output,
        *q.stride()[1:], *k.stride()[1:], *v.stride()[1:], *mask_strides, *output.stride()[1:],
        rows, keys, q.shape[1] // k.shape[1], scale,
        BK=64, LIVE_LENGTH=length is not None,
        num_warps=4, num_stages=2, enable_fp_fusion=False)
    return output


if triton is not None:
    @triton.jit
    def _attention(Q, K, V, Mask, Length, Out,
                   qh: tl.constexpr, qr: tl.constexpr, qd: tl.constexpr,
                   kh: tl.constexpr, kk: tl.constexpr, kd: tl.constexpr,
                   vh: tl.constexpr, vk: tl.constexpr, vd: tl.constexpr,
                   mh: tl.constexpr, mr: tl.constexpr, mk: tl.constexpr,
                   oh: tl.constexpr, orun: tl.constexpr, od: tl.constexpr,
                   rows: tl.constexpr, keys: tl.constexpr, groups: tl.constexpr,
                   scale: tl.constexpr, BK: tl.constexpr, LIVE_LENGTH: tl.constexpr):
        h, rb = tl.program_id(0), tl.program_id(1)
        r = rb * 16 + tl.arange(0, 16)
        d = tl.arange(0, 128)
        query = tl.load(Q + h * qh + r[:, None] * qr + d[None, :] * qd,
                        r[:, None] < rows, other=0)
        peak = tl.full((16,), -float("inf"), tl.float32)
        denom = tl.full((16,), 0., tl.float32)
        accum = tl.full((16, 128), 0., tl.float32)
        if LIVE_LENGTH:
            length = tl.load(Length)
        for split in range(tl.cdiv(keys, BK)):
            s = split * BK + tl.arange(0, BK)
            key = tl.load(K + (h // groups) * kh + s[:, None] * kk + d[None, :] * kd,
                          s[:, None] < keys, other=0)
            value = tl.load(V + (h // groups) * vh + s[:, None] * vk + d[None, :] * vd,
                            s[:, None] < keys, other=0)
            scores = tl.dot(query, tl.trans(key)) * scale
            if LIVE_LENGTH:
                scores = tl.where((s[None, :] < length + r[:, None]) & (s[None, :] < keys),
                                  scores, -float("inf"))
            else:
                bias = tl.load(Mask + h * mh + r[:, None] * mr + s[None, :] * mk,
                               (r[:, None] < rows) & (s[None, :] < keys), other=-float("inf"))
                scores = tl.where((bias > -10000) & (s[None, :] < keys), scores + bias, -float("inf"))
            local_peak = tl.max(scores, 1)
            new_peak = tl.maximum(peak, local_peak)
            safe_peak = tl.where(new_peak == -float("inf"), 0., new_peak)
            if LIVE_LENGTH:
                old_scale = tl.exp2((peak - safe_peak) * 1.4426950408889634)
                weights = tl.exp2((scores - safe_peak[:, None]) * 1.4426950408889634)
                contribution = tl.dot(weights, value.to(tl.float32), input_precision="tf32x3")
            else:
                old_scale = tl.exp(peak - safe_peak)
                weights = tl.exp(scores - safe_peak[:, None])
                contribution = tl.dot(weights.to(value.dtype), value)
            accum = accum * old_scale[:, None] + contribution
            denom = denom * old_scale + tl.sum(weights, 1)
            peak = new_peak
        result = accum / tl.where(denom == 0, 1., denom)[:, None]
        tl.store(Out + h * oh + r[:, None] * orun + d[None, :] * od,
                 result, r[:, None] < rows)
