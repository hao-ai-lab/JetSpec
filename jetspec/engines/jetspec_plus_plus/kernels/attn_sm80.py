"""Device-masked split-KV attention for short adapter queries on Ampere."""

from __future__ import annotations

import os

import torch


def enabled(device: torch.device, dtype: torch.dtype) -> bool:
    if os.environ.get("JETSPEC_ATTN_SM80", "0") != "1":
        return False
    if device.type != "cuda":
        raise ValueError("JETSPEC_ATTN_SM80 requires CUDA")
    if dtype != torch.bfloat16:
        raise ValueError("JETSPEC_ATTN_SM80 requires bf16")
    capability = torch.cuda.get_device_capability(device)
    return capability[0] == 8


def attend(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
           mask: torch.Tensor, scale: float) -> torch.Tensor:
    from jetspec.engines.jetspec_plus_plus._optional import triton

    global _partials, _merge
    if _partials is None:
        _partials, _merge = _kernels()

    if (q.ndim != 4 or q.shape[0] != 1 or q.shape[-1] != 128
            or k.shape != v.shape or k.shape[:2] != q.shape[:2]
            or k.shape[-1] != 128 or mask.shape != (*q.shape[:3], k.shape[2])):
        raise ValueError("JETSPEC_ATTN_SM80 requires B=1, equal heads, D=128, expanded mask")
    heads, rows, keys = q.shape[1], q.shape[2], k.shape[2]
    splits = triton.cdiv(keys, 128)
    partial = torch.empty((heads, rows, splits, 128), device=q.device, dtype=torch.float32)
    lse = torch.empty((heads, rows, splits), device=q.device, dtype=torch.float32)
    output = torch.empty_like(q)
    _partials[(heads, triton.cdiv(rows, 16), splits)](
        q, k, v, mask, partial, lse,
        *q.stride()[1:], *k.stride()[1:], *v.stride()[1:], *mask.stride()[1:],
        rows, keys, splits, scale, num_warps=4)
    _merge[(heads, triton.cdiv(rows, 16))](
        partial, lse, output, *output.stride()[1:], rows, splits,
        SPLITS=triton.next_power_of_2(splits), num_warps=4)
    return output


def _kernels():
    from jetspec.engines.jetspec_plus_plus._optional import triton
    from jetspec.engines.jetspec_plus_plus._optional import tl

    @triton.jit
    def partials(Q, K, V, M, P, L,
                 qh: tl.constexpr, qr: tl.constexpr, qd: tl.constexpr,
                 kh: tl.constexpr, kk: tl.constexpr, kd: tl.constexpr,
                 vh: tl.constexpr, vk: tl.constexpr, vd: tl.constexpr,
                 mh: tl.constexpr, mr: tl.constexpr, mk: tl.constexpr,
                 rows: tl.constexpr, keys: tl.constexpr, splits: tl.constexpr,
                 scale: tl.constexpr):
        h, rb, split = tl.program_id(0), tl.program_id(1), tl.program_id(2)
        r = rb * 16 + tl.arange(0, 16)
        s = split * 128 + tl.arange(0, 128)
        d = tl.arange(0, 128)
        query = tl.load(Q + h * qh + r[:, None] * qr + d[None, :] * qd,
                        r[:, None] < rows, other=0)
        key = tl.load(K + h * kh + s[:, None] * kk + d[None, :] * kd,
                      s[:, None] < keys, other=0)
        value = tl.load(V + h * vh + s[:, None] * vk + d[None, :] * vd,
                        s[:, None] < keys, other=0)
        bias = tl.load(M + h * mh + r[:, None] * mr + s[None, :] * mk,
                       (r[:, None] < rows) & (s[None, :] < keys), other=-float("inf"))
        scores = tl.dot(query, tl.trans(key)) * scale
        scores = tl.where((bias > -10000) & (s[None, :] < keys), scores + bias, -float("inf"))
        peak = tl.max(scores, 1)
        weights = tl.exp(scores - tl.where(peak == -float("inf"), 0., peak)[:, None])
        denom = tl.sum(weights, 1)
        numerator = tl.dot(weights.to(value.dtype), value)
        result = numerator / tl.where(denom == 0, 1., denom)[:, None]
        tl.store(P + (h * rows + r[:, None]) * splits * 128 + split * 128 + d[None, :],
                 result, r[:, None] < rows)
        tl.store(L + (h * rows + r) * splits + split,
                 tl.where(denom == 0, -float("inf"), peak + tl.log(denom)), r < rows)

    @triton.jit
    def merge(P, L, O, oh: tl.constexpr, orun: tl.constexpr, od: tl.constexpr,
              rows: tl.constexpr, splits: tl.constexpr, SPLITS: tl.constexpr):
        h, rb = tl.program_id(0), tl.program_id(1)
        r = rb * 16 + tl.arange(0, 16)
        s = tl.arange(0, SPLITS)
        d = tl.arange(0, 128)
        logsum = tl.load(L + (h * rows + r[:, None]) * splits + s[None, :],
                         (r[:, None] < rows) & (s[None, :] < splits), other=-float("inf"))
        peak = tl.max(logsum, 1)
        weight = tl.exp(logsum - tl.where(peak == -float("inf"), 0., peak)[:, None])
        weight = weight / tl.where(tl.sum(weight, 1) == 0, 1., tl.sum(weight, 1))[:, None]
        accum = tl.full((16, 128), 0., tl.float32)
        for i in range(SPLITS):
            part = tl.load(P + (h * rows + r[:, None]) * splits * 128 + i * 128 + d[None, :],
                           (r[:, None] < rows) & (i < splits), other=0)
            accum += part * tl.sum(tl.where(s == i, weight, 0.), 1)[:, None]
        tl.store(O + h * oh + r[:, None] * orun + d[None, :] * od,
                 accum, r[:, None] < rows)

    return partials, merge


_partials = _merge = None
