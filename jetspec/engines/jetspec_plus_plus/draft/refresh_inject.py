"""Default-off packed N=1 fixed-refresh injection."""
from types import SimpleNamespace
import os

import torch
from torch.nn import functional as F


@torch.inference_mode()
def prepare(head):
    """Pack once after load; retain the original projection parameters and routes."""
    if hasattr(head, '_refresh_inject_kv'):
        return
    attns = [layer.self_attn for layer in head.layers]
    shapes = {(a.config.num_key_value_heads, a.head_dim) for a in attns}
    eps = {a.k_norm.variance_epsilon for a in attns}
    if (not attns or len(shapes) != 1 or len(eps) != 1 or
            any(type(a.k_norm).__name__ != 'Qwen3RMSNorm' for a in attns) or
            type(head.hidden_norm).__name__ != 'Qwen3RMSNorm'):
        raise ValueError('REFRESH_INJECT_LEAN2 requires matching plain Qwen3 K norms')
    from jetspec.engines.jetspec_plus_plus.draft.projections import prepare_context_kv
    # Reuse Seg A's packing without enabling its dispatch or changing its weights.
    packed = prepare_context_kv(SimpleNamespace(layers=head.layers))
    nh, hd = next(iter(shapes))
    if packed[3] != nh * hd or any(a.v_proj.out_features != nh * hd for a in attns):
        raise ValueError('REFRESH_INJECT_LEAN2 requires matching K/V widths')
    head._refresh_inject_kv = packed


def inject(bucket, hidden, khat, reset):
    """Write entries 0..khat (except slot 15, which no query can read)."""
    commit(bucket, project(bucket, hidden, khat), khat, reset)


def project(bucket, hidden, khat):
    """Project eligible rows without changing KV or visibility.

    A later smaller cutoff may consume the same row projections.
    """
    head = bucket.head
    context = hidden.reshape(bucket.R * bucket.B, -1)
    from jetspec.engines.jetspec_plus_plus.serving.presets.budget import budget_type
    projected_khat = budget_type().projected_khat(khat, bucket.B)
    if context.is_cuda:
        from jetspec.engines.jetspec_plus_plus._optional import triton
        global _front, _postprocess
        if _front is None:
            _front, _postprocess = _kernels()
        prepared = torch.empty_like(context, dtype=bucket.dtype)
        _front[(context.shape[0], triton.cdiv(context.shape[1], 512))](
            context, projected_khat, head.adapted_position_embedding.weight, prepared,
            context.shape[1], bucket.B, bucket._refresh_mark, TILE=512,
            num_warps=4, enable_fp_fusion=False)
        from jetspec.engines.jetspec_plus_plus.draft.rows import fused_kernels
        state = fused_kernels().fast_rms_norm(
            head.fc(prepared), head.hidden_norm.weight, head.hidden_norm.variance_epsilon)
    else:
        flat = torch.arange(context.shape[0], device=context.device)
        visible = (flat % bucket.B < (projected_khat + 1).clamp(max=bucket.B - 1)[flat // bucket.B])
        context = torch.where(visible[:, None], context, torch.zeros_like(context)).to(bucket.dtype)
        if bucket._refresh_mark:
            context = context + head.adapted_position_embedding.weight[1]
        state = head.hidden_norm(head.fc(context))
    weight, bias, _, _ = head._refresh_inject_kv
    return F.linear(state, weight, bias)


def commit(bucket, projected, khat, reset):
    """Write only the final cutoff's rows and expose them to the next pass."""
    head = bucket.head
    _, _, norms, width = head._refresh_inject_kv
    nh, hd = head.layers[0].self_attn.config.num_key_value_heads, head.layers[0].self_attn.head_dim
    cos, sin = bucket.qcos[0], bucket.qsin[0]
    fuse_mask = (os.environ.get('JETSPEC_REFRESH_INJECT_MASK_FUSED', '0') == '1'
                 and os.environ.get('JETSPEC_REFRESH_INJECT_LEAN', '0') == '1'
                 and projected.dtype == torch.bfloat16 and bucket.N == 1
                 and bucket.B == 16 and bucket.backend == 'flashinfer'
                 and bucket._refresh_start % 8 == 0)
    mask_bytes = (projected.shape[0] * ((bucket._refresh_start + projected.shape[0]) // 8)
                  if fuse_mask else 0)
    if projected.is_cuda:
        from jetspec.engines.jetspec_plus_plus._optional import triton
        storage = bucket._refresh_inject_storage
        cache_programs = len(head.layers) * projected.shape[0] * nh
        _postprocess[(cache_programs + (mask_bytes + 255) // 256,)](
            projected, norms, cos, sin, storage, bucket.refresh_slots, khat,
            projected.shape[0], nh, hd, bucket.B, projected.stride(0), cos.stride(0),
            storage.stride(0), storage.stride(1), storage.stride(2),
            head.layers[0].self_attn.k_norm.variance_epsilon,
            BLOCK=triton.next_power_of_2(hd),
            MASK=bucket.custom_mask_buffer if fuse_mask else None,
            MASK_START=bucket._refresh_start if fuse_mask else 0,
            MASK_BYTES=mask_bytes, CACHE_PROGRAMS=cache_programs,
            FUSED_MASK=fuse_mask, RESET=reset, num_warps=4, enable_fp_fusion=False)
    else:
        from jetspec.engines.jetspec_plus_plus.draft.rows import _rotary
        flat = torch.arange(projected.shape[0], device=projected.device)
        visible = (flat % bucket.B < (khat + 1).clamp(max=bucket.B - 1)[flat // bucket.B])
        parts = projected.split(width, -1)
        slots = bucket.refresh_slots[visible]
        for i, (layer, cache) in enumerate(zip(head.layers, bucket.kv)):
            key = layer.self_attn.k_norm(parts[2 * i].reshape(-1, nh, hd))
            key = _rotary(key, cos, sin)
            value = parts[2 * i + 1].reshape(-1, nh, hd)
            cache[slots, 0, 0] = key[visible]
            cache[slots, 1, 0] = value[visible]
        if fuse_mask:
            _mask_reference(bucket.custom_mask_buffer, khat, bucket._refresh_start,
                            projected.shape[0], reset)
    if not fuse_mask:
        bucket._inject_mask(khat, projected.shape[0], reset)


from jetspec.engines.jetspec_plus_plus.draft.method.refresh import inject_mask_reference as _mask_reference


_front = _postprocess = None


def _kernels():
    from jetspec.engines.jetspec_plus_plus._optional import triton
    from jetspec.engines.jetspec_plus_plus._optional import tl

    @triton.jit
    def front(source, khat, marker, output, width: tl.constexpr, block: tl.constexpr,
              mark: tl.constexpr, TILE: tl.constexpr):
        token = tl.program_id(0)
        col = tl.program_id(1) * TILE + tl.arange(0, TILE)
        visible = (token % block <= tl.load(khat + token // block)) & (token % block < block - 1)
        value = tl.load(source + token * width + col, (col < width) & visible, other=0)
        dtype = output.dtype.element_ty
        value = value.to(dtype)
        if mark:
            value = (value + tl.load(marker + width + col, col < width, other=0)).to(dtype)
        tl.store(output + token * width + col, value, col < width)

    @triton.jit
    def postprocess(projected, weights, cos, sin, cache, slots, khat,
                    tokens: tl.constexpr, kv_heads: tl.constexpr, head_dim: tl.constexpr,
                    block: tl.constexpr, projected_row_stride: tl.constexpr,
                    cos_row_stride: tl.constexpr, cache_layer_stride: tl.constexpr,
                    cache_row_stride: tl.constexpr, cache_kv_stride: tl.constexpr,
                    eps: tl.constexpr, BLOCK: tl.constexpr,
                    MASK=None, MASK_START: tl.constexpr = 0,
                    MASK_BYTES: tl.constexpr = 0, CACHE_PROGRAMS: tl.constexpr = 0,
                    FUSED_MASK: tl.constexpr = False, RESET: tl.constexpr = False):
        program = tl.program_id(0)
        if not FUSED_MASK or program < CACHE_PROGRAMS:
            head = program % kv_heads
            token = (program // kv_heads) % tokens
            layer = program // (kv_heads * tokens)
            visible = (token % block <= tl.load(khat + token // block)) & (token % block < block - 1)
            if visible:
                # Seg A's arithmetic, including every BF16 rounding boundary.
                col = tl.arange(0, BLOCK)
                mask = col < head_dim
                width = kv_heads * head_dim
                base = token * projected_row_stride + layer * 2 * width + head * head_dim
                value = tl.load(projected + base + col, mask=mask, other=0).to(tl.float32)
                scale = tl.rsqrt(tl.sum(value * value, 0) / head_dim + eps)
                normed = (value * scale).to(tl.bfloat16)
                weight = tl.load(weights + layer * head_dim + col, mask=mask, other=0)
                normed = (normed * weight).to(tl.bfloat16)
                half = head_dim // 2
                partner = tl.where(col < half, col + half, col - half)
                rotated = tl.gather(normed, partner, 0)
                rotated = tl.where(col < half, -rotated, rotated)
                rope = token * cos_row_stride + col
                cosine = tl.load(cos + rope, mask=mask, other=0)
                sine = tl.load(sin + rope, mask=mask, other=0)
                first = (normed * cosine).to(tl.bfloat16)
                second = (rotated * sine).to(tl.bfloat16)
                cache_row = tl.load(slots + token)
                target = (layer * cache_layer_stride + cache_row * cache_row_stride + head * head_dim + col)
                tl.store(cache + target, (first + second).to(tl.bfloat16), mask=mask)
                copied = tl.load(projected + base + width + col, mask=mask, other=0)
                tl.store(cache + target + cache_kv_stride, copied, mask=mask)
        else:
            index = (program - CACHE_PROGRAMS) * 256 + tl.arange(0, 256)
            valid = index < MASK_BYTES
            row_bytes = (MASK_START + tokens) // 8
            query = index // row_bytes
            entry = (index % row_bytes) * 8 - MASK_START
            region = (entry >= 0) & (entry < tokens)
            own = region & (entry // block == query // block)
            cutoff = tl.load(khat + query // block, valid, other=0)
            limit = tl.minimum(tl.minimum(tl.maximum(cutoff, 0), block - 1),
                               query % block - 1)
            count = tl.minimum(tl.maximum(limit - entry % block + 1, 0), 8)
            packed = tl.where(own, (1 << count) - 1, 0).to(tl.uint8)
            prior = tl.load(MASK + index, valid, other=0)
            if RESET:
                prior = tl.where(region, 0, prior).to(tl.uint8)
            tl.store(MASK + index, prior | packed, valid)

    return front, postprocess
