"""Default-off packed N=1 fixed-refresh injection."""

from jetspec.engines.jetspec_plus_plus.profiles import kernel_enabled, kernel_value
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
    from jetspec.engines.jetspec_plus_plus.kernels.projections import prepare_context_kv
    # Reuse Seg A's packing without enabling its dispatch or changing its weights.
    packed = prepare_context_kv(SimpleNamespace(layers=head.layers))
    nh, hd = next(iter(shapes))
    if packed[3] != nh * hd or any(a.v_proj.out_features != nh * hd for a in attns):
        raise ValueError('REFRESH_INJECT_LEAN2 requires matching K/V widths')
    head._refresh_inject_kv = packed


def inject(bucket, hidden, khat, reset, run_config=None):
    """Write entries 0..khat (except slot 15, which no query can read)."""
    commit(bucket, project(bucket, hidden, khat, run_config=run_config), khat, reset)


def project(bucket, hidden, khat, run_config=None):
    """Project eligible rows without changing KV or visibility.

    A later smaller cutoff may consume the same row projections.
    """
    head = bucket.head
    context = hidden.reshape(bucket.R * bucket.B, -1)
    from jetspec.engines.jetspec_plus_plus.draft.serve_policy import projected_khat as method_projection
    projected_khat = method_projection(khat, bucket.B, run_config=run_config)
    if context.is_cuda:
        import triton
        global _front, _postprocess
        if _front is None:
            _front, _postprocess = _kernels()
        prepared = torch.empty_like(context, dtype=bucket.dtype)
        _front[(context.shape[0], triton.cdiv(context.shape[1], 512))](
            context, projected_khat, head.adapted_position_embedding.weight, prepared,
            context.shape[1], bucket.B, bucket._refresh_mark, TILE=512,
            num_warps=4, enable_fp_fusion=False)
        from jetspec.engines.jetspec_plus_plus.jetspec.rows import fused_kernels
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
    from jetspec.engines.jetspec_plus_plus.profiles import kernel_enabled
    head = bucket.head
    _, _, norms, width = head._refresh_inject_kv
    nh, hd = head.layers[0].self_attn.config.num_key_value_heads, head.layers[0].self_attn.head_dim
    cos, sin = bucket.qcos[0], bucket.qsin[0]
    fuse_mask = (kernel_enabled('refresh_inject_mask_fused', '1')
                 and kernel_enabled('refresh_inject_lean', '1')
                 and projected.dtype == torch.bfloat16 and bucket.N == 1
                 and bucket.B == 16 and bucket.backend == 'flashinfer'
                 and bucket._refresh_start % 8 == 0)
    mask_bytes = (projected.shape[0] * ((bucket._refresh_start + projected.shape[0]) // 8)
                  if fuse_mask else 0)
    if projected.is_cuda:
        import triton
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
        from jetspec.engines.jetspec_plus_plus.jetspec.rows import _rotary
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


from jetspec.engines.jetspec_plus_plus.kernels.refresh_inject import _kernels
