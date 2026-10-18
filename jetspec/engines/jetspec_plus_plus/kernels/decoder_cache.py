"""Opt-in decoder K norm, rotary and slot write in one BF16 launch."""
import torch
from torch.nn import functional as F


def project_into_cache(layer, fused, hidden, pending, rope, cache, prefix_slots, window_slots):
    from jetspec.engines.jetspec_plus_plus.models.decoder_head import decoder_lean_project
    source = hidden if pending is None else pending[0]
    if not source.is_cuda or source.dtype != torch.bfloat16:
        query, key, value, residual = decoder_lean_project(layer, fused, hidden, pending, rope)
        for target, projected in zip(cache, (key, value)):
            target.index_copy_(2, prefix_slots, projected[..., :prefix_slots.numel(), :])
            target.index_copy_(2, window_slots, projected[..., prefix_slots.numel():, :])
        return query, residual
    import triton
    attn = layer.self_attn
    if pending is None:
        normed = fused.fast_rms_norm(hidden, layer.input_layernorm.weight,
                                     layer.input_layernorm.variance_epsilon)
    else:
        normed, hidden = fused.fast_add_rms_norm(
            *pending, layer.input_layernorm.weight, layer.input_layernorm.variance_epsilon)
    weight, bias, widths = attn._decoder_qkv
    from jetspec.engines.jetspec_plus_plus.models.target_linear import packed_draft_linear
    qp, kp, vp = packed_draft_linear(normed, weight, bias).split(widths, -1)
    tokens, dim = source.shape[1], attn.head_dim
    cos, sin = rope
    from jetspec.engines.jetspec_plus_plus.profiles import kernel_enabled
    if kernel_enabled('draft_fuse_t12', '1'):
        from jetspec.engines.jetspec_plus_plus.kernels.decoder_t12 import qkv_postprocess
        query, _, _ = qkv_postprocess(
            fused, qp, kp, vp, attn.q_norm.weight, attn.k_norm.weight, cos, sin,
            attn.q_norm.variance_epsilon, attn.k_norm.variance_epsilon,
            widths[0] // dim, widths[1] // dim, dim,
            cache=cache, prefix_slots=prefix_slots, window_slots=window_slots)
        return query, hidden
    query = fused.jpass_q_norm_rotary(
        qp.view(tokens, -1), attn.q_norm.weight, cos[0], sin[0],
        attn.q_norm.variance_epsilon, widths[0] // dim, dim).transpose(0, 1)[None]
    global _write
    if _write is None:
        _write = _kernel()
    _write[(tokens, widths[1] // dim)](
        kp, vp, attn.k_norm.weight, cos, sin, *cache, prefix_slots, window_slots,
        prefix_slots.numel(), dim, kp.stride(1), vp.stride(1), cos.stride(1),
        cache[0].stride(1), cache[0].stride(2), cache[1].stride(1), cache[1].stride(2),
        attn.k_norm.variance_epsilon, BLOCK=triton.next_power_of_2(dim),
        num_warps=4, enable_fp_fusion=False)
    return query, hidden


_write = None


def _kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def write(key, value, weight, cos, sin, key_out, value_out, prefix, window,
              prefix_rows: tl.constexpr, dim: tl.constexpr,
              key_stride: tl.constexpr, value_stride: tl.constexpr, rope_stride: tl.constexpr,
              key_head: tl.constexpr, key_row: tl.constexpr,
              value_head: tl.constexpr, value_row: tl.constexpr,
              eps: tl.constexpr, BLOCK: tl.constexpr):
        row, head = tl.program_id(0), tl.program_id(1)
        col = tl.arange(0, BLOCK)
        mask = col < dim
        source = tl.load(key + row * key_stride + head * dim + col, mask, 0).to(tl.float32)
        rms = tl.rsqrt(tl.sum(source * source, 0) / dim + eps)
        normed = (source * rms).to(tl.bfloat16)
        normed = (normed * tl.load(weight + col, mask, 0)).to(tl.bfloat16)
        partner = tl.where(col < dim // 2, col + dim // 2, col - dim // 2)
        rotated = tl.gather(normed, partner, 0)
        rotated = tl.where(col < dim // 2, -rotated, rotated)
        first = (normed * tl.load(cos + row * rope_stride + col, mask, 0)).to(tl.bfloat16)
        second = (rotated * tl.load(sin + row * rope_stride + col, mask, 0)).to(tl.bfloat16)
        if row < prefix_rows:
            slot = tl.load(prefix + row)
        else:
            slot = tl.load(window + row - prefix_rows)
        tl.store(key_out + head * key_head + slot * key_row + col,
                 (first + second).to(tl.bfloat16), mask)
        copied = tl.load(value + row * value_stride + head * dim + col, mask, 0)
        tl.store(value_out + head * value_head + slot * value_row + col, copied, mask)

    return write
