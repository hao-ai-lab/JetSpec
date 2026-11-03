"""Device kernels for draft/rows.py."""
from __future__ import annotations

from types import SimpleNamespace

try:
    import triton
    import triton.language as tl
except ImportError:  # Triton is Linux/CUDA-only in the pinned environment.
    triton = None
    tl = None


if triton is not None:

    @triton.jit(do_not_specialize=["tokens"])
    def _context_kv_postprocess_kernel(projected, weights, cos, sin, cache, slots,
                                       tokens, layers: tl.constexpr, kv_heads: tl.constexpr,
                                       head_dim: tl.constexpr, projected_row_stride: tl.constexpr,
                                       cos_row_stride: tl.constexpr,
                                       cache_layer_stride: tl.constexpr,
                                       cache_row_stride: tl.constexpr,
                                       cache_kv_stride: tl.constexpr,
                                       cache_start: tl.constexpr, direct_span: tl.constexpr,
                                       eps: tl.constexpr,
                                       BLOCK: tl.constexpr):
        program = tl.program_id(0)
        head = program % kv_heads
        token = (program // kv_heads) % tokens
        layer = program // (kv_heads * tokens)
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
        cache_row = cache_start + token if direct_span else tl.load(slots + token)
        target = (layer * cache_layer_stride + cache_row * cache_row_stride
                  + head * head_dim + col)
        tl.store(cache + target, (first + second).to(tl.bfloat16), mask=mask)
        copied = tl.load(projected + base + width + col, mask=mask, other=0)
        tl.store(cache + target + cache_kv_stride, copied, mask=mask)

    @triton.jit(do_not_specialize=["tokens"])
    def _kv_rotary_into_cache_kernel(normalized_key_ptr, projected_value_ptr,
                                     cos_ptr, sin_ptr, key_out_ptr, value_out_ptr,
                                     tokens, kv_heads: tl.constexpr,
                                     groups: tl.constexpr, head_dim: tl.constexpr,
                                     in_token_stride: tl.constexpr,
                                     out_token_stride: tl.constexpr,
                                     out_head_stride: tl.constexpr,
                                     BLOCK: tl.constexpr):
        # The certified kv rotary (c1_fused._kv_rotary_repeat_kernel) arithmetic,
        # writing each rotated key and copied value straight into a strided
        # cache slice instead of a fresh [heads, tokens, head_dim] buffer.
        program = tl.program_id(0)
        token = program % tokens
        output_head = program // tokens
        input_head = output_head // groups
        offsets = tl.arange(0, BLOCK)
        mask = offsets < head_dim
        key_base = token * in_token_stride + input_head * head_dim
        normalized = tl.load(normalized_key_ptr + key_base + offsets, mask=mask, other=0.0)
        half = head_dim // 2
        rotated_offsets = tl.where(offsets < half, offsets + half, offsets - half)
        rotated = tl.load(normalized_key_ptr + key_base + rotated_offsets, mask=mask, other=0.0)
        rotated = tl.where(offsets < half, -rotated, rotated)
        rope_base = token * head_dim
        cos = tl.load(cos_ptr + rope_base + offsets, mask=mask, other=0.0)
        sin = tl.load(sin_ptr + rope_base + offsets, mask=mask, other=0.0)
        output_base = token * out_token_stride + output_head * out_head_stride
        first = (normalized * cos).to(tl.bfloat16)
        second = (rotated * sin).to(tl.bfloat16)
        tl.store(key_out_ptr + output_base + offsets, (first + second).to(tl.bfloat16), mask=mask)
        value = tl.load(projected_value_ptr + key_base + offsets, mask=mask, other=0.0)
        tl.store(value_out_ptr + output_base + offsets, value, mask=mask)

    _kv_rotary_into_cache_kernel._jetspec_runtime_count_indices = (6,)


    @triton.jit
    def _kv_rotary_into_slots_kernel(normalized_key_ptr, projected_value_ptr,
                                     cos_ptr, sin_ptr, key_out_ptr, value_out_ptr,
                                     slots_ptr, tokens: tl.constexpr,
                                     kv_heads: tl.constexpr, head_dim: tl.constexpr,
                                     in_token_stride: tl.constexpr,
                                     out_token_stride: tl.constexpr,
                                     out_head_stride: tl.constexpr,
                                     BLOCK: tl.constexpr):
        program = tl.program_id(0)
        token = program % tokens
        head = program // tokens
        col = tl.arange(0, BLOCK)
        valid = col < head_dim
        source = token * in_token_stride + head * head_dim
        normalized = tl.load(normalized_key_ptr + source + col, valid, other=0.0)
        half = head_dim // 2
        partner = tl.where(col < half, col + half, col - half)
        rotated = tl.load(normalized_key_ptr + source + partner, valid, other=0.0)
        rotated = tl.where(col < half, -rotated, rotated)
        rope = token * head_dim + col
        cosine = tl.load(cos_ptr + rope, valid, other=0.0)
        sine = tl.load(sin_ptr + rope, valid, other=0.0)
        first = (normalized * cosine).to(tl.bfloat16)
        second = (rotated * sine).to(tl.bfloat16)
        slot = tl.load(slots_ptr + token)
        destination = slot * out_token_stride + head * out_head_stride + col
        tl.store(key_out_ptr + destination, (first + second).to(tl.bfloat16), valid)
        value = tl.load(projected_value_ptr + source + col, valid, other=0.0)
        tl.store(value_out_ptr + destination, value, valid)


    @triton.jit
    def _top1_probability_kernel(logits_ptr, out_ptr, vocab, row_stride, slot_stride, slots,
                                 BLOCK: tl.constexpr):
        # max softmax probability of one [vocab] row: 1 / sum(exp(x - max)) in fp32 (the
        # softmax's own value at its argmax), reading the bf16 row twice
        program = tl.program_id(0)
        base = logits_ptr + (program // slots).to(tl.int64) * row_stride \
            + (program % slots).to(tl.int64) * slot_stride
        offsets = tl.arange(0, BLOCK)
        top = tl.full([BLOCK], float('-inf'), tl.float32)
        for start in range(0, vocab, BLOCK):
            x = tl.load(base + start + offsets, mask=start + offsets < vocab,
                        other=float('-inf')).to(tl.float32)
            top = tl.maximum(top, x)
        peak = tl.max(top, 0)
        total = tl.zeros([BLOCK], tl.float32)
        for start in range(0, vocab, BLOCK):
            x = tl.load(base + start + offsets, mask=start + offsets < vocab,
                        other=float('-inf')).to(tl.float32)
            total += tl.exp(x - peak)
        tl.store(out_ptr + program, 1.0 / tl.sum(total, 0))


    @triton.jit
    def _kv_rotary_into_cache_at_kernel(normalized_key_ptr, projected_value_ptr,
                                        cos_ptr, sin_ptr, key_out_ptr, value_out_ptr, start_ptr,
                                        tokens: tl.constexpr, kv_heads: tl.constexpr,
                                        groups: tl.constexpr, head_dim: tl.constexpr,
                                        in_token_stride: tl.constexpr,
                                        out_token_stride: tl.constexpr,
                                        out_head_stride: tl.constexpr,
                                        BLOCK: tl.constexpr):
        # _kv_rotary_into_cache_kernel's arithmetic, line for line; the output rows
        # start at the cache slot held in device memory at start_ptr (a graph replays
        # it at the live length, JETSPEC_C1_GRAPH_LAYOUT=live).
        program = tl.program_id(0)
        token = program % tokens
        output_head = program // tokens
        input_head = output_head // groups
        offsets = tl.arange(0, BLOCK)
        mask = offsets < head_dim
        key_base = token * in_token_stride + input_head * head_dim
        normalized = tl.load(normalized_key_ptr + key_base + offsets, mask=mask, other=0.0)
        half = head_dim // 2
        rotated_offsets = tl.where(offsets < half, offsets + half, offsets - half)
        rotated = tl.load(normalized_key_ptr + key_base + rotated_offsets, mask=mask, other=0.0)
        rotated = tl.where(offsets < half, -rotated, rotated)
        rope_base = token * head_dim
        cos = tl.load(cos_ptr + rope_base + offsets, mask=mask, other=0.0)
        sin = tl.load(sin_ptr + rope_base + offsets, mask=mask, other=0.0)
        start = tl.load(start_ptr)
        output_base = (start + token) * out_token_stride + output_head * out_head_stride
        first = (normalized * cos).to(tl.bfloat16)
        second = (rotated * sin).to(tl.bfloat16)
        tl.store(key_out_ptr + output_base + offsets, (first + second).to(tl.bfloat16), mask=mask)
        value = tl.load(projected_value_ptr + key_base + offsets, mask=mask, other=0.0)
        tl.store(value_out_ptr + output_base + offsets, value, mask=mask)

