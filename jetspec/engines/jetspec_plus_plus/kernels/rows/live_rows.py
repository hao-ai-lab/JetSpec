"""Device kernels for draft/rows_live.py."""
from __future__ import annotations

from types import SimpleNamespace

try:
    import triton
    import triton.language as tl
except ImportError:  # Triton is Linux/CUDA-only in the pinned environment.
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _q_rotary_scatter_kernel(normalized_ptr, cos_ptr, sin_ptr, output_ptr, index_ptr,
                                 heads: tl.constexpr, head_dim: tl.constexpr, BLOCK: tl.constexpr):
        # c1_fused._q_rotary_kernel at one token per row (the same loads and arithmetic), each
        # row's heads stored at bucket token index_ptr[row] of the attention's query layout
        program = tl.program_id(0)
        head = program % heads
        row = program // heads
        offsets = tl.arange(0, BLOCK)
        mask = offsets < head_dim
        input_base = (row * heads + head) * head_dim
        normalized = tl.load(normalized_ptr + input_base + offsets, mask=mask, other=0.0)
        half = head_dim // 2
        rotated_offsets = tl.where(offsets < half, offsets + half, offsets - half)
        rotated = tl.load(normalized_ptr + input_base + rotated_offsets, mask=mask, other=0.0)
        rotated = tl.where(offsets < half, -rotated, rotated)
        rope_base = row * head_dim
        cos = tl.load(cos_ptr + rope_base + offsets, mask=mask, other=0.0)
        sin = tl.load(sin_ptr + rope_base + offsets, mask=mask, other=0.0)
        output_base = (tl.load(index_ptr + row) * heads + head) * head_dim
        first = (normalized * cos).to(tl.bfloat16)
        second = (rotated * sin).to(tl.bfloat16)
        tl.store(output_ptr + output_base + offsets, (first + second).to(tl.bfloat16), mask=mask)


    @triton.jit
    def _kv_rotary_slots_kernel(normalized_key_ptr, projected_value_ptr, cos_ptr, sin_ptr, cache_ptr,
                                slots_ptr, slot_stride, kv_stride, head_stride,
                                kv_heads: tl.constexpr, head_dim: tl.constexpr, BLOCK: tl.constexpr):
        # c1_fused._kv_rotary_repeat_kernel at one token per row and one group (the same loads and
        # arithmetic), each row's key and value stored at its K/V slot slots_ptr[row] of the cache
        program = tl.program_id(0)
        head = program % kv_heads
        row = program // kv_heads
        offsets = tl.arange(0, BLOCK)
        mask = offsets < head_dim
        key_base = row * (kv_heads * head_dim) + head * head_dim
        normalized = tl.load(normalized_key_ptr + key_base + offsets, mask=mask, other=0.0)
        half = head_dim // 2
        rotated_offsets = tl.where(offsets < half, offsets + half, offsets - half)
        rotated = tl.load(normalized_key_ptr + key_base + rotated_offsets, mask=mask, other=0.0)
        rotated = tl.where(offsets < half, -rotated, rotated)
        rope_base = row * head_dim
        cos = tl.load(cos_ptr + rope_base + offsets, mask=mask, other=0.0)
        sin = tl.load(sin_ptr + rope_base + offsets, mask=mask, other=0.0)
        first = (normalized * cos).to(tl.bfloat16)
        second = (rotated * sin).to(tl.bfloat16)
        target = cache_ptr + tl.load(slots_ptr + row) * slot_stride + head * head_stride
        tl.store(target + offsets, (first + second).to(tl.bfloat16), mask=mask)
        value = tl.load(projected_value_ptr + key_base + offsets, mask=mask, other=0.0)
        tl.store(target + kv_stride + offsets, value, mask=mask)


    @triton.jit
    def _compact_rows_kernel(order_ptr, qcos_ptr, qsin_ptr, noise_slots_ptr, preceding_ptr, first_ptr,
                             keys_ptr, active_ptr, token_map_ptr, cos_ptr, sin_ptr, kv_slots_ptr,
                             preceding_out, first_out, keys_out, active_out, first_stride,
                             B: tl.constexpr, HD: tl.constexpr, BLOCK_B: tl.constexpr,
                             BLOCK_HD: tl.constexpr):
        # compact row -> its bucket row: the tokens' bucket positions, rotary rows and K/V slots,
        # and the row's anchor, first-pass ids, key and active flag (copies only)
        row = tl.program_id(0).to(tl.int64)
        source = tl.load(order_ptr + row)
        slot = tl.arange(0, BLOCK_B)
        in_block = slot < B
        token = source * B + slot
        target = row * B + slot
        tl.store(token_map_ptr + target, token, mask=in_block)
        tl.store(kv_slots_ptr + target, tl.load(noise_slots_ptr + token, mask=in_block, other=0),
                 mask=in_block)
        dim = tl.arange(0, BLOCK_HD)
        inside = in_block[:, None] & (dim[None, :] < HD)
        tl.store(cos_ptr + target[:, None] * HD + dim[None, :],
                 tl.load(qcos_ptr + token[:, None] * HD + dim[None, :], mask=inside), mask=inside)
        tl.store(sin_ptr + target[:, None] * HD + dim[None, :],
                 tl.load(qsin_ptr + token[:, None] * HD + dim[None, :], mask=inside), mask=inside)
        tl.store(first_out + row * B + slot, tl.load(first_ptr + source * first_stride + slot,
                                                    mask=in_block), mask=in_block)
        tl.store(preceding_out + row, tl.load(preceding_ptr + source))
        pair = tl.arange(0, 2)
        tl.store(keys_out + row * 2 + pair, tl.load(keys_ptr + source * 2 + pair))
        tl.store(active_out + row, tl.load(active_ptr + source))

