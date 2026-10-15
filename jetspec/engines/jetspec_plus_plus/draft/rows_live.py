"""JETSPEC_BATCH_ROWS_LIVE: the batched draft graph's Jacobi passes compute the live rows only.

The graphed batch (JETSPEC_BATCH_DRAFT_GRAPH=1) runs its rows in one flat bucket: N padded requests
x H rows (H = the stage's row capacity), one attention group per request.  A round publishes
min(requested, cap) of those rows per request (at least one row is drafted: the private row of an
empty round), so a fixed-H pass spends its per-token work on dead rows.

A compact pass keeps the bucket and changes only where the per-token work runs.  The bucket's own
state is untouched: its context K/V, its geometry (rotary rows and the packed mask) and its
attention plan.  The pass takes R rows: the round's live rows packed request-major, then dead rows
of the bucket up to R (the live count rounded up to a multiple of the row quantum, so a few
captured passes serve every round).  Per layer it runs the norms and the q/k/v/o and MLP
projections on the R rows only, writes each row's K/V into that row's own block slots of the
bucket, places the queries at their rows in the bucket's query layout, runs the bucket's attention
call unchanged and gathers the R rows' outputs back.  The lm_head then runs on R rows.

A live row sees exactly what it sees in the full pass: the same context K/V, its own block's K/V,
the same mask and plan (attention is per query row, and a masked key adds an exact zero).  Every
per-token kernel is the full pass's own.  Only the GEMM row count changes; the GPU check compares
the live rows' logits with the full pass's.  The slots of rows the pass does not write keep finite
values from earlier passes (the bucket's K/V is zeroed once, before its first compact pass), which
the mask hides from every live row.
"""
from __future__ import annotations

import os
from types import SimpleNamespace

import torch

from jetspec.engines.jetspec_plus_plus.jetspec.rows import RowDraftHead
from jetspec.engines.jetspec_plus_plus.jetspec.rows import _rotary
from jetspec.engines.jetspec_plus_plus.jetspec.rows import fused_kernels
from jetspec.engines.jetspec_plus_plus.jetspec.rows import rows_pass_fused
from jetspec.engines.jetspec_plus_plus.jetspec.rows import sdpa_attention_forward

try:
    import triton
    import triton.language as tl
except ImportError:  # Triton is Linux/CUDA-only in the pinned environment.
    triton = None
    tl = None












class SharedOutLinear(torch.nn.Module):
    """``linear`` writing its output into one static buffer: every compact pass of a batch
    bucket shares that buffer for its logits (one pass runs at a time and its logits are read
    before the next replay), so the captured passes do not each keep a full-vocabulary output.
    ``torch.mm(..., out=)`` is the matmul ``F.linear`` runs for a 2-D input without bias."""

    def __init__(self, linear, rows):
        super().__init__()
        self.weight = linear.weight
        self.bias = linear.bias
        self.out = linear.weight.new_empty(int(rows), linear.weight.shape[0])

    def forward(self, x):
        out = self.out[:x.shape[0]]
        if self.bias is None:
            return torch.mm(x, self.weight.t(), out=out)
        return torch.addmm(self.bias, x, self.weight.t(), out=out)


class _DrafterView:
    """The drafter with another lm_head (BucketPass.step reads ``drafter.lm_head``)."""

    def __init__(self, drafter, lm_head):
        self.__dict__['_drafter'] = drafter
        self.__dict__['lm_head'] = lm_head

    def __getattr__(self, name):
        return getattr(self.__dict__['_drafter'], name)


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


def kv_rotary_into_slots(fused, attn, source, cos, sin, cache, slots, projected=None):
    """RowDraftHead._kv_into_cache's scattered-slot path (the certified K/V postprocess, then two
    index puts) as one kernel writing each row's key and value into its slot: the same norm kernel
    and rotary arithmetic, so the same cache values."""
    rows = source.shape[0]
    nkv, hd = attn.config.num_key_value_heads, attn.head_dim
    if (triton is None or source.device.type != 'cuda' or cache.dtype != torch.bfloat16
            or cache.stride(-1) != 1 or cache.shape[2] != 1):
        RowDraftHead._kv_into_cache(fused, attn, source, cos, sin, cache, slots, None, projected)
        return
    key, value = projected if projected is not None else (attn.k_proj(source), attn.v_proj(source))
    if key.dtype != torch.bfloat16:
        RowDraftHead._kv_into_cache(fused, attn, source, cos, sin, cache, slots, None, projected)
        return
    normalized = fused.fast_rms_norm(key.view(rows, 1, nkv, hd), attn.k_norm.weight,
                                     attn.k_norm.variance_epsilon).contiguous()
    _kv_rotary_slots_kernel[(rows * nkv,)](
        normalized, value.contiguous(), cos.contiguous(), sin.contiguous(), cache, slots,
        cache.stride(0), cache.stride(1), cache.stride(3), kv_heads=nkv, head_dim=hd,
        BLOCK=triton.next_power_of_2(hd), num_warps=4, enable_fp_fusion=False)


def q_rotary_into(fused, projected, attn, cos, sin, queries, token_map):
    """``fused.fast_q_postprocess`` of the compact rows written straight into their rows of the
    bucket's query layout ``queries`` (one kernel instead of the postprocess and a scatter; the
    same norm kernel and rotary arithmetic, so the same values)."""
    rows = projected.shape[0]
    nh, hd = attn.config.num_attention_heads, attn.head_dim
    if (triton is None or projected.device.type != 'cuda' or projected.dtype != torch.bfloat16
            or queries.dtype != torch.bfloat16):
        q = fused.fast_q_postprocess(projected.view(rows, 1, nh * hd), attn.q_norm.weight, cos, sin,
                                     attn.q_norm.variance_epsilon, nh, hd).view(rows, nh, hd)
        queries.index_copy_(0, token_map, q)
        return
    normalized = fused.fast_rms_norm(projected.view(rows, 1, nh, hd), attn.q_norm.weight,
                                     attn.q_norm.variance_epsilon).contiguous()
    _q_rotary_scatter_kernel[(rows * nh,)](
        normalized, cos.contiguous(), sin.contiguous(), queries, token_map, heads=nh, head_dim=hd,
        BLOCK=triton.next_power_of_2(hd), num_warps=4, enable_fp_fusion=False)


class LiveRowsShelf:
    """One padded batch bucket's static buffers, shared by every compact pass of every prefix
    bucket of that size: the row order a round writes and, from it, the compact tokens' bucket
    positions, rotary rows and K/V slots (written once per round, read by both passes), the query
    layout the attention reads, one graph memory pool and (captured on CUDA) the shared logits
    buffer."""

    def __init__(self, drafter, *, padded, rows_per_request, block_size, graphs):
        head = drafter.head
        attn = head.layers[0].self_attn
        parameter = next(head.parameters())
        device, dtype = parameter.device, parameter.dtype
        self.padded, self.H, self.B = int(padded), int(rows_per_request), int(block_size)
        self.rows = self.padded * self.H
        self.tokens = self.rows * self.B
        self.order = torch.arange(self.rows, device=device)
        self.slots = torch.arange(self.B, device=device)
        self.token_map = torch.arange(self.tokens, device=device)
        self.cos = torch.zeros(self.tokens, attn.head_dim, dtype=dtype, device=device)
        self.sin = torch.zeros(self.tokens, attn.head_dim, dtype=dtype, device=device)
        self.kv_slots = torch.zeros(self.tokens, dtype=torch.long, device=device)
        self.queries = torch.zeros(self.tokens, attn.config.num_attention_heads, attn.head_dim,
                                   dtype=dtype, device=device)
        captured = bool(graphs) and device.type == 'cuda'
        self.pool = torch.cuda.graph_pool_handle() if captured else None
        lm_head = drafter.lm_head
        self.drafter = (_DrafterView(drafter, SharedOutLinear(lm_head, self.tokens))
                        if captured and isinstance(getattr(lm_head, 'weight', None), torch.Tensor)
                        and lm_head.weight.ndim == 2 else drafter)



class CompactRowHead(torch.nn.Module):
    """The pass's per-token work over ``rows`` rows of the flat bucket head ``base`` (a
    BucketRowDraftHead): row r of the input is bucket row ``shelf.order[r]``; its tokens' positions,
    rotary rows and K/V slots are the shelf's, written by ``LiveRowsShelf.write_order`` each round
    and read when the pass runs."""

    def __init__(self, base, *, rows, shelf):
        super().__init__()
        if not base.flat or base.F != 1 or base.R != base.H or getattr(base, 'refresh', False):
            raise ValueError('live rows need a flat one-branch bucket without refresh slots')
        if shelf.rows != base.N * base.R or shelf.B != base.B:
            raise ValueError('the live rows shelf must match the bucket rows and block size')
        if not 1 <= int(rows) <= shelf.rows:
            raise ValueError('a compact pass runs 1..the bucket rows')
        self.__dict__['base'] = base
        self.__dict__['shelf'] = shelf
        self.N, self.H, self.F, self.R = int(rows), 1, 1, 1
        self.P, self.T, self.B = base.P, base.T, base.B
        self.graphs = base.graphs
        self.layout = SimpleNamespace(r=int(rows), block_size=base.B)
        tokens = int(rows) * base.B
        self.token_map, self.cos, self.sin, self.kv_slots = (
            shelf.token_map[:tokens], shelf.cos[:tokens], shelf.sin[:tokens], shelf.kv_slots[:tokens])

    def forward(self, noise_embedding, *, run_config=None):
        base = self.base
        if noise_embedding.shape[:2] != (self.layout.r, self.B):
            raise ValueError(f'noise_embedding must have shape [{self.layout.r},{self.B},hidden]')
        hidden = noise_embedding.reshape(-1, noise_embedding.shape[-1])
        token_map, cos, sin, slots = self.token_map, self.cos, self.sin, self.kv_slots
        if rows_pass_fused(hidden, run_config=run_config) and base.backend == 'flashinfer':
            return self._forward_fused(hidden, fused_kernels(), token_map, cos, sin, slots)
        queries = self.shelf.queries
        for layer, cache in zip(base.head.layers, base.kv):
            attn = layer.self_attn
            residual = hidden
            normalized = layer.input_layernorm(hidden)
            if hasattr(attn, '_fused_qkv'):
                weight, bias, widths = attn._fused_qkv
                qp, kp, vp = torch.nn.functional.linear(normalized, weight, bias).split(widths, dim=-1)
            else:
                qp, kp, vp = attn.q_proj(normalized), attn.k_proj(normalized), attn.v_proj(normalized)
            q = attn.q_norm(qp.reshape(-1, attn.config.num_attention_heads, attn.head_dim))
            k = attn.k_norm(kp.reshape(-1, attn.config.num_key_value_heads, attn.head_dim))
            v = vp.reshape(-1, attn.config.num_key_value_heads, attn.head_dim)
            q = _rotary(q, cos, sin)
            k = _rotary(k, cos, sin)
            cache[slots, 0, 0] = k
            cache[slots, 1, 0] = v
            queries.index_copy_(0, token_map, q)
            if base.backend == 'flashinfer':
                output = base.wrapper.run(queries, cache)
            else:
                output = sdpa_attention_forward(
                    attn, queries.transpose(0, 1)[None], cache[:, 0, 0].transpose(0, 1)[None],
                    cache[:, 1, 0].transpose(0, 1)[None], base.mask,
                    scaling=attn.scaling, is_causal=False,
                )[0][0]
            output = output.index_select(0, token_map)
            hidden = residual + attn.o_proj(output.reshape(hidden.shape))
            if layer.drafter_mlp:
                hidden = hidden + layer.mlp(layer.post_attention_layernorm(hidden))
        return base.head.norm(hidden).reshape(self.layout.r, self.B, -1)

    def _forward_fused(self, hidden, fused, token_map, cos, sin, slots):
        """RowDraftHead._forward_fused over the compact rows (the same kernels per token)."""
        base, queries = self.base, self.shelf.queries
        rows = hidden.shape[0]
        pending = None
        for layer, cache in zip(base.head.layers, base.kv):
            attn = layer.self_attn
            nh, hd = attn.config.num_attention_heads, attn.head_dim
            if pending is None:
                normalized = fused.fast_rms_norm(hidden, layer.input_layernorm.weight,
                                                 layer.input_layernorm.variance_epsilon)
            else:
                normalized, hidden = fused.fast_add_rms_norm(
                    *pending, layer.input_layernorm.weight, layer.input_layernorm.variance_epsilon)
            residual = hidden
            if hasattr(attn, '_fused_qkv'):
                weight, bias, widths = attn._fused_qkv
                qp, kp, vp = torch.nn.functional.linear(normalized, weight, bias).split(widths, dim=-1)
                projected_kv = (kp, vp.contiguous())
            else:
                qp, projected_kv = attn.q_proj(normalized), None
            q_rotary_into(fused, qp, attn, cos, sin, queries, token_map)
            kv_rotary_into_slots(fused, attn, normalized, cos, sin, cache, slots, projected_kv)
            output = base.wrapper.run(queries, cache).index_select(0, token_map)
            update = attn.o_proj(output.reshape(hidden.shape))
            if layer.drafter_mlp:
                normalized, hidden = fused.fast_add_rms_norm(
                    update, residual, layer.post_attention_layernorm.weight,
                    layer.post_attention_layernorm.variance_epsilon)
                pending = (layer.mlp(normalized), hidden)
            else:
                pending = (update, residual)
        head = base.head
        if pending is None:
            output = fused.fast_rms_norm(hidden, head.norm.weight, head.norm.variance_epsilon)
        else:
            output, _ = fused.fast_add_rms_norm(*pending, head.norm.weight, head.norm.variance_epsilon)
        return output.reshape(self.layout.r, self.B, -1)
