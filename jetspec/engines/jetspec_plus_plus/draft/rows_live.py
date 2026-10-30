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
from jetspec.engines.jetspec_plus_plus.models.target_linear import draft_linear, draft_mm, draft_precision

from jetspec.engines.jetspec_plus_plus.draft.rows import RowDraftHead
from jetspec.engines.jetspec_plus_plus.draft.rows import _rotary
from jetspec.engines.jetspec_plus_plus.draft.rows import fused_kernels
from jetspec.engines.jetspec_plus_plus.draft.rows import rows_pass_fused
from jetspec.engines.jetspec_plus_plus.draft.rows import sdpa_attention_forward

try:
    import triton
    import triton.language as tl
except ImportError:  # Triton is Linux/CUDA-only in the pinned environment.
    triton = None
    tl = None


def rows_live(run_config=None):
    """JETSPEC_BATCH_ROWS_LIVE (default 1; read only under JETSPEC_BATCH_DRAFT_GRAPH=1): the
    batched draft graph's Jacobi passes run over the round's live rows (the compact pass above);
    0 keeps the fixed row-cap pass."""
    from jetspec.engines.jetspec_plus_plus.draft.config import setting, required_setting, as_run_config
    value = setting(run_config, 'engine.engine_settings.batch_rows_live', '1')
    if value not in ('0', '1'):
        raise ValueError('JETSPEC_BATCH_ROWS_LIVE must be 0 or 1')
    return value == '1'


def rows_live_quantum(capacity, run_config=None):
    """JETSPEC_BATCH_ROWS_QUANTUM: the compact pass's row count is the live row count rounded up to
    a multiple of this (fewer captured passes, a few dead rows per round).  Unset, it is a
    sixteenth of the bucket's rows, at least 2 (2 up to 32 rows, 4 at 64, 8 at 128): at most ~16
    passes per bucket and prefix bucket, with the rounding small next to the live rows."""
    from jetspec.engines.jetspec_plus_plus.draft.config import setting, required_setting, as_run_config
    value = setting(run_config, 'engine.engine_settings.batch_rows_quantum')
    if value is None:
        return max(2, int(capacity) // 16)
    if not value.isdigit() or int(value) < 1:
        raise ValueError('JETSPEC_BATCH_ROWS_QUANTUM must be a positive integer')
    return int(value)


def live_row_bucket(total, capacity, quantum):
    """The compact pass's row count for ``total`` live rows of a ``capacity``-row bucket."""
    if not 1 <= int(total) <= int(capacity):
        raise ValueError('live rows must number 1..the bucket capacity')
    return min(int(capacity), -(-int(total) // int(quantum)) * int(quantum))


def live_row_buckets(padded, rows_per_request, quantum):
    """Every compact row count a round of the ``padded`` batch bucket can use.  A bucket serves
    more than half of its size (batch_draft_bucket: a smaller bucket serves fewer requests; the
    two-request bucket serves two), and every request drafts 1..rows_per_request rows."""
    capacity = int(padded) * int(rows_per_request)
    fewest = 2 if padded <= 2 else padded // 2 + 1
    return sorted({live_row_bucket(total, capacity, quantum) for total in range(fewest, capacity + 1)})


def live_row_order(counts, rows_per_request):
    """The bucket's rows in compact order: each request's first ``counts[n]`` rows, request-major,
    then every other row of the bucket (the dead rows a rounded-up compact pass also runs)."""
    H = int(rows_per_request)
    if any(not 0 <= int(count) <= H for count in counts):
        raise ValueError('a request drafts 0..H rows')
    live = [n * H + j for n, count in enumerate(counts) for j in range(int(count))]
    dead = [n * H + j for n, count in enumerate(counts) for j in range(int(count), H)]
    return live + dead


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
        return draft_mm(x, self.weight.t(), self.bias, out=out,
                        precision=draft_precision(self.weight))


class _DrafterView:
    """The drafter with another lm_head (BucketPass.step reads ``drafter.lm_head``)."""

    def __init__(self, drafter, lm_head):
        self.__dict__['_drafter'] = drafter
        self.__dict__['lm_head'] = lm_head

    def __getattr__(self, name):
        return getattr(self.__dict__['_drafter'], name)


if triton is not None:
    from jetspec.engines.jetspec_plus_plus.kernels.live_rows import (_q_rotary_scatter_kernel, _kv_rotary_slots_kernel, _compact_rows_kernel)


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

    def write_order(self, order, rows, base, inputs=()):
        """Upload the round's compact order (every bucket row, live first) in one copy and gather
        its first ``rows`` rows' token positions, query rotary rows (the bucket head ``base``'s,
        after this round's geometry) and noise K/V slots, and those rows of each bucket-row tensor
        in ``inputs`` (anchors [rows, 1], first-pass ids [rows, B], keys [rows, 2], active [rows]:
        one kernel on CUDA).  Returns (the device order of those rows, the gathered inputs)."""
        order = list(map(int, order))
        if sorted(order) != list(range(self.rows)):
            raise ValueError('the compact order must list every bucket row once')
        host = torch.tensor(order, dtype=torch.long)
        cuda = self.order.device.type == 'cuda'
        self.order.copy_(host.pin_memory() if cuda else host, non_blocking=cuda)
        rows = int(rows)
        tokens = rows * self.B
        order_rows = self.order[:rows]
        qcos, qsin, noise_slots = base.qcos[0], base.qsin[0], base.layout.noise_slots
        if (triton is not None and cuda and len(inputs) == 4
                and all(x.is_contiguous() for x in (qcos, qsin, noise_slots, *inputs))
                and inputs[1].shape[-1] == self.B and inputs[0].shape[-1] == 1
                and inputs[2].shape[-1] == 2 and qcos.shape[-1] == self.cos.shape[-1]):
            preceding, first, keys, active = inputs
            gathered = (preceding.new_empty(rows, 1), first.new_empty(rows, self.B),
                        keys.new_empty(rows, 2), active.new_empty(rows))
            _compact_rows_kernel[(rows,)](
                order_rows, qcos, qsin, noise_slots, preceding, first, keys, active, self.token_map,
                self.cos, self.sin, self.kv_slots, *gathered, first.stride(0), B=self.B,
                HD=qcos.shape[-1], BLOCK_B=triton.next_power_of_2(self.B),
                BLOCK_HD=triton.next_power_of_2(qcos.shape[-1]), num_warps=1)
            return order_rows, gathered
        token_map = self.token_map[:tokens]
        torch.add(order_rows[:, None] * self.B, self.slots, out=token_map.view(rows, self.B))
        torch.index_select(qcos, 0, token_map, out=self.cos[:tokens])
        torch.index_select(qsin, 0, token_map, out=self.sin[:tokens])
        torch.index_select(noise_slots, 0, token_map, out=self.kv_slots[:tokens])
        return order_rows, tuple(x.index_select(0, order_rows) for x in inputs)


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
                qp, kp, vp = draft_linear(normalized, weight, bias).split(widths, dim=-1)
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
            hidden = residual + attn.o_proj(output.reshape(hidden.shape[0], -1))
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
                qp, kp, vp = draft_linear(normalized, weight, bias).split(widths, dim=-1)
                projected_kv = (kp, vp.contiguous())
            else:
                qp, projected_kv = attn.q_proj(normalized), None
            q_rotary_into(fused, qp, attn, cos, sin, queries, token_map)
            kv_rotary_into_slots(fused, attn, normalized, cos, sin, cache, slots, projected_kv)
            output = base.wrapper.run(queries, cache).index_select(0, token_map)
            update = attn.o_proj(output.reshape(hidden.shape[0], -1))
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
