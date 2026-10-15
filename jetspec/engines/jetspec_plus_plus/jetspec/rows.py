"""Shared-prefix, 16-position rows for the refine path's T module.

N is sequences, k is one-based block position, F is branches, J is passes,
and r = N * len(k-set) * F. The existing wire uses hypothesis = k - 1.
"""
from __future__ import annotations

from jetspec.engines.jetspec_plus_plus.profiles import kernel_enabled, kernel_value

import importlib.util
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from jetspec.engines.jetspec_plus_plus.jetspec.env import context_append_enabled
from jetspec.engines.jetspec_plus_plus.jetspec.config_values import config_values
from jetspec.engines.jetspec_plus_plus.legacy_env import environment_name

try:
    import torch
    from torch import nn
    from transformers.integrations.sdpa_attention import sdpa_attention_forward
except ImportError:
    class _MissingTorch:
        long = object()

        @staticmethod
        def inference_mode():
            return lambda function: function

    class _MissingNN:
        class Module:
            pass

    def sdpa_attention_forward(*args, **kwargs):
        raise RuntimeError("rows execution requires the pinned serve dependencies")

    torch = _MissingTorch()
    nn = _MissingNN()

try:
    import triton
    import triton.language as tl
except ImportError:  # Triton is Linux/CUDA-only in the pinned environment.
    triton = None
    tl = None


class RowLayout:
    """One paged KV group per sequence; prefix pages occur exactly once.

    A group holds stale prefix, shared predicted tail, then each row's own
    block. The mask exposes prefix + k - 1 and the row's causal block only.
    Page size 1 permits arbitrary prefix lengths without copying tail pages.
    """

    def __init__(self, prefix_lengths, ksets, *, F=1, device='cuda', private_tails=False,
                 vectorized=False, row_capacity=None, _flat_rows=None, refresh_slots=0,
                 tail_rows=15, block_size=16):
        if F not in (1, 2, 3, 5):
            raise ValueError('F must be 1, 2, 3, or 5')
        if not prefix_lengths or len(prefix_lengths) != len(ksets):
            raise ValueError('one nonempty k-set is required per sequence')
        self.F = F
        self.private_tails = private_tails
        self.N = len(prefix_lengths)
        self.prefix_lengths = list(map(int, prefix_lengths))
        self.flat_rows = _flat_rows is not None
        self.block_size = int(block_size)
        if self.block_size not in (4, 8, 16, 32) or (self.block_size != 16 and not self.flat_rows):
            raise ValueError('non-16-position blocks require flat rows')
        if self.flat_rows:
            if F != 1 or private_tails or row_capacity is None:
                raise ValueError('flat rows require one branch, shared tails, and a row capacity')
            if len(_flat_rows) != len(prefix_lengths):
                raise ValueError('one flat row list is required per sequence')
            normalized = [[(int(k), int(anchor)) for k, anchor in rows] for rows in _flat_rows]
            if any(len(rows) > row_capacity or len(set(rows)) != len(rows) or
                   any(k < 0 or k >= self.block_size or anchor < 0 for k, anchor in rows)
                   for rows in normalized):
                raise ValueError('flat rows require unique (k=0..15, anchor>=0) keys within capacity')
            self.ksets = [[k + 1 for k, _ in rows] for rows in normalized]
            if any(not ks for ks in self.ksets):
                self.ksets = [ks or [1] for ks in self.ksets]
            padded_keys = [rows + [(-1, -1)] * (row_capacity - len(rows)) for rows in normalized]
            self.row_keys = torch.tensor(
                [pair for rows in padded_keys for pair in rows], dtype=torch.long, device=device)
            self.row_ks = torch.tensor(
                [k + 1 if k >= 0 else 1 for rows in padded_keys for k, _ in rows],
                dtype=torch.long, device=device)
        else:
            self.ksets = [list(map(int, ks)) for ks in ksets]
            self.row_keys = None
            self.row_ks = None
        if any(p < 1 for p in self.prefix_lengths):
            raise ValueError('prefix lengths must be positive')
        # JETSPEC_ADAPTER_REFRESH: 16 KV slots per row after the row blocks, for the
        # refreshed block entries.  Every mask term below leaves them hidden.
        self.refresh_slots = int(refresh_slots)
        if self.refresh_slots and (not self.flat_rows or not vectorized):
            raise ValueError('refresh slots require vectorized flat rows')
        if any(not ks or (not self.flat_rows and len(set(ks)) != len(ks)) or
               any(k < 1 or k > self.block_size for k in ks) for ks in self.ksets):
            raise ValueError('k-set must contain unique positions in 1..16')
        # serving: flat rows keep tail_rows staged tail rows (16 with 15 drafts)
        if int(tail_rows) not in (4, 8, 15, 16, 32) or (int(tail_rows) != 15 and not self.flat_rows):
            raise ValueError('tail_rows must match a supported flat block or legacy 15-row tail')
        self.tail_rows = int(tail_rows)
        self.context_lengths = ([p + self.tail_rows for p in self.prefix_lengths] if self.flat_rows else
            [p + (sum(k - 1 for k in ks) if private_tails else max(ks) - 1)
             for p, ks in zip(self.prefix_lengths, self.ksets)])
        if row_capacity is not None and (private_tails or any(row_capacity < len(ks)*F for ks in self.ksets)):
            raise ValueError('row capacity must cover all shared-tail branches')
        self.rows_per_sequence = [row_capacity or len(ks) * F for ks in self.ksets]
        self.r = sum(self.rows_per_sequence)
        self.group_masks = []
        qo, kv = [0], [0]
        context_slots, noise_slots, context_positions, query_positions = [], [], [], []
        for p, c, ks in zip(self.prefix_lengths, self.context_lengths, self.ksets):
            count = row_capacity or len(ks) * F
            length = c + count * self.block_size + self.refresh_slots
            mask = None if vectorized else torch.zeros(count * self.block_size, length, dtype=torch.bool)
            context_slots.extend(range(kv[-1], kv[-1] + c))
            noise_slots.extend(range(kv[-1] + c, kv[-1] + c + count * self.block_size))
            context_positions.extend(range(p))
            if private_tails:
                for k in ks:
                    context_positions.extend(range(p, p + k - 1))
            else:
                context_positions.extend(range(p, c))
            if vectorized:
                kr = torch.tensor(ks, device=device).repeat_interleave(F)
                if count > kr.numel():
                    kr = torch.cat((kr, kr.new_full((count-kr.numel(),),1)))
                q = torch.arange(count*self.block_size, device=device)
                col = torch.arange(length, device=device)
                row_ids = q // self.block_size
                if private_tails:
                    starts = torch.cat((kr.new_zeros(1), (torch.tensor(ks, device=device)-1).cumsum(0)[:-1]))
                    tail_start = p + starts.repeat_interleave(F)[row_ids]
                else:
                    tail_start = torch.full_like(q, p)
                prefix_allowed = col[None] < p
                tail_allowed = (col[None] >= tail_start[:, None]) & (col[None] < (tail_start+kr[row_ids]-1)[:, None])
                block_allowed = ((col[None]-c)//self.block_size == row_ids[:, None]) & ((col[None]-c)%self.block_size <= (q%self.block_size)[:, None]) & (col[None] >= c)
                mask = prefix_allowed | tail_allowed | block_allowed
                query_positions.append((p+kr[:, None]-1+torch.arange(self.block_size, device=device)).flatten())
            else:
                row_ks = [k for k in ks for _ in range(F)] + [1]*(count-len(ks)*F)
                for row, k in enumerate(row_ks):
                    block = slice(row * self.block_size, (row + 1) * self.block_size)
                    mask[block, :p] = True
                    tail_start = p + sum(x - 1 for x in ks[:row // F]) if private_tails else p
                    mask[block, tail_start:tail_start + k - 1] = True
                    mask[block, c + row * self.block_size:c + (row + 1) * self.block_size] = torch.ones(self.block_size, self.block_size, dtype=torch.bool).tril()
                    query_positions.extend(range(p + k - 1, p + k - 1 + self.block_size))
            self.group_masks.append(mask)
            qo.append(qo[-1] + count * self.block_size)
            kv.append(kv[-1] + length)
        self.num_kv = kv[-1]
        self.qo_indptr = torch.tensor(qo, dtype=torch.int32, device=device)
        self.kv_indptr = torch.tensor(kv, dtype=torch.int32, device=device)
        self.kv_indices = torch.arange(self.num_kv, dtype=torch.int32, device=device)
        self.last_page_len = torch.ones(self.N, dtype=torch.int32, device=device)
        self.custom_mask = torch.cat([x.flatten() for x in self.group_masks]).to(device)
        self.context_slots = torch.tensor(context_slots, dtype=torch.long, device=device)
        self.noise_slots = torch.tensor(noise_slots, dtype=torch.long, device=device)
        self.context_positions = torch.tensor(context_positions, dtype=torch.long, device=device)
        self.query_positions = (torch.cat(query_positions) if vectorized else
                                torch.tensor(query_positions, dtype=torch.long, device=device))

    @classmethod
    def flat(cls, prefix_lengths, rows, *, row_capacity, device='cuda', vectorized=False,
             refresh_slots=0, tail_rows=15, block_size=16):
        """Build repeated-k keyed rows; keys use public zero-based ``(k, anchor)``."""
        return cls(prefix_lengths, [[] for _ in prefix_lengths], F=1, device=device,
                   vectorized=vectorized, row_capacity=row_capacity, _flat_rows=rows,
                   refresh_slots=refresh_slots, tail_rows=tail_rows, block_size=block_size)

    def dense_mask(self, device):
        """CPU/fp32 oracle: still one attention operation over all groups."""
        return torch.block_diag(*self.group_masks).to(device)


def _rotary(x, cos, sin):
    half = x.shape[-1] // 2
    rotated = torch.cat((-x[..., half:], x[..., :half]), dim=-1)
    return x * cos[:, None, :] + rotated * sin[:, None, :]


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


def kv_rotary_into_cache(normalized_key, projected_value, cos, sin, key_out, value_out, *,
                         kv_heads, groups, head_dim):
    """Rotate ``normalized_key`` and copy ``projected_value`` into two cache views.

    ``normalized_key`` and ``projected_value`` are ``[tokens, kv_heads * head_dim]``
    (any row stride); ``key_out`` / ``value_out`` are ``[tokens, kv_heads * groups, head_dim]``
    views with a dense last dimension.  Off CUDA / off bf16 the reference
    arithmetic runs through plain tensor ops.
    """
    tokens = normalized_key.shape[0]
    heads = kv_heads * groups
    if (normalized_key.device.type != 'cuda' or normalized_key.dtype != torch.bfloat16
            or triton is None):
        key = normalized_key.reshape(tokens, kv_heads, head_dim)
        value = projected_value.reshape(tokens, kv_heads, head_dim)
        rotated = _rotary(key, cos, sin)
        if groups != 1:
            rotated = rotated.repeat_interleave(groups, dim=1)
            value = value.repeat_interleave(groups, dim=1)
        key_out.copy_(rotated)
        value_out.copy_(value)
        return
    if (normalized_key.stride(1) != 1 or projected_value.stride() != normalized_key.stride()
            or key_out.stride() != value_out.stride() or key_out.stride(2) != 1
            or key_out.shape != (tokens, heads, head_dim)):
        raise ValueError('cache kv write requires matching dense-row inputs and matching cache views')
    cos, sin = cos.contiguous(), sin.contiguous()
    _kv_rotary_into_cache_kernel[(tokens * heads,)](
        normalized_key, projected_value, cos, sin, key_out, value_out,
        tokens=tokens, kv_heads=kv_heads, groups=groups, head_dim=head_dim,
        in_token_stride=normalized_key.stride(0),
        out_token_stride=key_out.stride(0), out_head_stride=key_out.stride(1),
        BLOCK=triton.next_power_of_2(head_dim), num_warps=4, enable_fp_fusion=False)


def kv_rotary_into_cache_slots(normalized_key, projected_value, cos, sin,
                               key_cache, value_cache, slots, *, kv_heads, head_dim):
    """The cache rotary arithmetic with a device slot vector for graph extensions."""
    tokens = normalized_key.shape[0]
    if (normalized_key.device.type != 'cuda' or normalized_key.dtype != torch.bfloat16
            or triton is None):
        key = normalized_key.reshape(tokens, kv_heads, head_dim)
        key = _rotary(key, cos, sin)
        value = projected_value.reshape(tokens, kv_heads, head_dim)
        key_cache.index_copy_(0, slots, key)
        value_cache.index_copy_(0, slots, value)
        return
    if (normalized_key.shape != (tokens, kv_heads * head_dim)
            or projected_value.shape != normalized_key.shape
            or key_cache.shape[1:] != (kv_heads, head_dim)
            or key_cache.stride() != value_cache.stride()
            or key_cache.stride(2) != 1):
        raise ValueError('cache slot write requires matching KV geometry')
    _kv_rotary_into_slots_kernel[(tokens * kv_heads,)](
        normalized_key, projected_value, cos.contiguous(), sin.contiguous(),
        key_cache, value_cache, slots, tokens, kv_heads, head_dim,
        normalized_key.stride(0), key_cache.stride(0), key_cache.stride(1),
        triton.next_power_of_2(head_dim), num_warps=4, enable_fp_fusion=False)


def top1_probabilities_fast(logits):
    """[rows, slots, vocab] -> float32 [rows, slots] top-1 softmax probabilities in one kernel
    (JETSPEC_BATCH_DRAFT_GRAPH): the chunked float softmax's value at the argmax, up to the
    order of its fp32 sum. None when the input is not a CUDA tensor with a dense last dim."""
    if (triton is None or logits.device.type != 'cuda' or logits.ndim != 3
            or logits.stride(-1) != 1 or logits.numel() == 0):
        return None
    rows, slots, vocab = logits.shape
    out = torch.empty(rows, slots, dtype=torch.float32, device=logits.device)
    _top1_probability_kernel[(rows * slots,)](logits, out, vocab, logits.stride(0),
                                               logits.stride(1), slots, BLOCK=2048, num_warps=8)
    return out


def fused_kernels():
    """The certified C1 kernels, under the one module name the adapter loads."""
    name = 'jetspec_c1_fused'
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name('c1_fused.py'))
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def rows_fused(tensor, *, run_config=None):
    """Legacy eager fusion, also read by the decoder's residual dispatch."""
    from jetspec.engines.jetspec_plus_plus.profiles import kernel_enabled
    return (kernel_enabled('rows_fused', '1') and tensor.device.type == 'cuda'
            and tensor.dtype == torch.bfloat16
            and not (kernel_enabled('rows_fused_graph', '1')
                     and any(config_values(run_config).get(name, '0') == '1' for name in
                             ('engine.engine_settings.rows_graphs', 'engine.engine_settings.host_fast', 'engine.engine_settings.c1_graph'))))


def rows_pass_fused(tensor, *, run_config=None):
    """Fuse the row pass and context refresh without changing decoder dispatch."""
    from jetspec.engines.jetspec_plus_plus.profiles import kernel_enabled
    return (tensor.device.type == 'cuda' and tensor.dtype == torch.bfloat16
            and (rows_fused(tensor, run_config=run_config) or kernel_enabled('rows_fused_graph', '1')))


def check_rows_fused_regime(*, run_config=None):
    """Keep legacy eager refusal; the graph switch isolates row fusion from decoder.

    Under the graphed regime the anchor decoder's window layers run as graph parts
    and never hand the fused path its pending residual pair, so a served round
    would reach ``fast_add_rms_norm(*None)`` (B200, /decoder_ predict, round 4).
    """
    from jetspec.engines.jetspec_plus_plus.profiles import kernel_enabled
    if not kernel_enabled('rows_fused', '1'):
        return
    graphed = [name for name in ('engine.engine_settings.rows_graphs', 'engine.engine_settings.host_fast', 'engine.engine_settings.c1_graph')
               if config_values(run_config).get(name, '0') == '1']
    if graphed and not kernel_enabled('rows_fused_graph', '1'):
        raise ValueError('JETSPEC_ROWS_FUSED=1 is the eager rows path; unset it or run eager '
                         '(set here: ' + ', '.join(f'{environment_name(*name.split(".", 1))}=1' for name in graphed) + ')')


def rows_ctx_incr(*, run_config=None):
    """JETSPEC_ROWS_CTX_INCR=1 (default off, not exact, see _incremental_context): an
    eager N=1 flat bucket keeps the context K/V of the request it served last and
    projects only appended prefix rows and the 15 tail rows."""
    value = config_values(run_config).get('engine.engine_settings.rows_ctx_incr', '0')
    if value not in ('0', '1'):
        raise ValueError('JETSPEC_ROWS_CTX_INCR must be 0 or 1')
    return value == '1'


def rows_ctx_early(*, run_config=None):
    """JETSPEC_ROWS_CTX_EARLY=1 (default off): the flat bucket's context K/V is projected at
    the start of the proposal, on a side stream beside the anchor decoder (EarlyContextKV),
    and the bucket update copies it in; the same rows, kernels and shapes, so exact."""
    value = config_values(run_config).get('engine.engine_settings.rows_ctx_early', '0')
    if value not in ('0', '1'):
        raise ValueError('JETSPEC_ROWS_CTX_EARLY must be 0 or 1')
    return value == '1'


def refresh_inject_lean():
    """JETSPEC_REFRESH_INJECT_LEAN=1 (default off): inject_refresh in about 50 launches instead
    of about 125, with the same kernels on the same values: the five layers' k norm and rotary
    run once over the stacked keys (elementwise ops and a per-row mean over head_dim), the
    cache writes are index_copy_ instead of advanced-index puts, and the packed mask comes from
    a per-bucket (row, k-hat) bank (the rows' bits are disjoint, so their byte sum is their OR)."""
    from jetspec.engines.jetspec_plus_plus.profiles import kernel_value
    value = kernel_value('refresh_inject_lean', '0')
    if value not in ('0', '1'):
        raise ValueError('JETSPEC_REFRESH_INJECT_LEAN must be 0 or 1')
    return value == '1'


def _context_kv_views(storage, *, run_config=None):
    caches = list(storage.unbind(0))
    if (config_values(run_config).get('engine.engine_settings.ctx_kv_direct', '0') == '1'
            or config_values(run_config).get('engine.engine_settings.hybrid_context_dirty', '0') == '1'):
        # Inference tensors do not track a view's _base.
        caches[0]._jetspec_context_storage = storage
    return caches


def _context_kv_storage(head, caches, *, run_config=None):
    if not hasattr(head, '_context_kv'):
        return None
    if (config_values(run_config).get('engine.engine_settings.ctx_kv_direct', '0') == '1'
            or config_values(run_config).get('engine.engine_settings.hybrid_context_dirty', '0') == '1'):
        return getattr(caches[0], '_jetspec_context_storage', caches[0]._base)
    return caches[0]._base


class EarlyContextKV:
    """serving JETSPEC_ROWS_CTX_EARLY: one N=1 flat bucket's context K/V for prefix bucket P.

    ``issue`` writes the contexts exactly as BucketRowDraftHead.update does (zero, prefix,
    tails, the adapted markers) and projects all P + 15 (16 with 15 drafts) rows with the bucket refresh's own
    kernels and shapes (``_refresh_context_eager`` / ``_refresh_context_fused`` over the same
    rotary rows: the prefix rows from one rotary call over 0..P+14, the tail rows gathered from
    the rotary table over 0..P+31, as prepare() and the cached geometry build them), on a side
    stream, so it runs beside the anchor decoder.  ``copy_into`` makes the bucket's stream wait
    for it and copies the K/V rows into the bucket's context slots (one copy per layer)."""

    @torch.inference_mode()
    def __init__(self, head, P, tail_rows=15, N=1, *, run_config=None):
        self.run_config = run_config
        from jetspec.engines.jetspec_plus_plus.profiles import kernel_enabled
        parameter = next(head.parameters())
        self.head, self.P, self.N = head, P, N
        self.T = int(tail_rows)  # serving: 16 with 15 drafts
        self.device, self.dtype = parameter.device, parameter.dtype
        if (kernel_enabled('ctx_kv_fused', '1')
                or config_values(self.run_config).get('engine.engine_settings.hybrid_context_dirty', '0') == '1'):
            from jetspec.engines.jetspec_plus_plus.kernels.projections import prepare_context_kv
            prepare_context_kv(head)
        self.contexts = torch.zeros(N * (P + self.T), head.fc.in_features, device=self.device, dtype=self.dtype)
        self.kv = []
        if hasattr(head, '_context_kv'):
            attn = head.layers[0].self_attn
            storage = torch.empty(
                len(head.layers), N * (P + self.T), 2, 1,
                attn.config.num_key_value_heads, attn.head_dim,
                device=self.device, dtype=self.dtype)
            self.kv = _context_kv_views(storage, run_config=self.run_config)
        else:
            for layer in head.layers:
                attn = layer.self_attn
                self.kv.append(torch.empty(N * (P + self.T), 2, 1, attn.config.num_key_value_heads, attn.head_dim,
                                           device=self.device, dtype=self.dtype))
        stub = torch.empty(1, 1, head.config.hidden_size, device=self.device, dtype=self.dtype)
        positions = torch.arange(P + self.T, device=self.device)
        if N > 1:
            positions = positions.repeat(N)
        cos, sin = head.rotary_emb(stub, positions[None])
        self.cos, self.sin = cos[0].clone(), sin[0].clone()
        table = head.rotary_emb(stub, torch.arange(P + 32, device=self.device)[None])
        self._table = (table[0][0], table[1][0])
        self._tail_positions = torch.arange(self.T, device=self.device)
        self._slots = torch.arange(N * (P + self.T), device=self.device) if N > 1 else None
        self.stream = torch.cuda.Stream(device=self.device) if self.device.type == 'cuda' else None
        self.event = None
        self.start_event = self.end_event = None
        self.key = None

    @torch.inference_mode()
    def issue(self, prefix, tails, *, key):
        if self.N > 1:
            return self._issue_batch(prefix, tails, key=key)
        length = prefix.shape[0]
        if length > self.P:
            raise ValueError('prefix exceeds bucket')
        if self.stream is not None:
            self.stream.wait_stream(torch.cuda.current_stream(self.device))
            prefix.record_stream(self.stream)
            tails.record_stream(self.stream)
            context_manager = torch.cuda.stream(self.stream)
        else:
            import contextlib
            context_manager = contextlib.nullcontext()
        with context_manager:
            context = self.contexts.view(1, self.P + self.T, -1)
            context.zero_()
            context[:, :length].copy_(prefix[None])
            context[:, self.P:].copy_(tails)
            context[:, :self.P].add_(self.head.adapted_position_embedding.weight[0])
            context[:, self.P:].add_(self.head.adapted_position_embedding.weight[1])
            tail = length + self._tail_positions
            self.cos[self.P:].copy_(self._table[0].index_select(0, tail))
            self.sin[self.P:].copy_(self._table[1].index_select(0, tail))
            self._project()
            if self.stream is not None:
                self.event = torch.cuda.Event()
                self.event.record(self.stream)
        self.key = (key, length)

    @torch.inference_mode()
    def _issue_batch(self, prefixes, tails, *, key):
        self.start_event = self.end_event = None
        lengths = tuple(prefix.shape[0] for prefix in prefixes)
        if len(prefixes) != self.N or any(length > self.P for length in lengths):
            raise ValueError('batched prefix count or length exceeds bucket')
        if tails.shape[:2] != (self.N, self.T):
            raise ValueError('batched tails must have shape [N,T,hidden]')
        if self.stream is not None:
            self.stream.wait_stream(torch.cuda.current_stream(self.device))
            for prefix in prefixes:
                prefix.record_stream(self.stream)
            tails.record_stream(self.stream)
            context_manager = torch.cuda.stream(self.stream)
        else:
            import contextlib
            context_manager = contextlib.nullcontext()
        with context_manager:
            if self.stream is not None and config_values(self.run_config).get('engine.engine_settings.device_split', '0') == '1':
                self.start_event = torch.cuda.Event(enable_timing=True)
                self.start_event.record(self.stream)
            padded = torch.nn.utils.rnn.pad_sequence(prefixes, batch_first=True)
            context = self.contexts.view(self.N, self.P + self.T, -1)
            context.zero_()
            context[:, :padded.shape[1]].copy_(padded)
            context[:, self.P:].copy_(tails)
            context[:, :self.P].add_(self.head.adapted_position_embedding.weight[0])
            context[:, self.P:].add_(self.head.adapted_position_embedding.weight[1])
            tail = (device_tensor(lengths, self.device)[:, None] + self._tail_positions).flatten()
            width = self.cos.shape[-1]
            self.cos.view(self.N, self.P + self.T, width)[:, self.P:].copy_(
                self._table[0].index_select(0, tail).view(self.N, self.T, width))
            self.sin.view(self.N, self.P + self.T, width)[:, self.P:].copy_(
                self._table[1].index_select(0, tail).view(self.N, self.T, width))
            self._project()
            if self.start_event is not None:
                self.end_event = torch.cuda.Event(enable_timing=True)
                self.end_event.record(self.stream)
            if self.stream is not None:
                self.event = torch.cuda.Event()
                self.event.record(self.stream)
        self.key = (tuple(key), lengths)

    def _project(self):
        head, contexts, span = self.head, self.contexts, (0, self.N * (self.P + self.T))
        if rows_pass_fused(contexts, run_config=self.run_config):
            fused = fused_kernels()
            hidden = fused.fast_rms_norm(head.fc(contexts), head.hidden_norm.weight,
                                         head.hidden_norm.variance_epsilon)
            if RowDraftHead._context_into_cache(
                    fused, head, hidden, self.cos, self.sin, self.kv, self._slots,
                    span if self.N == 1 else None,
                    _context_kv_storage(head, self.kv, run_config=self.run_config), run_config=self.run_config):
                return
            for layer, cache in zip(head.layers, self.kv):
                RowDraftHead._kv_into_cache(fused, layer.self_attn, hidden, self.cos, self.sin,
                                            cache, self._slots, span if self.N == 1 else None)
            return
        hidden = head.hidden_norm(head.fc(contexts))
        for layer, cache in zip(head.layers, self.kv):
            attn = layer.self_attn
            nh, hd = attn.config.num_key_value_heads, attn.head_dim
            k = attn.k_norm(attn.k_proj(hidden).reshape(-1, nh, hd))
            k = _rotary(k, self.cos, self.sin)
            v = attn.v_proj(hidden).reshape(-1, nh, hd)
            cache[:, 0, 0] = k
            cache[:, 1, 0] = v

    def ready_for(self, key, length):
        if self.N > 1:
            return (self.key is not None and len(key) == self.N
                    and all(a is b for a, b in zip(self.key[0], key))
                    and self.key[1] == tuple(length))
        return self.key is not None and self.key[0] is key and self.key[1] == length

    def copy_into(self, bucket):
        if self.event is not None:
            torch.cuda.current_stream(self.device).wait_event(self.event)
        for cache, staged in zip(bucket.kv, self.kv):
            if self.N == 1:
                cache[:self.P + self.T].copy_(staged)
            else:
                _write_kv(cache, bucket.layout.context_slots, None, staged[:, 0, 0], staged[:, 1, 0])


def device_tensor(values, device, dtype=torch.long):
    """Host values to ``device`` without a stream drain: a pageable copy would
    synchronize the stream, so CUDA targets stage through pinned memory."""
    tensor = torch.tensor(values, dtype=dtype)
    if device.type == 'cuda' and torch.cuda.is_available():
        return tensor.pin_memory().to(device, non_blocking=True)
    return tensor.to(device)


def _contiguous_range(slots):
    values = slots.tolist()
    if values and values == list(range(values[0], values[0] + len(values))):
        return values[0], values[0] + len(values)
    return None


def pack_mask_bits(allowed):
    """flashinfer's packed custom mask: bit i of each byte is element i."""
    bits = allowed.reshape(-1, 8).to(torch.uint8)
    return (bits * (1 << torch.arange(8, device=allowed.device))).sum(-1).to(torch.uint8)


def packed_mask_banks(*, P, R, columns, queries, tail_rows=15, compact=False):
    """Per-bucket packed masks whose OR is the round's custom mask.

    prefix[length] covers ``col < length`` for every query row, tail[k] covers
    one row block's ``P <= col < P + k``, and block is the constant causal
    block.  Packing is bitwise, so OR of packed bytes is the packed OR.
    """
    col = columns[None, :]
    row = queries // 16
    c = P + tail_rows
    L = columns.numel()
    block = ((col - c) // 16 == row[:, None]) & ((col - c) % 16 <= (queries % 16)[:, None]) & (col >= c)
    k = torch.arange(17, device=columns.device)
    tail = ((col[None] >= P) & (col[None] < P + k[:, None, None])).expand(17, 16, L)
    # A prefix is identical for every query. Store only the shortest byte-
    # aligned repetition; odd column counts need up to eight query rows.
    prefix_rows = 8 // math.gcd(L, 8) if compact else R * 16
    chunks = []
    for start in range(0, P + 1, 32):
        lengths = torch.arange(start, min(P + 1, start + 32), device=columns.device)
        allowed = (col[None] < lengths[:, None, None]).expand(-1, prefix_rows, -1)
        chunks.append(pack_mask_bits(allowed).reshape(len(lengths), -1))
    return dict(prefix=torch.cat(chunks), tail=pack_mask_bits(tail).reshape(17, -1),
                block=pack_mask_bits(block))


def packed_mask(banks, lengths, ks):
    """Three gathers and two ORs replace the per-round mask rebuild."""
    prefix = banks['prefix'].index_select(0, lengths)
    tail = (banks['tail'].index_select(0, ks.reshape(-1)).reshape(lengths.numel(), -1)
            if ks.ndim > 1 else banks['tail'].index_select(0, ks).reshape(1, -1))
    if prefix.shape[-1] == banks['block'].numel():
        return (prefix | tail | banks['block'][None]).flatten()
    combined = tail | banks['block'][None]
    return (prefix[:, None, :] | combined.reshape(lengths.numel(), -1, prefix.shape[-1])).flatten()


def _write_kv(cache, slots, span, k, v):
    """One strided copy per tensor when the slots are one range; index_put otherwise."""
    if span is not None:
        cache[span[0]:span[1], 0, 0].copy_(k)
        cache[span[0]:span[1], 1, 0].copy_(v)
    else:
        cache[slots, 0, 0] = k
        cache[slots, 1, 0] = v


def chain_context_dirty_rows(old_length, length, *, prefix, tail, block, noise_start):
    """Fixed-shape gather and scatter rows for an append-only N=1 chain context."""
    offsets = torch.arange(block, device=length.device)
    appended = old_length.reshape(()) + offsets
    live = (appended < length.reshape(())) & (appended < prefix)
    source = torch.where(live, appended, torch.zeros_like(appended))
    destination = torch.where(live, appended, noise_start + offsets)
    tail_rows = torch.arange(prefix, prefix + tail, device=length.device)
    return torch.cat((source, tail_rows)), torch.cat((destination, tail_rows))


class RowDraftHead(nn.Module):
    """Use the checkpoint's exact parameters with one paged call per layer.

    prepare() projects each unique context once for the round. forward() is
    T(r): all r blocks in one model call, without per-row model/attention calls.
    Its KV buffers retain shared prefix pages across all J passes.
    """

    def __init__(self, head, *, backend='flashinfer', workspace_bytes_per_row=2 * 1024 * 1024, run_config=None):
        self.run_config = run_config
        from jetspec.engines.jetspec_plus_plus.profiles import kernel_enabled
        super().__init__()
        self.head = head
        self.backend = backend
        self.workspace_bytes_per_row = int(workspace_bytes_per_row)
        self.layout = None
        if (kernel_enabled('ctx_kv_fused', '1')
                or config_values(self.run_config).get('engine.engine_settings.hybrid_context_dirty', '0') == '1'):
            from jetspec.engines.jetspec_plus_plus.kernels.projections import prepare_context_kv
            prepare_context_kv(head)
        if (getattr(head, 'has_masked_inject_layers', False)
                or getattr(head, 'token_top_layer', None) is not None
                or getattr(head, 'markov_head', None) is not None):
            raise ValueError('rows requires the full-inject p-free head')

    @torch.inference_mode()
    def prepare(self, layout, contexts):
        self.layout = layout
        dtype = next(self.head.parameters()).dtype
        device = next(self.head.parameters()).device
        if contexts.shape != (sum(layout.context_lengths), self.head.fc.in_features):
            raise ValueError('contexts must contain each unique group context once')
        if contexts.device != device or contexts.dtype != dtype:
            raise ValueError('contexts must match the checkpoint device and dtype')
        self.kv = []
        self.context_span = _contiguous_range(layout.context_slots)
        self.noise_span = _contiguous_range(layout.noise_slots)
        hidden = torch.empty(1, 1, self.head.config.hidden_size, device=device, dtype=dtype)
        self.ccos, self.csin = self.head.rotary_emb(hidden, layout.context_positions[None])
        self.qcos, self.qsin = self.head.rotary_emb(hidden, layout.query_positions[None])
        if hasattr(self.head, '_context_kv'):
            attn = self.head.layers[0].self_attn
            storage = torch.empty(
                len(self.head.layers), layout.num_kv, 2, 1,
                attn.config.num_key_value_heads, attn.head_dim, device=device, dtype=dtype)
            self.kv = _context_kv_views(storage, run_config=self.run_config)
        else:
            for layer in self.head.layers:
                attn = layer.self_attn
                cache = torch.empty(layout.num_kv, 2, 1, attn.config.num_key_value_heads,
                                    attn.head_dim, device=device, dtype=dtype)
                self.kv.append(cache)
        self.refresh_context(contexts)
        if self.backend == 'flashinfer':
            import flashinfer
            # Split-K scratch grows with query rows. The full N=8,r=128
            # control needs 148,897,792 bytes, exceeding a fixed 128 MiB.
            q_width = self.head.config.num_attention_heads * self.head.config.head_dim
            minimum_workspace = 160 if q_width != self.head.config.hidden_size else 128
            workspace_bytes = max(minimum_workspace * 1024 * 1024,
                                  self.workspace_bytes_per_row * layout.r)
            self.workspace = torch.empty(workspace_bytes, dtype=torch.uint8, device=device)
            # The fork's megaspec primitive: paged prefill, custom mask, and
            # cu_seqlens_q per sequence group. Static buffers permit graph replay.
            self.custom_mask_buffer = torch.empty(layout.custom_mask.numel(), dtype=torch.uint8, device=device)
            self.wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
                self.workspace, 'NHD', use_cuda_graph=True,
                qo_indptr_buf=layout.qo_indptr,
                paged_kv_indptr_buf=layout.kv_indptr,
                paged_kv_indices_buf=layout.kv_indices,
                paged_kv_last_page_len_buf=layout.last_page_len,
                custom_mask_buf=self.custom_mask_buffer,
                mask_indptr_buf=torch.empty(layout.N + 1, dtype=torch.int32, device=device),
            )
            cfg = self.head.config
            self.wrapper.plan(layout.qo_indptr, layout.kv_indptr, layout.kv_indices,
                              layout.last_page_len, cfg.num_attention_heads,
                              cfg.num_key_value_heads, cfg.head_dim, 1,
                              custom_mask=layout.custom_mask,
                              q_data_type=dtype, kv_data_type=dtype,
                              sm_scale=cfg.head_dim ** -0.5)
        elif self.backend == 'sdpa':
            allowed = layout.dense_mask(device)
            self.mask = torch.zeros(allowed.shape, device=device, dtype=dtype).masked_fill(~allowed, torch.finfo(dtype).min)[None, None]
        else:
            raise ValueError('backend must be flashinfer or sdpa')

    def refresh_context(self, contexts):
        """Refresh only context KV data, retaining the prepared plan/buffers.

        This operation can be captured after adapter outputs change. It must
        execute once per round before any drafter pass reads the context.
        """
        if (not getattr(self, 'graphs', False) or contexts.device.type != 'cuda'
                or not hasattr(self, '_context_graph')):
            self._refresh_context_eager(contexts)
            return
        if self._context_graph is None:
            self._refresh_context_eager(contexts)
            expected = [cache.index_select(0, self.layout.context_slots).clone()
                        for cache in self.kv]
            stream = torch.cuda.Stream(device=contexts.device)
            stream.wait_stream(torch.cuda.current_stream(contexts.device))
            with torch.cuda.stream(stream):
                for _ in range(3):
                    self._refresh_context_eager(contexts)
            torch.cuda.current_stream(contexts.device).wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                self._refresh_context_eager(contexts)
            graph.replay()
            if not all(torch.equal(cache.index_select(0, self.layout.context_slots), reference)
                       for cache, reference in zip(self.kv, expected)):
                raise RuntimeError('context refresh graph changed KV values')
            self._context_graph = graph
            self._context_graph_input = contexts
            return
        if contexts is not self._context_graph_input:
            self._context_graph_input.copy_(contexts)
        self._context_graph.replay()

    def _refresh_context_eager(self, contexts):
        if rows_pass_fused(contexts, run_config=self.run_config):
            self._refresh_context_fused(contexts, fused_kernels())
            return
        hidden = self.head.hidden_norm(self.head.fc(contexts))
        for layer, cache in zip(self.head.layers, self.kv):
            attn = layer.self_attn
            nh, hd = attn.config.num_key_value_heads, attn.head_dim
            k = attn.k_norm(attn.k_proj(hidden).reshape(-1, nh, hd))
            k = _rotary(k, self.ccos[0], self.csin[0])
            v = attn.v_proj(hidden).reshape(-1, nh, hd)
            cache[self.layout.context_slots, 0, 0] = k
            cache[self.layout.context_slots, 1, 0] = v

    def _refresh_context_fused(self, contexts, fused):
        # The same projections; the norm, rotary and repeat-free K/V layout
        # come from the certified kernels with one token per program row.
        head = self.head
        hidden = fused.fast_rms_norm(head.fc(contexts), head.hidden_norm.weight,
                                     head.hidden_norm.variance_epsilon)
        rows = hidden.shape[0]
        cos, sin = self.ccos[0], self.csin[0]
        if self._context_into_cache(
                fused, head, hidden, cos, sin, self.kv,
                self.layout.context_slots, self.context_span,
                _context_kv_storage(head, self.kv, run_config=self.run_config)):
            return
        for layer, cache in zip(head.layers, self.kv):
            attn = layer.self_attn
            nh, hd = attn.config.num_key_value_heads, attn.head_dim
            self._kv_into_cache(fused, attn, hidden, cos, sin, cache,
                                self.layout.context_slots, self.context_span)

    @staticmethod
    def _context_into_cache(fused, head, hidden, cos, sin, caches, slots, span, storage,
                            source_rows=None, *, run_config=None):
        from jetspec.engines.jetspec_plus_plus.profiles import kernel_enabled
        packed = getattr(head, '_context_kv', None)
        if (not kernel_enabled('ctx_kv_fused', '1')
                and config_values(run_config).get('engine.engine_settings.hybrid_context_dirty', '0') != '1') or packed is None:
            return False
        weight, bias, norms, width = packed
        projected = torch.nn.functional.linear(hidden, weight, bias)
        if source_rows is not None:
            projected = projected.index_select(0, source_rows)
            hidden = hidden.index_select(0, source_rows)
        attn = head.layers[0].self_attn
        kv_heads, head_dim = attn.config.num_key_value_heads, attn.head_dim
        if (storage is not None and hidden.is_cuda
                and hidden.dtype == torch.bfloat16 and triton is not None):
            _context_kv_postprocess_kernel[(len(caches) * hidden.shape[0] * kv_heads,)](
                projected, norms, cos, sin, storage, slots,
                tokens=hidden.shape[0], layers=len(caches), kv_heads=kv_heads,
                head_dim=head_dim, projected_row_stride=projected.stride(0),
                cos_row_stride=cos.stride(0), cache_layer_stride=storage.stride(0),
                cache_row_stride=storage.stride(1), cache_kv_stride=storage.stride(2),
                cache_start=span[0] if span is not None else 0,
                direct_span=span is not None, eps=attn.k_norm.variance_epsilon,
                BLOCK=triton.next_power_of_2(head_dim), num_warps=4,
                enable_fp_fusion=False)
            return True
        parts = projected.split(width, dim=-1)
        for index, (layer, cache) in enumerate(zip(head.layers, caches)):
            if span is None:
                attn = layer.self_attn
                key = fused.fast_rms_norm(
                    parts[2 * index].reshape(-1, kv_heads, head_dim),
                    attn.k_norm.weight, attn.k_norm.variance_epsilon)
                value = parts[2 * index + 1].reshape(-1, kv_heads, head_dim)
                _write_kv(cache, slots, None, _rotary(key, cos, sin), value)
            else:
                RowDraftHead._kv_into_cache(
                    fused, layer.self_attn, hidden, cos, sin, cache, slots, span,
                    projected=(parts[2 * index], parts[2 * index + 1]))
        return True

    def forward(self, noise_embedding):
        layout = self.layout
        if layout is None:
            raise RuntimeError('prepare the row layout before forward')
        if noise_embedding.shape[:2] != (layout.r, layout.block_size):
            raise ValueError(f'noise_embedding must have shape [r,{layout.block_size},hidden]')
        hidden = noise_embedding.reshape(-1, noise_embedding.shape[-1])
        if rows_pass_fused(hidden, run_config=self.run_config) and self.backend == 'flashinfer':
            return self._forward_fused(hidden, fused_kernels())
        for layer, cache in zip(self.head.layers, self.kv):
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
            q = _rotary(q, self.qcos[0], self.qsin[0])
            k = _rotary(k, self.qcos[0], self.qsin[0])
            cache[layout.noise_slots, 0, 0] = k
            cache[layout.noise_slots, 1, 0] = v
            if self.backend == 'flashinfer':
                output = self._attention(q, cache, attn)
            else:
                output = sdpa_attention_forward(
                    attn, q.transpose(0, 1)[None], cache[:, 0, 0].transpose(0, 1)[None],
                    cache[:, 1, 0].transpose(0, 1)[None], self.mask,
                    scaling=attn.scaling, is_causal=False,
                )[0][0]
            hidden = residual + attn.o_proj(output.reshape(
                *hidden.shape[:-1], attn.config.num_attention_heads * attn.head_dim))
            if layer.drafter_mlp:
                hidden = hidden + layer.mlp(layer.post_attention_layernorm(hidden))
        return self.head.norm(hidden).reshape(layout.r, layout.block_size, -1)

    def _attention(self, q, cache, attn):
        if getattr(self, '_jpass_len_enabled', False):
            from jetspec.engines.jetspec_plus_plus.kernels.jpass_len_attention import attend
            return attend(q, cache, self.custom_mask_buffer, self._jpass_state.lengths,
                          groups=self.N, prefix=self.P, scale=attn.head_dim ** -0.5)
        return self.wrapper.run(q, cache)

    @staticmethod
    def _kv_into_cache(fused, attn, source, cos, sin, cache, slots, span, projected=None,
                       lean=False):
        """Project, norm, rotate and write K/V for ``source`` rows into ``cache``:
        one strided kernel when the slots are one range, the certified
        postprocess plus index_put otherwise."""
        rows = source.shape[0]
        nkv, hd = attn.config.num_key_value_heads, attn.head_dim
        key, value = projected if projected is not None else (attn.k_proj(source), attn.v_proj(source))
        if span is not None:
            if lean:
                fused.jpass_kv_norm_rotary_cache(
                    key, value, attn.k_norm.weight, cos, sin,
                    cache[span[0]:span[1], 0, 0], cache[span[0]:span[1], 1, 0],
                    attn.k_norm.variance_epsilon, nkv, hd)
                return
            normalized = fused.fast_rms_norm(key.view(rows, nkv, hd), attn.k_norm.weight,
                                             attn.k_norm.variance_epsilon).view(rows, nkv * hd)
            kv_rotary_into_cache(normalized, value.contiguous(), cos, sin, cache[span[0]:span[1], 0, 0],
                                 cache[span[0]:span[1], 1, 0], kv_heads=nkv, groups=1, head_dim=hd)
            return
        k, v = fused.fast_kv_postprocess(
            key.view(rows, 1, nkv * hd), value.view(rows, 1, nkv * hd),
            attn.k_norm.weight, cos, sin, attn.k_norm.variance_epsilon, nkv, 1, hd)
        _write_kv(cache, slots, None, k.view(rows, nkv, hd), v.view(rows, nkv, hd))

    def _forward_fused(self, hidden, fused):
        """The pass layers on the certified kernels: one launch per norm, norm
        plus rotary fused for q and k, the residual add fused into the post
        norm, and a strided cache copy per tensor.  Same projections, same
        attention call, same order of additions as the reference above."""
        from jetspec.engines.jetspec_plus_plus.profiles import kernel_enabled
        layout = self.layout
        rows = hidden.shape[0]
        cos, sin = self.qcos[0], self.qsin[0]
        lean = kernel_enabled('jpass_glue_lean', '1')
        gate_up_pack = (kernel_enabled('j0_gateup_packed', '1')
                        and getattr(self, 'N', 0) == 1)
        qkv_glue = (kernel_enabled('j0_qkv_glue_fused', '1')
                    and getattr(self, 'N', 0) == 1 and lean)
        # Each layer's closing residual add is deferred into the next norm's
        # add+norm kernel (the add is the same bf16 sum, the norm the same kernel).
        pending = None
        for layer, cache in zip(self.head.layers, self.kv):
            attn = layer.self_attn
            nh, nkv, hd = attn.config.num_attention_heads, attn.config.num_key_value_heads, attn.head_dim
            if pending is None:
                normalized = fused.fast_rms_norm(hidden, layer.input_layernorm.weight,
                                                 layer.input_layernorm.variance_epsilon)
            else:
                normalized, hidden = fused.fast_add_rms_norm(
                    *pending, layer.input_layernorm.weight, layer.input_layernorm.variance_epsilon)
            residual = hidden
            if hasattr(attn, '_fused_qkv'):
                weight, bias, widths = attn._fused_qkv
                packed = torch.nn.functional.linear(normalized, weight, bias)
                qp, kp, vp = packed.split(widths, dim=-1)
                projected_kv = (kp, vp if lean else vp.contiguous())
            else:
                qp, projected_kv = attn.q_proj(normalized), None
            if qkv_glue and projected_kv is not None:
                from jetspec.engines.jetspec_plus_plus.kernels.qkv_glue import qkv_norm_rotary_cache
                q = qkv_norm_rotary_cache(
                    qp, *projected_kv, attn.q_norm.weight, attn.k_norm.weight,
                    cos, sin, cache, layout.noise_slots, self.noise_span,
                    attn.q_norm.variance_epsilon, attn.k_norm.variance_epsilon, nh, nkv, hd)
            elif lean:
                q = fused.jpass_q_norm_rotary(
                    qp, attn.q_norm.weight, cos, sin,
                    attn.q_norm.variance_epsilon, nh, hd)
            else:
                q = fused.fast_q_postprocess(
                    qp.view(rows, 1, nh * hd), attn.q_norm.weight, cos, sin,
                    attn.q_norm.variance_epsilon, nh, hd).view(rows, nh, hd)
            if not (qkv_glue and projected_kv is not None):
                self._kv_into_cache(fused, attn, normalized, cos, sin, cache, layout.noise_slots,
                                    self.noise_span, projected_kv, lean=lean)
            output = self._attention(q, cache, attn)
            update = attn.o_proj(output.reshape(rows, nh * hd))
            if layer.drafter_mlp:
                normalized, hidden = fused.fast_add_rms_norm(
                    update, residual, layer.post_attention_layernorm.weight,
                    layer.post_attention_layernorm.variance_epsilon)
                if lean:
                    mlp = layer.mlp
                    if gate_up_pack:
                        weight, bias, widths = mlp._j0_gate_up_packed
                        gate, up = torch.nn.functional.linear(normalized, weight, bias).split(widths, dim=-1)
                    else:
                        gate = mlp.gate_proj(normalized)
                        up = mlp.up_proj(normalized)
                    pending = (mlp.down_proj(fused._silu_mul_extension().silu_mul(gate, up)), hidden)
                else:
                    pending = (layer.mlp(normalized), hidden)
            else:
                pending = (update, residual)
        if pending is None:
            output = fused.fast_rms_norm(hidden, self.head.norm.weight, self.head.norm.variance_epsilon)
        else:
            output, _ = fused.fast_add_rms_norm(*pending, self.head.norm.weight, self.head.norm.variance_epsilon)
        return output.reshape(layout.r, layout.block_size, -1)


def project_row_logits(lm_head, hidden):
    """Project all r*16 block positions through one two-dimensional GEMM."""
    logits = lm_head(hidden.reshape(-1, hidden.shape[-1]))
    return logits.reshape(*hidden.shape[:-1], logits.shape[-1])


class RefineRound:
    """Static-shape adapter -> refreshed KV -> J full-pin refine passes.

    Rows are ordered by sequence, k, then branch. Each k owns its adapter
    tail; all its F branches reuse that tail and the sequence's stale prefix.
    J=1 also selects the post-exclusion top-F slot-0 candidates. Subsequent
    passes feed back all argmax tokens while keeping those candidates pinned.
    Input preparation/attention planning and graph capture are outside replay.
    Embedding, adapter, context refresh, logits and token glue are inside it.
    """

    def __init__(self, row_head, adapter, embed_tokens, lm_head, *, prefix_h,
                 adapter_ids, adapter_kwargs, initial_ids, rejected_ids, J,
                 shared_prefix_projection=True):
        self.row_head, self.adapter = row_head, adapter
        self.embed_tokens, self.lm_head = embed_tokens, lm_head
        self.layout = layout = row_head.layout
        if not layout.private_tails:
            raise ValueError('round requires private per-k adapter tails')
        if J not in (1, 3):
            raise ValueError('round J must be 1 or 3')
        self.J, self.shared_prefix_projection = J, shared_prefix_projection
        self.prefix_h, self.adapter_ids = prefix_h, adapter_ids
        self.adapter_kwargs = adapter_kwargs
        self.initial_ids, self.rejected_ids = initial_ids, rejected_ids
        m = sum(len(ks) for ks in layout.ksets)
        if initial_ids.shape != (layout.r, 16) or adapter_ids.shape != (m, 16):
            raise ValueError('round IDs must be [r,16] and [m,16]')
        if rejected_ids.shape != (m,):
            raise ValueError('one rejected token is required per adapter row')
        if len(set(layout.prefix_lengths)) != 1 or prefix_h.shape[:2] != (layout.N, layout.prefix_lengths[0]):
            raise ValueError('round requires equal-length unique sequence prefixes')
        self.tokens = torch.empty_like(initial_ids)
        self.anchors = torch.empty_like(initial_ids[:, 0])
        device = prefix_h.device
        self.owners = torch.tensor([n for n, ks in enumerate(layout.ksets) for _ in ks],device=device)
        prefix_slots, prefix_sources, tail_slots, tail_rows, tail_positions = [], [], [], [], []
        offset = adapter_row = 0
        for n, (p, c, ks) in enumerate(zip(layout.prefix_lengths, layout.context_lengths, layout.ksets)):
            prefix_slots.extend(range(offset, offset + p))
            prefix_sources.extend(range(n*p, (n+1)*p))
            tail = offset + p
            for k in ks:
                tail_slots.extend(range(tail, tail + k - 1))
                tail_rows.extend([adapter_row] * (k - 1))
                tail_positions.extend(range(k - 1))
                tail += k - 1
                adapter_row += 1
            offset += c
        for name, values in (('prefix_slots',prefix_slots),('prefix_sources',prefix_sources),
                             ('tail_slots',tail_slots),('tail_rows',tail_rows),('tail_positions',tail_positions)):
            setattr(self,name,torch.tensor(values,dtype=torch.long,device=device))
        self.contexts = torch.empty(offset,prefix_h.shape[-1],device=device,dtype=prefix_h.dtype)

    def adapt(self):
        kwargs = dict(self.adapter_kwargs, token_embeddings=self.embed_tokens(self.adapter_ids))
        if self.shared_prefix_projection:
            kwargs.update(prefix_h=self.prefix_h,h_prev=self.prefix_h[:,-1:,:],
                          shared_prefix_projection=True,prefix_row_indices=self.owners)
        else:
            prefix = self.prefix_h.index_select(0,self.owners)
            kwargs.update(prefix_h=prefix,h_prev=prefix[:,-1:,:].expand(-1,16,-1))
        adapted = self.adapter(**kwargs)
        marker = self.row_head.head.adapted_position_embedding.weight
        self.contexts.index_copy_(0,self.prefix_slots,self.prefix_h.reshape(-1,self.prefix_h.shape[-1])[self.prefix_sources] + marker[0])
        self.contexts.index_copy_(0,self.tail_slots,adapted[self.tail_rows,self.tail_positions] + marker[1])
        self.row_head.refresh_context(self.contexts)
        return adapted

    def draft_step(self, index):
        inputs = self.initial_ids if index == 0 else self.tokens
        hidden = self.row_head(self.embed_tokens(inputs))
        return hidden, project_row_logits(self.lm_head,hidden)

    def update_tokens(self, index, logits):
        if index == 0:
            scores = logits[::self.layout.F,0].clone()
            scores.scatter_(1,self.rejected_ids[:,None],float('-inf'))
            self.anchors.copy_(scores.topk(self.layout.F,dim=-1).indices.reshape(-1))
        self.tokens.copy_(logits.argmax(-1))
        self.tokens[:,0].copy_(self.anchors)

    def pass_step(self, index):
        hidden, logits = self.draft_step(index)
        self.update_tokens(index,logits)
        return hidden, logits

    def eager(self):
        adapted = self.adapt()
        for index in range(self.J):
            hidden, logits = self.pass_step(index)
        return adapted, hidden, logits

    @torch.inference_mode()
    def capture(self, scope):
        """One-round captures glue too; per-pass retains eager token glue."""
        import time
        if scope not in ('per-pass','one-round'):
            raise ValueError('graph scope must be per-pass or one-round')
        torch.cuda.synchronize()
        started = time.perf_counter()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                self.eager()
        torch.cuda.current_stream().wait_stream(stream)
        graphs = []
        steps = []
        pool = torch.cuda.graph_pool_handle()
        if scope == 'one-round':
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph,pool=pool):
                output = self.eager()
            graphs.append(graph)
            steps.append(graph.replay)
        else:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph,pool=pool):
                adapted = self.adapt()
            graphs.append(graph)
            steps.append(graph.replay)
            for index in range(self.J):
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph,pool=pool):
                    hidden, logits = self.draft_step(index)
                graphs.append(graph)
                steps.append(graph.replay)
                glue = lambda index=index, logits=logits: self.update_tokens(index,logits)
                steps.append(glue)
                glue()
            output = adapted, hidden, logits
        torch.cuda.synchronize()
        return CapturedRefineRound(graphs,steps,output,(time.perf_counter()-started)*1000,
                                   (self.layout.N,tuple(map(len,self.layout.ksets)),self.layout.F,self.J))


class CapturedRefineRound:
    def __init__(self, graphs, steps, output, capture_ms, shape):
        self.graphs, self.steps, self.output = graphs, steps, output
        self.capture_ms, self.shape = capture_ms, shape

    def replay(self):
        for step in self.steps:
            step()
        return self.output


class BucketRowDraftHead(RowDraftHead):
    """Persistent KV, rotary and mask buffers for a fixed (N,H,prefix) bucket.

    Padding is masked; actual prefix/query positions are refreshed each call.
    Planning uses fixed page-size-one geometry, so changing lengths only changes
    the packed custom mask and rotary data, never the attention plan pointers.
    """
    @torch.inference_mode()
    def __init__(self, head, *, N, H, prefix_bucket, backend='flashinfer', F=1,
                 row_capacity=None, flat=False, graphs=False, refresh=False, tail_rows=15,
                 block_size=16, run_config=None):
        from jetspec.engines.jetspec_plus_plus.profiles import kernel_enabled
        self.run_config = run_config
        super().__init__(head,backend=backend,workspace_bytes_per_row=4*1024*1024,
                         run_config=run_config)
        self.refresh = bool(refresh)
        self._refresh_mark = False
        # serving: the staged tail rows (15; 16 with 15 drafts, flat only; serving: graphed too)
        self.T = int(tail_rows)
        self.B = int(block_size)
        if self.T != 15 and not flat:
            raise ValueError('a 16-row tail serves flat rows only')
        if self.refresh and not flat:
            raise ValueError('the adapter refresh requires flat rows')
        self.graphs = bool(graphs)
        self._geometry_graph = None
        self._context_graph = None
        self.N, self.H, self.P = N, H, prefix_bucket
        if N == 1 and (((kernel_enabled('j0_gateup_packed', '1')))):
            from jetspec.engines.jetspec_plus_plus.kernels.projections import prepare_j0_gate_up
            for layer in head.layers:
                if layer.drafter_mlp:
                    prepare_j0_gate_up(layer.mlp)
        self.flat = bool(flat)
        self.F, self.R = F, row_capacity or H*F
        parameter=next(head.parameters())
        self.device,self.dtype=parameter.device,parameter.dtype
        # Max k=15 reserves all staged tail slots regardless of the chosen set.
        if self.flat:
            if F != 1 or self.R != H:
                raise ValueError('flat row buckets require F=1 and R=H')
            layout = RowLayout.flat([prefix_bucket] * N,
                [[(0, row) for row in range(H)] for _ in range(N)],
                row_capacity=H, device=self.device, vectorized=True,
                refresh_slots=H * self.B if self.refresh else 0, tail_rows=self.T,
                block_size=self.B)
        else:
            layout=RowLayout([prefix_bucket]*N,[list(range(17-H,17))]*N,
                             device=self.device,vectorized=True,F=F,row_capacity=row_capacity)
        self.contexts=torch.zeros(N*(prefix_bucket+self.T),head.fc.in_features,
                                  device=self.device,dtype=self.dtype)
        self.prepare(layout,self.contexts)
        if refresh and kernel_enabled('refresh_inject_lean2', '1'):
            from jetspec.engines.jetspec_plus_plus.jetspec.profile import enabled as profile_enabled
            supported_widths = (16, 32) if profile_enabled() else (16,)
            if N != 1 or self.B not in supported_widths:
                raise ValueError('REFRESH_INJECT_LEAN2 requires N=1 fixed 16-slot refresh (32 in eval)')
            if self.device.type == 'cuda' and self.dtype != torch.bfloat16:
                raise ValueError('REFRESH_INJECT_LEAN2 requires BF16 drafter weights')
            from jetspec.engines.jetspec_plus_plus.jetspec.refresh_inject import prepare
            prepare(head)
            self._refresh_inject_storage = torch.stack(self.kv)
            self.kv = _context_kv_views(self._refresh_inject_storage, run_config=self.run_config)
            if self._context_graph is not None:
                # prepare() captured the old cache addresses; rebuild after
                # installing the shared allocation used by the inject writer.
                self._context_graph = None
                self.refresh_context(self.contexts)
        # The measured short bucket uses one kernel. Long buckets retain the
        # existing split-KV path until a measured crossover says otherwise.
        self._jpass_len_enabled = (backend == 'flashinfer' and
                                   kernel_enabled('jpass_attn_len', '1') and
                                   layout.num_kv // N <= 768)
        if self._jpass_len_enabled:
            self._jpass_state = SimpleNamespace(lengths=None)
        self.columns=torch.arange(prefix_bucket+self.T+self.R*self.B+layout.refresh_slots,device=self.device)
        if self.refresh:
            start = prefix_bucket + self.T + self.R * self.B
            group = start + self.R * self.B
            self.refresh_slots = (torch.arange(N, device=self.device)[:, None] * group
                                  + torch.arange(start, group, device=self.device)[None]).flatten()
            self._refresh_start = start  # host copy: inject_refresh reads no device value
            for cache in self.kv:
                cache.index_fill_(0, self.refresh_slots, 0)  # hidden until injected
        self.queries=torch.arange(self.R*self.B,device=self.device)
        self.prefix_positions=torch.arange(prefix_bucket,device=self.device)
        self.tail_positions=torch.arange(self.T,device=self.device)
        self.hidden_stub=torch.empty(1,1,head.config.hidden_size,device=self.device,dtype=self.dtype)
        # JETSPEC_ROWS_CTX_INCR: the request and length whose context K/V this bucket holds
        self._ctx_key = self._ctx_length = self._ctx_mark = None
        self.ctx_full_updates = self.ctx_incremental_updates = 0
        self.last_ctx_update, self.last_ctx_rows = 'none', 0
        if context_append_enabled(run_config=self.run_config) and self.N == 1 and self.flat:
            self._chain_ctx_length = torch.zeros((), dtype=torch.long, device=self.device)
            self._chain_ctx_previous = torch.zeros((), dtype=torch.long, device=self.device)
            self._chain_ctx_dirty_source = torch.zeros(self.B + self.T, dtype=torch.long,
                                                       device=self.device)
            self._chain_ctx_dirty_slots = torch.zeros_like(self._chain_ctx_dirty_source)

    def _incremental_context(self, prefix, tails, length, mark):
        """JETSPEC_ROWS_CTX_INCR=1: the hypotheses only set the query rotary rows and
        the mask, never a context row's K/V.  So a later round of the same request
        whose prefix only grew writes the appended prefix rows and the 15 tail rows
        into the same context rows the full update writes (the same values: the
        padding rows past the prefix keep the bucket's earlier zero-plus-marker rows,
        which the full update also rewrites to zero plus marker) and projects only
        those rows.  Every per-row step (fc, norm, k/v projections, rotary) is the
        full refresh's own operation on a subset of its rows, but NOT bitwise: a GEMM
        row's value depends on the call's row count (another cuBLAS kernel for 15 to
        ~50 rows than for P + 15), so this is a measurement arm, not an exact path."""
        old = self._ctx_length
        context = self.contexts.view(1, self.P + self.T, -1)
        if length > old:
            context[0, old:length].copy_(prefix[old:length])
            if mark:
                context[0, old:length].add_(self.head.adapted_position_embedding.weight[0])
        context[:, self.P:].copy_(tails)
        if mark:
            context[:, self.P:].add_(self.head.adapted_position_embedding.weight[1])
        return [(old, length), (self.P, self.P + self.T)] if length > old else [(self.P, self.P + self.T)]

    def _refresh_context_rows(self, spans):
        """Project and write the context K/V of the context rows in ``spans`` only."""
        rows = torch.cat([self.contexts[a:b] for a, b in spans]) if len(spans) > 1 else \
            self.contexts[spans[0][0]:spans[0][1]]
        head = self.head
        fused = fused_kernels() if rows_pass_fused(rows, run_config=self.run_config) else None
        if fused is not None:
            hidden = fused.fast_rms_norm(head.fc(rows), head.hidden_norm.weight,
                                         head.hidden_norm.variance_epsilon)
        else:
            hidden = head.hidden_norm(head.fc(rows))
        cos, sin = self.ccos[0], self.csin[0]
        starts, offset = [], 0
        for a, b in spans:
            starts.append((a, b, offset))
            offset += b - a
        for layer, cache in zip(head.layers, self.kv):
            attn = layer.self_attn
            nh, hd = attn.config.num_key_value_heads, attn.head_dim
            if fused is not None:
                key, value = attn.k_proj(hidden), attn.v_proj(hidden)
                normalized = fused.fast_rms_norm(key.view(-1, nh, hd), attn.k_norm.weight,
                                                 attn.k_norm.variance_epsilon).view(-1, nh * hd)
                for a, b, o in starts:
                    kv_rotary_into_cache(normalized[o:o + b - a], value[o:o + b - a], cos[a:b], sin[a:b],
                                         cache[a:b, 0, 0], cache[a:b, 1, 0],
                                         kv_heads=nh, groups=1, head_dim=hd)
            else:
                k = attn.k_norm(attn.k_proj(hidden).reshape(-1, nh, hd))
                v = attn.v_proj(hidden).reshape(-1, nh, hd)
                for a, b, o in starts:
                    cache[a:b, 0, 0] = _rotary(k[o:o + b - a], cos[a:b], sin[a:b])
                    cache[a:b, 1, 0] = v[o:o + b - a]

    @torch.inference_mode()
    def update(self, prefixes, tails, hypotheses, *, mark=False, prefix_key=None, early=None):
        expected = self.N * self.H if self.flat and self.N > 1 else self.H
        if len(prefixes)!=self.N or len(hypotheses)!=expected:
            raise ValueError('bucket update must supply padded N sequences and hypotheses')
        if any(p.shape[0]>self.P for p in prefixes):
            raise ValueError('prefix exceeds bucket')
        self._refresh_mark = bool(mark)
        if (early is not None and mark and self.N == 1 and self.flat and not self.graphs
                and early.P == self.P and early.ready_for(prefix_key, prefixes[0].shape[0])
                and getattr(self, 'context_span', None) == (0, self.P + self.T)):
            # JETSPEC_ROWS_CTX_EARLY: the context K/V was projected at the round's start
            self._update_geometry(device_tensor([prefixes[0].shape[0]], self.device),
                                  device_tensor(hypotheses, self.device))
            early.copy_into(self)
            self._ctx_key = self._ctx_length = self._ctx_mark = None
            self.ctx_full_updates += 1
            self.last_ctx_update, self.last_ctx_rows = 'early', self.P + self.T
            return
        if (early is not None and mark and self.N > 1 and self.flat and not self.graphs
                and early.P == self.P and early.N == self.N and early.T == self.T
                and early.ready_for(prefix_key, [p.shape[0] for p in prefixes])):
            ks = (hypotheses if isinstance(hypotheses, torch.Tensor) and hypotheses.device == self.device
                  else device_tensor(hypotheses, self.device))
            self._update_geometry(device_tensor([p.shape[0] for p in prefixes], self.device), ks)
            early.copy_into(self)
            self._ctx_key = self._ctx_length = self._ctx_mark = None
            self.ctx_full_updates += 1
            self.last_ctx_update, self.last_ctx_rows = 'early', self.N * (self.P + self.T)
            return
        if (prefix_key is not None and rows_ctx_incr(run_config=self.run_config) and self.N == 1 and self.flat
                and not self.graphs and getattr(self, 'context_span', None) == (0, self.P + self.T)):
            length = prefixes[0].shape[0]
            if (self._ctx_key is prefix_key and self._ctx_mark == bool(mark)
                    and self._ctx_length is not None and length >= self._ctx_length):
                spans = self._incremental_context(prefixes[0], tails, length, mark)
                self._update_geometry(device_tensor([length], self.device),
                                      device_tensor(hypotheses, self.device))
                self._refresh_context_rows(spans)
                self._ctx_length = length
                self.ctx_incremental_updates += 1
                self.last_ctx_update = 'incremental'
                self.last_ctx_rows = sum(b - a for a, b in spans)
                return
            self._ctx_key, self._ctx_mark, self._ctx_length = prefix_key, bool(mark), length
        else:
            self._ctx_key = self._ctx_length = self._ctx_mark = None
        self.ctx_full_updates += 1
        self.last_ctx_update, self.last_ctx_rows = 'full', self.N * (self.P + self.T)
        lengths=device_tensor([p.shape[0] for p in prefixes],self.device)
        ks=(hypotheses if self.N > 1 and isinstance(hypotheses, torch.Tensor) and hypotheses.device == self.device
            else device_tensor(hypotheses,self.device))
        if self.F != 1 or self.R != self.H:
            ks=ks.repeat_interleave(self.F)
            ks=torch.cat((ks,ks.new_zeros(self.R-ks.numel())))
        padded=torch.nn.utils.rnn.pad_sequence(prefixes,batch_first=True)
        context=self.contexts.view(self.N,self.P+self.T,-1)
        context.zero_()
        context[:,:padded.shape[1]].copy_(padded)
        context[:,self.P:].copy_(tails)
        if mark:
            context[:,:self.P].add_(self.head.adapted_position_embedding.weight[0])
            context[:,self.P:].add_(self.head.adapted_position_embedding.weight[1])
        self._update_geometry(lengths, ks)
        self.refresh_context(self.contexts)

    @torch.inference_mode()
    def update_chain(self, prefix_padded, tails, length, ks, *, mark=True,
                     incremental=False):
        """Fixed-P context copy and one device-length geometry update for N=1."""
        from jetspec.engines.jetspec_plus_plus.profiles import kernel_enabled
        if self.N != 1 or prefix_padded.shape != (self.P, self.contexts.shape[-1]) or \
                tails.shape != (1, self.T, self.contexts.shape[-1]):
            raise ValueError('chain context requires one fixed-P prefix and one fixed tail')
        self._refresh_mark = bool(mark)
        context = self.contexts.view(1, self.P + self.T, -1)
        if kernel_enabled('chain_sega_glue', '2') and mark:
            from jetspec.engines.jetspec_plus_plus.draft.glue import context_copy_mark
            context_copy_mark(length, prefix_padded, tails,
                              self.head.adapted_position_embedding.weight, context)
        else:
            valid = torch.arange(self.P, device=self.contexts.device) < length.reshape(())
            context[:, :self.P].copy_(torch.where(valid[None, :, None], prefix_padded[None],
                                                  torch.zeros_like(prefix_padded)[None]))
            context[:, self.P:].copy_(tails)
            if mark:
                context[:, :self.P].add_(self.head.adapted_position_embedding.weight[0])
                context[:, self.P:].add_(self.head.adapted_position_embedding.weight[1])
        self._prepare_geometry(length.reshape(1), ks)
        if incremental:
            source, slots = chain_context_dirty_rows(
                self._chain_ctx_length, length, prefix=self.P, tail=self.T,
                block=self.B, noise_start=self.P + self.T)
            self._chain_ctx_previous.copy_(self._chain_ctx_length)
            self._chain_ctx_dirty_source.copy_(source)
            self._chain_ctx_dirty_slots.copy_(slots)
            self._refresh_chain_dirty(source, slots)
        else:
            self._refresh_context_eager(self.contexts)
        if hasattr(self, '_chain_ctx_length'):
            self._chain_ctx_length.copy_(length.reshape(()))

    def _refresh_chain_dirty(self, source, slots):
        # Changing M changes BLAS reduction/rounding on real 27B BF16 shapes.
        # Keep the reference projection geometry and select only cache writes.
        preserve_projection = config_values(self.run_config).get('engine.engine_settings.hybrid_context_dirty', '0') == '1'
        hidden = self.contexts if preserve_projection else self.contexts.index_select(0, source)
        cos = self.ccos[0].index_select(0, source)
        sin = self.csin[0].index_select(0, source)
        if rows_pass_fused(hidden, run_config=self.run_config):
            fused = fused_kernels()
            state = fused.fast_rms_norm(self.head.fc(hidden),
                                        self.head.hidden_norm.weight,
                                        self.head.hidden_norm.variance_epsilon)
            if not self._context_into_cache(
                    fused, self.head, state, cos, sin, self.kv, slots, None,
                    _context_kv_storage(self.head, self.kv, run_config=self.run_config),
                    source_rows=source if preserve_projection else None):
                for layer, cache in zip(self.head.layers, self.kv):
                    attn = layer.self_attn
                    nh, hd = attn.config.num_key_value_heads, attn.head_dim
                    key, value = attn.k_proj(state), attn.v_proj(state)
                    if preserve_projection:
                        key, value = key.index_select(0, source), value.index_select(0, source)
                    normalized = fused.fast_rms_norm(
                        key.view(-1, nh, hd), attn.k_norm.weight,
                        attn.k_norm.variance_epsilon).view(-1, nh * hd)
                    kv_rotary_into_cache_slots(
                        normalized, value.contiguous(), cos, sin,
                        cache[:, 0, 0], cache[:, 1, 0], slots,
                        kv_heads=nh, head_dim=hd)
        else:
            state = self.head.hidden_norm(self.head.fc(hidden))
            for layer, cache in zip(self.head.layers, self.kv):
                attn = layer.self_attn
                nh, hd = attn.config.num_key_value_heads, attn.head_dim
                k = attn.k_norm(attn.k_proj(state).reshape(-1, nh, hd))
                v = attn.v_proj(state).reshape(-1, nh, hd)
                if preserve_projection:
                    k, v = k.index_select(0, source), v.index_select(0, source)
                cache[slots, 0, 0] = _rotary(k, cos, sin)
                cache[slots, 1, 0] = v

    @torch.inference_mode()
    def check_chain_context_kv(self, *, incremental=False):
        """Re-project into private storage after replay and compare every context slot."""
        original = self.kv
        if hasattr(self.head, '_context_kv'):
            scratch = list(_context_kv_storage(self.head, original, run_config=self.run_config).clone().unbind(0))
        else:
            scratch = [cache.clone() for cache in original]
        try:
            self.kv = scratch
            self._refresh_context_eager(self.contexts)
        finally:
            self.kv = original
        full_context = [cache.index_select(0, self.layout.context_slots)
                        for cache in scratch]
        width = 16 if self.dtype == torch.bfloat16 else 32
        bits = torch.int16 if width == 16 else torch.int32
        mask = (1 << width) - 1
        sign = 1 << (width - 1)

        def ordered(value):
            raw = value.contiguous().view(bits).to(torch.int64) & mask
            return torch.where((raw & sign) != 0,
                               sign - (raw & (sign - 1)), sign + raw)

        if incremental:
            expected_source, expected_slots = chain_context_dirty_rows(
                self._chain_ctx_previous, self._chain_ctx_length,
                prefix=self.P, tail=self.T, block=self.B, noise_start=self.P + self.T)
            selection_exact = (torch.equal(self._chain_ctx_dirty_source, expected_source) and
                               torch.equal(self._chain_ctx_dirty_slots, expected_slots))
            dirty_mask = torch.zeros(self.P + self.T, dtype=torch.bool, device=self.device)
            live_slots = self._chain_ctx_dirty_slots[
                self._chain_ctx_dirty_slots < self.P + self.T]
            dirty_mask[live_slots] = True
            try:
                self.kv = scratch
                self._refresh_chain_dirty(self._chain_ctx_dirty_source,
                                          self._chain_ctx_dirty_slots)
            finally:
                self.kv = original
        else:
            selection_exact = True
        # Keep an independent reference with the same full/dirty projection
        # history as the live cache. A retained row's measured GEMM noise need
        # not fit the current round's dirty-row noise maximum.
        references = getattr(self, '_chain_ctx_check_reference', None)
        if not incremental or references is None:
            references = [value.clone() for value in full_context]
        report = []
        for index, (actual_cache, scratch_cache) in enumerate(zip(original, scratch)):
            actual = actual_cache.index_select(0, self.layout.context_slots)
            expected = full_context[index]
            exact = torch.equal(actual, expected)
            dirty_reference = scratch_cache.index_select(0, self.layout.context_slots)
            reference = references[index]
            if incremental:
                reference[dirty_mask] = dirty_reference[dirty_mask]
            recompute_exact = (torch.equal(actual, reference) if incremental else exact)
            unchanged_exact = (torch.equal(actual[~dirty_mask], reference[~dirty_mask])
                               if incremental else exact)
            difference = (actual.float() - expected.float()).abs()
            ulp_diff = (ordered(actual) - ordered(expected)).abs()
            if incremental:
                # Neither reference uses live cache values: the difference is
                # full-M versus independently replayed full/dirty-M projections.
                noise_abs = (reference.float() - expected.float()).abs().reshape(-1)
                noise_ulp = (ordered(reference) - ordered(expected)).abs().reshape(-1)
                noise_max_abs = float(noise_abs.max().item())
                noise_p99_abs = float(torch.quantile(noise_abs, 0.99).item())
                noise_max_ulp = int(noise_ulp.max().item())
                noise_p99_ulp = float(torch.quantile(noise_ulp.float(), 0.99).item())
                over_floor = difference.reshape(self.P + self.T, -1).amax(1) > 2 * noise_max_abs
                row_norm = torch.linalg.vector_norm(difference.reshape(self.P + self.T, -1), dim=1)
                reference_norm = torch.linalg.vector_norm(
                    expected.float().reshape(self.P + self.T, -1), dim=1)
                relative = row_norm / reference_norm.clamp_min(1e-12)
                over_relative = relative > 1e-2
                stale = (actual != reference).reshape(self.P + self.T, -1).any(1)
                offending = over_floor | over_relative | stale
                offending_count = int(offending.sum().item())
                offending_row = int(offending.int().argmax().item()) if offending_count else -1
                relative_count = int(over_relative.sum().item())
                max_relative = float(relative.max().item())
                within_floor = not bool(over_floor.any().item())
            else:
                noise_max_abs = noise_p99_abs = noise_p99_ulp = 0.0
                noise_max_ulp = relative_count = offending_count = 0
                max_relative = 0.0
                offending_row = -1
                within_floor = exact
            accepted = (exact or (incremental and selection_exact and recompute_exact and
                                  within_floor and relative_count == 0))
            max_index = int(ulp_diff.reshape(-1).argmax().item())
            ulp = int(ulp_diff.reshape(-1)[max_index].item())
            per_row = difference.reshape(difference.shape[0], -1).amax(dim=1)
            report.append(dict(exact=exact, accepted=accepted,
                               recompute_exact=recompute_exact,
                               unchanged_exact=unchanged_exact,
                               noise_max_abs=noise_max_abs,
                               noise_p99_abs=noise_p99_abs,
                               noise_max_ulp=noise_max_ulp,
                               noise_p99_ulp=noise_p99_ulp,
                               within_floor=within_floor,
                               relative_count=relative_count,
                               max_relative=max_relative,
                               offending_count=offending_count,
                               offending_row=offending_row,
                               max_abs=difference.max().item(),
                               max_row=int(per_row.argmax().item()), max_ulp=ulp,
                               selection_exact=selection_exact,
                               ulp_row=max_index // (actual.numel() // actual.shape[0]),
                               ulp_values=(float(actual.reshape(-1)[max_index].item()),
                                           float(expected.reshape(-1)[max_index].item()))))
        self._chain_ctx_check_reference = references
        return report

    def _update_geometry(self, lengths, ks):
        if not self.graphs or self.device.type != 'cuda':
            self._prepare_geometry(lengths, ks)
            return
        if self._geometry_graph is None:
            self._geometry_inputs = (lengths.clone(), ks.clone())
            self._prepare_geometry(*self._geometry_inputs)
            buffers = (self.ccos, self.csin, self.qcos, self.qsin,
                       self.custom_mask_buffer if self.backend == 'flashinfer' else self.mask)
            expected = tuple(value.clone() for value in buffers)
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    self._prepare_geometry(*self._geometry_inputs)
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                self._prepare_geometry(*self._geometry_inputs)
            graph.replay()
            if not all(torch.equal(actual, reference)
                       for actual, reference in zip(buffers, expected)):
                raise RuntimeError('row geometry graph changed rotary or mask values')
            self._geometry_graph = graph
            return
        for target, source in zip(self._geometry_inputs, (lengths, ks)):
            target.copy_(source)
        self._geometry_graph.replay()

    def _prepare_geometry(self, lengths, ks):
        if self._jpass_len_enabled:
            self._jpass_state.lengths = lengths
        if self.B == 16 and not self.graphs and self.backend == 'flashinfer':
            self._prepare_geometry_cached(lengths, ks)
            return
        cpos=torch.cat((self.prefix_positions.expand(self.N,-1),lengths[:,None]+self.tail_positions),dim=1).flatten()
        row_ks = ks.reshape(self.N, self.R) if self.flat and self.N > 1 else ks[None, :].expand(self.N, -1)
        qpos=(lengths[:,None,None]+row_ks[:,:,None]+torch.arange(self.B,device=self.device)).flatten()
        cc,cs=self.head.rotary_emb(self.hidden_stub,cpos[None])
        qc,qs=self.head.rotary_emb(self.hidden_stub,qpos[None])
        self.ccos.copy_(cc);self.csin.copy_(cs);self.qcos.copy_(qc);self.qsin.copy_(qs)
        col=self.columns[None,None,:]
        row=self.queries//self.B
        c=self.P+self.T
        prefix_allowed=col<lengths[:,None,None]
        tail_allowed=(col>=self.P)&(col<self.P+row_ks[:,row,None])
        block_allowed=((col-c)//self.B==row[None,:,None])&((col-c)%self.B<=(self.queries%self.B)[None,:,None])&(col>=c)
        allowed=prefix_allowed|tail_allowed|block_allowed
        if self.backend=='flashinfer':
            # Every group mask contains H*16*L bits, hence byte-aligned. The
            # constructor's public custom_mask_buf is the persistent plan input.
            bits=allowed.flatten().reshape(-1,8).to(torch.uint8)
            packed=(bits*(1<<torch.arange(8,device=self.device))).sum(-1).to(torch.uint8)
            self.custom_mask_buffer[:packed.numel()].copy_(packed)
        else:
            dense=torch.block_diag(*allowed.unbind(0))
            self.mask.copy_(torch.zeros_like(self.mask).masked_fill_(~dense[None,None],torch.finfo(self.dtype).min))


    def _prepare_geometry_cached(self, lengths, ks):
        """Eager geometry from per-bucket tables: rotary rows are gathered from
        one cos/sin table (each position's value is one product, so the gather
        is bitwise the rotary_emb call), and the mask is the OR of packed banks.
        The prefix rows of ccos/csin never change after prepare()."""
        from jetspec.engines.jetspec_plus_plus.profiles import kernel_enabled
        if getattr(self, '_rope_table', None) is None:
            positions = torch.arange(self.P + 32, device=self.device)
            cos, sin = self.head.rotary_emb(self.hidden_stub, positions[None])
            self._rope_table = (cos[0], sin[0])
            self._block_positions = torch.arange(16, device=self.device)
            from jetspec.engines.jetspec_plus_plus.jetspec.profile import enabled as profile_enabled
            self._mask_banks = packed_mask_banks(P=self.P, R=self.R, columns=self.columns, queries=self.queries,
                                                tail_rows=self.T, compact=profile_enabled())
        cos, sin = self._rope_table
        tail = (lengths[:, None] + self.tail_positions).flatten()
        row_ks = ks.reshape(self.N, self.R) if self.flat and self.N > 1 else ks[None, :].expand(self.N, -1)
        qpos = (lengths[:, None, None] + row_ks[:, :, None] + self._block_positions).flatten()
        width = cos.shape[-1]
        if config_values(self.run_config).get('engine.engine_settings.chain_gather_vec', '0') == '1':
            tail_cos, tail_sin = cos[tail], sin[tail]
        else:
            tail_cos, tail_sin = cos.index_select(0, tail), sin.index_select(0, tail)
        self.ccos.view(self.N, self.P + self.T, width)[:, self.P:].copy_(tail_cos.view(self.N, self.T, width))
        self.csin.view(self.N, self.P + self.T, width)[:, self.P:].copy_(tail_sin.view(self.N, self.T, width))
        self.qcos[0].copy_(cos.index_select(0, qpos))
        self.qsin[0].copy_(sin.index_select(0, qpos))
        packed = packed_mask(self._mask_banks, lengths, row_ks if self.N > 1 else ks)
        self.custom_mask_buffer[:packed.numel()].copy_(packed)

    @torch.inference_mode()
    def inject_refresh(self, hidden, khat, entries=None, reset=False, fixed_layout=False, run_config=None):
        """JETSPEC_ADAPTER_REFRESH: write each row's refreshed block entries as extra
        context KV and expose them to the round's remaining passes.

        ``hidden`` [R,16,H] is the adapter's row for block slot j of row r, at that
        slot's own position (the row's query position), with the adapted marker as
        the stale tail rows carry it.  Row r's slot i reads entry j iff j < i and
        j <= ``khat[r]``.  The next update() rewrites the whole mask, hiding them.
        ``entries`` (JETSPEC_REFRESH_INJECT_PACK): the flat r*16+j indices to write,
        at least every entry the mask exposes; the other slots keep stale, hidden rows.
        ``reset`` (a second refresh in the round): hide every earlier refresh entry
        first, so the mask exposes exactly this call's entries.
        """
        from jetspec.engines.jetspec_plus_plus.profiles import kernel_enabled
        if not self.refresh:
            raise RuntimeError('bucket was built without refresh slots')
        head, rows = self.head, self.N * self.R * self.B
        if tuple(hidden.shape[:2]) != (self.N * self.R, self.B):
            raise ValueError(f'refresh entries must be [N*R,{self.B},hidden]')
        if kernel_enabled('refresh_inject_lean2', '1'):
            if (self.N != 1 or not fixed_layout or entries is not None or
                    config_values(self.run_config).get('engine.engine_settings.refresh_inject_inclusive', '0') != '1'):
                raise ValueError('REFRESH_INJECT_LEAN2 requires N=1 fixed INCLUSIVE refresh')
            from jetspec.engines.jetspec_plus_plus.jetspec.refresh_inject import inject
            inject(self, hidden, khat, reset, run_config=run_config)
            return
        context = hidden.reshape(rows, -1)
        slots, cos, sin = self.refresh_slots, self.qcos[0], self.qsin[0]
        inclusive = (self.N == 1 and
                     config_values(self.run_config).get('engine.engine_settings.refresh_inject_inclusive', '0') == '1')
        if fixed_layout:
            flat = torch.arange(rows, device=self.device)
            limit = (khat + 1).clamp(max=self.B - 1) if inclusive else khat
            visible = flat.remainder(self.B) < limit[flat // self.B]
            # The final block slot is never visible to a later query. It is the
            # scratch destination for every padded or hidden refresh entry.
            slots = torch.where(visible, slots, slots[self.B - 1])
            context = torch.where(visible[:, None], context, torch.zeros_like(context))
        elif entries is not None:
            context = context.index_select(0, entries)
            slots, cos, sin = slots[entries], cos[entries], sin[entries]
        context = context.to(self.dtype)
        if self._refresh_mark:
            context = context + head.adapted_position_embedding.weight[1]
        state = head.hidden_norm(head.fc(context))
        if refresh_inject_lean() and self._inject_lean_ready():
            self._inject_kv_lean(state, slots, cos, sin)
        else:
            self._inject_kv(state, slots, cos, sin)
        self._inject_mask(khat - 1 if fixed_layout and not inclusive else khat, rows, reset)

    def _inject_kv(self, state, slots, cos, sin):
        head = self.head
        for layer, cache in zip(head.layers, self.kv):
            attn = layer.self_attn
            nh, hd = attn.config.num_key_value_heads, attn.head_dim
            k = attn.k_norm(attn.k_proj(state).reshape(-1, nh, hd))
            k = _rotary(k, cos, sin)
            v = attn.v_proj(state).reshape(-1, nh, hd)
            cache[slots, 0, 0] = k
            cache[slots, 1, 0] = v

    def _inject_lean_ready(self):
        """The batched k norm is Qwen3RMSNorm.forward's own op sequence: every layer's k_norm
        must be that plain module with one epsilon, and k/v projections one width."""
        if getattr(self, '_inject_lean_ok', None) is None:
            norms = [layer.self_attn.k_norm for layer in self.head.layers]
            plain = all(type(norm).forward is type(norms[0]).forward and
                        type(norm).__name__ == 'Qwen3RMSNorm' for norm in norms)
            eps = {float(norm.variance_epsilon) for norm in norms}
            shapes = {(layer.self_attn.config.num_key_value_heads, layer.self_attn.head_dim)
                      for layer in self.head.layers}
            self._inject_lean_ok = plain and len(eps) == 1 and len(shapes) == 1
            if self._inject_lean_ok:
                self._k_norm_weights = torch.stack([norm.weight for norm in norms])[:, None, None, :]
                self._k_norm_eps = eps.pop()
        return self._inject_lean_ok

    def _inject_kv_lean(self, state, slots, cos, sin):
        layers = self.head.layers
        nh, hd = layers[0].self_attn.config.num_key_value_heads, layers[0].self_attn.head_dim
        keys = torch.stack([layer.self_attn.k_proj(state) for layer in layers]).view(len(layers), -1, nh, hd)
        values = [layer.self_attn.v_proj(state).view(-1, nh, hd) for layer in layers]
        # Qwen3RMSNorm.forward, op for op, over the five layers at once (per-layer weights)
        hidden = keys.to(torch.float32)
        variance = hidden.pow(2).mean(-1, keepdim=True)
        hidden = hidden * torch.rsqrt(variance + self._k_norm_eps)
        keys = _rotary(self._k_norm_weights * hidden.to(keys.dtype), cos, sin)
        for index, cache in enumerate(self.kv):
            cache[:, 0, 0].index_copy_(0, slots, keys[index])
            cache[:, 1, 0].index_copy_(0, slots, values[index])

    def _inject_mask(self, khat, rows, reset):
        if self.N > 1:
            if khat.numel() != self.N * self.R:
                raise ValueError('one refresh count is required per batched row')
            width, start = self.R * self.B, self._refresh_start
            group = start + width
            query = torch.arange(width, device=self.device)
            slot = query % self.B
            row = query // self.B
            entry = torch.arange(width, device=self.device)
            own = row[:, None] == entry[None, :] // self.B
            earlier = entry[None, :] % self.B < slot[:, None]
            visible = (entry[None, :] % self.B <= khat.reshape(self.N, self.R)[:, row, None])
            allowed = torch.zeros(self.N, width, group, device=self.device, dtype=torch.bool)
            allowed[:, :, start:] = own[None] & earlier[None] & visible
            if self.backend == 'flashinfer':
                packed = pack_mask_bits(allowed)
                if reset:
                    if getattr(self, '_refresh_region_bits', None) is None:
                        region = torch.zeros_like(allowed)
                        region[:, :, start:] = True
                        self._refresh_region_bits = pack_mask_bits(region).bitwise_not_()
                    self.custom_mask_buffer[:packed.numel()].bitwise_and_(self._refresh_region_bits)
                self.custom_mask_buffer[:packed.numel()].bitwise_or_(packed)
            else:
                dense = torch.block_diag(*allowed.unbind(0))
                if reset:
                    for n in range(self.N):
                        self.mask[0, 0, n * width:(n + 1) * width,
                                  n * group + start:(n + 1) * group].fill_(torch.finfo(self.mask.dtype).min)
                self.mask[0, 0].masked_fill_(dense, 0)
            return
        start = self._refresh_start
        lean_mask = self.B == 16 and self.backend == 'flashinfer' and refresh_inject_lean()
        if lean_mask:
            # serving JETSPEC_REFRESH_INJECT_LEAN: row r's packed bits at its k-hat, from a
            # per-bucket bank; the rows' bits are disjoint, so their byte sum is their OR
            bank = self._refresh_mask_bank()
            packed = bank[self._bank_rows, khat.to(self.device).clamp(0, 15)].sum(0, dtype=torch.int16).to(torch.uint8)
        else:
            slot = torch.arange(self.B, device=self.device)
            visible = ((slot[None, :, None] > slot[None, None, :])
                       & (slot[None, None, :] <= khat.to(self.device)[:, None, None]))
            own = torch.arange(self.R, device=self.device)
            refreshed = torch.zeros(self.R, self.B, self.R, self.B, dtype=torch.bool, device=self.device)
            refreshed[own, :, own] = visible
            allowed = torch.cat((torch.zeros(rows, start, dtype=torch.bool, device=self.device),
                                 refreshed.reshape(rows, rows)), dim=1)
        if self.backend == 'flashinfer':
            packed = packed if lean_mask else pack_mask_bits(allowed.flatten())
            if reset:
                if getattr(self, '_refresh_region_bits', None) is None:
                    region = torch.zeros(rows, start + rows, dtype=torch.bool, device=self.device)
                    region[:, start:] = True
                    self._refresh_region_bits = pack_mask_bits(region.flatten()).bitwise_not_()
                self.custom_mask_buffer[:packed.numel()].bitwise_and_(self._refresh_region_bits)
            self.custom_mask_buffer[:packed.numel()].bitwise_or_(packed)
        else:
            if reset:
                self.mask[0, 0, :, start:start + rows].fill_(torch.finfo(self.mask.dtype).min)
            self.mask[0, 0].masked_fill_(allowed, 0)


    def _refresh_mask_bank(self):
        """[R, 16, bytes]: the packed refresh mask of row r alone at k-hat v (every other row
        hidden), built once per bucket with inject_refresh's own mask terms."""
        if getattr(self, '_refresh_bank', None) is None:
            rows, start = self.R * 16, self._refresh_start
            slot = torch.arange(16, device=self.device)
            own = torch.arange(self.R, device=self.device)
            banks = []
            for v in range(16):
                visible = (slot[:, None] > slot[None, :]) & (slot[None, :] <= v)   # [16, 16]
                refreshed = torch.zeros(self.R, self.R, 16, self.R, 16, dtype=torch.bool, device=self.device)
                refreshed[own, own, :, own] = visible
                allowed = torch.cat((torch.zeros(self.R, rows, start, dtype=torch.bool, device=self.device),
                                     refreshed.reshape(self.R, rows, rows)), dim=2)
                banks.append(pack_mask_bits(allowed.reshape(-1)).reshape(self.R, -1))
            self._refresh_bank = torch.stack(banks, dim=1)
            self._bank_rows = own
        return self._refresh_bank


class FastBucketRowDraftHead(BucketRowDraftHead):
    """N=1 append-only context cache; explicit request identity owns reuse.

    Masks and rotary inputs keep the bucket's plan pointers. Two small graphs
    prepare dirty rows and refresh their KV. Unused dirty lanes write distinct
    noise slots, all overwritten by the next pass before attention reads them.
    A new request, shrink, changed hypotheses, or large append rebuilds fully.
    """
    @torch.inference_mode()
    def __init__(self, head, **kwargs):
        if kwargs['N'] != 1:
            raise ValueError('host context cache supports one sequence')
        super().__init__(head, **kwargs)
        self._prefix_key = None
        self._length = None
        self._signature = None
        self._graphs = {}
        self.incremental_updates = 0
        self.full_updates = 0
        self.last_update_kind = 'uninitialized'
        self.last_refresh_rows = 0
        self.last_refresh_context_rows = 0

    def _configure(self, hypotheses):
        ks = torch.tensor(hypotheses, device=self.device)
        self.used_tail = max(hypotheses)
        ks = ks.repeat_interleave(self.F)
        ks = torch.cat((ks, ks.new_zeros(self.R - ks.numel())))
        self.query_offsets = (ks[:, None] + torch.arange(16, device=self.device)).flatten()
        self.dirty_order = torch.arange(16 + self.used_tail, device=self.device)
        self.control_cpu = torch.empty(2, dtype=torch.long, pin_memory=self.device.type == 'cuda')
        self.control = torch.empty(2, dtype=torch.long, device=self.device)
        col = self.columns[None, None, :]
        row = self.queries // 16
        c = self.P + 15
        tail_allowed = (col >= self.P) & (col < self.P + ks[row][None, :, None])
        block_allowed = ((col-c)//16 == row[None, :, None]) & ((col-c)%16 <= (self.queries%16)[None, :, None]) & (col >= c)
        masks = []
        for start in range(0, self.P + 1, 32):
            lengths = torch.arange(start, min(self.P + 1, start + 32), device=self.device)
            allowed = (col < lengths[:, None, None]) | tail_allowed | block_allowed
            if self.backend == 'flashinfer':
                bits = allowed.reshape(len(lengths), -1, 8).to(torch.uint8)
                allowed = (bits * (1 << torch.arange(8, device=self.device))).sum(-1).to(torch.uint8)
            masks.append(allowed)
        self.mask_bank = torch.cat(masks)
        self._graphs.clear()

    def _prepare_incremental(self):
        old, length = self.control.unbind()
        cc, cs = self.head.rotary_emb(self.hidden_stub, (length + self.tail_positions)[None])
        qc, qs = self.head.rotary_emb(self.hidden_stub, (length + self.query_offsets)[None])
        self.ccos[:, self.P:].copy_(cc); self.csin[:, self.P:].copy_(cs)
        self.qcos.copy_(qc); self.qsin.copy_(qs)
        mask = self.mask_bank.index_select(0, self.control[1:2])
        if self.backend == 'flashinfer':
            self.custom_mask_buffer[:mask.numel()].copy_(mask.flatten())
        else:
            allowed = mask.reshape(self.mask.shape)
            self.mask.copy_(torch.zeros_like(self.mask).masked_fill_(~allowed, torch.finfo(self.dtype).min))
        delta = length - old
        index = self.dirty_order
        context_index = torch.where(index < delta, old + index, self.P + index - delta).clamp(0, self.P + 14)
        noise_index = (index - delta - self.used_tail).clamp(0, 15)
        self.dirty_slots = torch.where(index < delta + self.used_tail, context_index, self.layout.noise_slots[noise_index])
        self.dirty_contexts = self.contexts.index_select(0, context_index)
        self.dirty_cos = self.ccos[0].index_select(0, context_index)
        self.dirty_sin = self.csin[0].index_select(0, context_index)

    def _refresh_incremental_step(self):
        # Same pointwise projection and normalization as refresh_context; only
        # accepted-prefix and visible-tail rows replace their cached KV.
        hidden = self.head.hidden_norm(self.head.fc(self.dirty_contexts))
        for layer, cache in zip(self.head.layers, self.kv):
            attn = layer.self_attn
            nh, hd = attn.config.num_key_value_heads, attn.head_dim
            k = attn.k_norm(attn.k_proj(hidden).reshape(-1, nh, hd))
            k = _rotary(k, self.dirty_cos, self.dirty_sin)
            v = attn.v_proj(hidden).reshape(-1, nh, hd)
            cache[self.dirty_slots, 0, 0] = k
            cache[self.dirty_slots, 1, 0] = v

    def _replay(self, name, step):
        if self.device.type != 'cuda':
            step()
            return
        if name not in self._graphs:
            import time
            started = time.perf_counter()
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3): step()
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph): step()
            self._graphs[name] = graph
            print(f'W256_HOST_CAPTURE phase={name} H={self.H} F={self.F} rows={self.R} prefix={self.P} dirty={self.dirty_order.numel()} ms={(time.perf_counter()-started)*1000:.3f}', flush=True)
        self._graphs[name].replay()

    def refresh_incremental(self):
        """Separate replay boundary keeps CUDA refresh accounting comparable."""
        self._replay('refresh', self._refresh_incremental_step)

    @torch.inference_mode()
    def update(self, prefixes, tails, hypotheses, *, mark=False, prefix_key=None):
        if len(prefixes) != 1 or len(hypotheses) != self.H:
            raise ValueError('host context update requires one sequence and H hypotheses')
        length = prefixes[0].shape[0]
        if length > self.P:
            raise ValueError('prefix exceeds bucket')
        signature = (tuple(hypotheses), bool(mark))
        reset = (prefix_key is None or self._prefix_key is not prefix_key or self._length is None or
                 length < self._length or length - self._length > 16 or signature != self._signature)
        if reset:
            super().update(prefixes, tails, hypotheses, mark=mark)
            if signature != self._signature:
                self._configure(hypotheses)
            self._signature = signature
            self._prefix_key = prefix_key
            self._length = length
            self.full_updates += 1
            self.last_update_kind = 'full'
            self.last_refresh_rows = self.P + 15
            self.last_refresh_context_rows = self.P + 15
            return
        old = self._length
        context = self.contexts.view(1, self.P + 15, -1)
        if length > old:
            context[0, old:length].copy_(prefixes[0][old:length])
            if mark: context[0, old:length].add_(self.head.adapted_position_embedding.weight[0])
        context[:, self.P:].copy_(tails)
        if mark: context[:, self.P:].add_(self.head.adapted_position_embedding.weight[1])
        self.incremental_updates += 1
        self.last_update_kind = 'incremental'
        self.last_refresh_context_rows = length - old + self.used_tail
        self.last_refresh_rows = self.dirty_order.numel() if self.last_refresh_context_rows else 0
        if self.last_refresh_context_rows:
            self.control_cpu[0] = old; self.control_cpu[1] = length
            self.control.copy_(self.control_cpu, non_blocking=True)
            self._replay('prepare', self._prepare_incremental)
            self.refresh_incremental()
        self._length = length
