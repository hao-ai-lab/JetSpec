# Adaptation: package relocation and deferred optional dependencies.

from jetspec.engines.jetspec_plus_plus.profiles import kernel_enabled, kernel_value
import os
import sys

import torch
from torch import nn
from torch.nn import functional as torch_functional
from jetspec.engines.jetspec_plus_plus._optional import triton
from jetspec.engines.jetspec_plus_plus._optional import tl

# A spawned rank imports this module before it installs its engine. install_engine binds the
# backend again, here and in every module that imports the name.
BACKEND_IMPORTERS = ('jetspec.engines.jetspec_plus_plus.serving.model_runner', 'jetspec.engines.jetspec_plus_plus.serving.graphs',
                     'jetspec.engines.jetspec_plus_plus.verify.forward', 'jetspec.engines.jetspec_plus_plus.comparisons.fanout_graphs')


def attention_backend():
    backend = kernel_value('ssd_attention_backend', "sdpa")
    if backend not in {"flashinfer", "sdpa", "triton"}:
        raise ValueError(
            f"Unsupported SSD_ATTENTION_BACKEND={backend!r}; expected 'flashinfer', 'sdpa', or 'triton'"
        )
    return backend


def bind_attention_backend():
    global ATTENTION_BACKEND
    ATTENTION_BACKEND = attention_backend()
    for name in BACKEND_IMPORTERS:
        if name in sys.modules:
            sys.modules[name].ATTENTION_BACKEND = ATTENTION_BACKEND
    return ATTENTION_BACKEND


ATTENTION_BACKEND = attention_backend()

from jetspec.engines.jetspec_plus_plus.models.context import get_context


from jetspec.engines.jetspec_plus_plus.kernels.store_kvcache import store_kvcache_kernel


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)

class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
        draft: bool = False,
        speculate: bool = False,
        draft_async: bool = False,
        use_eagle: bool = False,
        F: int = 1,
        K: int = 1,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])
        self.draft = draft
        self.speculate = speculate
        self.draft_async = draft_async
        self.use_eagle = use_eagle
        self.prefill_wrappers = {}
        self.F = F # async_fan_out
        self.K = K # speculate_k
        self.only_prefill_wrapper = None
        self.flashinfer_ragged_wrapper = None
        self.flashinfer_paged_wrapper = None
        self.flashinfer_graph_wrappers = {}
        self.jetspec_paged_flashinfer = False

    def _sdpa(self, q, k, v, attn_mask=None, is_causal=False):
        output = torch_functional.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=is_causal,
            scale=self.scale,
            enable_gqa=self.num_heads != self.num_kv_heads,
        )
        return output.transpose(1, 2)

    def _sdpa_ragged(self, q, k, v, cu_seqlens_q, cu_seqlens_k):
        q_offsets = cu_seqlens_q.tolist()
        k_offsets = cu_seqlens_k.tolist()
        outputs = []
        for index in range(len(q_offsets) - 1):
            q_seq = q[q_offsets[index]:q_offsets[index + 1]].unsqueeze(0)
            k_seq = k[k_offsets[index]:k_offsets[index + 1]].unsqueeze(0)
            v_seq = v[k_offsets[index]:k_offsets[index + 1]].unsqueeze(0)
            outputs.append(self._sdpa(q_seq, k_seq, v_seq, is_causal=True)[0])
        return torch.cat(outputs)

    def _sdpa_paged(self, q, k_cache, v_cache, context_lens, block_tables):
        batch_size = context_lens.shape[0]
        assert q.shape[0] % batch_size == 0
        query_len = q.shape[0] // batch_size
        block_ids = block_tables.to(torch.long).clamp_min(0)
        k = k_cache[block_ids].flatten(1, 2)
        v = v_cache[block_ids].flatten(1, 2)
        q = q.view(batch_size, query_len, self.num_heads, self.head_dim)

        query_positions = (
            context_lens[:, None] - query_len
            + torch.arange(query_len, device=q.device)[None, :]
        )
        key_positions = torch.arange(k.shape[1], device=q.device)
        mask = key_positions[None, None, None, :] <= query_positions[:, None, :, None]
        return self._sdpa(q, k, v, attn_mask=mask).flatten(0, 1)

    def _triton_paged(self, q, k_cache, v_cache, context_lens, block_tables):
        if (self.num_heads, self.num_kv_heads, self.head_dim) != (24, 4, 256):
            return self._sdpa_paged(q, k_cache, v_cache, context_lens, block_tables)
        from jetspec.engines.jetspec_plus_plus.models.triton_paged import attend
        return attend(q, k_cache, v_cache, context_lens, block_tables, self.scale)

    def _sdpa_paged_ragged(
        self, q, k_cache, v_cache, cu_seqlens_q, cu_seqlens_k, block_tables,
    ):
        q_offsets = cu_seqlens_q.tolist()
        k_offsets = cu_seqlens_k.tolist()
        block_size = k_cache.shape[1]
        outputs = []
        for index in range(len(q_offsets) - 1):
            q_start, q_end = q_offsets[index:index + 2]
            kv_len = k_offsets[index + 1] - k_offsets[index]
            num_blocks = (kv_len + block_size - 1) // block_size
            block_ids = block_tables[index, :num_blocks].to(torch.long)
            k = k_cache[block_ids].flatten(0, 1)[:kv_len].unsqueeze(0)
            v = v_cache[block_ids].flatten(0, 1)[:kv_len].unsqueeze(0)
            q_seq = q[q_start:q_end].unsqueeze(0)
            query_positions = (
                kv_len - (q_end - q_start)
                + torch.arange(q_end - q_start, device=q.device)
            )
            key_positions = torch.arange(kv_len, device=q.device)
            mask = key_positions[None, None, None, :] <= query_positions[None, None, :, None]
            outputs.append(self._sdpa(q_seq, k, v, attn_mask=mask)[0])
        return torch.cat(outputs)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        o: torch.Tensor
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)

        k_cache, v_cache = self.k_cache, self.v_cache

        context = get_context()
        if self.k_cache.numel() and self.v_cache.numel():
            store_kvcache(k, v, self.k_cache, self.v_cache, context.slot_mapping)

        if context.is_prefill:
            if context.block_tables is not None:
                k, v = k_cache, v_cache

            if ATTENTION_BACKEND == "flashinfer":
                if context.block_tables is None:
                    o = self.flashinfer_ragged_wrapper.run(q, k, v)
                else:
                    o = self.flashinfer_paged_wrapper.run(q, (k, v))
            elif ATTENTION_BACKEND in {"sdpa", "triton"}:
                if context.block_tables is None:
                    o = self._sdpa_ragged(
                        q, k, v, context.cu_seqlens_q, context.cu_seqlens_k)
                else:
                    o = self._sdpa_paged_ragged(
                        q, k_cache, v_cache, context.cu_seqlens_q,
                        context.cu_seqlens_k, context.block_tables)
        else:
            # verify/glue decode: multi-query with cu_seqlens_q (K+1 or variable per seq)
            verify_or_glue = (
                self.speculate and context.cu_seqlens_q is not None
            )
            decode = not verify_or_glue
            tree_decode = (
                decode and self.speculate and self.draft and self.draft_async
                and not context.is_jit
            )

            if verify_or_glue:
                assert context.context_lens is not None
                if self.jetspec_paged_flashinfer:
                    wrapper = self.flashinfer_graph_wrappers[
                        (context.context_lens.shape[0], q.shape[0])]
                    o = wrapper.run(q, (k_cache, v_cache))
                elif ATTENTION_BACKEND == "flashinfer":
                    wrapper = self.flashinfer_graph_wrappers.get(
                        (context.context_lens.shape[0], q.shape[0]),
                        self.flashinfer_paged_wrapper)
                    o = wrapper.run(q, (k_cache, v_cache))
                elif ATTENTION_BACKEND == "sdpa":
                    o = self._sdpa_paged(
                        q, k_cache, v_cache, context.context_lens,
                        context.block_tables)
                elif ATTENTION_BACKEND == "triton":
                    o = self._triton_paged(
                        q, k_cache, v_cache, context.context_lens,
                        context.block_tables)

            elif tree_decode:
                if self.only_prefill_wrapper is not None:
                    prefill_wrapper = self.only_prefill_wrapper
                else:
                    mq_len = self.F * (self.K+1)
                    bs = q.shape[0] // mq_len
                    wrapper_bs = None
                    for available_bs in sorted(self.prefill_wrappers.keys()):
                        if available_bs >= bs:
                            wrapper_bs = available_bs
                            break
                    prefill_wrapper = self.prefill_wrappers[wrapper_bs]
                o = prefill_wrapper.run(q, (self.k_cache, self.v_cache))
            elif self.jetspec_paged_flashinfer or ATTENTION_BACKEND == "flashinfer":
                wrapper = self.flashinfer_graph_wrappers.get(
                    (context.context_lens.shape[0], q.shape[0]),
                    self.flashinfer_paged_wrapper)
                o = wrapper.run(q, (k_cache, v_cache))
            elif ATTENTION_BACKEND == "sdpa":
                o = self._sdpa_paged(
                    q, k_cache, v_cache, context.context_lens,
                    context.block_tables)
            elif ATTENTION_BACKEND == "triton":
                o = self._triton_paged(
                    q, k_cache, v_cache, context.context_lens,
                    context.block_tables)

        o = o.reshape(-1, self.num_heads * self.head_dim)
        return o
