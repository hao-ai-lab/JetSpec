"""Static-cache CUDA-graph execution for the linear block head."""

from __future__ import annotations

from jetspec.engines.jetspec_plus_plus.profiles import kernel_enabled, kernel_value

import os
import json
from pathlib import Path
import statistics
from time import perf_counter

import torch
from torch.nn import functional as F

from jetspec.engines.jetspec_plus_plus.jetspec.blockhead_serve import BlockheadProposal
from jetspec.engines.jetspec_plus_plus.serving.cache_capacity import CACHE_BUCKETS
from jetspec.engines.jetspec_plus_plus.run_config import RunConfig


CACHE_CAPACITY = 768
SCRATCH_SLOTS = 16


def validate_graph_context_length(
    length: int, *, capacity: int = CACHE_CAPACITY
) -> None:
    if length < 1 or length > capacity:
        raise ValueError(
            f"BLOCKHEAD GRAPH context length {length} outside [1, {capacity}]"
        )


def _rotate_half(value: torch.Tensor) -> torch.Tensor:
    half = value.shape[-1] // 2
    return torch.cat((-value[..., half:], value[..., :half]), dim=-1)


def _rotary(value: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return value * cos + _rotate_half(value) * sin


def _repeat_kv(value: torch.Tensor, groups: int) -> torch.Tensor:
    if groups == 1:
        return value
    return value.repeat_interleave(groups, dim=1)


class StaticBlockheadGraph:
    """One fixed extension, query, and Markov graph over a static K/V cache."""

    def __init__(self, drafter, *, capacity: int = CACHE_CAPACITY, _force_stock: bool = False,
                 run_config=None):
        from jetspec.engines.jetspec_plus_plus.profiles import kernel_enabled
        self.run_config = run_config if run_config is not None else RunConfig.from_environment({})
        self.drafter = drafter
        self.head = drafter.head
        self.target = drafter.target
        self.block_size = int(drafter.block_size)
        self.draft_offset = getattr(drafter, "draft_offset", 0)
        self.capacity = int(capacity)
        self.lanes = 32 if self.block_size in (31, 32) else 16 if self.block_size in (15, 16) else 8
        if self.block_size not in (7, 15, 16, 31, 32):
            raise ValueError("BLOCKHEAD GRAPH requires block size 7, 15, 16, 31 or 32")
        if self.block_size == 16 and (self.draft_offset != 1 or self.head.markov_head is not None):
            raise ValueError("BLOCKHEAD GRAPH block size 16 requires plain DFlash")
        if self.block_size == 32 and self.draft_offset == 1 and self.head.markov_head is not None:
            raise ValueError("BLOCKHEAD GRAPH DFlash block size 32 requires no Markov head")
        if self.capacity not in CACHE_BUCKETS:
            raise ValueError(f"BLOCKHEAD GRAPH cache capacity must be one of {CACHE_BUCKETS}")
        self.device = next(self.head.parameters()).device
        self.dtype = next(self.head.parameters()).dtype
        if self.device.type != "cuda":
            raise ValueError("BLOCKHEAD GRAPH requires CUDA")
        if self.head.resolve_causal_head("auto"):
            raise ValueError("BLOCKHEAD GRAPH requires the non-causal block head")
        if any(layer.self_attn.sliding_window is not None for layer in self.head.layers):
            raise ValueError("BLOCKHEAD GRAPH refuses sliding-window attention")

        self.hidden_size = int(self.head.config.hidden_size)
        self.target_hidden_size = int(self.head.fc.in_features)
        self.vocab_size = int(self.target.lm_head.weight.shape[0])
        self.attn_len = not _force_stock and kernel_enabled('blockhead_attn_len', '1')
        self.glue_lean = not _force_stock and kernel_enabled('blockhead_glue_lean', '1')
        self.qkv_fused = not _force_stock and kernel_enabled('blockhead_qkv_fused', '1')
        self.lmhead_kn = not _force_stock and kernel_enabled('blockhead_lmhead_kn', '1')
        if self.glue_lean:
            from jetspec.engines.jetspec_plus_plus.jetspec import c1_fused
            from jetspec.engines.jetspec_plus_plus.jetspec.rows import kv_rotary_into_cache, kv_rotary_into_cache_slots
            self._fused = c1_fused
            self._kv_rotary_into_cache = kv_rotary_into_cache
            self._kv_rotary_into_cache_slots = kv_rotary_into_cache_slots
        if self.attn_len:
            from jetspec.engines.jetspec_plus_plus.kernels.jpass_len_attention import attend
            from jetspec.engines.jetspec_plus_plus.kernels.jpass_len_attention import blockhead_mask
            self._attend = attend
            self._blockhead_mask = blockhead_mask
            if self.capacity % 64:
                raise ValueError("BLOCKHEAD ATTN LEN requires a 64-aligned cache capacity")
        if self.qkv_fused:
            self._packed_qkv = []
            for layer in self.head.layers:
                attn = layer.self_attn
                projections = (attn.q_proj, attn.k_proj, attn.v_proj)
                weight = torch.cat([projection.weight for projection in projections], dim=0).contiguous()
                bias = (torch.cat([projection.bias for projection in projections], dim=0)
                        if all(projection.bias is not None for projection in projections) else None)
                if bias is None and any(projection.bias is not None for projection in projections):
                    raise ValueError("BLOCKHEAD QKV FUSED requires matching projection biases")
                widths = tuple(int(projection.out_features) for projection in projections)
                self._packed_qkv.append((weight, bias, widths))
        if self.lmhead_kn:
            if getattr(self.target.lm_head, "bias", None) is not None:
                raise ValueError("BLOCKHEAD LMHEAD KN requires a bias-free target lm_head")
            self._lm_head_kn = self.target.lm_head.weight.detach().T.contiguous()
        self._cache_length = 0
        self._parity_round = 0
        self._graphs: dict[str, torch.cuda.CUDAGraph] = {}
        self._state = {
            "head_graph": True,
            "captures": 0,
            "query_replays": 0,
            "extension_replays": 0,
            "prefill_extensions": 0,
        }
        self.self_check = {}
        self._allocate()
        self._capture_all()
        self._self_check_all()
        if self.run_config.engine.engine_settings.blockhead_bench == '1':
            self.benchmark_graphs()
        self.reset_cache()
        self._audit_control = None
        audit_path = self.run_config.engine.engine_settings.blockhead_decision_audit
        if audit_path and not _force_stock:
            self._audit_path = Path(audit_path)
            self._audit_path.parent.mkdir(parents=True, exist_ok=True)
            self._audit_round = 0
            self._audit_control = StaticBlockheadGraph(
                drafter, capacity=capacity, _force_stock=True, run_config=self.run_config)

    @torch.inference_mode()
    def benchmark_graphs(self, *, replays: int = 200) -> None:
        """Measure the captured extension and query on synthetic fixed buffers."""
        self._extension_hidden.zero_()
        self._extension_positions.copy_(
            torch.arange(self.lanes, device=self.device).unsqueeze(0))
        self._extension_slots.copy_(self.capacity + self._extension_lanes)
        self._query_token_ids.fill_(int(self.head.mask_token_id))
        self._query_token_ids[:, 0] = 17
        for length in sorted({1, 16, 216, 768, self.capacity}):
            if length > self.capacity:
                continue
            self._query_positions.copy_(
                torch.arange(length, length + self.block_size,
                             device=self.device).unsqueeze(0))
            self._set_query_mask(length)
            for _ in range(3):
                self._graphs["extension"].replay()
                self._graphs["query"].replay()
            starts = [torch.cuda.Event(enable_timing=True) for _ in range(replays)]
            ends = [torch.cuda.Event(enable_timing=True) for _ in range(replays)]
            for start, end in zip(starts, ends):
                start.record()
                self._graphs["extension"].replay()
                self._graphs["query"].replay()
                end.record()
            torch.cuda.synchronize(self.device)
            times = sorted(start.elapsed_time(end) for start, end in zip(starts, ends))
            if length == 1:
                with torch.profiler.profile(activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ]) as profile:
                    self._graphs["extension"].replay()
                    self._graphs["query"].replay()
                    torch.cuda.synchronize(self.device)
                self._bench_kernel_count = sum(
                    event.device_type == torch.autograd.DeviceType.CUDA
                    for event in profile.events())
            print("W533_HEAD_BENCH " + json.dumps({
                "length": length, "capacity": self.capacity, "replays": replays,
                "p50_ms": statistics.median(times),
                "p90_ms": times[int(replays * .9) - 1],
                "kernels": self._bench_kernel_count,
                "flags": {name: bool(getattr(self, name)) for name in
                          ("attn_len", "glue_lean", "qkv_fused", "lmhead_kn")},
            }, sort_keys=True), flush=True)

    def _allocate(self) -> None:
        total = self.capacity + max(SCRATCH_SLOTS, self.lanes)
        self._kv = []
        self._attn_cache = []
        for layer in self.head.layers:
            attn = layer.self_attn
            kv_heads = int(attn.config.num_key_value_heads)
            shape = (1, kv_heads, total, int(attn.head_dim))
            if self.attn_len:
                cache = torch.zeros((total, 2, 1, kv_heads, int(attn.head_dim)),
                                    device=self.device, dtype=self.dtype)
                self._attn_cache.append(cache)
                self._kv.append((cache[:, 0, 0].permute(1, 0, 2).unsqueeze(0),
                                 cache[:, 1, 0].permute(1, 0, 2).unsqueeze(0)))
            else:
                self._kv.append(
                    (
                        torch.zeros(shape, device=self.device, dtype=self.dtype),
                        torch.zeros(shape, device=self.device, dtype=self.dtype),
                    )
                )

        self._extension_hidden = torch.zeros(
            1, self.lanes, self.target_hidden_size,
            device=self.device, dtype=self.dtype,
        )
        self._extension_positions = torch.zeros(
            1, self.lanes, device=self.device, dtype=torch.long
        )
        self._extension_lanes = torch.arange(
            self.lanes, device=self.device, dtype=torch.long
        )
        self._extension_slots = self.capacity + self._extension_lanes

        self._query_token_ids = torch.full(
            (1, self.block_size),
            int(self.head.mask_token_id),
            device=self.device,
            dtype=torch.long,
        )
        self._query_positions = torch.arange(
            self.block_size, device=self.device, dtype=torch.long
        ).unsqueeze(0)
        self._query_mask = torch.full(
            (1, 1, self.block_size, self.capacity + self.block_size),
            torch.finfo(self.dtype).min,
            device=self.device,
            dtype=self.dtype,
        )
        self._query_mask[..., self.capacity :] = 0
        self._query_hidden = torch.empty(
            1, self.block_size, self.hidden_size,
            device=self.device, dtype=self.dtype,
        )
        # Logits keep the target head's dtype (fp32 in the identity twins), as eager does.
        self._markov_logits = torch.empty(
            1, self.block_size, self.vocab_size,
            device=self.device, dtype=self.target.lm_head.weight.dtype,
        )
        self._query_logits = (self._markov_logits if self.lmhead_kn
                              else torch.empty_like(self._markov_logits))
        if self.attn_len:
            self._query_rows = ((self.block_size + 15) // 16) * 16
            self._query_length = torch.zeros(1, device=self.device, dtype=torch.int32)
            mask_bytes = (self._query_rows * (self.capacity + self.block_size) + 7) // 8
            self._query_packed_mask = torch.zeros(mask_bytes, device=self.device, dtype=torch.uint8)
        self._markov_anchor = torch.zeros(1, device=self.device, dtype=torch.long)
        self._markov_tokens = torch.empty(
            1, self.block_size, device=self.device, dtype=torch.long
        )
        self._markov_corrected = torch.empty_like(self._markov_logits)

    def _project_context(self, target_hidden: torch.Tensor) -> torch.Tensor:
        projected = self.head.fc(self.head.hidden_dim_adapter(target_hidden.to(self.dtype)))
        if self.glue_lean:
            norm = self.head.hidden_norm
            return self._fused.fast_rms_norm(projected, norm.weight, norm.variance_epsilon)
        return self.head.hidden_norm(projected)

    def _project_extension(self, target_hidden: torch.Tensor, positions: torch.Tensor):
        count = target_hidden.shape[1]
        source = self._project_context(target_hidden)
        cos, sin = self.head.rotary_emb(source, positions)
        projected = []
        for layer in self.head.layers:
            attn = layer.self_attn
            kv_heads = int(attn.config.num_key_value_heads)
            k = attn.k_proj(source).view(
                1, count, kv_heads, attn.head_dim
            )
            v = attn.v_proj(source).view(
                1, count, kv_heads, attn.head_dim
            )
            k = (self._fused.fast_rms_norm(k, attn.k_norm.weight,
                                          attn.k_norm.variance_epsilon)
                 if self.glue_lean else attn.k_norm(k)).transpose(1, 2)
            v = v.transpose(1, 2)
            projected.append((_rotary(k, cos, sin), v))
        return projected

    def _extension_step(self) -> None:
        if self.glue_lean:
            self._write_extension_lean(
                self._extension_hidden, self._extension_positions,
                slots=self._extension_slots)
            return
        projected = self._project_extension(
            self._extension_hidden, self._extension_positions
        )
        for (cache_k, cache_v), (new_k, new_v) in zip(self._kv, projected):
            cache_k.index_copy_(2, self._extension_slots, new_k)
            cache_v.index_copy_(2, self._extension_slots, new_v)

    def _write_extension_lean(self, target_hidden, positions, *, slots=None, start=0):
        count = target_hidden.shape[1]
        source = self._project_context(target_hidden)
        cos, sin = self.head.rotary_emb(source, positions)
        cos = cos.reshape(count, -1)
        sin = sin.reshape(count, -1)
        for layer, (cache_k, cache_v) in zip(self.head.layers, self._kv):
            attn = layer.self_attn
            kv_heads = int(attn.config.num_key_value_heads)
            width = kv_heads * int(attn.head_dim)
            key = attn.k_proj(source).reshape(count, width)
            value = attn.v_proj(source).reshape(count, width)
            normalized = self._fused.fast_rms_norm(
                key.reshape(count, kv_heads, attn.head_dim),
                attn.k_norm.weight, attn.k_norm.variance_epsilon).reshape(count, width)
            keys = cache_k.squeeze(0).transpose(0, 1)
            values = cache_v.squeeze(0).transpose(0, 1)
            if slots is None:
                self._kv_rotary_into_cache(
                    normalized, value, cos, sin,
                    keys[start:start + count], values[start:start + count],
                    kv_heads=kv_heads, groups=1, head_dim=attn.head_dim)
            else:
                self._kv_rotary_into_cache_slots(
                    normalized, value, cos, sin, keys, values, slots,
                    kv_heads=kv_heads, head_dim=attn.head_dim)

    def _query_forward(self) -> torch.Tensor:
        # An fp32 target (identity twins) feeds a bf16 head: cast at the boundary.
        hidden = self.target.model.embed_tokens(self._query_token_ids).to(self.dtype)
        cos, sin = self.head.rotary_emb(hidden, self._query_positions)
        if self.attn_len:
            self._blockhead_mask(self._query_length, self.capacity, self.block_size,
                                 self._query_rows, self._query_packed_mask)
        pending = None
        for index, layer in enumerate(self.head.layers):
            attn = layer.self_attn
            if self.glue_lean and pending is not None:
                normed, hidden = self._fused.fast_add_rms_norm(
                    *pending, layer.input_layernorm.weight,
                    layer.input_layernorm.variance_epsilon)
            elif self.glue_lean:
                normed = self._fused.fast_rms_norm(
                    hidden, layer.input_layernorm.weight,
                    layer.input_layernorm.variance_epsilon)
            else:
                normed = layer.input_layernorm(hidden)
            residual = hidden
            heads = int(attn.config.num_attention_heads)
            kv_heads = int(attn.config.num_key_value_heads)
            if self.qkv_fused:
                weight, bias, widths = self._packed_qkv[index]
                qp, kp, vp = F.linear(normed, weight, bias).split(widths, dim=-1)
            else:
                qp, kp, vp = attn.q_proj(normed), attn.k_proj(normed), attn.v_proj(normed)
            cache_k, cache_v = self._kv[index]
            if self.glue_lean:
                q = self._fused.jpass_q_norm_rotary(
                    qp.reshape(self.block_size, heads * attn.head_dim),
                    attn.q_norm.weight, cos.reshape(self.block_size, attn.head_dim),
                    sin.reshape(self.block_size, attn.head_dim),
                    attn.q_norm.variance_epsilon, heads, attn.head_dim)
                if self.attn_len:
                    key_out = cache_k[:, :, self.capacity:self.capacity + self.block_size].squeeze(0).transpose(0, 1)
                    value_out = cache_v[:, :, self.capacity:self.capacity + self.block_size].squeeze(0).transpose(0, 1)
                else:
                    key_out = torch.empty((self.block_size, kv_heads, attn.head_dim),
                                          device=self.device, dtype=self.dtype)
                    value_out = torch.empty_like(key_out)
                self._fused.jpass_kv_norm_rotary_cache(
                    kp.reshape(self.block_size, kv_heads * attn.head_dim),
                    vp.reshape(self.block_size, kv_heads * attn.head_dim),
                    attn.k_norm.weight, cos.reshape(self.block_size, attn.head_dim),
                    sin.reshape(self.block_size, attn.head_dim), key_out, value_out,
                    attn.k_norm.variance_epsilon, kv_heads, attn.head_dim)
                k_block = key_out.transpose(0, 1).unsqueeze(0)
                v_block = value_out.transpose(0, 1).unsqueeze(0)
            else:
                q = attn.q_norm(qp.view(1, self.block_size, heads, attn.head_dim)).transpose(1, 2)
                k_block = attn.k_norm(kp.view(1, self.block_size, kv_heads, attn.head_dim)).transpose(1, 2)
                v_block = vp.view(1, self.block_size, kv_heads, attn.head_dim).transpose(1, 2)
                q = _rotary(q, cos, sin)
                k_block = _rotary(k_block, cos, sin)
            if self.attn_len:
                if not self.glue_lean:
                    cache_k[:, :, self.capacity:self.capacity + self.block_size].copy_(k_block)
                    cache_v[:, :, self.capacity:self.capacity + self.block_size].copy_(v_block)
                q_rows = (q if self.glue_lean else q.squeeze(0).transpose(0, 1).contiguous())
                q_rows = F.pad(q_rows, (0, 0, 0, 0, 0, self._query_rows - self.block_size))
                attended = self._attend(
                    q_rows, self._attn_cache[index][:self.capacity + self.block_size],
                    self._query_packed_mask, self._query_length, groups=1,
                    prefix=self.capacity, scale=attn.scaling)[:self.block_size]
                attended = attended.reshape(1, self.block_size, heads * attn.head_dim)
            else:
                k = torch.cat((cache_k[:, :, : self.capacity], k_block), dim=2)
                v = torch.cat((cache_v[:, :, : self.capacity], v_block), dim=2)
                k = _repeat_kv(k, int(attn.num_key_value_groups))
                v = _repeat_kv(v, int(attn.num_key_value_groups))
                attended = F.scaled_dot_product_attention(
                    q.transpose(0, 1).unsqueeze(0) if self.glue_lean else q,
                    k, v, attn_mask=self._query_mask, dropout_p=0.0,
                    is_causal=False, scale=attn.scaling,
                ).transpose(1, 2).reshape(1, self.block_size, heads * attn.head_dim)
            update = attn.o_proj(attended)
            if self.glue_lean:
                normalized, hidden = self._fused.fast_add_rms_norm(
                    update, residual, layer.post_attention_layernorm.weight,
                    layer.post_attention_layernorm.variance_epsilon)
                mlp = layer.mlp
                gate, up = mlp.gate_proj(normalized), mlp.up_proj(normalized)
                pending = (mlp.down_proj(self._fused._silu_mul_extension().silu_mul(gate, up)), hidden)
            else:
                hidden = residual + update
                hidden = hidden + layer.mlp(layer.post_attention_layernorm(hidden))
        if self.glue_lean:
            output, _ = self._fused.fast_add_rms_norm(
                *pending, self.head.norm.weight, self.head.norm.variance_epsilon)
            return output
        return self.head.norm(hidden)

    def _query_step(self) -> None:
        hidden = self._query_forward()
        self._query_hidden.copy_(hidden)
        if self.lmhead_kn:
            torch.mm(hidden.reshape(-1, self.hidden_size).to(self._lm_head_kn.dtype),
                     self._lm_head_kn, out=self._markov_logits.reshape(-1, self.vocab_size))
        else:
            self._query_logits.copy_(self.target.lm_head(hidden.to(self.target.lm_head.weight.dtype)))

    def _markov_step(self) -> None:
        markov = self.head.markov_head
        if markov is None:
            self._markov_corrected.copy_(self._markov_logits)
            self._markov_tokens.copy_(self._markov_logits.argmax(dim=-1))
            return
        if getattr(markov, "markov_head_type", None) != "vanilla":
            raise ValueError("BLOCKHEAD GRAPH requires the fixed order-1 Markov head")
        prev = self._markov_anchor
        corrected = []
        tokens = []
        for position in range(self.block_size):
            step = self._markov_logits[:, position, :]
            step = step + markov.markov_w2(markov.markov_w1(prev.long()))
            corrected.append(step.unsqueeze(1))
            prev = step.argmax(dim=-1)
            tokens.append(prev.unsqueeze(1))
        self._markov_corrected.copy_(torch.cat(corrected, dim=1))
        self._markov_tokens.copy_(torch.cat(tokens, dim=1))

    @torch.inference_mode()
    def _capture(self, name: str, step) -> None:
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                step()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            step()
        self._graphs[name] = graph
        self._state["captures"] += 1
        print(
            "W277_BLOCKHEAD_CAPTURE "
            f"phase={name} block={self.block_size} capacity={self.capacity} "
            f"lanes={self.lanes}",
            flush=True,
        )

    def _capture_all(self) -> None:
        started = perf_counter()
        self._capture("extension", self._extension_step)
        self._capture("query", self._query_step)
        self._capture("markov", self._markov_step)
        torch.cuda.synchronize(self.device)
        print(
            "W277_BLOCKHEAD_CAPTURE_COMPLETE "
            f"captures={self._state['captures']} ms={(perf_counter() - started) * 1000:.3f}",
            flush=True,
        )

    def _self_check_extension(self) -> list[int]:
        checked = []
        start = 31
        for count in range(1, self.lanes + 1):
            self._extension_hidden.copy_(
                torch.arange(
                    self._extension_hidden.numel(),
                    device=self.device,
                    dtype=torch.float32,
                ).reshape_as(self._extension_hidden).to(self.dtype) / 1024
            )
            self._extension_positions.copy_(
                torch.arange(
                    start, start + self.lanes, device=self.device
                ).unsqueeze(0)
            )
            self._extension_slots.copy_(
                torch.where(
                    self._extension_lanes < count,
                    start + self._extension_lanes,
                    self.capacity + self._extension_lanes,
                )
            )
            self._extension_step()
            expected = [
                tensor.index_select(2, self._extension_slots).clone()
                for pair in self._kv
                for tensor in pair
            ]
            for pair in self._kv:
                for tensor in pair:
                    tensor.index_fill_(2, self._extension_slots, 0)
            self._graphs["extension"].replay()
            actual = [
                tensor.index_select(2, self._extension_slots)
                for pair in self._kv
                for tensor in pair
            ]
            if not all(torch.equal(left, right) for left, right in zip(expected, actual)):
                raise RuntimeError(
                    f"BLOCKHEAD GRAPH extension byte parity failed length={count}"
                )
            checked.append(count)
        return checked

    def _self_check_query(self) -> None:
        for index, (cache_k, cache_v) in enumerate(self._kv, start=1):
            cache_k[:, :, :47].fill_(index / 128)
            cache_v[:, :, :47].fill_(index / 64)
        self._query_token_ids[:, 0] = 17
        self._query_positions.copy_(
            torch.arange(47, 47 + self.block_size, device=self.device).unsqueeze(0)
        )
        self._set_query_mask(47)
        self._query_step()
        expected = (self._query_hidden.clone(), self._query_logits.clone())
        self._graphs["query"].replay()
        if not torch.equal(expected[0], self._query_hidden):
            raise RuntimeError("BLOCKHEAD GRAPH query byte parity failed tensor=hidden")
        if not torch.equal(expected[1], self._query_logits):
            raise RuntimeError("BLOCKHEAD GRAPH query byte parity failed tensor=logits")

    def _self_check_markov(self) -> None:
        self._markov_logits.copy_(self._query_logits)
        self._markov_anchor.fill_(17)
        self._markov_step()
        expected = (self._markov_tokens.clone(), self._markov_corrected.clone())
        self._graphs["markov"].replay()
        if not torch.equal(expected[0], self._markov_tokens):
            raise RuntimeError("BLOCKHEAD GRAPH Markov byte parity failed tensor=tokens")
        if not torch.equal(expected[1], self._markov_corrected):
            raise RuntimeError("BLOCKHEAD GRAPH Markov byte parity failed tensor=logits")

    @torch.inference_mode()
    def _self_check_all(self) -> None:
        lengths = self._self_check_extension()
        self._self_check_query()
        self._self_check_markov()
        torch.cuda.synchronize(self.device)
        self.self_check = {
            "extension_lengths": lengths,
            "extension_bitwise": True,
            "query_bitwise": True,
            "markov_bitwise": True,
        }
        print(
            "W277_BLOCKHEAD_SELF_CHECK "
            f"block={self.block_size} extension_lengths={lengths} "
            "query=bitwise markov=bitwise",
            flush=True,
        )

    def _eager_extend(
        self,
        target_hidden: torch.Tensor,
        positions: torch.Tensor,
        start: int,
        length: int,
    ) -> None:
        if self.glue_lean:
            self._write_extension_lean(target_hidden, positions, start=start)
            return
        projected = self._project_extension(target_hidden, positions)
        for (cache_k, cache_v), (new_k, new_v) in zip(self._kv, projected):
            cache_k[:, :, start:length].copy_(new_k)
            cache_v[:, :, start:length].copy_(new_v)

    def _graph_extend(
        self, target_hidden: torch.Tensor, positions: torch.Tensor, start: int
    ) -> None:
        count = target_hidden.shape[1]
        if count < 1 or count > self.lanes:
            raise ValueError(
                f"BLOCKHEAD GRAPH extension length {count} outside [1, {self.lanes}]"
            )
        self._extension_hidden.zero_()
        self._extension_hidden[:, :count].copy_(
            target_hidden.to(device=self.device, dtype=self.dtype)
        )
        self._extension_positions.zero_()
        self._extension_positions[:, :count].copy_(positions)
        self._extension_slots.copy_(
            torch.where(
                self._extension_lanes < count,
                start + self._extension_lanes,
                self.capacity + self._extension_lanes,
            )
        )
        self._graphs["extension"].replay()

    def _set_query_mask(self, length: int) -> None:
        if self.attn_len:
            self._query_length.fill_(length)
            return
        self._query_mask.fill_(torch.finfo(self.dtype).min)
        self._query_mask[..., :length] = 0
        self._query_mask[..., self.capacity :] = 0

    def reset_cache(self) -> None:
        self._cache_length = 0
        self._parity_round = 0
        control = getattr(self, "_audit_control", None)
        if control is not None:
            control.reset_cache()

    def graph_state(self) -> dict:
        return dict(self._state)

    def _record_parity(
        self,
        context_ids: torch.Tensor,
        target_hidden: torch.Tensor,
        hidden: torch.Tensor,
        base_logits: torch.Tensor,
        corrected: torch.Tensor,
        tokens: torch.Tensor,
    ) -> None:
        root = self.run_config.engine.engine_settings.w277_blockhead_parity_record
        if root is None or self._parity_round >= 8:
            return
        path = Path(root)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "context_ids": context_ids.detach().cpu(),
                "target_hidden": target_hidden.detach().cpu(),
                "block_hidden": hidden.detach().cpu(),
                "base_logits": base_logits.detach().cpu(),
                "corrected_logits": corrected.detach().cpu(),
                "tokens": tokens.detach().cpu(),
                "matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
                "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
            },
            path / f"round-{self._parity_round + 1:02d}.pt",
        )

    @torch.inference_mode()
    def propose(
        self,
        context_ids: torch.Tensor,
        k: int,
        *,
        target_hidden: torch.Tensor,
    ) -> BlockheadProposal:
        length = int(target_hidden.shape[1])
        validate_graph_context_length(length, capacity=self.capacity)
        if length < self._cache_length:
            raise ValueError("BLOCKHEAD GRAPH context shrank without reset")
        append = length - self._cache_length
        if append < 1:
            raise ValueError("BLOCKHEAD GRAPH requires a non-empty context extension")

        start_event = torch.cuda.Event(enable_timing=True)
        split_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        positions = torch.arange(
            self._cache_length, length, device=self.device
        ).unsqueeze(0)
        if self._cache_length == 0:
            self._eager_extend(
                target_hidden.to(device=self.device, dtype=self.dtype),
                positions,
                0,
                length,
            )
            self._state["prefill_extensions"] += 1
        else:
            self._graph_extend(
                target_hidden[:, self._cache_length : length],
                positions,
                self._cache_length,
            )
            self._state["extension_replays"] += 1
        self._cache_length = length

        self._query_token_ids.fill_(int(self.head.mask_token_id))
        self._query_token_ids[:, 0].copy_(context_ids[:, -1])
        self._query_positions.copy_(
            torch.arange(length, length + self.block_size, device=self.device).unsqueeze(0)
        )
        self._set_query_mask(length)
        self._graphs["query"].replay()
        self._state["query_replays"] += 1
        split_event.record()

        if not self.lmhead_kn:
            self._markov_logits.copy_(self._query_logits)
        self._markov_anchor.copy_(context_ids[:, -1])
        from jetspec.engines.jetspec_plus_plus.trace.current import active_recorder
        trace = active_recorder()
        if trace is None:
            self._graphs["markov"].replay()
        else:
            label = 'sampling' if self.draft_offset else 'markov_sampling'
            with trace.span(label, 'draft'):
                self._graphs["markov"].replay()
        corrected = self._markov_corrected[:, self.draft_offset:].clone()
        tokens = self._markov_tokens[:, self.draft_offset:].clone()
        parity_root = self.run_config.engine.engine_settings.w277_blockhead_parity_record
        hidden = self._query_hidden[:, self.draft_offset:].clone() if parity_root is not None else None
        base_logits = self._query_logits[:, self.draft_offset:].clone() if parity_root is not None else None
        end_event.record()
        end_event.synchronize()

        if parity_root is not None:
            self._record_parity(
                context_ids,
                target_hidden,
                hidden,
                base_logits,
                corrected,
                tokens,
            )
        if self._audit_control is not None:
            control = self._audit_control.propose(
                context_ids, k, target_hidden=target_hidden)
            reference = control.tokens.reshape(-1)
            differing = int((tokens.reshape(-1)[:k] != reference).sum().item())
            with self._audit_path.open("a") as sink:
                sink.write(json.dumps({
                    "round": self._audit_round, "context_length": length,
                    "draft_slots": int(reference.numel()), "decision_differences": differing,
                }) + "\n")
            self._audit_round += 1
        self._parity_round += 1
        return BlockheadProposal(
            tokens=tokens.reshape(-1)[:k],
            logits=corrected[:, :k, :],
            head_ms=float(start_event.elapsed_time(split_event)),
            markov_ms=float(split_event.elapsed_time(end_event)),
        )
