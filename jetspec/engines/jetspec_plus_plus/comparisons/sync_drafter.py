"""Serve a DFlash-class block drafter synchronously on the verifier's GPU.

The baseline form of speculative decoding: verify, then ONE drafter call that reads the target's
true (fresh) hidden states for every committed token and proposes 15 drafts after the known anchor
(block 16), chain, greedy. No adapter, no heads, no second process. It fills the same serial slot
as the block-head mode (speculator_blockhead.py) and reuses its round records; the drafter itself
loads through ``JetSpecDrafter.load(fresh_only=True)``, the loader the asynchronous path uses.

Default off: nothing here is imported unless ``JETSPEC_SYNC_DRAFTER`` names a checkpoint.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from time import perf_counter

import torch

from jetspec.engines.jetspec_plus_plus.legacy_env import environment_name
from jetspec.engines.jetspec_plus_plus.run_config import RunConfig

SYNC_DRAFTER_ENV = environment_name('model', 'model_settings.sync_drafter')
SYNC_PASSES_ENV = environment_name('method', 'method_settings.sync_passes')
SYNC_GRAPH_ENV = environment_name('engine', 'engine_settings.sync_graph')
SYNC_LOOKAHEAD = 15
SYNC_SEED = 9504
# One captured graph per context capacity; the engine serves max_model_len 768 on this path.
SYNC_CONTEXT_BUCKETS = (128, 192, 256, 320, 384, 448, 512, 640, 768, 1024, 2048, 4096)
INITS = ("mask", "random_context")


def sync_block_size_read(run_config=None) -> int:
    config = run_config if run_config is not None else RunConfig.from_environment({})
    value = config.method.method_settings.block_size_read
    if value not in ("8", "16", "32"):
        raise ValueError("JETSPEC_BLOCK_SIZE_READ must be 8, 16 or 32")
    return int(value)


def sync_drafter_dir(run_config=None) -> str | None:
    config = run_config if run_config is not None else RunConfig.from_environment({})
    return config.model.model_settings.sync_drafter or None


def sync_passes(run_config=None) -> int:
    config = run_config if run_config is not None else RunConfig.from_environment({})
    value = config.method.method_settings.sync_passes
    if value not in ("1", "2", "3", "4"):
        raise ValueError(f"{SYNC_PASSES_ENV} must be 1, 2, 3 or 4")
    return int(value)


def sync_graph_requested(run_config=None) -> bool:
    config = run_config if run_config is not None else RunConfig.from_environment({})
    value = config.engine.engine_settings.sync_graph
    if value not in ("0", "1"):
        raise ValueError(f"{SYNC_GRAPH_ENV} must be 0 or 1")
    return value == "1"


def resolve_init(dflash_config: dict | None, passes: int) -> str:
    """The first forward's 15 draft slots, as the checkpoint was trained.

    A Jacobi-forcing drafter is mask-free: random context tokens seeded ``9504 XOR prompt index``,
    and later passes refeed its own argmax. A drafter without the flag is the mask-initialised
    DFlash-class drafter: mask tokens, and exactly one pass (it never saw its own output)."""
    jacobi = bool((dflash_config or {}).get("jacobi_forcing", False))
    if jacobi:
        return "random_context"
    if passes != 1:
        raise ValueError(
            f"a mask-initialised drafter runs one pass; {SYNC_PASSES_ENV}={passes} needs "
            "dflash_config.jacobi_forcing")
    return "mask"


def sync_drafter_config_overrides(checkpoint_dir: str | None, config_kwargs: dict,
                                run_config=None) -> dict:
    """Apply the default-off launch selection before ``Config`` is built."""
    if checkpoint_dir is None:
        return dict(config_kwargs)
    config = run_config if run_config is not None else RunConfig.from_environment({})
    if config.model.model_settings.blockhead:
        raise ValueError(f"{SYNC_DRAFTER_ENV} and JETSPEC_BLOCKHEAD both claim the sync slot")
    if int(config_kwargs.get("num_gpus", 1)) != 1:
        raise ValueError("sync drafter serving requires one GPU")
    if int(config_kwargs.get("max_num_seqs", 1)) > 1 and sync_graph_requested(config):
        raise ValueError("sync drafter batching requires eager mode")
    if config_kwargs.get("draft_async", False):
        raise ValueError("sync drafter serving uses the synchronous serial slot")
    if config_kwargs.get("jetspec", False) or config_kwargs.get("use_eagle", False):
        raise ValueError("sync drafter serving cannot be combined with another draft mode")
    checkpoint = json.loads((Path(checkpoint_dir) / "config.json").read_text(encoding="utf-8"))
    if int(checkpoint.get("block_size", 0)) != SYNC_LOOKAHEAD + 1:
        raise ValueError(f"sync drafter block_size must be {SYNC_LOOKAHEAD + 1}")
    resolve_init(checkpoint.get("dflash_config"), sync_passes(config))
    overrides = dict(config_kwargs)
    overrides.update(speculate=True, draft_async=False, jetspec_taps=True,
                     speculate_k=sync_block_size_read(config) - 1)
    target_layer_ids = (checkpoint.get("dflash_config") or {}).get("target_layer_ids")
    if target_layer_ids:
        overrides["jetspec_target_layer_ids"] = tuple(target_layer_ids)
    return overrides


def build_first_block(context_ids, *, init, mask_token_id, prompt_index, block_size=16,
                      seed=SYNC_SEED, device_gather=False):
    """``[anchor, 15 slots]``: mask tokens, or the rail's seeded random context tokens."""
    if init == "mask":
        anchor = context_ids[:, -1:]
        masks = torch.full((1, block_size - 1), int(mask_token_id), dtype=anchor.dtype,
                           device=anchor.device)
        return torch.cat([anchor, masks], dim=1)
    if init == "random_context":
        from jetspec.engines.jetspec_plus_plus.jetspec.drafter import build_random_context_block_token_ids
        return build_random_context_block_token_ids(
            context_ids, block_size=block_size, seed=seed, prompt_index=prompt_index, mode="token",
            device_gather=device_gather)
    raise ValueError(f"init must be one of {INITS}")


class SyncChainDrafter:
    """One fresh-context drafter call per round: anchor known, 15 drafts, ``passes`` forwards."""

    def __init__(self, *, head, embed_tokens, lm_head, passes, init, graph=False,
                 seed=SYNC_SEED):
        if init not in INITS:
            raise ValueError(f"init must be one of {INITS}")
        if init == "mask" and passes != 1:
            raise ValueError("a mask-initialised drafter runs one pass")
        self.eager_head = head
        self.head = head
        self.embed_tokens = embed_tokens
        self.lm_head = lm_head
        self.passes = int(passes)
        self.init = init
        self.seed = int(seed)
        self.prompt_index = 0
        self.block_size = int(head.block_size)
        self.mask_token_id = int(head.mask_token_id)
        self.dtype = next(head.parameters()).dtype
        self.device = next(head.parameters()).device
        mark = getattr(head, "adapted_position_embedding", None)
        # Row 0 marks a true-tap context row. Every row here is one, so row 1 is never read.
        self.fresh_mark = None if mark is None else mark.weight[0].detach()
        self.graph = bool(graph)
        self.self_check = None
        self.passes_run = []
        if self.graph:
            raise ValueError('retired graphed draft head requires a sealed historical read')

    def reset_cache(self) -> None:
        """Nothing survives a round: the context is rebuilt from the taps each call."""

    def set_prompt_index(self, prompt_index) -> None:
        self.prompt_index = int(prompt_index or 0)

    def graph_state(self) -> dict:
        captures = len(self.head.graph_keys) if self.graph else 0
        return {"head_graph": self.graph, "captures": captures, "query_replays": 0,
                "extension_replays": 0, "prefill_extensions": 0}

    @torch.inference_mode()
    def warm(self, max_context: int) -> list[int]:
        """Run one call per context shape the read can reach, before anything is timed.

        Graphed, that is one capture per bucket up to the first that holds ``max_context``; a
        bucket first met inside the measured phase would charge its capture to a round."""
        max_context = int(max_context)
        if self.graph:
            lengths = [value for value in SYNC_CONTEXT_BUCKETS if value < max_context]
            lengths.append(next(value for value in SYNC_CONTEXT_BUCKETS if value >= max_context))
        else:
            lengths = [min(64, max_context)]
        width = int(self.eager_head.fc.in_features)
        for length in lengths:
            hidden = torch.zeros((1, length, width), dtype=self.dtype, device=self.device)
            ids = torch.zeros((1, length + 1), dtype=torch.long, device=self.device)
            self.propose(ids, self.block_size - 1, target_hidden=hidden)
        self.passes_run.clear()
        return lengths

    def context(self, target_hidden: torch.Tensor) -> torch.Tensor:
        exact = target_hidden.to(device=self.device, dtype=self.dtype)
        if self.fresh_mark is None:
            return exact
        return exact + self.fresh_mark.to(dtype=exact.dtype)

    def forward_block(self, contexts, block_ids):
        noise = self.embed_tokens(block_ids)
        position_ids = torch.arange(int(contexts.shape[1]) + self.block_size, dtype=torch.long,
                                    device=self.device).unsqueeze(0)
        hidden = self.head(target_hidden=contexts, noise_embedding=noise,
                           position_ids=position_ids, past_key_values=None, use_cache=False,
                           is_causal=True, attention_backend="sdpa")
        return self.lm_head(hidden[:, 1:, :])[:, : self.block_size - 1, :]

    @torch.inference_mode()
    def propose(self, context_ids, k, *, target_hidden):
        from jetspec.engines.jetspec_plus_plus.jetspec.blockhead_serve import BlockheadProposal
        if int(k) != self.block_size - 1:
            raise ValueError(f"sync drafter proposes {self.block_size - 1} drafts, asked for {k}")
        if int(target_hidden.shape[1]) != int(context_ids.shape[1]) - 1:
            raise ValueError("fresh context must end immediately before the anchor")
        use_cuda_events = context_ids.device.type == "cuda"
        if use_cuda_events:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
        else:
            start_s = perf_counter()
        contexts = self.context(target_hidden)
        block_ids = build_first_block(
            context_ids, init=self.init, mask_token_id=self.mask_token_id,
            prompt_index=self.prompt_index, block_size=self.block_size, seed=self.seed)
        anchor = context_ids[:, -1:]
        passes_run = 0
        for _ in range(self.passes):
            logits = self.forward_block(contexts, block_ids)
            tokens = logits.argmax(dim=-1)
            passes_run += 1
            if self.passes == passes_run or bool((tokens == block_ids[:, 1:]).all()):
                break  # the cap, or a fixed point: a further pass would return the same block
            block_ids = torch.cat([anchor, tokens.to(block_ids.dtype)], dim=1)
        if use_cuda_events:
            end.record()
            end.synchronize()
            head_ms = float(start.elapsed_time(end))
        else:
            head_ms = (perf_counter() - start_s) * 1000.0
        self.passes_run.append(passes_run)
        return BlockheadProposal(tokens=tokens.reshape(-1)[:k], logits=logits[:, :k, :].clone(),
                                 head_ms=head_ms, markov_ms=0.0)


SYNC_STATIC_ENV = environment_name('engine', 'engine_settings.sync_static')
STATIC_CAPACITY = 768
STATIC_LANES = 16


def sync_static_requested(run_config=None) -> bool:
    config = run_config if run_config is not None else RunConfig.from_environment({})
    value = config.engine.engine_settings.sync_static
    if value not in ("0", "1"):
        raise ValueError(f"{SYNC_STATIC_ENV} must be 0 or 1")
    return value == "1"


def _rotate_half(value: torch.Tensor) -> torch.Tensor:
    half = value.shape[-1] // 2
    return torch.cat((-value[..., half:], value[..., :half]), dim=-1)


def _rotary(value: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return value * cos + _rotate_half(value) * sin


def _repeat_kv(value: torch.Tensor, groups: int) -> torch.Tensor:
    return value if groups == 1 else value.repeat_interleave(groups, dim=1)


class StaticSyncDrafter(SyncChainDrafter):
    """The same drafter call over a static context K/V cache, extended by the committed rows.

    A context row's keys and values depend on that row alone (its projected hidden state and its
    position), so they are written once when the row is committed and the block forward reads them:
    the per-round cost follows the accepted length, not the context length. ``graph=True`` replays
    one extension graph (up to 16 rows) and one block graph; ``graph=False`` runs the same two steps
    eagerly. A prompt's first call extends its whole context in 16-row steps, inside round one."""

    def __init__(self, *, capacity=STATIC_CAPACITY, **kwargs):
        graph = bool(kwargs.pop("graph", False))
        super().__init__(graph=False, **kwargs)
        head = self.eager_head
        if not head.resolve_causal_head("auto"):
            raise ValueError("the static sync drafter serves the causal block head")
        if getattr(head, "token_top_layer", None) is not None:
            raise ValueError("the static sync drafter refuses a token-top layer")
        if getattr(head, "has_masked_inject_layers", False):
            raise ValueError("the static sync drafter requires context injection in every layer")
        if any(layer.self_attn.sliding_window is not None for layer in head.layers):
            raise ValueError("the static sync drafter refuses sliding-window attention")
        if self.device.type != "cuda" and graph:
            raise ValueError("the graphed static sync drafter requires CUDA")
        self.graph = graph
        self.capacity = int(capacity)
        self.lanes = STATIC_LANES
        self._cache_length = 0
        self._graphs = {}
        self._state = {"head_graph": graph, "captures": 0, "query_replays": 0,
                       "extension_replays": 0, "prefill_extensions": 0}
        self._allocate()
        if graph:
            self._capture("extension", self._extension_step)
            self._capture("query", self._query_step)
            self._self_check()
        self.reset_cache()

    def _allocate(self) -> None:
        head, device, dtype = self.eager_head, self.device, self.dtype
        total = self.capacity + self.lanes  # the last ``lanes`` slots are scratch, never attended
        self._kv = []
        for layer in head.layers:
            attn = layer.self_attn
            shape = (1, int(attn.config.num_key_value_heads), total, int(attn.head_dim))
            self._kv.append((torch.zeros(shape, device=device, dtype=dtype),
                             torch.zeros(shape, device=device, dtype=dtype)))
        width = int(head.fc.in_features)
        self._extension_hidden = torch.zeros(1, self.lanes, width, device=device, dtype=dtype)
        self._extension_positions = torch.zeros(1, self.lanes, device=device, dtype=torch.long)
        self._lane_index = torch.arange(self.lanes, device=device, dtype=torch.long)
        self._extension_slots = self.capacity + self._lane_index
        self._block_index = torch.arange(self.block_size, device=device, dtype=torch.long)
        self._query_token_ids = torch.zeros(1, self.block_size, device=device, dtype=torch.long)
        self._query_positions = self._block_index.unsqueeze(0).clone()
        floor = torch.finfo(dtype).min
        self._query_mask = torch.full((1, 1, self.block_size, self.capacity + self.block_size), floor,
                                      device=device, dtype=dtype)
        causal = self._block_index.unsqueeze(0) <= self._block_index.unsqueeze(1)
        self._query_mask[0, 0, :, self.capacity:] = torch.where(causal, 0.0, floor).to(dtype)
        self._mask_floor = floor
        vocab = int(self.lm_head.weight.shape[0])
        self._query_logits = torch.empty(1, self.block_size - 1, vocab, device=device, dtype=dtype)
        self._query_tokens = torch.zeros(1, self.block_size - 1, device=device, dtype=torch.long)

    def _extension_step(self) -> None:
        head = self.eager_head
        hidden = self._extension_hidden
        if self.fresh_mark is not None:
            hidden = hidden + self.fresh_mark
        source = head.hidden_norm(head.fc(hidden))
        cos, sin = head.rotary_emb(source, self._extension_positions)
        for layer, (cache_k, cache_v) in zip(head.layers, self._kv):
            attn = layer.self_attn
            kv_heads = int(attn.config.num_key_value_heads)
            k = attn.k_proj(source).view(1, self.lanes, kv_heads, attn.head_dim)
            v = attn.v_proj(source).view(1, self.lanes, kv_heads, attn.head_dim)
            k = _rotary(attn.k_norm(k).transpose(1, 2), cos, sin)
            cache_k.index_copy_(2, self._extension_slots, k)
            cache_v.index_copy_(2, self._extension_slots, v.transpose(1, 2))

    def _query_step(self) -> None:
        head, size = self.eager_head, self.block_size
        hidden = self.embed_tokens(self._query_token_ids)
        cos, sin = head.rotary_emb(hidden, self._query_positions)
        for layer, (cache_k, cache_v) in zip(head.layers, self._kv):
            attn = layer.self_attn
            heads = int(attn.config.num_attention_heads)
            kv_heads = int(attn.config.num_key_value_heads)
            residual = hidden
            normed = layer.input_layernorm(hidden)
            q = attn.q_norm(attn.q_proj(normed).view(1, size, heads, attn.head_dim)).transpose(1, 2)
            k = attn.k_norm(attn.k_proj(normed).view(1, size, kv_heads, attn.head_dim)).transpose(1, 2)
            v = attn.v_proj(normed).view(1, size, kv_heads, attn.head_dim).transpose(1, 2)
            q, k = _rotary(q, cos, sin), _rotary(k, cos, sin)
            groups = int(attn.num_key_value_groups)
            keys = _repeat_kv(torch.cat((cache_k[:, :, : self.capacity], k), dim=2), groups)
            values = _repeat_kv(torch.cat((cache_v[:, :, : self.capacity], v), dim=2), groups)
            attended = torch.nn.functional.scaled_dot_product_attention(
                q, keys, values, attn_mask=self._query_mask, dropout_p=0.0, is_causal=False,
                scale=attn.scaling)
            attended = attended.transpose(1, 2).reshape(1, size, heads * attn.head_dim)
            hidden = residual + attn.o_proj(attended)
            if getattr(layer, "drafter_mlp", True):
                hidden = hidden + layer.mlp(layer.post_attention_layernorm(hidden))
        logits = self.lm_head(head.norm(hidden)[:, 1:, :])
        self._query_logits.copy_(logits)
        self._query_tokens.copy_(logits.argmax(dim=-1))

    @torch.inference_mode()
    def _capture(self, name, step) -> None:
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
        print(f"JETSPEC_SYNC_STATIC_CAPTURE phase={name} capacity={self.capacity} lanes={self.lanes}",
              flush=True)

    @torch.inference_mode()
    def _self_check(self) -> None:
        """A replay returns what the same step returns eagerly, byte for byte."""
        generator = torch.Generator(device="cpu").manual_seed(383)
        rows = torch.randn(self._extension_hidden.shape, generator=generator).to(
            device=self.device, dtype=self.dtype)
        self._extension_hidden.copy_(rows)
        self._extension_positions.copy_((40 + self._lane_index).unsqueeze(0))
        self._extension_slots.copy_(40 + self._lane_index)
        self._extension_step()
        expected = [tensor[:, :, 40:56].clone() for pair in self._kv for tensor in pair]
        for pair in self._kv:
            for tensor in pair:
                tensor[:, :, 40:56].zero_()
        self._graphs["extension"].replay()
        actual = [tensor[:, :, 40:56] for pair in self._kv for tensor in pair]
        if not all(torch.equal(left, right) for left, right in zip(expected, actual)):
            raise RuntimeError("static sync drafter: extension replay differs from the eager step")
        self._set_mask(0, 56)
        self._query_token_ids.copy_((17 + self._block_index).unsqueeze(0))
        self._query_positions.copy_((56 + self._block_index).unsqueeze(0))
        self._query_step()
        expected = (self._query_logits.clone(), self._query_tokens.clone())
        self._graphs["query"].replay()
        if not (torch.equal(expected[0], self._query_logits)
                and torch.equal(expected[1], self._query_tokens)):
            raise RuntimeError("static sync drafter: query replay differs from the eager step")
        torch.cuda.synchronize(self.device)
        self.self_check = {"extension_bitwise": True, "query_bitwise": True}
        print("JETSPEC_SYNC_STATIC_SELF_CHECK extension=bitwise query=bitwise", flush=True)

    def _set_mask(self, start: int, length: int) -> None:
        if start == 0:
            self._query_mask[..., : self.capacity] = self._mask_floor
        self._query_mask[..., start:length] = 0

    def reset_cache(self) -> None:
        self._cache_length = 0

    def graph_state(self) -> dict:
        return dict(self._state)

    def _run(self, name, step) -> None:
        if self.graph:
            self._graphs[name].replay()
        else:
            step()

    def _extend(self, target_hidden: torch.Tensor) -> None:
        length = int(target_hidden.shape[1])
        if length > self.capacity:
            raise ValueError(f"context length {length} exceeds the static cache capacity {self.capacity}")
        start = self._cache_length
        if length < start:
            raise ValueError("the fresh context shrank without a cache reset")
        first = start == 0
        while start < length:
            count = min(self.lanes, length - start)
            self._extension_hidden.zero_()
            self._extension_hidden[:, :count].copy_(target_hidden[:, start:start + count])
            torch.add(self._lane_index, start, out=self._extension_positions[0])
            self._extension_slots.copy_(torch.where(
                self._lane_index < count, start + self._lane_index, self.capacity + self._lane_index))
            self._run("extension", self._extension_step)
            self._state["prefill_extensions" if first else "extension_replays"] += 1
            start += count
        self._set_mask(self._cache_length, length)
        self._cache_length = length

    def warm(self, max_context: int) -> list[int]:
        """One full-length call before anything is timed; the graphs were captured at build."""
        length = min(int(max_context), self.capacity)
        width = int(self.eager_head.fc.in_features)
        hidden = torch.zeros((1, length, width), dtype=self.dtype, device=self.device)
        ids = torch.zeros((1, length + 1), dtype=torch.long, device=self.device)
        self.reset_cache()
        self.propose(ids, self.block_size - 1, target_hidden=hidden)
        self.reset_cache()
        self.passes_run.clear()
        for key in ("query_replays", "extension_replays", "prefill_extensions"):
            self._state[key] = 0
        return [length]

    @torch.inference_mode()
    def parity(self, length: int = 40) -> dict:
        """The static call against the full-context forward on one random context (recorded)."""
        generator = torch.Generator(device="cpu").manual_seed(384)
        width = int(self.eager_head.fc.in_features)
        hidden = torch.randn((1, length, width), generator=generator).to(self.device, self.dtype)
        ids = torch.randint(0, 1000, (1, length + 1), generator=generator).to(self.device)
        block = build_first_block(ids, init=self.init, mask_token_id=self.mask_token_id,
                                  prompt_index=0, block_size=self.block_size, seed=self.seed)
        reference = self.forward_block(self.context(hidden), block)
        self.reset_cache()
        self._extend(hidden)
        self._query_token_ids.copy_(block)
        torch.add(self._block_index, length, out=self._query_positions[0])
        self._run("query", self._query_step)
        # A position whose static token differs is a near-tie only when the reference scores that token within
        # the load check's bound of its own top logit; any other disagreement is a real static-path defect.
        bound = 64 * torch.finfo(self.dtype).eps
        chosen = reference.gather(-1, self._query_tokens.reshape(*reference.shape[:-1], 1)).squeeze(-1)
        disagree = reference.argmax(-1) != self._query_tokens
        result = {"max_abs_logit_diff": float((reference - self._query_logits).abs().max()),
                  "argmax_agree": int((~disagree).sum()),
                  "argmax_disagree_beyond_tie": int((disagree & (reference.max(-1).values - chosen > bound)).sum()),
                  "of": self.block_size - 1}
        self.reset_cache()
        for key in ("query_replays", "extension_replays", "prefill_extensions"):
            self._state[key] = 0
        return result

    @torch.inference_mode()
    def propose(self, context_ids, k, *, target_hidden):
        from jetspec.engines.jetspec_plus_plus.jetspec.blockhead_serve import BlockheadProposal
        if int(k) != self.block_size - 1:
            raise ValueError(f"sync drafter proposes {self.block_size - 1} drafts, asked for {k}")
        length = int(target_hidden.shape[1])
        if length != int(context_ids.shape[1]) - 1:
            raise ValueError("fresh context must end immediately before the anchor")
        use_cuda_events = self.device.type == "cuda"
        if use_cuda_events:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
        else:
            start_s = perf_counter()
        self._extend(target_hidden.to(device=self.device, dtype=self.dtype))
        block_ids = build_first_block(
            context_ids, init=self.init, mask_token_id=self.mask_token_id,
            prompt_index=self.prompt_index, block_size=self.block_size, seed=self.seed,
            device_gather=True)
        self._query_token_ids.copy_(block_ids)
        torch.add(self._block_index, length, out=self._query_positions[0])
        passes_run = 0
        while True:
            self._run("query", self._query_step)
            self._state["query_replays"] += 1
            passes_run += 1
            if passes_run == self.passes or bool(
                    (self._query_tokens == self._query_token_ids[:, 1:]).all()):
                break  # the cap, or a fixed point: a further pass would return the same block
            self._query_token_ids[:, 1:].copy_(self._query_tokens)
        tokens = self._query_tokens.reshape(-1).clone()
        logits = self._query_logits.clone()
        if use_cuda_events:
            end.record()
            end.synchronize()
            head_ms = float(start.elapsed_time(end))
        else:
            head_ms = (perf_counter() - start_s) * 1000.0
        self.passes_run.append(passes_run)
        return BlockheadProposal(tokens=tokens[:k], logits=logits[:, :k, :], head_ms=head_ms,
                                 markov_ms=0.0)


def load_sync_drafter(checkpoint_dir, *, device, max_context=None,
                      run_config=None) -> SyncChainDrafter:
    """Load through the asynchronous path's loader, fresh-only, and bind the run to what it read."""
    from jetspec.engines.jetspec_plus_plus.jetspec.drafter import JetSpecDrafter, _run_manifest
    checkpoint = Path(checkpoint_dir)
    run_config = run_config if run_config is not None else RunConfig.from_environment({})
    release = run_config.model.model_settings.target_release_weights or str(
        checkpoint / "target_release_weights.safetensors")
    passes = sync_passes(run_config)
    graph = sync_graph_requested(run_config)
    loaded = JetSpecDrafter.load(str(checkpoint), None, None, release, device, fresh_only=True, run_config=run_config)
    config = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    init = resolve_init(config.get("dflash_config"), passes)
    read_block = sync_block_size_read(run_config)
    if read_block != 16:
        loaded.head.config.block_size = read_block
        loaded.head.block_size = read_block
    if run_config.model.model_settings.draft_dtype == "fp32":
        for module in (loaded.head, loaded.embed_tokens, loaded.lm_head):
            module.float()
    static = sync_static_requested(run_config)
    from jetspec.engines.jetspec_plus_plus.serving.cache_capacity import capacity_for
    capacity = capacity_for(max_context or STATIC_CAPACITY) if static else None
    drafter = (StaticSyncDrafter if static else SyncChainDrafter)(
        head=loaded.head, embed_tokens=loaded.embed_tokens, lm_head=loaded.lm_head, passes=passes,
        init=init, graph=graph, **({"capacity": capacity} if static else {}))
    run_manifest = _run_manifest()
    run_manifest.record_value("method_mode", "jetspec_sync", run_config=run_config)
    run_manifest.record_value("sync_passes", passes, run_config=run_config)
    run_manifest.record_value("sync_init", init, run_config=run_config)
    run_manifest.record_value("sync_graph", graph, run_config=run_config)
    run_manifest.record_value("sync_context", "fresh", run_config=run_config)
    run_manifest.record_value("sync_draft_tokens", read_block - 1, run_config=run_config)
    run_manifest.record_value("sync_verify_tokens", read_block, run_config=run_config)
    if read_block != 16:
        run_manifest.record_value("block_size_read", read_block, run_config=run_config)
    run_manifest.record_value("sync_draft_dtype", str(drafter.dtype).removeprefix("torch."), run_config=run_config)
    run_manifest.record_value("sync_fresh_mark", drafter.fresh_mark is not None, run_config=run_config)
    run_manifest.record_value("sync_static_cache", static, run_config=run_config)
    if static:
        run_manifest.record_value("sync_static_cache_capacity", capacity, run_config=run_config)
    if static and drafter.device.type == "cuda":
        parity = drafter.parity()
        run_manifest.record_value("sync_static_parity", parity, run_config=run_config)
        run_manifest.record_value("sync_static_self_check", drafter.self_check, run_config=run_config)
        print("JETSPEC_SYNC_STATIC_PARITY " + json.dumps(parity, sort_keys=True), flush=True)
        # Allow a few dtype ULPs at ordinary logit magnitudes across the eager and
        # static attention kernels; token choices must agree except at a reference near-tie within that bound
        # (b32 graphed refused at 29 of 31 argmax agree with max diff 0.47).
        max_diff = 64 * torch.finfo(drafter.dtype).eps
        beyond_tie = parity.get("argmax_disagree_beyond_tie", parity["of"] - parity["argmax_agree"])
        if (beyond_tie != 0
                or not math.isfinite(parity["max_abs_logit_diff"])
                or parity["max_abs_logit_diff"] > max_diff):
            raise ValueError(f"sync static parity failed: {parity}, max_abs_logit_diff bound={max_diff}")
    warmed = [] if max_context is None else drafter.warm(max_context)
    run_manifest.record_value("sync_warmed_contexts", warmed, run_config=run_config)
    print(f"JETSPEC_SYNC_DRAFTER_LOAD passes={passes} init={init} graph={int(graph)} "
          f"dtype={str(drafter.dtype).removeprefix('torch.')} "
          f"fresh_mark={int(drafter.fresh_mark is not None)}", flush=True)
    return drafter
