"""Offline single-stream LLM — a small JetFlow-style API.

Plain autoregressive greedy/temperature decode over an HF `DynamicCache`
(prefill the prompt once, then single-token decode steps reusing the cache).
This is the 1x baseline; the draft head + tree verify build on the same seam.
Single sequence, batch=1 (the offline-inference regime).
"""
from dataclasses import dataclass
import time

import torch
from transformers import DynamicCache, StaticCache

from jetflow.models.qwen3 import load_target
from jetflow.core.model_runner import ModelRunner
from jetflow.core.sampler import sample


@dataclass
class SamplingParams:
    temperature: float = 0.0
    max_new_tokens: int = 256


class _CompileFriendlyDynamicCache(DynamicCache):
    """DynamicCache variant that does not infer logical length from KV tensor shape.

    HF's DynamicCache is convenient for the reference implementation, but under
    `torch.compile` its `get_seq_length()` path inspects the growing `keys`
    tensor (`numel()`, shape[-2]), which creates unstable Dynamo guards. The
    decode loops here already know the logical cache length from `cache_position`
    and accepted-token accounting, so store that length explicitly.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._jetflow_seq_length = 0

    def set_seq_length(self, seq_length: int) -> None:
        self._jetflow_seq_length = int(seq_length)

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return self._jetflow_seq_length

    def get_mask_sizes(self, cache_position: torch.Tensor | int, layer_idx: int) -> tuple[int, int]:
        if isinstance(cache_position, torch.Tensor):
            query_length = int(cache_position.numel())
        else:
            query_length = 1
        return self._jetflow_seq_length + query_length, 0


class _CompileFriendlyStaticCache(StaticCache):
    """StaticCache with explicit logical length for compiled HF decode."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._jetflow_seq_length = 0

    def set_seq_length(self, seq_length: int) -> None:
        self._jetflow_seq_length = int(seq_length)

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return self._jetflow_seq_length

    def get_mask_sizes(self, cache_position: torch.Tensor | int, layer_idx: int) -> tuple[int, int]:
        return self.layers[layer_idx].get_max_cache_shape(), 0


def _select_kv_cache(cache, keep_index: torch.Tensor) -> None:
    """Gather a `DynamicCache` along the sequence dim, keeping only `keep_index`
    (a 1-D LongTensor of cache positions, in increasing order), in place.

    The tree KV-cache verify path uses this to drop the rejected branches' KV
    after accepting one root-to-leaf path: the accepted nodes are a
    non-contiguous subset of the tree-ordered slots, but their positions are
    `[past, past+1, …, past+acc]`, so the gathered cache is an ordinary causal
    prefix again. Mirrors `DynamicCache.crop` (which slices `[..., :max_length, :]`
    along the seq dim) but for a non-contiguous keep set.
    """
    def _prefix_len(idx: torch.Tensor) -> int:
        if idx.numel() == 0:
            return 0
        ar = torch.arange(idx.numel(), device=idx.device, dtype=idx.dtype)
        diff = (idx != ar).nonzero(as_tuple=False)
        return int(diff[0, 0].item()) if diff.numel() else int(idx.numel())

    def _select_or_compact(keys, values, idx):
        idx = idx.to(keys.device)
        new_len = int(idx.numel())
        prefix_len = _prefix_len(idx)
        if prefix_len > 0:
            # Fast path for tree verify: keep_index = [0..prefix_len-1, accepted_tree_slots...].
            # The prefix is already in place, so only move accepted tree KV down and
            # then slice. This avoids O(prefix_len) copies per tree round.
            suffix = idx[prefix_len:]
            if suffix.numel():
                kept_keys = keys.index_select(-2, suffix)
                kept_values = values.index_select(-2, suffix)
                keys[..., prefix_len:new_len, :].copy_(kept_keys)
                values[..., prefix_len:new_len, :].copy_(kept_values)
            return keys[..., :new_len, :], values[..., :new_len, :]
        return keys.index_select(-2, idx), values.index_select(-2, idx)

    is_static = isinstance(cache, _CompileFriendlyStaticCache)
    layers = getattr(cache, "layers", None)
    if layers is not None:                       # transformers >= 4.54 (DynamicLayer)
        for layer in layers:
            keys = getattr(layer, "keys", None)
            if keys is None:
                continue                         # uninitialized / sliding layer — nothing cached
            if is_static:
                new_keys, new_values = _select_or_compact(keys, layer.values, keep_index)
                new_len = new_keys.shape[-2]
                keys[..., :new_len, :].copy_(new_keys)
                layer.values[..., :new_len, :].copy_(new_values)
            else:
                layer.keys, layer.values = _select_or_compact(keys, layer.values, keep_index)
    else:                                        # legacy key_cache / value_cache lists
        for i in range(len(cache.key_cache)):
            cache.key_cache[i], cache.value_cache[i] = _select_or_compact(
                cache.key_cache[i], cache.value_cache[i], keep_index
            )


class LLM:
    def __init__(
        self,
        model_name_or_path: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        attn_implementation: str = "sdpa",
    ):
        self.model, self.tokenizer = load_target(
            model_name_or_path, device, dtype, attn_implementation
        )
        self.runner = ModelRunner(self.model)
        self.device = device
        self.dtype = dtype
        self.eos_token_ids = self._resolve_eos()

    def _resolve_eos(self) -> set:
        ids = set()
        gc = getattr(self.model, "generation_config", None)
        for src in (
            getattr(self.tokenizer, "eos_token_id", None),
            getattr(gc, "eos_token_id", None) if gc is not None else None,
        ):
            if src is None:
                continue
            if isinstance(src, (list, tuple, set)):
                ids.update(int(x) for x in src)
            else:
                ids.add(int(src))
        return ids

    def _sample_last(self, logits: torch.Tensor, temperature: float) -> torch.Tensor:
        """Sample exactly one next token from the final logits row.

        Some HF backends may ignore/reshape the final-logits-only hint in
        prefill-like calls, so keep this scalar boundary explicit and robust.
        """
        if logits.dim() == 2:
            logits = logits.unsqueeze(1)
        else:
            logits = logits[:, -1:, :]
        tok = sample(logits, temperature).reshape(-1)[-1]
        return tok.view(1, 1)

    def _reset_drafter_cache(self, drafter) -> None:
        reset = getattr(drafter, "reset_cache", None)
        if reset is not None:
            reset()

    def _model_config(self):
        if hasattr(self.model, "config"):
            return self.model.config
        orig = getattr(self.model, "_orig_mod", None)
        return getattr(orig, "config", None)

    def _is_compiled_model(self) -> bool:
        return hasattr(self.model, "_orig_mod")

    def _new_cache(self, max_cache_len: int | None = None) -> DynamicCache:
        """Create a fully layered HF cache so compiled forwards see stable state."""
        config = self._model_config()
        if self._is_compiled_model() and config is not None and max_cache_len is not None:
            cache = _CompileFriendlyStaticCache(config=config, max_cache_len=int(max_cache_len))
        else:
            cache = _CompileFriendlyDynamicCache(config=config) if config is not None else _CompileFriendlyDynamicCache()
        early_init = getattr(cache, "early_initialization", None)
        if early_init is not None and config is not None:
            param = next(self.model.parameters(), None)
            dtype = getattr(self, "dtype", None) or (param.dtype if param is not None else torch.bfloat16)
            device = torch.device(getattr(self, "device", param.device if param is not None else "cpu"))
            head_dim = getattr(config, "head_dim", None)
            if head_dim is None:
                head_dim = config.hidden_size // config.num_attention_heads
            num_kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
            early_init(
                batch_size=1,
                num_heads=num_kv_heads,
                head_dim=head_dim,
                dtype=dtype,
                device=device,
            )
            empty_kv = torch.empty((1, num_kv_heads, 0, head_dim), dtype=dtype, device=device)
            for layer in getattr(cache, "layers", []):
                keys = getattr(layer, "keys", None)
                if keys is not None and keys.dim() == 1 and keys.numel() == 0:
                    layer.keys = empty_kv
                    layer.values = empty_kv
        return cache

    def _timer(self) -> float:
        device = torch.device(self.device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        return time.perf_counter()

    @torch.inference_mode()
    def generate(self, prompt, sampling_params: SamplingParams = None) -> dict:
        """Greedy/temperature decode. `prompt` is a str (tokenized raw) or an
        already-tokenized `input_ids` tensor (1, T). Returns {token_ids, text}."""
        sp = sampling_params or SamplingParams()
        if isinstance(prompt, str):
            input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        else:
            input_ids = prompt.to(self.device)
        prompt_len = input_ids.shape[1]
        cache = self._new_cache(max_cache_len=prompt_len + sp.max_new_tokens + 1)

        # --- prefill: process the whole prompt once, sample the first token ---
        pos = torch.arange(prompt_len, device=self.device).unsqueeze(0)
        logits, cache, _ = self.runner.forward(
            input_ids, cache, pos, cache_position=pos.squeeze(0),
            last_position_logits_only=True,
        )
        cache.set_seq_length(prompt_len)
        next_tok = self._sample_last(logits, sp.temperature)  # (1, 1)
        out_ids = [int(next_tok.item())]
        cur = prompt_len
        decode_start = self._timer()

        # --- decode: single-token steps reusing the KV cache (no reprocess) ---
        for _ in range(sp.max_new_tokens - 1):
            if out_ids[-1] in self.eos_token_ids:
                break
            pos = torch.tensor([[cur]], device=self.device)
            logits, cache, _ = self.runner.forward(
                next_tok, cache, pos, cache_position=pos.squeeze(0),
                last_position_logits_only=True,
            )
            cache.set_seq_length(cur + 1)
            next_tok = self._sample_last(logits, sp.temperature)
            out_ids.append(int(next_tok.item()))
            cur += 1

        decode_time = self._timer() - decode_start
        text = self.tokenizer.decode(out_ids, skip_special_tokens=True)
        return {"token_ids": out_ids, "text": text, "decode_time": decode_time}

    @torch.inference_mode()
    def generate_chain(self, prompt, drafter, block_size: int = 4,
                       target_layer_ids=None, sampling_params: SamplingParams = None) -> dict:
        """Chain (linear) speculative decode over a persistent target KV cache.
        Each round the drafter proposes `block_size-1` tokens; the target verifies
        them in ONE forward that processes only [anchor | drafts] against the cached
        prefix (no prefix recompute); we accept the longest greedy-agreeing prefix
        plus one correction, then crop the cache to drop the rejected drafts' KV.

        Lossless by construction (commits only the verify forward's own greedy, for
        any drafter). The persistent cache makes the verify cheap — it processes
        only the new tokens, no prefix recompute — which is the wall-clock win.
        Output matches plain AR `generate()` to within rare bf16-borderline flips: a
        block forward (many queries) and a single-token forward differ in SDPA
        reduction order, so a borderline argmax can flip after ~tens of exact tokens.
        That gap is inherent to bf16 block verification (not a recompute artifact);
        fp32 would be exact. Returns {token_ids, text, tpf}, tpf = new / verify-forwards.

        `target_layer_ids` (the head's tapped layers): when set with block_size>1,
        each verify forward extracts `target_hidden` for the new tokens, accumulated
        in lockstep with the cache and threaded into the next `drafter.propose(...)`
        (the real DraftHead conditions on it; stubs ignore it)."""
        sp = sampling_params or SamplingParams()
        if isinstance(prompt, str):
            committed = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        else:
            committed = prompt.to(self.device)
        self._reset_drafter_cache(drafter)
        k = max(1, block_size - 1)
        need_hidden = target_layer_ids is not None and block_size > 1
        new_ids, rounds = [], 0
        target_hidden = None

        # --- prefill: populate the persistent cache with the prompt's KV ---
        cache = self._new_cache(max_cache_len=committed.shape[1] + sp.max_new_tokens + k + 1)
        pos = torch.arange(committed.shape[1], device=self.device).unsqueeze(0)
        logits, cache, full_hidden = self.runner.forward(
            committed, cache, pos, cache_position=pos.squeeze(0),
            output_hidden_states=need_hidden, target_layer_ids=target_layer_ids,
            last_position_logits_only=True,
        )
        cache.set_seq_length(committed.shape[1])
        if full_hidden is not None:
            target_hidden = full_hidden          # prompt context; next anchor (first_tok) fed via noise
        first_tok = self._sample_last(logits, sp.temperature)
        new_ids.append(int(first_tok.item()))
        committed = torch.cat([committed, first_tok.view(1, 1)], dim=1)   # anchor; NOT yet in cache
        if int(first_tok.item()) in self.eos_token_ids:
            new_ids = new_ids[: sp.max_new_tokens]
            return {"token_ids": new_ids, "text": self.tokenizer.decode(new_ids, skip_special_tokens=True), "tpf": 0.0}

        # Invariant each round: cache.get_seq_length() == committed.shape[1] - 1 ==
        # target_hidden.shape[1] (when need_hidden). The cache trails `committed` by
        # the anchor, which enters the cache as the first token of the next forward.
        while len(new_ids) < sp.max_new_tokens:
            drafts = drafter.propose(committed, k, target_hidden=target_hidden).to(self.device).view(-1)[:k]   # (k,)
            anchor = committed[:, -1:]                                  # (1,1) — not yet cached
            step_ids = torch.cat([anchor, drafts.view(1, -1)], dim=1)   # (1, 1+k): only the new tokens
            past_len = cache.get_seq_length()                           # == committed.shape[1] - 1
            span = torch.arange(past_len, past_len + step_ids.shape[1], device=self.device)
            logits, cache, new_hidden = self.runner.forward(
                step_ids, cache, span.unsqueeze(0), cache_position=span,
                output_hidden_states=need_hidden, target_layer_ids=target_layer_ids,
            )
            rounds += 1
            # step_ids[0,0] is the anchor, so logits[0,0] is already the post-anchor
            # prediction — NO p-1 offset (the cache holds the prefix).
            tgt = logits[0, :, :].argmax(-1)                            # (1+k,)
            acc = 0
            for i in range(k):
                if int(drafts[i]) == int(tgt[i]):
                    acc += 1
                else:
                    break
            correction = tgt[acc]
            if not isinstance(cache, _CompileFriendlyStaticCache):
                cache.crop(past_len + 1 + acc)    # keep prefix + anchor + accepted drafts; drop rejected KV
            cache.set_seq_length(past_len + 1 + acc)
            if new_hidden is not None:
                # append [anchor | accepted drafts] hidden (the correction has none
                # yet — it is the next anchor, fed via noise). Restores the invariant.
                target_hidden = torch.cat([target_hidden, new_hidden[:, :1 + acc, :]], dim=1)
            block_new = torch.cat([drafts[:acc], correction.view(1)])   # accepted + correction
            committed = torch.cat([committed, block_new.view(1, -1)], dim=1)
            for t in block_new.tolist():
                new_ids.append(int(t))
                if int(t) in self.eos_token_ids:
                    break
            if new_ids and new_ids[-1] in self.eos_token_ids:
                break
        new_ids = new_ids[: sp.max_new_tokens]
        text = self.tokenizer.decode(new_ids, skip_special_tokens=True)
        return {"token_ids": new_ids, "text": text, "tpf": (len(new_ids) / rounds if rounds else 0.0)}

    @torch.inference_mode()
    def generate_tree(self, prompt, tree_drafter, block_size: int = 4, tree_width: int = 2,
                      budget: int = 15, algo: str = "accum_logp", algo_kwargs: dict = None,
                      target_layer_ids=None, sampling_params: SamplingParams = None,
                      return_stats: bool = False, profile_phases: bool = False,
                      prompt_info: dict = None, profile_table: dict = None,
                      tree_attn: str = "sdpa") -> dict:
        """Tree speculative decode. Each round: the tree drafter emits per-depth
        logits, the tree algorithm builds a DraftTree, the target verifies all
        nodes against a persistent KV cache, and tree_accept takes the longest
        greedy-agreeing root-to-leaf path + a correction. Returns
        {token_ids, text, tpf}.

        `algo` selects a registered tree algorithm (see `jetflow.tree.list_algorithms`);
        `algo_kwargs` passes its constructor knobs (e.g. {"beta": 2.0} for
        top2gap_fanout). `prompt_info` (optional dict: task label / reasoning mode
        / decoded text) is forwarded to the algorithm's build() for the
        prompt-adaptive (semantic_aware) algorithms; None → they use their
        logit-fingerprint fallback. `profile_table` (optional dict: offline
        per-(depth,rank) acceptance, from bench/profiling/collect_depth_rank_stats.py) is forwarded to
        the profile-guided algorithms (depth_rank_histogram); None → they recover
        accum_logp. All bundled algorithms recover accum_logp at their identity
        knobs, so the choice is lossless regardless.

        `target_layer_ids` (the head's tapped layers): when set with block_size>1,
        each verify forward extracts `target_hidden`, threaded into the next
        `tree_drafter.propose_logits(...)`. The next anchor is the deepest accepted
        node's hidden (the correction token has no hidden yet).

        Verification always uses the persistent-cache wall-clock path. Historical
        full-recompute diagnostics belong in benchmark/debug scripts rather than
        this core execution path."""
        # tree contract (engine -> tree, one-way): import only the public jetflow.tree API
        from jetflow.tree import get_algorithm

        sp = sampling_params or SamplingParams()
        if isinstance(prompt, str):
            committed = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        else:
            committed = prompt.to(self.device)
        self._reset_drafter_cache(tree_drafter)
        D = max(1, block_size - 1)
        algo_obj = get_algorithm(algo, **(algo_kwargs or {}))
        if tree_attn not in {"sdpa", "triton"}:
            raise ValueError(f"unknown tree_attn {tree_attn!r}; expected 'sdpa' or 'triton'")
        return self._generate_tree_kv_cached(
            committed, tree_drafter, block_size, tree_width, budget, algo_obj,
            target_layer_ids, sp, return_stats, prompt_info, profile_table,
            tree_attn, profile_phases,
        )

    @torch.inference_mode()
    def _generate_tree_kv_cached(self, committed, tree_drafter, block_size, tree_width,
                                 budget, algo_obj, target_layer_ids, sp,
                                 return_stats, prompt_info, profile_table,
                                 tree_attn: str = "sdpa", profile_phases: bool = False) -> dict:
        """Tree spec decode over a PERSISTENT target KV cache (the wall-clock path).

        Mirrors `generate_chain`'s persistent-cache verify, extended to trees: each
        round forwards only the tree nodes against the cached prefix (no prefix
        recompute), then GATHERS the accepted root-to-leaf path's KV back into a
        linear cache (dropping the rejected branches). The accepted nodes are a
        non-contiguous subset of the tree-ordered slots, but their positions are
        `[past, past+1, …, past+acc]` (root + one node per accepted depth), so the
        gathered cache is an ordinary causal prefix again — `committed[:-1]`.

        Lossless by construction (commits the verify forward's own greedy along the
        accepted path). Dispatched from `generate_tree`; args are already parsed there."""
        # tree contract (engine -> tree, one-way): import only the public jetflow.tree API
        from jetflow.models.draft_head import extract_context_feature
        from jetflow.tree import build_ancestor_matrix, tree_accept
        if tree_attn == "triton":
            from jetflow.core.tree_attention_kernel import (
                use_attention_implementation,
                use_tree_attention,
            )

        dtype = next(self.model.parameters()).dtype
        neg = torch.finfo(dtype).min
        D = max(1, block_size - 1)
        need_hidden = target_layer_ids is not None and block_size > 1
        new_ids, rounds = [], 0
        accept_lengths, tree_sizes = [], []   # per-round (acc+1) and node count (return_stats)
        phase_times = {"draft": 0.0, "tree_build": 0.0, "verify": 0.0, "accept": 0.0, "kv_select": 0.0}
        target_hidden = None

        def _phase(name, fn):
            if not profile_phases:
                return fn()
            start = self._timer()
            out = fn()
            phase_times[name] += self._timer() - start
            return out

        def _tree_out(decode_time: float) -> dict:
            text = self.tokenizer.decode(new_ids, skip_special_tokens=True)
            out = {
                "token_ids": new_ids,
                "text": text,
                "tpf": (len(new_ids) / rounds if rounds else 0.0),
                "decode_time": decode_time,
            }
            if return_stats:
                out["accept_lengths"] = accept_lengths
                out["tree_sizes"] = tree_sizes
                out["rounds"] = rounds
                if profile_phases:
                    out["phase_times"] = dict(phase_times)
            return out

        # --- prefill: populate the persistent cache with the prompt's KV ---
        prompt_len = committed.shape[1]
        cache = DynamicCache()
        max_len = prompt_len + sp.max_new_tokens + max(1, budget)
        position_ids = torch.arange(max_len + 1, device=self.device).unsqueeze(0)
        prefill_out = self.model(
            input_ids=committed,
            position_ids=position_ids[:, :prompt_len],
            past_key_values=cache,
            cache_position=position_ids[0, :prompt_len],
            use_cache=True,
            output_hidden_states=need_hidden,
            logits_to_keep=1,
        )
        if need_hidden:
            target_hidden = extract_context_feature(prefill_out.hidden_states, target_layer_ids)
        first_tok = self._sample_last(prefill_out.logits, sp.temperature)
        new_ids.append(int(first_tok.item()))
        committed = torch.cat([committed, first_tok.view(1, 1)], dim=1)   # anchor; NOT yet cached
        if int(first_tok.item()) in self.eos_token_ids:
            new_ids = new_ids[: sp.max_new_tokens]
            return _tree_out(0.0)

        # Invariant each round: cache.get_seq_length() == committed.shape[1] - 1 ==
        # target_hidden.shape[1] (when need_hidden). The cache trails `committed` by
        # the anchor (= tree root), which enters the cache via the verify forward.
        decode_start = None
        while len(new_ids) < sp.max_new_tokens:
            draft_logits = _phase(
                "draft",
                lambda: tree_drafter.propose_logits(
                    committed, D, target_hidden=target_hidden
                ).to(self.device),
            )  # (1, D, V)
            if decode_start is None:
                decode_start = self._timer()
            tree = _phase(
                "tree_build",
                lambda: algo_obj.build(
                    int(committed[0, -1]), draft_logits, block_size, tree_width,
                    budget, self.device, prompt_info=prompt_info,
                    profile_table=profile_table,
                ),
            )
            N = tree.num_nodes
            past_len = committed.shape[1] - 1                          # == logical cache length
            # feed only the tree nodes (node 0 = anchor/root); the prefix KV is cached.
            seq_step = tree.token_ids.view(1, -1)                      # (1, N)
            posN = (past_len + tree.depth).unsqueeze(0).long()         # RoPE: depth-based
            cache_pos = torch.arange(past_len, past_len + N, device=self.device)        # contiguous append slots
            # 4D additive mask: queries = N nodes; keys = past_len cached prefix
            # (all visible) + N nodes (ancestor mask, incl self).
            ancestor = build_ancestor_matrix(tree).bool()
            if tree_attn == "triton":
                attn_ancestor = (
                    tree.ancestor_packed if tree.ancestor_packed is not None
                    else ancestor.to(dtype=torch.uint8)
                )
                with (
                    use_attention_implementation(self.model, "sdpa"),
                    use_tree_attention(attn_ancestor, prefix_len=past_len, attn_impl="triton"),
                ):
                    # The Triton tree-attn handler ignores HF's attention_mask and
                    # reads the ancestor mask from `use_tree_attention`. Passing a
                    # 4D placeholder makes HF skip DynamicCache mask-size inference,
                    # which otherwise inspects growing KV tensors under torch.compile.
                    mask = torch.empty((1, 1, N, past_len + N), dtype=torch.bool, device=self.device)
                    out = _phase(
                        "verify",
                        lambda: self.model(
                            input_ids=seq_step,
                            position_ids=posN,
                            attention_mask=mask,
                            past_key_values=cache,
                            cache_position=cache_pos,
                            use_cache=True,
                            output_hidden_states=need_hidden,
                        ),
                    )
            else:
                allowed = torch.zeros(N, past_len + N, dtype=torch.bool, device=self.device)
                allowed[:, :past_len] = True
                allowed[:, past_len:] = ancestor
                mask = torch.where(allowed, torch.zeros((), dtype=dtype, device=self.device),
                                   torch.full((), neg, dtype=dtype, device=self.device)).view(1, 1, N, past_len + N)
                out = _phase(
                    "verify",
                    lambda: self.model(
                        input_ids=seq_step,
                        position_ids=posN,
                        attention_mask=mask,
                        past_key_values=cache,
                        cache_position=cache_pos,
                        use_cache=True,
                        output_hidden_states=need_hidden,
                    ),
                )
            target_logits = out.logits                                # (1, N, V) — every row is a tree node
            accepted_path, acc, correction = _phase(
                "accept",
                lambda: tree_accept(tree, target_logits, sp.temperature),
            )
            # GATHER: keep prefix + accepted path (root + accepted nodes); drop the
            # rejected branches' KV. accepted_path = [0(root), …acc nodes], tree-ordered;
            # their cache slots are past_len + path -> contiguous positions after gather.
            keep = torch.cat([
                torch.arange(past_len, device=self.device),
                past_len + torch.tensor(accepted_path, device=self.device),
            ])
            _phase("kv_select", lambda: _select_kv_cache(cache, keep))
            if need_hidden:
                # append [root | accepted nodes] hidden (the correction has none yet —
                # it is the next anchor, fed via noise). Restores the invariant.
                sel = torch.tensor(accepted_path, device=self.device)
                new_hidden = extract_context_feature(out.hidden_states, target_layer_ids)
                target_hidden = torch.cat([target_hidden, new_hidden[:, sel, :]], dim=1)
            accepted = tree.token_ids[torch.tensor(accepted_path[1:], device=self.device)] if acc > 0 \
                else torch.empty(0, dtype=tree.token_ids.dtype, device=self.device)
            block = torch.cat([accepted, torch.tensor([correction], device=self.device)])
            committed = torch.cat([committed, block.view(1, -1)], dim=1)
            rounds += 1
            accept_lengths.append(int(block.numel()))   # acc + 1, matches reference accept-len
            tree_sizes.append(int(N))
            for t in block.tolist():
                new_ids.append(int(t))
                if int(t) in self.eos_token_ids:
                    break
            if new_ids and new_ids[-1] in self.eos_token_ids:
                break
        new_ids = new_ids[: sp.max_new_tokens]
        decode_time = (self._timer() - decode_start) if decode_start is not None else 0.0
        return _tree_out(decode_time)
