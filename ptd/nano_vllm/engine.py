"""NanoEngine — single-stream AR decode over a paged KV cache (nano_vllm N0).

The owned-substrate analogue of `ptd.engine.llm.LLM.generate()`: plain
greedy/temperature decode (prefill the prompt once, then single-token steps
reusing the cache), with the HF `DynamicCache` swapped for `PagedKVCache`. The
paging is invisible to the model forward (`ModelRunner` is reused as-is), so the
loop is the same — this is the N0 lossless gate: token-identical to `LLM.generate`
on the same model/device/dtype (fp32 exact on CPU; bf16 on b200, modulo the
documented SDPA-reduction-order borderline flips).

N0 is vanilla AR (no tree). N1 layers tree spec on the same paged cache, using
`PagedKVCache.gather` for the accepted-path KV (the paged `_select_kv_cache`).
N2a adds `generate_batch`: continuous-batched AR over the *multi-sequence* paged
pool (one shared cache, per-`seq_id` block tables), token-identical to running
`generate` on each prompt alone (the N2a lossless gate). N2b adds
`generate_tree_batch`: batched per-sequence TREE-spec over the same pool (each seq
builds its own tree, one batched verify forward under a padded per-seq 4D ancestor
mask, per-seq accept + ref-count-safe gather), token-identical to running
`generate_tree` on each prompt alone (the N2b lossless gate).
"""
import torch
from transformers import DynamicCache

from ptd.models.qwen3 import load_target
from ptd.engine.llm import SamplingParams
from ptd.engine.model_runner import ModelRunner
from ptd.engine.sampler import sample
from ptd.nano_vllm.paged_kv_cache import PagedKVCache
from ptd.nano_vllm.scheduler import Scheduler, SequenceRequest


class NanoEngine:
    def __init__(
        self,
        model_name_or_path: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        block_size: int = 16,
        attn_implementation: str = "sdpa",
        attn_backend: str = "sdpa",
    ):
        self.model, self.tokenizer = load_target(
            model_name_or_path, device, dtype, attn_implementation
        )
        self.runner = ModelRunner(self.model)
        self.device = device
        self.dtype = dtype
        self.block_size = block_size
        self.eos_token_ids = self._resolve_eos()
        # N3 attention backend (opt-in). "sdpa" (default) is byte-identical to the
        # pre-N3 engine. "triton_paged_tree" swaps the RUNTIME attention interface
        # for the paged tree-attention kernel (the model still loads with sdpa
        # weights/format; only the interface HF dispatches to is replaced). Affects
        # N0/N1/N2a; N2b stays on SDPA regardless (see generate_tree_batch).
        self.attn_backend = attn_backend
        if attn_backend == "triton_paged_tree":
            from ptd.nano_vllm.paged_attn_backend import register_ptd_paged_tree

            register_ptd_paged_tree()
            self.model.config._attn_implementation = "ptd_paged_tree"

    def _resolve_eos(self) -> set:
        """EOS ids from tokenizer + generation_config (mirrors `LLM._resolve_eos`)."""
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

    @torch.inference_mode()
    def generate(self, prompt, sampling_params: SamplingParams = None) -> dict:
        """Greedy/temperature decode over the paged cache. `prompt` is a str
        (tokenized raw) or an already-tokenized `input_ids` tensor (1, T). Returns
        {token_ids, text}. Token-identical to `LLM.generate` (the N0 gate)."""
        sp = sampling_params or SamplingParams()
        if isinstance(prompt, str):
            input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        else:
            input_ids = prompt.to(self.device)
        prompt_len = input_ids.shape[1]
        cache = PagedKVCache(
            block_size=self.block_size, device=torch.device(self.device), dtype=self.dtype
        )
        # N3 kernel path: route the single seq (id 0) through the paged tree-attn
        # kernel. Prefill runs on the kernel too (qq_bias=None -> pure causal, since
        # context_len = seq_len - S = 0). attention_mask=None: the kernel masks.
        # getattr default keeps object.__new__-built engines (test fixtures) on SDPA.
        kernel = getattr(self, "attn_backend", "sdpa") == "triton_paged_tree"
        if kernel:
            cache._paged_handoff = True
            cache._handoff_seq_ids = [0]
            cache._ptd_attn_meta = {"seq_ids": [0], "qq_bias": None}

        # --- prefill: process the whole prompt once, sample the first token ---
        # (attention_mask stays None for both paths: SDPA builds its own causal
        # mask from cache_position; the kernel masks internally.)
        pos = torch.arange(prompt_len, device=self.device).unsqueeze(0)
        logits, cache, _ = self.runner.forward(input_ids, cache, pos)
        next_tok = sample(logits[:, -1:, :], sp.temperature)  # (1, 1)
        out_ids = [int(next_tok.item())]
        cur = prompt_len

        # --- decode: single-token steps reusing the paged cache (no reprocess) ---
        for _ in range(sp.max_new_tokens - 1):
            if out_ids[-1] in self.eos_token_ids:
                break
            pos = torch.tensor([[cur]], device=self.device)
            logits, cache, _ = self.runner.forward(next_tok, cache, pos)
            next_tok = sample(logits[:, -1:, :], sp.temperature)
            out_ids.append(int(next_tok.item()))
            cur += 1

        text = self.tokenizer.decode(out_ids, skip_special_tokens=True)
        return {"token_ids": out_ids, "text": text}

    @torch.inference_mode()
    def generate_tree(self, prompt, tree_drafter, block_size: int = 4, tree_width: int = 2,
                      budget: int = 15, algo: str = "crossproduct", algo_kwargs: dict = None,
                      target_layer_ids=None, sampling_params: SamplingParams = None,
                      return_stats: bool = False, prompt_info: dict = None,
                      profile_table: dict = None) -> dict:
        """Tree speculative decode over a PERSISTENT paged KV cache (nano_vllm N1).

        The owned-substrate analogue of `ptd.engine.llm.LLM._generate_tree_kv_cached`:
        each round the tree drafter emits per-depth logits, the tree algorithm builds
        a DraftTree, the target verifies all nodes in ONE forward under a 4D ancestor
        mask (appending only the tree nodes against the cached prefix — no prefix
        recompute), `tree_accept` takes the longest greedy-agreeing root-to-leaf path
        plus a correction, and the accepted path's KV is GATHERED back into a linear
        prefix (dropping the rejected branches). The only difference from the
        `DynamicCache` reference is the cache class: `PagedKVCache.gather` replaces
        `_select_kv_cache`. Lossless by construction (commits the verify forward's own
        greedy along the accepted path) — token-identical to `LLM._generate_tree_kv_cached`
        and to plain greedy `generate()` in fp32 (the N1 gate); bf16 may flip a
        borderline argmax after ~tens of exact tokens (cached prefix KV vs a fresh
        recompute differ in SDPA reduction order). Returns {token_ids, text, tpf}.

        `algo` / `algo_kwargs` / `prompt_info` / `profile_table` / `target_layer_ids`
        mirror `LLM.generate_tree`; all bundled algorithms recover crossproduct at
        their identity knobs, so the choice is lossless regardless."""
        # tree contract (engine -> tree, one-way): import only the public ptd.tree API
        from ptd.tree import get_algorithm, build_ancestor_matrix, tree_accept

        sp = sampling_params or SamplingParams()
        if isinstance(prompt, str):
            committed = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        else:
            committed = prompt.to(self.device)
        # Drop any per-layer context K/V from a prior call; the optional DraftHead
        # context cache extends as target_hidden grows within THIS generation. No-op
        # for drafters without a persistent cache (recompute path).
        reset_ctx = getattr(tree_drafter, "reset_context_cache", None)
        if callable(reset_ctx):
            reset_ctx()
        D = max(1, block_size - 1)
        algo_obj = get_algorithm(algo, **(algo_kwargs or {}))
        dtype = self.dtype
        neg = torch.finfo(dtype).min
        need_hidden = target_layer_ids is not None and block_size > 1
        new_ids, rounds = [], 0
        accept_lengths, tree_sizes = [], []   # per-round (acc+1) and node count (return_stats)
        target_hidden = None

        kernel = getattr(self, "attn_backend", "sdpa") == "triton_paged_tree"

        # --- prefill: populate the persistent paged cache with the prompt's KV ---
        cache = PagedKVCache(
            block_size=self.block_size, device=torch.device(self.device), dtype=self.dtype
        )
        if kernel:
            # Prefill runs on the kernel too (single seq 0, qq_bias=None -> pure
            # causal over the prompt: context_len = seq_len - S = 0).
            cache._paged_handoff = True
            cache._handoff_seq_ids = [0]
            cache._ptd_attn_meta = {"seq_ids": [0], "qq_bias": None}
        pos = torch.arange(committed.shape[1], device=self.device).unsqueeze(0)
        logits, cache, full_hidden = self.runner.forward(
            committed, cache, pos,
            output_hidden_states=need_hidden, target_layer_ids=target_layer_ids,
        )
        if full_hidden is not None:
            target_hidden = full_hidden          # prompt context; anchor (first_tok) fed via noise
        first_tok = sample(logits[:, -1:, :], sp.temperature)
        new_ids.append(int(first_tok.item()))
        committed = torch.cat([committed, first_tok.view(1, 1)], dim=1)   # anchor; NOT yet cached
        if int(first_tok.item()) in self.eos_token_ids:
            new_ids = new_ids[: sp.max_new_tokens]
            return {"token_ids": new_ids, "text": self.tokenizer.decode(new_ids, skip_special_tokens=True), "tpf": 0.0}

        # Invariant each round: cache.get_seq_length() == committed.shape[1] - 1 ==
        # target_hidden.shape[1] (when need_hidden). The cache trails `committed` by
        # the anchor (= tree root), which enters the cache via the verify forward.
        while len(new_ids) < sp.max_new_tokens:
            draft_logits = tree_drafter.propose_logits(committed, D, target_hidden=target_hidden).to(self.device)  # (1, D, V)
            tree = algo_obj.build(int(committed[0, -1]), draft_logits, block_size, tree_width, budget, self.device,
                                  prompt_info=prompt_info, profile_table=profile_table)
            N = tree.num_nodes
            past_len = cache.get_seq_length()                          # == committed.shape[1] - 1
            # feed only the tree nodes (node 0 = anchor/root); the prefix KV is cached.
            seq_step = tree.token_ids.view(1, -1)                      # (1, N)
            depths = tree.depth.tolist()
            posN = torch.tensor([[past_len + d for d in depths]], device=self.device)   # RoPE: depth-based
            cache_pos = torch.arange(past_len, past_len + N, device=self.device)        # contiguous append slots
            if kernel:
                # Kernel path: prefix [0, past_len) is always-visible (handled by the
                # kernel); the N tree nodes attend per the ancestor mask folded in as
                # the fp32 (0/-inf) qq_bias. No dense 4D mask — attention_mask=None.
                anc = build_ancestor_matrix(tree).to(device=self.device, dtype=torch.bool)
                qq_bias = torch.where(
                    anc, torch.zeros((), dtype=torch.float32, device=self.device),
                    torch.full((), float("-inf"), dtype=torch.float32, device=self.device),
                )
                cache._handoff_seq_ids = [0]
                cache._ptd_attn_meta = {"seq_ids": [0], "qq_bias": qq_bias}
                logits, cache, new_hidden = self.runner.forward(
                    seq_step, cache, posN, attention_mask=None, cache_position=cache_pos,
                    output_hidden_states=need_hidden, target_layer_ids=target_layer_ids,
                )
            else:
                # 4D additive mask: queries = N nodes; keys = past_len cached prefix
                # (all visible) + N nodes (ancestor mask, incl self).
                allowed = torch.zeros(N, past_len + N, dtype=torch.bool, device=self.device)
                allowed[:, :past_len] = True
                allowed[:, past_len:] = build_ancestor_matrix(tree).bool()
                mask = torch.where(allowed, torch.zeros((), dtype=dtype, device=self.device),
                                   torch.full((), neg, dtype=dtype, device=self.device)).view(1, 1, N, past_len + N)
                logits, cache, new_hidden = self.runner.forward(
                    seq_step, cache, posN, attention_mask=mask, cache_position=cache_pos,
                    output_hidden_states=need_hidden, target_layer_ids=target_layer_ids,
                )
            target_logits = logits                                    # (1, N, V) — every row is a tree node
            accepted_path, acc, correction = tree_accept(tree, target_logits, sp.temperature)
            # GATHER: keep prefix + accepted path (root + accepted nodes); drop the
            # rejected branches' KV. accepted_path = [0(root), …acc nodes], tree-ordered;
            # their cache slots are past_len + path -> contiguous positions after gather.
            keep = torch.cat([
                torch.arange(past_len, device=self.device),
                past_len + torch.tensor(accepted_path, device=self.device),
            ])
            cache.gather(keep)                    # cache length -> past_len + (acc + 1)
            if new_hidden is not None:
                # append [root | accepted nodes] hidden (the correction has none yet —
                # it is the next anchor, fed via noise). Restores the invariant.
                sel = torch.tensor(accepted_path, device=self.device)
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
        text = self.tokenizer.decode(new_ids, skip_special_tokens=True)
        out = {"token_ids": new_ids, "text": text, "tpf": (len(new_ids) / rounds if rounds else 0.0)}
        if return_stats:
            out["accept_lengths"] = accept_lengths   # per-round (acc+1)
            out["tree_sizes"] = tree_sizes           # per-round node count
            out["rounds"] = rounds
        return out

    @torch.inference_mode()
    def generate_batch(self, prompts: list, sampling_params: SamplingParams = None) -> list:
        """Continuous-batched greedy/temperature AR over the shared multi-seq paged
        cache (nano_vllm N2a). Returns a list of `{token_ids, text}` aligned to
        `prompts`, each token-identical to `generate(prompt)` run alone — the N2a
        lossless gate.

        Each `prompts[i]` is a str (tokenized raw) or a tokenized `(1, T)` tensor.
        The `Scheduler` admits every prompt into one fixed `PagedKVCache` pool
        (per-`seq_id` block tables); a `SequenceRequest` carries each sequence's
        state. The loop is the pad-to-max batched forward of the N2a design: prefill
        each admitted prompt (its prefix KV lands in the pool under its `seq_id`),
        then each decode step reconstructs every live sequence's dense KV from the
        pool, pads to the batch max length, runs ONE forward under a 4D padding mask
        (so attention only sees each sequence's real positions — padded KV is masked,
        not attended), appends the new token's KV back into the pool per `seq_id`,
        and samples each sequence's next token. A sequence drops out of the batch on
        EOS or once it hits its token budget; the survivors keep decoding.

        Lossless because each sequence sees exactly the KV / RoPE positions / causal
        visibility it would see decoding alone — the only difference is the pooled
        storage and the per-step pad+mask, both of which are masked-out no-ops for
        attention. fp32 bitwise-equal on CPU; bf16 carries the same SDPA
        reduction-order caveat as N0/N1."""
        sp = sampling_params or SamplingParams()
        # Tokenize / normalize prompts to input_id lists.
        prompt_ids = []
        for p in prompts:
            if isinstance(p, str):
                ids = self.tokenizer(p, return_tensors="pt").input_ids[0].tolist()
            else:
                ids = p.to(self.device).view(-1).tolist()
            prompt_ids.append([int(t) for t in ids])

        cache = PagedKVCache(
            block_size=self.block_size, max_batch_size=max(2, len(prompt_ids)),
            device=torch.device(self.device), dtype=self.dtype,
        )
        scheduler = Scheduler(cache, max_batch_size=max(2, len(prompt_ids)))
        for seq_id, ids in enumerate(prompt_ids):
            scheduler.admit_request(SequenceRequest(
                seq_id=seq_id, input_ids=list(ids),
                max_new_tokens=sp.max_new_tokens, temperature=sp.temperature,
            ))
        scheduler.step()                              # FCFS admit all into the pool

        num_layers = self.model.config.num_hidden_layers
        results = {sid: {"token_ids": []} for sid in range(len(prompt_ids))}

        # --- prefill: each admitted prompt forwards once (batch=1) into the pool
        # under its seq_id; sample its first token. (Prefills are independent, so a
        # per-seq forward is simplest and identical to the single-stream prefill.)
        for sid, ids in enumerate(prompt_ids):
            input_ids = torch.tensor([ids], device=self.device)
            self._prefill_into_pool(cache, sid, input_ids, num_layers)
            logits = self._last_prefill_logits
            next_tok = int(sample(logits[:, -1:, :], sp.temperature).item())
            results[sid]["token_ids"].append(next_tok)
            scheduler.mark_decode_step(sid, next_tok, logits[:, -1:, :])

        # Active = sequences still decoding (not finished). Drop on EOS or budget.
        active = [sid for sid in range(len(prompt_ids))
                  if not self._is_finished(results[sid]["token_ids"], sp)]

        # --- decode: batched single-token steps over the live sequences ---------
        while active:
            toks = [results[sid]["token_ids"][-1] for sid in active]
            logits = self._batched_decode_forward(cache, active, toks, num_layers, sp.temperature)
            for i, sid in enumerate(active):
                next_tok = int(sample(logits[i:i + 1, -1:, :], sp.temperature).item())
                results[sid]["token_ids"].append(next_tok)
                scheduler.mark_decode_step(sid, next_tok, logits[i:i + 1, -1:, :])
            active = [sid for sid in active
                      if not self._is_finished(results[sid]["token_ids"], sp)]

        out = []
        for sid in range(len(prompt_ids)):
            ids = results[sid]["token_ids"][: sp.max_new_tokens]
            out.append({"token_ids": ids,
                        "text": self.tokenizer.decode(ids, skip_special_tokens=True)})
        return out

    def _is_finished(self, token_ids: list, sp: SamplingParams) -> bool:
        """A sequence is done once it hit EOS or its `max_new_tokens` budget."""
        return (len(token_ids) >= sp.max_new_tokens
                or (token_ids and token_ids[-1] in self.eos_token_ids))

    def _prefill_into_pool(self, cache: PagedKVCache, seq_id: int,
                           input_ids: torch.Tensor, num_layers: int) -> None:
        """Forward a prompt once (batch=1) and transfer its prefix KV into the
        shared pool under `seq_id`. HF's `update` can't route a per-row `seq_id`
        through a batched forward, so we prefill into a throwaway `DynamicCache`
        and copy each layer's KV into the pool via `append(..., seq_id=seq_id)`.
        Stashes the prompt's logits for the caller to sample the first token."""
        pos = torch.arange(input_ids.shape[1], device=self.device).unsqueeze(0)
        scratch = DynamicCache()
        logits, scratch, _ = self.runner.forward(input_ids, scratch, pos)
        for layer_idx in range(num_layers):
            keys = scratch.layers[layer_idx].keys        # (1, H, T, D)
            values = scratch.layers[layer_idx].values
            cache.append(keys, values, layer_idx, seq_id=seq_id)
        self._last_prefill_logits = logits

    def _batched_decode_forward(self, cache: PagedKVCache, seq_ids: list,
                                tokens: list, num_layers: int, temperature: float):
        """One pad-to-max batched decode step over `seq_ids` (the N2a forward).

        Reconstructs every sequence's dense KV from the pool, pads to the batch max
        cached length, builds a 4D additive mask that exposes each sequence's real
        prefix + its own new token (padded KV masked out), forwards once, then
        appends the new token's KV back into the pool per `seq_id`. Returns the
        batched logits `(B, 1, V)`.

        On the N3 kernel path (`attn_backend == "triton_paged_tree"`) there is no
        pad+mask+copy-back: `update` appends each row's new-token KV to its own seq
        and the kernel reads `[0, past_i + 1)` straight from the pool (each seq is a
        pure decode -> qq_bias=None). Returns the same `(B, 1, V)` logits."""
        B = len(seq_ids)
        if getattr(self, "attn_backend", "sdpa") == "triton_paged_tree":
            return self._batched_decode_forward_kernel(cache, seq_ids, tokens)
        seq_lens = [cache.get_seq_length(0, seq_id=s) for s in seq_ids]
        max_len = max(seq_lens)
        neg = torch.finfo(self.dtype).min

        # Build a padded DynamicCache from each seq's logical KV (right-padded with
        # zeros to max_len; the padding columns are masked out below).
        batched = DynamicCache()
        for layer_idx in range(num_layers):
            k_pad = torch.zeros((B, cache._num_heads, max_len, cache._head_dim),
                                dtype=self.dtype, device=self.device)
            v_pad = torch.zeros_like(k_pad)
            for i, s in enumerate(seq_ids):
                gk, gv = cache._logical_kv(layer_idx, seq_id=s)   # (1, H, S_i, D)
                S_i = gk.shape[2]
                k_pad[i, :, :S_i, :] = gk[0]
                v_pad[i, :, :S_i, :] = gv[0]
            batched.update(k_pad, v_pad, layer_idx)

        # New token per seq + its RoPE position (= the seq's cached length).
        input_ids = torch.tensor([[t] for t in tokens], device=self.device)
        position_ids = torch.tensor([[s] for s in seq_lens], device=self.device)
        # 4D additive mask (B, 1, Q=1, KV=max_len+1): real prefix cols [0, S_i)
        # allowed, padded cols [S_i, max_len) masked, the new-token col (max_len)
        # is self-visible. cache_position=max_len places the new KV uniformly.
        mask = torch.full((B, 1, 1, max_len + 1), neg, dtype=self.dtype, device=self.device)
        for i, S_i in enumerate(seq_lens):
            mask[i, 0, 0, :S_i] = 0.0
            mask[i, 0, 0, max_len] = 0.0
        cache_position = torch.tensor([max_len], device=self.device)
        logits, batched, _ = self.runner.forward(
            input_ids, batched, position_ids,
            attention_mask=mask, cache_position=cache_position,
        )
        # Append each seq's new-token KV (col max_len) back into the pool.
        for layer_idx in range(num_layers):
            keys = batched.layers[layer_idx].keys        # (B, H, max_len+1, D)
            values = batched.layers[layer_idx].values
            for i, s in enumerate(seq_ids):
                k_new = keys[i:i + 1, :, max_len:max_len + 1, :]
                v_new = values[i:i + 1, :, max_len:max_len + 1, :]
                cache.append(k_new, v_new, layer_idx, seq_id=s)
        return logits

    def _batched_decode_forward_kernel(self, cache: PagedKVCache, seq_ids: list,
                                       tokens: list):
        """N3 kernel decode step: forward the REAL pooled cache, no pad/mask/copy-back.

        Each seq is a pure decode (one query row): `update` routes each batch row's
        new-token KV to its own `seq_id` (per `_handoff_seq_ids`) and the kernel
        reads `[0, past_i + 1)` from the pool with qq_bias=None (causal decode).
        position_ids = per-seq past length; cache_position is irrelevant on this
        path (the cache appends by seq order, not cache_position). Returns `(B, 1, V)`."""
        seq_lens = [cache.get_seq_length(0, seq_id=s) for s in seq_ids]
        cache._paged_handoff = True
        cache._handoff_seq_ids = list(seq_ids)
        cache._ptd_attn_meta = {"seq_ids": list(seq_ids), "qq_bias": None}
        input_ids = torch.tensor([[t] for t in tokens], device=self.device)
        position_ids = torch.tensor([[s] for s in seq_lens], device=self.device)
        logits, _, _ = self.runner.forward(
            input_ids, cache, position_ids, attention_mask=None,
        )
        return logits

    @torch.inference_mode()
    def generate_tree_batch(self, prompts: list, tree_drafter, block_size: int = 4,
                            tree_width: int = 2, budget: int = 15, algo: str = "crossproduct",
                            algo_kwargs: dict = None, sampling_params: SamplingParams = None) -> list:
        """Batched per-sequence TREE-spec decode over the shared multi-seq paged
        cache (nano_vllm N2b). Returns a list of `{token_ids, text, tpf}` aligned to
        `prompts`, each token-identical to single-stream `generate_tree(prompt)` run
        alone — the N2b lossless gate.

        Always uses the SDPA path regardless of `self.attn_backend`: the N3 kernel
        path is a follow-on for N2b. It pads queries to `S = max_N`, so `total_q =
        B*max_N` no longer matches Unit-2's ragged `qq_bias` (sum N_i) and padding-
        node KV would pollute the per-seq pool; that needs a padded `qq_bias` or
        query compaction (out of scope here).

        This is the tree-spec analogue of `generate_batch` (N2a) and the batched
        analogue of `generate_tree` (N1). Each round, every live sequence builds its
        OWN draft tree (`get_algorithm(algo).build` on its drafter logits — the N1
        per-seq path, with possibly different node counts N_i), the trees are padded
        to `max_N` and verified in ONE batched forward under the design's padded 4D
        mask, and each sequence's accepted root-to-leaf path is taken by `tree_accept`
        on its own logit slice. Each sequence's KV is gathered independently
        (ref-count-safe `PagedKVCache` per-seq append), so dropping a finished
        sequence never perturbs the survivors.

        Per-seq isolation comes from the additive mask `(B, 1, max_N, max_len + max_N)`
        (`max_len = max(past_len[i])`): for seq i, query j < N_i sees its real prefix
        columns `[0, past_len[i])`, its own tree nodes `[max_len, max_len + N_i)`
        filtered by the ancestor matrix, and nothing else (padding prefix columns,
        other seqs' tree columns, and padding tree columns are all `-inf`); padding
        queries j >= N_i are fully masked. RoPE positions are per-seq depth-relative
        (`past_len[i] + depth[i, j]`), `cache_position` is uniform — exactly the N1
        single-stream geometry replicated per row, so each sequence's attention graph
        is isomorphic to verifying its tree alone.

        Lossless by construction (commits the verify forward's own greedy along each
        accepted path); fp32 bitwise-equal to single-stream `generate_tree` on CPU,
        bf16 carries the same SDPA reduction-order caveat as N0/N1/N2a.

        `algo` / `algo_kwargs` mirror `generate_tree`; all bundled algorithms recover
        crossproduct at their identity knobs, so the choice is lossless regardless.
        Hidden-state-conditioned (DraftHead) drafting is N1-only for now: this batched
        route runs the no-hidden path (`target_hidden=None`), which is what the stub
        drafters that gate it exercise."""
        # tree contract (engine -> tree, one-way): import only the public ptd.tree API
        from ptd.tree import get_algorithm, build_ancestor_matrix, tree_accept

        sp = sampling_params or SamplingParams()
        # Tokenize / normalize prompts to input_id lists.
        prompt_ids = []
        for p in prompts:
            if isinstance(p, str):
                ids = self.tokenizer(p, return_tensors="pt").input_ids[0].tolist()
            else:
                ids = p.to(self.device).view(-1).tolist()
            prompt_ids.append([int(t) for t in ids])
        n_seq = len(prompt_ids)

        D = max(1, block_size - 1)
        algo_obj = get_algorithm(algo, **(algo_kwargs or {}))
        num_layers = self.model.config.num_hidden_layers
        cache = PagedKVCache(
            block_size=self.block_size, max_batch_size=max(2, n_seq),
            device=torch.device(self.device), dtype=self.dtype,
        )
        scheduler = Scheduler(cache, max_batch_size=max(2, n_seq))
        for seq_id, ids in enumerate(prompt_ids):
            scheduler.admit_request(SequenceRequest(
                seq_id=seq_id, input_ids=list(ids),
                max_new_tokens=sp.max_new_tokens, temperature=sp.temperature,
            ))
        scheduler.step()                              # FCFS admit all into the pool

        # Per-seq decode state. `committed` is the full token stream incl. the anchor
        # (= each round's tree root); the pool trails it by that anchor (the root
        # enters via the verify forward, exactly as in N1). `new_ids` accumulates the
        # generated tokens; `rounds` counts verify forwards for tpf.
        committed = {}
        new_ids = {sid: [] for sid in range(n_seq)}
        rounds = {sid: 0 for sid in range(n_seq)}

        # --- prefill: each prompt forwards once into the pool under its seq_id;
        # sample its first token (the anchor / first tree root). Mirrors the N2a
        # prefill, plus N1's anchor bookkeeping. ---------------------------------
        for sid, ids in enumerate(prompt_ids):
            input_ids = torch.tensor([ids], device=self.device)
            self._prefill_into_pool(cache, sid, input_ids, num_layers)
            logits = self._last_prefill_logits
            first_tok = int(sample(logits[:, -1:, :], sp.temperature).item())
            new_ids[sid].append(first_tok)
            # committed = prompt + anchor; the anchor is NOT yet cached (it is the
            # tree root, fed via the next verify forward).
            committed[sid] = torch.cat(
                [input_ids, torch.tensor([[first_tok]], device=self.device)], dim=1)

        # Active = sequences still decoding (not finished on the first token).
        active = [sid for sid in range(n_seq)
                  if not self._is_finished(new_ids[sid], sp)]

        # --- decode: each round builds every live seq's tree, verifies in one
        # batched forward, then per-seq tree_accept + ref-count-safe pool append. --
        while active:
            trees, ancestors, past_lens = [], [], []
            for sid in active:
                draft_logits = tree_drafter.propose_logits(
                    committed[sid], D, target_hidden=None).to(self.device)   # (1, D, V)
                tree = algo_obj.build(
                    int(committed[sid][0, -1]), draft_logits, block_size, tree_width,
                    budget, self.device)
                trees.append(tree)
                ancestors.append(build_ancestor_matrix(tree).bool())
                past_lens.append(cache.get_seq_length(seq_id=sid))   # == committed-1
            max_N = max(t.num_nodes for t in trees)

            # ONE batched verify forward over the padded per-seq trees.
            tree_kv = self._batched_tree_verify_forward(
                cache, active, trees, ancestors, past_lens, max_N, num_layers)
            logits = tree_kv["logits"]                # (B, max_N, V)

            still = []
            for i, sid in enumerate(active):
                tree = trees[i]
                N = tree.num_nodes
                logits_i = logits[i:i + 1, :N, :]     # (1, N, V) — only real nodes
                accepted_path, acc, correction = tree_accept(tree, logits_i, sp.temperature)
                rounds[sid] += 1
                # Append [root | accepted nodes] tree-node KV to the pool (the N1
                # gather's keep set, applied per-seq). The pool grows from
                # past_len[i] to past_len[i] + (acc + 1); the correction has no KV
                # yet (it becomes the next round's anchor / root).
                self._append_tree_path_kv(cache, sid, tree_kv, i, accepted_path, num_layers)
                accepted = tree.token_ids[torch.tensor(accepted_path[1:], device=self.device)] \
                    if acc > 0 else torch.empty(0, dtype=tree.token_ids.dtype, device=self.device)
                block = torch.cat([accepted, torch.tensor([correction], device=self.device)])
                committed[sid] = torch.cat([committed[sid], block.view(1, -1)], dim=1)
                for t in block.tolist():
                    new_ids[sid].append(int(t))
                    if int(t) in self.eos_token_ids:
                        break
                if not self._is_finished(new_ids[sid], sp):
                    still.append(sid)
            active = still

        out = []
        for sid in range(n_seq):
            ids = new_ids[sid][: sp.max_new_tokens]
            r = rounds[sid]
            out.append({"token_ids": ids,
                        "text": self.tokenizer.decode(ids, skip_special_tokens=True),
                        "tpf": (len(ids) / r if r else 0.0)})
        return out

    def _batched_tree_verify_forward(self, cache: PagedKVCache, seq_ids: list,
                                     trees: list, ancestors: list, past_lens: list,
                                     max_N: int, num_layers: int) -> dict:
        """One padded batched tree-verify forward over `seq_ids` (the N2b forward).

        Reconstructs every sequence's dense prefix KV from the pool, pads to the
        batch max prefix length `max_len`, appends each seq's `N_i` tree-node columns
        (right-padded to `max_N`), and forwards once under the design's 4D additive
        mask so each sequence's tree nodes see ONLY that seq's prefix + their tree
        ancestors. Returns `{"logits": (B, max_N, V), "cache": DynamicCache,
        "max_len": max_len}`; the caller slices each seq's accepted-path KV out of
        the verify cache and appends it to the pool via `_append_tree_path_kv`.

        Geometry (per seq i, the N1 single-stream verify replicated per row):
          - input_ids[i, j]    = tree_i.token_ids[j]              (j < N_i; pad 0)
          - position_ids[i, j] = past_len[i] + depth_i[j]         (RoPE, depth-based)
          - cache_position[j]  = max_len + j                      (uniform append slot)
          - mask[i, 0, j, k]:  0 on real prefix cols [0, past_len[i]); 0 on tree cols
            [max_len, max_len + N_i) where ancestor_i[j, k - max_len]; -inf elsewhere
            (padding prefix cols, other seqs' / padding tree cols). Padding queries
            j >= N_i are fully -inf (never attend)."""
        B = len(seq_ids)
        max_len = max(past_lens)
        neg = torch.finfo(self.dtype).min
        kv_len = max_len + max_N

        # Build a padded DynamicCache from each seq's logical prefix KV (right-padded
        # with zeros to max_len; padding columns are masked out below). The verify
        # forward appends the max_N tree columns onto this, giving (B, H, kv_len, D).
        batched = DynamicCache()
        for layer_idx in range(num_layers):
            k_pad = torch.zeros((B, cache._num_heads, max_len, cache._head_dim),
                                dtype=self.dtype, device=self.device)
            v_pad = torch.zeros_like(k_pad)
            for i, s in enumerate(seq_ids):
                gk, gv = cache._logical_kv(layer_idx, seq_id=s)   # (1, H, past_len[i], D)
                P_i = gk.shape[2]
                k_pad[i, :, :P_i, :] = gk[0]
                v_pad[i, :, :P_i, :] = gv[0]
            batched.update(k_pad, v_pad, layer_idx)

        # Per-seq tree nodes (token_ids / RoPE positions), right-padded to max_N.
        input_ids = torch.zeros((B, max_N), dtype=torch.long, device=self.device)
        position_ids = torch.zeros((B, max_N), dtype=torch.long, device=self.device)
        for i, s in enumerate(seq_ids):
            tree, N, past = trees[i], trees[i].num_nodes, past_lens[i]
            input_ids[i, :N] = tree.token_ids
            position_ids[i, :N] = past + tree.depth.to(self.device)

        # 4D additive mask (B, 1, max_N, kv_len): per-seq prefix + ancestor isolation.
        mask = torch.full((B, 1, max_N, kv_len), neg, dtype=self.dtype, device=self.device)
        for i, s in enumerate(seq_ids):
            N, past = trees[i].num_nodes, past_lens[i]
            mask[i, 0, :N, :past] = 0.0                         # real prefix: visible
            tree_block = torch.where(                           # tree cols: ancestor relation
                ancestors[i],
                torch.zeros((), dtype=self.dtype, device=self.device),
                torch.full((), neg, dtype=self.dtype, device=self.device))
            mask[i, 0, :N, max_len:max_len + N] = tree_block    # (N, N) ancestor-masked
            # rows j >= N (padding queries) stay fully -inf; cols outside the two
            # blocks (padding prefix, other seqs' / padding tree cols) stay -inf.
        cache_position = torch.arange(max_len, max_len + max_N, device=self.device)
        logits, batched, _ = self.runner.forward(
            input_ids, batched, position_ids,
            attention_mask=mask, cache_position=cache_position,
        )
        return {"logits": logits, "cache": batched, "max_len": max_len}

    def _append_tree_path_kv(self, cache: PagedKVCache, seq_id: int, tree_kv: dict,
                             batch_idx: int, accepted_path: list, num_layers: int) -> None:
        """Append seq `seq_id`'s accepted-path tree-node KV from the verify cache
        into the pool (the per-seq, ref-count-safe analogue of N1's `gather`).

        `accepted_path` is `[0(root), …acc accepted nodes]` in tree order; their KV
        sits at columns `max_len + node_idx` of the verify cache. After this append
        the pool length for `seq_id` is `past_len + (acc + 1)`, restoring the N1
        invariant (pool == committed minus the next anchor)."""
        max_len = tree_kv["max_len"]
        verify = tree_kv["cache"]
        cols = max_len + torch.tensor(accepted_path, device=self.device)   # (acc+1,)
        for layer_idx in range(num_layers):
            keys = verify.layers[layer_idx].keys          # (B, H, max_len + max_N, D)
            values = verify.layers[layer_idx].values
            k_path = keys[batch_idx:batch_idx + 1, :, cols, :]    # (1, H, acc+1, D)
            v_path = values[batch_idx:batch_idx + 1, :, cols, :]
            cache.append(k_path, v_path, layer_idx, seq_id=seq_id)
