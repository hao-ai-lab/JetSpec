"""Offline single-stream LLM — a small nano-vLLM-style API.

Plain autoregressive greedy/temperature decode over an HF `DynamicCache`
(prefill the prompt once, then single-token decode steps reusing the cache).
This is the 1x baseline; the draft head + tree verify build on the same seam.
Single sequence, batch=1 (the offline-inference regime).
"""
from dataclasses import dataclass

import torch
from transformers import DynamicCache

from ptd.models.qwen3 import load_target
from ptd.engine.model_runner import ModelRunner
from ptd.engine.sampler import sample


@dataclass
class SamplingParams:
    temperature: float = 0.0
    max_new_tokens: int = 256


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
    layers = getattr(cache, "layers", None)
    if layers is not None:                       # transformers >= 4.54 (DynamicLayer)
        for layer in layers:
            keys = getattr(layer, "keys", None)
            if keys is None:
                continue                         # uninitialized / sliding layer — nothing cached
            idx = keep_index.to(keys.device)
            layer.keys = keys.index_select(-2, idx)
            layer.values = layer.values.index_select(-2, idx)
    else:                                        # legacy key_cache / value_cache lists
        for i in range(len(cache.key_cache)):
            idx = keep_index.to(cache.key_cache[i].device)
            cache.key_cache[i] = cache.key_cache[i].index_select(-2, idx)
            cache.value_cache[i] = cache.value_cache[i].index_select(-2, idx)


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
        cache = DynamicCache()

        # --- prefill: process the whole prompt once, sample the first token ---
        pos = torch.arange(prompt_len, device=self.device).unsqueeze(0)
        logits, cache, _ = self.runner.forward(input_ids, cache, pos)
        next_tok = sample(logits[:, -1:, :], sp.temperature)  # (1, 1)
        out_ids = [int(next_tok.item())]
        cur = prompt_len

        # --- decode: single-token steps reusing the KV cache (no reprocess) ---
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
        k = max(1, block_size - 1)
        need_hidden = target_layer_ids is not None and block_size > 1
        new_ids, rounds = [], 0
        target_hidden = None

        # --- prefill: populate the persistent cache with the prompt's KV ---
        cache = DynamicCache()
        pos = torch.arange(committed.shape[1], device=self.device).unsqueeze(0)
        logits, cache, full_hidden = self.runner.forward(
            committed, cache, pos,
            output_hidden_states=need_hidden, target_layer_ids=target_layer_ids,
        )
        if full_hidden is not None:
            target_hidden = full_hidden          # prompt context; next anchor (first_tok) fed via noise
        first_tok = sample(logits[:, -1:, :], sp.temperature)
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
            cache.crop(past_len + 1 + acc)        # keep prefix + anchor + accepted drafts; drop rejected KV
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
                      budget: int = 15, algo: str = "crossproduct", algo_kwargs: dict = None,
                      target_layer_ids=None, sampling_params: SamplingParams = None,
                      return_stats: bool = False, prompt_info: dict = None,
                      profile_table: dict = None, kv_cache_verify: bool = False) -> dict:
        """Tree speculative decode. Each round: the tree drafter emits per-depth
        logits, the tree algorithm builds a DraftTree, the target verifies all
        nodes in one forward under a 4D ancestor mask, and tree_accept takes the
        longest greedy-agreeing root-to-leaf path + a correction. Lossless —
        output equals plain greedy. Recompute-based (validates the tree verify,
        not speed). Returns {token_ids, text, tpf}.

        `algo` selects a registered tree algorithm (see `ptd.tree.list_algorithms`);
        `algo_kwargs` passes its constructor knobs (e.g. {"beta": 2.0} for
        top2gap_fanout). `prompt_info` (optional dict: task label / reasoning mode
        / decoded text) is forwarded to the algorithm's build() for the
        prompt-adaptive (semantic_aware) algorithms; None → they use their
        logit-fingerprint fallback. `profile_table` (optional dict: offline
        per-(depth,rank) acceptance, from bench/collect_profile.py) is forwarded to
        the profile-guided algorithms (depth_rank_histogram); None → they recover
        crossproduct. All bundled algorithms recover crossproduct at their identity
        knobs, so the choice is lossless regardless.

        `target_layer_ids` (the head's tapped layers): when set with block_size>1,
        each verify forward extracts `target_hidden`, threaded into the next
        `tree_drafter.propose_logits(...)`. The next anchor is the deepest accepted
        node's hidden (the correction token has no hidden yet).

        `kv_cache_verify` (default False = recompute): when True, route to the
        persistent-cache verify (`_generate_tree_kv_cached`) — each round forwards
        only the tree nodes against a cached prefix (no prefix recompute), the
        wall-clock path. Lossless by construction either way; the cached path is
        not bitwise-equal to recompute in bf16 (cached prefix KV vs a fresh
        recompute differ in SDPA reduction order; fp32 exact)."""
        # tree contract (engine -> tree, one-way): import only the public ptd.tree API
        from ptd.tree import get_algorithm, build_ancestor_matrix, tree_accept

        sp = sampling_params or SamplingParams()
        if isinstance(prompt, str):
            committed = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        else:
            committed = prompt.to(self.device)
        D = max(1, block_size - 1)
        algo_obj = get_algorithm(algo, **(algo_kwargs or {}))
        if kv_cache_verify:
            return self._generate_tree_kv_cached(
                committed, tree_drafter, block_size, tree_width, budget, algo_obj,
                target_layer_ids, sp, return_stats, prompt_info, profile_table,
            )
        dtype = self.model.dtype
        neg = torch.finfo(dtype).min
        need_hidden = target_layer_ids is not None and block_size > 1
        new_ids, rounds = [], 0
        accept_lengths, tree_sizes = [], []   # per-round (acc+1) and node count (return_stats)
        target_hidden = None

        # --- prefill: seed the first target_hidden anchor + the first token ---
        pos = torch.arange(committed.shape[1], device=self.device).unsqueeze(0)
        logits, _, full_hidden = self.runner.forward(
            committed, DynamicCache(), pos,
            output_hidden_states=need_hidden, target_layer_ids=target_layer_ids,
        )
        if full_hidden is not None:
            target_hidden = full_hidden          # full prompt context (next anchor = first_tok, fed via noise)
        first_tok = sample(logits[:, -1:, :], sp.temperature)
        new_ids.append(int(first_tok.item()))
        committed = torch.cat([committed, first_tok.view(1, 1)], dim=1)
        if int(first_tok.item()) in self.eos_token_ids:
            new_ids = new_ids[: sp.max_new_tokens]
            return {"token_ids": new_ids, "text": self.tokenizer.decode(new_ids, skip_special_tokens=True), "tpf": 0.0}

        while len(new_ids) < sp.max_new_tokens:
            draft_logits = tree_drafter.propose_logits(committed, D, target_hidden=target_hidden).to(self.device)  # (1, D, V)
            tree = algo_obj.build(int(committed[0, -1]), draft_logits, block_size, tree_width, budget, self.device,
                                  prompt_info=prompt_info, profile_table=profile_table)
            N = tree.num_nodes
            prefix = committed[:, :-1]              # tokens before the anchor (= tree root)
            P = prefix.shape[1]
            seq = torch.cat([prefix, tree.token_ids.view(1, -1)], dim=1)                  # (1, P+N)
            depths = tree.depth.tolist()
            pos = torch.tensor([list(range(P)) + [P + d for d in depths]], device=self.device)
            # 4D additive mask: prefix causal · every node sees all prefix · nodes see ancestors (incl self)
            T = P + N
            allowed = torch.zeros(T, T, dtype=torch.bool, device=self.device)
            if P > 0:
                allowed[:P, :P] = torch.tril(torch.ones(P, P, dtype=torch.bool, device=self.device))
                allowed[P:, :P] = True
            allowed[P:, P:] = build_ancestor_matrix(tree).bool()
            mask = torch.where(allowed, torch.zeros((), dtype=dtype, device=self.device),
                               torch.full((), neg, dtype=dtype, device=self.device)).view(1, 1, T, T)
            logits, _, full_hidden = self.runner.forward(
                seq, DynamicCache(), pos, attention_mask=mask,
                output_hidden_states=need_hidden, target_layer_ids=target_layer_ids,
            )
            target_logits = logits[:, P:, :]                                              # (1, N, V)
            accepted_path, acc, correction = tree_accept(tree, target_logits, sp.temperature)
            # Next-round context for the head = the new committed minus its last
            # token (the anchor = correction, fed via noise_embedding[0]). The new
            # committed[:-1] is prefix + root + accepted nodes; gather their
            # (non-contiguous, tree-ordered) positions in `seq`: prefix [0:P], root
            # = node 0 at P, accepted nodes at P+accepted_path[1:]. Full context —
            # not the deepest node alone — is what lifts acceptance.
            if full_hidden is not None:
                idx = list(range(P + 1)) + [P + j for j in accepted_path[1:]]
                target_hidden = full_hidden[:, idx, :]
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
    def _generate_tree_kv_cached(self, committed, tree_drafter, block_size, tree_width,
                                 budget, algo_obj, target_layer_ids, sp,
                                 return_stats, prompt_info, profile_table) -> dict:
        """Tree spec decode over a PERSISTENT target KV cache (the wall-clock path).

        Mirrors `generate_chain`'s persistent-cache verify, extended to trees: each
        round forwards only the tree nodes against the cached prefix (no prefix
        recompute), then GATHERS the accepted root-to-leaf path's KV back into a
        linear cache (dropping the rejected branches). The accepted nodes are a
        non-contiguous subset of the tree-ordered slots, but their positions are
        `[past, past+1, …, past+acc]` (root + one node per accepted depth), so the
        gathered cache is an ordinary causal prefix again — `committed[:-1]`.

        Lossless by construction (commits the verify forward's own greedy along the
        accepted path). See `generate_tree` for the bf16-vs-fp32 note. Dispatched
        from `generate_tree(kv_cache_verify=True)`; args are already parsed there."""
        # tree contract (engine -> tree, one-way): import only the public ptd.tree API
        from ptd.tree import build_ancestor_matrix, tree_accept

        dtype = self.model.dtype
        neg = torch.finfo(dtype).min
        D = max(1, block_size - 1)
        need_hidden = target_layer_ids is not None and block_size > 1
        new_ids, rounds = [], 0
        accept_lengths, tree_sizes = [], []   # per-round (acc+1) and node count (return_stats)
        target_hidden = None

        # --- prefill: populate the persistent cache with the prompt's KV ---
        cache = DynamicCache()
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
            _select_kv_cache(cache, keep)         # cache length -> past_len + (acc + 1)
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
