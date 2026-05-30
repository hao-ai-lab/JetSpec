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
        """Chain (linear) speculative decode. Each round: the drafter proposes
        `block_size-1` tokens, the target verifies them in one forward, and we
        accept the longest prefix the target agrees with (greedy) plus one
        correction token. Lossless — the output equals plain greedy regardless of
        draft quality. Recompute-based (no KV reuse yet — this validates the accept
        logic, not speed). Returns {token_ids, text, tpf}, tpf = new / forwards.

        `target_layer_ids` (the head's tapped layers): when set with block_size>1,
        each verify forward also extracts `target_hidden`, threaded into the next
        `drafter.propose(...)` (the real DraftHead conditions on it; stubs ignore
        it). The anchor for the next draft is the last committed token's hidden."""
        sp = sampling_params or SamplingParams()
        if isinstance(prompt, str):
            committed = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        else:
            committed = prompt.to(self.device)
        k = max(1, block_size - 1)
        need_hidden = target_layer_ids is not None and block_size > 1
        new_ids, rounds = [], 0
        target_hidden = None

        # --- prefill: seed the target cache and the first target_hidden anchor ---
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
            drafts = drafter.propose(committed, k, target_hidden=target_hidden).to(self.device).view(-1)[:k]   # (k,)
            seq = torch.cat([committed, drafts.view(1, -1)], dim=1)               # (1, p+k)
            p = committed.shape[1]
            pos = torch.arange(seq.shape[1], device=self.device).unsqueeze(0)
            logits, _, full_hidden = self.runner.forward(
                seq, DynamicCache(), pos,
                output_hidden_states=need_hidden, target_layer_ids=target_layer_ids,
            )
            rounds += 1
            tgt = logits[0, p - 1:, :].argmax(-1)    # (k+1,) target greedy at the boundary positions
            acc = 0
            for i in range(k):
                if int(drafts[i]) == int(tgt[i]):
                    acc += 1
                else:
                    break
            block_new = torch.cat([drafts[:acc], tgt[acc].view(1)])   # accepted drafts + 1 correction
            committed = torch.cat([committed, block_new.view(1, -1)], dim=1)
            # Next-round context for the head = the new committed minus its last
            # token (that token is the anchor, fed via noise_embedding[0]). The new
            # committed is committed[:p] + accepted drafts[:acc]; those occupy seq
            # positions [0 : p+acc] (accepted drafts == target greedy, so their
            # verify-forward hidden is exactly right). The correction (next anchor)
            # needs no hidden. Full context — not the anchor alone — is what lifts
            # acceptance; a length-1 slice starves the head.
            if full_hidden is not None:
                target_hidden = full_hidden[:, :p + acc, :]
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
                      budget: int = 15, algo: str = "crossproduct",
                      target_layer_ids=None, sampling_params: SamplingParams = None) -> dict:
        """Tree speculative decode. Each round: the tree drafter emits per-depth
        logits, the tree algorithm builds a DraftTree, the target verifies all
        nodes in one forward under a 4D ancestor mask, and tree_accept takes the
        longest greedy-agreeing root-to-leaf path + a correction. Lossless —
        output equals plain greedy. Recompute-based (validates the tree verify,
        not speed). Returns {token_ids, text, tpf}.

        `target_layer_ids` (the head's tapped layers): when set with block_size>1,
        each verify forward extracts `target_hidden`, threaded into the next
        `tree_drafter.propose_logits(...)`. The next anchor is the deepest accepted
        node's hidden (the correction token has no hidden yet)."""
        from ptd.tree import get_algorithm
        from ptd.tree._core.ancestor import build_ancestor_matrix
        from ptd.tree._core.accept import tree_accept

        sp = sampling_params or SamplingParams()
        if isinstance(prompt, str):
            committed = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        else:
            committed = prompt.to(self.device)
        D = max(1, block_size - 1)
        algo_obj = get_algorithm(algo)
        dtype = self.model.dtype
        neg = torch.finfo(dtype).min
        need_hidden = target_layer_ids is not None and block_size > 1
        new_ids, rounds = [], 0
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
            tree = algo_obj.build(int(committed[0, -1]), draft_logits, block_size, tree_width, budget, self.device)
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
            for t in block.tolist():
                new_ids.append(int(t))
                if int(t) in self.eos_token_ids:
                    break
            if new_ids and new_ids[-1] in self.eos_token_ids:
                break
        new_ids = new_ids[: sp.max_new_tokens]
        text = self.tokenizer.decode(new_ids, skip_special_tokens=True)
        return {"token_ids": new_ids, "text": text, "tpf": (len(new_ids) / rounds if rounds else 0.0)}
