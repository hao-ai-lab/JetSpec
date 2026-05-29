"""Offline single-stream LLM — a small nano-vLLM-style API.

M0: plain autoregressive greedy/temperature decode over an HF `DynamicCache`
(prefill the prompt once, then single-token decode steps reusing the cache).
This is the 1x baseline and the seam M1 extends with the draft head + tree
verify. Single sequence, batch=1 (the offline-inference regime).
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
        logits, cache = self.runner.forward(input_ids, cache, pos)
        next_tok = sample(logits[:, -1:, :], sp.temperature)  # (1, 1)
        out_ids = [int(next_tok.item())]
        cur = prompt_len

        # --- decode: single-token steps reusing the KV cache (no reprocess) ---
        for _ in range(sp.max_new_tokens - 1):
            if out_ids[-1] in self.eos_token_ids:
                break
            pos = torch.tensor([[cur]], device=self.device)
            logits, cache = self.runner.forward(next_tok, cache, pos)
            next_tok = sample(logits[:, -1:, :], sp.temperature)
            out_ids.append(int(next_tok.item()))
            cur += 1

        text = self.tokenizer.decode(out_ids, skip_special_tokens=True)
        return {"token_ids": out_ids, "text": text}

    @torch.inference_mode()
    def generate_chain(self, prompt, drafter, block_size: int = 4, sampling_params: SamplingParams = None) -> dict:
        """Chain (linear) speculative decode. Each round: the drafter proposes
        `block_size-1` tokens, the target verifies them in one forward, and we
        accept the longest prefix the target agrees with (greedy) plus one
        correction token. Lossless — the output equals plain greedy regardless of
        draft quality. Recompute-based (no KV reuse yet; M1a validates the accept
        logic, not speed). Returns {token_ids, text, tpf}, tpf = new / forwards."""
        sp = sampling_params or SamplingParams()
        if isinstance(prompt, str):
            committed = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        else:
            committed = prompt.to(self.device)
        k = max(1, block_size - 1)
        new_ids, rounds = [], 0
        while len(new_ids) < sp.max_new_tokens:
            drafts = drafter.propose(committed, k).to(self.device).view(-1)[:k]   # (k,)
            seq = torch.cat([committed, drafts.view(1, -1)], dim=1)               # (1, p+k)
            p = committed.shape[1]
            pos = torch.arange(seq.shape[1], device=self.device).unsqueeze(0)
            logits = self.model(input_ids=seq, position_ids=pos, use_cache=False).logits
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
                      budget: int = 15, algo: str = "crossproduct", sampling_params: SamplingParams = None) -> dict:
        """Tree speculative decode. Each round: the tree drafter emits per-depth
        logits, the tree algorithm builds a DraftTree, the target verifies all
        nodes in one forward under a 4D ancestor mask, and tree_accept takes the
        longest greedy-agreeing root-to-leaf path + a correction. Lossless —
        output equals plain greedy. Recompute-based (validates the tree verify,
        not speed). Returns {token_ids, text, tpf}."""
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
        new_ids, rounds = [], 0
        while len(new_ids) < sp.max_new_tokens:
            draft_logits = tree_drafter.propose_logits(committed, D).to(self.device)      # (1, D, V)
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
            logits = self.model(input_ids=seq, position_ids=pos, attention_mask=mask, use_cache=False).logits
            target_logits = logits[:, P:, :]                                              # (1, N, V)
            accepted_path, acc, correction = tree_accept(tree, target_logits, sp.temperature)
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
