"""Drafters for speculative decoding.

A `Drafter` proposes the next `k` tokens given the committed context; the engine's
verify loop accepts the longest prefix the target agrees with. Speculative
decoding is **lossless** — it accepts only what the target would have produced
greedily, regardless of draft quality — so any drafter (even a trivial stub)
yields output byte-identical to plain greedy. That property lets the chain
plumbing be validated before the real drafter checkpoint exists.

The real drafter (M1a', checkpoint-gated) is `DraftHead`: a JF-trained causal
head sharing the target's `embed_tokens` + `lm_head` and tapping `target_hidden`
(the DFlash convention). Its variants are *parameters*, not subclasses
(`draft_shift` = the I-DLM shift, `block_size`, `target_layer_ids`, ...).
"""
from abc import ABC, abstractmethod

import torch


class Drafter(ABC):
    @abstractmethod
    def propose(self, context_ids: torch.Tensor, k: int) -> torch.Tensor:
        """Given context `(1, T)`, return a `(k,)` int tensor of proposed next tokens."""
        ...


class RepeatDrafter(Drafter):
    """Trivial stub — repeat the last context token `k` times. Exercises the
    verify/accept/rollback plumbing and losslessness; accept length is usually
    low (the point is correctness, not speedup)."""

    def propose(self, context_ids: torch.Tensor, k: int) -> torch.Tensor:
        return context_ids[0, -1].repeat(k)


class TargetEchoDrafter(Drafter):
    """Testing stub — propose the target's OWN greedy next-`k` tokens (runs the
    target, so no real speedup). Every draft is accepted, exercising the
    multi-token-accept path end-to-end and proving losslessness at accept == k."""

    def __init__(self, model):
        self.model = model

    @torch.inference_mode()
    def propose(self, context_ids: torch.Tensor, k: int) -> torch.Tensor:
        from transformers import DynamicCache

        ids = context_ids
        cache = DynamicCache()
        pos = torch.arange(ids.shape[1], device=ids.device).unsqueeze(0)
        logits = self.model(
            input_ids=ids, position_ids=pos, past_key_values=cache, use_cache=True
        ).logits
        nxt = logits[:, -1:, :].argmax(-1)  # (1, 1)
        out = [nxt]
        cur = ids.shape[1]
        for _ in range(k - 1):
            p = torch.tensor([[cur]], device=ids.device)
            logits = self.model(
                input_ids=nxt, position_ids=p, past_key_values=cache, use_cache=True
            ).logits
            nxt = logits[:, -1:, :].argmax(-1)
            out.append(nxt)
            cur += 1
        return torch.cat(out, dim=1)[0]  # (k,)


class TreeDrafter(ABC):
    """A tree drafter emits per-depth logits `(1, D, vocab)`; the tree algorithm
    (crossproduct, ...) turns those into a DraftTree. The real DraftHead emits
    these from one forward; the stubs below validate the tree verify path."""

    @abstractmethod
    def propose_logits(self, context_ids: torch.Tensor, depth: int) -> torch.Tensor:
        ...


class RandomTreeDrafter(TreeDrafter):
    """Trivial stub — random per-depth logits → a random tree. Lossless."""

    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size

    def propose_logits(self, context_ids: torch.Tensor, depth: int) -> torch.Tensor:
        return torch.randn(1, depth, self.vocab_size, device=context_ids.device)


class TargetEchoTreeDrafter(TreeDrafter):
    """Testing stub — per-depth logits = the target's OWN greedy logits, so the
    crossproduct top-1 path IS the greedy chain → the verify accepts the full
    depth. Proves losslessness + the multi-node-accept path (runs the target, so
    no real speedup)."""

    def __init__(self, model):
        self.model = model

    @torch.inference_mode()
    def propose_logits(self, context_ids: torch.Tensor, depth: int) -> torch.Tensor:
        from transformers import DynamicCache

        ids = context_ids
        cache = DynamicCache()
        pos = torch.arange(ids.shape[1], device=ids.device).unsqueeze(0)
        logits = self.model(
            input_ids=ids, position_ids=pos, past_key_values=cache, use_cache=True
        ).logits
        cols = [logits[:, -1:, :]]            # logits predicting the depth-1 token
        nxt = logits[:, -1:, :].argmax(-1)
        cur = ids.shape[1]
        for _ in range(depth - 1):
            p = torch.tensor([[cur]], device=ids.device)
            logits = self.model(
                input_ids=nxt, position_ids=p, past_key_values=cache, use_cache=True
            ).logits
            cols.append(logits[:, -1:, :])
            nxt = logits[:, -1:, :].argmax(-1)
            cur += 1
        return torch.cat(cols, dim=1)         # (1, depth, vocab)
