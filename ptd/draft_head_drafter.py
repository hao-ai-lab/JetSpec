"""DraftHead-backed drafters — the real (checkpoint-gated) speculative drafters.

`DraftHeadDrafter` (chain) and `DraftHeadTreeDrafter` (tree) wrap a loaded DFlash
draft head (`ptd.models.draft_head.DFlashDraftModel`). The head owns neither
`embed_tokens` nor `lm_head` — it shares the *target's* (the DFlash convention),
so these drafters take the target module too and call `target.model.embed_tokens`
/ `target.lm_head` exactly as the reference `benchmark.py` does.

Kept in a separate module (not appended to `draft.py`) so the stub file stays
free of `transformers` / checkpoint imports; the real head pulls in `DynamicCache`
and the loaded model.

`_forward_head` mirrors `causal_parallel_drafting/benchmark.py` lines ~153-207
(chain, `tree_width=1`): build a noise embedding from `[anchor, mask_id*(block-1)]`,
run the head conditioned on the anchor's `target_hidden`, slice the `block_size-1`
real-prediction positions (gated on `draft_shift` — never hardcode), and apply the
target's `lm_head`. The recompute design feeds the running committed-context hidden
each round (the engine owns the decode loop; no draft-side KV reuse here).
"""
import torch
from transformers import DynamicCache

from ptd.draft import Drafter, TreeDrafter
from ptd.models.draft_head import DFlashContextCache


class _DraftHeadForward:
    """Shared head-forward helper for the chain + tree drafters.

    Both drafters need the same single-step head forward; this holds the head /
    target / config and exposes `_forward_head` returning raw `(1, depth, V)`
    logits. The two public drafters compose it (chain argmaxes, tree returns raw).

    Optional context cache (`use_context_cache=True`): the head writes the
    projected context + block K/V into a reused per-layer buffer instead of
    `torch.cat`-ing them each round, removing the `CatArrayBatched` copy (the #1 GPU
    bottleneck). The engine calls `reset_context_cache()` at the start of each
    `generate_tree`. Lossless by construction (see DFlashContextCache); OFF by
    default, so the recompute path is byte-unchanged.
    """

    def __init__(self, head, target, block_size: int, target_layer_ids, draft_shift: bool = False,
                 use_context_cache: bool = False):
        self.head = head
        self.target = target
        self.block_size = block_size
        self.target_layer_ids = target_layer_ids
        self.draft_shift = draft_shift
        self.use_context_cache = use_context_cache
        self._context_cache = DFlashContextCache() if use_context_cache else None
        # The head's device + dtype are the source of truth: every tensor we
        # build (anchor row, mask-id placeholders, position ids) goes here so
        # embed_tokens / fc / lm_head never hit a device or dtype mismatch.
        self.device = next(head.parameters()).device
        self.dtype = next(head.parameters()).dtype
        self.mask_token_id = head.mask_token_id

    def reset_context_cache(self) -> None:
        """Drop the persistent per-layer context K/V (call at the start of every
        generation). No-op when the cache mode is off."""
        if self._context_cache is not None:
            self._context_cache.reset()

    def _forward_head(self, context_ids: torch.Tensor, target_hidden: torch.Tensor, depth: int) -> torch.Tensor:
        """Run the head once and return `(1, depth, V)` draft logits.

        `context_ids` (1, T): committed context; its last token is the anchor (the
        speculative-block root). `target_hidden` (1, ctx_len, dim_concat): the
        tapped target hidden states the head conditions on (the K/V context).
        `depth`: number of real-prediction positions to return (= block_size - 1).
        """
        if target_hidden is None:
            raise ValueError(
                "DraftHead drafters require target_hidden; pass it from the "
                "ModelRunner forward (output_hidden_states + target_layer_ids)."
            )
        block_size = self.block_size
        anchor = context_ids[0, -1].view(1, 1).to(self.device)
        # block_output_ids = [anchor, mask_id, mask_id, ...] of length block_size
        # (benchmark.py:155,162 — the anchor seeds position 0, the rest are masked).
        mask_fill = torch.full(
            (1, block_size - 1), self.mask_token_id, dtype=anchor.dtype, device=self.device
        )
        block_output_ids = torch.cat([anchor, mask_fill], dim=1)  # (1, block_size)
        noise_embedding = self.target.model.embed_tokens(block_output_ids)  # (1, block_size, H)

        ctx_len = target_hidden.shape[1]
        # The head's K/V context is [ctx_tokens ; block positions]; rotary needs
        # absolute positions over that concatenation (benchmark.py:163 draft_pos_ids).
        position_ids = torch.arange(ctx_len + block_size, device=self.device).unsqueeze(0)

        # Cache mode: pass the context cache and NO DynamicCache (the head writes
        # context+block K/V into a reused per-layer buffer instead of torch.cat).
        # cached_kv_len stays 0 either way (fresh DynamicCache vs None), so the
        # block-causal mask is identical. Default: fresh DynamicCache (recompute).
        hidden = self.head(
            target_hidden=target_hidden.to(device=self.device, dtype=self.dtype),
            noise_embedding=noise_embedding,
            position_ids=position_ids,
            past_key_values=None if self.use_context_cache else DynamicCache(),
            use_cache=False,
            context_cache=self._context_cache,
            is_causal=self.head.resolve_causal_head("auto"),
        )  # (1, block_size, H)

        # Draft-logit slice: gated on draft_shift, never hardcode (the I-DLM bug).
        #   in-place (draft_shift=False): positions 1..block_size-1 are predictions
        #     -> slice(-block_size+1, None)
        #   shift (draft_shift=True): positions 0..block_size-2 are predictions
        #     -> slice(0, block_size-1)
        draft_slice = slice(0, block_size - 1) if self.draft_shift else slice(-block_size + 1, None)
        draft_logits = self.target.lm_head(hidden[:, draft_slice, :])  # (1, block_size-1, V)
        return draft_logits[:, :depth, :]


class DraftHeadDrafter(Drafter):
    """Chain drafter backed by a trained DFlash draft head.

    One head forward proposes `k = block_size - 1` next tokens (argmax of the
    per-position draft logits). The engine's chain verify loop accepts the longest
    target-agreeing prefix — lossless regardless of draft quality.
    """

    def __init__(self, head, target, block_size: int, target_layer_ids, draft_shift: bool = False,
                 use_context_cache: bool = False):
        self._fwd = _DraftHeadForward(head, target, block_size, target_layer_ids, draft_shift,
                                      use_context_cache)
        self.head = head
        self.target = target
        self.block_size = block_size
        self.target_layer_ids = target_layer_ids
        self.draft_shift = draft_shift

    def reset_context_cache(self) -> None:
        """Reset the persistent context cache (call at the start of each generation)."""
        self._fwd.reset_context_cache()

    @torch.inference_mode()
    def propose(
        self,
        context_ids: torch.Tensor,
        k: int,
        target_hidden: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:
        draft_logits = self._fwd._forward_head(context_ids, target_hidden, k)  # (1, k, V)
        return draft_logits.squeeze(0).argmax(dim=-1)  # (k,)


class DraftHeadTreeDrafter(TreeDrafter):
    """Tree drafter backed by a trained DFlash draft head.

    Emits the raw per-depth draft logits `(1, depth, V)` from one head forward;
    the tree algorithm turns them into a DraftTree and the engine verifies all
    nodes under a 4D ancestor mask. Lossless regardless of draft quality.
    """

    def __init__(self, head, target, block_size: int, target_layer_ids, draft_shift: bool = False,
                 use_context_cache: bool = False):
        self._fwd = _DraftHeadForward(head, target, block_size, target_layer_ids, draft_shift,
                                      use_context_cache)
        self.head = head
        self.target = target
        self.block_size = block_size
        self.target_layer_ids = target_layer_ids
        self.draft_shift = draft_shift

    def reset_context_cache(self) -> None:
        """Reset the persistent context cache (call at the start of each generation)."""
        self._fwd.reset_context_cache()

    @torch.inference_mode()
    def propose_logits(
        self,
        context_ids: torch.Tensor,
        depth: int,
        target_hidden: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:
        return self._fwd._forward_head(context_ids, target_hidden, depth)  # (1, depth, V)
