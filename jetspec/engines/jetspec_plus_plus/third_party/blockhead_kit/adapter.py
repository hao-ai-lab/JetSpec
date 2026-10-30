"""DSpark linear-block drafter for JetSpec (Phase 1).

DSpark's backbone + incremental draft KV-cache + noise-embedding are identical to
DFlash, so this reuses `jetspec.draft_head_adapter._DraftHeadForward` and only
changes the proposal:

  - consume ALL `block_size` output positions (the anchor slot at position 0
    predicts the first draft token), NOT the DFlash `[-block_size+1:]` slice;
  - AR Markov sampling (`head.sample_draft_tokens`) instead of per-position
    argmax — so a parent at depth d uses `base_logits[d] + markov(prev_token)`.

`num_draft_tokens = block_size` (e.g. 7 for block7). The target `lm_head` is
applied here (the head does not own it), matching the DFlash convention.
"""
import torch

from jetspec.draft_head_adapter import _DraftHeadForward


class _DSparkForward(_DraftHeadForward):
    """`_DraftHeadForward` that returns the full per-position backbone hidden.

    Identical caching / noise-embedding / RoPE logic as the DFlash path (which is
    validated), but returns `hidden (1, block_size, H)` over ALL positions rather
    than slicing and applying `lm_head`. The caller applies the target `lm_head`
    and the Markov head.
    """

    @torch.inference_mode()
    def forward_hidden(
        self,
        context_ids: torch.Tensor,
        target_hidden: torch.Tensor,
        fill_tokens=None,
    ) -> torch.Tensor:
        if target_hidden is None:
            raise ValueError(
                "DSpark drafter requires target_hidden; pass it from the verify "
                "forward (output_hidden_states + target_layer_ids)."
            )
        # noise input = [anchor, mask*(block_size-1)] of length block_size.
        noise_embedding = self.noise_embedding(context_ids, fill_tokens=fill_tokens).to(
            device=self.device, dtype=self.dtype
        )

        ctx_len = target_hidden.shape[1]
        cached_len = int(self.cache.get_seq_length())
        if cached_len > ctx_len:
            self.reset_cache()
            cached_len = 0
        target_hidden_step = target_hidden[:, cached_len:, :].to(
            device=self.device, dtype=self.dtype
        )
        position_ids = torch.arange(
            cached_len,
            ctx_len + self.block_size,
            device=self.device,
        ).unsqueeze(0)

        hidden = self.head(
            target_hidden=target_hidden_step,
            noise_embedding=noise_embedding,
            position_ids=position_ids,
            past_key_values=self.cache,
            use_cache=True,
            is_causal=self.head.resolve_causal_head("auto"),
        )  # (1, block_size, H)
        self.cache.crop(ctx_len)
        return hidden


class DSparkLinearDrafter:
    """Linear-block DSpark drafter.

    One backbone forward per round, then AR Markov sampling over all
    `block_size` positions -> `block_size` draft tokens. Greedy verify is owned
    by the engine / `dspark_generate` (lossless at temperature=0 for any drafter).
    """

    def __init__(
        self,
        head,
        target,
        block_size: int,
        target_layer_ids,
        temperature: float = 0.0,
    ):
        self._fwd = _DSparkForward(
            head, target, block_size, target_layer_ids, draft_shift=False
        )
        self.head = head
        self.target = target
        self.block_size = int(block_size)
        self.num_draft_tokens = int(block_size)  # DSpark proposes B tokens (anchor slot reused)
        self.target_layer_ids = target_layer_ids
        self.temperature = float(temperature)

    def reset_cache(self) -> None:
        self._fwd.reset_cache()

    @torch.inference_mode()
    def propose(
        self,
        context_ids: torch.Tensor,
        k: int,
        target_hidden: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:
        """Return up to `k` AR-Markov-sampled draft token ids (1D, len <= k)."""
        hidden = self._fwd.forward_hidden(context_ids, target_hidden)  # (1, B, H)
        B = self.block_size
        proposal_hidden = hidden[:, :B, :]
        base_logits = self.target.lm_head(proposal_hidden)  # (1, B, V)
        anchor = context_ids[:, -1]  # (1,) — first_prev for the Markov head
        sampled, _ = self.head.sample_draft_tokens(
            base_logits,
            first_prev_token_ids=anchor,
            temperature=self.temperature,
            hidden_states=proposal_hidden,
        )  # (1, B)
        return sampled.reshape(-1)[:k]


class DSparkTreeDrafter:
    """Tree drafter for DSpark.

    Emits the RAW per-depth base logits `(1, depth, V)` from one backbone forward
    (NO Markov bias). The Markov head is order-1 (vanilla), so the per-node bias is
    path-conditional (depends on the parent token) and CANNOT be applied here — it
    is applied per tree node by the `markov_accum_logp` tree algorithm. This keeps
    the same drafter usable for two ablations on the SAME checkpoint:

      - Markov-OFF (DSpark backbone-only / DFlash-style at block7): feed these base
        logits to `accum_logp` (position-independent per-depth top-k).
      - Markov-ON: feed them to `markov_accum_logp` with `head.markov_head`.

    Any accept-len gap between the two is purely the Markov head, at a fixed block
    size / checkpoint (controls for DFlash's larger-block resource advantage).
    """

    def __init__(
        self,
        head,
        target,
        block_size: int,
        target_layer_ids,
        temperature: float = 0.0,
    ):
        self._fwd = _DSparkForward(
            head, target, block_size, target_layer_ids, draft_shift=False
        )
        self.head = head
        self.target = target
        self.block_size = int(block_size)
        self.target_layer_ids = target_layer_ids
        self.temperature = float(temperature)
        # exposed so the harness can pass it to markov_accum_logp via algo_kwargs.
        self.markov_head = head.markov_head

    def reset_cache(self) -> None:
        self._fwd.reset_cache()

    @torch.inference_mode()
    def propose_logits(
        self,
        context_ids: torch.Tensor,
        depth: int,
        target_hidden: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:
        """Return `(1, depth, V)` raw base logits (no Markov bias)."""
        if depth > self.block_size:
            raise ValueError(
                f"DSparkTreeDrafter depth={depth} exceeds head block_size="
                f"{self.block_size}; the backbone only produces {self.block_size} "
                "base positions. Call generate_tree with block_size = head.block_size + 1."
            )
        hidden = self._fwd.forward_hidden(context_ids, target_hidden)  # (1, B, H)
        base_logits = self.target.lm_head(hidden[:, : self.block_size, :])  # (1, B, V)
        return base_logits[:, :depth, :]


__all__ = ["DSparkLinearDrafter", "DSparkTreeDrafter", "_DSparkForward"]
