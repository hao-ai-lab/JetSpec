"""Full-block hidden-state forward used by correction-layer tree drafters."""
from __future__ import annotations

import torch

from jetspec.draft_head_adapter import _DraftHeadForward


class _HiddenForward(_DraftHeadForward):
    """Return every post-normalization draft-head slot before projection.

    This preserves the validated cache, noise-embedding, and absolute-position
    behavior of the standard draft-head path. Unlike the standard helper, it
    returns (1, block_size, hidden_size) so a correction layer can produce
    path-conditioned logits for individual tree nodes.
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
                "Correction-layer drafter requires target_hidden; pass it from "
                "the verify forward (output_hidden_states + target_layer_ids)."
            )
        noise_embedding = self.noise_embedding(
            context_ids,
            fill_tokens=fill_tokens,
        )

        ctx_len = target_hidden.shape[1]
        cached_len = int(self.cache.get_seq_length())
        if cached_len > ctx_len:
            self.reset_cache()
            cached_len = 0
        target_hidden_step = target_hidden[:, cached_len:, :].to(
            device=self.device,
            dtype=self.dtype,
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
        )
        self.cache.crop(ctx_len)
        return hidden


__all__ = ["_HiddenForward"]
