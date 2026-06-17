"""The single place the engine touches the model forward.

Today: a thin pass-through to the HF target with an explicit `position_ids` and
an HF `DynamicCache`. The draft-head forward and the tree-mask path (a 4D additive
ancestor mask passed to the same HF forward) build on this; a dedicated
tree-attention kernel later swaps out the mask path.
"""
from contextlib import contextmanager

import torch


def _config_owner(model):
    if hasattr(model, "config"):
        return model
    orig = getattr(model, "_orig_mod", None)
    if orig is not None and hasattr(orig, "config"):
        return orig
    return None


@contextmanager
def _masked_verify_attention(model, attention_mask):
    """Use SDPA for explicit tree masks when the normal backend is FA2.

    FlashAttention2 is the right backend for normal target forwards, but HF's
    FA2 path does not support the 4D additive ancestor masks used by tree verify.
    The reference benchmark temporarily dispatches masked verify through SDPA;
    keep that behavior at the shared model-forward seam.
    """
    owner = _config_owner(model)
    if attention_mask is None or owner is None:
        yield
        return
    config = owner.config
    previous = getattr(config, "_attn_implementation", None)
    if previous != "flash_attention_2":
        yield
        return
    config._attn_implementation = "sdpa"
    try:
        yield
    finally:
        config._attn_implementation = previous


class ModelRunner:
    def __init__(self, model):
        self.model = model

    @torch.inference_mode()
    def forward(
        self,
        input_ids,
        past_key_values,
        position_ids=None,
        output_hidden_states: bool = False,
        target_layer_ids=None,
        attention_mask=None,
        cache_position=None,
        last_position_logits_only: bool = False,
    ):
        """One target forward step. Returns (logits, past_key_values, target_hidden).

        `position_ids` is passed explicitly (not inferred) so the offline loop's
        positions are deterministic and match HF greedy generation exactly.

        `cache_position` indexes the new tokens into a populated KV cache; left
        `None` for normal prefill/decode calls where HF derives it from past length.

        `target_hidden` is the pre-extracted concatenated tapped-layer hidden
        states (1, T, len(target_layer_ids)*H) when `output_hidden_states` and
        `target_layer_ids` are both set; else `None`. Extracting here (the single
        place the engine touches the model forward) avoids re-gathering the full
        hidden-states tuple downstream.

        `attention_mask` lets the tree-verify path route its 4D additive ancestor
        mask through this same forward (keeping the single-seam invariant).

        `last_position_logits_only` asks HF to materialize only the final logits
        position when callers only need next-token scores. This is equivalent to
        `logits_to_keep=1` in model-specific HF forwards, but keeps this runner's
        API named after the behavior the rest of the engine relies on.
        """
        with _masked_verify_attention(self.model, attention_mask):
            out = self.model(
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                cache_position=cache_position,
                use_cache=True,
                output_hidden_states=output_hidden_states,
                logits_to_keep=1 if last_position_logits_only else 0,
            )
        target_hidden = None
        if output_hidden_states and target_layer_ids is not None:
            from jetflow.models.draft_head import extract_context_feature

            target_hidden = extract_context_feature(out.hidden_states, target_layer_ids)
        return out.logits, out.past_key_values, target_hidden
