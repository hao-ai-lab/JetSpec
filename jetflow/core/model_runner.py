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


def _target_owner(model):
    return getattr(model, "_orig_mod", model)


def _can_capture_target_hidden(model, target_layer_ids) -> bool:
    if target_layer_ids is None:
        return False
    owner = _target_owner(model)
    layers = getattr(getattr(owner, "model", None), "layers", None)
    if layers is None:
        return False
    return all(0 <= int(layer_id) < len(layers) for layer_id in target_layer_ids)


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


@contextmanager
def _capture_target_hidden(model, target_layer_ids):
    """Capture selected decoder layer outputs without HF's hidden_states wrapper."""
    if target_layer_ids is None:
        yield None
        return
    owner = _target_owner(model)
    layers = getattr(getattr(owner, "model", None), "layers", None)
    if layers is None:
        yield None
        return

    captured = {}
    handles = []

    def _make_hook(layer_id):
        def _hook(_module, _args, output):
            captured[layer_id] = output[0] if isinstance(output, tuple) else output
        return _hook

    for layer_id in target_layer_ids:
        handles.append(layers[int(layer_id)].register_forward_hook(_make_hook(int(layer_id))))
    try:
        yield captured
    finally:
        for handle in handles:
            handle.remove()


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
        need_hidden = output_hidden_states and target_layer_ids is not None
        use_hook_hidden = need_hidden and _can_capture_target_hidden(self.model, target_layer_ids)
        with (
            _capture_target_hidden(self.model, target_layer_ids if use_hook_hidden else None) as hidden_by_layer,
            _masked_verify_attention(self.model, attention_mask),
        ):
            out = self.model(
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                cache_position=cache_position,
                use_cache=True,
                output_hidden_states=need_hidden and not use_hook_hidden,
                logits_to_keep=1 if last_position_logits_only else 0,
            )
        target_hidden = None
        if use_hook_hidden and hidden_by_layer is not None:
            target_hidden = torch.cat(
                [hidden_by_layer[int(layer_id)] for layer_id in target_layer_ids],
                dim=-1,
            )
        elif need_hidden:
            from jetflow.models.draft_head import extract_context_feature

            target_hidden = extract_context_feature(out.hidden_states, target_layer_ids)
        return out.logits, out.past_key_values, target_hidden
