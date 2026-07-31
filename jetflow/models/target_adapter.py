"""Architecture-neutral target-model helpers.

The HF/SDPA execution path intentionally works through the public causal-LM
contract instead of reaching into one model family's module layout.  Gemma 4's
full checkpoint, for example, nests its text decoder under
``model.language_model`` while Qwen3 and GPT-OSS expose ``model.embed_tokens``.
HF's ``get_input_embeddings`` / ``get_output_embeddings`` APIs normalize that
difference.

Hybrid targets (currently Gemma 4 and GPT-OSS) also need two prepared tree
masks: full-attention layers see the whole cached prefix, while sliding layers
see only keys inside their logical position window.  Passing one prebuilt 4-D
mask to HF bypasses its normal sliding-mask construction, so JetFlow must
preserve that restriction explicitly.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


QWEN3_MODEL_TYPES = frozenset({"qwen3", "qwen3_moe"})
SDPA_MODEL_TYPES = frozenset({
    "qwen3",
    "qwen3_moe",
    "gpt_oss",
    "gemma4",
    "gemma4_text",
})
REFERENCE_TRITON_MODEL_TYPES = SDPA_MODEL_TYPES


def _config_owner(model):
    if hasattr(model, "config"):
        return model
    original = getattr(model, "_orig_mod", None)
    if original is not None and hasattr(original, "config"):
        return original
    raise TypeError(f"target model {type(model).__name__} has no Hugging Face config")


def target_config(model):
    """Return the decoder/text config for plain and multimodal causal LMs."""
    config = _config_owner(model).config
    get_text_config = getattr(config, "get_text_config", None)
    return get_text_config() if get_text_config is not None else config


def target_model_type(model) -> str:
    """Return the outer HF model type used for backend capability dispatch."""
    return str(getattr(_config_owner(model).config, "model_type", ""))


def get_target_input_embeddings(model):
    try:
        owner = _config_owner(model)
    except TypeError:
        owner = model
    get_embeddings = getattr(owner, "get_input_embeddings", None)
    embeddings = get_embeddings() if get_embeddings is not None else None
    if embeddings is None:
        base = getattr(owner, "model", None)
        embeddings = getattr(base, "embed_tokens", None)
    if embeddings is None:
        raise TypeError(f"{type(model).__name__} exposes no input embeddings")
    return embeddings


def get_target_output_embeddings(model):
    try:
        owner = _config_owner(model)
    except TypeError:
        owner = model
    get_embeddings = getattr(owner, "get_output_embeddings", None)
    embeddings = get_embeddings() if get_embeddings is not None else None
    if embeddings is None:
        embeddings = getattr(owner, "lm_head", None)
    if embeddings is None:
        raise TypeError(f"{type(model).__name__} exposes no output embeddings")
    return embeddings


def require_backend_support(model, backend: str) -> None:
    """Fail before decoding when an architecture/backend pair is not validated."""
    model_type = target_model_type(model)
    if backend in {"sdpa", "triton"}:
        supported = (
            REFERENCE_TRITON_MODEL_TYPES if backend == "triton"
            else SDPA_MODEL_TYPES
        )
        if model_type not in supported:
            raise ValueError(
                f"target model_type={model_type!r} is not validated for JetFlow "
                f"{backend}; supported: {sorted(supported)}"
            )
        return
    if model_type not in QWEN3_MODEL_TYPES:
        raise ValueError(
            f"backend={backend!r} is currently Qwen3/Qwen3-MoE-only; "
            f"target model_type={model_type!r} must use backend='sdpa'"
        )


@dataclass(frozen=True)
class TargetCompatibility:
    model_type: str
    hidden_size: int
    vocab_size: int
    num_hidden_layers: int


def validate_draft_target_compatibility(head, target) -> TargetCompatibility:
    """Validate the shape/token contract shared by a DFlash head and target."""
    config = target_config(target)
    model_type = target_model_type(target)
    require_backend_support(target, "sdpa")

    expected = {
        "hidden_size": int(config.hidden_size),
        "vocab_size": int(config.vocab_size),
        "num_hidden_layers": int(config.num_hidden_layers),
    }
    actual = {
        "hidden_size": int(head.config.hidden_size),
        "vocab_size": int(head.config.vocab_size),
        "num_hidden_layers": int(head.config.num_target_layers),
    }
    mismatches = [
        f"{name}: target={expected[name]}, draft={actual[name]}"
        for name in expected
        if expected[name] != actual[name]
    ]
    layer_ids = [int(layer_id) for layer_id in head.target_layer_ids]
    invalid_layers = [
        layer_id for layer_id in layer_ids
        if layer_id < 0 or layer_id >= expected["num_hidden_layers"]
    ]
    if invalid_layers:
        mismatches.append(
            f"target_layer_ids outside [0, {expected['num_hidden_layers']}): {invalid_layers}"
        )
    mask_token_id = head.mask_token_id
    if mask_token_id is None or not 0 <= int(mask_token_id) < expected["vocab_size"]:
        mismatches.append(
            f"mask_token_id must be inside target vocabulary: {mask_token_id!r}"
        )
    if mismatches:
        raise ValueError(
            f"incompatible draft head and {model_type!r} target: " + "; ".join(mismatches)
        )
    return TargetCompatibility(model_type=model_type, **expected)


def _additive_mask(allowed: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    device = allowed.device
    zero = torch.zeros((), dtype=dtype, device=device)
    blocked = torch.full((), torch.finfo(dtype).min, dtype=dtype, device=device)
    return torch.where(allowed, zero, blocked).view(
        1, 1, allowed.shape[0], allowed.shape[1]
    )


def build_tree_attention_mask(
    model,
    ancestor: torch.Tensor,
    *,
    past_len: int,
    depths: torch.Tensor,
    dtype: torch.dtype,
):
    """Build the prepared HF tree mask, preserving hybrid sliding semantics.

    ``ancestor`` is ``(N, N)`` over tree-ordered nodes.  Cache slots for those
    nodes are physically contiguous, but their logical RoPE/window positions are
    ``past_len + depth``; sliding eligibility must therefore use ``depths`` rather
    than the physical node index.
    """
    device = ancestor.device
    depths = depths.to(device=device, dtype=torch.long).reshape(-1)
    num_nodes = int(ancestor.shape[0])
    if depths.numel() != num_nodes:
        raise ValueError(
            f"tree depths has {depths.numel()} entries for {num_nodes} nodes"
        )

    allowed = torch.zeros(
        num_nodes, past_len + num_nodes, dtype=torch.bool, device=device
    )
    allowed[:, :past_len] = True
    allowed[:, past_len:] = ancestor.to(device=device, dtype=torch.bool)
    full_mask = _additive_mask(allowed, dtype)

    config = target_config(model)
    layer_types = set(getattr(config, "layer_types", ()) or ())
    if "sliding_attention" not in layer_types:
        return full_mask

    sliding_window = getattr(config, "sliding_window", None)
    if sliding_window is None or int(sliding_window) <= 0:
        raise ValueError(
            f"model_type={target_model_type(model)!r} has sliding layers but "
            f"invalid sliding_window={sliding_window!r}"
        )
    query_positions = past_len + depths
    prefix_positions = torch.arange(past_len, device=device)
    node_positions = past_len + depths
    key_positions = torch.cat([prefix_positions, node_positions])
    in_window = key_positions.unsqueeze(0) > (
        query_positions.unsqueeze(1) - int(sliding_window)
    )
    sliding_mask = _additive_mask(allowed & in_window, dtype)

    masks = {}
    if "full_attention" in layer_types:
        masks["full_attention"] = full_mask
    masks["sliding_attention"] = sliding_mask
    return masks
