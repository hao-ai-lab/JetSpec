# Adaptation: package relocation and deferred optional dependencies.
"""Shared torch reference for the trained and frozen tap-aligned H2H adapter."""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Optional

import torch
import torch.utils.checkpoint
import torch.nn.functional as F
from torch import nn


CTXKV_HEAD_KEY = "ctxkv_head"

# Modes whose input, output and loss all live in tapped-hidden space. The projection
# term is structurally absent for every one of them, so a new H-space arm is added here
# rather than to each gate that used to name the mode string directly.
H2H_MODES = frozenset({"l1_attention_h2h", "c1_tap_aligned_h2h"})
H5_G_MASS_CONVENTION = "w136-acceptlen-to-g-equalweight-v2"  # Pinned adapter configuration convention identity.

# servingw29: where a stream block's FFN (pre-MLP norm + MLP) comes from. Absent = the
# block trains its own, exactly as before. "target_frozen" = the block runs the TARGET's
# own post_attention_layernorm + mlp of one layer, frozen, never saved with the head.
# servingw33: "target_init" = the same target tensors are copied in as the INITIAL values
# of ordinary trainable parameters (in the optimizer, saved, trained like the parent's).
FFN_SOURCES = frozenset({"target_frozen", "target_init"})
FROZEN_FFN_SOURCES = frozenset({"target_frozen"})
# (block attribute path, target module suffix). The pre-MLP norm travels WITH the MLP:
# it is the norm that MLP was trained behind.
TARGET_FFN_PIECES = (
    ("post_attention_norm", "post_attention_layernorm"),
    ("mlp.gate_proj", "mlp.gate_proj"),
    ("mlp.up_proj", "mlp.up_proj"),
    ("mlp.down_proj", "mlp.down_proj"),
)
# servingw37: where an EXTRA block's (block j >= 2 of a stream) attention starts. Absent =
# random, exactly as before. "target_same_layer" = the target module that block's FFN
# starts from (ffn_layer_map), with the block-1 transplant's pieces and its kv-side norm
# rule (kv_norm <- input_layernorm); block 1 keeps its tinit (model.layers[L+1]) source.
EXTRA_BLOCK_ATTENTION_INITS = frozenset({"target_same_layer"})
# servingw42: where block 1's attention starts. Absent = tinit (model.layers[L+1]), exactly
# as before. "target_same_layer" = the target module block 1's FFN starts from
# (ffn_layer_map[stream][0]); init_c1_from_target reads that module instead of L+1, with
# the same seven pieces. Every block is then a whole target layer.
BLOCK1_ATTENTION_INITS = frozenset({"target_same_layer"})
# (block attribute path, target module suffix), in init_c1_from_target's piece order.
TARGET_ATTENTION_PIECES = (
    ("q_proj", "self_attn.q_proj"),
    ("k_proj", "self_attn.k_proj"),
    ("v_proj", "self_attn.v_proj"),
    ("o_proj", "self_attn.o_proj"),
    ("q_norm", "self_attn.q_norm"),
    ("k_norm", "self_attn.k_norm"),
    ("kv_norm", "input_layernorm"),
)


def resolve_ffn_layer_map(target_layer_ids, stream_depths) -> list[list[int]]:
    """Target module index (model.layers[m], counted from 0) for every stream block.

    Tap L is the output of model.layers[L] (hidden_states[L + 1]; hidden_states[0] is
    the token embedding). The stream at tap L maps the tap below to h^L, so it stands in
    for modules L_below + 1 .. L; the first stream starts from the token embedding, so
    its L_below is -1. Block b (1..d_L) takes module L_below + ceil(b (L - L_below) / d_L),
    exact integer ceiling, no exceptions: the last block of every stream is module L.
    """
    table = []
    below = -1
    for tap, depth in zip(target_layer_ids, stream_depths, strict=True):
        tap = int(tap)
        depth = int(depth)
        span = tap - below
        if span < 1 or depth < 1:
            raise ValueError(
                f"ffn layer map needs ascending taps and positive depths: tap={tap} "
                f"below={below} depth={depth}"
            )
        table.append(
            [below + (block * span + depth - 1) // depth for block in range(1, depth + 1)]
        )
        below = tap
    return table

def _freeze_weight_as_target_buffer(module: nn.Module) -> None:
    """Turn ``module.weight`` into a frozen, never-saved, NaN-poisoned buffer.

    The module is constructed first exactly as the trainable path does (same RNG
    draws, so every trainable tensor initialises exactly as in the parent), then its
    weight stops being a parameter: no gradient, no optimizer slot, no broadcast, no
    state_dict entry. NaN until load_target_ffns copies the target's tensor in, so an
    unloaded FFN can never compute something plausible.
    """
    weight = module.weight
    del module.weight
    module.register_buffer(
        "weight", torch.full_like(weight.detach(), float("nan")), persistent=False
    )

def _resolve_h5_g_mass(resolved: dict) -> dict[str, float]:
    convention = resolved.get("g_mass_convention")
    if convention != H5_G_MASS_CONVENTION:
        raise ValueError(
            "unsupported ctxkv g-mass convention: "
            f"{convention!r}; expected {H5_G_MASS_CONVENTION!r}."
        )
    source_path = resolved.get("g_histogram_path")
    expected_sha256 = resolved.get("g_histogram_sha256")
    if not isinstance(source_path, str) or not source_path:
        raise ValueError("ctxkv H5 g-mass requires g_histogram_path.")
    if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
        raise ValueError("ctxkv H5 g-mass requires a SHA-256 source pin.")
    raw_bytes = Path(source_path).read_bytes()
    actual_sha256 = hashlib.sha256(raw_bytes).hexdigest()
    if actual_sha256 != expected_sha256:
        raise ValueError(
            "ctxkv H5 histogram SHA-256 mismatch: "
            f"expected={expected_sha256} actual={actual_sha256}."
        )
    artifact = json.loads(raw_bytes)
    datasets = ("gsm8k", "math500")
    per_dataset = {}
    for dataset in datasets:
        record = artifact.get(dataset)
        if not isinstance(record, dict):
            raise ValueError(f"ctxkv H5 histogram missing dataset {dataset}.")
        if record.get("cross_dataset_pooling_applied") is not False:
            raise ValueError(
                "ctxkv H5 source must preserve unpooled per-dataset histograms."
            )
        ceiling = record.get("ceiling")
        if not isinstance(ceiling, dict) or (
            ceiling.get("accept_len") != 16
            or ceiling.get("interpretation") != "TRUNCATION_NOT_TAIL"
        ):
            raise ValueError(
                "ctxkv H5 accept_len=16 must be stamped TRUNCATION_NOT_TAIL."
            )
        counts = {}
        for item in record.get("histogram", []):
            if not isinstance(item, dict):
                raise ValueError("ctxkv H5 histogram rows must be dictionaries.")
            accept_len = item.get("accept_len")
            count = item.get("count")
            if (
                not isinstance(accept_len, int)
                or isinstance(accept_len, bool)
                or not isinstance(count, int)
                or isinstance(count, bool)
                or count <= 0
                or accept_len in counts
            ):
                raise ValueError("ctxkv H5 histogram rows are invalid or duplicated.")
            counts[accept_len] = count
        if set(counts) != set(range(1, 17)):
            raise ValueError(
                "ctxkv H5 accept-length source support must be exactly 1..16."
            )
        retained_total = sum(counts[g] for g in range(1, 16))
        per_dataset[dataset] = {
            g: counts[g] / retained_total for g in range(1, 16)
        }
    distribution = {
        str(g): 0.5 * per_dataset["gsm8k"][g]
        + 0.5 * per_dataset["math500"][g]
        for g in range(1, 16)
    }
    served_g_support = sorted(int(g) for g in distribution)
    max_served_g = max(served_g_support)
    required_offset_rows = max_served_g + 1  # driver query span is 1 + g
    offset_margin_rows = 1
    offset_embedding_rows = required_offset_rows + offset_margin_rows
    resolved.update(
        {
            "g_mass_stamp": f"G_MASS={convention}",
            "g_mass_kind": "CONVENTION",
            "g_histogram_source_path": source_path,
            "g_histogram_source_sha256": actual_sha256,
            "g_mass_derivation": [
                "source=accept_len_histograms",
                "drop_accept_len=16 reason=TRUNCATION_NOT_TAIL",
                "renormalise_each_dataset_over_g=1..15",
                "equal_weight_datasets=gsm8k,math500",
            ],
            "raw_histograms": artifact,
            "served_g_support": served_g_support,
            "max_served_g": max_served_g,
            "query_span_formula": "1+g",
            "required_offset_rows": required_offset_rows,
            "offset_margin_rows": offset_margin_rows,
            "offset_embedding_rows": offset_embedding_rows,
            "offset_rows_stamp": (
                f"OFFSET_ROWS={offset_embedding_rows} "
                f"(required {required_offset_rows} at g_max={max_served_g}; "
                f"+{offset_margin_rows} deliberate margin)"
            ),
        }
    )
    return distribution

def _resolve_training_config(config, validate) -> dict:
    dflash_config = getattr(config, "dflash_config", None) or {}
    raw = dflash_config.get(CTXKV_HEAD_KEY)
    if not isinstance(raw, dict):
        raise ValueError("dflash_config.ctxkv_head must be a dictionary.")

    mode = raw.get("mode")
    if mode not in {
        "l1_attention",
        "l1_attention_h2h",
        "l1_attention_mlp",
        "l0_mlp",
        "c1_tap_aligned_h2h",
    }:
        raise ValueError(
            "dflash_config.ctxkv_head.mode must be 'l1_attention', "
            "'l1_attention_h2h', 'l1_attention_mlp', or 'l0_mlp'."
        )
    resolved = dict(raw)
    if "stream_depths" in resolved:
        stream_depths = resolved["stream_depths"]
        if (
            mode != "c1_tap_aligned_h2h"
            or not isinstance(stream_depths, (list, tuple))
            or not stream_depths
            or any(
                not isinstance(depth, int)
                or isinstance(depth, bool)
                or depth < 1
                for depth in stream_depths
            )
        ):
            raise ValueError(
                "dflash_config.ctxkv_head.stream_depths must contain positive "
                "integers for c1_tap_aligned_h2h."
            )
        resolved["stream_depths"] = list(stream_depths)
    if "ffn_source" in resolved:
        if mode != "c1_tap_aligned_h2h" or resolved["ffn_source"] not in FFN_SOURCES:
            raise ValueError(
                "dflash_config.ctxkv_head.ffn_source must be one of "
                f"{sorted(FFN_SOURCES)} and requires c1_tap_aligned_h2h; got "
                f"{resolved['ffn_source']!r} under mode {mode!r}."
            )
        taps = list(dflash_config.get("target_layer_ids", []))
        depths = resolved.get("stream_depths")
        if depths is None:
            depths = [int(resolved.get("layers_per_stream", 1))] * len(taps)
        # Derived, never configured: it is not in KNOWN_CTXKV_KEYS, so a raw config that
        # tried to supply it would be refused as an unread key.
        resolved["ffn_layer_map"] = resolve_ffn_layer_map(taps, depths)
    if "extra_block_attention_init" in resolved:
        # servingw37: the extra blocks' attention source is the module their FFN starts
        # from, so the FFN map must exist; block 1 keeps tinit, so tinit must be on.
        if (
            mode != "c1_tap_aligned_h2h"
            or not isinstance(resolved["extra_block_attention_init"], str)
            or resolved["extra_block_attention_init"] not in EXTRA_BLOCK_ATTENTION_INITS
            or "ffn_source" not in resolved
            or resolved.get("attention_init") != "target_layer_plus_one"
        ):
            raise ValueError(
                "dflash_config.ctxkv_head.extra_block_attention_init must be one of "
                f"{sorted(EXTRA_BLOCK_ATTENTION_INITS)} and requires c1_tap_aligned_h2h, "
                "ffn_source and attention_init=target_layer_plus_one; got "
                f"{resolved['extra_block_attention_init']!r} under mode {mode!r}."
            )
    if "block1_attention_init" in resolved:
        # servingw42: block 1's attention source is the module its FFN starts from, so the
        # FFN map must exist and tinit must be on; only on top of the servingw37 whole-layer
        # arm (extra_block_attention_init), so every block is a whole target layer.
        if (
            mode != "c1_tap_aligned_h2h"
            or not isinstance(resolved["block1_attention_init"], str)
            or resolved["block1_attention_init"] not in BLOCK1_ATTENTION_INITS
            or "ffn_source" not in resolved
            or resolved.get("attention_init") != "target_layer_plus_one"
            or resolved.get("extra_block_attention_init") != "target_same_layer"
        ):
            raise ValueError(
                "dflash_config.ctxkv_head.block1_attention_init must be one of "
                f"{sorted(BLOCK1_ATTENTION_INITS)} and requires c1_tap_aligned_h2h, "
                "ffn_source, attention_init=target_layer_plus_one and "
                "extra_block_attention_init=target_same_layer; got "
                f"{resolved['block1_attention_init']!r} under mode {mode!r}."
            )
    for key in (
        "residual_parameterization",
        "offset_conditioning",
        "consumption_space_loss",
        "allow_extrapolated_attention_mass",
    ):
        if key in resolved and not isinstance(resolved[key], bool):
            raise ValueError(f"dflash_config.ctxkv_head.{key} must be a boolean.")
    per_tap_loss = resolved.get("per_tap_loss", False)
    if not isinstance(per_tap_loss, bool):
        raise ValueError("dflash_config.ctxkv_head.per_tap_loss must be a boolean.")
    resolved["per_tap_loss"] = per_tap_loss
    per_tap_loss_p = resolved.get("per_tap_loss_p", 0.0)
    if (
        not isinstance(per_tap_loss_p, (int, float))
        or isinstance(per_tap_loss_p, bool)
        or not math.isfinite(per_tap_loss_p)
    ):
        raise ValueError(
            "dflash_config.ctxkv_head.per_tap_loss_p must be a finite number."
        )
    resolved["per_tap_loss_p"] = float(per_tap_loss_p)
    per_tap_norms = resolved.get("per_tap_norms", [1.0] * 5)
    if (
        not isinstance(per_tap_norms, (list, tuple))
        or len(per_tap_norms) != 5
        or any(
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
            or value <= 0
            for value in per_tap_norms
        )
    ):
        raise ValueError(
            "dflash_config.ctxkv_head.per_tap_norms must contain five finite positive numbers."
        )
    resolved["per_tap_norms"] = [float(value) for value in per_tap_norms]

    resolved.setdefault("consumption_weight", 0.0)
    for key in ("smooth_l1_weight", "cosine_lambda", "projection_weight", "consumption_weight"):
        value = resolved.get(key)
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
            or value < 0
        ):
            raise ValueError(
                f"dflash_config.ctxkv_head.{key} must be a finite non-negative number."
            )
        resolved[key] = float(value)

    if resolved.get("consumption_space_loss", False):
        if resolved.get("a2_aggregation") != "eng00-v1":
            raise ValueError(
                "ctxkv consumption loss requires a2_aggregation='eng00-v1'."
            )
        stats_path = resolved.get("consumption_stats_path")
        if not isinstance(stats_path, str) or not stats_path:
            raise ValueError(
                "ctxkv consumption loss requires consumption_stats_path."
            )
        ridge_basis = resolved.get("ridge_basis")
        if (
            not isinstance(ridge_basis, (int, float))
            or isinstance(ridge_basis, bool)
            or not math.isfinite(ridge_basis)
            or ridge_basis < 0
        ):
            raise ValueError(
                "dflash_config.ctxkv_head.ridge_basis must be a finite "
                "non-negative number."
            )
        resolved["ridge_basis"] = float(ridge_basis)

    distribution = resolved.get("g_distribution")
    if "g_mass_convention" in resolved:
        if distribution is not None:
            raise ValueError(
                "ctxkv H5 g-mass must be derived from raw histograms, not supplied "
                "alongside g_distribution."
            )
        distribution = _resolve_h5_g_mass(resolved)
    if not isinstance(distribution, dict) or not distribution:
        raise ValueError(
            "dflash_config.ctxkv_head.g_distribution must be a non-empty dictionary."
        )
    normalized_distribution = {}
    for raw_g, raw_probability in distribution.items():
        try:
            g = int(raw_g)
        except (TypeError, ValueError) as exc:
            raise ValueError("ctxkv_head g values must be positive integers.") from exc
        if isinstance(raw_g, float) and raw_g != g:
            raise ValueError("ctxkv_head g values must be positive integers.")
        if g <= 0 or str(g) != str(raw_g):
            raise ValueError(
                "ctxkv_head g values must be positive integers; g=0 is forbidden."
            )
        if (
            not isinstance(raw_probability, (int, float))
            or isinstance(raw_probability, bool)
            or not math.isfinite(raw_probability)
            or raw_probability < 0
        ):
            raise ValueError(
                "ctxkv_head g probabilities must be finite and non-negative."
            )
        normalized_distribution[str(g)] = float(raw_probability)
    if not math.isclose(sum(normalized_distribution.values()), 1.0, abs_tol=1e-6):
        raise ValueError("ctxkv_head g probabilities must sum to 1.0.")

    resolved["g_distribution"] = normalized_distribution
    if per_tap_loss and mode not in H2H_MODES:
        raise ValueError("per_tap_loss is only defined for H-space ctxkv modes.")
    if mode in H2H_MODES and resolved["projection_weight"] != 0.0:
        raise ValueError(
            "ctxkv h2h requires projection_weight=0 so the training loss is "
            "strictly H-space"
        )
    if mode in H2H_MODES:
        resolved.update(
            {
                "input_space": "tapped_hidden_concat",
                "output_space": "tapped_hidden_concat",
                "loss_space": "tapped_hidden_concat",
                "read_space": "post_hidden_norm",
            }
        )
    else:
        resolved["output_space"] = "post_hidden_norm"
    if mode == "c1_tap_aligned_h2h":
        # D1 is the one flag that changes it: a query may then also read the LOWER tap's
        # predictions for earlier window slots -- never its own layer's, which is D2.
        resolved["query_interaction"] = (
            "causal_over_tokens"
            if resolved.get("window_causal_over_tokens", False)
            else "self_only_parallel"
        )
    else:
        resolved["query_interaction"] = (
            "self_only_parallel"
            if mode in {"l1_attention", "l1_attention_h2h", "l1_attention_mlp"}
            else "none_prefix_blind"
        )
    validate(resolved, dflash_config)
    return resolved

def resolve_ctxkv_head_config(config, *, training=False, validate=None) -> dict:
    """Resolve the C1 inference configuration; training metadata stays inert."""
    if training:
        return _resolve_training_config(config, validate)
    raw = getattr(config, "dflash_config", {}).get("ctxkv_head")
    if not isinstance(raw, dict) or raw.get("mode") != "c1_tap_aligned_h2h":
        raise ValueError("C1 requires ctxkv_head.mode=c1_tap_aligned_h2h")
    if raw.get("prefix_kv_source") is not None:
        raise ValueError("ctxkv_head.prefix_kv_source requires target cache K/V, which serve cannot feed")
    # The published config carries attention_init as a training provenance claim.
    # Verify its one supported value here so it cannot pass through unread.
    if "attention_init" in raw and raw["attention_init"] != "target_layer_plus_one":
        raise ValueError("ctxkv_head.attention_init must be target_layer_plus_one")
    served_keys = {"mode", "attention_init", "smooth_l1_weight", "cosine_lambda",
                   "projection_weight", "g_distribution", "per_tap_width_multiplier",
                   "layers_per_stream", "stream_depths", "ffn_source", "window_causal_over_tokens",
                   "extra_block_attention_init", "block1_attention_init", "distance_weight_tilt",
                   "target_text_config", "hybrid_attention_source"}
    unread = set(raw) - served_keys
    if unread:
        raise ValueError(f"ctxkv_head carries keys serve does not read: {sorted(unread)}")
    if "target_text_config" in raw:
        target = raw["target_text_config"]
        if (not isinstance(target, dict) or target.get("model_type") != "qwen3_5_text"
                or raw.get("hybrid_attention_source") != "nearest_full_at_or_below"):
            raise ValueError("qwen3_5_text requires hybrid_attention_source=nearest_full_at_or_below")
        for key in ("hidden_size", "intermediate_size", "num_attention_heads",
                    "num_key_value_heads", "head_dim", "num_hidden_layers",
                    "full_attention_interval"):
            value = target.get(key)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"target_text_config.{key} must be a positive integer")
        if target["hidden_size"] != config.hidden_size:
            raise ValueError("target and drafter hidden sizes must match for tap-aligned H-to-H")
        if target["num_attention_heads"] % target["num_key_value_heads"]:
            raise ValueError("target q heads must be divisible by kv heads")
        if not isinstance(target.get("attn_output_gate"), bool):
            raise ValueError("target_text_config.attn_output_gate must be boolean")
        rope = target.get("rope_parameters")
        if not isinstance(rope, dict) or rope.get("rope_type") != "default":
            raise ValueError("target_text_config requires default rope_parameters")
        theta, factor = rope.get("rope_theta"), rope.get("partial_rotary_factor", 1.0)
        if (not isinstance(theta, (int, float)) or isinstance(theta, bool)
                or not math.isfinite(theta) or theta <= 0):
            raise ValueError("target rope_theta must be finite and positive")
        if (not isinstance(factor, (int, float)) or isinstance(factor, bool)
                or not math.isfinite(factor) or not 0 < factor <= 1
                or target["head_dim"] * factor != int(target["head_dim"] * factor)
                or int(target["head_dim"] * factor) < 2
                or int(target["head_dim"] * factor) % 2):
            raise ValueError("target partial rotary must select a positive even head width")
        if ("partial_rotary_factor" in target
                and target["partial_rotary_factor"] != factor):
            raise ValueError("target partial_rotary_factor disagrees with rope_parameters")
        if target.get("hidden_act", "silu") != "silu":
            raise ValueError("target C1 requires silu")
        eps = target.get("rms_norm_eps", 1e-6)
        if (not isinstance(eps, (int, float)) or isinstance(eps, bool)
                or not math.isfinite(eps) or eps <= 0):
            raise ValueError("target rms_norm_eps must be finite and positive")
        if not isinstance(target.get("attention_bias", False), bool):
            raise ValueError("target attention_bias must be boolean")
    elif "hybrid_attention_source" in raw:
        raise ValueError("hybrid_attention_source requires target_text_config")
    # These three published fields affected weight initialization or training loss only.
    # Validate their supported meanings; the trained tensor bytes carry the effect at serve time.
    if "extra_block_attention_init" in raw and (
            raw["extra_block_attention_init"] != "target_same_layer"
            or raw.get("ffn_source") != "target_init"
            or raw.get("attention_init") != "target_layer_plus_one"):
        raise ValueError("ctxkv_head.extra_block_attention_init requires target_same_layer, "
                         "target_init and target_layer_plus_one")
    if "block1_attention_init" in raw and (
            raw["block1_attention_init"] != "target_same_layer"
            or raw.get("extra_block_attention_init") != "target_same_layer"):
        raise ValueError("ctxkv_head.block1_attention_init requires target_same_layer "
                         "extra_block_attention_init")
    if "distance_weight_tilt" in raw and (
            not isinstance(raw["distance_weight_tilt"], (int, float))
            or isinstance(raw["distance_weight_tilt"], bool)
            or not math.isfinite(raw["distance_weight_tilt"])
            or not 0 < raw["distance_weight_tilt"] <= 1):
        raise ValueError("ctxkv_head.distance_weight_tilt must be in (0, 1]")
    resolved = dict(raw)
    for key in ("smooth_l1_weight", "cosine_lambda", "projection_weight"):
        value = resolved.get(key)
        if (not isinstance(value, (int, float)) or isinstance(value, bool)
                or not math.isfinite(value) or value < 0):
            raise ValueError(f"ctxkv_head.{key} must be finite and non-negative")
        resolved[key] = float(value)
    if resolved["projection_weight"] != 0.0:
        raise ValueError("C1 h2h requires projection_weight=0")
    distribution = resolved.get("g_distribution")
    if not isinstance(distribution, dict) or not distribution:
        raise ValueError("ctxkv_head.g_distribution must be non-empty")
    normalized = {}
    for raw_g, probability in distribution.items():
        g = int(raw_g)
        if g <= 0 or str(g) != str(raw_g):
            raise ValueError("ctxkv g values must be positive integers")
        if (not isinstance(probability, (int, float)) or isinstance(probability, bool)
                or not math.isfinite(probability) or probability < 0):
            raise ValueError("ctxkv g probabilities must be finite and non-negative")
        normalized[str(g)] = float(probability)
    if not math.isclose(sum(normalized.values()), 1.0, abs_tol=1e-6):
        raise ValueError("ctxkv g probabilities must sum to one")
    resolved.update(g_distribution=normalized, input_space="tapped_hidden_concat",
        output_space="tapped_hidden_concat", loss_space="tapped_hidden_concat",
        read_space="post_hidden_norm", query_interaction="causal_over_tokens"
        if resolved.get("window_causal_over_tokens", False) else "self_only_parallel")
    return resolved

class RMSNorm(nn.Module):
    """Qwen3 arithmetic: fp32 statistics, weight multiply after the input-dtype cast."""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.variance_epsilon = eps

    def forward(self, x):
        value = x.float()
        value = value * torch.rsqrt(value.pow(2).mean(-1, keepdim=True) + self.variance_epsilon)
        return self.weight * value.to(x.dtype)


class MLP(nn.Module):
    """The bias-free gated MLP, with checkpoint-compatible projection names."""

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = {"silu": nn.SiLU(), "relu": F.relu, "gelu": F.gelu}[config.hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class RotaryEmbedding(nn.Module):
    """Torch-only text RoPE; cosine and sine are computed in fp32."""

    def __init__(self, config):
        super().__init__()
        scaling = getattr(config, "rope_scaling", None) or {}
        rope_type = scaling.get("rope_type", scaling.get("type", "default"))
        if rope_type not in {"default", "linear"}:
            raise ValueError(f"unsupported adapter rope_type {rope_type!r}")
        width = int(getattr(config, "head_dim", config.hidden_size // config.num_attention_heads))
        inv_freq = 1.0 / (config.rope_theta ** (torch.arange(0, width, 2).float() / width))
        if rope_type == "linear":
            inv_freq = inv_freq / scaling["factor"]
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids):
        inv_freq = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        positions = position_ids[:, None, :].float()
        device_type = x.device.type if x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq.float() @ positions.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos, sin = emb.cos(), emb.sin()
        return cos.to(x.dtype), sin.to(x.dtype)


def _is_block_mask(mask):
    try:
        from torch.nn.attention.flex_attention import BlockMask
    except ImportError:
        return False

    return isinstance(mask, BlockMask)


def _attention(q, k, v, *, attention_mask, scale):
    return F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask, scale=scale)


class HybridC1RMSNorm(nn.Module):
    """Qwen3.5's offset weight and fp32 multiply, including the final dtype cast."""

    def __init__(self, dim, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(dim))
        self.variance_epsilon = eps

    def forward(self, x):
        value = x.float()
        value = value * torch.rsqrt(value.pow(2).mean(-1, keepdim=True) + self.variance_epsilon)
        return (value * (1.0 + self.weight.float())).to(x.dtype)

def _c1_rotary(x, cos, sin):
    """Rotate only the C1 query or key, preserving the pinned L1 helper."""
    cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
    if cos.shape[-1] != x.shape[-1]:
        width = cos.shape[-1]
        return torch.cat((_c1_rotary(x[..., :width], cos.squeeze(1), sin.squeeze(1)),
                          x[..., width:]), dim=-1)
    half = x.shape[-1] // 2
    rotated = torch.cat((-x[..., half:], x[..., :half]), dim=-1)
    return x * cos + rotated * sin

def _splice_prefix_kv(k, v, prefix_kv, prefix_count):
    """Replace the prefix part of a block's (post-norm, post-RoPE) keys and values.

    The capture covers the WHOLE sequence; this block attends its anchor's LEADING prefix
    [0, prefix_count). ctxkv_dense_forward builds prefix_h = raw_target_hidden[row, :anchor] with
    prefix_position_ids = arange(anchor), and layer_kv applies RoPE at positions 0..S-1, so row i
    of the capture IS position i and the leading slice is positionally exact -- the same slice the
    fidelity reader makes per anchor. Only a capture too SHORT is an error (a stale or wrong-batch
    one); a longer capture is the normal case.
    """
    target_k, target_v = prefix_kv
    if target_k.shape[2] < prefix_count:
        raise RuntimeError(
            f"FREE_A_PRIME_PREFIX_KV_TOO_SHORT: the capture has {target_k.shape[2]} positions, the "
            f"block prefix needs {prefix_count}. A capture from a shorter batch reached this step."
        )
    target_k, target_v = target_k[:, :, :prefix_count], target_v[:, :, :prefix_count]
    if target_k.shape[0] != k.shape[0]:
        raise RuntimeError(
            f"FREE_A_PRIME_PREFIX_KV_BATCH: the capture holds {target_k.shape[0]} rows and this "
            f"block was given {k.shape[0]}. ctxkv_dense_forward takes one row per sample "
            "(raw_target_hidden[batch_index]); a micro-batch wider than one row needs that row "
            "selected here, which this arm does not yet carry."
        )
    if target_k.shape[1] != k.shape[1] or target_k.shape[3] != k.shape[3]:
        raise RuntimeError(
            f"FREE_A_PRIME_PREFIX_KV_SHAPE prefix_kv {tuple(target_k.shape)} does not fit keys "
            f"{tuple(k.shape)}"
        )
    return (
        torch.cat([target_k.to(k.dtype), k[:, :, prefix_count:]], dim=2),
        torch.cat([target_v.to(v.dtype), v[:, :, prefix_count:]], dim=2),
    )

class CtxKVC1Stream(nn.Module):
    """One tap-aligned stream (arm C1): a transformer block in ``d_h`` that reads only
    its own tap's prefix.

    ``u`` is the lower tap's prediction for the same window slot (for the first tap, the
    token embedding). The block retrieves from ``h^L_{1..t}`` -- the SAME layer's prefix
    taps and nothing else -- then adds an MLP that has to compress the eight target
    layers between this tap and the one below it.

        a = Attn(norm(u); K,V from h^L_prefix)
        s = u + a
        z = s + MLP(norm(s))

    No offset embedding: RoPE at the query's absolute position carries the offset.
    """

    def __init__(self, config: object, *, mlp_intermediate: Optional[int] = None,
                 target_text_config=None, ffn_source=None, attention_fn=None):
        super().__init__()
        self._attention_fn = attention_fn or _attention
        self.hybrid_target = target_text_config is not None
        self.attn_output_gate = bool((target_text_config or {}).get("attn_output_gate", False))
        rotary_config = config
        norm_class = RMSNorm
        if self.hybrid_target:
            config = copy.deepcopy(config)
            for key in ("hidden_size", "intermediate_size", "num_attention_heads",
                        "num_key_value_heads", "head_dim", "rms_norm_eps",
                        "hidden_act", "attention_bias"):
                setattr(config, key, target_text_config.get(key, getattr(config, key)))
            norm_class = HybridC1RMSNorm
            # Text positions are identical in all three MRoPE axes. Their interleave
            # therefore equals ordinary RoPE on just the selected rotary channels.
            rotary_config = copy.deepcopy(config)
            rope = target_text_config["rope_parameters"]
            rotary_config.head_dim = int(config.head_dim * rope.get("partial_rotary_factor", 1.0))
            rotary_config.rope_theta = rope["rope_theta"]
            rotary_config.rope_scaling = None
            if hasattr(rotary_config, "rope_parameters"):
                rotary_config.rope_parameters = dict(rope_type="default", rope_theta=rope["rope_theta"])
        self.hidden_size = int(config.hidden_size)
        self.num_heads = int(config.num_attention_heads)
        self.num_key_value_heads = int(config.num_key_value_heads)
        self.head_dim = int(
            getattr(config, "head_dim", self.hidden_size // self.num_heads)
        )
        if self.num_heads % self.num_key_value_heads != 0:
            raise ValueError("ctxkv c1 q heads must be divisible by kv heads.")
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        bias = bool(config.attention_bias)

        self.input_norm = norm_class(self.hidden_size, eps=config.rms_norm_eps)
        # The key/value side has its OWN pre-norm because it has its own source. A Qwen3
        # block draws q, k and v all from LN(x); this stream draws q from u (the tap
        # below) and k/v from h^L (its own tap's prefix), so "pre-norms as in Qwen3"
        # requires one norm per source, not one norm. Two consequences, either of which
        # is sufficient on its own: without it a deep tap feeds k_proj a tensor whose RMS
        # is far larger than a shallow tap's, so the five streams train at wildly
        # different input scales; and the target's layer-(L+1) W_k/W_v were trained on
        # LN(h^L), so transferring them into an un-normalised path (arm C1b) would be a
        # half-load -- shape-legal and miscalibrated.
        self.kv_norm = norm_class(self.hidden_size, eps=config.rms_norm_eps)
        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim * (2 if self.attn_output_gate else 1), bias=bias
        )
        self.k_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=bias
        )
        self.v_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=bias
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=bias
        )
        self.q_norm = norm_class(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = norm_class(self.head_dim, eps=config.rms_norm_eps)
        self.rotary_emb = RotaryEmbedding(rotary_config)
        self.post_attention_norm = norm_class(
            self.hidden_size, eps=config.rms_norm_eps
        )
        mlp_config = copy.deepcopy(config)
        if mlp_intermediate is not None:
            mlp_config.intermediate_size = int(mlp_intermediate)
        self.mlp = MLP(mlp_config)
        if ffn_source is not None:
            if ffn_source not in FFN_SOURCES:
                raise ValueError(f"unknown ffn_source {ffn_source!r}")
            if ffn_source in FROZEN_FFN_SOURCES:
                for piece, _ in TARGET_FFN_PIECES:
                    _freeze_weight_as_target_buffer(self.get_submodule(piece))
            self.ffn_source = ffn_source

    def project_kv(self, source):
        packed = getattr(self, "_fused_kv", None)
        if packed is None:
            return self.k_proj(source), self.v_proj(source)
        weight, bias, widths = packed
        return F.linear(source, weight, bias).split(widths, dim=-1)

    def forward(
        self,
        *,
        u: torch.Tensor,
        prefix_tap: torch.Tensor,
        prefix_position_ids: torch.LongTensor,
        query_position_ids: torch.LongTensor,
        window_keys: Optional[torch.Tensor] = None,
        window_position_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        prefix_kv: Optional[tuple] = None,
    ) -> torch.Tensor:
        batch_size, query_count, _ = u.shape
        normed = self.input_norm(u)

        # Keys and values come from this tap's prefix. D1 appends the lower tap's
        # predictions for earlier window slots; C1 passes nothing here, so a query can
        # only ever see the prefix.
        if window_keys is None:
            key_source = self.kv_norm(prefix_tap)
            key_position_ids = prefix_position_ids
        else:
            key_source = self.kv_norm(torch.cat([prefix_tap, window_keys], dim=1))
            key_position_ids = torch.cat(
                [prefix_position_ids, window_position_ids], dim=1
            )
        key_count = key_source.shape[1]

        q = self.q_proj(normed).view(
            batch_size, query_count, self.num_heads, self.head_dim * (2 if self.attn_output_gate else 1)
        )
        if self.attn_output_gate:
            q, gate = q.chunk(2, dim=-1)
            gate = gate.reshape(batch_size, query_count, -1)
        projected_key, projected_value = self.project_kv(key_source)
        k = projected_key.view(
            batch_size, key_count, self.num_key_value_heads, self.head_dim
        )
        v = projected_value.view(
            batch_size, key_count, self.num_key_value_heads, self.head_dim
        )
        q = self.q_norm(q).transpose(1, 2)
        k = self.k_norm(k).transpose(1, 2)
        v = v.transpose(1, 2)

        cos_q, sin_q = self.rotary_emb(normed, query_position_ids)
        cos_k, sin_k = self.rotary_emb(key_source, key_position_ids)
        q = _c1_rotary(q, cos_q, sin_q)
        k = _c1_rotary(k, cos_k, sin_k)
        if prefix_kv is not None:
            k, v = _splice_prefix_kv(k, v, prefix_kv, prefix_tap.shape[1])
        if self.num_key_value_groups != 1:
            k = k.repeat_interleave(self.num_key_value_groups, dim=1)
            v = v.repeat_interleave(self.num_key_value_groups, dim=1)

        attended = self._attention_fn(
            q, k, v, attention_mask=attention_mask, scale=self.scaling
        )
        attended = attended.transpose(1, 2).reshape(
            batch_size, query_count, self.num_heads * self.head_dim
        )
        if self.attn_output_gate:
            attended = attended * torch.sigmoid(gate)
        state = u + self.o_proj(attended)
        return state + self.mlp(self.post_attention_norm(state))

class CtxKVC1Head(nn.Module):
    """Arm C1 -- tap-aligned, bottom-up.

    One stream per tap, ascending. Each stream takes the tap below it as its query
    content and retrieves from its own tap's prefix; the five outputs concatenate to the
    20480-d H the drafter's converter consumes. There are no input or output projections:
    the streams work in tap space throughout.
    """

    target_text_config = None
    ffn_frozen = False
    target_ffn_report = None
    checkpoint_blocks = 0

    def __init__(self, config, *, resolved_config=None, training=False, attention_fn=None):
        super().__init__()
        self.resolved_config = (resolve_ctxkv_head_config(config)
                                if resolved_config is None else resolved_config)
        self.target_text_config = None if training else self.resolved_config.get("target_text_config")
        self.hidden_size = int(config.hidden_size)
        dflash_config = getattr(config, "dflash_config", None) or {}
        self.target_layer_ids = list(dflash_config.get("target_layer_ids", []))
        if not self.target_layer_ids:
            raise ValueError("ctxkv c1 requires dflash_config.target_layer_ids.")
        self.hspace_dim = len(self.target_layer_ids) * self.hidden_size
        widths = self.resolved_config.get("per_tap_width_multiplier")
        if widths is not None and len(widths) != len(self.target_layer_ids):
            raise ValueError(
                "per_tap_width_multiplier must have one entry per tap "
                f"({len(self.target_layer_ids)})."
            )
        base_intermediate = int((self.target_text_config or {}).get("intermediate_size", config.intermediate_size))
        if training:
            self.mlp_intermediates = [
                round(base_intermediate * (1.0 if widths is None else float(widths[index])))
                for index in range(len(self.target_layer_ids))
            ]
            if any(intermediate <= 0 for intermediate in self.mlp_intermediates):
                raise ValueError("per_tap_width_multiplier must yield positive integer widths.")
        else:
            self.mlp_intermediates = [
                base_intermediate * (1 if widths is None else int(widths[index]))
                for index in range(len(self.target_layer_ids))
            ]
        stream_kwargs = {"target_text_config": self.target_text_config,
                         "attention_fn": attention_fn}
        ffn_source = self.resolved_config.get("ffn_source")
        if training and ffn_source is not None:
            if widths is not None or any(
                    intermediate != base_intermediate for intermediate in self.mlp_intermediates):
                raise ValueError(
                    f"ffn_source={ffn_source} requires the target's own MLP width; "
                    "per_tap_width_multiplier must be absent.")
            stream_kwargs["ffn_source"] = ffn_source
        self.streams = nn.ModuleList(
            CtxKVC1Stream(config, mlp_intermediate=intermediate, **stream_kwargs)
            for intermediate in self.mlp_intermediates
        )
        # C5's depth form. Kept ALONGSIDE `streams` rather than nesting them, so the
        # state-dict names stay `streams.N.<piece>` -- C1b's loader addresses those, and
        # C1 is measured under them. Empty at depth 1: no parameters, no RNG draws.
        layers_per_stream = int(self.resolved_config.get("layers_per_stream", 1))
        if layers_per_stream < 1:
            raise ValueError("ctxkv_head.layers_per_stream must be >= 1")
        self.layers_per_stream = layers_per_stream
        # servingw33 FFNINIT: an optional per-tap depth list (one entry per tap) replaces the
        # uniform layers_per_stream; absent, every stream is layers_per_stream deep.
        stream_depths = self.resolved_config.get("stream_depths")
        if stream_depths is None:
            stream_depths = [layers_per_stream] * len(self.target_layer_ids)
        if training and len(stream_depths) != len(self.target_layer_ids):
            raise ValueError("ctxkv_head.stream_depths must have one entry per tap "
                             f"({len(self.target_layer_ids)}).")
        if not training and (len(stream_depths) != len(self.target_layer_ids) or any(
                int(depth) < 1 for depth in stream_depths)):
            raise ValueError("ctxkv_head.stream_depths must be one positive depth per tap")
        self.stream_depths = [int(depth) for depth in stream_depths]
        # target_init FFNs are ordinary trained parameters saved in head.safetensors (the
        # strict load proves it); target_frozen FFNs are not saved and are not served.
        if not training and ffn_source not in (None, "target_init"):
            raise ValueError("C1 serving supports ffn_source absent or target_init only")
        self.extra_stream_layers = nn.ModuleList(
            nn.ModuleList(
                CtxKVC1Stream(config, mlp_intermediate=intermediate, **stream_kwargs)
                for _ in range(self.stream_depths[index] - 1)
            )
            for index, intermediate in enumerate(self.mlp_intermediates)
        )
        # Kept so the h2h wrapper's shared plumbing resolves; C1 owns no projections.
        self.input_projection = None
        self.output_projection = None
        self.residual_parameterization = False
        self.offset_conditioning = False

    def forward(
        self,
        *,
        token_embeddings: torch.Tensor,
        prefix_position_ids: torch.LongTensor,
        query_position_ids: torch.LongTensor,
        h_prev: Optional[torch.Tensor] = None,
        prefix_h: Optional[torch.Tensor] = None,
        c_prev: Optional[torch.Tensor] = None,
        prefix_c: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        prefix_kv: Optional[list] = None,
    ) -> torch.Tensor:
        if self.resolved_config.get("prefix_kv_source") is not None and prefix_kv is None:
            # A consumer that holds this head without wiring the target-K/V feed (the stock
            # fidelity reader, the stock tau rail) must be refused here, never served on the tap.
            raise RuntimeError(
                "FREE_A_PRIME_NO_PREFIX_KV: this head was trained with "
                "ctxkv_head.prefix_kv_source=target_layer_cache, so its prefix keys and values "
                "must come from the target's own layer-m cache. The caller passed none; serving "
                "it on the tap would silently read the wrong keys."
            )
        if self.ffn_frozen and self.target_ffn_report is None:
            raise RuntimeError(
                "ffn_source=target_frozen: the FFNs were never loaded from the target; "
                "call load_target_ffns(<target path>) before any forward."
            )
        if prefix_h is None:
            raise ValueError("ctxkv c1 is an h2h arm and requires prefix_h.")
        if c_prev is not None or prefix_c is not None:
            raise ValueError("ctxkv c1 forbids c-space inputs.")
        if prefix_h.shape[-1] != self.hspace_dim:
            raise ValueError(
                f"prefix_h last dimension must be {self.hspace_dim}, "
                f"got {prefix_h.shape[-1]}."
            )
        window_causal = bool(
            self.resolved_config.get("window_causal_over_tokens", False)
        )

        # The caller's mask is [batch, 1, queries, prefix + queries]. Its prefix half
        # carries the per-query block anchor AND the padding mask, neither of which can
        # be rebuilt from position ids, so it is sliced rather than reconstructed. Its
        # query half is v0's self-only identity under C1 -- meaningless here, because C1's
        # keys are the prefix only -- and under D1 it is the driver's same-block,
        # strictly-earlier mask, which is exactly what the window keys need.
        prefix_count = prefix_h.shape[1]
        if attention_mask is None:
            stream_mask = None
        elif window_causal:
            stream_mask = attention_mask
        elif _is_block_mask(attention_mask):
            raise ValueError("a flex BlockMask is defined for window-causal heads only")
        else:
            stream_mask = attention_mask[..., :prefix_count]

        u = token_embeddings
        outputs = []
        executed = 0

        def run_block(block, **kwargs):
            # Same block, same inputs: checkpointing only moves when its activations exist,
            # so the math is the forward's own. It trades one recomputed forward per block
            # for the memory a B200 lacks on the longest rows.
            nonlocal executed
            executed += 1
            if self.training and executed <= self.checkpoint_blocks:
                return torch.utils.checkpoint.checkpoint(block, use_reentrant=False, **kwargs)
            return block(**kwargs)

        for index, stream in enumerate(self.streams):
            start = index * self.hidden_size
            prefix_tap = prefix_h[..., start : start + self.hidden_size]
            # D1: the window keys are the LOWER tap's predictions for the other slots --
            # never this stream's own, which would be D2 and serial. They are fixed for
            # the whole tap, so a C5 depth stack re-reads the same lower-tap window
            # rather than its own partial outputs.
            window = u if window_causal else None
            # free_a_prime: one (K, V) per BLOCK, in ffn_layer_map order -- [index][0] is the
            # stream's own block, [index][j + 1] its j-th extra block. None on the base path.
            stream_kv = prefix_kv[index] if prefix_kv is not None else None
            state = run_block(
                stream,
                u=u,
                prefix_tap=prefix_tap,
                prefix_position_ids=prefix_position_ids,
                query_position_ids=query_position_ids,
                window_keys=window,
                window_position_ids=query_position_ids if window_causal else None,
                attention_mask=stream_mask,
                prefix_kv=stream_kv[0] if stream_kv is not None else None,
            )
            for extra_index, extra in enumerate(self.extra_stream_layers[index]):
                state = run_block(
                    extra,
                    u=state,
                    prefix_tap=prefix_tap,
                    prefix_position_ids=prefix_position_ids,
                    query_position_ids=query_position_ids,
                    window_keys=window,
                    window_position_ids=query_position_ids if window_causal else None,
                    attention_mask=stream_mask,
                    prefix_kv=stream_kv[extra_index + 1] if stream_kv is not None else None,
                )
            outputs.append(state)
            u = state
        return torch.cat(outputs, dim=-1)

    def final_output_parameters(self) -> dict[str, torch.nn.Parameter]:
        """The last parameterised op on each tap's path -- what output_projection's role
        devolves to when the streams emit tap space directly.

        RESOLVED from the structure, never named. `streams.N.mlp.down_proj` is final only
        while `extra_stream_layers[N]` is empty; under a C5 depth stack the extra layer
        runs AFTER its stream (see forward), so `streams.N` becomes an intermediate and
        naming it would check a mid-stack tensor while reporting that the output was
        checked. A gate that hardcodes the name cannot tell those two apart.
        """
        final: dict[str, torch.nn.Parameter] = {}
        for index, stream in enumerate(self.streams):
            extras = self.extra_stream_layers[index]
            if len(extras) == 0:
                name = f"streams.{index}.mlp.down_proj.weight"
                module = stream
            else:
                name = (
                    f"extra_stream_layers.{index}.{len(extras) - 1}"
                    ".mlp.down_proj.weight"
                )
                module = extras[len(extras) - 1]
            final[name] = module.mlp.down_proj.weight
        return final

    def parameter_count_split(self) -> dict[str, int]:
        per_stream = [
            sum(parameter.numel() for parameter in stream.parameters())
            for stream in self.streams
        ]
        # ⚠ `total` must be the WHOLE head. It summed `streams` alone, which under a C5
        # depth stack reported 964,752,640 -- C1's own banked count -- for a head holding
        # 1,929,505,280. A reporter that omits the delta returns the COMPARATOR's known
        # number, so it reads as confirmation rather than as a mismatch. Derive it from
        # self.parameters() so a future sub-module cannot fall out of it again.
        per_extra = [
            sum(
                parameter.numel()
                for layer in extras
                for parameter in layer.parameters()
            )
            for extras in self.extra_stream_layers
        ]
        total = sum(parameter.numel() for parameter in self.parameters())
        assert total == sum(per_stream) + sum(per_extra), (
            "ctxkv c1 parameter split does not account for every parameter: "
            f"{total} vs {sum(per_stream) + sum(per_extra)}"
        )
        return {
            "streams": per_stream,
            "extra_stream_layers": per_extra,
            "layers_per_stream": self.layers_per_stream,
            "total": total,
            "mlp_intermediates": list(self.mlp_intermediates),
        }

def build_adapter_query_mask(
    *,
    prefix_count: int,
    query_count: int,
    dtype: torch.dtype,
    device: torch.device,
    window_causal: bool = False,
) -> torch.Tensor:
    """Allow every query to see the true prefix and only its own query state."""
    prefix = torch.ones((query_count, prefix_count), dtype=torch.bool, device=device)
    if window_causal:
        positions = torch.arange(query_count, device=device)
        own = positions[:, None] > positions[None, :]
    else:
        own = torch.eye(query_count, dtype=torch.bool, device=device)
    allowed = torch.cat([prefix, own], dim=1)
    return torch.zeros(
        (1, 1, query_count, prefix_count + query_count),
        dtype=dtype,
        device=device,
    ).masked_fill(
        allowed.logical_not().unsqueeze(0).unsqueeze(0),
        torch.finfo(dtype).min,
    )

def build_head_attention_mask(
    *,
    head,
    prefix_position_ids: torch.Tensor,
    query_position_ids: torch.Tensor,
    block_anchors: torch.Tensor,
    dtype: torch.dtype,
    device: torch.device,
    prefix_valid: torch.Tensor | None = None,
    query_key_valid: torch.Tensor | None = None,
    query_row_valid: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build the trainer-equivalent prefix-plus-query mask for one read batch."""
    prefix_positions = prefix_position_ids.to(device=device)
    query_positions = query_position_ids.to(device=device)
    anchors = block_anchors.to(device=device)
    if prefix_positions.ndim != 2 or query_positions.ndim != 2:
        raise ValueError("reader position ids must have shape [batch, sequence]")
    if anchors.shape != query_positions.shape:
        raise ValueError("reader block anchors must match query position ids")

    prefix_allowed = prefix_positions.unsqueeze(1) < anchors.unsqueeze(-1)
    if prefix_valid is not None:
        prefix_allowed &= prefix_valid.to(device=device, dtype=torch.bool).unsqueeze(1)

    batch_size, query_count = query_positions.shape
    window_causal = bool(
        head.resolved_config.get("window_causal_over_tokens", False)
    )
    if window_causal:
        same_block = anchors.unsqueeze(-1) == anchors.unsqueeze(1)
        strictly_earlier = query_positions.unsqueeze(-1) > query_positions.unsqueeze(1)
        query_allowed = same_block & strictly_earlier
        if query_key_valid is not None:
            query_allowed &= query_key_valid.to(
                device=device, dtype=torch.bool
            ).unsqueeze(1)
        if query_row_valid is not None:
            query_allowed &= query_row_valid.to(
                device=device, dtype=torch.bool
            ).unsqueeze(-1)
    else:
        query_allowed = torch.eye(
            query_count, dtype=torch.bool, device=device
        ).unsqueeze(0).expand(batch_size, -1, -1)

    allowed = torch.cat([prefix_allowed, query_allowed], dim=-1)
    additive = torch.zeros(
        (batch_size, 1, query_count, allowed.shape[-1]),
        dtype=dtype,
        device=device,
    )
    return additive.masked_fill(
        allowed.logical_not().unsqueeze(1), torch.finfo(dtype).min
    )
