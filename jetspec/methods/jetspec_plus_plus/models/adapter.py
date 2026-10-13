"""Ported from the JetSpec++ reference implementation.

Public API: resolve_ctxkv_head_config(config: Qwen3Config) -> dict;
HybridC1RMSNorm(dim, eps).forward(x) -> Tensor.
build_head_attention_mask(*, head, prefix_position_ids, query_position_ids,
block_anchors, dtype, device, prefix_valid=None, query_key_valid=None,
query_row_valid=None) -> Tensor[batch, 1, queries, prefix + queries].
CtxKVC1Stream(config, *, mlp_intermediate=None, target_text_config=None).
Stream.forward(*, u, prefix_tap, prefix_position_ids, query_position_ids,
window_keys=None, window_position_ids=None, attention_mask=None) -> Tensor.
Stream.project_kv(source) -> tuple[Tensor, Tensor].
CtxKVC1Head(config).forward(*, token_embeddings, prefix_position_ids,
query_position_ids, h_prev=None, prefix_h=None, c_prev=None, prefix_c=None,
attention_mask=None) -> Tensor[batch, queries, taps * hidden_size].
Head.final_output_parameters() -> dict; Head.parameter_count_split() -> dict.
load_adapter(path, device, dtype) -> frozen, eval-mode CtxKVC1Head. The local
converter bank is validated as drafter provenance; the head emits tap space.
"""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from safetensors import safe_open
from safetensors.torch import load_file
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Config, Qwen3MLP, Qwen3RMSNorm, Qwen3RotaryEmbedding, rotate_half,
)

__all__ = ['resolve_ctxkv_head_config', 'HybridC1RMSNorm', 'build_head_attention_mask',
           'CtxKVC1Stream', 'CtxKVC1Head', 'load_adapter']


def resolve_ctxkv_head_config(config: Qwen3Config) -> dict:
    """Resolve the C1 inference configuration; training metadata stays inert."""
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


# Ported from the JetSpec++ reference implementation.
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


def _apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    del position_ids
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_len = q.size(-2)
    q_embed = (q * cos[..., -q_len:, :]) + (rotate_half(q) * sin[..., -q_len:, :])
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def _c1_rotary(x, cos, sin):
    """Rotate the query or key, preserving channels outside the rotary width."""
    cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
    if cos.shape[-1] != x.shape[-1]:
        width = cos.shape[-1]
        return torch.cat((_c1_rotary(x[..., :width], cos.squeeze(1), sin.squeeze(1)),
                          x[..., width:]), dim=-1)
    half = x.shape[-1] // 2
    rotated = torch.cat((-x[..., half:], x[..., :half]), dim=-1)
    return x * cos + rotated * sin


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

    def __init__(self, config: Qwen3Config, *, mlp_intermediate: Optional[int] = None,
                 target_text_config=None):
        super().__init__()
        self.hybrid_target = target_text_config is not None
        self.attn_output_gate = bool((target_text_config or {}).get("attn_output_gate", False))
        rotary_config = config
        norm_class = Qwen3RMSNorm
        if self.hybrid_target:
            config = copy.deepcopy(config)
            for key in ("hidden_size", "intermediate_size", "num_attention_heads",
                        "num_key_value_heads", "head_dim", "rms_norm_eps",
                        "hidden_act", "attention_bias"):
                setattr(config, key, target_text_config.get(key, getattr(config, key)))
            norm_class = HybridC1RMSNorm
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
        self.rotary_emb = Qwen3RotaryEmbedding(rotary_config)
        self.post_attention_norm = norm_class(
            self.hidden_size, eps=config.rms_norm_eps
        )
        mlp_config = copy.deepcopy(config)
        if mlp_intermediate is not None:
            mlp_config.intermediate_size = int(mlp_intermediate)
        self.mlp = Qwen3MLP(mlp_config)

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
    ) -> torch.Tensor:
        batch_size, query_count, _ = u.shape
        normed = self.input_norm(u)

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
        if self.hybrid_target:
            q = _c1_rotary(q, cos_q, sin_q)
            k = _c1_rotary(k, cos_k, sin_k)
        else:
            q, _ = _apply_rotary_pos_emb(q, q, cos_q, sin_q)
            k, _ = _apply_rotary_pos_emb(k, k, cos_k, sin_k)
        if self.num_key_value_groups != 1:
            k = k.repeat_interleave(self.num_key_value_groups, dim=1)
            v = v.repeat_interleave(self.num_key_value_groups, dim=1)

        attended = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attention_mask, scale=self.scaling
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

    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.resolved_config = resolve_ctxkv_head_config(config)
        self.target_text_config = self.resolved_config.get("target_text_config")
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
        self.mlp_intermediates = [
            base_intermediate * (1 if widths is None else int(widths[index]))
            for index in range(len(self.target_layer_ids))
        ]
        self.streams = nn.ModuleList(
            CtxKVC1Stream(config, mlp_intermediate=intermediate, target_text_config=self.target_text_config)
            for intermediate in self.mlp_intermediates
        )
        layers_per_stream = int(self.resolved_config.get("layers_per_stream", 1))
        if layers_per_stream < 1:
            raise ValueError("ctxkv_head.layers_per_stream must be >= 1")
        self.layers_per_stream = layers_per_stream
        stream_depths = self.resolved_config.get("stream_depths")
        if stream_depths is None:
            stream_depths = [layers_per_stream] * len(self.target_layer_ids)
        if len(stream_depths) != len(self.target_layer_ids) or any(
                int(depth) < 1 for depth in stream_depths):
            raise ValueError("ctxkv_head.stream_depths must be one positive depth per tap")
        self.stream_depths = [int(depth) for depth in stream_depths]
        if self.resolved_config.get("ffn_source") not in (None, "target_init"):
            raise ValueError("C1 serving supports ffn_source absent or target_init only")
        self.extra_stream_layers = nn.ModuleList(
            nn.ModuleList(
                CtxKVC1Stream(config, mlp_intermediate=intermediate, target_text_config=self.target_text_config)
                for _ in range(self.stream_depths[index] - 1)
            )
            for index, intermediate in enumerate(self.mlp_intermediates)
        )
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
    ) -> torch.Tensor:
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

        prefix_count = prefix_h.shape[1]
        # Preserve the caller's block-anchor and padding mask on the prefix keys.
        if attention_mask is None:
            stream_mask = None
        elif window_causal:
            stream_mask = attention_mask
        else:
            stream_mask = attention_mask[..., :prefix_count]

        u = token_embeddings
        outputs = []
        for index, stream in enumerate(self.streams):
            start = index * self.hidden_size
            prefix_tap = prefix_h[..., start : start + self.hidden_size]
            # Every depth reads the lower tap's window, rather than its own output.
            window = u if window_causal else None
            state = stream(
                u=u,
                prefix_tap=prefix_tap,
                prefix_position_ids=prefix_position_ids,
                query_position_ids=query_position_ids,
                window_keys=window,
                window_position_ids=query_position_ids if window_causal else None,
                attention_mask=stream_mask,
            )
            for extra in self.extra_stream_layers[index]:
                state = extra(
                    u=state,
                    prefix_tap=prefix_tap,
                    prefix_position_ids=prefix_position_ids,
                    query_position_ids=query_position_ids,
                    window_keys=window,
                    window_position_ids=query_position_ids if window_causal else None,
                    attention_mask=stream_mask,
                )
            outputs.append(state)
            u = state
        return torch.cat(outputs, dim=-1)

    def final_output_parameters(self) -> dict[str, torch.nn.Parameter]:
        """Resolve the final MLP projection on each tap's complete depth stack."""
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


def load_adapter(path, device, dtype) -> CtxKVC1Head:
    """Load a local HF adapter directory, rejecting incompatible checkpoint banks.

    The converter belongs to the drafter, as in the reference loader. It is checked
    here but does not project the head's tap-space output.
    """
    directory = Path(path)
    for name in ('config.json', 'head.safetensors', 'converter.safetensors'):
        if not (directory / name).is_file():
            raise FileNotFoundError(directory / name)
    raw = json.loads((directory / 'config.json').read_text())
    if not isinstance(raw, dict) or raw.get('model_type') != 'qwen3':
        raise ValueError('adapter config must describe qwen3')
    parameters = raw.get('rope_parameters')
    if parameters is not None and not isinstance(parameters, dict):
        raise ValueError('rope_parameters must be an object')
    rope_type = (parameters or {}).get('rope_type', raw.get('rope_type'))
    if rope_type not in (None, 'default'):
        raise ValueError(f'unsupported rope_type {rope_type!r}')
    if any(name in raw or name in (parameters or {})
           for name in ('partial_rotary_factor', 'partial_rotary_ratio', 'partial_rotary_dim')):
        raise ValueError('partial rotary is unsupported for the drafter')
    theta = raw.get('rope_theta', (parameters or {}).get('rope_theta'))
    if (not isinstance(theta, (int, float)) or isinstance(theta, bool)
            or not math.isfinite(theta) or theta <= 0):
        raise ValueError('rope_theta is absent or invalid')
    # from_dict reads no remote code and keeps the reference's nested-theta compatibility.
    config = Qwen3Config.from_dict(raw)
    config.rope_theta = float(theta)
    config._attn_implementation = 'sdpa'
    with safe_open(directory / 'head.safetensors', framework='pt') as checkpoint:
        if (checkpoint.metadata() or {}).get('PREFIX_KV_SOURCE') is not None:
            raise ValueError('head PREFIX_KV_SOURCE requires target cache K/V')
    head = CtxKVC1Head(config)
    converter = load_file(directory / 'converter.safetensors', device='cpu')
    # The training loader requires exactly these two tensors, unlike the reader's subset check.
    if set(converter) != {'fc.weight', 'hidden_norm.weight'}:
        raise ValueError(f'adapter converter keys differ: {sorted(converter)}')
    shapes = {'fc.weight': (head.hidden_size, head.hspace_dim),
              'hidden_norm.weight': (head.hidden_size,)}
    for name, shape in shapes.items():
        if tuple(converter[name].shape) != shape or not converter[name].is_floating_point():
            raise ValueError(f'adapter converter {name} must be floating point with shape {shape}')
    state = load_file(directory / 'head.safetensors', device='cpu')
    if any(not value.is_floating_point() for value in state.values()):
        raise ValueError('adapter head tensors must be floating point')
    head.load_state_dict(state, strict=True)
    head.to(device=device, dtype=dtype)
    head.eval()
    head.requires_grad_(False)
    return head
