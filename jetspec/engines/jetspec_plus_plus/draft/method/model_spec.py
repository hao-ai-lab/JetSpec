"""Checkpoint model properties shared by method and engine consumers.

No runtime policy (precision, tensor parallelism, kernels or state capacity) is
part of the spec. Head counts describe the complete, unsharded model.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from copy import deepcopy
import json
from typing import Literal


@dataclass(frozen=True)
class AttentionSpec:
    q_heads: int
    kv_heads: int
    head_dim: int
    rotary_dim: int
    output_gate: bool
    norm_offset: int
    # Cast normalized FP32 activations before or after multiplying the weight.
    norm_round: Literal["before_weight", "after_weight"]


@dataclass(frozen=True)
class GDNSpec:
    k_heads: int
    v_heads: int
    k_dim: int
    v_dim: int
    conv_kernel: int
    hidden_act: str


@dataclass(frozen=True)
class ModelSpec:
    num_layers: int
    hidden_size: int
    intermediate_size: int
    vocab_size: int
    rms_norm_eps: float
    max_position_embeddings: int
    rope_theta: float
    rope_scaling: dict | None
    rope_parameters: dict | None
    tie_word_embeddings: bool
    layer_kinds: tuple[Literal["full_attention", "linear_attention"], ...]
    attention: AttentionSpec
    gdn: GDNSpec | None
    taps: tuple[int, ...]
    tail: tuple[int, ...]
    weight_prefix: str

    @property
    def has_state(self) -> bool:
        return any(kind != "full_attention" for kind in self.layer_kinds)

    @classmethod
    def from_configs(cls, target_config: dict, adapter_config: dict | None = None) -> ModelSpec:
        """Describe a checkpoint; without an adapter there are no taps or tail.

        Weight prefixes follow the current dense and Qwen3.5 checkpoint layouts;
        resolving a different layout against a weight index remains a loader job.
        """
        text = target_config.get("text_config", target_config)
        model_type = text.get("model_type", "qwen3")
        if model_type not in {"qwen3", "qwen3_5_text"}:
            raise ValueError(f"unsupported model_type: {model_type}")

        def positive_int(name):
            value = text.get(name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
            return value

        depth = positive_int("num_hidden_layers")
        positive_int("hidden_size")
        positive_int("intermediate_size")
        positive_int("vocab_size")
        heads = positive_int("num_attention_heads")
        kv_heads = text.get("num_key_value_heads", heads)
        if type(kv_heads) is not int or kv_heads <= 0 or heads % kv_heads:
            raise ValueError("attention heads must be divisible by positive KV heads")
        dim = text.get("head_dim") or text["hidden_size"] // heads
        if type(dim) is not int or dim <= 0:
            raise ValueError("head_dim must be a positive integer")
        hybrid = model_type == "qwen3_5_text"
        rope = text.get("rope_parameters") or {}
        rotary_dim = int(dim * rope.get("partial_rotary_factor",
                                       text.get("partial_rotary_factor", 1.0)))
        if not 0 < rotary_dim <= dim or rotary_dim % 2:
            raise ValueError("rotary_dim must be positive, even and no larger than head_dim")
        kinds = text.get("layer_types")
        if kinds is None and not hybrid:
            kinds = ("full_attention",) * depth
        if (not isinstance(kinds, (tuple, list)) or len(kinds) != depth
                or any(kind not in {"full_attention", "linear_attention"} for kind in kinds)):
            raise ValueError("layer_types must name a supported kind for every layer")
        kinds = tuple(kinds)
        gdn = None
        if "linear_attention" in kinds:
            gdn = GDNSpec(positive_int("linear_num_key_heads"),
                          positive_int("linear_num_value_heads"),
                          positive_int("linear_key_head_dim"),
                          positive_int("linear_value_head_dim"),
                          positive_int("linear_conv_kernel_dim"), text.get("hidden_act", "silu"))
            if gdn.v_heads % gdn.k_heads:
                raise ValueError("GDN value heads must be divisible by key heads")
        adapter = (adapter_config or {}).get("dflash_config", adapter_config or {})
        taps = tuple(adapter.get("target_layer_ids", ()))
        if adapter_config is not None and (not taps or any(
                type(index) is not int or not 0 <= index < depth for index in taps)
                or len(set(taps)) != len(taps)):
            raise ValueError("adapter target_layer_ids must be nonempty, unique layer indices")
        return cls(
            num_layers=depth,
            hidden_size=text["hidden_size"],
            intermediate_size=text["intermediate_size"],
            vocab_size=text["vocab_size"],
            rms_norm_eps=text.get("rms_norm_eps", 1e-6),
            max_position_embeddings=text.get("max_position_embeddings", 32768),
            rope_theta=text.get("rope_theta", rope.get("rope_theta", 10000.0)),
            rope_scaling=deepcopy(text.get("rope_scaling")),
            rope_parameters=deepcopy(text.get("rope_parameters")),
            tie_word_embeddings=text.get("tie_word_embeddings", False),
            layer_kinds=kinds,
            # The current hybrid target gates unconditionally.
            attention=AttentionSpec(heads, kv_heads,
                                    dim, rotary_dim, hybrid, int(hybrid),
                                    "after_weight" if hybrid else "before_weight"),
            gdn=gdn,
            taps=taps,
            tail=tuple(range(max(taps) + 1, depth)) if taps else (),
            weight_prefix="model.language_model." if hybrid else "model.",
        )

    def to_json(self) -> str:
        """Stable, sorted JSON for records, independent of config key order."""
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"), allow_nan=False)
