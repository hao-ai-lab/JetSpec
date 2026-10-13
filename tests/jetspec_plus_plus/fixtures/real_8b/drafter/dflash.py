import logging
from typing import Callable, Optional

import torch
from torch import nn
from transformers import DynamicCache
from transformers.cache_utils import Cache
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.qwen3.modeling_qwen3 import (
    ALL_ATTENTION_FUNCTIONS,
    FlashAttentionKwargs,
    GradientCheckpointingLayer,
    Qwen3Config,
    Qwen3MLP,
    Qwen3PreTrainedModel,
    Qwen3RMSNorm,
    Qwen3RotaryEmbedding,
    eager_attention_forward,
    rotate_half,
)
from typing_extensions import Tuple, Unpack

logger = logging.getLogger(__name__)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_len = q.size(-2)
    q_embed = (q * cos[..., -q_len:, :]) + (rotate_half(q) * sin[..., -q_len:, :])
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def _to_additive_attention_mask(
    attention_mask: torch.Tensor,
    *,
    query_dtype: torch.dtype,
    device: torch.device,
    key_len: int,
) -> torch.Tensor:
    if attention_mask.ndim == 4:
        attention_mask = attention_mask[:, :, :, :key_len]
    if attention_mask.dtype == torch.bool:
        additive_mask = torch.zeros_like(attention_mask, dtype=query_dtype, device=device)
        return additive_mask.masked_fill(
            attention_mask.logical_not().to(device=device),
            torch.finfo(query_dtype).min,
        )
    return attention_mask.to(device=device, dtype=query_dtype)


def _build_dflash_causal_attention_mask(
    *,
    query: torch.Tensor,
    key: torch.Tensor,
    cached_kv_len: int,
    ctx_len: int,
) -> torch.Tensor:
    q_len = query.shape[-2]
    kv_len = key.shape[-2]
    key_positions = torch.arange(kv_len, device=query.device)
    query_positions = cached_kv_len + ctx_len + torch.arange(q_len, device=query.device)
    can_attend = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
    mask = torch.zeros((1, 1, q_len, kv_len), dtype=query.dtype, device=query.device)
    return mask.masked_fill(can_attend.logical_not().unsqueeze(0).unsqueeze(0), torch.finfo(query.dtype).min)


class Qwen3DFlashAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        self.num_key_value_groups = (
            config.num_attention_heads // config.num_key_value_heads
        )
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        dflash_config = getattr(config, "dflash_config", {}) or {}
        self.is_causal = bool(dflash_config.get("causal_head", False))
        self.q_proj = nn.Linear(
            config.hidden_size,
            config.num_attention_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.sliding_window = (
            config.sliding_window
            if config.layer_types[layer_idx] == "sliding_attention"
            else None
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        target_hidden: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        inject_target_context: bool = True,
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        bsz, q_len = hidden_states.shape[:-1]
        ctx_len = target_hidden.shape[1] if inject_target_context else 0
        is_causal = kwargs.pop("is_causal", None)
        if is_causal is None:
            is_causal = self.is_causal
        q = self.q_proj(hidden_states)
        q = q.view(bsz, q_len, -1, self.head_dim)
        q = self.q_norm(q).transpose(1, 2)
        if inject_target_context:
            k_ctx = self.k_proj(target_hidden)
            k_noise = self.k_proj(hidden_states)
            v_ctx = self.v_proj(target_hidden)
            v_noise = self.v_proj(hidden_states)
            k = torch.cat([k_ctx, k_noise], dim=1)
            v = torch.cat([v_ctx, v_noise], dim=1)
        else:
            k = self.k_proj(hidden_states)
            v = self.v_proj(hidden_states)
        k = k.view(bsz, ctx_len + q_len, -1, self.head_dim)
        v = v.view(bsz, ctx_len + q_len, -1, self.head_dim)
        k = self.k_norm(k).transpose(1, 2)
        v = v.transpose(1, 2)
        cos, sin = position_embeddings
        if not inject_target_context:
            cos = cos[..., -q_len:, :]
            sin = sin[..., -q_len:, :]
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        cached_kv_len = past_key_values.get_seq_length() if past_key_values is not None else 0
        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            k, v = past_key_values.update(k, v, self.layer_idx, cache_kwargs)

        attn_backend = kwargs.pop(
            "attention_backend",
            self.config._attn_implementation,
        )
        use_explicit_dflash_causal_mask = bool(is_causal) and attn_backend in {"eager", "sdpa"}
        if not inject_target_context and attn_backend in {"eager", "sdpa"}:
            block_local_mask = _build_block_local_causal_attention_mask(
                query=q,
                key=k,
                block_size=self.config.block_size,
            )
            if attention_mask is not None:
                if attention_mask.shape[-1] != k.shape[-2]:
                    attention_mask = attention_mask[..., -k.shape[-2] :]
                block_local_mask = block_local_mask + _to_additive_attention_mask(
                    attention_mask,
                    query_dtype=q.dtype,
                    device=q.device,
                    key_len=k.shape[-2],
                )
            attention_mask = block_local_mask
            is_causal = False
        elif use_explicit_dflash_causal_mask:
            dflash_causal_mask = _build_dflash_causal_attention_mask(
                query=q, key=k, cached_kv_len=cached_kv_len, ctx_len=ctx_len,
            )
            if attention_mask is not None:
                dflash_causal_mask = dflash_causal_mask + _to_additive_attention_mask(
                    attention_mask, query_dtype=q.dtype, device=q.device, key_len=k.shape[-2],
                )
            attention_mask = dflash_causal_mask
            is_causal = False

        kwargs["is_causal"] = is_causal

        attn_fn: Callable = eager_attention_forward
        if attn_backend != "eager":
            attn_fn = ALL_ATTENTION_FUNCTIONS[attn_backend]
        attn_output, attn_weights = attn_fn(
            self,
            q,
            k,
            v,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )
        attn_output = attn_output.reshape(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


def _resolve_drafter_mlp(config: Qwen3Config) -> bool:
    dflash_config = getattr(config, "dflash_config", {}) or {}
    raw_value = dflash_config.get("drafter_mlp", True)
    if not isinstance(raw_value, bool):
        raise ValueError(
            f"dflash_config.drafter_mlp must be a boolean, got {raw_value!r}."
        )
    return raw_value


def _resolve_inject_layer_ids(config: Qwen3Config) -> Optional[list[int]]:
    dflash_config = getattr(config, "dflash_config", {}) or {}
    if "inject_layer_ids" not in dflash_config:
        return None
    inject_layer_ids = dflash_config["inject_layer_ids"]
    if not isinstance(inject_layer_ids, list) or any(
        not isinstance(layer_idx, int) or isinstance(layer_idx, bool)
        for layer_idx in inject_layer_ids
    ):
        raise ValueError(
            "dflash_config.inject_layer_ids must be a list of integer layer indices, "
            f"got {inject_layer_ids!r}."
        )
    invalid_ids = [
        layer_idx
        for layer_idx in inject_layer_ids
        if layer_idx < 0 or layer_idx >= config.num_hidden_layers
    ]
    if invalid_ids:
        raise ValueError(
            "dflash_config.inject_layer_ids contains indices outside "
            f"[0, {config.num_hidden_layers}): {invalid_ids}."
        )
    return inject_layer_ids


class Qwen3DFlashDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.drafter_mlp = _resolve_drafter_mlp(config)
        inject_layer_ids = _resolve_inject_layer_ids(config)
        self.inject_target_context = (
            inject_layer_ids is None or layer_idx in inject_layer_ids
        )
        self.self_attn = Qwen3DFlashAttention(config=config, layer_idx=layer_idx)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        target_hidden: Optional[torch.Tensor] = None,
        hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[
            Tuple[torch.Tensor, torch.Tensor]
        ] = None,  # necessary, but kept here for BC
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            target_hidden=target_hidden,
            attention_mask=attention_mask,
            inject_target_context=self.inject_target_context,
            position_ids=position_ids,
            past_key_values=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )[0]
        hidden_states = residual + hidden_states
        if self.drafter_mlp:
            residual = hidden_states
            hidden_states = self.post_attention_layernorm(hidden_states)
            hidden_states = self.mlp(hidden_states)
            hidden_states = residual + hidden_states
        return hidden_states


def build_target_layer_ids(num_target_layers: int, num_draft_layers: int):
    if num_draft_layers == 1:
        return [(num_target_layers // 2)]
    start = 1
    end = num_target_layers - 3
    span = end - start
    target_layer_ids = [
        int(round(start + (i * span) / (num_draft_layers - 1)))
        for i in range(num_draft_layers)
    ]
    return target_layer_ids


def extract_context_feature(
    hidden_states: list[torch.Tensor],
    layer_ids: Optional[list[int]],
) -> torch.Tensor:
    offset = 1
    selected_states = []
    for layer_id in layer_ids:
        selected_states.append(hidden_states[layer_id + offset])
    target_hidden = torch.cat(selected_states, dim=-1)
    return target_hidden


class DFlashDraftModel(Qwen3PreTrainedModel):
    config_class = Qwen3Config
    _no_split_modules = ["Qwen3DFlashDecoderLayer"]

    def __init__(self, config) -> None:
        super().__init__(config)
        self.config = config
        if not hasattr(self.config, "dflash_config") or self.config.dflash_config is None:
            self.config.dflash_config = {}
        self.causal_head = bool(self.config.dflash_config.get("causal_head", False))
        self.drafter_mlp = _resolve_drafter_mlp(self.config)
        self.config.dflash_config["drafter_mlp"] = self.drafter_mlp
        configured_inject_layer_ids = _resolve_inject_layer_ids(self.config)
        self.inject_layer_ids = (
            list(range(config.num_hidden_layers))
            if configured_inject_layer_ids is None
            else list(configured_inject_layer_ids)
        )
        self.has_masked_inject_layers = set(self.inject_layer_ids) != set(
            range(config.num_hidden_layers)
        )
        self.layers = nn.ModuleList(
            [
                Qwen3DFlashDecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        dflash_config = getattr(config, "dflash_config", {}) or {}
        self.target_layer_ids = dflash_config.get(
            "target_layer_ids",
            build_target_layer_ids(config.num_target_layers, config.num_hidden_layers),
        )
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config)
        target_hidden_size = getattr(config, "target_hidden_size", config.hidden_size)
        self.fc = nn.Linear(
            len(self.target_layer_ids) * target_hidden_size,
            config.hidden_size,
            bias=False,
        )
        self.hidden_norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        jacobi_forcing = dflash_config.get("jacobi_forcing", False)
        if not isinstance(jacobi_forcing, bool):
            raise ValueError("dflash_config.jacobi_forcing must be a boolean.")
        self.jacobi_forcing = jacobi_forcing
        self.stock_drafter = dflash_config.get("stock_drafter", False)
        if not isinstance(self.stock_drafter, bool):
            raise ValueError("dflash_config.stock_drafter must be a boolean.")
        if self.stock_drafter and dflash_config.get("fresh_fraction") != 1.0:
            raise ValueError("stock_drafter requires an explicit fresh_fraction=1.0.")
        self.config.dflash_config["stock_drafter"] = self.stock_drafter
        # All-fresh adapter marking contributes only a learned context bias.
        # Stock DFlash omits this bias and its embedding tensor.
        if self.jacobi_forcing and not self.stock_drafter:
            self.adapted_position_embedding = nn.Embedding(
                2,
                len(self.target_layer_ids) * target_hidden_size,
            )
        self.block_size = config.block_size
        self.mask_token_id = dflash_config.get("mask_token_id", None)
        self.post_init()
        if self.jacobi_forcing and not self.stock_drafter:
            nn.init.zeros_(self.adapted_position_embedding.weight)
        if not self.drafter_mlp:
            for layer in self.layers:
                del layer.mlp


    def resolve_causal_head(self, head_type: str = "auto") -> bool:
        if head_type == "auto":
            return bool(self.causal_head)
        if head_type == "bidirectional":
            return False
        if head_type == "causal":
            return True
        raise ValueError(
            f"Unsupported head_type={head_type!r}. Expected one of: auto, bidirectional, causal."
        )

    def forward(
        self,
        position_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        noise_embedding: Optional[torch.Tensor] = None,
        target_hidden: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: bool = False,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        no_inject_attention_mask = kwargs.pop("no_inject_attention_mask", None)
        attention_backend = kwargs.get(
            "attention_backend",
            self.config._attn_implementation,
        )
        hidden_states = noise_embedding
        target_hidden = self.hidden_norm(self.fc(target_hidden))
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        for layer_idx, layer in enumerate(self.layers):
            layer_attention_mask = attention_mask
            if not layer.inject_target_context and no_inject_attention_mask is not None:
                layer_attention_mask = no_inject_attention_mask
            hidden_states = layer(
                hidden_states=hidden_states,
                target_hidden=target_hidden,
                attention_mask=layer_attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                use_cache=use_cache,
                position_embeddings=position_embeddings,
                **kwargs,
            )
        hidden_states = self.norm(hidden_states)
        return hidden_states

