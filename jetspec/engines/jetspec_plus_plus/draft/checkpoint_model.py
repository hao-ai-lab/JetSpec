import hashlib
import importlib
import logging
import sys
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
_token_top_inactive_logged = False


class _StockAdaptedMark(nn.Module):
    """Non-persistent identity mark used by the stock-only local checkpoint model."""

    def __init__(self, width: int, *, dtype: torch.dtype, device: torch.device) -> None:
        super().__init__()
        self.register_buffer(
            "weight", torch.full((2, int(width)), -0.0, dtype=dtype, device=device), persistent=False
        )

    def forward(self, indicator: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.embedding(indicator, self.weight)


def _build_markov_head(config):
    module_name = f"{__package__}.dspark_heads"
    try:
        heads_module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name != module_name:
            raise
        from transformers.utils import cached_file

        module_file = cached_file(config._name_or_path, "dspark_heads.py")
        module_spec = importlib.util.spec_from_file_location(module_name, module_file)
        if module_spec is None or module_spec.loader is None:
            raise ImportError(f"Could not load Markov head module from {module_file}.")
        heads_module = importlib.util.module_from_spec(module_spec)
        sys.modules[module_name] = heads_module
        try:
            module_spec.loader.exec_module(heads_module)
        except Exception:
            sys.modules.pop(module_name, None)
            raise
    return heads_module.build_markov_head(config)


def _log_token_top_inactive_once() -> None:
    global _token_top_inactive_logged
    if _token_top_inactive_logged:
        return
    logger.warning(
        "token_top_layer present but INACTIVE this forward (no prev-token input) "
        "— mask-mode/parallel semantics"
    )
    _token_top_inactive_logged = True




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


def build_token_top_causal_attention_mask(
    block_size: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    query_positions = torch.arange(block_size, device=device).view(block_size, 1)
    key_positions = torch.arange(block_size, device=device).view(1, block_size)
    can_attend = key_positions <= query_positions
    mask = torch.zeros((1, 1, block_size, block_size), dtype=dtype, device=device)
    return mask.masked_fill(can_attend.logical_not().unsqueeze(0).unsqueeze(0), torch.finfo(dtype).min)


def _build_block_local_causal_attention_mask(
    *,
    query: torch.Tensor,
    key: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    q_len = query.shape[-2]
    kv_len = key.shape[-2]
    query_positions = torch.arange(q_len, device=query.device)
    key_positions = torch.arange(kv_len, device=query.device)
    same_block = (
        query_positions.unsqueeze(1) // block_size
        == key_positions.unsqueeze(0) // block_size
    )
    causal = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
    can_attend = same_block & causal
    mask = torch.zeros((1, 1, q_len, kv_len), dtype=query.dtype, device=query.device)
    return mask.masked_fill(
        can_attend.logical_not().unsqueeze(0).unsqueeze(0),
        torch.finfo(query.dtype).min,
    )


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


class Qwen3TokenTopAttention(nn.Module):
    """Block-local causal self-attention for the token-fed top layer."""

    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.config = config
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        self.num_key_value_groups = (
            config.num_attention_heads // config.num_key_value_heads
        )
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
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

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        bsz, q_len = hidden_states.shape[:-1]
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        q = q.view(bsz, q_len, -1, self.head_dim)
        k = k.view(bsz, q_len, -1, self.head_dim)
        v = v.view(bsz, q_len, -1, self.head_dim)
        q = self.q_norm(q).transpose(1, 2)
        k = self.k_norm(k).transpose(1, 2)
        v = v.transpose(1, 2)

        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        attn_output, _ = eager_attention_forward(
            self,
            q,
            k,
            v,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
        )
        attn_output = attn_output.reshape(bsz, q_len, -1)
        return attn_output


class _TokenTopBlock(nn.Module):
    """One token-fed block-local residual adapter."""

    def __init__(self, config: Qwen3Config, token_top_config: dict):
        super().__init__()
        self.hidden_size = int(config.hidden_size)
        self.block_size = int(config.block_size)
        self.width = int(token_top_config.get("width", self.hidden_size))
        self.heads = int(token_top_config.get("heads", config.num_attention_heads))
        self.init = str(token_top_config.get("init", "zero_output"))
        self.fusion_mode = str(token_top_config.get("fusion", "concat")).lower()
        self.use_mlp = bool(token_top_config.get("mlp", False))
        if self.width != self.hidden_size:
            raise ValueError(
                f"token_top_layer.width must match hidden_size={self.hidden_size}, got {self.width}."
            )
        if self.heads != int(config.num_attention_heads):
            raise ValueError(
                f"token_top_layer.heads must match num_attention_heads={config.num_attention_heads}, got {self.heads}."
            )
        if self.fusion_mode not in ("concat", "h_only", "e_only", "add"):
            raise ValueError(
                "Unsupported token_top_layer.fusion="
                f"{self.fusion_mode!r}. Expected one of: concat, h_only, e_only, add."
            )

        if self.fusion_mode == "concat":
            self.fusion = nn.Linear(2 * self.hidden_size, self.hidden_size)
        elif self.fusion_mode == "e_only":
            self.fusion = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.input_layernorm = Qwen3RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = Qwen3TokenTopAttention(config)
        self.output_norm = Qwen3RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        if self.use_mlp:
            self.post_attention_layernorm = Qwen3RMSNorm(
                self.hidden_size, eps=config.rms_norm_eps
            )
            self.mlp = Qwen3MLP(config)
        self.rotary_emb = Qwen3RotaryEmbedding(config)


    def forward(
        self,
        hidden_states: torch.Tensor,
        prev_token_embedding: torch.Tensor,
        position_ids: torch.LongTensor,
        block_keep_mask: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bsz, flat_slots, hidden_size = hidden_states.shape
        if hidden_size != self.hidden_size:
            raise ValueError(
                f"Expected hidden size {self.hidden_size}, got {hidden_size}."
            )
        if attention_mask is None and flat_slots % self.block_size != 0:
            raise ValueError(
                f"Expected flat_slots divisible by block_size={self.block_size}, got {flat_slots}."
            )

        n_blocks = flat_slots // self.block_size if attention_mask is None else 1
        slots_per_block = self.block_size if attention_mask is None else flat_slots
        if block_keep_mask.shape != (bsz, n_blocks):
            raise ValueError(
                "block_keep_mask must match the token-top block layout; "
                f"got {tuple(block_keep_mask.shape)}, expected {(bsz, n_blocks)}."
            )
        if self.fusion_mode == "concat":
            token_top_input = self.fusion(
                torch.cat([hidden_states, prev_token_embedding], dim=-1)
            )
        elif self.fusion_mode == "h_only":
            token_top_input = hidden_states
        elif self.fusion_mode == "e_only":
            token_top_input = self.fusion(prev_token_embedding)
        else:
            token_top_input = hidden_states + prev_token_embedding
        block_hidden = token_top_input.reshape(
            bsz * n_blocks, slots_per_block, self.hidden_size
        )
        block_positions = position_ids.reshape(bsz * n_blocks, slots_per_block)
        position_embeddings = self.rotary_emb(block_hidden, block_positions)
        if attention_mask is None:
            attention_mask = build_token_top_causal_attention_mask(
                self.block_size,
                dtype=block_hidden.dtype,
                device=block_hidden.device,
            )
        else:
            attention_mask = attention_mask.to(
                device=block_hidden.device,
                dtype=block_hidden.dtype,
            )

        attn_input = self.input_layernorm(block_hidden)
        attn_output = self.self_attn(
            hidden_states=attn_input,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
        )
        delta = self.self_attn.o_proj(self.output_norm(attn_output))
        delta = delta.reshape(bsz, n_blocks, slots_per_block, self.hidden_size)
        block_mask = block_keep_mask.reshape(bsz, n_blocks, 1, 1).to(
            dtype=delta.dtype
        )
        delta = delta * block_mask
        hidden_states = hidden_states + delta.reshape(
            bsz, flat_slots, self.hidden_size
        )
        if self.use_mlp:
            block_hidden = hidden_states.reshape(
                bsz * n_blocks, slots_per_block, self.hidden_size
            )
            mlp_delta = self.mlp(self.post_attention_layernorm(block_hidden))
            mlp_delta = mlp_delta.reshape(
                bsz, n_blocks, slots_per_block, self.hidden_size
            )
            mlp_delta = mlp_delta * block_mask
            hidden_states = hidden_states + mlp_delta.reshape(
                bsz, flat_slots, self.hidden_size
            )
        return hidden_states


class TokenTopLayer(_TokenTopBlock):
    """Configurable stack of token-fed block-local residual adapters."""

    def __init__(self, config: Qwen3Config, token_top_config: dict):
        super().__init__(config, token_top_config)
        self.num_layers = token_top_config.get("num_layers", 1)
        if (
            isinstance(self.num_layers, bool)
            or not isinstance(self.num_layers, int)
            or self.num_layers < 1
        ):
            raise ValueError(
                f"token_top_layer.num_layers must be a positive int, got {self.num_layers!r}."
            )
        self.extra_layers = nn.ModuleList(
            _TokenTopBlock(config, token_top_config)
            for _ in range(self.num_layers - 1)
        )


    def forward(
        self,
        hidden_states: torch.Tensor,
        prev_token_embedding: torch.Tensor,
        position_ids: torch.LongTensor,
        block_keep_mask: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hidden_states = super().forward(
            hidden_states=hidden_states,
            prev_token_embedding=prev_token_embedding,
            position_ids=position_ids,
            block_keep_mask=block_keep_mask,
            attention_mask=attention_mask,
        )
        for layer in self.extra_layers:
            hidden_states = layer(
                hidden_states=hidden_states,
                prev_token_embedding=prev_token_embedding,
                position_ids=position_ids,
                block_keep_mask=block_keep_mask,
                attention_mask=attention_mask,
            )
        return hidden_states


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


def _resolve_split_stack_from(config: Qwen3Config) -> Optional[int]:
    dflash_config = getattr(config, "dflash_config", {}) or {}
    if "split_stack_from" not in dflash_config:
        return None
    split_stack_from = dflash_config["split_stack_from"]
    if not isinstance(split_stack_from, int) or isinstance(split_stack_from, bool):
        raise ValueError(
            "dflash_config.split_stack_from must be an integer layer index, "
            f"got {split_stack_from!r}."
        )
    if split_stack_from < 0 or split_stack_from >= config.num_hidden_layers:
        raise ValueError(
            "dflash_config.split_stack_from must be inside "
            f"[0, {config.num_hidden_layers}), got {split_stack_from}."
        )
    return split_stack_from


def _resolve_seam_projection(
    config: Qwen3Config,
    split_stack_from: Optional[int],
) -> bool:
    dflash_config = getattr(config, "dflash_config", {}) or {}
    seam_projection = dflash_config.get("seam_projection", False)
    if not isinstance(seam_projection, bool):
        raise ValueError(
            "dflash_config.seam_projection must be a boolean, "
            f"got {seam_projection!r}."
        )
    if seam_projection and split_stack_from is None:
        raise ValueError(
            "dflash_config.seam_projection=true requires "
            "dflash_config.split_stack_from."
        )
    return seam_projection


def _tensor_sha256(tensor: torch.Tensor) -> str:
    tensor_bytes = (
        tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    )
    return hashlib.sha256(tensor_bytes).hexdigest()


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




class DFlashDraftModel(Qwen3PreTrainedModel):
    config_class = Qwen3Config
    _no_split_modules = ["Qwen3DFlashDecoderLayer", "TokenTopLayer"]

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
        self.split_stack_from = _resolve_split_stack_from(self.config)
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
        if self.jacobi_forcing:
            self.adapted_position_embedding = nn.Embedding(
                2,
                len(self.target_layer_ids) * target_hidden_size,
            )
        self.block_size = config.block_size
        self.mask_token_id = dflash_config.get("mask_token_id", None)
        self.markov_config = self._normalize_markov_config(config)
        self.markov_head = (
            _build_markov_head(config)
            if self.markov_config.get("enabled", False)
            else None
        )
        self.token_top_config = self._normalize_token_top_config(config)
        self.token_top_layer = (
            TokenTopLayer(config, self.token_top_config)
            if self.token_top_config.get("enabled", False)
            else None
        )
        self.seam_projection_enabled = _resolve_seam_projection(
            self.config,
            self.split_stack_from,
        )
        self.seam_projection = (
            nn.Linear(config.hidden_size, config.hidden_size, bias=False)
            if self.seam_projection_enabled
            else None
        )
        seam_projection_trainable = dflash_config.get(
            "seam_projection_trainable", True
        )
        if self.seam_projection_enabled and not isinstance(
            seam_projection_trainable, bool
        ):
            raise ValueError(
                "dflash_config.seam_projection_trainable must be a boolean, "
                f"got {seam_projection_trainable!r}."
            )
        self.seam_projection_trainable = bool(
            self.seam_projection_enabled and seam_projection_trainable
        )
        has_init_seed = "seam_projection_init_seed" in dflash_config
        has_init_sha256 = "seam_projection_init_sha256" in dflash_config
        if self.seam_projection_enabled and has_init_seed != has_init_sha256:
            raise ValueError(
                "dflash_config must carry seam_projection_init_seed and "
                "seam_projection_init_sha256 together."
            )
        self._seam_projection_initialization_is_fresh = bool(
            self.seam_projection_enabled and not has_init_seed
        )
        self.seam_projection_parameter_count = (
            sum(
                parameter.numel()
                for parameter in self.seam_projection.parameters()
            )
            if self.seam_projection_enabled
            else 0
        )
        if self.seam_projection_enabled:
            seam_projection_weight_shape = list(self.seam_projection.weight.shape)
            self.config.dflash_config.update(
                {
                    "seam_projection": True,
                    "seam_projection_form": "linear_token_delta_residual",
                    "seam_projection_in_features": self.seam_projection.in_features,
                    "seam_projection_out_features": self.seam_projection.out_features,
                    "seam_projection_weight_shape": seam_projection_weight_shape,
                    "seam_projection_bias": self.seam_projection.bias is not None,
                    "seam_projection_layer_index": self.split_stack_from,
                    "seam_projection_parameter_count": (
                        self.seam_projection_parameter_count
                    ),
                    "seam_projection_trainable": self.seam_projection_trainable,
                }
            )
        self.compose_markov_token_top = bool(
            self.config.dflash_config.get("compose_markov_token_top", False)
        )
        if self.compose_markov_token_top and (
            self.markov_head is None or self.token_top_layer is None
        ):
            raise ValueError(
                "dflash_config.compose_markov_token_top=true requires both "
                "markov_head and token_top_layer.enabled=true."
            )
        self.config.dflash_config["compose_markov_token_top"] = (
            self.compose_markov_token_top
        )
        self.post_init()
        if self.jacobi_forcing:
            # Preserve the constructor's exact RNG consumption, then discard the retired trainable tensor.
            initialized_mark = self.adapted_position_embedding.weight
            self.adapted_position_embedding = _StockAdaptedMark(
                initialized_mark.shape[1], dtype=initialized_mark.dtype, device=initialized_mark.device
            )
        if self.seam_projection is not None:
            self._record_fresh_seam_projection_initialization()
            self.seam_projection.weight.requires_grad_(
                self.seam_projection_trainable
            )
        if not self.drafter_mlp:
            for layer in self.layers:
                del layer.mlp
        if self.markov_head is not None:
            nn.init.zeros_(self.markov_head.markov_w2.weight)
            if self.markov_config.get("order", 1) == 2:
                self.markov_head.maybe_zero_init_order2()
        if self.token_top_layer is not None:
            self.token_top_layer.maybe_zero_init_output()

    def _record_fresh_seam_projection_initialization(
        self,
        *,
        init_seed: Optional[int] = None,
    ) -> None:
        if (
            not self._seam_projection_initialization_is_fresh
            or self.seam_projection is None
            or self.seam_projection.weight.is_meta
        ):
            return
        if init_seed is None:
            init_seed = torch.initial_seed()
        self.config.dflash_config.update(
            {
                "seam_projection_init_seed": int(init_seed),
                "seam_projection_init_sha256": _tensor_sha256(
                    self.seam_projection.weight
                ),
            }
        )



    @staticmethod
    def _normalize_markov_config(config) -> dict:
        dflash_config = getattr(config, "dflash_config", None) or {}
        raw_config = getattr(config, "markov_head", None)
        if raw_config is None:
            raw_config = dflash_config.get("markov_head", None)
        if raw_config is None:
            return {"enabled": False, "rank": 0}

        markov_config = dict(raw_config)
        rank = int(markov_config.get("rank", 0))
        if rank <= 0:
            raise ValueError(f"markov_head.rank must be positive, got {rank}.")
        head_type = str(markov_config.get("type", "vanilla")).lower()
        if head_type != "vanilla":
            raise ValueError(
                "DFlash markov_head supports only the vanilla DSpark bigram head, "
                f"got type={head_type!r}."
            )

        order = int(markov_config.get("order", 1))
        if order not in (1, 2):
            raise ValueError(
                f"markov_head.order must be 1 or 2, got {order}."
            )

        markov_config = {"rank": rank}
        if "order" in raw_config:
            markov_config["order"] = order
        if order == 2:
            mode = str(raw_config.get("mode", "")).lower()
            if mode not in ("additive", "gated"):
                raise ValueError(
                    "markov_head.mode is required for order 2 and must be "
                    f"'additive' or 'gated', got {mode!r}."
                )
            bos_row_index = int(
                raw_config.get("bos_row_index", config.vocab_size)
            )
            if bos_row_index != int(config.vocab_size):
                raise ValueError(
                    "markov_head.bos_row_index must be the dedicated row after "
                    f"the vocabulary ({config.vocab_size}), got {bos_row_index}."
                )
            markov_config.update(
                {
                    "mode": mode,
                    "bos_row_index": bos_row_index,
                }
            )
        config.markov_head = markov_config
        config.markov_rank = rank
        config.markov_head_type = head_type
        return {"enabled": True, **markov_config}

    @staticmethod
    def _normalize_token_top_config(config) -> dict:
        dflash_config = getattr(config, "dflash_config", None) or {}
        raw_config = getattr(config, "token_top_layer", None)
        if raw_config is None:
            raw_config = dflash_config.get("token_top_layer", None)
        if raw_config is None:
            return {"enabled": False}
        token_top_config = dict(raw_config)
        token_top_config["enabled"] = bool(token_top_config.get("enabled", False))
        token_top_config.setdefault("width", int(config.hidden_size))
        token_top_config.setdefault("heads", int(config.num_attention_heads))
        token_top_config.setdefault("init", "zero_output")
        token_top_config.setdefault("num_layers", 1)
        token_top_config.setdefault("fusion", "concat")
        token_top_config.setdefault("mlp", False)
        token_top_config.setdefault("parent_swap", False)
        config.token_top_layer = token_top_config
        if not hasattr(config, "dflash_config") or config.dflash_config is None:
            config.dflash_config = {}
        config.dflash_config["token_top_layer"] = token_top_config
        return token_top_config





    def apply_token_top_layer(
        self,
        hidden_states: torch.Tensor,
        prev_token_embedding: torch.Tensor,
        position_ids: torch.LongTensor,
        block_keep_mask: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.token_top_layer is None:
            return hidden_states
        token_top_kwargs = dict(
            hidden_states=hidden_states,
            prev_token_embedding=prev_token_embedding,
            position_ids=position_ids,
            block_keep_mask=block_keep_mask,
        )
        if attention_mask is not None:
            token_top_kwargs["attention_mask"] = attention_mask
        return self.token_top_layer(**token_top_kwargs)

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
        token_top_prev_embedding: Optional[torch.Tensor] = None,
        token_top_block_keep_mask: Optional[torch.Tensor] = None,
        split_stack_token_embedding: Optional[torch.Tensor] = None,
        apply_token_top: bool = True,
        past_key_values: Optional[Cache] = None,
        use_cache: bool = False,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        no_inject_attention_mask = kwargs.pop("no_inject_attention_mask", None)
        if self.has_masked_inject_layers and past_key_values is not None:
            raise NotImplementedError(
                "KV-cached DFlash forward does not support per-layer inject_layer_ids."
            )
        attention_backend = kwargs.get(
            "attention_backend",
            self.config._attn_implementation,
        )
        if (
            self.has_masked_inject_layers
            and attention_backend == "flex_attention"
            and no_inject_attention_mask is None
        ):
            raise ValueError(
                "Masked flex_attention requires a block-local no-inject attention mask."
            )
        hidden_states = noise_embedding
        target_hidden = self.hidden_norm(self.fc(target_hidden))
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        for layer_idx, layer in enumerate(self.layers):
            if (
                split_stack_token_embedding is not None
                and layer_idx == self.split_stack_from
            ):
                if split_stack_token_embedding.shape != noise_embedding.shape:
                    raise ValueError(
                        "split-stack token and mask embeddings must have identical shapes."
                    )
                if self.seam_projection is not None:
                    # Keep the lower-stack residual intact and add a learned map
                    # from token-vs-MASK embedding space at the seam only.
                    hidden_states = hidden_states + self.seam_projection(
                        split_stack_token_embedding - noise_embedding
                    )
                else:
                    # Keep the path-independent lower-stack updates while replacing
                    # their MASK embedding component with the sampled token input.
                    hidden_states = (
                        hidden_states + split_stack_token_embedding - noise_embedding
                    )
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
        if self.token_top_layer is not None and apply_token_top:
            if token_top_prev_embedding is None:
                _log_token_top_inactive_once()
            else:
                if token_top_block_keep_mask is None:
                    raise ValueError(
                        "token_top_block_keep_mask is required when token_top_prev_embedding is provided."
                    )
                token_top_position_ids = position_ids[:, -hidden_states.shape[1] :]
                hidden_states = self.apply_token_top_layer(
                    hidden_states=hidden_states,
                    prev_token_embedding=token_top_prev_embedding.to(dtype=hidden_states.dtype),
                    position_ids=token_top_position_ids,
                    block_keep_mask=token_top_block_keep_mask,
                )
        return hidden_states

