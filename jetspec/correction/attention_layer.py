"""Attention correction layer for draft-head checkpoints.

This module preserves the trainer's block-local residual attention math. Public
identifiers use the ratified correction-layer vocabulary; the adapter owns all
checkpoint compatibility mapping. Do not refactor the computations here: their
ordering and tensor shapes are part of the training contract.
"""
from __future__ import annotations

import torch
from torch import nn
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Config,
    Qwen3MLP,
    Qwen3RMSNorm,
    Qwen3RotaryEmbedding,
    eager_attention_forward,
)

# apply_rotary_pos_emb is vendored byte-identical in draft_head.py; reuse it so
# the RoPE math is shared with the backbone path.
from jetspec.models.draft_head import apply_rotary_pos_emb


def build_attention_layer_causal_attention_mask(
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


class Qwen3CorrectionAttention(nn.Module):
    """Block-local causal self-attention for the attention correction layer."""

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


class _AttentionLayerBlock(nn.Module):
    """One attention-layer block-local residual adapter."""

    def __init__(self, config: Qwen3Config, correction_config: dict):
        super().__init__()
        self.hidden_size = int(config.hidden_size)
        self.block_size = int(config.block_size)
        self.width = int(correction_config.get("width", self.hidden_size))
        self.heads = int(correction_config.get("heads", config.num_attention_heads))
        self.init = str(correction_config.get("init", "zero_output"))
        self.fusion_mode = str(correction_config.get("fusion", "concat")).lower()
        self.use_mlp = bool(correction_config.get("mlp", False))
        if self.width != self.hidden_size:
            raise ValueError(
                f"correction_layer.width must match hidden_size={self.hidden_size}, got {self.width}."
            )
        if self.heads != int(config.num_attention_heads):
            raise ValueError(
                f"correction_layer.heads must match num_attention_heads={config.num_attention_heads}, got {self.heads}."
            )
        if self.fusion_mode not in ("concat", "h_only", "e_only", "add"):
            raise ValueError(
                "Unsupported correction_layer.fusion="
                f"{self.fusion_mode!r}. Expected one of: concat, h_only, e_only, add."
            )

        if self.fusion_mode == "concat":
            self.fusion = nn.Linear(2 * self.hidden_size, self.hidden_size)
        elif self.fusion_mode == "e_only":
            self.fusion = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.input_layernorm = Qwen3RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = Qwen3CorrectionAttention(config)
        self.output_norm = Qwen3RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        if self.use_mlp:
            self.post_attention_layernorm = Qwen3RMSNorm(
                self.hidden_size, eps=config.rms_norm_eps
            )
            self.mlp = Qwen3MLP(config)
        self.rotary_emb = Qwen3RotaryEmbedding(config)

    def maybe_zero_init_output(self) -> None:
        if self.init in ("zero_output", "zero"):
            nn.init.zeros_(self.self_attn.o_proj.weight)
            if self.self_attn.o_proj.bias is not None:
                nn.init.zeros_(self.self_attn.o_proj.bias)
            if self.use_mlp:
                nn.init.zeros_(self.mlp.down_proj.weight)
            return
        if self.init != "default":
            raise ValueError(
                f"Unsupported correction_layer.init={self.init!r}. Expected 'zero_output' or 'default'."
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        prev_token_embedding: torch.Tensor,
        position_ids: torch.LongTensor,
        block_keep_mask: torch.Tensor,
    ) -> torch.Tensor:
        bsz, flat_slots, hidden_size = hidden_states.shape
        if hidden_size != self.hidden_size:
            raise ValueError(
                f"Expected hidden size {self.hidden_size}, got {hidden_size}."
            )
        if flat_slots % self.block_size != 0:
            raise ValueError(
                f"Expected flat_slots divisible by block_size={self.block_size}, got {flat_slots}."
            )

        n_blocks = flat_slots // self.block_size
        if self.fusion_mode == "concat":
            correction_input = self.fusion(
                torch.cat([hidden_states, prev_token_embedding], dim=-1)
            )
        elif self.fusion_mode == "h_only":
            correction_input = hidden_states
        elif self.fusion_mode == "e_only":
            correction_input = self.fusion(prev_token_embedding)
        else:
            correction_input = hidden_states + prev_token_embedding
        block_hidden = correction_input.reshape(
            bsz * n_blocks, self.block_size, self.hidden_size
        )
        block_positions = position_ids.reshape(bsz * n_blocks, self.block_size)
        position_embeddings = self.rotary_emb(block_hidden, block_positions)
        attention_mask = build_attention_layer_causal_attention_mask(
            self.block_size,
            dtype=block_hidden.dtype,
            device=block_hidden.device,
        )

        attn_input = self.input_layernorm(block_hidden)
        attn_output = self.self_attn(
            hidden_states=attn_input,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
        )
        delta = self.self_attn.o_proj(self.output_norm(attn_output))
        delta = delta.reshape(bsz, n_blocks, self.block_size, self.hidden_size)
        block_mask = block_keep_mask.reshape(bsz, n_blocks, 1, 1).to(
            dtype=delta.dtype
        )
        delta = delta * block_mask
        hidden_states = hidden_states + delta.reshape(
            bsz, flat_slots, self.hidden_size
        )
        if self.use_mlp:
            block_hidden = hidden_states.reshape(
                bsz * n_blocks, self.block_size, self.hidden_size
            )
            mlp_delta = self.mlp(self.post_attention_layernorm(block_hidden))
            mlp_delta = mlp_delta.reshape(
                bsz, n_blocks, self.block_size, self.hidden_size
            )
            mlp_delta = mlp_delta * block_mask
            hidden_states = hidden_states + mlp_delta.reshape(
                bsz, flat_slots, self.hidden_size
            )
        return hidden_states


class AttentionLayer(_AttentionLayerBlock):
    """Configurable stack of attention-layer block-local residual adapters."""

    def __init__(self, config: Qwen3Config, correction_config: dict):
        super().__init__(config, correction_config)
        self.num_layers = correction_config.get("num_layers", 1)
        if (
            isinstance(self.num_layers, bool)
            or not isinstance(self.num_layers, int)
            or self.num_layers < 1
        ):
            raise ValueError(
                f"correction_layer.num_layers must be a positive int, got {self.num_layers!r}."
            )
        self.extra_layers = nn.ModuleList(
            _AttentionLayerBlock(config, correction_config)
            for _ in range(self.num_layers - 1)
        )

    def maybe_zero_init_output(self) -> None:
        # Every residual branch must start at zero for stack-level exact identity.
        super().maybe_zero_init_output()
        for layer in self.extra_layers:
            layer.maybe_zero_init_output()

    def forward(
        self,
        hidden_states: torch.Tensor,
        prev_token_embedding: torch.Tensor,
        position_ids: torch.LongTensor,
        block_keep_mask: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = super().forward(
            hidden_states=hidden_states,
            prev_token_embedding=prev_token_embedding,
            position_ids=position_ids,
            block_keep_mask=block_keep_mask,
        )
        for layer in self.extra_layers:
            hidden_states = layer(
                hidden_states=hidden_states,
                prev_token_embedding=prev_token_embedding,
                position_ids=position_ids,
                block_keep_mask=block_keep_mask,
            )
        return hidden_states


__all__ = [
    "build_attention_layer_causal_attention_mask",
    "Qwen3CorrectionAttention",
    "_AttentionLayerBlock",
    "AttentionLayer",
]
