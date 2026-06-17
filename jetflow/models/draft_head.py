"""DFlash draft head — vendored from PTD/causal_parallel_drafting/model/dflash.py.

A causal speculative-decoding head that subclasses Qwen3PreTrainedModel, shares
the *target's* embed_tokens + lm_head (it owns neither), and conditions on
`target_hidden` (concatenated hidden states tapped from selected target layers).
Engine-only: the reference's spec_generate / tree / triton paths are dropped —
the JetFlow engine (jetflow/core/llm.py) owns the decode + tree-verify loop and stays
on the SDPA attention backend (no Optimus kernel).

Forward contract (unchanged from reference):
    forward(position_ids, noise_embedding, target_hidden, ...) -> hidden (1, L, H)
The caller applies the target's lm_head to get logits.

The attention / decoder / adapter classes below are copied BYTE-FOR-BYTE from
dflash.py (lines 46-268) — they carry critical fixes (the KV-cache layer_idx fix
at lines 144-148 and the explicit block-causal mask logic); do not "clean them up".
"""
from typing import Optional, Callable
from typing_extensions import Unpack, Tuple
import warnings

import torch
from torch import nn
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3RMSNorm,
    Qwen3RotaryEmbedding,
    Qwen3Config,
    Qwen3PreTrainedModel,
    Qwen3MLP,
    GradientCheckpointingLayer,
    FlashAttentionKwargs,
    rotate_half,
    eager_attention_forward,
    ALL_ATTENTION_FUNCTIONS,
)
from transformers.cache_utils import Cache
from transformers.modeling_outputs import CausalLMOutputWithPast  # noqa: F401 (BC)


# ---- vendored from reference utils.py (kept local; JetFlow owns no tree deps) ----
def build_target_layer_ids(num_target_layers: int, num_draft_layers: int) -> list[int]:
    if num_draft_layers == 1:
        return [num_target_layers // 2]
    start = 1
    end = num_target_layers - 3
    span = end - start
    return [
        int(round(start + (i * span) / (num_draft_layers - 1)))
        for i in range(num_draft_layers)
    ]


def extract_context_feature(hidden_states, layer_ids: list[int]) -> torch.Tensor:
    """Concatenate selected target-layer hidden states along the feature dim.

    offset=1 because HF returns the embedding output at index 0, so target layer
    `L` is hidden_states[L + 1]. Returns (B, T, len(layer_ids)*H).
    """
    offset = 1
    selected_states = []
    for layer_id in layer_ids:
        selected_states.append(hidden_states[layer_id + offset])
    return torch.cat(selected_states, dim=-1)


# ============================================================================
# COPIED VERBATIM from dflash.py lines 46-268 — do not modify. These carry the
# KV-cache layer_idx fix (lines 144-148) and the block-causal mask logic.
# ============================================================================
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
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        dflash_config = getattr(config, "dflash_config", {}) or {}
        self.is_causal = bool(dflash_config.get("causal_head", False))
        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.sliding_window = config.sliding_window if config.layer_types[layer_idx] == "sliding_attention" else None

    def forward(
        self,
        hidden_states: torch.Tensor,
        target_hidden: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        bsz, q_len = hidden_states.shape[:-1]
        ctx_len = target_hidden.shape[1]
        is_causal = kwargs.pop("is_causal", None)
        if is_causal is None:
            is_causal = self.is_causal
        q = self.q_proj(hidden_states)
        q = q.view(bsz, q_len, -1, self.head_dim)
        q = self.q_norm(q).transpose(1, 2)
        k_ctx = self.k_proj(target_hidden)
        k_noise = self.k_proj(hidden_states)
        v_ctx = self.v_proj(target_hidden)
        v_noise = self.v_proj(hidden_states)
        k = torch.cat([k_ctx, k_noise], dim=1).view(bsz, ctx_len + q_len, -1, self.head_dim)
        v = torch.cat([v_ctx, v_noise], dim=1).view(bsz, ctx_len + q_len, -1, self.head_dim)
        k = self.k_norm(k).transpose(1, 2)
        v = v.transpose(1, 2)
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        # NOTE: DynamicCache.get_seq_length() defaults to layer_idx=0, which returns layer 0's
        # cached length — even when called from layer_idx > 0. Because layer 0 runs first and
        # updates its cache before layer 1's forward, layers 1..N would otherwise read layer 0's
        # post-update length (ctx_len+q_len) instead of their own still-empty cache (0). That bug
        # makes _build_dflash_causal_attention_mask produce an all-zero (non-causal) mask at layers
        # 1..N on the first speculative iteration, which mismatches training (training applies the
        # same block-causal mask uniformly at every layer (as in the reference). Pass
        # self.layer_idx to query THIS layer's own cached length.
        cached_kv_len = (
            past_key_values.get_seq_length(self.layer_idx)
            if past_key_values is not None
            else 0
        )
        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            k, v = past_key_values.update(k, v, self.layer_idx, cache_kwargs)

        attn_backend = self.config._attn_implementation
        use_explicit_dflash_causal_mask = bool(is_causal) and attn_backend in {"eager", "sdpa"}
        if use_explicit_dflash_causal_mask:
            dflash_causal_mask = _build_dflash_causal_attention_mask(
                query=q,
                key=k,
                cached_kv_len=cached_kv_len,
                ctx_len=ctx_len,
            )
            if attention_mask is not None:
                dflash_causal_mask = dflash_causal_mask + _to_additive_attention_mask(
                    attention_mask,
                    query_dtype=q.dtype,
                    device=q.device,
                    key_len=k.shape[-2],
                )
            attention_mask = dflash_causal_mask
            is_causal = False

        kwargs["is_causal"] = is_causal

        attn_fn: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attn_fn = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
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


class Qwen3DFlashDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Qwen3DFlashAttention(config=config, layer_idx=layer_idx)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

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
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            target_hidden=target_hidden,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )[0]
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class HiddenDimAdapter(nn.Module):
    """Single shared linear projection in_dim -> out_dim, applied to every
    target-layer slice of the concatenated target hidden. All L slices share
    the same projection weights (NOT L independent linears) — chosen because
    weights are random-init smoke anyway and per-layer differentiation needs
    Path B (fresh-train) to be meaningful.

    Random-initialized — produces noisy hidden states. Smoke-only; do NOT use
    for acceptance baselines. Use Path B (fresh-train a head against the new
    target) for real numbers.
    """

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.proj = nn.Linear(in_dim, out_dim, bias=False)
        warnings.warn(
            f"HiddenDimAdapter active (in_dim={in_dim}, out_dim={out_dim}, "
            "random-init). Path A smoke only — acceptance numbers will be "
            "garbage.",
            stacklevel=2,
        )

    def forward(self, target_hidden_concat: torch.Tensor) -> torch.Tensor:
        # [B, T, L * in_dim] -> [B, T, L * out_dim]; reshape (not view) to
        # tolerate non-contiguous callers.
        B, T, total = target_hidden_concat.shape
        L = total // self.in_dim
        return self.proj(
            target_hidden_concat.reshape(B, T, L, self.in_dim)
        ).reshape(B, T, L * self.out_dim)


# ============================================================================
# DFlashDraftModel — vendored from dflash.py lines 270-332. The reference's
# spec_generate method and all tree / tree_attention_kernel / triton imports are
# DROPPED: the JetFlow engine owns the decode + tree-verify loop and stays on SDPA.
# ============================================================================
class DFlashDraftModel(Qwen3PreTrainedModel):
    config_class = Qwen3Config
    _no_split_modules = ["Qwen3DFlashDecoderLayer"]

    def __init__(self, config) -> None:
        super().__init__(config)
        self.config = config
        if not hasattr(self.config, "dflash_config") or self.config.dflash_config is None:
            self.config.dflash_config = {}
        self.causal_head = bool(self.config.dflash_config.get("causal_head", False))
        self.layers = nn.ModuleList(
            [Qwen3DFlashDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.target_layer_ids = self.config.dflash_config.get(
            "target_layer_ids",
            build_target_layer_ids(config.num_target_layers, config.num_hidden_layers),
        )
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config)
        target_hidden_size = self.config.dflash_config.get("target_hidden_size", config.hidden_size)
        if target_hidden_size != config.hidden_size:
            self.hidden_dim_adapter = HiddenDimAdapter(target_hidden_size, config.hidden_size)
        else:
            self.hidden_dim_adapter = nn.Identity()
        self.fc = nn.Linear(len(self.target_layer_ids) * config.hidden_size, config.hidden_size, bias=False)
        self.hidden_norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.block_size = config.block_size
        self.mask_token_id = self.config.dflash_config.get("mask_token_id", None)
        self.post_init()

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
    ) -> torch.Tensor:
        hidden_states = noise_embedding
        target_hidden = self.hidden_norm(self.fc(self.hidden_dim_adapter(target_hidden)))
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        for layer in self.layers:
            hidden_states = layer(
                hidden_states=hidden_states,
                target_hidden=target_hidden,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                use_cache=use_cache,
                position_embeddings=position_embeddings,
                **kwargs,
            )
        return self.norm(hidden_states)  # (B, L, H); caller applies the target lm_head

    def project_target_hidden(self, target_hidden: torch.Tensor) -> torch.Tensor:
        """Apply target_hidden -> adapter -> fc -> norm. Helper for smoke tests
        that verify the adapter handles shape mismatch end-to-end without
        running the full draft forward."""
        return self.hidden_norm(self.fc(self.hidden_dim_adapter(target_hidden)))


def load_draft_head(
    repo_or_path: str,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    attn_implementation: str = "sdpa",
) -> DFlashDraftModel:
    """Load a trained DFlash draft head from an HF repo (or local dir).

    Standard HF from_pretrained — the safetensors keys are produced by this same
    class, so no remapping. The config.json carries dflash_config / block_size /
    num_target_layers, which round-trip onto Qwen3Config. Multi-cell sweep repos
    use the `'repo::subfolder'` form, mapped to HF `subfolder=`.
    """
    repo, _, subfolder = repo_or_path.partition("::")  # "repo::subfolder" optional form
    kwargs = {"dtype": dtype, "attn_implementation": attn_implementation}
    if subfolder:
        kwargs["subfolder"] = subfolder
    head = DFlashDraftModel.from_pretrained(repo, **kwargs).to(device).eval()
    return head
