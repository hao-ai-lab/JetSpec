"""JetSpec block drafter, frozen H-to-H adapter, and Jacobi refinement loop."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import importlib.util
import sys
import weakref
import numpy as np
from collections import OrderedDict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Literal, NamedTuple, Optional

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_file
from torch import nn
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Config,
    Qwen3MLP,
    Qwen3RMSNorm,
    Qwen3RotaryEmbedding,
    rotate_half,
)
from jetspec.engines.jetspec_plus_plus.wire.chain.contract import key_length_bucket; from jetspec.engines.jetspec_plus_plus.wire.chain.contract import key_length_buckets
import jetspec.engines.jetspec_plus_plus.draft.block_geometry as block_geometry

from jetspec.engines.jetspec_plus_plus.wire.chain.contract import FrontInputs; from jetspec.engines.jetspec_plus_plus.wire.chain.contract import Offset; from jetspec.engines.jetspec_plus_plus.wire.chain.contract import TokenBuffers


Mode = Literal["token", "kth", "noise"]
ADAPTER_STATE_SHA256 = (
    "1f995be3ba1fd7a1fb630d255879ea85943eb581a67d6d85fac53b7450d6cac6"
)
ADAPTER_FILE_SHA256 = (
    "3046a850e27eb56de78df198c80f9fa5275c3faad3204e92d0ed4ac4c963f6e5"
)






def chain_query_ids(rp, tokens: TokenBuffers):
    """The current committed token followed by the pending draft block."""
    current = tokens.committed_ids.index_select(0, rp[Offset.L].view(1))  # the correction: committed length is L + 1
    return torch.cat((current, tokens.pending_draft_ids))[None]


class PassTelemetry(NamedTuple):
    pass_index: int
    input_tokens: tuple[int, ...]
    edit_rate: float


class DraftForwardPass(NamedTuple):
    logits: torch.Tensor
    input_tokens: torch.Tensor
    hidden: torch.Tensor | None = None


class DraftSuperseded(RuntimeError):
    """A newer verifier state arrived between Jacobi passes."""


@dataclass(frozen=True)
class RefinementResult:
    logits: torch.Tensor
    final_tokens: torch.Tensor
    passes_run: int
    converged: bool
    telemetry: tuple[PassTelemetry, ...]


@dataclass(frozen=True)
class Block:
    hypothesis: int
    mode: Mode
    draft_ids: torch.Tensor
    logits: torch.Tensor | None
    passes_run: int
    converged: bool
    anchor: int
    correction_guess: int | None = None
    k_probabilities: torch.Tensor | None = None
    kpred_pass: str | None = None
    kpred_ms: float = 0.0


def anchor_known_correction(block: Block, *, correction: int) -> Block:
    """Convert a token-anchored 15-token result to a known-f kth wire block."""
    if block.mode != "token" or block.correction_guess is not None:
        raise ValueError("known-correction anchoring requires a token-mode block")
    if block.draft_ids.numel() != 15 or block.logits.shape[0] != 15:
        raise ValueError("token-mode anchor must contain exactly 15 predictions")
    correction = int(correction)
    return replace(
        block,
        mode="kth",
        draft_ids=block.draft_ids[:14].detach(),
        logits=block.logits[:14].detach(),
        anchor=correction,
        correction_guess=correction,
    )


def following_candidate_from_token_block(block: Block) -> Block:
    """Shift a token-anchored prediction into the next-round wire layout."""
    if block.mode != "token" or block.correction_guess is not None:
        raise ValueError("following candidate requires a token-mode block")
    if block.draft_ids.numel() != 15 or block.logits.shape[0] != 15:
        raise ValueError("token-mode candidate must contain exactly 15 predictions")
    return replace(
        block,
        mode="kth",
        draft_ids=block.draft_ids[1:].detach(),
        logits=block.logits[1:].detach(),
        correction_guess=int(block.draft_ids[0].item()),
    )


class OneRoundAdaptedContext:
    """Carry the adapter-predicted hidden tail into exactly the next round."""

    def __init__(self, max_adapted: int = 16) -> None:
        self.max_adapted = int(max_adapted)
        self.clear()

    def clear(self) -> None:
        self.start_position: int | None = None
        self.token_ids: torch.Tensor | None = None
        self.predicted_hidden: torch.Tensor | None = None

    def stage(
        self,
        *,
        start_position: int,
        token_ids: torch.Tensor,
        predicted_hidden: torch.Tensor,
    ) -> None:
        if token_ids.ndim != 2 or predicted_hidden.ndim != 3:
            raise ValueError("staged tokens/hidden must have shapes [1, q] and [1, q, h]")
        if token_ids.shape[:2] != predicted_hidden.shape[:2]:
            raise ValueError("staged token and hidden lengths differ")
        self.start_position = int(start_position)
        self.token_ids = token_ids.detach().clone()
        self.predicted_hidden = predicted_hidden.detach().clone()



@dataclass
class JetSpecSequence:
    """Per-sequence mirror owned by the speculator process."""

    taps: torch.Tensor
    committed_ids: torch.Tensor
    prompt_index: int = 0
    staged: OneRoundAdaptedContext = field(
        default_factory=lambda: OneRoundAdaptedContext(
            max_adapted=block_geometry.window_rows())
    )
    prepared_context: torch.Tensor | None = None
    pending_block: Block | None = None
    decoder_hidden_tail: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if self.taps.ndim != 3 or int(self.taps.shape[1]) != 5:
            raise ValueError("taps must have shape [positions, 5, hidden]")
        if self.committed_ids.ndim != 2 or int(self.committed_ids.shape[0]) != 1:
            raise ValueError("committed_ids must have shape [1, positions + 1]")
        if int(self.taps.shape[0]) != int(self.committed_ids.shape[1]) - 1:
            raise ValueError("tap mirror must end immediately before the committed anchor")

    def exact_hidden(self, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return self.taps.reshape(1, self.taps.shape[0], -1).to(
            device=device, dtype=dtype
        )



def _run_manifest():
    """The run-manifest recorder; by package when this module has one, else by path
    (the test harness loads drafter.py as a bare file)."""
    if __package__:
        import jetspec.engines.jetspec_plus_plus.draft.run_manifest as run_manifest
        return run_manifest
    name = "jetspec_run_manifest"
    if name not in sys.modules:
        path = Path(__file__).with_name("run_manifest.py")
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def _compile_pin_module():
    """The compile pin; by package when this module has one, else by path
    (the test harness loads drafter.py as a bare file)."""
    if __package__:
        import jetspec.engines.jetspec_plus_plus.draft.compile_pin as compile_pin
        return compile_pin
    name = "jetspec_compile_pin"
    if name not in sys.modules:
        path = Path(__file__).with_name("compile_pin.py")
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def build_random_context_block_token_ids(
    context_ids: torch.Tensor,
    *,
    block_size: int,
    seed: int,
    prompt_index: int,
    mode: Mode = "token",
    device_gather: bool = False,
) -> torch.Tensor:
    """Build the rail's anchor plus its paired random-context draft slots."""
    if mode not in {"token", "noise", "kth"}:
        raise ValueError("mode must be token, noise, or kth")
    context = context_ids.reshape(-1)
    if int(context.numel()) == 0:
        raise ValueError("context_ids must contain at least one token")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) ^ int(prompt_index))
    sample_indices = torch.randint(
        int(context.numel()),
        (int(block_size) - 1,),
        generator=generator,
    )
    if device_gather:
        # Pinned staging: a pageable host copy would drain the device stream.
        if context_ids.device.type == "cuda":
            sample_indices = sample_indices.pin_memory().to(context_ids.device, non_blocking=True)
        else:
            sample_indices = sample_indices.to(context_ids.device)
        sampled = context.detach().index_select(0, sample_indices).reshape(1, -1)
    else:
        sampled = context.detach().cpu()[sample_indices].to(context_ids.device).reshape(1, -1)
    if mode == "noise":
        slot_zero_index = torch.randint(
            int(context.numel()),
            (1,),
            generator=generator,
        )
        if device_gather:
            slot_zero = context.detach().index_select(0, slot_zero_index.to(context_ids.device)).reshape(1, 1)
        else:
            slot_zero = (
                context.detach()
                .cpu()[slot_zero_index]
                .to(context_ids.device)
                .reshape(1, 1)
            )
        return torch.cat([slot_zero, sampled], dim=1)
    if mode == "kth":
        if int(context_ids.shape[1]) < 2:
            raise ValueError("kth mode requires a token before the correction")
        return torch.cat([context_ids[:, -2:-1], sampled], dim=1)
    return torch.cat([context_ids[0, -1:].reshape(1, 1), sampled], dim=1)








# Batch-generalized refinement protocol.
def run_refinement_protocol(
    forward: Callable[..., DraftForwardPass],
    *,
    batch_size: int,
    depth: int,
    max_passes: int | None,
    convergence_limit: int = 64,
    abort_if: Callable[[], bool] | None = None,
) -> list[RefinementResult]:
    """Run pure Jacobi iteration per hypothesis until fixed point or the cap."""
    if batch_size < 1 or depth < 1:
        raise ValueError("batch_size and depth must be positive")
    if max_passes is not None and max_passes < 1:
        raise ValueError("max_passes must be positive or None")
    hard_limit = convergence_limit if max_passes is None else max_passes
    if hard_limit < 1:
        raise ValueError("convergence_limit must be positive")

    telemetry: list[list[PassTelemetry]] = [[] for _ in range(batch_size)]
    finished = torch.zeros(batch_size, dtype=torch.bool)
    converged = [False] * batch_size
    passes_run = [0] * batch_size
    final_logits: list[torch.Tensor | None] = [None] * batch_size
    final_tokens: list[torch.Tensor | None] = [None] * batch_size
    current = None

    for pass_index in range(hard_limit):
        if abort_if is not None and abort_if():
            raise DraftSuperseded("newer committed verifier state is ready")
        forward_pass = forward(fill_tokens=current, pass_index=pass_index)
        if not isinstance(forward_pass, DraftForwardPass):
            raise TypeError("forward must return DraftForwardPass")
        logits = forward_pass.logits
        if tuple(logits.shape[:2]) != (batch_size, depth):
            raise ValueError(
                f"forward must return [{batch_size}, {depth}, vocab], "
                f"got {tuple(logits.shape)}"
            )
        input_tokens = forward_pass.input_tokens
        if tuple(input_tokens.shape) != (batch_size, depth):
            raise ValueError(
                f"forward input tokens must have shape [{batch_size}, {depth}], "
                f"got {tuple(input_tokens.shape)}"
            )
        current = logits.argmax(dim=-1)
        now_fixed = (current == input_tokens).all(dim=1).cpu()
        for row in range(batch_size):
            if bool(finished[row]):
                continue
            edit_rate = float(
                (current[row] != input_tokens[row]).float().mean().item()
            )
            telemetry[row].append(
                PassTelemetry(
                    pass_index,
                    tuple(int(token) for token in input_tokens[row].detach().cpu().tolist()),
                    edit_rate,
                )
            )
            passes_run[row] = pass_index + 1
            if bool(now_fixed[row]):
                finished[row] = True
                converged[row] = True
                final_logits[row] = logits[row]
                final_tokens[row] = current[row]
        if bool(finished.all()):
            break

    assert current is not None
    for row in range(batch_size):
        if final_logits[row] is None:
            final_logits[row] = logits[row]
            final_tokens[row] = current[row]
    return [
        RefinementResult(
            logits=final_logits[row],
            final_tokens=final_tokens[row],
            passes_run=passes_run[row],
            converged=converged[row],
            telemetry=tuple(telemetry[row]),
        )
        for row in range(batch_size)
    ]


def _repeat_kv(hidden_states: torch.Tensor, groups: int) -> torch.Tensor:
    return hidden_states.repeat_interleave(groups, dim=1)


def _apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    del position_ids
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_len = q.size(-2)
    q_embed = (q * cos[..., -q_len:, :]) + (rotate_half(q) * sin[..., -q_len:, :])
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class _PinnedCtxKVAttention(nn.Module):
    """The attention submodule from the pinned H-to-H adapter."""

    def __init__(self, config) -> None:
        super().__init__()
        self.hidden_size = int(config.hidden_size)
        self.num_heads = int(config.num_attention_heads)
        self.num_key_value_heads = int(config.num_key_value_heads)
        self.head_dim = int(
            getattr(config, "head_dim", self.hidden_size // self.num_heads)
        )
        if self.num_heads % self.num_key_value_heads != 0:
            raise ValueError("ctxkv q heads must be divisible by kv heads")
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        bias = bool(config.attention_bias)
        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=bias
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
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config)

    def forward(
        self,
        *,
        query_states: torch.Tensor,
        prefix_h: torch.Tensor,
        prefix_position_ids: torch.Tensor,
        query_position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, query_count, _ = query_states.shape
        prefix_count = prefix_h.shape[1]
        key_value_states = torch.cat([prefix_h, query_states], dim=1)
        q = self.q_proj(query_states).view(
            batch_size, query_count, self.num_heads, self.head_dim
        )
        k = self.k_proj(key_value_states).view(
            batch_size,
            prefix_count + query_count,
            self.num_key_value_heads,
            self.head_dim,
        )
        v = self.v_proj(key_value_states).view(
            batch_size,
            prefix_count + query_count,
            self.num_key_value_heads,
            self.head_dim,
        )
        q = self.q_norm(q).transpose(1, 2)
        k = self.k_norm(k).transpose(1, 2)
        v = v.transpose(1, 2)
        all_position_ids = torch.cat(
            [prefix_position_ids, query_position_ids], dim=1
        )
        cos, sin = self.rotary_emb(key_value_states, all_position_ids)
        q, k = _apply_rotary_pos_emb(q, k, cos, sin)
        if self.num_key_value_groups != 1:
            k = _repeat_kv(k, self.num_key_value_groups)
            v = _repeat_kv(v, self.num_key_value_groups)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scaling
        weights = F.softmax(scores + attention_mask, dim=-1, dtype=torch.float32).to(
            dtype=q.dtype
        )
        output = torch.matmul(weights, v)
        output = output.transpose(1, 2).reshape(batch_size, query_count, -1)
        return self.o_proj(output)


class _PinnedCtxKVL1H2HHead(nn.Module):
    """Exact pinned H-to-H inference graph without its training wrapper."""

    def __init__(self, config) -> None:
        super().__init__()
        mode = config.dflash_config.get("ctxkv_head", {}).get("mode")
        if mode != "l1_attention_h2h":
            raise ValueError("the pinned adapter requires ctxkv mode l1_attention_h2h")
        self.hidden_size = int(config.hidden_size)
        self.hspace_dim = len(config.dflash_config["target_layer_ids"]) * int(
            getattr(config, "target_hidden_size", self.hidden_size)
        )
        self.fusion = nn.Linear(2 * self.hidden_size, self.hidden_size, bias=False)
        self.input_projection = nn.Linear(
            self.hspace_dim, self.hidden_size, bias=False
        )
        self.fusion_norm = Qwen3RMSNorm(
            self.hidden_size, eps=config.rms_norm_eps
        )
        self.attention = _PinnedCtxKVAttention(config)
        self.final_norm = Qwen3RMSNorm(
            self.hidden_size, eps=config.rms_norm_eps
        )
        self.output_projection = nn.Linear(
            self.hidden_size, self.hspace_dim, bias=False
        )

    def forward(
        self,
        *,
        token_embeddings: torch.Tensor,
        h_prev: torch.Tensor,
        prefix_h: torch.Tensor,
        prefix_position_ids: torch.Tensor,
        query_position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        shared_prefix_projection: bool = False,
        prefix_row_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Optionally project unique sequence inputs before mapping to rows.

        In shared mode prefix_h is [N,P,H] and h_prev is [N,1,H] or
        [N,Q,H]. prefix_row_indices maps each query row to its sequence;
        positions, masks and token embeddings remain per row. Without a map,
        each sequence has one row (including the stage() serving call).
        """
        expected = token_embeddings.shape[:2] + (self.hspace_dim,)
        if prefix_h.ndim != 3 or prefix_h.shape[-1] != self.hspace_dim:
            raise ValueError("prefix_h must have shape [batch, prefix, hspace_dim]")
        if shared_prefix_projection:
            if (h_prev.ndim != 3 or h_prev.shape[0] != prefix_h.shape[0]
                    or h_prev.shape[1] not in (1, expected[1])
                    or h_prev.shape[2] != self.hspace_dim):
                raise ValueError("shared h_prev must have shape [N,1 or query,hspace_dim]")
            if prefix_row_indices is None:
                if prefix_h.shape[0] != expected[0]:
                    raise ValueError("shared sequences require a row mapping")
            elif (prefix_row_indices.shape != (expected[0],)
                  or prefix_row_indices.dtype != torch.long):
                raise ValueError("prefix_row_indices must be int64 [rows]")
            repeated_c_prev = self.input_projection(h_prev)
            prefix_states = self.input_projection(prefix_h)
            if prefix_row_indices is not None:
                repeated_c_prev = repeated_c_prev.index_select(0, prefix_row_indices)
                prefix_states = prefix_states.index_select(0, prefix_row_indices)
            repeated_c_prev = repeated_c_prev.expand(-1, expected[1], -1)
        else:
            if h_prev.shape != expected:
                raise ValueError("h_prev must have shape [batch, query, hspace_dim]")
            if prefix_row_indices is not None:
                raise ValueError("row mapping requires shared_prefix_projection")
            repeated_c_prev = self.input_projection(h_prev)
            prefix_states = self.input_projection(prefix_h)
        fused = self.fusion(
            torch.cat([token_embeddings, repeated_c_prev], dim=-1)
        )
        fused = self.fusion_norm(fused)
        attended = self.attention(
            query_states=fused,
            prefix_h=prefix_states,
            prefix_position_ids=prefix_position_ids,
            query_position_ids=query_position_ids,
            attention_mask=attention_mask,
        )
        return self.output_projection(self.final_norm(fused + attended))


def tensor_state_sha256(module: nn.Module) -> str:
    """Hash a module's named tensor bytes in a stable order."""
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        digest.update(name.encode("utf-8"))
        value = tensor.detach().contiguous().cpu()
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


# C1 source sha256: 44147c53aa9bbc95ee2a88b52fdb3686443f35fac71798c7ea2465e80ed1d135
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
        # C5's depth form. Kept ALONGSIDE `streams` rather than nesting them, so the
        # state-dict names stay `streams.N.<piece>` -- C1b's loader addresses those, and
        # C1 is measured under them. Empty at depth 1: no parameters, no RNG draws.
        layers_per_stream = int(self.resolved_config.get("layers_per_stream", 1))
        if layers_per_stream < 1:
            raise ValueError("ctxkv_head.layers_per_stream must be >= 1")
        self.layers_per_stream = layers_per_stream
        # stream_depths: an optional per-tap depth list (one entry per tap) replaces the
        # uniform layers_per_stream; absent, every stream is layers_per_stream deep.
        stream_depths = self.resolved_config.get("stream_depths")
        if stream_depths is None:
            stream_depths = [layers_per_stream] * len(self.target_layer_ids)
        if len(stream_depths) != len(self.target_layer_ids) or any(
                int(depth) < 1 for depth in stream_depths):
            raise ValueError("ctxkv_head.stream_depths must be one positive depth per tap")
        self.stream_depths = [int(depth) for depth in stream_depths]
        # target_init FFNs are ordinary trained parameters saved in head.safetensors (the
        # strict load proves it); target_frozen FFNs are not saved and are not served.
        if self.resolved_config.get("ffn_source") not in (None, "target_init"):
            raise ValueError("C1 serving supports ffn_source absent or target_init only")
        self.extra_stream_layers = nn.ModuleList(
            nn.ModuleList(
                CtxKVC1Stream(config, mlp_intermediate=intermediate, target_text_config=self.target_text_config)
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
        else:
            stream_mask = attention_mask[..., :prefix_count]

        u = token_embeddings
        outputs = []
        for index, stream in enumerate(self.streams):
            start = index * self.hidden_size
            prefix_tap = prefix_h[..., start : start + self.hidden_size]
            # D1: the window keys are the LOWER tap's predictions for the other slots --
            # never this stream's own, which would be D2 and serial. They are fixed for
            # the whole tap, so a C5 depth stack re-reads the same lower-tap window
            # rather than its own partial outputs.
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






class FrozenH2HAdapter(nn.Module):
    """Load the pinned adapter and assert its exact post-load state."""

    def __init__(
        self,
        *,
        config_path: str,
        head_path: str,
        target_embed_tokens: nn.Module,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        head_file = Path(head_path)
        raw_digest = _file_sha256(head_file)
        if raw_digest != ADAPTER_FILE_SHA256:
            raise AssertionError(
                f"adapter file sha256 mismatch: expected={ADAPTER_FILE_SHA256} "
                f"actual={raw_digest}"
            )
        config = load_qwen3_config(config_path)
        config._attn_implementation = "eager"
        self.head = _PinnedCtxKVL1H2HHead(config).to(device=device, dtype=dtype)
        state = load_file(head_file, device="cpu")
        self.head.load_state_dict(state, strict=True)
        self.head.eval()
        for parameter in self.head.parameters():
            parameter.requires_grad_(False)
        self.target_embed_tokens = target_embed_tokens
        for parameter in self.target_embed_tokens.parameters():
            parameter.requires_grad_(False)
        actual = tensor_state_sha256(self.head)
        if actual != ADAPTER_STATE_SHA256:
            raise AssertionError(
                f"adapter state sha256 mismatch: expected={ADAPTER_STATE_SHA256} "
                f"actual={actual}"
            )
        self._sealed_state_sha256 = actual


    def assert_byte_identical(self) -> str:
        actual = tensor_state_sha256(self.head)
        if actual != self._sealed_state_sha256:
            raise AssertionError(
                "H-to-H head bytes changed: "
                f"expected={self._sealed_state_sha256} actual={actual}"
            )
        return actual

@torch.no_grad()
def _predict_frozen_h2h_windows(
    self,
    *,
    input_ids: torch.Tensor,
    true_hidden: torch.Tensor,
    local_position_ids: torch.Tensor,
    shared_boundaries: torch.Tensor,
) -> torch.Tensor:
    """Predict every forced window with the pinned lab mask construction."""
    batch_size, sequence_length = input_ids.shape
    if local_position_ids.shape[:2] != shared_boundaries.shape:
        raise ValueError(
            "local_position_ids and shared_boundaries disagree on batch/blocks"
        )
    block_count, window_size = local_position_ids.shape[1:]
    safe_positions = local_position_ids.clamp(min=0, max=sequence_length - 1)
    flat_positions = safe_positions.flatten(1, 2)
    query_ids = input_ids.gather(1, flat_positions)
    query_tokens = self.target_embed_tokens(query_ids).detach()

    previous_positions = (shared_boundaries - 1).clamp(
        min=0, max=sequence_length - 1
    )
    previous = true_hidden.gather(
        1,
        previous_positions.unsqueeze(-1).expand(-1, -1, true_hidden.shape[-1]),
    )
    previous = previous.unsqueeze(2).expand(-1, -1, window_size, -1)
    previous = previous.flatten(1, 2)

    prefix_positions = torch.arange(
        sequence_length, device=input_ids.device, dtype=torch.long
    ).unsqueeze(0).expand(batch_size, -1)
    flat_boundaries = shared_boundaries.unsqueeze(-1).expand(
        -1, -1, window_size
    ).flatten(1, 2)
    prefix_allowed = prefix_positions.unsqueeze(1) < flat_boundaries.unsqueeze(-1)
    query_count = block_count * window_size
    query_self = torch.eye(
        query_count, dtype=torch.bool, device=input_ids.device
    ).unsqueeze(0).expand(batch_size, -1, -1)
    allowed = torch.cat((prefix_allowed, query_self), dim=-1)
    additive_mask = torch.zeros(
        (batch_size, 1, query_count, sequence_length + query_count),
        dtype=true_hidden.dtype,
        device=true_hidden.device,
    ).masked_fill(
        allowed.logical_not().unsqueeze(1),
        torch.finfo(true_hidden.dtype).min,
    )

    predicted = self.head(
        token_embeddings=query_tokens,
        h_prev=previous,
        prefix_h=true_hidden,
        prefix_position_ids=prefix_positions,
        query_position_ids=flat_positions,
        attention_mask=additive_mask,
    )
    return predicted.view(
        batch_size, block_count, window_size, true_hidden.shape[-1]
    ).detach()


# Install the parity-only extension without changing the pinned reference
# class body.
FrozenH2HAdapter.predict_windows = _predict_frozen_h2h_windows


_D1_REBUILD_BUCKET = 64


def _d1_rebuild_rows(tokens):
    """The row count a D1 cache projection of ``tokens`` rows runs its Triton kernels on:
    the count itself up to 16 (the steady-state append), else the next 64-row bucket."""
    tokens = int(tokens)
    if tokens <= 16:
        return tokens
    return -(-tokens // _D1_REBUILD_BUCKET) * _D1_REBUILD_BUCKET


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


def _d1_rms(x, weight, eps):
    """Qwen3RMSNorm's arithmetic (fp32 statistics, the weight applied in the input dtype)."""
    hidden = x.float()
    hidden = hidden * torch.rsqrt(hidden.pow(2).mean(-1, keepdim=True) + eps)
    return weight * hidden.to(x.dtype)


def _d1_rope(x, cos, sin):
    half = x.shape[-1] // 2
    return x * cos + torch.cat((-x[..., half:], x[..., :half]), dim=-1) * sin


def _d1_compiled_pre_q(u, window, in_w, q_w, qn_w, kvn_w, kv_w, cos, sin, in_eps, qn_eps,
                       kvn_eps, heads: int, head_dim: int):
    """JETSPEC_C1_COMPILE with the lean cache-write kernel: q [1, heads, 15, head_dim] and
    the window's projected k|v rows [15, 2 * kv width] (normed, rotated and written into
    the cache by ``k_norm_rotary_cache``, the lean path's kernel)."""
    rows = u.shape[0]
    q = F.linear(_d1_rms(u, in_w, in_eps), q_w).view(rows, heads, head_dim)
    q = _d1_rope(_d1_rms(q, qn_w, qn_eps).transpose(0, 1), cos, sin)
    return q.unsqueeze(0).contiguous(), F.linear(_d1_rms(window, kvn_w, kvn_eps), kv_w)


def _d1_compiled_pre(u, window, in_w, q_w, qn_w, kvn_w, kv_w, kn_w, cos, sin, in_eps, qn_eps,
                     kvn_eps, kn_eps, heads: int, kv_heads: int, groups: int, head_dim: int):
    """JETSPEC_C1_COMPILE: one D1 block up to attention.  u, window: [15, width];
    cos, sin: [1, 15, head_dim].  Returns q [1, heads, 15, head_dim] and the window's
    key/value rows [heads, 15, head_dim] (repeated over the query groups, the cache's
    layout)."""
    rows = u.shape[0]
    q = F.linear(_d1_rms(u, in_w, in_eps), q_w).view(rows, heads, head_dim)
    q = _d1_rope(_d1_rms(q, qn_w, qn_eps).transpose(0, 1), cos, sin)
    key, value = F.linear(_d1_rms(window, kvn_w, kvn_eps), kv_w).split(
        kv_heads * head_dim, dim=-1)
    key = _d1_rope(_d1_rms(key.view(rows, kv_heads, head_dim), kn_w, kn_eps).transpose(0, 1),
                   cos, sin)
    value = value.view(rows, kv_heads, head_dim).transpose(0, 1)
    if groups != 1:
        key = key.repeat_interleave(groups, dim=0)
        value = value.repeat_interleave(groups, dim=0)
    return q.unsqueeze(0).contiguous(), key.contiguous(), value.contiguous()


def _d1_compiled_post(attended, u, o_w, post_w, gate_up, down_w, post_eps):
    """JETSPEC_C1_COMPILE: one D1 block after attention (attended [1, heads, 15, d])."""
    rows = u.shape[0]
    state = u + F.linear(attended.transpose(1, 2).reshape(rows, -1), o_w)
    gate, up = F.linear(_d1_rms(state, post_w, post_eps), gate_up).chunk(2, dim=-1)
    return state + F.linear(F.silu(gate) * up, down_w)


def fixed_refresh_layout(counts, tokens, width):
    """Dense device row/slot addresses with a scratch address for padded tokens."""
    rows = counts.numel()
    offsets = torch.cumsum(counts, 0)
    index = torch.arange(tokens, device=counts.device)
    row = torch.searchsorted(offsets, index, right=True)
    valid = row < rows
    starts = torch.cat((offsets.new_zeros(1), offsets[:-1]))
    slot = torch.where(valid, index - starts[row.clamp(max=rows - 1)], width)
    return row, slot, valid


class ServingCtxKVC1Head(CtxKVC1Head):
    """Reference serving translation plus default-off per-sequence C1 KV/replay."""

    @staticmethod
    def _c1_fused_module():
        name = "jetspec_c1_fused"
        if name not in sys.modules:
            path = Path(__file__).parents[1] / "kernels/c1_fused.py"
            spec = importlib.util.spec_from_file_location(name, path)
            module = importlib.util.module_from_spec(spec)
            assert spec.loader is not None
            sys.modules[name] = module
            spec.loader.exec_module(module)
        return sys.modules[name]

    @staticmethod
    def _rows_module():
        """The rows kernels; by package when this module has one, else by path
        (the test harness loads drafter.py as a bare file)."""
        if __package__:
            import jetspec.engines.jetspec_plus_plus.draft.rows as rows
            return rows
        name = "jetspec_rows"
        if name not in sys.modules:
            path = Path(__file__).with_name("rows.py")
            spec = importlib.util.spec_from_file_location(name, path)
            module = importlib.util.module_from_spec(spec)
            assert spec.loader is not None
            sys.modules[name] = module
            spec.loader.exec_module(module)
        return sys.modules[name]

    @staticmethod
    def _c1_fused_enabled():
        return os.environ.get("JETSPEC_C1_FUSED", "0") == "1"

    @staticmethod
    def _c1_fused2_enabled(tensor):
        return (
            os.environ.get("JETSPEC_C1_FUSED", "0") == "2"
            and tensor.device.type == "cuda"
            and tensor.dtype == torch.bfloat16
        )

    def _c1_fused2_prepare(self):
        if hasattr(self, "_c1_fast_kv_norm"):
            return
        blocks = [
            layer
            for index, stream in enumerate(self.streams)
            for layer in (stream, *self.extra_stream_layers[index])
        ]
        self._c1_fast_kv_norm = torch.stack([
            layer.kv_norm.weight for layer in blocks
        ])
        self._c1_fast_k_norm = torch.stack([
            layer.k_norm.weight for layer in blocks
        ])
        self._c1_fast_streams = torch.tensor(
            [
                index
                for index, stream in enumerate(self.streams)
                for _ in (stream, *self.extra_stream_layers[index])
            ],
            device=blocks[0].kv_norm.weight.device,
        )

    def forward(self, *, shared_prefix_projection=False, prefix_row_indices=None, prefix_key=None,
                prefix_keys=None, prefix_lengths=None,
                batched_uncached=False, **kwargs):
        rows = kwargs["token_embeddings"].shape[0]
        cached = os.environ.get("JETSPEC_C1_KVCACHE", "0") == "1"
        graphed = os.environ.get("JETSPEC_C1_GRAPH", "0") == "1"
        eval_mode = os.environ.get("JETSPEC_EVAL_MODE", "off") != "off"
        if graphed and (os.environ.get("TORCHDYNAMO_DISABLE") == "1"
                        or (os.environ.get("JETSPEC_HOST_FAST", "0") == "0"
                            and not eval_mode)):
            raise ValueError("C1 GRAPH refuses E")
        if graphed and not cached:
            raise ValueError("C1 GRAPH requires KVCACHE")
        if batched_uncached:
            if (not cached or (graphed and os.environ.get("JETSPEC_CHAIN_ADMIT_BATCH", "0") != "1")
                    or rows < 2 or prefix_key is not None or prefix_row_indices is not None):
                raise ValueError("batched C1 staging requires eager cached mode and independent rows")
            return super().forward(**kwargs)
        if prefix_keys is not None:
            if (not cached or graphed or prefix_key is not None or prefix_row_indices is not None
                    or rows < 2 or len(prefix_keys) != rows or prefix_lengths is None
                    or len(prefix_lengths) != rows):
                raise ValueError("multi-slot C1 requires eager cached N>1 and per-row identities/lengths")
            return self._forward_multi_cached(prefix_keys, prefix_lengths, **kwargs)
        if os.environ.get("JETSPEC_C1_COMPILE", "0") == "1" and not graphed:
            # The compiled kernel set serves in the graphed regime only
            raise ValueError("C1 COMPILE requires the graphed regime (JETSPEC_C1_GRAPH=1)")
        if cached:
            if os.environ.get("JETSPEC_C1_SHARED_PREFIX", "0") == "1":
                raise ValueError("C1 KVCACHE refuses SHARED_PREFIX")
            if rows != 1 or prefix_row_indices is not None or prefix_key is None:
                raise ValueError("C1 KVCACHE requires N1 and a sequence identity")
            output = self._forward_cached(prefix_key=prefix_key, graphed=graphed, **kwargs)
            if os.environ.get("W328_DEFER_ADAPTER"):
                from jetspec.engines.jetspec_plus_plus.draft.compile_pin import digest_defer
                digest_defer("adapter", "W328_DEFER_ADAPTER", embeddings=kwargs["token_embeddings"],
                             taps=kwargs["prefix_h"], output=output)
            if os.environ.get("W328_DIGEST"):
                from jetspec.engines.jetspec_plus_plus.draft.compile_pin import digest_log
                digest_log("adapter", embeddings=kwargs["token_embeddings"],
                           taps=kwargs["prefix_h"], positions=kwargs["query_position_ids"],
                           mask=kwargs.get("attention_mask"), output=output)
            return output
        if prefix_row_indices is not None:
            if not shared_prefix_projection:
                raise ValueError("row mapping requires shared_prefix_projection")
            if prefix_row_indices.shape != (rows,) or prefix_row_indices.dtype != torch.long:
                raise ValueError("prefix_row_indices must be int64 [rows]")
        if shared_prefix_projection and os.environ.get("JETSPEC_C1_SHARED_PREFIX", "0") == "1":
            if self.resolved_config.get("window_causal_over_tokens", False):
                raise ValueError("C1 prefix sharing requires prefix-only attention")
            owners = prefix_row_indices
            if owners is None:
                if kwargs["prefix_h"].shape[0] != rows:
                    raise ValueError("shared sequences require a row mapping")
                owners = torch.arange(rows, device=kwargs["prefix_h"].device)
            return self._forward_shared(owners=owners, **kwargs)
        if prefix_row_indices is not None:
            kwargs["prefix_h"] = kwargs["prefix_h"].index_select(0, prefix_row_indices)
        # C1 does not read h_prev. This baseline repeats each stream's K/V projections.
        return super().forward(**kwargs)

    def _multi_slot(self, sequence, active):
        key = id(sequence)
        entry = self._c1_multi_slots.get(key)
        if entry is not None and entry[1]() is sequence:
            return entry[0]
        if not self._c1_multi_free:
            # The engine can retain completed sequence objects until the read ends.
            # Evict an inactive slot; its next use rebuilds the exact prefix.
            for old_key, (old_slot, _, finalizer) in self._c1_multi_slots.items():
                if old_key not in active:
                    finalizer.detach()
                    self._c1_multi_slots.pop(old_key)
                    self._c1_multi_lengths[old_slot] = 0
                    self._c1_multi_free.append(old_slot)
                    break
            else:
                raise ValueError("C1 multi-slot cache has no inactive sequence slots")
        slot = self._c1_multi_free.pop()
        owner = weakref.ref(self)
        def drop():
            head = owner()
            if head is not None and head._c1_multi_slots.get(key, (None,))[0] == slot:
                head._c1_multi_slots.pop(key, None)
                head._c1_multi_lengths[slot] = 0
                head._c1_multi_free.append(slot)
        self._c1_multi_slots[key] = (slot, weakref.ref(sequence), weakref.finalize(sequence, drop))
        return slot

    def _forward_multi_cached(self, prefix_keys, prefix_lengths, *, token_embeddings, prefix_h,
                              prefix_position_ids, query_position_ids, attention_mask,
                              h_prev=None, c_prev=None, prefix_c=None):
        if c_prev is not None or prefix_c is not None or prefix_h.shape[-1] != self.hspace_dim:
            raise ValueError("multi-slot C1 requires tapped h-space inputs")
        count, Q = token_embeddings.shape[:2]
        lengths = [int(length) for length in prefix_lengths]
        if (len(set(map(id, prefix_keys))) != count or min(lengths) < 1
                or max(lengths) > int(os.environ.get("W259_C1_CACHE_CAPACITY", "768"))
                or attention_mask is None):
            raise ValueError("invalid C1 multi-slot lengths, identities or mask")
        if not hasattr(self, "_c1_multi_kv"):
            cap = int(os.environ.get("W259_C1_CACHE_CAPACITY", "768"))
            slots = max(int(os.environ.get("JETSPEC_C1_MULTI_SLOTS", "32")), count)
            self._c1_multi_blocks = [(index, layer) for index, stream in enumerate(self.streams)
                for layer in (stream, *self.extra_stream_layers[index])]
            layer = self._c1_multi_blocks[0][1]
            shape = (len(self._c1_multi_blocks), slots, layer.num_heads, cap + 16, layer.head_dim)
            self._c1_multi_kv = (prefix_h.new_zeros(shape), prefix_h.new_zeros(shape))
            self._c1_multi_lengths = [0] * slots
            self._c1_multi_slots = {}
            self._c1_multi_free = list(reversed(range(slots)))
        elif count > len(self._c1_multi_lengths):
            old_slots = len(self._c1_multi_lengths)
            grown = []
            for current in self._c1_multi_kv:
                shape = list(current.shape)
                shape[1] = count
                larger = current.new_zeros(shape)
                larger[:, :old_slots].copy_(current)
                grown.append(larger)
            self._c1_multi_kv = tuple(grown)
            self._c1_multi_lengths.extend([0] * (count - old_slots))
            self._c1_multi_free.extend(reversed(range(old_slots, count)))
        active = {id(sequence) for sequence in prefix_keys}
        slots = [self._multi_slot(sequence, active) for sequence in prefix_keys]
        starts = [self._c1_multi_lengths[slot] for slot in slots]
        starts = [0 if length < start or length - start > 16 else start
                  for length, start in zip(lengths, starts)]
        if os.environ.get("JETSPEC_BATCH_DRAFT_GRAPH", "0") == "1":
            # the admission tail: one projection per cold prefix (not 16-row tiles) and one
            # tile projection plus one scatter for every request's appended rows
            self._multi_append(prefix_h, prefix_position_ids, slots, starts, lengths)
            starts = list(lengths)
        # A cold prefix is filled in 16-row tiles. Steady-state calls issue one
        # projection over the appended rows of all requests for each stream layer.
        while any(start < length for start, length in zip(starts, lengths)):
            rows = prefix_h.new_zeros(count, 16, self.hspace_dim)
            positions = torch.zeros(count, 16, dtype=torch.long, device=prefix_h.device)
            for row, (start, length) in enumerate(zip(starts, lengths)):
                take = min(16, length - start)
                if take > 0:
                    rows[row, :take].copy_(prefix_h[row, start:start + take])
                    positions[row, :take] = prefix_position_ids[row, start:start + take]
            cos, sin = self.streams[0].rotary_emb(rows, positions)
            for block, (index, layer) in enumerate(self._c1_multi_blocks):
                source = layer.kv_norm(rows[..., index*self.hidden_size:(index+1)*self.hidden_size])
                projected_key, projected_value = layer.project_kv(source)
                k = layer.k_norm(projected_key.view(count, 16, layer.num_key_value_heads,
                                                         layer.head_dim)).transpose(1, 2)
                v = projected_value.view(count, 16, layer.num_key_value_heads,
                                             layer.head_dim).transpose(1, 2)
                k = _c1_rotary(k, cos, sin)
                if layer.num_key_value_groups != 1:
                    k = k.repeat_interleave(layer.num_key_value_groups, dim=1)
                    v = v.repeat_interleave(layer.num_key_value_groups, dim=1)
                for row, (slot, start, length) in enumerate(zip(slots, starts, lengths)):
                    take = min(16, length - start)
                    if take > 0:
                        self._c1_multi_kv[0][block, slot, :, start:start + take].copy_(k[row, :, :take])
                        self._c1_multi_kv[1][block, slot, :, start:start + take].copy_(v[row, :, :take])
            starts = [min(start + 16, length) for start, length in zip(starts, lengths)]
        for slot, length in zip(slots, lengths):
            self._c1_multi_lengths[slot] = length
        P = prefix_h.shape[1]
        cos, sin = self.streams[0].rotary_emb(token_embeddings, query_position_ids)
        window_causal = self.resolved_config.get("window_causal_over_tokens", False)
        mask = attention_mask[..., :P + Q] if window_causal else attention_mask[..., :P]
        u, outputs, block = token_embeddings, [], 0
        slot_ids = torch.tensor(slots, device=prefix_h.device)
        for index, stream in enumerate(self.streams):
            window = u
            for layer in (stream, *self.extra_stream_layers[index]):
                normed = layer.input_norm(u)
                q = layer.q_norm(layer.q_proj(normed).view(count, Q, layer.num_heads,
                                                          layer.head_dim)).transpose(1, 2)
                q = _c1_rotary(q, cos, sin)
                k = self._c1_multi_kv[0][block, :, :, :P].index_select(0, slot_ids)
                v = self._c1_multi_kv[1][block, :, :, :P].index_select(0, slot_ids)
                if window_causal:
                    source = layer.kv_norm(window)
                    projected_key, projected_value = layer.project_kv(source)
                    wk = layer.k_norm(projected_key.view(count, Q, layer.num_key_value_heads,
                                                               layer.head_dim)).transpose(1, 2)
                    wv = projected_value.view(count, Q, layer.num_key_value_heads,
                                                   layer.head_dim).transpose(1, 2)
                    wk = _c1_rotary(wk, cos, sin)
                    if layer.num_key_value_groups != 1:
                        wk = wk.repeat_interleave(layer.num_key_value_groups, dim=1)
                        wv = wv.repeat_interleave(layer.num_key_value_groups, dim=1)
                    k, v = torch.cat((k, wk), 2), torch.cat((v, wv), 2)
                attended = F.scaled_dot_product_attention(q, k, v, attn_mask=mask,
                                                          scale=layer.scaling)
                attended = attended.transpose(1, 2).reshape(count, Q, layer.num_heads * layer.head_dim)
                state = u + layer.o_proj(attended)
                u = state + layer.mlp(layer.post_attention_norm(state))
                block += 1
            outputs.append(u)
        return torch.cat(outputs, dim=-1)

    # --- JETSPEC_BATCH_DRAFT_GRAPH_FRONT=1 (default off): N>1 staging through a per-sequence prefix
    # K/V slot cache and one CUDA graph per (batch bucket, key bucket). The graph appends each
    # request's newly committed prefix rows (at most one window of them) into its slot and
    # runs the D1 window query over the slot's cached keys; a new or stale slot is filled
    # eagerly before the replay. Not bitwise to the reference: GEMM row counts differ, and
    # masked key padding contributes exact zeros. N=1 never reaches this path.


    def _bg_slot(self, state, sequence, active):
        key = id(sequence)
        entry = state["slots"].get(key)
        if entry is not None and entry[1]() is sequence:
            return entry[0]
        if not state["free"]:
            for old_key, (old_slot, _, finalizer) in list(state["slots"].items()):
                if old_key not in active:
                    finalizer.detach()
                    state["slots"].pop(old_key)
                    state["lengths"][old_slot] = 0
                    state["free"].append(old_slot)
                    break
            else:
                raise ValueError("batched C1 graph cache has no inactive sequence slots")
        slot = state["free"].pop()
        state["lengths"][slot] = 0
        owner = weakref.ref(self)

        def drop():
            head = owner()
            shelf = None if head is None else getattr(head, "_bg", None)
            if shelf is not None and shelf["slots"].get(key, (None,))[0] == slot:
                shelf["slots"].pop(key, None)
                shelf["lengths"][slot] = 0
                shelf["free"].append(slot)
        state["slots"][key] = (slot, weakref.ref(sequence), weakref.finalize(sequence, drop))
        return slot





    def _multi_project(self, rows, positions):
        """[M, hspace] rows at positions [M] -> keys, values [blocks, M, heads, head_dim]: the
        multi-slot tile's per-block kv norm, k/v projections, k norm, rotary and K/V repeat."""
        count = rows.shape[0]
        cos, sin = self.streams[0].rotary_emb(rows[None], positions[None])
        keys, values = [], []
        for index, layer in self._c1_multi_blocks:
            source = layer.kv_norm(rows[:, index * self.hidden_size:(index + 1) * self.hidden_size])
            k = layer.k_norm(layer.k_proj(source).view(1, count, layer.num_key_value_heads,
                                                       layer.head_dim)).transpose(1, 2)
            v = layer.v_proj(source).view(1, count, layer.num_key_value_heads,
                                         layer.head_dim).transpose(1, 2)
            k = _c1_rotary(k, cos, sin)
            if layer.num_key_value_groups != 1:
                k = k.repeat_interleave(layer.num_key_value_groups, dim=1)
                v = v.repeat_interleave(layer.num_key_value_groups, dim=1)
            keys.append(k[0].transpose(0, 1))
            values.append(v[0].transpose(0, 1))
        return torch.stack(keys), torch.stack(values)

    def _multi_append(self, prefix_h, prefix_position_ids, slots, starts, lengths):
        """JETSPEC_BATCH_DRAFT_GRAPH=1: the multi-slot cache's appends without its 16-row tile
        loop: a cold or stale prefix (start 0, more than one tile) in one projection padded to a
        64-row bucket, then every request's remaining rows (at most 16) in one tile projection
        and one scatter. The same per-row operations; other GEMM row counts, so not bitwise."""
        device = prefix_h.device
        for row, (slot, start, length) in enumerate(zip(slots, starts, lengths)):
            if start == 0 and length > 16:
                padded = ((length + 63) // 64) * 64
                keys, values = self._multi_project(
                    F.pad(prefix_h[row, :length], (0, 0, 0, padded - length)),
                    F.pad(prefix_position_ids[row, :length], (0, padded - length)))
                self._c1_multi_kv[0][:, slot, :, :length].copy_(keys[:, :length].transpose(1, 2))
                self._c1_multi_kv[1][:, slot, :, :length].copy_(values[:, :length].transpose(1, 2))
                starts[row] = length
        pending = [(row, slot, start, length) for row, (slot, start, length)
                   in enumerate(zip(slots, starts, lengths)) if start < length]
        if not pending:
            return
        count = prefix_h.shape[0]
        tile = prefix_h.new_zeros(count * 16, self.hspace_dim)
        source, target, slot_index, position_index = [], [], [], []
        for row, slot, start, length in pending:
            for j in range(length - start):
                source.append(row * prefix_h.shape[1] + start + j)
                target.append(row * 16 + j)
                slot_index.append(slot)
                position_index.append(start + j)
        indices = torch.tensor([source, target, slot_index, position_index], dtype=torch.long)
        indices = (indices.pin_memory().to(device, non_blocking=True) if device.type == "cuda"
                   else indices)
        tile.index_copy_(0, indices[1], prefix_h.reshape(-1, self.hspace_dim).index_select(0, indices[0]))
        positions = torch.zeros(count * 16, dtype=torch.long, device=device)
        positions.index_copy_(0, indices[1], indices[3])
        keys, values = self._multi_project(tile, positions)
        self._c1_multi_kv[0][:, indices[2], :, indices[3]] = keys[:, indices[1]].transpose(0, 1)
        self._c1_multi_kv[1][:, indices[2], :, indices[3]] = values[:, indices[1]].transpose(0, 1)

    def _cache_init(self, prefix_h):
        self._c1_capacity = int(os.environ.get("W259_C1_CACHE_CAPACITY", "768"))
        _run_manifest().record_value("c1_cache_capacity", self._c1_capacity)
        if self._c1_capacity <= 0:
            raise ValueError("C1 cache capacity must be positive")
        self._c1_blocks = [(index, layer) for index, stream in enumerate(self.streams)
            for layer in (stream, *self.extra_stream_layers[index])]
        window_capacity = block_geometry.window_rows()
        if os.environ.get('JETSPEC_REFRESH_FIXED', '0') == '1':
            cap = int(os.environ.get('JETSPEC_ROWS_RULE_CAP',
                                     os.environ.get('JETSPEC_ROWS_CAP', '16')))
            from jetspec.engines.jetspec_plus_plus.draft.serve_policy import rows_min_capacity
            cap = rows_min_capacity(cap)
            if cap not in (4, 8, 16):
                raise ValueError('fixed refresh requires JETSPEC_ROWS_RULE_CAP in 4,8,16')
            block = block_geometry.window_rows()
            window_capacity += ((cap * (block - 1) + 15) // 16) * 16
        if getattr(self, "target_text_config", None) is None and self._c1_fused2_enabled(prefix_h):
            layer = self._c1_blocks[0][1]
            shape = (
                len(self._c1_blocks), 1, layer.num_heads,
                self._c1_capacity + window_capacity, layer.head_dim,
            )
            self._c1_fast_key_cache = prefix_h.new_zeros(shape)
            self._c1_fast_value_cache = prefix_h.new_zeros(shape)
            self._c1_kv = [
                (self._c1_fast_key_cache[block], self._c1_fast_value_cache[block])
                for block in range(len(self._c1_blocks))
            ]
        else:
            self._c1_kv = [(prefix_h.new_zeros(1, layer.num_heads, self._c1_capacity + window_capacity, layer.head_dim),
                            prefix_h.new_zeros(1, layer.num_heads, self._c1_capacity + window_capacity, layer.head_dim))
                           for _, layer in self._c1_blocks]
        self._c1_prefix_key = None
        self.c1_state = dict(cache_hits=0, cache_rebuilds=0, cache_length=0,
            graph_replays=0, captures=0, extension_replays=0,
            extension_path="eager", query_path="eager")

    def _cache_project(self, prefix, positions, cos, sin):
        if getattr(self, "target_text_config", None) is not None:
            count, result = prefix.shape[1], []
            for index, layer in self._c1_blocks:
                source = layer.kv_norm(prefix[..., index*self.hidden_size:(index+1)*self.hidden_size])
                key, value = layer.project_kv(source)
                key = layer.k_norm(key.view(1, count, layer.num_key_value_heads,
                                           layer.head_dim)).transpose(1, 2)
                value = value.view(1, count, layer.num_key_value_heads, layer.head_dim).transpose(1, 2)
                key = _c1_rotary(key, cos, sin)
                result.append((key.repeat_interleave(layer.num_key_value_groups, dim=1),
                               value.repeat_interleave(layer.num_key_value_groups, dim=1)))
            return result
        # Keep D1's small appends on the same matrix kernel as the full reference;
        # a one-row GEMV can round differently from the training GEMM in fp32.
        if self.resolved_config.get("window_causal_over_tokens", False) and prefix.shape[1] < 16:
            count = prefix.shape[1]
            padded = F.pad(prefix, (0, 0, 0, 16 - count))
            projected = self._cache_project(padded, positions,
                F.pad(cos, (0, 0, 0, 16 - count)), F.pad(sin, (0, 0, 0, 16 - count)))
            return [(k[:, :, :count], v[:, :, :count]) for k, v in projected]
        if self._c1_fused2_enabled(prefix):
            key, value = self._cache_project_fused2(prefix, cos, sin)
            return [(key[block], value[block]) for block in range(key.shape[0])]
        if self._c1_fused_enabled() and prefix.dtype == torch.bfloat16:
            fused = self._c1_fused_module()
            result = []
            for index, layer in self._c1_blocks:
                source = fused.rms_norm(
                    prefix[..., index * self.hidden_size:(index + 1) * self.hidden_size],
                    layer.kv_norm.weight,
                    layer.kv_norm.variance_epsilon,
                )
                key, value = layer.project_kv(source)
                result.append(fused.fused_kv_postprocess(
                    key, value,
                    layer.k_norm.weight, cos, sin,
                    layer.k_norm.variance_epsilon, layer.num_key_value_heads,
                    layer.num_key_value_groups, layer.head_dim,
                ))
            return result
        count = prefix.shape[1]
        result = []
        for index, layer in self._c1_blocks:
            source = layer.kv_norm(prefix[..., index*self.hidden_size:(index+1)*self.hidden_size])
            key, value = layer.project_kv(source)
            k = layer.k_norm(key.view(1, count, layer.num_key_value_heads,
                                                     layer.head_dim)).transpose(1, 2)
            v = value.view(1, count, layer.num_key_value_heads, layer.head_dim).transpose(1, 2)
            k = _c1_rotary(k, cos, sin)
            if layer.num_key_value_groups != 1:
                k = k.repeat_interleave(layer.num_key_value_groups, dim=1)
                v = v.repeat_interleave(layer.num_key_value_groups, dim=1)
            result.append((k, v))
        return result

    def _d1_cache_project_batched(self, prefix, cos, sin):
        fused = self._c1_fused_module()
        self._c1_fused2_prepare()
        if not hasattr(self, "_d1_projection_weights"):
            layers = [layer for _, layer in self._c1_blocks]
            self._d1_projection_weights = ((None, None) if hasattr(layers[0], "_fused_kv") else (
                torch.stack([layer.k_proj.weight for layer in layers]),
                torch.stack([layer.v_proj.weight for layer in layers])))
            self._d1_projection_biases = (
                torch.stack([layer.k_proj.bias for layer in layers])[:, None]
                    if layers[0].k_proj.bias is not None else None,
                torch.stack([layer.v_proj.bias for layer in layers])[:, None]
                    if layers[0].v_proj.bias is not None else None)
        layer = self._c1_blocks[0][1]
        # JETSPEC_C1_PRELOOP: the same two Triton kernels through launch_cached (CUDA bf16)
        preloop = (os.environ.get("JETSPEC_C1_PRELOOP", "0") == "1"
                   and prefix.device.type == "cuda" and prefix.dtype == torch.bfloat16)
        norm = fused.lean_grouped_rms_norm if preloop else fused.fast_grouped_rms_norm
        # Both Triton kernels take the row count as a constexpr, so every new count is a
        # new ~0.1 s compile. A rebuild (a new prompt, or an append past 16 rows) has an
        # arbitrary count: run the two kernels on zero rows padded up to a 64-row bucket
        # (each program reads and writes only its own row, so the kept rows are the same
        # bits) and keep the GEMMs on the exact rows.
        tokens = prefix.shape[1]
        rows = _d1_rebuild_rows(tokens) if (prefix.device.type == "cuda"
                                             and prefix.dtype == torch.bfloat16) else tokens
        if rows != tokens:
            prefix = F.pad(prefix, (0, 0, 0, rows - tokens))
        source = norm(prefix, self._c1_fast_kv_norm, self._c1_fast_streams,
                      layer.kv_norm.variance_epsilon)
        if rows != tokens:
            source = source[:, :tokens].contiguous()
        if hasattr(layer, "_fused_kv"):
            widths = layer._fused_kv[2]
            projected = torch.bmm(source, layer._fused_kv_stack.transpose(1, 2))
            key, value = projected.split(widths, dim=-1)
            if self._d1_projection_biases[0] is not None:
                key = key + self._d1_projection_biases[0]
                value = value + self._d1_projection_biases[1]
        else:
            key = torch.bmm(source, self._d1_projection_weights[0].transpose(1, 2))
            value = torch.bmm(source, self._d1_projection_weights[1].transpose(1, 2))
            if self._d1_projection_biases[0] is not None:
                key = key + self._d1_projection_biases[0]
                value = value + self._d1_projection_biases[1]
        if rows != tokens:
            key = F.pad(key, (0, 0, 0, rows - tokens))
            value = F.pad(value, (0, 0, 0, rows - tokens))
            cos = F.pad(cos, (0, 0, 0, rows - tokens))
            sin = F.pad(sin, (0, 0, 0, rows - tokens))
        if preloop:
            key, value = fused.lean_batched_kv_postprocess(
                key, value, self._c1_fast_k_norm, cos.contiguous(), sin.contiguous(),
                layer.k_norm.variance_epsilon, layer.num_key_value_heads,
                layer.num_key_value_groups, layer.head_dim)
        else:
            key, value = fused.fast_batched_kv_postprocess(key, value, self._c1_fast_k_norm,
                cos, sin, layer.k_norm.variance_epsilon, layer.num_key_value_heads,
                layer.num_key_value_groups, layer.head_dim)
        if rows != tokens:
            key, value = key[:, :, :, :tokens], value[:, :, :, :tokens]
        return key, value

    def prewarm_rebuild_rows(self, like):
        """Compile ``_d1_cache_project_batched``'s bucketed rebuild kernels for every
        bucket up to the cache capacity, before any request is admitted (outputs unused)."""
        if (not self.resolved_config.get("window_causal_over_tokens", False)
                or like.device.type != "cuda" or like.dtype != torch.bfloat16
                or not hasattr(self, "_c1_fast_key_cache")
                or not self._c1_fused2_enabled(like)):
            return []
        buckets = sorted({_d1_rebuild_rows(count)
                          for count in range(17, self._c1_capacity + 1)})
        for rows in buckets:
            prefix = like.new_zeros(1, rows, self.hspace_dim)
            positions = torch.arange(rows, device=like.device)[None]
            cos, sin = self._c1_rope(prefix, positions)
            self._d1_cache_project_batched(prefix, cos, sin)
        return buckets

    def _cache_project_fused2(self, prefix, cos, sin):
        if self.resolved_config.get("window_causal_over_tokens", False):
            return self._d1_cache_project_batched(prefix, cos, sin)
        fused = self._c1_fused_module()
        keys, values = [], []
        for index, layer in self._c1_blocks:
            source = fused.rms_norm(
                prefix[..., index * self.hidden_size:(index + 1) * self.hidden_size],
                layer.kv_norm.weight,
                layer.kv_norm.variance_epsilon,
            )
            projected_key, projected_value = layer.project_kv(source)
            key, value = fused.fused_kv_postprocess(
                projected_key, projected_value,
                layer.k_norm.weight, cos, sin,
                layer.k_norm.variance_epsilon, layer.num_key_value_heads,
                layer.num_key_value_groups, layer.head_dim,
            )
            keys.append(key)
            values.append(value)
        return torch.stack(keys), torch.stack(values)

    def _d1_rotary_only(self, q, k, v, cos, sin, layer):
        count = q.shape[1]
        if q.device.type != "cuda" or q.dtype != torch.bfloat16:
            q = _c1_rotary(q.transpose(1, 2), cos, sin)
            k = _c1_rotary(k.transpose(1, 2), cos, sin)
            v = v.view(1, count, layer.num_key_value_heads, layer.head_dim).transpose(1, 2)
            if layer.num_key_value_groups != 1:
                k = k.repeat_interleave(layer.num_key_value_groups, dim=1)
                v = v.repeat_interleave(layer.num_key_value_groups, dim=1)
            return q, k, v
        fused = self._c1_fused_module()
        shape = (1, layer.num_heads, count, layer.head_dim)
        query, key, value = (q.new_empty(shape) for _ in range(3))
        cos, sin = cos.contiguous(), sin.contiguous()
        width = fused.triton.next_power_of_2(layer.head_dim)
        fused._q_rotary_kernel[(count * layer.num_heads,)](
            q.contiguous(), cos, sin, query, tokens=count, heads=layer.num_heads,
            head_dim=layer.head_dim, BLOCK=width, num_warps=4, enable_fp_fusion=False)
        fused._kv_rotary_repeat_kernel[(count * layer.num_heads,)](
            k.contiguous(), v.contiguous(), cos, sin, key, value, tokens=count,
            kv_heads=layer.num_key_value_heads, groups=layer.num_key_value_groups,
            head_dim=layer.head_dim, BLOCK=width, num_warps=4, enable_fp_fusion=False)
        return query, key, value

    def _d1_cache_query(self, embeddings, cos, sin, mask, length):
        """Refresh lower-stream window K/V while retaining the committed prefix."""
        if getattr(self, "target_text_config", None) is not None:
            return self._hybrid_cache_query(embeddings, cos, sin, mask, length)
        fused = (self._c1_fused_module() if embeddings.dtype == torch.bfloat16
                 and (self._c1_fused_enabled() or self._c1_fused2_enabled(embeddings))
                 else None)
        if (fused is not None and embeddings.device.type == "cuda"
                and os.environ.get("JETSPEC_C1_LEAN", "0") == "1"):
            return self._d1_cache_query_lean(fused, embeddings, cos, sin, mask, length)
        # The window holds Q = 15 (14 drafts) or 16 (15 drafts) query rows
        Q = int(embeddings.shape[1])
        u, outputs = embeddings, []
        block = 0
        for index, stream in enumerate(self.streams):
            window = u
            for layer in (stream, *self.extra_stream_layers[index]):
                if fused is None:
                    normed = layer.input_norm(u)
                    q = layer.q_norm(layer.q_proj(normed).view(
                        1, Q, layer.num_heads, layer.head_dim))
                    source = layer.kv_norm(window)
                    projected_key, projected_value = layer.project_kv(source)
                    wk = layer.k_norm(projected_key.view(
                        1, Q, layer.num_key_value_heads, layer.head_dim))
                    wv = projected_value
                    q, wk, wv = self._d1_rotary_only(q, wk, wv, cos, sin, layer)
                else:
                    normed = fused.rms_norm(
                        u, layer.input_norm.weight, layer.input_norm.variance_epsilon)
                    q = fused.fused_q_postprocess(
                        layer.q_proj(normed), layer.q_norm.weight, cos, sin,
                        layer.q_norm.variance_epsilon, layer.num_heads, layer.head_dim)
                    source = fused.rms_norm(
                        window, layer.kv_norm.weight, layer.kv_norm.variance_epsilon)
                    wk = wv = None
                    if embeddings.dtype == torch.bfloat16 and embeddings.device.type == "cuda":
                        # The same k norm and rotary arithmetic, written straight
                        # into the reserved tail slots (no separate window buffers).
                        kv_rotary_into_cache = self._rows_module().kv_rotary_into_cache
                        k, v = self._c1_kv[block]
                        count = layer.num_key_value_heads * layer.head_dim
                        projected_key, projected_value = layer.project_kv(source)
                        normalized_key = fused.rms_norm(
                            projected_key.view(1, Q, layer.num_key_value_heads, layer.head_dim),
                            layer.k_norm.weight, layer.k_norm.variance_epsilon).view(Q, count)
                        kv_rotary_into_cache(
                            normalized_key, projected_value.contiguous().view(Q, count), cos[0], sin[0],
                            k[0, :, length:length + Q].transpose(0, 1),
                            v[0, :, length:length + Q].transpose(0, 1),
                            kv_heads=layer.num_key_value_heads,
                            groups=layer.num_key_value_groups, head_dim=layer.head_dim)
                    else:
                        projected_key, projected_value = layer.project_kv(source)
                        wk, wv = fused.fused_kv_postprocess(
                            projected_key, projected_value,
                            layer.k_norm.weight, cos, sin, layer.k_norm.variance_epsilon,
                            layer.num_key_value_heads, layer.num_key_value_groups, layer.head_dim)
                k, v = self._c1_kv[block]
                if embeddings.dtype != torch.bfloat16:
                    keys = torch.cat((k[:, :, :length], wk), dim=2)
                    values = torch.cat((v[:, :, :length], wv), dim=2)
                else:
                    # The cache reserves 16 tail slots. A later prefix extension
                    # overwrites these ephemeral window rows before reading them.
                    if wk is not None:
                        k[:, :, length:length + Q].copy_(wk)
                        v[:, :, length:length + Q].copy_(wv)
                    keys = k[:, :, :length + Q]
                    values = v[:, :, :length + Q]
                attended = F.scaled_dot_product_attention(
                    q, keys, values, attn_mask=mask, scale=layer.scaling)
                attended = attended.transpose(1, 2).reshape(1, Q, -1)
                state = u + layer.o_proj(attended)
                if fused is None:
                    u = state + layer.mlp(layer.post_attention_norm(state))
                else:
                    post = fused.rms_norm(state, layer.post_attention_norm.weight,
                                          layer.post_attention_norm.variance_epsilon)
                    u = state + layer.mlp(post)
                block += 1
            outputs.append(u)
        return torch.cat(outputs, dim=-1)

    def _hybrid_cache_query(self, embeddings, cos, sin, mask, length,
                            window_slots=None, stop_before_attention=False, resume=None):
        """Gated/partial-RoPE fallback, including the chain's split first attention.

        The Qwen3 kernels assume ordinary norm weights, ungated Q and full RoPE.
        Module arithmetic here matches training and uses device index copies for
        live cache slots; no device scalar is read by Python during capture.
        """
        if os.environ.get("JETSPEC_HYBRID_C1_FUSED", "0") == "1":
            return self._c1_fused_module().lean_variant_query(
                self, embeddings, cos, sin, mask, length, window_slots,
                stop_before_attention, resume)
        count = embeddings.shape[1]
        lean = None
        if os.environ.get("JETSPEC_HYBRID_C1_LEAN", "0") == "1":
            import jetspec.engines.jetspec_plus_plus.draft.hybrid_c1_lean as lean
        live = window_slots is not None
        total = mask.shape[-1] if live else length + (
            count if self.resolved_config.get("window_causal_over_tokens", False) else 0)
        u, outputs = (embeddings, []) if resume is None else (resume["u"], list(resume["outputs"]))
        window = u if resume is None else resume["window"]
        first = 0 if resume is None else resume["block"]
        blocks = [(index, layer) for index, stream in enumerate(self.streams)
                  for layer in (stream, *self.extra_stream_layers[index])]
        for block, (index, layer) in enumerate(blocks):
            if block < first:
                continue
            if resume is not None and block == first:
                q, gate = resume["q"], resume["gate"]
            else:
                if block == 0 or index != blocks[block - 1][0]:
                    window = u
                normed = layer.input_norm(u) if lean is None else lean.norm(u, layer.input_norm)
                projected = layer.q_proj(normed).view(
                    1, count, layer.num_heads,
                    layer.head_dim * (2 if layer.attn_output_gate else 1))
                gate = None
                if layer.attn_output_gate:
                    projected, gate = projected.chunk(2, dim=-1)
                    if lean is None:
                        gate = gate.reshape(1, count, -1)
                q = (_c1_rotary(layer.q_norm(projected).transpose(1, 2), cos, sin)
                     if lean is None else lean.norm_rotary(projected, layer.q_norm, cos, sin))
                if self.resolved_config.get("window_causal_over_tokens", False):
                    source = layer.kv_norm(window) if lean is None else lean.norm(window, layer.kv_norm)
                    key, value = layer.project_kv(source)
                    k, v = self._c1_kv[block]
                    if lean is not None:
                        slots = (window_slots if live else
                                 torch.arange(length, total, device=u.device, dtype=torch.long))
                        lean.write_kv(key.view(1, count, layer.num_key_value_heads, layer.head_dim),
                                      value, layer.k_norm, cos, sin, (k, v), slots,
                                      layer.num_key_value_groups)
                    else:
                        key = layer.k_norm(key.view(1, count, layer.num_key_value_heads,
                                                   layer.head_dim)).transpose(1, 2)
                        value = value.view(1, count, layer.num_key_value_heads, layer.head_dim).transpose(1, 2)
                        key = _c1_rotary(key, cos, sin).repeat_interleave(layer.num_key_value_groups, dim=1)
                        value = value.repeat_interleave(layer.num_key_value_groups, dim=1)
                        if live:
                            k.index_copy_(2, window_slots, key)
                            v.index_copy_(2, window_slots, value)
                        else:
                            k[:, :, length:total].copy_(key)
                            v[:, :, length:total].copy_(value)
                if stop_before_attention:
                    if block != 0 or resume is not None:
                        raise ValueError("split C1 can suspend only before its first attention")
                    return dict(u=u, outputs=outputs, window=window, block=block, q=q, gate=gate)
            k, v = self._c1_kv[block]
            attended = F.scaled_dot_product_attention(q, k[:, :, :total], v[:, :, :total],
                                                      attn_mask=mask, scale=layer.scaling)
            attended = attended.transpose(1, 2).reshape(1, count, -1)
            if gate is not None:
                attended = (attended * torch.sigmoid(gate) if lean is None else
                            lean.gate_mul(attended, gate, layer.num_heads, layer.head_dim))
            state = u + layer.o_proj(attended)
            if lean is None:
                u = state + layer.mlp(layer.post_attention_norm(state))
            else:
                u = state + lean.mlp(lean.norm(state, layer.post_attention_norm), layer.mlp)
            if block == len(blocks) - 1 or blocks[block + 1][0] != index:
                outputs.append(u)
        return torch.cat(outputs, dim=-1)

    def _d1_lean_plan(self):
        """Per-block weights and constants bound once (JETSPEC_C1_LEAN)."""
        plan = getattr(self, "_d1_lean_blocks", None)
        if plan is None:
            plan = []
            for index, stream in enumerate(self.streams):
                for layer in (stream, *self.extra_stream_layers[index]):
                    linears = (layer.q_proj, layer.k_proj, layer.v_proj, layer.o_proj,
                               layer.mlp.gate_proj, layer.mlp.up_proj, layer.mlp.down_proj)
                    if any(linear.bias is not None for linear in linears):
                        raise ValueError("C1 LEAN requires bias-free projections")
                    if type(layer.mlp.act_fn).__name__ not in ("SiLU", "SiLUActivation"):
                        raise ValueError("C1 LEAN requires the silu MLP")
                    plan.append((
                        index, layer.input_norm.weight, layer.input_norm.variance_epsilon,
                        layer.q_proj.weight, layer.q_norm.weight, layer.q_norm.variance_epsilon,
                        layer.kv_norm.weight, layer.kv_norm.variance_epsilon,
                        layer.k_proj.weight, layer.k_norm.weight, layer.k_norm.variance_epsilon,
                        layer.v_proj.weight, layer.o_proj.weight,
                        layer.post_attention_norm.weight, layer.post_attention_norm.variance_epsilon,
                        layer.mlp.gate_proj.weight, layer.mlp.up_proj.weight,
                        layer.mlp.down_proj.weight, layer.num_heads, layer.num_key_value_heads,
                        layer.num_key_value_groups, layer.head_dim, layer.scaling))
            self._d1_lean_blocks = plan
        return plan

    def _d1_lean_loop(self, fused, norm, add_norm, silu_mul, rows, cos, sin, cos0, sin0, bias,
                      length, keys_total):
        """JETSPEC_C1_LOOP=1: the loop below (its folded, eager branch) in C++
        (``c1_fused._loop_extension``): the same ops and kernels on the same operands.
        None until both Triton rotary specializations are compiled (the Python loop
        compiles them), or when an operand is misaligned for them."""
        plan = self._d1_lean_plan()
        packed = getattr(self, "_d1_loop_packed", None)
        if packed is None or packed[0] is not self._c1_kv:
            weights, scalars, ints = [], [], []
            for (index, in_w, in_eps, q_w, qn_w, qn_eps, kvn_w, kvn_eps, k_w, kn_w, kn_eps, v_w,
                 o_w, post_w, post_eps, gate_w, up_w, down_w, heads, kv_heads, groups, head_dim,
                 scaling) in plan:
                weights += [in_w, q_w, qn_w, kvn_w, k_w, kn_w, v_w, o_w, post_w, gate_w, up_w, down_w]
                scalars += [in_eps, qn_eps, kvn_eps, kn_eps, post_eps, scaling]
                ints += [index, heads, kv_heads, groups, head_dim]
            caches = [tensor for pair in self._c1_kv for tensor in pair]
            packed = self._d1_loop_packed = (self._c1_kv, weights, scalars, ints, caches)
        _, weights, scalars, ints, caches = packed
        heads, kv_heads, groups, head_dim = ints[1:5]
        block = fused.triton.next_power_of_2(head_dim)
        device = rows.device.index
        # JETSPEC_C1_PRELOOP: the two compiled entries, looked up once per cache (their
        # key is the device, the constexprs and 16-byte alignment, which the loop checks
        # on every operand before it launches)
        preloop = os.environ.get("JETSPEC_C1_PRELOOP", "0") == "1"
        entries = getattr(self, "_d1_loop_entries", None)
        runtime = fused.triton.knobs.runtime
        if os.environ.get("JETSPEC_C1_ROPE_NORM", "0") == "1" and head_dim == 128:
            # the fused head-norm rotary kernels replace both Triton launches (and the Python
            # loop no longer compiles them); the loop refuses to fall back without them
            q_entry = kv_entry = ()
        elif (preloop and entries is not None and entries[0] is self._c1_kv
                and entries[1] == device and not fused._has_launch_hooks(runtime)):
            q_entry, kv_entry = entries[2:]
        else:
            rows_module = self._rows_module()
            options = dict(num_warps=4, enable_fp_fusion=False)
            q_entry = fused.compiled_launcher(
                fused._q_rotary_kernel, device,
                (cos, cos, sin, cos, rows.shape[0], heads, head_dim, block),
                **options)
            kv_entry = fused.compiled_launcher(
                rows_module._kv_rotary_into_cache_kernel, device,
                (cos0, cos0, cos0, sin0, cos0, cos0, rows.shape[0], kv_heads, groups, head_dim,
                 kv_heads * head_dim, head_dim, caches[0].stride(1), block), **options)
            if q_entry is None or kv_entry is None:
                return None
            if preloop:
                self._d1_loop_entries = (self._c1_kv, device, q_entry, kv_entry)
        kv_norm_stack = None
        if os.environ.get("JETSPEC_C1_KVNORM_BATCH", "0") == "1":
            # the stacked kv_norm weights, one row per block in plan order
            self._c1_fused2_prepare()
            kv_norm_stack = self._c1_fast_kv_norm
        native = (0, 0, 0)
        if os.environ.get("JETSPEC_C1_NATIVE", "0") == "1":
            native = (fused.native_entry(fused._wide_norm_extension(), "rms_norm"),
                      fused.native_entry(fused._lean_ops_extension(), "add_rms_norm"),
                      fused.native_entry(fused._silu_mul_extension(), "silu_mul"))
        try:
            return fused._loop_extension().d1_lean_loop(
                rows, cos, sin, cos0, sin0, bias, length, keys_total, weights, scalars, ints,
                caches, norm, add_norm, silu_mul, q_entry, kv_entry, block,
                os.environ.get("JETSPEC_C1_GEMM", "aten") == "direct", *native,
                os.environ.get("JETSPEC_C1_ROPE_NORM", "0") == "1", kv_norm_stack,
                rows.shape[0])
        except RuntimeError as error:
            if ("not 16-byte aligned" not in str(error)
                    and "needs the Triton rotary entries" not in str(error)):
                raise
            return None

    def _d1_cache_query_lean(self, fused, embeddings, cos, sin, mask, length,
                             window_slots=None, allow_flashinfer=True,
                             stop_before_attention=False, resume=None,
                             chain_length=None, append=None):
        """``_d1_cache_query``'s fused CUDA branch with the host work cut down.

        The same kernels on the same operands, so bitwise the same output: the
        projections call ``F.linear`` on the bound weights (what ``nn.Linear`` and
        ``Qwen3MLP`` call), the norms call the selected certified norm extension
        directly (every shape here takes its CUDA path), and attention defaults to the
        memory-efficient kernel that ``scaled_dot_product_attention`` selects for
        this float-masked shape, with the mask padded once per round the way its
        preprocessing pads it inside each of the 15 calls.

        ``window_slots`` (the graphed live layout, JETSPEC_C1_GRAPH_LAYOUT=live): the
        window rows go to the cache slots that device tensor names (the eager
        layout's ``length .. length+15``; the cache-write kernel reads its first
        entry as the row offset), and attention reads the first ``mask.shape[-1]``
        cache slots, whose mask hides every slot past the live prefix and the window.
        """
        if getattr(self, "target_text_config", None) is not None:
            if append is not None:
                raise ValueError("hybrid C1 requires the unfused prefix append")
            return self._hybrid_cache_query(embeddings, cos, sin, mask, length,
                window_slots=window_slots, stop_before_attention=stop_before_attention, resume=resume)
        # Q = 15 (14 drafts) or 16 (15 drafts) window rows
        Q = int(embeddings.shape[1])
        norm = (fused._wide_norm_extension() if os.environ.get("JETSPEC_C1_NORM") == "wide"
                else fused._norm_extension()).rms_norm
        rows = self._rows_module()
        kv_kernel, launch = rows._kv_rotary_into_cache_kernel, fused.launch_cached
        at_kernel = rows._kv_rotary_into_cache_at_kernel
        cos, sin = cos.contiguous(), sin.contiguous()
        if window_slots is not None:
            keys_total = mask.shape[-1]
            if (tuple(mask.shape[:3]) != (1, 1, Q) or keys_total % 16
                    or mask.dtype != embeddings.dtype or tuple(window_slots.shape) != (Q,)):
                raise ValueError("C1 LEAN live layout requires a [1,1,Q,16k] float mask")
        else:
            keys_total = length + Q
        if mask is None:
            mask = embeddings.new_zeros((1, 1, Q, keys_total))
        if tuple(mask.shape) != (1, 1, Q, keys_total) or mask.dtype != embeddings.dtype:
            raise ValueError("C1 LEAN requires the [1,1,Q,prefix+Q] float mask")
        use_len_attention = (chain_length is not None and embeddings.is_cuda
                             and os.environ.get("JETSPEC_C1_ATTN_LEN", "0") == "1")
        staged = getattr(self, "_d1_preloop_mask", None)
        if resume is not None:
            aligned = resume["aligned"]
        elif (window_slots is None and staged is not None and mask is staged[0]
                and staged[1].shape[-1] == (keys_total + 15) // 16 * 16):
            aligned = staged[1]  # JETSPEC_C1_PRELOOP wrote it in this layout
        elif use_len_attention:
            aligned = None
        else:
            aligned = embeddings.new_zeros((1, 1, Q, (keys_total + 15) // 16 * 16))
            aligned[..., :keys_total].copy_(mask)
        cos0, sin0 = cos[0], sin[0]
        triton_power = fused.triton.next_power_of_2
        from jetspec.engines.jetspec_plus_plus.kernels.attn_sm100 import attend; from jetspec.engines.jetspec_plus_plus.kernels.attn_sm100 import attend_c1; from jetspec.engines.jetspec_plus_plus.kernels.attn_sm100 import flashinfer_enabled; from jetspec.engines.jetspec_plus_plus.kernels.attn_sm100 import prepare_prefix
        # Rows as [Q, width]: F.linear folds a [1, Q, width] input to the same
        # [Q, width] matrix product, and the norm extension reads the same rows.
        u = embeddings.reshape(Q, embeddings.shape[-1]) if resume is None else resume["u"]
        outputs = [] if resume is None else list(resume["outputs"])
        window, current = ((u, 0) if resume is None else
                           (resume["window"], resume["current"]))
        # The residual adds ride in the next norm's launch and silu(gate) * up is one
        # launch (c1_fused's lean ops: torch's add and silu/mul arithmetic, the
        # certified norm); an add whose sum no norm reads stays a torch add.
        fold = os.environ.get("JETSPEC_C1_NORM") == "wide"
        gate_up_pack = os.environ.get("JETSPEC_C1_GATEUP_PACKED", "0") == "1"
        gate_up_linear = F.linear
        if os.environ.get("JETSPEC_ADAPTER_GEMM_TUNE", "0") == "1":
            if not hasattr(self, "_adapter_gemm"):
                from jetspec.engines.jetspec_plus_plus.kernels.adapter_gemm import AdapterGemm

                self._adapter_gemm = AdapterGemm(embeddings.device)
            gate_up_linear = self._adapter_gemm.linear
        add_norm = fused._lean_ops_extension().add_rms_norm if fold else None
        silu_mul = fused._silu_mul_extension().silu_mul if fold else None
        span_fused = (os.environ.get("JETSPEC_C1_SPAN_KV_FUSED", "0") == "1"
                      and hasattr(self, "_c1_span_kv"))
        if resume is not None:
            append = resume.get("append")
        ew_fused = (os.environ.get("JETSPEC_ADAPTER_EW_FUSED", "0") == "1"
                    or os.environ.get("JETSPEC_C1_GLUE_FUSED", "0") == "1")
        if append is not None and not ew_fused and resume is None:
            layer = self._c1_blocks[0][1]
            source = fused.fast_grouped_rms_norm(
                append[0], self._c1_fast_kv_norm, self._c1_fast_streams,
                layer.kv_norm.variance_epsilon)
            append = (source, *append[1:])
        if (window_slots is None and fold and Q in (15, 16)
                and os.environ.get("JETSPEC_C1_LOOP", "0") == "1"
                and os.environ.get("JETSPEC_C1_GLUON", "0") != "1"
                and os.environ.get("JETSPEC_FUSED_QKV", "0") != "1"
                and not span_fused
                and not flashinfer_enabled(embeddings.device, embeddings.dtype)):
            # the C++ loop handles both served window sizes.
            looped = self._d1_lean_loop(fused, norm, add_norm, silu_mul, u, cos, sin, cos0, sin0,
                                        aligned[..., :keys_total], length, keys_total)
            if looped is not None:
                return looped
        pending = None if resume is None else resume["pending"]
        # JETSPEC_C1_ROPE_NORM / JETSPEC_C1_KVNORM_BATCH in this loop too (the graphed live
        # layout runs it): the loop extension's kernels, bitwise the ones they replace
        rope_norm = os.environ.get("JETSPEC_C1_ROPE_NORM", "0") == "1"
        qk_fused = os.environ.get("JETSPEC_C1_QK_FUSED", "0") == "1"
        query_layout = os.environ.get("JETSPEC_ADAPTER_QUERY_LAYOUT_FUSED", "0") == "1"
        kv_stack = None
        if span_fused or os.environ.get("JETSPEC_C1_KVNORM_BATCH", "0") == "1":
            self._c1_fused2_prepare()
            kv_stack = self._c1_fast_kv_norm
        loop_ext = fused._loop_extension() if rope_norm or kv_stack is not None else None
        head_ops = loop_ext
        if rope_norm and os.environ.get("JETSPEC_C1_GLUON", "0") == "1":
            import jetspec.engines.jetspec_plus_plus.kernels.c1_gluon as c1_gluon
            if c1_gluon.available(u):
                head_ops = c1_gluon
        plan = self._d1_lean_plan()
        flash_plan = None
        if allow_flashinfer and flashinfer_enabled(
                embeddings.device, embeddings.dtype, chain=window_slots is not None):
            if window_slots is None:
                flash_plan = prepare_prefix(self, "c1", Q, self._c1_kv[0][0], length,
                                            plan[0][-1])
            else:
                flash_plan = self._flashinfer_c1_plan
        kv_sources, kv_first = ((None, None) if resume is None else
                                (resume["kv_sources"], resume["kv_first"]))
        span_projected = None if resume is None else resume.get("span_projected")
        if resume is not None:
            q, k, v, scaling = (resume[name] for name in ("q", "k", "v", "scaling"))
            (o_w, post_w, post_eps, gate_w, up_w, down_w) = (
                resume[name] for name in
                ("o_w", "post_w", "post_eps", "gate_w", "up_w", "down_w"))
            if use_len_attention:
                from jetspec.engines.jetspec_plus_plus.kernels.chain.len_attention import attend as attend_len
                attended = attend_len(q, k[:, :, :keys_total], v[:, :, :keys_total],
                                      chain_length, scaling, mode="c1")
            elif flash_plan is None:
                attended = attend(
                    q, k[:, :, :keys_total], v[:, :, :keys_total],
                    aligned[..., :keys_total].expand(1, q.shape[1], Q, keys_total),
                    scaling)
            else:
                attended = attend_c1(q, k[:, :, :keys_total], v[:, :, :keys_total],
                                     scaling, flash_plan)
            projected = F.linear(attended.transpose(1, 2).reshape(Q, -1), o_w)
            if fold:
                post, state = add_norm(projected, u, post_w, post_eps)
                if gate_up_pack:
                    gate, up = gate_up_linear(post, self._refresh_gate_up_weights[0]).chunk(2, dim=-1)
                    pending = (F.linear(silu_mul(gate, up), down_w), state)
                else:
                    pending = (F.linear(silu_mul(F.linear(post, gate_w), F.linear(post, up_w)),
                                        down_w), state)
            else:
                state = u + projected
                post = norm(state, post_w, post_eps)
                u = state + F.linear(F.silu(F.linear(post, gate_w)) * F.linear(post, up_w), down_w)
        blocks = enumerate(plan) if resume is None else enumerate(plan[1:], start=1)
        for block, (index, in_w, in_eps, q_w, qn_w, qn_eps, kvn_w, kvn_eps, k_w, kn_w, kn_eps,
                    v_w, o_w, post_w, post_eps, gate_w, up_w, down_w, heads, kv_heads,
                    groups, head_dim, scaling) in blocks:
            new_kv_group = kv_first is None or plan[kv_first][0] != index
            ew_source = None
            if ew_fused and (kv_stack is None or new_kv_group):
                span = 1
                if kv_stack is not None:
                    while (block + span < len(plan) and plan[block + span][0] == index
                           and plan[block + span][7] == kvn_eps):
                        span += 1
                kv_weights = (kvn_w.unsqueeze(0) if kv_stack is None else
                              kv_stack[block:block + span])
                normed, u, pre_sources = fused.ew_pre_norm(
                    u, window, in_w, kv_weights, in_eps, kvn_eps, pending,
                    kv_from_state=index != current,
                    append=(append[0], self._c1_fast_streams, block)
                    if append is not None else None)
                if kv_stack is None:
                    ew_source = pre_sources[0]
                else:
                    kv_sources, kv_first = pre_sources, block
            elif pending is None:
                normed = norm(u, in_w, in_eps)
            else:
                normed, u = add_norm(pending[0], pending[1], in_w, in_eps)
            if index != current:
                outputs.append(u)
                window, current = u, index
            fused_rope = rope_norm and head_dim == 128
            if fused_rope:
                projected_q = F.linear(normed, q_w)
                q = (fused.query_output(projected_q, heads, Q, head_dim) if qk_fused and query_layout
                     else projected_q.new_empty((1, heads, Q, head_dim)))
                if not qk_fused:
                    head_ops.q_norm_rotary(projected_q, qn_w, cos, sin, q, Q, heads, qn_eps)
            else:
                q = fused.lean_q_postprocess(F.linear(normed, q_w).view(1, Q, -1),
                                             qn_w, cos, sin, qn_eps, heads, head_dim, norm)
            source = ew_source
            if kv_stack is not None:
                if not ew_fused and new_kv_group:
                    span = 1
                    while (block + span < len(plan) and plan[block + span][0] == index
                           and plan[block + span][7] == kvn_eps):
                        span += 1
                    kv_sources = loop_ext.rms_norm_wide_rows(window, kv_stack[block:block + span],
                                                             kvn_eps)
                    kv_first = block if kv_sources is not None else None
                if kv_first is not None and block - kv_first < kv_sources.shape[0]:
                    source = kv_sources[block - kv_first]
                    if append is not None and ew_fused:
                        source = source[append[0].shape[1]:]
            if span_fused and kv_first is not None and (new_kv_group or span_projected is None):
                weights, _ = self._c1_span_kv
                count = kv_sources.shape[0]
                source_rows = kv_sources
                if append is not None and not ew_fused:
                    source_rows = torch.cat((append[0][kv_first:kv_first + count], kv_sources), dim=1)
                span_projected = torch.bmm(source_rows,
                    weights[kv_first:kv_first + count].transpose(1, 2))
                if append is not None:
                    layer = self._c1_blocks[0][1]
                    width = layer.num_key_value_heads * layer.head_dim
                    appended_key, appended_value = span_projected[:, :append[0].shape[1]].split(
                        width, dim=-1)
                    fused.folded_append_kv_into_cache(
                        appended_key, appended_value,
                        self._c1_fast_k_norm[kv_first:kv_first + count],
                        append[1], append[2], append[3],
                        self._c1_fast_key_cache[kv_first:kv_first + count],
                        self._c1_fast_value_cache[kv_first:kv_first + count],
                        layer.k_norm.variance_epsilon, layer.num_key_value_heads,
                        layer.num_key_value_groups, layer.head_dim)
            if source is None:
                source = norm(window, kvn_w, kvn_eps)
            k, v = self._c1_kv[block]
            width = kv_heads * head_dim
            # rows.kv_rotary_into_cache's CUDA launch on the same operands.
            if window_slots is None:
                key_out = k[0, :, length:keys_total].transpose(0, 1)
                value_out = v[0, :, length:keys_total].transpose(0, 1)
            else:
                # rows from slot 0; the kernel offsets them by window_slots[0]
                key_out = k[0, :, :Q].transpose(0, 1)
                value_out = v[0, :, :Q].transpose(0, 1)
            packed_kv = getattr(self._c1_blocks[block][1], "_fused_kv", None)
            if span_fused and span_projected is not None:
                projected = span_projected[block - kv_first]
                if append is not None:
                    projected = projected[append[0].shape[1]:]
                projected_key, projected_value = projected.split(width, dim=-1)
                if not fused_rope:
                    projected_key, projected_value = projected_key.contiguous(), projected_value.contiguous()
            elif packed_kv is None:
                projected_key, projected_value = F.linear(source, k_w), F.linear(source, v_w)
            else:
                projected_key, projected_value = F.linear(source, packed_kv[0], packed_kv[1]).split(
                    packed_kv[2], dim=-1)
                if not (fused_rope and os.environ.get("JETSPEC_C1_KV_SPLIT_LEAN", "0") == "1"):
                    projected_key, projected_value = projected_key.contiguous(), projected_value.contiguous()
            if fused_rope:
                if (projected_value.stride() != projected_key.stride()
                        or key_out.stride() != value_out.stride() or key_out.stride(2) != 1):
                    raise ValueError("C1 LEAN cache write requires matching dense operands")
                if qk_fused:
                    loop_ext.qk_norm_rotary_cache(
                        projected_q, projected_key, projected_value, qn_w, kn_w, cos, sin,
                        q, key_out, value_out, Q, heads, kv_heads, groups,
                        qn_eps, kn_eps, window_slots)
                else:
                    head_ops.k_norm_rotary_cache(projected_key, projected_value, kn_w, cos0, sin0,
                                                 key_out, value_out, Q, kv_heads, groups, kn_eps,
                                                 window_slots)
            else:
                normalized_key = norm(projected_key.view(Q, kv_heads, head_dim), kn_w,
                                      kn_eps).view(Q, width)
                if (projected_value.stride() != normalized_key.stride()
                        or key_out.stride() != value_out.stride() or key_out.stride(2) != 1):
                    raise ValueError("C1 LEAN cache write requires matching dense operands")
            if fused_rope:
                pass  # the fused kernel wrote the key and value rows
            elif window_slots is None:
                launch(kv_kernel, (Q * heads,),
                       (normalized_key, projected_value, cos0, sin0, key_out, value_out, Q,
                        kv_heads, groups, head_dim, normalized_key.stride(0), key_out.stride(0),
                        key_out.stride(1), triton_power(head_dim)),
                       num_warps=4, enable_fp_fusion=False)
            else:
                launch(at_kernel, (Q * heads,),
                       (normalized_key, projected_value, cos0, sin0, key_out, value_out,
                        window_slots, Q, kv_heads, groups, head_dim, normalized_key.stride(0),
                        key_out.stride(0), key_out.stride(1), triton_power(head_dim)),
                       num_warps=4, enable_fp_fusion=False)
            if stop_before_attention:
                if block != 0 or resume is not None:
                    raise ValueError("split C1 can suspend only before its first attention")
                state = dict(aligned=aligned, u=u, window=window, current=current,
                            outputs=outputs, pending=pending,
                            kv_sources=kv_sources, kv_first=kv_first,
                            span_projected=span_projected,
                            q=q, k=k, v=v, scaling=scaling,
                            o_w=o_w, post_w=post_w, post_eps=post_eps,
                            gate_w=gate_w, up_w=up_w, down_w=down_w)
                if append is not None:
                    state["append"] = append
                return state
            if use_len_attention:
                from jetspec.engines.jetspec_plus_plus.kernels.chain.len_attention import attend as attend_len
                attended = attend_len(q, k[:, :, :keys_total], v[:, :, :keys_total],
                                      chain_length, scaling, mode="c1")
            elif flash_plan is None:
                attended = attend(
                    q, k[:, :, :keys_total], v[:, :, :keys_total],
                    aligned[..., :keys_total].expand(1, heads, Q, keys_total),
                    scaling)
            else:
                attended = attend_c1(q, k[:, :, :keys_total], v[:, :, :keys_total],
                                     scaling, flash_plan)
            projected = F.linear(attended.transpose(1, 2).reshape(Q, -1), o_w)
            if fold:
                post, state = add_norm(projected, u, post_w, post_eps)
                if gate_up_pack:
                    gate, up = gate_up_linear(post, self._refresh_gate_up_weights[block]).chunk(2, dim=-1)
                    pending = (F.linear(silu_mul(gate, up), down_w), state)
                else:
                    pending = (F.linear(silu_mul(F.linear(post, gate_w), F.linear(post, up_w)),
                                        down_w), state)
            else:
                state = u + projected
                post = norm(state, post_w, post_eps)
                u = state + F.linear(F.silu(F.linear(post, gate_w)) * F.linear(post, up_w), down_w)
        if pending is not None and os.environ.get("JETSPEC_C1_GLUE_FUSED", "0") == "1":
            return fused.finish_streams(outputs, pending[1], pending[0]).view(1, Q, -1)
        if pending is not None:
            u = pending[1] + pending[0]
        outputs.append(u)
        return torch.cat(outputs, dim=-1).view(1, Q, -1)

    def _cache_query(self, embeddings, cos, sin, mask, length):
        if getattr(self, "target_text_config", None) is not None:
            return self._hybrid_cache_query(embeddings, cos, sin, mask, length)
        if self.resolved_config.get("window_causal_over_tokens", False):
            return self._d1_cache_query(embeddings, cos, sin, mask, length)
        if ((self._c1_fused_enabled() or self._c1_fused2_enabled(embeddings))
                and embeddings.dtype == torch.bfloat16):
            fused = self._c1_fused_module()
            u, outputs = embeddings, []
            block = 0
            for index, stream in enumerate(self.streams):
                for layer in (stream, *self.extra_stream_layers[index]):
                    normed = fused.rms_norm(
                        u, layer.input_norm.weight, layer.input_norm.variance_epsilon
                    )
                    q = fused.fused_q_postprocess(
                        F.linear(normed, layer.q_proj.weight, layer.q_proj.bias),
                        layer.q_norm.weight, cos, sin, layer.q_norm.variance_epsilon,
                        layer.num_heads, layer.head_dim,
                    )
                    k, v = self._c1_kv[block]
                    attended = F.scaled_dot_product_attention(
                        q, k[:, :, :length], v[:, :, :length],
                        attn_mask=mask, scale=layer.scaling,
                    )
                    attended = attended.transpose(1, 2).reshape(
                        1, 15, layer.num_heads * layer.head_dim
                    )
                    state = u + layer.o_proj(attended)
                    post = fused.rms_norm(
                        state, layer.post_attention_norm.weight,
                        layer.post_attention_norm.variance_epsilon,
                    )
                    hidden = layer.mlp.act_fn(layer.mlp.gate_proj(post)) * layer.mlp.up_proj(post)
                    u = state + layer.mlp.down_proj(hidden)
                    block += 1
                outputs.append(u)
            return torch.cat(outputs, dim=-1)
        u, outputs = embeddings, []
        block = 0
        for index, stream in enumerate(self.streams):
            for layer in (stream, *self.extra_stream_layers[index]):
                normed = layer.input_norm(u)
                q = layer.q_norm(layer.q_proj(normed).view(1, 15, layer.num_heads,
                                                          layer.head_dim)).transpose(1, 2)
                q = _c1_rotary(q, cos, sin)
                k, v = self._c1_kv[block]
                attended = F.scaled_dot_product_attention(q, k[:, :, :length], v[:, :, :length],
                    attn_mask=mask, scale=layer.scaling)
                attended = attended.transpose(1, 2).reshape(1, 15, layer.num_heads*layer.head_dim)
                state = u + layer.o_proj(attended)
                u = state + layer.mlp(layer.post_attention_norm(state))
                block += 1
            outputs.append(u)
        return torch.cat(outputs, dim=-1)

    def _forward_cached(self, *, prefix_key, graphed, token_embeddings, prefix_h,
                        prefix_position_ids, query_position_ids, attention_mask=None,
                        h_prev=None, c_prev=None, prefix_c=None):
        if (c_prev is not None or prefix_c is not None or prefix_h.shape[0] != 1
                or prefix_h.shape[-1] != self.hspace_dim
                or tuple(token_embeddings.shape[:2]) not in ((1, 4), (1, 8),
                                                              (1, 15), (1, 16), (1, 32))
                or (graphed and token_embeddings.shape[1] != 15
                    and os.environ.get("JETSPEC_C1_GRAPH_LAYOUT", "full") != "live")):
            # The graphed live layout takes the 16-row window (15 drafts) too
            raise ValueError("C1 cache requires N1 and a supported query width with tapped hidden")
        Q = int(token_embeddings.shape[1])
        self._c1_window = Q  # refresh_cached reads the staged window rows
        if not hasattr(self, "_c1_kv"):
            self._cache_init(prefix_h)
        length = prefix_h.shape[1]
        if length < 1 or length > self._c1_capacity:
            raise ValueError("C1 prefix exceeds cache capacity or is empty")
        old = self.c1_state["cache_length"]
        rebuild = self._c1_prefix_key is not prefix_key or length < old or length-old > 16
        start = 0 if rebuild else old
        appended = None
        if (not graphed and not rebuild and 1 <= length - start <= 16
                and os.environ.get("JETSPEC_C1_APPEND", "0") == "1"):
            appended = self._d1_append(token_embeddings, prefix_h, prefix_position_ids,
                                       query_position_ids, attention_mask, start, length)
        if appended is not None:
            cos_q, sin_q = appended[:2]
        else:
            cos_q, sin_q = self._c1_rope(token_embeddings, query_position_ids)
            if length > start:
                positions = prefix_position_ids[:, start:length]
                if graphed and not rebuild and self._c1_extension_enabled():
                    self._graph_extend(prefix_h[:, start:length], positions, start)
                else:
                    self._eager_extend(prefix_h[:, start:length], positions, start, length)
        self._c1_prefix_key = prefix_key
        self.c1_state["cache_length"] = length
        self.c1_state["cache_rebuilds" if rebuild else "cache_hits"] += 1
        self.c1_state.update(last_update="rebuild" if rebuild else "hit", last_start=start,
                             last_append=length-start)
        mask = None if attention_mask is None else attention_mask[..., :length]
        self._d1_preloop_mask = None
        if appended is not None:
            # JETSPEC_C1_APPEND wrote the PRELOOP mask too
            mask = appended[2][..., :length + 15]
            self._d1_preloop_mask = (mask, appended[2])
        elif (not graphed and self.resolved_config.get("window_causal_over_tokens", False)
                and os.environ.get("JETSPEC_C1_PRELOOP", "0") == "1"
                and os.environ.get("JETSPEC_C1_LOOP", "0") == "1"
                and mask is not None and mask.dtype == torch.bfloat16 == token_embeddings.dtype
                and mask.device.type == "cuda" and tuple(mask.shape[:3]) == (1, 1, 15)
                and mask.stride(-1) == 1 and query_position_ids.dtype == torch.long
                and tuple(query_position_ids.shape) == (1, 15)):
            # JETSPEC_C1_PRELOOP: the torch.cat below, already in the lean loop's aligned
            # layout (one kernel: the same values, the zero padding the lean path adds)
            aligned = self._c1_fused_module()._loop_extension().d1_window_mask(
                mask, query_position_ids.contiguous(), length)
            mask = aligned[..., :length + 15]
            self._d1_preloop_mask = (mask, aligned)
        elif self.resolved_config.get("window_causal_over_tokens", False):
            # Each served N1 staging call contains one block of 15 window slots.
            window_allowed = query_position_ids[:, :, None] > query_position_ids[:, None, :]
            window_mask = token_embeddings.new_zeros((1, 1, Q, Q)).masked_fill_(
                ~window_allowed[:, None], torch.finfo(token_embeddings.dtype).min)
            if mask is None:
                mask = token_embeddings.new_zeros((1, 1, Q, length))
            elif mask.dtype == torch.bool:
                mask = token_embeddings.new_zeros(mask.shape).masked_fill_(
                    ~mask, torch.finfo(token_embeddings.dtype).min)
            mask = torch.cat((mask, window_mask), dim=-1)
        if graphed:
            return self._graph_query(token_embeddings, cos_q, sin_q, mask, length)
        return self._cache_query(token_embeddings, cos_q, sin_q, mask, length)

    def prime_admission_prefix(self, sequence, prefix_h):
        """Retain the exact prefix K/V consumed by the first chain round."""
        from jetspec.engines.jetspec_plus_plus.wire.protocol import draft_width

        if not hasattr(self, "_c1_kv"):
            self._cache_init(prefix_h)
        length = prefix_h.shape[1]
        positions = torch.arange(length, device=prefix_h.device)[None]
        self._eager_extend(prefix_h, positions, 0, length)
        self._c1_prefix_key = sequence
        self.c1_state["cache_length"] = length
        self._c1_window = draft_width() + 1

    def _d1_append(self, token_embeddings, prefix_h, prefix_position_ids, query_position_ids,
                   attention_mask, start, length):
        """JETSPEC_C1_APPEND=1: the steady-state work of ``_forward_cached`` before the loop
        (the query and prefix rotary rows, ``_eager_extend``'s D1 fused2 append, the PRELOOP
        mask) in one C++ call on the same ops and compiled Triton entries; None when this
        round is not that steady state or an entry is not compiled yet (the Python path,
        which compiles them, runs instead)."""
        if getattr(self, "target_text_config", None) is not None:
            return None
        table = getattr(self, "_c1_rope_table", None)
        weights = getattr(self, "_d1_projection_weights", None)
        if (not self.resolved_config.get("window_causal_over_tokens", False)
                or os.environ.get("JETSPEC_FUSED_QKV", "0") == "1"
                or os.environ.get("JETSPEC_C1_PRELOOP", "0") != "1"
                or os.environ.get("JETSPEC_C1_LOOP", "0") != "1"
                or os.environ.get("JETSPEC_C1_ROPE_TABLE", "0") != "1"
                or table is None or weights is None
                or self._d1_projection_biases[0] is not None
                or not self._c1_fused2_enabled(prefix_h) or not hasattr(self, "_c1_fast_key_cache")
                or not hasattr(self, "_c1_fast_kv_norm")
                or token_embeddings.dtype != torch.bfloat16 or table[0].dtype != torch.bfloat16
                or table[0].device != prefix_h.device
                or attention_mask is None or attention_mask.dtype != torch.bfloat16
                or attention_mask.dim() != 4 or tuple(attention_mask.shape[:3]) != (1, 1, 15)
                or attention_mask.shape[3] < length or attention_mask.stride(-1) != 1
                or query_position_ids.dtype != torch.long
                or tuple(query_position_ids.shape) != (1, 15)
                or prefix_position_ids.dtype != torch.long):
            return None
        fused = self._c1_fused_module()
        layer = self._c1_blocks[0][1]
        device = prefix_h.device.index
        entries = getattr(self, "_d1_append_entries", None)
        if entries is None or entries[0] is not self._c1_kv:
            width = self._c1_fast_kv_norm.shape[1]
            stand_in = self._c1_fast_kv_norm  # a 16-byte aligned bf16 tensor: the key is dtype + alignment
            heads = layer.num_key_value_heads * layer.num_key_value_groups
            head_dim = layer.head_dim
            norm_entry = fused.compiled_launcher(
                fused._grouped_rms_norm_kernel, device,
                (stand_in, stand_in, self._c1_fast_streams, stand_in, 16,
                 self.hspace_dim // width, width, layer.kv_norm.variance_epsilon,
                 fused.triton.next_power_of_2(width)), num_warps=8)
            kv_width = layer.num_key_value_heads * head_dim
            post_entry = fused.compiled_launcher(
                fused._batched_kv_postprocess_kernel, device,
                (stand_in, stand_in, stand_in, stand_in, stand_in, stand_in, stand_in, 16,
                 layer.num_key_value_heads, layer.num_key_value_groups, head_dim, 16 * kv_width,
                 kv_width, layer.k_norm.variance_epsilon, fused.triton.next_power_of_2(head_dim)),
                num_warps=4)
            if norm_entry is None or post_entry is None:
                return None
            entries = self._d1_append_entries = (self._c1_kv, norm_entry, post_entry)
        try:
            return fused._loop_extension().d1_append(
                prefix_h[:, start:length], prefix_position_ids[:, start:length],
                query_position_ids.contiguous(), table[0], table[1], self._c1_fast_kv_norm,
                self._c1_fast_streams, weights[0], weights[1], self._c1_fast_k_norm,
                self._c1_fast_key_cache, self._c1_fast_value_cache, attention_mask, start, length,
                layer.kv_norm.variance_epsilon, layer.k_norm.variance_epsilon,
                layer.num_key_value_heads, layer.num_key_value_groups, layer.head_dim,
                entries[1], entries[2])
        except RuntimeError as error:
            if ("not 16-byte aligned" not in str(error)
                    and "C1 APPEND requires dense prefix rows" not in str(error)
                    and "C1 APPEND requires dense operands" not in str(error)):
                raise
            return None

    def _c1_rope(self, like, position_ids):
        """``rotary_emb(like, position_ids)``; with JETSPEC_C1_ROPE_TABLE=1 on CUDA, the rows
        gathered from one table of positions 0..8191 computed by that same call (each
        position's cos/sin is a single product, so the gather is bitwise the call; the
        cached path's position ids are cache indices, below the capacity + 16)."""
        rotary = self.streams[0].rotary_emb
        if ((os.environ.get("JETSPEC_C1_ROPE_TABLE", "0") != "1"
             and os.environ.get("JETSPEC_C1_PREP_LEAN", "0") != "1") or like.device.type != "cuda"
                or position_ids.dim() != 2 or position_ids.shape[0] != 1):
            return rotary(like, position_ids)
        table = getattr(self, "_c1_rope_table", None)
        if table is None or table[0].dtype != like.dtype or table[0].device != like.device:
            cos, sin = rotary(like, torch.arange(8192, device=like.device)[None])
            table = self._c1_rope_table = (cos[0], sin[0])
        flat = position_ids.reshape(-1)
        if os.environ.get("JETSPEC_CHAIN_GATHER_VEC", "0") == "1":
            return table[0][flat][None], table[1][flat][None]
        return table[0].index_select(0, flat)[None], table[1].index_select(0, flat)[None]

    def _eager_extend(self, prefix, positions, start, length):
        cos, sin = self._c1_rope(prefix, positions)
        if self._c1_fused2_enabled(prefix) and hasattr(self, "_c1_fast_key_cache"):
            count = prefix.shape[1]
            if self.resolved_config.get("window_causal_over_tokens", False) and count < 16:
                prefix = F.pad(prefix, (0, 0, 0, 16 - count))
                cos = F.pad(cos, (0, 0, 0, 16 - count))
                sin = F.pad(sin, (0, 0, 0, 16 - count))
            key, value = self._cache_project_fused2(prefix, cos, sin)
            self._c1_fast_key_cache[:, :, :, start:length].copy_(key[:, :, :, :count])
            self._c1_fast_value_cache[:, :, :, start:length].copy_(value[:, :, :, :count])
            return
        projected = self._cache_project(prefix, positions, cos, sin)
        for (k, v), (new_k, new_v) in zip(self._c1_kv, projected):
            k[:, :, start:length].copy_(new_k)
            v[:, :, start:length].copy_(new_v)

    def _c1_extension_enabled(self):
        return not getattr(self, "_c1_extension_failed", False)

    def _graph_init(self, embeddings):
        if embeddings.device.type != "cuda":
            raise ValueError("C1 GRAPH requires CUDA")
        self._c1_graphs = {}
        # 15 rows whatever the first call's window (a 16-row window has its own set)
        self._c1_embeddings = embeddings.new_empty(1, 15, embeddings.shape[-1])
        hd = self.streams[0].rotary_emb.inv_freq.numel() * 2
        self._c1_qcos = embeddings.new_empty(1, 15, hd)
        self._c1_qsin = torch.empty_like(self._c1_qcos)
        self._c1_mask = embeddings.new_full((1, 1, 15, self._c1_capacity +
            (15 if self.resolved_config.get("window_causal_over_tokens", False) else 0)),
                                            torch.finfo(embeddings.dtype).min)
        self._c1_output = embeddings.new_empty(1, 15, self.hspace_dim)
        lanes = block_geometry.window_rows()
        self._c1_dirty = embeddings.new_zeros(1, lanes, self.hspace_dim)
        self._c1_dirty_positions = torch.zeros(1, lanes, device=embeddings.device, dtype=torch.long)
        self._c1_lanes = torch.arange(lanes, device=embeddings.device)
        self._c1_slots = torch.empty_like(self._c1_lanes)
        # The 15-row window's buffers (the live layout adds a 16-row set on demand)
        self._c1_window_sets = {15: dict(_c1_embeddings=self._c1_embeddings,
                                         _c1_qcos=self._c1_qcos, _c1_qsin=self._c1_qsin,
                                         _c1_output=self._c1_output)}

    def _c1_query_step(self):
        self._c1_output.copy_(self._cache_query(self._c1_embeddings, self._c1_qcos,
            self._c1_qsin, self._c1_mask, self._c1_capacity))

    def _c1_extension_step(self, dirty=None, positions=None, slots=None):
        dirty = self._c1_dirty if dirty is None else dirty
        positions = self._c1_dirty_positions if positions is None else positions
        slots = self._c1_slots if slots is None else slots
        cos, sin = (self._c1_rope(dirty, positions)
                    if os.environ.get("JETSPEC_C1_PREP_LEAN", "0") == "1"
                    else self.streams[0].rotary_emb(dirty, positions))
        if (getattr(self, "target_text_config", None) is not None
                and os.environ.get("JETSPEC_HYBRID_C1_FUSED", "0") == "1"):
            self._c1_fused_module().hybrid_cache_extend(self, dirty, cos, sin, slots)
            return
        # The live layout (D1) takes the stacked write too: all 16 dirty rows are live
        # lanes or reserved-tail slots, so _cache_project would not pad them, and the
        # stacked projection is the per-block one (two index copies instead of 30).
        live = (os.environ.get("JETSPEC_C1_GRAPH_LAYOUT", "full") == "live"
                and os.environ.get("JETSPEC_C1_LEAN", "0") == "1")
        if (getattr(self, "target_text_config", None) is None and self._c1_fused2_enabled(dirty)
                and hasattr(self, "_c1_fast_key_cache")
                and (live or not self.resolved_config.get("window_causal_over_tokens", False))):
            key, value = self._cache_project_fused2(dirty, cos, sin)
            self._c1_fast_key_cache.index_copy_(3, slots, key)
            self._c1_fast_value_cache.index_copy_(3, slots, value)
            return
        projected = self._cache_project(dirty, positions, cos, sin)
        for (k, v), (new_k, new_v) in zip(self._c1_kv, projected):
            k.index_copy_(2, slots, new_k)
            v.index_copy_(2, slots, new_v)

    def _c1_fold_append(self, dirty, positions, slots):
        """Stage append sources for the span GEMMs without projecting K/V twice."""
        required = ("JETSPEC_C1_SPAN_KV_FUSED", "JETSPEC_FUSED_QKV",
                    "JETSPEC_C1_ROPE_NORM", "JETSPEC_C1_KV_SPLIT_LEAN",
                    "JETSPEC_FRONT_STATIC")
        if (any(os.environ.get(name, "0") != "1" for name in required)
                or os.environ.get("JETSPEC_C1_GRAPH_LAYOUT", "full") != "live"
                or not self._c1_fused2_enabled(dirty)
                or not hasattr(self, "_c1_fast_key_cache")
                or not hasattr(self, "_c1_span_kv")):
            raise ValueError("C1 append fold requires the fused live C1 chain")
        self._c1_fused2_prepare()
        cos, sin = self._c1_rope(dirty, positions)
        return dirty, cos, sin, slots

    def _c1_replay(self, name, step):
        if name not in self._c1_graphs:
            import time
            started = time.perf_counter()
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    step()
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                step()
            self._c1_graphs[name] = graph
            self.c1_state["captures"] += 1
            print(f"W259_C1_CAPTURE phase={name} batch=1 query={self._c1_embeddings.shape[1]} capacity={self._c1_capacity} lanes=16 ms={(time.perf_counter()-started)*1000:.3f}", flush=True)
            # Eager and replay here use exactly the same shapes and operations.
            step()
            expected = ([self._c1_output.clone()] if name.startswith("query") else
                [x.index_select(2, self._c1_slots).clone() for kv in self._c1_kv for x in kv])
            graph.replay()
            actual = ([self._c1_output] if name.startswith("query") else
                [x.index_select(2, self._c1_slots) for kv in self._c1_kv for x in kv])
            if not all(torch.equal(a, b) for a, b in zip(actual, expected)):
                raise RuntimeError(f"C1 {name} graph replay byte parity failed")
        self._c1_graphs[name].replay()

    def _graph_extend(self, prefix, positions, start):
        if not hasattr(self, "_c1_graphs"):
            self._graph_init(prefix.new_empty(1, 15, self.hidden_size))
        count = prefix.shape[1]
        self._c1_dirty.zero_()
        self._c1_dirty[:, :count].copy_(prefix)
        self._c1_dirty_positions.zero_()
        self._c1_dirty_positions[:, :count].copy_(positions)
        self._c1_slots.copy_(torch.where(self._c1_lanes < count, start + self._c1_lanes,
                                         self._c1_capacity + self._c1_lanes))
        self._c1_replay("extension", self._c1_extension_step)
        self.c1_state["extension_replays"] += 1
        self.c1_state["extension_path"] = "graph16"

    def _graph_live_layout(self, embeddings):
        """JETSPEC_C1_GRAPH_LAYOUT=live: the graphed D1 query on the eager layout.

        The window rows sit at cache slots ``length .. length+15`` (as in eager) and
        each graph attends over one key bucket (multiples of 128, the memory-efficient
        kernel's key block at head_dim 128, then the whole cache), masked past the
        live prefix and the window.  Only the certified lean kernel set runs here.
        """
        if os.environ.get("JETSPEC_C1_GRAPH_LAYOUT", "full") != "live":
            return False
        if not (self.resolved_config.get("window_causal_over_tokens", False)
                and self._c1_fused2_enabled(embeddings)
                and os.environ.get("JETSPEC_C1_LEAN", "0") == "1"):
            raise ValueError("C1 GRAPH_LAYOUT=live requires D1, JETSPEC_C1_FUSED=2 and "
                             "JETSPEC_C1_LEAN=1")
        return True

    def _c1_live_bucket(self, keys):
        top = self._c1_capacity + block_geometry.window_rows()
        if self._c1_capacity > 768:
            return (key_length_bucket(keys, top)
                    if os.environ.get("JETSPEC_KEYLEN_BUCKET", "0") == "1" else top)
        bucket = (keys + 127) // 128 * 128
        return top if bucket >= top else bucket

    def _c1_live_query_step(self, bucket):
        if os.environ.get("JETSPEC_C1_COMPILE", "0") == "1":
            self._c1_output.copy_(self._d1_compiled_query(self._c1_live_masks[bucket]))
            return
        fused = self._c1_fused_module()
        self._c1_output.copy_(self._d1_cache_query_lean(
            fused, self._c1_embeddings, self._c1_qcos, self._c1_qsin,
            self._c1_live_masks[bucket], None, window_slots=self._c1_window_slots))

    def _c1_compile_pin(self):
        """JETSPEC_C1_COMPILE_PIN: the pinned inductor cache (compile_pin.Pin), installed
        once before the first compile, or None when COMPILE or the pin is off."""
        if os.environ.get("JETSPEC_C1_COMPILE", "0") != "1":
            return None
        if not hasattr(self, "_c1_pin"):
            self._c1_pin = _compile_pin_module().from_env()
        return self._c1_pin

    def _d1_compiled_plan(self):
        """JETSPEC_C1_COMPILE: per-block operands for the compiled query, with k|v and
        gate|up each one GEMM.  The fused weights are built once and the original
        projections become row views of them (the same bytes; no second copy)."""
        plan = getattr(self, "_d1_compiled_blocks", None)
        if plan is not None:
            return plan
        import torch._dynamo
        self._c1_compile_pin()
        torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 64)
        mode = os.environ.get("JETSPEC_C1_COMPILE_MODE", "max-autotune-no-cudagraphs")
        self._d1_compiled_pre = torch.compile(_d1_compiled_pre, mode=mode, dynamic=False)
        self._d1_compiled_post = torch.compile(_d1_compiled_post, mode=mode, dynamic=False)
        self._d1_compiled_pre_q = torch.compile(_d1_compiled_pre_q, mode=mode, dynamic=False)
        plan = []
        for index, stream in enumerate(self.streams):
            for layer in (stream, *self.extra_stream_layers[index]):
                linears = (layer.q_proj, layer.k_proj, layer.v_proj, layer.o_proj,
                           layer.mlp.gate_proj, layer.mlp.up_proj, layer.mlp.down_proj)
                if any(linear.bias is not None for linear in linears):
                    raise ValueError("C1 COMPILE requires bias-free projections")
                if type(layer.mlp.act_fn).__name__ not in ("SiLU", "SiLUActivation"):
                    raise ValueError("C1 COMPILE requires the silu MLP")
                if hasattr(layer, "_fused_kv"):
                    kv = layer._fused_kv[0]
                else:
                    kv = torch.cat((layer.k_proj.weight, layer.v_proj.weight))
                    width = layer.k_proj.weight.shape[0]
                    layer.k_proj.weight.data, layer.v_proj.weight.data = kv[:width], kv[width:]
                gate_up = torch.cat((layer.mlp.gate_proj.weight, layer.mlp.up_proj.weight))
                inner = layer.mlp.gate_proj.weight.shape[0]
                layer.mlp.gate_proj.weight.data = gate_up[:inner]
                layer.mlp.up_proj.weight.data = gate_up[inner:]
                plan.append((
                    index, layer.input_norm.weight, layer.input_norm.variance_epsilon,
                    layer.q_proj.weight, layer.q_norm.weight, layer.q_norm.variance_epsilon,
                    layer.kv_norm.weight, layer.kv_norm.variance_epsilon, kv,
                    layer.k_norm.weight, layer.k_norm.variance_epsilon, layer.o_proj.weight,
                    layer.post_attention_norm.weight, layer.post_attention_norm.variance_epsilon,
                    gate_up, layer.mlp.down_proj.weight, layer.num_heads,
                    layer.num_key_value_heads, layer.num_key_value_groups, layer.head_dim,
                    layer.scaling))
        # the lean plan and the loop's packed operands hold the pre-view tensors
        for name in ("_d1_lean_blocks", "_d1_loop_packed"):
            if hasattr(self, name):
                delattr(self, name)
        self._d1_compiled_blocks = plan
        return plan

    def _d1_compiled_query(self, mask):
        """JETSPEC_C1_COMPILE=1 (graphed live layout only): the D1 query of
        ``_d1_cache_query_lean`` as torch ops, each block's element-wise chains and norms
        compiled (torch.compile, max-autotune without CUDA graphs) around its GEMMs, and
        captured inside the live layout's query graph.  A new kernel set: its outputs
        are checked against the eager bf16 path's own error vs fp32, not bitwise.
        Attention defaults to the memory-efficient kernel over the bucket's cache slots
        with the bucket's float mask; the window rows go to ``_c1_window_slots``."""
        plan = self._d1_compiled_plan()
        pre, post = self._d1_compiled_pre, self._d1_compiled_post
        from jetspec.engines.jetspec_plus_plus.kernels.attn_sm100 import attend; from jetspec.engines.jetspec_plus_plus.kernels.attn_sm100 import attend_c1; from jetspec.engines.jetspec_plus_plus.kernels.attn_sm100 import flashinfer_enabled
        keys_total = mask.shape[-1]
        slots = self._c1_window_slots
        cos, sin = self._c1_qcos, self._c1_qsin
        Q = int(self._c1_embeddings.shape[1])  # 15, or 16 with 15 drafts
        u = self._c1_embeddings.reshape(Q, self._c1_embeddings.shape[-1])
        window, current, outputs = u, 0, []
        cache_write = (self._c1_fused_module()._loop_extension().k_norm_rotary_cache
                       if os.environ.get("JETSPEC_C1_COMPILE_KVWRITE", "0") == "1" else None)
        for block, (index, in_w, in_eps, q_w, qn_w, qn_eps, kvn_w, kvn_eps, kv_w, kn_w, kn_eps,
                    o_w, post_w, post_eps, gate_up, down_w, heads, kv_heads, groups, head_dim,
                    scaling) in enumerate(plan):
            if index != current:
                outputs.append(u)
                window, current = u, index
            k, v = self._c1_kv[block]
            if cache_write is not None and head_dim == 128:
                # JETSPEC_C1_COMPILE_KVWRITE=1: the lean path's k norm + rotary + cache
                # write kernel on the compiled k|v projection (no repeated-K/V copies)
                q, projected = self._d1_compiled_pre_q(u, window, in_w, q_w, qn_w, kvn_w, kv_w,
                                                       cos, sin, in_eps, qn_eps, kvn_eps, heads,
                                                       head_dim)
                width = kv_heads * head_dim
                cache_write(projected[:, :width], projected[:, width:], kn_w, cos[0], sin[0],
                            k[0, :, :Q].transpose(0, 1), v[0, :, :Q].transpose(0, 1), Q,
                            kv_heads, groups, kn_eps, slots)
            else:
                q, key, value = pre(u, window, in_w, q_w, qn_w, kvn_w, kv_w, kn_w, cos, sin,
                                    in_eps, qn_eps, kvn_eps, kn_eps, heads, kv_heads, groups,
                                    head_dim)
                k[0].index_copy_(1, slots, key)
                v[0].index_copy_(1, slots, value)
            if flashinfer_enabled(q.device, q.dtype):
                attended = attend_c1(q, k[:, :, :keys_total], v[:, :, :keys_total],
                                     scaling, self._flashinfer_c1_plan)
            else:
                attended = attend(q, k[:, :, :keys_total], v[:, :, :keys_total],
                                  mask.expand(1, heads, Q, keys_total), scaling)
            u = post(attended, u, o_w, post_w, gate_up, down_w, post_eps)
        outputs.append(u)
        return torch.cat(outputs, dim=-1).view(1, Q, -1)

    def _c1_window_set(self, Q):
        """The graphed live layout's per-window buffers and graphs.  Q = 15 rows
        (14 drafts) keeps ``_graph_init``'s buffers and the ``query{bucket}`` graphs;
        Q = 16 (15 drafts: [c, d1..d15]) gets its own buffers and ``query{bucket}w16``
        graphs, captured on its first call.  The chosen set's buffers become the
        attributes the query steps read (each graph was captured on its own set)."""
        sets = self._c1_window_sets
        if Q not in sets:
            if Q in (16, 32):
                base = sets[15]
                sets[Q] = {name: base[name].new_empty(1, Q, base[name].shape[-1])
                           for name in ("_c1_embeddings", "_c1_qcos", "_c1_qsin", "_c1_output")}
            else:
                raise ValueError("C1 GRAPH_LAYOUT=live takes a 15-, 16- or 32-row window")
        for name, value in sets[Q].items():
            setattr(self, name, value)
        return sets[Q], ("" if Q == 15 else f"w{Q}")

    def _graph_query_live(self, embeddings, cos, sin, mask, length):
        from jetspec.engines.jetspec_plus_plus.kernels.attn_sm100 import flashinfer_enabled; from jetspec.engines.jetspec_plus_plus.kernels.attn_sm100 import prepare_prefix

        use_flashinfer = flashinfer_enabled(embeddings.device, embeddings.dtype)
        Q = int(embeddings.shape[1])
        if mask is None or tuple(mask.shape) != (1, 1, Q, length + Q):
            raise ValueError("C1 GRAPH_LAYOUT=live requires the [1,1,Q,prefix+Q] mask")
        window_set, suffix = self._c1_window_set(Q)
        self._c1_embeddings.copy_(embeddings)
        self._c1_qcos.copy_(cos)
        self._c1_qsin.copy_(sin)
        if "_c1_live_masks" not in window_set:
            self._c1_live_masks = {}
            self._c1_window_lanes = torch.arange(Q, device=embeddings.device)
            self._c1_window_slots = torch.empty_like(self._c1_window_lanes)
            torch.add(self._c1_window_lanes, length, out=self._c1_window_slots)
            window_set.update(_c1_live_masks=self._c1_live_masks,
                              _c1_window_lanes=self._c1_window_lanes,
                              _c1_window_slots=self._c1_window_slots)
            # The long-context buckets are captured by prewarm_keylen_graphs at startup.
            top = self._c1_capacity + block_geometry.window_rows()
            pin = self._c1_compile_pin()
            if pin is not None:
                pin.arm()
            keylen = (self._c1_capacity > 768
                      and os.environ.get("JETSPEC_KEYLEN_BUCKET", "0") == "1")
            buckets = ([] if use_flashinfer or keylen else [top] if self._c1_capacity > 768
                       else sorted({*range(128, top, 128), top}))
            for bucket in buckets:
                buffer = self._c1_live_masks[bucket] = embeddings.new_empty((1, 1, Q, bucket))
                buffer.fill_(torch.finfo(embeddings.dtype).min)
                self._c1_replay(f"query{bucket}{suffix}",
                                lambda bucket=bucket: self._c1_live_query_step(bucket))
            if pin is not None and not use_flashinfer and not keylen:
                # Every bucket's compile of this window hit the pinned cache
                pin.check(f"window{Q}")
        bucket = self._c1_live_bucket(length + Q)
        keylen = (self._c1_capacity > 768
                  and os.environ.get("JETSPEC_KEYLEN_BUCKET", "0") == "1")
        name = f"query{bucket}{suffix}"
        if keylen and not getattr(self, "_keylen_prewarming", False) and name not in self._c1_graphs:
            bucket = self._c1_capacity + block_geometry.window_rows()
            name = f"query{bucket}{suffix}"
            self.c1_state["keylen_fallback"] = self.c1_state.get("keylen_fallback", 0) + 1
            if name not in self._c1_graphs:
                raise RuntimeError(f"C1 whole-cache graph {name} was not prewarmed")
        if bucket not in self._c1_live_masks:
            self._c1_live_masks[bucket] = embeddings.new_empty((1, 1, Q, bucket))
        buffer = self._c1_live_masks[bucket]
        buffer.fill_(torch.finfo(embeddings.dtype).min)
        buffer[..., :length + Q].copy_(mask)
        torch.add(self._c1_window_lanes, length, out=self._c1_window_slots)
        if use_flashinfer:
            self._flashinfer_c1_plan = prepare_prefix(
                self, "c1", Q, self._c1_kv[0][0], length,
                self._d1_lean_plan()[0][-1])
        self._c1_replay(name, lambda: self._c1_live_query_step(bucket))
        self.c1_state["graph_replays"] += 1
        self.c1_state["query_path"] = f"graph{bucket}"
        return self._c1_output.clone()

    def c1_chain_body(self, rp, tokens: TokenBuffers, front: FrontInputs, bucket):
        """C1 append and live query with all lengths read from device RoundParams.

        ``front`` holds stable tensors prepared before capture. The rows pass's
        static prefix bucket chooses the graph's attention implementation.
        """
        if os.environ.get("JETSPEC_FRONT_STATIC", "0") != "1":
            raise ValueError("C1 chain body requires JETSPEC_FRONT_STATIC=1")
        use_len_attention = (os.environ.get("JETSPEC_C1_ATTN_LEN", "0") == "1"
                             and rp.is_cuda and bucket.head.P >= 2048)
        lanes = front.lanes
        lane = torch.arange(lanes, device=rp.device, dtype=torch.long)
        start, count, length = rp[Offset.C1_START], rp[Offset.C1_COUNT], rp[Offset.L]
        prep_lean = os.environ.get("JETSPEC_C1_PREP_LEAN", "0") == "1"
        if prep_lean:
            from jetspec.engines.jetspec_plus_plus.kernels.chain.glue2 import c1_front_prep
            slots = torch.empty(lanes, device=rp.device, dtype=torch.long)
            c1_front_prep(rp, tokens.committed_ids, tokens.pending_draft_ids,
                          front.c1_query_ids, front.prefix_positions, slots,
                          self._c1_capacity)
        else:
            front.c1_query_ids.copy_(chain_query_ids(rp, tokens))
            front.prefix_positions.copy_((start + lane)[None])
            slots = chain_c1_slots(start, count, self._c1_capacity, lanes)
        front.c1_query_embeddings.copy_(
            self._front_embed_tokens.weight[front.c1_query_ids]
            if os.environ.get("JETSPEC_CHAIN_GATHER_VEC", "0") == "1"
            else self._front_embed_tokens(front.c1_query_ids))
        append = None
        if os.environ.get("JETSPEC_C1_APPEND_FOLD", "0") == "1":
            append = self._c1_fold_append(front.taps.view(1, lanes, self.hspace_dim),
                                          front.prefix_positions, slots)
        else:
            self._c1_extension_step(front.taps.view(1, lanes, self.hspace_dim),
                                    front.prefix_positions, slots)
        query_positions = (length + lane)[None]
        cos, sin = self._c1_rope(front.c1_query_embeddings, query_positions)
        if use_len_attention:
            mask = front.c1_query_embeddings.new_empty(1, 1, lanes, front.c1_bucket)
        elif prep_lean:
            from jetspec.engines.jetspec_plus_plus.kernels.chain.glue2 import c1_mask
            mask = torch.empty((1, 1, lanes, front.c1_bucket),
                               device=rp.device, dtype=front.c1_query_embeddings.dtype)
            c1_mask(length, mask)
        else:
            mask = chain_c1_mask(length, front.c1_bucket, lanes,
                                 front.c1_query_embeddings.dtype)
        len_args = {"chain_length": length} if use_len_attention else {}
        append_args = {"append": append} if append is not None else {}
        front.staged_hidden.copy_(self._d1_cache_query_lean(
            self._c1_fused_module(), front.c1_query_embeddings, cos, sin, mask,
            None, window_slots=length + lane, allow_flashinfer=False,
            **append_args, **len_args))
        return front.staged_hidden



    @torch.inference_mode()
    def prewarm_live_query_window(self, query_rows=15):
        """Capture the live window shape missed by the synthetic session path."""
        if (os.environ.get("JETSPEC_C1_GRAPH_LAYOUT", "full") != "live"
                or not hasattr(self, "_c1_window_sets")):
            return False
        base = self._c1_window_sets[query_rows]
        embeddings = base["_c1_embeddings"].new_zeros(base["_c1_embeddings"].shape)
        cos = base["_c1_qcos"].new_ones(base["_c1_qcos"].shape)
        sin = base["_c1_qsin"].new_zeros(base["_c1_qsin"].shape)
        length = max(1, min(self._c1_capacity, int(self.c1_state["cache_length"])))
        mask = embeddings.new_zeros((1, 1, query_rows, length + query_rows))
        self._keylen_prewarming = True
        try:
            self._graph_query_live(embeddings, cos, sin, mask, length)
        finally:
            self._keylen_prewarming = False
        if "extension" not in self._c1_graphs:
            self._graph_extend(embeddings.new_zeros(1, 1, self.hspace_dim),
                               torch.zeros(1, 1, device=embeddings.device, dtype=torch.long), 0)
        print(f"W409_C1_QUERY_PREWARM rows={query_rows} extension=1", flush=True)
        return True

    @torch.inference_mode()
    def prewarm_keylen_graphs(self):
        """Capture bounded C1 and refresh key ranges before the first timed round."""
        if os.environ.get("JETSPEC_KEYLEN_BUCKET", "0") != "1" or self._c1_capacity <= 768:
            return False
        if not hasattr(self, "_c1_window_sets"):
            raise RuntimeError("C1 graph buffers are required before key-length prewarm")
        import time
        from jetspec.engines.jetspec_plus_plus.kernels.attn_sm100 import flashinfer_enabled

        def measured(stage, bucket, tokens, call):
            torch.cuda.synchronize()
            before = torch.cuda.memory_allocated()
            reserved = torch.cuda.memory_reserved()
            start_time = time.perf_counter()
            call()
            torch.cuda.synchronize()
            print(f"KEYLEN_PREWARM stage={stage} bucket={bucket} tokens={tokens} "
                  f"ms={(time.perf_counter() - start_time) * 1000:.3f} "
                  f"allocated_delta={torch.cuda.memory_allocated() - before} "
                  f"reserved_delta={torch.cuda.memory_reserved() - reserved}", flush=True)

        top = self._c1_capacity + self._c1_dirty.shape[1]
        self._keylen_prewarming = True
        try:
            for q in ((15, 16, 32) if self._c1_dirty.shape[1] == 32 else (15, 16)):
                base = self._c1_window_sets.get(q, self._c1_window_sets[15])
                embeddings = base["_c1_embeddings"].new_zeros(1, q, base["_c1_embeddings"].shape[-1])
                cos = base["_c1_qcos"].new_ones(1, q, base["_c1_qcos"].shape[-1])
                sin = torch.zeros_like(cos)
                for bucket in key_length_buckets(top):
                    length = min(self._c1_capacity, bucket - q)
                    mask = embeddings.new_zeros(1, 1, q, length + q)
                    measured("c1", bucket, q,
                             lambda: self._graph_query_live(embeddings, cos, sin, mask, length))
            if (os.environ.get("JETSPEC_REFRESH_GRAPH", "0") == "1"
                    and os.environ.get("JETSPEC_REFRESH_LEAN", "0") == "1"
                    and os.environ.get("JETSPEC_C1_NORM") == "wide"):
                fused = self._c1_fused_module()
                device = self._c1_embeddings.device
                dtype = self._c1_embeddings.dtype
                use_flashinfer = flashinfer_enabled(device, dtype)
                for tokens in range(16, 129, 16):
                    widths = key_length_buckets(top) if tokens in (64, 128) else (top,)
                    for width in widths:
                        rows = torch.zeros(tokens, self.hidden_size, device=device, dtype=dtype)
                        start = width - tokens
                        positions = torch.arange(start, width, device=device)[None]
                        cos, sin = self._c1_rope(rows[None], positions)
                        mask = torch.zeros(1, 1, tokens, width, device=device, dtype=dtype)
                        suffix_meta = None
                        if use_flashinfer:
                            row_count = tokens // 16
                            row_index = (torch.arange(tokens, device=device) // 15).clamp(
                                max=row_count)
                            row_starts = torch.arange(row_count + 1, device=device) * 15
                            zero = torch.zeros(row_count + 1, device=device, dtype=torch.long)
                            suffix_meta = (row_index, row_starts, zero, zero)
                        measured("refresh", width, tokens,
                                 lambda: self._refresh_graph_replay(
                                     fused, rows, cos, sin, mask, start, tokens, tokens,
                                     suffix_meta=suffix_meta))
        finally:
            self._keylen_prewarming = False
        return True

    def _graph_query(self, embeddings, cos, sin, mask, length):
        if not hasattr(self, "_c1_graphs"):
            self._graph_init(embeddings)
        if self._graph_live_layout(embeddings):
            return self._graph_query_live(embeddings, cos, sin, mask, length)
        if embeddings.shape[1] != 15:
            raise ValueError("the full-layout C1 graph takes a 15-row window only")
        self._c1_embeddings.copy_(embeddings)
        self._c1_qcos.copy_(cos)
        self._c1_qsin.copy_(sin)
        self._c1_mask.fill_(torch.finfo(embeddings.dtype).min)
        if mask is None:
            self._c1_mask[..., :length].zero_()
        else:
            self._c1_mask[..., :length].copy_(mask[..., :length])
            if self.resolved_config.get("window_causal_over_tokens", False):
                self._c1_mask[..., self._c1_capacity:].copy_(mask[..., length:])
        self._c1_replay("query", self._c1_query_step)
        self.c1_state["graph_replays"] += 1
        self.c1_state["query_path"] = "graph"
        return self._c1_output.clone()

    def _forward_shared(self, *, owners, token_embeddings, prefix_h, prefix_position_ids,
                        query_position_ids, attention_mask=None, h_prev=None,
                        c_prev=None, prefix_c=None):
        if c_prev is not None or prefix_c is not None:
            raise ValueError("ctxkv c1 forbids c-space inputs")
        if prefix_h.shape[-1] != self.hspace_dim:
            raise ValueError("C1 requires tapped hidden prefix")
        rows, queries = token_embeddings.shape[:2]
        sequences, prefix = prefix_h.shape[:2]
        first_rows = torch.full((sequences,), rows, dtype=torch.long, device=owners.device)
        first_rows.scatter_reduce_(0, owners, torch.arange(rows, device=owners.device), reduce="amin")
        positions = prefix_position_ids.index_select(0, first_rows)
        mask = None if attention_mask is None else attention_mask[..., :prefix]
        u, outputs = token_embeddings, []
        for index, stream in enumerate(self.streams):
            prefix_tap = prefix_h[..., index*self.hidden_size:(index+1)*self.hidden_size]
            for layer in (stream, *self.extra_stream_layers[index]):
                normed = layer.input_norm(u)
                key_source = layer.kv_norm(prefix_tap)
                q = layer.q_norm(layer.q_proj(normed).view(rows, queries, layer.num_heads, layer.head_dim)).transpose(1,2)
                k = layer.k_norm(layer.k_proj(key_source).view(sequences, prefix, layer.num_key_value_heads, layer.head_dim)).transpose(1,2)
                v = layer.v_proj(key_source).view(sequences, prefix, layer.num_key_value_heads, layer.head_dim).transpose(1,2)
                cos_q, sin_q = layer.rotary_emb(normed, query_position_ids)
                cos_k, sin_k = layer.rotary_emb(key_source, positions)
                q, _ = _apply_rotary_pos_emb(q, q, cos_q, sin_q)
                k, _ = _apply_rotary_pos_emb(k, k, cos_k, sin_k)
                if layer.num_key_value_groups != 1:
                    k = k.repeat_interleave(layer.num_key_value_groups, dim=1)
                    v = v.repeat_interleave(layer.num_key_value_groups, dim=1)
                attended = F.scaled_dot_product_attention(q, k.index_select(0, owners),
                    v.index_select(0, owners), attn_mask=mask, scale=layer.scaling)
                attended = attended.transpose(1,2).reshape(rows, queries, layer.num_heads*layer.head_dim)
                state = u + layer.o_proj(attended)
                u = state + layer.mlp(layer.post_attention_norm(state))
            outputs.append(u)
        return torch.cat(outputs, dim=-1)

    @torch.inference_mode()
    def refresh_window(self, *, tail_embeddings, block_embeddings, length, tail_counts,
                       prefix_key, row_slots=None):
        """JETSPEC_ADAPTER_REFRESH: the adapter's rows for each drafted row's block.

        One call over [staged tail, row block] per row, the adapter refresh rule on the
        served layout: the prefix is the exact rows 0..length-1 the staging call cached
        (never a consumed adapter row), the window is the staged tail's first
        ``tail_counts[r]`` tokens followed by the row's 16 block tokens at their true
        positions, and under D1 each window slot reads only strictly earlier slots.
        The window-causal tail slots do not depend on any block, so they run once for
        all rows (``tail_out``, which reproduces the staged prediction).  All tokens
        (tail, then every row's block) share one projection per layer and one
        attention call whose mask carries the per-row window.  Returns
        ``(tail_out [T, H], block_out [R, 16, H])``, or None when the cache does not
        hold this sequence's exact prefix.

        ``row_slots`` (JETSPEC_REFRESH_PACK): a host list, row r's first
        ``row_slots[r]`` block slots only.  A window slot reads only earlier slots, so
        those outputs are the full call's; the rest of ``block_out`` is zero.
        """
        if (not hasattr(self, "_c1_kv") or self._c1_prefix_key is not prefix_key
                or self.c1_state["cache_length"] != length):
            return None
        fused = self._c1_fused_module()
        window_causal = bool(self.resolved_config.get("window_causal_over_tokens", False))
        tail_count, hidden = int(tail_embeddings.shape[0]), int(tail_embeddings.shape[-1])
        rows, width = int(block_embeddings.shape[0]), int(block_embeddings.shape[1])
        device, dtype = block_embeddings.device, block_embeddings.dtype
        counts = tail_counts.to(device=device, dtype=torch.long).clamp(0, tail_count)
        if row_slots is None:
            slot = torch.arange(width, device=device).repeat(rows)
            row_index = torch.arange(rows, device=device).repeat_interleave(width)
            block_rows = block_embeddings.reshape(rows * width, hidden)
        else:
            if len(row_slots) != rows or any(not 0 < n <= width for n in row_slots):
                raise ValueError("row_slots must hold one count in 1..width per row")
            slot = torch.tensor([s for n in row_slots for s in range(n)], device=device)
            row_index = torch.tensor([r for r, n in enumerate(row_slots) for _ in range(n)],
                                     device=device)
            block_rows = block_embeddings[row_index, slot]
        tokens = tail_count + int(slot.shape[0])
        positions = torch.cat((torch.arange(length, length + tail_count, device=device),
                               length + counts[row_index] + slot))
        cos, sin = self._c1_rope(tail_embeddings[None], positions[None])
        # Window keys: token a reads window token b (a, b in tail-then-rows order) iff
        # b is an earlier tail slot within a's own tail span, or an earlier slot of a's
        # own row block.  The prefix is always visible.
        tail_index = torch.arange(tail_count, device=device)
        row_of = torch.cat((torch.full((tail_count,), -1, device=device), row_index))
        slot_of = torch.cat((tail_index, slot))
        span = torch.cat((tail_index, counts[row_index]))
        tail_key = row_of[None, :] < 0
        allowed = torch.where(tail_key, slot_of[None, :] < span[:, None],
                              (row_of[None, :] == row_of[:, None])
                              & (slot_of[None, :] < slot_of[:, None]))
        if not window_causal:
            allowed = torch.zeros_like(allowed)
        mask = torch.zeros(1, 1, tokens, length + tokens, device=device, dtype=dtype)
        mask[0, 0, :, length:].masked_fill_(~allowed, torch.finfo(dtype).min)

        def norm(x, module):
            return fused.rms_norm(x, module.weight, module.variance_epsilon)

        u = torch.cat((tail_embeddings, block_rows))[None]
        outputs, block = [], 0
        for index, stream in enumerate(self.streams):
            window = u
            for layer in (stream, *self.extra_stream_layers[index]):
                heads, kv_heads, dim = layer.num_heads, layer.num_key_value_heads, layer.head_dim
                q = fused.fused_q_postprocess(
                    layer.q_proj(norm(u, layer.input_norm)), layer.q_norm.weight, cos, sin,
                    layer.q_norm.variance_epsilon, heads, dim)
                cached_k, cached_v = self._c1_kv[block]
                keys, values = cached_k[:, :, :length], cached_v[:, :, :length]
                if window_causal:
                    source = norm(window, layer.kv_norm)
                    k = fused.fused_q_postprocess(
                        layer.k_proj(source), layer.k_norm.weight, cos, sin,
                        layer.k_norm.variance_epsilon, kv_heads, dim)
                    v = layer.v_proj(source).view(1, tokens, kv_heads, dim).transpose(1, 2)
                    if layer.num_key_value_groups != 1:
                        k = k.repeat_interleave(layer.num_key_value_groups, dim=1)
                        v = v.repeat_interleave(layer.num_key_value_groups, dim=1)
                    keys = torch.cat((keys, k), dim=2)
                    values = torch.cat((values, v), dim=2)
                    step_mask = mask
                else:
                    step_mask = mask[..., :length]
                attended = F.scaled_dot_product_attention(
                    q, keys, values, attn_mask=step_mask, scale=layer.scaling)
                state = u + layer.o_proj(attended.transpose(1, 2).reshape(1, tokens, heads * dim))
                u = state + layer.mlp(norm(state, layer.post_attention_norm))
                block += 1
            outputs.append(u[0])
        out = torch.cat(outputs, dim=-1)
        if row_slots is None:
            return out[:tail_count], out[tail_count:].view(rows, width, -1)
        block_out = out.new_zeros(rows, width, out.shape[-1])
        block_out[row_index, slot] = out[tail_count:]
        return out[:tail_count], block_out

    @torch.inference_mode()
    def refresh_cached(self, *, block_embeddings, length, tail_counts, prefix_key, row_slots,
                       pad_to=0, slot_counts=None, first_slots=None, fixed_tokens=None,
                       fixed_start=None):
        """JETSPEC_REFRESH_PACK=2: ``refresh_window(row_slots=...)``'s block rows with the
        staged tail read from the cache instead of recomputed.

        The bf16 D1 staging call wrote its 15 window rows' keys and values to cache
        slots ``length .. length+14`` (the ephemeral tail slots a later extension
        overwrites before reading).  The packed block tokens write theirs after them,
        from slot ``length+15``, and attend over the cache in place: the prefix, tail
        slot t iff t < the row's k, and the row's own earlier block slots.  ``pad_to``
        appends masked filler tokens up to that count (a shape choice only: no real
        token reads a filler).  ``slot_counts``: ``row_slots`` already on the device.
        ``first_slots`` (JETSPEC_REFRESH_REUSE0, a device 0/1 per row): a row with 1
        computes block slots 1 .. row_slots[r] and reads tail slot k in place of its
        block slot 0 (the caller has checked they hold the same token at the same
        position, so they are the same window slot).  Returns ``block_out [R, 16, H]``
        (zero outside each row's computed slots), or None when the cache does not hold this sequence's exact
        prefix and window or has no room for the block tokens.
        """
        if (not hasattr(self, "_c1_kv") or self._c1_prefix_key is not prefix_key
                or self.c1_state["cache_length"] != length
                or not self.resolved_config.get("window_causal_over_tokens", False)
                or block_embeddings.dtype != torch.bfloat16
                or block_embeddings.device.type != "cuda"):
            return None
        rows, width = int(block_embeddings.shape[0]), int(block_embeddings.shape[1])
        hidden = int(block_embeddings.shape[-1])
        if fixed_tokens is not None:
            if slot_counts is None or first_slots is not None or fixed_tokens < rows * (width - 1):
                raise ValueError("fixed refresh needs device counts and R x (block-1) slots")
            window_rows = getattr(self, "_c1_window", 15)
            if length + window_rows + fixed_tokens > self._c1_kv[0][0].shape[2]:
                return None
            return self._refresh_cached_fixed(
                block_embeddings, length, tail_counts, slot_counts, fixed_tokens, fixed_start)
        if len(row_slots) != rows or any(not 0 < n <= width - 1 * (first_slots is not None)
                                         for n in row_slots):
            raise ValueError("row_slots must hold one count in 1..width per row")
        real = sum(row_slots)
        tokens = max(real, int(pad_to))
        window_rows = getattr(self, "_c1_window", 15)  # 16 with 15 drafts
        start = length + window_rows
        keys_total = start + tokens
        if keys_total > self._c1_kv[0][0].shape[2]:
            return None
        fused = self._c1_fused_module()
        device, dtype = block_embeddings.device, block_embeddings.dtype
        counts = tail_counts.to(device=device, dtype=torch.long).clamp(0, window_rows)
        filler = tokens - real
        # the packed (row, slot) pairs built on device from the counts (no host copy)
        need = (torch.as_tensor(row_slots, device=device) if slot_counts is None
                else slot_counts.to(device=device, dtype=torch.long))
        row_index = torch.repeat_interleave(torch.arange(rows, device=device), need,
                                            output_size=real)
        row_starts = torch.cumsum(need, 0) - need
        slot = torch.arange(real, device=device) - row_starts[row_index]
        first = (counts.new_zeros(rows + 1) if first_slots is None else
                 torch.cat((first_slots.to(device=device, dtype=torch.long),
                            counts.new_zeros(1))))
        if first_slots is not None:
            slot = slot + first[row_index]
        if filler:
            row_index = torch.cat((row_index, row_index.new_full((filler,), rows)))
            slot = torch.cat((slot, slot.new_full((filler,), width)))
        counts_with_filler = torch.cat((counts, counts.new_zeros(1)))
        from jetspec.engines.jetspec_plus_plus.kernels.attn_sm100 import flashinfer_enabled
        use_flashinfer = flashinfer_enabled(device, dtype)
        suffix_meta = None
        if use_flashinfer:
            row_starts = torch.cat((row_starts, row_starts.new_full((1,), real)))
            suffix_meta = (row_index, row_starts, counts_with_filler, first)
        block_rows = block_embeddings[row_index[:real], slot[:real]]
        if filler:
            block_rows = torch.cat((block_rows, block_rows.new_zeros(filler, hidden)))
        k_of = counts_with_filler[row_index]
        positions = length + k_of + slot.clamp(max=width - 1)
        span = k_of + first[row_index]
        cos, sin = self._c1_rope(block_rows[None], positions[None])
        tail = torch.arange(window_rows, device=device)
        allowed = torch.cat((
            tail[None, :] < span[:, None],
            (row_index[None, :] == row_index[:, None]) & (slot[None, :] < slot[:, None])), dim=1)
        lean = (os.environ.get("JETSPEC_REFRESH_LEAN", "0") == "1"
                and os.environ.get("JETSPEC_C1_NORM") == "wide")
        graph = lean and (os.environ.get("JETSPEC_REFRESH_GRAPH", "0") == "1"
                          or os.environ.get("JETSPEC_ROWS_FUSED_GRAPH", "0") == "1")
        # The graphed path keeps the exact mask values and slices K/V to its live bucket.
        if graph and os.environ.get("JETSPEC_KEYLEN_BUCKET", "0") == "1":
            aligned = self._refresh_key_bucket(keys_total, self._c1_kv[0][0].shape[2])
        else:
            aligned = self._c1_kv[0][0].shape[2] if graph else (keys_total + 15) // 16 * 16
        mask = torch.zeros(1, 1, tokens, aligned, device=device, dtype=dtype)
        mask[0, 0, :, length:keys_total].masked_fill_(~allowed, torch.finfo(dtype).min)
        if graph:
            mask[0, 0, :, keys_total:].fill_(torch.finfo(dtype).min)
            out = self._refresh_graph_replay(fused, block_rows, cos, sin, mask, start, tokens,
                                             real, suffix_meta=suffix_meta)
        elif lean:
            from jetspec.engines.jetspec_plus_plus.kernels.attn_sm100 import prepare_prefix

            if use_flashinfer:
                self._flashinfer_refresh_plan = prepare_prefix(
                    self, "refresh", tokens, self._c1_kv[0][0], length,
                    self._d1_lean_plan()[0][-1])
            out = self._refresh_cached_lean(fused, block_rows, cos, sin, mask, start,
                                            keys_total, tokens, real, suffix_meta=suffix_meta)
        else:
            out = self._refresh_cached_plain(fused, block_rows, cos, sin,
                                             mask[..., :keys_total], start, keys_total,
                                             tokens, real)
        block_out = out.new_zeros(rows, width, out.shape[-1])
        block_out[row_index[:real], slot[:real]] = out
        return block_out

    def _refresh_cached_fixed(self, embeddings, length, tail_counts, slot_counts,
                              tokens, fixed_start, key_width=None):
        """Static refresh body: device counts choose dense slots, never a host shape."""
        if os.environ.get("JETSPEC_REFRESH_PREP_LEAN", "0") == "1":
            return self._refresh_cached_fixed_prep(
                embeddings, length, tail_counts, slot_counts, tokens, fixed_start, key_width)
        rows, width, hidden = embeddings.shape
        device = embeddings.device
        window_rows = getattr(self, "_c1_window", 15)
        length_dev = torch.as_tensor(length, device=device, dtype=torch.long)
        start_dev = (length_dev + window_rows if fixed_start is None else
                     torch.as_tensor(fixed_start, device=device, dtype=torch.long))
        row, slot, valid = fixed_refresh_layout(slot_counts, tokens, width)
        data = embeddings[row.clamp(max=rows - 1), slot.clamp(max=width - 1)]
        block_rows = torch.where(valid[:, None], data, torch.zeros_like(data))
        counts = tail_counts.to(device=device, dtype=torch.long).clamp(0, window_rows)
        k_of = torch.cat((counts, counts.new_zeros(1)))[row]
        positions = torch.where(valid, length_dev + k_of + slot, length_dev)
        cos, sin = self._c1_rope(block_rows[None], positions[None])

        capacity = self._c1_kv[0][0].shape[2]
        key_bucket = os.environ.get("JETSPEC_CHAIN_REFRESH_KEY_BUCKET", "0") == "1"
        aligned = key_width if key_bucket and key_width is not None else capacity
        if not 0 < aligned <= capacity:
            raise ValueError("refresh key bucket exceeds cache capacity")
        refresh_meta = None
        if os.environ.get("JETSPEC_REFRESH_ATTN_LEAN", "0") == "1":
            offsets = torch.cat((slot_counts.new_zeros(1), slot_counts.cumsum(0)))
            refresh_meta = (length_dev.reshape(1), k_of, start_dev.reshape(1),
                            torch.where(valid, offsets[row], tokens))
            mask = None
        else:
            col = torch.arange(aligned, device=device)
            tail_col = col - length_dev
            block_col = col - start_dev
            source = block_col.clamp(0, tokens - 1)
            prefix = col[None, :] < length_dev
            tail = ((tail_col[None, :] >= 0) & (tail_col[None, :] < window_rows)
                    & (tail_col[None, :] < k_of[:, None]))
            earlier = ((block_col[None, :] >= 0) & (block_col[None, :] < tokens)
                       & (row[source][None, :] == row[:, None])
                       & (slot[source][None, :] < slot[:, None]))
            allowed = prefix | tail | earlier
            mask = torch.zeros(1, 1, tokens, aligned, device=device, dtype=embeddings.dtype)
            mask[0, 0].masked_fill_(~allowed, torch.finfo(mask.dtype).min)
        fused = self._c1_fused_module()
        if not (os.environ.get("JETSPEC_REFRESH_LEAN", "0") == "1"
                and os.environ.get("JETSPEC_C1_NORM") == "wide"):
            raise ValueError("fixed refresh requires JETSPEC_REFRESH_LEAN=1 and JETSPEC_C1_NORM=wide")
        chain_flashinfer = os.environ.get("JETSPEC_CHAIN_ATTN3", "0") == "1"
        suffix_meta = None
        if chain_flashinfer:
            starts = torch.cumsum(slot_counts, 0) - slot_counts
            row_starts = torch.cat((starts, slot_counts.sum().reshape(1)))
            counts_with_filler = torch.cat((counts, counts.new_zeros(1)))
            suffix_meta = (row, row_starts, counts_with_filler,
                           counts_with_filler.new_zeros(rows + 1))
        if refresh_meta is not None:
            output = self._refresh_cached_lean(
                fused, block_rows, cos, sin, mask, 0, aligned, tokens, tokens,
                start_slot=start_dev.reshape(1), allow_flashinfer=False,
                refresh_meta=refresh_meta)
        elif chain_flashinfer:
            output = self._refresh_cached_lean(
                fused, block_rows, cos, sin, mask, 0, aligned, tokens, tokens,
                start_slot=start_dev.reshape(1), suffix_meta=suffix_meta,
                allow_flashinfer=True, chain_flashinfer=True)
        else:
            output = self._refresh_cached_lean(
                fused, block_rows, cos, sin, mask, 0, aligned, tokens, tokens,
                start_slot=start_dev.reshape(1), allow_flashinfer=False)
        padded = output.new_zeros(rows + 1, width + 1, output.shape[-1])
        padded[row, slot] = output
        return padded[:rows, :width]

    def _refresh_cached_fixed_prep(self, embeddings, length, tail_counts, slot_counts,
                                   tokens, fixed_start, key_width=None):
        from jetspec.engines.jetspec_plus_plus.kernels.refresh_prep import prepare; from jetspec.engines.jetspec_plus_plus.kernels.refresh_prep import scatter

        if not (os.environ.get("JETSPEC_REFRESH_ATTN_LEAN", "0") == "1"
                and os.environ.get("JETSPEC_REFRESH_LEAN", "0") == "1"
                and os.environ.get("JETSPEC_C1_NORM") == "wide"):
            raise ValueError("refresh prep requires fixed lean refresh with attention metadata")
        window = getattr(self, "_c1_window", 15)
        length_dev = torch.as_tensor(length, device=embeddings.device, dtype=torch.long)
        start_dev = (length_dev + window if fixed_start is None else
                     torch.as_tensor(fixed_start, device=embeddings.device, dtype=torch.long))
        table = getattr(self, "_c1_rope_table", None)
        if table is None or table[0].dtype != embeddings.dtype or table[0].device != embeddings.device:
            cos, sin = self.streams[0].rotary_emb(
                embeddings[:1], torch.arange(8192, device=embeddings.device)[None])
            table = self._c1_rope_table = (cos[0], sin[0])
        prep = prepare(embeddings, length_dev, tail_counts, slot_counts, tokens,
                       *table, window)
        capacity = self._c1_kv[0][0].shape[2]
        key_bucket = os.environ.get("JETSPEC_CHAIN_REFRESH_KEY_BUCKET", "0") == "1"
        aligned = key_width if key_bucket and key_width is not None else capacity
        if not 0 < aligned <= capacity:
            raise ValueError("refresh key bucket exceeds cache capacity")
        meta = (length_dev.reshape(1), prep['k_of'], start_dev.reshape(1), prep['starts'])
        output = self._refresh_cached_lean(
            self._c1_fused_module(), prep['data'], prep['cos'], prep['sin'], None,
            0, aligned, tokens, tokens, start_slot=start_dev.reshape(1),
            allow_flashinfer=False, refresh_meta=meta)
        return scatter(output, slot_counts, embeddings.shape[1])

    @staticmethod
    def _refresh_key_bucket(keys_total, capacity):
        return key_length_bucket(keys_total, capacity)

    def _refresh_graph_replay(self, fused, rows, cos, sin, mask, start, tokens, real,
                              suffix_meta=None):
        """JETSPEC_REFRESH_GRAPH=1: the lean refresh loop as one CUDA graph per padded token
        count (captured on first use, byte-checked against its eager run), reading
        every cache slot under ``mask`` and writing the block keys and values from the
        slot held in device memory.  The graph's output rows are overwritten by the
        next replay; the caller copies them out at once."""
        graphs = getattr(self, "_refresh_graphs", None)
        from jetspec.engines.jetspec_plus_plus.kernels.attn_sm100 import flashinfer_enabled; from jetspec.engines.jetspec_plus_plus.kernels.attn_sm100 import prepare_prefix

        use_flashinfer = flashinfer_enabled(rows.device, rows.dtype)
        if use_flashinfer:
            window_rows = getattr(self, "_c1_window", 15)
            length = start - window_rows
            self._flashinfer_refresh_plan = prepare_prefix(
                self, "refresh", tokens, self._c1_kv[0][0], length,
                self._d1_lean_plan()[0][-1])
            if suffix_meta is None:
                raise ValueError("FlashInfer refresh requires device suffix metadata")
        if graphs is None or graphs[0] is not self._c1_kv:
            graphs = self._refresh_graphs = (self._c1_kv, OrderedDict())
        keylen = os.environ.get("JETSPEC_KEYLEN_BUCKET", "0") == "1"
        key = ((tokens, mask.shape[-1], suffix_meta[2].numel(), window_rows) if use_flashinfer
               else (tokens, mask.shape[-1]) if keylen else tokens)
        entry = graphs[1].get(key)
        if (keylen and not use_flashinfer
                and not getattr(self, "_keylen_prewarming", False) and entry is None):
            capacity = self._c1_kv[0][0].shape[2]
            key = (tokens, capacity)
            entry = graphs[1].get(key)
            self.c1_state["keylen_fallback"] = self.c1_state.get("keylen_fallback", 0) + 1
            if entry is None:
                raise RuntimeError(f"refresh whole-cache graph {key} was not prewarmed")
            if mask.shape[-1] != capacity:
                padded = mask.new_full((*mask.shape[:-1], capacity), torch.finfo(mask.dtype).min)
                padded[..., :mask.shape[-1]].copy_(mask)
                mask = padded
        if entry is None:
            static = dict(rows=rows.clone(), cos=cos.clone(), sin=sin.clone(), mask=mask.clone(),
                          start=torch.full((1,), start, device=rows.device, dtype=torch.long))
            if use_flashinfer:
                static["suffix_meta"] = tuple(value.clone() for value in suffix_meta)
            width = mask.shape[-1]

            def step():
                return self._refresh_cached_lean(
                    fused, static["rows"], static["cos"], static["sin"], static["mask"], 0,
                    width, tokens, tokens, start_slot=static["start"],
                    suffix_meta=static.get("suffix_meta"))

            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(2):
                    step()
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                output = step()
            expected = step().clone()
            graph.replay()
            if not torch.equal(output, expected):
                raise RuntimeError("refresh graph replay byte parity failed")
            entry = graphs[1][key] = (graph, static, output)
        graph, static, output = entry
        static["rows"].copy_(rows)
        static["cos"].copy_(cos)
        static["sin"].copy_(sin)
        static["mask"].copy_(mask)
        if use_flashinfer:
            for destination, source in zip(static["suffix_meta"], suffix_meta):
                destination.copy_(source)
        static["start"].fill_(start)
        graph.replay()
        return output[:real]

    def _prepare_refresh_gate_up_pack(self):
        """Pack immutable gate/up weights once for refresh and optional C1 reuse."""
        if hasattr(self, "_refresh_gate_up_weights"):
            return
        self._refresh_gate_up_weights = tuple(
            torch.cat((part[15], part[16])).detach() for part in self._d1_lean_plan())

    def _prepare_refresh_kv_pack(self):
        """Reuse Seg A's K/V packs, or bind them once after checkpoint load."""
        if hasattr(self, "_refresh_kv_weights"):
            return
        layers = [layer for index, stream in enumerate(self.streams)
                  for layer in (stream, *self.extra_stream_layers[index])]
        self._refresh_kv_weights = tuple(
            layer._fused_kv[0] if hasattr(layer, "_fused_kv") else
            torch.cat((layer.k_proj.weight, layer.v_proj.weight)).detach()
            for layer in layers)

    def _refresh_cached_lean(self, fused, rows, cos, sin, aligned, start, keys_total, tokens,
                             real, start_slot=None, suffix_meta=None,
                             allow_flashinfer=True, chain_flashinfer=False, refresh_meta=None):
        """JETSPEC_REFRESH_LEAN=1 (with JETSPEC_C1_NORM=wide): refresh_cached's layer loop
        on the served adapter's lean eager ops (_d1_cache_query_lean's kernels at this
        token count): the wide norm with the residual add folded into the next norm,
        one silu*up launch, the fused head-norm rotary for q and for k written straight
        into the cache with v, attention defaulting to the memory-efficient kernel over
        the 16-aligned mask, and F.linear on the bound weights."""
        if getattr(self, "target_text_config", None) is not None:
            if aligned is None:
                raise ValueError("hybrid cached refresh requires an explicit attention mask")
            first = start if start_slot is None else start_slot
            slots = torch.arange(tokens, device=rows.device, dtype=torch.long) + first
            return self._hybrid_cache_query(
                rows[None], cos, sin, aligned[..., :keys_total], None,
                window_slots=slots)[0, :real]
        norm = fused._wide_norm_extension().rms_norm
        add_norm = fused._lean_ops_extension().add_rms_norm
        silu_mul = fused._silu_mul_extension().silu_mul
        loop_ext = fused._loop_extension()
        from jetspec.engines.jetspec_plus_plus.kernels.attn_sm100 import attend; from jetspec.engines.jetspec_plus_plus.kernels.attn_sm100 import attend_refresh; from jetspec.engines.jetspec_plus_plus.kernels.attn_sm100 import flashinfer_enabled
        head_ops = loop_ext
        if os.environ.get("JETSPEC_C1_GLUON", "0") == "1":
            import jetspec.engines.jetspec_plus_plus.kernels.c1_gluon as c1_gluon
            if c1_gluon.available(rows):
                head_ops = c1_gluon
        use_flashinfer = allow_flashinfer and flashinfer_enabled(
            rows.device, rows.dtype, chain=chain_flashinfer)
        if use_flashinfer:
            window_rows = getattr(self, "_c1_window", 15)
            if suffix_meta is None:
                raise ValueError("FlashInfer refresh requires device suffix metadata")
            if start_slot is None:
                if not hasattr(self, "_flashinfer_refresh_start"):
                    self._flashinfer_refresh_start = torch.empty(
                        1, device=rows.device, dtype=torch.long)
                self._flashinfer_refresh_start.fill_(start)
                start_slot = self._flashinfer_refresh_start
        cos, sin = cos.contiguous(), sin.contiguous()
        cos0, sin0 = cos[0], sin[0]
        u = rows
        outputs, pending = [], None
        window, current = u, 0
        kv_pack = os.environ.get("JETSPEC_REFRESH_KV_PACK", "0") == "1"
        gate_up_pack = os.environ.get("JETSPEC_REFRESH_GATEUP_PACKED", "0") == "1"
        linear = F.linear
        if os.environ.get("JETSPEC_REFRESH_GEMM_TUNE", "0") == "1":
            if not hasattr(self, "_refresh_gemm"):
                from jetspec.engines.jetspec_plus_plus.kernels.refresh_gemm import RefreshGemm

                self._refresh_gemm = RefreshGemm(rows.device)
            linear = self._refresh_gemm.linear
        gate_up_linear = linear
        if os.environ.get("JETSPEC_ADAPTER_GEMM_TUNE", "0") == "1":
            if not hasattr(self, "_adapter_gemm"):
                from jetspec.engines.jetspec_plus_plus.kernels.adapter_gemm import AdapterGemm

                self._adapter_gemm = AdapterGemm(rows.device)
            gate_up_linear = self._adapter_gemm.linear
        ew_fused = (os.environ.get("JETSPEC_ADAPTER_EW_FUSED", "0") == "1"
                    or os.environ.get("JETSPEC_C1_GLUE_FUSED", "0") == "1")
        if os.environ.get("JETSPEC_REFRESH_ADAPTER_LOOP", "0") == "1":
            packed = getattr(self, "_refresh_loop_packed", None)
            if packed is None or packed[0] is not self._c1_kv:
                weights, scalars, ints = [], [], []
                for (index, in_w, in_eps, q_w, qn_w, qn_eps, kvn_w, kvn_eps, k_w,
                     kn_w, kn_eps, v_w, o_w, post_w, post_eps, gate_w, up_w, down_w,
                     heads, kv_heads, groups, head_dim, scaling) in self._d1_lean_plan():
                    weights += [in_w, q_w, qn_w, kvn_w, k_w, kn_w, v_w, o_w,
                                post_w, gate_w, up_w, down_w]
                    scalars += [in_eps, qn_eps, kvn_eps, kn_eps, post_eps, scaling]
                    ints += [index, heads, kv_heads, groups, head_dim]
                caches = [tensor for pair in self._c1_kv for tensor in pair]
                packed = self._refresh_loop_packed = (self._c1_kv, weights, scalars, ints, caches)
            ops = dict(norm=norm, add_norm=add_norm, silu_mul=silu_mul,
                       q_norm_rotary=loop_ext.q_norm_rotary,
                       k_norm_rotary_cache=loop_ext.k_norm_rotary_cache)
            if rows.is_cuda:
                ops.update(native_norm=fused.native_entry(fused._wide_norm_extension(), 'rms_norm'),
                           native_add=fused.native_entry(fused._lean_ops_extension(), 'add_rms_norm'),
                           native_silu=fused.native_entry(fused._silu_mul_extension(), 'silu_mul'))
            if linear is not F.linear:
                ops['linear'] = linear
            if ew_fused:
                ops['ew_pre_norm'] = fused.ew_pre_norm
            finish = os.environ.get("JETSPEC_C1_GLUE_FUSED", "0") == "1"
            if finish:
                ops['finish_streams'] = fused.finish_streams
            if refresh_meta is not None:
                from jetspec.engines.jetspec_plus_plus.kernels.refresh_attn import attend as attend_lean_refresh
                tune = os.environ.get("JETSPEC_REFRESH_ATTN_TUNE", "0") == "1"
                ops['attention'] = lambda q, k, v, scale, groups: attend_lean_refresh(
                    q, k[:, ::groups, :keys_total], v[:, ::groups, :keys_total],
                    scale, *refresh_meta, tune=tune)
            elif use_flashinfer:
                ops['attention'] = lambda q, k, v, scale, groups: attend_refresh(
                    q, k[:, :, :keys_total], v[:, :, :keys_total], scale,
                    self._flashinfer_refresh_plan, start_slot, window_rows, suffix_meta)
            else:
                from jetspec.engines.jetspec_plus_plus.kernels.attn_sm80 import enabled as sm80_enabled
                from jetspec.engines.jetspec_plus_plus.kernels.attn_sm100 import native_enabled
                if (rows.device.type != 'cuda' or native_enabled(rows.device)
                        or (keys_total > 512 and sm80_enabled(rows.device, rows.dtype))):
                    ops['attention'] = lambda q, k, v, scale, groups: attend(
                        q, k[:, :, :keys_total], v[:, :, :keys_total],
                        aligned[..., :keys_total].expand(1, q.shape[1], tokens, keys_total), scale)
            return loop_ext.refresh_loop(
                rows, cos, sin, aligned, start, keys_total, tokens, real, start_slot,
                *packed[1:], list(self._refresh_kv_weights) if kv_pack else [], ops,
                ew_fused, finish)
        qk_fused = os.environ.get("JETSPEC_REFRESH_QK_FUSED", "0") == "1"
        query_layout = os.environ.get("JETSPEC_ADAPTER_QUERY_LAYOUT_FUSED", "0") == "1"
        for block, (index, in_w, in_eps, q_w, qn_w, qn_eps, kvn_w, kvn_eps, k_w, kn_w, kn_eps,
                    v_w, o_w, post_w, post_eps, gate_w, up_w, down_w, heads, kv_heads,
                    groups, head_dim, scaling) in enumerate(self._d1_lean_plan()):
            if head_dim != 128:
                raise ValueError("JETSPEC_REFRESH_LEAN requires head_dim 128")
            if ew_fused:
                normed, u, source = fused.ew_pre_norm(
                    u, window, in_w, kvn_w.unsqueeze(0), in_eps, kvn_eps,
                    pending, kv_from_state=index != current)
                source = source[0]
            elif pending is None:
                normed = norm(u, in_w, in_eps)
            else:
                normed, u = add_norm(pending[0], pending[1], in_w, in_eps)
            if index != current:
                outputs.append(u[:real])
                window, current = u, index
            q = (fused.query_output(u, heads, tokens, head_dim) if qk_fused and query_layout
                 else u.new_empty((1, heads, tokens, head_dim)))
            if qk_fused:
                projected_q = linear(normed, q_w)
            else:
                head_ops.q_norm_rotary(linear(normed, q_w), qn_w, cos, sin, q, tokens,
                                       heads, qn_eps)
            if not ew_fused:
                source = norm(window, kvn_w, kvn_eps)
            k, v = self._c1_kv[block]
            # start_slot (the graph): rows from slot 0, the kernel offsets them
            first = 0 if start_slot is not None else start
            key_out = k[0, :, first:first + tokens].transpose(0, 1)
            value_out = v[0, :, first:first + tokens].transpose(0, 1)
            if kv_pack:
                projected_key, projected_value = linear(
                    source, self._refresh_kv_weights[block]).split(
                        (k_w.shape[0], v_w.shape[0]), dim=-1)
            else:
                projected_value = linear(source, v_w)
                projected_key = linear(source, k_w)
            if qk_fused:
                loop_ext.qk_norm_rotary_cache(projected_q, projected_key, projected_value,
                    qn_w, kn_w, cos, sin, q, key_out, value_out, tokens, heads, kv_heads,
                    groups, qn_eps, kn_eps, start_slot)
            else:
                head_ops.k_norm_rotary_cache(projected_key, projected_value, kn_w, cos0, sin0,
                                             key_out, value_out, tokens, kv_heads, groups, kn_eps,
                                             start_slot)
            if refresh_meta is not None:
                from jetspec.engines.jetspec_plus_plus.kernels.refresh_attn import attend as attend_lean_refresh

                attended = attend_lean_refresh(
                    q, k[:, ::groups, :keys_total], v[:, ::groups, :keys_total],
                    scaling, *refresh_meta, tune=os.environ.get("JETSPEC_REFRESH_ATTN_TUNE", "0") == "1")
            elif use_flashinfer:
                attended = attend_refresh(
                    q, k[:, :, :keys_total], v[:, :, :keys_total], scaling,
                    self._flashinfer_refresh_plan, start_slot, window_rows, suffix_meta)
            else:
                attended = attend(
                    q, k[:, :, :keys_total], v[:, :, :keys_total],
                    aligned[..., :keys_total].expand(1, heads, tokens, keys_total),
                    scaling)
            projected = linear(attended.transpose(1, 2).reshape(tokens, -1), o_w)
            post, state = add_norm(projected, u, post_w, post_eps)
            if gate_up_pack:
                gate, up = gate_up_linear(post, self._refresh_gate_up_weights[block]).chunk(2, dim=-1)
                pending = (linear(silu_mul(gate, up), down_w), state)
            else:
                pending = (linear(silu_mul(linear(post, gate_w), linear(post, up_w)), down_w),
                           state)
        if os.environ.get("JETSPEC_C1_GLUE_FUSED", "0") == "1":
            return fused.finish_streams(outputs, pending[1], pending[0], real)
        u = pending[1] + pending[0]
        outputs.append(u[:real])
        return torch.cat(outputs, dim=-1)

    def _refresh_cached_plain(self, fused, block_rows, cos, sin, mask, start, keys_total,
                              tokens, real):
        def norm(x, module):
            return fused.rms_norm(x, module.weight, module.variance_epsilon)

        u = block_rows[None]
        outputs, block = [], 0
        for index, stream in enumerate(self.streams):
            window = u
            for layer in (stream, *self.extra_stream_layers[index]):
                heads, kv_heads, dim = layer.num_heads, layer.num_key_value_heads, layer.head_dim
                q = fused.fused_q_postprocess(
                    layer.q_proj(norm(u, layer.input_norm)), layer.q_norm.weight, cos, sin,
                    layer.q_norm.variance_epsilon, heads, dim)
                cached_k, cached_v = self._c1_kv[block]
                source = norm(window, layer.kv_norm)
                k = fused.fused_q_postprocess(
                    layer.k_proj(source), layer.k_norm.weight, cos, sin,
                    layer.k_norm.variance_epsilon, kv_heads, dim)
                v = layer.v_proj(source).view(1, tokens, kv_heads, dim).transpose(1, 2)
                groups = layer.num_key_value_groups
                cached_k[:, :, start:keys_total].view(1, kv_heads, groups, tokens, dim).copy_(
                    k[:, :, None].expand(1, kv_heads, groups, tokens, dim))
                cached_v[:, :, start:keys_total].view(1, kv_heads, groups, tokens, dim).copy_(
                    v[:, :, None].expand(1, kv_heads, groups, tokens, dim))
                attended = F.scaled_dot_product_attention(
                    q, cached_k[:, :, :keys_total], cached_v[:, :, :keys_total],
                    attn_mask=mask, scale=layer.scaling)
                state = u + layer.o_proj(attended.transpose(1, 2).reshape(1, tokens, heads * dim))
                u = state + layer.mlp(norm(state, layer.post_attention_norm))
                block += 1
            outputs.append(u[0, :real])
        return torch.cat(outputs, dim=-1)


def verify_pinned_files(paths, pins_path) -> None:
    """Check every consumed file against the dispatch-owned pin manifest."""
    if pins_path is None:
        raise ValueError("the new stack requires JETSPEC_PINS=pins_w258.json")
    pins = json.loads(Path(pins_path).read_text())
    expected = {}
    for checkpoint in pins["checkpoints"].values():
        directory = Path(checkpoint["directory"]).resolve()
        for name, item in checkpoint["files"].items():
            expected[directory / name] = item["sha256"]
    for path in paths:
        path = Path(path)
        path = path.parent.resolve() / path.name  # an HF snapshot symlinks each file to a blob
        if path not in expected:
            raise ValueError(f"consumed file is not pinned: {path}")
        actual = _file_sha256(path)
        if actual != expected[path]:
            raise AssertionError(f"sha256 mismatch: {path} expected={expected[path]} actual={actual}")


def load_qwen3_config(path):
    """Load the checkpoint's theta even when a newer writer nests it under rope_parameters."""
    config_file = Path(path)
    if config_file.is_dir():
        config_file /= "config.json"
    raw = json.loads(config_file.read_text())
    parameters = raw.get("rope_parameters")
    if parameters is not None and not isinstance(parameters, dict):
        raise ValueError(f"{config_file}: rope_parameters must be an object")
    rope_type = (parameters or {}).get("rope_type", raw.get("rope_type"))
    if rope_type not in (None, "default"):
        raise ValueError(f"{config_file}: unsupported rope_type {rope_type!r}")
    if any(name in raw or name in (parameters or {})
           for name in ("partial_rotary_factor", "partial_rotary_ratio", "partial_rotary_dim")):
        raise ValueError(f"{config_file}: partial rotary is unsupported for the drafter")
    config = Qwen3Config.from_pretrained(path)
    if "rope_theta" not in raw:
        theta = (parameters or {}).get("rope_theta")
        if not isinstance(theta, (int, float)) or isinstance(theta, bool) or not math.isfinite(theta) or theta <= 0:
            raise ValueError(f"{config_file}: rope_theta is absent or invalid")
        config.rope_theta = float(theta)
    return config


def load_frozen_h2h_adapter(*, config_path, head_path, target_embed_tokens, device, dtype,
                            pins_path=None):
    config = load_qwen3_config(config_path)
    mode = config.dflash_config.get("ctxkv_head", {}).get("mode")
    kwargs = dict(config_path=config_path, head_path=head_path,
                  target_embed_tokens=target_embed_tokens, device=device, dtype=dtype)
    if mode == "l1_attention_h2h":
        adapter = FrozenH2HAdapter(**kwargs)
    elif mode == "c1_tap_aligned_h2h":
        adapter = FrozenC1H2HAdapter(**kwargs, pins_path=pins_path)
    else:
        raise ValueError(f"unsupported served ctxkv head mode: {mode!r}")
    run_manifest = _run_manifest()
    config_file = Path(config_path)
    run_manifest.record_file(
        "adapter/config", config_file / "config.json" if config_file.is_dir() else config_file)
    run_manifest.record_file("adapter/head", head_path)
    if pins_path is not None:
        run_manifest.record_file("pins", pins_path)
    run_manifest.record_value("adapter_mode", mode)
    return adapter


class FrozenC1H2HAdapter(FrozenH2HAdapter):
    """Strict C1 checkpoint load with file pins; retain the frozen-head checks."""

    def __init__(self, *, config_path, head_path, target_embed_tokens, device, dtype, pins_path):
        nn.Module.__init__(self)
        config_file = Path(config_path)
        if config_file.is_dir():
            config_file = config_file / "config.json"
        verify_pinned_files([config_file, head_path], pins_path)
        with safe_open(head_path, framework="pt") as head_file:
            if (head_file.metadata() or {}).get("PREFIX_KV_SOURCE") is not None:
                raise ValueError("head.safetensors PREFIX_KV_SOURCE requires target cache K/V, which serve cannot feed")
        config = load_qwen3_config(config_path)
        config._attn_implementation = "sdpa"
        self.head = ServingCtxKVC1Head(config).to(device=device, dtype=dtype)
        self.head.load_state_dict(load_file(Path(head_path), device="cpu"), strict=True)
        self.head.eval()
        self.target_embed_tokens = target_embed_tokens
        for module in (self.head, self.target_embed_tokens):
            for parameter in module.parameters():
                parameter.requires_grad_(False)
        self._sealed_state_sha256 = tensor_state_sha256(self.head)
        if os.environ.get("JETSPEC_FUSED_QKV", "0") == "1":
            from jetspec.engines.jetspec_plus_plus.draft.projections import prepare_kv_group
            layers = [layer for index, stream in enumerate(self.head.streams)
                      for layer in (stream, *self.head.extra_stream_layers[index])]
            prepare_kv_group(layers)
        if os.environ.get("JETSPEC_C1_SPAN_KV_FUSED", "0") == "1":
            from jetspec.engines.jetspec_plus_plus.draft.projections import prepare_c1_span_kv
            prepare_c1_span_kv(self.head, reuse_append=(
                os.environ.get("JETSPEC_C1_APPEND_FOLD", "0") == "1"))
        if os.environ.get("JETSPEC_REFRESH_KV_PACK", "0") == "1":
            self.head._prepare_refresh_kv_pack()
        if (os.environ.get("JETSPEC_REFRESH_GATEUP_PACKED", "0") == "1"
                or os.environ.get("JETSPEC_C1_GATEUP_PACKED", "0") == "1"):
            self.head._prepare_refresh_gate_up_pack()


CORRECTION_HEAD_PREFIX = "markov_head."
ADAPTED_MARK_TENSOR = "adapted_position_embedding.weight"
RETIRED_DRAFTER = "the 59-tensor drafter is retired (2026-09-22)"


def resolve_stock_drafter(dflash_config, source) -> bool:
    """True for the stock 58-tensor Jacobi drafter, False for a mask-initialised DFlash-class head (58 tensors,
    no Jacobi forcing; served only by the fresh-only sync load). A Jacobi-forcing config that is not stock
    builds the retired 59-tensor model, so it is refused before any model code runs."""
    dflash_config = dflash_config or {}
    stock = dflash_config.get("stock_drafter", False)
    if not isinstance(stock, bool):
        raise ValueError(f"{source}: dflash_config.stock_drafter must be true or false, got {stock!r}")
    jacobi = dflash_config.get("jacobi_forcing", False)
    if not isinstance(jacobi, bool):
        raise ValueError(f"{source}: dflash_config.jacobi_forcing must be true or false, got {jacobi!r}")
    if jacobi and not stock:
        raise ValueError(f"{source}: {RETIRED_DRAFTER}; a Jacobi-forcing drafter must declare "
                         "dflash_config.stock_drafter true")
    if stock and dflash_config.get("fresh_fraction", 1.0) != 1.0:
        raise ValueError(f"{source}: a stock drafter is trained all-fresh; stock_drafter is true but "
                         f"fresh_fraction is {dflash_config.get('fresh_fraction')!r}")
    # Serve draws J0 from random earlier context; a mask-noise drafter was trained on [anchor, mask x 15].
    noise_source = dflash_config.get("noise_source", "context")
    if noise_source != "context":
        raise ValueError(f"{source}: dflash_config.noise_source is {noise_source!r}; serve initialises J0 from "
                         "random context only")
    return stock


def resolve_causal_head(dflash_config, source) -> bool:
    """``dflash_config.causal_head`` must be true: propose drafts with ``is_causal=True`` and serve has
    no bidirectional path. The public engine reads false or absent as a bidirectional block, so a
    checkpoint declaring either would be drafted here under a mask it was not trained with."""
    value = (dflash_config or {}).get("causal_head")
    if value is None:
        raise ValueError(f"{source}: dflash_config.causal_head is absent, which the public engine reads as a "
                         "bidirectional block; serve drafts causally only, so a checkpoint must declare it true")
    if not isinstance(value, bool):
        raise ValueError(f"{source}: dflash_config.causal_head must be true or false, got {value!r}")
    if not value:
        raise ValueError(f"{source}: dflash_config.causal_head is false (a bidirectional block); serve drafts "
                         "causally only")
    return value


def check_stock_drafter_keys(head: nn.Module, state, source) -> None:
    """A stock drafter's file holds exactly the model's keys and no adapted mark, checked BEFORE
    the zero mark is attached so the attachment can never stand in for a tensor the file lacks."""
    if ADAPTED_MARK_TENSOR in state:
        raise ValueError(f"{source}: {RETIRED_DRAFTER}; the file carries {ADAPTED_MARK_TENSOR}")
    expected = set(head.state_dict())
    if ADAPTED_MARK_TENSOR in expected:
        raise ValueError(f"{source}: the bundled dflash.py builds the retired mark "
                         f"{ADAPTED_MARK_TENSOR}")
    if set(state) != expected:
        raise ValueError(f"{source}: stock drafter key set differs: missing={sorted(expected - set(state))} "
                         f"unexpected={sorted(set(state) - expected)}")


class StockAdaptedMark(nn.Module):
    """The adapted mark of a drafter that has none: two rows of negative zero.

    Every serving site adds ``adapted_position_embedding.weight[0|1]`` to the context, several of them
    inside captured graphs. x + (-0.0) is x bit for bit for every x (x + (+0.0) would turn a -0.0 into
    +0.0), so a stock drafter walks the same sites and sees the context it was trained on. The weight is
    a non-persistent buffer: it is not a parameter, never enters ``state_dict`` and was never loaded."""

    def __init__(self, width: int, *, dtype: torch.dtype, device: torch.device) -> None:
        super().__init__()
        self.register_buffer("weight", torch.full((2, int(width)), -0.0, dtype=dtype, device=device),
                             persistent=False)

    def forward(self, indicator: torch.Tensor) -> torch.Tensor:
        return F.embedding(indicator, self.weight)


CORRECTION_HEAD_ENV = "JETSPEC_DRAFTER_CORRECTION_HEAD"


def strip_correction_head_state(state: dict) -> dict:
    """Drop the co-trained correction head tensors; the drafter loads its stock base keys."""
    return {key: value for key, value in state.items() if not key.startswith(CORRECTION_HEAD_PREFIX)}


def disable_correction_head_config(config) -> None:
    """Keep the bundled module from building its (relative-import) head."""
    if getattr(config, "markov_head", None) is not None:
        config.markov_head = None
    dflash_config = getattr(config, "dflash_config", None)
    if isinstance(dflash_config, dict):
        dflash_config.pop("markov_head", None)


class CorrectionHead:
    """Order-1 bigram correction: bias(prev) = W2 . W1[prev], applied at the row logits.

    Tables are the co-trained head's (V, r) embedding W1 and (V, r) projection
    W2, in the drafter dtype (the reference read casts the tables to bf16).
    """

    def __init__(self, w1: torch.Tensor, w2: torch.Tensor) -> None:
        if w1.dim() != 2 or w1.shape != w2.shape:
            raise ValueError(f"correction tables must be two (V, r) tensors, got {tuple(w1.shape)} {tuple(w2.shape)}")
        self.w1, self.w2 = w1, w2
        self.vocab_size, self.rank = w1.shape
        self.calls = 0

    def bias(self, prev_tokens: torch.Tensor) -> torch.Tensor:
        """(R,) previous tokens -> (R, V) logit bias in the table dtype."""
        return F.linear(self.w1[prev_tokens.long()], self.w2)

    def chain(self, logits: torch.Tensor, anchors: torch.Tensor, pinned=None) -> torch.Tensor:
        """Spine read: slot d = argmax(logits[d] + bias(token at d-1)); slot 0 keys on the anchor.

        Sequential across slots (each slot keys on the corrected token before
        it), batched across rows. An optional pinned slot-0 token replaces the
        slot-0 read and keys slot 1.
        """
        if logits.dim() != 3 or logits.shape[-1] != self.vocab_size:
            raise ValueError(f"logits must be (R, D, V={self.vocab_size}); got {tuple(logits.shape)}")
        self.calls += 1
        prev = anchors.reshape(-1).long()
        tokens = []
        for depth in range(logits.shape[1]):
            if depth == 0 and pinned is not None:
                token = torch.as_tensor(pinned, device=logits.device, dtype=torch.long).reshape(-1)
            else:
                bias = self.bias(prev).to(dtype=logits.dtype)
                token = (logits[:, depth] + bias).argmax(dim=-1)
            tokens.append(token)
            prev = token
        return torch.stack(tokens, dim=1)




def served_target_layer_ids(config=None) -> tuple[int, ...]:
    """The configured taps, or Qwen3's legacy default for callers without a config.

    Read from the module's source rather than imported: qwen3.py imports the attention
    layers (triton, flashinfer), and this module otherwise loads on a CPU-only host.
    """
    configured = getattr(config, "jetspec_target_layer_ids", None)
    if configured is not None:
        return tuple(configured)
    import ast
    source = Path(__file__).resolve().parents[1] / "models" / "qwen3.py"
    for node in ast.parse(source.read_text()).body:
        if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "JETSPEC_TARGET_LAYER_IDS"
                for target in node.targets):
            return tuple(int(layer) for layer in ast.literal_eval(node.value))
    raise RuntimeError(f"{source} no longer assigns JETSPEC_TARGET_LAYER_IDS")


def check_rotary_agreement(drafter_freq, adapter_freq, *, target_text_config=None) -> None:
    """Check legacy shared RoPE or the hybrid adapter's declared target RoPE.

    The adapter loads in the serving dtype, which casts its inv_freq buffer (bf16 on the served recipe)
    while the drafter keeps fp32, so compare at the adapter's precision.
    """
    expected = drafter_freq
    if target_text_config is not None:
        # A trained hybrid adapter rotates target D256/partial-RoPE channels;
        # the separately trained drafter retains its own full D128 geometry.
        rope = target_text_config["rope_parameters"]
        dim = int(target_text_config["head_dim"] * rope.get("partial_rotary_factor", 1.0))
        expected = 1.0 / (rope["rope_theta"] ** (
            torch.arange(0, dim, 2, dtype=torch.float32, device=adapter_freq.device) / dim))
    if not torch.equal(expected.to(adapter_freq.dtype), adapter_freq):
        raise ValueError("JetSpec drafter/adapter inv_freq disagree")


def check_tap_contract(dflash_config, source, *, served, required=True) -> None:
    """Require a checkpoint's tap declaration to match the served target taps."""
    declared = (dflash_config or {}).get("target_layer_ids")
    if declared is None:
        if required:
            raise ValueError(f"{source} declares no target_layer_ids; the served taps "
                             f"{served} cannot be checked against it")
        return
    if tuple(int(layer) for layer in declared) != served:
        raise ValueError(f"tap mismatch: {source} declares target_layer_ids {tuple(declared)} "
                         f"but the target serves {served}")


class JetSpecDrafter:
    """Self-contained block drafter used by the speculator process."""

    def __init__(
        self,
        *,
        head: nn.Module,
        adapter,
        embed_tokens: nn.Module,
        lm_head: nn.Module,
        device: torch.device,
        seed: int = 9504,
        convergence_limit: int = 64,
        shared_prefix_projection: bool = True,
        correction_head: CorrectionHead | None = None,
    ) -> None:
        self.head = head
        self.adapter = adapter
        self.correction_head = correction_head
        self.embed_tokens = embed_tokens
        self.lm_head = lm_head
        self.device = torch.device(device)
        self.dtype = next(head.parameters()).dtype
        self.seed = int(seed)
        self.convergence_limit = int(convergence_limit)
        self.shared_prefix_projection = bool(shared_prefix_projection)
        if int(head.block_size) != 16:
            raise ValueError("the pinned drafter requires block_size=16")
        from jetspec.engines.jetspec_plus_plus.wire.protocol import draft_width
        # The synchronous JetSpec baseline never sets JETSPEC_DRAFTS; keep its pre-width default (14).
        width = draft_width() if 'JETSPEC_DRAFTS' in os.environ else 14
        self.block_size = 16 if width == 14 else width + 1
        if any("clean_prefix_embedding" in name for name, _ in head.named_parameters()):
            raise AssertionError("p-free Jacobi serving forbids clean_prefix_embedding")

    @classmethod
    def load(
        cls,
        checkpoint_dir: str,
        adapter_config: str,
        adapter_head: str,
        target_release_weights: str,
        device: str | torch.device,
        *,
        fresh_only: bool = False,
    ) -> "JetSpecDrafter":
        """Load only the checkpoint-owned drafter, adapter, embedding, and head.

        ``fresh_only`` is the synchronous one-GPU baseline's load (sync_drafter.py): the drafter, the
        embedding and the LM head, and nothing else. No adapter is read (``adapter_config`` and
        ``adapter_head`` must be None), no adapted mark is read or attached, and a head that is
        not Jacobi-forcing (a mask-initialised DFlash-class drafter) is accepted. The default is unchanged."""
        run_manifest = _run_manifest()
        checkpoint = Path(checkpoint_dir)
        config = load_qwen3_config(checkpoint)
        stock_drafter = resolve_stock_drafter(config.dflash_config, checkpoint / "config.json")
        resolve_causal_head(config.dflash_config, checkpoint / "config.json")

        pins_path = os.environ.get("JETSPEC_PINS")
        if fresh_only and (adapter_config is not None or adapter_head is not None):
            raise ValueError("a fresh-only load reads no adapter; pass adapter_config=None, adapter_head=None")
        if not fresh_only and not stock_drafter:
            raise ValueError(f"{checkpoint / 'config.json'}: checkpoint is not a Jacobi-forcing head; a "
                             "mask-initialised drafter is served only by the fresh-only sync load")
        adapter_model_config = None if fresh_only else load_qwen3_config(adapter_config)
        adapter_mode = None if fresh_only else adapter_model_config.dflash_config.get("ctxkv_head", {}).get("mode")
        if adapter_mode == "c1_tap_aligned_h2h" and pins_path is None:
            raise ValueError("the new stack requires JETSPEC_PINS before checkpoint import")
        if pins_path is not None:
            verify_pinned_files([checkpoint / "config.json", checkpoint / "dflash.py",
                                 checkpoint / "model.safetensors", target_release_weights], pins_path)
        model_file = checkpoint / "dflash.py"
        if not model_file.is_file():
            raise FileNotFoundError(f"checkpoint lacks bundled dflash.py: {model_file}")
        module_name = "jetspec_checkpoint_dflash"
        spec = importlib.util.spec_from_file_location(module_name, model_file)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot import {model_file}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

        declared_taps = (config.dflash_config or {}).get("target_layer_ids")
        if declared_taps is None:
            raise ValueError(f"{checkpoint / 'config.json'} declares no target_layer_ids")
        if (not isinstance(declared_taps, list) or not declared_taps
                or any(type(layer) is not int or layer < 0 or layer >= config.num_target_layers
                       for layer in declared_taps)
                or any(left >= right for left, right in zip(declared_taps, declared_taps[1:]))):
            raise ValueError(f"{checkpoint / 'config.json'}: target_layer_ids must be strictly "
                             f"increasing integers below num_target_layers={config.num_target_layers}")
        served_taps = tuple(declared_taps)
        check_tap_contract(config.dflash_config, checkpoint / "config.json", served=served_taps)
        if not fresh_only:
            check_tap_contract(adapter_model_config.dflash_config, adapter_config, served=served_taps)
        config._attn_implementation = "sdpa"
        disable_correction_head_config(config)
        previous_default = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.bfloat16)
            head = module.DFlashDraftModel(config)
        finally:
            torch.set_default_dtype(previous_default)
        if getattr(head, "target_layer_ids", None) is not None:
            check_tap_contract({"target_layer_ids": getattr(head, "target_layer_ids", None)},
                               f"{model_file} (taps derived by the bundle)", served=served_taps)
        if getattr(head, "markov_head", None) is not None:
            raise ValueError("the drafter must not build its own correction head")
        with safe_open(checkpoint / "model.safetensors", framework="pt") as drafter_file:
            check_stock_drafter_keys(head, strip_correction_head_state(dict.fromkeys(drafter_file.keys())),
                                     checkpoint / "model.safetensors")
        head.load_state_dict(
            strip_correction_head_state(load_file(checkpoint / "model.safetensors", device="cpu")),
            strict=True,
        )
        if not fresh_only:
            converter_path = Path(adapter_head).with_name("converter.safetensors")
            ctxkv = adapter_model_config.dflash_config.get("ctxkv_head", {})
            if "projection_weight" not in ctxkv:
                raise ValueError("ctxkv_head.projection_weight is required; no serving default is defined")
            if adapter_mode == "c1_tap_aligned_h2h":
                projection_weight = resolve_ctxkv_head_config(adapter_model_config)["projection_weight"]
            else:
                projection_weight = ctxkv["projection_weight"]
                if (not isinstance(projection_weight, (int, float)) or isinstance(projection_weight, bool)
                        or not math.isfinite(projection_weight) or projection_weight < 0):
                    raise ValueError("ctxkv_head.projection_weight must be finite and non-negative")
            if converter_path.is_file():
                if pins_path is not None:
                    verify_pinned_files([converter_path], pins_path)
                converter = load_file(converter_path, device="cpu")
                if projection_weight == 0:
                    provenance = {}
                    for key in ("fc.weight", "hidden_norm.weight"):
                        recorded = converter.get(key)
                        if recorded is None:
                            raise ValueError(f"adapter converter lacks {key}")
                        provenance[key] = hashlib.sha256(
                            recorded.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
                    run_manifest.record_value("adapter_converter_provenance", provenance)
                else:
                    for key in ("fc.weight", "hidden_norm.weight"):
                        expected = head.state_dict()[key].detach().cpu().contiguous()
                        recorded = converter.get(key)
                        if (recorded is None or recorded.dtype != expected.dtype
                                or recorded.shape != expected.shape
                                or not torch.equal(recorded.contiguous().view(torch.uint8),
                                                   expected.view(torch.uint8))):
                            raise ValueError(f"adapter converter mismatch with served drafter: {key}")
                run_manifest.record_file("adapter/converter", converter_path)
        for name in ("config.json", "dflash.py", "model.safetensors"):
            run_manifest.record_file(f"drafter/{name}", checkpoint / name)
        if not fresh_only and not getattr(head, "jacobi_forcing", False):
            raise ValueError("checkpoint is not a Jacobi-forcing head")
        run_manifest.record_value("stock_drafter", stock_drafter)
        run_manifest.record_value("causal_head", True)
        if fresh_only:
            # no context row is ever adapter-predicted and the file carries no mark, so none is attached
            adapted_mark = "fresh_only_absent"
        else:
            # attached only after the strict load saw the file's own key set, and outside state_dict
            parameter = next(head.parameters())
            head.adapted_position_embedding = StockAdaptedMark(
                head.fc.in_features, dtype=parameter.dtype, device=parameter.device)
            adapted_mark = "stock"
        run_manifest.record_value("adapted_mark", adapted_mark)
        correction_head = None
        correction_path = os.environ.get(CORRECTION_HEAD_ENV, "")
        if correction_path:
            correction_head = load_correction_head(
                correction_path, device=device, dtype=next(head.parameters()).dtype, pins_path=pins_path)
            print(f"JETSPEC_CORRECTION_HEAD={correction_path} rank={correction_head.rank}", flush=True)
            run_manifest.record_file("correction_head", correction_path)

        release = load_file(Path(target_release_weights), device="cpu")
        run_manifest.record_file("target_release_weights", target_release_weights)
        expected_keys = {"lm_head.weight", "model.embed_tokens.weight"}
        if set(release) != expected_keys:
            raise ValueError(
                f"target release keys differ: expected={sorted(expected_keys)} "
                f"actual={sorted(release)}"
            )
        vocab_size, hidden_size = release["lm_head.weight"].shape
        embed_tokens = nn.Embedding(vocab_size, hidden_size, dtype=torch.bfloat16)
        lm_head = nn.Linear(hidden_size, vocab_size, bias=False, dtype=torch.bfloat16)
        embed_tokens.load_state_dict(
            {"weight": release["model.embed_tokens.weight"]}, strict=True
        )
        lm_head.load_state_dict({"weight": release["lm_head.weight"]}, strict=True)

        resolved_device = torch.device(device)
        head = head.to(resolved_device).eval()
        embed_tokens = embed_tokens.to(resolved_device).eval()
        lm_head = lm_head.to(resolved_device).eval()
        for module_to_freeze in (head, embed_tokens, lm_head):
            for parameter in module_to_freeze.parameters():
                parameter.requires_grad_(False)
        if os.environ.get("JETSPEC_FUSED_QKV", "0") == "1":
            from jetspec.engines.jetspec_plus_plus.draft.projections import prepare_qkv
            for layer in head.layers:
                prepare_qkv(layer.self_attn)
        if os.environ.get("JETSPEC_CTX_KV_FUSED", "0") == "1":
            from jetspec.engines.jetspec_plus_plus.draft.projections import prepare_context_kv
            prepare_context_kv(head)
        if os.environ.get("JETSPEC_REFRESH_INJECT_LEAN2", "0") == "1":
            from jetspec.engines.jetspec_plus_plus.draft.refresh_inject import prepare
            prepare(head)
        if fresh_only:
            adapter = None
            print("JETSPEC_ADAPTER=absent (fresh-only load)", flush=True)
        else:
            adapter = load_frozen_h2h_adapter(
                config_path=adapter_config,
                head_path=adapter_head,
                target_embed_tokens=embed_tokens,
                device=resolved_device,
                dtype=next(head.parameters()).dtype,
                pins_path=pins_path,
            )
            digest = adapter.assert_byte_identical()
            print(f"JETSPEC_ADAPTER_STATE_SHA256={digest}", flush=True)
            check_rotary_agreement(head.rotary_emb.inv_freq, adapter.head.streams[0].rotary_emb.inv_freq,
                                   target_text_config=getattr(adapter.head, "target_text_config", None))
        return cls(
            head=head,
            adapter=adapter,
            embed_tokens=embed_tokens,
            lm_head=lm_head,
            device=resolved_device,
            correction_head=correction_head,
        )




    @torch.inference_mode()
    def propose(
        self,
        seq: JetSpecSequence,
        *,
        mode: Mode,
        k_hypotheses: list[int],
        passes: int | None,
        abort_if: Callable[[], bool] | None = None,
    ) -> list[Block]:
        if mode not in {"token", "noise", "kth"}:
            raise ValueError("mode must be token, noise, or kth")
        hypotheses = [int(value) for value in k_hypotheses]
        if not hypotheses or len(set(hypotheses)) != len(hypotheses):
            raise ValueError("k_hypotheses must be non-empty and unique")
        if any(value < 0 or value >= self.block_size for value in hypotheses):
            raise ValueError("k_hypotheses must be inside [0, 16)")
        if passes is not None and passes < 1:
            raise ValueError("passes must be positive or None")

        contexts = self._target_contexts(seq, mode=mode, hypotheses=hypotheses)
        batch_size = len(hypotheses)
        context_ids = seq.committed_ids.to(device=self.device)
        first_block = build_random_context_block_token_ids(
            context_ids,
            block_size=self.block_size,
            seed=self.seed,
            prompt_index=seq.prompt_index,
            mode=mode,
        ).repeat(batch_size, 1)
        depth = self.block_size if mode == "noise" else self.block_size - 1

        # Batched refinement forward.
        def forward(*, fill_tokens, pass_index):
            del pass_index
            if fill_tokens is None:
                block_ids = first_block
            elif mode == "noise":
                if tuple(fill_tokens.shape) != (batch_size, self.block_size):
                    raise ValueError("noise passes must feed all 16 slots")
                block_ids = fill_tokens.to(device=self.device, dtype=context_ids.dtype)
            else:
                if tuple(fill_tokens.shape) != (batch_size, self.block_size - 1):
                    raise ValueError("later passes must feed all 15 variable slots")
                anchor = (
                    context_ids[:, -2:-1]
                    if mode == "kth"
                    else context_ids[:, -1:]
                ).repeat(batch_size, 1)
                block_ids = torch.cat(
                    [anchor, fill_tokens.to(device=self.device, dtype=context_ids.dtype)],
                    dim=1,
                )
            if bool((block_ids == int(self.head.mask_token_id)).any()):
                raise ValueError("Jacobi input must be mask-free")
            noise = self.embed_tokens(block_ids)
            position_ids = torch.arange(
                int(contexts.shape[1]) + self.block_size,
                dtype=torch.long,
                device=self.device,
            ).unsqueeze(0).repeat(batch_size, 1)
            hidden = self.head(
                target_hidden=contexts,
                noise_embedding=noise,
                position_ids=position_ids,
                past_key_values=None,
                use_cache=False,
                is_causal=True,
                attention_backend="sdpa",
            )
            if mode == "noise":
                logits = self.lm_head(hidden)[:, :depth, :]
                input_tokens = block_ids[:, :depth]
            else:
                logits = self.lm_head(hidden[:, 1:, :])[:, :depth, :]
                input_tokens = block_ids[:, 1 : 1 + depth]
            return DraftForwardPass(logits=logits, input_tokens=input_tokens)

        results = run_refinement_protocol(
            forward,
            batch_size=batch_size,
            depth=depth,
            max_passes=passes,
            convergence_limit=self.convergence_limit,
            abort_if=abort_if,
        )
        anchor_index = -2 if mode == "kth" else -1
        true_anchor = int(context_ids[0, anchor_index].item())
        blocks = []
        for hypothesis, result in zip(hypotheses, results):
            correction_guess = None
            if mode in {"noise", "kth"}:
                correction_guess = int(result.final_tokens[0].item())
                draft_ids = result.final_tokens[1:]
                logits = result.logits[1:]
            else:
                draft_ids = result.final_tokens
                logits = result.logits
            blocks.append(
                Block(
                    hypothesis=hypothesis,
                    mode=mode,
                    draft_ids=draft_ids.detach(),
                    logits=logits.detach(),
                    passes_run=result.passes_run,
                    converged=result.converged,
                    anchor=true_anchor,
                    correction_guess=correction_guess,
                )
            )
        return blocks


    @torch.inference_mode()
    def propose_protocol(
        self,
        seq: JetSpecSequence,
        *,
        mode: Mode,
        k_hypotheses: list[int],
        passes: int | None,
        abort_if: Callable[[], bool] | None = None,
    ) -> list[Block]:
        """Produce the fixed H-row wire payload used by the serving protocol.

        Kth mode verifies 14 drafts, so the verifier can return k=0..14.  The
        H=16 wire contract nevertheless carries an unused k=15 row.  Staging
        predicts the 15 reachable tail positions; duplicate its final hidden
        only while constructing that unreachable row.

        Unlike the initial proposal, each next-round hypothesis has a distinct
        preceding token: k=0 follows the current correction, while k>0 follows
        draft k-1 from the block currently being verified.  Keeping the initial
        round's single anchor for every row shifts correction guesses back one
        round and destroys the hit rate.
        """
        if mode != "kth":
            return self.propose(
                seq,
                mode=mode,
                k_hypotheses=k_hypotheses,
                passes=passes,
                abort_if=abort_if,
            )
        hypotheses = [int(value) for value in k_hypotheses]
        if not hypotheses or len(set(hypotheses)) != len(hypotheses):
            raise ValueError("k_hypotheses must be non-empty and unique")
        if any(value < 0 or value >= self.block_size for value in hypotheses):
            raise ValueError("k_hypotheses must be inside [0, 16)")
        if passes is not None and passes < 1:
            raise ValueError("passes must be positive or None")

        if seq.pending_block is None:
            raise RuntimeError("protocol proposal requires a staged pending block")
        predicted = seq.staged.predicted_hidden
        required = max(map(int, k_hypotheses)) + 1
        if predicted is None:
            raise RuntimeError("protocol proposal requires staged predicted hidden")
        if required > int(predicted.shape[1]) and (
            required != self.block_size
            or int(predicted.shape[1]) != self.block_size - 1
        ):
            raise ValueError("protocol hypotheses exceed the staged adapter window")
        if required > int(predicted.shape[1]):
            seq.staged.predicted_hidden = torch.cat(
                [predicted, predicted[:, -1:, :]], dim=1
            )

        try:
            contexts, attention_mask, position_ids = (
                build_protocol_hypothesis_contexts(
                    seq.exact_hidden(device=self.device, dtype=self.dtype),
                    predicted_hidden=seq.staged.predicted_hidden,
                    k_hypotheses=hypotheses,
                    adapted_position_embedding=self.head.adapted_position_embedding,
                    block_size=self.block_size,
                )
            )
            context_ids = seq.committed_ids.to(device=self.device)
            pending_drafts = seq.pending_block.draft_ids.to(
                device=self.device, dtype=context_ids.dtype
            )
            if int(pending_drafts.numel()) != self.block_size - 2:
                raise ValueError("protocol pending block must carry 14 drafts")
            anchors = []
            for hypothesis in hypotheses:
                if hypothesis == 0:
                    anchors.append(context_ids[0, -1])
                else:
                    anchors.append(
                        pending_drafts[
                            min(hypothesis - 1, pending_drafts.numel() - 1)
                        ]
                    )
            anchor_ids = torch.stack(anchors).reshape(-1, 1)
            batch_size = len(hypotheses)
            first_block = build_random_context_block_token_ids(
                context_ids,
                block_size=self.block_size,
                seed=self.seed,
                prompt_index=seq.prompt_index,
                mode=mode,
            ).repeat(batch_size, 1)
            first_block[:, :1] = anchor_ids
            depth = self.block_size - 1

            def forward(*, fill_tokens, pass_index):
                del pass_index
                if fill_tokens is None:
                    block_ids = first_block
                else:
                    if tuple(fill_tokens.shape) != (batch_size, depth):
                        raise ValueError("later passes must feed all 15 variable slots")
                    block_ids = torch.cat(
                        [
                            anchor_ids,
                            fill_tokens.to(
                                device=self.device, dtype=context_ids.dtype
                            ),
                        ],
                        dim=1,
                    )
                if bool((block_ids == int(self.head.mask_token_id)).any()):
                    raise ValueError("Jacobi input must be mask-free")
                noise = self.embed_tokens(block_ids)
                hidden = self.head(
                    target_hidden=contexts,
                    noise_embedding=noise,
                    position_ids=position_ids,
                    attention_mask=attention_mask,
                    past_key_values=None,
                    use_cache=False,
                    is_causal=True,
                    attention_backend="sdpa",
                )
                logits = self.lm_head(hidden[:, 1:, :])[:, :depth, :]
                return DraftForwardPass(
                    logits=logits,
                    input_tokens=block_ids[:, 1 : 1 + depth],
                )

            results = run_refinement_protocol(
                forward,
                batch_size=batch_size,
                depth=depth,
                max_passes=passes,
                convergence_limit=self.convergence_limit,
                abort_if=abort_if,
            )
            blocks = []
            for hypothesis, anchor_id, result in zip(
                hypotheses, anchor_ids[:, 0], results
            ):
                blocks.append(
                    Block(
                        hypothesis=hypothesis,
                        mode=mode,
                        draft_ids=result.final_tokens[1:].detach(),
                        logits=result.logits[1:].detach(),
                        passes_run=result.passes_run,
                        converged=result.converged,
                        anchor=int(anchor_id.item()),
                        correction_guess=int(result.final_tokens[0].item()),
                    )
                )
            return blocks
        finally:
            seq.staged.predicted_hidden = predicted


    @torch.inference_mode()
    def stage(self, seq: JetSpecSequence, block: Block) -> None:
        block_tokens = torch.cat(
            [
                seq.committed_ids[:, -1:].to(self.device),
                block.draft_ids.reshape(1, -1).to(self.device),
            ],
            dim=1,
        )
        prefix = seq.exact_hidden(device=self.device, dtype=self.dtype)
        query_count = int(block_tokens.shape[1])
        token_embeddings = self.embed_tokens(block_tokens)
        previous = prefix[:, -1:, :]
        projection_kwargs = {}
        if self.shared_prefix_projection:
            projection_kwargs["shared_prefix_projection"] = True
        else:
            previous = previous.expand(-1, query_count, -1)
        prefix_count = int(prefix.shape[1])
        prefix_positions = torch.arange(
            prefix_count, dtype=torch.long, device=self.device
        ).unsqueeze(0)
        query_positions = torch.arange(
            prefix_count,
            prefix_count + query_count,
            dtype=torch.long,
            device=self.device,
        ).unsqueeze(0)
        predicted = self.adapter.head(
            **projection_kwargs,
            token_embeddings=token_embeddings,
            h_prev=previous,
            prefix_h=prefix,
            prefix_position_ids=prefix_positions,
            query_position_ids=query_positions,
            attention_mask=build_adapter_query_mask(
                prefix_count=prefix_count,
                query_count=query_count,
                dtype=prefix.dtype,
                device=self.device,
            ),
        ).detach()
        seq.staged.stage(
            start_position=prefix_count,
            token_ids=block_tokens,
            predicted_hidden=predicted,
        )
        seq.pending_block = block



def commit_sequences_batch(sequences, accept_lengths, corrections):
    """Commit independent C>1 token prefixes with one concatenation."""
    if not sequences:
        return
    if len(sequences) != len(accept_lengths) or len(sequences) != len(corrections):
        raise ValueError("batched commit inputs must align")
    first = sequences[0].committed_ids
    pins = torch.tensor(corrections, dtype=first.dtype, device=first.device)
    parts, lengths = [], []
    for row, (seq, accept_len) in enumerate(zip(sequences, accept_lengths)):
        if seq.pending_block is None:
            raise RuntimeError("cannot commit without a staged block")
        accepted = int(accept_len)
        if accepted < 0 or accepted > int(seq.pending_block.draft_ids.numel()):
            raise ValueError("accept_len is outside the staged draft block")
        if seq.committed_ids.device != first.device or seq.committed_ids.dtype != first.dtype:
            raise ValueError("batched commit requires one token device and dtype")
        parts.extend((seq.committed_ids.reshape(-1),
                      seq.pending_block.draft_ids[:accepted].to(device=first.device, dtype=first.dtype),
                      pins[row:row + 1]))
        lengths.append(seq.committed_ids.numel() + accepted + 1)
    joined = torch.cat(parts)
    offset = 0
    for seq, length in zip(sequences, lengths):
        seq.committed_ids = joined[offset:offset + length].reshape(1, -1)
        seq.prepared_context = None
        offset += length


def bind_front_inputs(drafter, decoder, capacity):
    """Bind one key's U1 inputs to the existing C1 and decoder graph buffers.

    ``capacity`` is the maximum live prefix length in this chain key. The
    decoder window graph and rotary tables are prepared here. The caller fills
    these tensors, runs one eager body to warm lazy C1 launchers, then captures.
    """
    from jetspec.engines.jetspec_plus_plus.wire.protocol import draft_width

    head = drafter.adapter.head
    if not isinstance(head, ServingCtxKVC1Head):
        raise ValueError("front chain requires the served C1 cache head")
    head._front_embed_tokens = drafter.embed_tokens
    width = draft_width()
    if width not in (15, 31):
        raise ValueError("front chain requires 15 or 31 pending drafts")
    lanes = width + 1
    device, dtype = drafter.device, drafter.dtype
    hidden = decoder.hidden_size
    if head.hspace_dim != 5 * hidden:
        raise ValueError("front taps must contain five target-width rows")
    if not hasattr(head, "_c1_kv"):
        head._cache_init(torch.empty(1, 1, head.hspace_dim, device=device, dtype=dtype))
    if capacity > head._c1_capacity:
        raise ValueError("front chain prefix exceeds the C1 cache capacity")
    if not hasattr(head, "_c1_graphs"):
        embedding_dim = drafter.embed_tokens.weight.shape[1]
        head._graph_init(torch.empty(1, 15, embedding_dim, device=device, dtype=dtype))
    window, _ = head._c1_window_set(lanes)
    c1_embeddings = window["_c1_embeddings"]
    staged_hidden = window["_c1_output"]
    taps = head._c1_dirty[0].view(lanes, 5, hidden)
    positions = head._c1_dirty_positions
    c1_top = head._c1_capacity + lanes
    key_bucket = os.environ.get("JETSPEC_CHAIN_C1_KEY_BUCKET", "0") == "1"
    c1_bucket = (c1_top if head._c1_capacity > 768 and not key_bucket else
                 min(c1_top, ((capacity + lanes + 127) // 128) * 128))
    if c1_bucket < capacity + lanes:
        raise ValueError("front chain query exceeds the C1 key bucket")
    head._c1_rope(c1_embeddings, torch.zeros(1, lanes, device=device, dtype=torch.long))

    if os.environ.get("JETSPEC_HYBRID_PAIR", "0") == "1":
        state = decoder.bind_front_state(device=device, dtype=dtype,
                                         capacity=capacity + lanes)
        return FrontInputs(
            c1_query_ids=torch.empty(1, lanes, device=device, dtype=torch.long),
            c1_query_embeddings=c1_embeddings,
            staged_hidden=staged_hidden, prefix_positions=positions,
            taps=taps,
            decoder_prefix_taps=torch.empty(lanes, hidden, device=device, dtype=dtype),
            decoder_stale=torch.empty(4, hidden, device=device, dtype=dtype),
            rejected_ids=torch.empty(lanes, device=device, dtype=torch.long),
            selection_thresholds=torch.empty(2, device=device, dtype=torch.float64),
            decoder_prefix_kv={}, c1_bucket=c1_bucket,
            decoder_bucket=decoder._window_buckets()[0], lanes=lanes,
            select_cap=int(os.environ.get("JETSPEC_ANCHOR_CAP", "16")),
            hybrid_recurrent=state["recurrent"], hybrid_conv=state["conv"],
            hybrid_prefix_key=state["key"], hybrid_prefix_value=state["value"])

    rows = 4 + lanes
    decoder_bucket = next((b for b in decoder._window_buckets() if b >= capacity + rows), None)
    if decoder_bucket is None:
        raise ValueError("front chain window exceeds the decoder cache capacity")
    like = torch.empty(1, rows, hidden, device=device, dtype=dtype)
    state = decoder._window_graph(like, decoder_bucket)
    decoder._rope(like, torch.zeros(1, rows, device=device, dtype=torch.long))
    return FrontInputs(
        c1_query_ids=torch.empty(1, lanes, device=device, dtype=torch.long),
        c1_query_embeddings=c1_embeddings,
        staged_hidden=staged_hidden,
        prefix_positions=positions,
        taps=taps,
        decoder_prefix_taps=torch.empty(lanes, hidden, device=device, dtype=dtype),
        decoder_stale=torch.empty(4, hidden, device=device, dtype=dtype),
        rejected_ids=torch.empty(lanes, device=device, dtype=torch.long),
        selection_thresholds=torch.empty(2, device=device, dtype=torch.float64),
        decoder_prefix_kv=state["kv"],
        c1_bucket=c1_bucket,
        decoder_bucket=decoder_bucket,
        lanes=lanes,
        select_cap=int(os.environ.get("JETSPEC_ANCHOR_CAP", "16")),
    )
