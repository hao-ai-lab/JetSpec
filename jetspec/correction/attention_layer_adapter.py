"""Checkpoint adapter for the attention correction layer.

The public draft model intentionally omits correction modules. This adapter
validates the checkpoint contract, restores the standalone attention layer from
the checkpoint state, and exposes a tree drafter that returns the full hidden
block required for path-conditioned correction.

The public config key and state prefix are correction_layer. Pre-ratification
checkpoints are accepted through an explicit legacy fallback only when the
public key or prefix is absent.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open

from jetspec.correction import cost_counters
from jetspec.draft import TreeDrafter
from jetspec.correction.hidden_forward import _HiddenForward
from jetspec.models.draft_head import (
    load_draft_head,
    load_qwen3_config_with_rope_compat,
)
from jetspec.correction.attention_layer import AttentionLayer


_MODEL_WEIGHTS = "model.safetensors"
_TARGET_WEIGHTS = "target_release_weights.safetensors"
_CORRECTION_KEY = "correction_layer"
_LEGACY_CORRECTION_KEY = "token_top_layer"
_CORRECTION_PREFIX = f"{_CORRECTION_KEY}."
_LEGACY_CORRECTION_PREFIX = f"{_LEGACY_CORRECTION_KEY}."
_FUSION_MODES = ("concat", "h_only", "e_only", "add")
_MLP_STATE_KEYS = frozenset(
    {
        "post_attention_layernorm.weight",
        "mlp.gate_proj.weight",
        "mlp.up_proj.weight",
        "mlp.down_proj.weight",
    }
)


@dataclass(frozen=True)
class AttentionLayerLayout:
    native_block: int
    proposal_len: int
    engine_block: int
    target_layer_ids: list[int]
    correction_config: dict
    causal_head: bool = True


def resolve_attention_layer_layout(config) -> AttentionLayerLayout:
    """Validate the attention-layer checkpoint contract.

    The public correction_layer key is primary. The pre-ratification key is
    read only when the public key is absent so existing checkpoints remain
    loadable without exposing the old vocabulary in public APIs.
    """
    architectures = list(getattr(config, "architectures", None) or [])
    if architectures != ["DFlashDraftModel"]:
        raise ValueError(
            "Attention-layer adapter requires architectures=['DFlashDraftModel']; "
            f"got {architectures}"
        )

    dflash_config = dict(getattr(config, "dflash_config", None) or {})
    # Both causal and bidirectional base attention are supported.
    # DFlashDraftModel reads the same flag and builds/skips the block-causal mask.
    # Fail-closed: the key must be EXPLICIT in the checkpoint config — a missing key
    # would otherwise silently run bidirectional attention on a legacy causal head.
    if "causal_head" not in dflash_config:
        raise ValueError(
            "Attention-layer adapter requires dflash_config.causal_head to be set "
            "explicitly (true=causal, false=bidirectional)"
        )
    causal_head = bool(dflash_config["causal_head"])

    correction = getattr(config, _CORRECTION_KEY, None)
    if correction is None:
        correction = dflash_config.get(_CORRECTION_KEY)
    if correction is None:
        correction = getattr(config, _LEGACY_CORRECTION_KEY, None)
    if correction is None:
        correction = dflash_config.get(_LEGACY_CORRECTION_KEY)
    if correction is None:
        raise ValueError("Attention-layer adapter requires a correction_layer config")
    correction = dict(correction)
    if not bool(correction.get("enabled", False)):
        raise ValueError(
            "Attention-layer adapter requires correction_layer.enabled=true"
        )
    heads = int(correction.get("heads", 0))
    if heads != 32:
        raise ValueError(f"correction_layer must use heads=32, got {heads}")
    width = int(correction.get("width", 0))
    if width != 4096:
        raise ValueError(f"correction_layer must use width=4096, got {width}")
    num_layers = correction.get("num_layers", 1)
    if num_layers != 1:
        raise ValueError(
            f"correction_layer must use num_layers=1, got {num_layers}"
        )
    init = str(correction.get("init", ""))
    if init != "zero_output":
        raise ValueError(
            f"correction_layer must use init=zero_output, got {init!r}"
        )
    fusion = str(correction.get("fusion", "concat")).lower()
    if fusion not in _FUSION_MODES:
        raise ValueError(
            f"correction_layer.fusion={fusion!r} is unsupported; "
            f"expected one of {_FUSION_MODES}"
        )
    use_mlp = bool(correction.get("mlp", False))

    native_block = int(getattr(config, "block_size", 0))
    if native_block not in (8, 16):
        raise ValueError(
            "Attention-layer checkpoint must use native block_size in (8, 16), "
            f"got {native_block}"
        )
    num_hidden_layers = int(getattr(config, "num_hidden_layers", 0))
    if num_hidden_layers != 5:
        raise ValueError(
            f"Attention-layer checkpoint must use 5 draft layers, got {num_hidden_layers}"
        )
    target_layer_ids = [int(x) for x in dflash_config.get("target_layer_ids", [])]
    expected_target_layers = [1, 9, 17, 25, 33]
    if target_layer_ids != expected_target_layers:
        raise ValueError(
            f"Attention-layer checkpoint must use target_layer_ids={expected_target_layers}, "
            f"got {target_layer_ids}"
        )
    mask_token_id = int(dflash_config.get("mask_token_id", -1))
    if mask_token_id != 151669:
        raise ValueError(
            f"Attention-layer checkpoint must use mask_token_id=151669, got {mask_token_id}"
        )

    proposal_len = native_block - 1
    return AttentionLayerLayout(
        native_block=native_block,
        proposal_len=proposal_len,
        engine_block=native_block,
        causal_head=causal_head,
        target_layer_ids=target_layer_ids,
        correction_config={
            "enabled": True,
            "heads": heads,
            "width": width,
            "num_layers": int(num_layers),
            "init": init,
            "fusion": fusion,
            "mlp": use_mlp,
        },
    )


def _checkpoint_file(repo_or_path: str, filename: str) -> str:
    repo, _, subfolder = repo_or_path.partition("::")
    local_root = Path(repo)
    if local_root.is_dir():
        path = local_root / subfolder / filename if subfolder else local_root / filename
        if not path.is_file():
            raise FileNotFoundError(f"missing attention-layer checkpoint file: {path}")
        return str(path)
    return hf_hub_download(
        repo_id=repo,
        filename=filename,
        subfolder=subfolder or None,
    )


def _load_attention_layer_state(repo_or_path: str) -> dict[str, torch.Tensor]:
    """Read correction-layer tensors, using the legacy prefix only as fallback."""
    weights_path = _checkpoint_file(repo_or_path, _MODEL_WEIGHTS)
    state: dict[str, torch.Tensor] = {}
    legacy_state: dict[str, torch.Tensor] = {}
    with safe_open(weights_path, framework="pt", device="cpu") as reader:
        for key in reader.keys():
            if key.startswith(_CORRECTION_PREFIX):
                state[key[len(_CORRECTION_PREFIX):]] = reader.get_tensor(key)
            elif key.startswith(_LEGACY_CORRECTION_PREFIX):
                legacy_state[key[len(_LEGACY_CORRECTION_PREFIX):]] = reader.get_tensor(
                    key
                )
    if state:
        return state
    if legacy_state:
        return legacy_state
    raise KeyError(
        f"{weights_path} carries no {_CORRECTION_PREFIX}* tensors and no "
        "compatible legacy correction tensors"
    )


def _describe_checkpoint_fusion(
    state: dict[str, torch.Tensor], hidden_size: int
) -> str:
    fusion_keys = {key for key in state if key.startswith("fusion.")}
    if not fusion_keys:
        fusion_description = "'h_only' or 'add'"
    else:
        weight = state.get("fusion.weight")
        if (
            weight is not None
            and tuple(weight.shape) == (hidden_size, 2 * hidden_size)
            and "fusion.bias" in fusion_keys
        ):
            fusion_description = "'concat'"
        elif (
            weight is not None
            and tuple(weight.shape) == (hidden_size, hidden_size)
            and fusion_keys == {"fusion.weight"}
        ):
            fusion_description = "'e_only'"
        else:
            fusion_description = f"unknown ({sorted(fusion_keys)})"

    mlp_keys = set(state) & _MLP_STATE_KEYS
    if mlp_keys == _MLP_STATE_KEYS:
        mlp_description = "True"
    elif not mlp_keys:
        mlp_description = "False"
    else:
        mlp_description = f"incomplete ({sorted(mlp_keys)})"
    return f"{fusion_description}, mlp={mlp_description}"


def _strict_load_attention_layer_state(
    attention_layer: AttentionLayer,
    state: dict[str, torch.Tensor],
    *,
    fusion_mode: str,
) -> None:
    """Strict-load with a mode-specific diagnostic for incompatible key maps."""
    state = dict(state)
    expected_state = attention_layer.state_dict()
    expected = set(expected_state)
    # rotary_emb.inv_freq is a derived (non-learned) buffer whose persistence flips
    # across transformers versions; drop any such stray key rather than fail on it.
    for key in [
        key
        for key in state
        if key not in expected and "rotary_emb" in key and "inv_freq" in key
    ]:
        state.pop(key)

    got = set(state)
    expected_fusion = {key for key in expected if key.startswith("fusion.")}
    got_fusion = {key for key in got if key.startswith("fusion.")}
    fusion_shape_mismatch = any(
        key in state
        and key in expected_state
        and state[key].shape != expected_state[key].shape
        for key in expected_fusion & got_fusion
    )
    if expected_fusion != got_fusion or fusion_shape_mismatch:
        checkpoint_mode = _describe_checkpoint_fusion(
            state, attention_layer.hidden_size
        )
        raise RuntimeError(
            "Attention-layer fusion mode mismatch: "
            f"configured fusion={fusion_mode!r} expects keys "
            f"{sorted(expected_fusion)}, but checkpoint looks like fusion="
            f"{checkpoint_mode} and carries keys {sorted(got_fusion)}"
        )

    expected_mlp = expected & _MLP_STATE_KEYS
    got_mlp = got & _MLP_STATE_KEYS
    if expected_mlp != got_mlp:
        checkpoint_mode = _describe_checkpoint_fusion(
            state, attention_layer.hidden_size
        )
        raise RuntimeError(
            "Attention-layer MLP mode mismatch: "
            f"configured mlp={attention_layer.use_mlp} expects keys "
            f"{sorted(expected_mlp)}, but checkpoint looks like "
            f"{checkpoint_mode} and carries keys {sorted(got_mlp)}"
        )

    missing = sorted(expected - got)
    unexpected = sorted(got - expected)
    if missing:
        raise RuntimeError(
            f"Attention-layer adapter is missing correction tensors: {missing}"
        )
    if unexpected:
        raise RuntimeError(
            f"Attention-layer adapter found unexpected correction tensors: {unexpected}"
        )
    attention_layer.load_state_dict(state, strict=True)


def load_attention_layer_head(
    repo_or_path: str,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    attn_implementation: str = "sdpa",
):
    """Load the backbone and attach its standalone attention layer."""
    repo, _, subfolder = repo_or_path.partition("::")
    config_kwargs = {"subfolder": subfolder} if subfolder else {}
    config = load_qwen3_config_with_rope_compat(repo, **config_kwargs)
    layout = resolve_attention_layer_layout(config)

    head = load_draft_head(
        repo_or_path,
        device=device,
        dtype=dtype,
        attn_implementation=attn_implementation,
    )

    attention_layer = AttentionLayer(config, layout.correction_config)
    state = _load_attention_layer_state(repo_or_path)
    _strict_load_attention_layer_state(
        attention_layer, state, fusion_mode=layout.correction_config["fusion"]
    )
    attention_layer = attention_layer.to(device=device, dtype=dtype).eval()
    head.attention_layer = attention_layer
    return head.eval(), layout


def load_residual_attention_layer(
    repo_or_path: str,
    base_config,
    *,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[AttentionLayer, AttentionLayerLayout]:
    """Load only the residual layer and validate it against an Markov-head substrate."""
    repo, _, subfolder = repo_or_path.partition("::")
    config_kwargs = {"subfolder": subfolder} if subfolder else {}
    layer_config = load_qwen3_config_with_rope_compat(repo, **config_kwargs)
    layout = resolve_attention_layer_layout(layer_config)

    base_dflash = dict(getattr(base_config, "dflash_config", None) or {})
    base_target_layers = [
        int(x) for x in base_dflash.get("target_layer_ids", [])
    ]
    mismatches = []
    if int(getattr(base_config, "block_size", 0)) != layout.native_block:
        mismatches.append(
            f"block_size={getattr(base_config, 'block_size', None)} "
            f"vs {layout.native_block}"
        )
    if int(getattr(base_config, "hidden_size", 0)) != int(
        getattr(layer_config, "hidden_size", 0)
    ):
        mismatches.append(
            f"hidden_size={getattr(base_config, 'hidden_size', None)} "
            f"vs {getattr(layer_config, 'hidden_size', None)}"
        )
    if base_target_layers != layout.target_layer_ids:
        mismatches.append(
            f"target_layer_ids={base_target_layers} vs {layout.target_layer_ids}"
        )
    if bool(base_dflash.get("causal_head", False)) != layout.causal_head:
        mismatches.append(
            f"causal_head={base_dflash.get('causal_head', None)} "
            f"vs {layout.causal_head}"
        )
    if mismatches:
        raise ValueError(
            "Residual attention-layer checkpoint is incompatible with the "
            "Markov-head substrate: "
            + "; ".join(mismatches)
        )

    attention_layer = AttentionLayer(base_config, layout.correction_config)
    state = _load_attention_layer_state(repo_or_path)
    _strict_load_attention_layer_state(
        attention_layer, state, fusion_mode=layout.correction_config["fusion"]
    )
    return attention_layer.to(device=device, dtype=dtype).eval(), layout


@torch.inference_mode()
def load_attention_layer_target_weights(
    target,
    repo_or_path: str,
    *,
    arm: str = "attention_layer",
) -> str:
    """Restore the released target embedding and lm-head into the target model."""
    weights_path = _checkpoint_file(repo_or_path, _TARGET_WEIGHTS)
    destinations = {
        "model.embed_tokens.weight": target.model.embed_tokens.weight,
        "lm_head.weight": target.lm_head.weight,
    }
    with safe_open(weights_path, framework="pt", device="cpu") as state:
        missing = [key for key in destinations if key not in state.keys()]
        if missing:
            raise KeyError(f"{weights_path} is missing attention-layer target tensors: {missing}")
        for key, destination in destinations.items():
            source = state.get_tensor(key)
            if source.shape != destination.shape:
                raise ValueError(
                    f"attention-layer target tensor {key} has shape {tuple(source.shape)}; "
                    f"target expects {tuple(destination.shape)}"
                )
            destination.copy_(source.to(device=destination.device, dtype=destination.dtype))
    print(
        f"[gate G-TARGET] arm={arm} "
        f"embed_shape={tuple(destinations['model.embed_tokens.weight'].shape)} "
        f"lm_head_shape={tuple(destinations['lm_head.weight'].shape)} "
        f"source={_TARGET_WEIGHTS}"
    )
    return weights_path


@torch.inference_mode()
def assert_live_attention_layer(
    attention_layer: AttentionLayer,
    base_hidden: torch.Tensor | None = None,
    prev_embedding: torch.Tensor | None = None,
    block_positions: torch.Tensor | None = None,
    *,
    arm: str = "attention_layer",
) -> dict[str, bool | int | float | str | None]:
    """Reject a zero-weight or false-null attention layer before generation.

    Fails loudly if the loaded o_proj is identically zero (untrained / accidentally
    zero-init) OR if the applied residual delta is zero on a probe block — either
    means the correction would silently behave as the uncorrected control.
    """
    o_proj = attention_layer.self_attn.o_proj.weight
    o_proj_nonzero = int(torch.count_nonzero(o_proj).item())
    fusion = getattr(attention_layer, "fusion", None)
    fusion_nonzero = (
        int(torch.count_nonzero(fusion.weight).item()) if fusion is not None else None
    )
    mlp_enabled = bool(attention_layer.use_mlp)
    mlp_down_proj_nonzero = (
        int(torch.count_nonzero(attention_layer.mlp.down_proj.weight).item())
        if mlp_enabled
        else None
    )
    if o_proj_nonzero == 0:
        raise ValueError("attention layer o_proj is identically zero (untrained/bypassed)")
    if fusion is not None and fusion_nonzero == 0:
        raise ValueError("attention layer fusion is identically zero")

    device = o_proj.device
    dtype = o_proj.dtype
    block_size = attention_layer.block_size
    hidden_size = attention_layer.hidden_size
    if base_hidden is None:
        base_hidden = torch.randn(1, block_size, hidden_size, device=device, dtype=dtype)
    if prev_embedding is None:
        prev_embedding = torch.randn(1, block_size, hidden_size, device=device, dtype=dtype)
    if block_positions is None:
        block_positions = torch.arange(block_size, device=device).unsqueeze(0)
    keep = torch.ones((1, 1), dtype=dtype, device=device)
    corrected = attention_layer(base_hidden, prev_embedding, block_positions, keep)
    delta = (corrected - base_hidden).float()
    delta_nonzero = int(torch.count_nonzero(delta).item())
    if delta_nonzero == 0:
        raise ValueError("attention layer applied delta is identically zero on the probe block")

    stats: dict[str, bool | int | float | str | None] = {
        "o_proj_nonzero": o_proj_nonzero,
        "fusion_nonzero": fusion_nonzero,
        "fusion_mode": attention_layer.fusion_mode,
        "mlp_enabled": mlp_enabled,
        "mlp_down_proj_nonzero": mlp_down_proj_nonzero,
        "delta_nonzero": delta_nonzero,
        "delta_abs_max": float(delta.abs().max().item()),
    }
    print(
        f"[gate G-ATTENTION-LIVE] arm={arm} fusion={attention_layer.fusion_mode} "
        f"mlp={mlp_enabled} mlp_down_proj_nonzero={mlp_down_proj_nonzero} "
        f"o_proj_nonzero={o_proj_nonzero} "
        f"fusion_nonzero={fusion_nonzero} delta_nonzero={delta_nonzero} "
        f"delta_abs_max={stats['delta_abs_max']:.6g}"
    )
    return stats


class AttentionLayerDrafter(TreeDrafter):
    """Return path-independent base hidden states for attention-layer trees.

    `propose_logits` returns the FULL post-`norm` backbone hidden
    `(1, block_size, H)` (NO lm_head, NO draft slice) — a `forward_hidden` variant
    like the full-block path — because the tree algorithm needs the raw per-slot hidden
    to feed the attention layer per node. It also caches the block RoPE positions
    used and exposes the layer, embedding, and projection modules for
    `attention_layer_accum_logp` via `algo_kwargs`.
    """

    def __init__(
        self,
        head,
        target,
        block_size: int,
        target_layer_ids,
        temperature: float = 0.0,
    ) -> None:
        self._fwd = _HiddenForward(
            head, target, block_size, target_layer_ids, draft_shift=False
        )
        self.head = head
        self.target = target
        self.block_size = int(block_size)
        self.target_layer_ids = target_layer_ids
        self.temperature = float(temperature)
        self.attention_layer = getattr(head, "attention_layer", None)
        self.embed_tokens = target.model.embed_tokens
        self.lm_head = target.lm_head
        self.mask_token_id = head.mask_token_id
        # The absolute block RoPE positions the most recent forward used
        # (position_ids[:, -block_size:]); refreshed every propose_logits.
        self.block_positions = None

    def reset_cache(self) -> None:
        self._fwd.reset_cache()

    @torch.inference_mode()
    def propose_logits(
        self,
        context_ids: torch.Tensor,
        depth: int,
        target_hidden: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:
        """Return `(1, block_size, H)` post-norm base hidden (path-independent)."""
        del depth  # the whole block is returned; the tree reads slots per node
        if cost_counters.active("attention_layer"):
            with cost_counters.base_forward("attention_layer", context_ids.device):
                hidden = self._fwd.forward_hidden(
                    context_ids, target_hidden
                )  # (1, B, H)
        else:
            hidden = self._fwd.forward_hidden(context_ids, target_hidden)  # (1, B, H)
        ctx_len = int(target_hidden.shape[1])
        self.block_positions = torch.arange(
            ctx_len, ctx_len + self.block_size, device=hidden.device
        ).unsqueeze(0)
        return hidden[:, : self.block_size, :]


__all__ = [
    "AttentionLayerLayout",
    "AttentionLayerDrafter",
    "assert_live_attention_layer",
    "load_attention_layer_head",
    "load_residual_attention_layer",
    "load_attention_layer_target_weights",
    "resolve_attention_layer_layout",
]
