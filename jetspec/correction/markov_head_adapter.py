"""Checkpoint adapter for the Markov-head correction layer.

The adapter validates the fail-closed draft-head contract, restores the Markov
tables without transposition, and exposes both chain and tree drafters. Public
names use the ratified Markov-head vocabulary.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open

from jetspec.correction import cost_counters
from jetspec.draft_head_adapter import _DraftHeadForward
from jetspec.models.draft_head import (
    load_draft_head,
    load_qwen3_config_with_rope_compat,
)
from jetspec.correction.markov_head import VanillaMarkov


_MODEL_WEIGHTS = "model.safetensors"
_TARGET_WEIGHTS = "target_release_weights.safetensors"
_W1_KEY = "markov_head.markov_w1.weight"
_W2_KEY = "markov_head.markov_w2.weight"
_CORRECTION_KEY = "correction_layer"
_LEGACY_CORRECTION_KEY = "token_top_layer"
_CORRECTION_PREFIXES = (
    f"{_CORRECTION_KEY}.",
    f"{_LEGACY_CORRECTION_KEY}.",
)


@dataclass(frozen=True)
class MarkovHeadLayout:
    markov_rank: int
    native_block: int
    proposal_len: int
    engine_block: int
    target_layer_ids: list[int]


class MarkovHeadLinearDrafter:
    """The causal DFlash slots plus an autoregressive vanilla-bias walk."""

    def __init__(
        self,
        head,
        target,
        layout: MarkovHeadLayout,
        temperature: float = 0.0,
    ) -> None:
        self._fwd = _DraftHeadForward(
            head,
            target,
            layout.native_block,
            layout.target_layer_ids,
            draft_shift=False,
        )
        self.head = head
        self.target = target
        self.layout = layout
        self.block_size = layout.native_block
        self.target_layer_ids = layout.target_layer_ids
        self.temperature = float(temperature)

    def reset_cache(self) -> None:
        self._fwd.reset_cache()

    @torch.inference_mode()
    def _propose(self, context_ids, k: int, target_hidden):
        depth = min(int(k), self.layout.proposal_len)
        base_logits = self._fwd._forward_head(context_ids, target_hidden, depth)
        return self.head.markov_head.sample_block_tokens(
            base_logits,
            first_prev_token_ids=context_ids[:, -1],
            hidden_states=None,
            temperature=self.temperature,
        )

    @torch.inference_mode()
    def propose(self, context_ids, k: int, target_hidden=None, **kwargs) -> torch.Tensor:
        del kwargs
        sampled, _ = self._propose(context_ids, k, target_hidden)
        return sampled.reshape(-1)[:k]

    @torch.inference_mode()
    def propose_logits(
        self,
        context_ids,
        k: int,
        target_hidden=None,
        **kwargs,
    ) -> torch.Tensor:
        del kwargs
        _, corrected_logits = self._propose(context_ids, k, target_hidden)
        return corrected_logits


class MarkovHeadDrafter:
    """The raw causal DFlash logits for bias-OFF or per-node bias-ON trees."""

    def __init__(
        self,
        head,
        target,
        layout: MarkovHeadLayout,
        temperature: float = 0.0,
    ) -> None:
        self._fwd = _DraftHeadForward(
            head,
            target,
            layout.native_block,
            layout.target_layer_ids,
            draft_shift=False,
        )
        self.head = head
        self.target = target
        self.layout = layout
        self.block_size = layout.native_block
        self.target_layer_ids = layout.target_layer_ids
        self.temperature = float(temperature)
        self.markov_head = head.markov_head

    def reset_cache(self) -> None:
        self._fwd.reset_cache()

    @torch.inference_mode()
    def propose_logits(
        self,
        context_ids,
        depth: int,
        target_hidden=None,
        **kwargs,
    ) -> torch.Tensor:
        del kwargs
        if depth > self.layout.proposal_len:
            raise ValueError(
                f"Markov-head tree depth={depth} exceeds proposal_len={self.layout.proposal_len}"
            )
        if cost_counters.active("markov_head"):
            with cost_counters.base_forward("markov_head", context_ids.device):
                return self._fwd._forward_head(context_ids, target_hidden, depth)
        return self._fwd._forward_head(context_ids, target_hidden, depth)


def resolve_markov_head_layout(config) -> MarkovHeadLayout:
    """Validate the checkpoint features that determine Markov-head semantics.

    The public correction_layer key is checked first. The pre-ratification key
    is checked only when the public key is absent.
    """
    architectures = list(getattr(config, "architectures", None) or [])
    if architectures != ["DFlashDraftModel"]:
        raise ValueError(
            "Markov-head adapter requires architectures=['DFlashDraftModel']; "
            f"got {architectures}"
        )

    dflash_config = dict(getattr(config, "dflash_config", None) or {})
    # Both causal and bidirectional base attention are supported.
    # DFlashDraftModel reads the same flag and builds/skips the block-causal mask.
    # Fail-closed: the key must be EXPLICIT (missing would silently run bidirectional).
    if "causal_head" not in dflash_config:
        raise ValueError(
            "Markov-head adapter requires dflash_config.causal_head to be set "
            "explicitly (true=causal, false=bidirectional)"
        )

    correction = getattr(config, _CORRECTION_KEY, None)
    if correction is None:
        correction = dflash_config.get(_CORRECTION_KEY)
    if correction is None:
        correction = getattr(config, _LEGACY_CORRECTION_KEY, None)
    if correction is None:
        correction = dflash_config.get(_LEGACY_CORRECTION_KEY)
    if correction and bool(dict(correction).get("enabled", True)):
        raise ValueError(
            "Markov-head adapter does not accept checkpoints with an enabled "
            "attention correction layer"
        )

    markov_config = getattr(config, "markov_head", None)
    if markov_config is None:
        markov_config = dflash_config.get("markov_head")
    if markov_config is None:
        raise ValueError("Markov-head adapter requires config.markov_head")
    markov_rank = int(dict(markov_config).get("rank", 0))
    if markov_rank != 256:
        raise ValueError(f"Markov-head checkpoint must use markov rank=256, got {markov_rank}")

    native_block = int(getattr(config, "block_size", 0))
    if native_block not in (8, 16):
        raise ValueError(
            "Markov-head checkpoint must use native block_size in (8, 16), "
            f"got {native_block}"
        )
    mask_token_id = int(dflash_config.get("mask_token_id", -1))
    if mask_token_id != 151669:
        raise ValueError(
            f"Markov-head checkpoint must use mask_token_id=151669, got {mask_token_id}"
        )
    target_layer_ids = [int(x) for x in dflash_config.get("target_layer_ids", [])]
    expected_target_layers = [1, 9, 17, 25, 33]
    if target_layer_ids != expected_target_layers:
        raise ValueError(
            f"Markov-head checkpoint must use target_layer_ids={expected_target_layers}, "
            f"got {target_layer_ids}"
        )
    num_hidden_layers = int(getattr(config, "num_hidden_layers", 0))
    if num_hidden_layers != 5:
        raise ValueError(f"Markov-head checkpoint must use 5 draft layers, got {num_hidden_layers}")

    proposal_len = native_block - 1
    return MarkovHeadLayout(
        markov_rank=markov_rank,
        native_block=native_block,
        proposal_len=proposal_len,
        engine_block=proposal_len + 1,
        target_layer_ids=target_layer_ids,
    )


def _checkpoint_file(repo_or_path: str, filename: str) -> str:
    repo, _, subfolder = repo_or_path.partition("::")
    local_root = Path(repo)
    if local_root.is_dir():
        path = local_root / subfolder / filename if subfolder else local_root / filename
        if not path.is_file():
            raise FileNotFoundError(f"missing Markov-head checkpoint file: {path}")
        return str(path)
    return hf_hub_download(
        repo_id=repo,
        filename=filename,
        subfolder=subfolder or None,
    )


def _load_bias_tables(
    repo_or_path: str,
) -> tuple[torch.Tensor, torch.Tensor, set[str]]:
    weights_path = _checkpoint_file(repo_or_path, _MODEL_WEIGHTS)
    with safe_open(weights_path, framework="pt", device="cpu") as state:
        checkpoint_keys = set(state.keys())
        missing = [key for key in (_W1_KEY, _W2_KEY) if key not in checkpoint_keys]
        if missing:
            raise KeyError(f"{weights_path} is missing Markov-head bias tensors: {missing}")
        return (
            state.get_tensor(_W1_KEY),
            state.get_tensor(_W2_KEY),
            checkpoint_keys,
        )


def load_markov_head(
    repo_or_path: str,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    attn_implementation: str = "sdpa",
    allow_attention_layer: bool = False,
):
    """Load a draft head and attach its checkpoint-native Markov tables."""
    repo, _, subfolder = repo_or_path.partition("::")
    config_kwargs = {"subfolder": subfolder} if subfolder else {}
    config = load_qwen3_config_with_rope_compat(repo, **config_kwargs)
    dflash_config = dict(getattr(config, "dflash_config", None) or {})
    public_correction = getattr(config, _CORRECTION_KEY, None)
    legacy_correction = getattr(config, _LEGACY_CORRECTION_KEY, None)
    nested_public = dflash_config.pop(_CORRECTION_KEY, None)
    nested_legacy = dflash_config.pop(_LEGACY_CORRECTION_KEY, None)
    if allow_attention_layer:
        setattr(config, _CORRECTION_KEY, None)
        setattr(config, _LEGACY_CORRECTION_KEY, None)
        config.dflash_config = dflash_config
    layout = resolve_markov_head_layout(config)
    if allow_attention_layer:
        setattr(config, _CORRECTION_KEY, public_correction)
        setattr(config, _LEGACY_CORRECTION_KEY, legacy_correction)
        if nested_public is not None:
            dflash_config[_CORRECTION_KEY] = nested_public
        if nested_legacy is not None:
            dflash_config[_LEGACY_CORRECTION_KEY] = nested_legacy
        config.dflash_config = dflash_config

    head = load_draft_head(
        repo_or_path,
        device=device,
        dtype=dtype,
        attn_implementation=attn_implementation,
    )
    w1, w2, checkpoint_keys = _load_bias_tables(repo_or_path)
    base_keys = set(head.state_dict())
    allowed_extra = {_W1_KEY, _W2_KEY}
    if allow_attention_layer:
        allowed_extra.update(
            key
            for key in checkpoint_keys
            if key.startswith(_CORRECTION_PREFIXES)
        )
    unexpected = sorted(checkpoint_keys - base_keys - allowed_extra)
    missing = sorted(base_keys - checkpoint_keys)
    if unexpected:
        raise RuntimeError(f"Markov-head adapter found unexpected checkpoint tensors: {unexpected}")
    if missing:
        raise RuntimeError(f"Markov-head adapter found missing DFlash tensors: {missing}")
    expected_shape = (int(config.vocab_size), layout.markov_rank)
    if tuple(w1.shape) != expected_shape or tuple(w2.shape) != expected_shape:
        raise ValueError(
            f"Markov-head config expects bias tables {expected_shape}; "
            f"got {tuple(w1.shape)} and {tuple(w2.shape)}"
        )
    markov_head = build_markov_head_from_weights(w1, w2).to(device=device, dtype=dtype)
    head.markov_head = markov_head
    head.markov_rank = layout.markov_rank
    return head.eval(), layout


@torch.inference_mode()
def load_markov_head_target_weights(
    target,
    repo_or_path: str,
    *,
    arm: str = "markov_head",
) -> str:
    """Restore the released target embedding and lm_head into the target model."""
    weights_path = _checkpoint_file(repo_or_path, _TARGET_WEIGHTS)
    destinations = {
        "model.embed_tokens.weight": target.model.embed_tokens.weight,
        "lm_head.weight": target.lm_head.weight,
    }
    with safe_open(weights_path, framework="pt", device="cpu") as state:
        missing = [key for key in destinations if key not in state.keys()]
        if missing:
            raise KeyError(f"{weights_path} is missing Markov-head target tensors: {missing}")
        for key, destination in destinations.items():
            source = state.get_tensor(key)
            if source.shape != destination.shape:
                raise ValueError(
                    f"Markov-head target tensor {key} has shape {tuple(source.shape)}; "
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


def build_markov_head_from_weights(
    markov_w1: torch.Tensor,
    markov_w2: torch.Tensor,
) -> VanillaMarkov:
    """Build the tree-facing head without changing checkpoint tensor orientation.

    The training wrapper suppresses bias for mask placeholders, but both the
    autoregressive chain and the tree call this head only with real anchor/node
    tokens. On that live path its formula is exactly VanillaMarkov's w2(w1[token]).
    """
    if (
        markov_w1.ndim != 2
        or markov_w2.ndim != 2
        or markov_w1.shape != markov_w2.shape
    ):
        raise ValueError(
            "Markov tensors must have matching [vocab_size, rank] shapes; "
            f"got {tuple(markov_w1.shape)} and {tuple(markov_w2.shape)}"
        )
    if markov_w1.dtype != markov_w2.dtype:
        raise ValueError(
            "Markov tensors must have matching dtypes; "
            f"got {markov_w1.dtype} and {markov_w2.dtype}"
        )

    vocab_size, markov_rank = markov_w1.shape
    head = VanillaMarkov(vocab_size=vocab_size, markov_rank=markov_rank)
    head = head.to(device=markov_w1.device, dtype=markov_w1.dtype)
    with torch.no_grad():
        head.markov_w1.weight.copy_(markov_w1)
        # nn.Linear stores [out_features, in_features]. The checkpoint stores that
        # [vocab, rank] layout, so copying (not transposing) preserves
        # bias(prev) = w1[prev] @ w2.T.
        head.markov_w2.weight.copy_(markov_w2)
    return head.eval()


@torch.inference_mode()
def assert_live_markov_bias(
    markov_head: VanillaMarkov,
    probe_token_ids: torch.Tensor | None = None,
    *,
    arm: str = "markov_head",
) -> dict[str, int | float]:
    """Reject zero tables and a false-null applied bias before generation."""
    w1 = markov_head.markov_w1.weight
    w2 = markov_head.markov_w2.weight
    w1_nonzero = int(torch.count_nonzero(w1).item())
    w2_nonzero = int(torch.count_nonzero(w2).item())
    if w1_nonzero == 0:
        raise ValueError("Markov markov_w1 is identically zero")
    if w2_nonzero == 0:
        raise ValueError("Markov markov_w2 is identically zero")

    if probe_token_ids is None:
        last = markov_head.vocab_size - 1
        probe_token_ids = torch.tensor(
            [0, last // 3, (2 * last) // 3, last],
            dtype=torch.long,
            device=w1.device,
        )
    else:
        probe_token_ids = probe_token_ids.to(device=w1.device, dtype=torch.long)
    probe_bias = markov_head.compute_step_bias(probe_token_ids, None)
    probe_nonzero = int(torch.count_nonzero(probe_bias).item())
    if probe_nonzero == 0:
        raise ValueError("Markov bias is identically zero on the probe batch")

    stats: dict[str, int | float] = {
        "w1_nonzero": w1_nonzero,
        "w2_nonzero": w2_nonzero,
        "probe_nonzero": probe_nonzero,
        "probe_abs_max": float(probe_bias.float().abs().max().item()),
    }
    print(
        f"[gate G-BIAS-LIVE] arm={arm} w1_nonzero={w1_nonzero} "
        f"w2_nonzero={w2_nonzero} probe_nonzero={probe_nonzero} "
        f"probe_abs_max={stats['probe_abs_max']:.6g}"
    )
    return stats


__all__ = [
    "MarkovHeadLayout",
    "MarkovHeadLinearDrafter",
    "MarkovHeadDrafter",
    "assert_live_markov_bias",
    "build_markov_head_from_weights",
    "load_markov_head",
    "load_markov_head_target_weights",
    "resolve_markov_head_layout",
]
