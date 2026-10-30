"""DSpark draft head for JetSpec.

DSpark = the DFlash backbone (identical to `jetspec.models.draft_head.DFlashDraftModel`)
plus an order-1 Markov head (and an optional confidence head). `DSparkDraftModel`
therefore SUBCLASSES `DFlashDraftModel` and only adds those two modules + the
sampling/confidence helpers lifted from DeepSpec's `Qwen3DSparkModel`
(`deepspec/modeling/dspark/qwen3/modeling.py`).

Like the DFlash head, DSpark does NOT own `embed_tokens` / `lm_head` — it shares
the target's (the JetSpec convention). The released DeepSpec checkpoint *does*
carry frozen copies of `embed_tokens.weight` / `lm_head.weight` (initialized from
the target and frozen), so `load_dspark_head` intentionally drops them and uses
the target's at inference time.

Convention (block7): noise input is `[anchor, mask*(block_size-1)]` of length
`block_size` (the anchor is reused as the first prediction slot, per DSpark
paper sec 3.1), and ALL `block_size` output positions are consumed -> `block_size`
draft tokens. The caller applies the target `lm_head` to get `base_logits`
(shape `(1, block_size, V)`); `sample_draft_tokens` then runs the AR Markov
sampling over those positions with `first_prev = anchor`.
"""
from typing import Optional

import torch
from torch import nn
from jetspec.core.sampler import sample as sample_tokens
from jetspec.models.draft_head import DFlashDraftModel, load_qwen3_config_with_rope_compat
from jetspec.models.markov_head import build_markov_head


class AcceptRatePredictor(nn.Module):
    """Confidence head — copied from deepspec/modeling/dspark/common.py.

    Predicts a per-position acceptance logit. Loaded for weight completeness; in
    Phase 1 it is unused (we run with confidence_threshold=0 -> full block).
    """

    def __init__(self, input_dim: int):
        super().__init__()
        self.proj = nn.Linear(int(input_dim), 1)

    def forward(self, features):
        return self.proj(features).squeeze(-1)


class DSparkDraftModel(DFlashDraftModel):
    """DFlash backbone + Markov head (+ optional confidence head).

    Inherits `forward` (backbone, non-causal within block) unchanged from
    `DFlashDraftModel`. Adds the DSpark proposal-time helpers.
    """

    def __init__(self, config) -> None:
        super().__init__(config)
        # Markov head (None when markov_rank == 0 -> pure DFlash behavior).
        self.markov_rank = int(getattr(config, "markov_rank", 0))
        self.markov_head = build_markov_head(config) if self.markov_rank > 0 else None

        # Confidence head (optional; loaded but unused in Phase 1).
        self.enable_confidence_head = bool(getattr(config, "enable_confidence_head", False))
        self.confidence_head_with_markov = False
        if self.enable_confidence_head:
            self.confidence_head_with_markov = bool(
                getattr(config, "confidence_head_with_markov", False)
            )
        if self.enable_confidence_head and self.confidence_head_with_markov:
            assert self.markov_head is not None, (
                "confidence_head_with_markov requires a Markov head (markov_rank > 0)."
            )
        self.confidence_head = None
        if self.enable_confidence_head:
            input_dim = int(config.hidden_size)
            if self.confidence_head_with_markov:
                input_dim += int(self.markov_rank)
            self.confidence_head = AcceptRatePredictor(input_dim=input_dim)

    # --- lifted from Qwen3DSparkModel (operate on externally-computed base_logits) ---

    def predict_confidence_step(
        self,
        hidden_states: torch.Tensor,
        prev_token_ids: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        if self.confidence_head is None:
            return None
        if self.confidence_head_with_markov:
            assert self.markov_head is not None
            assert prev_token_ids is not None
            prev_embeddings = self.markov_head.get_prev_embeddings(prev_token_ids).to(
                dtype=hidden_states.dtype
            )
            features = torch.cat([hidden_states, prev_embeddings], dim=-1)
            return self.confidence_head(features).float()
        return self.confidence_head(hidden_states).float()

    def sample_draft_tokens(
        self,
        base_logits: torch.Tensor,
        *,
        first_prev_token_ids: torch.Tensor,
        temperature: float = 0.0,
        hidden_states: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, proposal_len = base_logits.shape[:2]
        if proposal_len == 0:
            empty_tokens = torch.empty(
                batch_size,
                0,
                dtype=torch.long,
                device=base_logits.device,
            )
            return empty_tokens, base_logits
        if self.markov_head is None:
            return sample_tokens(base_logits, temperature), base_logits
        return self.markov_head.sample_block_tokens(
            base_logits,
            first_prev_token_ids=first_prev_token_ids,
            hidden_states=hidden_states,
            temperature=temperature,
        )

    def sample_draft_token_step(
        self,
        base_logits: torch.Tensor,
        *,
        prev_token_ids: torch.Tensor,
        temperature: float = 0.0,
        hidden_states: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert base_logits.ndim == 2, (
            "sample_draft_token_step expects base_logits shaped [batch, vocab], "
            f"got {tuple(base_logits.shape)}."
        )
        if self.markov_head is None:
            step_logits = base_logits
        else:
            step_logits = self.markov_head.apply_step_logits(
                base_logits,
                token_ids=prev_token_ids,
                hidden_states=hidden_states,
            )
        sampled_token_ids = sample_tokens(
            step_logits.unsqueeze(1),
            temperature=temperature,
        ).squeeze(1)
        return sampled_token_ids, step_logits


def _is_embed_or_lm_head_key(key: str) -> bool:
    """Keys intentionally NOT owned by the head (shared from the target)."""
    return "embed_tokens" in key or key.startswith("lm_head") or ".lm_head" in key


def _is_ignorable_missing(key: str) -> bool:
    """Buffers that legitimately may be absent from the checkpoint."""
    return "rotary_emb" in key or "inv_freq" in key


def load_dspark_head(
    repo_or_path: str,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    attn_implementation: str = "sdpa",
) -> DSparkDraftModel:
    """Load a DeepSpec DSpark draft head into JetSpec's `DSparkDraftModel`.

    The DeepSpec config.json sets DSpark fields at the TOP level (see
    `deepspec/modeling/dspark/qwen3/config.py::build_draft_config`):
    `block_size`, `num_target_layers`, `target_layer_ids`, `mask_token_id`,
    `num_anchors`, `enable_confidence_head`, `confidence_head_with_markov`,
    `markov_rank`, `markov_head_type`. We translate those into the
    `dflash_config` dict that `DFlashDraftModel` expects, then strict-load the
    weights (dropping the target-shared `embed_tokens`/`lm_head`).
    """
    repo, _, subfolder = repo_or_path.partition("::")  # "repo::subfolder" optional form
    sub_kwargs = {"subfolder": subfolder} if subfolder else {}

    config = load_qwen3_config_with_rope_compat(repo, **sub_kwargs)

    # DSpark top-level fields -> the dflash_config dict DFlashDraftModel reads.
    if not hasattr(config, "target_layer_ids"):
        dflash = getattr(config, "dflash_config", {}) or {}
        if config.block_size != 16 or dflash.get("causal_head") is not False:
            raise ValueError("plain DFlash requires block_size=16 and causal_head=false")
        config.target_layer_ids = dflash["target_layer_ids"]
        config.mask_token_id = dflash["mask_token_id"]
    target_layer_ids = [int(x) for x in config.target_layer_ids]
    config.dflash_config = {
        "target_layer_ids": target_layer_ids,
        "mask_token_id": int(config.mask_token_id),
        "causal_head": False,  # DSpark backbone is non-causal within block
        "target_hidden_size": int(config.hidden_size),
    }
    config._attn_implementation = attn_implementation

    # Normalize markov / confidence fields (defaults keep markov_rank=0 -> DFlash).
    config.markov_rank = int(getattr(config, "markov_rank", 0))
    if config.markov_rank > 0 and not hasattr(config, "markov_head_type"):
        config.markov_head_type = "vanilla"
    config.enable_confidence_head = bool(getattr(config, "enable_confidence_head", False))
    config.confidence_head_with_markov = bool(
        getattr(config, "confidence_head_with_markov", False)
    )
    config.num_anchors = int(getattr(config, "num_anchors", 1))

    model, loading_info = DSparkDraftModel.from_pretrained(
        repo,
        config=config,
        dtype=dtype,
        attn_implementation=attn_implementation,
        output_loading_info=True,
        **sub_kwargs,
    )

    # --- strict load asserts: only embed_tokens/lm_head may be unexpected ---
    unexpected = list(loading_info.get("unexpected_keys", []))
    missing = list(loading_info.get("missing_keys", []))
    bad_unexpected = [k for k in unexpected if not _is_embed_or_lm_head_key(k)]
    bad_missing = [k for k in missing if not _is_ignorable_missing(k)]
    if bad_unexpected:
        raise RuntimeError(
            f"load_dspark_head: unexpected checkpoint keys (not embed/lm_head): {bad_unexpected}"
        )
    if bad_missing:
        raise RuntimeError(
            f"load_dspark_head: missing model keys not loaded from checkpoint: {bad_missing}"
        )

    # Markov / confidence heads must actually have been built + loaded.
    if config.markov_rank > 0:
        assert model.markov_head is not None, "markov_rank>0 but markov_head is None"
        markov_keys = [
            k for k in model.state_dict() if k.startswith("markov_head.")
        ]
        loaded_markov = [k for k in markov_keys if k not in missing]
        assert loaded_markov, "markov_head present but no markov_head.* weights loaded"
    if config.enable_confidence_head:
        assert model.confidence_head is not None, (
            "enable_confidence_head but confidence_head is None"
        )

    return model.to(device).eval()


__all__ = [
    "AcceptRatePredictor",
    "DSparkDraftModel",
    "load_dspark_head",
]
