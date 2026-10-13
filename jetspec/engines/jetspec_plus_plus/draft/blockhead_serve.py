"""Serve a linear block head from target hidden-state taps."""

from __future__ import annotations

import __future__
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from time import perf_counter
from types import ModuleType, SimpleNamespace
from typing import NamedTuple

import torch
from torch import nn
from torch.nn import functional as F
from safetensors import safe_open
from jetspec.engines.jetspec_plus_plus.draft.drafter import served_target_layer_ids


TARGET_LAYER_IDS = list(served_target_layer_ids())
KIT_DIR = Path(__file__).parent / "vendor/blockhead_kit"
_VENDOR_API = None
BLOCKHEAD_GRAPH_METRICS = (
    "head_graph",
    "captures",
    "query_replays",
    "extension_replays",
    "prefill_extensions",
)


CHECKPOINT_CODE_FILES = ("dspark.py", "dspark_heads.py")
# A Domino head (arXiv 2605.29707): the DFlash-shape bidirectional backbone, then a GRU prefix walk
# (prefix_gru, embed_proj) that corrects each slot's logits sequentially. The same declarations and
# tensors the offline Domino reader requires.
DOMINO_PROJECTOR = "domino"
DOMINO_HEAD_TENSORS = (
    "prefix_gru.weight_ih_l0",
    "prefix_gru.weight_hh_l0",
    "embed_proj.0.weight",
    "embed_proj.2.weight",
)






def record_blockhead_runtime(tf32_state: dict, *, graph: bool) -> None:
    """Record what the speculator resolved after the load: the TF32 pin and the regime."""
    run_manifest = _run_manifest()
    run_manifest.record_value("blockhead_tf32", dict(tf32_state))
    run_manifest.record_value("blockhead_graph", bool(graph))


def blockhead_batch_requested() -> bool:
    value = os.environ.get("JETSPEC_BLOCKHEAD_BATCH", "0")
    if value not in ("0", "1"):
        raise ValueError("JETSPEC_BLOCKHEAD_BATCH must be 0 or 1")
    return value == "1"


def blockhead_graph_requested() -> bool:
    value = os.environ.get("JETSPEC_BLOCKHEAD_GRAPH", "0")
    if value not in ("0", "1"):
        raise ValueError("JETSPEC_BLOCKHEAD_GRAPH must be 0 or 1")
    if value == "1" and os.environ.get("TORCHDYNAMO_DISABLE") == "1":
        raise ValueError("BLOCKHEAD GRAPH refuses E")
    return value == "1"


def configure_blockhead_tf32() -> dict[str, bool]:
    """Apply the explicit fp32 parity pin without changing the default path."""
    value = os.environ.get("JETSPEC_BLOCKHEAD_TF32")
    if value not in (None, "0"):
        raise ValueError("JETSPEC_BLOCKHEAD_TF32 must be 0 when set")
    if value == "0":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    return {
        "matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
    }


class _SharedEmbedding(nn.Module):
    def __init__(self, weight: nn.Parameter):
        super().__init__()
        self.weight = weight

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return F.embedding(token_ids, self.weight)


class _SharedLMHead(nn.Module):
    def __init__(self, weight: nn.Parameter):
        super().__init__()
        self.weight = weight

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return F.linear(hidden, self.weight)


class EngineTargetShim:
    """HF-shaped view over the engine target's existing weight tensors."""

    def __init__(self, target):
        # Under target TP the engine tensors hold one vocab shard; rank 0 then carries the
        # all-gathered full rows (ModelRunner.gather_full_vocab_weights).
        full = getattr(target, "jetspec_full_vocab", None)
        embed, head = full if full is not None else (
            target.model.embed_tokens.weight, target.lm_head.weight)
        self.model = SimpleNamespace(
            embed_tokens=_SharedEmbedding(embed)
        )
        self.lm_head = _SharedLMHead(head)


class BlockheadProposal(NamedTuple):
    tokens: torch.Tensor
    logits: torch.Tensor
    head_ms: float
    markov_ms: float
    # The Domino GRU walk's time; None on every other head (its records carry no domino_ms).
    domino_ms: float | None = None


class CheckpointContract(NamedTuple):
    lookahead: int
    target_layer_ids: list[int]
    target_weights: str
    query_size: int
    draft_offset: int
    trained_block_size: int
    # The Domino declarations (domino_options) when the checkpoint is a Domino head, else None.
    domino: dict | None = None


class LoadedBlockhead(NamedTuple):
    drafter: TimedLinearBlockhead
    lookahead: int
    target_weights: str
    target_weights_path: str | None












def blockhead_config_overrides(
    checkpoint_dir: str | None,
    config_kwargs: dict,
) -> dict:
    """Apply the default-off environment selection before ``Config`` is built."""
    checkpoint_dir = checkpoint_dir or os.environ.get("JETSPEC_BLOCKHEAD") or None
    if checkpoint_dir is None:
        return dict(config_kwargs)
    # Two GPUs without async = the target tensor-parallel (tp2); the head stays on rank 0.
    if int(config_kwargs.get("num_gpus", 1)) not in (1, 2):
        raise ValueError("block-head serving requires one GPU (or two for target TP)")
    if (int(config_kwargs.get("max_num_seqs", 1)) > 1
            and blockhead_graph_requested() and not blockhead_batch_requested()):
        raise ValueError("batched block-head graph requires JETSPEC_BLOCKHEAD_BATCH=1")
    if config_kwargs.get("draft_async", False):
        raise ValueError("block-head serving uses the synchronous serial slot")
    if config_kwargs.get("jetspec", False) or config_kwargs.get("use_eagle", False):
        raise ValueError("block-head serving cannot be combined with another draft mode")
    contract = read_checkpoint_contract(
        checkpoint_dir, block_size_read=block_size_read(
            batch=int(config_kwargs.get("max_num_seqs", 1))),
        verify_block=blockhead_verify_block())
    overrides = dict(config_kwargs)
    overrides.update(
        speculate=True,
        draft_async=False,
        jetspec_taps=True,
        speculate_k=contract.lookahead,
    )
    if contract.target_layer_ids != TARGET_LAYER_IDS:
        overrides["jetspec_target_layer_ids"] = tuple(contract.target_layer_ids)
    return overrides




def complete_metric_rounds(metrics: dict) -> None:
    """Join each head record to its exact outer engine and verifier timings."""
    completed = []
    self_check = None
    for partial in metrics["blockhead_round_records"]:
        partial = dict(partial)
        graph_state = partial.pop("blockhead_graph_state", None)
        self_check = partial.pop("blockhead_graph_self_check", self_check)
        if graph_state is not None:
            partial.update(
                {key: graph_state[key] for key in BLOCKHEAD_GRAPH_METRICS}
            )
        if "round_ms" in partial:
            completed.append(partial)
            continue
        step_index = int(partial["step_index"])
        verify_index = int(partial["verify_index"])
        joined = complete_round_anatomy(
            [partial],
            accepted=[metrics["accepted_suffix_lens_with_recovery"][
                int(partial.get("accepted_index", verify_index))]],
            verify_ms=metrics["target_verify_times"][verify_index] * 1000.0,
            round_ms=metrics["target_step_times"][step_index] * 1000.0,
        )[0]
        completed.append(joined)
    metrics["blockhead_round_records"] = completed
    if completed and all(key in completed[-1] for key in BLOCKHEAD_GRAPH_METRICS):
        final = completed[-1]
        for key in BLOCKHEAD_GRAPH_METRICS[1:]:
            metrics[f"blockhead_graph_{key}"] = final[key]
        metrics["blockhead_head_graph"] = final["head_graph"]
        metrics["blockhead_graph_self_check"] = self_check






def load_blockhead(
    checkpoint_dir: str | Path,
    *,
    target,
    device: str | torch.device,
    max_num_seqs: int = 1,
) -> LoadedBlockhead:
    """Load one pinned head beside the single-GPU verifier target."""
    checkpoint = Path(checkpoint_dir)
    contract = read_checkpoint_contract(
        checkpoint, block_size_read=block_size_read(batch=max_num_seqs),
        verify_block=blockhead_verify_block())
    target_layers = getattr(target.model, "layers", None)
    if target_layers is not None and any(layer >= len(target_layers) for layer in contract.target_layer_ids):
        raise ValueError(f"block-head target_layer_ids must be below target "
                         f"num_hidden_layers={len(target_layers)}")
    dtype = resolve_draft_dtype()
    torch.set_float32_matmul_precision("high")

    weights_path = None
    if contract.target_weights == "checkpoint_embedded":
        if getattr(target, "jetspec_full_vocab", None) is not None:
            raise ValueError("checkpoint-embedded target weights are not supported under target TP")
        weights_path = load_checkpoint_target_weights(checkpoint, target)

    api = load_vendor_api()
    domino_walk = None
    if contract.domino:
        if max_num_seqs != 1:
            raise ValueError("Domino serving runs batch 1 only")
        if blockhead_graph_requested():
            raise ValueError("Domino serving is eager only (JETSPEC_BLOCKHEAD_GRAPH=0)")
        head = load_domino_backbone(checkpoint, api, device=device, dtype=dtype)
        domino_walk = load_domino_walk(
            checkpoint, contract.domino, head.config, device=device, dtype=dtype)
    else:
        head = api.load_head(
            str(checkpoint),
            device=str(device),
            dtype=dtype,
            attn_implementation="sdpa",
        )
    loaded_layers = [int(value) for value in head.config.target_layer_ids]
    if loaded_layers != contract.target_layer_ids:
        raise ValueError(
            "loaded block-head target_layer_ids mismatch: "
            f"expected={contract.target_layer_ids} actual={loaded_layers}"
        )
    if contract.draft_offset and head.markov_head is not None:
        raise ValueError("plain DFlash serving requires no Markov head")
    if contract.query_size != head.block_size:
        head.config.block_size = contract.query_size
        head.block_size = contract.query_size
    record_blockhead_manifest(
        checkpoint,
        contract,
        head.config,
        dtype=dtype,
        target_weights_path=weights_path,
    )
    target_shim = EngineTargetShim(target)
    linear = api.linear_drafter(
        head,
        target_shim,
        block_size=contract.query_size,
        target_layer_ids=contract.target_layer_ids,
        temperature=0.0,
    )
    linear.lookahead = contract.lookahead
    linear.draft_offset = contract.draft_offset
    print(
        "W273_BLOCKHEAD_LOAD "
        f"arm={'domino' if contract.domino else 'blockhead'}-{contract.lookahead} "
        f"lookahead={contract.lookahead} "
        f"dtype={str(dtype).removeprefix('torch.')} "
        f"target_weights={contract.target_weights}",
        flush=True,
    )
    return LoadedBlockhead(
        drafter=TimedLinearBlockhead(linear, domino=domino_walk),
        lookahead=contract.lookahead,
        target_weights=contract.target_weights,
        target_weights_path=str(weights_path) if weights_path else None,
    )






class DominoWalk(nn.Module):
    """The Domino correction head: prefix_gru over the realized prefix, then per slot
    ``embed_proj([hidden_i; state])`` added to the slot's base logits, the GRU stepping on
    each corrected draft. Module shapes are the bundled dflash.py's."""

    def __init__(self, *, hidden_size: int, vocab_size: int, options: dict):
        super().__init__()
        self.prefix_len = int(options["pure_draft_prefix_len"])
        gru_hidden_dim = int(options["gru_hidden_dim"])
        emb_dim = int(options["emb_dim"])
        self.prefix_gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=gru_hidden_dim,
            num_layers=1,
            batch_first=True,
            bias=False,
        )
        self.embed_proj = nn.Sequential(
            nn.Linear(hidden_size + gru_hidden_dim, emb_dim, bias=False),
            nn.SiLU(),
            nn.Linear(emb_dim, vocab_size, bias=False),
        )





class TimedLinearBlockhead:
    """Time the eager block forward separately from its serial token sampler."""

    def __init__(self, drafter, domino: DominoWalk | None = None):
        self.drafter = drafter
        self.domino = domino
        self._parity_round = 0















def pack_target_taps(taps: torch.Tensor, *, ctx_len: int) -> torch.Tensor:
    """Convert engine taps ``[tokens, layers, hidden]`` to HF ``[1, T, L*H]``."""
    if taps.ndim != 3 or taps.shape[1] != len(TARGET_LAYER_IDS):
        raise ValueError(
            "target taps must have shape "
            f"[tokens, {len(TARGET_LAYER_IDS)}, hidden], got {tuple(taps.shape)}"
        )
    if not 0 <= ctx_len <= taps.shape[0]:
        raise ValueError(
            f"ctx_len={ctx_len} outside available target taps [0, {taps.shape[0]}]"
        )
    selected = taps[:ctx_len]
    return selected.reshape(1, ctx_len, -1)
