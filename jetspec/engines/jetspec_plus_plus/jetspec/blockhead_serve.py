# Adaptation: package relocation and deferred optional dependencies.
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
from jetspec.engines.jetspec_plus_plus.jetspec.drafter import served_target_layer_ids
from jetspec.engines.jetspec_plus_plus.jetspec.config_values import config_values


TARGET_LAYER_IDS = list(served_target_layer_ids())
KIT_DIR = Path(__file__).parent / "../../third_party/blockhead_kit"
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
# tensors the offline reader requires.
DOMINO_PROJECTOR = "domino"
DOMINO_HEAD_TENSORS = (
    "prefix_gru.weight_ih_l0",
    "prefix_gru.weight_hh_l0",
    "embed_proj.0.weight",
    "embed_proj.2.weight",
)


def _run_manifest():
    """The run-manifest recorder; by package when this module has one, else by path
    (the test harness loads this file bare)."""
    if __package__:
        import jetspec.engines.jetspec_plus_plus.jetspec.run_manifest as run_manifest
        return run_manifest
    name = "jetspec_run_manifest"
    if name not in sys.modules:
        path = (Path(__file__).parent.parent / 'read/run_manifest.py')
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def record_blockhead_manifest(
    checkpoint: Path,
    contract: CheckpointContract,
    head_config,
    *,
    dtype: torch.dtype,
    target_weights_path: Path | None,
run_config=None) -> None:
    """Bind the run to the checkpoint bytes it loaded and to the values it resolved."""
    run_manifest = _run_manifest()
    weights = sorted(
        path.name
        for path in checkpoint.iterdir()
        if path.name.startswith("model") and (
            path.name.endswith(".safetensors")
            or path.name.endswith(".safetensors.index.json")
        )
    )
    for name in ("config.json", *weights):
        run_manifest.record_file(f"blockhead/{name}", checkpoint / name, run_config=run_config)
    if target_weights_path is not None and target_weights_path.name not in weights:
        run_manifest.record_file(
            f"blockhead/{target_weights_path.name}", target_weights_path
        , run_config=run_config)
    # The checkpoint's own modeling code identifies it, but nothing here imports it:
    # the pinned kit under vendor/blockhead_kit is what runs.
    code_files = (
        ("dflash.py",) if contract.draft_offset or contract.domino
        else ("dflash.py", "dspark.py") if contract.target_layer_ids != TARGET_LAYER_IDS
        else CHECKPOINT_CODE_FILES
    )
    unread = [name for name in code_files if (checkpoint / name).is_file()]
    for name in unread:
        run_manifest.record_file(f"blockhead/{name}", checkpoint / name, run_config=run_config)
    run_manifest.record_value("blockhead_files_recorded_not_read", unread, run_config=run_config)
    run_manifest.record_file("blockhead_kit/manifest.json", (KIT_DIR / 'manifest.json'), run_config=run_config)
    # A Domino head is served on the plain DFlash route (serve_read's method dflash, whose
    # method_mode it asserts); its declarations are recorded beside it.
    run_manifest.record_value(
        "method_mode", "dflash" if contract.draft_offset or contract.domino else "blockhead", run_config=run_config)
    if contract.domino:
        run_manifest.record_value("blockhead_domino", dict(contract.domino), run_config=run_config)
    run_manifest.record_value("blockhead_block_size", contract.trained_block_size, run_config=run_config)
    if contract.query_size in (4, 8, 32):
        run_manifest.record_value("block_size_read", contract.query_size, run_config=run_config)
    run_manifest.record_value("blockhead_draft_tokens", contract.lookahead, run_config=run_config)
    run_manifest.record_value("blockhead_verify_tokens", contract.lookahead + 1, run_config=run_config)
    run_manifest.record_value("blockhead_target_weights", contract.target_weights, run_config=run_config)
    run_manifest.record_value("blockhead_dtype", str(dtype).removeprefix("torch."), run_config=run_config)
    markov_rank = int(getattr(head_config, "markov_rank", 0))
    run_manifest.record_value("blockhead_markov_rank", markov_rank, run_config=run_config)
    run_manifest.record_value(
        "blockhead_markov_head_type",
        getattr(head_config, "markov_head_type", None) if markov_rank > 0 else None,
    run_config=run_config)
    # No threshold exists on this path: neither this module, blockhead_graph.py nor the
    # kit's adapter.py calls the confidence head, so every round drafts the full block.
    run_manifest.record_value(
        "blockhead_confidence_head",
        {
            "enabled_in_checkpoint": bool(
                getattr(head_config, "enable_confidence_head", False)
            ),
            "called_by_serving_path": False,
        },
    run_config=run_config)
    run_manifest.record_value(
        "float32_matmul_precision", torch.get_float32_matmul_precision()
    , run_config=run_config)
    run_manifest.record_value(
        "blockhead_parity_record", config_values(run_config).get('engine.engine_settings.w277_blockhead_parity_record')
    , run_config=run_config)


def record_blockhead_runtime(tf32_state: dict, *, graph: bool, run_config=None) -> None:
    """Record what the speculator resolved after the load: the TF32 pin and the regime."""
    run_manifest = _run_manifest()
    run_manifest.record_value("blockhead_tf32", dict(tf32_state), run_config=run_config)
    run_manifest.record_value("blockhead_graph", bool(graph), run_config=run_config)


def blockhead_batch_requested(*, run_config=None) -> bool:
    value = config_values(run_config).get('engine.engine_settings.blockhead_batch', "0")
    if value not in ("0", "1"):
        raise ValueError("JETSPEC_BLOCKHEAD_BATCH must be 0 or 1")
    return value == "1"


def blockhead_graph_requested(*, run_config=None) -> bool:
    value = config_values(run_config).get('engine.engine_settings.blockhead_graph', "0")
    if value not in ("0", "1"):
        raise ValueError("JETSPEC_BLOCKHEAD_GRAPH must be 0 or 1")
    if value == "1" and config_values(run_config).get('engine.engine_settings.torchdynamo_disable') == "1":
        raise ValueError("BLOCKHEAD GRAPH refuses E")
    return value == "1"


def configure_blockhead_tf32(*, run_config=None) -> dict[str, bool]:
    """Apply the explicit fp32 parity pin without changing the default path."""
    value = config_values(run_config).get('engine.engine_settings.blockhead_tf32')
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


def block_size_read(*, batch: int = 1, run_config=None) -> int:
    if batch > 1 and 'method.method_settings.drafts' in config_values(run_config):
        drafts = int(config_values(run_config)['method.method_settings.drafts'])
        if drafts not in (3, 7, 15, 31):
            raise ValueError("batched block-head JETSPEC_DRAFTS must be 3, 7, 15 or 31")
        return drafts + 1
    value = config_values(run_config).get('method.method_settings.block_size_read', "16")
    if value not in ("8", "16", "32"):
        raise ValueError("JETSPEC_BLOCK_SIZE_READ must be 8, 16 or 32")
    return int(value)


def domino_options(dflash_config: dict) -> dict:
    """The Domino declarations the walk depends on, each checked for its kind (as the reader checks them)."""
    if dflash_config.get("projector_type") != DOMINO_PROJECTOR:
        raise ValueError(f"dflash_config.projector_type must be {DOMINO_PROJECTOR!r}, "
                         f"got {dflash_config.get('projector_type')!r}")
    if not isinstance(dflash_config.get("mask_token_id"), int):
        raise ValueError("Domino serving needs an integer dflash_config.mask_token_id")
    shift_label = dflash_config.get("shift_label", False)
    if not isinstance(shift_label, bool):
        raise ValueError(f"dflash_config.shift_label must be a boolean, got {shift_label!r}")
    prefix_len = dflash_config.get("pure_draft_prefix_len", 0)
    if not isinstance(prefix_len, int) or isinstance(prefix_len, bool) or prefix_len < 0:
        raise ValueError("dflash_config.pure_draft_prefix_len must be a non-negative integer, "
                         f"got {prefix_len!r}")
    for key in ("gru_hidden_dim", "emb_dim"):
        value = dflash_config.get(key)
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(f"Domino serving needs an integer dflash_config.{key}")
    # The backbone runs bidirectionally (the bundled loop passes is_causal=False). A SpecForge
    # export declares that as causal_head=false; any other declared value is not represented.
    if dflash_config.get("causal_head", False) is not False:
        raise ValueError("a Domino backbone runs bidirectionally; dflash_config.causal_head "
                         f"must be false or absent, got {dflash_config.get('causal_head')!r}")
    return {"projector_type": DOMINO_PROJECTOR, "mask_token_id": dflash_config["mask_token_id"],
            "shift_label": shift_label, "pure_draft_prefix_len": prefix_len,
            "gru_hidden_dim": dflash_config["gru_hidden_dim"], "emb_dim": dflash_config["emb_dim"]}


def blockhead_verify_block(*, run_config=None) -> int | None:
    """JETSPEC_BLOCKHEAD_BLOCK: a verify block of 4 or 8 (anchor included); unset keeps the trained block."""
    value = config_values(run_config).get('engine.engine_settings.blockhead_block')
    if value is None:
        return None
    if value not in ("4", "8"):
        raise ValueError("JETSPEC_BLOCKHEAD_BLOCK must be 4 or 8")
    return int(value)


def read_checkpoint_contract(checkpoint_dir: str | Path, *, block_size_read: int = 16,
                             verify_block: int | None = None) -> CheckpointContract:
    checkpoint = Path(checkpoint_dir)
    config = json.loads((checkpoint / "config.json").read_text())
    dflash = config.get("dflash_config") or {}
    plain_dflash = "target_layer_ids" not in config
    if dflash.get("projector_type") == "dspark" and "dspark_config" in config:
        target_layer_ids = config.get("target_layer_ids")
        layers = config.get("num_target_layers")
        if (not isinstance(target_layer_ids, list) or not target_layer_ids
                or any(type(value) is not int or value < 0 or value >= layers
                       for value in target_layer_ids)
                or any(left >= right for left, right in zip(target_layer_ids, target_layer_ids[1:]))
                or target_layer_ids != dflash.get("target_layer_ids")
                or target_layer_ids != config["dspark_config"].get("target_layer_ids")):
            raise ValueError("hybrid block-head target taps disagree or exceed target layers")
        served = int(config.get("block_size", 0))
        trained = int(config.get("training_block_size", 0))
        if served < 1 or trained < served or block_size_read not in (served + 1, trained):
            raise ValueError("hybrid block-head read must match serving or training block")
        query_size = served if block_size_read == served + 1 else trained
        return CheckpointContract(query_size, target_layer_ids, "stock",
                                  query_size, 0, trained)
    if dflash.get("projector_type") == DOMINO_PROJECTOR or any(
            key in dflash for key in ("gru_hidden_dim", "emb_dim", "pure_draft_prefix_len")):
        return _domino_contract(config, dflash, block_size_read=block_size_read)
    if dflash.get("projector_type") is not None:
        raise ValueError(f"unsupported block-head projector_type={dflash['projector_type']!r}")
    if plain_dflash and (config.get("block_size") != 16 or dflash.get("causal_head") is not False):
        raise ValueError("plain DFlash serving requires block_size=16 and causal_head=false")
    target_layer_ids = dflash.get("target_layer_ids", []) if plain_dflash else config["target_layer_ids"]
    layers = config.get("num_target_layers")
    # Older 8B exports omit the target depth. Keep their established contract;
    # full checkpoint configs validate against their own target architecture.
    if layers is None:
        if target_layer_ids != TARGET_LAYER_IDS:
            raise ValueError(f"block-head target_layer_ids mismatch: "
                             f"expected={TARGET_LAYER_IDS} actual={target_layer_ids}")
    elif (type(layers) is not int or layers < 1
          or not isinstance(target_layer_ids, list) or not target_layer_ids
          or any(type(value) is not int or not 0 <= value < layers for value in target_layer_ids)
          or any(left >= right for left, right in zip(target_layer_ids, target_layer_ids[1:]))
          or ("target_layer_ids" in dflash and target_layer_ids != dflash["target_layer_ids"])):
        raise ValueError("block-head target_layer_ids disagree or exceed num_target_layers")
    trained = int(config.get("block_size", 0))
    trained_lookahead = trained - 1 if plain_dflash else trained
    if trained_lookahead not in (7, 15):
        raise ValueError(f"block-head block_size must be 7 or 15, got {trained_lookahead}")
    if block_size_read not in (4, 8, 16, 32) or (block_size_read != 16 and trained_lookahead != 15):
        raise ValueError("block size 4, 8 or 32 requires a head trained for 15 drafts")
    query_size = block_size_read if block_size_read != 16 else trained
    lookahead = block_size_read - 1 if block_size_read != 16 else trained_lookahead
    if verify_block is not None:
        # The head keeps its trained query block (the drafts it was trained to make); the round
        # verifies only the first verify_block - 1 of them.
        if block_size_read != 16 or trained_lookahead != 15:
            raise ValueError("JETSPEC_BLOCKHEAD_BLOCK needs the trained 16-wide block "
                             "(JETSPEC_DRAFTS=15, JETSPEC_BLOCK_SIZE_READ=16) and a head trained for 15 drafts")
        lookahead = verify_block - 1
    target_weights = "stock" if trained_lookahead == 15 else "checkpoint_embedded"
    return CheckpointContract(lookahead, target_layer_ids, target_weights,
                              query_size, int(plain_dflash), trained)


def _domino_contract(config: dict, dflash: dict, *, block_size_read: int) -> CheckpointContract:
    """A Domino head: taps in dflash_config, a 16-slot block of [anchor, mask x 15], 15 drafts.

    With shift_label the backbone's position i drafts slot i+1 (the anchor's own position drafts
    slot 1, offset 0); without it the anchor position is dropped (offset 1), as the bundled loop
    slices it. Either way the walk reads slots 1..15."""
    options = domino_options(dflash)
    if config.get("block_size") != 16:
        raise ValueError("Domino serving requires block_size=16")
    if block_size_read != 16:
        raise ValueError("Domino serving reads the trained 16-slot block only")
    target_layer_ids = [int(value) for value in dflash.get("target_layer_ids", [])]
    if target_layer_ids != TARGET_LAYER_IDS:
        raise ValueError(
            "block-head target_layer_ids mismatch: "
            f"expected={TARGET_LAYER_IDS} actual={target_layer_ids}"
        )
    offset = 0 if options["shift_label"] else 1
    return CheckpointContract(15, target_layer_ids, "stock", 16, offset, 16, options)


def blockhead_config_overrides(
    checkpoint_dir: str | None,
    config_kwargs: dict,
*, run_config=None) -> dict:
    """Apply the default-off environment selection before ``Config`` is built."""
    checkpoint_dir = checkpoint_dir or config_values(run_config).get('model.model_settings.blockhead') or None
    if checkpoint_dir is None:
        return dict(config_kwargs)
    # Two GPUs without async = the target tensor-parallel (tp2); the head stays on rank 0.
    if int(config_kwargs.get("num_gpus", 1)) not in (1, 2):
        raise ValueError("block-head serving requires one GPU (or two for target TP)")
    if (int(config_kwargs.get("max_num_seqs", 1)) > 1
            and blockhead_graph_requested(run_config=run_config) and not blockhead_batch_requested(run_config=run_config)):
        raise ValueError("batched block-head graph requires JETSPEC_BLOCKHEAD_BATCH=1")
    if config_kwargs.get("draft_async", False):
        raise ValueError("block-head serving uses the synchronous serial slot")
    if config_kwargs.get("jetspec", False) or config_kwargs.get("use_eagle", False):
        raise ValueError("block-head serving cannot be combined with another draft mode")
    contract = read_checkpoint_contract(
        checkpoint_dir, block_size_read=block_size_read(
            batch=int(config_kwargs.get("max_num_seqs", 1)), run_config=run_config),
        verify_block=blockhead_verify_block(run_config=run_config))
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


def complete_round_anatomy(
    partial: list[dict],
    *,
    accepted: list[int],
    verify_ms: float,
    round_ms: float,
) -> list[dict]:
    if len(partial) != len(accepted):
        raise ValueError("block-head anatomy and accepted rows differ")
    records = []
    for timing, accepted_count in zip(partial, accepted):
        record = dict(timing)
        accepted_raw = int(accepted_count)
        if "completion_before" in timing and "max_new_tokens" in timing:
            remaining = max(
                0,
                int(timing["max_new_tokens"])
                - 1
                - int(timing["completion_before"]),
            )
            accepted_count = min(accepted_raw, remaining)
        record.update(
            verify_ms=float(verify_ms),
            residual_ms=(
                float(round_ms)
                - float(timing["head_ms"])
                - float(timing["markov_ms"])
                - float(timing.get("domino_ms", 0.0))
                - float(verify_ms)
            ),
            round_ms=float(round_ms),
            accepted_raw=accepted_raw,
            accepted=int(accepted_count),
        )
        records.append(record)
    return records


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


def load_checkpoint_target_weights(checkpoint_dir: str | Path, target) -> Path:
    """Load a released head's frozen target tensors into the live engine target."""
    checkpoint = Path(checkpoint_dir)
    separate = checkpoint / "target_release_weights.safetensors"
    source = separate if separate.is_file() else checkpoint / "model.safetensors"
    if not source.is_file():
        raise FileNotFoundError(f"block-head target weights missing: {source}")
    destinations = {
        "embed_tokens.weight": target.model.embed_tokens.weight,
        "lm_head.weight": target.lm_head.weight,
    }
    with safe_open(source, framework="pt", device="cpu") as handle:
        available = set(handle.keys())
        for short_name, destination in destinations.items():
            candidates = (short_name, f"model.{short_name}")
            source_name = next((name for name in candidates if name in available), None)
            if source_name is None:
                raise KeyError(
                    f"block-head target tensor missing from {source.name}: {short_name}"
                )
            tensor = handle.get_tensor(source_name)
            if tensor.shape != destination.shape:
                raise ValueError(
                    f"block-head target shape mismatch for {short_name}: "
                    f"checkpoint={tuple(tensor.shape)} engine={tuple(destination.shape)}"
                )
            with torch.no_grad():
                destination.copy_(
                    tensor.to(device=destination.device, dtype=destination.dtype)
                )
            del tensor
    return source


def resolve_draft_dtype(name: str | None = None, *, run_config=None) -> torch.dtype:
    if name is None:
        name = config_values(run_config).get('model.model_settings.draft_dtype', "bf16")
    by_name = {"bf16": torch.bfloat16, "fp32": torch.float32}
    if name not in by_name:
        raise ValueError("JETSPEC_DRAFT_DTYPE must be bf16 or fp32")
    return by_name[name]


def load_blockhead(
    checkpoint_dir: str | Path,
    *,
    target,
    device: str | torch.device,
    max_num_seqs: int = 1,
run_config=None) -> LoadedBlockhead:
    """Load one pinned head beside the single-GPU verifier target."""
    checkpoint = Path(checkpoint_dir)
    contract = read_checkpoint_contract(
        checkpoint, block_size_read=block_size_read(batch=max_num_seqs, run_config=run_config),
        verify_block=blockhead_verify_block(run_config=run_config))
    target_layers = getattr(target.model, "layers", None)
    if target_layers is not None and any(layer >= len(target_layers) for layer in contract.target_layer_ids):
        raise ValueError(f"block-head target_layer_ids must be below target "
                         f"num_hidden_layers={len(target_layers)}")
    dtype = resolve_draft_dtype(run_config=run_config)
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
        if blockhead_graph_requested(run_config=run_config):
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
    run_config=run_config)
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
        drafter=TimedLinearBlockhead(linear, domino=domino_walk, run_config=run_config),
        lookahead=contract.lookahead,
        target_weights=contract.target_weights,
        target_weights_path=str(weights_path) if weights_path else None,
    )


def _is_domino_head_key(key: str) -> bool:
    return key.startswith("prefix_gru.") or key.startswith("embed_proj.")


def load_domino_backbone(checkpoint: Path, api, *, device, dtype: torch.dtype):
    """The Domino backbone through the pinned kit's backbone class (no Markov, no confidence head).

    The kit's own loader refuses a config without causal_head=false and a state with the walk's
    tensors, so the Domino declarations are translated here the way it translates a plain DFlash
    config, and the load is held to the same strict key rule plus the four walk tensors."""
    config = api.load_config(str(checkpoint))
    dflash = dict(getattr(config, "dflash_config", {}) or {})
    domino_options(dflash)
    target_layer_ids = [int(value) for value in dflash["target_layer_ids"]]
    config.target_layer_ids = target_layer_ids
    config.mask_token_id = int(dflash["mask_token_id"])
    config.dflash_config = {
        "target_layer_ids": target_layer_ids,
        "mask_token_id": int(dflash["mask_token_id"]),
        "causal_head": False,  # the bundled Domino loop runs the backbone with is_causal=False
        "target_hidden_size": int(config.hidden_size),
    }
    config._attn_implementation = "sdpa"
    config.markov_rank = 0
    config.enable_confidence_head = False
    config.confidence_head_with_markov = False
    config.num_anchors = 1
    model, loading_info = api.head_model.from_pretrained(
        str(checkpoint),
        config=config,
        dtype=dtype,
        attn_implementation="sdpa",
        output_loading_info=True,
    )
    unexpected = list(loading_info.get("unexpected_keys", []))
    missing = list(loading_info.get("missing_keys", []))
    bad_unexpected = [
        key for key in unexpected
        if not (api.is_shared_key(key) or _is_domino_head_key(key))
    ]
    bad_missing = [key for key in missing if not api.is_ignorable_missing(key)]
    if bad_unexpected:
        raise RuntimeError(f"Domino backbone: unexpected checkpoint keys: {bad_unexpected}")
    if bad_missing:
        raise RuntimeError(f"Domino backbone: missing model keys: {bad_missing}")
    if model.markov_head is not None or model.confidence_head is not None:
        raise RuntimeError("Domino backbone must carry no Markov or confidence head")
    return model.to(device).eval()


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

    @torch.inference_mode()
    def walk(
        self,
        hidden: torch.Tensor,
        base_logits: torch.Tensor,
        anchor: torch.Tensor,
        embed,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """(tokens (1, depth), corrected logits (1, depth, V)) for one 15-slot block.

        ``hidden`` is the backbone's per-slot hidden, ``base_logits`` the target head over it,
        ``anchor`` the block's first token (1,), ``embed`` the target's token embedding."""
        dtype = self.embed_proj[0].weight.dtype
        depth = int(base_logits.shape[1])
        prefix = min(self.prefix_len, depth)
        realized = torch.cat(
            [anchor.reshape(1, 1), base_logits[:, :prefix, :].argmax(dim=-1)], dim=1)
        _, state = self.prefix_gru(embed(realized).to(dtype))
        slots = [base_logits[:, :prefix, :]]
        for i in range(prefix, depth):
            bias = self.embed_proj(
                torch.cat([hidden[:, i:i + 1, :].to(dtype), state.transpose(0, 1)], dim=-1))
            corrected = base_logits[:, i:i + 1, :] + bias
            slots.append(corrected)
            if i + 1 < depth:
                _, state = self.prefix_gru(embed(corrected.argmax(dim=-1)).to(dtype), state)
        logits = torch.cat(slots, dim=1)
        return logits.argmax(dim=-1), logits


def load_domino_walk(checkpoint: Path, options: dict, head_config, *, device, dtype: torch.dtype) -> DominoWalk:
    """Strictly load prefix_gru and embed_proj from the checkpoint; refuse when any is absent."""
    source = checkpoint / "model.safetensors"
    with safe_open(source, framework="pt", device="cpu") as handle:
        keys = [key for key in handle.keys() if _is_domino_head_key(key)]
        absent = [key for key in DOMINO_HEAD_TENSORS if key not in keys]
        if absent:
            raise ValueError(f"the checkpoint carries no Domino head: missing {absent} in {source.name}")
        state = {key: handle.get_tensor(key) for key in keys}
    walk = DominoWalk(
        hidden_size=int(head_config.hidden_size),
        vocab_size=int(head_config.vocab_size),
        options=options,
    )
    walk.load_state_dict(state, strict=True)
    del state
    return walk.to(device=device, dtype=dtype).eval()


class TimedLinearBlockhead:
    """Time the eager block forward separately from its serial token sampler."""

    def __init__(self, drafter, domino: DominoWalk | None = None, *, run_config=None):
        self.run_config = run_config
        self.drafter = drafter
        self.domino = domino
        self._parity_round = 0

    def reset_cache(self) -> None:
        self.drafter.reset_cache()
        self._parity_round = 0

    def graph_state(self) -> dict:
        return {
            "head_graph": False,
            "captures": 0,
            "query_replays": 0,
            "extension_replays": 0,
            "prefill_extensions": 0,
        }

    def _record_parity(
        self,
        context_ids: torch.Tensor,
        target_hidden: torch.Tensor,
        hidden: torch.Tensor,
        base_logits: torch.Tensor,
        corrected: torch.Tensor,
        tokens: torch.Tensor,
    ) -> None:
        root = config_values(self.run_config).get('engine.engine_settings.w277_blockhead_parity_record')
        if root is None or self._parity_round >= 8:
            return
        path = Path(root)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "context_ids": context_ids.detach().cpu(),
                "target_hidden": target_hidden.detach().cpu(),
                "block_hidden": hidden.detach().cpu(),
                "base_logits": base_logits.detach().cpu(),
                "corrected_logits": corrected.detach().cpu(),
                "tokens": tokens.detach().cpu(),
                "matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
                "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
            },
            path / f"round-{self._parity_round + 1:02d}.pt",
        )

    @torch.inference_mode()
    def propose(
        self,
        context_ids: torch.Tensor,
        k: int,
        *,
        target_hidden: torch.Tensor,
    ) -> BlockheadProposal:
        use_cuda_events = context_ids.device.type == "cuda"
        if use_cuda_events:
            start = torch.cuda.Event(enable_timing=True)
            split = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
        else:
            start_s = perf_counter()

        hidden = self.drafter._fwd.forward_hidden(context_ids, target_hidden)
        offset = getattr(self.drafter, "draft_offset", 0)
        proposal_hidden = hidden[:, offset : offset + k, :]
        lm_head = self.drafter.target.lm_head
        lm_input = proposal_hidden.to(dtype=lm_head.weight.dtype) if hasattr(lm_head, "weight") else proposal_hidden
        base_logits = lm_head(lm_input)

        if use_cuda_events:
            split.record()
        else:
            split_s = perf_counter()

        anchor = context_ids[:, -1]
        from jetspec.engines.jetspec_plus_plus.trace.current import active_recorder
        trace = active_recorder()
        if self.domino is not None:
            embed = self.drafter.target.model.embed_tokens
            if trace is None:
                sampled, corrected = self.domino.walk(
                    proposal_hidden, base_logits, anchor, embed)
            else:
                with trace.span('domino_walk', 'draft'):
                    sampled, corrected = self.domino.walk(
                        proposal_hidden, base_logits, anchor, embed)
        elif trace is None:
            sampled, corrected = self.drafter.head.sample_draft_tokens(
                base_logits, first_prev_token_ids=anchor, temperature=0.0,
                hidden_states=proposal_hidden)
        else:
            label = 'sampling' if offset else 'markov_sampling'
            with trace.span(label, 'draft'):
                sampled, corrected = self.drafter.head.sample_draft_tokens(
                    base_logits, first_prev_token_ids=anchor, temperature=0.0,
                    hidden_states=proposal_hidden)

        if use_cuda_events:
            end.record()
            end.synchronize()
            head_ms = float(start.elapsed_time(split))
            markov_ms = float(split.elapsed_time(end))
        else:
            end_s = perf_counter()
            head_ms = (split_s - start_s) * 1000.0
            markov_ms = (end_s - split_s) * 1000.0
        domino_ms = None
        if self.domino is not None:
            # The sequential corrector here is the Domino walk; no Markov walk ran.
            domino_ms, markov_ms = markov_ms, 0.0
        self._record_parity(
            context_ids,
            target_hidden,
            proposal_hidden,
            base_logits,
            corrected,
            sampled,
        )
        self._parity_round += 1
        return BlockheadProposal(
            tokens=sampled.reshape(-1)[:k],
            logits=corrected[:, :k, :],
            head_ms=head_ms,
            markov_ms=markov_ms,
            domino_ms=domino_ms,
        )

    @torch.inference_mode()
    def propose_batch(
        self,
        context_ids: list[torch.Tensor],
        k: int,
        *,
        target_hidden: list[torch.Tensor],
    ) -> BlockheadProposal:
        """Draft all rows together, masking ragged context and recomputing it each round."""
        fwd = self.drafter._fwd
        if len(context_ids) != len(target_hidden) or not context_ids:
            raise ValueError("block-head batch inputs differ or are empty")
        widths = [int(row.shape[1]) for row in target_hidden]
        if any(context.shape[1] != width + 1 for context, width in zip(context_ids, widths)):
            raise ValueError("block-head context and target taps differ")
        max_width = max(widths)
        block_size = fwd.block_size
        device = fwd.device
        use_cuda_events = device.type == "cuda"
        if use_cuda_events:
            start = torch.cuda.Event(enable_timing=True)
            split = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
        else:
            start_s = perf_counter()

        anchors = torch.stack([row[0, -1] for row in context_ids]).to(device)
        noise_ids = torch.cat((anchors[:, None], torch.full(
            (len(widths), block_size - 1), fwd.mask_token_id,
            dtype=anchors.dtype, device=device)), dim=1)
        noise_embedding = fwd.target.model.embed_tokens(noise_ids).to(dtype=fwd.dtype)
        hidden_padded = torch.cat([
            F.pad(row.to(device=device, dtype=fwd.dtype), (0, 0, 0, max_width - width))
            for row, width in zip(target_hidden, widths)
        ], dim=0)
        lengths = torch.tensor(widths, device=device)
        context_valid = torch.arange(max_width, device=device)[None, :] < lengths[:, None]
        key_valid = torch.cat((context_valid, torch.ones(
            (len(widths), block_size), dtype=torch.bool, device=device)), dim=1)
        attention_mask = key_valid[:, None, None, :].expand(-1, 1, block_size, -1)
        position_ids = torch.cat((
            torch.arange(max_width, device=device)[None, :].expand(len(widths), -1),
            lengths[:, None] + torch.arange(block_size, device=device)[None, :],
        ), dim=1)
        hidden = fwd.head(
            target_hidden=hidden_padded,
            noise_embedding=noise_embedding,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=None,
            use_cache=False,
            is_causal=fwd.head.resolve_causal_head("auto"),
        )
        offset = getattr(self.drafter, "draft_offset", 0)
        proposal_hidden = hidden[:, offset : offset + k, :]
        lm_head = self.drafter.target.lm_head
        lm_input = (proposal_hidden.to(dtype=lm_head.weight.dtype)
                    if hasattr(lm_head, "weight") else proposal_hidden)
        base_logits = lm_head(lm_input)

        if use_cuda_events:
            split.record()
        else:
            split_s = perf_counter()
        from jetspec.engines.jetspec_plus_plus.trace.current import active_recorder
        trace = active_recorder()
        if trace is None:
            sampled, corrected = self.drafter.head.sample_draft_tokens(
                base_logits, first_prev_token_ids=anchors, temperature=0.0,
                hidden_states=proposal_hidden)
        else:
            label = "sampling" if offset else "markov_sampling"
            with trace.span(label, "draft"):
                sampled, corrected = self.drafter.head.sample_draft_tokens(
                    base_logits, first_prev_token_ids=anchors, temperature=0.0,
                    hidden_states=proposal_hidden)

        if use_cuda_events:
            end.record()
            end.synchronize()
            head_ms = float(start.elapsed_time(split))
            markov_ms = float(split.elapsed_time(end))
        else:
            end_s = perf_counter()
            head_ms = (split_s - start_s) * 1000.0
            markov_ms = (end_s - split_s) * 1000.0
        return BlockheadProposal(sampled[:, :k], corrected[:, :k, :],
                                 head_ms, markov_ms)


def validate_vendor_manifest() -> int:
    """Fail closed if any byte in the pinned serving kit has changed."""
    manifest = json.loads(((KIT_DIR / 'manifest.json')).read_text())
    for name, expected in manifest["files"].items():
        actual = hashlib.sha256((KIT_DIR / name).read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(
                f"block-head kit hash mismatch for {name}: expected={expected} actual={actual}"
            )
    return len(manifest["files"])


def _package(name: str) -> ModuleType:
    module = sys.modules.get(name)
    if module is None:
        module = ModuleType(name)
        module.__path__ = []
        sys.modules[name] = module
    return module


def _load_vendor_module(name: str, filename: str, *, KIT_DIR: Path = KIT_DIR) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, KIT_DIR / filename)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load block-head kit module {filename}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    source_path = KIT_DIR / filename
    code = compile(
        source_path.read_bytes(),
        str(source_path),
        "exec",
        flags=__future__.annotations.compiler_flag,
        dont_inherit=True,
    )
    exec(code, module.__dict__)
    return module


def load_vendor_api():
    """Load byte-exact kit files behind neutral serving module names."""
    global _VENDOR_API
    if _VENDOR_API is not None:
        return _VENDOR_API
    validate_vendor_manifest()
    for package in ("jetspec", "jetspec.core", "jetspec.models", "jetspec.blockhead"):
        _package(package)

    sampler = _load_vendor_module("jetspec.blockhead.sampler", "sampler.py")
    sys.modules["jetspec.core.sampler"] = sampler
    draft = _load_vendor_module("jetspec.blockhead.draft", "draft.py")
    sys.modules["jetspec.draft"] = draft
    backbone = _load_vendor_module("jetspec.blockhead.backbone", "backbone.py")
    sys.modules["jetspec.models.draft_head"] = backbone
    markov = _load_vendor_module("jetspec.blockhead.markov", "markov.py")
    sys.modules["jetspec.models.markov_head"] = markov
    forward_base = _load_vendor_module(
        "jetspec.blockhead.forward_base", "forward_base.py"
    )
    sys.modules["jetspec.draft_head_adapter"] = forward_base
    adapter = _load_vendor_module("jetspec.blockhead.adapter", "adapter.py")
    head = _load_vendor_module("jetspec.blockhead.head", "head.py")
    _VENDOR_API = SimpleNamespace(
        linear_drafter=adapter.DSparkLinearDrafter,
        load_head=head.load_dspark_head,
        head_model=head.DSparkDraftModel,
        load_config=head.load_qwen3_config_with_rope_compat,
        is_shared_key=head._is_embed_or_lm_head_key,
        is_ignorable_missing=head._is_ignorable_missing,
    )
    return _VENDOR_API


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
