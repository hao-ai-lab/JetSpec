"""Ported from the JetSpec++ reference implementation.

Also mirrors the reference draft-model construction.

Public API: load_drafter(checkpoint=CHECKPOINT, *, device='cpu',
dtype=torch.bfloat16, revision=REVISION) -> bundled DFlashDraftModel;
check_state_keys(expected, found, *, jacobi=False) -> None;
resolve_stock_drafter(options, source) -> bool; resolve_causal_head(options, source) -> bool;
check_stock_drafter_keys(head, state, source) -> None; StockAdaptedMark(width, *, dtype, device).
load_qwen3_config_with_rope_compat(repo_or_path, **kwargs) -> Qwen3Config;
restore_rope_theta(config) -> None.

The public DFlashDraftModel is not bitwise equivalent on a cached batch.
Load the stock checkpoint's bundled dflash.py and attach only a negative-zero mark.
Hub identifiers resolve only from the local cache at the pinned revision;
local checkpoint directories must carry the same architecture contract.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import torch
import torch.nn.functional as F
from torch import nn

__all__ = ["CHECKPOINT", "REVISION", "load_drafter", "check_state_keys",
           "resolve_stock_drafter", "resolve_causal_head", "check_stock_drafter_keys",
           "StockAdaptedMark",
           "load_qwen3_config_with_rope_compat", "restore_rope_theta"]

CHECKPOINT = 'JetSpec/jetspec-plus-plus-qwen3-8b'
REVISION = '43d3aa9d'
ADAPTED_MARK_TENSOR = 'adapted_position_embedding.weight'
RETIRED_DRAFTER = 'the 59-tensor drafter is retired'


def resolve_stock_drafter(dflash_config, source) -> bool:
    """Refuse retired Jacobi configs and noise unlike the served random context."""
    dflash_config = dflash_config or {}
    stock = dflash_config.get('stock_drafter', False)
    if not isinstance(stock, bool):
        raise ValueError(f'{source}: dflash_config.stock_drafter must be true or false, got {stock!r}')
    jacobi = dflash_config.get('jacobi_forcing', False)
    if not isinstance(jacobi, bool):
        raise ValueError(f'{source}: dflash_config.jacobi_forcing must be true or false, got {jacobi!r}')
    if jacobi and not stock:
        raise ValueError(f'{source}: {RETIRED_DRAFTER}; a Jacobi-forcing drafter must declare '
                         'dflash_config.stock_drafter true')
    if stock and dflash_config.get('fresh_fraction', 1.0) != 1.0:
        raise ValueError(f'{source}: a stock drafter is trained all-fresh; '
                         f'fresh_fraction is {dflash_config.get("fresh_fraction")!r}')
    noise_source = dflash_config.get('noise_source', 'context')
    if noise_source != 'context':
        raise ValueError(f'{source}: dflash_config.noise_source is {noise_source!r}; '
                         'the drafter initialises from random context only')
    return stock


def resolve_causal_head(dflash_config, source) -> bool:
    """Require the causal mask declared by the served checkpoint."""
    value = (dflash_config or {}).get('causal_head')
    if value is None:
        raise ValueError(f'{source}: dflash_config.causal_head is absent; it must declare true')
    if not isinstance(value, bool):
        raise ValueError(f'{source}: dflash_config.causal_head must be true or false, got {value!r}')
    if not value:
        raise ValueError(f'{source}: dflash_config.causal_head is false; the drafter requires a causal head')
    return value


def check_stock_drafter_keys(head: nn.Module, state, source) -> None:
    """Check the file's exact keys before attaching the non-persistent stock mark."""
    if ADAPTED_MARK_TENSOR in state:
        raise ValueError(f'{source}: {RETIRED_DRAFTER}; the file carries {ADAPTED_MARK_TENSOR}')
    expected = set(head.state_dict())
    if ADAPTED_MARK_TENSOR in expected:
        raise ValueError(f'{source}: the bundled dflash.py builds the retired mark {ADAPTED_MARK_TENSOR}')
    if set(state) != expected:
        raise ValueError(f'{source}: stock drafter key set differs: missing={sorted(expected - set(state))} '
                         f'unexpected={sorted(set(state) - expected)}')


class StockAdaptedMark(nn.Module):
    """Two non-persistent negative-zero rows preserve context bits, including -0.0."""

    def __init__(self, width: int, *, dtype: torch.dtype, device: torch.device) -> None:
        super().__init__()
        self.register_buffer('weight', torch.full((2, int(width)), -0.0, dtype=dtype, device=device),
                             persistent=False)

    def forward(self, indicator: torch.Tensor) -> torch.Tensor:
        return F.embedding(indicator, self.weight)


def restore_rope_theta(config) -> None:
    if hasattr(config, "rope_theta"):
        return
    rope_parameters = getattr(config, "rope_parameters", None) or {}
    if "rope_theta" not in rope_parameters:
        raise ValueError("config lacks rope_theta compatibility metadata")
    config.rope_theta = rope_parameters["rope_theta"]


def load_qwen3_config_with_rope_compat(repo_or_path: str, **kwargs):
    """Load a Qwen3 config preserving transformers-5 nested RoPE settings."""
    from transformers import Qwen3Config

    raw_config, _ = Qwen3Config.get_config_dict(repo_or_path, **kwargs)
    config = Qwen3Config.from_pretrained(repo_or_path, **kwargs)
    rope_parameters = raw_config.get("rope_parameters")
    theta = raw_config.get('rope_theta', (rope_parameters or {}).get('rope_theta'))
    if theta is not None:
        # Transformers 5 consumes nested RoPE and can discard the legacy attribute.
        config.rope_theta = theta
        if hasattr(config, 'rope_parameters'):
            config.rope_parameters = dict(config.rope_parameters or {}, rope_theta=theta)
    return config


def check_state_keys(expected: set[str], found: set[str], *, jacobi: bool = False) -> None:
    """Refuse any unexpected, missing, or dropped tensor before loading."""
    del jacobi
    unexpected = sorted(found - expected)
    missing = sorted(expected - found)
    if unexpected or missing:
        raise RuntimeError(
            f"drafter checkpoint does not match its model: unexpected={unexpected} missing={missing}"
        )


def load_drafter(checkpoint: str | Path = CHECKPOINT, *, device: str = 'cpu',
                 dtype: torch.dtype = torch.bfloat16, revision: str = REVISION,
                 model_family: str = 'auto'):
    """Build the checkpoint's bundled model and load every safetensor strictly."""
    from safetensors import safe_open
    from safetensors.torch import load_file

    if revision != REVISION:
        raise ValueError(f'drafter revision must be {REVISION}')
    checkpoint_path = Path(checkpoint).expanduser()
    if not checkpoint_path.is_dir():
        if str(checkpoint) != CHECKPOINT:
            raise FileNotFoundError(f'expected a local checkpoint directory or {CHECKPOINT}')
        from huggingface_hub import snapshot_download
        # The drafter sits at the repository root; adapter/ and calibration/ are separate inputs.
        checkpoint_path = Path(snapshot_download(CHECKPOINT, revision=revision, local_files_only=True,
                                                 allow_patterns=['config.json', 'dflash.py', 'model.safetensors']))
    raw_config = json.loads((checkpoint_path / 'config.json').read_text(encoding='utf-8'))
    options = raw_config.get('dflash_config') or {}
    family = ('8b' if raw_config.get('num_target_layers') == 36 else '27b') if model_family == 'auto' else model_family
    taps, target_layers = options.get('target_layer_ids'), raw_config.get('num_target_layers')
    if family not in ('8b', '27b'):
        raise ValueError('model_family must be auto, 8b or 27b')
    valid_taps = (isinstance(taps, list) and len(taps) == 5
                  and isinstance(target_layers, int) and target_layers >= 7
                  and all(type(tap) is int and 0 <= tap < target_layers for tap in taps)
                  and taps == sorted(set(taps)) and taps[-1] == target_layers - 3)
    if (raw_config.get('model_type') != 'qwen3'
            or raw_config.get('num_hidden_layers') != 5
            or raw_config.get('block_size') != 16
            or not valid_taps
            or (family == '8b' and (target_layers != 36 or taps != [1, 9, 17, 25, 33]))
            or raw_config.get('auto_map', {}).get('AutoModel') != 'dflash.DFlashDraftModel'):
        raise ValueError('expected a Qwen3 B16 five-layer drafter with five ordered target taps, '
                         'two target layers after the deepest tap, and auto_map dflash.DFlashDraftModel')
    stock = resolve_stock_drafter(options, checkpoint_path / 'config.json')
    resolve_causal_head(options, checkpoint_path / 'config.json')
    if not stock or options.get('jacobi_forcing') is not True:
        raise ValueError('expected a stock Jacobi-forcing drafter')
    model_file = checkpoint_path / 'dflash.py'
    if not model_file.is_file():
        raise FileNotFoundError(f'checkpoint lacks bundled dflash.py: {model_file}')
    with safe_open(str(checkpoint_path / 'model.safetensors'), framework='pt', device='cpu') as state_file:
        found_keys = set(state_file.keys())
    # Keep separate local checkpoints' Python modules distinct.
    module_name = 'jetspec_drafter_' + hashlib.sha256(str(model_file.resolve()).encode()).hexdigest()
    spec = importlib.util.spec_from_file_location(module_name, model_file)
    if spec is None or spec.loader is None:
        raise ImportError(f'cannot import {model_file}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    config = load_qwen3_config_with_rope_compat(str(checkpoint_path), local_files_only=True)
    config._attn_implementation = 'sdpa'
    previous_default = torch.get_default_dtype()
    try:
        torch.set_default_dtype(dtype)
        head = module.DFlashDraftModel(config)
    finally:
        torch.set_default_dtype(previous_default)
    if not getattr(head, 'jacobi_forcing', False):
        raise ValueError('checkpoint is not a Jacobi-forcing drafter')
    check_stock_drafter_keys(head, found_keys, checkpoint_path / 'model.safetensors')
    state = load_file(checkpoint_path / 'model.safetensors', device='cpu')
    head.load_state_dict(state, strict=True)
    del state
    parameter = next(head.parameters())
    head.adapted_position_embedding = StockAdaptedMark(
        head.fc.in_features, dtype=parameter.dtype, device=parameter.device)
    restore_rope_theta(head.config)
    return head.to(device).eval()
