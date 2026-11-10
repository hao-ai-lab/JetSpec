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
from ...drafter import (resolve_stock_drafter, resolve_causal_head,
                        check_stock_drafter_keys, StockAdaptedMark)

__all__ = ["CHECKPOINT", "REVISION", "load_drafter", "check_state_keys",
           "resolve_stock_drafter", "resolve_causal_head", "check_stock_drafter_keys",
           "StockAdaptedMark",
           "load_qwen3_config_with_rope_compat", "restore_rope_theta"]

CHECKPOINT = 'JetSpec/jetspec-plus-plus-qwen3-8b'
REVISION = '9bd8b157'


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
