"""Tiny checkpoints retaining the published 8B metadata and tensor names."""
import importlib.util
import json
from pathlib import Path
import sys

import torch
from safetensors.torch import save_file
from transformers import Qwen3Config

REAL_8B = Path(__file__).parent / 'fixtures/real_8b'


def tiny_config(family):
    raw = json.loads((REAL_8B / family / 'config.json').read_text())
    raw.update(hidden_size=16, intermediate_size=32, head_dim=8,
               num_attention_heads=2, num_key_value_heads=1, vocab_size=41)
    return raw


def bundled_model(config):
    name = 'real_8b_dflash_fixture'
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, REAL_8B / 'drafter/dflash.py')
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name].DFlashDraftModel(config).eval()


def real_checkpoint(directory, seed):
    torch.manual_seed(seed)
    raw = tiny_config('drafter')
    config = Qwen3Config.from_dict(raw)
    config._attn_implementation = 'sdpa'
    head = bundled_model(config)
    keys = json.loads((REAL_8B / 'drafter/keys.json').read_text())['drafter']
    assert set(head.state_dict()) == set(keys)
    (directory / 'config.json').write_text(json.dumps(raw))
    (directory / 'dflash.py').write_bytes((REAL_8B / 'drafter/dflash.py').read_bytes())
    save_file({name: head.state_dict()[name] for name in keys}, directory / 'model.safetensors')
    return head
