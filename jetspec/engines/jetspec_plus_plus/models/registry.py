"""Checkpoint identity is resolved here; consumers use model properties."""
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers import AutoConfig

from jetspec.engines.jetspec_plus_plus.draft.method.model_spec import ModelSpec


def load_config(path):
    raw = json.loads((Path(path) / 'config.json').read_text())
    if raw.get('model_type') not in {'qwen3_5', 'qwen3_5_text'}:
        return AutoConfig.from_pretrained(path)
    text = dict(raw.get('text_config', raw))
    if text.get('model_type') != 'qwen3_5_text':
        raise ValueError('hybrid target needs a qwen3_5_text text_config')
    ModelSpec.from_configs(raw)
    text['rope_theta'] = text.get('rope_theta', (text.get('rope_parameters') or {}).get('rope_theta'))
    dtype = text.get('dtype', 'bfloat16')
    if dtype not in {'bfloat16', 'float32'}:
        raise ValueError(f'unsupported hybrid target dtype: {dtype}')
    text['torch_dtype'] = {'bfloat16': torch.bfloat16, 'float32': torch.float32}[dtype]
    return SimpleNamespace(**text)


def model_spec(config):
    raw = config.to_dict() if hasattr(config, 'to_dict') else vars(config)
    if raw.get('model_type') == 'llama':
        return None  # The retained comparison model is outside the target refactor.
    return ModelSpec.from_configs(raw)


def model_class(config, *, draft=False):
    if getattr(config, 'model_type', None) == 'llama':
        from jetspec.engines.jetspec_plus_plus.models.llama3 import LlamaForCausalLM
        return LlamaForCausalLM
    spec = model_spec(config)
    if draft and spec.has_state:
        raise ValueError('stateful target cannot be used as the draft model')
    from jetspec.engines.jetspec_plus_plus.models.target import TargetForCausalLM
    return TargetForCausalLM
