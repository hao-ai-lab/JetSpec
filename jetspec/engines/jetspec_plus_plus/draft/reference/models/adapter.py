"""Load and validate local adapter artifacts for the eager route."""
from pathlib import Path
import json
import math

from safetensors import safe_open
from safetensors.torch import load_file
from transformers import Qwen3Config

from ...method.adapter import CtxKVC1Head


def load_adapter(path, device, dtype) -> CtxKVC1Head:
    """Load a local HF adapter directory, rejecting incompatible checkpoint banks.

    The converter belongs to the drafter, as in the reference loader. It is checked
    here but does not project the head's tap-space output.
    """
    directory = Path(path)
    for name in ('config.json', 'head.safetensors', 'converter.safetensors'):
        if not (directory / name).is_file():
            raise FileNotFoundError(directory / name)
    raw = json.loads((directory / 'config.json').read_text())
    if not isinstance(raw, dict) or raw.get('model_type') != 'qwen3':
        raise ValueError('adapter config must describe qwen3')
    parameters = raw.get('rope_parameters')
    if parameters is not None and not isinstance(parameters, dict):
        raise ValueError('rope_parameters must be an object')
    rope_type = (parameters or {}).get('rope_type', raw.get('rope_type'))
    if rope_type not in (None, 'default'):
        raise ValueError(f'unsupported rope_type {rope_type!r}')
    if any(name in raw or name in (parameters or {})
           for name in ('partial_rotary_factor', 'partial_rotary_ratio', 'partial_rotary_dim')):
        raise ValueError('partial rotary is unsupported for the drafter')
    theta = raw.get('rope_theta', (parameters or {}).get('rope_theta'))
    if (not isinstance(theta, (int, float)) or isinstance(theta, bool)
            or not math.isfinite(theta) or theta <= 0):
        raise ValueError('rope_theta is absent or invalid')
    # from_dict reads no remote code and keeps the reference's nested-theta compatibility.
    config = Qwen3Config.from_dict(raw)
    config.rope_theta = float(theta)
    config._attn_implementation = 'sdpa'
    with safe_open(directory / 'head.safetensors', framework='pt') as checkpoint:
        if (checkpoint.metadata() or {}).get('PREFIX_KV_SOURCE') is not None:
            raise ValueError('head PREFIX_KV_SOURCE requires target cache K/V')
    head = CtxKVC1Head(config)
    converter = load_file(directory / 'converter.safetensors', device='cpu')
    # The training loader requires exactly these two tensors, unlike the reader's subset check.
    if set(converter) != {'fc.weight', 'hidden_norm.weight'}:
        raise ValueError(f'adapter converter keys differ: {sorted(converter)}')
    shapes = {'fc.weight': (head.hidden_size, head.hspace_dim),
              'hidden_norm.weight': (head.hidden_size,)}
    for name, shape in shapes.items():
        if tuple(converter[name].shape) != shape or not converter[name].is_floating_point():
            raise ValueError(f'adapter converter {name} must be floating point with shape {shape}')
    state = load_file(directory / 'head.safetensors', device='cpu')
    if any(not value.is_floating_point() for value in state.values()):
        raise ValueError('adapter head tensors must be floating point')
    head.load_state_dict(state, strict=True)
    head.to(device=device, dtype=dtype)
    head.eval()
    head.requires_grad_(False)
    return head
