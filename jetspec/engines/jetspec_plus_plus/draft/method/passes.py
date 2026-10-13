"""Jacobi pass math and completed-depth labels."""
from dataclasses import dataclass
import re

import torch

from jetspec.engines.jetspec_plus_plus.draft.method.ops import reference_op


def normalized_label(label):
    return isinstance(label, str) and re.fullmatch(r'J0d0|J[1-9][0-9]*', label) is not None


def completed_label(passes_run, *, max_depth=None):
    passes_run = int(passes_run)
    if passes_run < 1:
        raise ValueError('a served block must have at least one computed pass')
    depth = passes_run - 1
    if max_depth is not None:
        depth = min(depth, max_depth)
    return 'J0d0' if depth == 0 else f'J{depth}'


def resolve_tau(table, label, *, bundle=None, recipe=None):
    """Dataset pooling is already resolved; this fallback pools pass labels only."""
    if not normalized_label(label):
        raise ValueError(f'invalid normalized pass label {label!r}')
    key = label if label in table else 'pooled'
    if key not in table:
        raise ValueError(f'missing tau for {label}; bundle={bundle}; recipe={recipe}; '
                         'no explicit label or pooled key')
    value = table[key]
    return float(value['value'] if isinstance(value, dict) else value)


@dataclass
class PassValues:
    tokens: torch.Tensor
    fixed: torch.Tensor
    p_top1: torch.Tensor


@reference_op("jacobi_tokens")
def jacobi_tokens(logits, pinned_first=None, pinned_second=None, pins=None):
    """Argmax and token constraints, independent of pass bookkeeping."""
    tokens = logits.argmax(dim=-1)
    if pinned_first is not None:
        tokens[..., 0] = pinned_first
    if pinned_second is not None:
        tokens[..., 1] = pinned_second
    if pins is not None:
        tokens = apply_pins(tokens, pins)
    return tokens


@reference_op("apply_pins")
def apply_pins(tokens, pins):
    if pins[0].ndim == 2:
        return torch.where(pins[0], pins[1], tokens)
    lane = torch.arange(tokens.shape[1], device=tokens.device)
    return torch.where(lane[None, :] == pins[0][:, None], pins[1][:, None], tokens)


@reference_op("jacobi_converged")
def jacobi_converged(tokens, inputs):
    return tokens.eq(inputs).all(dim=-1)


@reference_op("jacobi_pass")
def reference_pass(logits, input_tokens, pinned_first, pinned_second=None, pins=None):
    """Argmax, pins, convergence and float32 softmax in served order."""
    tokens = jacobi_tokens(logits, pinned_first, pinned_second, pins)
    fixed = jacobi_converged(tokens, input_tokens)
    p_top1 = logits.float().softmax(-1).amax(-1)
    return PassValues(tokens, fixed, p_top1)
