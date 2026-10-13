"""Qualified optimizations shared by eval and fast, with method fallback."""
import torch

from jetspec.engines.jetspec_plus_plus.draft.method.ops import reference; from jetspec.engines.jetspec_plus_plus.draft.method.ops import register_impl
import jetspec.engines.jetspec_plus_plus.draft.method.passes as passes  # Declare the operations before registering implementations.


def scatter_pins(tokens, pins):
    """Avoid constructing a full lane mask for one pinned slot per row."""
    if pins[0].ndim == 2 or (tokens.is_cuda and torch.are_deterministic_algorithms_enabled()):
        return reference('apply_pins')(tokens, pins)
    slots = pins[0].clamp(0, tokens.shape[1] - 1)
    active = (pins[0] >= 0) & (pins[0] < tokens.shape[1])
    values = torch.where(active, pins[1], tokens.gather(1, slots[:, None])[:, 0])
    return tokens.clone().scatter_(1, slots[:, None], values[:, None])


IMPLEMENTATIONS = {'apply_pins': scatter_pins}


def register_optimization(op_name, implementation):
    """An optimized op reaches both graphed tiers; all other ops use the oracle."""
    for engine in ('eval', 'fast'):
        register_impl(op_name, implementation, engine=engine)


def install(op_names):
    for name in op_names:
        register_optimization(name, IMPLEMENTATIONS[name])
