# Adaptation: package relocation and deferred optional dependencies.
"""Capture every recurrent GDN state in one launch during hybrid verification."""

import torch
from jetspec.engines.jetspec_plus_plus._optional import triton
from jetspec.engines.jetspec_plus_plus._optional import tl


from jetspec.engines.jetspec_plus_plus.kernels.hybrid_snapshot import _snapshot_states


def snapshot_states(k: torch.Tensor, v: torch.Tensor, g: torch.Tensor,
                    beta: torch.Tensor, initial: torch.Tensor,
                    scratch: torch.Tensor) -> None:
    """Write scratch[1:T+1] without mutating the committed initial state."""
    _, width, heads, key_dim = k.shape
    value_dim = v.shape[-1]
    if v.shape[:3] != (1, width, heads) or g.shape != (1, width, heads):
        raise ValueError("GDN snapshot geometry mismatch")
    if beta.shape != g.shape or scratch.shape[0] <= width:
        raise ValueError("GDN snapshot capacity mismatch")
    _snapshot_states[(heads, triton.cdiv(value_dim, 8))](
        k, v, g, beta, initial, scratch,
        *k.stride()[1:], *v.stride()[1:], *g.stride()[1:], *beta.stride()[1:],
        *initial.stride()[1:], scratch.stride(0), *scratch.stride()[2:],
        width, key_dim, value_dim, triton.next_power_of_2(key_dim), 8,
    )
