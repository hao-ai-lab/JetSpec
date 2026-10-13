"""Ported from the JetSpec++ reference implementation.

Also ports the agreed-run pin helper from the served refresh reference.

Public API:
apply_pins(tokens, pins) -> tokens; pins is (slot, token) [R] or (mask, values) [R,W].
_pin_agreed(state, ids, khat_used) -> newly pinned slot lists; updates state.pins
    (token IDs, -1 means free) and state.agreed (bool mask or None).
"""
import torch

__all__ = ['apply_pins', '_pin_agreed']


def apply_pins(tokens, pins):
    if pins[0].ndim == 2:
        return torch.where(pins[0], pins[1], tokens)
    lane = torch.arange(tokens.shape[1], device=tokens.device)
    return torch.where(lane[None, :] == pins[0][:, None], pins[1][:, None], tokens)


def _pin_agreed(state, ids, khat_used):
    """Pin agreed slots 1..khat_used, preserving existing pins and slot zero."""
    device = state.pins.device
    slots = torch.arange(ids.shape[1], device=device)[None, :]
    agree = (slots >= 1) & (slots <= khat_used.to(device)[:, None]) & (state.pins < 0)
    state.pins[agree] = ids.to(device)[agree]
    state.agreed = agree if state.agreed is None else state.agreed | agree
    return [row.nonzero()[:, 0].tolist() for row in agree]

