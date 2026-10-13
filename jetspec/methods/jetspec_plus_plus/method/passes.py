"""Ported from the JetSpec++ reference implementation.

Public API:
    DraftForwardPass(logits, input_tokens, hidden=None).
    PassValues(tokens, fixed, p_top1).
    draft_forward(head, embed_tokens, lm_head, ids, target_hidden, *,
                  position_ids=None, attention_mask=None, start=0) -> DraftForwardPass.
    apply_pins(tokens, pins) -> tokens.
    jacobi_tokens(logits, pinned_first=None, pinned_second=None, pins=None) -> tokens.
    jacobi_converged(tokens, inputs) -> [rows] bool.
    reference_pass(logits, input_tokens, pinned_first, pinned_second=None,
                   pins=None) -> PassValues.

The head call mirrors the reference uncached draft pass: no cached
context K/V survive a pass. start=0 projects every block slot for the served
anchor0 layout; start=1 selects the kth reader's slots.
Pins are applied before convergence, while p_top1 reads the original logits.
"""
from dataclasses import dataclass
from typing import NamedTuple

import torch

__all__ = ['DraftForwardPass', 'PassValues', 'draft_forward', 'apply_pins',
           'jacobi_tokens', 'jacobi_converged', 'reference_pass']


class DraftForwardPass(NamedTuple):
    logits: torch.Tensor
    input_tokens: torch.Tensor
    hidden: torch.Tensor | None = None


@dataclass
class PassValues:
    tokens: torch.Tensor
    fixed: torch.Tensor
    p_top1: torch.Tensor


def draft_forward(head, embed_tokens, lm_head, ids, target_hidden, *,
                  position_ids=None, attention_mask=None, start=0):
    """Run one uncached causal head pass; the caller applies pins and convergence."""
    if position_ids is None:
        position_ids = torch.arange(target_hidden.shape[1] + ids.shape[1],
                                    dtype=torch.long, device=ids.device)[None]
        position_ids = position_ids.repeat(ids.shape[0], 1)
    # Lift the reference's forward closure into a callable, with slot selection explicit.
    hidden = head(target_hidden=target_hidden, noise_embedding=embed_tokens(ids),
                  position_ids=position_ids, past_key_values=None, use_cache=False,
                  is_causal=True, attention_backend='sdpa',
                  **({} if attention_mask is None else {'attention_mask': attention_mask}))
    return DraftForwardPass(lm_head(hidden[:, start:, :]), ids[:, start:], hidden)


def apply_pins(tokens, pins):
    if pins[0].ndim == 2:
        return torch.where(pins[0], pins[1], tokens)
    lane = torch.arange(tokens.shape[1], device=tokens.device)
    return torch.where(lane[None, :] == pins[0][:, None], pins[1][:, None], tokens)


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


def jacobi_converged(tokens, inputs):
    return tokens.eq(inputs).all(dim=-1)


def reference_pass(logits, input_tokens, pinned_first, pinned_second=None, pins=None):
    """Argmax, pins, convergence and float32 softmax in served order."""
    tokens = jacobi_tokens(logits, pinned_first, pinned_second, pins)
    fixed = jacobi_converged(tokens, input_tokens)
    p_top1 = logits.float().softmax(-1).amax(-1)
    return PassValues(tokens, fixed, p_top1)
