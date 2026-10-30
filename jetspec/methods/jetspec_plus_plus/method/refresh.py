"""Ported from the JetSpec++ reference implementation.

Public API:
    refresh_cutoff(probabilities, threshold, *, start=1, final=None) -> [rows] cutoff.
    refresh_khat_anchor0(p_drafts, threshold) -> [rows] cutoff (B16, final=15).
    inject(target_hidden, hidden, khat, *, block_size=16) -> (context, positions, mask).

The served p_drafts equal p_top1[:, 1:]: the reference softmaxes
logits[:, 1:], and the recipe's wide anchor0 pass projects all 16 block slots
(the bucketed head pass); the chain saves pass_top1[:, 1:].
Thus the B16 cutoff is refresh_cutoff(p[:, 1:], t, final=15), with start=1.

The recipe's inclusive inject setting makes khat a last-visible index, rather
than an entry count. RefreshSwap.apply returns khat_used+1; its caller subtracts
one in both the served and eager refresh paths. Entries 0..khat_used
are exposed, matching round.py's inject_hidden <= cutoff selection. The served
entries are extra context K/V, rather than replacements for existing context:
block query i reads refreshed entry j only if j < i and j <= khat. The final
block entry has no later query and is never visible.

inject mirrors the reference stale-reader tensor layout and refreshed-row
visibility. Its fixed block
layout takes target_hidden [rows, prefix, features], hidden [rows, block_size,
features], and khat [rows]. positions [rows, prefix+2*block_size] duplicate the
block positions for extra context keys and block queries. mask is additive,
[rows, 1, block_size, prefix+2*block_size]; the head adds block causality.
Each call rebuilds visibility, including when a later cutoff shrinks. The
caller passes these three tensors to draft_forward with no cached K/V.
"""
import torch

__all__ = ['refresh_cutoff', 'refresh_khat_anchor0', 'inject']


def refresh_cutoff(probabilities, threshold, *, start=1, final=None):
    """First slot below the cumulative probability threshold; equality continues."""
    chained = torch.cumprod(probabilities.double(), dim=1)
    below = chained < float(threshold)
    first = below.to(torch.int8).argmax(dim=1) + start
    final = probabilities.shape[1] + start - 1 if final is None else final
    return torch.where(below.any(dim=1), first, torch.full_like(first, final))


def refresh_khat_anchor0(p_drafts, threshold):
    return refresh_cutoff(p_drafts, threshold, final=15)


def inject(target_hidden, hidden, khat, *, block_size=16):
    """Expose refreshed block entries as extra context for the next causal J pass."""
    rows, base, _ = target_hidden.shape
    if hidden.shape != (rows, block_size, target_hidden.shape[-1]):
        raise ValueError('refresh hidden must have shape [rows, block_size, features]')
    if khat.shape != (rows,):
        raise ValueError('one refresh cutoff is required per row')
    slots = torch.arange(block_size, device=target_hidden.device)
    # The recipe projects all eligible rows; the unread final slot is zeroed in project().
    extra = torch.where((slots < block_size - 1)[None, :, None], hidden,
                        torch.zeros_like(hidden))
    context = torch.cat([target_hidden, extra], dim=1)
    position_ids = torch.cat((torch.arange(base + block_size, device=context.device),
                              torch.arange(base, base + block_size, device=context.device)))
    position_ids = position_ids[None].repeat(rows, 1)
    allowed = torch.ones(rows, block_size, base + 2 * block_size,
                         dtype=torch.bool, device=context.device)
    allowed[:, :, base:base + block_size] = (
        (slots[None, :, None] > slots[None, None, :]) &
        (slots[None, None, :] <= khat[:, None, None]))
    mask = torch.zeros((rows, 1, block_size, allowed.shape[-1]), dtype=context.dtype,
                       device=context.device).masked_fill_(~allowed[:, None],
                                                           torch.finfo(context.dtype).min)
    return context, position_ids, mask
