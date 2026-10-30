"""Post-swap fill modes, selected by method.explore.after_swap_fill."""
import torch
from jetspec.engines.jetspec_plus_plus.draft.explore import register

def fill_after_swap(kind, ids, slots, applied, *, mask_token, context=None, noise=None,
                    decoder_argmax=None, slot_counts=None, pins=None):
    """Fill unpinned slots after applied swaps in place; return this pass's guard exemptions.

    Decoder row j-1 predicts block slot j. Only covered decoder rows are filled.
    The filled slots remain unpinned and the next drafter pass can replace them.
    """
    lane = torch.arange(ids.shape[1], device=ids.device)[None]
    free = applied[:, None] & (lane > slots[:, None])
    if pins is not None:
        if isinstance(pins, tuple):
            mask = (pins[0] if pins[0].ndim == 2 else lane == pins[0][:, None])
            free &= ~mask
        else:
            free &= pins.to(ids.device) < 0
    if kind == 'decoder':
        top = decoder_argmax.to(ids.device)
        values = torch.cat((top[:, :1], top), dim=1)
        free &= (lane - 1 < slot_counts[:, None]) & (values >= 0)
    elif kind == 'mask':
        values = torch.full_like(ids, mask_token)
    else:
        values = context if kind == 'context' else noise
    ids.copy_(torch.where(free, values, ids))
    return free


def _refresh_swap_state():
    from .refresh import RefreshSwap
    return RefreshSwap

METHOD_KEYS = ("explore.after_swap_fill",)
register(METHOD_KEYS[0], refresh_swap_state=_refresh_swap_state)
