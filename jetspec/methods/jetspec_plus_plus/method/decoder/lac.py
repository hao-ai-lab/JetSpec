"""Ported from the JetSpec++ reference implementation."""
import torch
from ..ops import reference_op

def keep_device(data):
    """Like ``torch.as_tensor`` but a tensor stays where it is: the engine sets a CPU default device, which
    would otherwise copy a CUDA tensor to the host and run the rest of the selection there."""
    return data if isinstance(data, torch.Tensor) else torch.as_tensor(data)

@reference_op('lac_candidates')
def lac_candidates(probabilities, *, q_hat, cap, excluded=()):
    """Return stable probability-ranked members of ``p >= 1 - q_hat``."""
    probabilities = keep_device(probabilities)
    if probabilities.ndim != 1 or not bool(torch.isfinite(probabilities).all()):
        raise ValueError('LAC probabilities must be one finite vector')
    q_hat = float(q_hat)
    cap = int(cap)
    if not 0.0 <= q_hat <= 1.0 or cap < 1:
        raise ValueError('LAC q_hat must be in [0,1] and cap must be positive')
    keep = probabilities >= 1.0 - q_hat
    if excluded:
        rejected = torch.as_tensor(sorted(set(map(int, excluded))), device=probabilities.device)
        if bool(((rejected < 0) | (rejected >= probabilities.numel())).any()):
            raise ValueError('excluded LAC ids are outside the probability vector')
        keep[rejected] = False
    ranked = torch.argsort(probabilities, descending=True, stable=True)
    ranked = ranked[keep.index_select(0, ranked)][:cap]
    return (ranked, probabilities.index_select(0, ranked))
