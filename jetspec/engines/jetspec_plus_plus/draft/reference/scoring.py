"""Ported from the JetSpec++ reference implementation.

Public API: blend(p_top1, logits, draft_ids, probabilities, sizes, *, coefficients)
-> float64 wire mass[n+2], with P(k) at index k+1 and a leading zero.
The probabilities are post-exclusion anchor offers, padded to [n+1, pool].
"""
import math

import torch

from ..method.cache_rows import blend_mass, blend_features

__all__ = ['blend']

def blend(p_top1, logits, draft_ids, probabilities, sizes, *, coefficients):
    """Score decoder features with the reference's clipping and operation order."""
    # Explicit tensors and coefficients replace the serving block and env config.
    names = ('a', 'b', 'c', 'argmax', 'top', 'pos')
    if set(coefficients) != set(names):
        raise ValueError('BLEND requires exactly a, b, c, argmax, top and pos')
    a, b, c, g, t, p = (float(coefficients[name]) for name in names)
    if not all(math.isfinite(value) for value in (a, b, c, g, t, p)):
        raise ValueError('BLEND coefficients must be finite')
    logits = torch.as_tensor(logits).float()
    drafts = torch.as_tensor(draft_ids, device=logits.device, dtype=torch.long).reshape(-1)
    n = drafts.numel()
    p_top1 = torch.as_tensor(p_top1).to(device=logits.device).double().reshape(-1)
    probabilities = torch.as_tensor(probabilities, device=logits.device).double()
    sizes = torch.as_tensor(sizes, device=logits.device, dtype=torch.long)
    if (n < 1 or logits.ndim != 2 or logits.shape[0] != n + 1
            or p_top1.shape != (n + 1,) or probabilities.ndim != 2
            or probabilities.shape[0] != n + 1 or probabilities.shape[1] < 1
            or sizes.shape != (n + 1,)):
        raise ValueError('BLEND requires n drafts and n+1 aligned decoder, drafter and offer rows')
    if bool(((drafts < 0) | (drafts >= logits.shape[1])).any()):
        raise ValueError('BLEND requires a valid draft at every scored decoder row')
    if bool(((sizes[:n] < 1) | (sizes[:n] > probabilities.shape[1])).any()):
        raise ValueError('BLEND requires an anchor offer at every scored decoder row')
    rejected = torch.cat((drafts, drafts.new_full((1,), -1)))
    q = logits.softmax(-1).gather(1, rejected.clamp_min(0)[:, None])[:, 0].double()
    argmax_eq = (logits.argmax(-1)[:n] == rejected[:n]).double()
    width = probabilities.shape[1]
    valid = torch.arange(width, device=probabilities.device)[None, :] < sizes[:n, None]
    top = torch.where(valid, probabilities[:n], float('-inf')).amax(-1).to(device=logits.device)
    z = blend_features(p_top1[1:n + 1], q[:n], (a, b, c, g, t, p),
                       argmax_eq=argmax_eq, top=top)
    mass = blend_mass(z)
    if not bool(torch.isfinite(mass).all()):
        raise ValueError('BLEND produced a non-finite k score')
    return torch.cat((mass.new_zeros(1), mass))
