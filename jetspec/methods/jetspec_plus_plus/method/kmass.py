"""Ported from the JetSpec++ reference implementation.

Public API: blend_mass(logits) -> mass[..., n+1];
blend(p_top1, logits, draft_ids, probabilities, sizes, *, coefficients)
-> float64 wire mass[n+2], with P(k) at index k+1 and a leading zero.
The probabilities are post-exclusion anchor offers, padded to [n+1, pool].
"""
import math

import torch

__all__ = ['blend_mass', 'blend']

GRID = 2.0 ** 32
EPS = 1e-6
POSITION_SCALE = 14.0


def blend_mass(logits):
    """Convert BLEND hazard logits to the quantized per-offset mass."""
    # Use the torch reference directly; serving op dispatch is outside this layer.
    hazards = 1.0 / (1.0 + torch.exp(-logits))
    survival = torch.cat((hazards.new_ones(hazards.shape[:-1] + (1,)),
                          hazards.cumprod(-1)), dim=-1)
    mass = torch.cat((survival[..., :-1] * (1.0 - hazards),
                      survival[..., -1:]), dim=-1)
    return torch.round(mass * GRID) / GRID


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
    drafter = p_top1[1:n + 1].clamp(EPS, 1.0 - EPS)
    decoder = q[:n].clamp(EPS, 1.0 - EPS)
    z = a * (drafter.log() - (-drafter).log1p()) + b * (decoder.log() - (-decoder).log1p()) + c
    argmax_eq = (logits.argmax(-1)[:n] == rejected[:n]).double()
    width = probabilities.shape[1]
    valid = torch.arange(width, device=probabilities.device)[None, :] < sizes[:n, None]
    top = torch.where(valid, probabilities[:n], float('-inf')).amax(-1).to(device=logits.device)
    top = top.clamp(EPS, 1.0 - EPS)
    position = torch.arange(1, n + 1, device=logits.device, dtype=torch.float64) / POSITION_SCALE
    z = z + g * argmax_eq + t * (top.log() - (-top).log1p()) + p * position
    mass = blend_mass(z)
    if not bool(torch.isfinite(mass).all()):
        raise ValueError('BLEND produced a non-finite k score')
    return torch.cat((mass.new_zeros(1), mass))
