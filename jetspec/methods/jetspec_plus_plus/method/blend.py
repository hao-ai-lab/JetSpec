"""Ported from the JetSpec++ reference implementation."""
import torch
from .rows import blend_mass


def blend_features(logits, drafts, p_top1, p64, sizes, coefficients):
    """The six decoder-feature terms, in the served order, followed by BLEND mass."""
    a, b, c = (float(coefficients[name]) for name in ('a', 'b', 'c'))
    g, t, p = (float(coefficients[name]) for name in ('argmax', 'top', 'pos'))
    n = 15
    logits = torch.as_tensor(logits).float()
    rejected = torch.cat((drafts.reshape(-1).to(logits.device),
                          drafts.new_full((1,), -1).to(logits.device)))
    p_top1 = torch.as_tensor(p_top1).to(device=logits.device).double().reshape(-1)
    if logits.ndim != 2 or logits.shape[0] != 16 or rejected.shape != (16,) or p_top1.shape != (16,):
        raise ValueError('BLEND requires 16 decoder rows, 15 drafts and 16 probabilities')
    if bool((rejected[:n] < 0).any()) or bool((sizes[:n] < 1).any()):
        raise ValueError('BLEND requires a draft and an anchor offer at every scored offset')
    q = logits.softmax(-1).gather(1, rejected.clamp_min(0)[:, None])[:, 0].double()
    drafter = p_top1[1:n + 1].clamp(1e-6, 1.0 - 1e-6)
    decoder = q[:n].clamp(1e-6, 1.0 - 1e-6)
    z = a * (drafter.log() - (-drafter).log1p()) + b * (decoder.log() - (-decoder).log1p()) + c
    argmax_eq = (logits.argmax(-1)[:n] == rejected[:n]).double()
    valid = torch.arange(p64.shape[1], device=p64.device)[None, :] < sizes[:n, None]
    top = torch.where(valid, p64[:n], float('-inf')).amax(-1).to(device=logits.device)
    top = top.clamp(1e-6, 1.0 - 1e-6)
    position = torch.arange(1, n + 1, device=logits.device, dtype=torch.float64) / 14.0
    z = z + g * argmax_eq + t * (top.log() - (-top).log1p()) + p * position
    # Public offsets are k=0..15; the serving wire reserves an unused leading zero.
    return blend_mass(z)
