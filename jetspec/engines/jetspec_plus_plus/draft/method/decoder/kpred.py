"""Torch features and survival probabilities for the k predictor."""

import torch

from .offers import keep_device

from ..ops import reference_op

@reference_op("probabilities_from_logits")
def probabilities_from_logits(logits):
    """Match the vendored fp32 sigmoid followed by float64 survival hazards."""
    q = torch.sigmoid(keep_device(logits).float()).double()
    survival = torch.cumprod(q, dim=-1)
    before = torch.cat((torch.ones_like(survival[..., :1]), survival[..., :-1]), dim=-1)
    return torch.cat((before * (1.0 - q), survival[..., -1:]), dim=-1)
