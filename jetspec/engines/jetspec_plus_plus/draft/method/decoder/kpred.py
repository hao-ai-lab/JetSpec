"""Torch features and survival probabilities for the k predictor."""

import torch

from jetspec.engines.jetspec_plus_plus.draft.method.decoder.lac import keep_device

from jetspec.engines.jetspec_plus_plus.draft.method.ops import reference_op

@reference_op("assemble_fr_features")
def assemble_fr_features(hidden, p_top1, stable, outputs, embedding, *, scales):
    """Assemble FR groups exactly as ``pred/features.py`` registers them."""
    device = embedding.weight.device
    hidden = keep_device(hidden).to(device=device)
    p_top1 = keep_device(p_top1).to(device=device, dtype=torch.float32)
    stable = keep_device(stable).to(device=device, dtype=torch.float32)
    outputs = keep_device(outputs).to(device=device, dtype=torch.long)
    if hidden.ndim != 2 or p_top1.shape != hidden.shape[:1] or stable.shape != hidden.shape[:1]:
        raise ValueError("predictor hidden, p_top1, and stable inputs must share 15 slots")
    if outputs.shape != hidden.shape[:1]:
        raise ValueError("predictor output ids must have one token per slot")
    required = {"s", "p_top1", "stable", "emb"}
    if set(scales) != required or any(float(scales[key]) <= 0 for key in required):
        raise ValueError("predictor scales must contain positive FR group RMS values")
    embedded = embedding(outputs).float()
    features = torch.cat(
        (
            hidden.float() / float(scales["s"]),
            (p_top1 / float(scales["p_top1"]))[:, None],
            (stable / float(scales["stable"]))[:, None],
            embedded / float(scales["emb"]),
        ),
        dim=-1,
    )
    floor = torch.logit(p_top1.double().clamp(1e-6, 1 - 1e-6)).float()
    return features, floor

@reference_op("probabilities_from_logits")
def probabilities_from_logits(logits):
    """Match the vendored fp32 sigmoid followed by float64 survival hazards."""
    q = torch.sigmoid(keep_device(logits).float()).double()
    survival = torch.cumprod(q, dim=-1)
    before = torch.cat((torch.ones_like(survival[..., :1]), survival[..., :-1]), dim=-1)
    return torch.cat((before * (1.0 - q), survival[..., -1:]), dim=-1)
