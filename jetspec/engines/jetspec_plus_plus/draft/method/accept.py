"""Candidate acceptance math, independent of transport and timing."""
import torch

from .ops import reference_op
from .batching import sequencewise


@reference_op("greedy_accept_device")
@sequencewise('proposals', 3)
@torch.inference_mode()
def greedy_accept_device(logits, proposals):
    """Return accepted draft counts and recovery IDs without a host read."""
    predictions = logits.argmax(dim=-1)
    matching_prefix = predictions[:, :-1].eq(proposals[:, 1:])
    counts = matching_prefix.to(torch.int64).cumprod(dim=1).sum(dim=1)
    recovery = predictions.gather(1, counts[:, None]).squeeze(1)
    return counts, recovery


@reference_op("point_mass_accept_device")
@sequencewise('proposals', 3)
@torch.inference_mode()
def point_mass_accept_device(logits, proposals, temperatures, *, sampler_x=None,
                             async_fan_out=None):
    """Host verify's sampled decision for a greedy draft, without a host read.

    The caller selects this only for positive target temperatures. Keep both
    multinomial draws (residual and unadjusted) in host order so the same torch
    request seed gives the same decisions and advances the same RNG stream.
    """
    width = proposals.shape[1]
    temperature = temperatures[:, None, None].clamp(min=1e-8)
    probs = torch.softmax((logits / temperature).to(torch.float32), dim=-1)
    batch = torch.arange(logits.shape[0], device=logits.device)
    if width == 1:
        counts = torch.zeros(logits.shape[0], dtype=torch.int64, device=logits.device)
        fallback = probs[:, 0] / probs[:, 0].sum(-1, keepdim=True)
        return counts, torch.multinomial(fallback, 1).squeeze(1)

    # Rescaling a point mass leaves it unchanged for positive sampler_x.
    if sampler_x is not None:
        if async_fan_out is None:
            raise ValueError("async_fan_out must be provided with sampler_x")
        if sampler_x <= 0:
            raise ValueError("point-mass device verify requires positive sampler_x")
    p_values = probs[:, :-1].gather(2, proposals[:, 1:, None]).squeeze(2)
    accept_probs = (p_values / (1.0 + 1e-10)).clamp(max=1.0)
    accepts = torch.rand_like(accept_probs) <= accept_probs
    counts = accepts.to(torch.int64).cumprod(1).sum(1)
    selected = probs[batch, counts]
    fallback = selected / selected.sum(1, keepdim=True)
    rejected = proposals[:, 1:].gather(1, counts.clamp(max=width - 2)[:, None])
    residual = selected.clone()
    residual.scatter_(1, rejected, 0.0)
    sums = residual.sum(1, keepdim=True)
    adjusted = torch.where(sums > 0, residual / sums, fallback)
    from_residual = torch.multinomial(adjusted, 1).squeeze(1)
    from_target = torch.multinomial(fallback, 1).squeeze(1)
    recovery = torch.where(counts < width - 1, from_residual, from_target)
    return counts, recovery


@reference_op("greedy_accept")
@sequencewise('proposals', 3)
@torch.inference_mode()
def greedy_accept(logits, proposals, *, return_predictions=False, proposal_rows=None):
    """Native greedy decisions with one packed device-to-host materialization."""
    predictions = logits.argmax(dim=-1)
    if proposal_rows is not None:
        # Proposals already live on the host in the verify sequence. Materialize
        # only predictions, avoiding a chain of tiny acceptance kernels.
        rows = predictions.tolist()
        accepted, recoveries = [], []
        for proposal, row in zip(proposal_rows, rows):
            count = next((i for i, (token, prediction) in
                          enumerate(zip(proposal[1:], row)) if token != prediction),
                         len(proposal) - 1)
            accepted.append(proposal[:count + 1])
            recoveries.append(row[count])
        if return_predictions:
            return accepted, recoveries, rows
        return accepted, recoveries
    mismatch = proposals[:, 1:] != predictions[:, :-1]
    first = mismatch.int().argmax(dim=1)
    counts = torch.where(mismatch.any(dim=1), first, torch.full_like(first, proposals.shape[1] - 1))
    recovery = predictions[torch.arange(proposals.shape[0], device=logits.device), counts]
    packed = torch.cat([proposals, counts[:, None], recovery[:, None]], dim=1)
    if return_predictions:
        width = proposals.shape[1] + 2
        rows = torch.cat([packed, predictions], dim=1).tolist()
        accepted = [row[:row[width - 2] + 1] for row in rows]
        return accepted, [row[width - 1] for row in rows], [row[width:] for row in rows]
    rows = packed.tolist()
    return [row[:row[-2] + 1] for row in rows], [row[-1] for row in rows]
