"""Ported from the JetSpec++ reference implementation.

Public API: greedy_accept_device(logits, proposals) -> (counts, recovery);
greedy_accept(logits, proposals, *, return_predictions=False,
proposal_rows=None) -> (accepted, recoveries[, predictions]) as host lists.
Proposals include the anchor in slot zero; counts exclude that anchor.
"""
import torch

__all__ = ['greedy_accept_device', 'greedy_accept']


# Use the torch references directly; serving op dispatch is outside this layer.
@torch.inference_mode()
def greedy_accept_device(logits, proposals):
    """Return accepted draft counts and recovery IDs without a host read."""
    predictions = logits.argmax(dim=-1)
    matching_prefix = predictions[:, :-1].eq(proposals[:, 1:])
    counts = matching_prefix.to(torch.int64).cumprod(dim=1).sum(dim=1)
    recovery = predictions.gather(1, counts[:, None]).squeeze(1)
    return counts, recovery


@torch.inference_mode()
def greedy_accept(logits, proposals, *, return_predictions=False, proposal_rows=None):
    """Native greedy decisions with one packed device-to-host materialization."""
    predictions = logits.argmax(dim=-1)
    if proposal_rows is not None:
        # Proposals already live on the host in the verify sequence.
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
