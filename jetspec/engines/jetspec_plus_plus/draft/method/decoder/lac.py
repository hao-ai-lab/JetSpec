"""Conformal membership and keyed-row probability math."""


from __future__ import annotations


from dataclasses import dataclass


import torch


from jetspec.engines.jetspec_plus_plus.draft.method.ops import reference_op


@dataclass(frozen=True)
class KeyedRows:
    keys: torch.Tensor
    joint_probabilities: torch.Tensor
    requested: int
    truncated: int


def keep_device(data):
    """Like ``torch.as_tensor`` but a tensor stays where it is: the engine sets a CPU default device, which
    would otherwise copy a CUDA tensor to the host and run the rest of the selection there."""
    return data if isinstance(data, torch.Tensor) else torch.as_tensor(data)


@reference_op("lac_candidates")
def lac_candidates(probabilities, *, q_hat, cap, excluded=()):
    """Return stable probability-ranked members of ``p >= 1 - q_hat``."""
    probabilities = keep_device(probabilities)
    if probabilities.ndim != 1 or not bool(torch.isfinite(probabilities).all()):
        raise ValueError("LAC probabilities must be one finite vector")
    q_hat = float(q_hat)
    cap = int(cap)
    if not 0.0 <= q_hat <= 1.0 or cap < 1:
        raise ValueError("LAC q_hat must be in [0,1] and cap must be positive")
    keep = probabilities >= 1.0 - q_hat
    if excluded:
        rejected = torch.as_tensor(sorted(set(map(int, excluded))), device=probabilities.device)
        if bool(((rejected < 0) | (rejected >= probabilities.numel())).any()):
            raise ValueError("excluded LAC ids are outside the probability vector")
        keep[rejected] = False
    ranked = torch.argsort(probabilities, descending=True, stable=True)
    ranked = ranked[keep.index_select(0, ranked)][:cap]
    return ranked, probabilities.index_select(0, ranked)




def row_bucket(live_count, *, cap, graphed, fixed_graph=None, budget=False):
    """Use a static graph bucket and the smallest power-of-two eager bucket."""
    live_count, cap = int(live_count), int(cap)
    if live_count < 0 or cap < 1 or live_count > cap:
        raise ValueError("live row count must be within the positive row capacity")
    fixed_graph = graphed if fixed_graph is None else fixed_graph
    bucket = cap if fixed_graph else 1 << (max(1, live_count) - 1).bit_length()
    return min(cap, bucket) if budget else bucket
