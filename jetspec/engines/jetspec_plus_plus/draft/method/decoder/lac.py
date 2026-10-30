"""Conformal membership and keyed-row probability math."""


from __future__ import annotations


from dataclasses import dataclass


import torch


from ..ops import reference_op


@dataclass(frozen=True)
class KeyedRows:
    keys: torch.Tensor
    joint_probabilities: torch.Tensor
    requested: int
    truncated: int








@reference_op("keyed_rows")
def keyed_rows(k_ids, k_probabilities, anchor_ids, anchor_probabilities, *, cap, rank_parameters=(1.0, 1.0, False)):
    """Rank ragged ``(k, anchor)`` rows by joint probability, stably."""
    k_ids = torch.as_tensor(k_ids, dtype=torch.long)
    k_probabilities = torch.as_tensor(k_probabilities, dtype=torch.float64, device=k_ids.device)
    if k_ids.ndim != 1 or k_probabilities.shape != k_ids.shape:
        raise ValueError("k ids and probabilities must be equal one-dimensional vectors")
    a, b, length_weight = rank_parameters
    pairs, scores = [], []
    for k, pk in zip(k_ids.tolist(), k_probabilities):
        ids = torch.as_tensor(anchor_ids[int(k)], dtype=torch.long, device=k_ids.device)
        probs = torch.as_tensor(anchor_probabilities[int(k)], dtype=torch.float64, device=k_ids.device)
        if ids.ndim != 1 or probs.shape != ids.shape:
            raise ValueError("each anchor id/probability offer must be one-dimensional")
        pairs.extend((int(k), int(anchor)) for anchor in ids.tolist())
        pk = pk if a == 1 else pk.pow(a)
        probs = probs if b == 1 else probs.pow(b)
        if length_weight:
            pk = pk * (int(k) + 1)
        scores.extend(pk * probability for probability in probs)
    requested = len(pairs)
    if requested:
        score_tensor = torch.stack(scores)
        order = torch.argsort(score_tensor, descending=True, stable=True)[: int(cap)]
        key_tensor = torch.tensor(pairs, dtype=torch.long, device=k_ids.device).index_select(0, order)
        score_tensor = score_tensor.index_select(0, order)
    else:
        key_tensor = torch.empty((0, 2), dtype=torch.long, device=k_ids.device)
        score_tensor = torch.empty((0,), dtype=torch.float64, device=k_ids.device)
    return KeyedRows(
        keys=key_tensor,
        joint_probabilities=score_tensor,
        requested=requested,
        truncated=max(0, requested - int(cap)),
    )


def row_bucket(live_count, *, cap, graphed, fixed_graph=None, budget=False):
    """Use a static graph bucket and the smallest power-of-two eager bucket."""
    live_count, cap = int(live_count), int(cap)
    if live_count < 0 or cap < 1 or live_count > cap:
        raise ValueError("live row count must be within the positive row capacity")
    fixed_graph = graphed if fixed_graph is None else fixed_graph
    bucket = cap if fixed_graph else 1 << (max(1, live_count) - 1).bit_length()
    return min(cap, bucket) if budget else bucket
