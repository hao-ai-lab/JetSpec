"""Stable front metadata for intake and the first adapter embedding norm."""
from types import SimpleNamespace

import torch


def prepare(front, weight, capacity):
    """Bind per-key pointers on the private ID storage, not shared window storage."""
    slots = torch.empty(front.lanes, device=weight.device, dtype=torch.int64)
    recipe = SimpleNamespace(weight=weight, ids=front.c1_query_ids,
        positions=front.prefix_positions, slots=slots, query_slots=torch.empty_like(slots),
        mask=front.c1_query_embeddings.new_empty(1, 1, front.lanes, front.c1_bucket),
        capacity=capacity, neg=float(torch.finfo(front.c1_query_embeddings.dtype).min))
    front.c1_query_ids.untyped_storage()._w4a8_front_norm = recipe
    return recipe


def update_pending(front, pending):
    """Miss redrafting replaces pending IDs after intake has prepared the front."""
    if getattr(front.c1_query_ids.untyped_storage(), '_w4a8_front_norm', None) is not None:
        front.c1_query_ids.reshape(-1)[1:].copy_(pending)
