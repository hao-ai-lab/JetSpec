"""Row staging and compatibility imports for the method cache-row rule."""

import torch

from .ops import reference_op
from .cache_rows import (GRID, DeviceOffers, blend_mass, joint_scores, floor_passing,
                         joint_rule, budget_rule, select_rows, select_rows_batch)


@reference_op("stage_rows_reference")
def stage_rows_reference(score, tau, sizes, miss, ids, *, cap, capacity):
    """The eager selection (_assemble_rows_batch: the joint rule's passing pairs, BatchedRowStage,
    the non-finite score check): score [C, rows, pool] float64, tau [C], sizes [C, rows], miss [C]
    bool, ids [C, rows, >= pool] -> (requested [C], keys [C, capacity, 2], score_bad [C])."""
    count, rows, pool = score.shape
    device = score.device
    valid = torch.arange(pool, device=device)[None, None, :] < sizes[:, :, None]
    valid = valid & (~miss[:, None, None] | (torch.arange(rows, device=device)[None, :, None] == 0))
    passing = (score >= tau[:, None, None]) & (score > 0.0) & valid
    requested, keys = select_rows_batch(score.flatten(1), passing.flatten(1), ids[:, :, :pool],
                                        cap=cap, capacity=capacity)
    scored_row = ~miss[:, None] | (torch.arange(rows, device=device)[None, :] == 0)
    score_bad = (~torch.isfinite(score) & scored_row[:, :, None]).flatten(1).any(1).long()
    return requested, keys, score_bad


@reference_op("row_inputs_reference")
def row_inputs_reference(requested, keys, guesses, source, samples, *, count, padded, H, cap):
    """The eager row inputs (_flat_rows_and_passes_batch's deferred stage, padded to the bucket):
    -> keys [padded, H, 2], drafted row counts [padded], geometry [padded * H], active rows
    [padded * H], anchors [padded * H, 1], first-pass ids [padded * H, B]."""
    keys_device = keys[:, :H].clone()
    private = torch.stack((torch.zeros_like(guesses), guesses), dim=1)
    keys_device[:, 0] = torch.where((requested == 0)[:, None], private, keys_device[:, 0])
    live_counts = requested.clamp(max=cap).clamp(min=1)
    keys_device = keys_device.clamp_min(0)
    if padded > count:
        extra = padded - count
        keys_device = torch.cat((keys_device, keys_device.new_zeros(extra, H, 2)))
        live_counts = torch.cat((live_counts, live_counts.new_zeros(extra)))
    geometry = (keys_device[:, :, 0] + 1).reshape(-1)
    active_rows = (torch.arange(H, device=keys.device)[None, :] < live_counts[:, None]).reshape(-1)
    preceding = source.gather(1, keys_device[:, :, 0]).reshape(-1, 1)
    first = samples[:, None, :].expand(-1, H, -1).clone()
    first[:, :, 0] = keys_device[:, :, 1]
    return keys_device, live_counts, geometry, active_rows, preceding, first.reshape(padded * H, -1)
