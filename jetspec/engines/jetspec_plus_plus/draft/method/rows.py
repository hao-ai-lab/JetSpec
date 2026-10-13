"""Stable keyed-row scoring and selection, independent of graph capture."""

from dataclasses import dataclass

import torch

from jetspec.engines.jetspec_plus_plus.draft.method.ops import reference_op

GRID = 2.0 ** 32


@reference_op("blend_mass")
def blend_mass(logits):
    """Convert BLEND hazard logits to the quantized per-offset mass."""
    hazards = 1.0 / (1.0 + torch.exp(-logits))
    survival = torch.cat((hazards.new_ones(hazards.shape[:-1] + (1,)),
                          hazards.cumprod(-1)), dim=-1)
    mass = torch.cat((survival[..., :-1] * (1.0 - hazards),
                      survival[..., -1:]), dim=-1)
    return torch.round(mass * GRID) / GRID


@reference_op("joint_scores")
def joint_scores(mass, probabilities, temperature):
    scaled = probabilities.clamp_min(1e-300).pow(1.0 / temperature)
    anchor = scaled / scaled.sum(-1, keepdim=True).clamp_min(1e-300)
    return torch.round(mass[..., None] * anchor * GRID) / GRID

@dataclass
class DeviceOffers:
    """Anchor offers of the scored decoder rows, on the device: row r offers
    ``ids[r, :sizes[r]]`` with ``probabilities[r, :sizes[r]]`` (``valid`` marks them)."""
    ids: torch.Tensor            # [rows, width] long
    probabilities: torch.Tensor  # [rows, width] float64
    valid: torch.Tensor          # [rows, width] bool
    ks: torch.Tensor             # [rows] long, the wire k of each row

@reference_op("rows_floor")
def floor_passing(score, passing, valid, n, *, fill_zero=False):
    """Add the first n valid positive pairs in the joint rule's stable order."""
    if not n:
        return passing
    order = torch.argsort(score, descending=True, stable=True)
    positive = ((score > 0) | fill_zero) & valid.reshape(-1)
    ranked = positive.index_select(0, order)
    added = ranked & (ranked.long().cumsum(0) <= n)
    floor = torch.zeros_like(passing).scatter(0, order, added)
    return passing | floor

def joint_rule(tau, temperature, *, floor=0, fill_zero=False):
    """E04_ROWRULE=joint: s = P(k) P_T(a | k) on the 2^-32 grid, P_T = p^(1/T) renormalized in
    the row's pool; pairs with s >= tau and s > 0 pass (live_rows.e04_rowrule_joint)."""
    tau, temperature = float(tau), float(temperature)
    if not 0.0 <= tau <= 1.0 or not (0.0 < temperature < float('inf')):
        raise ValueError('the joint row rule requires tau in [0, 1] and a finite positive temperature')

    def rule(mass, offers):
        score = joint_scores(mass, offers.probabilities, temperature).reshape(-1)
        passing = (score >= tau) & (score > 0.0) & offers.valid.reshape(-1)
        return score, floor_passing(score, passing, offers.valid, floor, fill_zero=fill_zero)
    return rule

def budget_rule(a=1.0, b=1.0, length_weight=False):
    """lac.keyed_rows: s = P(k)^a p^b (times k + 1 with the length weight); every offered pair
    passes and the stage keeps the top ``cap``."""
    a, b = float(a), float(b)

    def rule(mass, offers):
        pk = mass if a == 1 else mass.pow(a)
        if length_weight:
            pk = pk * (offers.ks + 1)
        p = offers.probabilities if b == 1 else offers.probabilities.pow(b)
        score = (pk[:, None] * p).reshape(-1)
        return score, offers.valid.reshape(-1).clone()
    return rule

@reference_op("select_rows")
def select_rows(score, passing, offers, cap, capacity):
    device = score.device
    count = score.numel()
    order = torch.argsort(score, descending=True, stable=True)
    chosen = passing.index_select(0, order)
    position = chosen.long().cumsum(0) - 1           # rank among the passing pairs
    requested = chosen.long().sum()
    keep = chosen & (position < cap)
    slot = torch.where(keep, position, torch.full_like(position, capacity))  # dump slot
    width = offers.ids.shape[1]
    k_of = offers.ks.repeat_interleave(width)
    pairs = torch.stack((k_of, offers.ids.reshape(-1)), dim=1).index_select(0, order)
    keys = torch.full((capacity + 1, 2), -1, dtype=torch.long, device=device)
    if count:
        # every kept pair has its own slot; the repeated dump slot is overwritten below
        keys.index_copy_(0, slot, pairs)
        keys[capacity].fill_(-1)
    return requested, keys[:capacity]






