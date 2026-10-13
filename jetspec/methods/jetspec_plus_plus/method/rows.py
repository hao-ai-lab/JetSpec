"""Ported from the JetSpec++ reference implementation.

Public API: DeviceOffers(ids, probabilities, valid, ks);
blend_mass(logits); joint_scores(mass, probabilities, temperature);
floor_passing(score, passing, valid, n, *, fill_zero=False);
joint_rule(tau, temperature, *, floor=8, fill_zero=False) -> scoring callable;
select_rows(score, passing, offers, cap, capacity) -> (requested, keys).
Keys are (k, anchor) pairs with unused capacity padded by (-1, -1).
"""
from dataclasses import dataclass

import torch

from .kmass import GRID, blend_mass

__all__ = ['DeviceOffers', 'blend_mass', 'joint_scores', 'floor_passing',
           'joint_rule', 'select_rows']


def joint_scores(mass, probabilities, temperature):
    # Use torch references directly; serving op dispatch is outside this layer.
    scaled = probabilities.clamp_min(1e-300).pow(1.0 / temperature)
    anchor = scaled / scaled.sum(-1, keepdim=True).clamp_min(1e-300)
    return torch.round(mass[..., None] * anchor * GRID) / GRID


@dataclass
class DeviceOffers:
    """Padded offers; valid marks live (k, anchor) pairs."""
    ids: torch.Tensor            # [rows, width] long
    probabilities: torch.Tensor  # [rows, width] float64
    valid: torch.Tensor          # [rows, width] bool
    ks: torch.Tensor             # [rows] long


def floor_passing(score, passing, valid, n, *, fill_zero=False):
    """Add the first n valid positive pairs in stable score order."""
    if not n:
        return passing
    order = torch.argsort(score, descending=True, stable=True)
    positive = ((score > 0) | fill_zero) & valid.reshape(-1)
    ranked = positive.index_select(0, order)
    added = ranked & (ranked.long().cumsum(0) <= n)
    floor = torch.zeros_like(passing).scatter(0, order, added)
    return passing | floor


# The evaluated policy requires eight rows; the reference leaf defaults to zero.
def joint_rule(tau, temperature, *, floor=8, fill_zero=False):
    """Quantized P(k) P_T(anchor | k), with a stable positive-pair floor."""
    tau, temperature = float(tau), float(temperature)
    if not 0.0 <= tau <= 1.0 or not (0.0 < temperature < float('inf')):
        raise ValueError('the joint row rule requires tau in [0, 1] and a finite positive temperature')

    def rule(mass, offers):
        score = joint_scores(mass, offers.probabilities, temperature).reshape(-1)
        passing = (score >= tau) & (score > 0.0) & offers.valid.reshape(-1)
        return score, floor_passing(score, passing, offers.valid, floor, fill_zero=fill_zero)
    return rule


def select_rows(score, passing, offers, cap, capacity):
    """Count all passing pairs and retain up to cap in stable score order."""
    device = score.device
    count = score.numel()
    order = torch.argsort(score, descending=True, stable=True)
    chosen = passing.index_select(0, order)
    position = chosen.long().cumsum(0) - 1
    requested = chosen.long().sum()
    keep = chosen & (position < cap)
    slot = torch.where(keep, position, torch.full_like(position, capacity))
    width = offers.ids.shape[1]
    k_of = offers.ks.repeat_interleave(width)
    pairs = torch.stack((k_of, offers.ids.reshape(-1)), dim=1).index_select(0, order)
    keys = torch.full((capacity + 1, 2), -1, dtype=torch.long, device=device)
    if count:
        # Kept pairs have unique slots; clear the shared dump slot afterwards.
        keys.index_copy_(0, slot, pairs)
        keys[capacity].fill_(-1)
    return requested, keys[:capacity]
