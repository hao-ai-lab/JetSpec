"""Stable keyed-row scoring and selection, independent of graph capture."""

from dataclasses import dataclass
from types import SimpleNamespace

import torch

from .ops import reference_op
from .batching import sequence_slice, sequencewise

GRID = 2.0 ** 32


@reference_op("blend_mass")
def blend_mass(logits, *, sigmoid=False):
    """Quantized offset mass; sigmoid retains calibration replay's hazard rounding."""
    hazards = torch.sigmoid(logits) if sigmoid else 1.0 / (1.0 + torch.exp(-logits))
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
@sequencewise('score', 2)
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
@sequencewise('score', 2)
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


@reference_op("select_rows_batch")
def select_rows_batch(score, passing, ids, *, cap, capacity=None):
    if score.ndim != 2 or score.shape != passing.shape or ids.shape[:2] != (score.shape[0], score.shape[1] // ids.shape[-1]):
        raise ValueError('batched row scores and offers must align')
    if cap < 1:
        raise ValueError('the row cap must be positive')
    capacity = max(cap, int(capacity or cap))
    count, width = score.shape
    order = torch.argsort(score, dim=1, descending=True, stable=True)
    chosen = passing.gather(1, order)
    position = chosen.long().cumsum(1) - 1
    requested = chosen.long().sum(1)
    slot = torch.where(chosen & (position < cap), position, capacity)
    k = torch.arange(ids.shape[1], device=ids.device).repeat_interleave(ids.shape[2])
    pairs = torch.stack((k.expand(count, -1), ids.reshape(count, width)), dim=-1)
    ranked = pairs.gather(1, order[..., None].expand(-1, -1, 2))
    keys = torch.full((count, capacity + 1, 2), -1, device=ids.device, dtype=torch.long)
    keys.scatter_(1, slot[..., None].expand(-1, -1, 2), ranked)
    keys[:, capacity] = -1
    return requested, keys[:, :capacity]


EPS = 1e-6
POSITION_SCALE = 14.0


@reference_op("blend_features")
def blend_features(drafter, decoder, coefficients, *, argmax_eq=None, top=None,
                   position=None, multiply_position_first=False):
    """BLEND logits in float64, with optional decoder terms in their served order.

    Inputs are the hazard slots (p_top1[1:], q[:-1]). Calibration replay retains
    its historical coefficient-times-position-before-division rounding.
    """
    drafter = drafter.double().clamp(EPS, 1.0 - EPS)
    decoder = decoder.double().clamp(EPS, 1.0 - EPS)
    a, b, c = coefficients[:3]
    z = a * (drafter.log() - (-drafter).log1p()) + b * (
        decoder.log() - (-decoder).log1p()) + c
    if argmax_eq is not None:
        g, t, p = coefficients[3:]
        top = top.double().clamp(EPS, 1.0 - EPS)
        z = z + g * argmax_eq + t * (top.log() - (-top).log1p())
        if position is None:
            position = torch.arange(1, drafter.shape[-1] + 1, device=drafter.device,
                                    dtype=torch.float64)
            term = (p * position / POSITION_SCALE if multiply_position_first else
                    p * (position / POSITION_SCALE))
        else:
            term = p * position
        z = z + term
    return z


@reference_op("hit_lookup")
@sequencewise('keys', 3, shared=('index',))
def hit_lookup(keys, true_k, correction, *, pins=None, index=None, legacy=False):
    """First surviving exact row, or None; host indexes share the same key rule."""
    key = (int(true_k), int(correction))
    if index is not None:
        return index.get(key)
    if legacy:
        if not 0 <= true_k < len(pins):
            return None
        return next((i for i, pin in enumerate(pins[true_k]) if pin == correction), None)
    if keys is None:
        matches = ((pins >= 0) & (pins == key[1])).nonzero()
        return int(matches[0, 0]) if matches.numel() else None
    if isinstance(keys, (set, frozenset)):
        return key if key in keys else None
    if isinstance(keys, torch.Tensor) and keys.device.type != "cpu":
        matching = (keys[:, 0] == key[0]) & (keys[:, 1] == key[1])
        if pins is not None:
            matching = matching & (pins >= 0)
        matches = matching.nonzero()
        return int(matches[0, 0]) if matches.numel() else None
    keys = keys.tolist() if isinstance(keys, torch.Tensor) else keys
    pins = pins.tolist() if isinstance(pins, torch.Tensor) else pins
    return next((i for i, row in enumerate(keys)
                 if tuple(row) == key and (pins is None or pins[i] >= 0)), None)


def after_miss(snapshot, *, round_id, true_k, recovery, eos=False, snapshot_error=RuntimeError):
    """The admission, terminal, negative-key and redraft cache-miss rule."""
    if isinstance(true_k, torch.Tensor) and true_k.ndim == 1:
        count = true_k.shape[0]
        results = []
        for c in range(count):
            single = None if snapshot is None else SimpleNamespace(
                round_id=sequence_slice(snapshot.round_id, c, count),
                live_keys=sequence_slice(snapshot.live_keys, c, count),
                legacy_pins=sequence_slice(snapshot.legacy_pins, c, count),
                redraft=sequence_slice(snapshot.redraft, c, count))
            results.append(after_miss(single, round_id=sequence_slice(round_id, c, count),
                                      true_k=true_k[c], recovery=sequence_slice(recovery, c, count),
                                      eos=sequence_slice(eos, c, count), snapshot_error=snapshot_error))
        return true_k.new_tensor(results)
    round_id, true_k, recovery = int(round_id), int(true_k), int(recovery)
    if round_id == 0 or eos:
        return 0
    if true_k < 0 or recovery < 0:
        return 1
    if snapshot is None or snapshot.round_id != round_id:
        raise snapshot_error(
            f"chain key snapshot round {None if snapshot is None else snapshot.round_id} "
            f"differs from command round {round_id}")
    hit = hit_lookup(snapshot.live_keys, true_k, recovery,
                     pins=snapshot.legacy_pins, legacy=snapshot.legacy_pins is not None) is not None
    if snapshot.redraft and not hit:
        raise RuntimeError("chain redraft delivery lacks its exact key")
    return int(not hit)


@reference_op("cache_rows_chain")
@sequencewise('offers_ids', 3, shared=('tau_table', 'coefficients'))
def cache_rows_chain(draft_ids, offers_ids, probabilities, pack, logits, p_top1, R, *,
                     coefficients, miss, tau_idx, correction, tau_table, temperature,
                     out, previous_mass16=None, floor=0, fill_zero=False, control=None):
    """Fixed-shape cache-row rule and packed checks; oracle for the K3 override.

    The optional preallocated control packet lets that override keep its wire
    loads; the reference uses the explicit miss, tau index and correction.
    """
    lanes, pool = offers_ids.shape[0], 32
    n, device = lanes - 1, logits.device
    drafts = draft_ids.to(device=device, dtype=torch.long).reshape(-1)
    rejected = torch.cat((drafts, drafts.new_full((1,), -1)))
    p_top1 = torch.as_tensor(p_top1).to(device=device).double().reshape(-1)
    q = logits.float().softmax(-1).gather(1, rejected.clamp_min(0)[:, None])[:, 0].double()
    eq = top = None
    missing_offer = torch.zeros((), dtype=torch.bool, device=device)
    if len(coefficients) == 6:
        eq = (logits.argmax(-1)[:n] == rejected[:n]).double()
        valid = torch.arange(probabilities.shape[1], device=device)[None, :] < pack[0, :n, None]
        top = torch.where(valid, probabilities[:n], float('-inf')).amax(-1)
        missing_offer = (pack[0, :n] < 1).any()
    mass = blend_mass(blend_features(p_top1[1:], q[:n], coefficients, argmax_eq=eq, top=top))
    mass16 = torch.cat((mass.new_zeros(1), mass))
    mass_home = (torch.where(miss.bool(), previous_mass16, mass16)
                 if previous_mass16 is not None else mass16)
    row = torch.arange(lanes, device=device)
    offers = DeviceOffers(offers_ids[:, :pool], probabilities[:, :pool],
                          torch.arange(pool, device=device)[None, :] < pack[0, :, None], row)
    scoring_mass = torch.where(miss.bool(), (row == 0).to(mass.dtype), mass)
    score = joint_scores(scoring_mass, offers.probabilities, temperature).reshape(-1)
    tau = tau_table.index_select(0, tau_idx.reshape(1))[0]
    valid = offers.valid & (~miss.bool() | (row[:, None] == 0))
    passing = (score >= tau) & (score > 0.) & valid.reshape(-1)
    passing = floor_passing(score, passing, valid, floor, fill_zero=fill_zero)
    requested, keys = select_rows(score, passing, offers, R, R)
    active = torch.arange(R, device=device) < requested.clamp(min=1, max=R)
    private = torch.stack((correction.new_zeros(()), correction))
    keys = torch.cat((torch.where(requested > 0, keys[0], private)[None], keys[1:]), dim=0)
    scored = ~miss.bool() | (row == 0)
    flags = torch.stack((
        (~pack[3].bool()).any(), pack[4].bool().any(),
        (rejected[:n] < 0).any() & ~miss.bool(), missing_offer & ~miss.bool(),
        (~torch.isfinite(mass16)).any() & ~miss.bool(),
        ((pack[0] < pool) & scored).any(),
        ((~torch.isfinite(score.reshape(lanes, pool))).any(1) & scored).any(),
    )).long()
    err = (flags * (1 << torch.arange(flags.numel(), device=device))).sum().reshape(1)
    requested = requested.reshape(1)
    home = torch.cat((requested, keys.reshape(-1), pack.reshape(-1), mass_home.view(torch.int64), err))
    for name, value in (('keys', keys), ('active', active), ('requested', requested),
                        ('row_home', home), ('err', err)):
        out[name].copy_(value)
    return out['keys'], out['active'], out['requested'], out['row_home'], out['err']


@reference_op("blend_rows_batch")
def blend_rows_batch(inputs, miss, sizes):
    """e04_kscore_blend_batch_device after its vocabulary reductions, the after-miss overrides
    and the joint score, in the eager path's torch operations -> (mass16, checks, score)."""
    C, n = inputs.count, inputs.n
    # e04_kscore_blend_batch_device
    missing_draft = (inputs.rejected[:, :n] < 0).any(1)
    drafter = inputs.p_top1[:, 1:n + 1].clamp(EPS, 1.0 - EPS)
    decoder = inputs.q[:, :n].clamp(EPS, 1.0 - EPS)
    argmax_eq = top = None
    missing_offer = inputs.no_offer
    if inputs.decoder_features:
        missing_offer = (sizes[:, :n] < 1).any(1)
        argmax_eq = (inputs.argmax[:, :n] == inputs.rejected[:, :n]).double()
        valid = inputs.offer_index[None, None, :] < sizes[:, :n, None]
        top = torch.where(valid, inputs.p64[:, :n], float('-inf')).amax(-1)
        top = top.clamp(EPS, 1.0 - EPS)
    z = blend_features(drafter, decoder, inputs.coefficients,
                       argmax_eq=argmax_eq, top=top, position=inputs.position)
    mass = blend_mass(z)
    mass16 = torch.cat((inputs.zeros1, mass), dim=1)
    checks = torch.stack((missing_draft, missing_offer), dim=1).long()
    scoring_mass = torch.where(miss[:, None], inputs.after_miss_mass, mass)
    # _assemble_rows_batch: an after-miss round keeps its saved mass (a no-op without misses)
    mass16 = torch.where(miss[:, None], inputs.saved, mass16)
    checks = torch.where(miss[:, None], torch.zeros_like(checks), checks)
    score = joint_scores(scoring_mass, inputs.p64[:, :, :inputs.pool], inputs.temperature)
    return mass16, checks, score
