"""Refresh cutoffs, agreement, swap decisions and visibility math."""

import torch

from .ops import reference_op
from .batching import sequencewise



@reference_op("refresh_cutoff")
@sequencewise('probabilities', 3)
def refresh_cutoff(probabilities, threshold, *, start=1, final=None, chained=False):
    """First slot below threshold; ``chained`` consumes already rounded survival values.

    Equality continues, and raw probabilities accumulate in double precision.
    """
    survival = probabilities if chained else torch.cumprod(probabilities.double(), dim=1)
    below = survival < float(threshold)
    first = below.to(torch.int8).argmax(dim=1) + start
    final = probabilities.shape[1] + start - 1 if final is None else final
    return torch.where(below.any(dim=1), first, torch.full_like(first, final))


def refresh_khat_threshold(again=False, *, threshold=.1, second_threshold=None):
    value = float(second_threshold if again and second_threshold is not None else threshold)
    if not 0.0 < value < 1.0:
        raise ValueError('refresh threshold must be in (0, 1)')
    return value


def refresh_khat_cap(again=False, *, second_cap=None):
    if not again or second_cap is None:
        return None
    cap = int(second_cap)
    if not 0 <= cap <= 15:
        raise ValueError('second refresh cap must be in 0..15')
    return cap


@reference_op("refresh_bookkeeping")
@sequencewise('chained', 3)
def refresh_bookkeeping(chained, active, keys, length, window, threshold, cap):
    """Select the refresh prefix from already rounded double survival probabilities."""
    khat = refresh_cutoff(chained, threshold, chained=True)
    if cap is not None:
        khat = khat.clamp(max=cap)
    khat = torch.where(active, khat, torch.zeros_like(khat))
    counts = torch.where(active, khat.clamp(max=chained.shape[1] - 1) + 1, torch.zeros_like(khat))
    return khat, counts, keys[:, 0] + 1, length + window


@reference_op("rule_f")
@sequencewise('ids', 3)
def rule_f(decoder_argmax, ids, slot_counts):
    """Spec rule F: ``agree[r, j] = decoder_argmax[r, j-1] == ids[r, j]`` for j = 1..n_r-1
    (n_r = slot_counts[r], the decoder rows computed); F[r] = the number of leading agreements,
    0..n_r-1.  ``decoder_argmax`` [R, >= W-1] holds the argmax of block slot j's decoder row (the
    prediction for slot j + 1)."""
    width = ids.shape[1]
    slot_counts = torch.as_tensor(slot_counts, dtype=torch.long, device=ids.device).reshape(-1, 1)
    agree = decoder_argmax[:, :width - 1].to(ids.device) == ids[:, 1:]
    valid = torch.arange(1, width, device=ids.device)[None, :] < slot_counts
    return torch.cumprod((agree & valid).long(), dim=1).sum(dim=1)


@reference_op("margin_swap")
@sequencewise('logits', 3)
def margin_swap(logits, current, margin, *, exclude_current):
    """Per row: the decoder's top-1 (``exclude_current``: the best token other than
    ``current``) and whether p(top-1) - p(current) >= margin under the row's softmax."""
    probabilities = torch.softmax(logits.float(), dim=-1)
    scores = logits.float().clone()
    if exclude_current:
        scores.scatter_(1, current[:, None], float('-inf'))
    new = scores.argmax(dim=-1)
    gain = probabilities.gather(1, new[:, None])[:, 0] - probabilities.gather(1, current[:, None])[:, 0]
    return new, (gain >= margin) & (new != current)


@reference_op("swap_decision")
@sequencewise('ids', 3)
def swap_decision(logits, ids, khat, slot_counts, active, pin_slot=None):
    """The cutoff and the swap from the refresh-context decoder logits, on device.

    ``logits`` [R, W-1, V]: decoder row j of row r (the distribution for block slot j + 1), valid
    for j < slot_counts[r]; ``ids`` [R, W] the block (the next pass's input); ``khat`` [R] the
    draft cutoff.  Returns ``(khat_used, slot, new, apply)``: khat_used = min(khat, F); slot =
    khat_used + 1 (clamped to W - 1); ``apply`` marks the rows whose ``ids[r, slot]`` becomes
    ``new[r]``.  A row with a pinned slot (``pin_slot`` >= 0) is not swapped again.
    A compact ``(top, alternative, margin)`` tuple consumes the decoder's rounded
    margin comparison directly and uses the same cutoff, eligibility and slot rule.
    """
    rows, width = ids.shape
    compact = isinstance(logits, tuple)
    top = logits[0] if compact else logits.argmax(dim=-1)
    f = rule_f(top, ids, slot_counts)
    khat_used = torch.minimum(khat, f.to(khat.device))
    target = khat_used.to(ids.device) + 1
    slot = target.clamp(max=width - 1)
    eligible = active & (target <= width - 1)
    if pin_slot is not None:
        eligible = eligible & (pin_slot < 0)
    # decoder row khat_used is block slot khat_used's output: the distribution for slot
    # khat_used + 1; it was computed since khat_used <= min(khat, W - 2)
    if compact:
        current = ids.gather(1, slot[:, None])[:, 0]
        _, alternative, margin = logits
        index = (slot - 1).clamp(0, top.shape[1] - 1)[:, None]
        new = alternative.gather(1, index)[:, 0]
        apply = margin.gather(1, index)[:, 0] & (new != current)
    else:
        row_logits = logits[torch.arange(rows, device=logits.device),
                            (slot - 1).clamp(0, logits.shape[1] - 1).to(logits.device)]
        current = ids.gather(1, slot[:, None])[:, 0]
        new, apply = margin_swap(row_logits, current.to(row_logits.device), 0.0, exclude_current=True)
    return khat_used, slot, new.to(ids.device), apply.to(ids.device) & eligible


@sequencewise('ids', 3)
def compact_logits(logits, ids):
    """Reference compact epilogue, including float32 softmax's near-tie rounding."""
    current = ids[:, 1:]
    scores = logits.float().clone()
    scores.scatter_(-1, current[..., None], float('-inf'))
    alternative = scores.argmax(-1)
    probabilities = torch.softmax(logits.float(), -1)
    margin = (probabilities.gather(-1, alternative[..., None])[..., 0]
              >= probabilities.gather(-1, current[..., None])[..., 0])
    return logits.argmax(-1), alternative, margin


@reference_op("refresh_compact_head")
@sequencewise('ids', 3, shared=('weight',))
def refresh_compact_head(hidden, weight, ids, storage=None, *, persistent=False):
    """Torch compact decoder head; storage holds the engine's preallocated outputs."""
    logits = (hidden @ weight.T).reshape(ids.shape[0], ids.shape[1] - 1, weight.shape[0])
    result = compact_logits(logits, ids)
    if storage is None:
        return result
    for output, value in zip(storage[1], result):
        output.copy_(value)
    return storage[1]


@reference_op("refresh_swap_write")
@sequencewise('ids', 3)
def swap_write(compact, ids, khat, slot_counts, active, pin_slot=None):
    """Consume a refresh decision and write the selected draft ids in place."""
    result = swap_decision(compact, ids, khat, slot_counts, active, pin_slot)
    _, slot, new, apply = result
    old = ids.gather(1, slot[:, None])[:, 0]
    ids.scatter_(1, slot[:, None], torch.where(apply, new, old)[:, None])
    return result


@reference_op("refresh_agreed_pins")
@sequencewise('ids', 3, shared=('lanes',))
def agreed_pins(ids, khat_used, active, pin_mask, pin_values, *, lanes=None):
    """Pin the refresh input at slots 1..khat_used, retaining every existing pin."""
    if lanes is None:
        lanes = torch.arange(ids.shape[1], device=ids.device)[None, :]
    agreed = ((lanes >= 1) & (lanes <= khat_used[:, None]) & active[:, None] & ~pin_mask)
    pin_values.copy_(torch.where(agreed, ids, pin_values))
    pin_mask.logical_or_(agreed)
    return agreed






@reference_op("refresh_visibility")
def inject_mask_reference(mask, khat, start, rows, reset):
    row_bytes = (start + rows) // 8
    index = torch.arange(rows * row_bytes, device=mask.device)
    query = index // row_bytes
    entry = (index % row_bytes) * 8 - start
    region = (entry >= 0) & (entry < rows)
    own = region & (entry // 16 == query // 16)
    limit = torch.minimum(khat.clamp(0, 15)[query // 16], query % 16 - 1)
    count = (limit - entry % 16 + 1).clamp(0, 8)
    packed = torch.where(own, (1 << count) - 1, 0).to(torch.uint8)
    target = mask[:index.numel()]
    prior = torch.where(region, 0, target).to(torch.uint8) if reset else target
    target.copy_(prior | packed)
