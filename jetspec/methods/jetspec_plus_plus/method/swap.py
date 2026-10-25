"""Ported from the JetSpec++ reference implementation.

RefreshSwap also mirrors the reference refresh-swap operation.

Public API:
rule_f(decoder_argmax, ids, slot_counts) -> leading agreement counts.
margin_swap(logits, current, margin, *, exclude_current) -> (new, apply).
swap_decision(logits, ids, khat, slot_counts, active, pin_slot=None)
    -> (khat_used, slot, new, apply), with logits [R, W-1, V] and ids [R, W].
RefreshSwap(*, rows, lanes, device=None, pin_agreed=True) owns round pin state.
configure_pins(refresh_at, *, rows, lanes) enables accumulated multi-refresh pins;
reset() starts J0; pins() -> (mask, values) or (slot, token).
apply(logits, *, ids, khat, slot_counts, active, finished) -> inclusive inject counts;
    edits ids, pins and finished in place. Inactive rows inject zero entries.
"""
import torch

__all__ = ['rule_f', 'margin_swap', 'swap_decision', 'RefreshSwap']


def rule_f(decoder_argmax, ids, slot_counts):
    """Count valid leading decoder agreements with block slots 1..W-1."""
    width = ids.shape[1]
    slot_counts = torch.as_tensor(slot_counts, dtype=torch.long, device=ids.device).reshape(-1, 1)
    agree = decoder_argmax[:, :width - 1].to(ids.device) == ids[:, 1:]
    valid = torch.arange(1, width, device=ids.device)[None, :] < slot_counts
    return torch.cumprod((agree & valid).long(), dim=1).sum(dim=1)


def margin_swap(logits, current, margin, *, exclude_current):
    """Choose the best alternative with an inclusive softmax-probability margin."""
    probabilities = torch.softmax(logits.float(), dim=-1)
    scores = logits.float().clone()
    if exclude_current:
        scores.scatter_(1, current[:, None], float('-inf'))
    new = scores.argmax(dim=-1)
    gain = probabilities.gather(1, new[:, None])[:, 0] - probabilities.gather(1, current[:, None])[:, 0]
    return new, (gain >= margin) & (new != current)


def swap_decision(logits, ids, khat, slot_counts, active, pin_slot=None):
    """The cutoff and the swap from the refresh-context decoder logits, on device.

    ``logits`` [R, W-1, V]: decoder row j of row r (the distribution for block slot j + 1), valid
    for j < slot_counts[r]; ``ids`` [R, W] the block (the next pass's input); ``khat`` [R] the
    draft cutoff.  Returns ``(khat_used, slot, new, apply)``: khat_used = min(khat, F); slot =
    khat_used + 1 (clamped to W - 1); ``apply`` marks the rows whose ``ids[r, slot]`` becomes
    ``new[r]``.  A row with a pinned slot (``pin_slot`` >= 0) is not swapped again.
    """
    rows, width = ids.shape
    f = rule_f(logits.argmax(dim=-1), ids, slot_counts)
    khat_used = torch.minimum(khat, f.to(khat.device))
    target = khat_used.to(ids.device) + 1
    slot = target.clamp(max=width - 1)
    eligible = active & (target <= width - 1)
    if pin_slot is not None:
        eligible = eligible & (pin_slot < 0)
    # decoder row khat_used is block slot khat_used's output: the distribution for slot
    # khat_used + 1; it was computed since khat_used <= min(khat, W - 2)
    row_logits = logits[torch.arange(rows, device=logits.device),
                        (slot - 1).clamp(0, logits.shape[1] - 1).to(logits.device)]
    current = ids.gather(1, slot[:, None])[:, 0]
    new, apply = margin_swap(row_logits, current.to(row_logits.device), 0.0, exclude_current=True)
    return khat_used, slot, new.to(ids.device), apply.to(ids.device) & eligible


class RefreshSwap:
    """The margin-0 refresh epilogue and round-owned legacy or dense pins."""

    def __init__(self, *, rows, lanes, device=None, pin_agreed=True):
        # Explicit method inputs replace the reference's environment and graph-buffer allocation.
        self.pin_agreed = pin_agreed
        self.dense_pins = pin_agreed
        self.multi_refresh = False
        self.pin_slot = torch.full((rows,), -1, dtype=torch.long, device=device)
        self.pin_token = torch.zeros(rows, dtype=torch.long, device=device)
        if pin_agreed:
            self.pin_mask = torch.zeros((rows, lanes), dtype=torch.bool, device=device)
            self.pin_values = torch.zeros((rows, lanes), dtype=torch.long, device=device)
            self.pin_lanes = torch.arange(lanes, device=device)[None, :]

    def configure_pins(self, refresh_at, *, rows, lanes):
        """Accumulate swap pins across multiple refreshes within one round."""
        self.multi_refresh = len(refresh_at) > 1
        if len(refresh_at) > 1 and not self.dense_pins:
            device = self.pin_slot.device
            self.pin_mask = torch.zeros((rows, lanes), dtype=torch.bool, device=device)
            self.pin_values = torch.zeros((rows, lanes), dtype=torch.long, device=device)
            self.pin_lanes = torch.arange(lanes, device=device)[None, :]
            self.dense_pins = True

    def reset(self):
        """J0: no pin survives from the previous round."""
        self.pin_slot.fill_(-1)
        if self.dense_pins:
            self.pin_mask.zero_()

    def pins(self):
        if self.dense_pins:
            return self.pin_mask, self.pin_values
        return self.pin_slot, self.pin_token

    def apply(self, logits, *, ids, khat, slot_counts, active, finished):
        """Swap the first disagreement, pin it and reopen rows that changed."""
        # The torch round supplies decoder logits instead of replaying the reference's graph read here.
        khat_used, slot, new, apply = swap_decision(logits, ids, khat, slot_counts, active)
        if self.dense_pins:
            apply &= ~self.pin_mask.gather(1, slot[:, None])[:, 0]
        if self.pin_agreed:
            agreed = ((self.pin_lanes >= 1) & (self.pin_lanes <= khat_used[:, None])
                      & active[:, None] & ~self.pin_mask)
            self.pin_values.copy_(torch.where(agreed, ids, self.pin_values))
            self.pin_mask.logical_or_(agreed)
        if self.dense_pins:
            swapped = (self.pin_lanes == slot[:, None]) & apply[:, None]
            self.pin_values.copy_(torch.where(swapped, new[:, None], self.pin_values))
            self.pin_mask.logical_or_(swapped)
        ids.scatter_(1, slot[:, None], torch.where(apply, new, ids.gather(1, slot[:, None])[:, 0])[:, None])
        self.pin_slot.copy_(torch.where(apply, slot, self.pin_slot))
        self.pin_token.copy_(torch.where(apply, new, self.pin_token) if self.multi_refresh else new)
        finished.logical_and_(~apply)
        return torch.where(active, khat_used + 1, torch.zeros_like(khat_used))
