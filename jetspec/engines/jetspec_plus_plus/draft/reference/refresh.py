"""Eager refresh context and pin state; decisions use the shared method."""
import torch

from ..method.refresh import swap_decision, agreed_pins


def inject(target_hidden, hidden, khat, *, block_size=16):
    """Expose refreshed block entries as extra context for the next causal J pass."""
    rows, base, _ = target_hidden.shape
    if hidden.shape != (rows, block_size, target_hidden.shape[-1]):
        raise ValueError('refresh hidden must have shape [rows, block_size, features]')
    if khat.shape != (rows,):
        raise ValueError('one refresh cutoff is required per row')
    slots = torch.arange(block_size, device=target_hidden.device)
    # The recipe projects all eligible rows; the unread final slot is zeroed in project().
    extra = torch.where((slots < block_size - 1)[None, :, None], hidden,
                        torch.zeros_like(hidden))
    context = torch.cat([target_hidden, extra], dim=1)
    position_ids = torch.cat((torch.arange(base + block_size, device=context.device),
                              torch.arange(base, base + block_size, device=context.device)))
    position_ids = position_ids[None].repeat(rows, 1)
    allowed = torch.ones(rows, block_size, base + 2 * block_size,
                         dtype=torch.bool, device=context.device)
    allowed[:, :, base:base + block_size] = (
        (slots[None, :, None] > slots[None, None, :]) &
        (slots[None, None, :] <= khat[:, None, None]))
    mask = torch.zeros((rows, 1, block_size, allowed.shape[-1]), dtype=context.dtype,
                       device=context.device).masked_fill_(~allowed[:, None],
                                                           torch.finfo(context.dtype).min)
    return context, position_ids, mask


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
            agreed_pins(ids, khat_used, active, self.pin_mask, self.pin_values,
                        lanes=self.pin_lanes)
        if self.dense_pins:
            swapped = (self.pin_lanes == slot[:, None]) & apply[:, None]
            self.pin_values.copy_(torch.where(swapped, new[:, None], self.pin_values))
            self.pin_mask.logical_or_(swapped)
        ids.scatter_(1, slot[:, None], torch.where(apply, new, ids.gather(1, slot[:, None])[:, 0])[:, None])
        self.pin_slot.copy_(torch.where(apply, slot, self.pin_slot))
        self.pin_token.copy_(torch.where(apply, new, self.pin_token) if self.multi_refresh else new)
        finished.logical_and_(~apply)
        return torch.where(active, khat_used + 1, torch.zeros_like(khat_used))
