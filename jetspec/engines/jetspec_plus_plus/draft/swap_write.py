"""Fuse compact refresh decisions and the draft-id write without changing rules."""
import torch
from jetspec.engines.jetspec_plus_plus.draft.method.ops import reference
from jetspec.engines.jetspec_plus_plus.draft.method import refresh
from jetspec.engines.jetspec_plus_plus.kernels import register_optimization

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = tl = None


if triton is not None:
    from jetspec.engines.jetspec_plus_plus.kernels.swap_write import (_decide)


def decide_and_write(compact, ids, khat, slot_counts, active, pin_slot=None):
    """Return compact_decision's four tensors and perform its conditional id write.

    The decoder's rounded margin boolean is consumed unchanged; no logits,
    probabilities, acceptance, or published state arithmetic is introduced.
    """
    if ids.is_cuda and triton is not None:
        from jetspec.engines.jetspec_plus_plus.jetspec.profile import enabled as profile_enabled
        top, alternative, margin = compact
        width = ids.shape[1]
        if width not in ((16, 32) if profile_enabled() else (16,)) or not all(
                x.is_contiguous() for x in (*compact, ids, khat, slot_counts, active)):
            raise ValueError('fused swap write requires contiguous 16-slot rows (32 in eval)')
        result = (torch.empty_like(khat), torch.empty_like(khat),
                  torch.empty_like(khat), torch.empty_like(active))
        _decide[(ids.shape[0],)](*compact, ids, khat, slot_counts, active,
            pin_slot if pin_slot is not None else khat, *result, width, top.shape[1],
            pin_slot is not None, triton.next_power_of_2(width - 1), num_warps=4)
        return result
    return reference('refresh_swap_write')(compact, ids, khat, slot_counts, active, pin_slot)


register_optimization('refresh_swap_write', decide_and_write)


def decide_and_finish(compact, ids, khat, slot_counts, active,
                      pin_slot, pin_token, finished, *, inclusive=False):
    """Write the legacy single-refresh swap, pins and injection count in one launch."""
    if ids.is_cuda and triton is not None:
        from jetspec.engines.jetspec_plus_plus.jetspec.profile import enabled as profile_enabled
        width = ids.shape[1]
        if width not in ((16, 32) if profile_enabled() else (16,)) or not all(
                x.is_contiguous() for x in (*compact, ids, khat, slot_counts, active,
                                            pin_slot, pin_token, finished)):
            raise ValueError('fused swap finish requires contiguous 16-slot rows (32 in eval)')
        output = torch.empty_like(khat)
        _decide[(ids.shape[0],)](
            *compact, ids, khat, slot_counts, active, pin_slot,
            khat, khat, khat, active, width, compact[0].shape[1], False,
            triton.next_power_of_2(width - 1), PIN_TOKEN=pin_token, FINISHED=finished,
            INJECT_COUNT=output, FINISH=True, INCLUSIVE=inclusive, num_warps=4)
        return output
    used, slot, new, apply = reference('refresh_swap_write')(
        compact, ids, khat, slot_counts, active)
    pin_slot.copy_(torch.where(apply, slot, pin_slot))
    pin_token.copy_(new)
    finished.logical_and_(~apply)
    return torch.where(active, used + 1, torch.zeros_like(used)) - int(inclusive)
