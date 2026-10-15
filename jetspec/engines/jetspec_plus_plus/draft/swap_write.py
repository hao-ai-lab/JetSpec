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

    @triton.jit
    def _decide(TOP, ALT, MARGIN, IDS, KHAT, COUNTS, ACTIVE, PIN,
                USED, SLOT, NEW, APPLY, WIDTH: tl.constexpr, SLOTS: tl.constexpr,
                PINNED: tl.constexpr, BLOCK: tl.constexpr,
                PIN_TOKEN=None, FINISHED=None, INJECT_COUNT=None,
                FINISH: tl.constexpr = False):
        row = tl.program_id(0)
        lane = tl.arange(0, BLOCK)
        top = tl.load(TOP + row * SLOTS + lane, lane < WIDTH - 1, other=-1)
        token = tl.load(IDS + row * WIDTH + lane + 1, lane < WIDTH - 1, other=-2)
        count = tl.load(COUNTS + row)
        agree = (top == token) & (lane + 1 < count)
        first = tl.min(tl.where((lane < WIDTH - 1) & ~agree, lane, WIDTH - 1), 0)
        used = tl.minimum(tl.load(KHAT + row), first)
        target = used + 1
        slot = tl.minimum(target, WIDTH - 1)
        index = tl.minimum(tl.maximum(slot - 1, 0), SLOTS - 1)
        new = tl.load(ALT + row * SLOTS + index)
        current = tl.load(IDS + row * WIDTH + slot)
        apply = tl.load(MARGIN + row * SLOTS + index) & tl.load(ACTIVE + row)
        apply = apply & (target < WIDTH) & (new != current)
        if PINNED:
            apply = apply & (tl.load(PIN + row) < 0)
        tl.store(IDS + row * WIDTH + slot, new, apply)
        tl.store(USED + row, used)
        tl.store(SLOT + row, slot)
        tl.store(NEW + row, new)
        tl.store(APPLY + row, apply)


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
