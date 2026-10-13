"""Fuse compact refresh decisions and the draft-id write without changing rules."""
import torch

try:
    from jetspec.engines.jetspec_plus_plus._optional import triton
    from jetspec.engines.jetspec_plus_plus._optional import tl
except ImportError:
    triton = tl = None


if triton is not None:
    @triton.jit
    def _true_taps(TAPS, LENGTH, LANES, OUT,
                   TS: tl.constexpr, SS: tl.constexpr, DS: tl.constexpr,
                   STREAM: tl.constexpr, WIDTH: tl.constexpr, TILE: tl.constexpr):
        row = tl.program_id(0)
        col = tl.program_id(1) * TILE + tl.arange(0, TILE)
        position = tl.maximum(tl.load(LENGTH) + tl.load(LANES + row), 0)
        value = tl.load(TAPS + position * TS + STREAM * SS + col * DS,
                        col < WIDTH, other=0)
        tl.store(OUT + row * WIDTH + col, value, col < WIDTH)

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
        if FINISH:
            tl.store(PIN + row, tl.where(apply, slot, tl.load(PIN + row)))
            tl.store(PIN_TOKEN + row, new)
            tl.store(FINISHED + row, tl.load(FINISHED + row) & ~apply)
            tl.store(INJECT_COUNT + row, tl.where(tl.load(ACTIVE + row), used + 1, 0))
        else:
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
        from jetspec.engines.jetspec_plus_plus.draft.eval_profile import enabled as eval_enabled
        top, alternative, margin = compact
        width = ids.shape[1]
        if width not in ((16, 32) if eval_enabled() else (16,)) or not all(
                x.is_contiguous() for x in (*compact, ids, khat, slot_counts, active)):
            raise ValueError('fused swap write requires contiguous 16-slot rows (32 in eval)')
        result = (torch.empty_like(khat), torch.empty_like(khat),
                  torch.empty_like(khat), torch.empty_like(active))
        _decide[(ids.shape[0],)](*compact, ids, khat, slot_counts, active,
            pin_slot if pin_slot is not None else khat, *result, width, top.shape[1],
            pin_slot is not None, triton.next_power_of_2(width - 1), num_warps=4)
        return result
    from jetspec.engines.jetspec_plus_plus.draft.chain.refresh_swap import compact_decision
    result = compact_decision(compact, ids, khat, slot_counts, active, pin_slot)
    used, slot, new, apply = result
    old = ids.gather(1, slot[:, None])[:, 0]
    ids.scatter_(1, slot[:, None], torch.where(apply, new, old)[:, None])
    return result


def read_true_taps(storage, length, lanes):
    """Gather the decoder's last-stream taps directly from the live prefix."""
    if not storage.is_cuda or triton is None:
        positions = (length + lanes).clamp_min(0)
        return storage[:, -1].index_select(0, positions)
    output = storage.new_empty(lanes.numel(), storage.shape[-1])
    _true_taps[(lanes.numel(), triton.cdiv(storage.shape[-1], 512))](
        storage, length, lanes, output, *storage.stride(), storage.shape[1] - 1,
        storage.shape[-1], 512, num_warps=4)
    return output


def decide_and_finish(compact, ids, khat, slot_counts, active,
                      pin_slot, pin_token, finished):
    """Apply the existing unpinned swap decision and write refresh metadata."""
    if ids.is_cuda and triton is not None:
        from jetspec.engines.jetspec_plus_plus.draft.eval_profile import enabled as eval_enabled
        width = ids.shape[1]
        if width not in ((16, 32) if eval_enabled() else (16,)) or not all(x.is_contiguous() for x in (
                *compact, ids, khat, slot_counts, active, pin_slot, pin_token, finished)):
            raise ValueError('fused swap finish requires contiguous 16-slot rows (32 in eval)')
        output = torch.empty_like(khat)
        _decide[(ids.shape[0],)](*compact, ids, khat, slot_counts, active, pin_slot,
            khat, khat, khat, active, width, compact[0].shape[1], False,
            triton.next_power_of_2(width - 1), PIN_TOKEN=pin_token, FINISHED=finished,
            INJECT_COUNT=output, FINISH=True, num_warps=4)
        return output
    used, slot, new, apply = decide_and_write(compact, ids, khat, slot_counts, active)
    pin_slot.copy_(torch.where(apply, slot, pin_slot))
    pin_token.copy_(new)
    finished.logical_and_(~apply)
    return torch.where(active, used + 1, torch.zeros_like(used))
