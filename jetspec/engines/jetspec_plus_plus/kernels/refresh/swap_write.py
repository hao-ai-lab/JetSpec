"""Device kernels for draft/swap_write.py."""

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
                FINISH: tl.constexpr = False, INCLUSIVE: tl.constexpr = False):
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
            inject = tl.where(tl.load(ACTIVE + row), used + 1, 0)
            tl.store(INJECT_COUNT + row, inject - INCLUSIVE)
        else:
            tl.store(USED + row, used)
            tl.store(SLOT + row, slot)
            tl.store(NEW + row, new)
            tl.store(APPLY + row, apply)
