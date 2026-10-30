"""J0-owned fp64 refresh prefix scan; threshold ties require a served tau gate."""
import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None


def allocate_refresh(rows, lanes, device, window, threshold):
    """Allocate refresh outputs once, outside graph capture."""
    return dict(REFRESH=True, RefreshChained=torch.empty(rows, lanes - 1,
                    device=device, dtype=torch.float64),
                RefreshKhat=torch.empty(rows, device=device, dtype=torch.long),
                RefreshCounts=torch.empty(rows, device=device, dtype=torch.long),
                RefreshTails=torch.empty(rows, device=device, dtype=torch.long),
                RefreshStart=torch.empty((), device=device, dtype=torch.long),
                REFRESH_WINDOW=window, REFRESH_THRESHOLD=threshold)


def reference_refresh(top1, active, keys, length, state):
    """CPU pass fallback with the original double cumprod rounding points."""
    from jetspec.engines.jetspec_plus_plus.draft.method.refresh import refresh_bookkeeping
    chained = top1[:, 1:].double().cumprod(1)
    values = refresh_bookkeeping(
        chained, active, keys, length, state['REFRESH_WINDOW'],
        state['REFRESH_THRESHOLD'], None)
    state['RefreshChained'].copy_(chained)
    for name, value in zip(('RefreshKhat', 'RefreshCounts', 'RefreshTails', 'RefreshStart'), values):
        state[name].copy_(value.reshape_as(state[name]))


if triton is not None:
    @triton.jit
    def _refresh(top1, lane, row, ACTIVE, KEYS, RP, CHAINED, KHAT, COUNTS,
                 TAILS, START, L: tl.constexpr, RP_L: tl.constexpr,
                 WINDOW: tl.constexpr, THRESHOLD: tl.constexpr):
        # A balanced scan changes only fp64 product association. The fp32
        # probabilities and argmax from _finish remain untouched. Bound: 32 ulp
        # for the served 15-factor prefix; integer outputs need the tau gate.
        probability = tl.where(lane > 0, top1, 1.).to(tl.float64)
        chained = tl.associative_scan(probability, 0, _multiply)
        tl.store(CHAINED + row * (L - 1) + lane - 1, chained, lane > 0)
        threshold = tl.full((), THRESHOLD, tl.float64)
        khat = tl.min(tl.where((lane > 0) & (chained < threshold), lane, L - 1), 0)
        active = tl.load(ACTIVE + row)
        tl.store(KHAT + row, tl.where(active, khat, 0))
        tl.store(COUNTS + row, tl.where(active, tl.minimum(khat, L - 2) + 1, 0))
        tl.store(TAILS + row, tl.load(KEYS + row * 2) + 1)
        if row == 0:
            tl.store(START, tl.load(RP + RP_L) + WINDOW)


    @triton.jit
    def _multiply(left, right):
        return left * right
