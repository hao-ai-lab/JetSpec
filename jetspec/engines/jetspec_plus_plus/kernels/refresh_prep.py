"""Fixed-refresh integer layout and bitwise gathers, with no activation arithmetic."""

import torch
from jetspec.engines.jetspec_plus_plus.draft.method.ops import reference
from jetspec.engines.jetspec_plus_plus.draft.method import refresh
from . import register_optimization

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None


if triton is not None:
    @triton.jit
    def _bookkeeping(CHAINED, ACTIVE, KEYS, LENGTH, KHAT, COUNTS, TAILS, START,
                     D: tl.constexpr, R: tl.constexpr, WINDOW: tl.constexpr,
                     THRESHOLD: tl.constexpr, CAP: tl.constexpr,
                     K0: tl.constexpr, K1: tl.constexpr, B: tl.constexpr):
        row = tl.program_id(0)
        col = tl.arange(0, B)
        value = tl.load(CHAINED + row * D + col, col < D, other=1)
        threshold = tl.full((), THRESHOLD, tl.float64)
        khat = tl.min(tl.where((col < D) & (value < threshold), col + 1, D), 0)
        if CAP >= 0:
            khat = tl.minimum(khat, CAP)
        active = tl.load(ACTIVE + row)
        khat = tl.where(active, khat, 0)
        tl.store(KHAT + row, khat)
        tl.store(COUNTS + row, tl.where(active, tl.minimum(khat, D - 1) + 1, 0))
        tl.store(TAILS + row, tl.load(KEYS + row * K0) + 1)
        if row == 0:
            tl.store(START, tl.load(LENGTH) + WINDOW)

    @triton.jit
    def _prepare(EMBED, LENGTH, TAILS, COUNTS, CT, ST, DATA, COS, SIN,
                 ROW, SLOT, VALID, KOF, POS, STARTS,
                 R: tl.constexpr, W: tl.constexpr, H: tl.constexpr, D: tl.constexpr,
                 T: tl.constexpr, WINDOW: tl.constexpr,
                 E0: tl.constexpr, E1: tl.constexpr, E2: tl.constexpr,
                 C0: tl.constexpr, C1: tl.constexpr, S0: tl.constexpr, S1: tl.constexpr,
                 RB: tl.constexpr, B: tl.constexpr):
        token = tl.program_id(0)
        part = tl.program_id(1)
        r = tl.arange(0, RB)
        count = tl.load(COUNTS + r, r < R, other=0).to(tl.int64)
        ends = tl.cumsum(count)
        row = tl.sum(((token >= ends) & (r < R)).to(tl.int64), 0)
        valid = row < R
        start = tl.sum(tl.where(r < row, count, 0), 0)
        slot = tl.where(valid, token - start, W)
        tail = tl.load(TAILS + row, valid, other=0).to(tl.int64)
        tail = tl.minimum(tl.maximum(tail, 0), WINDOW)
        length = tl.load(LENGTH).to(tl.int64)
        position = tl.where(valid, length + tail + slot, length)
        col = part * B + tl.arange(0, B)
        value = tl.load(EMBED + row * E0 + slot * E1 + col * E2,
                        valid & (col < H), other=0)
        tl.store(DATA + token * H + col, value, col < H)
        if part == 0:
            d = tl.arange(0, B)
            cos = tl.load(CT + position * C0 + d * C1, d < D, other=0)
            sin = tl.load(ST + position * S0 + d * S1, d < D, other=0)
            tl.store(COS + token * D + d, cos, d < D)
            tl.store(SIN + token * D + d, sin, d < D)
            tl.store(ROW + token, row)
            tl.store(SLOT + token, slot)
            tl.store(VALID + token, valid)
            tl.store(KOF + token, tail)
            tl.store(POS + token, position)
            tl.store(STARTS + token, tl.where(valid, start, T))

    @triton.jit
    def _scatter(OUTPUT, COUNTS, PADDED, R: tl.constexpr, W: tl.constexpr,
                 H: tl.constexpr, T: tl.constexpr, RB: tl.constexpr, B: tl.constexpr):
        flat = tl.program_id(0) * B + tl.arange(0, B)
        row = flat // ((W + 1) * H)
        slot = flat // H % (W + 1)
        col = flat % H
        start = tl.full((B,), 0, tl.int64)
        # Inverse gather writes zeros and real outputs together; filler writes never race.
        for r in tl.static_range(R):
            start += tl.where(r < row, tl.load(COUNTS + r).to(tl.int64), 0)
        count = tl.load(COUNTS + row, row < R, other=0)
        valid = (row < R) & (slot < count) & (slot < W) & (start + slot < T)
        value = tl.load(OUTPUT + (start + slot) * H + col, valid, other=0)
        tl.store(PADDED + flat, value, flat < (R + 1) * (W + 1) * H)


def bookkeeping(chained, active, keys, length, window, threshold, cap):
    """Fuse selection after the unchanged torch double cumprod rounding points."""
    rows, drafts = chained.shape
    khat = torch.empty(rows, device=keys.device, dtype=torch.long)
    counts, tails = torch.empty_like(khat), torch.empty_like(khat)
    start = torch.empty_like(length)
    if chained.is_cuda:
        if triton is None:
            raise RuntimeError('refresh bookkeeping requires CUDA Triton')
        _bookkeeping[(rows,)](chained, active, keys, length, khat, counts, tails, start,
                              drafts, rows, window, threshold, -1 if cap is None else cap,
                              *keys.stride(), triton.next_power_of_2(drafts))
    else:
        return reference('refresh_bookkeeping')(chained, active, keys, length, window, threshold, cap)
    return khat, counts, tails, start


register_optimization('refresh_bookkeeping', bookkeeping)


def prepare(embeddings, length, tails, counts, tokens, cos_table, sin_table, window):
    """Pack rows, positions, attention metadata and table gathers in one launch."""
    rows, width, hidden = embeddings.shape
    dim = cos_table.shape[1]
    result = {name: torch.empty(tokens, device=embeddings.device,
                               dtype=torch.bool if name == 'valid' else torch.long)
              for name in ('row', 'slot', 'valid', 'k_of', 'positions', 'starts')}
    result['data'] = embeddings.new_empty(tokens, hidden)
    result['cos'] = cos_table.new_empty(1, tokens, dim)
    result['sin'] = sin_table.new_empty(1, tokens, dim)
    if embeddings.is_cuda:
        if triton is None:
            raise RuntimeError('refresh prep requires CUDA Triton')
        block = max(256, triton.next_power_of_2(dim))
        _prepare[(tokens, triton.cdiv(hidden, block))](
            embeddings, length, tails, counts, cos_table, sin_table,
            result['data'], result['cos'], result['sin'], result['row'], result['slot'],
            result['valid'], result['k_of'], result['positions'], result['starts'],
            rows, width, hidden, dim, tokens, window, *embeddings.stride(),
            *cos_table.stride(), *sin_table.stride(), triton.next_power_of_2(rows), block)
    else:
        offsets = torch.cat((counts.new_zeros(1), counts.cumsum(0)))
        index = torch.arange(tokens, device=counts.device)
        row = torch.searchsorted(offsets[1:], index, right=True)
        valid = row < rows
        slot = torch.where(valid, index - offsets[row], width)
        k_of = torch.cat((tails.long().clamp(0, window), tails.new_zeros(1)))[row]
        positions = torch.where(valid, length + k_of + slot, length)
        result.update(row=row, slot=slot, valid=valid, k_of=k_of, positions=positions,
                      starts=torch.where(valid, offsets[row], tokens),
                      data=torch.where(valid[:, None],
                          embeddings[row.clamp(max=rows-1), slot.clamp(max=width-1)], 0),
                      cos=cos_table[positions][None], sin=sin_table[positions][None])
    return result


def scatter(output, counts, width):
    """Same returned shape, strides and zero slots as the eager padded scatter."""
    rows, hidden = counts.numel(), output.shape[1]
    padded = output.new_empty(rows + 1, width + 1, hidden)
    if output.is_cuda:
        if triton is None:
            raise RuntimeError('refresh scatter requires CUDA Triton')
        _scatter[(triton.cdiv(padded.numel(), 256),)](
            output, counts, padded, rows, width, hidden, output.shape[0],
            triton.next_power_of_2(rows), 256)
    else:
        offsets = torch.cat((counts.new_zeros(1), counts.cumsum(0)))
        index = torch.arange(output.shape[0], device=counts.device)
        row = torch.searchsorted(offsets[1:], index, right=True)
        slot = torch.where(row < rows, index - offsets[row], width)
        padded.zero_()
        padded[row, slot] = output
    return padded[:rows, :width]
