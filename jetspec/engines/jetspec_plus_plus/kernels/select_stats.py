"""Unit T4 level 2: one read of the BF16 chain decoder logits for the select and the K3 hazard.

The default N=1 chain select (anchor_select_fused + decoder_select radix) widens the BF16
logits to FP32, masks d_rej into a second FP32 tensor, runs torch's FP32 softmax and reads
the probability tensor again in the radix counts, the emit and the tie repair. The K3 hazard
then reads the same logits twice more for its unfiltered max, argmax and sum-exp.

Here one split-row kernel reads the BF16 logits once. Per 1024-column chunk it writes the
unfiltered max, first argmax and sum-exp, the d_rej-filtered max and sum-exp, the radix
pass-0 byte histogram and, optionally, the BF16 copy that the chain hands to its rows.
One row kernel merges the chunks (sum = sum_c s_c * exp(m_c - m)) and takes the pass-0
threshold. Later kernels compute p = exp(l - m) / s from the logits, with no [rows, V]
probability tensor. Every p comes from the same expression (IEEE exp and division), so
ties stay ties and p is monotone in the logit, as the radix pivot requires.

Parity class: tau gate. The filtered max is exact and the radix histograms are bitwise; the
sum-exp merge order differs from torch's softmax and the K3 hazard's two-pass sum, so p and
the hazard move by a few ulp. The LAC test p >= 1 - q can flip at one ulp.
"""
import torch

try:
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice
except ImportError:  # the Mac CPU reference does not need Triton
    triton = None

TILE = 1024
BIN_TILE = 16


if triton is not None:
    from jetspec.engines.jetspec_plus_plus.kernels.decoder_select import (
        _bf16_key, _radix_find_pivot, _radix_hist, _radix_offsets, _radix_threshold,
        _radix_totals_coalesced)

    @triton.jit
    def _probability(value, maximum, total):
        return libdevice.div_rn(libdevice.exp(value - maximum), total)

    @triton.jit
    def _stats(LOGITS, REJECTED, COPY, PARTS, WINNERS, HIST,
               V: tl.constexpr, C: tl.constexpr, TILE: tl.constexpr,
               WRITE_COPY: tl.constexpr):
        row, chunk = tl.program_id(0), tl.program_id(1)
        col = chunk * TILE + tl.arange(0, TILE)
        inside = col < V
        raw = tl.load(LOGITS + row * V + col, inside, other=0.)
        if WRITE_COPY:
            tl.store(COPY + row * V + col, raw, inside)
        value = tl.where(inside, raw.to(tl.float32), -float('inf'))
        rejected = tl.load(REJECTED + row)
        # Unfiltered, for the K3 hazard and the rejected-was-top1 flag.
        top = tl.max(value, 0)
        winner = tl.min(tl.where(inside & (value == top), col, 0x7FFFFFFF), 0)
        base = tl.where(top == -float('inf'), 0., top)
        total = tl.sum(tl.where(inside, tl.exp(value - base), 0.), 0)
        # d_rej filtered, for the select probabilities.
        live = inside & (col != rejected)
        kept = tl.where(live, value, -float('inf'))
        kept_top = tl.max(kept, 0)
        kept_base = tl.where(kept_top == -float('inf'), 0., kept_top)
        kept_total = tl.sum(tl.where(live, libdevice.exp(kept - kept_base), 0.), 0)
        part = PARTS + (row * C + chunk) * 4
        tl.store(part, top)
        tl.store(part + 1, total)
        tl.store(part + 2, kept_top)
        tl.store(part + 3, kept_total)
        tl.store(WINNERS + row * C + chunk, winner)
        # Radix pass 0 on the BF16 logit keys, as decoder_select._radix_hist.
        bins = (_bf16_key(raw) >> 8) & 255
        counts = tl.histogram(bins, 256, mask=live)
        tl.store(HIST + (row * C + chunk) * 256 + tl.arange(0, 256), counts)

    @triton.jit
    def _stats_finish(PARTS, WINNERS, TOTAL, PREFIX, REJECTED, TOP1, STATS, ARGMAX,
                      ROWS: tl.constexpr, C: tl.constexpr, BLOCK: tl.constexpr,
                      WIDTH: tl.constexpr):
        row = tl.program_id(0)
        chunk = tl.arange(0, BLOCK)
        valid = chunk < C
        part = PARTS + (row * C + chunk) * 4
        tops = tl.load(part, valid, other=-float('inf'))
        totals = tl.load(part + 1, valid, other=0.)
        kept_tops = tl.load(part + 2, valid, other=-float('inf'))
        kept_totals = tl.load(part + 3, valid, other=0.)
        winners = tl.load(WINNERS + row * C + chunk, valid, other=0x7FFFFFFF)
        top = tl.max(tops, 0)
        winner = tl.min(tl.where(tops == top, winners, 0x7FFFFFFF), 0)
        total = tl.sum(tl.where(valid & (totals > 0), totals * tl.exp(tops - top), 0.), 0)
        kept_top = tl.max(kept_tops, 0)
        kept_total = tl.sum(tl.where(valid & (kept_totals > 0),
                                     kept_totals * libdevice.exp(kept_tops - kept_top), 0.), 0)
        tl.store(STATS + row, top)
        tl.store(STATS + ROWS + row, total)
        tl.store(STATS + 2 * ROWS + row, kept_top)
        tl.store(STATS + 3 * ROWS + row, kept_total)
        tl.store(ARGMAX + row, winner)
        tl.store(TOP1 + row, winner == tl.load(REJECTED + row))
        # Radix pass-0 threshold, as decoder_select._radix_threshold(PASS=0, BITS=16).
        slot = tl.arange(0, 256)
        count = tl.load(TOTAL + row * 256 + 255 - slot)
        cumulative = tl.cumsum(count, 0)
        crossing = tl.min(tl.where(cumulative >= WIDTH, slot, 256), 0)
        above = tl.sum(tl.where(slot < crossing, count, 0), 0)
        tl.store(PREFIX + row * 3, (255 - crossing) << 8)
        tl.store(PREFIX + row * 3 + 1, WIDTH - above)

    @triton.jit
    def _cutoff(LOGITS, STATS, PIVOTS, PREFIX,
                ROWS: tl.constexpr, V: tl.constexpr, C: tl.constexpr, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        chunk = tl.arange(0, BLOCK)
        pivot = tl.min(tl.load(PIVOTS + row * C + chunk, chunk < C, other=V), 0)
        value = tl.load(LOGITS + row * V + pivot).to(tl.float32)
        p = _probability(value, tl.load(STATS + 2 * ROWS + row),
                         tl.load(STATS + 3 * ROWS + row))
        tl.store(PREFIX + row * 3 + 2, p.to(tl.int32, bitcast=True))

    @triton.jit
    def _row_probabilities(LOGITS, STATS, row, col,
                           ROWS: tl.constexpr, V: tl.constexpr):
        value = tl.load(LOGITS + row * V + col, col < V, other=0.).to(tl.float32)
        return _probability(value, tl.load(STATS + 2 * ROWS + row),
                            tl.load(STATS + 3 * ROWS + row))

    @triton.jit
    def _counts(LOGITS, REJECTED, STATS, PREFIX, COUNTS,
                ROWS: tl.constexpr, V: tl.constexpr, C: tl.constexpr, TILE: tl.constexpr):
        row, chunk = tl.program_id(0), tl.program_id(1)
        col = chunk * TILE + tl.arange(0, TILE)
        key = _row_probabilities(LOGITS, STATS, row, col, ROWS, V).to(tl.int32, bitcast=True)
        threshold = tl.load(PREFIX + row * 3 + 2)
        live = (col < V) & (col != tl.load(REJECTED + row))
        tl.store(COUNTS + (row * C + chunk) * 2,
                 tl.sum((live & (key > threshold)).to(tl.int32), 0))
        tl.store(COUNTS + (row * C + chunk) * 2 + 1,
                 tl.sum((live & (key == threshold)).to(tl.int32), 0))

    @triton.jit
    def _emit(LOGITS, REJECTED, STATS, PREFIX, OFFSETS, ABOVE, VALUES, IDS,
              ROWS: tl.constexpr, V: tl.constexpr, C: tl.constexpr, WIDTH: tl.constexpr,
              TILE: tl.constexpr):
        row, chunk = tl.program_id(0), tl.program_id(1)
        col = chunk * TILE + tl.arange(0, TILE)
        probability = _row_probabilities(LOGITS, STATS, row, col, ROWS, V)
        key = probability.to(tl.int32, bitcast=True)
        threshold = tl.load(PREFIX + row * 3 + 2)
        live = (col < V) & (col != tl.load(REJECTED + row))
        strict, tied = live & (key > threshold), live & (key == threshold)
        strict_index = tl.load(OFFSETS + (row * C + chunk) * 2) + tl.cumsum(strict.to(tl.int32), 0) - 1
        tied_index = tl.load(OFFSETS + (row * C + chunk) * 2 + 1) + tl.cumsum(tied.to(tl.int32), 0) - 1
        above = tl.load(ABOVE + row)
        index = tl.where(strict, strict_index, above + tied_index)
        keep = strict | (tied & (tied_index < WIDTH - above))
        tl.store(VALUES + row * WIDTH + index, probability, keep)
        tl.store(IDS + row * WIDTH + index, col, keep)

    @triton.jit
    def _tie_candidates(LOGITS, REJECTED, STATS, selected_p, inexact, candidates,
                        ROWS: tl.constexpr, VOCAB: tl.constexpr, WIDTH: tl.constexpr,
                        CAP: tl.constexpr, CHUNKS: tl.constexpr, TILE: tl.constexpr):
        row = tl.program_id(0)
        chunk = tl.program_id(1)
        if tl.load(inexact + row):
            slot = tl.arange(0, CAP)
            base = candidates + (row * CHUNKS + chunk) * CAP
            tl.store(base + slot, 0x7FFFFFFF)
            col = chunk * TILE + tl.arange(0, TILE)
            threshold = tl.load(selected_p + row * WIDTH + CAP - 1).to(tl.float32)
            # d_rej carries p - 1 = -1 in the default probability tensor.
            p = tl.where(col == tl.load(REJECTED + row), -1.0,
                         _row_probabilities(LOGITS, STATS, row, col, ROWS, VOCAB))
            match = (col < VOCAB) & (p == threshold)
            rank = tl.cumsum(match.to(tl.int32), 0)
            tl.store(base + rank - 1, col, match & (rank <= CAP))


def available(logits, rejected, width, cap):
    """Shapes the one-read select serves; anything else keeps the default select."""
    return (triton is not None and logits.is_cuda and logits.dtype == torch.bfloat16
            and logits.ndim == 2 and logits.is_contiguous() and rejected.is_contiguous()
            and rejected.shape == logits.shape[:1] and width <= 64
            and width < logits.shape[1] and cap <= width and cap & (cap - 1) == 0)


def select_topk(logits, rejected, width, *, copy=None, stats=None, argmax=None):
    """Radix top-``width`` candidates of the filtered probabilities, from one logits read.

    Returns (values, ids, before_top1, stats, argmax). ``stats`` is float32 [4, rows]:
    unfiltered max and sum-exp, filtered max and sum-exp. ``argmax`` is the unfiltered
    first argmax (int32). ``copy`` receives the BF16 logits when given."""
    rows, vocab = logits.shape
    device = logits.device
    chunks = triton.cdiv(vocab, TILE)
    block = triton.next_power_of_2(chunks)
    scratch = lambda shape: torch.empty(shape, device=device, dtype=torch.int32)
    histogram = scratch((rows, chunks, 256))
    totals, prefix = scratch((rows, 256)), scratch((rows, 3))
    counts, offsets, above = scratch((rows, chunks, 2)), scratch((rows, chunks, 2)), scratch((rows,))
    pivots, winners = scratch((rows, chunks)), scratch((rows, chunks))
    parts = torch.empty((rows, chunks, 4), device=device, dtype=torch.float32)
    stats = torch.empty((4, rows), device=device, dtype=torch.float32) if stats is None else stats
    argmax = scratch((rows,)) if argmax is None else argmax
    before_top1 = torch.empty((rows,), device=device, dtype=torch.bool)
    values = torch.empty((rows, width), device=device, dtype=torch.float32)
    ids = torch.empty((rows, width), device=device, dtype=torch.long)
    _stats[(rows, chunks)](logits, rejected, logits if copy is None else copy, parts, winners,
                           histogram, vocab, chunks, TILE, copy is not None,
                           num_warps=4, enable_fp_fusion=False)
    _radix_totals_coalesced[(rows, 256 // BIN_TILE)](histogram, totals, chunks, block, BIN_TILE,
                                                    num_warps=4)
    _stats_finish[(rows,)](parts, winners, totals, prefix, rejected, before_top1, stats, argmax,
                           rows, chunks, block, width, num_warps=4, enable_fp_fusion=False)
    _radix_hist[(rows, chunks)](logits, rejected, prefix, histogram, vocab, chunks, TILE, 1, True,
                                num_warps=4)
    _radix_totals_coalesced[(rows, 256 // BIN_TILE)](histogram, totals, chunks, block, BIN_TILE,
                                                    num_warps=4)
    _radix_threshold[(rows,)](totals, prefix, width, 1, 16, num_warps=4)
    _radix_find_pivot[(rows, chunks)](logits, rejected, prefix, pivots, vocab, chunks, TILE,
                                      num_warps=4)
    _cutoff[(rows,)](logits, stats, pivots, prefix, rows, vocab, chunks, block,
                     num_warps=4, enable_fp_fusion=False)
    _counts[(rows, chunks)](logits, rejected, stats, prefix, counts, rows, vocab, chunks, TILE,
                            num_warps=4, enable_fp_fusion=False)
    _radix_offsets[(rows,)](counts, offsets, above, chunks, block, num_warps=4)
    _emit[(rows, chunks)](logits, rejected, stats, prefix, offsets, above, values, ids,
                          rows, vocab, chunks, width, TILE, num_warps=4, enable_fp_fusion=False)
    return values, ids, before_top1, stats, argmax


def repair_ties(logits, rejected, stats, ids, p64, pack, cap):
    """anchor_select_fused.repair_ties with the probabilities computed from the logits."""
    from jetspec.engines.jetspec_plus_plus.jetspec.heads.anchor_select_fused import _tie_merge
    rows, vocab = logits.shape
    width = ids.shape[1]
    tile = 2048
    chunks = triton.cdiv(vocab, tile)
    candidates = torch.empty((rows, chunks, cap), device=ids.device, dtype=torch.int32)
    _tie_candidates[(rows, chunks)](
        logits, rejected, stats, p64, pack[4], candidates,
        rows, vocab, width, cap, chunks, tile, num_warps=4, enable_fp_fusion=False)
    _tie_merge[(rows,)](
        ids, p64, pack[4], candidates, width, cap, chunks,
        triton.next_power_of_2(chunks * cap), num_warps=4)
    return ids, p64


def reference(logits, rejected):
    """CPU FP32 emulation of the level-2 arithmetic, for tests: the filtered probabilities
    p = exp(l - m) / s (d_rej at -1) and the unfiltered (max, argmax, sum-exp)."""
    values = logits.float()
    rows, vocab = values.shape
    col = torch.arange(vocab)
    filtered = values.masked_fill(col[None, :] == rejected[:, None], float('-inf'))
    chunks = [(start, min(start + TILE, vocab)) for start in range(0, vocab, TILE)]

    def merged(source):
        tops = torch.stack([source[:, a:b].amax(-1) for a, b in chunks], 1)
        top = tops.amax(-1)
        sums = torch.stack([(source[:, a:b] - torch.where(tops[:, i:i + 1].isinf(), 0.,
                                                           tops[:, i:i + 1])).exp().sum(-1)
                            for i, (a, b) in enumerate(chunks)], 1)
        return top, (sums * (tops - top[:, None]).exp()).sum(-1)

    top, total = merged(values)
    kept_top, kept_total = merged(filtered)
    p = (filtered - kept_top[:, None]).exp() / kept_total[:, None]
    p = torch.where(col[None, :] == rejected[:, None], torch.full_like(p, -1.0), p)
    return p, top, values.argmax(-1), total
