"""Opt-in BF16 decoder radix selection and retained selection experiments.

The retained experiments reproduce the measured selection comparison.
"""
import torch

try:
    from jetspec.engines.jetspec_plus_plus._optional import triton
    from jetspec.engines.jetspec_plus_plus._optional import tl
except ImportError:
    triton = None


if triton is not None:
    @triton.jit
    def _maxima(P, MAX, V: tl.constexpr, C: tl.constexpr, TILE: tl.constexpr):
        row, chunk = tl.program_id(0), tl.program_id(1)
        col = chunk * TILE + tl.arange(0, TILE)
        value = tl.load(P + row * V + col, col < V, other=-float('inf'))
        tl.store(MAX + row * C + chunk, tl.max(value, 0))

    @triton.jit
    def _threshold(MAX, THRESHOLD, C: tl.constexpr, WIDTH: tl.constexpr,
                   BLOCK: tl.constexpr, TOP: tl.constexpr):
        row = tl.program_id(0)
        col = tl.arange(0, BLOCK)
        values = tl.load(MAX + row * C + col, col < C, other=-float('inf'))
        selected = tl.topk(values, k=TOP, dim=0)
        slot = tl.arange(0, TOP)
        threshold = tl.max(tl.where(slot == WIDTH - 1, selected, -float('inf')), 0)
        tl.store(THRESHOLD + row, threshold)

    @triton.jit
    def _candidates(P, MAX, THRESHOLD, VALUES, IDS,
                    V: tl.constexpr, C: tl.constexpr, TILE: tl.constexpr,
                    BITS: tl.constexpr, TOP: tl.constexpr):
        row, chunk = tl.program_id(0), tl.program_id(1)
        slot = tl.arange(0, TOP)
        out = (row * C + chunk) * TOP
        threshold = tl.load(THRESHOLD + row)
        if threshold != threshold:
            # A non-finite softmax row must still fail the existing finite guard.
            tl.store(VALUES + out + slot, float('nan'))
            tl.store(IDS + out + slot, chunk * TILE + slot)
        else:
            if tl.load(MAX + row * C + chunk) >= threshold:
                local = tl.arange(0, TILE)
                col = chunk * TILE + local
                value = tl.load(P + row * V + col, col < V, other=-float('inf'))
                live = (col < V) & (value >= threshold)
                count = tl.sum(live.to(tl.int32), 0)
                if count <= TOP:
                    # Empty-tail stores and compaction stores address disjoint slots.
                    tl.store(VALUES + out + slot, -float('inf'), slot >= count)
                    tl.store(IDS + out + slot, 0, slot >= count)
                    index = tl.cumsum(live.to(tl.int32), 0) - 1
                    tl.store(VALUES + out + index, value, live)
                    tl.store(IDS + out + index, col, live)
                elif tl.max(value, 0) == threshold:
                    # A uniform overflowing chunk needs only its first TOP ids.
                    index = tl.cumsum(live.to(tl.int32), 0) - 1
                    keep = live & (index < TOP)
                    tl.store(VALUES + out + index, value, keep)
                    tl.store(IDS + out + index, col, keep)
                else:
                    # Concentrated or tied chunks still retain at least WIDTH winners.
                    bits = value.to(tl.int32, bitcast=True).to(tl.int64)
                    key = (bits << BITS) | (TILE - 1 - local)
                    key = tl.where(live, key, -9223372036854775807)
                    selected = tl.topk(key, k=TOP, dim=0)
                    p = (selected >> BITS).to(tl.int32).to(tl.float32, bitcast=True)
                    token = chunk * TILE + TILE - 1 - (selected & (TILE - 1))
                    tl.store(VALUES + out + slot, p)
                    tl.store(IDS + out + slot, token)
            else:
                tl.store(VALUES + out + slot, -float('inf'))
                tl.store(IDS + out + slot, 0)

    @triton.jit
    def _merge(VALUES, IDS, OUT_VALUES, OUT_IDS,
               COUNT: tl.constexpr, WIDTH: tl.constexpr, TOP: tl.constexpr,
               ID_BITS: tl.constexpr, ID_MASK: tl.constexpr, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        offset = tl.arange(0, BLOCK)
        p = tl.load(VALUES + row * COUNT + offset, offset < COUNT, other=0.)
        ids = tl.load(IDS + row * COUNT + offset, offset < COUNT, other=0)
        bits = p.to(tl.int32, bitcast=True).to(tl.int64)
        key = (bits << ID_BITS) | (ID_MASK - ids)
        key = tl.where(offset < COUNT, key, -9223372036854775807)
        selected = tl.topk(key, k=TOP, dim=0)
        slot = tl.arange(0, TOP)
        values = (selected >> ID_BITS).to(tl.int32).to(tl.float32, bitcast=True)
        tokens = ID_MASK - (selected & ID_MASK)
        tl.store(OUT_VALUES + row * WIDTH + slot, values, slot < WIDTH)
        tl.store(OUT_IDS + row * WIDTH + slot, tokens, slot < WIDTH)

    @triton.jit
    def _bf16_key(value):
        bits = value.to(tl.int16, bitcast=True).to(tl.int32) & 65535
        return tl.where((bits & 32768) != 0, (~bits) & 65535, bits ^ 32768)

    @triton.jit
    def _radix_hist(P, REJECTED, PREFIX, HIST,
                    V: tl.constexpr, C: tl.constexpr, TILE: tl.constexpr,
                    PASS: tl.constexpr, LOGIT_KEYS: tl.constexpr):
        row, chunk = tl.program_id(0), tl.program_id(1)
        col = chunk * TILE + tl.arange(0, TILE)
        value = tl.load(P + row * V + col, col < V, other=0.)
        if LOGIT_KEYS:
            key = _bf16_key(value)
            bits: tl.constexpr = 16
        else:
            key = value.to(tl.int32, bitcast=True)
            bits: tl.constexpr = 32
        live = (col < V) & (col != tl.load(REJECTED + row))
        if PASS > 0:
            prefix = tl.load(PREFIX + row * 3)
            live = live & ((key >> (bits - PASS * 8)) == (prefix >> (bits - PASS * 8)))
        bins = (key >> (bits - 8 - PASS * 8)) & 255
        counts = tl.histogram(bins, 256, mask=live)
        tl.store(HIST + (row * C + chunk) * 256 + tl.arange(0, 256), counts)

    @triton.jit
    def _radix_totals(HIST, TOTAL, C: tl.constexpr, BLOCK: tl.constexpr):
        row, bucket = tl.program_id(0), tl.program_id(1)
        chunk = tl.arange(0, BLOCK)
        count = tl.load(HIST + (row * C + chunk) * 256 + bucket, chunk < C, other=0)
        tl.store(TOTAL + row * 256 + bucket, tl.sum(count, 0))

    @triton.jit
    def _radix_totals_coalesced(HIST, TOTAL, C: tl.constexpr,
                                BLOCK: tl.constexpr, BIN_TILE: tl.constexpr):
        row, group = tl.program_id(0), tl.program_id(1)
        chunk = tl.arange(0, BLOCK)
        bucket = group * BIN_TILE + tl.arange(0, BIN_TILE)
        count = tl.load(HIST + (row * C + chunk[:, None]) * 256 + bucket[None, :],
                        chunk[:, None] < C, other=0)
        tl.store(TOTAL + row * 256 + bucket, tl.sum(count, 0))

    @triton.jit
    def _radix_threshold(TOTAL, PREFIX, WIDTH: tl.constexpr, PASS: tl.constexpr,
                         BITS: tl.constexpr):
        row = tl.program_id(0)
        slot = tl.arange(0, 256)
        count = tl.load(TOTAL + row * 256 + 255 - slot)
        rank = tl.load(PREFIX + row * 3 + 1) if PASS > 0 else WIDTH
        cumulative = tl.cumsum(count, 0)
        crossing = tl.min(tl.where(cumulative >= rank, slot, 256), 0)
        bucket = (255 - crossing) << (BITS - 8 - PASS * 8)
        if PASS > 0:
            bucket = bucket | tl.load(PREFIX + row * 3)
        above = tl.sum(tl.where(slot < crossing, count, 0), 0)
        tl.store(PREFIX + row * 3, bucket)
        tl.store(PREFIX + row * 3 + 1, rank - above)
        if PASS == BITS // 8 - 1:
            tl.store(PREFIX + row * 3 + 2, bucket)

    @triton.jit
    def _radix_find_pivot(LOGITS, REJECTED, PREFIX, PIVOTS,
                          V: tl.constexpr, C: tl.constexpr, TILE: tl.constexpr):
        row, chunk = tl.program_id(0), tl.program_id(1)
        col = chunk * TILE + tl.arange(0, TILE)
        value = tl.load(LOGITS + row * V + col, col < V, other=0.)
        key = _bf16_key(value)
        threshold = tl.load(PREFIX + row * 3 + 2)
        live = (col < V) & (col != tl.load(REJECTED + row)) & (key == threshold)
        tl.store(PIVOTS + row * C + chunk, tl.min(tl.where(live, col, V), 0))

    @triton.jit
    def _radix_probability_cutoff(P, PIVOTS, PREFIX,
                                  V: tl.constexpr, C: tl.constexpr, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        chunk = tl.arange(0, BLOCK)
        pivot = tl.min(tl.load(PIVOTS + row * C + chunk, chunk < C, other=V), 0)
        probability = tl.load(P + row * V + pivot)
        tl.store(PREFIX + row * 3 + 2, probability.to(tl.int32, bitcast=True))

    @triton.jit
    def _radix_counts(P, REJECTED, PREFIX, COUNTS,
                      V: tl.constexpr, C: tl.constexpr, TILE: tl.constexpr):
        row, chunk = tl.program_id(0), tl.program_id(1)
        col = chunk * TILE + tl.arange(0, TILE)
        value = tl.load(P + row * V + col, col < V, other=0.)
        key = value.to(tl.int32, bitcast=True)
        threshold = tl.load(PREFIX + row * 3 + 2)
        live = (col < V) & (col != tl.load(REJECTED + row))
        tl.store(COUNTS + (row * C + chunk) * 2,
                 tl.sum((live & (key > threshold)).to(tl.int32), 0))
        tl.store(COUNTS + (row * C + chunk) * 2 + 1,
                 tl.sum((live & (key == threshold)).to(tl.int32), 0))

    @triton.jit
    def _radix_offsets(COUNTS, OFFSETS, ABOVE, C: tl.constexpr, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        chunk = tl.arange(0, BLOCK)
        strict = tl.load(COUNTS + (row * C + chunk) * 2, chunk < C, other=0)
        tied = tl.load(COUNTS + (row * C + chunk) * 2 + 1, chunk < C, other=0)
        tl.store(OFFSETS + (row * C + chunk) * 2, tl.cumsum(strict, 0) - strict, chunk < C)
        tl.store(OFFSETS + (row * C + chunk) * 2 + 1, tl.cumsum(tied, 0) - tied, chunk < C)
        tl.store(ABOVE + row, tl.sum(strict, 0))

    @triton.jit
    def _radix_emit(P, REJECTED, PREFIX, OFFSETS, ABOVE, VALUES, IDS,
                    V: tl.constexpr, C: tl.constexpr, WIDTH: tl.constexpr, TILE: tl.constexpr):
        row, chunk = tl.program_id(0), tl.program_id(1)
        col = chunk * TILE + tl.arange(0, TILE)
        value = tl.load(P + row * V + col, col < V, other=0.)
        key = value.to(tl.int32, bitcast=True)
        threshold = tl.load(PREFIX + row * 3 + 2)
        live = (col < V) & (col != tl.load(REJECTED + row))
        strict, tied = live & (key > threshold), live & (key == threshold)
        strict_index = tl.load(OFFSETS + (row * C + chunk) * 2) + tl.cumsum(strict.to(tl.int32), 0) - 1
        tied_index = tl.load(OFFSETS + (row * C + chunk) * 2 + 1) + tl.cumsum(tied.to(tl.int32), 0) - 1
        above = tl.load(ABOVE + row)
        index = tl.where(strict, strict_index, above + tied_index)
        keep = strict | (tied & (tied_index < WIDTH - above))
        probability = tl.load(P + row * V + col, keep, other=0.)
        tl.store(VALUES + row * WIDTH + index, probability, keep)
        tl.store(IDS + row * WIDTH + index, col, keep)




def radix_topk(logits, probabilities, rejected, width, *, logit_pivot=False, bin_tile=1):
    """Select unchanged FP32 probabilities with an optional BF16 logit pivot.

    Probability ties are resolved by the existing select_post/repair_ties pair.
    The BF16 pivot uses two byte passes, then gathers its original probability.
    Counts and tie compaction compare probability bits, including rounded ties.
    Four probability-byte passes are retained for the comparison screens.
    """
    if bin_tile < 1 or bin_tile > 256 or bin_tile & (bin_tile - 1):
        raise ValueError('bin_tile must be a power of two between 1 and 256')
    if logits.dtype != torch.bfloat16:
        return probabilities.topk(width, dim=-1, sorted=False)
    rows, vocab = logits.shape
    if not logits.is_cuda or triton is None:
        return probabilities.topk(width, dim=-1, sorted=False)
    if not logits.is_contiguous() or width >= vocab:
        return probabilities.topk(width, dim=-1, sorted=False)
    tile = 1024
    chunks = triton.cdiv(vocab, tile)
    scratch = lambda shape: torch.empty(shape, device=logits.device, dtype=torch.int32)
    histogram = scratch((rows, chunks, 256))
    totals, prefix = scratch((rows, 256)), scratch((rows, 3))
    counts, offsets, above = scratch((rows, chunks, 2)), scratch((rows, chunks, 2)), scratch((rows,))
    values = torch.empty((rows, width), device=logits.device, dtype=torch.float32)
    ids = torch.empty((rows, width), device=logits.device, dtype=torch.long)
    bits = 16 if logit_pivot else 32
    source = logits if logit_pivot else probabilities
    for digit in range(bits // 8):
        _radix_hist[(rows, chunks)](source, rejected, prefix, histogram, vocab, chunks, tile, digit, logit_pivot, num_warps=4)
        if bin_tile == 1:
            _radix_totals[(rows, 256)](histogram, totals, chunks, triton.next_power_of_2(chunks), num_warps=4)
        else:
            _radix_totals_coalesced[(rows, 256 // bin_tile)](
                histogram, totals, chunks, triton.next_power_of_2(chunks), bin_tile, num_warps=4)
        _radix_threshold[(rows,)](totals, prefix, width, digit, bits, num_warps=4)
    if logit_pivot:
        pivots = scratch((rows, chunks))
        _radix_find_pivot[(rows, chunks)](logits, rejected, prefix, pivots, vocab, chunks, tile, num_warps=4)
        _radix_probability_cutoff[(rows,)](probabilities, pivots, prefix, vocab, chunks,
                                          triton.next_power_of_2(chunks), num_warps=4)
    _radix_counts[(rows, chunks)](probabilities, rejected, prefix, counts, vocab, chunks, tile, num_warps=4)
    _radix_offsets[(rows,)](counts, offsets, above, chunks, triton.next_power_of_2(chunks), num_warps=4)
    _radix_emit[(rows, chunks)](probabilities, rejected, prefix, offsets, above,
                              values, ids, vocab, chunks, width, tile, num_warps=4)
    return values, ids
