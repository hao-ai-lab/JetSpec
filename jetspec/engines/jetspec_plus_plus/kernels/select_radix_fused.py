"""T5: six-launch BF16 radix select built on T4's split-row statistics.

Each histogram reduction also chooses its threshold. Pass 1 visits only the
winning high byte. Emit computes exp(l-m)/s and retains the best WIDTH keys per
chunk, including probability ties outside the winning logit bucket (underflow
can tie different BF16 logits). The final anchor post merges those keys in
(probability descending, id ascending) order, so no tie scan/repair is needed.

Parity: tau gate against torch softmax, with T4's 64 FP32-ULP probability bound.
The statistics arithmetic is T4's; the radix restructuring adds no rounding.
The larger final merge is a timing tradeoff that requires a served A/B gate.
"""
import torch

from jetspec.engines.jetspec_plus_plus.kernels import select_stats

try:
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice
except ImportError:
    triton = None


if triton is not None:
    from jetspec.engines.jetspec_plus_plus.kernels.decoder_select import _radix_hist, _bf16_key
    from jetspec.engines.jetspec_plus_plus.kernels.select_stats import _stats, _probability

    @triton.jit
    def _hist_threshold(HIST, PREFIX, row, C: tl.constexpr, BLOCK: tl.constexpr,
                        WIDTH: tl.constexpr, PASS: tl.constexpr, SLAB: tl.constexpr = 0,
                        ATOMIC: tl.constexpr = False):
        slot = tl.arange(0, 256)
        if ATOMIC:
            count = tl.load(HIST + row * 256 + 255 - slot)
        elif SLAB:
            count = tl.full((256,), 0, tl.int32)
            for start in tl.static_range((C + SLAB - 1) // SLAB):
                chunk = start * SLAB + tl.arange(0, SLAB)
                counts = tl.load(HIST + (row * C + chunk[:, None]) * 256 + 255 - slot[None, :],
                                 chunk[:, None] < C, other=0)
                count += tl.sum(counts, 0)
        else:
            chunk = tl.arange(0, BLOCK)
            counts = tl.load(HIST + (row * C + chunk[:, None]) * 256 + 255 - slot[None, :],
                             chunk[:, None] < C, other=0)
            count = tl.sum(counts, 0)
        rank = tl.load(PREFIX + row * 3 + 1) if PASS else WIDTH
        cumulative = tl.cumsum(count, 0)
        crossing = tl.min(tl.where(cumulative >= rank, slot, 256), 0)
        bucket = (255 - crossing) << (8 - PASS * 8)
        if PASS:
            bucket = bucket | tl.load(PREFIX + row * 3)
        above = tl.sum(tl.where(slot < crossing, count, 0), 0)
        tl.store(PREFIX + row * 3, bucket)
        tl.store(PREFIX + row * 3 + 1, rank - above)
        return bucket

    @triton.jit
    def _finish0(PARTS, WINNERS, HIST, PREFIX, REJECTED, TOP1, STATS, ARGMAX,
                 ROWS: tl.constexpr, C: tl.constexpr, BLOCK: tl.constexpr,
                 WIDTH: tl.constexpr, SLAB: tl.constexpr = 0, ATOMIC: tl.constexpr = False):
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
        _hist_threshold(HIST, PREFIX, row, C, BLOCK, WIDTH, 0, SLAB, ATOMIC)

    @triton.jit
    def _finish1(HIST, PREFIX, STATS, ROWS: tl.constexpr, C: tl.constexpr,
                 BLOCK: tl.constexpr, WIDTH: tl.constexpr, SLAB: tl.constexpr = 0,
                 COUNTER=0, COMPACT: tl.constexpr = 0, ATOMIC: tl.constexpr = False):
        row = tl.program_id(0)
        key = _hist_threshold(HIST, PREFIX, row, C, BLOCK, WIDTH, 1, SLAB, ATOMIC)
        # Invert the BF16 ordered key directly: no pivot-search or gather launch.
        bits = tl.where((key & 32768) != 0, key ^ 32768, (~key) & 65535)
        value = bits.to(tl.int16).to(tl.bfloat16, bitcast=True).to(tl.float32)
        p = _probability(value, tl.load(STATS + 2 * ROWS + row),
                         tl.load(STATS + 3 * ROWS + row))
        tl.store(PREFIX + row * 3 + 2, p.to(tl.int32, bitcast=True))
        if COMPACT:
            tl.store(COUNTER + row, 0)

    @triton.jit
    def _radix_hist_atomic(LOGITS, REJECTED, TOTAL0, TOTAL1,
                           V: tl.constexpr, TILE: tl.constexpr, WIDTH: tl.constexpr):
        row, chunk = tl.program_id(0), tl.program_id(1)
        slot = tl.arange(0, 256)
        count = tl.load(TOTAL0 + row * 256 + 255 - slot)
        crossing = tl.min(tl.where(tl.cumsum(count, 0) >= WIDTH, slot, 256), 0)
        bucket = 255 - crossing
        col = chunk * TILE + tl.arange(0, TILE)
        value = tl.load(LOGITS + row * V + col, col < V, other=0.)
        key = _bf16_key(value)
        live = (col < V) & (col != tl.load(REJECTED + row)) & ((key >> 8) == bucket)
        counts = tl.histogram(key & 255, 256, mask=live)
        tl.atomic_add(TOTAL1 + row * 256 + slot, counts, counts > 0, sem='relaxed')

    @triton.jit
    def _finish_atomic(PARTS, WINNERS, TOTAL0, TOTAL1, PREFIX, REJECTED, TOP1,
                       STATS, ARGMAX, COUNTER,
                       ROWS: tl.constexpr, C: tl.constexpr, BLOCK: tl.constexpr,
                       WIDTH: tl.constexpr, SMALL: tl.constexpr):
        # Inlining preserves _finish0's BLOCK and eight-warp FP32 reduction tree.
        _finish0(PARTS, WINNERS, TOTAL0, PREFIX, REJECTED, TOP1, STATS, ARGMAX,
                 ROWS, C, BLOCK, WIDTH, ATOMIC=True)
        # The two inlined stages communicate through PREFIX and STATS. Their
        # layouts can put the scalar store and reload in different warps.
        tl.debug_barrier()
        _finish1(TOTAL1, PREFIX, STATS, ROWS, C, BLOCK, WIDTH,
                 COUNTER=COUNTER, COMPACT=SMALL, ATOMIC=True)
        row = tl.program_id(0)
        tl.store(TOTAL0 + row * 256 + tl.arange(0, 256), 0)

    @triton.jit
    def _emit(LOGITS, REJECTED, STATS, PREFIX, KEYS,
              ROWS: tl.constexpr, V: tl.constexpr, C: tl.constexpr,
              TOP: tl.constexpr, TILE: tl.constexpr, ID_BITS: tl.constexpr,
              ID_MASK: tl.constexpr, COUNTER=0, COMPACT_KEYS=0, SMALL: tl.constexpr = 0):
        row, chunk = tl.program_id(0), tl.program_id(1)
        slot = tl.arange(0, TOP)
        base = KEYS + (row * C + chunk) * TOP
        col = chunk * TILE + tl.arange(0, TILE)
        value = tl.load(LOGITS + row * V + col, col < V, other=0.).to(tl.float32)
        p = _probability(value, tl.load(STATS + 2 * ROWS + row),
                         tl.load(STATS + 3 * ROWS + row))
        cutoff = tl.load(PREFIX + row * 3 + 2)
        bits = p.to(tl.int32, bitcast=True)
        live = (col < V) & (col != tl.load(REJECTED + row)) & (bits >= cutoff)
        count = tl.sum(live.to(tl.int32), 0)
        key = (bits.to(tl.int64) << ID_BITS) | (ID_MASK - col)
        if count <= TOP:
            tl.store(base + slot, -9223372036854775807, slot >= count)
            index = tl.cumsum(live.to(tl.int32), 0) - 1
            tl.store(base + index, key, live)
            if SMALL:
                if count > 0:
                    old = tl.atomic_add(COUNTER + row, count, sem='relaxed')
                    tl.store(COMPACT_KEYS + row * SMALL + old + index, key,
                             live & (old + index < SMALL))
        else:
            # Includes all-uniform rows and rounded zero ties. Sorting by p then
            # id here and in _post retains the global lowest IDs at every tie.
            ordered = tl.topk(tl.where(live, key, -9223372036854775807), TOP)
            tl.store(base + slot, ordered)
            if SMALL:
                tl.atomic_add(COUNTER + row, SMALL + 1, sem='relaxed')


def select(logits, rejected, q_a, q_b, cap, width, *, copy=None, stats=None,
           argmax=None, out=None, totals=None, counter=None, debug=False):
    """Return (ids, p64, pack, stats, argmax) using six capture-safe launches.

    Caller checks select_stats.available and supplies the optional T4 handoffs.
    T4 levels 1/2 never call this function unless draft_fuse_t5 is selected.
    Atomic totals use five launches with the chain's persistent zeroed totals;
    debug=True checks the zero invariant before the first histogram update.
    """
    from jetspec.engines.jetspec_plus_plus.draft.heads.anchor_select_fused import select_post
    from jetspec.engines.jetspec_plus_plus.profiles import kernel_value
    slab = 8 if kernel_value('draft_fuse_sf', '0') == '1' else 0
    mode = kernel_value('draft_fuse_sel', '0')
    small = 128 if mode in ('1', '3') else 0
    atomic = mode in ('2', '3')
    rows, vocab = logits.shape
    device = logits.device
    tile = select_stats.TILE
    chunks = triton.cdiv(vocab, tile)
    block = triton.next_power_of_2(chunks)
    scratch = lambda shape: torch.empty(shape, device=device, dtype=torch.int32)
    hist = scratch((rows, 256) if atomic else (rows, chunks, 256))
    if atomic:
        # The served chain supplies zeros allocated outside capture. Standalone
        # callers initialize their own buffer; never trust graph-pool empty memory.
        if totals is None:
            totals = torch.zeros((rows, 256), device=device, dtype=torch.int32)
        if debug:
            torch._assert_async((totals == 0).all(), 'T5 pass-0 histogram was not reset')
    if small:
        counter = scratch((rows,)) if counter is None else counter
        compact = torch.empty((rows, small), device=device, dtype=torch.int64)
    prefix, winners = scratch((rows, 3)), scratch((rows, chunks))
    parts = torch.empty((rows, chunks, 4), device=device, dtype=torch.float32)
    stats = torch.empty((4, rows), device=device) if stats is None else stats
    argmax = scratch((rows,)) if argmax is None else argmax
    top1 = torch.empty((rows,), device=device, dtype=torch.bool)
    hist_warps = 4
    emit_tile = 4096
    if ((rows, vocab, width, cap) == (16, 248320, 49, 32) and mode == '3'
            and kernel_value('draft_fuse_head27', '0') == '1'):
        # Class-A segment gate: smaller emit CTAs preserve every probability
        # and id while reducing the 27B vocabulary scan's critical wave time.
        emit_tile = 1024
    if ((rows, vocab, width, cap) == (16, 151936, 49, 32) and mode == '3'
            and kernel_value('draft_fuse_head8', '0') == '1'):
        # Only launch geometry changes; stats retain their served reduction tree.
        emit_tile = 1024
        hist_warps = 2
    emit_chunks = triton.cdiv(vocab, emit_tile)
    top = triton.next_power_of_2(width)
    keys = torch.empty((rows, emit_chunks * top), device=device, dtype=torch.int64)
    values = torch.empty((rows, width), device=device, dtype=torch.float32)
    ids = torch.empty((rows, width), device=device, dtype=torch.long)
    id_bits = max(1, (vocab - 1).bit_length())
    if atomic:
        _stats[(rows, chunks)](logits, rejected, logits if copy is None else copy, parts, winners,
                               totals, vocab, chunks, tile, copy is not None,
                               TOTAL1=hist, COUNTER=counter if small else 0, ATOMIC=True,
                               COMPACT=small, num_warps=4, enable_fp_fusion=False)
        _radix_hist_atomic[(rows, chunks)](logits, rejected, totals, hist, vocab, tile, width,
                                          num_warps=hist_warps)
        _finish_atomic[(rows,)](parts, winners, totals, hist, prefix, rejected, top1, stats,
                                argmax, counter if small else 0, rows, chunks, block, width,
                                small, num_warps=8, enable_fp_fusion=False)
    else:
        _stats[(rows, chunks)](logits, rejected, logits if copy is None else copy, parts, winners,
                               hist, vocab, chunks, tile, copy is not None,
                               num_warps=4, enable_fp_fusion=False)
        _finish0[(rows,)](parts, winners, hist, prefix, rejected, top1, stats, argmax,
                          rows, chunks, block, width, SLAB=slab,
                          num_warps=8, enable_fp_fusion=False)
        _radix_hist[(rows, chunks)](logits, rejected, prefix, hist, vocab, chunks, tile, 1, True,
                                    num_warps=4)
        _finish1[(rows,)](hist, prefix, stats, rows, chunks, block, width,
                          SLAB=slab, COUNTER=counter if small else 0, COMPACT=small,
                          num_warps=8, enable_fp_fusion=False)
    _emit[(rows, emit_chunks)](logits, rejected, stats, prefix, keys, rows, vocab,
                               emit_chunks, top, emit_tile, id_bits, (1 << id_bits) - 1,
                               COUNTER=counter if small else 0,
                               COMPACT_KEYS=compact if small else 0, SMALL=small,
                               num_warps=4, enable_fp_fusion=False)
    ids, p64, pack = select_post(values, ids, top1, q_a, q_b, cap, vocab,
                                 candidate_keys=keys, out=out,
                                 compact_keys=compact if small else None,
                                 compact_count=counter if small else None)
    return ids, p64, pack, stats, argmax


def reference(logits, rejected, width):
    """CPU oracle for the two radix thresholds and the probability-key emit/merge."""
    probabilities, *_ = select_stats.reference(logits, rejected)
    rows, vocab = logits.shape
    raw = logits.view(torch.int16).int() & 65535
    ordered = torch.where(raw & 32768 != 0, ~raw & 65535, raw ^ 32768)
    live = torch.arange(vocab)[None, :] != rejected[:, None]
    cutoffs = []
    for row in range(rows):
        hi = ordered[row] >> 8
        totals = torch.bincount(hi[live[row]], minlength=256)
        crossing = int((totals.flip(0).cumsum(0) >= width).nonzero()[0])
        bucket = 255 - crossing
        rank = width - int(totals[bucket + 1:].sum())
        lo = ordered[row] & 255
        totals = torch.bincount(lo[live[row] & (hi == bucket)], minlength=256)
        crossing = int((totals.flip(0).cumsum(0) >= rank).nonzero()[0])
        pivot = (live[row] & (ordered[row] == (bucket << 8 | (255 - crossing)))).nonzero()[0]
        cutoffs.append(probabilities[row, pivot].reshape(()))
    bits = max(1, (vocab - 1).bit_length())
    mask = (1 << bits) - 1
    keys = (probabilities.view(torch.int32).long() << bits) | (mask - torch.arange(vocab))
    candidates = []
    for start in range(0, vocab, 4096):
        end = min(start + 4096, vocab)
        keep = live[:, start:end] & (probabilities[:, start:end] >= torch.stack(cutoffs)[:, None])
        local = keys[:, start:end].masked_fill(~keep, -9223372036854775807)
        candidates.append(local.topk(min(width, end - start), -1).values)
    selected = torch.cat(candidates, -1).topk(width, -1).values
    return (selected >> bits).int().view(torch.float32), mask - (selected & mask)
