"""Fused candidate selection, threshold, and metadata for anchor selection.

The fp32 softmax remains torch's operation for bitwise probability parity.
Pinned CUDA multi-block top-k shapes use Triton; other cases retain torch top-k.
"""
from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None


_EAGER_KERNELS = {}




if triton is not None:




    @triton.jit
    def _mask_logits(logits, rejected, filtered, partial_max, partial_id,
                     VOCAB: tl.constexpr, CHUNKS: tl.constexpr, TILE: tl.constexpr,
                     NAN_ARGMAX: tl.constexpr = False):
        row = tl.program_id(0)
        chunk = tl.program_id(1)
        col = chunk * TILE + tl.arange(0, TILE)
        value = tl.load(logits + row * VOCAB + col, col < VOCAB, other=float('-inf'))
        rejected_id = tl.load(rejected + row)
        tl.store(filtered + row * VOCAB + col,
                 tl.where(col == rejected_id, float('-inf'), value), col < VOCAB)
        maximum = tl.max(value, 0)
        winner = tl.min(tl.where((col < VOCAB) & (value == maximum), col, 0x7FFFFFFF), 0)
        if NAN_ARGMAX:
            nan_id = tl.min(tl.where((col < VOCAB) & (value != value), col, 0x7FFFFFFF), 0)
            maximum = tl.where(nan_id != 0x7FFFFFFF, float('nan'), maximum)
            winner = tl.where(nan_id != 0x7FFFFFFF, nan_id, winner)
        tl.store(partial_max + row * CHUNKS + chunk, maximum)
        tl.store(partial_id + row * CHUNKS + chunk, winner)


    @triton.jit
    def _top1(partial_max, partial_id, rejected, before_top1,
              CHUNKS: tl.constexpr, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        chunk = tl.arange(0, BLOCK)
        maximums = tl.load(partial_max + row * CHUNKS + chunk,
                           chunk < CHUNKS, other=float('-inf'))
        ids = tl.load(partial_id + row * CHUNKS + chunk,
                      chunk < CHUNKS, other=0x7FFFFFFF)
        maximum = tl.max(maximums, 0)
        first_id = tl.min(tl.where(maximums == maximum, ids, 0x7FFFFFFF), 0)
        tl.store(before_top1 + row, first_id == tl.load(rejected + row))


    @triton.jit
    def _mark_rejected(probabilities, rejected, VOCAB: tl.constexpr):
        row = tl.program_id(0)
        rejected_id = tl.load(rejected + row)
        if rejected_id >= 0:
            address = probabilities + row * VOCAB + rejected_id
            p = tl.load(address)
            tl.store(address, p - 1.0)


    @triton.jit
    def _post(values, candidate_ids, top1, qa, qb, out_ids, out_p, out_pack,
              partial_max, partial_id, rejected, combined,
              ROWS: tl.constexpr, WIDTH: tl.constexpr, CAP: tl.constexpr,
              VOCAB: tl.constexpr, ID_BITS: tl.constexpr, ID_MASK: tl.constexpr,
              DEVICE_Q: tl.constexpr, Q_ROWS: tl.constexpr,
              Q_A: tl.constexpr, Q_B: tl.constexpr,
              BLOCK: tl.constexpr, TOP1_CHUNKS: tl.constexpr = 0,
              TOP1_BLOCK: tl.constexpr = 1, PACK_HOME: tl.constexpr = False,
              MERGE_COUNT: tl.constexpr = 0, MERGE_BLOCK: tl.constexpr = 1):
        row = tl.program_id(0)
        if TOP1_CHUNKS:
            chunk = tl.arange(0, TOP1_BLOCK)
            maxima = tl.load(partial_max + row * TOP1_CHUNKS + chunk,
                             chunk < TOP1_CHUNKS, other=float('-inf'))
            winners = tl.load(partial_id + row * TOP1_CHUNKS + chunk,
                              chunk < TOP1_CHUNKS, other=0x7FFFFFFF)
            nan_id = tl.min(tl.where(maxima != maxima, winners, 0x7FFFFFFF), 0)
            maximum = tl.max(maxima, 0)
            winner = tl.min(tl.where(maxima == maximum, winners, 0x7FFFFFFF), 0)
            winner = tl.where(nan_id != 0x7FFFFFFF, nan_id, winner)
            before_top1 = winner == tl.load(rejected + row)
        else:
            before_top1 = tl.load(top1 + row)
        col = tl.arange(0, BLOCK)
        live = col < WIDTH
        if MERGE_COUNT:
            offset = tl.arange(0, MERGE_BLOCK)
            key = tl.load(values + row * MERGE_COUNT + offset,
                          offset < MERGE_COUNT, other=-9223372036854775807)
            ordered = tl.topk(key, BLOCK)
            bits = (ordered >> ID_BITS).to(tl.int32)
        else:
            p = tl.load(values + row * WIDTH + col, live, other=0)
            ids = tl.load(candidate_ids + row * WIDTH + col, live, other=0)
            bits = p.to(tl.int32, bitcast=True).to(tl.int64)
            key = (bits << ID_BITS) | (ID_MASK - ids)
            key = tl.where(live, key, -9223372036854775807)
            ordered = tl.sort(key, descending=True)
        sorted_ids = ID_MASK - (ordered & ID_MASK)
        sorted_p = (ordered >> ID_BITS).to(tl.int32).to(tl.float32, bitcast=True)
        tl.store(out_ids + row * WIDTH + col, sorted_ids, live)
        tl.store(out_p + row * WIDTH + col, sorted_p.to(tl.float64), live)

        if DEVICE_Q:
            q_offset = row if Q_ROWS else 0
            q = tl.where(before_top1, tl.load(qa + q_offset), tl.load(qb + q_offset))
        else:
            q = tl.where(before_top1,
                         tl.full((), Q_A, tl.float64), tl.full((), Q_B, tl.float64))
        kept = sorted_p.to(tl.float64) >= (1.0 - q)
        size = tl.sum((kept & (col < CAP)).to(tl.int32), 0)
        cap_last = tl.sum(tl.where(col == CAP - 1, sorted_p, 0.0), 0)
        width_last = tl.sum(tl.where(col == WIDTH - 1, sorted_p, 0.0), 0)
        cap_next = tl.sum(tl.where(col == CAP, sorted_p, 0.0), 0)
        next_kept = tl.sum((kept & (col == CAP)).to(tl.int32), 0) > 0
        crossing = (WIDTH > CAP) & next_kept & (cap_last == cap_next)
        last_kept = tl.sum((kept & (col == CAP - 1)).to(tl.int32), 0) > 0
        inexact = (WIDTH < VOCAB) & last_kept & (width_last == cap_last)
        nonfinite = tl.sum((((bits & 0x7F800000) == 0x7F800000) & live).to(tl.int32), 0)
        tl.store(out_pack + row, size)
        tl.store(out_pack + ROWS + row, before_top1.to(tl.int64))
        tl.store(out_pack + 2 * ROWS + row, crossing.to(tl.int64))
        tl.store(out_pack + 3 * ROWS + row, (nonfinite == 0).to(tl.int64))
        tl.store(out_pack + 4 * ROWS + row, inexact.to(tl.int64))
        if PACK_HOME:
            tl.store(combined + row, size)
            tl.store(combined + ROWS + row, before_top1.to(tl.int64))
            tl.store(combined + 2 * ROWS + row, crossing.to(tl.int64))
            tl.store(combined + 3 * ROWS + row, (nonfinite == 0).to(tl.int64))
            tl.store(combined + 4 * ROWS + row, inexact.to(tl.int64))
            tl.store(combined + 5 * ROWS + row * WIDTH + col, sorted_ids, live)
            p_bits = sorted_p.to(tl.float64).to(tl.int64, bitcast=True)
            tl.store(combined + 5 * ROWS + ROWS * WIDTH + row * WIDTH + col, p_bits, live)


    @triton.jit
    def _tie_candidates(probabilities, selected_p, inexact, candidates,
                        VOCAB: tl.constexpr, WIDTH: tl.constexpr, CAP: tl.constexpr,
                        CHUNKS: tl.constexpr, TILE: tl.constexpr):
        row = tl.program_id(0)
        chunk = tl.program_id(1)
        if tl.load(inexact + row):
            slot = tl.arange(0, CAP)
            base = candidates + (row * CHUNKS + chunk) * CAP
            tl.store(base + slot, 0x7FFFFFFF)
            col = chunk * TILE + tl.arange(0, TILE)
            threshold = tl.load(selected_p + row * WIDTH + CAP - 1).to(tl.float32)
            p = tl.load(probabilities + row * VOCAB + col, col < VOCAB, other=-1.0)
            match = (col < VOCAB) & (p == threshold)
            rank = tl.cumsum(match.to(tl.int32), 0)
            tl.store(base + rank - 1, col, match & (rank <= CAP))


    @triton.jit
    def _tie_merge(selected_ids, selected_p, inexact, candidates,
                   WIDTH: tl.constexpr, CAP: tl.constexpr, CHUNKS: tl.constexpr,
                   BLOCK: tl.constexpr):
        row = tl.program_id(0)
        if tl.load(inexact + row):
            offset = tl.arange(0, BLOCK)
            candidate = tl.load(candidates + row * CHUNKS * CAP + offset,
                                offset < CHUNKS * CAP, other=0x7FFFFFFF)
            pos = tl.arange(0, CAP)
            p = tl.load(selected_p + row * WIDTH + pos)
            threshold = tl.load(selected_p + row * WIDTH + CAP - 1)
            higher = tl.sum((p > threshold).to(tl.int32), 0)
            valid = candidate != 0x7FFFFFFF
            rank = tl.cumsum(valid.to(tl.int32), 0)
            destination = higher + rank - 1
            tl.store(selected_ids + row * WIDTH + destination, candidate,
                     valid & (destination < CAP))


def select_post(values, ids, before_top1, q_a, q_b, cap, vocab, *,
                top1_partials=None, rejected=None, combined=None, candidate_keys=None,
                _kernels=None):
    if triton is None or values.device.type != "cuda":
        raise RuntimeError("fused anchor selection requires CUDA Triton")
    rows, width = values.shape
    id_bits = max(1, (vocab - 1).bit_length())
    out_ids = torch.empty_like(ids) if candidate_keys is None else ids
    out_p = torch.empty((rows, width), device=values.device, dtype=torch.float64)
    pack = torch.empty((5, rows), device=values.device, dtype=torch.long)
    device_q = isinstance(q_a, torch.Tensor)
    q_rows = device_q and q_a.numel() == rows
    chunks = 0 if top1_partials is None else top1_partials[0].shape[1]
    maxima, winners = (values, ids) if top1_partials is None else top1_partials
    args = (
        values if candidate_keys is None else candidate_keys,
        ids, values if before_top1 is None else before_top1,
        q_a if device_q else values, q_b if device_q else values,
        out_ids, out_p, pack, maxima, winners,
        ids if rejected is None else rejected, pack if combined is None else combined,
        rows, width, cap, vocab, id_bits,
        (1 << id_bits) - 1, device_q, q_rows,
        0.0 if device_q else q_a, 0.0 if device_q else q_b,
        triton.next_power_of_2(width), chunks, triton.next_power_of_2(max(1, chunks)),
        combined is not None,
        0 if candidate_keys is None else candidate_keys.shape[1],
        1 if candidate_keys is None else triton.next_power_of_2(candidate_keys.shape[1]))
    _post[(rows,)](*args, num_warps=4, enable_fp_fusion=False)
    return out_ids, out_p, pack






def prepare_probabilities(logits, rejected):
    rows, vocab = logits.shape
    chunks = triton.cdiv(vocab, 2048)
    filtered = torch.empty_like(logits)
    partial_max = torch.empty((rows, chunks), device=logits.device, dtype=torch.float32)
    partial_id = torch.empty((rows, chunks), device=logits.device, dtype=torch.int32)
    before_top1 = torch.empty((rows,), device=logits.device, dtype=torch.bool)
    _mask_logits[(rows, chunks)](
        logits, rejected, filtered, partial_max, partial_id,
        vocab, chunks, 2048, num_warps=4)
    _top1[(rows,)](partial_max, partial_id, rejected, before_top1,
                   chunks, triton.next_power_of_2(chunks), num_warps=4)
    probabilities = filtered.softmax(-1)
    _mark_rejected[(rows,)](probabilities, rejected, vocab, num_warps=4,
                            enable_fp_fusion=False)
    return probabilities, before_top1


def repair_ties(probabilities, ids, p64, pack, cap):
    """Replace tied cap suffix by the lowest ids in the full row, on device."""
    rows, vocab = probabilities.shape
    width = ids.shape[1]
    tile = 2048
    chunks = triton.cdiv(vocab, tile)
    candidates = torch.empty((rows, chunks, cap), device=ids.device, dtype=torch.int32)
    _tie_candidates[(rows, chunks)](
        probabilities, p64, pack[4], candidates,
        vocab, width, cap, chunks, tile, num_warps=4)
    _tie_merge[(rows,)](
        ids, p64, pack[4], candidates, width, cap, chunks,
        triton.next_power_of_2(chunks * cap), num_warps=4)
    return ids, p64
