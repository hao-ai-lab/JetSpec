"""Vocab-tiled BF16 head with fp32 online-softmax partials for chain passes."""
from __future__ import annotations

import torch
from typing import NamedTuple

from jetspec.engines.jetspec_plus_plus.wire.chain.contract import Offset
import jetspec.engines.jetspec_plus_plus.kernels.chain.refine_k3 as refine_k3

try:
    from jetspec.engines.jetspec_plus_plus._optional import triton
    from jetspec.engines.jetspec_plus_plus._optional import tl
except ImportError:  # CPU-only gate
    triton = None
    tl = None


VOCAB_TILE = 128




if triton is not None:
    @triton.jit
    def _head(H, W, PMAX, PSUM, PIDX,
              M: tl.constexpr, V: tl.constexpr, K: tl.constexpr,
              HS0: tl.constexpr, HS1: tl.constexpr,
              WS0: tl.constexpr, WS1: tl.constexpr,
              C: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
        rows = tl.program_id(0) * BM + tl.arange(0, BM)
        cols = tl.program_id(1) * BN + tl.arange(0, BN)
        depth = tl.arange(0, BK)
        acc = tl.full((BM, BN), 0, tl.float32)
        for block in range(tl.cdiv(K, BK)):
            k = block * BK + depth
            h = tl.load(H + rows[:, None] * HS0 + k[None, :] * HS1,
                        (rows[:, None] < M) & (k[None, :] < K), other=0)
            w = tl.load(W + cols[:, None] * WS0 + k[None, :] * WS1,
                        (cols[:, None] < V) & (k[None, :] < K), other=0)
            acc = tl.dot(h, tl.trans(w), acc)
        # nn.Linear returns BF16 logits. Round before argmax and fp32 softmax.
        value = acc.to(tl.bfloat16).to(tl.float32)
        value = tl.where(cols[None, :] < V, value, -float('inf'))
        maximum = tl.max(value, 1)
        total = tl.sum(tl.exp(value - maximum[:, None]), 1)
        index = tl.min(tl.where(value == maximum[:, None], cols[None, :],
                                2147483647), 1)
        out = rows * C + tl.program_id(1)
        tl.store(PMAX + out, maximum, rows < M)
        tl.store(PSUM + out, total, rows < M)
        tl.store(PIDX + out, index, rows < M)


    @triton.jit
    def _merge(PMAX, PSUM, PIDX, TOKEN, PTOP1,
               C: tl.constexpr, PC: tl.constexpr):
        row = tl.program_id(0)
        chunk = tl.arange(0, PC)
        maximum_part = tl.load(PMAX + row * C + chunk, chunk < C,
                               other=-float('inf'))
        sum_part = tl.load(PSUM + row * C + chunk, chunk < C, other=0)
        index_part = tl.load(PIDX + row * C + chunk, chunk < C,
                             other=2147483647)
        maximum = tl.max(maximum_part, 0)
        denominator = tl.sum(sum_part * tl.exp(maximum_part - maximum), 0)
        token = tl.min(tl.where(maximum_part == maximum, index_part,
                                2147483647), 0)
        tl.store(TOKEN + row, token)
        tl.store(PTOP1 + row, 1.0 / denominator)


def allocate_partials(rows, lanes, vocab, device):
    chunks = (vocab + VOCAB_TILE - 1) // VOCAB_TILE
    shape = (rows, lanes, chunks)
    return (torch.empty(shape, device=device, dtype=torch.float32),
            torch.empty(shape, device=device, dtype=torch.float32),
            torch.empty(shape, device=device, dtype=torch.int32))


def head_partials(hidden, weight, partials):
    """Write only three scalars per row and vocabulary tile, never logits."""
    if triton is None or hidden.device.type != 'cuda':
        raise RuntimeError('fused lm_head requires CUDA Triton')
    if hidden.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        raise ValueError('fused lm_head requires BF16 hidden states and weight')
    rows, lanes, width = hidden.shape
    vocab, head_width = weight.shape
    if width != head_width or partials[0].shape != (rows, lanes, (vocab + VOCAB_TILE - 1) // VOCAB_TILE):
        raise ValueError('fused lm_head shape mismatch')
    flat = hidden.reshape(-1, width)
    chunks = partials[0].shape[-1]
    _head[(triton.cdiv(rows * lanes, 64), chunks)](
        flat, weight, *partials, rows * lanes, vocab, width,
        *flat.stride(), *weight.stride(), chunks, 64, VOCAB_TILE, 64,
        num_warps=4, num_stages=4)




def chain_pass(hidden, weight, ids, keys, active, outputs, previous_top1,
               tokens, rp, partials, index, *, save_previous, pins=None):
    """Project and finish the existing K3 chain contract in two launches.

    ``pins``: legacy ``(slot [R], token [R])`` or policy ``(mask [R,L], tokens [R,L])``
    as in ``refine_k3.reference_pass``; default None."""
    head_partials(hidden, weight, partials)
    rows, lanes = ids.shape
    chunks = partials[0].shape[-1]
    refine_k3._finish[(rows,)](
        *partials, ids, keys, active, outputs['finished'], outputs['passes_run'],
        outputs['final_tokens'], outputs['final_p_top1'], previous_top1,
        outputs['meta'], tokens.committed_ids, tokens.pending_draft_ids, rp, keys,
        *(pins if pins is not None else (keys, keys)),
        lanes, chunks, triton.next_power_of_2(chunks), index, save_previous,
        False, int(Offset.L), pins is not None,
        pins is not None and pins[0].ndim == 2, num_warps=16)


class DecoderEpilogue(NamedTuple):
    ids: torch.Tensor
    p64: torch.Tensor
    pack: torch.Tensor
    rejected_probability: torch.Tensor
    rejected_was_top1: torch.Tensor




if triton is not None:
    @triton.jit
    def _decoder_logits_partials(LOGITS, R, RQ, PMAX, PSUM, RAWMAX, RAWIDX, RLOGIT, TOP,
                                 V: tl.constexpr, C: tl.constexpr, WIDTH: tl.constexpr,
                                 BLOCK: tl.constexpr, LOCAL_BITS: tl.constexpr,
                                 TOPK: tl.constexpr, PREFILTER: tl.constexpr):
        row = tl.program_id(0)
        chunk = tl.program_id(1)
        local = tl.arange(0, BLOCK)
        col = chunk * BLOCK + local
        value = tl.load(LOGITS + row * V + col, col < V,
                        other=-float('inf')).to(tl.float32)
        rejected = tl.load(R + row)
        score_rejected = tl.load(RQ + row)
        raw_max = tl.max(value, 0)
        raw_idx = tl.min(tl.where(value == raw_max, col, 2147483647), 0)
        rejected_logit = tl.max(tl.where(col == tl.maximum(score_rejected, 0),
                                             value, -float('inf')), 0)
        filtered = tl.where(col == rejected, -float('inf'), value)
        maximum = tl.max(filtered, 0)
        safe_maximum = tl.where(maximum == -float('inf'), 0.0, maximum)
        total = tl.sum(tl.exp(filtered - safe_maximum), 0)
        out = row * C + chunk
        tl.store(PMAX + out, maximum)
        tl.store(PSUM + out, total)
        tl.store(RAWMAX + out, raw_max)
        tl.store(RAWIDX + out, raw_idx)
        tl.store(RLOGIT + out, rejected_logit)
        if not PREFILTER:
            bits = filtered.to(tl.bfloat16).to(tl.uint16, bitcast=True).to(tl.uint32)
            ordered = tl.where((bits & 0x8000) != 0, bits ^ 0xffff, bits ^ 0x8000)
            key = (ordered << LOCAL_BITS) | (BLOCK - 1 - local)
            key = tl.where((col < V) & (col != rejected), key, 0)
            selected = tl.topk(key, k=TOPK, dim=0)
            slot = tl.arange(0, TOPK)
            tl.store(TOP + out * WIDTH + slot, selected, slot < WIDTH)


    @triton.jit
    def _decoder_chunk_threshold(PMAX, THRESHOLD, C: tl.constexpr,
                                 PC: tl.constexpr, WIDTH: tl.constexpr,
                                 TOPK: tl.constexpr):
        row = tl.program_id(0)
        chunk = tl.arange(0, PC)
        maxima = tl.load(PMAX + row * C + chunk, chunk < C, other=-float('inf'))
        ranked = tl.topk(maxima, k=TOPK, dim=0)
        slot = tl.arange(0, TOPK)
        threshold = tl.max(tl.where(slot == WIDTH - 1, ranked, -float('inf')), 0)
        tl.store(THRESHOLD + row, threshold)


    @triton.jit
    def _decoder_prefilter_topk(LOGITS, R, PMAX, THRESHOLD, TOP,
                                V: tl.constexpr, C: tl.constexpr, WIDTH: tl.constexpr,
                                BLOCK: tl.constexpr, LOCAL_BITS: tl.constexpr,
                                TOPK: tl.constexpr):
        row = tl.program_id(0)
        chunk = tl.program_id(1)
        out = row * C + chunk
        slot = tl.arange(0, TOPK)
        if tl.load(PMAX + out) >= tl.load(THRESHOLD + row):
            local = tl.arange(0, BLOCK)
            col = chunk * BLOCK + local
            value = tl.load(LOGITS + row * V + col, col < V,
                            other=-float('inf')).to(tl.float32)
            rejected = tl.load(R + row)
            filtered = tl.where(col == rejected, -float('inf'), value)
            bits = filtered.to(tl.bfloat16).to(tl.uint16, bitcast=True).to(tl.uint32)
            ordered = tl.where((bits & 0x8000) != 0, bits ^ 0xffff, bits ^ 0x8000)
            key = (ordered << LOCAL_BITS) | (BLOCK - 1 - local)
            key = tl.where((col < V) & (col != rejected), key, 0)
            selected = tl.topk(key, k=TOPK, dim=0)
            tl.store(TOP + out * WIDTH + slot, selected, slot < WIDTH)
        else:
            tl.store(TOP + out * WIDTH + slot, 0, slot < WIDTH)




