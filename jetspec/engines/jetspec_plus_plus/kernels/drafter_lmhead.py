"""Vocab-tiled BF16 head with fp32 online-softmax partials for chain passes."""
from __future__ import annotations

import torch
from typing import NamedTuple

from jetspec.engines.jetspec_plus_plus.wire.contract import Offset
from jetspec.engines.jetspec_plus_plus.kernels import drafter_pass as refine_k3
from jetspec.engines.jetspec_plus_plus.kernels import register_optimization
from jetspec.engines.jetspec_plus_plus.draft.method.ops import reference
from jetspec.engines.jetspec_plus_plus.draft.method.passes import PassValues, apply_pins, jacobi_converged

try:
    import triton
    import triton.language as tl
except ImportError:  # CPU-only gate
    triton = None
    tl = None


VOCAB_TILE = 128


def reference_epilogue(logits, tile=VOCAB_TILE):
    """CPU oracle: chunked fp32 softmax and lowest-index argmax."""
    values = logits.float().reshape(-1, logits.shape[-1])
    maxima, sums, indices = [], [], []
    for start in range(0, values.shape[-1], tile):
        part = values[:, start:start + tile]
        maximum, local = part.max(-1)
        maxima.append(maximum)
        safe_maximum = torch.where(maximum == float('-inf'), 0.0, maximum)
        sums.append((part - safe_maximum[:, None]).exp().sum(-1))
        indices.append(local + start)
    maxima = torch.stack(maxima, -1)
    sums = torch.stack(sums, -1)
    indices = torch.stack(indices, -1)
    maximum = maxima.max(-1).values
    denominator = (sums * (maxima - maximum[:, None]).exp()).sum(-1)
    index = torch.where(maxima == maximum[:, None], indices,
                        torch.iinfo(indices.dtype).max).min(-1).values
    shape = logits.shape[:-1]
    return index.reshape(shape), (1.0 / denominator).reshape(shape)


if triton is not None:
    @triton.jit
    def _head(H, W, PMAX, PSUM, PIDX,
              M: tl.constexpr, V: tl.constexpr, K: tl.constexpr,
              HS0: tl.constexpr, HS1: tl.constexpr,
              WS0: tl.constexpr, WS1: tl.constexpr,
              C: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
              FROM_LOGITS: tl.constexpr = False, TOKEN_IDS=None,
              REDUCED: tl.constexpr = False):
        rows = tl.program_id(0) * BM + tl.arange(0, BM)
        cols = tl.program_id(1) * BN + tl.arange(0, BN)
        if FROM_LOGITS:
            value = tl.load(H + rows[:, None] * V + cols[None, :],
                            (rows[:, None] < M) & (cols[None, :] < V),
                            other=-float('inf')).to(tl.float32)
        else:
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
        safe_maximum = tl.where(maximum == -float('inf'), 0.0, maximum)
        total = tl.sum(tl.exp(value - safe_maximum[:, None]), 1)
        index = tl.min(tl.where(value == maximum[:, None], cols[None, :],
                                2147483647), 1)
        if REDUCED:
            index = tl.load(TOKEN_IDS + index, index < V, other=2147483647).to(tl.int32)
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
    """Write top-1 partials, using warmed W4A8 scratch logits under the head mode."""
    if triton is None or hidden.device.type != 'cuda':
        raise RuntimeError('fused lm_head requires CUDA Triton')
    if hidden.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        raise ValueError('fused lm_head requires BF16 hidden states and weight')
    rows, lanes, width = hidden.shape
    token_ids = None
    reduced = getattr(weight, '_draft_vocab', None)
    if reduced is not None:
        token_ids, weight = reduced
    vocab, head_width = weight.shape
    if width != head_width or partials[0].shape != (rows, lanes, (vocab + VOCAB_TILE - 1) // VOCAB_TILE):
        raise ValueError('fused lm_head shape mismatch')
    flat = hidden.reshape(-1, width)
    chunks = partials[0].shape[-1]
    from jetspec.engines.jetspec_plus_plus.models.target_linear import w4a8_head_logits
    logits = w4a8_head_logits(hidden, weight)
    _head[(triton.cdiv(rows * lanes, 64), chunks)](
        flat if logits is None else logits, weight, *partials, rows * lanes, vocab, width,
        *flat.stride(), *weight.stride(), chunks, 64, VOCAB_TILE, 64,
        FROM_LOGITS=logits is not None, TOKEN_IDS=token_ids,
        REDUCED=reduced is not None, num_warps=4, num_stages=4)


def head_top1(hidden, weight, partials, token, p_top1):
    """Two-launch head interface used by the GPU parity and timing check."""
    head_partials(hidden, weight, partials)
    rows = hidden.shape[0] * hidden.shape[1]
    chunks = partials[0].shape[-1]
    _merge[(rows,)](*partials, token, p_top1, chunks,
                    triton.next_power_of_2(chunks), num_warps=4)


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
        pins is not None and pins[0].ndim == 2,
        **(outputs.get('_t14_publication', {}) if index == 1 else {}),
        **(outputs.get('_t17_refresh', {}) if index == 0 else {}), num_warps=16)


def optimized_pass(hidden, input_tokens, pinned_first=None, pinned_second=None, pins=None,
                   *, p_top1=None, head_weight=None, partials=None):
    """The same pass op with an optional fused BF16 projection."""
    if head_weight is None:
        return refine_k3.optimized_pass(hidden, input_tokens, pinned_first, pinned_second,
                                        pins, p_top1=p_top1, partials=partials)
    if (triton is None or not hidden.is_cuda or p_top1 is not None
            or hidden.dtype != torch.bfloat16 or head_weight.dtype != torch.bfloat16):
        if getattr(head_weight, '_draft_vocab', None) is not None:
            from jetspec.engines.jetspec_plus_plus.models.target_linear import draft_linear
            hidden = draft_linear(hidden, head_weight)
            head_weight = None
        return reference('jacobi_pass')(hidden, input_tokens, pinned_first, pinned_second,
                                        pins, p_top1=p_top1, head_weight=head_weight)
    rows, lanes = input_tokens.shape
    if partials is None:
        reduced = getattr(head_weight, '_draft_vocab', None)
        vocab = head_weight.shape[0] if reduced is None else reduced[1].shape[0]
        partials = allocate_partials(rows, lanes, vocab, hidden.device)
    tokens = torch.empty_like(input_tokens)
    probabilities = torch.empty_like(input_tokens, dtype=torch.float32)
    head_top1(hidden, head_weight, partials, tokens, probabilities)
    if pinned_first is not None:
        tokens[..., 0] = pinned_first
    if pinned_second is not None:
        tokens[..., 1] = pinned_second
    if pins is not None:
        tokens = apply_pins(tokens, pins)
    return PassValues(tokens, jacobi_converged(tokens, input_tokens), probabilities)


register_optimization('jacobi_pass', optimized_pass)


class DecoderEpilogue(NamedTuple):
    ids: torch.Tensor
    p64: torch.Tensor
    pack: torch.Tensor
    rejected_probability: torch.Tensor
    rejected_was_top1: torch.Tensor


def decoder_reference_epilogue(logits, rejected, cap, *, score_rejected=None):
    """CPU oracle for the budget selector and BLEND decoder features."""
    values = logits.float().reshape(-1, logits.shape[-1])
    rejected = rejected.reshape(-1).long()
    score_rejected = (rejected if score_rejected is None
                      else score_rejected.reshape(-1).long())
    rows, vocab = values.shape
    if rejected.shape != (rows,) or score_rejected.shape != (rows,) or cap < 1 or cap >= vocab:
        raise ValueError('invalid decoder reference shapes or cap')
    top1 = values.argmax(-1) == rejected
    score_top1 = values.argmax(-1) == score_rejected
    q = values.softmax(-1).gather(1, score_rejected.clamp_min(0)[:, None])[:, 0]
    filtered = values.clone()
    valid = rejected >= 0
    filtered[torch.arange(rows)[valid], rejected[valid]] = float('-inf')
    probabilities = filtered.softmax(-1)
    width = cap + 1
    # Stable sort is the same value-descending, id-ascending total order as topk_select.
    ids = torch.argsort(probabilities, dim=-1, descending=True, stable=True)[:, :width]
    p64 = probabilities.gather(1, ids).double()
    crosses = (p64[:, cap - 1] == p64[:, cap]).long()
    pack = torch.stack((torch.full((rows,), cap, dtype=torch.long), top1.long(),
                        crosses, torch.ones(rows, dtype=torch.long),
                        torch.zeros(rows, dtype=torch.long)))
    return DecoderEpilogue(ids, p64, pack, q, score_top1)


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


def head_topk_logits(logits, rejected, cap, *, score_rejected=None, prefilter=False):
    """One BF16 logits read pass. After a miss, offers keep row zero while BLEND
    still gathers q for the original rejected token in score_rejected."""
    if triton is None or logits.device.type != 'cuda' or logits.dtype != torch.bfloat16:
        raise RuntimeError('batched decoder epilogue requires CUDA BF16 Triton')
    vocab = logits.shape[-1]
    rows = logits.numel() // vocab
    if (not logits.is_contiguous() or rejected.numel() != rows
            or cap < 1 or cap >= vocab):
        raise ValueError('batched decoder epilogue shape mismatch')
    rejected = rejected.reshape(rows)
    score_rejected = (rejected if score_rejected is None
                      else score_rejected.reshape(rows))
    if score_rejected.numel() != rows:
        raise ValueError('batched decoder BLEND rejected shape mismatch')
    tile = 1024 if vocab >= 1024 else triton.next_power_of_2(vocab)
    topk = triton.next_power_of_2(cap + 1)
    if topk > tile:
        raise ValueError('batched decoder epilogue cap exceeds a vocabulary tile')
    chunks = triton.cdiv(vocab, tile)
    candidates = cap + 1
    parts = [torch.empty((rows, chunks), device=logits.device, dtype=torch.float32)
             for _ in range(3)]
    raw_idx = torch.empty((rows, chunks), device=logits.device, dtype=torch.int32)
    rejected_logit = torch.empty_like(parts[0])
    keys = torch.empty((rows, chunks, candidates), device=logits.device, dtype=torch.int32)
    prefilter = prefilter and chunks >= candidates
    _decoder_logits_partials[(rows, chunks)](
        logits, rejected, score_rejected, parts[0], parts[1], parts[2], raw_idx,
        rejected_logit, keys, vocab, chunks, candidates, tile,
        tile.bit_length() - 1, topk, prefilter, num_warps=4)
    if prefilter:
        threshold = torch.empty((rows,), device=logits.device, dtype=torch.float32)
        _decoder_chunk_threshold[(rows,)](
            parts[0], threshold, chunks, triton.next_power_of_2(chunks),
            candidates, topk, num_warps=4)
        _decoder_prefilter_topk[(rows, chunks)](
            logits, rejected, parts[0], threshold, keys, vocab, chunks,
            candidates, tile, tile.bit_length() - 1, topk, num_warps=4)
    return _finish_decoder_partials(parts, raw_idx, rejected_logit, keys,
                                    rejected, score_rejected, cap, tile, vocab)


def _finish_decoder_partials(parts, raw_idx, rejected_logit, keys, rejected,
                             score_rejected, cap, tile, vocab):
    chunks = parts[0].shape[1]
    candidates = cap + 1
    maximum = parts[0].amax(-1)
    denominator = (parts[1] * torch.exp(parts[0] - maximum[:, None])).sum(-1)
    raw_max = parts[2].amax(-1)
    raw_token = torch.where(parts[2] == raw_max[:, None], raw_idx,
                            torch.iinfo(raw_idx.dtype).max).amin(-1)
    rejected_value = rejected_logit.amax(-1)
    # A valid offer exclusion is also BLEND's rejected id. With no exclusion,
    # denominator already covers the full vocabulary, including score_rejected.
    raw_denominator = (denominator * torch.exp(maximum - raw_max)
                       + torch.exp(rejected_value - raw_max))
    q_excluded = torch.exp(rejected_value - raw_max) / raw_denominator
    q_present = torch.exp(rejected_value - maximum) / denominator
    q = torch.where(rejected >= 0, q_excluded, q_present)

    local = keys.bitwise_and(tile - 1)
    index = (torch.arange(chunks, device=keys.device)[None, :, None] * tile
             + tile - 1 - local)
    ordered = keys.bitwise_right_shift(tile.bit_length() - 1).long()
    id_bits = max(1, (vocab - 1).bit_length())
    id_mask = (1 << id_bits) - 1
    rank_key = (ordered << id_bits) | (id_mask - index)
    selected = rank_key.flatten(1).topk(candidates, dim=-1, sorted=True).values
    ids = (id_mask - (selected & id_mask)).long()
    selected_order = (selected >> id_bits).int()
    raw_bits = torch.where((selected_order & 0x8000) != 0,
                           selected_order ^ 0x8000, selected_order ^ 0xffff)
    selected_logits = raw_bits.to(torch.uint16).view(torch.bfloat16).float()
    probabilities = (selected_logits - maximum[:, None]).exp() / denominator[:, None]
    p64 = probabilities.double()
    offer_top1 = raw_token.long() == rejected
    score_top1 = raw_token.long() == score_rejected
    pack = torch.stack((torch.full_like(rejected, cap), offer_top1.long(),
                        (selected_order[:, cap - 1] == selected_order[:, cap]).long(),
                        torch.ones_like(rejected), torch.zeros_like(rejected)))
    return DecoderEpilogue(ids, p64, pack, q, score_top1)
