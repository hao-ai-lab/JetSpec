"""Two-launch, preallocated bookkeeping for a fixed-shape Jacobi pass.

The Torch reference is also usable on a CPU-only development host. The CUDA
body uses one vocabulary-partial kernel and one row-finalization kernel.
"""
from __future__ import annotations

import torch

from jetspec.engines.jetspec_plus_plus.wire.contract import Offset
from jetspec.engines.jetspec_plus_plus.draft.method.passes import PassValues, reference_pass
from jetspec.engines.jetspec_plus_plus.draft.method.ops import reference
from jetspec.engines.jetspec_plus_plus.kernels import register_optimization

try:
    import triton
    import triton.language as tl
except ImportError:  # The Mac CPU gate has no Triton installation.
    triton = None
    tl = None


def reference_chain_pass(logits, ids, keys, active, outputs, previous_top1,
                         tokens, rp, index, *, save_previous, pins=None, p_top1=None):
    """CPU oracle for the CUDA body, including freeze and publication metadata."""
    values = reference_pass(logits, ids, keys[:, 1].clamp_min(0), pins=pins, p_top1=p_top1)
    was_finished = ~active if index == 0 else outputs['finished']
    selected_tokens = (values.tokens if index == 0 else torch.where(
        was_finished[:, None], outputs['final_tokens'], values.tokens))
    selected_top1 = (values.p_top1 if index == 0 else torch.where(
        was_finished[:, None], outputs['final_p_top1'], values.p_top1))
    outputs['final_tokens'].copy_(selected_tokens.masked_fill(~active[:, None], -1))
    outputs['final_p_top1'].copy_(selected_top1.masked_fill(~active[:, None], 0))
    outputs['passes_run'].masked_fill_(~was_finished, index + 1)
    outputs['finished'].copy_(was_finished | values.fixed)
    ids.copy_(values.tokens)
    if save_previous:
        previous_top1.copy_(values.p_top1[:, 1:])
    if index == 0 and '_t17_refresh' in outputs:
        from jetspec.engines.jetspec_plus_plus.kernels.refresh_prefix_scan import reference_refresh
        reference_refresh(values.p_top1, active, keys, rp[Offset.L], outputs['_t17_refresh'])
    source = torch.cat((tokens.committed_ids.index_select(0, rp[Offset.L].reshape(1)),
                        tokens.pending_draft_ids, tokens.pending_draft_ids[-1:]))
    preceding = source.index_select(0, keys[:, 0].clamp(0, ids.shape[1]))
    outputs['meta'].copy_(torch.stack((preceding, outputs['final_tokens'][:, 0],
                                       outputs['passes_run'],
                                       outputs['finished'].long()), dim=1))


if triton is not None:
    from jetspec.engines.jetspec_plus_plus.kernels.chain_publish import _finish_publish
    from jetspec.engines.jetspec_plus_plus.kernels.refresh_prefix_scan import _refresh
    @triton.jit
    def _partials(LOGITS, PMAX, PSUM, PIDX, V: tl.constexpr,
                  L: tl.constexpr, C: tl.constexpr, S0: tl.constexpr,
                  S1: tl.constexpr, S2: tl.constexpr, B: tl.constexpr):
        slot = tl.program_id(0)
        chunk = tl.program_id(1)
        x = chunk * B + tl.arange(0, B)
        value = tl.load(LOGITS + (slot // L) * S0 + (slot % L) * S1 + x * S2,
                        x < V, other=-float('inf')).to(tl.float32)
        maximum = tl.max(value, 0)
        safe_maximum = tl.where(maximum == -float('inf'), 0.0, maximum)
        total = tl.sum(tl.exp(value - safe_maximum), 0)
        index = tl.min(tl.where(value == maximum, x, 2147483647), 0)
        out = slot * C + chunk
        tl.store(PMAX + out, maximum)
        tl.store(PSUM + out, total)
        tl.store(PIDX + out, index)


    @triton.jit
    def _finish(PMAX, PSUM, PIDX, IDS, KEYS, ACTIVE, FINISHED,
                PASSES, FINAL_TOKENS, FINAL_TOP1, PREV_TOP1, META,
                COMMITTED, PENDING, RP, PIN2, PIN_SLOT, PIN_TOKEN,
                L: tl.constexpr, C: tl.constexpr, PC: tl.constexpr,
                PASS: tl.constexpr, SAVE_PREV: tl.constexpr,
                HAS_PIN2: tl.constexpr, RP_L: tl.constexpr,
                HAS_PIN: tl.constexpr, DENSE_PIN: tl.constexpr = False,
                PUBLISH: tl.constexpr = False, Packet=None, Published=None,
                Wire=None, Scores=None, WirePasses=None, RingRows=None, RingKeys=None,
                RingTop1=None, RingPasses=None, RingRounds=None,
                ROWS: tl.constexpr = 0, CAP: tl.constexpr = 0, RING: tl.constexpr = 0,
                WORDS: tl.constexpr = 0, BLOCK_OFF: tl.constexpr = 0,
                KEY_OFF: tl.constexpr = 0, PAD_BLOCK: tl.constexpr = 1,
                ROW=0, USE_ROW: tl.constexpr = False,
                REFRESH: tl.constexpr = False, RefreshChained=None, RefreshKhat=None,
                RefreshCounts=None, RefreshTails=None, RefreshStart=None,
                REFRESH_WINDOW: tl.constexpr = 0, REFRESH_THRESHOLD: tl.constexpr = 0.):
        row = ROW if USE_ROW else tl.program_id(0)
        lane = tl.arange(0, L)
        chunk = tl.arange(0, PC)
        offsets = row * L * C + lane[:, None] * C + chunk[None, :]
        partial_max = tl.load(PMAX + offsets, chunk[None, :] < C,
                              other=-float('inf'))
        partial_sum = tl.load(PSUM + offsets, chunk[None, :] < C, other=0)
        partial_idx = tl.load(PIDX + offsets, chunk[None, :] < C,
                              other=2147483647)
        maximum = tl.max(partial_max, 1)
        sum_exp = tl.sum(partial_sum * tl.exp(partial_max - maximum[:, None]), 1)
        token = tl.min(tl.where(partial_max == maximum[:, None], partial_idx,
                                2147483647), 1).to(tl.int64)
        top1 = 1.0 / sum_exp
        first = tl.maximum(tl.load(KEYS + row * 2 + 1), 0)
        token = tl.where(lane == 0, first, token)
        if HAS_PIN2:
            token = tl.where(lane == 1, tl.load(PIN2 + row), token)
        if HAS_PIN:
            if DENSE_PIN:
                pin = tl.load(PIN_SLOT + row * L + lane)
                token = tl.where(pin, tl.load(PIN_TOKEN + row * L + lane), token)
            else:
                # serving JETSPEC_REFRESH_SWAP: one swapped slot per row.
                token = tl.where(lane == tl.load(PIN_SLOT + row), tl.load(PIN_TOKEN + row), token)
        input_ids = tl.load(IDS + row * L + lane)
        fixed = tl.sum((token != input_ids).to(tl.int32), 0) == 0
        active = tl.load(ACTIVE + row)
        if PASS == 0:
            was_finished = ~active
        else:
            was_finished = tl.load(FINISHED + row)
        old_token = tl.load(FINAL_TOKENS + row * L + lane) if PASS else token
        old_top1 = tl.load(FINAL_TOP1 + row * L + lane) if PASS else top1
        selected_token = tl.where(was_finished, old_token, token)
        selected_top1 = tl.where(was_finished, old_top1, top1)
        selected_token = tl.where(active, selected_token, -1)
        selected_top1 = tl.where(active, selected_top1, 0)
        tl.store(FINAL_TOKENS + row * L + lane, selected_token)
        tl.store(FINAL_TOP1 + row * L + lane, selected_top1)
        tl.store(IDS + row * L + lane, token)
        if SAVE_PREV:
            tl.store(PREV_TOP1 + row * (L - 1) + lane - 1, top1, lane > 0)
        if REFRESH:
            _refresh(top1, lane, row, ACTIVE, KEYS, RP, RefreshChained,
                     RefreshKhat, RefreshCounts, RefreshTails, RefreshStart,
                     L, RP_L, REFRESH_WINDOW, REFRESH_THRESHOLD)
        passes = tl.load(PASSES + row) if PASS else 0
        passes = tl.where(was_finished, passes, PASS + 1)
        finished = was_finished | fixed
        tl.store(PASSES + row, passes)
        tl.store(FINISHED + row, finished)
        key = tl.maximum(tl.load(KEYS + row * 2), 0)
        length = tl.load(RP + RP_L)
        committed = tl.load(COMMITTED + length)
        pending = tl.load(PENDING + tl.minimum(tl.maximum(key - 1, 0), L - 2))
        preceding = tl.where(key == 0, committed, pending)
        tl.store(META + row * 4, preceding)
        tl.store(META + row * 4 + 1, tl.sum(tl.where(lane == 0, selected_token, 0), 0))
        tl.store(META + row * 4 + 2, passes)
        tl.store(META + row * 4 + 3, finished.to(tl.int64))
        if PUBLISH:
            _finish_publish(row, selected_token, selected_top1, passes, Packet, KEYS,
                FINAL_TOKENS, FINAL_TOP1, Published, Wire, Scores, WirePasses,
                RingRows, RingKeys, RingTop1, RingPasses, RingRounds,
                ROWS, CAP, RING, L, WORDS, BLOCK_OFF, KEY_OFF, PAD_BLOCK)


def allocate_partials(rows, lanes, vocab, device, *, block=2048):
    """Allocate outside CUDA graph capture, once per fixed row bucket."""
    chunks = triton.cdiv(vocab, block) if triton is not None else (vocab + block - 1) // block
    shape = (rows, lanes, chunks)
    partials = (torch.empty(shape, device=device, dtype=torch.float32),
                torch.empty(shape, device=device, dtype=torch.float32),
                torch.empty(shape, device=device, dtype=torch.int32))
    from jetspec.engines.jetspec_plus_plus.profiles import kernel_enabled
    if kernel_enabled('draft_fuse_t6'):
        from jetspec.engines.jetspec_plus_plus.kernels.draft_glue_tune import PassPartials
        return PassPartials(partials, torch.zeros(rows, device=device, dtype=torch.int32))
    return partials


def chain_pass(logits, ids, keys, active, outputs, previous_top1, tokens, rp,
               partials, index, *, save_previous, pinned_second=None, pins=None):
    """Update all post-forward chain pass outputs without a host tensor read.

    ``pins``: legacy ``(slot [R], token [R])`` or policy ``(mask [R,L], tokens [R,L])``
    as in ``reference_pass``; default None."""
    if triton is None:
        raise RuntimeError('JETSPEC_REFINE_K3 on CUDA requires Triton')
    from jetspec.engines.jetspec_plus_plus.profiles import kernel_enabled
    if kernel_enabled('draft_fuse_t6'):
        from jetspec.engines.jetspec_plus_plus.kernels.draft_glue_tune import chain_pass_into
        return chain_pass_into(logits, ids, keys, active, outputs, previous_top1,
                               tokens, rp, partials, index, save_previous,
                               pinned_second, pins)
    rows, lanes, vocab = logits.shape
    chunks = partials[0].shape[-1]
    block = triton.cdiv(vocab, chunks)
    block = triton.next_power_of_2(block)
    _partials[(rows * lanes, chunks)](
        logits, *partials, vocab, lanes, chunks, *logits.stride(), block,
        num_warps=4)
    _finish[(rows,)](
        *partials, ids, keys, active, outputs['finished'], outputs['passes_run'],
        outputs['final_tokens'], outputs['final_p_top1'], previous_top1,
        outputs['meta'], tokens.committed_ids, tokens.pending_draft_ids, rp,
        pinned_second if pinned_second is not None else keys,
        *(pins if pins is not None else (keys, keys)),
        lanes, chunks, triton.next_power_of_2(chunks), index, save_previous,
        pinned_second is not None, int(Offset.L), pins is not None,
        pins is not None and pins[0].ndim == 2,
        **(outputs.get('_t14_publication', {}) if index == 1 else {}),
        **(outputs.get('_t17_refresh', {}) if index == 0 else {}), num_warps=4)


def optimized_pass(logits, input_tokens, pinned_first=None, pinned_second=None, pins=None,
                   *, p_top1=None, head_weight=None, partials=None):
    """Expose the fused vocabulary reduction through the method pass contract."""
    if (triton is None or not logits.is_cuda or p_top1 is not None
            or head_weight is not None):
        return reference('jacobi_pass')(logits, input_tokens, pinned_first, pinned_second,
                                        pins, p_top1=p_top1, head_weight=head_weight)
    from jetspec.engines.jetspec_plus_plus.kernels.drafter_lmhead import _merge
    from jetspec.engines.jetspec_plus_plus.draft.method.passes import apply_pins, jacobi_converged
    rows, lanes, vocab = logits.shape
    if partials is None:
        partials = allocate_partials(rows, lanes, vocab, logits.device)
    chunks = partials[0].shape[-1]
    block = triton.next_power_of_2(triton.cdiv(vocab, chunks))
    _partials[(rows * lanes, chunks)](
        logits, *partials, vocab, lanes, chunks, *logits.stride(), block, num_warps=4)
    tokens = torch.empty_like(input_tokens)
    probabilities = torch.empty_like(input_tokens, dtype=torch.float32)
    _merge[(rows * lanes,)](*partials, tokens, probabilities, chunks,
                           triton.next_power_of_2(chunks), num_warps=4)
    if pinned_first is not None:
        tokens[..., 0] = pinned_first
    if pinned_second is not None:
        tokens[..., 1] = pinned_second
    if pins is not None:
        tokens = apply_pins(tokens, pins)
    return PassValues(tokens, jacobi_converged(tokens, input_tokens), probabilities)


register_optimization('jacobi_pass', optimized_pass)
