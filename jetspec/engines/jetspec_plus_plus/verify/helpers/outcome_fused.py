"""One launch for the N=1 verify outcome, token mirror, and tap block."""

import torch

from jetspec.engines.jetspec_plus_plus.wire.chain.contract import OUTCOME_WORDS; from jetspec.engines.jetspec_plus_plus.wire.chain.contract import OutcomeOffset
from jetspec.engines.jetspec_plus_plus.wire.chain.intake import HASH_WEIGHTS

try:
    from jetspec.engines.jetspec_plus_plus._optional import triton
    from jetspec.engines.jetspec_plus_plus._optional import tl
except ImportError:  # The CPU serve suite runs on a Mac without Triton.
    triton = None




if triton is not None:
    @triton.jit(do_not_specialize=["start", "tap_base", "seq_id", "round_id",
                                   "row_k", "row_a", "prompt_tokens",
                                   "max_new_tokens", "max_model_len", "eos_id"])
    def _outcome_kernel(proposal, count_ptr, recovery_ptr, token_storage,
                        taps, tap_storage, weights, outcome,
                        start, tap_base, seq_id, round_id, row_k, row_a,
                        prompt_tokens, max_new_tokens, max_model_len, eos_id,
                        WIDTH: tl.constexpr, TAP_ELEMENTS: tl.constexpr,
                        TAP_LANES: tl.constexpr, TAP_HIDDEN: tl.constexpr,
                        SRC_S0: tl.constexpr, SRC_S1: tl.constexpr,
                        SRC_S2: tl.constexpr, DST_S0: tl.constexpr,
                        DST_S1: tl.constexpr, DST_S2: tl.constexpr,
                        HAS_TAPS: tl.constexpr, IGNORE_EOS: tl.constexpr,
                        RECOVERY_EOS_TERMINAL: tl.constexpr,
                        ID_BLOCK: tl.constexpr, TAP_BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        if pid == 0:
            count = tl.load(count_ptr).to(tl.int64)
            recovery = tl.load(recovery_ptr).to(tl.int64)
            suffix_len = count + 1
            committed = start + suffix_len
            ids = tl.arange(0, ID_BLOCK)
            staged = tl.load(proposal + ids, ids < WIDTH, other=0).to(tl.int64)
            tl.store(token_storage + start + ids, staged, ids < WIDTH)
            tl.store(token_storage + committed, recovery)

            positions = committed + 1 - 16 + tl.arange(0, 16)
            old = tl.load(token_storage + positions,
                          (positions >= 0) & (positions < start), other=0)
            new = tl.load(proposal + positions - start,
                          (positions >= start) & (positions < committed), other=0)
            values = tl.where(positions < start, old, new)
            values = tl.where(positions == committed, recovery, values)
            values = tl.where(positions >= 0, values, 0).to(tl.int64)
            hash_weights = tl.load(weights + tl.arange(0, 16))
            digest = tl.sum((values + 1) * hash_weights, 0) + (committed + 1) * 1000003

            eos = tl.full((), 0, tl.int1)
            if not IGNORE_EOS:
                proposed_eos = tl.sum(tl.where((ids < suffix_len) & (ids < WIDTH)
                                               & (staged == eos_id), 1, 0), 0) > 0
                eos = (eos_id >= 0) & (proposed_eos |
                    (RECOVERY_EOS_TERMINAL & (recovery == eos_id)))
            limit = ((committed - prompt_tokens >= max_new_tokens)
                     | (committed >= max_model_len))

            offsets = tl.arange(0, 16)
            words = tl.full((16,), 0, tl.int64)
            words = tl.where(offsets == 0, 1, words)
            words = tl.where(offsets == 1, seq_id, words)
            words = tl.where(offsets == 2, count, words)
            words = tl.where(offsets == 3, recovery, words)
            words = tl.where(offsets == 4, round_id, words)
            words = tl.where(offsets == 5, suffix_len, words)
            words = tl.where(offsets == 6, (eos | limit).to(tl.int64), words)
            words = tl.where(offsets == 8, committed, words)
            words = tl.where(offsets == 9, digest, words)
            words = tl.where(offsets == 10, row_k, words)
            words = tl.where(offsets == 11, row_a, words)
            tl.store(outcome + offsets, words)

        if HAS_TAPS:
            offsets = pid * TAP_BLOCK + tl.arange(0, TAP_BLOCK)
            token = offsets // (TAP_LANES * TAP_HIDDEN)
            lane = offsets // TAP_HIDDEN % TAP_LANES
            hidden = offsets % TAP_HIDDEN
            src = token * SRC_S0 + lane * SRC_S1 + hidden * SRC_S2
            dst = (tap_base + token) * DST_S0 + lane * DST_S1 + hidden * DST_S2
            tap_values = tl.load(taps + src, offsets < TAP_ELEMENTS, other=0)
            tl.store(tap_storage + dst, tap_values,
                     offsets < TAP_ELEMENTS)


def launch_outcome_fused(proposal, count, recovery, token_storage, taps,
                         tap_storage, weights, outcome, *, start, tap_base,
                         seq_id, round_id, row_k, row_a, prompt_tokens,
                         max_new_tokens, max_model_len, eos_id, ignore_eos,
                         recovery_eos_terminal=True):
    """Launch once; all scalar metadata is host known before the accept read."""
    if triton is None or not proposal.is_cuda:
        raise RuntimeError("fused verify outcome requires CUDA Triton")
    if proposal.ndim != 1 or proposal.dtype != torch.int64:
        raise ValueError("proposal must be int64[width]")
    if outcome.shape != (OUTCOME_WORDS,) or outcome.dtype != torch.int64:
        raise ValueError("outcome must be int64[16]")
    if weights.shape != (16,) or weights.device != proposal.device:
        raise ValueError("hash weights must be int64[16] on the proposal device")
    has_taps = taps is not None
    if has_taps and (tap_storage is None or taps.shape[0] != proposal.numel()
                     or taps.shape[1:] != tap_storage.shape[1:]):
        raise ValueError("tap storage shape mismatch")
    tap_elements = taps.numel() if has_taps else 0
    tap_lanes, tap_hidden = taps.shape[1:] if has_taps else (1, 1)
    src_strides = taps.stride() if has_taps else (1, 1, 1)
    dst_strides = tap_storage.stride() if has_taps else (1, 1, 1)
    grid = (max(1, triton.cdiv(tap_elements, 256)),)
    _outcome_kernel[grid](
        proposal, count, recovery, token_storage,
        taps if has_taps else proposal,
        tap_storage if has_taps else proposal,
        weights, outcome, start, tap_base, seq_id, round_id, row_k,
        row_a, prompt_tokens, max_new_tokens, max_model_len,
        -1 if eos_id is None else eos_id, proposal.numel(), tap_elements,
        tap_lanes, tap_hidden, *src_strides, *dst_strides,
        has_taps, ignore_eos, recovery_eos_terminal,
        triton.next_power_of_2(proposal.numel()), 256,
        num_warps=4)


def prepare_outcome_storage(sequence, proposal, taps, *, token_capacity,
                            tap_capacity):
    """Prepare the same persistent buffers and pending metadata as eager staging."""
    start = sequence.num_cached_tokens
    width = proposal.shape[0]
    if start + width + 1 > token_capacity:
        raise ValueError("committed ID capacity is too small for the recovery")
    storage = getattr(sequence, "_jetspec_token_storage", None)
    if storage is None:
        storage = torch.empty(token_capacity, dtype=torch.int64, device=proposal.device)
        prefix = torch.tensor(sequence.token_ids[:start], dtype=torch.int64,
                              pin_memory=proposal.is_cuda)
        storage[:start].copy_(prefix, non_blocking=proposal.is_cuda)
        sequence._jetspec_token_storage = storage
    elif getattr(sequence, "_jetspec_device_valid_length", None) != start:
        raise RuntimeError("verify device token length differs from host metadata")
    sequence._jetspec_pending_tokens = (start, width)

    tap_storage = None
    tap_base = 0
    if taps is not None:
        if taps.ndim != 3:
            raise ValueError("candidate_taps must be [tokens, taps, hidden]")
        previous = sequence.jetspec_taps
        tap_base = 0 if previous is None else previous.shape[0]
        if tap_base + width > tap_capacity:
            raise ValueError("tap storage capacity is too small for the verify block")
        tap_storage = getattr(sequence, "_jetspec_tap_storage", None)
        if tap_storage is None:
            tap_storage = taps.new_empty((tap_capacity, *taps.shape[1:]))
            if previous is not None:
                tap_storage[:tap_base].copy_(previous)
            sequence._jetspec_tap_storage = tap_storage
        sequence._jetspec_tap_pending = (tap_base, width)
        sequence._jetspec_tap_candidate_block = tap_storage[tap_base:tap_base + width]
    return storage, tap_storage, tap_base
