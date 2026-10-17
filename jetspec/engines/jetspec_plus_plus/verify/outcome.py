"""One launch for the N=1 verify outcome, token mirror, and tap block."""

import torch

from jetspec.engines.jetspec_plus_plus.wire.contract import OUTCOME_WORDS
from jetspec.engines.jetspec_plus_plus.wire.contract import OutcomeOffset
from jetspec.engines.jetspec_plus_plus.wire.contract import HASH_WINDOW, HASH_LENGTH_FACTOR
from jetspec.engines.jetspec_plus_plus.wire.contract import hash_last16_host
from jetspec.engines.jetspec_plus_plus.wire.triton_offsets import (  # noqa: F401  (read inside the kernel)
    A, CMD, EOS_FLAG, HASH16, K, L, RECOVERY, RESERVED, ROUND, SEQ_ID, TAP_COUNT, TRUE_K)

try:
    import triton
    import triton.language as tl
except ImportError:  # The CPU serve suite runs on a Mac without Triton.
    triton = None


def outcome_reference(prefix, proposal, count, recovery, *, seq_id, round_id,
                      row_k, row_a, prompt_tokens, max_new_tokens,
                      max_model_len, eos_id, ignore_eos, recovery_eos_terminal=True):
    """CPU oracle with the same signed-int64 wrap and left-padded hash window."""
    start = len(prefix)
    width = len(proposal)
    if not 0 <= count < width:
        raise ValueError("accept count outside proposal")
    committed = start + count + 1
    storage = list(prefix) + list(proposal)
    if committed == len(storage):
        storage.append(recovery)
    else:
        storage[committed] = recovery
    hash_length = committed + 1
    digest = hash_last16_host(storage[max(0, hash_length - HASH_WINDOW):hash_length], hash_length)
    eos = (eos_id is not None and not ignore_eos and
           (eos_id in proposal[:count + 1] or
            (recovery_eos_terminal and recovery == eos_id)))
    limit = (committed - prompt_tokens >= max_new_tokens or
             committed >= max_model_len)
    words = [0] * OUTCOME_WORDS
    words[OutcomeOffset.CMD] = 1
    words[OutcomeOffset.SEQ_ID] = seq_id
    words[OutcomeOffset.TRUE_K] = count
    words[OutcomeOffset.RECOVERY] = recovery
    words[OutcomeOffset.ROUND] = round_id
    words[OutcomeOffset.TAP_COUNT] = count + 1
    words[OutcomeOffset.EOS_FLAG] = int(eos or limit)
    words[OutcomeOffset.L] = committed
    words[OutcomeOffset.HASH16] = digest
    words[OutcomeOffset.K] = row_k
    words[OutcomeOffset.A] = row_a
    return words, storage


if triton is not None:
    from jetspec.engines.jetspec_plus_plus.kernels.verify_outcome import (_outcome_kernel)


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
    if weights.shape != (HASH_WINDOW,) or weights.device != proposal.device:
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
        HASH_WINDOW, HASH_LENGTH_FACTOR, OUTCOME_WORDS,
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
