"""Fused metadata and exact-tap intake for a captured chain round."""
from __future__ import annotations

from jetspec.engines.jetspec_plus_plus.profiles import kernel_enabled, kernel_value

import os

from jetspec.engines.jetspec_plus_plus.wire.contract import HASH_WINDOW, HASH_LENGTH_FACTOR
from jetspec.engines.jetspec_plus_plus.wire.triton_offsets import (  # noqa: F401  (read inside the kernels)
    A, AFTER_MISS, C1_COUNT, C1_START, CMD, CORRECTION, DEC_LEN, EOS_FLAG, HASH16, J_ROUND, K, L,
    RECOVERY, RESERVED, RNG_IDX, ROUND, RP_L, RP_TRUE_K, SEQ_ID, SLOT, TAP_COUNT, TAU_IDX, TRUE_K)
from types import SimpleNamespace

try:
    import triton
    import triton.language as tl
except ImportError:  # The pinned Mac CPU test environment has no Triton.
    triton = SimpleNamespace(jit=lambda fn: fn)
    tl = SimpleNamespace(constexpr=object())

# Field offsets and hash parameters are contract-emitted Triton constants.
# Error literals mirror CHAIN_ERR_* in intake.py.
from jetspec.engines.jetspec_plus_plus.kernels.chain_intake import _select


from jetspec.engines.jetspec_plus_plus.kernels.chain_intake import _metadata


from jetspec.engines.jetspec_plus_plus.kernels.chain_intake import _tap_front


def fused_intake(buffers: dict, *, max_length: int | None) -> None:
    """Run the two kernels using only static shapes and device pointers."""
    from jetspec.engines.jetspec_plus_plus.profiles import kernel_enabled
    packet = buffers["packet"]
    if not packet.is_cuda:
        raise ValueError("fused intake requires CUDA tensors")
    tokens, ring = buffers["tokens"], buffers["row_ring"]
    lanes = ring.rows.shape[-1]
    from jetspec.engines.jetspec_plus_plus.jetspec.profile import enabled as profile_enabled
    if lanes not in ((16, 32) if profile_enabled() else (16,)):
        raise ValueError("fused intake supports b16 (b32 in eval)")
    rows = ring.rows.shape[1]
    capacity = tokens.committed_ids.numel()
    limit = capacity - 1 if max_length is None else min(capacity - 1, max_length)
    control = "control_round" in buffers
    rng = "rng_table" in buffers
    prior = "prior_p_top1" in buffers
    taps = "tap_store" in buffers and "taps" in buffers
    front = "front" in buffers and "taps" in buffers
    admission = "admission_hidden" in buffers
    after_miss = ("control_after_miss" in buffers and
                  kernel_enabled('chain_mr_glue_lean', '1'))
    fallback = packet
    _metadata[(1,)](
        packet, buffers["rp"], buffers["expected_round"], buffers["safe_length"],
        buffers.get("control_round", fallback), buffers.get("control_L", fallback),
        tokens.committed_ids, tokens.pending_draft_ids,
        buffers.get("prior_p_top1", fallback), ring.rows, ring.p_top1,
        ring.passes_run, ring.keys, ring.rounds, buffers["hash_weights"],
        buffers["rng_table"].rows if rng else fallback, buffers["intake_err"],
        buffers["intake_selected"],
        buffers["control_after_miss"] if after_miss else fallback,
        capacity, limit, lanes, rows, lanes - 1,
        control, rng, prior, after_miss, 3 if profile_enabled() else 2,
        HASH_WINDOW, HASH_LENGTH_FACTOR, num_warps=4)
    if taps:
        store = buffers["tap_store"]
        bound = buffers.get("front")
        hidden = store.storage.shape[-1]
        _tap_front[(triton.cdiv(5 * hidden, 256),)](
            packet, buffers["taps"], store.backing,
            bound.taps if front else fallback,
            bound.decoder_prefix_taps if front else fallback,
            bound.staged_hidden if front else fallback,
            buffers["admission_hidden"] if admission else fallback,
            buffers["intake_selected"], store.storage.shape[0],
            hidden, lanes, front, admission, 256,
            **({'TrueTaps': buffers['refresh_swap_true_taps'], 'HAS_TRUE_TAPS': True}
               if 'refresh_swap_true_taps' in buffers else {}), num_warps=4)
