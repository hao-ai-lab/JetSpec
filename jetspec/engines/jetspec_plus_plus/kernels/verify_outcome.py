"""Device kernels for verify/outcome.py."""

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
                        ID_BLOCK: tl.constexpr, TAP_BLOCK: tl.constexpr,
                        HASH_WINDOW: tl.constexpr, HASH_LENGTH_FACTOR: tl.constexpr,
                        OUTCOME_WORDS: tl.constexpr):
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

            positions = committed + 1 - HASH_WINDOW + tl.arange(0, HASH_WINDOW)
            old = tl.load(token_storage + positions,
                          (positions >= 0) & (positions < start), other=0)
            new = tl.load(proposal + positions - start,
                          (positions >= start) & (positions < committed), other=0)
            values = tl.where(positions < start, old, new)
            values = tl.where(positions == committed, recovery, values)
            values = tl.where(positions >= 0, values, 0).to(tl.int64)
            hash_weights = tl.load(weights + tl.arange(0, HASH_WINDOW))
            digest = tl.sum((values + 1) * hash_weights, 0) + (committed + 1) * HASH_LENGTH_FACTOR

            eos = tl.full((), 0, tl.int1)
            if not IGNORE_EOS:
                proposed_eos = tl.sum(tl.where((ids < suffix_len) & (ids < WIDTH)
                                               & (staged == eos_id), 1, 0), 0) > 0
                eos = (eos_id >= 0) & (proposed_eos |
                    (RECOVERY_EOS_TERMINAL & (recovery == eos_id)))
            limit = ((committed - prompt_tokens >= max_new_tokens)
                     | (committed >= max_model_len))

            offsets = tl.arange(0, OUTCOME_WORDS)
            words = tl.full((OUTCOME_WORDS,), 0, tl.int64)
            words = tl.where(offsets == CMD, 1, words)
            words = tl.where(offsets == SEQ_ID, seq_id, words)
            words = tl.where(offsets == TRUE_K, count, words)
            words = tl.where(offsets == RECOVERY, recovery, words)
            words = tl.where(offsets == ROUND, round_id, words)
            words = tl.where(offsets == TAP_COUNT, suffix_len, words)
            words = tl.where(offsets == EOS_FLAG, (eos | limit).to(tl.int64), words)
            words = tl.where(offsets == RESERVED, 0, words)
            words = tl.where(offsets == L, committed, words)
            words = tl.where(offsets == HASH16, digest, words)
            words = tl.where(offsets == K, row_k, words)
            words = tl.where(offsets == A, row_a, words)
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

