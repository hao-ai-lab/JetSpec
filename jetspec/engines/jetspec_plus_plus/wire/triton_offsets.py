"""The contract's packet offsets as Triton constexpr globals.

Triton 3.4 cannot unpack or index a tuple constexpr inside a kernel, so a kernel reads these
module-level names instead of taking an offset tuple as a parameter. Every value comes from
wire.contract, in its tuple order. On a host without Triton the names are plain ints.
"""
from jetspec.engines.jetspec_plus_plus.wire.contract import CANDIDATE_OFFSETS, OUTCOME_OFFSETS, ROUND_OFFSETS

try:
    import triton.language as tl
except ImportError:  # CPU hosts without Triton never compile a kernel.
    tl = None


def _constexpr(value):
    return value if tl is None else tl.constexpr(value)


# A length change in the contract fails these unpackings at import.
(CMD, SEQ_ID, TRUE_K, RECOVERY, ROUND, TAP_COUNT, EOS_FLAG, RESERVED,
 L, HASH16, K, A) = map(_constexpr, OUTCOME_OFFSETS)
(RP_L, C1_START, C1_COUNT, DEC_LEN, RP_TRUE_K, CORRECTION, AFTER_MISS,
 TAU_IDX, J_ROUND, SLOT, RNG_IDX) = map(_constexpr, ROUND_OFFSETS)
(AVAILABLE, CANDIDATE_SEQ_ID, CANDIDATE_ROUND, BLOCKS) = map(_constexpr, CANDIDATE_OFFSETS)
