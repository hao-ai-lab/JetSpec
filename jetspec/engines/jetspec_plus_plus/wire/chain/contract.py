"""The shared N=1 device contract for the static JetSpec++ draft chain.

All offsets are into one int64[64] round packet. U1, U2, and U3 import these
names; a segment may read them on device without materializing host scalars.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import Protocol

import torch


class Offset(IntEnum):
    L = 0  # exact C1/tap prefix; committed_ids has L + 1 valid IDs
    C1_START = 1
    C1_COUNT = 2
    DEC_LEN = 3
    TRUE_K = 4
    CORRECTION = 5
    AFTER_MISS = 6
    TAU_IDX = 7
    J_ROUND = 23
    SLOT = 24  # reserved request slot; N=1 writes zero
    RNG_IDX = 32  # 32..62 inclusive; the active width selects the used prefix


PARAM_WORDS = 64
RNG_WORDS_MAX = 31
OUTCOME_WORDS = 16
KEYLEN_STEP = 256


def key_length_bucket(live: int, capacity: int) -> int:
    """Shared capture key for C1 and refresh attention's live key range."""
    if not 0 < live <= capacity:
        raise ValueError("live key length must fit the bound cache capacity")
    return min(capacity, ((live + KEYLEN_STEP - 1) // KEYLEN_STEP) * KEYLEN_STEP)


def key_length_buckets(capacity: int) -> tuple[int, ...]:
    """Finite startup capture set, including the whole-cache fallback."""
    if capacity < 1:
        raise ValueError("key capacity must be positive")
    return (*range(KEYLEN_STEP, capacity, KEYLEN_STEP), capacity)


class SegmentId(str, Enum):
    A = "A"
    B1R = "B1r"  # refresh followed by J1
    B1N = "B1n"  # J1 without refresh
    B2 = "B2"
    C = "C"
    P = "P"


def rng_words(lanes: int) -> int:
    if not 1 < lanes <= RNG_WORDS_MAX + 1:
        raise ValueError("chain lanes must be between 2 and 32")
    return lanes - 1


class OutcomeOffset(IntEnum):
    CMD = 0
    SEQ_ID = 1
    TRUE_K = 2
    RECOVERY = 3
    ROUND = 4
    TAP_COUNT = 5
    EOS_FLAG = 6
    RESERVED = 7
    L = 8  # tap-bearing count; the new recovery is at committed_ids[L]
    HASH16 = 9
    K = 10
    A = 11


@dataclass(frozen=True)
class OutcomePacket:
    """Verifier→draft int64[16] header; fixed taps follow in one NCCL message."""

    cmd: int
    seq_id: int
    true_k: int
    recovery: int
    round: int
    tap_count: int
    eos_flag: int
    reserved: int = 0
    L: int = 0
    hash16: int = 0
    k: int = 0
    a: int = 0

    def pack(self, out: torch.Tensor | None = None) -> torch.Tensor:
        if out is None:
            out = torch.empty(OUTCOME_WORDS, dtype=torch.int64)
        if out.shape != (OUTCOME_WORDS,) or out.dtype != torch.int64:
            raise ValueError("OutcomePacket output must be int64[16]")
        out.copy_(torch.tensor((self.cmd, self.seq_id, self.true_k, self.recovery,
                                self.round, self.tap_count, self.eos_flag, self.reserved,
                                self.L, self.hash16, self.k, self.a, 0, 0, 0, 0),
                               dtype=torch.int64, device=out.device))
        return out

    @classmethod
    def unpack(cls, words: torch.Tensor) -> OutcomePacket:
        if words.shape != (OUTCOME_WORDS,) or words.dtype != torch.int64:
            raise ValueError("OutcomePacket input must be int64[16]")
        return cls(*words[:12].tolist())


def outcome_message_views(message: torch.Tensor, *, lanes: int, hidden: int,
                          tap_dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """Interpret one static uint8 NCCL message as header and fixed tap block."""
    if tap_dtype not in (torch.bfloat16, torch.float32):
        raise ValueError("outcome taps must be bf16 or fp32")
    tap_bytes = lanes * 5 * hidden * torch.empty((), dtype=tap_dtype).element_size()
    if message.dtype != torch.uint8 or message.shape != (OUTCOME_WORDS * 8 + tap_bytes,):
        raise ValueError("outcome message has the wrong static byte length")
    packet = message[:OUTCOME_WORDS * 8].view(torch.int64)
    taps = message[OUTCOME_WORDS * 8:].view(tap_dtype).view(lanes, 5, hidden)
    return packet, taps


def patch_round_params_from_outcome(rp: torch.Tensor, packet: torch.Tensor) -> None:
    """Apply the verifier outcome; the final recovery has no exact tap yet."""
    if rp.shape != (PARAM_WORDS,) or rp.dtype != torch.int64 or packet.shape != (OUTCOME_WORDS,) \
            or packet.dtype != torch.int64 or rp.device != packet.device:
        raise ValueError("round and outcome packets must be int64 on the same device")
    rp[Offset.TRUE_K].copy_(packet[OutcomeOffset.TRUE_K])
    rp[Offset.CORRECTION].copy_(packet[OutcomeOffset.RECOVERY])
    rp[Offset.C1_COUNT].copy_(packet[OutcomeOffset.TAP_COUNT])
    rp[Offset.L].copy_(packet[OutcomeOffset.L])
    # One recovery ID follows the exact tap prefix in committed_ids.
    rp[Offset.C1_START].copy_(packet[OutcomeOffset.L] - packet[OutcomeOffset.TAP_COUNT])
    # The decoder prefix covers every verified true tap.
    rp[Offset.DEC_LEN].copy_(packet[OutcomeOffset.L])


class BucketProtocol(Protocol):
    """The U2/U3 handoff: U2 owns geometry; U3 owns refinement launches."""

    def update_context(self, rp: torch.Tensor, keys: torch.Tensor,
                       active: torch.Tensor) -> None: ...

    def run_pass(self, rp: torch.Tensor, pass_index: int) -> None: ...


@dataclass(frozen=True)
class TokenBuffers:
    """Sequence-owned IDs with an explicit single-slot backing dimension.

    Current N=1 bodies read the slot-zero 1-D views. The stable backing tensors
    retain a batch dimension for the future C>1 chain key.
    """

    committed_ids: torch.Tensor   # int64[capacity], valid prefix length = rp[L] + 1
    pending_draft_ids: torch.Tensor  # int64[lanes-1]
    committed_batch: torch.Tensor | None = None  # int64[1, capacity]
    pending_batch: torch.Tensor | None = None  # int64[1, lanes-1]

    def __post_init__(self) -> None:
        for view, backing, name in (
                (self.committed_ids, self.committed_batch, "committed"),
                (self.pending_draft_ids, self.pending_batch, "pending")):
            if view.ndim != 1 or view.dtype != torch.int64:
                raise ValueError(f"{name} token view must be int64[width]")
            if backing is not None and (backing.shape != (1, view.numel())
                                        or backing.dtype != view.dtype
                                        or backing.device != view.device
                                        or backing[0].data_ptr() != view.data_ptr()):
                raise ValueError(f"{name} token backing must alias int64[1, width]")
        if self.committed_batch is None:
            object.__setattr__(self, "committed_batch", self.committed_ids.view(1, -1))
        if self.pending_batch is None:
            object.__setattr__(self, "pending_batch", self.pending_draft_ids.view(1, -1))

    def seed_committed(self, prompt_ids: torch.Tensor) -> None:
        """Copy the admission prompt into stable sequence storage once at prefill."""
        if (prompt_ids.ndim != 1 or prompt_ids.dtype != torch.int64
                or prompt_ids.device != self.committed_ids.device
                or prompt_ids.numel() > self.committed_ids.numel()):
            raise ValueError("prefill IDs do not match static committed-token storage")
        self.committed_ids.zero_()
        self.committed_ids[:prompt_ids.numel()].copy_(prompt_ids)

    def fill(self, committed_ids: torch.Tensor, pending_draft_ids: torch.Tensor) -> None:
        for target, source in ((self.committed_ids, committed_ids),
                               (self.pending_draft_ids, pending_draft_ids)):
            if source.shape != target.shape or source.dtype != torch.int64 or source.device != target.device:
                raise ValueError("token source must match its static int64 device buffer")
            target.copy_(source, non_blocking=True)


def allocate_token_buffers(*, device: torch.device | str, capacity: int,
                           lanes: int) -> TokenBuffers:
    if capacity < 1:
        raise ValueError("token capacity must be positive")
    committed = torch.zeros((1, capacity), dtype=torch.int64, device=device)
    pending = torch.zeros((1, rng_words(lanes)), dtype=torch.int64, device=device)
    return TokenBuffers(committed[0], pending[0], committed, pending)


@dataclass(frozen=True)
class FrontInputs:
    """Stable U1 input pointers, bound once per chain key before graph capture.

    All tensors live on the draft device. ``lanes`` is a chain-key parameter; ``H`` is one
    target tap width and ``E`` is the C1 token embedding width. The accepted
    tap count is ``rp[Offset.C1_COUNT]``, not a host field on this object.

    c1_query_ids: int64[1, lanes]
    c1_query_embeddings: bf16[1, lanes, E]
    staged_hidden: bf16[1, lanes, 5*H] (C1 output, decoder input)
    prefix_positions: int64[1, lanes] (append positions)
    taps: bf16[lanes, 5, H] (accepted block, unused lanes padded)
    decoder_prefix_taps: bf16[lanes, H] (tap-4 rows starting at dec_len-count)
    rejected_ids: int64[lanes]
    selection_thresholds: float64[2] (drej-top1, other)
    decoder_prefix_kv: layer index -> (key, value), each bf16[1, heads, capacity, head_dim]
    c1_bucket, decoder_bucket, lanes and select_cap are host-known capture keys.
    """

    c1_query_ids: torch.Tensor
    c1_query_embeddings: torch.Tensor
    staged_hidden: torch.Tensor
    prefix_positions: torch.Tensor
    taps: torch.Tensor
    decoder_prefix_taps: torch.Tensor
    rejected_ids: torch.Tensor
    selection_thresholds: torch.Tensor
    decoder_prefix_kv: dict[int, tuple[torch.Tensor, torch.Tensor]]
    c1_bucket: int
    decoder_bucket: int
    lanes: int
    select_cap: int
    hybrid_recurrent: torch.Tensor | None = None
    hybrid_conv: torch.Tensor | None = None
    hybrid_prefix_key: torch.Tensor | None = None
    hybrid_prefix_value: torch.Tensor | None = None


@dataclass(frozen=True)
class RoundParams:
    L: int
    c1_start: int
    c1_count: int
    dec_len: int
    true_k: int
    correction: int
    after_miss: int
    tau_idx: int
    rng_idx: tuple[int, ...]
    J_round: int

    def pack(self, out: torch.Tensor | None = None) -> torch.Tensor:
        if not 1 <= len(self.rng_idx) <= RNG_WORDS_MAX:
            raise ValueError("RoundParams requires 1..31 RNG indices")
        if out is None:
            out = torch.empty(PARAM_WORDS, dtype=torch.int64)
        if out.shape != (PARAM_WORDS,) or out.dtype != torch.int64 or out.device.type != "cpu":
            raise ValueError("RoundParams output must be CPU int64[64]")
        values = (self.L, self.c1_start, self.c1_count, self.dec_len, self.true_k,
                  self.correction, self.after_miss, self.tau_idx)
        out.zero_()
        out[:Offset.TAU_IDX + 1] = torch.tensor(values, dtype=torch.int64)
        out[Offset.J_ROUND] = self.J_round
        out[Offset.RNG_IDX:Offset.RNG_IDX + len(self.rng_idx)] = torch.tensor(self.rng_idx, dtype=torch.int64)
        return out

    @classmethod
    def unpack(cls, words: torch.Tensor, *, lanes: int = 16) -> RoundParams:
        if words.shape != (PARAM_WORDS,) or words.dtype != torch.int64:
            raise ValueError("RoundParams input must be int64[64]")
        values = words.tolist()
        return cls(*values[:Offset.TAU_IDX + 1], tuple(values[Offset.RNG_IDX:Offset.RNG_IDX + rng_words(lanes)]),
                   values[Offset.J_ROUND])


@dataclass(frozen=True)
class ChainKey:
    L_bucket: int
    R: int
    J: int
    refresh: str
    lanes: int
    c1_key_bucket: int = 0
    refresh_key_bucket: int = 0
    batch_size: int = 1  # N=1 served now; capture identity reserves the future batch width
    decoder_key_bucket: int = 0  # per-key decoder window state when opted in



def allocate_buffers(*, device: torch.device | str, R: int, lanes: int, hidden: int,
                     vocab: int, row_home_words: int) -> dict[str, torch.Tensor]:
    """Allocate the fixed-shape U1→U2→U3→U4 exchange tensors."""
    if R not in (4, 8, 16) or not 1 < lanes <= RNG_WORDS_MAX + 1:
        raise ValueError("invalid chain row cap or lane count")
    shapes = {
        "rp": ((PARAM_WORDS,), torch.int64),
        "staged_hidden": ((1, lanes, hidden), torch.bfloat16),
        "offers_ids": ((lanes, 49), torch.int64),
        "offers_p64": ((lanes, 49), torch.float64),
        "offers_pack": ((5, lanes), torch.int64),
        "dec_logits": ((lanes, vocab), torch.bfloat16),
        "prior_p_top1": ((lanes,), torch.float32),
        "keys": ((R, 2), torch.int64),
        "active": ((R,), torch.bool),
        "requested": ((1,), torch.int64),
        "row_home": ((row_home_words,), torch.int64),
        "err": ((1,), torch.int64),
        "final_tokens": ((R, lanes), torch.int64),
        "final_p_top1": ((R, lanes), torch.float32),
        "passes_run": ((R,), torch.int64),
        "finished": ((R,), torch.bool),
        "meta": ((R, 4), torch.int64),
    }
    return {name: torch.empty(shape, dtype=dtype, device=device)
            for name, (shape, dtype) in shapes.items()}


class PinnedRoundParams:
    """Two host slots; never overwrite a slot while its H2D copy is pending."""

    def __init__(self, device: torch.device | str, *, pinned: bool | None = None,
                 event_factory=None):
        self.device = torch.device(device)
        if pinned is None:
            pinned = self.device.type == "cuda"
        self.host = [torch.empty(PARAM_WORDS, dtype=torch.int64, pin_memory=pinned)
                     for _ in range(2)]
        self.device_words = torch.empty(PARAM_WORDS, dtype=torch.int64, device=device)
        self.events = [None, None]
        self.next_slot = 0
        self.event_factory = event_factory or (lambda: torch.cuda.Event())
