"""Verify-rank storage for the shared chain outcome message."""
from __future__ import annotations

from jetspec.engines.jetspec_plus_plus.profiles import kernel_enabled, kernel_value

from functools import lru_cache
import os
import time
import torch

from jetspec.engines.jetspec_plus_plus.wire.contract import OUTCOME_WORDS
from jetspec.engines.jetspec_plus_plus.wire.contract import OutcomeOffset
from jetspec.engines.jetspec_plus_plus.wire.contract import outcome_message_views
from jetspec.engines.jetspec_plus_plus.wire.contract import HASH_WEIGHTS, HASH_WINDOW
from jetspec.engines.jetspec_plus_plus.wire.contract import hash_last16 as hash16_committed


CHAIN_COMMAND_KEY = "chain_command_"
CHAIN_COMMAND_COUNTER = "chain_command_next"
CHAIN_RECV_READY_KEY = "chain_recv_ready_"
_REFERENCE_PADDING_TERM = sum(HASH_WEIGHTS)
_local_command_store = None
_local_command_next = 0


def announce_chain_command(store, command: str, *, round_id=None, L=None,
                           after_miss=None, eos=None, true_k=None, recovery=None,
                           redraft=None, prepost=None, attn3=None, local_counter=None, run_config=None) -> int:
    """Announce a chain payload shape on the host channel before its NCCL send."""
    from jetspec.engines.jetspec_plus_plus.serving.settings import setting
    from jetspec.engines.jetspec_plus_plus.profiles import kernel_enabled
    if command not in {"prefill", "warmup", "pin_served", "outcome", "exit"}:
        raise ValueError("unknown chain command")
    if redraft is None:
        redraft = setting(run_config, 'engine.engine_settings.chain_miss_redraft', '0') == "1"
    if prepost is None:
        prepost = setting(run_config, 'engine.engine_settings.chain_prepost', '0') == "1"
    if local_counter is None:
        local_counter = setting(run_config, 'engine.engine_settings.chain_cmd_local_counter', '0') == "1"
    if command == "outcome" and redraft:
        # 5f: the verifier sends its synced TRUE_K and RECOVERY words; the draft
        # host computes AFTER_MISS from its own key snapshot. EOS rides along so
        # the terminal command is never probed, with or without PREPOST.
        if (round_id is None or L is None or true_k is None or recovery is None
                or round_id < 0 or L < 1 or true_k < 0 or recovery < 0
                or eos not in (0, 1)):
            raise ValueError("miss-redraft outcome command needs round, L, TRUE_K, "
                             "RECOVERY and EOS")
        wire = f"outcome:{round_id}:{L}:{true_k}:{recovery}:{eos}".encode("ascii")
    elif command == "outcome" and prepost:
        if (round_id is None or L is None or round_id < 0 or L < 1
                or after_miss not in (0, 1) or eos not in (0, 1)):
            raise ValueError("prepost outcome command needs round, L, AFTER_MISS and EOS")
        wire = f"outcome:{round_id}:{L}:{after_miss}:{eos}".encode("ascii")
    else:
        wire = command.encode("ascii")
    if local_counter:
        global _local_command_store, _local_command_next
        if _local_command_store is not store:
            _local_command_store = store
            _local_command_next = 0
        command_id = _local_command_next
        _local_command_next += 1
    else:
        command_id = int(store.add(CHAIN_COMMAND_COUNTER, 1)) - 1
    store.set(f"{CHAIN_COMMAND_KEY}{command_id}", wire)
    return command_id


def await_chain_receive(store, command_id: int, *, timing=None, run_config=None) -> None:
    """Do not enqueue an outcome send until the draft posted its fixed receive."""
    from jetspec.engines.jetspec_plus_plus.serving.settings import setting
    if timing is None:
        timing = setting(run_config, 'engine.engine_settings.chain_prepost_timing', '0') == "1"
    started = time.perf_counter() if timing else None
    if store.get(f"{CHAIN_RECV_READY_KEY}{command_id}") != b"1":
        raise RuntimeError("draft did not post the chain outcome receive")
    if started is not None:
        print(f"CHAIN_PREPOST_ACK command_id={command_id} wait_ms="
              f"{(time.perf_counter() - started) * 1000:.3f}", flush=True)


def make_hash16_weights(device: torch.device | str) -> torch.Tensor:
    """Stage U4's fixed digest weights during prefill, outside timed rounds."""
    return torch.tensor(HASH_WEIGHTS, dtype=torch.int64, device=device)




def admission_outcome(sequence, *, capacity: int, device, weights: torch.Tensor) -> torch.Tensor:
    """Seed the verify token mirror and make the device round-zero outcome."""
    L = int(sequence.num_cached_tokens)
    if len(sequence.token_ids) != L + 1 or capacity <= L:
        raise ValueError("admission needs P taps and one untapped recovery token")
    if hasattr(sequence, "_jetspec_token_storage"):
        raise ValueError("admission device token mirror was already seeded")
    storage = torch.empty(capacity, dtype=torch.int64, device=device)
    source = torch.tensor(sequence.token_ids, dtype=torch.int64, device="cpu",
                          pin_memory=storage.is_cuda)
    storage[:L + 1].copy_(source, non_blocking=storage.is_cuda)
    sequence._jetspec_token_storage = storage
    sequence._jetspec_device_valid_length = L
    packet = torch.zeros(OUTCOME_WORDS, dtype=torch.int64, device=device)
    packet[OutcomeOffset.CMD] = 1
    packet[OutcomeOffset.SEQ_ID] = int(sequence.seq_id)
    packet[OutcomeOffset.RECOVERY].copy_(storage[L])
    packet[OutcomeOffset.ROUND] = 0
    packet[OutcomeOffset.L] = L
    packet[OutcomeOffset.HASH16].copy_(hash16_committed(storage, packet[OutcomeOffset.L] + 1,
                                                        weights))
    packet[OutcomeOffset.K] = -1
    packet[OutcomeOffset.A] = -1
    return packet





class OutcomeWire:
    """One fixed-size NCCL byte tensor: contract header, then [lanes,5,H] taps."""

    def __init__(self, lanes, hidden, dtype, device):
        self.lanes = lanes
        self.hidden = hidden
        self.dtype = dtype
        tap_bytes = lanes * 5 * hidden * torch.empty((), dtype=dtype).element_size()
        self.tensor = torch.empty(OUTCOME_WORDS * 8 + tap_bytes,
                                  dtype=torch.uint8, device=device)
        self.packet, self.taps = outcome_message_views(
            self.tensor, lanes=lanes, hidden=hidden, tap_dtype=dtype)

    def pack(self, packet, taps, *, skip_full_zero=False):
        if packet.shape != (OUTCOME_WORDS,) or packet.dtype != torch.int64:
            raise ValueError(f"OutcomePacket must be int64[{OUTCOME_WORDS}]")
        if taps.ndim != 3 or taps.shape[0] > self.lanes or taps.shape[1:] != (5, self.hidden):
            raise ValueError("outcome taps exceed the fixed lane block")
        self.packet.copy_(packet)
        if not skip_full_zero or taps.shape[0] != self.lanes:
            self.taps.zero_()
        self.taps[:taps.shape[0]].copy_(taps.to(self.dtype))
        return self.tensor

    def prepare_taps(self, taps):
        """Copy the invariant full tap block before host acceptance completes."""
        if taps.shape != (self.lanes, 5, self.hidden):
            raise ValueError("early outcome preparation requires every tap lane")
        self.taps.copy_(taps.to(self.dtype))

    def pack_prepared_header(self, packet):
        if packet.shape != (OUTCOME_WORDS,) or packet.dtype != torch.int64:
            raise ValueError(f"OutcomePacket must be int64[{OUTCOME_WORDS}]")
        self.packet.copy_(packet)
        return self.tensor


class OutcomeWireRing:
    """Keep each packed envelope until its asynchronous send completes."""

    def __init__(self, lanes, hidden, dtype, device):
        self.wires = [OutcomeWire(lanes, hidden, dtype, device) for _ in range(2)]
        self.works = [[], []]
        self.next_slot = 0
        self.pending_slot = None

    def acquire(self):
        slot = self.next_slot
        for work in self.works[slot]:
            work.wait()
        self.works[slot] = []
        self.pending_slot = slot
        self.next_slot = 1 - slot
        return self.wires[slot]

    def sent(self, works):
        if self.pending_slot is None:
            raise RuntimeError("outcome envelope send has no acquired slot")
        self.works[self.pending_slot] = list(works)
        self.pending_slot = None
