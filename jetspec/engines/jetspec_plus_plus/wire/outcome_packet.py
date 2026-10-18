"""Verify-rank storage for the shared chain outcome message."""
from __future__ import annotations

from jetspec.engines.jetspec_plus_plus.profiles import kernel_enabled, kernel_value

from functools import lru_cache
import os
import time
import torch

from jetspec.engines.jetspec_plus_plus.wire.contract import OUTCOME_WORDS
from jetspec.engines.jetspec_plus_plus.wire.contract import OutcomeLayout
from jetspec.engines.jetspec_plus_plus.wire.contract import OutcomeOffset
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
    if setting(run_config, 'engine.engine_settings.chain_tail_trace', '0') == '1':
        print(f"MAILBOX_ID side=verify event=command_publish id={command_id} command={wire.decode('ascii')}", flush=True)
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
    """Fixed NCCL storage; each sequence owns its dense core and shipped state.

    ``packet`` and ``taps`` retain the single-sequence ABI. Batched callers use
    ``sequences`` and pass one packet, tap block and state mapping per sequence.
    """

    def __init__(self, lanes, hidden, dtype, device, *, layout=None):
        self.layout = layout if layout is not None else OutcomeLayout(lanes, hidden, dtype)
        self.lanes = self.layout.lanes
        self.hidden = self.layout.hidden
        self.dtype = self.layout.tap_dtype
        self.tensor = torch.empty(self.layout.outcome_bytes, dtype=torch.uint8, device=device)
        self.sequences = self.layout.views(self.tensor)
        self.packet, self.taps = self.sequences[0].packet, self.sequences[0].taps

    def pack(self, packet, taps, *, skip_full_zero=False, states=None):
        if self.layout.sequence_count > 1:
            if (len(packet) != self.layout.sequence_count or len(taps) != self.layout.sequence_count
                    or states is not None and len(states) != self.layout.sequence_count):
                raise ValueError('outcome batch must match the sequence count')
            rows = zip(self.sequences, packet, taps,
                       states if states is not None else [{} for _ in self.sequences])
        else:
            rows = [(self.sequences[0], packet, taps, states or {})]
        for row, header, block, state in rows:
            self._pack_sequence(row, header, block, state, skip_full_zero=skip_full_zero)
        return self.tensor

    def _pack_sequence(self, row, packet, taps, states, *, skip_full_zero):
        if packet.shape != (OUTCOME_WORDS,) or packet.dtype != torch.int64:
            raise ValueError(f"OutcomePacket must be int64[{OUTCOME_WORDS}]")
        if taps.ndim != 3 or taps.shape[0] > self.lanes or taps.shape[1:] != row.taps.shape[1:]:
            raise ValueError("outcome taps exceed the fixed lane block")
        row.packet.copy_(packet)
        if not skip_full_zero or taps.shape[0] != self.lanes:
            row.taps.zero_()
        row.taps[:taps.shape[0]].copy_(taps.to(self.dtype))
        self._copy_state(row, states)

    @staticmethod
    def _copy_state(row, states):
        if states.keys() != row.states.keys():
            raise ValueError('outcome state must match the shipped layers')
        for layer, destination in row.states.items():
            source = states[layer]
            if len(source) != len(destination) or any(
                    a.shape != b.shape or a.dtype != b.dtype for a, b in zip(source, destination)):
                raise ValueError('outcome state geometry or dtype changed')
            for target, value in zip(destination, source):
                target.copy_(value)

    def prepare_taps(self, taps):
        """Copy the invariant full tap block before host acceptance completes."""
        blocks = [taps] if self.layout.sequence_count == 1 else taps
        if len(blocks) != self.layout.sequence_count:
            raise ValueError('outcome batch must match the sequence count')
        for row, block in zip(self.sequences, blocks):
            if block.shape != row.taps.shape:
                raise ValueError("early outcome preparation requires every tap lane")
            row.taps.copy_(block.to(self.dtype))

    def pack_prepared_header(self, packet, *, states=None):
        packets = [packet] if self.layout.sequence_count == 1 else packet
        state_rows = [states or {}] if self.layout.sequence_count == 1 else states
        if state_rows is None:
            state_rows = [{} for _ in self.sequences]
        if len(packets) != self.layout.sequence_count or len(state_rows) != self.layout.sequence_count:
            raise ValueError('outcome batch must match the sequence count')
        for row, header, state in zip(self.sequences, packets, state_rows):
            if header.shape != (OUTCOME_WORDS,) or header.dtype != torch.int64:
                raise ValueError(f"OutcomePacket must be int64[{OUTCOME_WORDS}]")
            row.packet.copy_(header)
            self._copy_state(row, state)
        return self.tensor

    def state_digests(self):
        """Optional audit: one committed-state digest per sequence and round."""
        from jetspec.engines.jetspec_plus_plus.wire.hybrid import state_digest
        return tuple((int(row.packet[OutcomeOffset.SEQ_ID]), int(row.packet[OutcomeOffset.ROUND]),
                      state_digest(*(tensor for section in self.layout.sections
                          if section.kind == 'linear_attention'
                          for tensor in row.states[section.layer_index])))
                     for row in self.sequences)


class OutcomeWireRing:
    """Keep each packed envelope until its asynchronous send completes."""

    def __init__(self, lanes, hidden, dtype, device, *, layout=None):
        self.wires = [OutcomeWire(lanes, hidden, dtype, device, layout=layout) for _ in range(2)]
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


def outcome_layout(owner):
    """Resolve a session layout, retaining the geometry-only public ABI."""
    layout = getattr(owner, '_outcome_layout', None)
    if layout is not None:
        return layout
    legacy = getattr(owner, '_hybrid_pair_layout', None)
    if legacy is not None:
        return legacy.layout
    return OutcomeLayout(owner.K + 1, owner.jetspec_tap_hidden_size, owner.draft_dtype)


def make_outcome_wire(owner, *, ring=False):
    """Build every verify-side wire from the session layout.

    Geometry-only callers retain their historical constructors; serving sessions
    supply the ModelSpec-derived layout once, before the first round.
    """
    layout = getattr(owner, '_outcome_layout', None)
    if layout is not None:
        cls = OutcomeWireRing if ring else OutcomeWire
        return cls(layout.lanes, layout.hidden, layout.tap_dtype, owner.device, layout=layout)
    legacy = getattr(owner, '_hybrid_pair_layout', None)
    if legacy is not None:
        from jetspec.engines.jetspec_plus_plus.wire.hybrid import HybridOutcomeWire, HybridOutcomeWireRing
        cls = HybridOutcomeWireRing if ring else HybridOutcomeWire
        return cls(legacy, owner.device)
    cls = OutcomeWireRing if ring else OutcomeWire
    return cls(owner.K + 1, owner.jetspec_tap_hidden_size, owner.draft_dtype, owner.device)
