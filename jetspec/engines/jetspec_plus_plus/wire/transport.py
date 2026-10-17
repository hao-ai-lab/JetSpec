
from __future__ import annotations

from jetspec.engines.jetspec_plus_plus.profiles import kernel_enabled, kernel_value
from dataclasses import dataclass
import os
import time
import torch
from jetspec.engines.jetspec_plus_plus.wire.contract import OUTCOME_WORDS
from jetspec.engines.jetspec_plus_plus.wire.contract import OutcomeLayout
from jetspec.engines.jetspec_plus_plus.wire.contract import FrontInputs
from jetspec.engines.jetspec_plus_plus.wire.contract import Offset
from jetspec.engines.jetspec_plus_plus.wire.contract import OutcomeOffset
from jetspec.engines.jetspec_plus_plus.wire.contract import OutcomePacket
from jetspec.engines.jetspec_plus_plus.wire.contract import TokenBuffers
from jetspec.engines.jetspec_plus_plus.wire.contract import outcome_message_views
from jetspec.engines.jetspec_plus_plus.wire.contract import rng_words
from jetspec.engines.jetspec_plus_plus.jetspec.chain.intake import CHAIN_COMMAND_PREFIX

# After a request's last EOS the target writes the read's outputs, audits its compile caches
# and only then sends "exit"; on a shared filesystem that can take longer than 30 s.
BOUNDARY_WAIT_S = 600.0

class ChainCommandInbox:
    """Read host-known command types; NCCL carries payloads only."""

    def __init__(self, store, transport=None, *, run_config=None):
        from jetspec.engines.jetspec_plus_plus.serving.settings import setting
        self.run_config = run_config
        self.store = store
        self.transport = transport
        self.pair_check = setting(self, 'engine.engine_settings.pair_outcome_push_check', '0') == "1"
        self.pair_source = "tcp"
        self.next_id = 0
        self.last_outcome = None
        self.last_eos = None
        self.last_raw_command = None
        self.last_command_id = None
        self.last_header_mismatch = False

    def next(self) -> tuple[int, str]:
        from jetspec.engines.jetspec_plus_plus.serving.settings import setting
        from jetspec.engines.jetspec_plus_plus.profiles import kernel_enabled
        command_id = self.next_id
        if self.transport is not None and self.pair_source == "boundary":
            # A device EOS can precede the host's request-end decision. Accept
            # either a continued pushed round or the next TCP control.
            key = f"{CHAIN_COMMAND_PREFIX}{command_id}"
            deadline = time.monotonic() + BOUNDARY_WAIT_S
            while True:
                if self.store.check([key]):
                    boundary = self.store.get(key).decode("ascii")
                    if boundary in ("prefill", "pin_served", "exit"):
                        self.pair_source = "tcp"
                        break
                try:
                    self.transport.poll(command_id, timeout_s=0)
                except TimeoutError:
                    pass
                else:
                    self.pair_source = "push"
                    print(f"PAIR_OUTCOME_CONTINUE command_id={command_id}", flush=True)
                    break
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"pair command {command_id} missing after EOS")
                time.sleep(0)
        if self.transport is not None and self.pair_source == "push":
            words = self.transport.poll(command_id)
            if words[OutcomeOffset.CMD] != 1 or words[OutcomeOffset.ROUND] < 0:
                raise RuntimeError("pushed chain outcome has an invalid packet")
            eos = int(words[OutcomeOffset.EOS_FLAG])
            redraft = setting(self, 'engine.engine_settings.chain_miss_redraft', '0') == "1"
            prepost = setting(self, 'engine.engine_settings.chain_prepost', '0') == "1"
            if redraft:
                self.last_outcome = (int(words[OutcomeOffset.ROUND]),
                                     int(words[OutcomeOffset.L]),
                                     int(words[OutcomeOffset.TRUE_K]),
                                     int(words[OutcomeOffset.RECOVERY]))
                raw = "outcome:" + ":".join(map(str, (*self.last_outcome, eos)))
            elif prepost:
                self.last_outcome = (int(words[OutcomeOffset.ROUND]),
                                     int(words[OutcomeOffset.L]), 0)
                raw = "outcome:" + ":".join(map(str, (*self.last_outcome, eos)))
            else:
                self.last_outcome = None
                raw = "outcome"
            self.last_eos = eos
            self.last_raw_command = raw
            self.last_header_mismatch = False
            if self.pair_check:
                original = self.store.get(f"{CHAIN_COMMAND_PREFIX}{command_id}").decode("ascii")
                self.last_header_mismatch = original != raw
            self.last_command_id = command_id
            self.next_id += 1
            if eos:
                self.pair_source = "boundary"
            return command_id, "outcome"
        command = self.store.get(f"{CHAIN_COMMAND_PREFIX}{command_id}").decode("ascii")
        self.last_raw_command = command
        self.last_outcome = None
        self.last_eos = None
        if setting(self, 'engine.engine_settings.chain_miss_redraft', '0') == "1" and command.startswith("outcome:"):
            # 5f: outcome:<round>:<L>:<TRUE_K>:<RECOVERY>:<EOS>, with or without PREPOST.
            parts = command.split(":")
            if (len(parts) != 6 or any(not part.isdecimal() for part in parts[1:])
                    or parts[5] not in ("0", "1") or int(parts[2]) < 1):
                raise RuntimeError("malformed miss-redraft outcome command")
            self.last_outcome = tuple(map(int, parts[1:5]))
            self.last_eos = int(parts[5])
            command = "outcome"
        elif setting(self, 'engine.engine_settings.chain_prepost', '0') == "1" and command.startswith("outcome:"):
            parts = command.split(":")
            if (len(parts) != 5 or any(not part.isdecimal() for part in parts[1:])
                    or parts[3] not in ("0", "1") or parts[4] not in ("0", "1")
                    or int(parts[2]) < 1):
                raise RuntimeError("malformed prepost outcome command")
            self.last_outcome = tuple(map(int, parts[1:4]))
            self.last_eos = int(parts[4])
            command = "outcome"
        if command not in ("prefill", "warmup", "pin_served", "exit", "outcome"):
            raise RuntimeError(f"unknown chain host command: {command}")
        if (command == "outcome" and
                (setting(self, 'engine.engine_settings.chain_miss_redraft', '0') == "1" or
                 setting(self, 'engine.engine_settings.chain_prepost', '0') == "1") and
                self.last_outcome is None):
            raise RuntimeError("chain outcome omitted exact host control")
        self.last_command_id = command_id
        self.next_id += 1
        if self.transport is not None and command == "prefill":
            self.pair_source = "push"
        return command_id, command

from dataclasses import dataclass
import os
import time
import torch
from jetspec.engines.jetspec_plus_plus.wire.contract import OUTCOME_WORDS
from jetspec.engines.jetspec_plus_plus.wire.contract import FrontInputs
from jetspec.engines.jetspec_plus_plus.wire.contract import Offset
from jetspec.engines.jetspec_plus_plus.wire.contract import OutcomeOffset
from jetspec.engines.jetspec_plus_plus.wire.contract import OutcomePacket
from jetspec.engines.jetspec_plus_plus.wire.contract import TokenBuffers
from jetspec.engines.jetspec_plus_plus.wire.contract import outcome_message_views
from jetspec.engines.jetspec_plus_plus.wire.contract import rng_words
from jetspec.engines.jetspec_plus_plus.jetspec.chain.intake import CHAIN_RECV_READY_PREFIX

class OutcomeReceiveRing:
    """Two fixed NCCL receive envelopes with async packet mirrors.

    The sender waits for the host ready key before posting its outcome send.
    ``take`` orders the receive on the current stream without reading fields.
    The caller releases a slot only after the Seg P publish event.
    """

    def __init__(self, *, device, lanes: int, hidden: int,
                 tap_dtype=torch.bfloat16, recv=None, event_factory=None,
                 hybrid_layout=None, run_config=None):
        from jetspec.engines.jetspec_plus_plus.serving.settings import setting
        self.run_config = run_config
        import torch.distributed as dist

        self.recv = recv or dist.irecv
        self.event_factory = event_factory or (lambda: torch.cuda.Event())
        layout = (hybrid_layout.layout if hybrid_layout is not None
                  else OutcomeLayout(lanes, hidden, tap_dtype))
        size = layout.outcome_bytes
        self.hybrid_transfer_bytes = (size if hybrid_layout is not None else None)
        self.hybrid_transfer_timing = (hybrid_layout is not None and
                                       setting(self, 'engine.engine_settings.hybrid_transfer_timing', '0') == "1")
        self._hybrid_transfer_pending = []
        self.buffers = [torch.empty(size, dtype=torch.uint8, device=device) for _ in range(2)]
        if hybrid_layout is None:
            self.views = [outcome_message_views(buffer, lanes=lanes, hidden=hidden,
                                                tap_dtype=tap_dtype) for buffer in self.buffers]
            self.hybrid_views = None
        else:
            from jetspec.engines.jetspec_plus_plus.wire.hybrid import hybrid_outcome_views
            parsed = [hybrid_outcome_views(buffer, hybrid_layout) for buffer in self.buffers]
            self.views = [row[:2] for row in parsed]
            self.hybrid_views = [row[2:] for row in parsed]
        pinned = torch.device(device).type == "cuda"
        self.mirrors = [torch.empty(OUTCOME_WORDS, dtype=torch.int64,
                                    pin_memory=pinned) for _ in range(2)]
        self.pending = [None, None]
        self.posted_command_ids = [None, None]
        self.mirror_events = [None, None]
        self.mirror_rounds = [None, None]
        self.release_events = [None, None]
        self.last_received_round = -1
        self.eos_round = None

    def post(self, round_id: int, *, group, store, command_id: int, src=0) -> None:
        slot = round_id % 2
        if self.pending[slot] is not None or self.mirror_events[slot] is not None:
            raise RuntimeError("chain receive slot is still occupied")
        released = self.release_events[slot]
        if released is not None and not released.query():
            raise RuntimeError("chain receive slot still has device consumers")
        self.release_events[slot] = None
        self.pending[slot] = (round_id, self.recv(self.buffers[slot], src=src, group=group))
        self.posted_command_ids[slot] = command_id
        store.set(f"{CHAIN_RECV_READY_PREFIX}{command_id}", b"1")

    def is_posted(self, round_id: int, command_id: int) -> bool:
        slot = round_id % 2
        return (self.pending[slot] is not None and self.pending[slot][0] == round_id
                and self.posted_command_ids[slot] == command_id)

    def take(self, round_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        slot = round_id % 2
        pending = self.pending[slot]
        if pending is None or pending[0] != round_id:
            raise RuntimeError("chain outcome receive was not posted")
        if self.hybrid_transfer_timing:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
        pending[1].wait()
        if self.hybrid_transfer_timing:
            end.record()
            self._hybrid_transfer_pending.append((round_id, start, end))
        self.pending[slot] = None
        self.posted_command_ids[slot] = None
        self.last_received_round = round_id
        return self.views[slot]

    def stage_mirror(self, round_id: int) -> None:
        slot = round_id % 2
        if self.last_received_round < round_id or self.mirror_events[slot] is not None:
            raise RuntimeError("chain mirror requires a newly received outcome")
        self.mirrors[slot].copy_(self.views[slot][0], non_blocking=True)
        event = self.event_factory()
        event.record()
        self.mirror_events[slot] = event
        self.mirror_rounds[slot] = round_id

    def poll_mirrors(self) -> list[OutcomePacket]:
        completed = []
        if self.hybrid_transfer_timing:
            waiting = []
            for round_id, start, end in self._hybrid_transfer_pending:
                if end.query():
                    print("HYBRID_OUTCOME_TRANSFER "
                          f"round={round_id} bytes={self.hybrid_transfer_bytes} "
                          f"recv_wait_ms={start.elapsed_time(end):.6f}", flush=True)
                else:
                    waiting.append((round_id, start, end))
            self._hybrid_transfer_pending = waiting
        for slot in range(2):
            event = self.mirror_events[slot]
            if event is None or not event.query():
                continue
            packet = OutcomePacket.unpack(self.mirrors[slot])
            if packet.round != self.mirror_rounds[slot]:
                raise RuntimeError("chain outcome mirror round mismatch")
            if packet.eos_flag:
                self.eos_round = packet.round
            completed.append(packet)
            self.mirror_events[slot] = None
            self.mirror_rounds[slot] = None
        return sorted(completed, key=lambda packet: packet.round)

    def release(self, round_id: int, event) -> None:
        self.release_events[round_id % 2] = event

    def can_retire(self, round_id: int) -> bool:
        if self.eos_round is None or self.last_received_round < self.eos_round:
            return False
        if any(pending is not None for pending in self.pending):
            return False
        event = self.release_events[self.eos_round % 2]
        return event is not None and event.query() and round_id >= self.eos_round

    @staticmethod
    def slot_for_round(round_id: int) -> int:
        return round_id % 2

from dataclasses import dataclass
import os
import time
import torch
from jetspec.engines.jetspec_plus_plus.wire.contract import OUTCOME_WORDS
from jetspec.engines.jetspec_plus_plus.wire.contract import FrontInputs
from jetspec.engines.jetspec_plus_plus.wire.contract import Offset
from jetspec.engines.jetspec_plus_plus.wire.contract import OutcomeOffset
from jetspec.engines.jetspec_plus_plus.wire.contract import OutcomePacket
from jetspec.engines.jetspec_plus_plus.wire.contract import TokenBuffers
from jetspec.engines.jetspec_plus_plus.wire.contract import outcome_message_views
from jetspec.engines.jetspec_plus_plus.wire.contract import rng_words

class PushedOutcomeRing:
    """Read the draft-visible slot, with the NCCL receive as an optional check."""

    def __init__(self, *, transport, inbox: ChainCommandInbox, lanes: int, hidden: int,
                 tap_dtype: torch.dtype, shadow: OutcomeReceiveRing | None = None,
                 hybrid_layout=None):
        self.transport = transport
        self.inbox = inbox
        self.shadow = shadow
        self.buffers = [transport.slot(slot) for slot in range(3)]
        if hybrid_layout is None:
            self.views = [outcome_message_views(buffer, lanes=lanes, hidden=hidden,
                                                tap_dtype=tap_dtype) for buffer in self.buffers]
            self.hybrid_views = None
        else:
            from jetspec.engines.jetspec_plus_plus.wire.hybrid import hybrid_outcome_views
            parsed = [hybrid_outcome_views(buffer, hybrid_layout) for buffer in self.buffers]
            self.views = [row[:2] for row in parsed]
            self.hybrid_views = [row[2:] for row in parsed]
        self.mirrors = [torch.empty(OUTCOME_WORDS, dtype=torch.int64, pin_memory=True)
                        for _ in range(3)]
        self.mirror_events = [None, None, None]
        self.mirror_rounds = [None, None, None]
        self.release_events = [None, None, None]
        self.copy_events = [None, None, None]
        self.round_slots = {}
        self.round_commands = {}
        self.last_received_round = -1
        self.eos_round = None
        self.check_rounds = 0
        self.check_mismatches = 0

    def slot_for_round(self, round_id: int) -> int:
        return self.round_slots[round_id]

    def post(self, round_id: int, *, group, store, command_id: int, src=0) -> None:
        if self.shadow is not None:
            self.shadow.post(round_id, group=group, store=store, command_id=command_id, src=src)

    def is_posted(self, round_id: int, command_id: int) -> bool:
        return self.shadow is None or self.shadow.is_posted(round_id, command_id)

    def take(self, round_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        command_id = self.inbox.last_command_id
        slot = command_id % 3
        self.round_slots[round_id] = slot
        self.round_commands[round_id] = command_id
        pushed = self.views[slot]
        if self.shadow is not None:
            received = self.shadow.take(round_id)
            mismatch = (self.inbox.last_header_mismatch or
                        not torch.equal(self.buffers[slot], self.shadow.buffers[round_id % 2]))
            self.check_rounds += 1
            self.check_mismatches += int(mismatch)
            print(f"PAIR_OUTCOME_CHECK rounds={self.check_rounds} "
                  f"mismatches={self.check_mismatches}", flush=True)
            if mismatch:
                raise RuntimeError("pushed outcome differs from TCP/NCCL outcome")
            del received
        self.last_received_round = round_id
        return pushed

    def stage_mirror(self, round_id: int) -> None:
        slot = self.slot_for_round(round_id)
        if self.last_received_round < round_id or self.mirror_events[slot] is not None:
            raise RuntimeError("chain mirror requires a newly received outcome")
        self.mirrors[slot].copy_(self.views[slot][0], non_blocking=True)
        event = torch.cuda.Event()
        event.record()
        self.mirror_events[slot] = event
        self.mirror_rounds[slot] = round_id

    def poll_mirrors(self) -> list[OutcomePacket]:
        completed = []
        for slot, event in enumerate(self.mirror_events):
            if event is None or not event.query():
                continue
            packet = OutcomePacket.unpack(self.mirrors[slot])
            if packet.round != self.mirror_rounds[slot]:
                raise RuntimeError("chain outcome mirror round mismatch")
            if packet.eos_flag:
                self.eos_round = packet.round
            completed.append(packet)
            self.mirror_events[slot] = None
            self.mirror_rounds[slot] = None
        return sorted(completed, key=lambda packet: packet.round)

    def mark_copy_done(self, round_id: int) -> None:
        slot = self.slot_for_round(round_id)
        event = torch.cuda.Event()
        event.record()
        self.copy_events[slot] = event

    def release(self, round_id: int, event) -> None:
        slot = self.slot_for_round(round_id)
        copied = self.copy_events[slot]
        if copied is None:
            raise RuntimeError("pair slot release lacks a completed device copy")
        self.transport.release(self.round_commands.pop(round_id), copied)
        self.release_events[slot] = event
        if self.shadow is not None:
            self.shadow.release(round_id, event)

    def can_retire(self, round_id: int) -> bool:
        if self.eos_round is None or self.last_received_round < self.eos_round:
            return False
        slot = self.slot_for_round(self.eos_round)
        event = self.release_events[slot]
        return event is not None and event.query() and round_id >= self.eos_round
