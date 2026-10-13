"""Device-side chain outcome intake and the published-row mirror.

The ring owns copies of the exact rows placed on the wire. Its entries never
alias live refinement outputs, which may change after an early publication.
"""
from __future__ import annotations

from dataclasses import dataclass
import os
import time

import torch

from jetspec.engines.jetspec_plus_plus.wire.chain.contract import OUTCOME_WORDS; from jetspec.engines.jetspec_plus_plus.wire.chain.contract import FrontInputs; from jetspec.engines.jetspec_plus_plus.wire.chain.contract import Offset; from jetspec.engines.jetspec_plus_plus.wire.chain.contract import OutcomeOffset; from jetspec.engines.jetspec_plus_plus.wire.chain.contract import OutcomePacket; from jetspec.engines.jetspec_plus_plus.wire.chain.contract import TokenBuffers; from jetspec.engines.jetspec_plus_plus.wire.chain.contract import outcome_message_views; from jetspec.engines.jetspec_plus_plus.wire.chain.contract import rng_words


CHAIN_ERR_CAPACITY = 1 << 16
CHAIN_ERR_ROW_MISSING = 1 << 17
CHAIN_ERR_DRIFT = 1 << 18
CHAIN_ERR_PACKET = 1 << 19
CHAIN_ERR_PENDING = 1 << 20
CHAIN_ERR_CONTROL = 1 << 21
CHAIN_ERR_AFTER_MISS = 1 << 22
CHAIN_COMMAND_PREFIX = "chain_command_"
CHAIN_RECV_READY_PREFIX = "chain_recv_ready_"
HASH_WEIGHTS = (
    5915587277, 12608681917, 20056049013, 27877307239,
    36117146159, 44762267993, 53812224539, 63267073067,
    73126780073, 83391345563, 94060769543, 105534498047,
    116007347237, 127284625129, 138965789647, 151352719069,
)


class ChainCommandInbox:
    """Read host-known command types; NCCL carries payloads only."""

    def __init__(self, store, transport=None):
        self.store = store
        self.transport = transport
        self.pair_check = os.environ.get("JETSPEC_PAIR_OUTCOME_PUSH_CHECK", "0") == "1"
        self.pair_source = "tcp"
        self.next_id = 0
        self.last_outcome = None
        self.last_eos = None
        self.last_raw_command = None
        self.last_command_id = None
        self.last_header_mismatch = False

    def next(self) -> tuple[int, str]:
        command_id = self.next_id
        if self.transport is not None and self.pair_source == "boundary":
            # A device EOS can precede the host's request-end decision. Accept
            # either a continued pushed round or the next TCP control.
            key = f"{CHAIN_COMMAND_PREFIX}{command_id}"
            deadline = time.monotonic() + 30.0
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
            redraft = os.environ.get("JETSPEC_CHAIN_MISS_REDRAFT", "0") == "1"
            prepost = os.environ.get("JETSPEC_CHAIN_PREPOST", "0") == "1"
            attn3 = os.environ.get("JETSPEC_CHAIN_ATTN3", "0") == "1"
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
            elif attn3:
                self.last_outcome = (int(words[OutcomeOffset.ROUND]),
                                     int(words[OutcomeOffset.L]))
                raw = "outcome:" + ":".join(map(str, self.last_outcome))
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
        if os.environ.get("JETSPEC_CHAIN_MISS_REDRAFT", "0") == "1" and command.startswith("outcome:"):
            # 5f: outcome:<round>:<L>:<TRUE_K>:<RECOVERY>:<EOS>, with or without PREPOST.
            parts = command.split(":")
            if (len(parts) != 6 or any(not part.isdecimal() for part in parts[1:])
                    or parts[5] not in ("0", "1") or int(parts[2]) < 1):
                raise RuntimeError("malformed miss-redraft outcome command")
            self.last_outcome = tuple(map(int, parts[1:5]))
            self.last_eos = int(parts[5])
            command = "outcome"
        elif os.environ.get("JETSPEC_CHAIN_PREPOST", "0") == "1" and command.startswith("outcome:"):
            parts = command.split(":")
            if (len(parts) != 5 or any(not part.isdecimal() for part in parts[1:])
                    or parts[3] not in ("0", "1") or parts[4] not in ("0", "1")
                    or int(parts[2]) < 1):
                raise RuntimeError("malformed prepost outcome command")
            self.last_outcome = tuple(map(int, parts[1:4]))
            self.last_eos = int(parts[4])
            command = "outcome"
        elif os.environ.get("JETSPEC_CHAIN_ATTN3", "0") == "1" and command.startswith("outcome:"):
            parts = command.split(":")
            if len(parts) != 3 or any(not part.isdecimal() for part in parts[1:]) or int(parts[2]) < 1:
                raise RuntimeError("malformed ATTN3 outcome command")
            self.last_outcome = tuple(map(int, parts[1:]))
            command = "outcome"
        if command not in ("prefill", "warmup", "pin_served", "exit", "outcome"):
            raise RuntimeError(f"unknown chain host command: {command}")
        if (command == "outcome" and
                (os.environ.get("JETSPEC_CHAIN_MISS_REDRAFT", "0") == "1" or
                 os.environ.get("JETSPEC_CHAIN_ATTN3", "0") == "1" or
                 os.environ.get("JETSPEC_CHAIN_PREPOST", "0") == "1") and
                self.last_outcome is None):
            raise RuntimeError("chain outcome omitted exact host control")
        self.last_command_id = command_id
        self.next_id += 1
        if self.transport is not None and command == "prefill":
            self.pair_source = "push"
        return command_id, command


class OutcomeReceiveRing:
    """Two fixed NCCL receive envelopes with async packet mirrors.

    The sender waits for the host ready key before posting its outcome send.
    ``take`` orders the receive on the current stream without reading fields.
    The caller releases a slot only after the Seg P publish event.
    """

    def __init__(self, *, device, lanes: int, hidden: int,
                 tap_dtype=torch.bfloat16, recv=None, event_factory=None,
                 hybrid_layout=None):
        import torch.distributed as dist

        self.recv = recv or dist.irecv
        self.event_factory = event_factory or (lambda: torch.cuda.Event())
        tap_bytes = lanes * 5 * hidden * torch.empty((), dtype=tap_dtype).element_size()
        size = (hybrid_layout.outcome_bytes if hybrid_layout is not None
                else OUTCOME_WORDS * 8 + tap_bytes)
        self.hybrid_transfer_bytes = (size if hybrid_layout is not None else None)
        self.hybrid_transfer_timing = (hybrid_layout is not None and
                                       os.environ.get("JETSPEC_HYBRID_TRANSFER_TIMING", "0") == "1")
        self._hybrid_transfer_pending = []
        self.buffers = [torch.empty(size, dtype=torch.uint8, device=device) for _ in range(2)]
        if hybrid_layout is None:
            self.views = [outcome_message_views(buffer, lanes=lanes, hidden=hidden,
                                                tap_dtype=tap_dtype) for buffer in self.buffers]
            self.hybrid_views = None
        else:
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


def fill_front_from_outcome(front: FrontInputs, packet: torch.Tensor,
                            taps: torch.Tensor, tap_store) -> None:
    """Roll accepted old predictions into the decoder tail, then stage new taps."""
    if taps.shape != front.taps.shape or taps.device != front.taps.device:
        raise ValueError("outcome taps do not match bound front inputs")
    stale = front.decoder_stale
    hidden = stale.shape[-1]
    # The target queries the old correction as well as the accepted drafts.
    # All tap_count predicted rows advance the decoder's four-row window.
    accepted = packet[OutcomeOffset.TAP_COUNT].clamp(0, front.lanes)
    old_predictions = front.staged_hidden[0, :, -hidden:]
    combined = torch.cat((stale, old_predictions), dim=0)
    positions = accepted + torch.arange(stale.shape[0], device=stale.device)
    stale.copy_(combined.index_select(0, positions))
    front.taps.copy_(taps)
    front.decoder_prefix_taps.copy_(tap_store.decoder_prefix(packet))


def hash_last16(ids: torch.Tensor, length: torch.Tensor,
                weights: torch.Tensor | None = None) -> torch.Tensor:
    """Shared verifier/draft signed-int64 rolling digest of the valid last 16 IDs.

    Arithmetic uses int64 wraparound on both ranks. Invalid left padding is
    zero; the length term distinguishes otherwise identical short prefixes.
    """
    if ids.ndim != 1 or ids.dtype != torch.int64 or length.numel() != 1:
        raise ValueError("hash16 expects int64[capacity] IDs and one length")
    positions = length.to(torch.int64).reshape(()) - 16 + torch.arange(16, device=ids.device)
    selected = ids[positions.clamp(0, ids.numel() - 1)]
    selected = torch.where((positions >= 0) & (positions < length), selected, 0)
    if weights is None:
        if ids.is_cuda:
            raise ValueError("capture requires pre-staged hash16 weights")
        weights = torch.tensor(HASH_WEIGHTS, dtype=torch.int64)
    if weights.shape != (16,) or weights.dtype != torch.int64 or weights.device != ids.device:
        raise ValueError("hash16 weights must be int64[16] on the token device")
    return ((selected + 1) * weights).sum() + length.reshape(()) * 1000003


class PublishedRowRing:
    """Three reusable device snapshots, indexed by published round modulo 3."""

    def __init__(self, *, device: torch.device | str, rows: int, lanes: int):
        self.rows = torch.zeros(3, rows, lanes, dtype=torch.int64, device=device)
        self.p_top1 = torch.zeros(3, rows, lanes, dtype=torch.float32, device=device)
        self.passes_run = torch.zeros(3, rows, dtype=torch.int64, device=device)
        self.keys = torch.full((3, rows, 2), -1, dtype=torch.int64, device=device)
        self.rounds = torch.full((3,), -1, dtype=torch.int64, device=device)
        from jetspec.engines.jetspec_plus_plus.draft.eval_profile import capture_candidates
        if capture_candidates():
            self.draft_top16_ids = torch.full((3, rows, lanes, 16), -1, dtype=torch.long, device=device)
            self.draft_top16_probabilities = torch.zeros((3, rows, lanes, 16), device=device)
            self.draft_token_probabilities = torch.zeros((3, rows, lanes), device=device)
            self.draft_pinned = torch.zeros((3, rows, lanes), device=device, dtype=torch.bool)

    def publish_candidates(self, round_id, ids, probabilities, token_probabilities, pinned):
        """Optional graph-owned distribution snapshots in the same row order."""
        slot = torch.arange(3, device=self.rows.device) == round_id % 3
        self.draft_top16_ids.copy_(torch.where(slot[:, None, None, None], ids[None],
                                             self.draft_top16_ids))
        self.draft_top16_probabilities.copy_(torch.where(slot[:, None, None, None], probabilities[None],
                                                       self.draft_top16_probabilities))
        self.draft_token_probabilities.copy_(torch.where(slot[:, None, None], token_probabilities[None],
                                                       self.draft_token_probabilities))
        self.draft_pinned.copy_(torch.where(slot[:, None, None], pinned[None], self.draft_pinned))

    def select_pending_candidates(self, packet):
        index, _, valid = self.index_for_pending(packet)
        slot = packet[OutcomeOffset.ROUND].remainder(3).reshape(1)
        return tuple(torch.where(valid, value.index_select(0, slot).squeeze(0)
                                 .index_select(0, index).squeeze(0), torch.zeros_like(value[0, 0]))
                     for value in (self.draft_top16_ids, self.draft_top16_probabilities,
                                   self.draft_token_probabilities, self.draft_pinned))

    def publish(self, round_id: int | torch.Tensor, rows: torch.Tensor,
                keys: torch.Tensor, p_top1: torch.Tensor | None = None,
                passes_run: torch.Tensor | None = None) -> None:
        if rows.shape != self.rows.shape[1:] or keys.shape != self.keys.shape[1:]:
            raise ValueError("published row snapshot has wrong shape")
        if p_top1 is None:
            p_top1 = torch.zeros_like(self.p_top1[0])
        if (p_top1.shape != self.p_top1.shape[1:] or p_top1.dtype != torch.float32
                or p_top1.device != self.p_top1.device):
            raise ValueError("published top-1 snapshot must be float32[R, lanes]")
        if passes_run is None:
            passes_run = torch.zeros_like(self.passes_run[0])
        if (passes_run.shape != self.passes_run.shape[1:] or passes_run.dtype != torch.int64
                or passes_run.device != self.passes_run.device):
            raise ValueError("published pass labels must be int64[R]")
        if isinstance(round_id, torch.Tensor):
            if round_id.numel() != 1 or round_id.dtype != torch.int64 or round_id.device != self.rows.device:
                raise ValueError("published round must be an int64 device scalar")
            round_word = round_id
        else:
            round_word = torch.as_tensor(round_id, dtype=torch.int64, device=self.rows.device)
        slot = torch.arange(3, device=self.rows.device) == round_word.remainder(3)
        self.rows.copy_(torch.where(slot[:, None, None], rows[None], self.rows))
        self.p_top1.copy_(torch.where(slot[:, None, None], p_top1[None], self.p_top1))
        self.passes_run.copy_(torch.where(slot[:, None], passes_run[None], self.passes_run))
        self.keys.copy_(torch.where(slot[:, None, None], keys[None], self.keys))
        self.rounds.copy_(torch.where(slot, round_word, self.rounds))

    def empty(self, round_id: int | torch.Tensor) -> None:
        self.publish(round_id, torch.zeros_like(self.rows[0]),
                     torch.full_like(self.keys[0], -1),
                     torch.zeros_like(self.p_top1[0]),
                     torch.zeros_like(self.passes_run[0]))

    def publish_masked(self, round_id: int | torch.Tensor, rows: torch.Tensor,
                       keys: torch.Tensor, published: torch.Tensor,
                       p_top1: torch.Tensor | None = None,
                       passes_run: torch.Tensor | None = None) -> None:
        """Snapshot the wire rows, or advance the ring with an empty round."""
        if published.shape != () or published.device != self.rows.device:
            raise ValueError("published must be a device scalar")
        visible = published.to(torch.bool)
        self.publish(round_id, torch.where(visible, rows, torch.zeros_like(rows)),
                     torch.where(visible, keys, torch.full_like(keys, -1)),
                     None if p_top1 is None else torch.where(
                         visible, p_top1, torch.zeros_like(p_top1)),
                     None if passes_run is None else torch.where(
                         visible, passes_run, torch.zeros_like(passes_run)))

    def select(self, packet: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        row, _, found = self.select_with_top1(packet)
        return row, found




    def index_for_pending(self, packet: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor,
                                                                torch.Tensor]:
        """Select the current candidate row that eager stages for the next round."""
        published_round = packet[OutcomeOffset.ROUND]
        slot = published_round.remainder(3).reshape(1)
        keys = self.keys.index_select(0, slot).squeeze(0)
        valid = ((self.rounds.index_select(0, slot).squeeze(0) == published_round)
                 & (keys[:, 0] >= 0) & (keys[:, 1] >= 0))
        same_k = valid & (keys[:, 0] == packet[OutcomeOffset.TRUE_K])
        exact = same_k & (keys[:, 1] == packet[OutcomeOffset.RECOVERY])
        fallback = torch.where(same_k.any(), same_k.to(torch.int64).argmax(),
                               valid.to(torch.int64).argmax())
        index = torch.where(exact.any(), exact.to(torch.int64).argmax(), fallback).reshape(1)
        return index, exact.any(), valid.any()

    def select_pending(self, packet: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor,
                                                             torch.Tensor, torch.Tensor,
                                                             torch.Tensor]:
        index, exact, valid = self.index_for_pending(packet)
        slot = packet[OutcomeOffset.ROUND].remainder(3).reshape(1)
        row = self.rows.index_select(0, slot).squeeze(0).index_select(0, index).squeeze(0)
        top1 = self.p_top1.index_select(0, slot).squeeze(0).index_select(0, index).squeeze(0)
        passes = self.passes_run.index_select(0, slot).squeeze(0).index_select(0, index).squeeze(0)
        return row, top1, passes, exact, valid


class ChainTapBuffers:
    """Sequence-owned fixed exact-tap prefix, seeded at prefill and extended in Seg A."""

    def __init__(self, *, device, capacity: int, lanes: int, hidden: int,
                 dtype=torch.bfloat16):
        # Inactive lanes write into distinct scratch rows after the live prefix.
        self.backing = torch.zeros(capacity + lanes, 5, hidden, device=device, dtype=dtype)
        self.storage = self.backing[:capacity]
        self.lanes = lanes
        self.lane = torch.arange(lanes, device=device)

    def seed(self, prompt_taps: torch.Tensor) -> None:
        if (prompt_taps.ndim != 3 or prompt_taps.shape[1:] != self.storage.shape[1:]
                or prompt_taps.shape[0] > self.storage.shape[0]
                or prompt_taps.device != self.storage.device
                or prompt_taps.dtype != self.storage.dtype):
            raise ValueError("prefill taps do not match static exact-tap storage")
        self.storage[:prompt_taps.shape[0]].copy_(prompt_taps)

    def append(self, packet: torch.Tensor, taps: torch.Tensor) -> None:
        if taps.shape != (self.lanes, *self.storage.shape[1:]) or taps.device != self.storage.device:
            raise ValueError("outcome tap block does not match static storage")
        length = packet[OutcomeOffset.L]
        count = packet[OutcomeOffset.TAP_COUNT]
        # The final recovery ID has no exact target tap until next round.
        start = length - count
        capacity = self.storage.shape[0]
        write = ((length <= capacity) & (start >= 0)
                 & (count >= 0) & (count <= self.lanes) & (self.lane < count))
        rows = torch.where(write, start + self.lane, capacity + self.lane)
        self.backing.index_copy_(0, rows, taps)

    def prefix_padded(self, rows: int) -> torch.Tensor:
        if rows > self.storage.shape[0]:
            raise ValueError("tap prefix bucket exceeds sequence capacity")
        return self.storage[:rows].reshape(rows, -1)



def rng_rows_from_one_draw(capacity: int, width: int, seed: int, prompt_index: int):
    """Recover the shared CPU randint words, or return None if the rail differs."""
    def draw(high):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed) ^ int(prompt_index))
        return torch.randint(high, (width,), generator=generator)

    low, high = 1 << 20, (1 << 20) - 1
    first, second = draw(low), draw(high)
    words = first + low * torch.remainder(second - first, high)
    if bool((words >= 1 << 32).any()):
        return None
    rows = torch.zeros(capacity + 1, width, dtype=torch.int64)
    committed = torch.arange(2, capacity + 1, dtype=torch.int64)[:, None]
    rows[2:] = torch.remainder(words[None, :], committed)
    for length in (2, capacity):
        if not torch.equal(rows[length], draw(length)):
            return None
    return rows


class ChainRngTable:
    """Rail-identical CPU RNG indices for every possible committed length.

    The seeded CPU generator resets for each length in the eager first-block
    build. Precompute these tiny index rows at admission; Seg A gathers the
    row selected by the device L without a host read or a pageable upload.
    """

    def __init__(self, *, capacity: int, lanes: int, seed: int,
                 prompt_index: int, device):
        if capacity < 2:
            raise ValueError("RNG table needs room for the query token")
        width = rng_words(lanes)
        one_draw = os.environ.get("JETSPEC_CHAIN_ADMIT_RNG_ONE_DRAW", "0") == "1"
        check = one_draw and os.environ.get("JETSPEC_CHAIN_ADMIT_RNG_ONE_DRAW_CHECK", "0") == "1"
        rows = (rng_rows_from_one_draw(capacity, width, seed, prompt_index)
                if one_draw else None)
        fallback = one_draw and rows is None


        if rows is None:
            rows = loop_rows()
        if check:
            loop = rows if fallback else loop_rows()
            mismatches = int((rows != loop).sum())
            if mismatches:
                rows, fallback = loop, True
            assert torch.equal(rows, loop)
            print(f"CHAIN_ADMIT_RNG_ONE_DRAW_CHECK mismatch={mismatches} fallback={int(fallback)}")
        if one_draw and torch.device(device).type == "cuda":
            self.rows = rows.pin_memory().to(device=device, non_blocking=True)
        else:
            self.rows = rows.to(device=device)

    def fill(self, rp: torch.Tensor) -> None:
        if rp.dtype != torch.int64 or rp.shape[0] < Offset.RNG_IDX + self.rows.shape[1]:
            raise ValueError("RNG table requires the static round params")
        committed = rp[Offset.L] + 1
        indices = self.rows.index_select(0, committed.clamp(0, self.rows.shape[0] - 1).reshape(1))
        rp[Offset.RNG_IDX:Offset.RNG_IDX + self.rows.shape[1]].copy_(indices[0])


def append_outcome(tokens: TokenBuffers, ring: PublishedRowRing,
                   packet: torch.Tensor, err: torch.Tensor,
                   *, hash_weights: torch.Tensor | None = None,
                   max_length: int | None = None,
                   lane_positions: torch.Tensor | None = None) -> torch.Tensor:
    """Append the verified old row; pending comes from the current ring slot."""
    committed = tokens.committed_ids
    row, _, found, _ = ring.select_for_outcome(packet)
    admission = packet[OutcomeOffset.ROUND] == 0
    count = packet[OutcomeOffset.TRUE_K]
    length = packet[OutcomeOffset.L]
    start = length - count
    capacity = committed.numel()
    limit = capacity - 1 if max_length is None else min(capacity - 1, max_length)
    valid_capacity = ((length <= limit) & (start >= 0)
                      & (count >= 0) & (count < row.numel()))
    valid_row = (count == 0) | found
    if lane_positions is None:
        positions = torch.arange(capacity, device=committed.device)
        offsets = positions - start
        # Row lane zero is the already committed query anchor; only draft lanes
        # one through true_k are new IDs before the recovery token.
        accepted = row[(offsets + 1).clamp(0, row.numel() - 1)]
        replacement = torch.where(offsets == count, packet[OutcomeOffset.RECOVERY], accepted)
        write = (~admission) & valid_capacity & valid_row & (offsets >= 0) & (offsets <= count)
        committed.copy_(torch.where(write, replacement, committed))
    else:
        lanes = row.numel()
        if (capacity <= lanes or lane_positions.shape != (lanes,)
                or lane_positions.device != committed.device
                or lane_positions.dtype != torch.int64):
            raise ValueError('lane-sized committed append requires capacity above lanes')
        # At most `lanes` consecutive positions are written. Since capacity is
        # larger, slot 0 or the final slot lies outside that interval and can
        # absorb every inactive lane without changing the committed prefix.
        scratch = torch.where(start > 0, start.new_zeros(()), start.new_full((), capacity - 1))
        write = (~admission) & valid_capacity & valid_row & (lane_positions <= count)
        indices = torch.where(write, start + lane_positions, scratch)
        accepted = row[(lane_positions + 1).clamp_max(lanes - 1)]
        replacement = torch.where(lane_positions == count,
                                  packet[OutcomeOffset.RECOVERY], accepted)
        saved = committed.index_select(0, scratch.reshape(1))[0]
        committed.index_copy_(0, indices, torch.where(write, replacement, saved))
    digest = hash_last16(committed, (length + 1).clamp(0, capacity), hash_weights)
    faults = ((~valid_capacity).to(torch.int64) * CHAIN_ERR_CAPACITY
              | ((count > 0) & ~found).to(torch.int64) * CHAIN_ERR_ROW_MISSING
              | (digest != packet[OutcomeOffset.HASH16]).to(torch.int64) * CHAIN_ERR_DRIFT)
    err[0].bitwise_or_(faults)
    return ~found


@dataclass
class ChainLengthTracker:
    """Host key estimate from the latest completed async packet mirror."""

    capacity: int
    lanes: int
    L_known: int
    mirrored_round: int
    eos: bool = False

    def key_length(self, round_id: int) -> int:
        age = round_id - self.mirrored_round
        if age < 0:
            raise ValueError("chain round precedes the mirrored packet")
        return self.L_known + (age + 1) * self.lanes

    def bucket(self, round_id: int, *, step: int) -> int:
        if step not in (128, 256):
            raise ValueError("chain key step must be 128 or 256")
        length = self.key_length(round_id)
        return ((length + step - 1) // step) * step


    def mirror(self, *, round_id: int, L: int, eos: bool) -> None:
        if round_id <= self.mirrored_round:
            return
        self.L_known, self.mirrored_round, self.eos = L, round_id, eos
