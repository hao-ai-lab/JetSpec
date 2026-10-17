"""N=1 outcome handoff between the verify and draft ranks."""
from __future__ import annotations

from dataclasses import dataclass
import atexit
import ctypes
import mmap
from multiprocessing.shared_memory import SharedMemory
import os
from pathlib import Path
import struct
import sys
import time

import torch


@dataclass(frozen=True)
class OutcomeHeader:
    command_id: int
    seq_id: int
    round_id: int
    L: int = 0
    true_k: int = 0
    recovery: int = 0
    eos: bool = False
    limit: bool = False

    @property
    def slot(self) -> int:
        return self.command_id % 3


class PairRingState:
    """Guard slot ownership and the global command order, including TCP boundaries."""

    def __init__(self):
        self.command_source = "tcp"
        self.next_push_id = 0
        self.next_take_id = 0
        self.slots: list[OutcomeHeader | None] = [None, None, None]
        self.taken: set[int] = set()
        self.released = [-1, -1, -1]
        self.release_waits = 0
        self.ahead_max = 0
        self._slot_commands = [-1, -1, -1]

    def tcp_command(self, command_id: int, command: str) -> None:
        assert self.command_source in ("tcp", "boundary"), "unexpected TCP command"
        assert command_id == self.next_push_id == self.next_take_id, "command id out of order"
        if command not in ("warmup", "prefill", "exit", "pin_served"):
            raise ValueError("unknown TCP command")
        self.next_push_id += 1
        self.next_take_id += 1
        self.command_source = "push" if command == "prefill" else "tcp"

    def push(self, header: OutcomeHeader, *, timeout_s: float = 0.5,
             release_poll=None) -> int:
        assert self.command_source in ("push", "boundary"), "outcome outside request"
        assert header.command_id == self.next_push_id, "command id out of order"
        slot = header.slot
        previous = self.slots[slot]
        if previous is not None:
            deadline = time.monotonic() + timeout_s
            while self.released[slot] < previous.command_id:
                if release_poll is not None:
                    self.released[slot] = max(self.released[slot], release_poll(slot))
                if self.released[slot] >= previous.command_id:
                    break
                self.release_waits += 1
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"outcome slot {slot} unreleased at command {header.command_id}")
                time.sleep(0)
        self.slots[slot] = header
        self.next_push_id += 1
        self.command_source = "push"
        return slot

    def take(self, header: OutcomeHeader) -> int:
        assert self.command_source == "push", "outcome outside request"
        assert header.command_id == self.next_take_id, "command id out of order"
        assert self.slots[header.slot] == header, "outcome header mismatch"
        self.next_take_id += 1
        self.taken.add(header.command_id)
        self.ahead_max = max(self.ahead_max, self.next_push_id - self.next_take_id)
        if header.eos or header.limit:
            self.command_source = "boundary"
        return header.slot

    def release(self, command_id: int) -> None:
        assert command_id in self.taken, "release before take"
        slot = command_id % 3
        assert self.slots[slot] is not None and self.slots[slot].command_id == command_id, \
            "release after overwrite"
        self.released[slot] = command_id
        self.taken.remove(command_id)


_PACKET = struct.Struct("<16q")
_WORD = struct.Struct("<q")
_SLOT_BYTES = _PACKET.size + 3 * _WORD.size
_command_store = None
_next_command_id = 0


class _AttachedSegment:
    """Attach on Linux without registering a second unlink owner.

    Spawned engine ranks share a multiprocessing resource tracker. Registering
    the attached name there and unregistering it on rank 1 removes rank 0's
    entry too, so the owner's later unlink produces a tracker traceback.
    """

    def __init__(self, name: str):
        descriptor = os.open(f"/dev/shm/{name.lstrip('/')}", os.O_RDWR)
        try:
            self.size = os.fstat(descriptor).st_size
            self._mapping = mmap.mmap(descriptor, self.size)
        finally:
            os.close(descriptor)
        self.name = name

    @property
    def buf(self):
        return memoryview(self._mapping)

    def close(self) -> None:
        self._mapping.close()


def start_command_ids(store, warmup_id: int) -> None:
    """Continue the process-lifetime TCP id after its warmup command."""
    global _command_store, _next_command_id
    if _command_store is not None and _command_store is not store:
        raise RuntimeError("pair command store changed during a live pair")
    _command_store = store
    _next_command_id = warmup_id + 1


def claim_command_id(store) -> int:
    global _next_command_id
    if _command_store is not store:
        raise RuntimeError("pair command ids were not initialized at warmup")
    command_id = _next_command_id
    _next_command_id += 1
    return command_id


def announce_tcp(store, command: str) -> int:
    if command not in ("prefill", "pin_served", "exit"):
        raise ValueError("pair TCP command must be prefill, pin_served or exit")
    command_id = claim_command_id(store)
    store.set(f"chain_command_{command_id}", command.encode("ascii"))
    return command_id


def announce_check_outcome(store, command_id: int, *, round_id: int, L: int,
                           true_k=None, recovery=None, eos=None,
                           redraft=False, prepost=False, attn3=False) -> None:
    if redraft:
        command = f"outcome:{round_id}:{L}:{true_k}:{recovery}:{eos}"
    elif prepost:
        command = f"outcome:{round_id}:{L}:0:{eos}"
    else:
        command = "outcome"
    store.set(f"chain_command_{command_id}", command.encode("ascii"))


class PairMailbox:
    """Pinned-memory layout; the sequence word is published after the packet DMA."""

    def __init__(self, segment: SharedMemory, *, owner: bool = False):
        if segment.size < 3 * _SLOT_BYTES:
            raise ValueError("pair mailbox segment is too short")
        self.segment = segment
        self.owner = owner
        self.buffer = segment.buf
        if owner:
            for slot in range(3):
                _WORD.pack_into(self.buffer, self.release_offset(slot), -1)

    @staticmethod
    def create() -> PairMailbox:
        return PairMailbox(SharedMemory(create=True, size=3 * _SLOT_BYTES), owner=True)

    @staticmethod
    def attach(name: str) -> PairMailbox:
        segment = (_AttachedSegment(name) if sys.platform == "linux"
                   else SharedMemory(name=name))
        return PairMailbox(segment)

    @staticmethod
    def packet_offset(slot: int) -> int:
        return slot * _SLOT_BYTES

    @staticmethod
    def sequence_offset(slot: int) -> int:
        return slot * _SLOT_BYTES + _PACKET.size

    @staticmethod
    def release_offset(slot: int) -> int:
        return slot * _SLOT_BYTES + _PACKET.size + _WORD.size

    @staticmethod
    def stamp_offset(slot: int) -> int:
        return slot * _SLOT_BYTES + _PACKET.size + 2 * _WORD.size

    def write_stamp(self, slot: int, nanoseconds: int) -> None:
        _WORD.pack_into(self.buffer, self.stamp_offset(slot), nanoseconds)

    def read_stamp(self, slot: int) -> int:
        return _WORD.unpack_from(self.buffer, self.stamp_offset(slot))[0]

    def publish_cpu(self, command_id: int, words: tuple[int, ...]) -> None:
        """CPU fixture for the same packet-before-sequence order as the stream copy."""
        slot = command_id % 3
        _PACKET.pack_into(self.buffer, self.packet_offset(slot), *words)
        _WORD.pack_into(self.buffer, self.sequence_offset(slot), command_id + 1)

    def read(self, expected_id: int, *, timeout_s: float = 0.5) -> tuple[int, ...]:
        slot = expected_id % 3
        offset = self.sequence_offset(slot)
        deadline = time.monotonic() + timeout_s
        while True:
            first = _WORD.unpack_from(self.buffer, offset)[0]
            if first > expected_id + 1:
                raise AssertionError(f"pair command id skipped: expected {expected_id}, got {first - 1}")
            if first == expected_id + 1:
                # The second sequence read is the acquire side of the mailbox
                # protocol on Grace: reject a concurrent slot reuse.
                words = _PACKET.unpack_from(self.buffer, self.packet_offset(slot))
                second = _WORD.unpack_from(self.buffer, offset)[0]
                if second == first:
                    return words
            if time.monotonic() >= deadline:
                raise TimeoutError(f"pair command {expected_id} did not arrive")

    def release(self, command_id: int) -> None:
        _WORD.pack_into(self.buffer, self.release_offset(command_id % 3), command_id)

    def read_release(self, slot: int) -> int:
        offset = self.release_offset(slot)
        while True:
            first = _WORD.unpack_from(self.buffer, offset)[0]
            second = _WORD.unpack_from(self.buffer, offset)[0]
            if first == second:
                return first

    def close(self) -> None:
        self.buffer.release()
        self.segment.close()
        if self.owner:
            self.segment.unlink()


class PairOutcomeTransport:
    """Verify-stream peer copy plus a pinned, host-polled command mailbox."""

    def __init__(self, *, store, group, device, lanes: int, hidden: int,
                 tap_dtype: torch.dtype, envelope_bytes: int | None = None):
        import torch.distributed as dist
        from torch.distributed import _symmetric_memory as symm

        self.rank = dist.get_rank(group)
        if dist.get_world_size(group) != 2 or self.rank not in (0, 1):
            raise ValueError("pair outcome transport requires exactly two ranks")
        self.device = torch.device(device)
        from jetspec.engines.jetspec_plus_plus.wire.contract import OutcomeLayout
        dense_bytes = OutcomeLayout(lanes, hidden, tap_dtype).outcome_bytes
        if envelope_bytes is not None and envelope_bytes < dense_bytes:
            raise ValueError("pair envelope cannot truncate the dense packet and taps")
        self.envelope_bytes = dense_bytes if envelope_bytes is None else envelope_bytes
        if self.rank == 0:
            self.mailbox = PairMailbox.create()
            store.set("pair_outcome_mailbox_name", self.mailbox.segment.name.encode())
        else:
            self.mailbox = PairMailbox.attach(
                store.get("pair_outcome_mailbox_name").decode())
        self._host_address = ctypes.addressof(ctypes.c_char.from_buffer(self.mailbox.buffer))
        error = torch.cuda.cudart().cudaHostRegister(
            self._host_address, self.mailbox.segment.size, 0)
        if int(error) != 0:
            self.mailbox.close()
            raise RuntimeError(f"cudaHostRegister pair mailbox failed: {error}")
        self._registered = True

        local = symm.empty(3 * self.envelope_bytes, dtype=torch.uint8, device=self.device)
        self.mapping = symm.rendezvous(local, group)
        self.local = local
        self.peer = self.mapping.get_buffer(
            1 - self.rank, (3 * self.envelope_bytes,), torch.uint8)
        self._sequence_base = 0
        self._sequence_values = (torch.arange(1, 65537, dtype=torch.int64,
                                              device=self.device)
                                 if self.rank == 0 else None)
        self._cudart = None
        self.release_waits = 0
        self.ahead_max = 0
        self._slot_commands = [-1, -1, -1]
        self._closed = False
        atexit.register(self.close)

        if self.rank == 0:
            import subprocess
            driver = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                text=True).splitlines()[0]
            print(f"PAIR_OUTCOME_TRANSPORT driver={driver} "
                  f"peer={torch.cuda.can_device_access_peer(self.device.index, 1)} "
                  f"mechanism=symmetric_memory_CUDA "
                  f"mailbox=cudaHostRegister", flush=True)

    def _memcpy_async(self, destination: int, source: int, size: int,
                      kind: int, stream: int) -> None:
        if self._cudart is None:
            library = Path(torch.__file__).parents[1] / "nvidia/cuda_runtime/lib/libcudart.so.12"
            self._cudart = ctypes.CDLL(str(library))
            self._cudart.cudaMemcpyAsync.argtypes = (
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
                ctypes.c_int, ctypes.c_void_p)
            self._cudart.cudaMemcpyAsync.restype = ctypes.c_int
        error = self._cudart.cudaMemcpyAsync(
            destination, source, size, kind, stream)
        if error:
            raise RuntimeError(f"cudaMemcpyAsync pair mailbox failed: {error}")

    def push(self, command_id: int, message: torch.Tensor, *, timeout_s: float = 0.5,
             measure_stamp: bool = False) -> int:
        if self.rank != 0 or message.dtype != torch.uint8 or message.numel() != self.envelope_bytes:
            raise ValueError("verify push needs one fixed outcome envelope")
        slot = command_id % 3
        previous_id = self._slot_commands[slot]
        if previous_id >= 0:
            deadline = time.monotonic() + timeout_s
            waited = False
            while self.mailbox.read_release(slot) < previous_id:
                waited = True
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"outcome slot {slot} unreleased at command {command_id}")
                time.sleep(0)
            self.release_waits += int(waited)
        self._slot_commands[slot] = command_id
        outstanding = sum(
            old_id >= 0 and self.mailbox.read_release(index) < old_id
            for index, old_id in enumerate(self._slot_commands))
        self.ahead_max = max(self.ahead_max, outstanding - 1)
        if command_id >= self._sequence_base + self._sequence_values.numel():
            # This happens at most once per 65536 commands. Finish old copies
            # before replacing the table that supplies the sequence word.
            torch.cuda.current_stream(self.device).synchronize()
            self._sequence_base = command_id
            self._sequence_values = torch.arange(
                command_id + 1, command_id + 65537,
                dtype=torch.int64, device=self.device)
        stream = torch.cuda.current_stream(self.device).cuda_stream
        self._memcpy_async(
            self.peer.data_ptr() + slot * self.envelope_bytes,
            message.data_ptr(), self.envelope_bytes, 3, stream)  # cudaMemcpyDeviceToDevice
        self._memcpy_async(
            self._host_address + self.mailbox.packet_offset(slot), message.data_ptr(),
            _PACKET.size, 2, stream)  # cudaMemcpyDeviceToHost
        if measure_stamp:
            # The smoke clocks final publication enqueue -> peer host poll.
            self.mailbox.write_stamp(slot, time.perf_counter_ns())
        self._memcpy_async(
            self._host_address + self.mailbox.sequence_offset(slot),
            self._sequence_values.data_ptr() + 8 * (command_id - self._sequence_base),
            _WORD.size, 2, stream)
        return slot

    def poll(self, expected_id: int, *, timeout_s: float = 0.5) -> tuple[int, ...]:
        if self.rank != 1:
            raise ValueError("only the draft polls pushed outcomes")
        return self.mailbox.read(expected_id, timeout_s=timeout_s)

    def slot(self, command_id: int) -> torch.Tensor:
        index = command_id % 3
        return self.local[index * self.envelope_bytes:(index + 1) * self.envelope_bytes]

    def release(self, command_id: int, copy_done: torch.cuda.Event) -> None:
        if self.rank != 1:
            raise ValueError("only the draft releases outcome slots")
        copy_done.synchronize()
        self.mailbox.release(command_id)

    def close(self) -> None:
        if self._closed:
            return
        torch.cuda.synchronize(self.device)
        if self.rank == 0:
            print(f"PAIR_OUTCOME_PUSH_STATS release_waits={self.release_waits} "
                  f"ahead_max={self.ahead_max}", flush=True)
        if self._registered:
            error = torch.cuda.cudart().cudaHostUnregister(self._host_address)
            if int(error) != 0:
                raise RuntimeError(f"cudaHostUnregister pair mailbox failed: {error}")
            self._registered = False
        self.mailbox.close()
        self._closed = True
