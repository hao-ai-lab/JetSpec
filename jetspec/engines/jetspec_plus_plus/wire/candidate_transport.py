"""Reference-pair candidate bytes over a same-host Unix socket.

Candidate IDs and immutable payload bytes keep the existing protocol. The optional
reverse channel carries ordered chain commands; other keys keep their store.
"""
from __future__ import annotations

from jetspec.engines.jetspec_plus_plus.profiles import transport_enabled

import atexit
import hashlib
import json
import os
import select
import socket
import struct
import sys
import uuid


_PREFIX = 'candidate_payload_'
_COMMAND_PREFIX = 'chain_command_'
_HEADER = struct.Struct('<QQ')
_MAX_BYTES = 65536


class CandidateSocketStore:
    def __init__(self, store, connection, *, writer, shadow=False, spin=False, main=False,
                 command_socket=None, run_config=None):
        self.run_config = run_config
        self.store = store
        self.connection = connection
        self.writer = writer
        self.shadow = shadow
        self.spin = spin
        self.main = main
        self.command_socket = (transport_enabled('reference_command_socket', run_config=self)
                               if command_socket is None else command_socket)
        self.receive_buffer = bytearray()
        self.poller = None
        if spin:
            self.poller = select.poll()
            self.poller.register(connection, select.POLLIN)

    def __getattr__(self, name):
        return getattr(self.store, name)

    def clone(self):
        return CandidateSocketStore(self.store.clone(), self.connection,
                                    writer=self.writer, shadow=self.shadow, spin=self.spin,
                                    main=self.main, command_socket=self.command_socket, run_config=getattr(self, 'run_config', None))

    def set(self, key, value):
        if self.command_socket and key.startswith(_COMMAND_PREFIX):
            if self.writer:
                raise RuntimeError('draft cannot publish chain commands')
            payload = value.encode() if isinstance(value, str) else bytes(value)
            if not 0 < len(payload) <= _MAX_BYTES:
                raise ValueError('reference command exceeds socket capacity')
            if self.shadow:
                self.store.set(key, payload)
            self.connection.sendall(_HEADER.pack(int(key[len(_COMMAND_PREFIX):]), len(payload)) + payload)
            return
        if not key.startswith(_PREFIX):
            return self.store.set(key, value)
        if not self.writer:
            raise RuntimeError('target cannot publish candidate bytes')
        payload = bytes(value)
        if not 0 < len(payload) <= _MAX_BYTES:
            raise ValueError('reference candidate packet exceeds socket capacity')
        if self.shadow:
            self.store.set(key, payload)
        self.connection.sendall(_HEADER.pack(int(key[len(_PREFIX):]), len(payload)) + payload)

    def _read_exact(self, size):
        payload = bytearray(size)
        view = memoryview(payload)
        offset = 0
        while offset < size:
            got = self.connection.recv_into(view[offset:])
            if got == 0:
                raise EOFError('reference candidate publisher closed')
            offset += got
        return bytes(payload)

    def get(self, key):
        if self.command_socket and key.startswith(_COMMAND_PREFIX):
            if not self.writer:
                raise RuntimeError('target cannot consume chain commands')
            if self.connection.type == socket.SOCK_SEQPACKET:
                packet, _, flags, _ = self.connection.recvmsg(_MAX_BYTES + _HEADER.size)
                if not packet:
                    raise EOFError('reference command publisher closed')
                if flags & socket.MSG_TRUNC or len(packet) < _HEADER.size:
                    raise ValueError('truncated reference command')
                command, size = _HEADER.unpack_from(packet)
                payload = packet[_HEADER.size:]
            else:
                command, size = _HEADER.unpack(self._read_exact(_HEADER.size))
                if not 0 < size <= _MAX_BYTES:
                    raise ValueError('reference command exceeds socket capacity')
                payload = self._read_exact(size)
            if command != int(key[len(_COMMAND_PREFIX):]):
                raise RuntimeError(f'command id mismatch: requested {key}, received {command}')
            if not 0 < size <= _MAX_BYTES or len(payload) != size:
                raise ValueError('reference command size mismatch')
            if self.shadow:
                if payload != self.store.get(key):
                    raise RuntimeError('reference command socket shadow mismatch')
                print('REFERENCE_COMMAND_SHADOW ' + json.dumps(dict(
                    id=command, bytes=size, equal=True)), flush=True)
            return payload
        if not key.startswith(_PREFIX):
            return self.store.get(key)
        if self.main:
            return self._get_main(key, block=True)
        if self.writer:
            raise RuntimeError('draft cannot consume candidate bytes')
        if self.poller is not None:
            # poll releases the GIL; dedicate one CPU reader to avoid a sleeping wakeup.
            while not self.poller.poll(0):
                pass
        packet = None
        if self.connection.type == socket.SOCK_SEQPACKET:
            packet, _, flags, _ = self.connection.recvmsg(_MAX_BYTES + _HEADER.size)
            if not packet:
                raise EOFError('reference candidate publisher closed')
            if flags & socket.MSG_TRUNC or len(packet) < _HEADER.size:
                raise ValueError('truncated reference candidate packet')
            command, size = _HEADER.unpack_from(packet)
        else:
            command, size = _HEADER.unpack(self._read_exact(_HEADER.size))
        if command != int(key[len(_PREFIX):]):
            raise RuntimeError(f'candidate id mismatch: requested {key}, received {command}')
        if not 0 < size <= _MAX_BYTES:
            raise ValueError('reference candidate packet exceeds socket capacity')
        payload = packet[_HEADER.size:] if packet is not None else self._read_exact(size)
        if len(payload) != size:
            raise ValueError('reference candidate packet size mismatch')
        if self.shadow:
            if payload != self.store.get(key):
                raise RuntimeError('reference candidate socket shadow mismatch')
            print('REFERENCE_CANDIDATE_SHADOW ' + json.dumps(dict(
                id=command, bytes=size, sha256=hashlib.sha256(payload).hexdigest(),
                equal=True)), flush=True)
        return payload

    def get_nowait(self, key):
        return self._get_main(key, block=False)

    def _get_main(self, key, *, block):
        if self.writer or not key.startswith(_PREFIX):
            raise ValueError('main candidate receive requires a target payload key')
        while True:
            if self.connection.type != socket.SOCK_SEQPACKET and len(self.receive_buffer) >= _HEADER.size:
                _, size = _HEADER.unpack_from(self.receive_buffer)
                if not 0 < size <= _MAX_BYTES:
                    raise ValueError('reference candidate packet exceeds socket capacity')
                if len(self.receive_buffer) >= _HEADER.size + size:
                    packet = bytes(self.receive_buffer[:_HEADER.size + size])
                    del self.receive_buffer[:_HEADER.size + size]
                    break
            try:
                if self.connection.type == socket.SOCK_SEQPACKET:
                    packet, _, flags, _ = self.connection.recvmsg(
                        _MAX_BYTES + _HEADER.size, 0, socket.MSG_DONTWAIT)
                    if flags & socket.MSG_TRUNC or 0 < len(packet) < _HEADER.size:
                        raise ValueError('truncated reference candidate packet')
                    if packet:
                        break
                else:
                    packet = self.connection.recv(_MAX_BYTES + _HEADER.size, socket.MSG_DONTWAIT)
                    if packet:
                        self.receive_buffer.extend(packet)
                        continue
                raise EOFError('reference candidate publisher closed')
            except BlockingIOError:
                if not block:
                    return None
                # Only intake waits here. A buffered packet never enters this wait.
                select.select([self.connection], [], [])
        command, size = _HEADER.unpack_from(packet)
        if command != int(key[len(_PREFIX):]):
            raise RuntimeError(f'candidate id mismatch: requested {key}, received {command}')
        if not 0 < size <= _MAX_BYTES:
            raise ValueError('reference candidate packet exceeds socket capacity')
        payload = packet[_HEADER.size:]
        if len(payload) != size:
            raise ValueError('reference candidate packet size mismatch')
        if self.shadow:
            if payload != self.store.get(key):
                raise RuntimeError('reference candidate socket shadow mismatch')
            print('REFERENCE_CANDIDATE_SHADOW ' + json.dumps(dict(
                id=command, bytes=size, sha256=hashlib.sha256(payload).hexdigest(),
                equal=True)), flush=True)
        return payload


class MainThreadCandidateMessage:
    """Lazy intake completion; posting performs no I/O and starts no worker."""

    def __init__(self, receiver, key):
        self.receiver = receiver
        self.key = key
        self.ready = self
        self.complete = False
        self.value = None
        self.error = None

    def _receive(self, *, block):
        try:
            payload = (self.receiver.store.get(self.key) if block else
                       self.receiver.store.get_nowait(self.key))
            if payload is None:
                return False
            import torch
            wire = torch.frombuffer(bytearray(payload), dtype=torch.int64)
            self.value = self.receiver.decode(wire)
        except Exception as error:
            self.error = error
        self.complete = True
        return True

    def done(self):
        return self.complete or self._receive(block=False)

    def cpu(self):
        if not self.complete:
            self._receive(block=True)
        if self.error is not None:
            raise self.error
        return self.value


class MainThreadCandidateReceiver:
    def __init__(self, store, decode, *, run_config=None):
        self.store = store.clone()
        self.decode = decode

    def post(self, key):
        return MainThreadCandidateMessage(self, key)


def decode_reference_candidate(wire, *, rows, lanes, run_config=None):
    """Decode the private N=1 keyed wire with one host read and two tensor views."""
    import torch
    from jetspec.engines.jetspec_plus_plus.wire.protocol import CandidateDelivery
    from jetspec.engines.jetspec_plus_plus.wire.protocol import CandidateSet
    if (wire.ndim != 1 or wire.dtype != torch.int64 or wire.device.type != 'cpu'
            or wire.numel() != 6 + rows * (lanes + 2)):
        raise ValueError('reference candidate decode requires its N=1 CPU wire')
    words = wire.tolist()
    timing = 3 + rows * lanes
    coalesced, gap = words[timing + 1:timing + 3]
    if not words[0]:
        return CandidateDelivery([None], [], coalesced, [gap])
    key_start = timing + 3
    keys = list(zip(words[key_start::2], words[key_start + 1::2]))
    candidate = CandidateSet(words[1], words[2],
        wire.as_strided((rows, lanes), (lanes, 1), storage_offset=3),
        row_keys=wire.as_strided((rows, 2), (2, 1), storage_offset=key_start),
        row_keys_host=keys, run_config=run_config)
    index = {}
    for offset, key in enumerate(keys):
        if key[0] >= 0 and words[3 + offset * lanes] >= 0:
            index.setdefault(key, offset)
    object.__setattr__(candidate, '_row_index', index)
    if transport_enabled('reference_host_select', run_config=run_config):
        object.__setattr__(candidate, '_blocks_host',
                           [words[3 + row * lanes:3 + (row + 1) * lanes] for row in range(rows)])
        object.__setattr__(candidate, '_row_keys_host', [list(key) for key in keys])
    draft_ms = struct.unpack('<d', struct.pack('<q', words[timing]))[0]
    return CandidateDelivery([candidate], [draft_ms], coalesced, [gap])


def reference_candidate_store(store, *, is_draft, max_num_seqs, run_config=None):
    """Off by default; establish the one N=1 reference pair's private channel."""
    from jetspec.engines.jetspec_plus_plus.engine import current_engine
    if not transport_enabled('reference_candidate_socket', run_config=run_config):
        return store
    if current_engine().kind != 'reference' or max_num_seqs != 1:
        raise ValueError('candidate socket requires the N=1 reference pair')
    if sys.platform != 'linux':
        raise ValueError('candidate socket requires a same-host Linux pair')
    main = transport_enabled('reference_candidate_main', run_config=run_config)
    if main and transport_enabled('reference_candidate_spin', run_config=run_config):
        raise ValueError('main candidate intake cannot use a polling reader')
    if not is_draft and not main:
        # A socket wakeup still needs the GIL before it can complete the Future.
        sys.setswitchinterval(min(sys.getswitchinterval(), .00005))
    shadow = transport_enabled('reference_candidate_shadow', run_config=run_config)
    if is_draft:
        endpoint = json.loads(store.get('reference_candidate_endpoint'))
        if endpoint['host'] != socket.gethostname():
            raise ValueError('candidate socket requires a same-host pair')
        connection = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        connection.settimeout(60)
        connection.connect('\0' + endpoint['name'])
    else:
        name = 'jetspec-candidate-' + uuid.uuid4().hex
        with socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET) as listener:
            listener.settimeout(60)
            listener.bind('\0' + name)
            listener.listen(1)
            store.set('reference_candidate_endpoint', json.dumps(dict(
                host=socket.gethostname(), name=name)).encode())
            connection, _ = listener.accept()
    connection.settimeout(None)
    atexit.register(connection.close)
    spin = not is_draft and transport_enabled('reference_candidate_spin', run_config=run_config)
    return CandidateSocketStore(store, connection, writer=is_draft, shadow=shadow, spin=spin,
                                main=main and not is_draft, run_config=run_config)
