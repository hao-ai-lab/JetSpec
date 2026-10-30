"""Stateful target-tail transfer and compatibility views for the pair wire."""

import os
import hashlib
from dataclasses import dataclass
from dataclasses import replace

import torch

from jetspec.engines.jetspec_plus_plus.wire.contract import OutcomeLayout, StateSection
from jetspec.engines.jetspec_plus_plus.models.state import has_state, pair_enabled, pair_layout


def target_wire_spec(runner):
    """Attach the serving adapter's taps to the runner's checkpoint spec."""
    spec = runner.model_spec
    if spec.taps:
        return spec
    taps = runner.config.jetspec_target_layer_ids
    if taps is None:
        taps = runner.model.model.jetspec_tap_layer_ids
    taps = tuple(taps)
    return replace(spec, taps=taps, tail=tuple(range(max(taps) + 1, spec.num_layers)))


def state_digest(*tensors):
    """Diagnostic SHA256 of the complete state bytes on the calling GPU."""
    digest = hashlib.sha256()
    for tensor in tensors:
        digest.update(tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return digest.hexdigest()


def eager_pair(runner):
    """The hybrid pair protocol may execute eagerly without enabling graph knobs."""
    from jetspec.engines.jetspec_plus_plus.serving.settings import setting
    # An N=1 protocol: a C>1 serve exchanges its hybrid state through the batch envelope instead.
    return (getattr(runner, 'enforce_eager', False)
            and has_state(runner)
            and pair_enabled(runner)
            and getattr(getattr(runner, 'config', None), 'max_num_seqs', 1) == 1
            and setting(runner, 'engine.engine_settings.chain_graph', '0') == '0')


# The frozen reference verifier imports this spelling.
eager_hybrid_pair = eager_pair


def seed_eager_hybrid_state(sequence, *, capacity):
    """Admission sends full K/V once; retain a private sentinel for later deltas."""
    for name in ('key', 'value'):
        source = getattr(sequence, 'hybrid_prefix_' + name)
        if source.shape[0] >= capacity:
            raise ValueError('eager hybrid admission exceeds its prefix capacity')
        storage = source.new_zeros((capacity, *source.shape[1:]))
        storage[:source.shape[0]].copy_(source)
        setattr(sequence, 'hybrid_prefix_' + name, storage)


def receive_eager_hybrid_outcome(owner, *, receive):
    """Consume the existing fixed envelope, appending only its accepted delta."""
    from jetspec.engines.jetspec_plus_plus.wire.contract import OutcomeOffset
    from jetspec.engines.jetspec_plus_plus.wire.nccl_pack import append_jetspec_tap_mirrors
    wire = getattr(owner, '_eager_hybrid_wire', None)
    if wire is None:
        wire = owner._eager_hybrid_wire = HybridOutcomeWire(pair_layout(owner), owner.device)
        owner._eager_hybrid_words = torch.empty_like(wire.packet, device='cpu',
                                                     pin_memory=wire.tensor.is_cuda)
    receive(wire.tensor)
    words = owner._eager_hybrid_words
    words.copy_(wire.packet, non_blocking=wire.tensor.is_cuda)
    # This replaces the old request receive's one completion wait. All further
    # metadata reads are CPU-only; K/V append uses the device packet directly.
    if wire.tensor.is_cuda:
        event = torch.cuda.Event(enable_timing=False)
        event.record()
        event.synchronize()
    seq_id = int(words[OutcomeOffset.SEQ_ID])
    count = int(words[OutcomeOffset.TAP_COUNT])
    length = int(words[OutcomeOffset.L])
    if not 1 <= count <= pair_layout(owner).lanes:
        raise ValueError('eager hybrid outcome has an invalid accepted count')
    state = owner.jetspec_sequences[seq_id]
    if length >= state.hybrid_prefix_key.shape[0]:
        raise ValueError('eager hybrid outcome exceeds its prefix capacity')
    state.hybrid_recurrent.copy_(wire.recurrent)
    state.hybrid_conv.copy_(wire.conv)
    append_hybrid_kv(state.hybrid_prefix_key, state.hybrid_prefix_value,
                     wire.keys, wire.values, wire.packet[OutcomeOffset.L],
                     wire.packet[OutcomeOffset.TAP_COUNT])
    control = torch.tensor([[seq_id, int(words[OutcomeOffset.TRUE_K]),
        int(words[OutcomeOffset.RECOVERY]), int(words[OutcomeOffset.ROUND]), count]], dtype=torch.long)
    taps = wire.taps[:count].to(dtype=owner.hf_config.torch_dtype)
    append_jetspec_tap_mirrors(owner.jetspec_tap_mirrors, control[:, 0], taps, control[:, 4])
    if owner.jetspec_tap_mirrors[seq_id].shape[0] != length:
        raise ValueError('eager hybrid delta does not extend the committed prefix')
    owner._jetspec_tap_payload_bytes.append(wire.tensor.numel())
    return dict(cache_keys=control[:, :3], seq_ids=control[:, 0], rounds=control[:, 3],
                tap_lengths=control[:, 4], accepted_tree_ids=None, tap_payload=taps)


@dataclass(frozen=True)
class HybridPairLayout:
    lanes: int
    hidden: int
    key_heads: int
    value_heads: int
    key_dim: int
    value_dim: int
    conv_kernel: int
    kv_heads: int
    attention_dim: int
    dtype: torch.dtype

    @classmethod
    def from_config(cls, config, lanes=16):
        from jetspec.engines.jetspec_plus_plus.draft.method.model_spec import ModelSpec
        config = getattr(config, "text_config", config)
        raw = config.to_dict() if hasattr(config, 'to_dict') else vars(config)
        spec = ModelSpec.from_configs(raw)
        if not spec.has_state:
            raise ValueError("state transfer requires stateful layer geometry")
        gdn = spec.gdn
        return cls(lanes, spec.hidden_size, gdn.k_heads, gdn.v_heads,
                   gdn.k_dim, gdn.v_dim, gdn.conv_kernel,
                   spec.attention.kv_heads, spec.attention.head_dim, config.torch_dtype)

    @property
    def layout(self):
        # Positional section IDs preserve the old geometry-only constructor.
        return OutcomeLayout(self.lanes, self.hidden, self.dtype, sections=(
            StateSection(0, 'linear_attention', ((self.recurrent_shape, torch.float32),
                                                (self.conv_shape, self.dtype))),
            StateSection(1, 'full_attention', ((self.kv_shape, self.dtype),
                                              (self.kv_shape, self.dtype)))))

    @property
    def recurrent_shape(self):
        return (1, self.value_heads, self.key_dim, self.value_dim)

    @property
    def conv_shape(self):
        width = 2 * self.key_heads * self.key_dim + self.value_heads * self.value_dim
        return (1, width, self.conv_kernel)

    @property
    def kv_shape(self):
        return (self.lanes, self.kv_heads, self.attention_dim)

    @property
    def base_bytes(self):
        return self.layout.base_bytes

    @property
    def state_offset(self):
        return (self.base_bytes + 3) & ~3

    @property
    def outcome_bytes(self):
        return self.layout.outcome_bytes


def hybrid_outcome_views(message, layout):
    """Return the legacy packet, taps and shipped tail tensor views."""
    if message.dtype != torch.uint8 or message.numel() != layout.outcome_bytes:
        raise ValueError("hybrid outcome envelope has the wrong byte length")
    row = layout.layout.views(message)[0]
    return row.packet, row.taps, *row.states[0], *row.states[1]


class HybridOutcomeWire:
    def __init__(self, layout, device):
        self.tensor = torch.empty(layout.outcome_bytes, dtype=torch.uint8, device=device)
        (self.packet, self.taps, self.recurrent, self.conv,
         self.keys, self.values) = hybrid_outcome_views(self.tensor, layout)

    def pack(self, packet, taps, recurrent, conv, keys, values):
        self.packet.copy_(packet)
        self.taps.zero_()
        self.taps[:taps.shape[0]].copy_(taps)
        self.recurrent.copy_(recurrent)
        self.conv.copy_(conv)
        self.keys.copy_(keys)
        self.values.copy_(values)
        return self.tensor

    def prepare_taps(self, taps):
        """Only taps are invariant before the reference host acceptance."""
        if taps.shape != self.taps.shape:
            raise ValueError("early outcome preparation requires every tap lane")
        self.taps.copy_(taps)

    def pack_prepared_header(self, packet, recurrent, conv, keys, values):
        self.packet.copy_(packet)
        self.recurrent.copy_(recurrent)
        self.conv.copy_(conv)
        self.keys.copy_(keys)
        self.values.copy_(values)
        return self.tensor


class HybridOutcomeWireRing:
    """Two fixed hybrid envelopes, released after their asynchronous sends."""

    def __init__(self, layout, device):
        self.wires = [HybridOutcomeWire(layout, device) for _ in range(2)]
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
            raise RuntimeError("hybrid outcome send has no acquired slot")
        self.works[self.pending_slot] = list(works)
        self.pending_slot = None


def append_hybrid_kv(prefix_key, prefix_value, keys, values, length, count):
    """Append accepted keys only; unused lanes overwrite a reserved last slot."""
    if prefix_key.shape != prefix_value.shape or keys.shape != values.shape:
        raise ValueError("target-tail K/V shapes differ")
    if prefix_key.shape[1:] != keys.shape[1:] or prefix_key.shape[0] < keys.shape[0] + 1:
        raise ValueError("target-tail K/V capacity or geometry changed")
    lane = torch.arange(keys.shape[0], device=keys.device)
    slots = torch.where(lane < count, length - count + lane,
                        prefix_key.shape[0] - 1)
    prefix_key.index_copy_(0, slots, keys)
    prefix_value.index_copy_(0, slots, values)


def gather_paged_kv(key_cache, value_cache, block_table, positions):
    """Read the target's tail K/V by logical token position."""
    if key_cache.shape != value_cache.shape or key_cache.ndim != 4:
        raise ValueError("target-tail paged K/V geometry changed")
    block_size = key_cache.shape[1]
    page = block_table.index_select(0, positions.div(block_size,
                                                    rounding_mode="floor").long())
    offset = positions.remainder(block_size)
    return key_cache[page, offset], value_cache[page, offset]


def target_tail(runner):
    spec = target_wire_spec(runner)
    layers = runner.model.model.layers
    kinds = tuple(spec.layer_kinds[index] for index in spec.tail)
    if kinds != ('linear_attention', 'full_attention'):
        raise ValueError('pair compatibility views require a GDN / attention tail')
    gdn_index, attention_index = spec.tail
    return layers[gdn_index].linear_attn, layers[attention_index].self_attn.attn


def target_prefix_kv(runner, sequence, positions):
    _, attention = target_tail(runner)
    from jetspec.engines.jetspec_plus_plus.draft.rows import device_tensor
    pages = device_tensor(sequence.block_table, runner.device, dtype=torch.long)
    return gather_paged_kv(attention.k_cache, attention.v_cache, pages, positions)


def target_committed_state(runner, sequence, *, layer_index=None):
    """Select the sequence's live F3 slot while retaining the C=1 buffers."""
    gdn = (target_tail(runner)[0] if layer_index is None else
           runner.model.model.layers[layer_index].linear_attn)
    if gdn.committed_recurrent.shape[0] == 1:
        return gdn.committed_recurrent, gdn.committed_conv
    state = runner.state
    slot = state.owners.get(sequence.seq_id)
    if slot is None:
        raise ValueError('hybrid state owner is not live')
    if slot.shape != (1,) or slot.dtype != torch.long:
        raise ValueError('hybrid state owner must name one slot')
    index = int(slot.item())
    if (not 0 <= index < state.capacity or index in state.slots.available
            or index >= gdn.committed_recurrent.shape[0]):
        raise ValueError('hybrid state slot is not live')
    if any(other_id != sequence.seq_id and torch.equal(other, slot)
           for other_id, other in state.owners.items()):
        raise ValueError('hybrid state slot has a wrong owner')
    return (gdn.committed_recurrent[index:index + 1],
            gdn.committed_conv[index:index + 1])


def pack_batch_hybrid_outcomes(runner, sequences, lengths, layout):
    """Append one F6 envelope containing each request row's committed state."""
    from jetspec.engines.jetspec_plus_plus.wire.contract import OutcomePacket, hash_last16_host
    from jetspec.engines.jetspec_plus_plus.wire.outcome_packet import OutcomeWire
    if not sequences or len(sequences) != len(lengths):
        raise ValueError('hybrid batch sequence count does not match tap counts')
    layout = replace(layout, sequence_count=len(sequences))
    wire = OutcomeWire(layout.lanes, layout.hidden, layout.tap_dtype, runner.device,
                       layout=layout)
    packets, taps, states = [], [], []
    for sequence, count, row in zip(sequences, lengths, wire.sequences):
        length = len(sequence.jetspec_taps)
        if not 0 <= count <= min(layout.lanes, length):
            raise ValueError('hybrid batch outcome has an invalid accepted count')
        packet = OutcomePacket(cmd=1, seq_id=sequence.seq_id,
            true_k=max(0, sequence.last_spec_step_accepted_len - 1),
            recovery=sequence.recovery_token_id, round=sequence.jetspec_round,
            tap_count=count, eos_flag=0, L=length,
            hash16=hash_last16_host(sequence.token_ids[-16:], len(sequence.token_ids))).pack(
                torch.empty_like(row.packet))
        state = {}
        for section in layout.sections:
            if section.kind == 'linear_attention':
                state[section.layer_index] = target_committed_state(
                    runner, sequence, layer_index=section.layer_index)
            elif count:
                lane = torch.arange(layout.lanes, device=runner.device)
                positions = torch.where(lane < count, length - count + lane, length - count)
                attention = runner.model.model.layers[section.layer_index].self_attn.attn
                from jetspec.engines.jetspec_plus_plus.draft.rows import device_tensor
                pages = device_tensor(sequence.block_table, runner.device, dtype=torch.long)
                state[section.layer_index] = gather_paged_kv(
                    attention.k_cache, attention.v_cache, pages, positions)
            else:
                state[section.layer_index] = tuple(torch.zeros_like(tensor)
                    for tensor in row.states[section.layer_index])
        packets.append(packet)
        taps.append(row.taps[:0])
        states.append(state)
    # OutcomeWire's C=1 ABI takes a single row rather than a row list.
    message = (wire.pack(packets[0], taps[0], states=states[0]) if len(sequences) == 1
               else wire.pack(packets, taps, states=states))
    from jetspec.engines.jetspec_plus_plus.serving.settings import setting
    if setting(runner, 'engine.engine_settings.hybrid_state_audit', '0') == '1':
        for seq_id, round_id, digest in wire.state_digests():
            print(f'HYBRID_STATE_TARGET seq={seq_id} round={round_id} sha256={digest}', flush=True)
    return message


def receive_batch_hybrid_outcomes(owner, payload, *, receive):
    """Validate the entire F6 extension before updating any request owner."""
    from jetspec.engines.jetspec_plus_plus.wire.contract import OutcomeOffset as O
    from jetspec.engines.jetspec_plus_plus.wire.outcome_packet import OutcomeWire, outcome_layout
    layout = replace(outcome_layout(owner), sequence_count=len(payload['seq_ids']))
    wire = OutcomeWire(layout.lanes, layout.hidden, layout.tap_dtype, owner.device, layout=layout)
    receive(wire.tensor)
    words = torch.stack([row.packet for row in wire.sequences]).cpu().tolist()
    seq_ids = payload['seq_ids'].tolist()
    if len(set(seq_ids)) != len(seq_ids):
        raise ValueError('hybrid batch outcome has duplicate owners')
    for index, packet in enumerate(words):
        seq_id, count, length = packet[O.SEQ_ID], packet[O.TAP_COUNT], packet[O.L]
        if seq_id != seq_ids[index] or seq_id not in owner.jetspec_sequences:
            raise ValueError('hybrid batch outcome has a wrong owner')
        if count != int(payload['tap_lengths'][index]) or not 0 <= count <= layout.lanes:
            raise ValueError('hybrid batch outcome has an invalid accepted count')
        if (packet[O.ROUND] != int(payload['rounds'][index])
                or packet[O.TRUE_K:O.RECOVERY + 1] != payload['cache_keys'][index, 1:].tolist()):
            raise ValueError('hybrid batch outcome does not match the request round/key')
        state = owner.jetspec_sequences[seq_id]
        if not count <= length < state.hybrid_prefix_key.shape[0]:
            raise ValueError('hybrid batch outcome has an invalid prefix capacity')
        mirrors = getattr(owner, 'jetspec_tap_mirrors', None)
        if mirrors is not None and len(mirrors[seq_id]) != length:
            raise ValueError('hybrid batch delta does not extend the committed prefix')
    from jetspec.engines.jetspec_plus_plus.serving.settings import setting
    for row, packet in zip(wire.sequences, words):
        state = owner.jetspec_sequences[packet[O.SEQ_ID]]
        # The draft pair exposes one recurrent tail and one attention prefix.
        recurrent = [section for section in layout.sections if section.kind == 'linear_attention']
        attention = [section for section in layout.sections if section.kind == 'full_attention']
        if len(recurrent) != 1 or len(attention) != 1:
            raise ValueError('hybrid batch draft requires a recurrent/attention tail pair')
        state.hybrid_recurrent.copy_(row.states[recurrent[0].layer_index][0])
        state.hybrid_conv.copy_(row.states[recurrent[0].layer_index][1])
        keys, values = row.states[attention[0].layer_index]
        dtype = state.hybrid_prefix_key.dtype
        append_hybrid_kv(state.hybrid_prefix_key, state.hybrid_prefix_value, keys.to(dtype),
                         values.to(dtype), row.packet[O.L], row.packet[O.TAP_COUNT])
        if setting(owner, 'engine.engine_settings.hybrid_state_audit', '0') == '1':
            print(f'HYBRID_STATE_DRAFT seq={packet[O.SEQ_ID]} round={packet[O.ROUND]} '
                  f'sha256={state_digest(state.hybrid_recurrent, state.hybrid_conv)}', flush=True)


def pack_target_hybrid_outcome(runner, sequence, wire, packet, taps, *, prepared_taps=False):
    """Enqueue accepted state and newly accepted K/V, with no device-to-host read."""
    from jetspec.engines.jetspec_plus_plus.serving.settings import setting
    recurrent, conv = target_committed_state(runner, sequence)
    count = packet[5]
    start = packet[8] - count
    lane = torch.arange(wire.keys.shape[0], device=runner.device)
    positions = torch.where(lane < count, start + lane, start)
    keys, values = target_prefix_kv(runner, sequence, positions)
    if setting(runner, 'engine.engine_settings.hybrid_state_audit', '0') == "1":
        print("HYBRID_STATE_TARGET " +
              f"round={int(packet[4].item())} sha256={state_digest(recurrent, conv)}",
              flush=True)
    if prepared_taps:
        return wire.pack_prepared_header(packet, recurrent, conv, keys, values)
    return wire.pack(packet, taps, recurrent, conv, keys, values)


def target_prefill_tensors(runner, sequence, positions, layout):
    """Ship the committed state and complete prefix in the layout's layer order."""
    from jetspec.engines.jetspec_plus_plus.draft.rows import device_tensor
    layers = runner.model.model.layers
    result = []
    for section in layout.sections:
        layer = layers[section.layer_index]
        if section.kind == 'linear_attention':
            result.extend(target_committed_state(runner, sequence, layer_index=section.layer_index))
        else:
            attention = layer.self_attn.attn
            pages = device_tensor(sequence.block_table, runner.device, dtype=torch.long)
            result.extend(gather_paged_kv(attention.k_cache, attention.v_cache, pages, positions))
    return result


def pack_target_outcome(runner, sequence, wire, packet, taps, *, skip_full_zero=False,
                        prepared_taps=False, admission=False):
    """Pack the dense core and each shipped tail layer, without host reads."""
    from jetspec.engines.jetspec_plus_plus.serving.settings import setting
    if isinstance(wire, HybridOutcomeWire):
        if not admission:
            return pack_target_hybrid_outcome(runner, sequence, wire, packet, taps,
                                             prepared_taps=prepared_taps)
        recurrent, conv = target_committed_state(runner, sequence)
        wire.keys.zero_()
        wire.values.zero_()
        return wire.pack(packet, taps, recurrent, conv, wire.keys, wire.values)
    if (not admission and wire.layout.sections
            and setting(runner, 'engine.engine_settings.verify_cut_pack27b', '0') == '1'):
        message = pack_target_outcome_cut(runner, sequence, wire, packet, taps,
                                          prepared_taps=prepared_taps)
        if message is not None:
            return message
    states = {}
    audits = []
    for section in wire.layout.sections:
        layer = runner.model.model.layers[section.layer_index]
        if section.kind == 'linear_attention':
            states[section.layer_index] = target_committed_state(
                runner, sequence, layer_index=section.layer_index)
            if not admission and setting(runner, 'engine.engine_settings.hybrid_state_audit', '0') == '1':
                audits.append(states[section.layer_index])
        else:
            if admission:
                keys, values = wire.sequences[0].states[section.layer_index]
                keys.zero_()
                values.zero_()
            else:
                count = packet[5]
                start = packet[8] - count
                lane = torch.arange(wire.layout.lanes, device=runner.device)
                positions = torch.where(lane < count, start + lane, start)
                attention = layer.self_attn.attn
                from jetspec.engines.jetspec_plus_plus.draft.rows import device_tensor
                pages = device_tensor(sequence.block_table, runner.device, dtype=torch.long)
                keys, values = gather_paged_kv(attention.k_cache, attention.v_cache, pages, positions)
            states[section.layer_index] = (keys, values)
    for state in audits:
        print('HYBRID_STATE_TARGET ' +
              f'round={int(packet[4].item())} sha256={state_digest(*state)}', flush=True)
    if prepared_taps:
        return wire.pack_prepared_header(packet, states=states)
    # Stateful C=1 packets historically zeroed the taps even for a full block.
    # Retain that operation as well as its bytes; dense rings may skip it.
    return wire.pack(packet, taps, states=states,
                     skip_full_zero=skip_full_zero and not wire.layout.sections)


def tail_lane_offsets(wire, device):
    """Lane offsets per accepted count, built once per wire.

    Row ``c`` is ``where(lane < c, lane - c, -c)``, so ``L + row[c]`` are the
    accepted tail positions. The rows cover the counts 0..lanes."""
    table = getattr(wire, '_jetspec_tail_lane_offsets', None)
    if table is None or table.device != device:
        lane = torch.arange(wire.layout.lanes, dtype=torch.long)
        count = torch.arange(wire.layout.lanes + 1, dtype=torch.long)[:, None]
        host = torch.where(lane < count, lane - count, -count)
        if device.type == 'cuda':
            host = host.pin_memory()
        table = host.to(device, non_blocking=device.type == 'cuda')
        wire._jetspec_tail_lane_offsets = table
    return table


def tail_slot_table(sequence, block_size, device):
    """Per-token flat cache slots, page * block_size + offset, kept on the device.

    The table persists on the sequence. A round uploads only the pages whose
    IDs changed since the last call, so a steady round issues no host copy."""
    pages = tuple(sequence.block_table)
    state = getattr(sequence, '_jetspec_tail_slots', None)
    previous, storage = (), None
    if state is not None and state[0] == (block_size, device):
        _, previous, storage = state
    tokens = len(pages) * block_size
    changed = next((index for index, (old, new) in enumerate(zip(previous, pages))
                    if old != new), min(len(previous), len(pages)))
    if storage is None or storage.numel() < tokens:
        capacity = max(tokens, 2 * (storage.numel() if storage is not None else 0), block_size)
        storage = torch.empty(capacity, dtype=torch.long, device=device)
        changed = 0
    if changed < len(pages):
        ids = torch.tensor(pages[changed:], dtype=torch.long)
        host = (ids[:, None] * block_size + torch.arange(block_size)).reshape(-1)
        if device.type == 'cuda':
            host = host.pin_memory()
        storage[changed * block_size:tokens].copy_(host, non_blocking=device.type == 'cuda')
    sequence._jetspec_tail_slots = ((block_size, device), pages, storage)
    return storage[:tokens]


def pack_target_outcome_cut(runner, sequence, wire, packet, taps, *, prepared_taps=False):
    """The stateful round pack with fewer launches and the same packet bytes.

    The knob is engine.engine_settings.verify_cut_pack27b. The accepted tail
    K/V rows are gathered in one ``index_select`` per cache, straight into the
    wire, through a persistent device slot table. The old pack instead uploads
    the whole block table each round and gathers with ten glue kernels plus two
    copies. A full tap block also skips the redundant zero.
    Returns ``None`` before any device work when the geometry does not allow
    it; the caller then runs the old pack."""
    from jetspec.engines.jetspec_plus_plus.serving.settings import setting
    if wire.layout.sequence_count != 1:
        return None
    row = wire.sequences[0]
    gathers = []
    for section in wire.layout.sections:
        if section.kind == 'linear_attention':
            continue
        attention = runner.model.model.layers[section.layer_index].self_attn.attn
        key_cache, value_cache = attention.k_cache, attention.v_cache
        destinations = row.states[section.layer_index]
        if (key_cache.shape != value_cache.shape or key_cache.ndim != 4
                or key_cache.dtype != value_cache.dtype or key_cache.device != packet.device
                or value_cache.device != packet.device or len(destinations) != 2
                or any(cache.stride(0) != cache.shape[1] * cache.stride(1)
                       for cache in (key_cache, value_cache))
                or any(target.shape != (wire.layout.lanes, *key_cache.shape[2:])
                       or target.dtype != key_cache.dtype for target in destinations)):
            return None
        gathers.append((section.layer_index, key_cache, value_cache, destinations))
    states = {}
    audits = []
    offsets = None
    for section in wire.layout.sections:
        if section.kind == 'linear_attention':
            states[section.layer_index] = target_committed_state(
                runner, sequence, layer_index=section.layer_index)
            if setting(runner, 'engine.engine_settings.hybrid_state_audit', '0') == '1':
                audits.append(states[section.layer_index])
    for layer, key_cache, value_cache, (key_out, value_out) in gathers:
        if offsets is None:
            # packet[5] is the accepted count c, packet[8] the committed length L.
            offsets = torch.add(tail_lane_offsets(wire, packet.device).index_select(
                0, packet[5:6]).view(-1), packet[8:9])
        slots = tail_slot_table(sequence, key_cache.shape[1], packet.device).index_select(0, offsets)
        torch.index_select(key_cache.flatten(0, 1), 0, slots, out=key_out)
        torch.index_select(value_cache.flatten(0, 1), 0, slots, out=value_out)
        # The wire's own views: the pack's state copies are then no-ops.
        states[layer] = (key_out, value_out)
    for state in audits:
        print('HYBRID_STATE_TARGET ' +
              f'round={int(packet[4].item())} sha256={state_digest(*state)}', flush=True)
    if prepared_taps:
        return wire.pack_prepared_header(packet, states=states)
    return wire.pack(packet, taps, states=states, skip_full_zero=True)
