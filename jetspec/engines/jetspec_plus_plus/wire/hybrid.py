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
    return (getattr(runner, 'enforce_eager', False)
            and has_state(runner)
            and pair_enabled(runner)
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
    from jetspec.engines.jetspec_plus_plus.jetspec.rows import device_tensor
    pages = device_tensor(sequence.block_table, runner.device, dtype=torch.long)
    return gather_paged_kv(attention.k_cache, attention.v_cache, pages, positions)


def pack_target_hybrid_outcome(runner, sequence, wire, packet, taps, *, prepared_taps=False):
    """Enqueue accepted state and newly accepted K/V, with no device-to-host read."""
    from jetspec.engines.jetspec_plus_plus.serving.settings import setting
    gdn, _ = target_tail(runner)
    count = packet[5]
    start = packet[8] - count
    lane = torch.arange(wire.keys.shape[0], device=runner.device)
    positions = torch.where(lane < count, start + lane, start)
    keys, values = target_prefix_kv(runner, sequence, positions)
    if setting(runner, 'engine.engine_settings.hybrid_state_audit', '0') == "1":
        print("HYBRID_STATE_TARGET " +
              f"round={int(packet[4].item())} sha256={state_digest(gdn.committed_recurrent, gdn.committed_conv)}",
              flush=True)
    if prepared_taps:
        return wire.pack_prepared_header(packet, gdn.committed_recurrent,
                                         gdn.committed_conv, keys, values)
    return wire.pack(packet, taps, gdn.committed_recurrent,
                     gdn.committed_conv, keys, values)


def target_prefill_tensors(runner, sequence, positions, layout):
    """Ship the committed state and complete prefix in the layout's layer order."""
    from jetspec.engines.jetspec_plus_plus.jetspec.rows import device_tensor
    layers = runner.model.model.layers
    result = []
    for section in layout.sections:
        layer = layers[section.layer_index]
        if section.kind == 'linear_attention':
            result.extend((layer.linear_attn.committed_recurrent, layer.linear_attn.committed_conv))
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
        gdn, _ = target_tail(runner)
        wire.keys.zero_()
        wire.values.zero_()
        return wire.pack(packet, taps, gdn.committed_recurrent,
                         gdn.committed_conv, wire.keys, wire.values)
    states = {}
    audits = []
    for section in wire.layout.sections:
        layer = runner.model.model.layers[section.layer_index]
        if section.kind == 'linear_attention':
            gdn = layer.linear_attn
            states[section.layer_index] = (gdn.committed_recurrent, gdn.committed_conv)
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
                from jetspec.engines.jetspec_plus_plus.jetspec.rows import device_tensor
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
