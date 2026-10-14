"""Private anchor windows, prefix caches and prediction values."""
from __future__ import annotations
from dataclasses import dataclass
import torch
from jetspec.engines.jetspec_plus_plus.draft.method.ops import reference_op


@reference_op("window_mask")
def window_mask(anchors, context_length, dtype, size=15):
    """Causal private blocks following every verified true prefix row."""
    anchors = torch.as_tensor(anchors).flatten()
    device = anchors.device
    query = torch.arange(len(anchors) * size, device=device)
    key = torch.arange(int(context_length) + len(query), device=device)
    block = query // size
    prefix = key[None, :] < anchors[block, None]
    local = key[None, :] - int(context_length)
    private = ((local >= 0) & (local // size == block[:, None]) &
               (local % size <= query[:, None] % size))
    allowed = prefix | private
    return torch.zeros((1, 1, len(query), len(key)), device=device, dtype=dtype).masked_fill(
        ~allowed[None, None], torch.finfo(dtype).min)


@reference_op("chain_decoder_window_mask")
def chain_decoder_window_mask(length, bucket, rows, dtype, *, out=None):
    """One decoder window's mask with device prefix length and a fixed key bucket."""
    query = torch.arange(rows, device=length.device)[:, None]
    key = torch.arange(bucket, device=length.device)[None]
    allowed = (key < length) | ((key >= length) & (key - length <= query))
    if out is None:
        out = torch.zeros((1, 1, rows, bucket), device=length.device, dtype=dtype)
    else:
        if out.shape != (1, 1, rows, bucket) or out.dtype != dtype:
            raise ValueError('decoder window mask output shape or dtype changed')
        out.zero_()
    return out.masked_fill_(~allowed[None, None], torch.finfo(dtype).min)


@reference_op("stage_chain_window_state")
def stage_chain_window_state(state, front, length, *, glue_level, fused_mask=False):
    """Stage exact decoder-window inputs, using static destinations for glue level 1."""
    rows = state['hidden'].shape[1]
    hidden = front.staged_hidden[:, :, -state['hidden'].shape[-1]:]
    if glue_level:
        state['hidden'].copy_(hidden)
        torch.add(state['lanes'], length, out=state['slots'])
        positions = state['slots'][None]
        if not fused_mask:
            chain_decoder_window_mask(length, front.decoder_bucket, rows,
                                      state['hidden'].dtype,
                                      out=state['masks'][front.decoder_bucket])
    else:
        state['hidden'].copy_(hidden)
        positions = (length + torch.arange(rows, device=length.device))[None]
        state['slots'].copy_(positions[0])
        state['masks'][front.decoder_bucket].copy_(chain_decoder_window_mask(
            length, front.decoder_bucket, rows, state['hidden'].dtype))
    return positions


@reference_op("chain_decoder_prefix_slots")
def chain_decoder_prefix_slots(length, count, bucket, lanes):
    lane = torch.arange(lanes, device=length.device)
    return torch.where(lane < count, length - count + lane, bucket - 1)


@reference_op("chain_decoder_prefix_mask")
def chain_decoder_prefix_mask(length, count, bucket, lanes, dtype):
    lane = torch.arange(lanes, device=length.device)[:, None]
    key = torch.arange(bucket, device=length.device)[None]
    positions = length - count + lane
    allowed = ((lane < count) & (key <= positions) & (key < length)) | \
              ((lane >= count) & (key == 0))
    return torch.zeros((1, 1, lanes, bucket), device=length.device, dtype=dtype).masked_fill_(
        ~allowed[None, None], torch.finfo(dtype).min)


@reference_op("decoder_window")
def decoder_window(current):
    """The private current block [c, d1..d_width], starting at the anchor."""
    current = current if isinstance(current, torch.Tensor) else torch.as_tensor(current)
    if current.ndim != 2 or current.shape[0] not in (4, 8, 15, 16, 32):
        raise ValueError('decoder window requires a supported current block')
    return current[None]


@dataclass(frozen=True)
class AnchorOffers:
    ids: dict[int, torch.Tensor]
    probabilities: dict[int, torch.Tensor]
    set_sizes: list[int]
    rejected_was_top1: list[bool]
    lac_fallbacks: int = 0


class FrozenPrefix:
    def __init__(self, kv):
        self.kv = kv

    def update(self, key, value, layer_idx, cache_kwargs=None):
        prefix_key, prefix_value = self.kv[layer_idx]
        return torch.cat((prefix_key, key), dim=-2), torch.cat((prefix_value, value), dim=-2)


class FrozenBatchPrefix:
    """Reuse the gathered prefix allocation for both private window layers."""
    def __init__(self, kv, capacity):
        self.kv = kv
        self.capacity = capacity

    def update(self, key, value, layer_idx, cache_kwargs=None):
        stored_key, stored_value = self.kv[layer_idx]
        end = self.capacity + key.shape[-2]
        stored_key[..., self.capacity:end, :].copy_(key)
        stored_value[..., self.capacity:end, :].copy_(value)
        return stored_key[..., :end, :], stored_value[..., :end, :]


class SlotBatchPrefix:
    """Read pooled prefix K/V by slot in the N>1 captured decoder paths."""

    def __init__(self, pool, slots, lengths, capacity):
        self.pool, self.slots, self.lengths, self.capacity = pool, slots, lengths, capacity

    def update(self, key, value, layer_idx, cache_kwargs=None):
        self.latest = (key, value)
        return key, value


class PackedPrefix:
    """Independent prefixes followed by packed append rows for one layer call."""
    def __init__(self, kv):
        self.kv = kv
        self.appended = {}

    def update(self, key, value, layer_idx, cache_kwargs=None):
        self.appended[layer_idx] = (key, value)
        prefix_key, prefix_value = self.kv[layer_idx]
        return torch.cat((prefix_key, key), dim=-2), torch.cat((prefix_value, value), dim=-2)


class StaticPrefixPool:
    """Stable sequence slots in zero-initialized per-layer K/V tensors."""
    def __init__(self, capacity):
        self.capacity = capacity
        self.owners = [None] * 16
        self.free = list(range(15, -1, -1))
        self.kv = {}







class IncrementalPrefix:
    """Append-only layer-34/35 K/V owned by one served sequence."""
    def __init__(self):
        self.length = 0
        self.kv = {}

    def update(self, key, value, layer_idx, cache_kwargs=None):
        if layer_idx in self.kv:
            key = torch.cat((self.kv[layer_idx][0], key), dim=-2)
            value = torch.cat((self.kv[layer_idx][1], value), dim=-2)
        self.kv[layer_idx] = (key.detach(), value.detach())
        return key, value


class StaticPrefix:
    """Sequence-owned fixed-capacity K/V storage for target layers 34-35."""
    def __init__(self, capacity, pool=None):
        self.capacity = int(capacity)
        self.length = 0
        self.kv = {}
        self.pool = pool
        self.slot = pool.acquire(self) if pool is not None else None

    def _storage(self, layer_idx, key, value):
        if self.pool is not None:
            return self.pool.storage(self, layer_idx, key, value)
        if layer_idx not in self.kv:
            shape = (*key.shape[:-2], self.capacity, key.shape[-1])
            self.kv[layer_idx] = (
                torch.empty(shape, device=key.device, dtype=key.dtype),
                torch.empty(shape, device=value.device, dtype=value.dtype),
            )
        stored_key, stored_value = self.kv[layer_idx]
        if stored_key.shape[:-2] != key.shape[:-2] or stored_key.shape[-1] != key.shape[-1]:
            raise ValueError('static decoder cache shape changed')
        return stored_key, stored_value

    def update(self, key, value, layer_idx, cache_kwargs=None):
        cache_kwargs = {} if cache_kwargs is None else cache_kwargs
        positions = cache_kwargs.get('cache_position')
        if positions is None:
            positions = torch.arange(
                self.length, self.length + key.shape[-2], device=key.device)
        positions = torch.as_tensor(positions, device=key.device, dtype=torch.long).reshape(-1)
        end = self.length + key.shape[-2]
        if positions.numel() != key.shape[-2] or end > self.capacity:
            raise ValueError('decoder prefix exceeds its static cache capacity')
        stored_key, stored_value = self._storage(layer_idx, key, value)
        stored_key.index_copy_(-2, positions, key)
        stored_value.index_copy_(-2, positions, value)
        return stored_key[..., :end, :], stored_value[..., :end, :]


class StaticWindowPrefix:
    """Append one private window into the sequence prefix's spare storage."""
    def __init__(self, prefix, context_length):
        self.prefix = prefix
        self.context_length = int(context_length)

    def update(self, key, value, layer_idx, cache_kwargs=None):
        start = self.context_length
        end = start + key.shape[-2]
        if end > self.prefix.capacity:
            raise ValueError('decoder window exceeds its static cache capacity')
        stored_key, stored_value = self.prefix._storage(layer_idx, key, value)
        stored_key[..., start:end, :].copy_(key)
        stored_value[..., start:end, :].copy_(value)
        return stored_key[..., :end, :], stored_value[..., :end, :]


@dataclass(frozen=True)
class DecoderPrediction:
    hidden: torch.Tensor
    logits: torch.Tensor | None
    offers: AnchorOffers
    elapsed_ms: float
    total_ms: float = 0.0
    prefix_ms: float = 0.0
    select_ms: float = 0.0
    stage_ms: dict[str, float] | None = None
    rejected: torch.Tensor | None = None
    compact_q: torch.Tensor | None = None
    compact_argmax_eq: torch.Tensor | None = None
