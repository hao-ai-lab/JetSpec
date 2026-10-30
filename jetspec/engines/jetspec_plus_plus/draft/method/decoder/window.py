"""Private anchor windows, prefix caches and prediction values."""
from __future__ import annotations
from dataclasses import dataclass
import torch
from ..ops import reference_op


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
