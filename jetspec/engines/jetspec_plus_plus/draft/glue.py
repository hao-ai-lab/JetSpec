"""Capture-safe level-2 decoder window glue."""

from __future__ import annotations

import torch
from jetspec.engines.jetspec_plus_plus.draft.method.passes import j0_fill
from jetspec.engines.jetspec_plus_plus.draft.method.ops import reference
from jetspec.engines.jetspec_plus_plus.kernels import register_optimization
from jetspec.engines.jetspec_plus_plus.wire.contract import Offset
from jetspec.engines.jetspec_plus_plus.wire.triton_offsets import (  # noqa: F401  (read inside the kernels)
    AFTER_MISS, C1_COUNT, C1_START, CORRECTION, DEC_LEN, J_ROUND, RNG_IDX, RP_L, RP_TRUE_K, SLOT,
    TAU_IDX)

try:
    import triton
    import triton.language as tl
except ImportError:  # CPU development hosts do not install Triton.
    triton = None
    tl = None


def available() -> bool:
    return triton is not None


def reference_window_mask_rope(length: torch.Tensor, cos_table: torch.Tensor,
                               sin_table: torch.Tensor, mask: torch.Tensor,
                               cos: torch.Tensor, sin: torch.Tensor) -> None:
    """Torch oracle with level-1 mask and rotary gather rounding points."""
    rows, bucket = mask.shape[-2:]
    query = torch.arange(rows, device=length.device)[:, None]
    key = torch.arange(bucket, device=length.device)[None]
    allowed = (key < length) | ((key >= length) & (key - length <= query))
    mask.zero_().masked_fill_(~allowed[None, None], torch.finfo(mask.dtype).min)
    positions = length + torch.arange(rows, device=length.device)
    cos.copy_(cos_table.index_select(0, positions)[None])
    sin.copy_(sin_table.index_select(0, positions)[None])


if triton is not None:
    from jetspec.engines.jetspec_plus_plus.kernels.chain_glue import (_window_mask_rope, _mask_token_guard, _j0_prep, _c1_front_prep, _c1_mask, _context_copy_mark)


def window_mask_rope(length: torch.Tensor, cos_table: torch.Tensor,
                     sin_table: torch.Tensor, mask: torch.Tensor,
                     cos: torch.Tensor, sin: torch.Tensor) -> None:
    """Write the level-1 mask and both table gathers in one CUDA launch."""
    if triton is None or not length.is_cuda:
        raise RuntimeError("window glue fusion requires CUDA Triton")
    rows, bucket = mask.shape[-2:]
    dim = cos.shape[-1]
    if (length.numel() != 1 or mask.shape != (1, 1, rows, bucket)
            or cos.shape != sin.shape or cos.shape != (1, rows, dim)
            or cos_table.ndim != 2 or cos_table.shape != sin_table.shape
            or cos_table.shape[1] != dim):
        raise ValueError("window glue tensors do not match the captured shape")
    block = 256
    mask_blocks = triton.cdiv(rows * bucket, block)
    rope_blocks = triton.cdiv(rows * dim, block)
    _window_mask_rope[(mask_blocks + rope_blocks,)](
        length, cos_table, sin_table, mask, cos, sin, rows, bucket, dim,
        float(torch.finfo(mask.dtype).min), mask_blocks, block)


def mask_token_guard(ids: torch.Tensor, active: torch.Tensor,
                     err: torch.Tensor, token: int) -> None:
    """Set the existing mask-token error bit for active rows."""
    rows, lanes = ids.shape
    if (active.shape != (rows,) or err.shape != (1,)
            or ids.dtype != torch.int64 or err.dtype != torch.int64):
        raise ValueError("mask-token guard tensors do not match the captured shape")
    if ids.is_cuda and triton is not None:
        _mask_token_guard[(1,)](ids, active, err, rows, lanes, token,
                                triton.next_power_of_2(rows * lanes))
    else:
        err.bitwise_or_((ids.eq(token) & active[:, None]).any().long().reshape(1) * (1 << 7))


def j0_prep(committed, rp, keys, active, ids, finished, passes, err,
            *, owns_err: bool, pin_slot=None, pin_mask=None) -> None:
    """Fill J0 ids and reset pass outputs with device-side RNG indices."""
    lanes = ids.shape[1]
    if pin_slot is not None:
        if ids.is_cuda and triton is not None:
            _j0_prep[(1,)](committed, rp[Offset.RNG_IDX:], keys[:, 1], active,
                ids, finished, passes, err, ids.shape[0], lanes, owns_err,
                triton.next_power_of_2(ids.numel()), PIN_SLOT=pin_slot,
                PIN_MASK=pin_mask, RESET_PINS=True, DENSE_PINS=pin_mask is not None)
            return
        pin_slot.fill_(-1)
        if pin_mask is not None:
            pin_mask.zero_()
    j0_fill(committed, block_size=lanes,
            indices=rp[Offset.RNG_IDX:Offset.RNG_IDX + lanes - 1],
            pinned_first=keys[:, 1:2], out=ids, active=active,
            finished=finished, passes=passes, err=err if owns_err else None)


def optimized_j0_fill(context_ids=None, *, block_size, seed=None, prompt_index=0,
                      init_window=None, indices=None, pinned_first=None, gather_vec=False,
                      noise_key=None, rows=1, vocab=None, mask_token=None,
                      out=None, active=None, finished=None, passes=None, err=None):
    if (triton is None or out is None or not out.is_cuda or indices is None
            or pinned_first.stride(0) != 2):
        return reference('j0_fill')(
            context_ids, block_size=block_size, seed=seed, prompt_index=prompt_index,
            init_window=init_window, indices=indices, pinned_first=pinned_first,
            gather_vec=gather_vec, noise_key=noise_key, rows=rows, vocab=vocab,
            mask_token=mask_token, out=out, active=active, finished=finished,
            passes=passes, err=err)
    rows, lanes = out.shape
    _j0_prep[(1,)](context_ids, indices, pinned_first, active, out, finished, passes,
                    err if err is not None else passes, rows, lanes, err is not None,
                    triton.next_power_of_2(rows * lanes))
    return out


register_optimization('j0_fill', optimized_j0_fill)


def c1_front_prep(rp, committed, pending, ids, positions, slots, capacity) -> None:
    lanes = ids.shape[1]
    if triton is None or not ids.is_cuda:
        raise RuntimeError("C1 front prep fusion requires CUDA Triton")
    _c1_front_prep[(1,)](rp, committed, pending, ids, positions, slots,
                          capacity, lanes, triton.next_power_of_2(lanes))


def c1_mask(length, mask) -> None:
    rows, bucket = mask.shape[-2:]
    if triton is None or not mask.is_cuda:
        raise RuntimeError("C1 mask fusion requires CUDA Triton")
    _c1_mask[(triton.cdiv(rows * bucket, 256),)](
        length, mask, rows, bucket, float(torch.finfo(mask.dtype).min), 256)


def reference_context_copy_mark(length: torch.Tensor, prefix: torch.Tensor,
                                tail: torch.Tensor, embedding: torch.Tensor,
                                context: torch.Tensor) -> None:
    """The level-1 fixed-P copy, zero padding, and two embedding adds."""
    p = prefix.shape[0]
    valid = torch.arange(p, device=prefix.device) < length.reshape(())
    context[:, :p].copy_(torch.where(valid[None, :, None], prefix[None],
                                     torch.zeros_like(prefix)[None]))
    context[:, p:].copy_(tail)
    context[:, :p].add_(embedding[0])
    context[:, p:].add_(embedding[1])


def context_copy_mark(length: torch.Tensor, prefix: torch.Tensor,
                      tail: torch.Tensor, embedding: torch.Tensor,
                      context: torch.Tensor) -> None:
    """Fuse the captured fixed-P context preparation into one launch."""
    p, hidden = prefix.shape
    t = tail.shape[1]
    if (length.numel() != 1 or tail.shape != (1, t, hidden)
            or context.shape != (1, p + t, hidden)
            or embedding.shape != (2, hidden)):
        raise ValueError("context glue tensors do not match the captured shape")
    if triton is None or not length.is_cuda:
        reference_context_copy_mark(length, prefix, tail, embedding, context)
    else:
        if not all(x.is_contiguous() for x in (prefix, tail, embedding, context)):
            raise ValueError("context glue requires contiguous capture buffers")
        block = 256
        _context_copy_mark[(triton.cdiv((p + t) * hidden, block),)](
            length, prefix, tail, embedding, context, p, t, hidden, block)


def context_gather_mark(old_length, length, prefix, tail, embedding, context,
                        cos_table, sin_table, ccos, csin, hidden, cos, sin,
                        source, slots, previous) -> None:
    """T9: stage fixed-width dirty inputs and update only appended/tail context rows.

    Inactive append lanes gather prefix row zero and write to private noise KV
    slots, exactly as chain_context_dirty_rows. Keep the touched context and tail
    rotary rows current for the independent context-KV audit and T11 geometry.
    The caller advances old_length after this launch; writing it here would race
    other CTAs reading the old length.
    """
    p, width = prefix.shape
    t, dim = tail.shape[1], cos.shape[1]
    block = source.numel() - t
    if length.is_cuda and triton is not None:
        from jetspec.engines.jetspec_plus_plus.kernels.chain_glue import _context_gather_mark
        _context_gather_mark[(block + t, triton.cdiv(width, 256))](
            old_length, length, prefix, tail, embedding, context,
            cos_table, sin_table, ccos, csin, hidden, cos, sin, source, slots, previous,
            p, t, width, block, dim, cos_table.stride(0), cos_table.stride(1),
            sin_table.stride(0), sin_table.stride(1), 256, triton.next_power_of_2(dim),
            enable_fp_fusion=False)
        return
    from jetspec.engines.jetspec_plus_plus.jetspec.rows import chain_context_dirty_rows
    selected, destinations = chain_context_dirty_rows(
        old_length, length, prefix=p, tail=t, block=block, noise_start=p + t)
    previous.copy_(old_length)
    source.copy_(selected)
    slots.copy_(destinations)
    prefix_rows = selected[:block]
    base = prefix.index_select(0, prefix_rows)
    base = torch.where((prefix_rows < length)[:, None], base, 0)
    hidden[:block].copy_(base + embedding[0])
    hidden[block:].copy_(tail[0] + embedding[1])
    live = destinations[:block] < p
    context[0, destinations[:block][live]] = hidden[:block][live]
    context[0, p:].copy_(hidden[block:])
    positions = torch.where(selected < p, selected, length + selected - p)
    cos.copy_(cos_table.index_select(0, positions))
    sin.copy_(sin_table.index_select(0, positions))
    ccos[p:].copy_(cos[block:])
    csin[p:].copy_(sin[block:])
