"""Decoder-entry norm reads the adapter's immutable last-stream operands."""
from dataclasses import dataclass

import torch

from jetspec.engines.jetspec_plus_plus.profiles import kernel_value


@dataclass
class AdapterTail:
    streams: tuple
    residual: torch.Tensor
    update: torch.Tensor
    counts: torch.Tensor | None = None
    window: int = 0
    small: 'AdapterTail | None' = None


def defer_finish(streams, residual, update, *, out=None, counts=None, window=0):
    """Keep the operands alive until both captured branches have consumed them."""
    if out is None:
        rows, width = counts.numel(), residual.shape[-1]
        out = residual.new_empty(rows + 1, window + 1, 5 * width)[:rows, :window]
    out.untyped_storage()._w4a8_adapter_tail = AdapterTail(
        tuple(streams), residual, update, counts, window)
    return out


def materialize_tail(output):
    """Issue the unchanged five-stream finish on the existing auxiliary branch."""
    tail = getattr(output.untyped_storage(), '_w4a8_adapter_tail', None)
    if tail is None:
        return output
    if tail.counts is None:
        from jetspec.engines.jetspec_plus_plus.kernels.c1_fused import finish_streams
        finish_streams(tail.streams, tail.residual, tail.update,
                       out=output.view(tail.residual.shape[0], -1))
    else:
        import triton
        from jetspec.engines.jetspec_plus_plus.kernels.adapter_glue import _finish_refresh
        width = tail.residual.shape[-1]
        if tail.small is not None:
            _finish_bucketed_refresh[((tail.counts.numel() + 1) * (tail.window + 1), 5)](
                (*tail.streams, tail.residual, tail.update),
                (*tail.small.streams, tail.small.residual, tail.small.update),
                tail.counts, output, tail.counts.numel(), tail.window, width,
                tail.residual.shape[0], tail.small.residual.shape[0],
                triton.next_power_of_2(tail.counts.numel()), triton.next_power_of_2(width),
                num_warps=8)
            return output
        _finish_refresh[((tail.counts.numel() + 1) * (tail.window + 1), 5)](
            *tail.streams, tail.residual, tail.update, tail.counts, output,
            tail.counts.numel(), tail.window, width, tail.residual.shape[0],
            triton.next_power_of_2(tail.counts.numel()), triton.next_power_of_2(width),
            num_warps=8)
    return output


def stage_entry(hidden, source, prefix, tail_rows, stage, pointers, constants, blocks):
    """Attach a first-norm recipe instead of launching the hidden/metadata copy."""
    if kernel_value('draft_fuse_decoder_entry', '0') != '1':
        return False
    adapter = getattr(source.untyped_storage(), '_w4a8_adapter_tail', None)
    if adapter is None:
        return False
    hidden.untyped_storage()._w4a8_entry = (
        adapter, prefix, tail_rows, stage, pointers, constants, blocks)
    return True


def entry_norm(hidden, weight, eps):
    import triton
    from .producers import allocate_activation, _arguments
    recipe = hidden.untyped_storage()._w4a8_entry
    adapter, prefix, tail, stage, pointers, constants, blocks = recipe
    width = hidden.shape[-1]
    rows = hidden.numel() // width
    output = torch.empty_like(hidden)
    produced = allocate_activation(output)
    small_args = ({} if adapter.small is None else dict(
        SMALL_X=adapter.small.update, SMALL_RESIDUAL=adapter.small.residual,
        SMALL_TOKENS=adapter.small.residual.shape[0]))
    _entry_norm[(rows + blocks,)](
        adapter.update, adapter.residual, prefix, tail, adapter.counts,
        hidden, weight, output, *_arguments(produced)[:2], pointers,
        width, eps, triton.next_power_of_2(width), rows,
        prefix.numel() // width if adapter.counts is None else 20,
        0 if tail is None else tail.stride(0),
        0 if adapter.counts is None else adapter.counts.numel(),
        1 if adapter.counts is None else triton.next_power_of_2(adapter.counts.numel()),
        0 if adapter.counts is None else (rows - 20) // adapter.counts.numel(),
        stage, constants, *_arguments(produced)[2:],
        TRIGGER=kernel_value('draft_fuse_pdl', '0') == '1',
        num_warps=8, enable_fp_fusion=False, **small_args)
    return output


try:
    import triton
    import triton.language as tl
    from .producers import store_mx8, _launch_dependents
except ImportError:
    triton = tl = None


if triton is not None:
    from jetspec.engines.jetspec_plus_plus.kernels.adapter_glue import _finish_refresh

    @triton.jit
    def _finish_bucketed_refresh(LARGE, SMALL, COUNTS, OUT,
                                 ROWS: tl.constexpr, WINDOW: tl.constexpr, WIDTH: tl.constexpr,
                                 TOKENS: tl.constexpr, SMALL_TOKENS: tl.constexpr,
                                 R: tl.constexpr, BLOCK: tl.constexpr):
        r = tl.arange(0, R)
        small = tl.sum(tl.load(COUNTS + r, r < ROWS, other=0), 0) <= SMALL_TOKENS
        _finish_refresh(
            tl.where(small, SMALL[0], LARGE[0]), tl.where(small, SMALL[1], LARGE[1]),
            tl.where(small, SMALL[2], LARGE[2]), tl.where(small, SMALL[3], LARGE[3]),
            tl.where(small, SMALL[4], LARGE[4]), tl.where(small, SMALL[5], LARGE[5]),
            COUNTS, OUT, ROWS, WINDOW, WIDTH, TOKENS, R, BLOCK)

    @triton.jit
    def _entry_norm(X, Residual, Prefix, Tail, Counts, Hidden, Gamma, Output, Q, SF,
                    StagePointers, WIDTH: tl.constexpr, EPS: tl.constexpr,
                    BLOCK: tl.constexpr, ROWS: tl.constexpr, PREFIX: tl.constexpr,
                    TAIL_STRIDE: tl.constexpr, GROUPS: tl.constexpr, R: tl.constexpr,
                    WINDOW: tl.constexpr, STAGE: tl.constexpr, STAGE_CONST: tl.constexpr,
                    TILES: tl.constexpr, STRIDES: tl.constexpr, TRIGGER: tl.constexpr,
                    SMALL_X=None, SMALL_RESIDUAL=None, SMALL_TOKENS: tl.constexpr = 0):
        row = tl.program_id(0)
        if row < ROWS:
            if TRIGGER:
                _launch_dependents(row)
            col = tl.arange(0, BLOCK)
            mask = col < WIDTH
            source = row - PREFIX
            valid = row >= PREFIX
            if GROUPS:
                group = tl.maximum(source, 0) // WINDOW
                slot = tl.maximum(source, 0) % WINDOW
                r = tl.arange(0, R)
                counts = tl.load(Counts + r, r < GROUPS, 0)
                if SMALL_TOKENS:
                    small = tl.sum(counts, 0) <= SMALL_TOKENS
                    X = tl.where(small, SMALL_X, X)
                    Residual = tl.where(small, SMALL_RESIDUAL, Residual)
                start = tl.sum(tl.where(r < group, counts, 0), 0)
                count = tl.sum(tl.where(r == group, counts, 0), 0)
                valid = valid & (slot < count)
                source = start + slot
                a = tl.load(Prefix + row * WIDTH + col, mask & (row < 4), 0)
                b = tl.load(Tail + (row - 4) * TAIL_STRIDE + col,
                            mask & (row >= 4) & (row < PREFIX), 0)
                prefix = tl.where(row < 4, a, b)
            else:
                prefix = tl.load(Prefix + row * WIDTH + col, mask & (row < PREFIX), 0)
            value = tl.load(X + source * WIDTH + col, mask & valid, 0)
            residual = tl.load(Residual + source * WIDTH + col, mask & valid, 0)
            value = (value + residual).to(tl.bfloat16)
            value = tl.where(row < PREFIX, prefix, value)
            tl.store(Hidden + row * WIDTH + col, value, mask)
            floats = value.to(tl.float32)
            scale = tl.rsqrt(tl.sum(floats * floats, 0) / WIDTH + EPS)
            normalized = (floats * scale).to(tl.bfloat16)
            weight = tl.load(Gamma + col, mask, 0)
            output = (normalized * weight).to(tl.bfloat16)
            tl.store(Output + row * WIDTH + col, output, mask)
            store_mx8(output, Q, SF, row, 0, 0, WIDTH, BLOCK, TILES, STRIDES)
        else:
            # Triton 3.4 unpacks runtime tuples, but not a constexpr Python tuple.
            if GROUPS:
                STAGE(*StagePointers, STAGE_CONST[0], STAGE_CONST[1], STAGE_CONST[2],
                      STAGE_CONST[3], STAGE_CONST[4], STAGE_CONST[5], STAGE_CONST[6],
                      STAGE_CONST[7], STAGE_CONST[8], STAGE_CONST[9], STAGE_CONST[10],
                      STAGE_CONST[11], STAGE_CONST[12], STAGE_CONST[13], STAGE_CONST[14],
                      OFFSET=ROWS)
            else:
                STAGE(*StagePointers, STAGE_CONST[0], STAGE_CONST[1], STAGE_CONST[2],
                      STAGE_CONST[3], STAGE_CONST[4], STAGE_CONST[5], STAGE_CONST[6],
                      STAGE_CONST[7], STAGE_CONST[8], STAGE_CONST[9], STAGE_CONST[10],
                      STAGE_CONST[11], STAGE_CONST[12], STAGE_CONST[13], STAGE_CONST[14],
                      STAGE_CONST[15], STAGE_CONST[16], STAGE_CONST[17], OFFSET=ROWS)
