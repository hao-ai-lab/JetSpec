"""Default-off T13 adapter glue; arithmetic stays at the existing bf16 boundaries."""
import torch

from jetspec.engines.jetspec_plus_plus.wire.contract import Offset

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None


if triton is not None:
    @triton.jit
    def _finish_refresh(S0, S1, S2, S3, RESIDUAL, UPDATE, COUNTS, OUT,
                        ROWS: tl.constexpr, WINDOW: tl.constexpr, WIDTH: tl.constexpr,
                        TOKENS: tl.constexpr, R: tl.constexpr, BLOCK: tl.constexpr):
        padded_row = tl.program_id(0)
        row = padded_row // (WINDOW + 1)
        slot = padded_row % (WINDOW + 1)
        stream = tl.program_id(1)
        r = tl.arange(0, R)
        counts = tl.load(COUNTS + r, r < ROWS, other=0)
        start = tl.sum(tl.where(r < row, counts, 0), 0)
        count = tl.sum(tl.where(r == row, counts, 0), 0)
        col = tl.arange(0, BLOCK)
        valid = (row < ROWS) & (slot < count) & (slot < WINDOW) & (start + slot < TOKENS)
        source = (start + slot) * WIDTH + col
        if stream == 0:
            value = tl.load(S0 + source, valid & (col < WIDTH), other=0)
        elif stream == 1:
            value = tl.load(S1 + source, valid & (col < WIDTH), other=0)
        elif stream == 2:
            value = tl.load(S2 + source, valid & (col < WIDTH), other=0)
        elif stream == 3:
            value = tl.load(S3 + source, valid & (col < WIDTH), other=0)
        else:
            base = tl.load(RESIDUAL + source, valid & (col < WIDTH), other=0)
            delta = tl.load(UPDATE + source, valid & (col < WIDTH), other=0)
            value = (base + delta).to(tl.bfloat16)
        tl.store(OUT + padded_row * (5 * WIDTH) + stream * WIDTH + col,
                 value, col < WIDTH)


def finish_refresh(streams, residual, update, counts, window):
    """Finish five streams directly into the refresh consumer's padded layout."""
    if not residual.is_cuda:
        from jetspec.engines.jetspec_plus_plus.jetspec.c1_fused import finish_streams
        from jetspec.engines.jetspec_plus_plus.kernels.refresh_prep import scatter
        return scatter(finish_streams(streams, residual, update), counts, window)
    rows, width = counts.numel(), residual.shape[-1]
    padded = residual.new_empty(rows + 1, window + 1, 5 * width)
    _finish_refresh[((rows + 1) * (window + 1), 5)](
        *streams, residual, update, counts, padded, rows, window, width,
        residual.shape[0], triton.next_power_of_2(rows), triton.next_power_of_2(width),
        num_warps=8)
    return padded[:rows, :window]


def front_prep(rp, committed, pending, weight, ids, positions, slots, query_slots,
               embeddings, mask, capacity, *, write_mask=True):
    """Gather embeddings and prepare all C1 front metadata in one launch."""
    lanes, width = embeddings.shape[-2:]
    if embeddings.is_cuda:
        from jetspec.engines.jetspec_plus_plus.kernels.chain_glue import _c1_front_prep
        bucket = mask.shape[-1] if write_mask else 0
        _c1_front_prep[(triton.cdiv(lanes * max(width, bucket), 256),)](
            rp, committed, pending, ids, positions, slots, capacity, lanes, 256,
            weight, embeddings, query_slots, mask, width, bucket,
            float(torch.finfo(embeddings.dtype).min), True)
        return
    lane = torch.arange(lanes, device=rp.device)
    length, start, count = rp[Offset.L], rp[Offset.C1_START], rp[Offset.C1_COUNT]
    ids.copy_(torch.cat((committed[length.reshape(1)], pending[:lanes - 1]))[None])
    positions.reshape(-1).copy_(start + lane)
    slots.copy_(torch.where(lane < count, start + lane, capacity + lane))
    query_slots.copy_(length + lane)
    embeddings.copy_(weight[ids])
    if write_mask:
        allowed = torch.arange(mask.shape[-1], device=rp.device)[None] < length + lane[:, None]
        mask.copy_(torch.where(allowed, 0., torch.finfo(mask.dtype).min)[None, None])


def rotary_tables(owner, like):
    """Reuse the same rotary table as the existing C1 gather path."""
    table = getattr(owner, '_c1_rope_table', None)
    if table is None or table[0].dtype != like.dtype or table[0].device != like.device:
        cos, sin = owner.streams[0].rotary_emb(
            like, torch.arange(8192, device=like.device)[None])
        table = owner._c1_rope_table = (cos[0], sin[0])
    return table
