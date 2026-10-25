"""Replay stored scan deltas with separate FP32 multiplies/adds and no reductions.

The private keep is (normalized key, delta, exp(gate)). Committed tensors keep
their dense layout; the all-layer pointer table retains its seven-column ABI.
"""
import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None

BLOCK = 2048


if triton is not None:
    @triton.jit
    def _raw_window(SOURCE, TARGET, count, offsets, live, ROWS: tl.constexpr):
        # Each channel's four taps belong to one CTA. All old taps are loaded
        # before any stores, so the in-place shift has no cross-CTA race.
        channel, tap = offsets // 4, offsets % 4
        index = count + tap
        old = tl.load(TARGET + channel * 4 + index, live & (index < 4), 0)
        new = tl.load(SOURCE + channel * ROWS + index - 4, live & (index >= 4), 0)
        return tl.where(index < 4, old, new)

    @triton.jit
    def _commit_conv_raw(SOURCE, COUNT, TARGET, N: tl.constexpr, ROWS: tl.constexpr, BLOCK: tl.constexpr):
        count = tl.minimum(tl.maximum(tl.load(COUNT), 0), ROWS)
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        value = _raw_window(SOURCE, TARGET, count, offsets, offsets < N, ROWS)
        tl.store(TARGET + offsets, value, offsets < N)

    @triton.jit
    def _stream(KEY, DELTA, EXP, INITIAL, OUT, count, head, tile,
                ROWS: tl.constexpr, HK: tl.constexpr, HV: tl.constexpr,
                DK: tl.constexpr, DV: tl.constexpr, BLOCK: tl.constexpr):
        elements = tile * BLOCK + tl.arange(0, BLOCK)
        live = elements < DK * DV
        key_dim, value_dim = elements // DV, elements % DV
        offsets = head * DK * DV + elements
        state = tl.load(INITIAL + offsets, live, 0.)
        key_head = head // (HV // HK)
        for step in range(count):
            decay = tl.load(EXP + step * HV + head)
            key = tl.load(KEY + (step * HK + key_head) * DK + key_dim, live, 0.)
            delta = tl.load(DELTA + (step * HV + head) * DV + value_dim, live, 0.)
            state = state * decay
            state = state + key * delta
        tl.store(OUT + offsets, state, live)

    @triton.jit
    def _commit_delta(KEY, DELTA, EXP, COUNT, INITIAL, OUT,
                      ROWS: tl.constexpr, HK: tl.constexpr, HV: tl.constexpr,
                      DK: tl.constexpr, DV: tl.constexpr, BLOCK: tl.constexpr):
        count = tl.minimum(tl.maximum(tl.load(COUNT), 0), ROWS)
        _stream(KEY, DELTA, EXP, INITIAL, OUT, count, tl.program_id(0), tl.program_id(1),
                ROWS, HK, HV, DK, DV, BLOCK)

    @triton.jit
    def _commit_delta_layers(TABLE, COUNT,
                            ROWS: tl.constexpr, HK: tl.constexpr, HV: tl.constexpr,
                            DK: tl.constexpr, DV: tl.constexpr, TILES: tl.constexpr,
                            CONV_N: tl.constexpr, CONV_ROWS: tl.constexpr,
                            CONV_TYPE: tl.constexpr, BLOCK: tl.constexpr, RAWKEEP: tl.constexpr = False):
        head, tile = tl.program_id(0), tl.program_id(1)
        row = TABLE + tl.program_id(2) * 7
        raw = tl.load(COUNT)
        if tile < TILES:
            count = tl.minimum(tl.maximum(raw, 0), ROWS)
            if count > 0:
                key = tl.load(row).to(tl.pointer_type(tl.float32))
                delta = tl.load(row + 1).to(tl.pointer_type(tl.float32))
                decay = tl.load(row + 2).to(tl.pointer_type(tl.float32))
                state = tl.load(row + 4).to(tl.pointer_type(tl.float32))
                _stream(key, delta, decay, state, state, count, head, tile,
                        ROWS, HK, HV, DK, DV, BLOCK)
        else:
            index = tl.minimum(tl.maximum(raw, 0), CONV_ROWS - 1)
            source = tl.load(row + 5).to(tl.pointer_type(CONV_TYPE))
            target = tl.load(row + 6).to(tl.pointer_type(CONV_TYPE))
            offsets = ((tile - TILES) * HV + head) * BLOCK + tl.arange(0, BLOCK)
            live = offsets < CONV_N
            if RAWKEEP:
                value = _raw_window(source, target, index, offsets, live, CONV_ROWS - 1)
            else:
                value = tl.load(source + index * CONV_N + offsets, live)
            tl.store(target + offsets, value, live)


def keep_buffers(rows, key_heads, value_heads, key_dim, value_dim, *, device=None):
    return (torch.zeros(rows, key_heads, key_dim, dtype=torch.float32, device=device),
            torch.zeros(rows, value_heads, value_dim, dtype=torch.float32, device=device),
            torch.zeros(rows, value_heads, dtype=torch.float32, device=device))


def commit_state(keep, count, initial, out=None):
    """Replay the accepted scan prefix, reading the device count during graph replay."""
    key, delta, decay = keep
    rows, hk, dk = key.shape
    hv, dv = delta.shape[1:]
    out = initial if out is None else out
    if (initial.shape != (1, hv, dk, dv) or out.shape != initial.shape
            or initial.dtype != torch.float32 or out.dtype != torch.float32
            or decay.shape != (rows, hv) or delta.shape[0] != rows
            or count.numel() != 1 or count.dtype != torch.int64
            or any(t.dtype != torch.float32 or not t.is_contiguous() for t in keep)):
        raise ValueError('commitdelta27b buffer geometry mismatch')
    _commit_delta[(hv, triton.cdiv(dk * dv, BLOCK))](
        key, delta, decay, count, initial, out, rows, hk, hv, dk, dv, BLOCK,
        num_warps=4, enable_fp_fusion=False)


def commit_conv_raw(scratch, count, target):
    """Gather the accepted four-tap window from committed state and raw tokens."""
    _commit_conv_raw[(triton.cdiv(target.numel(), BLOCK),)](
        scratch.view(torch.int16), count, target.view(torch.int16), target.numel(), scratch.shape[0] - 1,
        BLOCK, num_warps=4)


def launch_layers(table, count, geometry, *, rawkeep=False):
    rows, hk, hv, dk, dv, conv_rows, conv_n, conv_size, _ = geometry
    tiles = triton.cdiv(dk * dv, BLOCK)
    conv_tiles = triton.cdiv(conv_n, hv * BLOCK)
    _commit_delta_layers[(hv, tiles + conv_tiles, table.shape[0])](
        table, count, rows, hk, hv, dk, dv, tiles, conv_n, conv_rows,
        tl.int16 if conv_size == 2 else tl.int32, BLOCK,
        **({'RAWKEEP': True} if rawkeep else {}),
        num_warps=4, enable_fp_fusion=False)
