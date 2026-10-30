# Adaptation: package relocation and deferred optional dependencies.
"""CUDA-only NVFP4 activation kernels; imported lazily by the CPU-safe API."""
from jetspec.engines.jetspec_plus_plus._optional import triton
from jetspec.engines.jetspec_plus_plus._optional import tl


@triton.jit
def _partial_amax(X, Partial, COUNT: tl.constexpr, TILE: tl.constexpr):
    offsets = tl.program_id(0) * TILE + tl.arange(0, TILE)
    values = tl.load(X + offsets, offsets < COUNT, other=0).to(tl.float32)
    tl.store(Partial + tl.program_id(0), tl.max(tl.abs(values), 0))


@triton.jit
def _global_scale(Partial, Global, COUNT: tl.constexpr, TILE: tl.constexpr):
    offsets = tl.arange(0, TILE)
    maximum = tl.max(tl.load(Partial + offsets, offsets < COUNT, other=0), 0)
    tl.store(Global, tl.where(maximum == 0, 1., maximum / 2688.))


@triton.jit
def _quantize(X, Packed, Scales, Swizzled, Global,
              ROWS: tl.constexpr, K: tl.constexpr, PAD_BLOCKS: tl.constexpr,
              PER_ROW: tl.constexpr = False):
    row = tl.program_id(0)
    block = tl.program_id(1) * 32 + tl.arange(0, 32)
    column = block[:, None] * 16 + tl.arange(0, 16)[None, :]
    valid = (row < ROWS) & (column < K)
    x = tl.load(X + row * K + column, valid, other=0).to(tl.float32)
    if PER_ROW:
        global_scale = tl.load(Global + row, row < ROWS, other=1.)
    else:
        global_scale = tl.load(Global)
    block_max = tl.max(tl.abs(x), 1)
    if PER_ROW:
        scale = tl.div_rn(block_max, 6. * global_scale)
    else:
        scale = block_max / (6. * global_scale)
    scale = tl.where(scale == 0, 1., scale)
    scale = tl.minimum(tl.maximum(scale, 0.001953125), 448.).to(tl.float8e4nv)
    if PER_ROW:
        normalized = tl.div_rn(x, (scale.to(tl.float32) * global_scale)[:, None])
    else:
        normalized = x / (scale.to(tl.float32) * global_scale)[:, None]
    magnitude = tl.abs(normalized)
    code = ((magnitude > .25).to(tl.uint8) + (magnitude >= .75).to(tl.uint8)
            + (magnitude > 1.25).to(tl.uint8) + (magnitude >= 1.75).to(tl.uint8)
            + (magnitude > 2.5).to(tl.uint8) + (magnitude >= 3.5).to(tl.uint8)
            + (magnitude > 5.).to(tl.uint8))
    sign = ((x.to(tl.uint32, bitcast=True) >> 31) << 3).to(tl.uint8)
    code = code | sign
    pair = tl.broadcast_to(tl.arange(0, 8)[None, :] * 2, (32, 8))
    packed = tl.gather(code, pair, 1) | (tl.gather(code, pair + 1, 1) << 4)
    byte_column = block[:, None] * 8 + tl.arange(0, 8)[None, :]
    tl.store(Packed + row * (K // 2) + byte_column, packed,
             (row < ROWS) & (byte_column < K // 2))
    tl.store(Scales + row * (K // 16) + block, scale, (row < ROWS) & (block < K // 16))
    address = ((row // 128) * (PAD_BLOCKS // 4) + block // 4) * 512
    address += (row % 32) * 16 + ((row % 128) // 32) * 4 + block % 4
    tl.store(Swizzled + address, tl.where((row < ROWS) & (block < K // 16),
                                        scale.to(tl.float32), 0.), block < PAD_BLOCKS)


def quantize_into(x, workspace):
    count = x.numel()
    partials = workspace.partial_amax.numel()
    _partial_amax[(partials,)](x, workspace.partial_amax, count, 1024)
    _global_scale[(1,)](workspace.partial_amax, workspace.value.global_scale,
                        partials, triton.next_power_of_2(partials))
    rows, k = x.shape
    padded_rows, padded_blocks = workspace.value.swizzled_scales.shape
    _quantize[(padded_rows, triton.cdiv(padded_blocks, 32))](
        x, workspace.value.data, workspace.value.block_scales,
        workspace.value.swizzled_scales, workspace.value.global_scale, rows, k, padded_blocks)


@triton.jit
def _fake_quantize_rows(X, Out, K: tl.constexpr, TILE: tl.constexpr):
    row = tl.program_id(0)
    column = tl.arange(0, TILE)
    x = tl.load(X + row * K + column, column < K, other=0).to(tl.float32)
    maximum = tl.max(tl.abs(x), 0)
    global_scale = tl.where(maximum == 0, 1., tl.div_rn(maximum, 2688.))
    blocks = tl.reshape(x, (TILE // 16, 16))
    scale = tl.div_rn(tl.max(tl.abs(blocks), 1), 6. * global_scale)
    scale = tl.where(scale == 0, 1., scale)
    scale = tl.minimum(tl.maximum(scale, 0.001953125), 448.).to(tl.float8e4nv)
    descale = scale.to(tl.float32) * global_scale
    magnitude = tl.abs(tl.div_rn(blocks, descale[:, None]))
    code = ((magnitude > .25).to(tl.int32) + (magnitude >= .75).to(tl.int32)
            + (magnitude > 1.25).to(tl.int32) + (magnitude >= 1.75).to(tl.int32)
            + (magnitude > 2.5).to(tl.int32) + (magnitude >= 3.5).to(tl.int32)
            + (magnitude > 5.).to(tl.int32))
    rounded = tl.where(code == 7, 6., tl.where(code == 6, 4., tl.where(code == 5, 3., code * .5)))
    negative = (blocks.to(tl.uint32, bitcast=True) >> 31) != 0
    decoded = tl.where(negative, -rounded, rounded) * descale[:, None]
    tl.store(Out + row * K + column, tl.reshape(decoded, (TILE,)), column < K)


def fake_quantize_rows_into(x, out):
    rows, k = x.shape
    _fake_quantize_rows[(rows,)](x, out, k, triton.next_power_of_2(k), enable_fp_fusion=False)

@triton.jit
def _row_global_scale(X, Global, K: tl.constexpr, TILE: tl.constexpr):
    row = tl.program_id(0)
    column = tl.arange(0, TILE)
    x = tl.load(X + row * K + column, column < K, other=0).to(tl.float32)
    maximum = tl.max(tl.abs(x), 0)
    tl.store(Global + row, tl.where(maximum == 0, 1., tl.div_rn(maximum, 2688.)))


def quantize_rows_into(x, workspace):
    value = workspace.value
    rows, k = x.shape
    padded_rows, padded_blocks = value.swizzled_scales.shape
    _row_global_scale[(rows,)](x, value.global_scale, k, triton.next_power_of_2(k), enable_fp_fusion=False)
    _quantize[(padded_rows, triton.cdiv(padded_blocks, 32))](
        x, value.data, value.block_scales, value.swizzled_scales, value.global_scale,
        rows, k, padded_blocks, True, enable_fp_fusion=False)
