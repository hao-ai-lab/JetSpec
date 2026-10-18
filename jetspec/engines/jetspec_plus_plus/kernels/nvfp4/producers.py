"""BF16 draft producers with an MXFP8 side output owned by the produced tensor.

Each tuned token tile gets its own scale atoms in the same launch. This lets two
consumers retain their tuned configurations without another quantizer. Storage
owns the side output, including inference-mode views that have no base tensor.
Copies and tracked mutations fall back to the ordinary quantizer. Inference
outputs are private producer values and must remain immutable until consumed.
"""
from dataclasses import dataclass

import torch

from .w4a8 import MAX_M, TUNED, token_tile

try:
    import triton
    import triton.language as tl
    from ._w4a8_quant import quantize_activation_block
except ImportError:
    triton = tl = None


@dataclass(frozen=True)
class W4A8Activation:
    data: torch.Tensor
    scales: torch.Tensor
    tile: int


def enabled(x):
    from jetspec.engines.jetspec_plus_plus.engine import current_engine
    from jetspec.engines.jetspec_plus_plus.profiles import kernel_enabled
    return (x.is_cuda and x.dtype == torch.bfloat16 and x.shape[-1] % 128 == 0
            and 128 <= x.shape[-1] and 1 <= x.numel() // x.shape[-1] <= MAX_M
            and current_engine().precision_draft == 'mxfp4_w4a8'
            and kernel_enabled('draft_fuse_w1', '1'))


def tiles_for(m, k):
    bucket = token_tile(m)
    return tuple(sorted({bucket, *(config.tile_n for (n, width, rows), config in TUNED.items()
                                  if width == k and rows == bucket)}))


def _version(x):
    try:
        return x._version
    except RuntimeError:  # Private inference outputs have no version counter.
        return None


class _Produced:
    def __init__(self, output, group_rows):
        self.rows, self.width = group_rows, output.shape[-1]
        self.groups = output.numel() // (self.rows * self.width)
        self.data = torch.empty((self.groups * self.rows, self.width),
                                dtype=torch.float8_e4m3fn, device=output.device)
        self.tiles = tiles_for(self.rows, self.width)
        self.scales = {}
        for tile in self.tiles:
            atoms = triton_cdiv(self.rows, tile) * triton_cdiv(tile, 128)
            self.scales[tile] = torch.empty((self.groups, atoms * 128, self.width // 32),
                                            dtype=torch.uint8, device=output.device)
        self.pointer, self.version = output.data_ptr(), _version(output)


def triton_cdiv(a, b):
    return (a + b - 1) // b


def allocate_activation(output, *, group_rows=None):
    """Attach owned buffers before a producer launch; no device work or global cache."""
    produced = _Produced(output, group_rows or output.numel() // output.shape[-1])
    output.untyped_storage()._w4a8_activation = produced
    return produced


def activation_for(x, tile):
    """Read an exact produced view (or one complete group), rejecting stale aliases."""
    if not x.is_contiguous():
        return None
    produced = getattr(x.untyped_storage(), '_w4a8_activation', None)
    if produced is None or _version(x) != produced.version or tile not in produced.scales:
        return None
    if x.shape[-1] != produced.width or x.numel() != produced.rows * produced.width:
        return None
    offset = (x.data_ptr() - produced.pointer) // x.element_size()
    if offset < 0 or offset % x.numel() or offset // x.numel() >= produced.groups:
        return None
    group = offset // x.numel()
    return W4A8Activation(produced.data[group * produced.rows:(group + 1) * produced.rows],
                          produced.scales[tile][group], tile)


def _arguments(produced):
    scales = tuple(produced.scales[tile] for tile in produced.tiles)
    strides = tuple(sf[0].numel() for sf in scales)
    return produced.data, scales, produced.tiles, strides


if triton is not None:
    @triton.jit
    def store_mx8(values, Q, SF, row, group, column,
                  WIDTH: tl.constexpr, BLOCK: tl.constexpr, TILES: tl.constexpr,
                  STRIDES: tl.constexpr, ROWS=0):
        values = values.to(tl.bfloat16).to(tl.float32)
        q, byte = quantize_activation_block(values, BLOCK)
        col = column + tl.arange(0, BLOCK)
        tl.store(Q + (group * ROWS + row) * WIDTH + col, q, col < WIDTH)
        block = column // 32 + tl.arange(0, BLOCK // 32)
        for i in tl.static_range(len(TILES)):
            atom = min(TILES[i], 128)
            within = row % atom
            physical = ((row // atom) * (WIDTH // 128) + block // 4) * 512
            physical += (within % 32) * 16 + (within // 32) * 4 + block % 4
            tl.store(SF[i] + group * STRIDES[i] + physical, byte.to(tl.uint8), block < WIDTH // 32)

    @triton.jit
    def _norm(X, Residual, Gamma, Output, State, Q, SF, Append, Ids,
              WIDTH: tl.constexpr, EPS: tl.constexpr, BLOCK: tl.constexpr,
              ADD: tl.constexpr, APPEND: tl.constexpr, EMBEDDING: tl.constexpr,
              TILES: tl.constexpr, STRIDES: tl.constexpr):
        row = tl.program_id(0)
        col = tl.arange(0, BLOCK)
        mask = col < WIDTH
        source_row = row - APPEND
        if EMBEDDING:
            source_row = tl.load(Ids + source_row, row >= APPEND, 0)
        value = tl.load(X + source_row * WIDTH + col, mask & (row >= APPEND), 0)
        if ADD:
            residual = tl.load(Residual + (row - APPEND) * WIDTH + col,
                               mask & (row >= APPEND), 0)
            value = (value + residual).to(tl.bfloat16)
        tl.store(State + (row - APPEND) * WIDTH + col, value, mask & (row >= APPEND))
        if APPEND:
            appended = tl.load(Append + row * WIDTH + col, mask & (row < APPEND), 0)
            value = tl.where(row < APPEND, appended, value)
        floats = value.to(tl.float32)
        scale = tl.rsqrt(tl.sum(floats * floats, 0) / WIDTH + EPS)
        normalized = (floats * scale).to(tl.bfloat16)
        weight = tl.load(Gamma + col, mask, 0)
        output = (normalized * weight).to(tl.bfloat16)
        tl.store(Output + row * WIDTH + col, output, mask)
        store_mx8(output, Q, SF, row, 0, 0, WIDTH, BLOCK, TILES, STRIDES)

    @triton.jit
    def _silu(Gate, Up, Output, Q, SF, WIDTH: tl.constexpr,
              GSTRIDE: tl.constexpr, USTRIDE: tl.constexpr,
              BLOCK: tl.constexpr, TILES: tl.constexpr, STRIDES: tl.constexpr):
        row, chunk = tl.program_id(0), tl.program_id(1)
        col = chunk * BLOCK + tl.arange(0, BLOCK)
        gate = tl.load(Gate + row * GSTRIDE + col, col < WIDTH, 0).to(tl.float32)
        up = tl.load(Up + row * USTRIDE + col, col < WIDTH, 0)
        # Torch SiLU rounds before the BF16 multiply by up.
        exponential = tl.exp(-gate)
        activated = (gate / (1. + exponential)).to(tl.bfloat16)
        output = (activated * up).to(tl.bfloat16)
        tl.store(Output + row * WIDTH + col, output, col < WIDTH)
        store_mx8(output, Q, SF, row, 0, chunk * BLOCK, WIDTH, BLOCK, TILES, STRIDES)


def norm(x, weight, eps, *, residual=None, append=None, ids=None):
    """Norm, optional residual add / appended rows / embedding gather, and MX8 in one launch."""
    width = weight.numel()
    rows = ids.numel() if ids is not None else x.numel() // width
    appended = 0 if append is None else append.numel() // width
    if not x.is_cuda or x.dtype != torch.bfloat16 or rows + appended > MAX_M:
        from jetspec.engines.jetspec_plus_plus.jetspec.c1_fused import _reference_rms_norm
        state = x if residual is None else x + residual
        source = state if append is None else torch.cat((append, state))
        return _reference_rms_norm(source, weight, eps), state
    shape = (rows + appended, width) if append is not None or ids is not None else x.shape
    output = torch.empty(shape, device=x.device, dtype=x.dtype)
    state = torch.empty((rows, width), device=x.device, dtype=x.dtype)
    produced = allocate_activation(output)
    _norm[(rows + appended,)](x, residual, weight, output, state, *_arguments(produced)[:2],
                              append, ids, width, eps, triton.next_power_of_2(width),
                              residual is not None, appended, ids is not None,
                              *_arguments(produced)[2:], num_warps=8, enable_fp_fusion=False)
    return output, state.reshape(x.shape) if ids is None else state


def add_norm(update, residual, weight, eps):
    return norm(update, weight, eps, residual=residual)


def rms_norm(x, weight, eps):
    return norm(x, weight, eps)[0]


def silu_mul(gate, up):
    shape, width = gate.shape, gate.shape[-1]
    if (not gate.is_cuda or gate.dtype != torch.bfloat16 or width % 128
            or gate.numel() // width > MAX_M):
        return torch.nn.functional.silu(gate) * up
    output = torch.empty(shape, device=gate.device, dtype=gate.dtype)
    produced = allocate_activation(output)
    _silu[(gate.numel() // width, triton.cdiv(width, 1024))](
        gate, up, output, *_arguments(produced)[:2], width, gate.stride(-2), up.stride(-2),
        1024, *_arguments(produced)[2:], num_warps=4, enable_fp_fusion=False)
    return output


def grouped_norm(prefix, weights, stream_indices, eps):
    from jetspec.engines.jetspec_plus_plus.jetspec.c1_fused import _grouped_rms_norm_w1_kernel
    blocks, width = weights.shape
    rows = prefix.shape[1]
    output = prefix.new_empty((blocks, rows, width))
    produced = allocate_activation(output, group_rows=rows)
    _grouped_rms_norm_w1_kernel[(blocks * rows,)](
        prefix, weights, stream_indices, output, rows, prefix.shape[-1] // width,
        width, eps, triton.next_power_of_2(width), *_arguments(produced), num_warps=8)
    return output
