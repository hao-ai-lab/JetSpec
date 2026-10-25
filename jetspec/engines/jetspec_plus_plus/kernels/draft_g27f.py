"""Share immutable hybrid activation quantization across independent projections."""
import torch

try:
    import triton
    import triton.language as tl
    from jetspec.engines.jetspec_plus_plus.kernels.nvfp4.producers import store_mx8
    from jetspec.engines.jetspec_plus_plus.kernels.nvfp4._w4a8_quant import _launch_dependents
except ImportError:
    triton = tl = None


if triton is not None:
    @triton.jit
    def _shared_quant(X, Q, SF, WIDTH: tl.constexpr, TILES: tl.constexpr,
                      STRIDES: tl.constexpr, BLOCK: tl.constexpr):
        row, chunk = tl.program_id(0), tl.program_id(1)
        _launch_dependents(row)
        column = chunk * BLOCK + tl.arange(0, BLOCK)
        values = tl.load(X + row * WIDTH + column, column < WIDTH, 0)
        store_mx8(values, Q, SF, row, 0, chunk * BLOCK, WIDTH, BLOCK, TILES, STRIDES)


def share_activation(x):
    """Retain BF16 storage and quantize once into every tuned consumer layout."""
    from jetspec.engines.jetspec_plus_plus.kernels.nvfp4.producers import enabled, allocate_activation, _arguments
    if triton is None or not enabled(x) or not x.is_contiguous():
        return
    produced = allocate_activation(x)
    q, sf, tiles, strides = _arguments(produced)
    _shared_quant[(x.numel() // x.shape[-1], triton.cdiv(x.shape[-1], 1024))](
        x, q, sf, x.shape[-1], tiles, strides, 1024, num_warps=4)
