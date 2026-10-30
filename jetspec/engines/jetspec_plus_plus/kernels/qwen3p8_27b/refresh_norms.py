"""Bitwise before-weight refresh norms with the native contiguous reduction order."""
import torch

try:
    import triton
    import triton.language as tl
    from jetspec.engines.jetspec_plus_plus.kernels.c1_fused import _module_rms_scale
    from jetspec.engines.jetspec_plus_plus.kernels.qwen3p8_27b.draft_w1h27 import _packed_rms_scale
    from jetspec.engines.jetspec_plus_plus.kernels.nvfp4.producers import store_mx8
    from jetspec.engines.jetspec_plus_plus.kernels.nvfp4._w4a8_quant import _launch_dependents
except ImportError:
    triton = tl = None


if triton is not None:
    @triton.jit
    def _plain_norms(X, W, Y, ROWS: tl.constexpr, WIDTH: tl.constexpr,
                     THREADS: tl.constexpr, EPS: tl.constexpr, BLOCK: tl.constexpr,
                     Q=None, SF=(), TILES: tl.constexpr = (), STRIDES: tl.constexpr = ()):
        row = tl.program_id(0) % ROWS
        group = tl.program_id(0) // ROWS
        if len(TILES):
            _launch_dependents(row)
        for i in tl.static_range(len(X)):
            if group == i:
                column = tl.arange(0, BLOCK)
                value = tl.load(X[i] + row * WIDTH + column, column < WIDTH, 0).to(tl.float32)
                if WIDTH > 128:
                    scale = _packed_rms_scale(X[i], X[i], X[i], row, WIDTH, THREADS, EPS, False)
                else:
                    scale = _module_rms_scale(value, WIDTH, THREADS, EPS, BLOCK)
                normed = (value * scale).to(tl.bfloat16).to(tl.float32)
                weight = tl.load(W[i] + column, column < WIDTH, 0).to(tl.float32)
                tl.store(Y[i] + row * WIDTH + column, normed * weight, column < WIDTH)
                if len(TILES):
                    store_mx8(normed * weight, Q, SF, row, 0, 0, WIDTH, BLOCK, TILES, STRIDES)


def plain_norms(inputs, weights, eps, *, mx8=False):
    """Co-launch independent norms, preserving stacked-mean and BF16 boundaries."""
    from jetspec.engines.jetspec_plus_plus.kernels.c1_fused import _module_reduce_threads
    rows, width = inputs[0].shape
    outputs = tuple(torch.empty_like(x) for x in inputs)
    extra = ()
    if mx8:
        from jetspec.engines.jetspec_plus_plus.kernels.nvfp4.producers import allocate_activation, _arguments
        assert len(inputs) == 1
        extra = _arguments(allocate_activation(outputs[0]))
    _plain_norms[(len(inputs) * rows,)](
        inputs, weights, outputs, rows, width, _module_reduce_threads(width, rows * len(inputs)),
        eps, triton.next_power_of_2(width), *extra, num_warps=4, enable_fp_fusion=False)
    return outputs
