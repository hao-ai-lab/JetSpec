"""Bitwise elementwise controls for the hybrid decoder tail."""
import torch

try:
    import triton
    import triton.language as tl
    from triton.language.extra.cuda import libdevice
    from jetspec.engines.jetspec_plus_plus.kernels.verify_mlp_glue import _ratio, _product
    from jetspec.engines.jetspec_plus_plus.kernels.nvfp4.producers import store_mx8
    from jetspec.engines.jetspec_plus_plus.kernels.nvfp4._w4a8_quant import _launch_dependents
except ImportError:
    triton = tl = None


if triton is not None:
    @triton.jit
    def _g27d_gate(A, Gate, Y, N: tl.constexpr, HEADS: tl.constexpr, DIM: tl.constexpr,
                   AT: tl.constexpr, AH: tl.constexpr, GT: tl.constexpr, GH: tl.constexpr,
                   SILU: tl.constexpr, BLOCK: tl.constexpr,
                   Q=None, SF=(), TILES: tl.constexpr = (), STRIDES: tl.constexpr = ()):
        index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        if len(TILES):
            _launch_dependents(tl.program_id(0))
        token, head, col = index // (HEADS * DIM), index // DIM % HEADS, index % DIM
        value = tl.load(A + token * AT + head * AH + col, index < N, 0).to(tl.float32)
        gate = tl.load(Gate + token * GT + head * GH + col, index < N, 0).to(tl.float32)
        numerator = gate if SILU else tl.full((BLOCK,), 1., tl.float32)
        activated = _ratio(numerator, libdevice.exp(-gate)).to(tl.bfloat16).to(tl.float32)
        tl.store(Y + index, _product(value, activated), index < N)
        if len(TILES):
            store_mx8(_product(value, activated), Q, SF, tl.program_id(0) // (HEADS * DIM // BLOCK),
                      0, (tl.program_id(0) % (HEADS * DIM // BLOCK)) * BLOCK,
                      HEADS * DIM, BLOCK, TILES, STRIDES)

    @triton.jit
    def _gdn_controls(A, B, Bias, Logs, Beta, G, N: tl.constexpr,
                      HEADS: tl.constexpr, BLOCK: tl.constexpr):
        index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        head = index % HEADS
        a = tl.load(A + index, index < N, 0).to(tl.float32)
        b = tl.load(B + index, index < N, 0).to(tl.float32)
        bias = tl.load(Bias + head).to(tl.float32)
        logs = tl.load(Logs + head).to(tl.float32)
        value = a + bias
        exponential = libdevice.exp(value)
        softplus = tl.where(value > 20., value,
                            tl.where(exponential < 2. ** -126, exponential, libdevice.log1p(exponential)))
        tl.store(G + index, -libdevice.exp(logs) * softplus, index < N)
        beta = tl.inline_asm_elementwise('rcp.rn.f32 $0, $1;', '=f,f',
                                        [1. + libdevice.exp(-b)], dtype=tl.float32,
                                        is_pure=True, pack=1)
        tl.store(Beta + index, beta, index < N)


def gate_mul(value, gate, *, activation='sigmoid'):
    """Fuse activation and multiply with the eager BF16 rounding between them."""
    if triton is None or not value.is_cuda or value.dtype != torch.bfloat16:
        return value * (gate.sigmoid() if activation == 'sigmoid' else torch.nn.functional.silu(gate))
    from jetspec.engines.jetspec_plus_plus.kernels.decoder_refresh import _geometry
    heads, at, ah = _geometry(value)
    _, gt, gh = _geometry(gate)
    output = torch.empty_like(value, memory_format=torch.contiguous_format)
    extra = ()
    from jetspec.engines.jetspec_plus_plus.profiles import kernel_value
    if kernel_value('draft_fuse_g27f', '0') == '1':
        from jetspec.engines.jetspec_plus_plus.kernels.nvfp4.producers import enabled, allocate_activation, _arguments
        flat = output.reshape(-1, heads * value.shape[-1])
        if enabled(flat) and flat.shape[-1] % 256 == 0:
            extra = _arguments(allocate_activation(flat))
    _g27d_gate[(triton.cdiv(value.numel(), 256),)](
        value, gate, output, value.numel(), heads, value.shape[-1], at, ah, gt, gh,
        activation == 'silu', 256, *extra, enable_fp_fusion=False)
    return output


def gdn_controls(a, b, bias, logs):
    """Keep BF16 sigmoid and FP32 softplus intermediates exactly as eager ops."""
    if triton is None or not a.is_cuda:
        return b.sigmoid(), -logs.float().exp() * torch.nn.functional.softplus(a.float() + bias)
    beta, g = torch.empty_like(b), torch.empty_like(a, dtype=torch.float32)
    _gdn_controls[(triton.cdiv(a.numel(), 256),)](
        a, b, bias, logs, beta, g, a.numel(), a.shape[-1], 256, enable_fp_fusion=False)
    return beta, g
