"""27B GDN per-head RMSNorm + FP32 SiLU gate with a W1 activation side output.

Retains FLA's single BF16 rounding at the output. The reduction may differ at
rounding boundaries; the producer and GEMM chain is gated at Frobenius 2e-3.
"""
import torch

try:
    import triton
    import triton.language as tl
    from jetspec.engines.jetspec_plus_plus.kernels.nvfp4._w4a8_quant import store_activation_row
    from jetspec.engines.jetspec_plus_plus.kernels.verify_w1 import store_activation_packed
except ImportError:
    triton = None


if triton is not None:
    from jetspec.engines.jetspec_plus_plus.kernels.qwen3p8_27b.verify_pdl27b import producer_start

    @triton.jit
    def _gated_norm(X, Z, W, Y, Q, SF, xs, zt, zs, eps,
                    HEADS: tl.constexpr, D: tl.constexpr, TILE: tl.constexpr, PDL: tl.constexpr,
                    FP8STORE: tl.constexpr = False, SECONDARY: tl.constexpr = False):
        if PDL:
            producer_start(tl.program_id(0))
        row = tl.program_id(0)
        head = tl.program_id(1)
        col = tl.arange(0, D)
        source = row * HEADS + head
        if SECONDARY:
            z = tl.load(Z + row * zt + head * zs + col).to(tl.float32)
            w = tl.load(W + col).to(tl.float32)
            source = tl.inline_asm_elementwise('griddepcontrol.wait; mov.u32 $0, $1;',
                '=r,r,~{memory}', [source], dtype=tl.int32, is_pure=False, pack=1)
            x = tl.load(X + source * xs + col).to(tl.float32)
        else:
            x = tl.load(X + source * xs + col).to(tl.float32)
            z = tl.load(Z + row * zt + head * zs + col).to(tl.float32)
            w = tl.load(W + col).to(tl.float32)
        scale = 1. / tl.sqrt(tl.sum(x * x, 0) / D + eps)
        y = ((x * scale) * w) * z * tl.sigmoid(z)
        output = y.to(tl.bfloat16)
        tl.store(Y + source * D + col, output)
        if FP8STORE:
            store_activation_packed(output, Q, SF, row, head * D, HEADS * D, D, TILE)
        else:
            store_activation_row(output, Q, SF, row, head * D, HEADS * D, D, TILE)


def gated_norm(x, z, norm, heads, *, activation, pdl=False, fp8store=False, secondary=False):
    """Normalize BF16 heads, accepting flat z or a token-strided [1, tokens, heads * D] view."""
    layout, q, sf = activation
    width = x.shape[-1]
    if width not in (128, 256) or x.stride(-1) != 1 or z.stride(-1) != 1:
        raise ValueError('GDN W1 producer requires contiguous 128/256-value heads')
    out = torch.empty_like(x).view(-1, heads * width)
    z_head = width if z.ndim == 3 else z.stride(0)
    z_token = z.stride(1) if z.ndim == 3 else heads * z_head
    _gated_norm[(out.shape[0], heads)](
        x, z, norm.weight, out, q, sf, x.stride(0), z_token, z_head, float(norm.eps),
        HEADS=heads, D=width, TILE=layout.token_tile, PDL=pdl, FP8STORE=fp8store, num_warps=1,
        **({'SECONDARY': True, 'launch_pdl': True} if secondary else {}))
    out._verify_w1 = activation
    return out
