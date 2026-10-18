"""27B GDN per-head RMSNorm + FP32 SiLU gate with a W1 activation side output.

Retains FLA's single BF16 rounding at the output. The reduction may differ at
rounding boundaries; the producer and GEMM chain is gated at Frobenius 2e-3.
"""
import torch

try:
    import triton
    import triton.language as tl
    from .nvfp4._w4a8_quant import store_activation_row
except ImportError:
    triton = None


if triton is not None:
    @triton.jit
    def _gated_norm(X, Z, W, Y, Q, SF, xs, zs, eps,
                    HEADS: tl.constexpr, D: tl.constexpr, TILE: tl.constexpr):
        row = tl.program_id(0)
        head = tl.program_id(1)
        col = tl.arange(0, D)
        source = row * HEADS + head
        x = tl.load(X + source * xs + col).to(tl.float32)
        z = tl.load(Z + source * zs + col).to(tl.float32)
        w = tl.load(W + col).to(tl.float32)
        scale = 1. / tl.sqrt(tl.sum(x * x, 0) / D + eps)
        y = ((x * scale) * w) * z * tl.sigmoid(z)
        output = y.to(tl.bfloat16)
        tl.store(Y + source * D + col, output)
        store_activation_row(output, Q, SF, row, head * D, HEADS * D, D, TILE)


def gated_norm(x, z, norm, heads, *, activation):
    """Per-head normalization of BF16 [tokens * heads, D] into contiguous token rows."""
    layout, q, sf = activation
    width = x.shape[-1]
    if width not in (128, 256) or x.stride(-1) != 1 or z.stride(-1) != 1:
        raise ValueError('GDN W1 producer requires contiguous 128/256-value heads')
    out = torch.empty_like(x).view(-1, heads * width)
    _gated_norm[(out.shape[0], heads)](
        x, z, norm.weight, out, q, sf, x.stride(0), z.stride(0), float(norm.eps),
        HEADS=heads, D=width, TILE=layout.token_tile, num_warps=1)
    out._verify_w1 = activation
    return out
