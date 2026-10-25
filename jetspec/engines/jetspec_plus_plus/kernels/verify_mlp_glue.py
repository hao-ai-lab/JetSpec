"""Hybrid target gated-attention gate and MLP SiLU glue, bitwise to the eager ops.

The eager ops are a gate reshape copy, sigmoid and multiply, and SiLU and multiply.
ATen computes them in FP32 and rounds to BF16 after the activation and after the product.
The add, divide and multiply run as round-to-nearest PTX without flush-to-zero, as nvcc
compiles ATen. The exponential is the same libdevice expf.
"""
import torch
from torch.nn import functional as F

try:
    import triton
    import triton.language as tl
    from triton.language.extra.cuda import libdevice
except ImportError:  # CPU hosts import the eager fallbacks only.
    triton = None


if triton is not None:
    @triton.jit
    def _ratio(numerator, exp_neg):
        # ATen: numerator / (1 + exp(-x)) in FP32, IEEE division, subnormals kept.
        return tl.inline_asm_elementwise(
            '{ .reg .f32 t; add.rn.f32 t, $2, 0f3F800000; div.rn.f32 $0, $1, t; }',
            '=f,f,f', [numerator, exp_neg], dtype=tl.float32, is_pure=True, pack=1)

    @triton.jit
    def _product(a, b):
        return tl.inline_asm_elementwise('mul.rn.f32 $0, $1, $2;', '=f,f,f', [a, b],
                                         dtype=tl.float32, is_pure=True, pack=1)

    @triton.jit
    def _gate_mul(A, G, Y, a_token, g_token, g_head, features: tl.constexpr,
                  dim: tl.constexpr, BLOCK: tl.constexpr):
        token = tl.program_id(0).to(tl.int64)
        feature = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        live = feature < features
        g = tl.load(G + token * g_token + (feature // dim) * g_head + feature % dim,
                    live, 0.).to(tl.float32)
        a = tl.load(A + token * a_token + feature, live, 0.).to(tl.float32)
        one = tl.full([BLOCK], 1.0, tl.float32)
        sigmoid = _ratio(one, libdevice.exp(-g)).to(tl.bfloat16).to(tl.float32)
        tl.store(Y + token * features + feature, _product(a, sigmoid).to(tl.bfloat16), live)

    @triton.jit
    def _silu_mul(GATE, UP, Y, count, BLOCK: tl.constexpr):
        index = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        live = index < count
        x = tl.load(GATE + index, live, 0.).to(tl.float32)
        up = tl.load(UP + index, live, 0.).to(tl.float32)
        activated = _ratio(x, libdevice.exp(-x)).to(tl.bfloat16).to(tl.float32)
        tl.store(Y + index, _product(activated, up).to(tl.bfloat16), live)


def _cuda_bf16(*tensors):
    return triton is not None and all(
        t.is_cuda and t.dtype == torch.bfloat16 and t.device == tensors[0].device for t in tensors)


def gate_mul(mixed, gate):
    """``mixed * torch.sigmoid(gate.reshape(width, -1))`` in one launch, reading the strided gate.

    ``mixed`` is the [width, heads * dim] attention output. ``gate`` is the [width, heads, dim]
    gate half of the query projection.
    """
    width = mixed.shape[0]
    if (not _cuda_bf16(mixed, gate) or mixed.dim() != 2 or gate.dim() != 3
            or not mixed.is_contiguous() or gate.stride(-1) != 1 or gate.shape[0] != width
            or gate.shape[1] * gate.shape[2] != mixed.shape[1] or width == 0):
        return mixed * torch.sigmoid(gate.reshape(width, -1))
    features, dim = mixed.shape[1], gate.shape[2]
    out = torch.empty_like(mixed)
    block = min(1024, triton.next_power_of_2(features))
    _gate_mul[(width, triton.cdiv(features, block))](
        mixed, gate, out, mixed.stride(0), gate.stride(0), gate.stride(1),
        features, dim, block, num_warps=4)
    return out


def silu_mul(gate, up):
    """``F.silu(gate) * up`` in one launch with the BF16 round after the SiLU."""
    if (not _cuda_bf16(gate, up) or gate.shape != up.shape
            or not gate.is_contiguous() or not up.is_contiguous() or gate.numel() == 0):
        return F.silu(gate) * up
    out = torch.empty_like(gate)
    count = gate.numel()
    _silu_mul[(triton.cdiv(count, 1024),)](gate, up, out, count, 1024, num_warps=4)
    return out
