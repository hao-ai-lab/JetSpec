"""27B (offset, after-weight) RMSNorm and residual add + RMSNorm in one launch each.

The eager after-weight norm of models/target_layers.py RMSNorm is about nine launches: the
FP32 copy, square, mean, eps add, rsqrt, the scale product, the offset weight add, the weight
product and the BF16 copy. The residual add before it is one more. Here one program per row
(or per row and head) keeps the eager arithmetic: the add rounds to BF16 once, then
FP32 x * rsqrt(mean(x * x) + eps) * (offset + w) with one BF16 rounding. Only the FP32 sum
order of the mean differs from the ATen reduction, so results are near the BF16 rounding
error, not bitwise (class B). The residual sum is bitwise to the eager add.
"""
import torch

try:
    import triton
    import triton.language as tl
    from triton.language.extra.cuda import libdevice
    from .nvfp4._w4a8_quant import store_activation_row
except ImportError:  # The CPU suite runs without Triton.
    triton = None


if triton is not None:
    @triton.jit
    def _rms_norm(X, R, W, Y, S, x_row, x_head, r_row, heads, eps, offset, inverse,
                  D: tl.constexpr, BD: tl.constexpr, ADD: tl.constexpr,
                  Q, SF, TILE: tl.constexpr):
        program = tl.program_id(0)
        row = (program // heads).to(tl.int64)
        head = program % heads
        cols = tl.arange(0, BD)
        live = cols < D
        x = tl.load(X + row * x_row + head * x_head + cols, live, 0.).to(tl.float32)
        if ADD:
            # The eager bf16 add: FP32 sum, one BF16 rounding; the norm reads the rounded sum.
            total = (x + tl.load(R + row * r_row + cols, live, 0.).to(tl.float32)).to(tl.bfloat16)
            tl.store(S + row * D + cols, total, live)
            x = total.to(tl.float32)
        # ATen MeanOps: the FP32 sum times the FP32 factor 1 / D.
        mean = tl.sum(x * x, 0) * inverse
        scale = libdevice.rsqrt(mean + eps)
        weight = tl.load(W + cols, live, 0.).to(tl.float32) + offset
        output = ((x * scale) * weight).to(tl.bfloat16)
        tl.store(Y + program.to(tl.int64) * D + cols, output, live)
        if TILE:
            store_activation_row(output, Q, SF, row, 0, D, BD, TILE)


def supported(x, norm):
    """True for a CUDA BF16 input of an after-weight RMSNorm with its last dim the norm width."""
    return (triton is not None and x.is_cuda and x.dtype == torch.bfloat16
            and norm.norm_round == 'after_weight' and x.shape[-1] == norm.weight.shape[0]
            and x.stride(-1) == 1 and x.numel() > 0 and x.dim() in (2, 3))


def _launch(x, residual, norm, out, total, activation=None):
    width = x.shape[-1]
    if x.dim() == 3:
        rows, heads = x.shape[0], x.shape[1]
        x_row, x_head = x.stride(0), x.stride(1)
    else:
        rows, heads = x.shape[0], 1
        x_row, x_head = x.stride(0), 0
    layout, q, sf = activation if activation is not None else (None, None, None)
    _rms_norm[(rows * heads,)](
        x, residual if residual is not None else x, norm.weight, out, total if total is not None else out,
        x_row, x_head, residual.stride(0) if residual is not None else 0, heads, float(norm.eps),
        float(norm.offset), float(torch.tensor(1.0 / width, dtype=torch.float32)), D=width,
        BD=triton.next_power_of_2(width), ADD=residual is not None,
        Q=q, SF=sf, TILE=0 if layout is None else layout.token_tile,
        num_warps=8 if width >= 2048 else 2, enable_fp_fusion=False)
    if activation is not None:
        out._verify_w1 = activation


def rms_norm(x, norm, *, activation=None):
    """``norm(x)`` for [rows, D] or [rows, heads, D] BF16 (any row and head strides); contiguous output."""
    out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    _launch(x, None, norm, out, None, activation)
    return out


def add_rms_norm(x, residual, norm, *, activation=None):
    """``(x + residual, norm(x + residual))`` for [rows, D] BF16 rows, the sum rounded to BF16 first."""
    if x.dim() != 2 or residual.shape != x.shape or residual.stride(-1) != 1 or residual.dtype != x.dtype:
        raise ValueError('verify_norm27b add+norm needs matching [rows, D] BF16 rows')
    total = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    out = torch.empty_like(total)
    _launch(x, residual, norm, out, total, activation)
    return total, out
