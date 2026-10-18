"""Row-strided GDN gate/beta and MLP SiLU glue for the packed 27B target projections.

Under engine setting ``verify_cut_pack27b2`` the projection outputs are column views of
one packed GEMM output, so each row has its own stride. These launches read those views
directly. Each one is bitwise equal to its contiguous kernel (``gdn_gate.gate_beta``,
``verify_mlp_glue.silu_mul``) on the same values, at the same rounding points.
"""
import torch
from torch.nn import functional as F

from jetspec.engines.jetspec_plus_plus.kernels import gdn_gate, verify_mlp_glue

triton = gdn_gate.triton
libdevice = gdn_gate.libdevice
if triton is not None:
    import triton.language as tl


if triton is not None and libdevice is not None:
    FLT_MIN = tl.constexpr(2.0 ** -126)

    @triton.jit
    def _gate_beta_rows(a_ptr, b_ptr, dt_ptr, decay_ptr, g_ptr, beta_ptr, a_row, b_row,
                        H: tl.constexpr, BLOCK: tl.constexpr):
        row = tl.program_id(0).to(tl.int64)
        head = tl.arange(0, BLOCK)
        mask = head < H
        # The arithmetic of gdn_gate._gate_beta; only the input addressing differs.
        x = (tl.load(a_ptr + row * a_row + head, mask=mask, other=0).to(tl.float32)
             + tl.load(dt_ptr + head, mask=mask, other=0).to(tl.float32))
        e = libdevice.exp(x)
        soft = tl.where(x > 20.0, x, tl.where(e < FLT_MIN, e, libdevice.log1p(e)))
        g = tl.load(decay_ptr + head, mask=mask, other=0) * soft
        tl.store(g_ptr + row * H + head, g, mask=mask)
        b = tl.load(b_ptr + row * b_row + head, mask=mask, other=0).to(tl.float32)
        beta = tl.inline_asm_elementwise('rcp.rn.f32 $0, $1;', '=f,f', [1.0 + libdevice.exp(-b)],
                                         dtype=tl.float32, is_pure=True, pack=1)
        tl.store(beta_ptr + row * H + head, beta.to(beta_ptr.dtype.element_ty), mask=mask)


if verify_mlp_glue.triton is not None:
    from jetspec.engines.jetspec_plus_plus.kernels.verify_mlp_glue import _product, _ratio, libdevice as _mlp_libdevice
    from .nvfp4._w4a8_quant import store_activation_row

    @triton.jit
    def _silu_mul_rows(GATE, UP, Y, gate_row, up_row, features: tl.constexpr, BLOCK: tl.constexpr,
                       Q, SF, TILE: tl.constexpr):
        row = tl.program_id(0).to(tl.int64)
        feature = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        live = feature < features
        # The arithmetic of verify_mlp_glue._silu_mul; only the input addressing differs.
        x = tl.load(GATE + row * gate_row + feature, live, 0.).to(tl.float32)
        up = tl.load(UP + row * up_row + feature, live, 0.).to(tl.float32)
        activated = _ratio(x, _mlp_libdevice.exp(-x)).to(tl.bfloat16).to(tl.float32)
        output = _product(activated, up).to(tl.bfloat16)
        tl.store(Y + row * features + feature, output, live)
        if TILE:
            store_activation_row(output, Q, SF, row, tl.program_id(1) * BLOCK, features, BLOCK, TILE)


def _rows(tensor):
    """The [rows, width] view of a tensor whose last dim is contiguous, or None."""
    if tensor.dim() < 1 or tensor.stride(-1) != 1 or tensor.numel() == 0:
        return None
    if tensor.dim() == 1:
        return tensor.unsqueeze(0)
    try:
        return tensor.view(-1, tensor.shape[-1])
    except RuntimeError:
        return None


def gate_beta_rows(a, b, dt_bias, decay):
    """``gdn_gate.gate_beta`` for row-strided a and b views; g and beta come back contiguous."""
    if triton is None or libdevice is None or not a.is_cuda:
        return gdn_gate.gate_beta(a, b, dt_bias, decay)
    rows_a, rows_b = _rows(a), _rows(b)
    heads = a.shape[-1] if a.dim() else 0
    if (rows_a is None or rows_b is None or a.shape != b.shape or heads == 0 or heads > 1024
            or dt_bias.shape != (heads,) or decay.shape != (heads,)
            or a.dtype not in gdn_gate._DTYPES or b.dtype not in gdn_gate._DTYPES
            or dt_bias.dtype not in gdn_gate._DTYPES or decay.dtype != torch.float32
            or any(t.device != a.device for t in (b, dt_bias, decay))
            or not dt_bias.is_contiguous() or not decay.is_contiguous()
            or (torch.is_grad_enabled() and any(t.requires_grad for t in (a, b, dt_bias, decay)))):
        return gdn_gate.gate_beta(a, b, dt_bias, decay)
    g = torch.empty(a.shape, dtype=torch.float32, device=a.device)
    beta = torch.empty(b.shape, dtype=b.dtype, device=b.device)
    _gate_beta_rows[(rows_a.shape[0],)](
        rows_a, rows_b, dt_bias, decay, g, beta, rows_a.stride(0), rows_b.stride(0),
        H=heads, BLOCK=triton.next_power_of_2(heads), num_warps=1, enable_fp_fusion=False)
    return g, beta


def silu_mul_rows(gate, up, *, activation=None):
    """``verify_mlp_glue.silu_mul`` for row-strided gate and up views, in one launch."""
    rows_gate, rows_up = _rows(gate), _rows(up)
    if (not verify_mlp_glue._cuda_bf16(gate, up) or gate.shape != up.shape
            or rows_gate is None or rows_up is None):
        return F.silu(gate) * up
    features = gate.shape[-1]
    out = torch.empty(gate.shape, dtype=gate.dtype, device=gate.device)
    block = min(1024, triton.next_power_of_2(features))
    layout, q, sf = activation if activation is not None else (None, None, None)
    _silu_mul_rows[(rows_gate.shape[0], triton.cdiv(features, block))](
        rows_gate, rows_up, out, rows_gate.stride(0), rows_up.stride(0), features, block,
        q, sf, 0 if layout is None else layout.token_tile, num_warps=4)
    if activation is not None:
        out._verify_w1 = activation
    return out
