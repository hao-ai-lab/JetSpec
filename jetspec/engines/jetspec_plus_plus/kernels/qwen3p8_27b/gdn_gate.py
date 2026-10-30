"""GDN gate g and write strength beta in one launch, at the eager rounding points."""
import torch
from torch.nn import functional as F

try:
    import triton
    import triton.language as tl
except ImportError:  # The CPU suite runs without Triton.
    triton = None
libdevice = None
if triton is not None:
    try:
        from triton.language.extra import libdevice
    except ImportError:
        try:
            from triton.language.extra.cuda import libdevice
        except ImportError:
            libdevice = None

_DTYPES = (torch.bfloat16, torch.float32)
BLOCK = 1024
FLT_MIN = 2.0 ** -126  # the smallest normal fp32 value


def gate_beta_reference(a, b, dt_bias, decay):
    """The eager chain with ``decay = -exp(A_log.float())`` given: g in fp32, beta in b's dtype."""
    beta = b.sigmoid()
    g = decay * F.softplus(a.float() + dt_bias)
    return g, beta


if triton is not None and libdevice is not None:
    FLT_MIN = tl.constexpr(FLT_MIN)

    @triton.jit
    def _gate_beta(a_ptr, b_ptr, dt_ptr, decay_ptr, g_ptr, beta_ptr, n,
                   H: tl.constexpr, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        head = offs % H
        # aten softplus (beta 1, threshold 20): x > 20 ? x : log1p(exp(x)), all in fp32.
        x = (tl.load(a_ptr + offs, mask=mask, other=0).to(tl.float32)
             + tl.load(dt_ptr + head, mask=mask, other=0).to(tl.float32))
        e = libdevice.exp(x)
        # Triton links libdevice with flush-to-zero: its log1p reads a subnormal exp(x) as 0.
        # aten's log1pf returns a subnormal input unchanged, so pass it through.
        soft = tl.where(x > 20.0, x, tl.where(e < FLT_MIN, e, libdevice.log1p(e)))
        g = tl.load(decay_ptr + head, mask=mask, other=0) * soft
        tl.store(g_ptr + offs, g, mask=mask)
        # aten sigmoid: 1 / (1 + exp(-b)) in fp32, one rounding to the output dtype.
        # rcp.rn.f32 is the correctly rounded reciprocal without flush-to-zero (div_rn flushes).
        b = tl.load(b_ptr + offs, mask=mask, other=0).to(tl.float32)
        beta = tl.inline_asm_elementwise('rcp.rn.f32 $0, $1;', '=f,f', [1.0 + libdevice.exp(-b)],
                                         dtype=tl.float32, is_pure=True, pack=1)
        tl.store(beta_ptr + offs, beta.to(beta_ptr.dtype.element_ty), mask=mask)


def fused_supported(a, b, dt_bias, decay):
    """True when the one-launch kernel serves these inputs; the caller keeps the eager chain otherwise."""
    if triton is None or libdevice is None or not a.is_cuda:
        return False
    heads = a.shape[-1] if a.dim() else 0
    if (a.shape != b.shape or heads == 0 or dt_bias.shape != (heads,) or decay.shape != (heads,)
            or a.dtype not in _DTYPES or b.dtype not in _DTYPES or dt_bias.dtype not in _DTYPES
            or decay.dtype != torch.float32):
        return False
    if any(t.device != a.device or not t.is_contiguous() for t in (a, b, dt_bias, decay)):
        return False
    if torch.is_grad_enabled() and any(t.requires_grad for t in (a, b, dt_bias, decay)):
        return False
    return a.numel() < 2 ** 31


def gate_beta(a, b, dt_bias, decay):
    """Return ``(g, beta)`` bit-identical to ``gate_beta_reference``; one launch on CUDA."""
    if not fused_supported(a, b, dt_bias, decay):
        return gate_beta_reference(a, b, dt_bias, decay)
    g = torch.empty(a.shape, dtype=torch.float32, device=a.device)
    beta = torch.empty_like(b)
    n = a.numel()
    _gate_beta[(triton.cdiv(n, BLOCK),)](
        a, b, dt_bias, decay, g, beta, n, H=a.shape[-1], BLOCK=BLOCK,
        num_warps=4, enable_fp_fusion=False)
    return g, beta
