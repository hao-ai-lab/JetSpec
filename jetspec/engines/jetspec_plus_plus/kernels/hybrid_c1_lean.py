"""Default-off C1 glue for offset RMSNorm, gated Q and partial RoPE.

The fp32 path uses module arithmetic. BF16 retains PyTorch's variance reduction
order; fusing that reduction changes rounding at trained activation boundaries.
The elementwise kernels retain the intermediate
roundings of eager RoPE and sigmoid/multiply; cache writes broadcast KV heads
directly into the existing expanded cache without a repeated temporary.
"""

from jetspec.engines.jetspec_plus_plus.profiles import kernel_enabled, kernel_value
import os

import torch

try:
    import triton
    import triton.language as tl
    from triton.language.extra.cuda import libdevice
except ImportError:
    triton = tl = None


if triton is not None:
    @triton.jit
    def _offset_norm(X, W, Stats, Y, stride: tl.constexpr, width: tl.constexpr,
                     eps: tl.constexpr, B: tl.constexpr):
        row = tl.program_id(0)
        col = tl.arange(0, B)
        x = tl.load(X + row * stride + col, col < width, 0).to(tl.float32)
        w = tl.load(W + col, col < width, 0).to(tl.float32)
        r = libdevice.rsqrt(tl.load(Stats + row) + eps)
        tl.store(Y + row * width + col, (x * r) * (1.0 + w), col < width)

    @triton.jit
    def _head_norm_rope(X, W, Stats, C, S, Y, V, VC, Slots,
                        tokens: tl.constexpr, heads: tl.constexpr, dim: tl.constexpr,
                        rotary: tl.constexpr, xs_token: tl.constexpr, xs_head: tl.constexpr,
                        vs_token: tl.constexpr,
                        eps: tl.constexpr, CACHE: tl.constexpr, groups: tl.constexpr,
                        capacity: tl.constexpr, B: tl.constexpr):
        token, head = tl.program_id(0), tl.program_id(1)
        col = tl.arange(0, B)
        x = tl.load(X + token * xs_token + head * xs_head + col, col < dim, 0).to(tl.float32)
        w = tl.load(W + col, col < dim, 0).to(tl.float32)
        r = libdevice.rsqrt(tl.load(Stats + token * heads + head) + eps)
        dtype: tl.constexpr = X.dtype.element_ty
        normed = ((x * r) * (1.0 + w)).to(dtype).to(tl.float32)
        other = tl.where(col < rotary // 2, col + rotary // 2, col - rotary // 2)
        ox = tl.load(X + token * xs_token + head * xs_head + other,
                     col < rotary, 0).to(tl.float32)
        ow = tl.load(W + other, col < rotary, 0).to(tl.float32)
        on = ((ox * r) * (1.0 + ow)).to(dtype).to(tl.float32)
        on = tl.where(col < rotary // 2, -on, on)
        c = tl.load(C + token * rotary + col, col < rotary, 0).to(tl.float32)
        s = tl.load(S + token * rotary + col, col < rotary, 0).to(tl.float32)
        a = (normed * c).to(dtype).to(tl.float32)
        b = (on * s).to(dtype).to(tl.float32)
        result = tl.where(col < rotary, a + b, normed)
        if CACHE:
            slot = tl.load(Slots + token)
            value = tl.load(V + token * vs_token + head * dim + col, col < dim, 0)
            for group in range(groups):
                offset = ((head * groups + group) * capacity + slot) * dim + col
                tl.store(Y + offset, result, col < dim)
                tl.store(VC + offset, value, col < dim)
        else:
            tl.store(Y + (head * tokens + token) * dim + col, result, col < dim)

    @triton.jit
    def _sigmoid_mul(A, G, Y, count: tl.constexpr, heads: tl.constexpr,
                     dim: tl.constexpr, B: tl.constexpr):
        i = tl.program_id(0) * B + tl.arange(0, B)
        token = i // (heads * dim)
        head = (i // dim) % heads
        col = i % dim
        g = tl.load(G + token * heads * dim * 2 + head * dim * 2 + col,
                    i < count, 0).to(tl.float32)
        a = tl.load(A + i, i < count, 0).to(tl.float32)
        sigmoid = (1.0 / (1.0 + libdevice.exp(-g))).to(G.dtype.element_ty).to(tl.float32)
        tl.store(Y + i, a * sigmoid, i < count)


def _cuda(x):
    return triton is not None and x.device.type == 'cuda' and x.dtype == torch.bfloat16


def norm(x, module):
    if not _cuda(x):
        return module(x)
    out = torch.empty_like(x, memory_format=torch.contiguous_format)
    width = x.shape[-1]
    _offset_norm[(x.numel() // width,)](
        x, module.weight, x.float().square().mean(-1), out,
        x.stride(-2), width, module.variance_epsilon,
        triton.next_power_of_2(width), num_warps=8, enable_fp_fusion=False)
    return out


def norm_rotary(x, module, cos, sin):
    from jetspec.engines.jetspec_plus_plus.profiles import kernel_enabled
    from jetspec.engines.jetspec_plus_plus.jetspec.drafter import _c1_rotary
    if not _cuda(x):
        return _c1_rotary(module(x).transpose(1, 2), cos, sin)
    if kernel_enabled('hybrid_c1_gluon', '1') and cos.shape[-1] == 64:
        from jetspec.engines.jetspec_plus_plus.kernels import gluon_block
        if gluon_block.available(x):
            return gluon_block.norm_rotary(x, module, cos, sin)
    _, tokens, heads, dim = x.shape
    out = x.new_empty(1, heads, tokens, dim)
    _head_norm_rope[(tokens, heads)](
        x, module.weight, x.float().square().mean(-1),
        cos.contiguous(), sin.contiguous(), out, x, out, x,
        tokens, heads, dim, cos.shape[-1], x.stride(1), x.stride(2),
        0, module.variance_epsilon, False, 1, 0, triton.next_power_of_2(dim),
        num_warps=4, enable_fp_fusion=False)
    return out


def write_kv(key, value, module, cos, sin, cache, slots, groups):
    from jetspec.engines.jetspec_plus_plus.profiles import kernel_enabled
    from jetspec.engines.jetspec_plus_plus.jetspec.drafter import _c1_rotary
    k, v = cache
    _, tokens, heads, dim = key.shape
    if not _cuda(key):
        rotated = _c1_rotary(module(key).transpose(1, 2), cos, sin)
        k.index_copy_(2, slots, rotated.repeat_interleave(groups, dim=1))
        v.index_copy_(2, slots, value.view(1, tokens, heads, dim).transpose(1, 2)
                      .repeat_interleave(groups, dim=1))
        return
    if kernel_enabled('hybrid_c1_gluon', '1') and cos.shape[-1] == 64:
        from jetspec.engines.jetspec_plus_plus.kernels import gluon_block
        if gluon_block.available(key):
            return gluon_block.write_kv(key, value, module, cos, sin, cache, slots, groups)
    _head_norm_rope[(tokens, heads)](
        key, module.weight, key.float().square().mean(-1),
        cos.contiguous(), sin.contiguous(), k, value, v, slots,
        tokens, heads, dim, cos.shape[-1], key.stride(1), key.stride(2),
        value.stride(-2), module.variance_epsilon, True, groups, k.shape[2], triton.next_power_of_2(dim),
        num_warps=4, enable_fp_fusion=False)


def gate_mul(attended, gate, heads, dim):
    if not _cuda(attended):
        return attended * torch.sigmoid(gate.reshape_as(attended))
    # The Q projection interleaves query and gate per head. Preserve that view.
    out = torch.empty_like(attended)
    _sigmoid_mul[(triton.cdiv(attended.numel(), 256),)](
        attended, gate, out, attended.numel(), heads, dim, 256,
        enable_fp_fusion=False)
    return out


def mlp(x, module):
    if not _cuda(x):
        return module(x)
    from jetspec.engines.jetspec_plus_plus.jetspec.c1_fused import _silu_mul_extension
    return module.down_proj(_silu_mul_extension().silu_mul(
        module.gate_proj(x), module.up_proj(x)))
