"""Opt-in dense 8B head RMSNorm, full RoPE and direct KV writes.

Ports serving's Gluon head kernel, with ordinary norm's two BF16 roundings
and the 128-wide Torch reduction used by the dense adapter.
"""
import torch

try:
    from triton.experimental import gluon
    from triton.experimental.gluon import language as gl
    from triton.experimental.gluon.language.extra import libdevice
except ImportError:
    gluon = None


if gluon is not None:
    @gluon.jit
    def _head_rope(X, W, C, S, Y, V, VC, Start,
                   TOKENS: gl.constexpr, HEADS: gl.constexpr,
                   XT: gl.constexpr, VT: gl.constexpr,
                   YT: gl.constexpr, YH: gl.constexpr,
                   EPS: gl.constexpr, CACHE: gl.constexpr,
                   GROUPS: gl.constexpr, HAS_START: gl.constexpr):
        token, head = gl.program_id(0), gl.program_id(1)
        layout: gl.constexpr = gl.BlockedLayout([1], [32], [1], [0])
        lane = gl.arange(0, 32, layout=layout)
        partial = gl.full((32,), 0, gl.float32, layout=layout)
        for item in gl.static_range(4):
            value = gl.load(X + token * XT + head * 128 + lane + item * 32).to(gl.float32)
            partial = partial + value * value
        for step in gl.static_range(5):
            other = gl.inline_asm_elementwise(
                "shfl.sync.down.b32 $0, $1, $2, 31, -1;",
                constraints="=f,f,r", args=[partial, 1 << step],
                dtype=gl.float32, is_pure=True, pack=1)
            partial = partial + other
        mean = gl.sum(gl.where(lane == 0, partial, 0), 0) / 128
        rms = libdevice.rsqrt(mean + EPS)
        columns: gl.constexpr = gl.BlockedLayout([4], [32], [1], [0])
        col = gl.arange(0, 128, layout=columns)
        paired = gl.where(col < 64, col + 64, col - 64)
        x = gl.load(X + token * XT + head * 128 + col).to(gl.float32)
        ox = gl.load(X + token * XT + head * 128 + paired).to(gl.float32)
        w = gl.load(W + col).to(gl.float32)
        ow = gl.load(W + paired).to(gl.float32)
        normed = ((x * rms).to(gl.bfloat16).to(gl.float32) * w).to(gl.bfloat16).to(gl.float32)
        on = ((ox * rms).to(gl.bfloat16).to(gl.float32) * ow).to(gl.bfloat16).to(gl.float32)
        on = gl.where(col < 64, -on, on)
        c = gl.load(C + token * 128 + col).to(gl.float32)
        s = gl.load(S + token * 128 + col).to(gl.float32)
        a = (normed * c).to(gl.bfloat16).to(gl.float32)
        b = (on * s).to(gl.bfloat16).to(gl.float32)
        result = a + b
        if CACHE:
            slot = token
            if HAS_START:
                slot = slot + gl.load(Start)
            value = gl.load(V + token * VT + head * 128 + col)
            for group in gl.static_range(GROUPS):
                offset = slot * YT + (head * GROUPS + group) * YH + col
                gl.store(Y + offset, result)
                gl.store(VC + offset, value)
        else:
            gl.store(Y + token * YT + head * YH + col, result)


def available(x):
    return (gluon is not None and x.is_cuda and x.dtype == torch.bfloat16
            and x.ndim == 2 and x.stride(-1) == 1
            and torch.cuda.get_device_capability(x.device)[0] == 10)


def _reference(projected, weight, cos, sin, tokens, heads, eps):
    x = projected.view(tokens, heads, 128)
    f = x.float()
    x = weight * (f * torch.rsqrt(f.square().mean(-1, keepdim=True) + eps)).to(x.dtype)
    x = x.transpose(0, 1)
    cos, sin = cos.reshape(tokens, 128), sin.reshape(tokens, 128)
    return x * cos + torch.cat((-x[..., 64:], x[..., :64]), -1) * sin


def q_norm_rotary(projected, weight, cos, sin, out, tokens, heads, eps):
    """The lean loop's query-head API, preserving ordinary-norm roundings."""
    if not available(projected):
        out.copy_(_reference(projected, weight, cos, sin, tokens, heads, eps)[None])
        return
    _head_rope[(tokens, heads)](
        projected, weight, cos, sin, out, projected, out, projected,
        tokens, heads, projected.stride(0), 0, out.stride(2), out.stride(1), eps, False, 1, False,
        num_warps=1, enable_fp_fusion=False)


def k_norm_rotary_cache(key, value, weight, cos, sin, key_out, value_out,
                        tokens, heads, groups, eps, start=None):
    """Write strided [token, repeated-head, 128] cache views at a device start."""
    if not available(key):
        k = _reference(key, weight, cos, sin, tokens, heads, eps)
        v = value.view(tokens, heads, 128).transpose(0, 1)
        slots = torch.arange(tokens, device=key.device) + (0 if start is None else start[0])
        key_out.index_copy_(0, slots, k.repeat_interleave(groups, 0).transpose(0, 1))
        value_out.index_copy_(0, slots, v.repeat_interleave(groups, 0).transpose(0, 1))
        return
    if (key_out.stride() != value_out.stride() or key_out.stride(-1) != 1):
        raise ValueError('dense Gluon KV write requires matching cache layouts')
    _head_rope[(tokens, heads)](
        key, weight, cos, sin, key_out, value, value_out,
        key if start is None else start, tokens, heads, key.stride(0), value.stride(0),
        key_out.stride(0), key_out.stride(1), eps, True, groups, start is not None,
        num_warps=1, enable_fp_fusion=False)
