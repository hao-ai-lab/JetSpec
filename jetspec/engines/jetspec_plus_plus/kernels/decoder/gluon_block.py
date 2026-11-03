"""Opt-in Gluon head normalization, partial RoPE and adapter KV writes."""
import torch

try:
    import triton
    from triton.experimental import gluon
    from triton.experimental.gluon import language as gl
    from triton.experimental.gluon.language.extra import libdevice
except ImportError:
    gluon = None


if gluon is not None:
    @gluon.jit
    def _head_rope(X, W, C, S, Y, V, VC, Slots, TOKENS: gl.constexpr,
                   DIM: gl.constexpr, ROTARY: gl.constexpr, XT: gl.constexpr,
                   XH: gl.constexpr, VT: gl.constexpr, EPS: gl.constexpr,
                   CACHE: gl.constexpr, GROUPS: gl.constexpr,
                   CAPACITY: gl.constexpr, BLOCK: gl.constexpr):
        token, head = gl.program_id(0), gl.program_id(1)
        col = gl.arange(0, BLOCK, layout=gl.BlockedLayout([1], [32], [4], [0]))
        x = gl.load(X + token * XT + head * XH + col, col < DIM, 0).to(gl.float32)
        w = gl.load(W + col, col < DIM, 0).to(gl.float32)
        if DIM == 256:
            # Match Torch's vectorized mean: four accumulators per lane,
            # combined in order, then ascending shuffle-down offsets.
            lane = gl.arange(0, 32, layout=gl.BlockedLayout([1], [32], [4], [0]))
            base = X + token * XT + head * XH + lane * 4
            partial = gl.full((32,), 0, gl.float32,
                              layout=gl.BlockedLayout([1], [32], [4], [0]))
            for item in gl.static_range(4):
                first = gl.load(base + item).to(gl.float32)
                second = gl.load(base + 128 + item).to(gl.float32)
                partial = partial + (first * first + second * second)
            for shuffle_step in gl.static_range(5):
                other_partial = gl.inline_asm_elementwise(
                    "shfl.sync.down.b32 $0, $1, $2, 31, -1;",
                    constraints="=f,f,r", args=[partial, 1 << shuffle_step],
                    dtype=gl.float32, is_pure=True, pack=1)
                partial = partial + other_partial
            mean = gl.sum(gl.where(lane == 0, partial, 0), 0) / DIM
        else:
            mean = gl.sum(x * x, 0) / DIM
        rms = libdevice.rsqrt(mean + EPS)
        normed = ((x * rms) * (1.0 + w)).to(gl.bfloat16).to(gl.float32)
        other = gl.where(col < ROTARY // 2, col + ROTARY // 2, col - ROTARY // 2)
        ox = gl.load(X + token * XT + head * XH + other, col < ROTARY, 0).to(gl.float32)
        ow = gl.load(W + other, col < ROTARY, 0).to(gl.float32)
        on = ((ox * rms) * (1.0 + ow)).to(gl.bfloat16).to(gl.float32)
        on = gl.where(col < ROTARY // 2, -on, on)
        c = gl.load(C + token * ROTARY + col, col < ROTARY, 0).to(gl.float32)
        s = gl.load(S + token * ROTARY + col, col < ROTARY, 0).to(gl.float32)
        a = (normed * c).to(gl.bfloat16).to(gl.float32)
        b = (on * s).to(gl.bfloat16).to(gl.float32)
        result = gl.where(col < ROTARY, a + b, normed)
        if CACHE:
            slot = gl.load(Slots + token)
            value = gl.load(V + token * VT + head * DIM + col, col < DIM, 0)
            for group in range(GROUPS):
                offset = ((head * GROUPS + group) * CAPACITY + slot) * DIM + col
                gl.store(Y + offset, result, col < DIM)
                gl.store(VC + offset, value, col < DIM)
        else:
            gl.store(Y + (head * TOKENS + token) * DIM + col, result, col < DIM)



def _blackwell(x):
    return (gluon is not None and x.is_cuda and x.dtype == torch.bfloat16
            and torch.cuda.get_device_capability(x.device)[0] == 10)


def available(x):
    return (_blackwell(x) and x.ndim == 4 and x.shape[0] == 1
            and x.shape[-1] == 256 and x.stride(-1) == 1)


def norm_rotary(x, module, cos, sin):
    """Head offset RMSNorm and partial RoPE without a statistics temporary."""
    if not _blackwell(x):
        from jetspec.engines.jetspec_plus_plus.kernels.hybrid_c1_lean import norm_rotary as reference
        return reference(x, module, cos, sin)
    _, tokens, heads, dim = x.shape
    out = x.new_empty(1, heads, tokens, dim)
    _head_rope[(tokens, heads)](x, module.weight, cos.contiguous(), sin.contiguous(),
        out, x, out, x, tokens, dim, cos.shape[-1], x.stride(1), x.stride(2),
        0, module.variance_epsilon, False, 1, 0, triton.next_power_of_2(dim),
        num_warps=4, enable_fp_fusion=False)
    return out


def write_kv(key, value, module, cos, sin, cache, slots, groups):
    """Fuse head norm, partial RoPE and repeated KV-head cache writes."""
    if not _blackwell(key):
        from jetspec.engines.jetspec_plus_plus.kernels.hybrid_c1_lean import write_kv as reference
        return reference(key, value, module, cos, sin, cache, slots, groups)
    k, v = cache
    _, tokens, heads, dim = key.shape
    _head_rope[(tokens, heads)](key, module.weight, cos.contiguous(), sin.contiguous(),
        k, value, v, slots, tokens, dim, cos.shape[-1], key.stride(1), key.stride(2),
        value.stride(-2), module.variance_epsilon, True, groups, k.shape[2],
        triton.next_power_of_2(dim), num_warps=4, enable_fp_fusion=False)
