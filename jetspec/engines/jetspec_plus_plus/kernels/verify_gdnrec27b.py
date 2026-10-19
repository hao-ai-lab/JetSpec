"""27B verify gated delta recurrence with value tiles and warp-local key reductions.

Each program owns a value-head tile and runs the verify tokens in order. Its FP32
state stays in registers; each warp owns complete key reductions for its value
columns. The 16-column tiles expose eight programs per head at the 27B geometry.
Query/key normalization and the recurrent update use FP32, with one BF16 rounding
at the output. The reduction order can differ from fla (class B). The committed
state is read only; optional final/per-token stores exercise the same recurrence.
"""
import torch

try:
    import triton
    import triton.language as tl
    from triton.experimental import gluon
    from triton.experimental.gluon import language as gl
except ImportError:  # The CPU suite runs without Triton.
    triton = None

BLOCK_V = 16
WARPS = 4
MAX_WIDTH = 16


if triton is not None:
    @gluon.jit
    def _serial_verify(Q, K, V, G, BETA, H0, O, HT, HS,
                       q_t, q_h, k_t, k_h, v_t, v_h, g_t, g_h, b_t, b_h, o_t, o_h,
                       W: gl.constexpr, HK: gl.constexpr, HV: gl.constexpr,
                       DK: gl.constexpr, DV: gl.constexpr, BV: gl.constexpr, NW: gl.constexpr,
                       FINAL: gl.constexpr, STATES: gl.constexpr):
        # Every warp owns value columns and all K lanes. Both state reductions stay in a warp.
        layout: gl.constexpr = gl.BlockedLayout([1, BV // NW], [32, 1], [1, NW], [1, 0])
        d = gl.arange(0, DK, layout=gl.SliceLayout(1, layout))
        c = gl.program_id(0) * BV + gl.arange(0, BV, layout=gl.SliceLayout(0, layout))
        head = gl.program_id(1)
        key_head = head // (HV // HK)
        offset = head * DK * DV + d[:, None] * DV + c[None, :]
        state = gl.load(H0 + offset)
        for step in tl.static_range(W):
            q = gl.load(Q + step * q_t + key_head * q_h + d).to(gl.float32)
            k = gl.load(K + step * k_t + key_head * k_h + d).to(gl.float32)
            q = q * gl.rsqrt(gl.sum(q * q, 0) + 1e-6) * (DK ** -0.5)
            k = k * gl.rsqrt(gl.sum(k * k, 0) + 1e-6)
            value = gl.load(V + step * v_t + head * v_h + c).to(gl.float32)
            gate = gl.load(G + step * g_t + head * g_h).to(gl.float32)
            beta = gl.load(BETA + step * b_t + head * b_h).to(gl.float32)
            state = state * gl.exp(gate)
            delta = (value - gl.sum(state * k[:, None], 0)) * beta
            state = state + k[:, None] * delta[None, :]
            output = gl.sum(state * q[:, None], 0)
            gl.store(O + step * o_t + head * o_h + c, output)
            if STATES:
                gl.store(HS + step * HV * DK * DV + offset, state)
        if FINAL:
            gl.store(HT + offset, state)


def supported(q, k, v, g, beta, initial):
    """True for a C=1 CUDA verify call with BF16 queries/keys and a dense FP32 state."""
    if triton is None or not q.is_cuda or q.dim() != 4 or v.dim() != 4:
        return False
    _, width, key_heads, key_dim = k.shape
    value_heads, value_dim = v.shape[2:]
    return (q.shape == k.shape and q.shape[0] == 1 and v.shape[:2] == (1, width)
            and 0 < width <= MAX_WIDTH and key_heads and value_heads % key_heads == 0
            and g.shape == (1, width, value_heads) and beta.shape == g.shape
            and initial.shape == (1, value_heads, key_dim, value_dim) and initial.dtype == torch.float32
            and initial.is_contiguous() and key_dim in (64, 128) and value_dim % BLOCK_V == 0
            and q.stride(3) == 1 and k.stride(3) == 1 and v.stride(3) == 1
            and q.dtype == torch.bfloat16 and k.dtype == torch.bfloat16)


def recurrent_verify(q, k, v, g, beta, initial, *, final=False, states=False,
                     block_v=BLOCK_V, warps=WARPS):
    """Return BF16 output, optional FP32 final state, and optional FP32 per-token states.

    Inputs use fla's [1, W, head, dim] layout (G/BETA omit dim); initial is
    [1, value_heads, key_dim, value_dim]. Per-token states are [W, value_heads,
    key_dim, value_dim]. The input state is never written. Tile/warp overrides
    serve the focused timing sweep.
    """
    if not supported(q, k, v, g, beta, initial) or v.shape[-1] % block_v:
        raise ValueError('verify_gdnrec27b geometry mismatch')
    _, width, key_heads, key_dim = k.shape
    value_heads, value_dim = v.shape[2:]
    out = q.new_empty(1, width, value_heads, value_dim)
    last = torch.empty_like(initial) if final else None
    every = initial.new_empty(width, value_heads, key_dim, value_dim) if states else None
    _serial_verify[(value_dim // block_v, value_heads)](
        q, k, v, g, beta, initial, out, last, every,
        q.stride(1), q.stride(2), k.stride(1), k.stride(2), v.stride(1), v.stride(2),
        g.stride(1), g.stride(2), beta.stride(1), beta.stride(2), out.stride(1), out.stride(2),
        W=width, HK=key_heads, HV=value_heads, DK=key_dim, DV=value_dim, BV=block_v, NW=warps,
        FINAL=final, STATES=states, num_warps=warps, enable_fp_fusion=False)
    return out, last, every
