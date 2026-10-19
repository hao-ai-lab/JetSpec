"""Device kernels for models/hybrid_snapshot.py."""

# Adaptation: defer optional accelerator imports for CPU package inspection.
from jetspec.engines.jetspec_plus_plus._optional import triton
from jetspec.engines.jetspec_plus_plus._optional import tl


@triton.jit
def _snapshot_states(
    k_ptr, v_ptr, g_ptr, beta_ptr, initial_ptr, scratch_ptr,
    k_t: tl.constexpr, k_h: tl.constexpr, k_d: tl.constexpr,
    v_t: tl.constexpr, v_h: tl.constexpr, v_d: tl.constexpr,
    g_t: tl.constexpr, g_h: tl.constexpr,
    b_t: tl.constexpr, b_h: tl.constexpr,
    initial_h: tl.constexpr, initial_k: tl.constexpr, initial_v: tl.constexpr,
    scratch_t: tl.constexpr, scratch_h: tl.constexpr,
    scratch_k: tl.constexpr, scratch_v: tl.constexpr,
    T: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    BK: tl.constexpr, BV: tl.constexpr,
):
    head = tl.program_id(0)
    v_tile = tl.program_id(1)
    kk = tl.arange(0, BK)
    vv = v_tile * BV + tl.arange(0, BV)
    state = tl.load(
        initial_ptr + head * initial_h + kk[:, None] * initial_k + vv[None, :] * initial_v,
        mask=(kk[:, None] < K) & (vv[None, :] < V), other=0,
    ).to(tl.float32)
    for step in range(T):
        key = tl.load(k_ptr + step * k_t + head * k_h + kk * k_d,
                      mask=kk < K, other=0).to(tl.float32)
        key = key * tl.rsqrt(tl.sum(key * key, 0) + 1e-6)
        value = tl.load(v_ptr + step * v_t + head * v_h + vv * v_d,
                        mask=vv < V, other=0).to(tl.float32)
        gate = tl.load(g_ptr + step * g_t + head * g_h).to(tl.float32)
        beta = tl.load(beta_ptr + step * b_t + head * b_h).to(tl.float32)
        state = state * tl.exp(gate)
        delta = (value - tl.sum(state * key[:, None], 0)) * beta
        state = state + key[:, None] * delta[None, :]
        tl.store(
            scratch_ptr + (step + 1) * scratch_t + head * scratch_h
            + kk[:, None] * scratch_k + vv[None, :] * scratch_v,
            state, mask=(kk[:, None] < K) & (vv[None, :] < V),
        )

