"""Lazy GDN verify state: keep the verify inputs, then write only the accepted recurrent state at commit."""
import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # The CPU suite runs without Triton.
    triton = None

BV = 8  # the value tile of models/hybrid_snapshot.py snapshot_states


if triton is not None:
    @triton.jit
    def _verify_inputs(q_ptr, k_ptr, v_ptr, g_ptr, beta_ptr,
                       q_out, k_out, v_out, k_keep, v_keep, g_keep, beta_keep,
                       q_t, q_h, q_d, k_t, k_h, k_d, v_t, v_h, v_d, g_t, g_h, b_t, b_h,
                       KEY_HEADS: tl.constexpr, VALUE_HEADS: tl.constexpr,
                       K: tl.constexpr, V: tl.constexpr, BK: tl.constexpr, BVALUE: tl.constexpr,
                       SPLIT: tl.constexpr, KEEP: tl.constexpr):
        # One program per (token, value head). Pure data movement: every element keeps its bits.
        step = tl.program_id(0)
        head = tl.program_id(1)
        vv = tl.arange(0, BVALUE)
        value = tl.load(v_ptr + step * v_t + head * v_h + vv * v_d, mask=vv < V)
        row = step * VALUE_HEADS + head
        if SPLIT:
            tl.store(v_out + row * V + vv, value, mask=vv < V)
        if KEEP:
            tl.store(v_keep + row * V + vv, value.to(tl.float32), mask=vv < V)
            tl.store(g_keep + row, tl.load(g_ptr + step * g_t + head * g_h).to(tl.float32))
            tl.store(beta_keep + row, tl.load(beta_ptr + step * b_t + head * b_h).to(tl.float32))
        group = VALUE_HEADS // KEY_HEADS
        if head % group == 0:
            key_head = head // group
            kk = tl.arange(0, BK)
            key_row = step * KEY_HEADS + key_head
            key = tl.load(k_ptr + step * k_t + key_head * k_h + kk * k_d, mask=kk < K)
            if SPLIT:
                tl.store(k_out + key_row * K + kk, key, mask=kk < K)
                query = tl.load(q_ptr + step * q_t + key_head * q_h + kk * q_d, mask=kk < K)
                tl.store(q_out + key_row * K + kk, query, mask=kk < K)
            if KEEP:
                tl.store(k_keep + key_row * K + kk, key.to(tl.float32), mask=kk < K)

    @triton.jit
    def _commit_state(k_ptr, v_ptr, g_ptr, beta_ptr, count_ptr, initial_ptr, state_ptr,
                      initial_h: tl.constexpr, initial_k: tl.constexpr, initial_v: tl.constexpr,
                      state_h: tl.constexpr, state_k: tl.constexpr, state_v: tl.constexpr,
                      ROWS: tl.constexpr, KEY_HEADS: tl.constexpr, VALUE_HEADS: tl.constexpr,
                      K: tl.constexpr, V: tl.constexpr, BK: tl.constexpr, BVALUE: tl.constexpr):
        # The step body is kernels/hybrid_snapshot.py _snapshot_states, unchanged. The loop runs
        # the device count of steps, and only the final state is stored.
        head = tl.program_id(0)
        v_tile = tl.program_id(1)
        kk = tl.arange(0, BK)
        vv = v_tile * BVALUE + tl.arange(0, BVALUE)
        state = tl.load(
            initial_ptr + head * initial_h + kk[:, None] * initial_k + vv[None, :] * initial_v,
            mask=(kk[:, None] < K) & (vv[None, :] < V), other=0,
        ).to(tl.float32)
        count = tl.minimum(tl.maximum(tl.load(count_ptr), 0), ROWS)
        key_head = head // (VALUE_HEADS // KEY_HEADS)
        for step in range(count):
            key = tl.load(k_ptr + (step * KEY_HEADS + key_head) * K + kk,
                          mask=kk < K, other=0).to(tl.float32)
            key = key * tl.rsqrt(tl.sum(key * key, 0) + 1e-6)
            value = tl.load(v_ptr + (step * VALUE_HEADS + head) * V + vv,
                            mask=vv < V, other=0).to(tl.float32)
            gate = tl.load(g_ptr + step * VALUE_HEADS + head).to(tl.float32)
            beta = tl.load(beta_ptr + step * VALUE_HEADS + head).to(tl.float32)
            state = state * tl.exp(gate)
            delta = (value - tl.sum(state * key[:, None], 0)) * beta
            state = state + key[:, None] * delta[None, :]
        tl.store(
            state_ptr + head * state_h + kk[:, None] * state_k + vv[None, :] * state_v,
            state, mask=(kk[:, None] < K) & (vv[None, :] < V),
        )


def keep_buffers(rows, key_heads, value_heads, key_dim, value_dim, *, device=None):
    """The fp32 verify inputs of one GDN layer: keys per key head, values, gates and betas."""
    return (torch.zeros((rows, key_heads, key_dim), dtype=torch.float32, device=device),
            torch.zeros((rows, value_heads, value_dim), dtype=torch.float32, device=device),
            torch.zeros((rows, value_heads), dtype=torch.float32, device=device),
            torch.zeros((rows, value_heads), dtype=torch.float32, device=device))


def verify_inputs(q, k, v, g, beta, *, keep=None, split=False):
    """Copy the GDN verify inputs in one launch.

    ``q`` and ``k`` hold the key heads, (1, T, H, K); ``v`` is (1, T, HV, V); ``g`` and ``beta``
    are (1, T, HV). Any strides are allowed. With ``keep``, write k, v, g and beta into those
    fp32 buffers (exact for every source dtype). With ``split``, return contiguous q, k and v in
    their own dtype, the layout the fla GVA kernel reads without a copy.
    """
    _, width, key_heads, key_dim = k.shape
    value_heads, value_dim = v.shape[2:]
    if v.shape[:2] != (1, width) or g.shape != (1, width, value_heads) or beta.shape != g.shape:
        raise ValueError('GDN verify input geometry mismatch')
    if value_heads % key_heads or (split and q.shape != k.shape):
        raise ValueError('GDN verify head grouping mismatch')
    outputs = (torch.empty((1, width, key_heads, key_dim), dtype=q.dtype, device=q.device),
               torch.empty((1, width, key_heads, key_dim), dtype=k.dtype, device=k.device),
               torch.empty((1, width, value_heads, value_dim), dtype=v.dtype, device=v.device)
               ) if split else (None, None, None)
    if keep is not None:
        k_keep, v_keep, g_keep, beta_keep = keep
        if (k_keep.shape[0] < width or k_keep.shape[1:] != (key_heads, key_dim)
                or v_keep.shape[1:] != (value_heads, value_dim)
                or any(t.dtype != torch.float32 or not t.is_contiguous() for t in keep)):
            raise ValueError('GDN lazy keep buffers do not fit the verify inputs')
    else:
        k_keep = v_keep = g_keep = beta_keep = None
    query = q if split else k
    _verify_inputs[(width, value_heads)](
        query, k, v, g, beta, *outputs, k_keep, v_keep, g_keep, beta_keep,
        *query.stride()[1:], *k.stride()[1:], *v.stride()[1:], *g.stride()[1:], *beta.stride()[1:],
        key_heads, value_heads, key_dim, value_dim,
        triton.next_power_of_2(key_dim), triton.next_power_of_2(value_dim),
        bool(split), keep is not None,
    )
    return outputs if split else None


def commit_state(keep, count, initial, out=None):
    """Write the GDN recurrent state after ``count`` kept steps; ``out`` defaults to ``initial``.

    ``count`` is a one-element int64 device tensor, read on the device. ``initial`` is
    (1, HV, K, V) fp32. The update in place is safe: each program reads its state tile before
    it writes the same tile.
    """
    k_keep, v_keep, g_keep, beta_keep = keep
    rows, key_heads, key_dim = k_keep.shape
    value_heads, value_dim = v_keep.shape[1:]
    out = initial if out is None else out
    if (initial.shape != (1, value_heads, key_dim, value_dim) or out.shape != initial.shape
            or initial.dtype != torch.float32 or out.dtype != torch.float32):
        raise ValueError('GDN lazy commit state geometry mismatch')
    if count.numel() != 1 or count.dtype != torch.int64:
        raise ValueError('GDN lazy commit needs one int64 count')
    _commit_state[(value_heads, triton.cdiv(value_dim, BV))](
        k_keep, v_keep, g_keep, beta_keep, count, initial, out,
        *initial.stride()[1:], *out.stride()[1:],
        rows, key_heads, value_heads, key_dim, value_dim, triton.next_power_of_2(key_dim), BV,
    )
