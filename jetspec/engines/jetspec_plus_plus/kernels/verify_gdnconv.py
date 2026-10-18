"""27B GDN verify conv: causal depthwise conv1d, SiLU, q/k/v split, lazy keep and conv snapshots in one launch.

The eager chain (with verify_cut_convsnap, gdnlazy and gdnrepeat) is the cat of the committed
conv window and the in_proj_qkv rows, the cuDNN depthwise conv1d, the SiLU, the verify_inputs
split and keep launch and the conv snapshot copy. Here one program per channel block computes
the W verify tokens of its channels from the committed window and the raw rows. The conv sums
its K products in FP32 in tap order and rounds to BF16 once, as the conv output; the SiLU is
ATen's x / (1 + exp(-x)) with one BF16 rounding. cuDNN may sum in another order, so the result
is near the BF16 rounding error, not bitwise (class B). The split, keep and snapshot values
are copies, bitwise to the eager ones for the same conv output.
"""
import torch

try:
    import triton
    import triton.language as tl
    from triton.language.extra.cuda import libdevice
except ImportError:  # The CPU suite runs without Triton.
    triton = None

BLOCK_CHANNELS = 32
WARPS = 2


if triton is not None:
    from jetspec.engines.jetspec_plus_plus.kernels.verify_mlp_glue import _ratio

    @triton.jit
    def _window(RAW, COMMITTED, channels, live_c, index, live, r_c, r_t, KS: tl.constexpr):
        # Element ``index`` of cat(committed_conv, raw) along time, per channel.
        from_committed = (index < KS)[None, :]
        old = tl.load(COMMITTED + channels[:, None] * KS + index[None, :],
                      live_c[:, None] & live & from_committed, 0.)
        new = tl.load(RAW + channels[:, None] * r_c + (index[None, :] - KS) * r_t,
                      live_c[:, None] & live & (index[None, :] >= KS), 0.)
        return tl.where(from_committed, old, new)

    @triton.jit
    def _conv_verify(RAW, COMMITTED, WEIGHT, G, BETA, QO, KO, VO, KKEEP, VKEEP, GKEEP, BKEEP, SCRATCH,
                     r_c, r_t, g_t, g_h, b_t, b_h, s_step, C, W: tl.constexpr, WP: tl.constexpr,
                     KS: tl.constexpr, KW: tl.constexpr, VW: tl.constexpr, HV: tl.constexpr,
                     HVP: tl.constexpr, BC: tl.constexpr):
        block = tl.program_id(0)
        channels = block * BC + tl.arange(0, BC)
        live_c = channels < C
        tokens = tl.arange(0, WP)
        live_t = tokens < W
        acc = tl.zeros([BC, WP], tl.float32)
        for tap in tl.static_range(KS):
            # Token t's window is cat[t + 1 .. t + KS]; tap j reads cat[t + 1 + j].
            value = _window(RAW, COMMITTED, channels, live_c, tokens + 1 + tap, live_t[None, :], r_c, r_t, KS)
            weight = tl.load(WEIGHT + channels * KS + tap, live_c, 0.).to(tl.float32)
            acc += weight[:, None] * value.to(tl.float32)
        x = acc.to(tl.bfloat16).to(tl.float32)
        y = _ratio(x, libdevice.exp(-x)).to(tl.bfloat16)
        live = live_c[:, None] & live_t[None, :]
        t64 = tokens.to(tl.int64)[None, :]
        c2 = channels[:, None]
        is_q = c2 < KW
        is_k = (c2 >= KW) & (c2 < 2 * KW)
        is_v = c2 >= 2 * KW
        tl.store(QO + t64 * KW + c2, y, live & is_q)
        tl.store(KO + t64 * KW + (c2 - KW), y, live & is_k)
        tl.store(KKEEP + t64 * KW + (c2 - KW), y.to(tl.float32), live & is_k)
        tl.store(VO + t64 * VW + (c2 - 2 * KW), y, live & is_v)
        tl.store(VKEEP + t64 * VW + (c2 - 2 * KW), y.to(tl.float32), live & is_v)
        # Conv snapshots: scratch[j] is the window cat[j .. j + KS - 1], j = 0 .. W.
        taps = tl.arange(0, KS)
        for step in tl.static_range(W + 1):
            window = _window(RAW, COMMITTED, channels, live_c, step + taps, taps[None, :] < KS, r_c, r_t, KS)
            tl.store(SCRATCH + step * s_step + channels[:, None] * KS + taps[None, :], window,
                     live_c[:, None] & (taps[None, :] < KS))
        if block == 0:
            # The lazy keep of the gates and betas, fp32 as verify_inputs writes them.
            heads = tl.arange(0, HVP)
            live_h = live_t[:, None] & (heads[None, :] < HV)
            gate = tl.load(G + tokens[:, None] * g_t + heads[None, :] * g_h, live_h, 0.)
            beta = tl.load(BETA + tokens[:, None] * b_t + heads[None, :] * b_h, live_h, 0.)
            tl.store(GKEEP + tokens[:, None] * HV + heads[None, :], gate.to(tl.float32), live_h)
            tl.store(BKEEP + tokens[:, None] * HV + heads[None, :], beta.to(tl.float32), live_h)


def supported(raw, committed_conv, weight, scratch_conv):
    """True when the fused conv can serve this C=1 verify call (CUDA BF16, one slot, dense state)."""
    return (triton is not None and raw.is_cuda and raw.dtype == torch.bfloat16
            and committed_conv.dtype == torch.bfloat16 and scratch_conv.dtype == torch.bfloat16
            and weight.dtype == torch.bfloat16 and raw.dim() == 3 and raw.shape[0] == 1
            and committed_conv.shape[0] == 1 and committed_conv.is_contiguous()
            and weight.is_contiguous() and scratch_conv.is_contiguous() and scratch_conv.shape[1] == 1
            and 0 < raw.shape[2] < scratch_conv.shape[0])


def conv_verify(raw, committed_conv, weight, g, beta, keep, scratch_conv, key_heads, value_heads,
                key_dim, value_dim):
    """(q, k, v) for the GDN GVA recurrent kernel; the lazy keep and conv snapshots are written.

    ``raw`` is the [1, C, W] transposed in_proj_qkv output (any strides); ``committed_conv`` is
    [1, C, KS]; ``weight`` is the [C, 1, KS] depthwise conv weight; ``g`` and ``beta`` are
    [1, W, HV]; ``keep`` is gdn_lazy's (k, v, g, beta) fp32 buffers; ``scratch_conv`` is
    [rows, 1, C, KS]. q and k are [1, W, key_heads, key_dim], v is [1, W, value_heads, value_dim].
    """
    _, channels, width = raw.shape
    kernel = committed_conv.shape[-1]
    key_width, value_width = key_heads * key_dim, value_heads * value_dim
    k_keep, v_keep, g_keep, beta_keep = keep
    if (channels != 2 * key_width + value_width or committed_conv.shape != (1, channels, kernel)
            or weight.shape != (channels, 1, kernel) or g.shape != (1, width, value_heads)
            or beta.shape != g.shape or scratch_conv.shape[2:] != (channels, kernel)
            or k_keep.shape[0] < width or k_keep.shape[1:] != (key_heads, key_dim)
            or v_keep.shape[1:] != (value_heads, value_dim)
            or any(t.dtype != torch.float32 or not t.is_contiguous() for t in keep)):
        raise ValueError('verify_gdnconv geometry mismatch')
    q = raw.new_empty(1, width, key_heads, key_dim)
    k = raw.new_empty(1, width, key_heads, key_dim)
    v = raw.new_empty(1, width, value_heads, value_dim)
    _conv_verify[(triton.cdiv(channels, BLOCK_CHANNELS),)](
        raw, committed_conv, weight, g, beta, q, k, v, k_keep, v_keep, g_keep, beta_keep, scratch_conv,
        raw.stride(1), raw.stride(2), g.stride(1), g.stride(2), beta.stride(1), beta.stride(2),
        scratch_conv.stride(0), channels, W=width, WP=max(2, triton.next_power_of_2(width)), KS=kernel,
        KW=key_width, VW=value_width, HV=value_heads, HVP=triton.next_power_of_2(value_heads),
        BC=BLOCK_CHANNELS, num_warps=WARPS)
    return q, k, v
