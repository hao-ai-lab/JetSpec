"""Draft refresh gates/state kernels and measured normalization experiments.

Norm experiments are not wired into serving: local kernel ULP parity did not
preserve the required full-decoder output bound through repeated projections.
"""
import torch
from torch.nn import functional as F

try:
    import triton
    import triton.language as tl
    from triton.language.extra.cuda import libdevice
except ImportError:
    triton = tl = None


if triton is not None:
    @triton.jit
    def _stage_hidden(Tail, Block, Counts, Out, LANES: tl.constexpr, SLOTS: tl.constexpr,
                      TOKENS: tl.constexpr, WIDTH: tl.constexpr, TS: tl.constexpr,
                      BR: tl.constexpr, BS: tl.constexpr, TAP: tl.constexpr,
                      B: tl.constexpr):
        flat = tl.program_id(0) * B + tl.arange(0, B)
        token, column = flat // WIDTH, flat % WIDTH
        is_block = token >= LANES
        row, slot = tl.maximum(token - LANES, 0) // SLOTS, (token - LANES) % SLOTS
        count = tl.load(Counts + row, (token < TOKENS) & is_block, 0)
        a = tl.load(Tail + token * TS + column, (token < LANES), 0)
        b = tl.load(Block + row * BR + slot * BS + TAP + column,
                    (token < TOKENS) & is_block & (slot < count), 0)
        tl.store(Out + flat, tl.where(is_block, b, a), token < TOKENS)

    @triton.jit
    def _stage_conv(Raw, Committed, Tails, Out,
                    LANES: tl.constexpr, SLOTS: tl.constexpr, TOKENS: tl.constexpr,
                    CHANNELS: tl.constexpr, KERNEL: tl.constexpr,
                    RS: tl.constexpr, CS: tl.constexpr, CK: tl.constexpr,
                    B: tl.constexpr):
        flat = tl.program_id(0) * B + tl.arange(0, B)
        tap = flat % KERNEL
        channel = flat // KERNEL % CHANNELS
        token = flat // (KERNEL * CHANNELS)
        block = token >= LANES
        row = tl.maximum(token - LANES, 0) // SLOTS
        tail = tl.load(Tails + row, (token < TOKENS) & block, 0)
        position = tl.where(block, (token - LANES) % SLOTS, token) + tap + 1 - KERNEL
        own = block & (position >= 0)
        tail_position = position + tl.where(block, tail, 0)
        committed = ~own & (tail_position < 0)
        raw_index = tl.where(own, LANES + row * SLOTS + position, tail_position)
        a = tl.load(Committed + channel * CS + (KERNEL + tail_position) * CK,
                    (token < TOKENS) & committed, 0)
        b = tl.load(Raw + raw_index * RS + channel,
                    (token < TOKENS) & ~committed, 0)
        tl.store(Out + flat, tl.where(committed, a, b), token < TOKENS)

    @triton.jit
    def _selected_states(KP, VP, GP, BP, Initial, Tails, Out,
                         KT: tl.constexpr, KH: tl.constexpr, KD: tl.constexpr,
                         VT: tl.constexpr, VH: tl.constexpr, VD: tl.constexpr,
                         GT: tl.constexpr, GH: tl.constexpr,
                         BT: tl.constexpr, BH: tl.constexpr,
                         IH: tl.constexpr, IK: tl.constexpr, IV: tl.constexpr,
                         HEADS: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
                         ROWS: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
                         BR: tl.constexpr):
        head, tile = tl.program_id(0), tl.program_id(1)
        kk, vv = tl.arange(0, BK), tile * BV + tl.arange(0, BV)
        mask = (kk[:, None] < K) & (vv[None, :] < V)
        state = tl.load(Initial + head * IH + kk[:, None] * IK + vv[None, :] * IV,
                        mask, 0).to(tl.float32)
        rows = tl.arange(0, BR)
        tails = tl.load(Tails + rows, rows < ROWS, 0)
        last = tl.max(tails, 0)
        for row in tl.static_range(ROWS):
            tail = tl.load(Tails + row)
            tl.store(Out + ((row * HEADS + head) * K + kk[:, None]) * V + vv[None, :],
                     state, mask & (tail == 0))
        for step in range(last):
            key = tl.load(KP + step * KT + head * KH + kk * KD, kk < K, 0).to(tl.float32)
            key = key * tl.rsqrt(tl.sum(key * key, 0) + 1e-6)
            value = tl.load(VP + step * VT + head * VH + vv * VD, vv < V, 0).to(tl.float32)
            gate = tl.load(GP + step * GT + head * GH).to(tl.float32)
            beta = tl.load(BP + step * BT + head * BH).to(tl.float32)
            state = state * tl.exp(gate)
            delta = (value - tl.sum(state * key[:, None], 0)) * beta
            state = state + key[:, None] * delta[None, :]
            for row in tl.static_range(ROWS):
                tail = tl.load(Tails + row)
                tl.store(Out + ((row * HEADS + head) * K + kk[:, None]) * V + vv[None, :],
                         state, mask & (tail == step + 1))

    @triton.jit
    def _initial_states(Snapshot, Committed, Tails, Out,
                        ROW_WIDTH: tl.constexpr, N: tl.constexpr, B: tl.constexpr):
        i = tl.program_id(0) * B + tl.arange(0, B)
        row, col = i // ROW_WIDTH, i % ROW_WIDTH
        tail = tl.load(Tails + row, i < N, 0)
        committed = tl.load(Committed + col, (i < N) & (tail == 0), 0)
        snapshot = tl.load(Snapshot + tail * ROW_WIDTH + col,
                           (i < N) & (tail != 0), 0)
        tl.store(Out + i, tl.where(tail == 0, committed, snapshot), i < N)

    @triton.jit
    def _norm(X, R, W, S, Y, N: tl.constexpr, H: tl.constexpr,
              XT: tl.constexpr, XH: tl.constexpr, RT: tl.constexpr,
              RH: tl.constexpr, D: tl.constexpr, EPS: tl.constexpr,
              RESIDUAL: tl.constexpr, FUSED_WEIGHT: tl.constexpr, B: tl.constexpr):
        row = tl.program_id(0)
        col = tl.arange(0, B)
        x = tl.load(X + row // H * XT + row % H * XH + col,
                    col < D, 0).to(tl.float32)
        if RESIDUAL:
            r = tl.load(R + row // H * RT + row % H * RH + col,
                        col < D, 0).to(tl.float32)
            x = (x + r).to(X.dtype.element_ty).to(tl.float32)
            tl.store(S + row * D + col, x, col < D)
        variance = tl.sum(x * x, 0) / D
        weight = tl.load(W + col, col < D, 0).to(tl.float32)
        factor = 1.0 + weight
        if FUSED_WEIGHT:
            factor = factor.to(X.dtype.element_ty).to(tl.float32)
        y = (x * libdevice.rsqrt(variance + EPS)) * factor
        tl.store(Y + row * D + col, y, col < D)

    @triton.jit
    def _gate(A, G, Y, N: tl.constexpr, H: tl.constexpr, D: tl.constexpr,
              AT: tl.constexpr, AH: tl.constexpr, GT: tl.constexpr,
              GH: tl.constexpr, SILU: tl.constexpr, B: tl.constexpr):
        i = tl.program_id(0) * B + tl.arange(0, B)
        token, head, col = i // (H * D), (i // D) % H, i % D
        a = tl.load(A + token * AT + head * AH + col, i < N, 0).to(tl.float32)
        g = tl.load(G + token * GT + head * GH + col, i < N, 0).to(tl.float32)
        activated = 1.0 / (1.0 + libdevice.exp(-g))
        if SILU:
            activated = g * activated
        activated = activated.to(G.dtype.element_ty).to(tl.float32)
        tl.store(Y + i, a * activated, i < N)

    @triton.jit
    def _norm_rotary(X, W, C, S, Y, H: tl.constexpr, D: tl.constexpr,
                     ROTARY: tl.constexpr, XT: tl.constexpr, XH: tl.constexpr,
                     EPS: tl.constexpr, FUSED_WEIGHT: tl.constexpr, B: tl.constexpr):
        token, head = tl.program_id(0), tl.program_id(1)
        col = tl.arange(0, B)
        base = token * XT + head * XH
        x = tl.load(X + base + col, col < D, 0).to(tl.float32)
        weight = tl.load(W + col, col < D, 0).to(tl.float32)
        inverse = libdevice.rsqrt(tl.sum(x * x, 0) / D + EPS)
        dtype: tl.constexpr = X.dtype.element_ty
        factor = 1.0 + weight
        if FUSED_WEIGHT:
            factor = factor.to(dtype).to(tl.float32)
        normalized = ((x * inverse) * factor).to(dtype).to(tl.float32)
        other = tl.where(col < ROTARY // 2, col + ROTARY // 2, col - ROTARY // 2)
        shifted = tl.load(X + base + other, col < ROTARY, 0).to(tl.float32)
        shifted_weight = tl.load(W + other, col < ROTARY, 0).to(tl.float32)
        shifted_factor = 1.0 + shifted_weight
        if FUSED_WEIGHT:
            shifted_factor = shifted_factor.to(dtype).to(tl.float32)
        shifted = ((shifted * inverse) * shifted_factor).to(dtype).to(tl.float32)
        shifted = tl.where(col < ROTARY // 2, -shifted, shifted)
        cos = tl.load(C + token * ROTARY + col, col < ROTARY, 0).to(tl.float32)
        sin = tl.load(S + token * ROTARY + col, col < ROTARY, 0).to(tl.float32)
        first = (normalized * cos).to(dtype).to(tl.float32)
        second = (shifted * sin).to(dtype).to(tl.float32)
        out = tl.where(col < ROTARY, first + second, normalized)
        tl.store(Y + (token * H + head) * D + col, out, col < D)


def _cuda(x):
    return triton is not None and x.is_cuda and x.dtype == torch.bfloat16


def stage_conv(raw, committed, tails, *, lanes, slots, out=None):
    """Gather directly into the convolution layout; never mutate the carry."""
    tokens, channels = raw.shape
    kernel = committed.shape[-1]
    if out is None:
        out = raw.new_empty(tokens, channels, kernel)
    if triton is not None and raw.is_cuda:
        _stage_conv[(triton.cdiv(out.numel(), 256),)](
            raw, committed, tails, out, lanes, slots, tokens, channels, kernel,
            raw.stride(0), committed.stride(1), committed.stride(2), 256)
        return out
    token = torch.arange(tokens, device=raw.device)
    block = token >= lanes
    row = ((token - lanes).clamp_min(0) // slots).clamp(max=tails.numel() - 1)
    index = torch.where(block, (token - lanes).clamp_min(0) % slots, token)
    previous = index[:, None] + torch.arange(1 - kernel, 1, device=raw.device)
    own = block[:, None] & (previous >= 0)
    tail_position = previous + torch.where(block, tails[row], 0)[:, None]
    raw_index = torch.where(own, lanes + row[:, None] * slots + previous, tail_position)
    gather = torch.where(~own & (tail_position < 0), kernel + tail_position, kernel + raw_index)
    out.copy_(torch.cat((committed[0].T.to(raw.dtype), raw))[gather].transpose(1, 2))
    return out


def stage_hidden(tail, block, counts, *, lanes, slots, out=None):
    width = tail.shape[-1]
    tokens = lanes + block.shape[0] * slots
    if out is None:
        out = tail.new_empty(tokens, width)
    if triton is not None and tail.is_cuda:
        _stage_hidden[(triton.cdiv(out.numel(), 256),)](
            tail, block, counts, out, lanes, slots, tokens, width, tail.stride(0),
            block.stride(0), block.stride(1), block.shape[-1] - width, 256)
    else:
        valid = torch.arange(slots, device=tail.device)[None] < counts[:, None]
        out.copy_(torch.cat((tail[:lanes], torch.where(
            valid[..., None], block[:, :slots, -width:], 0).reshape(-1, width))))
    return out


def _geometry(x):
    if x.ndim == 2:
        return 1, x.stride(0), 0
    return x.shape[1], x.stride(0), x.stride(1)


def norm(x, module):
    if not _cuda(x):
        return module(x)
    heads, token_stride, head_stride = _geometry(x)
    out = torch.empty_like(x, memory_format=torch.contiguous_format)
    width = x.shape[-1]
    _norm[(x.numel() // width,)](
        x, x, module.weight, out, out, x.numel(), heads,
        token_stride, head_stride, 0, 0, width, module.eps, False, module.fused,
        triton.next_power_of_2(width), num_warps=4, enable_fp_fusion=False)
    return out


def norm_rotary(x, module, cos, sin):
    if not _cuda(x):
        normalized = module(x)
        rotary = cos.shape[-1]
        half = rotary // 2
        part = normalized[..., :rotary]
        shifted = torch.cat((-part[..., half:], part[..., :half]), -1)
        return torch.cat((part * cos + shifted * sin, normalized[..., rotary:]), -1)
    out = torch.empty_like(x, memory_format=torch.contiguous_format)
    tokens, heads, width = x.shape
    _norm_rotary[(tokens, heads)](
        x, module.weight, cos, sin, out, heads, width, cos.shape[-1],
        x.stride(0), x.stride(1), module.eps, module.fused, triton.next_power_of_2(width),
        num_warps=4, enable_fp_fusion=False)
    return out


def residual_norm(x, residual, module):
    if not _cuda(x):
        summed = x + residual
        return summed, module(summed)
    heads, token_stride, head_stride = _geometry(x)
    _, residual_token, residual_head = _geometry(residual)
    summed = torch.empty_like(x, memory_format=torch.contiguous_format)
    out = torch.empty_like(summed)
    width = x.shape[-1]
    _norm[(x.numel() // width,)](
        x, residual, module.weight, summed, out, x.numel(), heads,
        token_stride, head_stride, residual_token, residual_head, width,
        module.eps, True, module.fused, triton.next_power_of_2(width),
        num_warps=4, enable_fp_fusion=False)
    return summed, out


def gate_mul(value, gate, *, activation='sigmoid'):
    if activation not in ('sigmoid', 'silu'):
        raise ValueError('decoder gate activation must be sigmoid or silu')
    if not _cuda(value):
        return value * (gate.sigmoid() if activation == 'sigmoid' else F.silu(gate))
    heads, value_token, value_head = _geometry(value)
    _, gate_token, gate_head = _geometry(gate)
    out = torch.empty_like(value, memory_format=torch.contiguous_format)
    _gate[(triton.cdiv(value.numel(), 256),)](
        value, gate, out, value.numel(), heads, value.shape[-1],
        value_token, value_head, gate_token, gate_head, activation == 'silu',
        256, enable_fp_fusion=False)
    return out


def initial_states(snapshots, committed, tails):
    """Gather row states without needing a copy of committed into snapshots[0]."""
    if triton is None or not committed.is_cuda:
        selected = snapshots.index_select(0, tails)[:, 0]
        return torch.where((tails == 0)[:, None, None, None], committed, selected)
    out = torch.empty((tails.numel(),) + committed.shape[1:],
                      device=committed.device, dtype=committed.dtype)
    _initial_states[(triton.cdiv(out.numel(), 256),)](
        snapshots, committed, tails, out, committed.numel(), out.numel(), 256)
    return out


def selected_states(k, v, g, beta, committed, tails):
    """Draft row initial states; omit unselected snapshots and their gather."""
    if triton is None or not committed.is_cuda:
        key = k.float()
        key = key * torch.rsqrt(key.square().sum(-1, keepdim=True) + 1e-6)
        state = committed
        states = [state]
        for step in range(k.shape[1]):
            state = state * g[:, step].float().exp()[..., None, None]
            delta = (v[:, step].float() - (state * key[:, step, :, :, None]).sum(-2)) \
                    * beta[:, step].float()[:, :, None]
            state = state + key[:, step, :, :, None] * delta[:, :, None, :]
            states.append(state)
        return torch.stack(states).index_select(0, tails)[:, 0].contiguous()
    _, _, heads, key_dim = k.shape
    value_dim, rows = v.shape[-1], tails.numel()
    out = torch.empty((rows, heads, key_dim, value_dim), device=committed.device, dtype=torch.float32)
    _selected_states[(heads, triton.cdiv(value_dim, 8))](
        k, v, g, beta, committed, tails, out,
        *k.stride()[1:], *v.stride()[1:], *g.stride()[1:], *beta.stride()[1:],
        *committed.stride()[1:], heads, key_dim, value_dim, rows,
        triton.next_power_of_2(key_dim), 8, triton.next_power_of_2(rows))
    return out


def recurrent_output(gdn, q, k, v, g, beta, initial):
    """The draft read never consumes the final state returned by FLA."""
    if not q.is_cuda:
        return gdn._delta(q, k, v, g, beta, initial, chunk=False)[0]
    if gdn._recurrent is None:
        raise RuntimeError('hybrid GDN on CUDA requires fla gated delta rule kernels')
    from jetspec.engines.jetspec_plus_plus.profiles import kernel_value
    if kernel_value('draft_fuse_gdn27', '0') == '1':
        from jetspec.engines.jetspec_plus_plus.kernels.qwen3p8_27b.draft_gdn27 import recurrent
        tuned = recurrent(q, k, v, g, beta, initial)
        if tuned is not None:
            return tuned[0]
    return gdn._recurrent(q, k, v, g=g, beta=beta, initial_state=initial,
                          output_final_state=False, use_qk_l2norm_in_kernel=True)[0]
