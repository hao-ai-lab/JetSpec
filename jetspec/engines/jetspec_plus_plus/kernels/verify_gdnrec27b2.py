"""27B FP32 recurrence with output reduction before the rank-one update.

q @ (state + k * delta) = q @ state + (q @ k) * delta lets the output
reduction run alongside prediction, shortening the dependency chain. State
updates retain the original FP32 operations. Four value columns per one-warp
program expose 1536 programs at the 27B geometry. Width 1 uses the original
kernel. The committed input state is never written.
"""
import torch

from .verify_gdnrec27b import recurrent_verify as serial_verify, supported, triton

if triton is not None:
    import triton.language as tl
    from triton.experimental import gluon
    from triton.experimental.gluon import language as gl

    @gluon.jit
    def _algebra_verify(Q, K, V, G, B, H, O, HT, HS,
                        qt, qh, kt, kh, vt, vh, gt, gh, bt, bh,
                        W: gl.constexpr, HK: gl.constexpr, HV: gl.constexpr,
                        DK: gl.constexpr, DV: gl.constexpr, BV: gl.constexpr,
                        NW: gl.constexpr, FINAL: gl.constexpr, STATES: gl.constexpr,
                        STAGED: gl.constexpr = False,
                        KD=None, DELTA=None, EG=None, KEEP_DELTA: gl.constexpr = False,
                        SERIAL_OUTPUT: gl.constexpr = False,
                        PDL_SECONDARY: gl.constexpr = False, PDL_PRODUCER: gl.constexpr = False):
        if PDL_PRODUCER:
            gl.inline_asm_elementwise('griddepcontrol.launch_dependents; mov.u32 $0, $1;',
                '=r,r', [gl.program_id(0)], dtype=gl.int32, is_pure=False, pack=1)
        layout: gl.constexpr = gl.BlockedLayout([1, BV // NW], [32, 1], [1, NW], [1, 0])
        d = gl.arange(0, DK, layout=gl.SliceLayout(1, layout))
        c = gl.program_id(0) * BV + gl.arange(0, BV, layout=gl.SliceLayout(0, layout))
        head = gl.program_id(1)
        offset = head * DK * DV + d[:, None] * DV + c[None, :]
        state = gl.load(H + offset)
        if PDL_SECONDARY:
            # Issue H reads before waiting, without waiting for their results.
            # The memory clobber preserves those reads; every input pointer
            # below also depends on the head returned by the wait.
            head = gl.inline_asm_elementwise('griddepcontrol.wait; mov.u32 $0, $1;',
                '=r,r,~{memory}', [head], dtype=gl.int32, is_pure=False, pack=1)
        key_head = head // (HV // HK)
        if STAGED:
            # Load every token before the first output store. Shared rows are read
            # with the original scan layouts, retaining each reduction tree.
            rows: gl.constexpr = triton.next_power_of_2(W)
            kl: gl.constexpr = gl.BlockedLayout([1, 1], [1, 32], [1, NW], [1, 0])
            vl: gl.constexpr = gl.BlockedLayout([1, BV // NW], [1, 32], [1, NW], [1, 0])
            sl: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [1, 0])
            t = gl.arange(0, rows, layout=gl.SliceLayout(1, kl))
            kd = gl.arange(0, DK, layout=gl.SliceLayout(0, kl))
            qs = gl.allocate_shared_memory(Q.dtype.element_ty, [rows, DK], sl,
                gl.load(Q + t[:, None] * qt + key_head * qh + kd[None, :], t[:, None] < W))
            ks = gl.allocate_shared_memory(K.dtype.element_ty, [rows, DK], sl,
                gl.load(K + t[:, None] * kt + key_head * kh + kd[None, :], t[:, None] < W))
            tv = gl.arange(0, rows, layout=gl.SliceLayout(1, vl))
            vc = gl.program_id(0) * BV + gl.arange(0, BV, layout=gl.SliceLayout(0, vl))
            vs = gl.allocate_shared_memory(V.dtype.element_ty, [rows, BV], sl,
                gl.load(V + tv[:, None] * vt + head * vh + vc[None, :], tv[:, None] < W))
            gs = gl.allocate_shared_memory(G.dtype.element_ty, [rows, BV], sl,
                gl.load(G + tv[:, None] * gt + head * gh + gl.full([1, BV], 0, gl.int32, vl),
                        tv[:, None] < W))
            bs = gl.allocate_shared_memory(B.dtype.element_ty, [rows, BV], sl,
                gl.load(B + tv[:, None] * bt + head * bh + gl.full([1, BV], 0, gl.int32, vl),
                        tv[:, None] < W))
        for step in tl.static_range(W):
            if STAGED:
                # Keep shared descriptors rank two: Gluon 3.4's rank-reducing
                # index leaves an invalid CTA layout in LLVM lowering.
                q = gl.convert_layout(gl.sum(qs.slice(step, 1).load(kl), 0), gl.SliceLayout(1, layout)).to(gl.float32)
                k = gl.convert_layout(gl.sum(ks.slice(step, 1).load(kl), 0), gl.SliceLayout(1, layout)).to(gl.float32)
                value = gl.convert_layout(gl.sum(vs.slice(step, 1).load(vl), 0), gl.SliceLayout(0, layout)).to(gl.float32)
                gate = gl.convert_layout(gl.sum(gs.slice(step, 1).load(vl), 0), gl.SliceLayout(0, layout)).to(gl.float32)
                beta = gl.convert_layout(gl.sum(bs.slice(step, 1).load(vl), 0), gl.SliceLayout(0, layout)).to(gl.float32)
            else:
                q = gl.load(Q + step * qt + key_head * qh + d).to(gl.float32)
                k = gl.load(K + step * kt + key_head * kh + d).to(gl.float32)
            q = q * gl.rsqrt(gl.sum(q * q, 0) + 1e-6) * (DK ** -0.5)
            k = k * gl.rsqrt(gl.sum(k * k, 0) + 1e-6)
            if not STAGED:
                value = gl.load(V + step * vt + head * vh + c).to(gl.float32)
                gate = gl.load(G + step * gt + head * gh).to(gl.float32)
                beta = gl.load(B + step * bt + head * bh).to(gl.float32)
            if STAGED:
                decay = gl.exp(gate)
                state = state * decay[None, :]
            else:
                decay = gl.exp(gate)
                state = state * decay
            prior_output = gl.sum(state * q[:, None], 0)
            qk = gl.sum(q * k, 0)
            delta = (value - gl.sum(state * k[:, None], 0)) * beta
            state = state + k[:, None] * delta[None, :]
            output = prior_output + qk * delta
            if SERIAL_OUTPUT:
                output = gl.sum(state * q[:, None], 0)
            if KEEP_DELTA:
                gl.store(DELTA + (step * HV + head) * DV + c, delta)
                if gl.program_id(0) == 0:
                    if head % (HV // HK) == 0:
                        gl.store(KD + (step * HK + key_head) * DK + d, k)
                    if STAGED:
                        gl.store(EG + step * HV + head + gl.full([BV], 0, gl.int32, gl.SliceLayout(0, layout)),
                                 decay, gl.arange(0, BV, layout=gl.SliceLayout(0, layout)) == 0)
                    else:
                        gl.store(EG + step * HV + head, decay)
            gl.store(O + step * HV * DV + head * DV + c, output)
            if STATES:
                gl.store(HS + step * HV * DK * DV + offset, state)
        if FINAL:
            gl.store(HT + offset, state)



def recurrent_verify(q, k, v, g, beta, initial, *, final=False, states=False,
                     variant=None, block_v=32, warps=4, staged=False, delta_keep=None,
                     pdl=False, producer=False):
    """Return output, optional final state and every token state without mutating initial."""
    if variant is None and q.shape[1] == 1 and delta_keep is None:
        return serial_verify(q, k, v, g, beta, initial, final=final, states=states)
    if variant is None:
        variant, block_v, warps = 'algebra', 4, 1
    if (pdl or producer) and (variant != 'algebra' or block_v != 4 or warps != 1 or delta_keep is None):
        raise ValueError('GDN PDL requires the one-warp delta recurrence')
    if variant == 'bv8':
        return serial_verify(q, k, v, g, beta, initial, final=final, states=states,
                             block_v=8, warps=2)
    if not supported(q, k, v, g, beta, initial) or v.shape[-1] % block_v:
        raise ValueError('verify_gdnrec27b2 geometry mismatch')
    _, width, hk, dk = k.shape
    hv, dv = v.shape[2:]
    out = q.new_empty(1, width, hv, dv)
    last = torch.empty_like(initial) if final else None
    every = initial.new_empty(width, hv, dk, dv) if states else None
    args = (q, k, v, g, beta, initial, out, last, every,
            q.stride(1), q.stride(2), k.stride(1), k.stride(2), v.stride(1), v.stride(2),
            g.stride(1), g.stride(2), beta.stride(1), beta.stride(2))
    opts = dict(W=width, HK=hk, HV=hv, DK=dk, DV=dv, BV=block_v,
                FINAL=final, STATES=states, STAGED=staged and width > 1,
                num_warps=warps, enable_fp_fusion=False)
    if delta_keep is not None:
        key, delta, decay = delta_keep
        if (key.shape[1:] != (hk, dk) or key.shape[0] < width
                or delta.shape != (key.shape[0], hv, dv) or decay.shape != (key.shape[0], hv)
                or any(t.dtype != torch.float32 or not t.is_contiguous() for t in delta_keep)):
            raise ValueError('verify_gdnrec27b2 delta keep geometry mismatch')
        opts.update(KD=key, DELTA=delta, EG=decay, KEEP_DELTA=True, SERIAL_OUTPUT=width == 1)
    if pdl or producer:
        opts.update(PDL_SECONDARY=pdl, PDL_PRODUCER=producer)
    if pdl:
        opts['launch_pdl'] = True
    if variant != 'algebra':
        raise ValueError(f'unknown recurrence variant {variant}')
    _algebra_verify[(dv // block_v, hv)](*args, NW=warps, **opts)
    return out, last, every
