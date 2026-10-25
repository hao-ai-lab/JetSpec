"""Draft-only launch tiling of the installed FLA gated DeltaNet recurrence."""


def recurrent(q, k, v, g, beta, initial, *, output_final_state=False, bv=None,
              warps=1, stages=3):
    """Return None outside the segment-gated B200 draft geometry."""
    import torch
    if (not q.is_cuda or q.dtype != torch.bfloat16 or k.shape != q.shape
            or v.shape != q.shape or k.dtype != q.dtype or v.dtype != q.dtype
            or q.ndim != 4 or q.shape[2:] != (48, 128)
            or not 1 <= q.shape[0] <= 4 or not 1 <= q.shape[1] <= 16
            or (q.shape[0] > 1 and (q.shape[1] != 15 or output_final_state))
            or g.shape != q.shape[:3] or beta.shape != g.shape
            or g.dtype != torch.float32 or beta.dtype != torch.bfloat16
            or initial.shape != (q.shape[0], 48, 128, 128) or initial.dtype != torch.float32
            or torch.cuda.get_device_capability(q.device) != (10, 0)):
        return None
    import triton
    from fla.ops.gated_delta_rule.fused_recurrent import fused_recurrent_gated_delta_rule_fwd_kernel
    q, k, v, g, beta, initial = [x.contiguous() for x in (q, k, v, g, beta, initial)]
    b, t, h, dim = q.shape
    # Full tail scans favor more small CTAs; the four-row refresh favors BV=16.
    if bv is None:
        if b == 1 and t == 16:
            bv = 4
        elif not output_final_state and (b == 2 or (b == 1 and t in (7, 9, 15))):
            bv = 8  # Retain production tiling where the complete-chain sweep regressed.
        else:
            bv = 16
    out = torch.empty_like(v)
    state = torch.empty_like(initial) if output_final_state else None
    fused_recurrent_gated_delta_rule_fwd_kernel[(triton.cdiv(128, bv), b * h)](
        q=q, k=k, v=v, g=g, gk=None, gv=None, beta=beta, A_log=None, dt_bias=None,
        o=out, h0=initial, ht=state, cu_seqlens=None, scale=dim ** -.5, T=t,
        H=h, HV=h, K=dim, V=128, BK=128, BV=bv,
        IS_BETA_HEADWISE=True, USE_QK_L2NORM_IN_KERNEL=True,
        APPLY_BETA_SIGMOID=False, ALLOW_NEG_EIGVAL=False, STATE_V_FIRST=False,
        num_warps=warps, num_stages=stages)
    return out, state
