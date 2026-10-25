"""Opt-in GDN glue folds. Mask bits: 1 strided z, 2 conv gates.

The packed projection's z has a gap between tokens. Flattening its tokens and
heads copies it; read token/head strides directly in the current W1 gated-norm
producer, retaining its PDL early trigger. Without W1 retain the existing norm
and copy, so mask 1 changes no arithmetic on that fallback either.
Only mask 1 is selectable. Conv gate bit 2 below is retained as an unreachable
ablation after masks 2/3 regressed width 16 by 0.38/0.20 ms on the current stack.
Recurrent norm bit 4 is unreachable: it regressed width 16 by 2.97 ms.
"""
import torch

from jetspec.engines.jetspec_plus_plus.kernels.verify_gdnrec27b import recurrent_verify
from jetspec.engines.jetspec_plus_plus.kernels.verify_gdnconv import conv_verify
from jetspec.engines.jetspec_plus_plus.kernels.verify_pack_glue import gate_beta_rows

def verify(layer, raw, z, a, b, initial, committed_conv, scratch_conv):
    """The C=1 lazy GDN verify, preserving kept inputs and acceptance-time commits."""
    from jetspec.engines.jetspec_plus_plus.kernels.gdn_lazy import commit_state
    mask = layer.verify_cut_gdnfuse27b
    width = raw.shape[-1]
    if width >= scratch_conv.shape[0]:
        raise ValueError('verify width exceeds the GDN scratch capacity')
    keep = layer._lazy_keep()
    if layer.verify_cut_gbeta:
        decay = layer.gate_decay()
    else:
        decay = -layer.A_log.float().exp()
    gate_inputs = None
    if mask & 2:
        gate_inputs = a, b, layer.dt_bias, decay
        g, beta = (t[:width].unsqueeze(0) for t in keep[2:])
    else:
        g, beta = gate_beta_rows(a, b, layer.dt_bias, decay)
    q, k, v = conv_verify(raw, committed_conv, layer.conv1d.weight, g, beta, keep, scratch_conv,
                          layer.key_heads, layer.value_heads, layer.key_dim, layer.value_dim,
                          gate_inputs=gate_inputs,
                          **({'pdl': True} if layer.verify_cut_pdlgdn27b & 1 else {}),
                          **({'conv27b': True} if layer.verify_cut_conv27b else {}),
                          **({'rawkeep': True} if layer.verify_cut_conv27b_rawkeep else {}))
    if not layer._lazy_warm and not torch.cuda.is_current_stream_capturing():
        commit_state(keep, torch.zeros(1, dtype=torch.int64, device=initial.device), initial,
                     out=torch.empty_like(initial))
        layer._lazy_warm = True
    # Bit 1 can be measured independently; without it retain the original z copy.
    if not mask & 1:
        z = z.reshape(-1, layer.value_dim).view(1, width, layer.value_width)
    if layer.verify_cut_gdnrec27b2:
        from jetspec.engines.jetspec_plus_plus.kernels.verify_gdnrec27b2 import recurrent_verify as recurrence
        output = recurrence(q, k, v, g, beta, initial,
                            **({'staged': True} if layer.verify_cut_gdnrec27b3 else {}),
                            **({'delta_keep': layer._delta_keep()} if layer.verify_cut_commitdelta27b else {}),
                            **({'pdl': bool(layer.verify_cut_pdlgdn27b & 1),
                                'producer': bool(layer.verify_cut_pdlgdn27b & 2)} if layer.verify_cut_pdlgdn27b else {}))[0]
    else:
        output = recurrent_verify(q, k, v, g, beta, initial)[0]
    if layer.verify_cut_w1prod27b and output.dtype == torch.bfloat16:
        from jetspec.engines.jetspec_plus_plus.kernels.verify_gdn_w1 import gated_norm
        from jetspec.engines.jetspec_plus_plus.kernels.verify_w1 import activation_buffers, w1_linear
        output = gated_norm(output.reshape(-1, layer.value_dim), z, layer.norm, layer.value_heads,
                            activation=activation_buffers(layer.out_proj, width),
                            pdl=bool(layer.verify_cut_pdl27b & 1),
                            **({'secondary': True} if layer.verify_cut_pdlgdn27b & 2 else {}),
                            **({'fp8store': True} if layer.verify_cut_fp8store27b else {}))
        return w1_linear(layer.out_proj, output)
    output = layer.norm(output.reshape(-1, layer.value_dim), z.reshape(-1, layer.value_dim))
    return layer.out_proj(output.reshape(width, layer.value_width))
