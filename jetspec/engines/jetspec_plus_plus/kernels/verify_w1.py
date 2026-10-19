"""Shared verify producer buffers and consumers, extracted from verify_layer8b W1.

The exact BF16 output owns its immutable FP8 side output. Copies and views take
the ordinary path unless the caller explicitly hands the activation through.
Warm each owning projection before capture; allocations then live in the graph pool.
"""
import torch


def activation_buffers(linear, rows):
    """Private verify producer buffers in the consumer's public W1 layout."""
    from .nvfp4.w4a8 import activation_layout, pack_w4a8
    pack = getattr(linear, '_verify_w1_pack', None)
    if pack is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError('warm the W1 verify projection before CUDA graph capture')
        pack = linear._verify_w1_pack = pack_w4a8(linear.weight.detach())
    # Preserve the landed 8B diagnostic attribute; both names borrow the same pack.
    linear._verify_w1prod8b_pack = pack
    layout = activation_layout(pack, rows)
    return (layout, torch.empty(layout.data_shape, dtype=layout.data_dtype, device=linear.weight.device),
            torch.empty(layout.scale_shape, dtype=layout.scale_dtype, device=linear.weight.device))


def w1_linear(linear, x, *, activation=None):
    """Consume one immutable producer result through the public W4A8 API."""
    if activation is None:
        activation = getattr(x, '_verify_w1', None)
        if activation is None:
            activation = getattr(x, '_verify_w1prod8b', None)
    if activation is None:
        return linear(x)
    from .nvfp4.w4a8 import w4a8_linear
    layout, q, sf = activation
    bias = linear.bias if linear.layout != 'row' or linear.tp_rank == 0 else None
    out = w4a8_linear(x.reshape(-1, x.shape[-1]), linear._verify_w1_pack, bias,
                      config=layout.config, x_fp8=q, x_scale=sf)
    if linear.layout == 'row' and linear.tp_size > 1:
        torch.distributed.all_reduce(out, group=linear.tp_group)
    return out.view(*x.shape[:-1], out.shape[-1])


def packed_activation(owner, rows):
    """Buffers for the enabled packed projection group, or the ordinary path."""
    if not owner.verify_cut_pack27b2:
        return None
    from jetspec.engines.jetspec_plus_plus.models.target_pack import module_pack
    pack = module_pack(owner, owner.pack27b2_members)
    return None if pack is None else activation_buffers(pack.linear, rows)


def w1_packed(pack, x, *, activation=None):
    """The packed GEMM and its member views, retaining an explicit side output."""
    return w1_linear(pack.linear, x, activation=activation).split(pack.sizes, dim=-1)[:len(pack.members)]
