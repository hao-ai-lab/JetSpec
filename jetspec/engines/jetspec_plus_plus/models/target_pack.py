"""verify_cut_pack27b2: one GEMM per hybrid target projection group.

The groups are the GDN in_proj q|k|v, z, b, a; the gated attention q|k|v; and the MLP
gate|up. A ``ProjectionPack`` is one ``TargetLinear`` that holds the members' weight rows
in order, padded with zero rows to a multiple of 128 (a W4A8 kernel shape). Each member
weight becomes a row view of that weight, so the checkpoint loader, the member modules
and in-place weight rounding still work. The packed GEMM goes through the pack's own
``ProjectionPrecision``. Its output is split with views.

Only target-role projections without bias, with plain storage and a bf16 or MX (W4A8)
precision mode pack. NVFP4 modes have a per-tensor global scale, which a pack would
change. Any other group keeps its separate GEMMs.
"""
import torch
from torch import nn

ROWS = 128
# The precision modes whose rounding is per row, so a pack rounds every member row alike.
PACKABLE_MODES = ('bf16', 'mxfp4_w4a8')


class ProjectionPack:
    """A packed TargetLinear over ``members`` and the row views it gave them."""

    def __init__(self, linear, members, sizes):
        self.linear, self.members = linear, tuple(members)
        self.sizes = tuple(sizes) + (linear.weight.shape[0] - sum(sizes),)
        self._pointers = self._layout()

    def _layout(self):
        return tuple((m.weight.data_ptr(), tuple(m.weight.shape), m.weight.dtype) for m in self.members) + (
            (self.linear.weight.data_ptr(), tuple(self.linear.weight.shape), self.linear.weight.dtype),)

    def current(self):
        """True while every member weight is still the row view the pack gave it."""
        return self._layout() == self._pointers

    def __call__(self, x):
        return self.linear(x).split(self.sizes, dim=-1)[:len(self.members)]


def pack_projections(members):
    """A ProjectionPack over ``members``, or None when they cannot share one GEMM."""
    from jetspec.engines.jetspec_plus_plus.models.target_linear import TargetLinear
    first = members[0]
    weight = first.weight
    mode = first.precision.weight_format
    for member in members:
        if (type(member) is not TargetLinear or member.layout != 'replicated' or member.tp_size != 1
                or member.precision_role != 'target' or member.bias is not None or member.fp32_bf16_storage
                or member.precision.weight_format != mode or member.precision.fake_quant
                or mode.mode not in PACKABLE_MODES or member.weight.ndim != 2
                or member.weight.shape[1] != weight.shape[1] or member.weight.dtype != weight.dtype
                or member.weight.device != weight.device or member.weight.is_meta):
            return None
    if weight.is_cuda and torch.cuda.is_current_stream_capturing():
        raise RuntimeError('verify_cut_pack27b2 packs the projections before CUDA graph capture')
    sizes = [member.weight.shape[0] for member in members]
    rows = sum(sizes)
    padded = -(-rows // ROWS) * ROWS
    # The pack matches its members: ordinary parameters stay ordinary inside an inference-mode forward.
    with torch.inference_mode(weight.is_inference()), torch.no_grad():
        storage = torch.empty((padded, weight.shape[1]), dtype=weight.dtype, device=weight.device)
        torch.cat([member.weight.detach() for member in members], out=storage[:rows])
        storage[rows:].zero_()
        with torch.device('meta'):
            linear = TargetLinear(weight.shape[1], padded, layout='merged',
                                  output_sizes=sizes + [padded - rows], precision_role='target')
        if linear.precision.weight_format != mode:
            linear.precision = type(first.precision)(mode, role='target')
        linear.weight = nn.Parameter(storage, requires_grad=weight.requires_grad)
        linear.weight.weight_loader = linear.weight_loader
        start = 0
        for member, size in zip(members, sizes):
            member.weight.data = storage[start:start + size]
            start += size
    return ProjectionPack(linear, members, sizes)


def module_pack(owner, names):
    """The owner's pack over the named member projections, built or rebuilt when stale or regrouped.

    The pack lives outside the module's registered children, so parameters, state dicts
    and the strict loader check see the members only. A group that cannot pack is
    remembered and keeps its separate GEMMs.
    """
    pack = owner.__dict__.get('_projection_pack')
    if pack is False:
        return None
    members = [getattr(owner, name) for name in names]
    if pack is None or not pack.current() or [id(m) for m in pack.members] != [id(m) for m in members]:
        pack = pack_projections(members)
        owner.__dict__['_projection_pack'] = False if pack is None else pack
    return pack


def pack_model_projections(model):
    """Pack every group whose module has verify_cut_pack27b2 on; returns the pack count."""
    packed = 0
    for module in model.modules():
        names = getattr(module, 'pack27b2_members', None)
        if names and getattr(module, 'verify_cut_pack27b2', False):
            packed += module_pack(module, names) is not None
    return packed


def model_packs(model):
    """The live ProjectionPacks of a model, for warm-up and census tools."""
    for module in model.modules():
        pack = module.__dict__.get('_projection_pack')
        if pack:
            yield pack
