"""The target's mxfp4_w4a8 projections: MX rounding at load, W4A8 warm-up before capture, the census.

``ProjectionPrecision`` (target_linear.py) routes every target projection: M up
to MAX_M (256) runs the W4A8 kernel, larger M the BF16 copy. ``precision_target`` sets
the target-role projections. ``precision_target_head`` sets the target verify
head (layout ``head``); ``precision_lm_head`` stays the draft-lane vocabulary
heads. At load, ``round_nvfp4_weights`` packs each such weight once and stores
``dequantize_w4a8(pack)`` as its BF16 copy. With both settings bf16 every
function here returns at once. The 27B hybrid target binds every projection
(GDN in_proj and out_proj, gated attention, MLP) the same way; in_proj_b and
in_proj_a (N 48) use padded packs. Small-N shapes route through the kernel only
in the measured faster token buckets; the other buckets use their MX-rounded BF16 copy.
"""
import torch

MODE = 'mxfp4_w4a8'


def target_w4a8_enabled():
    from jetspec.engines.jetspec_plus_plus.engine import current_engine
    engine = current_engine()
    return MODE in (getattr(engine, 'precision_target', 'bf16'), getattr(engine, 'precision_target_head', 'bf16'))


def target_w4a8_linears(model):
    """The target TargetLinear modules in the mode (role target, or the verify head) with bf16 math."""
    from jetspec.engines.jetspec_plus_plus.models.target_linear import TargetLinear
    # verify_cut_pack27b2: a packed group's GEMM is its pack; the members' own GEMMs do not run.
    packs = [(name + '._projection_pack', pack.linear) for name, pack in _current_packs(model)]
    members = {id(member) for _, pack in _current_packs(model) for member in pack.members}
    for name, child in list(model.named_modules()) + packs:
        if (isinstance(child, TargetLinear) and (child.precision_role == 'target' or child.layout == 'head')
                and child.precision.weight_format.mode == MODE and not child.fp32_bf16_storage
                and child.weight.dtype == torch.bfloat16 and child.weight.ndim == 2
                and id(child) not in members):
            yield name, child


def _current_packs(model):
    """The verify_cut_pack27b2 packs of a model whose members are still their row views."""
    for name, module in model.named_modules():
        pack = module.__dict__.get('_projection_pack')
        if pack and pack.current():
            yield name, pack


def bind_target_head_w4a8(model):
    """Bind the verify head (layout head) to mxfp4_w4a8 when precision_target_head asks for it."""
    from jetspec.engines.jetspec_plus_plus.engine import current_engine
    from jetspec.engines.jetspec_plus_plus.models.target_linear import ProjectionPrecision, TargetLinear, WeightFormat
    if getattr(current_engine(), 'precision_target_head', 'bf16') != MODE:
        return 0
    if getattr(getattr(model, 'spec', None), 'tie_word_embeddings', False):
        raise ValueError('precision_target_head=mxfp4_w4a8 would round the tied input embedding')
    bound = 0
    for _, child in model.named_modules():
        if isinstance(child, TargetLinear) and child.layout == 'head':
            if child.precision.weight_format.mode != MODE:
                child.precision = ProjectionPrecision(WeightFormat(mode=MODE))
            bound += 1
    return bound


@torch.no_grad()
def round_target_w4a8_weights(model):
    """After the checkpoint load: bind the verify head, then round and pack the target weights in place.

    Returns the number of weights rounded. Call before the first forward, so
    the BF16 fallback reads the rounded weight itself and keeps no copy.
    """
    if not target_w4a8_enabled():
        return 0
    from jetspec.engines.jetspec_plus_plus.models.target_linear import TargetLinear, round_nvfp4_weights
    # FP32 math on BF16 storage (hybrid_fp32_linear_bf16_storage) has no W4A8 kernel: refuse it.
    stored = [name for name, child in model.named_modules()
              if isinstance(child, TargetLinear) and child.fp32_bf16_storage
              and child.precision.weight_format.mode == MODE]
    if stored:
        raise ValueError(f'{MODE} target projections need bf16 activations; FP32-math projections '
                         f'(hybrid_fp32_linear_bf16_storage=1): {stored[:4]}')
    bind_target_head_w4a8(model)
    # verify_cut_pack27b2: round each packed weight first. Its member rows then count as rounded,
    # so the members get no W4A8 pack of their own. MX rounding is per row, so the values agree.
    rounded = sum(round_nvfp4_weights(pack.linear, 'target') for _, pack in _current_packs(model))
    return rounded + round_nvfp4_weights(model, 'target')


@torch.inference_mode()
def warm_target_w4a8(model, m_buckets):
    """Build the W4A8 kernel state of every target projection and the head for each M bucket up to MAX_M.

    Run before CUDA graph capture: a bucket first met inside capture falls back
    to the BF16 copy. The head runs eagerly after the replay at the same M (the
    verify width). The warm-up calls the routing's own entry with zero
    activations (no RNG draw) and records them under one census site.
    """
    if not target_w4a8_enabled():
        return 0
    from jetspec.engines.jetspec_plus_plus.kernels.nvfp4.w4a8 import MAX_M
    from jetspec.engines.jetspec_plus_plus.models.target_linear import w4a8_draft_linear
    buckets = sorted({int(m) for m in m_buckets if 1 <= int(m) <= MAX_M})
    warmed = 0
    for _, child in target_w4a8_linears(model):
        # The head projects without a bias; a row shard adds it on rank 0 only.
        bias = (None if child.layout == 'head' else
                child.bias if child.layout != 'row' or child.tp_rank == 0 else None)
        for m in buckets:
            x = torch.zeros(m, child.input_size_per_partition, dtype=torch.bfloat16,
                            device=child.weight.device)
            if w4a8_draft_linear(x, child.weight, bias, site='target_w4a8.py:warm_target_w4a8') is not None:
                child.precision.warmed_m_buckets.add(m)
                warmed += 1
    return warmed


def report_target_w4a8(prefix='W4A8_TARGET_SITE'):
    """Print the W4A8 site census of this process for the target; None when both settings are bf16."""
    if not target_w4a8_enabled():
        return None
    from jetspec.engines.jetspec_plus_plus.models.target_linear import report_w4a8_sites
    return report_w4a8_sites(prefix)
