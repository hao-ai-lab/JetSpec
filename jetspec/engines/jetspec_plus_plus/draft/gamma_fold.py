"""Load-time gamma folding for one Q + all K/V projections of a C1 stream."""
from dataclasses import dataclass

import torch
from torch import nn

from jetspec.engines.jetspec_plus_plus.models.target_linear import draft_linear, round_nvfp4_weights


@dataclass(frozen=True)
class GammaSpan:
    stop: int
    index: int
    eps: float
    weight: torch.Tensor
    widths: tuple
    unit: torch.Tensor
    empty_kv: torch.Tensor


@torch.no_grad()
def prepare(head):
    """Fold ORIGINAL checkpoint weights before their first FP4 conversion.

    Separate packed parameters retain the original projections for prefix fill
    and other routes. They are outside state_dict and the checkpoint seal.
    """
    if hasattr(head, '_draft_w5'):
        return head._draft_w5
    from jetspec.engines.jetspec_plus_plus.engine import current_engine
    if current_engine().precision_draft not in ('bf16', 'mxfp4_w4a8'):
        raise ValueError('draft_fuse_w5 supports bf16 and mxfp4_w4a8 projections')
    plan = head._d1_lean_plan()
    groups = []
    first = 0
    while first < len(plan):
        stop = first + 1
        while stop < len(plan) and plan[stop][0] == plan[first][0]:
            stop += 1
        eps = plan[first][2]
        if any(part[7] != eps for part in plan[first:stop]):
            raise ValueError('draft_fuse_w5 requires in_eps == kv_eps across each stream')
        groups.append((first, stop))
        first = stop
    spans = {}
    for first, stop in groups:
        part = plan[first]
        if any(getattr(p[i], '_nvfp4_rounded', False)
               for p in plan[first:stop] for i in (3, 8, 11)):
            raise ValueError('prepare draft_fuse_w5 before rounding checkpoint weights')
        weights = [(part[3], part[1])]
        weights.extend((p[i], p[6]) for p in plan[first:stop] for i in (8, 11))
        folded = torch.cat([(w.float() * gamma.float()).to(w.dtype)
                            for w, gamma in weights]).detach()
        # Reuse load-time rounding/pack registration without initializing any
        # random parameters or changing the checkpoint's projections.
        projection = nn.Linear(folded.shape[1], folded.shape[0], bias=False,
                               device='meta', dtype=folded.dtype)
        projection.weight = nn.Parameter(folded, requires_grad=False)
        round_nvfp4_weights(projection, 'draft')
        unit = torch.ones_like(part[1])
        spans[first] = GammaSpan(stop, part[0], part[2], projection.weight,
                                tuple(w.shape[0] for w, _ in weights), unit,
                                unit.new_empty((0, unit.numel())))
    head._draft_w5 = spans
    return spans


def project(span, fused, u, pending, append=None):
    """One common unit-gamma norm and one concatenated GEMM at a stream start."""
    from jetspec.engines.jetspec_plus_plus.profiles import kernel_enabled
    if kernel_enabled('draft_fuse_w1', '1'):
        from jetspec.engines.jetspec_plus_plus.kernels.nvfp4.producers import enabled, norm
        if enabled(u) and u.shape[0] + (0 if append is None else append.shape[0]) <= 256:
            normalized, state = norm(u if pending is None else pending[0], span.unit, span.eps,
                                     residual=None if pending is None else pending[1],
                                     append=None if append is None else append.contiguous())
            pieces = draft_linear(normalized, span.weight).split(span.widths, dim=-1)
            return state, (pieces[0].contiguous(), *pieces[1:])
    normalized, state, _ = fused.ew_pre_norm(
        u, u, span.unit, span.empty_kv, span.eps, span.eps, pending)
    if append is not None:
        appended = fused._wide_norm_extension().rms_norm(
            append.contiguous(), span.unit, span.eps)
        normalized = torch.cat((appended, normalized))
    pieces = draft_linear(normalized, span.weight).split(span.widths, dim=-1)
    # Native Q norm/rotary requires dense rows; K/V consumers accept strides.
    return state, (pieces[0].contiguous(), *pieces[1:])
