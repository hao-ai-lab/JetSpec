"""Target projections with the existing layouts and one precision boundary.

The original parallel implementations remain pinned for comparison models.
Their weight loaders are reused here without changing shard order.
"""
from dataclasses import dataclass, field
import math

import torch
from torch import nn
from torch.nn import functional as F
import torch.distributed as dist

from jetspec.engines.jetspec_plus_plus.models.context import get_context
from jetspec.engines.jetspec_plus_plus.models.embed_head import VocabParallelEmbedding
from jetspec.engines.jetspec_plus_plus.models.linear import (LinearBase, ReplicatedLinear,
    QKVParallelLinear, MergedColumnParallelLinear, RowParallelLinear, divide)


@dataclass(frozen=True)
class WeightFormat:
    mode: str = 'bf16'
    block_size: int = 16
    scale_dtype: str = 'e4m3'
    scale_layout: str = 'swizzled'
    global_scale: bool = True
    activation_scale_policy: str | None = None


@dataclass
class ProjectionPrecision:
    weight_format: WeightFormat = field(default_factory=WeightFormat)
    fake_quant: bool = False
    warmed_m_buckets: set[int] = field(default_factory=set)
    _weights: dict = field(default_factory=dict, init=False, repr=False, compare=False)
    _activations: dict = field(default_factory=dict, init=False, repr=False, compare=False)

    @classmethod
    def from_config(cls, run_config, role):
        from jetspec.engines.jetspec_plus_plus.models.config import setting
        if role not in ('target', 'draft', 'lm_head'):
            raise ValueError('projection role must be target, draft or lm_head')
        key = {'target': 'engine_settings.precision_target', 'draft': 'engine_settings.precision_draft',
               'lm_head': 'engine_settings.precision_lm_head'}[role]
        mode = setting(run_config, 'engine', key, 'bf16')
        return cls(WeightFormat(mode=mode,
            activation_scale_policy='per_row' if mode == 'fq_w4a4' else None),
            fake_quant=mode in ('fq_w4a16', 'fq_w4a4'))

    def _weight(self, weight):
        from jetspec.engines.jetspec_plus_plus.kernels.nvfp4.format import fake_quantize
        # Slices of packed stacks and KN transposes are recreated by callers.
        # Retain the source to keep its storage alive and key views by layout.
        key = (weight.data_ptr(), tuple(weight.shape), tuple(weight.stride()),
               weight.dtype, weight.device)
        if key not in self._weights:
            if weight.is_cuda and torch.cuda.is_current_stream_capturing():
                raise RuntimeError('warm the projection weight before CUDA graph capture')
            self._weights[key] = (weight.detach(), fake_quantize(
                weight.reshape(-1, weight.shape[-1]).to(torch.bfloat16))
                .reshape(weight.shape).to(weight.dtype))
        return self._weights[key][1]

    def _inputs(self, x, weight):
        mode = self.weight_format.mode
        if mode not in ('fq_w4a16', 'fq_w4a4'):
            if self.fake_quant:
                raise NotImplementedError('fake quant requires fq_w4a16 or fq_w4a4')
            if mode != 'bf16':
                raise NotImplementedError(mode)
            return x, weight
        weight = self._weight(weight)
        if mode == 'fq_w4a4':
            from jetspec.engines.jetspec_plus_plus.kernels.nvfp4.activation import fake_quantize_activation
            source = x.reshape(-1, x.shape[-1]).to(torch.bfloat16).contiguous()
            key = (tuple(source.shape), x.dtype, x.device)
            if key not in self._activations:
                if x.is_cuda and torch.cuda.is_current_stream_capturing():
                    raise RuntimeError('warm the activation bucket before CUDA graph capture')
                self._activations[key] = torch.empty_like(source)
            x = fake_quantize_activation(source, out=self._activations[key]).reshape(x.shape).to(x.dtype)
        return x, weight

    def linear(self, x, weight, bias, *, implementation=None):
        x, weight = self._inputs(x, weight)
        if implementation is not None:
            if bias is None:
                return implementation(x, weight)
            return implementation(x, weight, bias)
        return F.linear(x, weight, bias)

    def bmm(self, x, weight):
        x, weight = self._inputs(x, weight)
        return torch.bmm(x, weight.transpose(1, 2))

    def mm(self, x, weight, bias=None, *, out=None):
        if self.weight_format.mode == 'bf16':
            x, weight = self._inputs(x, weight)
        else:
            x, weight = self._inputs(x, weight.t())
            weight = weight.t()
        if bias is not None:
            return torch.addmm(bias, x, weight, out=out)
        return torch.mm(x, weight, out=out)


_logged_precision = set()


def log_precision_exclusions(engine):
    modes = (engine.precision_target, engine.precision_draft, engine.precision_lm_head)
    if modes == ('bf16', 'bf16', 'bf16') or modes in _logged_precision:
        return
    _logged_precision.add(modes)
    print(f'PROJECTION_PRECISION target={modes[0]} draft={modes[1]} lm_head={modes[2]} '
          'bf16_bypass=c1_native_loop,c1_fused_direct_linear,packed_qkv_kv,'
          'drafter_lmhead,compact_head,comparison_linears,blockhead', flush=True)


def _precision_for_role(role):
    from jetspec.engines.jetspec_plus_plus.engine import current_engine
    if role not in ('target', 'draft', 'lm_head'):
        raise ValueError('projection role must be target, draft or lm_head')
    mode = getattr(current_engine(), 'precision_' + role)
    return ProjectionPrecision(WeightFormat(mode=mode,
        activation_scale_policy='per_row' if mode == 'fq_w4a4' else None),
        fake_quant=mode in ('fq_w4a16', 'fq_w4a4'))


def bind_weight_role(weight, role):
    """Attach a role to a parameter or pack without changing its storage."""
    from jetspec.engines.jetspec_plus_plus.engine import current_engine
    mode = getattr(current_engine(), 'precision_' + role)
    precision = getattr(weight, '_projection_precision', None)
    if precision is None or precision.weight_format.mode != mode:
        precision = weight._projection_precision = _precision_for_role(role)
    return precision


def bind_projection_role(module, role):
    """Assign decoder-tail and vocabulary roles before weights are warmed."""
    from jetspec.engines.jetspec_plus_plus.engine import current_engine
    for name, child in module.named_modules():
        selected = 'lm_head' if name.rsplit('.', 1)[-1] == 'lm_head' else role
        if isinstance(child, TargetLinear):
            child.precision_role = selected
            if child.precision.weight_format.mode != getattr(current_engine(), 'precision_' + selected):
                child.precision = _precision_for_role(selected)
        elif isinstance(child, DraftLinear):
            bind_weight_role(child.weight, selected)
    return module


def draft_precision(weight):
    """The boundary belongs to the actual parameter or packed weight tensor.

    Packs keep independent precision metadata; no registry retains their storage.
    Binding neither copies weights nor consumes the model's initialization RNG.
    """
    precision = getattr(weight, '_projection_precision', None)
    if precision is None:
        precision = bind_weight_role(weight, 'draft')
    return precision


def draft_linear(x, weight, bias=None, *, implementation=None):
    return draft_precision(weight).linear(x, weight, bias, implementation=implementation)


def draft_bmm(x, weight, *, precision=None):
    # A slice of a stack uses the stack owner's boundary, rather than metadata
    # on an ephemeral view created during graph capture.
    return (draft_precision(weight) if precision is None else precision).bmm(x, weight)


def draft_mm(x, weight, bias=None, *, out=None, precision=None):
    # KN packs and static output buffers keep their original mm/addmm route.
    return (draft_precision(weight) if precision is None else precision).mm(x, weight, bias, out=out)


class DraftLinear(nn.Linear):
    """Keep checkpoint nn.Linear storage/layout, sharing the target boundary.

    TargetLinear allocates and installs target shard loaders. Draft modules are
    already built by the pure-torch method or the checkpoint bundle, sometimes
    with frozen weights registered as buffers. Bind those objects in place so
    their parameters, buffers, ties, hooks and initialization stay intact.
    """

    def __init__(self, *args, precision_role='draft', **kwargs):
        super().__init__(*args, **kwargs)
        bind_weight_role(self.weight, precision_role)

    @property
    def precision(self):
        return draft_precision(self.weight)

    def forward(self, x):
        return draft_linear(x, self.weight, self.bias)

    @staticmethod
    def bind(module, *, precision_role='draft'):
        for child in module.modules():
            if type(child) is nn.Linear:
                child.__class__ = DraftLinear
        return bind_projection_role(module, precision_role)


class TargetLinear(LinearBase):
    def __init__(self, input_size, output_size, *, layout='replicated', bias=False,
                 tp_group=None, tp_size=1, head_size=None, total_num_heads=None,
                 total_num_kv_heads=None, output_sizes=None, fp32_bf16_storage=False,
                 precision_role=None):
        tp_dim = 1 if layout == 'row' else 0
        super().__init__(input_size, output_size, tp_dim, tp_group, tp_size)
        self.layout = layout
        self.precision_role = precision_role or ('lm_head' if layout == 'head' else 'target')
        self.precision = _precision_for_role(self.precision_role)
        self.fp32_bf16_storage = fp32_bf16_storage
        if layout == 'replicated' and tp_size != 1:
            raise ValueError('replicated target projections currently require TP=1')
        if layout == 'qkv':
            self.head_size = head_size
            self.total_num_heads = total_num_heads
            self.total_num_kv_heads = total_num_kv_heads or total_num_heads
            self.num_heads = divide(self.total_num_heads, tp_size)
            self.num_kv_heads = divide(self.total_num_kv_heads, tp_size)
        if layout == 'merged':
            self.output_sizes = output_sizes
        self.input_size_per_partition = divide(input_size, tp_size) if layout == 'row' else input_size
        self.output_size_per_partition = (output_size if layout in {'row', 'replicated'}
                                          else divide(output_size, tp_size))
        storage = {'dtype': torch.bfloat16} if fp32_bf16_storage else {}
        self.weight = nn.Parameter(torch.empty(self.output_size_per_partition,
                                               self.input_size_per_partition, **storage))
        self.weight.weight_loader = self.weight_loader
        if bias:
            self.bias = nn.Parameter(torch.empty(self.output_size_per_partition, **storage))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter('bias', None)
        if layout == 'replicated':
            # Match nn.Linear initialization for the existing GDN/attention path.
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
            if self.bias is not None:
                bound = 1 / math.sqrt(input_size) if input_size > 0 else 0
                nn.init.uniform_(self.bias, -bound, bound)

    def weight_loader(self, parameter, source, shard=None):
        if self.fp32_bf16_storage:
            if source.dtype == torch.bfloat16:
                parameter.data.copy_(source)
            else:
                parameter.data = source.to(device=parameter.device, dtype=torch.float32)
        elif self.layout == 'qkv':
            QKVParallelLinear.weight_loader(self, parameter, source, shard)
        elif self.layout == 'merged':
            MergedColumnParallelLinear.weight_loader(self, parameter, source, shard)
        elif self.layout == 'row':
            RowParallelLinear.weight_loader(self, parameter, source)
        elif self.layout == 'head':
            VocabParallelEmbedding.weight_loader(self, parameter, source)
        else:
            ReplicatedLinear.weight_loader(self, parameter, source)

    def project(self, x, bias=None):
        weight = self.weight
        if self.fp32_bf16_storage:
            if x.dtype != torch.float32:
                raise ValueError('FP32 projection storage requires FP32 activations')
            weight = weight.float()
            bias = None if bias is None else bias.float()
        return self.precision.linear(x, weight, bias)

    @torch.inference_mode()
    def warmup(self, m_bucket):
        if type(m_bucket) is not int or m_bucket <= 0:
            raise ValueError('M bucket must be a positive integer')
        dtype = torch.float32 if self.fp32_bf16_storage else self.weight.dtype
        x = torch.zeros(m_bucket, self.input_size_per_partition, device=self.weight.device, dtype=dtype)
        self.project(x, self.bias)
        self.precision.warmed_m_buckets.add(m_bucket)

    def forward(self, x, last_only=True):
        if self.layout == 'head':
            return self.head_forward(x, last_only)
        bias = self.bias if self.layout != 'row' or self.tp_rank == 0 else None
        output = self.project(x, bias)
        if self.layout == 'row' and self.tp_size > 1:
            dist.all_reduce(output, group=self.tp_group)
        return output

    def head_forward(self, x: torch.Tensor, last_only: bool = True): # x is always [nt = B*S, D] -> [nt, V]
        context = get_context()
        if context.cu_seqlens_q is not None:  # mq decode (prefill, glue, verify, tree decode)
            if context.is_prefill:
                if last_only:
                    # [nt, D] -> [b, D] which later becomes [b, V]
                    last_indices = context.cu_seqlens_q[1:] - 1
                    x = x[last_indices].contiguous()
                else:
                    # Return logits for all tokens in prefill
                    flat_logits = self.project(x)
                    if self.tp_size > 1:
                        parts = [torch.empty_like(flat_logits) for _ in range(self.tp_size)] if self.tp_rank == 0 else None
                        dist.gather(flat_logits, parts, 0, group=self.tp_group)
                        flat_logits = torch.cat(parts, dim=-1) if self.tp_rank == 0 else None
                    return flat_logits
            else: # multi-query decode path (glue, verify, tree)
                flat_logits = self.project(x)
                if self.tp_size > 1:
                    parts = [torch.empty_like(flat_logits) for _ in range(self.tp_size)] if self.tp_rank == 0 else None
                    dist.gather(flat_logits, parts, 0, group=self.tp_group)
                    flat_logits = torch.cat(parts, dim=-1) if self.tp_rank == 0 else None
                if flat_logits is None:
                    return None
                # Check if constant query len (verify/tree) or variable (glue)
                batch_size = context.cu_seqlens_q.size(0) - 1
                total_tokens = x.size(0)
                if total_tokens % batch_size == 0:
                    constant_query_len = total_tokens // batch_size
                    return flat_logits.view(batch_size, constant_query_len, flat_logits.size(-1))
                return flat_logits  # variable-length: return flat [N, V]

        # decode, get single token
        logits = self.project(x)
        if self.tp_size > 1:
            all_logits = [torch.empty_like(logits) for _ in range(self.tp_size)] if self.tp_rank == 0 else None
            dist.gather(logits, all_logits, 0, group=self.tp_group)
            logits = torch.cat(all_logits, -1) if self.tp_rank == 0 else None
        return logits
