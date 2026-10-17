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
    _packed_weights: dict = field(default_factory=dict, init=False, repr=False, compare=False)
    _plans: dict = field(default_factory=dict, init=False, repr=False, compare=False)

    @classmethod
    def from_config(cls, run_config, role):
        from jetspec.engines.jetspec_plus_plus.models.config import setting
        if role not in ('target', 'draft', 'lm_head'):
            raise ValueError('projection role must be target, draft or lm_head')
        key = {'target': 'engine_settings.precision_target', 'draft': 'engine_settings.precision_draft',
               'lm_head': 'engine_settings.precision_lm_head'}[role]
        mode = setting(run_config, 'engine', key, 'bf16')
        return cls(WeightFormat(mode=mode,
            activation_scale_policy='per_row' if mode in ('fq_w4a4', 'nvfp4_w4a4') else None),
            fake_quant=mode in ('fq_w4a16', 'fq_w4a4'))

    def _weight(self, weight, *, dtype=None):
        from jetspec.engines.jetspec_plus_plus.kernels.nvfp4.format import fake_quantize
        # Slices of packed stacks and KN transposes are recreated by callers.
        # Retain the source to keep its storage alive and key views by layout.
        key = (weight.data_ptr(), tuple(weight.shape), tuple(weight.stride()),
               weight.dtype, weight.device)
        cache_key = key if dtype is None else (key, dtype)
        if cache_key not in self._weights:
            if weight.is_cuda and torch.cuda.is_current_stream_capturing():
                raise RuntimeError('warm the projection weight before CUDA graph capture')
            if self.weight_format.mode == 'nvfp4_w4a4':
                from jetspec.engines.jetspec_plus_plus.kernels.nvfp4.format import dequantize
                rounded = dequantize(self._packed_weight(weight, key)).to(dtype or weight.dtype)
            elif self.weight_format.mode == 'nvfp4_w4a16':
                rounded = _nvfp4_weight_only(self, weight, key)
            else:
                rounded = fake_quantize(weight.reshape(-1, weight.shape[-1]).to(torch.bfloat16))
                rounded = rounded.reshape(weight.shape).to(weight.dtype)
            self._weights[cache_key] = (weight.detach(), rounded)
        return self._weights[cache_key][1]

    def _packed_weight(self, weight, key):
        from jetspec.engines.jetspec_plus_plus.kernels.nvfp4.format import quantize_weight
        if key not in self._packed_weights:
            self._packed_weights[key] = (weight.detach(), quantize_weight(weight.to(torch.bfloat16)))
        return self._packed_weights[key][1]

    def _inputs(self, x, weight):
        mode = self.weight_format.mode
        if mode not in ('fq_w4a16', 'fq_w4a4', 'nvfp4_w4a4', 'nvfp4_w4a16'):
            if self.fake_quant:
                raise NotImplementedError('fake quant requires fq_w4a16 or fq_w4a4')
            if mode != 'bf16':
                raise NotImplementedError(mode)
            return x, weight
        weight = self._weight(weight, dtype=x.dtype) if mode == 'nvfp4_w4a4' else self._weight(weight)
        if mode in ('fq_w4a4', 'nvfp4_w4a4'):
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
        if self.weight_format.mode == 'nvfp4_w4a4' and x.numel() // x.shape[-1] <= 80:
            return self._nvfp4_linear(x, weight, bias)
        x, weight = self._inputs(x, weight)
        if implementation is not None:
            if bias is None:
                return implementation(x, weight)
            return implementation(x, weight, bias)
        return F.linear(x, weight, bias)

    def _nvfp4_linear(self, x, weight, bias):
        from jetspec.engines.jetspec_plus_plus.kernels.nvfp4 import K1Plan
        source = x.reshape(-1, x.shape[-1]).to(torch.bfloat16).contiguous()
        key = (weight.data_ptr(), tuple(weight.shape), tuple(weight.stride()),
               weight.dtype, weight.device)
        plan_key = (key, source.shape[0])
        if plan_key not in self._plans:
            if x.is_cuda and torch.cuda.is_current_stream_capturing():
                raise RuntimeError('warm the NVFP4 projection M bucket before CUDA graph capture')
            plan = K1Plan(self._packed_weight(weight, key), source.shape[0], per_row_scale=True)
            plan.warmup(source)
            self._plans[plan_key] = plan
            self.warmed_m_buckets.add(source.shape[0])
        output = self._plans[plan_key](source).reshape(*x.shape[:-1], weight.shape[0])
        if bias is not None:
            output = output + bias
        return output.to(x.dtype)

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
    # nvfp4_w4a16 rounds the stored draft-lane weights at load, so every kernel reads them.
    bypass = [site for site, role in (('c1_native_loop', 1), ('c1_fused_direct_linear', 1),
              ('packed_qkv_kv', 1), ('drafter_lmhead', 2), ('compact_head', 2))
              if modes[role] != 'nvfp4_w4a16']
    print(f'PROJECTION_PRECISION target={modes[0]} draft={modes[1]} lm_head={modes[2]} '
          f'bf16_bypass={",".join(bypass + ["comparison_linears", "blockhead"])}'
          + (' target_verify_head=bf16' if modes[2] == 'nvfp4_w4a16' else ''), flush=True)


class _WeightRows:
    """Exact row fingerprints of draft-lane weights before and after NVFP4 rounding.

    A row key is an int64 dot product of the row's 16-bit patterns with fixed
    coefficients below 2**33, so no sum overflows and every copy of a row
    (packs, stacks, KN transposes) has the same key on any device.
    """

    def __init__(self):
        self.original, self.rounded, self.kept, self.names = {}, {}, {}, []
        self._coefficients = {}

    def keys(self, rows):
        rows = rows.reshape(-1, rows.shape[-1])
        width = rows.shape[-1]
        if (width, rows.device) not in self._coefficients:
            generator = torch.Generator().manual_seed(689 + width)
            self._coefficients[width, rows.device] = torch.randint(
                1, 2 ** 33, (width,), generator=generator, dtype=torch.int64).to(rows.device)
        coefficients = self._coefficients[width, rows.device]
        keys = torch.empty(rows.shape[0], dtype=torch.int64, device=rows.device)
        for start in range(0, rows.shape[0], 1024):
            bits = rows[start:start + 1024].contiguous().view(torch.int16).long()
            keys[start:start + 1024] = (bits * coefficients).sum(-1)
        return keys * 16411 + width

    def add(self, table, weight):
        table.setdefault(weight.shape[-1], []).append(self.keys(weight.detach()))

    def member(self, table, keys, width):
        stored = table.get(width)
        if not stored:
            return torch.zeros_like(keys, dtype=torch.bool)
        if len(stored) > 1:
            stored[:] = [torch.cat([item.to(keys.device) for item in stored]).unique()]
        return torch.isin(keys, stored[0].to(keys.device))

    def rounded_only(self, weight):
        """True when every row is a stored rounded row: a pack or view of converted weights."""
        if weight.dtype not in (torch.bfloat16, torch.float16) or weight.shape[-1] not in self.rounded:
            return False
        keys = self.keys(weight.detach())
        return bool(self.member(self.rounded, keys, weight.shape[-1]).all())


_weight_rows = _WeightRows()


def _nvfp4_weight_only(precision, weight, key):
    """Weight-only NVFP4: weights rounded at load pass through; others use a dequantized pack."""
    from jetspec.engines.jetspec_plus_plus.kernels.nvfp4.format import dequantize
    if _weight_rows.rounded_only(weight):
        return weight.detach()
    rows = weight.detach().reshape(-1, weight.shape[-1]).to(torch.bfloat16)
    if key not in precision._packed_weights:
        from jetspec.engines.jetspec_plus_plus.kernels.nvfp4.format import quantize_weight
        precision._packed_weights[key] = (weight.detach(), quantize_weight(rows))
    return dequantize(precision._packed_weights[key][1]).reshape(weight.shape).to(weight.dtype)


@torch.no_grad()
def round_nvfp4_weights(module, role):
    """Store NVFP4-rounded weights in place for roles set to nvfp4_w4a16.

    Call after checkpoint load and before any pack, stack or KN copy is built,
    so every draft-lane kernel (native loops, Triton heads, packed GEMMs) reads
    the rounded values with its kernel unchanged. Each parameter keeps its pack
    on its precision. Embeddings, norms and biases are not projections and stay.
    """
    from jetspec.engines.jetspec_plus_plus.engine import current_engine
    from jetspec.engines.jetspec_plus_plus.kernels.nvfp4.format import dequantize, quantize_weight
    engine = current_engine()
    if 'nvfp4_w4a16' not in (engine.precision_draft, engine.precision_lm_head):
        return 0
    rounded = 0
    for name, child in module.named_modules():
        if not isinstance(child, (nn.Linear, TargetLinear)):
            continue
        weight = child.weight
        if (weight.dtype != torch.bfloat16 or weight.ndim != 2 or weight.shape[1] % 16
                or getattr(weight, '_nvfp4_rounded', False)):
            continue
        if isinstance(child, TargetLinear):
            precision = child.precision
        else:
            selected = 'lm_head' if name.rsplit('.', 1)[-1] == 'lm_head' else role
            precision = bind_weight_role(weight, selected)
        if precision.weight_format.mode != 'nvfp4_w4a16':
            _weight_rows.add(_weight_rows.kept, weight)
            continue
        if _weight_rows.rounded_only(weight):
            # A copy of a rounded weight (a decoder head built from the drafter head).
            weight._nvfp4_rounded = True
            continue
        _weight_rows.add(_weight_rows.original, weight)
        pack = quantize_weight(weight.detach())
        weight.detach().copy_(dequantize(pack))
        key = (weight.data_ptr(), tuple(weight.shape), tuple(weight.stride()), weight.dtype, weight.device)
        precision._packed_weights[key] = (weight.detach(), pack)
        _weight_rows.add(_weight_rows.rounded, weight)
        _weight_rows.names.append((name, tuple(weight.shape)))
        weight._nvfp4_rounded = True
        rounded += 1
    return rounded


def _named_tensors(roots):
    """Name tensors reachable from module and object attributes, depth-first."""
    names, seen = {}, set()
    stack = [(name, root, 0) for name, root in reversed(list(roots.items()))]
    while stack:
        name, value, depth = stack.pop()
        if id(value) in seen or depth > 6:
            continue
        seen.add(id(value))
        if isinstance(value, torch.Tensor):
            names.setdefault((value.data_ptr(), tuple(value.shape), tuple(value.stride())), name)
        elif isinstance(value, (list, tuple)):
            stack.extend((f'{name}[{index}]', item, depth + 1) for index, item in enumerate(value))
        elif isinstance(value, dict):
            stack.extend((f'{name}[{key!r}]', item, depth + 1) for key, item in value.items()
                         if isinstance(key, (str, int, tuple)))
        elif isinstance(value, nn.Module) or hasattr(value, '__dict__'):
            items = dict(vars(value)) if hasattr(value, '__dict__') else {}
            if isinstance(value, nn.Module):
                items.update(value._parameters)
                items.update(value._buffers)
                items.update(value._modules)
                for key in ('_parameters', '_buffers', '_modules'):
                    items.pop(key, None)
            stack.extend((f'{name}.{key}', item, depth + 1) for key, item in items.items()
                         if isinstance(item, (torch.Tensor, list, tuple, dict, nn.Module))
                         or (hasattr(item, '__dict__') and type(item).__module__.startswith('jetspec')))
    return names


@torch.no_grad()
def nvfp4_census(roots, device=None):
    """Classify every live 16-bit matrix that holds draft-lane weight rows.

    Scans the process's live tensors (and the named roots) for rows or KN columns
    of weights seen by round_nvfp4_weights. A tensor is nvfp4 when all its weight
    rows are rounded rows, bf16_kept when it holds rows of a role set to bf16, and
    bf16_leftover when it still holds an original row of a rounded weight.
    """
    import gc
    import warnings
    names = _named_tensors(roots)
    widths = set(_weight_rows.original) | set(_weight_rows.rounded) | set(_weight_rows.kept)
    candidates, parameters = {}, set()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        live = [value for value in gc.get_objects() if isinstance(value, torch.Tensor)]
    for value in live:
        try:
            if value.dtype not in (torch.bfloat16, torch.float16) or value.ndim < 2 or value.is_meta:
                continue
            if device is not None and value.device != torch.device(device):
                continue
            key = (value.data_ptr(), tuple(value.shape), tuple(value.stride()))
        except Exception:  # fake and functional tensors carry no storage
            continue
        candidates.setdefault(key, value)
        if isinstance(value, nn.Parameter):
            parameters.add(key)
    rows = []
    for key, value in candidates.items():
        matched = False
        for layout, matrix in (('rows', value), ('kn', value.t() if value.ndim == 2 else None)):
            if matrix is None or matrix.shape[-1] not in widths:
                continue
            keys = _weight_rows.keys(matrix.detach())
            width = matrix.shape[-1]
            rounded = _weight_rows.member(_weight_rows.rounded, keys, width)
            original = _weight_rows.member(_weight_rows.original, keys, width) & ~rounded
            kept = _weight_rows.member(_weight_rows.kept, keys, width) & ~rounded
            if not bool((rounded | original | kept).any()):
                continue
            matched = True
            label = ('bf16_leftover' if bool(original.any()) else
                     'bf16_kept' if bool(kept.any()) else 'nvfp4')
            rows.append(dict(name=names.get(key, 'unnamed'), shape=tuple(value.shape), layout=layout,
                             precision=label, weight_rows=int((rounded | original | kept).sum()),
                             rounded_rows=int(rounded.sum()), original_rows=int(original.sum()),
                             kept_rows=int(kept.sum())))
        if not matched and key in parameters and value.ndim == 2:
            # A matrix parameter no rounding pass saw: an embedding, or a GEMM weight to inspect.
            rows.append(dict(name=names.get(key, 'unnamed'), shape=tuple(value.shape), layout='rows',
                             precision='unseen_parameter', weight_rows=0, rounded_rows=0,
                             original_rows=0, kept_rows=0))
    return rows


def report_nvfp4_census(roots, device=None):
    """Print the census once per process; returns the bf16 leftover count."""
    rows = nvfp4_census(roots, device)
    for row in sorted(rows, key=lambda item: (item['precision'], item['name'])):
        print('DRAFT_PRECISION_CENSUS ' + ' '.join(f'{key}={value}' for key, value in row.items()), flush=True)
    counts = {label: sum(row['precision'] == label for row in rows)
              for label in ('nvfp4', 'bf16_kept', 'bf16_leftover', 'unseen_parameter')}
    print(f'DRAFT_PRECISION_CENSUS_SUMMARY rounded_weights={len(_weight_rows.names)} tensors={len(rows)} '
          + ' '.join(f'{label}={count}' for label, count in counts.items())
          + ' excluded=comparison_linears,blockhead,embeddings,norms,biases', flush=True)
    return counts['bf16_leftover']


def _precision_for_role(role):
    from jetspec.engines.jetspec_plus_plus.engine import current_engine
    if role not in ('target', 'draft', 'lm_head'):
        raise ValueError('projection role must be target, draft or lm_head')
    mode = getattr(current_engine(), 'precision_' + role)
    return ProjectionPrecision(WeightFormat(mode=mode,
        activation_scale_policy='per_row' if mode in ('fq_w4a4', 'nvfp4_w4a4') else None),
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
        if layout == 'head' and self.precision.weight_format.mode == 'nvfp4_w4a16':
            # Weight-only NVFP4 is a draft-lane mode: the target's verify head stays bf16.
            self.precision = ProjectionPrecision()
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
            if self.precision.weight_format.mode != 'nvfp4_w4a4':
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
