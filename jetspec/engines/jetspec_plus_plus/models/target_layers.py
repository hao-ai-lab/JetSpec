"""Target layers parameterized by checkpoint geometry and rounding order."""
import torch
from torch import nn
from torch.nn import functional as F

from jetspec.engines.jetspec_plus_plus.draft.method.model_spec import ModelSpec
from jetspec.engines.jetspec_plus_plus.models.activation import SiluAndMul
from jetspec.engines.jetspec_plus_plus.models.layernorm import RMSDNorm, RMSHeadNorm
from jetspec.engines.jetspec_plus_plus.models.target_linear import TargetLinear
from jetspec.engines.jetspec_plus_plus.models.target_rotary import get_rope


class RMSNorm(nn.Module):
    def __init__(self, width, eps=1e-6, *, offset=0, norm_round='before_weight', head=False,
                 residual_in_activation_dtype=False):
        super().__init__()
        self.eps, self.offset, self.norm_round, self.head = eps, offset, norm_round, head
        self.residual_in_activation_dtype = residual_in_activation_dtype
        self.weight = nn.Parameter(torch.zeros(width) if offset else torch.ones(width))
        if norm_round == 'after_weight':
            from jetspec.engines.jetspec_plus_plus.profiles import kernel_enabled
            self.fused = kernel_enabled('hybrid_fused_norm', '1')
            if self.fused:
                from fla.modules.layernorm import rms_norm
                self._rms_norm = rms_norm
                self._fused_weight = None

    @property
    def variance_epsilon(self):
        return self.eps

    def forward(self, x, residual=None):
        if self.norm_round == 'before_weight':
            if self.residual_in_activation_dtype:
                value = x if residual is None else x + residual
                dtype = value.dtype
                summed = value
                value = value.float()
                value = value * torch.rsqrt(value.pow(2).mean(-1, keepdim=True) + self.eps)
                value = (self.offset + self.weight if self.offset else self.weight) * value.to(dtype)
                return value if residual is None else (value, summed)
            if self.offset:
                dtype = x.dtype
                value = x.float()
                if residual is not None:
                    value = value + residual.float()
                    residual = value.to(dtype)
                value = value * torch.rsqrt(value.pow(2).mean(-1, keepdim=True) + self.eps)
                value = value.to(dtype) * (self.offset + self.weight)
                return value if residual is None else (value, residual)
            if residual is not None:
                method = RMSHeadNorm.add_rms_forward if self.head else RMSDNorm.add_norm_forward
                return method(self, x, residual)
            method = RMSHeadNorm.rms_forward if self.head else RMSDNorm.norm_forward
            return method(self, x)
        if residual is not None:
            raise ValueError('after-weight norm uses explicit residual additions')
        if self.fused and x.is_cuda:
            weight = self._fused_weight
            if weight is None or weight.dtype != x.dtype or weight.device != x.device:
                weight = (self.offset + self.weight.float()).to(x.dtype)
                self._fused_weight = weight
            return self._rms_norm(x, weight, None, eps=self.eps)
        x_float = x.float()
        return (x_float * torch.rsqrt(x_float.square().mean(-1, keepdim=True) + self.eps)
                * (self.offset + self.weight.float())).to(x.dtype)


def HybridRMSNorm(width, eps):
    """Constructor retained for the F5 tail until that consumer is migrated."""
    return RMSNorm(width, eps, offset=1, norm_round='after_weight')


class FullAttention(nn.Module):
    def __init__(self, spec, config, *, draft=False, speculate=False, spec_k=1,
                 async_fan_out=1, draft_async=False, tp_group=None, tp_size=1, run_config=None,
                 separate_projections=None):
        super().__init__()
        self.spec = spec.attention
        attention = self.spec
        self.head_dim = attention.head_dim
        self.num_heads = attention.q_heads // tp_size
        self.num_kv_heads = attention.kv_heads // tp_size
        self.heads, self.kv_heads, self.dim = self.num_heads, self.num_kv_heads, self.head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.rotary_dim = attention.rotary_dim
        bias = getattr(config, 'attention_bias', False)
        self.separate_projections = (attention.output_gate if separate_projections is None
                                     else separate_projections)
        if self.separate_projections:
            from jetspec.engines.jetspec_plus_plus.models.gdn import hybrid_linear
            linear = hybrid_linear if attention.output_gate else lambda config, i, o, **kw: TargetLinear(
                i, o, bias=kw.get('bias', False))
            self.q_proj = linear(config, spec.hidden_size, (1 + attention.output_gate) * attention.q_heads * self.head_dim,
                                        bias=bias, run_config=run_config)
            self.k_proj = linear(config, spec.hidden_size, attention.kv_heads * self.head_dim,
                                        bias=bias, run_config=run_config)
            self.v_proj = linear(config, spec.hidden_size, attention.kv_heads * self.head_dim,
                                        bias=bias, run_config=run_config)
            self.o_proj = linear(config, attention.q_heads * self.head_dim, spec.hidden_size,
                                        bias=bias, run_config=run_config)
            self.q_norm = RMSNorm(self.head_dim, spec.rms_norm_eps, offset=attention.norm_offset,
                                  norm_round=attention.norm_round, head=True,
                                  residual_in_activation_dtype=separate_projections is True)
            self.k_norm = RMSNorm(self.head_dim, spec.rms_norm_eps, offset=attention.norm_offset,
                                  norm_round=attention.norm_round, head=True,
                                  residual_in_activation_dtype=separate_projections is True)
        else:
            self.qkv_proj = TargetLinear(spec.hidden_size,
                (attention.q_heads + 2 * attention.kv_heads) * self.head_dim, layout='qkv', bias=bias,
                head_size=self.head_dim, total_num_heads=attention.q_heads,
                total_num_kv_heads=attention.kv_heads, tp_group=tp_group, tp_size=tp_size)
            self.o_proj = TargetLinear(attention.q_heads * self.head_dim, spec.hidden_size,
                                       layout='row', tp_group=tp_group, tp_size=tp_size)
            self.rotary_emb = get_rope(self.head_dim, attention.rotary_dim,
                spec.max_position_embeddings, spec.rope_theta, spec.rope_scaling,
                norm_round=attention.norm_round)
        if separate_projections is None:
            from jetspec.engines.jetspec_plus_plus.models.attention import Attention
            self.attn = Attention(self.num_heads, self.head_dim, self.head_dim ** -0.5,
                self.num_kv_heads, draft=draft, speculate=speculate,
                draft_async=draft_async, F=async_fan_out, K=spec_k)
        if self.separate_projections:
            if separate_projections is None:
                self.rotary_emb = get_rope(self.head_dim, attention.rotary_dim,
                    spec.max_position_embeddings, spec.rope_theta, spec.rope_scaling,
                    norm_round=attention.norm_round)
        else:
            self.q_norm = RMSNorm(self.head_dim, spec.rms_norm_eps, offset=attention.norm_offset,
                                  norm_round=attention.norm_round, head=True)
            self.k_norm = RMSNorm(self.head_dim, spec.rms_norm_eps, offset=attention.norm_offset,
                                  norm_round=attention.norm_round, head=True)

    def forward(self, positions, x, cos=None, sin=None):
        if self.separate_projections and self.spec.output_gate:
            width = x.shape[0]
            q, gate = self.q_proj(x).view(width, self.num_heads, 2 * self.head_dim).chunk(2, dim=-1)
            q = self.q_norm(q)
            k = self.k_norm(self.k_proj(x).view(width, self.num_kv_heads, self.head_dim))
            v = self.v_proj(x)
            q, k = self.rotary_emb(q, k, cos, sin)
            mixed = self.attn(q.reshape(width, -1), k.reshape(width, -1), v)
            return self.o_proj(mixed * torch.sigmoid(gate.reshape(width, -1)))
        if self.separate_projections:
            q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        else:
            qkv = self.qkv_proj(x)
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = self.q_norm(q.reshape(-1, self.head_dim)).reshape(q.shape)
        k = self.k_norm(k.reshape(-1, self.head_dim)).reshape(k.shape)
        if self.spec.norm_round == 'after_weight':
            width = x.shape[0]
            q, k = self.rotary_emb(q.view(width, self.num_heads, self.head_dim),
                                  k.view(width, self.num_kv_heads, self.head_dim), cos, sin)
            q, k = q.reshape(width, -1), k.reshape(width, -1)
        else:
            q, k = self.rotary_emb(positions, q, k)
        return self.o_proj(self.attn(q, k, v))


class MLP(nn.Module):
    def __init__(self, spec, config, *, tp_group=None, tp_size=1, run_config=None,
                 separate_projections=None):
        super().__init__()
        self.separate_projections = (spec.attention.norm_round == 'after_weight'
                                     if separate_projections is None else separate_projections)
        hidden, intermediate = spec.hidden_size, spec.intermediate_size
        if self.separate_projections:
            from jetspec.engines.jetspec_plus_plus.models.gdn import hybrid_linear
            linear = hybrid_linear if spec.attention.norm_round == 'after_weight' else lambda config, i, o, **kw: TargetLinear(i, o)
            self.gate_proj = linear(config, hidden, intermediate, run_config=run_config)
            self.up_proj = linear(config, hidden, intermediate, run_config=run_config)
            self.down_proj = linear(config, intermediate, hidden, run_config=run_config)
            if separate_projections is True and spec.attention.norm_round == 'before_weight':
                self.act_fn = nn.SiLU()
        else:
            self.gate_up_proj = TargetLinear(hidden, 2 * intermediate, layout='merged',
                output_sizes=[intermediate] * 2, tp_group=tp_group, tp_size=tp_size)
            self.down_proj = TargetLinear(intermediate, hidden, layout='row',
                                           tp_group=tp_group, tp_size=tp_size)
            assert getattr(config, 'hidden_act', 'silu') == 'silu'
            self.act_fn = SiluAndMul()

    def forward(self, x):
        if self.separate_projections:
            gate = self.gate_proj(x)
            activated = self.act_fn(gate) if hasattr(self, 'act_fn') else F.silu(gate)
            return self.down_proj(activated * self.up_proj(x))
        return self.down_proj(self.act_fn(self.gate_up_proj(x)))


class DecoderLayer(nn.Module):
    def __init__(self, spec, layer_type='full_attention', *, config=None, draft=False,
                 speculate=False, spec_k=1, async_fan_out=1, draft_async=False,
                 tp_group=None, tp_size=1, max_verify_width=None, run_config=None,
                 separate_projections=None, residual_in_activation_dtype=False):
        super().__init__()
        if not isinstance(spec, ModelSpec):
            # The F5 compatibility constructor takes the normalized checkpoint config.
            from jetspec.engines.jetspec_plus_plus.models.registry import model_spec
            config, spec = spec, model_spec(spec)
        self.explicit_residual = spec.attention.norm_round == 'after_weight'
        self.residual_in_activation_dtype = residual_in_activation_dtype
        if layer_type == 'linear_attention':
            from jetspec.engines.jetspec_plus_plus.models.gdn import HybridGDN
            self.linear_attn = HybridGDN(config, max_verify_width or spec_k + 1, run_config=run_config)
        else:
            self.self_attn = FullAttention(spec, config, draft=draft, speculate=speculate,
                spec_k=spec_k, async_fan_out=async_fan_out, draft_async=draft_async,
                tp_group=tp_group, tp_size=tp_size, run_config=run_config,
                separate_projections=separate_projections)
        def norm():
            return RMSNorm(spec.hidden_size, spec.rms_norm_eps,
                           offset=spec.attention.norm_offset, norm_round=spec.attention.norm_round,
                           residual_in_activation_dtype=residual_in_activation_dtype)
        if self.explicit_residual:
            self.input_layernorm = norm()
            self.post_attention_layernorm = norm()
            self.mlp = MLP(spec, config, run_config=run_config,
                           separate_projections=separate_projections)
        else:
            self.mlp = MLP(spec, config, tp_group=tp_group, tp_size=tp_size, run_config=run_config,
                           separate_projections=separate_projections)
            self.input_layernorm = norm()
            self.post_attention_layernorm = norm()

    def forward(self, positions, hidden, residual=None, *, prefill=False, verify=False):
        if self.explicit_residual:
            # Preserve the F5 (hidden, cos, sin) calling convention during migration.
            x, cos, sin = positions, hidden, residual
            normed = self.input_layernorm(x)
            mixed = (self.linear_attn(normed, prefill=prefill, verify=verify)
                     if hasattr(self, 'linear_attn') else self.self_attn(None, normed, cos, sin))
            x = x + mixed
            return x + self.mlp(self.post_attention_layernorm(x))
        if self.residual_in_activation_dtype:
            x = hidden if residual is None else hidden + residual
            normed = self.input_layernorm(x)
            mixed = self.self_attn(positions, normed)
            x = x + mixed
            return x + self.mlp(self.post_attention_layernorm(x))
        if residual is None:
            residual = hidden
            hidden = self.input_layernorm(hidden)
        else:
            hidden, residual = self.input_layernorm(hidden, residual)
        hidden = (self.linear_attn(hidden, prefill=prefill, verify=verify)
                  if hasattr(self, 'linear_attn') else self.self_attn(positions, hidden))
        hidden, residual = self.post_attention_layernorm(hidden, residual)
        return self.mlp(hidden), residual


HybridDecoderLayer = DecoderLayer
