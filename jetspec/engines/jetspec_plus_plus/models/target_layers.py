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
        # verify_cut_norm27b: the owning layer routes this after-weight norm to kernels/verify_norm27b.py.
        self.norm27b = False
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
        if self.norm27b and x.is_cuda:
            from jetspec.engines.jetspec_plus_plus.kernels import verify_norm27b
            if verify_norm27b.supported(x, self):
                return verify_norm27b.rms_norm(x, self)
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
        from jetspec.engines.jetspec_plus_plus.models.config import setting
        self.verify_cut_mlpglue = setting(
            run_config, 'engine', 'engine_settings.verify_cut_mlpglue', '0') == '1'
        # verify_cut_attn27b: the gated N=1 verify attention as split-KV launches (kernels/verify_attn27b.py).
        self.verify_cut_attn27b = (self.separate_projections and attention.output_gate and setting(
            run_config, 'engine', 'engine_settings.verify_cut_attn27b', '0') == '1')
        cut = setting(run_config, 'engine', 'engine_settings.verify_cut_persist8b', '0')
        self.verify_cut_persist8b = (
            not self.separate_projections and attention.norm_round == 'before_weight'
            and attention.rotary_dim == attention.head_dim and cut in ('1', '2', '3', '4'))
        self.verify_cut_persist8b_pdl = self.verify_cut_persist8b and cut in ('2', '3', '4')
        self.verify_cut_persist8b_attention = self.verify_cut_persist8b and cut in ('3', '4')
        # Value '4': fixed-tile split attention and a grouped merge (kernels/verify_attn8b.py).
        self.verify_cut_persist8b_one_launch = self.verify_cut_persist8b and cut == '4'
        # verify_cut_pack27b2: the gated target q|k|v as one GEMM (models/target_pack.py).
        self.verify_cut_pack27b2 = (separate_projections is None and attention.output_gate and setting(
            run_config, 'engine', 'engine_settings.verify_cut_pack27b2', '0') == '1')
        self.pack27b2_members = ('q_proj', 'k_proj', 'v_proj')
        self.verify_cut_w1prod8b = setting(
            run_config, 'engine', 'engine_settings.verify_cut_w1prod8b', '0') == '1'
        if self.verify_cut_w1prod8b:
            from jetspec.engines.jetspec_plus_plus.engine import current_engine
            if (not self.verify_cut_persist8b_attention
                    or current_engine().precision_target != 'mxfp4_w4a8'):
                raise ValueError('verify_cut_w1prod8b requires dense persist8b>=3 and '
                                 'precision_target=mxfp4_w4a8')

    def _verify_glue8b(self, positions, x):
        """The fused N=1 verify path: one glue launch for q/k norm, RoPE and the K/V write."""
        from jetspec.engines.jetspec_plus_plus.kernels import verify_layer8b
        from jetspec.engines.jetspec_plus_plus.models.context import get_context
        attn, context = self.attn, get_context()
        if (context.is_prefill or not attn.jetspec_paged_flashinfer or not attn.speculate
                or context.cu_seqlens_q is None or context.context_lens is None
                or not verify_layer8b.supported(x, attn.k_cache, attn.v_cache, self.num_heads,
                                                self.num_kv_heads, self.head_dim)):
            return None
        if self.verify_cut_w1prod8b:
            qkv = verify_layer8b.w1_linear(self.qkv_proj, x)
        else:
            qkv = self.qkv_proj(x)
        q = verify_layer8b.qkv_glue(
            qkv, positions, self.rotary_emb.cos_sin_cache, self.q_norm.weight, self.k_norm.weight,
            self.q_norm.eps, self.k_norm.eps, attn.k_cache, attn.v_cache, context.slot_mapping,
            self.num_heads, self.num_kv_heads, self.head_dim,
            pdl=self.verify_cut_persist8b_pdl and verify_layer8b.pdl_available())
        if (self.verify_cut_persist8b_attention and context.context_lens.shape[0] == 1
                and context.block_tables is not None and verify_layer8b.pdl_available()):
            if self.verify_cut_persist8b_one_launch:
                from jetspec.engines.jetspec_plus_plus.kernels import verify_attn8b
                return self.o_proj(verify_attn8b.split_attention(
                    q, attn.k_cache, attn.v_cache, context.block_tables[0], context.context_lens,
                    attn.scale, self.num_heads, self.num_kv_heads, self.head_dim))
            return self.o_proj(verify_layer8b.split_attention(
                q, attn.k_cache, attn.v_cache, context.block_tables[0], context.context_lens,
                attn.scale, self.num_heads, self.num_kv_heads, self.head_dim))
        wrapper = attn.flashinfer_graph_wrappers[(context.context_lens.shape[0], q.shape[0])]
        o = wrapper.run(q.view(-1, self.num_heads, self.head_dim), (attn.k_cache, attn.v_cache))
        return self.o_proj(o.reshape(-1, self.num_heads * self.head_dim))

    def _verify_attn27b(self, q, k, v, gate):
        """The gated N=1 verify attention: K/V write, split-KV attention, merge with the output gate."""
        from jetspec.engines.jetspec_plus_plus.kernels import verify_attn27b
        from jetspec.engines.jetspec_plus_plus.models.attention import store_kvcache
        from jetspec.engines.jetspec_plus_plus.models.context import get_context
        attn, context = self.attn, get_context()
        width = q.shape[0]
        query = q.reshape(width, -1)
        if (context.is_prefill or not attn.jetspec_paged_flashinfer or not attn.speculate
                or context.cu_seqlens_q is None or context.context_lens is None
                or context.context_lens.shape[0] != 1 or context.block_tables is None
                or not attn.k_cache.numel() or not verify_attn27b.supported(
                    query, attn.k_cache, attn.v_cache, self.num_heads, self.num_kv_heads, self.head_dim)):
            return None
        store_kvcache(k.reshape(width, self.num_kv_heads, self.head_dim),
                      v.view(width, self.num_kv_heads, self.head_dim), attn.k_cache, attn.v_cache,
                      context.slot_mapping)
        # The merge applies the gate with gate_mul's arithmetic, which is bitwise to the eager gate.
        return self.o_proj(verify_attn27b.split_attention(
            query, attn.k_cache, attn.v_cache, context.block_tables[0], context.context_lens, attn.scale,
            self.num_heads, self.num_kv_heads, self.head_dim, gate=gate))

    def forward(self, positions, x, cos=None, sin=None):
        if self.verify_cut_persist8b and x.is_cuda:
            out = self._verify_glue8b(positions, x)
            if out is not None:
                return out
        if self.separate_projections and self.spec.output_gate:
            width = x.shape[0]
            pack = None
            if self.verify_cut_pack27b2:
                from jetspec.engines.jetspec_plus_plus.models.target_pack import module_pack
                pack = module_pack(self, self.pack27b2_members)
            if pack is not None:
                if getattr(x, '_verify_w1', None) is not None:
                    from jetspec.engines.jetspec_plus_plus.kernels.verify_w1 import w1_packed
                    query, key, v = w1_packed(pack, x)
                else:
                    query, key, v = pack(x)
                q, gate = query.view(width, self.num_heads, 2 * self.head_dim).chunk(2, dim=-1)
                q = self.q_norm(q)
                k = self.k_norm(key.view(width, self.num_kv_heads, self.head_dim))
            else:
                q, gate = self.q_proj(x).view(width, self.num_heads, 2 * self.head_dim).chunk(2, dim=-1)
                q = self.q_norm(q)
                k = self.k_norm(self.k_proj(x).view(width, self.num_kv_heads, self.head_dim))
                v = self.v_proj(x)
            q, k = self.rotary_emb(q, k, cos, sin)
            if self.verify_cut_attn27b and x.is_cuda:
                out = self._verify_attn27b(q, k, v, gate)
                if out is not None:
                    return out
            mixed = self.attn(q.reshape(width, -1), k.reshape(width, -1), v)
            if self.verify_cut_mlpglue:
                from jetspec.engines.jetspec_plus_plus.kernels.verify_mlp_glue import gate_mul
                return self.o_proj(gate_mul(mixed, gate))
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
        from jetspec.engines.jetspec_plus_plus.models.config import setting
        self.verify_cut_mlpglue = setting(
            run_config, 'engine', 'engine_settings.verify_cut_mlpglue', '0') == '1'
        # verify_cut_pack27b2: the hybrid target gate|up as one GEMM (models/target_pack.py).
        self.verify_cut_pack27b2 = (separate_projections is None and self.separate_projections and setting(
            run_config, 'engine', 'engine_settings.verify_cut_pack27b2', '0') == '1')
        self.pack27b2_members = ('gate_proj', 'up_proj')
        self.verify_cut_w1prod27b = setting(
            run_config, 'engine', 'engine_settings.verify_cut_w1prod27b', '0') == '1'

    def forward(self, x):
        if self.separate_projections:
            pack = None
            if self.verify_cut_pack27b2:
                from jetspec.engines.jetspec_plus_plus.models.target_pack import module_pack
                pack = module_pack(self, self.pack27b2_members)
            if pack is not None:
                if getattr(x, '_verify_w1', None) is not None:
                    from jetspec.engines.jetspec_plus_plus.kernels.verify_w1 import w1_packed
                    gate, up = w1_packed(pack, x)
                else:
                    gate, up = pack(x)
                if self.verify_cut_w1prod27b and getattr(x, '_verify_w1', None) is not None:
                    from jetspec.engines.jetspec_plus_plus.kernels.verify_pack_glue import silu_mul_rows
                    from jetspec.engines.jetspec_plus_plus.kernels.verify_w1 import activation_buffers, w1_linear
                    return w1_linear(self.down_proj, silu_mul_rows(
                        gate, up, activation=activation_buffers(self.down_proj, x.shape[0])))
                if self.verify_cut_mlpglue:
                    from jetspec.engines.jetspec_plus_plus.kernels.verify_pack_glue import silu_mul_rows
                    return self.down_proj(silu_mul_rows(gate, up))
                return self.down_proj(F.silu(gate) * up)
            gate = self.gate_proj(x)
            if self.verify_cut_mlpglue and not hasattr(self, 'act_fn'):
                from jetspec.engines.jetspec_plus_plus.kernels.verify_mlp_glue import silu_mul
                return self.down_proj(silu_mul(gate, self.up_proj(x)))
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
        from jetspec.engines.jetspec_plus_plus.models.config import setting
        # verify_cut_norm27b: every after-weight norm of the layer as one launch, residual adds fused in.
        self.verify_cut_norm27b = self.explicit_residual and setting(
            run_config, 'engine', 'engine_settings.verify_cut_norm27b', '0') == '1'
        self.verify_cut_w1prod27b = setting(
            run_config, 'engine', 'engine_settings.verify_cut_w1prod27b', '0') == '1'
        if self.verify_cut_w1prod27b:
            from jetspec.engines.jetspec_plus_plus.engine import current_engine
            if not self.verify_cut_norm27b or current_engine().precision_target != 'mxfp4_w4a8':
                raise ValueError('w1prod27b requires norm27b and precision_target=mxfp4_w4a8')
        self.norm27b_defer = False  # set by the model shell: return (x, mlp out) for the next norm
        if self.verify_cut_norm27b:
            self.input_layernorm.norm27b = self.post_attention_layernorm.norm27b = True
            attention = getattr(self, 'self_attn', None)
            for name in ('q_norm', 'k_norm'):
                if attention is not None and isinstance(getattr(attention, name, None), RMSNorm):
                    getattr(attention, name).norm27b = True

    def forward(self, positions, hidden, residual=None, *, prefill=False, verify=False, taps=None):
        if self.explicit_residual:
            # Preserve the F5 (hidden, cos, sin) calling convention during migration.
            x, cos, sin = positions, hidden, residual
            if self.verify_cut_norm27b:
                return self._norm27b_forward(x, cos, sin, prefill, verify)
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
            hidden, residual = self._add_norm(self.input_layernorm, hidden, residual, prefill)
            if taps is not None:
                # This residual is the previous layer's tap: the same fp32 add, one bf16 rounding.
                taps.append(residual)
        hidden = (self.linear_attn(hidden, prefill=prefill, verify=verify)
                  if hasattr(self, 'linear_attn') else self.self_attn(positions, hidden))
        hidden, residual = self._add_norm(self.post_attention_layernorm, hidden, residual, prefill)
        if self._pdl_glue(hidden, prefill) and not self.mlp.separate_projections:
            if self.self_attn.verify_cut_w1prod8b:
                from jetspec.engines.jetspec_plus_plus.kernels.verify_layer8b import activation_buffers, silu_mul, w1_linear
                gate_up = w1_linear(self.mlp.gate_up_proj, hidden)
                # Keep this producer separate so a gate_up SwiGLU epilogue can replace it.
                activation = activation_buffers(self.mlp.down_proj, hidden.shape[0])
                return w1_linear(self.mlp.down_proj, silu_mul(gate_up, activation=activation)), residual
            from jetspec.engines.jetspec_plus_plus.kernels.verify_layer8b import silu_mul
            return self.mlp.down_proj(silu_mul(self.mlp.gate_up_proj(hidden))), residual
        return self.mlp(hidden), residual

    def _norm27b_forward(self, x, cos, sin, prefill, verify):
        """verify_cut_norm27b: the explicit-residual layer with add+norm launches.

        ``x`` is the hidden rows, or the previous layer's deferred (x, mlp out) pair. With
        ``norm27b_defer`` the layer returns its own pair, so the next norm adds it in one launch.
        """
        from jetspec.engines.jetspec_plus_plus.kernels import verify_norm27b
        input_activation = post_activation = None
        source = x[0] if isinstance(x, tuple) else x
        if (self.verify_cut_w1prod27b and verify and not prefill and source.dim() == 2
                and verify_norm27b.supported(source, self.input_layernorm)):
            from jetspec.engines.jetspec_plus_plus.kernels.verify_w1 import packed_activation
            mixer = self.linear_attn if hasattr(self, 'linear_attn') else self.self_attn
            input_activation = packed_activation(mixer, source.shape[0])
            post_activation = packed_activation(self.mlp, source.shape[0])
        if isinstance(x, tuple):
            x, normed = verify_norm27b.add_rms_norm(
                x[0], x[1], self.input_layernorm, activation=input_activation)
        elif input_activation is not None:
            normed = verify_norm27b.rms_norm(x, self.input_layernorm, activation=input_activation)
        else:
            normed = self.input_layernorm(x)
        mixed = (self.linear_attn(normed, prefill=prefill, verify=verify)
                 if hasattr(self, 'linear_attn') else self.self_attn(None, normed, cos, sin))
        fused = (verify_norm27b.supported(x, self.post_attention_layernorm) and x.dim() == 2
                 and mixed.shape == x.shape and mixed.stride(-1) == 1 and mixed.dtype == x.dtype)
        if fused:
            x, normed = verify_norm27b.add_rms_norm(
                x, mixed, self.post_attention_layernorm, activation=post_activation)
        else:
            x = x + mixed
            normed = self.post_attention_layernorm(x)
        out = self.mlp(normed)
        if fused and self.norm27b_defer and out.shape == x.shape and out.stride(-1) == 1:
            return x, out
        return x + out

    def _pdl_glue(self, hidden, prefill):
        """verify_cut_persist8b=2, 3 or 4: the add+norm and SiLU-mul glue as PDL Triton launches in verify."""
        if prefill or not hidden.is_cuda or hidden.dtype != torch.bfloat16 or hidden.dim() != 2:
            return False
        attn = getattr(self, 'self_attn', None)
        if attn is None or not attn.verify_cut_persist8b_pdl:
            return False
        from jetspec.engines.jetspec_plus_plus.kernels.verify_layer8b import pdl_available
        return pdl_available()

    def _add_norm(self, norm, hidden, residual, prefill):
        if norm.offset or norm.head or not self._pdl_glue(hidden, prefill):
            return norm(hidden, residual)
        from jetspec.engines.jetspec_plus_plus.kernels.verify_layer8b import add_norm
        if self.self_attn.verify_cut_w1prod8b:
            from jetspec.engines.jetspec_plus_plus.kernels.verify_layer8b import activation_buffers
            linear = (self.self_attn.qkv_proj if norm is self.input_layernorm
                      else self.mlp.gate_up_proj)
            return add_norm(hidden, residual, norm.weight, norm.eps,
                            activation=activation_buffers(linear, hidden.shape[0]))
        return add_norm(hidden, residual, norm.weight, norm.eps)


HybridDecoderLayer = DecoderLayer
