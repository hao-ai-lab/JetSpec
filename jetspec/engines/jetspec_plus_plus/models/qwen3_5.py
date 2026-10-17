"""Text-only Qwen3.5 hybrid target for the N=1 serving path."""

from __future__ import annotations

from jetspec.engines.jetspec_plus_plus.profiles import kernel_enabled, kernel_value

import os
import torch
from jetspec.engines.jetspec_plus_plus.models.config import setting
from torch import nn
from torch.nn import functional as F

from jetspec.engines.jetspec_plus_plus.models.embed_head import ParallelLMHead
from jetspec.engines.jetspec_plus_plus.models.embed_head import VocabParallelEmbedding
from jetspec.engines.jetspec_plus_plus.models.context import get_context


class FP32ComputeLinear(nn.Linear):
    """Keep source BF16 values exact while evaluating the projection in FP32."""

    def __init__(self, input_size, output_size, *, bias):
        super().__init__(input_size, output_size, bias=bias, dtype=torch.bfloat16)
        for parameter in (self.weight, self.bias):
            if parameter is not None:
                parameter.weight_loader = self.weight_loader

    @staticmethod
    def weight_loader(parameter, source):
        if source.dtype == torch.bfloat16:
            parameter.data.copy_(source)
        else:
            # A checkpoint with more precise values must never be quantized.
            parameter.data = source.to(device=parameter.device, dtype=torch.float32)

    def forward(self, x):
        if x.dtype != torch.float32:
            raise ValueError("FP32 projection storage requires FP32 activations")
        bias = None if self.bias is None else self.bias.float()
        return F.linear(x, self.weight.float(), bias)


def hybrid_linear(config, input_size, output_size, *, bias=False, run_config=None):
    if (setting(run_config, 'engine', 'engine_settings.hybrid_fp32_linear_bf16_storage', '0') == '1'
            and getattr(config, "torch_dtype", torch.get_default_dtype()) == torch.float32):
        return FP32ComputeLinear(input_size, output_size, bias=bias)
    return nn.Linear(input_size, output_size, bias=bias)


class HybridRMSNorm(nn.Module):
    def __init__(self, width: int, eps: float):
        from jetspec.engines.jetspec_plus_plus.profiles import kernel_enabled
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(width))
        self.eps = eps
        self.fused = kernel_enabled('hybrid_fused_norm', '1')
        if self.fused:
            from fla.modules.layernorm import rms_norm
            self._rms_norm = rms_norm
            self._fused_weight = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.fused and x.is_cuda:
            weight = self._fused_weight
            if weight is None or weight.dtype != x.dtype or weight.device != x.device:
                weight = (1.0 + self.weight.float()).to(x.dtype)
                self._fused_weight = weight
            return self._rms_norm(x, weight, None, eps=self.eps)
        x_float = x.float()
        return (x_float * torch.rsqrt(x_float.square().mean(-1, keepdim=True) + self.eps)
                * (1.0 + self.weight.float())).to(x.dtype)


def _torch_delta_rule(q, k, v, g, beta, initial):
    """Small-shape CPU reference with the FLA recurrent state convention."""
    q = q.float()
    k = k.float()
    q = q * torch.rsqrt(q.square().sum(-1, keepdim=True) + 1e-6) * (q.shape[-1] ** -0.5)
    k = k * torch.rsqrt(k.square().sum(-1, keepdim=True) + 1e-6)
    v, g, beta = v.float(), g.float(), beta.float()
    state = initial.float()
    outputs = []
    for step in range(q.shape[1]):
        state = state * g[:, step].exp()[..., None, None]
        delta = (v[:, step] - (state * k[:, step, :, :, None]).sum(-2)) * beta[:, step, :, None]
        state = state + k[:, step, :, :, None] * delta[:, :, None, :]
        outputs.append((state * q[:, step, :, :, None]).sum(-2))
    return torch.stack(outputs, dim=1).to(v.dtype), state


class HybridGDN(nn.Module):
    def __init__(self, config, max_verify_width: int, *, run_config=None):
        super().__init__()
        self.run_config = run_config
        hidden = config.hidden_size
        self.key_heads = config.linear_num_key_heads
        self.value_heads = config.linear_num_value_heads
        self.key_dim = config.linear_key_head_dim
        self.value_dim = config.linear_value_head_dim
        self.conv_kernel = config.linear_conv_kernel_dim
        self.fast_snapshot = setting(self.run_config, 'engine', 'engine_settings.hybrid_fast_snapshot', '0') == "1"
        key_width = self.key_heads * self.key_dim
        value_width = self.value_heads * self.value_dim
        conv_width = 2 * key_width + value_width
        self.key_width = key_width
        self.value_width = value_width
        self.in_proj_qkv = hybrid_linear(config, hidden, conv_width, run_config=run_config)
        self.in_proj_z = hybrid_linear(config, hidden, value_width, run_config=run_config)
        self.in_proj_b = hybrid_linear(config, hidden, self.value_heads, run_config=run_config)
        self.in_proj_a = hybrid_linear(config, hidden, self.value_heads, run_config=run_config)
        self.conv1d = nn.Conv1d(conv_width, conv_width, self.conv_kernel,
                                groups=conv_width, padding=self.conv_kernel - 1, bias=False)
        self.dt_bias = nn.Parameter(torch.ones(self.value_heads))
        self.A_log = nn.Parameter(torch.zeros(self.value_heads))
        self.out_proj = hybrid_linear(config, value_width, hidden, run_config=run_config)
        self.eps = config.rms_norm_eps
        try:
            from fla.modules import FusedRMSNormGated
            from fla.ops.gated_delta_rule import (chunk_gated_delta_rule,
                                                   fused_recurrent_gated_delta_rule)
        except ImportError:
            self._chunk = self._recurrent = None
            self.norm = HybridRMSNorm(self.value_dim, self.eps)
            with torch.no_grad():
                self.norm.weight.fill_(1)
        else:
            self._chunk = chunk_gated_delta_rule
            self._recurrent = fused_recurrent_gated_delta_rule
            self.norm = FusedRMSNormGated(self.value_dim, eps=self.eps,
                                          activation=config.hidden_act)
        state_shape = (1, self.value_heads, self.key_dim, self.value_dim)
        conv_shape = (1, conv_width, self.conv_kernel)
        self.register_buffer("committed_recurrent", torch.zeros(state_shape, dtype=torch.float32), persistent=False)
        self.register_buffer("committed_conv", torch.zeros(conv_shape), persistent=False)
        self.register_buffer("scratch_recurrent", torch.zeros((max_verify_width + 1,) + state_shape,
                                                              dtype=torch.float32),
                             persistent=False)
        self.register_buffer("scratch_conv", torch.zeros((max_verify_width + 1,) + conv_shape),
                             persistent=False)

    def _apply(self, fn):
        super()._apply(fn)
        self.committed_recurrent = self.committed_recurrent.float()
        self.scratch_recurrent = self.scratch_recurrent.float()
        return self

    def reset_state(self):
        self.committed_recurrent.zero_()
        self.committed_conv.zero_()

    def _delta(self, q, k, v, g, beta, initial, *, chunk):
        if q.is_cuda:
            if self._recurrent is None:
                raise RuntimeError("hybrid GDN on CUDA requires fla gated delta rule kernels")
            kernel = self._chunk if chunk else self._recurrent
            return kernel(q, k, v, g=g, beta=beta, initial_state=initial,
                          output_final_state=True, use_qk_l2norm_in_kernel=True)
        return _torch_delta_rule(q, k, v, g, beta, initial)

    def forward(self, hidden: torch.Tensor, *, prefill: bool, verify: bool) -> torch.Tensor:
        width = hidden.shape[0]
        x = hidden.unsqueeze(0)
        raw = self.in_proj_qkv(x).transpose(1, 2)
        z = self.in_proj_z(x).reshape(-1, self.value_dim)
        beta = self.in_proj_b(x).sigmoid()
        a = self.in_proj_a(x)
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)

        conv_input = raw if prefill else torch.cat((self.committed_conv, raw), dim=-1)
        convolved = F.silu(self.conv1d(conv_input)[:, :, :conv_input.shape[-1]])
        convolved = convolved[:, :, -width:].transpose(1, 2)
        q, k, v = convolved.split((self.key_width, self.key_width, self.value_width), dim=-1)
        q = q.reshape(1, width, self.key_heads, self.key_dim)
        k = k.reshape(1, width, self.key_heads, self.key_dim)
        v = v.reshape(1, width, self.value_heads, self.value_dim)
        repeat = self.value_heads // self.key_heads
        if repeat > 1:
            q = q.repeat_interleave(repeat, dim=2)
            k = k.repeat_interleave(repeat, dim=2)
        initial = None if prefill else self.committed_recurrent
        if initial is None:
            initial = torch.zeros_like(self.committed_recurrent)
        output, final = self._delta(q, k, v, g, beta, initial, chunk=prefill)

        if verify:
            if width >= self.scratch_recurrent.shape[0]:
                raise ValueError("verify width exceeds the GDN scratch capacity")
            self.scratch_recurrent[0].copy_(self.committed_recurrent)
            self.scratch_conv[0].copy_(self.committed_conv)
            if self.fast_snapshot and hidden.is_cuda:
                from jetspec.engines.jetspec_plus_plus.models.hybrid_snapshot import snapshot_states
                snapshot_states(k, v, g, beta, self.committed_recurrent,
                                self.scratch_recurrent)
            else:
                state = self.committed_recurrent
                for step in range(width):
                    _, state = self._delta(q[:, step:step + 1], k[:, step:step + 1],
                                           v[:, step:step + 1], g[:, step:step + 1],
                                           beta[:, step:step + 1], state, chunk=False)
                    self.scratch_recurrent[step + 1].copy_(state)
                    state = self.scratch_recurrent[step + 1]
            for step in range(width):
                self.scratch_conv[step + 1].copy_(
                    torch.cat((self.committed_conv, raw[:, :, :step + 1]), dim=-1)
                    [:, :, -self.conv_kernel:])
        else:
            self.committed_recurrent.copy_(final)
            self.committed_conv.copy_(
                F.pad(conv_input, (max(0, self.conv_kernel - conv_input.shape[-1]), 0))
                [:, :, -self.conv_kernel:])

        output = output.reshape(-1, self.value_dim)
        if self._recurrent is None or not output.is_cuda:
            x_float = output.float()
            output = (x_float * torch.rsqrt(x_float.square().mean(-1, keepdim=True) + self.eps)
                      * self.norm.weight.float() * F.silu(z.float())).to(output.dtype)
        else:
            output = self.norm(output, z)
        return self.out_proj(output.reshape(width, self.value_width))

    def commit(self, accepted: torch.Tensor):
        index = accepted.to(torch.long).reshape(1)
        self.committed_recurrent.copy_(self.scratch_recurrent.index_select(0, index)[0])
        self.committed_conv.copy_(self.scratch_conv.index_select(0, index)[0])


class HybridFullAttention(nn.Module):
    def __init__(self, config, *, speculate: bool, spec_k: int, run_config=None):
        super().__init__()
        from jetspec.engines.jetspec_plus_plus.models.attention import Attention
        hidden = config.hidden_size
        heads = config.num_attention_heads
        kv_heads = config.num_key_value_heads
        dim = config.head_dim
        self.heads, self.kv_heads, self.dim = heads, kv_heads, dim
        self.q_proj = hybrid_linear(config, hidden, 2 * heads * dim, bias=config.attention_bias, run_config=run_config)
        self.k_proj = hybrid_linear(config, hidden, kv_heads * dim, bias=config.attention_bias, run_config=run_config)
        self.v_proj = hybrid_linear(config, hidden, kv_heads * dim, bias=config.attention_bias, run_config=run_config)
        self.o_proj = hybrid_linear(config, heads * dim, hidden, bias=config.attention_bias, run_config=run_config)
        self.q_norm = HybridRMSNorm(dim, config.rms_norm_eps)
        self.k_norm = HybridRMSNorm(dim, config.rms_norm_eps)
        self.attn = Attention(heads, dim, dim ** -0.5, kv_heads,
                              speculate=speculate, K=spec_k)
        self.rotary_dim = int(dim * config.rope_parameters["partial_rotary_factor"])

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        width = x.shape[0]
        q, gate = self.q_proj(x).view(width, self.heads, 2 * self.dim).chunk(2, dim=-1)
        q = self.q_norm(q)
        k = self.k_norm(self.k_proj(x).view(width, self.kv_heads, self.dim))
        v = self.v_proj(x)

        def rotate(tensor):
            rot = tensor[..., :self.rotary_dim]
            half = self.rotary_dim // 2
            shifted = torch.cat((-rot[..., half:], rot[..., :half]), dim=-1)
            changed = rot * cos[:, None, :] + shifted * sin[:, None, :]
            return torch.cat((changed, tensor[..., self.rotary_dim:]), dim=-1)

        q = rotate(q).reshape(width, -1)
        k = rotate(k).reshape(width, -1)
        mixed = self.attn(q, k, v)
        return self.o_proj(mixed * torch.sigmoid(gate.reshape(width, -1)))


class HybridDecoderLayer(nn.Module):
    def __init__(self, config, layer_type: str, *, speculate: bool,
                 spec_k: int, max_verify_width: int, run_config=None):
        super().__init__()
        self.run_config = run_config
        if layer_type == "linear_attention":
            self.linear_attn = HybridGDN(config, max_verify_width, run_config=run_config)
        else:
            self.self_attn = HybridFullAttention(config, speculate=speculate, spec_k=spec_k, run_config=run_config)
        hidden = config.hidden_size
        intermediate = config.intermediate_size
        self.input_layernorm = HybridRMSNorm(hidden, config.rms_norm_eps)
        self.post_attention_layernorm = HybridRMSNorm(hidden, config.rms_norm_eps)
        self.mlp = nn.Module()
        self.mlp.gate_proj = hybrid_linear(config, hidden, intermediate, run_config=run_config)
        self.mlp.up_proj = hybrid_linear(config, hidden, intermediate, run_config=run_config)
        self.mlp.down_proj = hybrid_linear(config, intermediate, hidden, run_config=run_config)

    def forward(self, x, cos, sin, *, prefill: bool, verify: bool):
        normed = self.input_layernorm(x)
        if hasattr(self, "linear_attn"):
            mixed = self.linear_attn(normed, prefill=prefill, verify=verify)
        else:
            mixed = self.self_attn(normed, cos, sin)
        x = x + mixed
        normed = self.post_attention_layernorm(x)
        return x + self.mlp.down_proj(F.silu(self.mlp.gate_proj(normed))
                                        * self.mlp.up_proj(normed))


class HybridTextModel(nn.Module):
    def __init__(self, config, *, speculate: bool, spec_k: int,
                 jetspec_taps: bool, jetspec_tap_layer_ids, run_config=None):
        super().__init__()
        self.run_config = run_config
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            HybridDecoderLayer(config, kind, speculate=speculate, spec_k=spec_k,
                               max_verify_width=spec_k + 1, run_config=run_config)
            for kind in config.layer_types)
        self.norm = HybridRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.jetspec_taps = jetspec_taps
        self.tap_layer_ids = tuple(jetspec_tap_layer_ids or ())
        self.gdn_verify_mode = False
        rotary_dim = int(config.head_dim * config.rope_parameters["partial_rotary_factor"])
        inv_freq = 1.0 / (config.rope_theta **
                          (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def reset_gdn_state(self):
        for layer in self.layers:
            if hasattr(layer, "linear_attn"):
                layer.linear_attn.reset_state()

    def commit_gdn(self, accepted: torch.Tensor):
        for layer in self.layers:
            if hasattr(layer, "linear_attn"):
                layer.linear_attn.commit(accepted)

    def forward(self, input_ids, positions):
        prefill = get_context().is_prefill
        if prefill:
            self.reset_gdn_state()
        x = self.embed_tokens(input_ids)
        freqs = positions.float()[:, None] * self.inv_freq[None, :]
        repeated = torch.cat((freqs, freqs), dim=-1)
        cos, sin = repeated.cos().to(x.dtype), repeated.sin().to(x.dtype)
        taps = []
        for index, layer in enumerate(self.layers):
            x = layer(x, cos, sin, prefill=prefill,
                      verify=self.gdn_verify_mode and not prefill)
            if self.jetspec_taps and index in self.tap_layer_ids:
                taps.append(x)
        x = self.norm(x)
        if self.jetspec_taps:
            return x, torch.stack(taps, dim=1)
        return x


class Qwen3_5TextForCausalLM(nn.Module):
    packed_modules_mapping = {}
    hybrid_text_only = True

    def __init__(self, config, *, draft=False, speculate=False, spec_k=1,
                 async_fan_out=1, draft_async=False, tp_group=None, tp_size=1,
                 jetspec_taps=False, jetspec_tap_layer_ids=None, run_config=None):
        super().__init__()
        self.run_config = run_config
        if draft or tp_size != 1 or (draft_async and setting(self.run_config, 'engine', 'engine_settings.hybrid_pair', '0') != "1"):
            raise ValueError("hybrid target currently supports N=1 on one GPU")
        self.model = HybridTextModel(config, speculate=speculate, spec_k=spec_k,
                                     jetspec_taps=jetspec_taps,
                                     jetspec_tap_layer_ids=jetspec_tap_layer_ids, run_config=run_config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)

    def forward(self, input_ids, positions):
        return self.model(input_ids, positions)

    def compute_logits(self, hidden_states, last_only=True):
        return self.lm_head(hidden_states, last_only=last_only)

    def reset_gdn_state(self):
        self.model.reset_gdn_state()

    def commit_gdn(self, accepted: torch.Tensor):
        self.model.commit_gdn(accepted)
