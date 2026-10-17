"""One target shell, with checkpoint arithmetic selected by ModelSpec."""
from types import SimpleNamespace

import torch
from torch import nn

from jetspec.engines.jetspec_plus_plus.models.config import setting
from jetspec.engines.jetspec_plus_plus.models.context import get_context
from jetspec.engines.jetspec_plus_plus.models.embed_head import VocabParallelEmbedding
from jetspec.engines.jetspec_plus_plus.models.forward_context import get_forward_context
from jetspec.engines.jetspec_plus_plus.models.target_layers import DecoderLayer, RMSNorm
from jetspec.engines.jetspec_plus_plus.models.target_linear import TargetLinear
from jetspec.engines.jetspec_plus_plus.models.registry import model_spec


PACKED_MODULES = {
    'q_proj': ('qkv_proj', 'q'), 'k_proj': ('qkv_proj', 'k'),
    'v_proj': ('qkv_proj', 'v'), 'gate_proj': ('gate_up_proj', 0),
    'up_proj': ('gate_up_proj', 1),
}

JETSPEC_TARGET_LAYER_IDS = (1, 9, 17, 25, 33)


def layer_config(spec, config):
    """Adapt existing mixers without letting them rediscover checkpoint identity."""
    values = dict(hidden_size=spec.hidden_size, intermediate_size=spec.intermediate_size,
                  rms_norm_eps=spec.rms_norm_eps, num_attention_heads=spec.attention.q_heads,
                  num_key_value_heads=spec.attention.kv_heads, head_dim=spec.attention.head_dim,
                  max_position_embeddings=spec.max_position_embeddings, rope_theta=spec.rope_theta,
                  rope_scaling=spec.rope_scaling, rope_parameters=spec.rope_parameters,
                  hidden_act=getattr(config, 'hidden_act', 'silu'),
                  attention_bias=getattr(config, 'attention_bias', False),
                  torch_dtype=getattr(config, 'torch_dtype', torch.get_default_dtype()))
    if spec.gdn is not None:
        values.update(linear_num_key_heads=spec.gdn.k_heads, linear_num_value_heads=spec.gdn.v_heads,
                      linear_key_head_dim=spec.gdn.k_dim, linear_value_head_dim=spec.gdn.v_dim,
                      linear_conv_kernel_dim=spec.gdn.conv_kernel, hidden_act=spec.gdn.hidden_act)
    return SimpleNamespace(**values)


class TargetModel(nn.Module):
    def __init__(self, spec, config, *, draft=False, speculate=False, spec_k=1,
                 async_fan_out=1, draft_async=False, tp_group=None, tp_size=1,
                 jetspec_taps=False, jetspec_tap_layer_ids=None, use_eagle=False,
                 eagle_layers=None, run_config=None):
        super().__init__()
        self.spec = spec
        self.jetspec_taps = jetspec_taps
        self.jetspec_tap_layer_ids = tuple(jetspec_tap_layer_ids if jetspec_tap_layer_ids is not None
                                          else spec.taps or (() if spec.has_state else JETSPEC_TARGET_LAYER_IDS))
        self.use_eagle, self.eagle_layers = use_eagle, eagle_layers
        assert not (use_eagle and jetspec_taps), 'EAGLE and JetSpec taps are mutually exclusive'
        if jetspec_taps:
            assert self.jetspec_tap_layer_ids and max(self.jetspec_tap_layer_ids) < spec.num_layers
        if use_eagle:
            assert eagle_layers and max(eagle_layers) < spec.num_layers, 'eagle_layers must be inside target'
        self.embed_tokens = VocabParallelEmbedding(spec.vocab_size, spec.hidden_size,
            draft_async=draft_async, tp_group=tp_group, tp_size=tp_size)
        config = layer_config(spec, config)
        self.layers = nn.ModuleList()
        for kind in spec.layer_kinds:
            layer = DecoderLayer(spec, kind, config=config, draft=draft, speculate=speculate,
                                 spec_k=spec_k, async_fan_out=async_fan_out, draft_async=draft_async,
                                 tp_group=tp_group, tp_size=tp_size, run_config=run_config)
            self.layers.append(layer)
        self.norm = RMSNorm(spec.hidden_size, spec.rms_norm_eps,
                            offset=spec.attention.norm_offset, norm_round=spec.attention.norm_round)
        if spec.attention.norm_round == 'after_weight':
            rotary_dim = spec.attention.rotary_dim
            inv_freq = 1.0 / (spec.rope_theta ** (
                torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim))
            self.register_buffer('inv_freq', inv_freq, persistent=False)

    def reset_gdn_state(self):
        for layer in self.layers:
            if hasattr(layer, 'linear_attn'):
                layer.linear_attn.reset_state()

    def commit_gdn(self, accepted):
        for layer in self.layers:
            if hasattr(layer, 'linear_attn'):
                layer.linear_attn.commit(accepted)

    def forward(self, input_ids, positions):
        prefill = get_context().is_prefill
        hidden = self.embed_tokens(input_ids)
        residual, taps, eagle = None, [], []
        explicit_residual = self.spec.attention.norm_round == 'after_weight'
        if explicit_residual:
            freqs = positions.float()[:, None] * self.inv_freq[None, :]
            repeated = torch.cat((freqs, freqs), dim=-1)
            cos, sin = repeated.cos().to(hidden.dtype), repeated.sin().to(hidden.dtype)
        for index, layer in enumerate(self.layers):
            if self.use_eagle and index in self.eagle_layers:
                eagle.append(hidden if residual is None else hidden + residual)
            if explicit_residual:
                hidden = layer(hidden, cos, sin, prefill=prefill,
                               verify=get_forward_context().verify and not prefill)
            else:
                hidden, residual = layer(positions, hidden, residual, prefill=prefill,
                                         verify=get_forward_context().verify and not prefill)
            if self.jetspec_taps and index in self.jetspec_tap_layer_ids:
                taps.append(hidden if explicit_residual else
                            (hidden.float() + residual.float()).to(hidden.dtype))
        hidden = self.norm(hidden) if explicit_residual else self.norm(hidden, residual)[0]
        if self.jetspec_taps:
            return hidden, torch.stack(taps, dim=1)
        if eagle:
            return hidden, torch.cat(eagle, dim=-1)
        return hidden


class TargetForCausalLM(nn.Module):
    def __init__(self, config, *, draft=False, speculate=False, spec_k=1,
                 async_fan_out=1, draft_async=False, tp_group=None, tp_size=1,
                 jetspec_taps=False, jetspec_tap_layer_ids=None, use_eagle=False,
                 eagle_layers=None, run_config=None):
        super().__init__()
        self.spec = model_spec(config)
        self.run_config = run_config
        self.hybrid_text_only = self.spec.has_state
        self.packed_modules_mapping = {} if self.spec.attention.output_gate else PACKED_MODULES
        if self.spec.has_state and (draft or tp_size != 1 or (draft_async and
                setting(run_config, 'engine', 'engine_settings.hybrid_pair', '0') != '1')):
            raise ValueError('hybrid target currently supports N=1 on one GPU')
        self.model = TargetModel(self.spec, config, draft=draft, speculate=speculate,
            spec_k=spec_k, async_fan_out=async_fan_out, draft_async=draft_async,
            tp_group=tp_group, tp_size=tp_size, jetspec_taps=jetspec_taps,
            jetspec_tap_layer_ids=jetspec_tap_layer_ids, use_eagle=use_eagle,
            eagle_layers=eagle_layers, run_config=run_config)
        self.lm_head = TargetLinear(self.spec.hidden_size, self.spec.vocab_size,
                                   layout='head', tp_group=tp_group, tp_size=tp_size, precision_role='lm_head')
        if self.spec.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def forward(self, input_ids, positions):
        return self.model(input_ids, positions)

    def compute_logits(self, hidden_states, last_only=True):
        return self.lm_head(hidden_states, last_only=last_only)

    def reset_gdn_state(self):
        self.model.reset_gdn_state()

    def commit_gdn(self, accepted):
        self.model.commit_gdn(accepted)
