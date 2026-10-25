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
from jetspec.engines.jetspec_plus_plus.models.target_rotary import RopeRows, rope_table
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
        self.verify_cut_embedgather = not draft and tp_size == 1 and setting(
            run_config, 'engine', 'engine_settings.verify_cut_embedgather', '0') == '1'
        self.verify_cut_tap27b_from_next_addnorm = setting(
            run_config, 'engine', 'engine_settings.verify_cut_tap27b_from_next_addnorm', '0') == '1'
        self.jetspec_taps = jetspec_taps
        self.jetspec_tap_layer_ids = tuple(jetspec_tap_layer_ids if jetspec_tap_layer_ids is not None
                                          else spec.taps or (() if spec.has_state else JETSPEC_TARGET_LAYER_IDS))
        self.use_eagle, self.eagle_layers = use_eagle, eagle_layers
        assert not (use_eagle and jetspec_taps), 'EAGLE and JetSpec taps are mutually exclusive'
        if jetspec_taps:
            assert self.jetspec_tap_layer_ids and max(self.jetspec_tap_layer_ids) < spec.num_layers
        if use_eagle:
            assert eagle_layers and max(eagle_layers) < spec.num_layers, 'eagle_layers must be inside target'
        # Layer i's tap is layer i + 1's input add_norm residual (the final norm after the last layer).
        self.jetspec_taps_from_residual = (
            jetspec_taps and spec.attention.norm_round != 'after_weight'
            and setting(run_config, 'engine', 'engine_settings.verify_cut_glue8b', '0') == '1')
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
        # verify_cut_norm27b: layers hand (x, mlp out) to the next norm, which adds it in one launch.
        if any(getattr(layer, 'verify_cut_norm27b', False) for layer in self.layers):
            self.norm.norm27b = True
            for layer in self.layers:
                layer.norm27b_defer = not use_eagle
        if spec.attention.norm_round == 'after_weight':
            rotary_dim = spec.attention.rotary_dim
            inv_freq = 1.0 / (spec.rope_theta ** (
                torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim))
            self.register_buffer('inv_freq', inv_freq, persistent=False)
            self.verify_cut_grope = setting(
                run_config, 'engine', 'engine_settings.verify_cut_grope', '0') == '1'
            if self.verify_cut_grope:
                # Built on the serving device so the gathered rows equal the per-forward values.
                cos_table, sin_table = rope_table(inv_freq, spec.max_position_embeddings,
                                                  self.embed_tokens.weight.dtype)
                self.register_buffer('rope_cos', cos_table, persistent=False)
                self.register_buffer('rope_sin', sin_table, persistent=False)
                self.rope_device = inv_freq.device.type

    def reset_gdn_state(self):
        for layer in self.layers:
            if hasattr(layer, 'linear_attn'):
                layer.linear_attn.reset_state()

    def commit_gdn(self, accepted):
        for layer in self.layers:
            if hasattr(layer, 'linear_attn'):
                layer.linear_attn.commit(accepted)

    def forward(self, input_ids, positions, tap_out=None):
        prefill = get_context().is_prefill
        if self.verify_cut_embedgather and get_forward_context().verify and not prefill:
            hidden = self.embed_tokens(input_ids, parallel_gather=True)
        else:
            hidden = self.embed_tokens(input_ids)
        residual, taps, eagle = None, [], []
        explicit_residual = self.spec.attention.norm_round == 'after_weight'
        deferred_taps = (self.verify_cut_tap27b_from_next_addnorm and explicit_residual
                         and self.norm.norm27b and self.jetspec_taps
                         and hidden.is_cuda and get_forward_context().verify and not prefill)
        if explicit_residual and self.verify_cut_grope:
            if (self.rope_cos.dtype != hidden.dtype or positions.device.type != self.rope_device
                    or self.inv_freq.dtype != torch.float32):
                raise ValueError('verify_cut_grope tables were built for another dtype or device')
            cos, sin = RopeRows(self.rope_cos, self.rope_sin, positions), None
        elif explicit_residual:
            freqs = positions.float()[:, None] * self.inv_freq[None, :]
            repeated = torch.cat((freqs, freqs), dim=-1)
            cos, sin = repeated.cos().to(hidden.dtype), repeated.sin().to(hidden.dtype)
        for index, layer in enumerate(self.layers):
            if self.use_eagle and index in self.eagle_layers:
                eagle.append(hidden if residual is None else hidden + residual)
            if explicit_residual:
                hidden = layer(hidden, cos, sin, prefill=prefill,
                               verify=get_forward_context().verify and not prefill,
                               **({'taps': taps} if deferred_taps and index - 1 in self.jetspec_tap_layer_ids else {}))
                if isinstance(hidden, tuple) and (index == len(self.layers) - 1 and not (
                        self.norm.norm27b) or self.jetspec_taps and index in self.jetspec_tap_layer_ids
                        and not deferred_taps):
                    hidden = hidden[0] + hidden[1]
            elif self.jetspec_taps_from_residual:
                hidden, residual = layer(positions, hidden, residual, prefill=prefill,
                                         verify=get_forward_context().verify and not prefill,
                                         taps=taps if index - 1 in self.jetspec_tap_layer_ids else None)
                continue
            else:
                hidden, residual = layer(positions, hidden, residual, prefill=prefill,
                                         verify=get_forward_context().verify and not prefill)
            if self.jetspec_taps and index in self.jetspec_tap_layer_ids:
                if not deferred_taps:
                    taps.append(hidden if explicit_residual else
                                (hidden.float() + residual.float()).to(hidden.dtype))
                elif index == len(self.layers) - 1 and not isinstance(hidden, tuple):
                    taps.append(hidden)
        if self.jetspec_taps_from_residual:
            hidden, residual = self.norm(hidden, residual)
            if len(self.layers) - 1 in self.jetspec_tap_layer_ids:
                taps.append(residual)
        else:
            if isinstance(hidden, tuple):
                from jetspec.engines.jetspec_plus_plus.kernels.verify_norm27b import add_rms_norm
                total, hidden = add_rms_norm(hidden[0], hidden[1], self.norm)
                if deferred_taps and len(self.layers) - 1 in self.jetspec_tap_layer_ids:
                    taps.append(total)
            else:
                hidden = self.norm(hidden) if explicit_residual else self.norm(hidden, residual)[0]
        if self.jetspec_taps:
            if tap_out is not None:
                return hidden, torch.stack(taps, dim=1, out=tap_out)
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

    def forward(self, input_ids, positions, tap_out=None):
        if tap_out is not None:
            return self.model(input_ids, positions, tap_out=tap_out)
        return self.model(input_ids, positions)

    def compute_logits(self, hidden_states, last_only=True):
        return self.lm_head(hidden_states, last_only=last_only)

    def reset_gdn_state(self):
        self.model.reset_gdn_state()

    def commit_gdn(self, accepted):
        self.model.commit_gdn(accepted)
