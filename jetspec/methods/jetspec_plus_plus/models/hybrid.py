"""Ported from the JetSpec++ reference implementation.

The pinned tail and row windows use the reference implementation's torch
recurrence and SDPA window math. State belongs to each causal sequence, including rollback
of rejected GDN tokens. No serving runtime or optional kernel is imported.
"""
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from .adapter import HybridC1RMSNorm
from .decoder import load_target_tensors


def text_config(raw):
    raw = raw.get('text_config', raw)
    if raw.get('model_type') != 'qwen3_5_text':
        raise ValueError('hybrid target requires qwen3_5_text')
    config = SimpleNamespace(**raw)
    if (len(config.layer_types) != config.num_hidden_layers
            or set(config.layer_types) != {'linear_attention', 'full_attention'}
            or config.layer_types[-2:] != ['linear_attention', 'full_attention']):
        raise ValueError('hybrid target requires a GDN/full-attention tail')
    rope = config.rope_parameters
    if rope.get('rope_type') != 'default':
        raise ValueError('hybrid target requires default RoPE')
    config.rope_theta = rope['rope_theta']
    config._attn_implementation = 'sdpa'
    return config


def delta_rule(q, k, v, g, beta, initial):
    """The reference implementation's fp32 recurrent-state convention, including Q/K epsilon."""
    q, k = q.float(), k.float()
    q = q * torch.rsqrt(q.square().sum(-1, keepdim=True) + 1e-6) * (q.shape[-1] ** -0.5)
    k = k * torch.rsqrt(k.square().sum(-1, keepdim=True) + 1e-6)
    v, g, beta = v.float(), g.float(), beta.float()
    state, outputs = initial.float(), []
    for step in range(q.shape[1]):
        state = state * g[:, step].exp()[..., None, None]
        delta = (v[:, step] - (state * k[:, step, :, :, None]).sum(-2)) * beta[:, step, :, None]
        state = state + k[:, step, :, :, None] * delta[:, :, None, :]
        outputs.append((state * q[:, step, :, :, None]).sum(-2))
    return torch.stack(outputs, dim=1), state


class HybridGDN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.key_heads, self.value_heads = config.linear_num_key_heads, config.linear_num_value_heads
        self.key_dim, self.value_dim = config.linear_key_head_dim, config.linear_value_head_dim
        self.conv_kernel = config.linear_conv_kernel_dim
        self.key_width, self.value_width = self.key_heads * self.key_dim, self.value_heads * self.value_dim
        width = 2 * self.key_width + self.value_width
        self.in_proj_qkv = nn.Linear(config.hidden_size, width, bias=False)
        self.in_proj_z = nn.Linear(config.hidden_size, self.value_width, bias=False)
        self.in_proj_b = nn.Linear(config.hidden_size, self.value_heads, bias=False)
        self.in_proj_a = nn.Linear(config.hidden_size, self.value_heads, bias=False)
        self.conv1d = nn.Conv1d(width, width, self.conv_kernel, groups=width,
                                padding=self.conv_kernel - 1, bias=False)
        self.dt_bias = nn.Parameter(torch.ones(self.value_heads))
        self.A_log = nn.Parameter(torch.zeros(self.value_heads))
        self.out_proj = nn.Linear(self.value_width, config.hidden_size, bias=False)
        self.norm = nn.Module()
        self.norm.weight = nn.Parameter(torch.ones(self.value_dim))
        self.eps = config.rms_norm_eps

    def empty_state(self, like):
        return (torch.zeros(1, self.value_heads, self.key_dim, self.value_dim,
                            device=like.device, dtype=torch.float32),
                like.new_zeros(1, self.conv1d.in_channels, self.conv_kernel))

    def forward(self, hidden, state=None):
        recurrent, conv = self.empty_state(hidden) if state is None else state
        if recurrent.dtype != torch.float32:
            raise ValueError('decoder GDN recurrent state must stay fp32')
        width = hidden.shape[0]
        x = hidden.unsqueeze(0)
        raw = self.in_proj_qkv(x).transpose(1, 2)
        z = self.in_proj_z(x).reshape(-1, self.value_dim)
        beta, a = self.in_proj_b(x).sigmoid(), self.in_proj_a(x)
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
        conv_input = torch.cat((conv.to(raw.dtype), raw), dim=-1)
        convolved = F.silu(self.conv1d(conv_input)[:, :, :conv_input.shape[-1]])
        convolved = convolved[:, :, -width:].transpose(1, 2)
        q, k, v = convolved.split((self.key_width, self.key_width, self.value_width), dim=-1)
        q = q.reshape(1, width, self.key_heads, self.key_dim)
        k = k.reshape(1, width, self.key_heads, self.key_dim)
        v = v.reshape(1, width, self.value_heads, self.value_dim)
        repeat = self.value_heads // self.key_heads
        if repeat > 1:
            q, k = q.repeat_interleave(repeat, 2), k.repeat_interleave(repeat, 2)
        output, final = delta_rule(q, k, v, g, beta, recurrent)
        output = output.reshape(-1, self.value_dim)
        output = (output * torch.rsqrt(output.square().mean(-1, keepdim=True) + self.eps)
                  * self.norm.weight.float() * F.silu(z.float()))
        # The torch delta reference is fp32; projection reads the checkpoint dtype.
        projected = self.out_proj(output.reshape(width, self.value_width).to(hidden.dtype))
        return projected, (final, conv_input[:, :, -self.conv_kernel:].clone())


class HybridAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads, self.kv_heads, self.dim = config.num_attention_heads, config.num_key_value_heads, config.head_dim
        self.rotary_dim = int(self.dim * config.rope_parameters['partial_rotary_factor'])
        self.theta = config.rope_parameters['rope_theta']
        self.q_proj = nn.Linear(config.hidden_size, 2 * self.heads * self.dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, self.kv_heads * self.dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, self.kv_heads * self.dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.heads * self.dim, config.hidden_size, bias=config.attention_bias)
        self.q_norm = HybridC1RMSNorm(self.dim, config.rms_norm_eps)
        self.k_norm = HybridC1RMSNorm(self.dim, config.rms_norm_eps)

    def forward(self, hidden, *, start, kv=None):
        width = hidden.shape[0]
        q, gate = self.q_proj(hidden).view(width, self.heads, 2 * self.dim).chunk(2, -1)
        q = self.q_norm(q)
        key = self.k_norm(self.k_proj(hidden).view(width, self.kv_heads, self.dim))
        value = self.v_proj(hidden).view(width, self.kv_heads, self.dim)
        frequency = 1.0 / (self.theta ** (torch.arange(0, self.rotary_dim, 2,
            device=hidden.device, dtype=torch.float32) / self.rotary_dim))
        angles = (torch.arange(width, device=hidden.device, dtype=torch.float32) + start)[:, None] * frequency[None]
        angles = torch.cat((angles, angles), -1)
        cos, sin = angles.cos().to(q.dtype), angles.sin().to(q.dtype)
        def rotate(tensor):
            rotary = tensor[..., :self.rotary_dim]
            half = self.rotary_dim // 2
            shifted = torch.cat((-rotary[..., half:], rotary[..., :half]), -1)
            return torch.cat((rotary * cos[:, None] + shifted * sin[:, None],
                              tensor[..., self.rotary_dim:]), -1)
        q, key = rotate(q), rotate(key)
        empty = key[:0]
        prefix_key, prefix_value = (empty, empty) if kv is None else kv
        if prefix_key.shape[0] != start:
            raise ValueError('hybrid attention prefix must end at start-1')
        keys, values = torch.cat((prefix_key, key)), torch.cat((prefix_value, value))
        groups = self.heads // self.kv_heads
        allowed = torch.arange(start + width, device=hidden.device)[None] <= (
            start + torch.arange(width, device=hidden.device))[:, None]
        attended = F.scaled_dot_product_attention(q.transpose(0, 1)[None],
            keys.repeat_interleave(groups, 1).transpose(0, 1)[None],
            values.repeat_interleave(groups, 1).transpose(0, 1)[None],
            attn_mask=allowed[None, None], dropout_p=0., scale=self.dim ** -0.5)[0].transpose(0, 1)
        return self.o_proj((attended * gate.sigmoid()).reshape(width, -1)), (keys, values)


class HybridLayer(nn.Module):
    def __init__(self, config, kind):
        super().__init__()
        if kind == 'linear_attention':
            self.linear_attn = HybridGDN(config)
        else:
            self.self_attn = HybridAttention(config)
        self.input_layernorm = HybridC1RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = HybridC1RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = nn.Module()
        self.mlp.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.mlp.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.mlp.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, hidden, *, start, state=None):
        source = self.input_layernorm(hidden)
        if hasattr(self, 'linear_attn'):
            mixed, state = self.linear_attn(source, state)
        else:
            mixed, state = self.self_attn(source, start=start, kv=state)
        hidden = hidden + mixed
        source = self.post_attention_layernorm(hidden)
        return hidden + self.mlp.down_proj(F.silu(self.mlp.gate_proj(source)) * self.mlp.up_proj(source)), state


class HybridCache:
    """Sequence-owned recurrence and K/V with per-verify-token commit snapshots."""
    sequential_verify = True

    def __init__(self, config):
        self.config, self.states, self.length = config, [None] * config.num_hidden_layers, 0
        self.snapshots = []

    def get_seq_length(self):
        return self.length

    def snapshot(self):
        # All state updates are functional; snapshots share immutable K/V storage.
        self.snapshots.append(list(self.states))

    def commit_tree(self, prefix_len, accepted):
        count = int(accepted.numel())
        if not torch.equal(accepted, torch.arange(count, device=accepted.device)):
            raise ValueError('hybrid target supports only causal width-one trees')
        self.states = self.snapshots[count - 1]
        self.length = prefix_len + count
        self.snapshots.clear()


class HybridTarget(nn.Module):
    """Text-only hybrid target through the existing HF-shaped ModelRunner seam."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.model.layers = nn.ModuleList(HybridLayer(config, kind) for kind in config.layer_types)
        self.model.norm = HybridC1RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.generation_config = SimpleNamespace(eos_token_id=getattr(config, 'eos_token_id', None))
        self._cache = None

    def new_cache(self):
        return HybridCache(self.config)

    def forward(self, input_ids, past_key_values=None, position_ids=None,
                attention_mask=None, logits_to_keep=0, **kwargs):
        if input_ids.shape[0] != 1:
            raise ValueError('hybrid target requires batch one')
        cache = past_key_values if past_key_values is not None else self.new_cache()
        start, width = cache.length, input_ids.shape[1]
        expected = torch.arange(start, start + width, device=input_ids.device)[None]
        if position_ids is not None and not torch.equal(position_ids, expected):
            raise ValueError('hybrid target supports only consecutive causal positions')
        if attention_mask is not None:
            allowed = torch.arange(start + width, device=input_ids.device)[None] <= expected[0, :, None]
            if attention_mask.ndim != 4 or not torch.equal(attention_mask[0, 0] == 0, allowed):
                raise ValueError('hybrid target supports only causal width-one masks')
        hidden = self.model.embed_tokens(input_ids)[0]
        for index, layer in enumerate(self.model.layers):
            hidden, cache.states[index] = layer(hidden, start=start, state=cache.states[index])
        cache.length += width
        self._cache = cache
        hidden = self.model.norm(hidden)[None]
        if logits_to_keep:
            hidden = hidden[:, -logits_to_keep:]
        return SimpleNamespace(logits=self.lm_head(hidden), past_key_values=cache)

    @classmethod
    def from_checkpoint(cls, root, *, device, dtype):
        from safetensors import safe_open
        root = Path(root).resolve(strict=True)
        config = text_config(json.loads((root / 'config.json').read_text()))
        config._name_or_path = str(root)
        with torch.device('meta'):
            target = cls(config)
        target.to(dtype=dtype)
        target.to_empty(device=device)
        names = target.state_dict()
        index = json.loads((root / 'model.safetensors.index.json').read_text())['weight_map']
        def local(name):
            return name.replace('model.language_model.', 'model.', 1)
        text_names = {local(name): name for name in index
                      if name.startswith(('model.language_model.', 'model.layers.', 'model.embed_tokens.',
                                          'model.norm.', 'lm_head.'))}
        if set(text_names) != set(names):
            raise ValueError(f'hybrid target keys differ: missing={sorted(set(names)-set(text_names))} '
                             f'unexpected={sorted(set(text_names)-set(names))}')
        for shard in sorted({index[name] for name in text_names.values()}):
            path = (root / shard).resolve(strict=True)
            if not path.is_relative_to(root):
                raise ValueError('target shard escapes checkpoint root')
            with safe_open(str(path), framework='pt', device='cpu') as handle:
                for name, source in text_names.items():
                    if index[source] == shard:
                        tensor = handle.get_tensor(source)
                        if tensor.shape != names[name].shape:
                            raise ValueError(f'hybrid target shape differs: {source}')
                        names[name].copy_(tensor.to(device=device, dtype=dtype))
        return target.eval().requires_grad_(False)


class HybridAnchorDecoder(nn.Module):
    """The last GDN/full pair reading committed state and isolated causal rows."""
    hybrid = True

    def __init__(self, target, *, tap_ids):
        super().__init__()
        deepest = max(tap_ids)
        indices = list(range(deepest + 1, target.config.num_hidden_layers))
        if len(indices) != 2 or target.config.layer_types[deepest + 1:] != ['linear_attention', 'full_attention']:
            raise ValueError('hybrid decoder requires two GDN/full layers after the last tap')
        self.layers = nn.ModuleList(target.model.layers[index] for index in indices)
        self.norm, self.lm_head = target.model.norm, target.lm_head
        self.hidden_size = target.config.hidden_size
        # Keep the target out of the module tree: the decoder views its pinned tail.
        object.__setattr__(self, 'target', target)
        self.indices, self._prefixes = indices, {}

    @classmethod
    def from_checkpoint(cls, target, *, tap_ids, file_pins):
        if 'config.json' not in file_pins or 'model.safetensors.index.json' not in file_pins:
            raise ValueError('hybrid decoder needs config and index SHA256 pins')
        decoder = cls(target, tap_ids=tap_ids)
        names = [f'model.language_model.layers.{index}.{name}' for index, layer in
                 zip(decoder.indices, decoder.layers) for name in layer.state_dict()]
        names += ['model.language_model.norm.weight', 'lm_head.weight']
        # Verify the exact tail tensor set and its pinned source shards, even when shared.
        load_target_tensors(target.config._name_or_path, names, file_pins=file_pins,
            prefixes=tuple(f'model.language_model.layers.{i}.' for i in decoder.indices)
                     + ('model.language_model.norm.', 'lm_head.'))
        return decoder

    def drop_sequence(self, sequence):
        self._prefixes.pop(id(sequence), None)

    def _prefix(self, sequence, desired_length):
        cached = self._prefixes.get(id(sequence))
        if cached is not None and cached.length == desired_length:
            return cached
        cache = self.target._cache
        if cache is not None and cache.length == desired_length:
            states = [cache.states[index] for index in self.indices]
        else:
            base = cached.length if cached is not None and cached.length < desired_length else 0
            states = list(cached.states) if base else [None, None]
            if desired_length > base:
                hidden = sequence.taps[base:desired_length, -1].to(next(self.layers.parameters()))
                for index, layer in enumerate(self.layers):
                    hidden, states[index] = layer(hidden, start=base, state=states[index])
        prefix = SimpleNamespace(length=desired_length, states=states)
        self._prefixes[id(sequence)] = prefix
        return prefix

    def refresh_logits(self, sequence, *, length, tail_hidden, tail_counts, block_hidden, slot_counts):
        prefix = self._prefix(sequence, length)
        parameter = next(self.layers.parameters())
        logits, row_of, slot_of = [], [], []
        for row, (tail_count, slots) in enumerate(zip(tail_counts, slot_counts)):
            tail_count, slots = int(tail_count), int(slots)
            if not slots:
                continue
            block = block_hidden[row, :slots, -self.hidden_size:].to(parameter)
            hidden = (block if not tail_count else torch.cat((tail_hidden[:tail_count].to(parameter), block)))
            for index, layer in enumerate(self.layers):
                hidden, _ = layer(hidden, start=length, state=prefix.states[index])
            logits.append(self.lm_head(self.norm(hidden))[tail_count:])
            row_of.extend([row] * slots)
            slot_of.extend(range(slots))
        return (torch.cat(logits), torch.tensor(row_of, device=parameter.device),
                torch.tensor(slot_of, device=parameter.device))


def load_hybrid_llm(*, model_name_or_path, device, dtype, attn_implementation='sdpa'):
    from transformers import AutoTokenizer
    from jetspec.core.llm import LLM
    from jetspec.core.model_runner import ModelRunner
    if attn_implementation != 'sdpa':
        raise ValueError('hybrid torch reader requires SDPA')
    llm = LLM.__new__(LLM)
    llm.model = HybridTarget.from_checkpoint(model_name_or_path, device=device, dtype=dtype)
    llm.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, local_files_only=True)
    llm.runner, llm.device, llm.dtype = ModelRunner(llm.model), device, dtype
    llm.eos_token_ids = llm._resolve_eos()
    return llm
