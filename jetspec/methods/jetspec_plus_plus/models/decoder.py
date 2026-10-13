"""Ported from the JetSpec++ reference implementation.

Also mirrors the reference decoder core, causal window and head.

Public API:
AnchorDecoder(*, layers, norm, lm_head, hidden_size, prefix_layers=None,
              rotary=None, root=None); from_target(target, *, tap_ids=(1,9,17,25,33)).
from_checkpoint(target, *, tap_ids, device, dtype, file_pins) loads an independent slice.
forward(hidden, *, mask=None, positions=None, rope=None, prefix=None)
    -> (decoded, logits); _prefix(sequence, desired_length) -> IncrementalPrefix;
drop_sequence(sequence) releases its prefix; sequence.taps is [L, 5, H].
decode_layer(layer, hidden, mask, positions, rope, prefix) -> hidden.
decode_hidden(layers, hidden, *, mask=None, positions=None, rope=None, prefix=None).
decoder_prefix_kv(layer, hidden, rope) -> (key, value).
IncrementalPrefix() has length, kv and update(key, value, layer_idx, cache_kwargs=None).
refresh_window_layout(tail_tokens, tail_counts, slot_counts, *, length, device=None)
    -> (allowed, positions, row_of, slot_of).
refresh_decoder_logits(decoder, sequence, *, length, tail_hidden, tail_counts,
                       block_hidden, slot_counts) -> (logits, row_of, slot_of).
dense_rows(values, row_of, slot_of, rows, width, fill) -> [rows, width].
load_target_tensors(root, names, *, file_pins=None, prefixes=()) -> checkpoint state.
"""
from pathlib import Path
import copy
import hashlib
import json

import torch
from torch import nn
from transformers.models.qwen3.modeling_qwen3 import rotate_half

__all__ = ['AnchorDecoder', 'decode_layer', 'decode_hidden', 'decoder_prefix_kv',
           'IncrementalPrefix', 'refresh_window_layout', 'refresh_decoder_logits', 'dense_rows',
           'load_target_tensors']


def load_target_tensors(root, names, *, file_pins=None, prefixes=()):
    """Read selected local checkpoint tensors; pinned slices fail closed."""
    from safetensors import safe_open

    root = Path(root).resolve(strict=True)
    index = root / 'model.safetensors.index.json'

    def checked(name):
        path = (root / name).resolve(strict=True)
        if Path(name).is_absolute() or '..' in Path(name).parts or not path.is_relative_to(root):
            raise ValueError(f'target-file path escapes the target directory: {name}')
        if file_pins is not None:
            if name not in file_pins:
                raise ValueError(f'target-slice decoder lacks pin for {name}')
            with path.open('rb') as stream:
                actual = hashlib.file_digest(stream, 'sha256').hexdigest()
            if actual != file_pins[name]:
                raise ValueError(f'target-file SHA256 mismatch: {name}')
        return path

    if index.is_file():
        weight_map = json.loads(checked(index.name).read_text())['weight_map']
    else:
        with safe_open(checked('model.safetensors'), framework='pt') as handle:
            weight_map = {name: 'model.safetensors' for name in handle.keys()}
    missing = set(names) - set(weight_map)
    if missing:
        raise ValueError(f'target-slice decoder missing tensors: {sorted(missing)}')
    unexpected = {name for name in weight_map if name.startswith(prefixes)} - set(names)
    if unexpected:
        raise ValueError(f'target-slice decoder unexpected tensors: {sorted(unexpected)}')
    state = {}
    for shard in sorted({weight_map[name] for name in names}):
        with safe_open(checked(shard), framework='pt') as handle:
            for name in names:
                if weight_map[name] == shard:
                    state[name] = handle.get_tensor(name)
    return state


def refresh_window_layout(tail_tokens, tail_counts, slot_counts, *, length, device=None):
    """Pack staged tail and causal, isolated block rows after the committed prefix."""
    tail_counts = torch.as_tensor(tail_counts, dtype=torch.long, device=device).reshape(-1)
    slot_counts = torch.as_tensor(slot_counts, dtype=torch.long, device=device).reshape(-1)
    device = tail_counts.device
    if tail_counts.shape != slot_counts.shape:
        raise ValueError('tail_counts and slot_counts need one entry per row')
    if bool((tail_counts < (1 if tail_tokens else 0)).any()) or bool((tail_counts > tail_tokens).any()):
        raise ValueError('every row needs 1..tail_tokens staged tail slots (none in the exact round)')
    if bool((slot_counts < 1).any()):
        raise ValueError('every row needs at least one block slot')
    rows = int(tail_counts.numel())
    total = int(slot_counts.sum())
    row_index = torch.repeat_interleave(torch.arange(rows, device=device), slot_counts)
    starts = torch.cumsum(slot_counts, 0) - slot_counts
    slot = torch.arange(total, device=device) - starts[row_index]
    tail = torch.arange(tail_tokens, device=device)
    row_of = torch.cat((torch.full((tail_tokens,), -1, dtype=torch.long, device=device), row_index))
    slot_of = torch.cat((tail, slot))
    positions = torch.cat((length + tail, length + tail_counts[row_index] + slot))
    query_tail = row_of[:, None] < 0
    key_tail = row_of[None, :] < 0
    span = torch.cat((tail + 1, tail_counts[row_index]))
    allowed = torch.where(
        key_tail, slot_of[None, :] < span[:, None],
        (~query_tail) & (row_of[None, :] == row_of[:, None]) & (slot_of[None, :] <= slot_of[:, None]))
    return allowed, positions, row_of, slot_of


def decode_layer(layer, hidden, mask, positions, rope, prefix):
    return layer(hidden, attention_mask=mask, position_ids=positions,
                 position_embeddings=rope, past_key_values=prefix, use_cache=False)


def decode_hidden(layers, hidden, *, mask=None, positions=None, rope=None, prefix=None):
    for layer in layers:
        hidden = decode_layer(layer, hidden, mask, positions, rope, prefix)
    return hidden


class IncrementalPrefix:
    """Append-only layer-34/35 K/V owned by one served sequence."""
    def __init__(self):
        self.length = 0
        self.kv = {}

    def update(self, key, value, layer_idx, cache_kwargs=None):
        if layer_idx in self.kv:
            key = torch.cat((self.kv[layer_idx][0], key), dim=-2)
            value = torch.cat((self.kv[layer_idx][1], value), dim=-2)
        self.kv[layer_idx] = (key.detach(), value.detach())
        return key, value


def decoder_prefix_kv(layer, hidden, rope):
    source = layer.input_layernorm(hidden)
    attn = layer.self_attn
    shape = (*source.shape[:-1], -1, attn.head_dim)
    key_states = attn.k_norm(attn.k_proj(source).view(shape)).transpose(1, 2)
    value_states = attn.v_proj(source).view(shape).transpose(1, 2)
    cos, sin = rope
    key_states = (key_states * cos.unsqueeze(1) +
                  rotate_half(key_states) * sin.unsqueeze(1))
    return key_states, value_states


class AnchorDecoder(nn.Module):
    """One decoder definition; engines supply optimized ops and model loading."""
    def __init__(self, *, layers, norm, lm_head, hidden_size, prefix_layers=None,
                 rotary=None, root=None):
        super().__init__()
        self.layers = layers
        self.prefix_layers = layers if prefix_layers is None else prefix_layers
        self.norm = norm
        self.lm_head = lm_head
        self.rotary = rotary
        self.root = Path(root) if root is not None else None
        self.hidden_size = int(hidden_size)
        self._prefixes = {}

    def forward(self, hidden, *, mask=None, positions=None, rope=None, prefix=None):
        if positions is None:
            positions = torch.arange(hidden.shape[1], device=hidden.device)[None]
        if rope is None and self.rotary is not None:
            rope = self.rotary(hidden, positions)
        decoded = self.norm(decode_hidden(self.layers, hidden, mask=mask, positions=positions,
                                         rope=rope, prefix=prefix))
        return decoded, self.lm_head(decoded)

    @classmethod
    def from_target(cls, target, *, tap_ids=(1, 9, 17, 25, 33)):
        """View the loaded Qwen3's two layers after the deepest adapter tap."""
        if not tap_ids:
            raise ValueError('target-slice decoder requires adapter target_layer_ids')
        deepest = max(int(index) for index in tap_ids)
        indices = tuple(range(deepest + 1, int(target.config.num_hidden_layers)))
        if len(indices) != 2:
            raise ValueError(f'target-slice decoder requires two layers after tap {deepest}; got {indices}')
        types = target.config.layer_types
        for index in indices:
            if target.config.model_type != 'qwen3' or types[index] != 'full_attention':
                raise ValueError(f'unsupported decoder layer {index}')
        # Share loaded target modules rather than loading a second set of weights.
        layers = nn.ModuleList([target.model.layers[index] for index in indices])
        return cls(layers=layers, prefix_layers=layers, norm=target.model.norm,
                   lm_head=target.lm_head, hidden_size=target.config.hidden_size,
                   rotary=target.model.rotary_emb)

    @classmethod
    def from_checkpoint(cls, target, *, tap_ids, device, dtype, file_pins):
        """Load an independent target slice at its checkpoint dtype, as served."""
        from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer, Qwen3RotaryEmbedding

        # The reference validates the two layers after the deepest tap.
        cls.from_target(target, tap_ids=tap_ids)
        indices = range(max(tap_ids) + 1, target.config.num_hidden_layers)
        config = copy.deepcopy(target.config)
        layers = nn.ModuleList([Qwen3DecoderLayer(config, index) for index in indices])
        names = {f'model.layers.{index}.{name}': f'{local}.{name}'
                 for local, index in enumerate(indices) for name in layers[local].state_dict()}
        names.update({'model.norm.weight': 'norm.weight', 'lm_head.weight': 'lm_head.weight'})
        # The reference reads the pinned slice, never the fp32 target view.
        raw = load_target_tensors(target.config._name_or_path, names, file_pins=file_pins,
                  prefixes=tuple(f'model.layers.{index}.' for index in indices) + ('model.norm.', 'lm_head.'))
        state = {names[name]: value for name, value in raw.items()}
        for name, tensor in state.items():
            # The reference refuses a cast of checkpoint slice weights.
            if tensor.dtype != dtype:
                raise ValueError(f'target-slice decoder dtype mismatch: {name} '
                                 f'target={tensor.dtype} runner={dtype}')
        layers.load_state_dict({name: state[name] for name in layers.state_dict()}, strict=True, assign=True)
        norm = copy.deepcopy(target.model.norm)
        lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False, dtype=dtype)
        norm.load_state_dict({'weight': state['norm.weight']}, strict=True, assign=True)
        lm_head.load_state_dict({'weight': state['lm_head.weight']}, strict=True, assign=True)
        # The reference owns separate layers, norm and LM head.
        decoder = cls(layers=layers, norm=norm, lm_head=lm_head, hidden_size=config.hidden_size,
                      rotary=Qwen3RotaryEmbedding(config, device=device))
        # The reference leaves the rotary frequencies in fp32.
        return decoder.to(device=device).eval().requires_grad_(False)

    def drop_sequence(self, sequence):
        """Release the committed-prefix cache at the end of a sequence."""
        self._prefixes.pop(id(sequence), None)

    @torch.inference_mode()
    def _prefix(self, sequence, desired_length):
        """Append true deepest-tap K/V; rebuild when the committed prefix shrinks."""
        if not 0 <= desired_length <= sequence.taps.shape[0]:
            raise ValueError('decoder prefix needs the true taps of every committed position')
        key = id(sequence)
        prefix = self._prefixes.get(key)
        if prefix is None or prefix.length > desired_length:
            prefix = IncrementalPrefix()
            self._prefixes[key] = prefix
        if prefix.length == desired_length:
            return prefix
        parameter = next(self.layers.parameters())
        # The reference casts true deepest taps before prefix K/V.
        tap33 = sequence.taps[:, 4, :].to(device=parameter.device, dtype=parameter.dtype)
        start = prefix.length
        hidden = tap33[start:desired_length][None]
        positions = torch.arange(start, desired_length, device=hidden.device, dtype=torch.long)
        rope = self.rotary(hidden, positions[None])
        keys = torch.arange(desired_length, device=hidden.device)
        mask = _additive_mask(keys[None, :] <= positions[:, None], hidden)
        # Dynamic storage keeps the reference's static-prefix CPU arithmetic without a capacity knob.
        for index, layer in enumerate(self.prefix_layers):
            if index == len(self.prefix_layers) - 1 and hasattr(layer, 'self_attn'):
                key_states, value_states = decoder_prefix_kv(layer, hidden, rope)
                prefix.update(key_states, value_states, layer.self_attn.layer_idx,
                              {'sin': rope[1], 'cos': rope[0], 'cache_position': positions})
            else:
                hidden = layer(hidden, attention_mask=mask, position_ids=positions[None],
                               position_embeddings=rope, past_key_values=prefix,
                               cache_position=positions, use_cache=True)
        prefix.length = desired_length
        return prefix


class _WindowCache:
    """Prefix K/V per layer index, extended by each call's keys; ``append`` keeps them."""

    def __init__(self, kv, append):
        self.kv = dict(kv)
        self.append = append

    def update(self, key, value, layer_idx, cache_kwargs=None):
        if layer_idx in self.kv:
            prefix_key, prefix_value = self.kv[layer_idx]
            key = torch.cat((prefix_key, key), dim=-2)
            value = torch.cat((prefix_value, value), dim=-2)
        if self.append:
            self.kv[layer_idx] = (key, value)
        return key, value


def _additive_mask(allowed, like):
    return like.new_zeros(allowed.shape).masked_fill_(~allowed, torch.finfo(like.dtype).min)[None, None]


@torch.inference_mode()
def refresh_decoder_logits(decoder, sequence, *, length, tail_hidden, tail_counts, block_hidden,
                           slot_counts):
    """Read true committed taps, staged tail and isolated refreshed block rows.

    Returns packed (logits, row_of, slot_of); output at slot j predicts slot j+1.
    The last four committed rows extend a private cache, leaving the round's
    cached prefix unchanged. No block row attends to another block row.
    """
    if getattr(decoder, 'hybrid', False):
        return decoder.refresh_logits(sequence, length=length, tail_hidden=tail_hidden,
            tail_counts=tail_counts, block_hidden=block_hidden, slot_counts=slot_counts)
    parameter = next(decoder.layers.parameters())
    hidden_size = decoder.hidden_size
    tail_counts = torch.as_tensor(tail_counts, dtype=torch.long).reshape(-1).cpu()
    slot_counts = torch.as_tensor(slot_counts, dtype=torch.long).reshape(-1).cpu()
    tail_tokens = int(tail_counts.max())
    if tail_tokens > (0 if tail_hidden is None else int(tail_hidden.shape[0])):
        raise ValueError('a row reads more tail slots than the staged block holds')
    taps = sequence.taps
    if int(taps.shape[0]) < length:
        raise ValueError('the decoder refresh prefix needs the true taps of every committed position')
    existing = getattr(decoder, '_prefixes', {}).get(id(sequence))
    base = int(existing.length) if existing is not None and int(existing.length) <= length else 0
    kv = ({index: (pair[0][..., :base, :], pair[1][..., :base, :])
           for index, pair in existing.kv.items()} if base else {})
    cache = _WindowCache(kv, append=True)
    if base < length:
        # The reference casts committed taps for the refresh prefix.
        hidden = taps[base:length, 4, :].to(device=parameter.device, dtype=parameter.dtype)[None]
        positions = torch.arange(base, length, device=parameter.device, dtype=torch.long)
        keys = torch.arange(length, device=parameter.device)
        mask = _additive_mask(keys[None, :] <= positions[:, None], hidden)
        rope = decoder.rotary(hidden, positions[None])
        for layer in decoder.prefix_layers:
            hidden = layer(hidden, attention_mask=mask, position_ids=positions[None],
                           position_embeddings=rope, past_key_values=cache,
                           cache_position=positions, use_cache=True)
    allowed, positions, row_of, slot_of = refresh_window_layout(
        tail_tokens, tail_counts, slot_counts, length=length)
    rows = torch.arange(int(slot_counts.numel()))
    block_index = torch.repeat_interleave(rows, slot_counts)
    tail = None if tail_hidden is None else tail_hidden[:tail_tokens, -hidden_size:]
    block = block_hidden[block_index.to(block_hidden.device), slot_of[tail_tokens:].to(block_hidden.device),
                         -hidden_size:]
    # The reference casts window predictions to decoder parameters.
    if tail is None:
        hidden = block.to(parameter)[None]
    else:
        hidden = torch.cat((tail.to(parameter), block.to(parameter)), dim=0)[None]
    positions = positions.to(parameter.device)
    full = torch.cat((torch.ones(allowed.shape[0], length, dtype=torch.bool), allowed), dim=1)
    mask = _additive_mask(full.to(parameter.device), hidden)
    rope = decoder.rotary(hidden, positions[None])
    frozen = _WindowCache(cache.kv, append=False)
    for layer in decoder.layers:
        hidden = layer(hidden, attention_mask=mask, position_ids=positions[None],
                       position_embeddings=rope, past_key_values=frozen, use_cache=False)
    decoded = decoder.norm(hidden)[0, tail_tokens:]
    return decoder.lm_head(decoded), row_of[tail_tokens:], slot_of[tail_tokens:]


def dense_rows(values, row_of, slot_of, rows, width, fill):
    """Scatter packed per-token values into [rows, width] (``fill`` elsewhere)."""
    out = torch.full((rows, width), fill, dtype=values.dtype, device=values.device)
    out[row_of.to(values.device), slot_of.to(values.device)] = values
    return out
