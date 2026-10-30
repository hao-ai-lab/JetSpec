"""Served two-layer anchor decoder and exact flat anchor offers.

``decoder/manifest.json`` selects ``target_slice`` with ``target_files`` mapping
target-relative index/shard names to sha256 digests, or ``file`` with the pinned
``decoder/checkpoint-0.pt`` path and sha256. Bundles without it retain the jd01
``decoder/checkpoint-final-model.pt`` load.
"""
from __future__ import annotations

from jetspec.engines.jetspec_plus_plus.profiles import kernel_enabled, kernel_value

import json
import copy
import hashlib
import os
import time
import weakref
from functools import partial
from types import SimpleNamespace
import numpy as np
from dataclasses import dataclass, replace
from pathlib import Path

import torch
from jetspec.engines.jetspec_plus_plus.models.config import setting
from safetensors import SafetensorError, safe_open
from torch import nn
from torch.nn import functional as F
from jetspec.engines.jetspec_plus_plus.draft.method.model_spec import ModelSpec
from jetspec.engines.jetspec_plus_plus.draft.method.decoder import AnchorDecoder as MethodAnchorDecoder, decode_layer
from jetspec.engines.jetspec_plus_plus.draft.method.decoder.anchor import decode_hidden
from transformers import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Attention,
    Qwen3DecoderLayer,  # Compatibility export used by bundle tooling.
    Qwen3RotaryEmbedding,
    rotate_half,
)

import jetspec.engines.jetspec_plus_plus.runtime.run_manifest as run_manifest
from jetspec.engines.jetspec_plus_plus.models.lac import keep_device
from jetspec.engines.jetspec_plus_plus.models.lac import lac_candidates
from jetspec.engines.jetspec_plus_plus.draft.rows import fused_kernels
from jetspec.engines.jetspec_plus_plus.draft.rows import rows_fused
from jetspec.engines.jetspec_plus_plus.models.fast import dataset_entry
from jetspec.engines.jetspec_plus_plus.wire.contract import FrontInputs
from jetspec.engines.jetspec_plus_plus.wire.contract import Offset
from jetspec.engines.jetspec_plus_plus.wire.contract import TokenBuffers
from jetspec.engines.jetspec_plus_plus.kernels.decoder_select import register_offer_twins

register_offer_twins()


def _fill_prefetch_indices(host_ints, host_index, starts, slots, deltas, trash, bucket, window):
    """Stage one prefetch bucket in reusable host buffers; return live index count."""
    count = len(starts)
    packed = host_ints.numpy()
    packed[:bucket] = 0
    packed[:count] = starts
    packed[bucket:2 * bucket] = trash
    packed[bucket:bucket + count] = slots
    packed[2 * bucket:2 * bucket + bucket * window] = trash
    packed[2 * bucket + bucket * window:] = np.tile(np.arange(window), bucket)
    index = host_index.numpy()
    offset = 0
    for n, (start, slot, take) in enumerate(zip(starts, slots, deltas)):
        row = n * window
        packed[2 * bucket + row:2 * bucket + row + take] = slot
        packed[2 * bucket + bucket * window + row:
               2 * bucket + bucket * window + row + take] = np.arange(start, start + take)
        index[offset:offset + take] = np.arange(row, row + take)
        offset += take
    return offset


REUSED_DECODER_SHA256 = '41c85c871acd7bce2904a8f16d4f2beb0a4261044fc30bfd1aa605edbc1c90b7'


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _required_model_path(run_config) -> str:
    value = setting(run_config, 'hardware', 'hardware_settings.target_model_path')
    if value:
        return value
    raise RuntimeError('model decoder requires the target model directory in hardware config')


from jetspec.engines.jetspec_plus_plus.draft.method.decoder.window import window_mask


from jetspec.engines.jetspec_plus_plus.draft.method.decoder.window import chain_decoder_window_mask


from jetspec.engines.jetspec_plus_plus.draft.method.decoder.window import stage_chain_window_state


from jetspec.engines.jetspec_plus_plus.draft.method.decoder.window import chain_decoder_prefix_slots


from jetspec.engines.jetspec_plus_plus.draft.method.decoder.window import chain_decoder_prefix_mask


def chain_rejected_tokens(rp, tokens: TokenBuffers, *, run_config=None):
    """Decoder exclusion frame built entirely from device token buffers."""
    if setting(run_config, 'engine', 'engine_settings.decoder_frame', None) == 'wire':
        rejected = torch.cat((tokens.pending_draft_ids,
                              tokens.pending_draft_ids.new_full((1,), -1)))
    else:
        rejected = torch.cat((rp[Offset.CORRECTION].view(1),
                              tokens.pending_draft_ids))
    if setting(run_config, 'method', 'method_settings.kset_after_miss', None) == 'zero':
        rejected = torch.cat((torch.where(rp[Offset.AFTER_MISS] != 0,
                                          rejected.new_full((), -1), rejected[0]).view(1),
                              rejected[1:]))
    return rejected


from jetspec.engines.jetspec_plus_plus.draft.method.decoder.window import decoder_window


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


def decoder_window_project(layer, hidden, rope):
    from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb
    normed = layer.input_layernorm(hidden)
    attn = layer.self_attn
    shape = (*hidden.shape[:-1], -1, attn.head_dim)
    query = attn.q_norm(attn.q_proj(normed).view(shape)).transpose(1, 2)
    key = attn.k_norm(attn.k_proj(normed).view(shape)).transpose(1, 2)
    value = attn.v_proj(normed).view(shape).transpose(1, 2)
    query, key = apply_rotary_pos_emb(query, key, *rope)
    return query, key, value


def decoder_window_finish(layer, hidden, attended):
    attended = attended.reshape(*hidden.shape[:-1], -1).contiguous()
    hidden = hidden + layer.self_attn.o_proj(attended)
    return hidden + layer.mlp(layer.post_attention_layernorm(hidden))


def target_tail_forward(layer, hidden, *, attention_mask=None, position_ids=None,
                        position_embeddings=None, past_key_values=None, cache_position=None,
                        use_cache=False, **kwargs):
    """Execute shared target parameters with the decoder's private SDPA cache."""
    from transformers.integrations.sdpa_attention import sdpa_attention_forward
    query, key, value = decoder_window_project(layer, hidden, position_embeddings)
    attn = layer.self_attn
    if past_key_values is not None:
        key, value = past_key_values.update(key, value, attn.layer_idx,
            {'sin': position_embeddings[1], 'cos': position_embeddings[0],
             'cache_position': cache_position})
    attended = sdpa_attention_forward(attn, query, key, value, attention_mask,
        dropout=0.0, scaling=attn.scaling, sliding_window=attn.sliding_window)[0]
    return decoder_window_finish(layer, hidden, attended)


def TargetTailLayer(spec, index, *, config, max_verify_width=None, run_config=None):
    """Build a shared target layer with separate decoder checkpoint projections."""
    from jetspec.engines.jetspec_plus_plus.models.target_layers import DecoderLayer
    layer = DecoderLayer(spec, spec.layer_kinds[index], config=config,
                         separate_projections=None if spec.attention.output_gate else True,
                         residual_in_activation_dtype=True,
                         max_verify_width=max_verify_width, run_config=run_config)
    from jetspec.engines.jetspec_plus_plus.models.target_linear import bind_projection_role
    bind_projection_role(layer, 'draft')
    layer.tail_index = index
    if hasattr(layer, 'self_attn'):
        attn = layer.self_attn
        attn.layer_idx = index
        attn._decoder_dense = not spec.attention.output_gate
        attn.num_key_value_groups = spec.attention.q_heads // spec.attention.kv_heads
        attn.scaling = spec.attention.head_dim ** -0.5
        attn.sliding_window = getattr(config, 'sliding_window', None)
        attn.is_causal = True
        for projection in (attn.q_proj, attn.k_proj, attn.v_proj, attn.o_proj,
                           layer.mlp.gate_proj, layer.mlp.up_proj, layer.mlp.down_proj):
            projection.out_features, projection.in_features = projection.weight.shape
        layer.forward = partial(target_tail_forward, layer)
    return layer


def _tail_layers(config, indices, *, run_config=None):
    spec = getattr(config, '_tail_spec', None) or ModelSpec.from_configs(config.to_dict())
    from jetspec.engines.jetspec_plus_plus.wire.protocol import draft_width
    width = draft_width(run_config=run_config) + 1 if spec.has_state else None
    return nn.ModuleList([TargetTailLayer(spec, index, config=config, max_verify_width=width, run_config=run_config)
                          for index in indices])


def _tail_norm(spec):
    from jetspec.engines.jetspec_plus_plus.models.target_layers import RMSNorm
    return RMSNorm(spec.hidden_size, spec.rms_norm_eps, offset=spec.attention.norm_offset,
                   norm_round=spec.attention.norm_round, residual_in_activation_dtype=True)


def _tail_lm_head(spec=None, *, source=None):
    """Keep decoder vocabulary projections on the common precision boundary."""
    from jetspec.engines.jetspec_plus_plus.models.target_linear import TargetLinear
    if source is None:
        return TargetLinear(spec.hidden_size, spec.vocab_size, layout='replicated', precision_role='lm_head')
    output_size, input_size = source.weight.shape
    with torch.device('meta'):
        head = TargetLinear(input_size, output_size, layout='replicated', bias=source.bias is not None,
                            precision_role='lm_head')
    head.to(dtype=source.weight.dtype).to_empty(device=source.weight.device)
    head.load_state_dict(source.state_dict(), strict=True)
    return head


def decoder_layer_lean(layer, hidden, rope, mask, cache, cache_position, fused, *, pending=None, run_config=None):
    """One eager Qwen3 decoder layer on the certified fused kernels.

    Same projections, cache update and sdpa call as the vendored layer; the
    norms, the rotary and the residual add come from ``fused`` (whose CPU
    and fp32 fallbacks spell the reference arithmetic, so the layer is bitwise
    the vendored one there).  ``pending=(update, residual)`` hands in the
    previous layer's unsummed residual pair: the add lands in this layer's
    input add+norm kernel.  Returns the same ``(mlp_output, residual)`` pair
    for the caller to sum or to hand on.
    """
    from jetspec.engines.jetspec_plus_plus.profiles import kernel_enabled
    from transformers.integrations.sdpa_attention import sdpa_attention_forward
    attn = layer.self_attn
    cos, sin = rope
    head_dim = attn.head_dim
    heads = attn.q_proj.out_features // head_dim
    kv_heads = attn.k_proj.out_features // head_dim
    if pending is None:
        normed = fused.fast_rms_norm(hidden, layer.input_layernorm.weight,
                                     layer.input_layernorm.variance_epsilon)
    else:
        normed, hidden = fused.fast_add_rms_norm(
            *pending, layer.input_layernorm.weight, layer.input_layernorm.variance_epsilon)
    query = fused.fast_q_postprocess(attn.q_proj(normed), attn.q_norm.weight, cos, sin,
                                     attn.q_norm.variance_epsilon, heads, head_dim)
    key, value = fused.fast_kv_postprocess(attn.k_proj(normed), attn.v_proj(normed),
                                           attn.k_norm.weight, cos, sin,
                                           attn.k_norm.variance_epsilon, kv_heads, 1, head_dim)
    if cache is not None:
        key, value = cache.update(key, value, attn.layer_idx,
                                  {'sin': sin, 'cos': cos, 'cache_position': cache_position})
    lean = (kernel_enabled('c1_lean', '1')
            and (mask is not None or isinstance(cache, SlotBatchPrefix))
            and query.device.type == 'cuda' and query.dtype == torch.bfloat16)
    if isinstance(cache, SlotBatchPrefix):
        from jetspec.engines.jetspec_plus_plus.kernels.attn_sm100 import attend_slots
        groups = attn.num_key_value_groups
        cache_key, cache_value = cache.pool.kv[attn.layer_idx]
        attended = attend_slots(
            query.reshape(query.shape[0], kv_heads, groups * query.shape[-2], head_dim),
            cache_key, cache_value, key, value, cache.slots, cache.lengths,
            cache.capacity, attn.scaling, inclusive=True, run_config=run_config)
        attended = attended.reshape(query.shape).transpose(1, 2).contiguous()
    elif lean:
        # sdpa_attention_forward's float-mask path (repeat_kv, mask cut to the keys, the
        # memory-efficient kernel SDPA selects), without the per-call backend selection.
        from transformers.integrations.sdpa_attention import repeat_kv
        groups = attn.num_key_value_groups
        key, value = repeat_kv(key, groups), repeat_kv(value, groups)
        attended = fused.efficient_attention(query, key, value,
                                             mask[:, :, :, :key.shape[-2]],
                                             attn.scaling).transpose(1, 2).contiguous()
    else:
        attended = sdpa_attention_forward(
            attn, query, key, value, mask, dropout=0.0, scaling=attn.scaling,
            sliding_window=attn.sliding_window)[0]
    attended = attended.reshape(*hidden.shape[:-1], -1).contiguous()
    normed, hidden = fused.fast_add_rms_norm(
        attn.o_proj(attended), hidden, layer.post_attention_layernorm.weight,
        layer.post_attention_layernorm.variance_epsilon)
    if lean:
        mlp = layer.mlp
        return mlp.down_proj(fused._silu_mul_extension().silu_mul(
            *_decoder_gate_up(mlp, normed))), hidden
    return layer.mlp(normed), hidden


def _decoder_gate_up(mlp, source):
    """One opt-in gate/up GEMM, with independent singleton and batched knobs."""
    from jetspec.engines.jetspec_plus_plus.profiles import kernel_enabled
    singleton = (kernel_enabled('decoder_single_gateup_packed', '1')
                 and source.ndim == 3 and source.shape[0] == 1)
    if ((not singleton)):
        return mlp.gate_proj(source), mlp.up_proj(source)
    _prepare_decoder_gate_up(mlp)
    weight, bias, width = mlp._decoder_gate_up_packed
    precision = getattr(mlp.gate_proj, 'precision', None)
    from jetspec.engines.jetspec_plus_plus.models.target_linear import packed_draft_linear
    project = packed_draft_linear if precision is None else precision.linear
    output = project(source.reshape(-1, source.shape[-1]), weight, bias)
    return output.view(*source.shape[:-1], -1).split(width, dim=-1)


def _prepare_decoder_gate_up(mlp):
    """Retain immutable packed weights before singleton graph warm-up."""
    if getattr(mlp, '_decoder_gate_up_packed', None) is None:
        weight = torch.cat((mlp.gate_proj.weight, mlp.up_proj.weight)).detach()
        biases = (mlp.gate_proj.bias, mlp.up_proj.bias)
        bias = torch.cat(biases).detach() if biases[0] is not None else None
        mlp._decoder_gate_up_packed = (weight, bias, mlp.gate_proj.out_features)


def decoder_prefix_kv_lean(layer, hidden, rope, fused, *, pending=None):
    """``decoder_prefix_kv`` with one norm launch and the fused k norm + rotary."""
    from jetspec.engines.jetspec_plus_plus.profiles import kernel_enabled
    attn = layer.self_attn
    cos, sin = rope
    head_dim = attn.head_dim
    kv_heads = attn.k_proj.out_features // head_dim
    if pending is None:
        source = fused.fast_rms_norm(hidden, layer.input_layernorm.weight,
                                     layer.input_layernorm.variance_epsilon)
    else:
        source, _ = fused.fast_add_rms_norm(
            *pending, layer.input_layernorm.weight, layer.input_layernorm.variance_epsilon)
    if (kernel_enabled('decoder_qkv_fused', '1')
            and getattr(attn, '_decoder_dense', type(attn) is Qwen3Attention) and hasattr(attn, '_decoder_prefix_kv')):
        _, key, value = _decoder_packed_project(attn, fused, source, rope, prefix=True)
        return key, value
    return fused.fast_kv_postprocess(attn.k_proj(source), attn.v_proj(source),
                                     attn.k_norm.weight, cos, sin,
                                     attn.k_norm.variance_epsilon, kv_heads, 1, head_dim)


def decoder_graph_lean():
    """JETSPEC_DECODER_GRAPH_LEAN=1: the graphed regime's decoder parts on the
    certified kernels (``decoder_layer_lean`` with JETSPEC_C1_LEAN=1, split around
    its attention so each side is one shape-only graph)."""
    from jetspec.engines.jetspec_plus_plus.profiles import kernel_enabled
    return kernel_enabled('decoder_graph_lean', '1')


def decoder_graph_window(*, run_config=None):
    """JETSPEC_DECODER_GRAPH_WINDOW=1 (with JETSPEC_DECODER_GRAPH_LEAN=1): the predict
    window (both trained layers, their attention, the norm and the lm_head) as one graph
    per key bucket, attending over a decoder-owned copy of the sequence's K/V."""
    return setting(run_config, 'engine', 'engine_settings.decoder_graph_window', None) == '1'


def batch_head_epilogue_enabled(count, dtype, cap, *, run_config=None):
    """The C>1 budget/BLEND graph consumes offers and two decoder scalars only."""
    from jetspec.engines.jetspec_plus_plus.profiles import kernel_enabled
    return (kernel_enabled('w632_head_fuse', '1')
            and count > 1 and dtype == torch.bfloat16
            and setting(run_config, 'engine', 'engine_settings.batch_draft_graph', '0') == '1'
            and setting(run_config, 'engine', 'engine_settings.rows_assembly', '0') == '1'
            and setting(run_config, 'method', 'method_settings.anchor_select', None) == 'topk'
            and setting(run_config, 'method', 'method_settings.anchor_select_home', None) == 'one'
            and setting(run_config, 'engine', 'engine_settings.heads_fast', None) == '1'
            and setting(run_config, 'method', 'method_settings.row_select', None) == 'budget'
            and setting(run_config, 'method', 'method_settings.kscore', None) == 'blend_dec'
            and not setting(run_config, 'engine', 'engine_settings.w280_record'))


def check_decoder_graph_lean_regime(*, run_config=None):
    from jetspec.engines.jetspec_plus_plus.profiles import kernel_enabled
    if decoder_graph_window(run_config=run_config) and not decoder_graph_lean():
        raise ValueError('JETSPEC_DECODER_GRAPH_WINDOW=1 requires JETSPEC_DECODER_GRAPH_LEAN=1')
    if not decoder_graph_lean():
        return
    if (setting(run_config, 'engine', 'engine_settings.rows_graphs', None) != '1' or setting(run_config, 'engine', 'engine_settings.heads_fast', None) != '1'
            or not kernel_enabled('c1_lean', '1')):
        raise ValueError('JETSPEC_DECODER_GRAPH_LEAN=1 is the graphed rows regime on the lean '
                         'kernels: it requires JETSPEC_ROWS_GRAPHS=1, JETSPEC_HEADS_FAST=1 and '
                         'JETSPEC_C1_LEAN=1')


def decoder_lean_project(layer, fused, hidden, pending, rope):
    """``decoder_layer_lean`` up to its attention: the input (add+)norm, the q and
    k/v projections with their norm + rotary.  Returns (query, key, value, hidden)."""
    from jetspec.engines.jetspec_plus_plus.profiles import kernel_enabled
    attn = layer.self_attn
    cos, sin = rope
    head_dim = attn.head_dim
    heads = attn.q_proj.out_features // head_dim
    kv_heads = attn.k_proj.out_features // head_dim
    if pending is None:
        normed = fused.fast_rms_norm(hidden, layer.input_layernorm.weight,
                                     layer.input_layernorm.variance_epsilon)
    else:
        normed, hidden = fused.fast_add_rms_norm(
            *pending, layer.input_layernorm.weight, layer.input_layernorm.variance_epsilon)
    if (kernel_enabled('decoder_qkv_fused', '1')
            and getattr(attn, '_decoder_dense', type(attn) is Qwen3Attention) and hasattr(attn, '_decoder_qkv')):
        query, key, value = _decoder_packed_project(attn, fused, normed, rope)
        return query, key, value, hidden
    query = fused.fast_q_postprocess(attn.q_proj(normed), attn.q_norm.weight, cos, sin,
                                     attn.q_norm.variance_epsilon, heads, head_dim)
    key, value = fused.fast_kv_postprocess(attn.k_proj(normed), attn.v_proj(normed),
                                           attn.k_norm.weight, cos, sin,
                                           attn.k_norm.variance_epsilon, kv_heads, 1, head_dim)
    return query, key, value, hidden


def _decoder_packed_project(attn, fused, source, rope, *, prefix=False):
    """One packed GEMM; CUDA postprocessing reads the split's row stride."""
    from jetspec.engines.jetspec_plus_plus.profiles import kernel_enabled
    weight, bias, widths = (attn._decoder_prefix_kv if prefix else attn._decoder_qkv)
    projection = attn.k_proj if prefix else attn.q_proj
    precision = getattr(projection, 'precision', None)
    from jetspec.engines.jetspec_plus_plus.models.target_linear import packed_draft_linear
    project = packed_draft_linear if precision is None else precision.linear
    parts = project(source, weight, bias).split(widths, dim=-1)
    cos, sin = rope
    head_dim = attn.head_dim
    kv_heads = attn.k_proj.out_features // head_dim
    if (not prefix and source.is_cuda and source.dtype == torch.bfloat16
            and source.shape[0] == 1 and kernel_enabled('draft_fuse_t12', '1')):
        from jetspec.engines.jetspec_plus_plus.kernels.decoder_staging import qkv_postprocess
        return qkv_postprocess(fused, *parts, attn.q_norm.weight, attn.k_norm.weight,
                               cos, sin, attn.q_norm.variance_epsilon,
                               attn.k_norm.variance_epsilon,
                               attn.q_proj.out_features // head_dim, kv_heads, head_dim)
    if prefix:
        query, projected_key, projected_value = None, *parts
    else:
        projected_query, projected_key, projected_value = parts
        heads = attn.q_proj.out_features // head_dim
        if source.is_cuda and source.dtype == torch.bfloat16 and source.shape[0] == 1:
            query = fused.jpass_q_norm_rotary(
                projected_query.view(-1, widths[0]), attn.q_norm.weight, cos[0], sin[0],
                attn.q_norm.variance_epsilon, heads, head_dim).transpose(0, 1)[None]
        else:
            query = fused.fast_q_postprocess(
                projected_query, attn.q_norm.weight, cos, sin,
                attn.q_norm.variance_epsilon, heads, head_dim)
    if source.is_cuda and source.dtype == torch.bfloat16 and source.shape[0] == 1:
        tokens = source.shape[1]
        key = source.new_empty((1, kv_heads, tokens, head_dim))
        value = source.new_empty((1, kv_heads, tokens, head_dim))
        fused.jpass_kv_norm_rotary_cache(
            projected_key.view(tokens, -1), projected_value.view(tokens, -1),
            attn.k_norm.weight, cos[0], sin[0], key[0].transpose(0, 1),
            value[0].transpose(0, 1), attn.k_norm.variance_epsilon, kv_heads, head_dim)
    else:
        key, value = fused.fast_kv_postprocess(
            projected_key, projected_value, attn.k_norm.weight, cos, sin,
            attn.k_norm.variance_epsilon, kv_heads, 1, head_dim)
    return query, key, value


def decoder_grouped_attention(fused, query, key, value, mask, scale):
    """Treat KV heads as batches so each group's keys are broadcast views."""
    _, heads, rows, dim = query.shape
    kv_heads, keys = key.shape[1:3]
    groups = heads // kv_heads
    grouped_query = query.view(kv_heads, groups, rows, dim)
    grouped_key = key.view(kv_heads, 1, keys, dim).expand(-1, groups, -1, -1)
    grouped_value = value.view(kv_heads, 1, keys, dim).expand(-1, groups, -1, -1)
    attended = fused.efficient_attention(grouped_query, grouped_key, grouped_value,
                                         mask[..., :keys], scale)
    return attended.reshape(1, heads, rows, dim)


def decoder_lean_attention(layer, fused, query, key, value, mask, *, length=None, count=None):
    """``decoder_layer_lean``'s lean attention (repeat_kv, the memory-efficient kernel)."""
    from jetspec.engines.jetspec_plus_plus.profiles import kernel_enabled
    if (length is not None and query.is_cuda
            and (((kernel_enabled('decoder_attn_len', '1'))))):
        from jetspec.engines.jetspec_plus_plus.kernels.len_attention import attend as attend_len
        return attend_len(query, key, value, length, layer.self_attn.scaling,
                          mode='decoder', count=count).transpose(1, 2).contiguous()
    from transformers.integrations.sdpa_attention import repeat_kv
    attn = layer.self_attn
    groups = attn.num_key_value_groups
    if (kernel_enabled('decoder_gqa_view', '1')
            and query.is_cuda and query.dtype == torch.bfloat16
            and query.shape[0] == 1 and groups > 1):
        return decoder_grouped_attention(fused, query, key, value, mask,
                                          attn.scaling).transpose(1, 2).contiguous()
    key, value = repeat_kv(key, groups), repeat_kv(value, groups)
    return fused.efficient_attention(query, key, value, mask[:, :, :, :key.shape[-2]],
                                     attn.scaling).transpose(1, 2).contiguous()


def decoder_merged_attention(layer, query, key, value, length, count, prefix):
    """Prefix rows (count rule) and window rows (key <= length + w) in one
    length attention; the result is already [1, tokens, heads, dim]."""
    from jetspec.engines.jetspec_plus_plus.kernels.len_attention import attend_merged
    return attend_merged(query, key, value, length, count, layer.self_attn.scaling,
                         prefix=prefix)


def decoder_lean_finish(layer, fused, hidden, attended):
    """``decoder_layer_lean`` after its attention: o_proj, the post add+norm and the
    silu MLP.  Returns the (mlp_output, residual) pair."""
    attended = attended.reshape(*hidden.shape[:-1], -1).contiguous()
    normed, hidden = fused.fast_add_rms_norm(
        layer.self_attn.o_proj(attended), hidden, layer.post_attention_layernorm.weight,
        layer.post_attention_layernorm.variance_epsilon)
    mlp = layer.mlp
    return mlp.down_proj(fused._silu_mul_extension().silu_mul(
        *_decoder_gate_up(mlp, normed))), hidden


def rejected_tokens(pending_block, *, after_miss=False, run_config=None):
    """Return the drafter token at every decoder k offset (0 through 14)."""
    drafts = pending_block.draft_ids.reshape(-1)
    from jetspec.engines.jetspec_plus_plus.wire.protocol import draft_width
    if drafts.numel() != draft_width(run_config=run_config):
        raise ValueError('decoder pending block requires JETSPEC_DRAFTS (14) draft ids')
    # Device fills, not host tensors: a host list would drain the stream.
    if setting(run_config, 'engine', 'engine_settings.decoder_frame', None) == 'wire':
        rejected = torch.cat((drafts, drafts.new_full((1,), -1)))
    else:
        rejected = torch.cat((drafts.new_full((1,), int(pending_block.correction_guess)), drafts))
    if after_miss and setting(run_config, 'method', 'method_settings.kset_after_miss', None) == 'zero':
        rejected = rejected.clone()
        rejected[0] = -1
    return rejected


def offer_dump_record(logits, reference_rejected, wire_rejected, *, seq_id,
                      round_id, top_n=32):
    """Build the raw, pre-exclusion decoder evidence row for one round."""
    logits = torch.as_tensor(logits).float()
    reference_rejected = torch.as_tensor(
        reference_rejected, device=logits.device, dtype=torch.long)
    wire_rejected = torch.as_tensor(wire_rejected, device=logits.device,
                                    dtype=torch.long)
    if logits.ndim != 2 or reference_rejected.shape != logits.shape[:1] or \
            wire_rejected.shape != logits.shape[:1]:
        raise ValueError('offer dump expects logits [offset,vocab] and two offset vectors')
    ranked = torch.argsort(logits, dim=-1, descending=True, stable=True)
    ids = ranked[:, :min(int(top_n), logits.shape[1])]
    values = logits.gather(1, ids)

    def rejected_logits(tokens):
        valid = tokens >= 0
        gathered = logits.new_zeros(tokens.shape)
        gathered[valid] = logits[valid, tokens[valid]]
        return [float(value) if bool(ok) else None
                for value, ok in zip(gathered.tolist(), valid.tolist())]

    return dict(
        seq_id=int(seq_id), round_id=int(round_id),
        pre_exclusion_logsumexp=logits.logsumexp(-1).tolist(),
        top_ids=ids.tolist(), top_logits=values.tolist(),
        reference_drej=reference_rejected.tolist(),
        reference_drej_logits=rejected_logits(reference_rejected),
        wire_drej=wire_rejected.tolist(),
        wire_drej_logits=rejected_logits(wire_rejected),
    )


from jetspec.engines.jetspec_plus_plus.draft.method.decoder.window import AnchorOffers


_GRAPHED_SELECT = None


class DeviceAnchorOffers:
    """serving JETSPEC_ROWS_ASSEMBLY=1: the eager top-k offers left on the device.

    ``device`` holds (ids, fp64 probabilities, pack) as _select_body returns them; the
    assembly stage reads them there and brings home only what the round needs.  Every
    AnchorOffers field is still served: the first read of a host field makes the same one-copy
    trip (topk_select._home) the HOME=one path makes, or takes the stage's host values."""

    def __init__(self, ids, p64, pack, probabilities, cap, width):
        self.device = (ids, p64, pack)
        self._probabilities, self._cap, self._width = probabilities, cap, width
        self._host = None
        self._summary = None

    def set_summary(self, sizes, top1, fallbacks):
        """The stage's host copy of the pack: set sizes, d_rej-was-top1 flags, fallbacks."""
        self._summary = ([int(v) for v in sizes], [bool(v) for v in top1], int(fallbacks))

    def _materialize(self):
        if self._host is None:
            from jetspec.engines.jetspec_plus_plus.draft.heads.topk_select import _home
            ids, p64, pack = self.device
            self._host = _home(ids, p64, pack, self._probabilities, self._cap, self._width)
        return self._host

    @property
    def ids(self):
        return dict(enumerate(self._materialize()[0]))

    @property
    def probabilities(self):
        return dict(enumerate(self._materialize()[1]))

    @property
    def set_sizes(self):
        return self._summary[0] if self._summary else [int(v) for v in self._materialize()[2]]

    @property
    def rejected_was_top1(self):
        return self._summary[1] if self._summary else list(self._materialize()[3])

    @property
    def lac_fallbacks(self):
        return self._summary[2] if self._summary else self._materialize()[4]


def _topk_offers(logits, rejected, q_hats, cap, *, run_config=None):
    """JETSPEC_ANCHOR_SELECT=topk: the exact top-k selection (heads/topk_select.py), eager, or
    one CUDA graph per shape under the graphed rows regime; offers come home in one copy."""
    from jetspec.engines.jetspec_plus_plus.profiles import kernel_enabled
    global _GRAPHED_SELECT
    from jetspec.engines.jetspec_plus_plus.draft.heads.topk_select import GraphedSelect
    from jetspec.engines.jetspec_plus_plus.draft.heads.topk_select import fast_offers_topk
    from jetspec.engines.jetspec_plus_plus.draft.heads.topk_select import fast_offers_topk_home
    if (setting(run_config, 'method', 'method_settings.anchor_select_home', None) == 'one'
            and not (logits.device.type == 'cuda' and setting(run_config, 'engine', 'engine_settings.rows_graphs', None) == '1')
            and isinstance(q_hats, (tuple, int, float))):
        if (setting(run_config, 'engine', 'engine_settings.rows_assembly', '0') == '1'
                and setting(run_config, 'method', 'method_settings.anchor_select_graph', '0') == '1'
                and torch.as_tensor(logits).device.type == 'cuda'):
            # serving: the same selection body replayed as one CUDA graph (the graphed regime's
            # GraphedSelect), left on the device for the assembly stage
            if _GRAPHED_SELECT is None:
                _GRAPHED_SELECT = GraphedSelect()
            body, cap, width = _GRAPHED_SELECT.device(logits, rejected, q_hats, cap)
            return DeviceAnchorOffers(*body, cap, width)
        if setting(run_config, 'engine', 'engine_settings.rows_assembly', '0') == '1':
            # serving: the same eager selection body, left on the device for the assembly stage
            from jetspec.engines.jetspec_plus_plus.draft.heads.topk_select import SLACK
            from jetspec.engines.jetspec_plus_plus.draft.heads.topk_select import _check
            from jetspec.engines.jetspec_plus_plus.draft.heads.topk_select import _q_pair
            from jetspec.engines.jetspec_plus_plus.draft.heads.topk_select import _select_body
            logits = torch.as_tensor(logits).float()
            rejected = torch.as_tensor(rejected, device=logits.device, dtype=torch.long)
            _check(logits, rejected, cap)
            cap = int(cap)
            width = min(logits.shape[1], cap + 1 + int(SLACK))
            q_a, q_b = _q_pair(q_hats)
            return DeviceAnchorOffers(*_select_body(logits, rejected, q_a, q_b, cap, width), cap, width)
        # the eager selection, its offers home in one device-to-host copy (serving)
        ids, probabilities, sizes, top1, fallbacks = fast_offers_topk_home(
            logits, rejected, q_hats, cap)
        return AnchorOffers(dict(enumerate(ids)), dict(enumerate(probabilities)),
                            [int(size) for size in sizes], list(top1), fallbacks)
    if logits.device.type == 'cuda' and setting(run_config, 'engine', 'engine_settings.rows_graphs', None) == '1':
        if _GRAPHED_SELECT is None:
            _GRAPHED_SELECT = GraphedSelect()
        if (setting(run_config, 'method', 'method_settings.anchor_select_home', None) == 'one'
                and setting(run_config, 'engine', 'engine_settings.rows_assembly', '0') == '1'
                and isinstance(q_hats, (tuple, int, float))):
            # serving: the graphed regime's same selection graph, its offers left on the device
            # for the assembly stage (serving wired only the eager regime)
            body, cap, width = _GRAPHED_SELECT.device(logits, rejected, q_hats, cap)
            return DeviceAnchorOffers(*body, cap, width)
        if setting(run_config, 'method', 'method_settings.anchor_select_home', None) == 'one':
            # the same offers, home in one device-to-host copy
            ids, probabilities, sizes, top1, fallbacks = _GRAPHED_SELECT.home(
                logits, rejected, q_hats, cap)
            return AnchorOffers(dict(enumerate(ids)), dict(enumerate(probabilities)),
                                [int(size) for size in sizes], list(top1), fallbacks)
        ids, probabilities, sizes, top1, fallbacks = _GRAPHED_SELECT(
            logits, rejected, q_hats, cap)
    else:
        ids, probabilities, sizes, top1, fallbacks = fast_offers_topk(
            logits, rejected, q_hats, cap)
    sizes = [int(size) for size in sizes]
    total = sum(sizes)
    host = torch.cat((torch.cat(ids).long(),
                      torch.cat(probabilities).double().view(torch.int64))).cpu()
    ids = host[:total].split(sizes)
    probabilities = host[total:].view(torch.float64).split(sizes)
    return AnchorOffers(dict(enumerate(ids)), dict(enumerate(probabilities)), sizes,
                        list(top1), fallbacks)


_THRESHOLD_FALLBACK_LOGGED = set()


def _threshold_dataset(dataset, coverage, *, run_config=None):
    """Serve an uncalibrated benchmark from the gsm8k anchor entry only where that is behaviour-identical.

    The e6v2 bundle carries anchor thresholds for gsm8k and math500 only. Under E04_ROWRULE=joint the anchor
    q_hat at coverage 0.99 is exactly 1.0 for every dataset (the joint rule needs all 32 offers), so reading the
    gsm8k entry cannot change served behaviour. Any other combination still refuses below.
    """
    if dataset in ('gsm8k', 'math500') or str(coverage) != '0.99' or setting(run_config, 'method', 'method_settings.rowrule', None) != 'joint':
        return dataset
    if dataset not in _THRESHOLD_FALLBACK_LOGGED:
        _THRESHOLD_FALLBACK_LOGGED.add(dataset)
        print(f"DECODER_THRESHOLD_FALLBACK dataset={dataset} -> gsm8k coverage=0.99 rowrule=joint", flush=True)
    return 'gsm8k'


def e04_joint_offers(logits, rejected, *, root, dataset, coverage, grouped, cap, run_config=None):
    """e04 opt-in (E04_ROWRULE=joint, fast path only): the decoder offers in one device batch, exact without fallback.

    anchor_offers converts its inputs with torch.as_tensor, which the serving process's default-device context
    (model_runner sets torch.set_default_device('cpu')) resolves to the host: the fast path's softmax and LAC run on
    the CPU. At the joint rule's pool (anchor q-hat 1.0, cap 32) a bf16 probability tie crosses the cap on about 5 of
    the 15 rows of a round, and each crossing sends fast_lac_candidates to lac_candidates, a full-vocabulary stable
    sort on the host (about 14 ms each). Here the post-exclusion softmax and the LAC top-cap run on the logits' own
    device for all rows at once and are exact without a fallback: every kept member above the cap-th largest kept
    probability, plus the lowest-id kept members equal to it (as many as the cap leaves), ordered by (probability
    descending, id ascending), which are the ids and probabilities lac_candidates returns. The sizes, the
    d_rej-was-top-1 flags, the input checks, the ids and the probabilities come to the host in one copy, and the
    offers are returned on the host, as the fast path returns them in the serving process; lac_fallbacks is 0.
    With the fast path off it returns None and anchor_offers' own path runs.
    """
    if setting(run_config, 'method', 'method_settings.rowrule', None) != 'joint':
        raise ValueError('E04_ROWRULE must be unset or joint')
    if setting(run_config, 'engine', 'engine_settings.heads_fast', None) != '1':
        return None
    logits = (logits if isinstance(logits, torch.Tensor) else torch.as_tensor(logits)).float()
    rejected = (rejected if isinstance(rejected, torch.Tensor) else torch.as_tensor(rejected)).to(
        device=logits.device, dtype=torch.long)
    if logits.ndim != 2 or rejected.shape != logits.shape[:1]:
        raise ValueError('decoder logits and rejected ids must have shapes [15,V] and [15]')
    dataset = _threshold_dataset(dataset, coverage, run_config=run_config)
    if not dataset or str(coverage) not in ('0.94', '0.99'):
        raise ValueError('decoder selection requires a known dataset and coverage 0.94 or 0.99')
    cap = int(cap)
    if cap < 1:
        raise ValueError('fast LAC cap must be positive')
    from jetspec.engines.jetspec_plus_plus.models.fast import read_thresholds
    thresholds = read_thresholds(run_config)['anchor_predictor']
    rows, vocab = logits.shape
    valid = rejected >= 0
    exclude = torch.zeros((rows, vocab), dtype=torch.bool, device=logits.device).scatter_(
        1, rejected.clamp(0, vocab - 1)[:, None], valid[:, None])
    before_top1 = logits.argmax(-1) == rejected
    probabilities = logits.masked_fill(exclude, float('-inf')).softmax(-1).double()
    if grouped:
        values = dataset_entry(thresholds['drej_top1_grouped'], dataset)[str(coverage)]
        q_hats = torch.where(before_top1, probabilities.new_tensor(float(values['drej-top1']['value'])),
                             probabilities.new_tensor(float(values['not-drej-top1']['value'])))
    else:
        q_hats = probabilities.new_full((rows,), float(dataset_entry(thresholds['pooled'], dataset)[str(coverage)]['value']))
    keep = (probabilities >= (1.0 - q_hats[:, None])) & ~exclude
    masked = probabilities.masked_fill(~keep, float('-inf'))
    width = min(cap, vocab)
    edge = torch.topk(masked, k=width, dim=-1, largest=True, sorted=True).values[:, width - 1:width]
    above = masked > edge
    tied = keep & (masked == edge)
    chosen = above | (tied & (tied.cumsum(-1) <= width - above.sum(-1, keepdim=True)))
    values, ids = torch.topk(masked.masked_fill(~chosen, float('-inf')), k=width, dim=-1, largest=True, sorted=False)
    order = torch.argsort(ids, dim=-1, stable=True)
    ids, values = ids.gather(1, order), values.gather(1, order)
    order = torch.argsort(values, dim=-1, descending=True, stable=True)
    ids, values = ids.gather(1, order), values.gather(1, order)
    bad = (~torch.isfinite(probabilities).all()) | (~((q_hats >= 0) & (q_hats <= 1)).all()) | \
        (rejected >= vocab).any()
    host = torch.cat((keep.sum(-1).clamp_max(width).double(), before_top1.double(), bad.double().reshape(1),
                      ids.double().reshape(-1), values.reshape(-1))).cpu()
    if bool(host[2 * rows] != 0):
        raise ValueError('fast LAC probabilities, q values and rejected ids must be finite and in range')
    sizes = [int(value) for value in host[:rows].tolist()]
    host_ids = host[2 * rows + 1:2 * rows + 1 + rows * width].reshape(rows, width).long()
    host_values = host[2 * rows + 1 + rows * width:].reshape(rows, width)
    return AnchorOffers({k: host_ids[k, :size] for k, size in enumerate(sizes)},
                        {k: host_values[k, :size] for k, size in enumerate(sizes)},
                        sizes, [bool(value) for value in host[rows:2 * rows].tolist()], 0)


def anchor_offers(logits, rejected, *, root, dataset, coverage, grouped, cap, run_config=None):
    """Remove d_rej, then apply the pinned dataset/group LAC threshold."""
    if setting(run_config, 'method', 'method_settings.rowrule', None) and None == '1':
        # serving: heads-lane patch 0010 behind E04_JOINT_OFFERS=1 (unset = the 0009 path)
        offers = e04_joint_offers(logits, rejected, root=root, dataset=dataset, coverage=coverage,
                                  grouped=grouped, cap=cap, run_config=run_config)
        if offers is not None:
            return offers
    logits = keep_device(logits).float()
    rejected = torch.as_tensor(rejected, device=logits.device, dtype=torch.long)
    if logits.ndim != 2 or rejected.shape != logits.shape[:1]:
        raise ValueError('decoder logits and rejected ids must have shapes [15,V] and [15]')
    dataset = _threshold_dataset(dataset, coverage, run_config=run_config)
    if not dataset or str(coverage) not in ('0.94', '0.99'):
        raise ValueError('decoder selection requires a known dataset and coverage 0.94 or 0.99')
    profile = None == '1'
    if profile and logits.device.type == 'cuda':
        torch.cuda.synchronize(logits.device)
    profile_started = time.perf_counter()
    fast_path = setting(run_config, 'engine', 'engine_settings.heads_fast', None) == '1'
    budget = setting(run_config, 'method', 'method_settings.row_select', None) == 'budget'
    if budget:
        pass  # Budget offers do not read calibrated coverage thresholds.
    elif fast_path:
        from jetspec.engines.jetspec_plus_plus.models.fast import fast_lac_candidates
        from jetspec.engines.jetspec_plus_plus.models.fast import read_thresholds
        thresholds = read_thresholds(run_config)['anchor_predictor']
    else:
        from jetspec.engines.jetspec_plus_plus.models.fast import read_thresholds
        thresholds = read_thresholds(run_config)['anchor_predictor']
    if profile and logits.device.type == 'cuda':
        torch.cuda.synchronize(logits.device)
    threshold_ms = (time.perf_counter() - profile_started) * 1000.0
    if setting(run_config, 'method', 'method_settings.anchor_select', None) == 'topk' and (budget or fast_path):
        if budget:
            q_hats = 1.0
        elif grouped:
            values = dataset_entry(thresholds['drej_top1_grouped'], dataset)[str(coverage)]
            q_hats = (float(values['drej-top1']['value']),
                      float(values['not-drej-top1']['value']))
        else:
            q_hats = float(dataset_entry(thresholds['pooled'], dataset)[str(coverage)]['value'])
        lac_started = time.perf_counter()
        result = _topk_offers(logits, rejected, q_hats, cap, run_config=run_config)
        if profile:
            if logits.device.type == 'cuda':
                torch.cuda.synchronize(logits.device)
            print('W278_OFFER_PROFILE ' + json.dumps(dict(
                thresholds_ms=threshold_ms, softmax_ms=0.0,
                lac_ms=(time.perf_counter() - lac_started) * 1000.0,
                fast=fast_path, select='topk', lac_fallbacks=result.lac_fallbacks)), flush=True)
        return result
    softmax_started = time.perf_counter()
    before_top1 = logits.argmax(-1) == rejected
    filtered = logits.clone()
    valid = rejected >= 0
    reject_index = rejected.clamp_min(0)[:, None]
    replacement = torch.where(valid, logits.new_full(valid.shape, float('-inf')),
                              filtered[:, 0])
    filtered.scatter_(1, reject_index, replacement[:, None])
    probabilities = filtered.softmax(-1).double()
    if profile and logits.device.type == 'cuda':
        torch.cuda.synchronize(logits.device)
    softmax_ms = (time.perf_counter() - softmax_started) * 1000.0
    lac_started = time.perf_counter()
    if budget:
        from jetspec.engines.jetspec_plus_plus.models.fast import fast_lac_candidates
        # q=1 admits every non-rejected token; this reuses stable top-cap
        # selection (including tied boundaries) without a coverage cutoff.
        row_ids, row_probabilities, fallbacks, top1 = fast_lac_candidates(
            probabilities, probabilities.new_ones(logits.shape[0]),
            cap=cap, rejected=rejected, also=before_top1)
        result = AnchorOffers(
            dict(enumerate(row_ids)), dict(enumerate(row_probabilities)),
            [int(value.numel()) for value in row_ids],
            list(map(bool, top1)), fallbacks)
    elif fast_path:
        if grouped:
            values = dataset_entry(thresholds['drej_top1_grouped'], dataset)[str(coverage)]
            top = float(values['drej-top1']['value'])
            other = float(values['not-drej-top1']['value'])
            q_hats = torch.where(before_top1, probabilities.new_tensor(top),
                                 probabilities.new_tensor(other))
        else:
            q_hats = probabilities.new_full(
                (logits.shape[0],),
                float(dataset_entry(thresholds['pooled'], dataset)[str(coverage)]['value']))
        row_ids, row_probabilities, fallbacks = fast_lac_candidates(
            probabilities, q_hats, cap=cap, rejected=rejected)
        result = AnchorOffers(
            {k: value for k, value in enumerate(row_ids)},
            {k: value for k, value in enumerate(row_probabilities)},
            [int(value.numel()) for value in row_ids],
            list(map(bool, before_top1.tolist())), fallbacks)
    else:
        ids, selected, sizes = {}, {}, []
        for k in range(logits.shape[0]):
            if grouped:
                group = 'drej-top1' if bool(before_top1[k]) else 'not-drej-top1'
                q_hat = dataset_entry(thresholds['drej_top1_grouped'], dataset)[str(coverage)][group]['value']
            else:
                q_hat = dataset_entry(thresholds['pooled'], dataset)[str(coverage)]['value']
            row_ids, row_probabilities = lac_candidates(
                probabilities[k], q_hat=q_hat, cap=cap,
                excluded=() if int(rejected[k]) < 0 else {int(rejected[k])})
            ids[k], selected[k] = row_ids, row_probabilities
            sizes.append(int(row_ids.numel()))
        result = AnchorOffers(ids, selected, sizes,
                              list(map(bool, before_top1.tolist())))
    if profile:
        if logits.device.type == 'cuda':
            torch.cuda.synchronize(logits.device)
        print('W278_OFFER_PROFILE ' + json.dumps(dict(
            thresholds_ms=threshold_ms, softmax_ms=softmax_ms,
            lac_ms=(time.perf_counter() - lac_started) * 1000.0,
            fast=fast_path, lac_fallbacks=result.lac_fallbacks)), flush=True)
    return result


from jetspec.engines.jetspec_plus_plus.draft.method.decoder.window import DecoderPrediction


def _load_target_prefix_layers(root, config, *, device, dtype, indices, key_prefix,
                               file_pins=None, run_config=None):
    """Load the frozen target tail for reference prefix K/V."""
    root = Path(root)
    if file_pins is not None:
        if not isinstance(file_pins, dict) or not file_pins:
            raise ValueError('file decoder requires target file pins')
        def verify(name):
            if name not in file_pins or Path(name).is_absolute() or '..' in Path(name).parts:
                raise ValueError(f'target-slice decoder lacks safe pin for {name}')
            actual = _sha256(root / name)
            if actual != file_pins[name]:
                raise ValueError(f'target-slice decoder sha256 mismatch: {name} expected={file_pins[name]} actual={actual}')
        verify('model.safetensors.index.json')
        if 'config.json' in file_pins:
            verify('config.json')
    index = json.loads((root / 'model.safetensors.index.json').read_text())['weight_map']
    layers = _tail_layers(config, indices, run_config=run_config)
    state = {}
    expected = {}
    for local_index, target_index in enumerate(indices):
        for name in layers[local_index].state_dict():
            full_name = f'{key_prefix}layers.{target_index}.{name}'
            if full_name not in index:
                raise RuntimeError(f'target checkpoint lacks {full_name}')
            expected[full_name] = f'{local_index}.{name}'
    by_shard = {}
    for full_name in expected:
        by_shard.setdefault(index[full_name], []).append(full_name)
    if file_pins is not None:
        for shard in by_shard:
            verify(shard)
    for shard, names in by_shard.items():
        with safe_open(str(root / shard), framework='pt', device='cpu') as handle:
            for full_name in names:
                state[expected[full_name]] = handle.get_tensor(full_name)
    layers.load_state_dict(state, strict=True)
    run_manifest.record_file('decoder_prefix/index', root / 'model.safetensors.index.json', run_config=run_config)
    for shard in sorted(by_shard):
        run_manifest.record_file(f'decoder_prefix/{shard}', root / shard, run_config=run_config)
    return layers.to(device=device, dtype=dtype).eval().requires_grad_(False)


def _load_target_norm_weight(root, key_prefix, *, run_config=None):
    """The target's final norm weight: the trained decoders learned against it."""
    root = Path(root)
    name = f'{key_prefix}norm.weight'
    index = json.loads((root / 'model.safetensors.index.json').read_text())['weight_map']
    if name not in index:
        raise RuntimeError(f'target checkpoint lacks {name}')
    shard = index[name]
    with safe_open(str(root / shard), framework='pt', device='cpu') as handle:
        weight = handle.get_tensor(name)
    run_manifest.record_file(f'decoder_norm/{shard}', root / shard, run_config=run_config)
    return weight


def _decoder_indices(config, tap_ids):
    if not tap_ids:
        raise ValueError('target-slice decoder requires adapter target_layer_ids')
    deepest = max(int(index) for index in tap_ids)
    indices = tuple(range(deepest + 1, int(config.num_hidden_layers)))
    if not indices:
        raise ValueError(f'target-slice decoder requires layers after tap {deepest}')
    return indices


def _decoder_config(target_path, drafter):
    root = Path(target_path)
    raw = json.loads((root / 'config.json').read_text())
    raw = raw.get('text_config', raw)
    tap_ids = getattr(getattr(drafter.adapter, 'head', None), 'target_layer_ids', None)
    spec = ModelSpec.from_configs(raw, {'target_layer_ids': tap_ids})
    indices = spec.tail
    if not indices:
        raise ValueError('target-slice decoder requires layers after its deepest tap')
    config = Qwen3Config(**raw)
    config._attn_implementation = 'sdpa'
    config._tail_spec = spec
    key_prefix = spec.weight_prefix
    index_path = root / 'model.safetensors.index.json'
    if index_path.is_file():
        names = json.loads(index_path.read_text())['weight_map']
        suffix = f'layers.{indices[0]}.input_layernorm.weight'
        candidates = [name[:-len(suffix)] for name in names if name.endswith(suffix)]
        if key_prefix + suffix not in names and len(candidates) == 1:
            spec = replace(spec, weight_prefix=candidates[0])
            config._tail_spec = spec
            key_prefix = spec.weight_prefix
    return config, indices, key_prefix


def _target_slice_state(root, config, indices, key_prefix, file_pins, *, run_config=None):
    """Read the decoder tail, target norm and LM head only after checking file pins."""
    root = Path(root)
    if key_prefix is None:
        raise ValueError(f'target-slice decoder missing tensor for layer {indices[0]}')
    index_name = 'model.safetensors.index.json'
    if index_name not in file_pins:
        raise ValueError(f'target-slice decoder lacks pin for {index_name}')
    def verify(name):
        if name not in file_pins or Path(name).is_absolute() or '..' in Path(name).parts:
            raise ValueError(f'target-slice decoder lacks safe pin for {name}')
        path = root / name
        actual = _sha256(path)
        if actual != file_pins[name]:
            raise ValueError(f'target-slice decoder sha256 mismatch: {name} expected={file_pins[name]} actual={actual}')
        run_manifest.record_file(f'heads/decoder_target/{name}', path, run_config=run_config)
    verify(index_name)
    if 'config.json' in file_pins:
        verify('config.json')
    index = json.loads((root / index_name).read_text())['weight_map']
    layers = _tail_layers(config, indices, run_config=run_config)
    names = {f'{key_prefix}layers.{target}.{name}': f'{local}.{name}'
             for local, target in enumerate(indices)
             for name in layers[local].state_dict()}
    names.update({f'{key_prefix}norm.weight': 'norm.weight', 'lm_head.weight': 'lm_head.weight'})
    for name in names:
        if name not in index:
            raise ValueError(f'target-slice decoder missing tensor {name}')
    shards = sorted({index[name] for name in names})
    for shard in shards:
        verify(shard)
    state = {}
    by_shard = {shard: [] for shard in shards}
    for name in names:
        by_shard[index[name]].append(name)
    for shard, keys in by_shard.items():
        with safe_open(str(root / shard), framework='pt', device='cpu') as handle:
            for name in keys:
                try:
                    state[names[name]] = handle.get_tensor(name)
                except (KeyError, SafetensorError) as exc:
                    raise ValueError(f'target-slice decoder missing tensor {name} in {shard}') from exc
    layers.load_state_dict({key: state[key] for key in layers.state_dict()}, strict=True, assign=True)
    return layers, state


def _check_reused_fallback(layers, root, expected_sha256, *, run_config=None):
    fallback = Path(root) / 'decoder/checkpoint-0.pt'
    if not fallback.is_file():
        return
    actual = _sha256(fallback)
    if actual != expected_sha256:
        raise ValueError(f'reused decoder fallback sha256 mismatch: {fallback} expected={expected_sha256} actual={actual}')
    reference = torch.load(fallback, map_location='cpu', weights_only=True)['layers']
    if set(reference) != set(layers.state_dict()):
        raise ValueError('reused decoder fallback tensor keys mismatch')
    for name, tensor in layers.state_dict().items():
        pinned = reference[name]
        if (tensor.shape != pinned.shape or tensor.dtype != pinned.dtype or
                not torch.equal(tensor.contiguous().view(torch.uint8), pinned.contiguous().view(torch.uint8))):
            raise ValueError(f'reused decoder fallback tensor mismatch: {name}')
    run_manifest.record_file('heads/decoder_fallback', fallback, run_config=run_config)
    print(f'REUSED_DECODER_LOAD_EQUALITY=PASS fallback=decoder/checkpoint-0.pt bitwise_tensors={len(reference)}', flush=True)


def tail_mlp(layer, hidden, gate_glue=None):
    if gate_glue is None and kernel_value('draft_fuse_g27d', '0') == '1' and hidden.is_cuda:
        from jetspec.engines.jetspec_plus_plus.kernels.qwen3p8_27b import decoder_tail as gate_glue
    source = layer.post_attention_layernorm(hidden)
    if kernel_value('draft_fuse_g27f', '0') == '1':
        from jetspec.engines.jetspec_plus_plus.kernels.qwen3p8_27b.activation_quant import share_activation
        share_activation(source)
    mlp = layer.mlp
    if gate_glue is None and type(mlp).forward is not nn.Module.forward:
        return hidden + mlp(source)
    gate, up = mlp.gate_proj(source), mlp.up_proj(source)
    product = (gate_glue.gate_mul(up, gate, activation='silu') if gate_glue is not None
               else F.silu(gate) * up)
    return hidden + mlp.down_proj(product)


def gdn_project(gdn, hidden):
    if kernel_value('draft_fuse_g27f', '0') == '1':
        from jetspec.engines.jetspec_plus_plus.kernels.qwen3p8_27b.activation_quant import share_activation
        share_activation(hidden)
    raw = gdn.in_proj_qkv(hidden)
    z = gdn.in_proj_z(hidden).reshape(-1, gdn.value_dim)
    b = gdn.in_proj_b(hidden)
    if kernel_value('draft_fuse_g27d', '0') == '1' and hidden.is_cuda:
        from jetspec.engines.jetspec_plus_plus.kernels.qwen3p8_27b.decoder_tail import gdn_controls
        a = gdn.in_proj_a(hidden)
        beta, g = gdn_controls(a, b, gdn.dt_bias, gdn.A_log)
        return raw, z, beta, g
    beta = b.sigmoid()
    a = gdn.in_proj_a(hidden)
    g = -gdn.A_log.float().exp() * F.softplus(a.float() + gdn.dt_bias)
    return raw, z, beta, g


def gdn_finish(gdn, output, z, *, projection_dtype=None):
    output = output.reshape(-1, gdn.value_dim)
    if gdn._recurrent is None:
        value = output.float()
        output = (value * torch.rsqrt(value.square().mean(-1, keepdim=True) + gdn.eps)
                  * gdn.norm.weight.float() * F.silu(z.float())).to(
                      output.dtype if projection_dtype is None else projection_dtype)
    else:
        output = gdn.norm(output, z)
    output = output.reshape(-1, gdn.value_width)
    return gdn.out_proj(output if projection_dtype is None else output.to(projection_dtype))


def stateful_attention_project(layer, hidden, positions, rope_theta, *, refresh_positions=None):
    source = layer.input_layernorm(hidden)
    if kernel_value('draft_fuse_g27f', '0') == '1':
        from jetspec.engines.jetspec_plus_plus.kernels.qwen3p8_27b.activation_quant import share_activation
        share_activation(source)
    attn, width = layer.self_attn, hidden.shape[0]
    output_gate = getattr(getattr(attn, 'spec', None), 'output_gate', True)
    projected = attn.q_proj(source)
    if output_gate:
        q, gate = projected.view(width, attn.heads, 2 * attn.dim).chunk(2, dim=-1)
    else:
        q, gate = projected.view(width, attn.heads, attn.dim), None
    q = attn.q_norm(q)
    key = attn.k_norm(attn.k_proj(source).view(width, attn.kv_heads, attn.dim))
    value = attn.v_proj(source).view(width, attn.kv_heads, attn.dim)
    table = getattr(attn, '_g27_rotary_table', None)
    if table is not None and kernel_enabled('draft_fuse_g27', '1'):
        from jetspec.engines.jetspec_plus_plus.draft.glue import g27_rotary
        extra = {} if refresh_positions is None else {'refresh_positions': refresh_positions}
        q, key = g27_rotary(q, key, positions, table, **extra)
        return q, gate, key, value
    frequency = 1.0 / (rope_theta ** (torch.arange(
        0, attn.rotary_dim, 2, device=hidden.device, dtype=torch.float32) / attn.rotary_dim))
    angles = positions.float()[:, None] * frequency[None]
    angles = torch.cat((angles, angles), -1)
    cos, sin = angles.cos().to(q.dtype)[:, None], angles.sin().to(q.dtype)[:, None]
    def rotate(tensor):
        rotary = tensor[..., :attn.rotary_dim]
        half = attn.rotary_dim // 2
        shifted = torch.cat((-rotary[..., half:], rotary[..., :half]), -1)
        return torch.cat((rotary * cos + shifted * sin, tensor[..., attn.rotary_dim:]), -1)
    return rotate(q), gate, rotate(key), value


def stateful_attention_finish(layer, hidden, attended, gate, gate_glue=None):
    if gate_glue is None and kernel_value('draft_fuse_g27d', '0') == '1' and hidden.is_cuda:
        from jetspec.engines.jetspec_plus_plus.kernels.qwen3p8_27b import decoder_tail as gate_glue
    product = attended if gate is None else (
        gate_glue.gate_mul(attended, gate) if gate_glue is not None else attended * gate.sigmoid())
    return tail_mlp(layer, hidden + layer.self_attn.o_proj(product.reshape(hidden.shape[0], -1)),
                    gate_glue)


def scan_gdn_window(gdn, hidden, recurrent, conv, *, return_state=False):
    """Run one decoder window without changing the target or copied state."""
    if recurrent.dtype != torch.float32:
        raise ValueError("decoder GDN recurrent state must stay fp32")
    if hidden.ndim != 2 or recurrent.shape != gdn.committed_recurrent.shape \
            or conv.shape != gdn.committed_conv.shape:
        raise ValueError("decoder GDN state or window shape changed")
    width = hidden.shape[0]
    x = hidden.unsqueeze(0)
    raw, z, beta, g = gdn_project(gdn, x)
    raw = raw.transpose(1, 2)
    conv_input = torch.cat((conv.to(raw.dtype), raw), dim=-1)
    if kernel_value('draft_fuse_g27g', '0') == '1' and hidden.is_cuda and hidden.dtype == torch.bfloat16:
        from jetspec.engines.jetspec_plus_plus.kernels.qwen3p8_27b.gdn_silu import gdn_inputs
        convolved = gdn.conv1d(conv_input)[:, :, :conv_input.shape[-1]][0, :, -width:].T
        q, k, v, initial = gdn_inputs(convolved, gdn.key_heads,
            gdn.value_heads // gdn.key_heads, gdn.key_dim, gdn.value_dim, state=recurrent)
        q, k, v = q[None], k[None], v[None]
    else:
        convolved = F.silu(gdn.conv1d(conv_input)[:, :, :conv_input.shape[-1]])
        convolved = convolved[:, :, -width:].transpose(1, 2)
        q, k, v = convolved.split((gdn.key_width, gdn.key_width, gdn.value_width), dim=-1)
        q = q.reshape(1, width, gdn.key_heads, gdn.key_dim)
        k = k.reshape(1, width, gdn.key_heads, gdn.key_dim)
        v = v.reshape(1, width, gdn.value_heads, gdn.value_dim)
        repeat = gdn.value_heads // gdn.key_heads
        if repeat > 1:
            q = q.repeat_interleave(repeat, dim=2)
            k = k.repeat_interleave(repeat, dim=2)
        initial = recurrent.clone()
    tuned = None
    if kernel_value('draft_fuse_gdn27', '0') == '1' and hidden.is_cuda:
        from jetspec.engines.jetspec_plus_plus.kernels.qwen3p8_27b.draft_gdn27 import recurrent as tuned_recurrent
        tuned = tuned_recurrent(q, k, v, g, beta, initial, output_final_state=True)
    if tuned is None:
        output, final = gdn._delta(q, k, v, g, beta, initial, chunk=False)
    else:
        output, final = tuned
    projected = gdn_finish(gdn, output, z)
    if return_state:
        return projected, final, conv_input[:, :, -gdn.conv_kernel:]
    return projected


def hybrid_gdn_tail(gdn_layer, hidden, *, recurrent, conv, **state):
    """The target GDN layer reads a private copy of committed recurrence."""
    first = gdn_layer.input_layernorm(hidden)
    x = hidden + scan_gdn_window(gdn_layer.linear_attn, first, recurrent, conv)
    return tail_mlp(gdn_layer, x)


def hybrid_attention_tail(attention_layer, hidden, *, prefix_key, prefix_value,
                          start, rope_theta, window_kv=None, **state):
    """The target attention layer appends only private window K/V."""
    if prefix_key.shape != prefix_value.shape or prefix_key.ndim != 3 \
            or (isinstance(start, int) and prefix_key.shape[0] != start):
        raise ValueError("hybrid decoder needs target attention KV through the committed prefix")
    width = hidden.shape[0]
    attn = attention_layer.self_attn
    if prefix_key.shape[1:] != (attn.kv_heads, attn.dim):
        raise ValueError("decoder prefix KV has the wrong head geometry")
    q, gate, key, value = stateful_attention_project(attention_layer, hidden,
        torch.arange(width, device=hidden.device, dtype=torch.float32) + start, rope_theta)
    from jetspec.engines.jetspec_plus_plus.kernels import draft_splitkv
    if kernel_enabled('draft_fuse_g27', '1') and draft_splitkv.a256_enabled(q, 'decoder'):
        from jetspec.engines.jetspec_plus_plus.draft.glue import g27_attention_inputs
        all_key, all_value, bias = g27_attention_inputs(
            prefix_key, prefix_value, key, value, start, start, lanes=width)
        attended = draft_splitkv.attend(q.transpose(0, 1)[None],
            all_key.transpose(0, 1)[None], all_value.transpose(0, 1)[None],
            attn.dim ** -0.5, mode='dense', mask=bias)[0].transpose(0, 1)
        x = stateful_attention_finish(attention_layer, hidden, attended, gate)
        if window_kv is not None:
            window_kv['kv'] = (key, value)
        return x
    all_key = torch.cat((prefix_key, key), dim=0)
    all_value = torch.cat((prefix_value, value), dim=0)
    groups = attn.heads // attn.kv_heads
    queries = q.transpose(0, 1)[None]
    prefix_allowed = (torch.arange(prefix_key.shape[0], device=hidden.device)
                      < start).expand(width, -1)
    window_allowed = (torch.arange(width, device=hidden.device)[None, :]
                      <= torch.arange(width, device=hidden.device)[:, None])
    allowed = torch.cat((prefix_allowed, window_allowed), dim=-1)
    from jetspec.engines.jetspec_plus_plus.kernels import draft_splitkv
    if draft_splitkv.a256_enabled(queries, 'decoder'):
        bias = torch.where(allowed[None, None], 0., torch.finfo(queries.dtype).min).to(queries.dtype)
        attended = draft_splitkv.attend(queries, all_key.transpose(0, 1)[None],
            all_value.transpose(0, 1)[None], attn.dim ** -0.5,
            mode='dense', mask=bias)[0].transpose(0, 1)
    else:
        keys = all_key.repeat_interleave(groups, dim=1).transpose(0, 1)[None]
        values = all_value.repeat_interleave(groups, dim=1).transpose(0, 1)[None]
        attended = F.scaled_dot_product_attention(
            queries, keys, values, attn_mask=allowed[None, None],
            dropout_p=0.0, scale=attn.dim ** -0.5)[0].transpose(0, 1)
    x = stateful_attention_finish(attention_layer, hidden, attended, gate)
    if window_kv is not None:
        window_kv['kv'] = (key, value)
    return x


def tail_call(layer, index, hidden, *, layer_states=None, **state):
    if layer_states is not None:
        state.update(layer_states[index])
    call = hybrid_gdn_tail if hasattr(layer, 'linear_attn') else hybrid_attention_tail
    return call(layer, hidden, **state)


def hybrid_tail(layers):
    """Layer callables supplied to the common method AnchorDecoder."""
    return tuple(partial(tail_call, layer, getattr(layer, 'tail_index', index))
                 for index, layer in enumerate(layers))


def hybrid_anchor_window(gdn_layer, attention_layer, norm, lm_head, hidden,
                         recurrent, conv, prefix_key, prefix_value, *, start,
                         rope_theta):
    """Decode deepest-tap rows using committed GDN and target attention KV."""
    window_kv = {}
    decoded = decode_hidden((gdn_layer, attention_layer), hidden,
        tail=hybrid_tail((gdn_layer, attention_layer)), recurrent=recurrent, conv=conv,
        prefix_key=prefix_key, prefix_value=prefix_value, start=start,
        rope_theta=rope_theta, window_kv=window_kv)
    return lm_head(norm(decoded)), window_kv['kv']



class AnchorDecoder(MethodAnchorDecoder):
    """Two trained target layers over one private adapter window per round."""
    def __init__(self, layers, norm, lm_head, *, prefix_layers=None, rotary=None, root=None,
                 hidden_size=None, rope_theta=None, capacity=None, spec=None,
                 state_policy=None, run_config=None):
        self.run_config = run_config
        super().__init__(layers=layers, prefix_layers=prefix_layers, norm=norm,
                         lm_head=lm_head, rotary=rotary, root=root, hidden_size=hidden_size or norm.weight.numel())
        self.spec = spec
        self.tap_position = spec.taps.index(max(spec.taps)) if spec is not None and spec.taps else -1
        self._uses_shipped_state = rope_theta is not None
        self.state_policy = dict(
            {'full_attention': 'ship', 'linear_attention': 'ship'} if self._uses_shipped_state
            else {'full_attention': 'rebuild', 'linear_attention': 'recompute'})
        self.state_policy.update(state_policy or {})
        allowed_sources = {'full_attention': {'ship', 'rebuild'},
                           'linear_attention': {'ship', 'recompute'}}
        if any(kind not in allowed_sources or source not in allowed_sources[kind]
               for kind, source in self.state_policy.items()):
            raise ValueError('unsupported decoder state source')
        if self._uses_shipped_state:
            self.tail = hybrid_tail(layers)
            self.rope_theta = float(rope_theta)
            self.hidden_size = norm.weight.numel()
            self.capacity = capacity
            self._state = None
        self._lm_head_kn = None
        self._prefixes = {}
        self._prefix_finalizers = {}
        self._static_prefix_pool = None
        self._batch_frozen = {}
        self._recorded_inputs = 0
        self._window_graphs = {}
        self._window_graph_states = {}
        self._graphs_warmed = False
        self._window_mask_cache = None
        self._rope_table = None
        self._loop_plans = {}
        self._loop_entries = {}
        self._causal_table = None

    def _tap_hidden(self, hidden):
        """Select the spec's deepest tap from packed adapter predictions."""
        if self.spec is None or hidden.shape[-1] == self.hidden_size:
            return hidden[..., -self.hidden_size:]
        start = self.tap_position * self.hidden_size
        return hidden[..., start:start + self.hidden_size]

    def forward(self, hidden, **state):
        if not self._uses_shipped_state:
            return super().forward(hidden, **state)
        if hidden.ndim == 3:
            count = hidden.shape[0]
            names = ('recurrent', 'conv', 'prefix_key', 'prefix_value')
            if any(state[name].shape[0] != count for name in names if name in state):
                raise ValueError('one private decoder state is required per sequence')
            outputs = []
            for row in range(count):
                private = {name: value[row] if name in names else value
                           for name, value in state.items()}
                if 'layer_states' in private:
                    private['layer_states'] = {index: {name: tensor[row] for name, tensor in values.items()}
                                               for index, values in private['layer_states'].items()}
                if isinstance(private.get('start'), torch.Tensor) and private['start'].ndim:
                    private['start'] = private['start'][row]
                outputs.append(super().forward(hidden[row], **private))
            return tuple(torch.stack([output[i] for output in outputs]) for i in range(2))
        return super().forward(hidden, **state)

    def _sequence_state(self, sequence, length):
        """Resolve each kind's prefix source without writing committed target state."""
        indices = self.spec.tail if self.spec is not None else tuple(range(len(self.layers)))
        supplied = getattr(sequence, 'decoder_state', None)
        if supplied is None and 'ship' in self.state_policy.values():
            self._ensure_warmup_state(sequence)
            supplied = {}
            for index, layer in zip(indices, self.layers):
                if hasattr(layer, 'linear_attn'):
                    supplied[index] = dict(recurrent=sequence.hybrid_recurrent, conv=sequence.hybrid_conv)
                else:
                    supplied[index] = dict(prefix_key=sequence.hybrid_prefix_key[:length],
                                           prefix_value=sequence.hybrid_prefix_value[:length])
        if all(source == 'ship' for source in self.state_policy.values()):
            return {'layer_states': supplied}
        parameter = next(self.layers.parameters())
        hidden = sequence.taps[:length, self.tap_position, :].to(parameter)
        states = {}
        for index, layer in zip(indices, self.layers):
            kind = 'linear_attention' if hasattr(layer, 'linear_attn') else 'full_attention'
            if kind == 'linear_attention':
                gdn = layer.linear_attn
                recurrent = torch.zeros_like(gdn.committed_recurrent)
                conv = torch.zeros_like(gdn.committed_conv)
                if length:
                    mixed, recurrent, conv = scan_gdn_window(gdn,
                        layer.input_layernorm(hidden), recurrent, conv, return_state=True)
                    hidden = tail_mlp(layer, hidden + mixed)
                rebuilt = dict(recurrent=recurrent, conv=conv)
            else:
                attn = layer.self_attn
                empty = hidden.new_empty(0, attn.kv_heads, attn.dim)
                window = {}
                if length:
                    hidden = hybrid_attention_tail(layer, hidden, prefix_key=empty,
                        prefix_value=empty, start=0, rope_theta=self.rope_theta, window_kv=window)
                    key, value = window['kv']
                else:
                    key = value = empty
                rebuilt = dict(prefix_key=key, prefix_value=value)
            states[index] = supplied[index] if self.state_policy[kind] == 'ship' else rebuilt
        return {'layer_states': states}

    def _rope(self, hidden, position_ids, *, run_config=None):
        """Rotary rows gathered from one table on the fused path; each position's
        cos/sin is a single product, so the gather is bitwise the rotary call.
        The graphed lean decoder (JETSPEC_DECODER_GRAPH_LEAN) gathers too."""
        if not (rows_fused(hidden, run_config=self.run_config) or (decoder_graph_lean() and hidden.device.type == 'cuda')):
            return self.rotary(hidden, position_ids)
        capacity = int(setting(self.run_config, 'engine', 'engine_settings.decoder_cache_capacity', '800')) + 32
        table = self._rope_table
        if table is None or table[0].shape[0] < capacity or table[0].device != hidden.device:
            positions = torch.arange(capacity, device=hidden.device)[None]
            cos, sin = self.rotary(hidden, positions)
            table = self._rope_table = (cos[0], sin[0])
        flat = position_ids.reshape(-1)
        return table[0].index_select(0, flat)[None], table[1].index_select(0, flat)[None]

    def _rope_range(self, hidden, start, stop):
        """``_rope`` for the contiguous positions start..stop (JETSPEC_DECODER_TABLES=1,
        serving): the same table rows as a slice instead of a gather.  A row slice of the
        contiguous table is contiguous and 16-byte aligned (a row is 128 values)."""
        table = self._rope_table
        if table is None or table[0].device != hidden.device:
            self._rope(hidden, torch.zeros((1, 1), dtype=torch.long, device=hidden.device))
            table = self._rope_table
        if stop > table[0].shape[0]:
            positions = torch.arange(start, stop, device=hidden.device, dtype=torch.long)[None]
            return self._rope(hidden, positions)
        return table[0][start:stop][None], table[1][start:stop][None]

    def _causal_rows(self, hidden, start, stop):
        """The prefix append's mask (key j visible to position i iff j <= i) for rows
        start..stop over keys 0..stop, as a slice of one cached table whose rows are
        16-aligned (JETSPEC_DECODER_TABLES=1, serving): the same values as the mask ``_prefix``
        builds, already in the layout ``efficient_attention`` pads a mask into."""
        table = self._causal_table
        if (table is None or table.shape[0] < stop or table.device != hidden.device
                or table.dtype != hidden.dtype):
            rows = max(stop, int(setting(self.run_config, 'engine', 'engine_settings.decoder_cache_capacity', '800')))
            width = (rows + 15) // 16 * 16
            keys = torch.arange(width, device=hidden.device)
            positions = torch.arange(rows, device=hidden.device)
            allowed = keys[None, :] <= positions[:, None]
            table = self._causal_table = hidden.new_zeros((rows, width)).masked_fill_(
                ~allowed, torch.finfo(hidden.dtype).min)
        return table[start:stop, :stop][None, None]

    @classmethod
    def load(cls, root, drafter, target_model_path=None, manifest=None, *, run_config=None):
        from jetspec.engines.jetspec_plus_plus.profiles import kernel_enabled
        from jetspec.engines.jetspec_plus_plus.draft.rows import check_rows_fused_regime
        check_rows_fused_regime(run_config=run_config)
        check_decoder_graph_lean_regime(run_config=run_config)
        target_model_path = target_model_path or _required_model_path(run_config)
        raw = json.loads((Path(target_model_path) / 'config.json').read_text())
        spec = ModelSpec.from_configs(raw, {'target_layer_ids': drafter.adapter.head.target_layer_ids})
        if spec.has_state or spec.attention.output_gate:
            manifest_path = Path(root) / 'decoder/manifest.json'
            manifest = manifest if manifest is not None else (json.loads(manifest_path.read_text()) if manifest_path.is_file() else None)
            return cls._load_stateful(root, drafter, target_model_path, manifest, run_config=run_config)
        root = Path(root)
        dtype = drafter.dtype
        device = drafter.device
        manifest_path = root / 'decoder/manifest.json'
        manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else None
        source = None if manifest is None else manifest.get('source')
        if manifest is not None:
            run_manifest.record_file('heads/decoder_manifest', manifest_path, run_config=run_config)
            if source not in ('target_slice', 'file'):
                raise ValueError(f'unknown decoder source: {source}')
        target_model_path = target_model_path or _required_model_path(run_config)
        draft_prefix = source == 'file' and manifest.get('prefix') == 'decoder'
        if source == 'file' and manifest.get('prefix', 'target') not in ('target', 'decoder'):
            raise ValueError(f"unknown decoder prefix: {manifest['prefix']}")
        if draft_prefix:
            config = copy.deepcopy(drafter.head.config)
            config._attn_implementation = 'sdpa'
            indices = (0, 1)
            key_prefix = None
        else:
            config, indices, key_prefix = _decoder_config(target_model_path, drafter)
        if source == 'target_slice':
            layers, state = _target_slice_state(target_model_path, config, indices, key_prefix,
                                                manifest.get('target_files', {}), run_config=run_config)
            _check_reused_fallback(layers, root,
                                   manifest.get('fallback_sha256', REUSED_DECODER_SHA256), run_config=run_config)
            for name, tensor in state.items():
                if tensor.dtype != dtype:
                    raise ValueError(f'target-slice decoder dtype mismatch: {name} target={tensor.dtype} runner={dtype}')
            prefix_layers = layers
            norm = _tail_norm(spec)
            norm.load_state_dict({'weight': state['norm.weight']}, strict=True)
            lm_head = _tail_lm_head(source=drafter.lm_head)
            lm_head.load_state_dict({'weight': state['lm_head.weight']}, strict=True)
            layers = layers.to(device=device, dtype=dtype).eval().requires_grad_(False)
            norm = norm.to(device=device, dtype=dtype).eval().requires_grad_(False)
            lm_head = lm_head.to(device=device, dtype=dtype).eval().requires_grad_(False)
        else:
            if source == 'file':
                if manifest.get('path') != 'decoder/checkpoint-0.pt' or not manifest.get('sha256'):
                    raise ValueError('reused decoder file manifest must pin decoder/checkpoint-0.pt')
                path = root / 'decoder/checkpoint-0.pt'
                actual = _sha256(path)
                if actual != manifest['sha256']:
                    raise ValueError(f"reused decoder fallback sha256 mismatch: {path} expected={manifest['sha256']} actual={actual}")
                state = torch.load(path, map_location='cpu', weights_only=True)['layers']
            else:
                path = root / 'decoder/checkpoint-final-model.pt'
                state = torch.load(path, map_location='cpu', weights_only=True)
            layers = _tail_layers(config, indices, run_config=run_config)
            layers.load_state_dict(state, strict=True)
            run_manifest.record_file('heads/decoder', path, run_config=run_config)
            if draft_prefix:
                prefix_layers = layers
            else:
                prefix_layers = _load_target_prefix_layers(
                    target_model_path, config, device=device, dtype=dtype,
                    indices=indices, key_prefix=key_prefix,
                    file_pins=manifest.get('target_files') if source == 'file' else None, run_config=run_config)
            if source is None:
                # The trained CE and KL decoders learned against the target's final norm,
                # not the drafter head's norm.
                norm = _tail_norm(spec)
                norm.load_state_dict({'weight': _load_target_norm_weight(
                    target_model_path, key_prefix, run_config=run_config)}, strict=True)
            else:
                norm = _tail_norm(spec)
                norm.load_state_dict(drafter.head.norm.state_dict(), strict=True)
            norm = norm.to(device=device, dtype=dtype).eval().requires_grad_(False)
            lm_head = _tail_lm_head(source=drafter.lm_head)
            layers = layers.to(device=device, dtype=dtype).eval().requires_grad_(False)
        # Construct RoPE on the runner device.  The serving process installs a
        # torch default-device context; constructing on CPU and moving later can
        # leave Qwen3's non-persistent ``original_inv_freq`` alias on CPU.
        rotary = Qwen3RotaryEmbedding(config, device=device).to(device=device)
        if any(buffer.device != device for buffer in rotary.buffers()):
            raise RuntimeError('decoder rotary buffers are not on the runner device')
        from jetspec.engines.jetspec_plus_plus.models.target_linear import round_nvfp4_weights
        for module, role in ((layers, 'draft'), (prefix_layers, 'draft'), (lm_head, 'lm_head')):
            round_nvfp4_weights(module, role)
        from jetspec.engines.jetspec_plus_plus.models.draft_vocab import configure_draft_vocab
        configure_draft_vocab(lm_head, run_config)
        if setting(run_config, 'engine', 'engine_settings.heads_fast', None) == '1':
            from jetspec.engines.jetspec_plus_plus.models.fast import read_thresholds
            read_thresholds(run_config)
        decoder = cls(layers=layers, prefix_layers=prefix_layers,
                      norm=norm, lm_head=lm_head,
                      rotary=rotary, root=root, hidden_size=config.hidden_size,
                      state_policy=(manifest or {}).get('state_policy'), run_config=run_config).eval()
        decoder.spec = getattr(config, '_tail_spec', None) or spec
        if decoder.spec is not None:
            decoder.tap_position = decoder.spec.taps.index(max(decoder.spec.taps))
        if kernel_enabled('decoder_lmhead_kn', '1'):
            if source != 'target_slice' or lm_head.bias is not None:
                raise ValueError('JETSPEC_DECODER_LMHEAD_KN requires a bias-free target_slice lm_head')
            decoder._lm_head_kn = lm_head.weight.detach().T.contiguous()
        if kernel_enabled('decoder_qkv_fused', '1'):
            from jetspec.engines.jetspec_plus_plus.kernels.projections import prepare_decoder_qkv
            for layer in (*layers, *prefix_layers):
                if getattr(layer.self_attn, '_decoder_dense', type(layer.self_attn) is Qwen3Attention):
                    prepare_decoder_qkv(layer.self_attn)
        if kernel_enabled('decoder_single_gateup_packed', '1'):
            for layer in (*layers, *prefix_layers):
                _prepare_decoder_gate_up(layer.mlp)
        if (device.type == 'cuda' and setting(run_config, 'engine', 'engine_settings.heads_fast', None) == '1'
                and setting(run_config, 'engine', 'engine_settings.rows_graphs', None) == '1'):
            decoder.warm_graphs()
        elif rows_fused(next(layers.parameters()), run_config=run_config):
            decoder.warm_kernels(run_config=run_config)
        return decoder

    @classmethod
    def _load_stateful(cls, root, drafter, target_model_path, manifest, *, run_config=None):
        import jetspec.engines.jetspec_plus_plus.runtime.run_manifest as run_manifest

        if manifest is None or manifest.get("source") != "target_slice":
            raise ValueError("hybrid decoder requires a pinned target_slice manifest")
        run_manifest.record_file("heads/decoder_manifest", Path(root) / "decoder/manifest.json", run_config=run_config)
        target = Path(target_model_path)
        raw = json.loads((target / "config.json").read_text())
        raw = raw.get('text_config', raw)
        spec = ModelSpec.from_configs(raw, {'target_layer_ids': drafter.adapter.head.target_layer_ids})
        if not spec.tail:
            raise ValueError('decoder needs a nonempty tail after the deepest tap')
        files = manifest.get("target_files", {})
        index_name = "model.safetensors.index.json"
        if not isinstance(files, dict) or index_name not in files or "config.json" not in files:
            raise ValueError("hybrid decoder needs config and index SHA256 pins")
        def verify(name):
            if name not in files or Path(name).is_absolute() or ".." in Path(name).parts:
                raise ValueError(f"hybrid decoder lacks a safe pin for {name}")
            path = target / name
            if _sha256(path) != files[name]:
                raise ValueError(f"hybrid decoder SHA256 mismatch for {name}")
            run_manifest.record_file(f"heads/decoder_target/{name}", path, run_config=run_config)
        verify("config.json")
        verify(index_name)
        config = SimpleNamespace(**raw)
        config.rope_theta = spec.rope_theta
        config._tail_spec = spec
        config.torch_dtype = drafter.dtype
        from jetspec.engines.jetspec_plus_plus.wire.protocol import draft_width
        lanes = draft_width(run_config=run_config) + 1
        with torch.device("meta"):
            layers = _tail_layers(config, spec.tail, run_config=run_config)
            norm = _tail_norm(spec)
            lm_head = _tail_lm_head(spec)
        wanted = {}
        for local, absolute in enumerate(spec.tail):
            for name in layers[local].state_dict():
                wanted[f"{spec.weight_prefix}layers.{absolute}.{name}"] = f"layers.{local}.{name}"
        wanted[f"{spec.weight_prefix}norm.weight"] = "norm.weight"
        wanted["lm_head.weight"] = "lm_head.weight"
        index = json.loads((target / index_name).read_text())["weight_map"]
        missing = set(wanted) - set(index)
        if missing:
            raise ValueError(f"hybrid decoder target lacks {sorted(missing)[:3]}")
        shards = sorted({index[name] for name in wanted})
        state = {}
        for shard in shards:
            verify(shard)
            with safe_open(str(target / shard), framework="pt", device="cpu") as handle:
                for name in wanted:
                    if index[name] == shard:
                        state[wanted[name]] = handle.get_tensor(name)
        module = cls(layers, norm, lm_head, root=root,
                     rope_theta=config.rope_theta, spec=spec, state_policy=manifest.get('state_policy'),
                     capacity=int(setting(run_config, 'engine', 'engine_settings.decoder_cache_capacity', '800')) + lanes, run_config=run_config)
        from jetspec.engines.jetspec_plus_plus.wire.hybrid import HybridPairLayout
        module.layout = HybridPairLayout.from_config(config, lanes=lanes)
        module.to_empty(device="cpu")
        module.load_state_dict(state, strict=True, assign=True)
        if any(value.dtype != drafter.dtype for value in module.parameters()):
            raise ValueError("hybrid decoder target dtype differs from draft dtype")
        module = module.to(device=drafter.device).eval().requires_grad_(False)
        from jetspec.engines.jetspec_plus_plus.engine import current_engine
        engine = current_engine()
        if 'mxfp4_w4a8' in (engine.precision_draft, engine.precision_lm_head):
            # The hybrid loader returns before the dense loader's rounding pass.
            from jetspec.engines.jetspec_plus_plus.models.target_linear import round_nvfp4_weights
            round_nvfp4_weights(module.layers, 'draft')
            round_nvfp4_weights(module.lm_head, 'lm_head')
        return module


    @property
    def bind_front_state(self):
        # Admission still probes this capability until F3/F6 migrate its callers.
        if not self._uses_shipped_state:
            raise AttributeError('this tail has no shipped recurrent state')
        return self._bind_front_state

    def _bind_front_state(self, *, device, dtype, capacity):
        if capacity > self.capacity:
            raise ValueError("hybrid decoder key exceeds its prefix capacity")
        if self._state is None:
            gdn = self.layers[0].linear_attn
            attn = self.layers[1].self_attn
            if setting(getattr(self, 'run_config', None), 'engine', 'engine_settings.draft_fuse_g27', '0') == '1':
                from jetspec.engines.jetspec_plus_plus.draft.glue import g27_rotary_table
                # Prefix capacity plus both the causal tail and one private row.
                from jetspec.engines.jetspec_plus_plus.wire.protocol import draft_width
                rows = self.capacity + 2 * (draft_width(run_config=getattr(self, 'run_config', None)) + 1)
                attn._g27_rotary_table = g27_rotary_table(
                    rows, attn.rotary_dim, self.rope_theta,
                    like=torch.empty((), device=device, dtype=dtype))
            self._state = dict(
                recurrent=torch.empty_like(gdn.committed_recurrent, device=device),
                conv=torch.empty(gdn.committed_conv.shape, device=device, dtype=dtype),
                key=torch.zeros(self.capacity, attn.kv_heads, attn.dim,
                                dtype=dtype, device=device),
                value=torch.zeros(self.capacity, attn.kv_heads, attn.dim,
                                  dtype=dtype, device=device))
        return self._state


    def _stateful_refresh_window_state(self, prefix_kv, bucket, *, rows, lanes, like, glue=None):
        """Bind the target's committed state, and allocate one fixed refresh tree.

        Unlike the dense decoder, the hybrid prefix is current through length-1.
        There are no true-tap lanes: the window is [tail | rows x slots]. All
        projections and MLPs read this single token matrix once per refresh.
        """
        from jetspec.engines.jetspec_plus_plus.profiles import kernel_enabled
        del prefix_kv
        if setting(self.run_config, 'engine', 'engine_settings.refresh_swap_glue', '0') == '1' if glue is None else glue:
            raise ValueError('JETSPEC_REFRESH_SWAP_GLUE refuses the hybrid decoder (GDN, D256)')
        prefix = self.bind_front_state(device=like.device, dtype=like.dtype,
                                       capacity=self.capacity)
        gdn = self.layers[0].linear_attn
        slots = lanes - 1
        tokens = lanes + rows * slots
        lane = torch.arange(tokens, device=like.device)
        is_block = lane >= lanes
        row = ((lane - lanes).clamp_min(0) // slots).clamp(max=rows - 1)
        index = torch.where(is_block, (lane - lanes).clamp_min(0) % slots, lane)
        state = dict(prefix=prefix, bucket=bucket, rows=rows, lanes=lanes,
                     refresh_staging=setting(self.run_config, 'engine', 'engine_settings.hybrid_decoder_refresh_staging', '0') == '1',
                     refresh_copy_staging=setting(getattr(self, 'run_config', None), 'engine', 'engine_settings.draft_fuse_g27b', '0') == '1',
                     refresh_gates=setting(self.run_config, 'engine', 'engine_settings.hybrid_decoder_refresh_gates', '0') == '1',
                     refresh_state_lean=kernel_enabled('hybrid_decoder_refresh_state_lean', '1'),
                     refresh_selected_states=kernel_enabled('hybrid_decoder_refresh_state_lean', '1'),
                     slots=slots, tokens=tokens, lane=lane, is_block=is_block,
                     row=row, index=index,
                     conv_offset=torch.arange(1 - gdn.conv_kernel, 1, device=like.device),
                     prefix_column=torch.arange(self.capacity, device=like.device),
                     recurrent=torch.empty((lanes + 1,) + gdn.committed_recurrent.shape,
                                           device=like.device, dtype=torch.float32))
        if state['refresh_staging'] or state['refresh_copy_staging']:
            state['conv_input'] = like.new_empty(tokens, gdn.conv1d.weight.shape[0], gdn.conv_kernel)
            state['hidden_input'] = like.new_empty(tokens, self.hidden_size)
        if kernel_enabled('refresh_swap_lean', '1') and like.is_cuda and like.dtype == torch.bfloat16:
            from jetspec.engines.jetspec_plus_plus.draft.chain.refresh_swap import allocate_compact
            state['compact'] = allocate_compact(rows, slots, self.lm_head.weight.shape[0], like.device)
        return state


    def _stateful_refresh_window_compact(self, state, *, ids, **kwargs):
        return self.refresh_window_logits(state, _compact_ids=ids, **kwargs)


    def _stateful_refresh_window_logits(self, state, *, length, true_taps, tail, tail_counts,
                              block, slot_counts, _compact_ids=None):
        """Return [rows, lanes-1, vocab] in the dense refresh-swap layout.

        Row r starts from the committed recurrent/conv state plus its own
        tail[:tail_counts[r]], then reads its causal slots. Target K/V below
        length is shared; neither attention nor recurrence crosses rows.
        true_taps is already inside the committed target state and is unused.
        No data-dependent shapes or device-to-host reads occur during capture.
        """
        del true_taps
        rows, lanes, slots = state['rows'], state['lanes'], state['slots']
        tokens, prefix = state['tokens'], state['prefix']
        gdn_layer, attention_layer = self.layers
        gdn = gdn_layer.linear_attn
        gate_glue = None
        if state.get('refresh_gates', False):
            from jetspec.engines.jetspec_plus_plus.kernels import decoder_refresh as gate_glue
        state_glue = None
        if state.get('refresh_state_lean', False):
            from jetspec.engines.jetspec_plus_plus.kernels import decoder_refresh as state_glue
        if prefix['recurrent'].dtype != torch.float32:
            raise ValueError('decoder GDN recurrent state must stay fp32')
        row, index, is_block = state['row'], state['index'], state['is_block']
        counts, tails = slot_counts.long(), tail_counts.long()
        if state.get('refresh_staging', False) or state.get('refresh_copy_staging', False):
            from jetspec.engines.jetspec_plus_plus.kernels.decoder_refresh import stage_hidden
            hidden = stage_hidden(tail, block, counts, lanes=lanes, slots=slots,
                                  out=state['hidden_input'])
        else:
            valid = state['index'][lanes:].view(rows, slots) < counts[:, None]
            blocks = torch.where(valid[..., None], block[:, :slots, -self.hidden_size:], 0)
            hidden = torch.cat((tail[:lanes], blocks.reshape(rows * slots, self.hidden_size)))
        source = gdn_layer.input_layernorm(hidden)
        raw, z, beta, g = gdn_project(gdn, source)

        # For every token gather its last kernel inputs. Negative logical tail
        # positions address committed conv; positive slot positions address only
        # that row. This includes the carry when tail_counts < kernel width.
        if state.get('refresh_staging', False) or state.get('refresh_copy_staging', False):
            from jetspec.engines.jetspec_plus_plus.kernels.decoder_refresh import stage_conv
            conv_input = stage_conv(raw, prefix['conv'], tails, lanes=lanes,
                                    slots=slots, out=state['conv_input'])
        else:
            previous = index[:, None] + state['conv_offset'][None]
            own_slot = is_block[:, None] & (previous >= 0)
            tail_position = previous + torch.where(is_block, tails[row], 0)[:, None]
            raw_index = torch.where(own_slot, lanes + row[:, None] * slots + previous,
                                    tail_position)
            committed = prefix['conv'][0].transpose(0, 1)
            all_raw = torch.cat((committed.to(raw.dtype), raw))
            gather = torch.where(~own_slot & (tail_position < 0),
                                 gdn.conv_kernel + tail_position,
                                 gdn.conv_kernel + raw_index)
            conv_input = all_raw[gather].transpose(1, 2).contiguous()
        if kernel_value('draft_fuse_g27g', '0') == '1' and hidden.is_cuda and hidden.dtype == torch.bfloat16:
            from jetspec.engines.jetspec_plus_plus.kernels.qwen3p8_27b.gdn_silu import gdn_inputs
            convolved = F.conv1d(conv_input, gdn.conv1d.weight, groups=conv_input.shape[1])[:, :, 0]
            q, k, v, _ = gdn_inputs(convolved, gdn.key_heads,
                gdn.value_heads // gdn.key_heads, gdn.key_dim, gdn.value_dim)
        else:
            convolved = F.silu(F.conv1d(conv_input, gdn.conv1d.weight,
                                       groups=conv_input.shape[1]))[:, :, 0]
            q, k, v = convolved.split((gdn.key_width, gdn.key_width, gdn.value_width), -1)
            q = q.reshape(tokens, gdn.key_heads, gdn.key_dim)
            k = k.reshape(tokens, gdn.key_heads, gdn.key_dim)
            v = v.reshape(tokens, gdn.value_heads, gdn.value_dim)
            repeat = gdn.value_heads // gdn.key_heads
            if repeat > 1:
                q, k = q.repeat_interleave(repeat, 1), k.repeat_interleave(repeat, 1)
        scratch = state['recurrent']
        if state_glue is None or not hidden.is_cuda:
            scratch[0].copy_(prefix['recurrent'])
        if hidden.is_cuda:
            from jetspec.engines.jetspec_plus_plus.models.hybrid_snapshot import snapshot_states
            if state_glue is not None:
                tail_output = state_glue.recurrent_output(gdn, q[:lanes][None], k[:lanes][None],
                    v[:lanes][None], g[:lanes][None], beta[:lanes][None], prefix['recurrent'])
            else:
                tail_output, _ = gdn._delta(q[:lanes][None], k[:lanes][None], v[:lanes][None],
                                          g[:lanes][None], beta[:lanes][None],
                                          prefix['recurrent'], chunk=False)
            if not state.get('refresh_selected_states', False):
                snapshot_states(k[:lanes][None], v[:lanes][None], g[:lanes][None],
                                beta[:lanes][None], prefix['recurrent'], scratch)
        else:
            outputs = []
            for step in range(lanes):
                output, final = gdn._delta(q[step:step+1][None], k[step:step+1][None],
                    v[step:step+1][None], g[step:step+1][None], beta[step:step+1][None],
                    scratch[step], chunk=False)
                scratch[step+1].copy_(final)
                outputs.append(output)
            tail_output = torch.cat(outputs, 1)
        if state.get('refresh_selected_states', False):
            from jetspec.engines.jetspec_plus_plus.kernels.decoder_refresh import selected_states
            initial = selected_states(k[:lanes][None], v[:lanes][None], g[:lanes][None],
                                      beta[:lanes][None], prefix['recurrent'], tails)
        else:
            initial = (state_glue.initial_states(scratch, prefix['recurrent'], tails)
                   if state_glue is not None else scratch.index_select(0, tails)[:, 0].contiguous())
        delta_args = (q[lanes:].reshape(rows, slots, gdn.value_heads, gdn.key_dim),
            k[lanes:].reshape(rows, slots, gdn.value_heads, gdn.key_dim),
            v[lanes:].reshape(rows, slots, gdn.value_heads, gdn.value_dim),
            g[lanes:].reshape(rows, slots, gdn.value_heads),
            beta[lanes:].reshape(rows, slots, gdn.value_heads), initial)
        if state_glue is not None:
            block_output = state_glue.recurrent_output(gdn, *delta_args)
        else:
            block_output, _ = gdn._delta(*delta_args, chunk=False)
        output = torch.cat((tail_output.reshape(lanes, -1), block_output.reshape(rows * slots, -1)))
        projected = gdn_finish(gdn, output, z, projection_dtype=hidden.dtype)
        x = tail_mlp(gdn_layer, hidden + projected, gate_glue)
        attn = attention_layer.self_attn
        fused_positions = (kernel_value('draft_fuse_g27g', '0') == '1' and hidden.is_cuda
                           and kernel_enabled('draft_fuse_g27', '1')
                           and getattr(attn, '_g27_rotary_table', None) is not None)
        positions = None if fused_positions else length + index + torch.where(is_block, tails[row], 0)
        position_args = {'refresh_positions': (length, tails, lanes, slots)} if fused_positions else {}
        q, gate, key, value = stateful_attention_project(
            attention_layer, x, positions, self.rope_theta, **position_args)
        groups = attn.heads // attn.kv_heads
        from jetspec.engines.jetspec_plus_plus.kernels import draft_splitkv
        fuse_inputs = kernel_enabled('draft_fuse_g27', '1') and draft_splitkv.a256_enabled(q, 'decoder')
        if fuse_inputs:
            from jetspec.engines.jetspec_plus_plus.draft.glue import g27_attention_inputs
            all_key, all_value, bias = g27_attention_inputs(
                prefix['key'], prefix['value'], key, value, length, tails, lanes=lanes, slots=slots)
        else:
            all_key, all_value = torch.cat((prefix['key'], key)), torch.cat((prefix['value'], value))
            allowed = (~is_block[None]) & (torch.where(is_block, tails[row], index + 1)[:, None]
                                           > index[None])
            allowed |= is_block[:, None] & is_block[None] & (row[:, None] == row[None]) \
                       & (index[:, None] >= index[None])
            allowed = torch.cat(((state['prefix_column'] < length).expand(tokens, -1), allowed), -1)
        if draft_splitkv.a256_enabled(q, 'decoder'):
            if not fuse_inputs:
                bias = torch.where(allowed[None, None], 0., torch.finfo(q.dtype).min).to(q.dtype)
            attended = draft_splitkv.attend(q.transpose(0, 1)[None],
                all_key.transpose(0, 1)[None], all_value.transpose(0, 1)[None],
                attn.dim ** -0.5, mode='dense', mask=bias)[0].transpose(0, 1)
        else:
            attended = F.scaled_dot_product_attention(q.transpose(0, 1)[None],
                all_key.repeat_interleave(groups, 1).transpose(0, 1)[None],
                all_value.repeat_interleave(groups, 1).transpose(0, 1)[None],
                attn_mask=allowed[None, None], dropout_p=0, scale=attn.dim ** -0.5)[0].transpose(0, 1)
        x = stateful_attention_finish(attention_layer, x, attended, gate, gate_glue)
        decoded = self.norm(x[lanes:])
        if _compact_ids is not None:
            from jetspec.engines.jetspec_plus_plus.draft.chain.refresh_swap import compact_head
            from jetspec.engines.jetspec_plus_plus.draft.chain.refresh_swap import compact_logits
            if 'compact' in state:
                return compact_head(decoded.contiguous(), self.lm_head.weight, _compact_ids, state['compact'],
                                    persistent=state.get('refresh_staging', False))
            return compact_logits(self.lm_head(decoded).reshape(rows, slots, -1), _compact_ids)
        return self.lm_head(decoded).reshape(rows, slots, -1)


    def _stateful_window_buckets(self):
        return (self.capacity + 20,)


    def _ensure_warmup_state(self, sequence):
        if hasattr(sequence, "hybrid_recurrent"):
            return
        # Warm-up sequences carry a negative id (-1 for N=1, others for the C>1 prewarm); a live
        # sequence, tagged or not, must arrive with the target's committed state.
        seq_id = getattr(sequence, "_jetspec_seq_id", None)
        if seq_id is None or seq_id >= 0:
            raise ValueError("live hybrid admission lacks target committed state")
        parameter = next(self.layers.parameters())
        layout = self.layout
        sequence.hybrid_recurrent = torch.zeros(
            layout.recurrent_shape, device=parameter.device, dtype=torch.float32)
        sequence.hybrid_conv = torch.zeros(
            layout.conv_shape, device=parameter.device, dtype=parameter.dtype)
        # The C>1 prewarm grows its prefixes in place, so it takes the decoder capacity.
        rows = sequence.taps.shape[0] if seq_id == -1 else self.capacity
        shape = (rows, layout.kv_heads, layout.attention_dim)
        sequence.hybrid_prefix_key = torch.zeros(shape, device=parameter.device,
                                                 dtype=parameter.dtype)
        sequence.hybrid_prefix_value = torch.zeros_like(sequence.hybrid_prefix_key)


    def _stateful_prefix(self, sequence, desired_length):
        states = self._sequence_state(sequence, desired_length)['layer_states']
        prefix = SimpleNamespace(length=desired_length,
            kv={index: (values['prefix_key'], values['prefix_value'])
                for index, values in states.items() if 'prefix_key' in values})
        self._prefixes[id(sequence)] = prefix
        return prefix


    def _stateful_prefetch_prefix(self, sequence, *, run_config=None):
        return None


    def _stateful_prefetch_batch(self, sequences, *, run_config=None):
        return None


    def _stateful_prewarm_prefix_appends(self, make_sequence, counts):
        return []


    @torch.inference_mode()
    def _stateful_predict(self, sequence, pending_block, *, dataset, coverage, grouped, cap,
                after_miss=False, run_config=None):
        staged = sequence.staged.predicted_hidden
        if staged is None:
            return None
        parameter = next(self.layers.parameters())
        from jetspec.engines.jetspec_plus_plus.wire.protocol import draft_width
        hidden = self._tap_hidden(staged[0, :draft_width(run_config=run_config if run_config is not None else getattr(self, 'run_config', None)) + 1]).to(parameter)
        start = sequence.taps.shape[0]
        began = time.perf_counter()
        _, logits = self(hidden, **self._sequence_state(sequence, start),
                         start=start, rope_theta=self.rope_theta)
        rejected = rejected_tokens(pending_block, after_miss=after_miss, run_config=self.run_config).to(logits.device)
        offers = anchor_offers(logits, rejected, root=self.root, dataset=dataset,
                               coverage=coverage, grouped=grouped, cap=cap, run_config=self.run_config)
        elapsed = (time.perf_counter() - began) * 1000
        return DecoderPrediction(hidden=self.norm(hidden), logits=logits,
                                 offers=offers, elapsed_ms=elapsed,
                                 total_ms=elapsed)


    def _stateful_predict_batch(self, sequences, pending_blocks, *, dataset, coverage, grouped,
                      cap, after_misses, run_config=None):
        if not (len(sequences) == len(pending_blocks) == len(after_misses)):
            raise ValueError('one pending block and miss flag are required per sequence')
        return [self._stateful_predict(sequence, pending, dataset=dataset,
                    coverage=coverage, grouped=grouped, cap=cap, after_miss=miss,
                    run_config=run_config) for sequence, pending, miss in
                zip(sequences, pending_blocks, after_misses)]

    def _stateful_decoder_chain_body(self, rp, tokens, front, bucket):
        from jetspec.engines.jetspec_plus_plus.wire.contract import Offset
        from jetspec.engines.jetspec_plus_plus.draft.heads.topk_select import GraphedSelect
        del bucket
        hidden = self._tap_hidden(front.staged_hidden[0, :front.lanes])
        logits, _ = hybrid_anchor_window(
            self.layers[0], self.layers[1], self.norm, self.lm_head,
            hidden, front.hybrid_recurrent, front.hybrid_conv,
            front.hybrid_prefix_key, front.hybrid_prefix_value,
            start=rp[Offset.L], rope_theta=self.rope_theta)
        front.rejected_ids.copy_(chain_rejected_tokens(rp, tokens, run_config=self.run_config))
        ids, p64, pack = GraphedSelect.chain_body(
            logits, front.rejected_ids, front.selection_thresholds, front.select_cap)
        return logits, ids, p64, pack


    @torch.inference_mode()
    def warm_kernels(self, *, run_config=None):
        """Compile every fused-kernel shape the eager decoder serves before any
        sequence is admitted: prefix appends of 1..16 positions and the current-slot
        window.  The kernels specialise on the row count, so a first sighting
        mid-round would otherwise stall that round on a Triton compile."""
        parameter = next(self.layers.parameters())
        fused = fused_kernels()
        for length in range(1, 17):
            hidden = parameter.new_zeros(1, length, self.hidden_size)
            positions = torch.arange(length, device=parameter.device)
            rope = self._rope(hidden, positions[None])
            mask = hidden.new_zeros((1, 1, length, length))
            prefix = StaticPrefix(32)
            pending = None
            for index, layer in enumerate(self.prefix_layers):
                if index == len(self.prefix_layers) - 1:
                    decoder_prefix_kv_lean(layer, hidden, rope, fused, pending=pending)
                else:
                    pending = decoder_layer_lean(layer, hidden, rope, mask, prefix, positions, fused,
                                                 pending=pending, run_config=self.run_config)
        hidden = parameter.new_zeros(1, self._window_rows(run_config=getattr(self, 'run_config', None)), self.hidden_size)
        rope = self._rope(hidden, torch.arange(self._window_rows(run_config=getattr(self, 'run_config', None)), device=parameter.device)[None])
        mask = hidden.new_zeros((1, 1, self._window_rows(run_config=getattr(self, 'run_config', None)), self._window_rows(run_config=getattr(self, 'run_config', None))))
        pending = None
        for layer in self.layers:
            pending = decoder_layer_lean(layer, hidden, rope, mask, None, None, fused, pending=pending, run_config=self.run_config)
        fused.fast_add_rms_norm(*pending, self.norm.weight, self.norm.variance_epsilon)

    @torch.inference_mode()
    def warm_graphs(self, *, run_config=None):
        """Capture every projection shape before any sequence is admitted.

        Prefix appends use 1..16 positions; the decoder window uses draft_width() + 1 rows.
        Attention and cache updates stay outside these shape-only graphs.
        """
        parameter = next(self.layers.parameters())
        if decoder_graph_lean():
            self._warm_lean_graphs(parameter, run_config=run_config if run_config is not None else getattr(self, 'run_config', None))
            self._graphs_warmed = True
            return
        for length in range(1, 17):
            hidden = parameter.new_zeros(1, length, self.hidden_size)
            positions = torch.arange(length, device=parameter.device)[None]
            rope = self.rotary(hidden, positions)
            for index, layer in enumerate(self.prefix_layers):
                if index == len(self.prefix_layers) - 1:
                    self._replay_window_part(
                        (id(layer), 'prefix_kv', tuple(hidden.shape), hidden.dtype),
                        lambda state, cos, sin: decoder_prefix_kv(layer, state, (cos, sin)),
                        (hidden, *rope))
                else:
                    hidden = self._window_layer(layer, hidden, None, rope, None)
        hidden = parameter.new_zeros(1, self._window_rows(run_config=run_config if run_config is not None else getattr(self, 'run_config', None)), self.hidden_size)
        rope = self.rotary(hidden, torch.arange(self._window_rows(run_config=run_config if run_config is not None else getattr(self, 'run_config', None)), device=parameter.device)[None])
        for layer in self.layers:
            hidden = self._window_layer(layer, hidden, None, rope, None)
        self._graphs_warmed = True

    def _warm_lean_graphs(self, parameter, *, run_config=None):
        """``warm_graphs`` for JETSPEC_DECODER_GRAPH_LEAN: every lean part shape (prefix
        appends of 1..16 positions, the current-slot window) captured before admission."""
        def layer_parts(layers, length, prefix_kv_last):
            hidden = parameter.new_zeros(1, length, self.hidden_size)
            rope = self.rotary(hidden, torch.arange(length, device=parameter.device)[None])
            pending = None
            for index, layer in enumerate(layers):
                if prefix_kv_last and index == len(layers) - 1:
                    self._lean_prefix_kv(layer, hidden, pending, rope)
                    return
                query, key, value, state = self._lean_project(layer, hidden, pending, rope)
                attended = query.new_zeros(query.shape[0], query.shape[2], query.shape[1],
                                           query.shape[3])
                pending = self._lean_finish(layer, state, attended)
        for length in range(1, 17):
            layer_parts(self.prefix_layers, length, True)
        layer_parts(self.layers, self._window_rows(run_config=getattr(self, 'run_config', None)), False)
        if decoder_graph_window(run_config=self.run_config):
            hidden = parameter.new_zeros(1, self._window_rows(run_config=getattr(self, 'run_config', None)), self.hidden_size)
            for bucket in self._window_buckets():
                self._window_graph(hidden, bucket)

    @staticmethod
    def _window_rows(run_config=None):
        """The current decoder slots [c, d1..d_width], following every true tap."""
        from jetspec.engines.jetspec_plus_plus.wire.protocol import draft_width
        return draft_width(run_config=run_config) + 1

    def _chain_lm_head(self, decoded):
        if getattr(self.lm_head.weight, '_draft_vocab', None) is not None:
            return self.lm_head(decoded)
        if self._lm_head_kn is not None:
            precision = getattr(self.lm_head, 'precision', None)
            if precision is not None and precision.weight_format.mode != 'bf16':
                from jetspec.engines.jetspec_plus_plus.models.target_linear import draft_mm
                logits = draft_mm(decoded.reshape(-1, decoded.shape[-1]), self._lm_head_kn,
                                  precision=precision)
                return logits.reshape(*decoded.shape[:-1], logits.shape[-1])
            return torch.matmul(decoded, self._lm_head_kn)
        return self.lm_head(decoded)

    def _check_prefix_merge(self, state, front):
        if self.prefix_layers is not self.layers or len(self.layers) != 2:
            raise ValueError('JETSPEC_DECODER_PREFIX_MERGE requires two shared target_slice layers')
        if front.decoder_prefix_kv is not state['kv']:
            raise ValueError('decoder prefix and window K/V must share storage')
        if any(not getattr(layer.self_attn, '_decoder_dense', type(layer.self_attn) is Qwen3Attention) or
               not hasattr(layer.self_attn, '_decoder_qkv') for layer in self.layers):
            raise ValueError('JETSPEC_DECODER_PREFIX_MERGE requires fused decoder QKV')

    def _merged_chain_layers(self, prefix_hidden, rope, mask, slots, state, front,
                             length, count, use_len_attention, fused):
        """Share the two layers' projection reads across prefix and window rows."""
        from jetspec.engines.jetspec_plus_plus.profiles import kernel_enabled
        self._check_prefix_merge(state, front)
        lanes = prefix_hidden.shape[1]
        if kernel_enabled('draft_fuse_t12', '1') and 'combined_hidden' in state:
            combined_rope = state['combined_cos'], state['combined_sin']
            source = state['combined_hidden']
        else:
            combined_rope = tuple(torch.cat((p, w), dim=1)
                                  for p, w in zip(rope, (state['cos'], state['sin'])))
            source = torch.cat((prefix_hidden, state['hidden']), dim=1)
        pending = None
        merged = length.is_cuda and kernel_enabled('draft_fuse_t3', '1')
        for index, layer in enumerate(self.layers):
            stored_key, stored_value = state['kv'][layer.self_attn.layer_idx]
            direct = kernel_enabled('decoder_slot_write', '1')
            if direct:
                from jetspec.engines.jetspec_plus_plus.kernels.decoder_cache import project_into_cache
                query, hidden = project_into_cache(
                    layer, fused, source if pending is None else None, pending, combined_rope,
                    (stored_key, stored_value), slots, state['slots'])
            else:
                query, key, value, hidden = decoder_lean_project(
                    layer, fused, source if pending is None else None, pending, combined_rope)
                stored_key.index_copy_(2, slots, key[..., :lanes, :])
                stored_value.index_copy_(2, slots, value[..., :lanes, :])
            if merged:
                if not direct:
                    stored_key.index_copy_(2, state['slots'], key[..., lanes:, :])
                    stored_value.index_copy_(2, state['slots'], value[..., lanes:, :])
                attended = decoder_merged_attention(
                    layer, query if index == 0 else query[..., lanes:, :],
                    stored_key[..., :front.decoder_bucket, :],
                    stored_value[..., :front.decoder_bucket, :], length, count,
                    lanes if index == 0 else 0)
                pending = decoder_lean_finish(
                    layer, fused, hidden if index == 0 else hidden[:, lanes:], attended)
                continue
            if index == 0:
                prefix_attended = decoder_lean_attention(
                    layer, fused, query[..., :lanes, :],
                    stored_key[..., :front.decoder_bucket, :],
                    stored_value[..., :front.decoder_bucket, :], mask,
                    length=length if use_len_attention else None,
                    count=count if use_len_attention else None)
            if not direct:
                stored_key.index_copy_(2, state['slots'], key[..., lanes:, :])
                stored_value.index_copy_(2, state['slots'], value[..., lanes:, :])
            window_attended = decoder_lean_attention(
                layer, fused, query[..., lanes:, :], stored_key[..., :front.decoder_bucket, :],
                stored_value[..., :front.decoder_bucket, :],
                state['masks'][front.decoder_bucket])
            if index == 0:
                attended = torch.cat((prefix_attended, window_attended), dim=1)
                pending = decoder_lean_finish(layer, fused, hidden, attended)
            else:
                pending = decoder_lean_finish(layer, fused, hidden[:, lanes:], window_attended)
        decoded, _ = fused.fast_add_rms_norm(*pending, self.norm.weight,
                                             self.norm.variance_epsilon)
        decoded = decoded.reshape(-1, state['hidden'].shape[1], self.hidden_size)[0]
        return self._chain_lm_head(decoded)

    def _window_buckets(self):
        if getattr(self, '_uses_shipped_state', False):
            return self._stateful_window_buckets()
        capacity = int(setting(self.run_config, 'engine', 'engine_settings.decoder_cache_capacity', '800'))
        if (capacity > 800 and
                setting(self.run_config, 'engine', 'engine_settings.chain_decoder_key_bucket', '0') != '1'):
            return [capacity]  # one graph for the selected run capacity
        return [*range(128, capacity, 128), capacity]

    def _window_state(self, like, bucket):
        """Keep each opted-in decoder key's static inputs and K/V distinct."""
        from jetspec.engines.jetspec_plus_plus.profiles import kernel_enabled
        bucketed = setting(self.run_config, 'engine', 'engine_settings.chain_decoder_key_bucket', '0') == '1'
        if bucketed:
            states = self._window_graph_states
            state = states.get(bucket)
        else:
            state = getattr(self, '_window_graph_state', None)
        if state is not None:
            return state
        capacity = bucket if bucketed else self._window_buckets()[-1]
        kv = {}
        for layer in self.layers:
            attn = layer.self_attn
            shape = (1, attn.k_proj.out_features // attn.head_dim, capacity, attn.head_dim)
            kv[attn.layer_idx] = (like.new_zeros(shape), like.new_zeros(shape))
        head_dim = self.layers[0].self_attn.head_dim
        lanes = torch.arange(like.shape[1], device=like.device)
        state = dict(hidden=like.new_zeros(1, like.shape[1], self.hidden_size), kv=kv,
                     cos=like.new_zeros(1, like.shape[1], head_dim),
                     sin=like.new_zeros(1, like.shape[1], head_dim),
                     lanes=lanes, slots=lanes.clone(), masks={}, graphs={}, steps={})
        if kernel_enabled('decoder_glue_fused', '1'):
            prefix_lanes = like.shape[1]
            state.update(prefix_positions=torch.empty((1, prefix_lanes), device=like.device,
                                                      dtype=torch.long),
                         prefix_slots=torch.empty(prefix_lanes, device=like.device,
                                                  dtype=torch.long),
                         prefix_masks={},
                         prefix_cos=like.new_empty(1, prefix_lanes, head_dim),
                         prefix_sin=like.new_empty(1, prefix_lanes, head_dim))
            if kernel_enabled('draft_fuse_t12', '1') and kernel_enabled('decoder_prefix_merge', '1'):
                state['combined_hidden'] = like.new_zeros(1, prefix_lanes + like.shape[1],
                                                         self.hidden_size)
                for name in ('cos', 'sin'):
                    combined = state['combined_' + name] = like.new_zeros(
                        1, prefix_lanes + like.shape[1], head_dim)
                    state['prefix_' + name], state[name] = combined[:, :prefix_lanes], combined[:, prefix_lanes:]
                state['hidden'] = state['combined_hidden'][:, prefix_lanes:]
        if bucketed:
            states[bucket] = state
        else:
            self._window_graph_state = state
        return state

    def _window_graph(self, like, bucket):
        """The predict window's graph for ``bucket`` keys (captured on first use).

        Its static inputs: the current-slot window, its rotary rows, the window slots
        ``context .. context+rows`` and the bucket's mask (the eager window mask, then
        the dtype minimum); the trained layers attend over decoder-owned K/V
        (the sequence's prefix rows copied in, finite everywhere past them)."""
        state = self._window_state(like, bucket)
        if bucket not in state['graphs']:
            if self._graphs_warmed and setting(self.run_config, 'engine', 'engine_settings.front_static', '0') != '1':
                raise RuntimeError(f'decoder window graph missing after warm-up: {bucket}')
            rows = state['hidden'].shape[1]
            mask = state['masks'][bucket] = like.new_zeros(1, 1, rows, bucket)
            if 'prefix_masks' in state:
                state['prefix_masks'][bucket] = like.new_empty(1, 1, rows, bucket)
            fused = fused_kernels()

            def step():
                pending = None
                for layer in self.layers:
                    query, key, value, hidden = decoder_lean_project(
                        layer, fused, state['hidden'] if pending is None else None, pending,
                        (state['cos'], state['sin']))
                    stored_key, stored_value = state['kv'][layer.self_attn.layer_idx]
                    stored_key.index_copy_(2, state['slots'], key)
                    stored_value.index_copy_(2, state['slots'], value)
                    attended = decoder_lean_attention(layer, fused, query,
                                                      stored_key[:, :, :bucket],
                                                      stored_value[:, :, :bucket], mask)
                    pending = decoder_lean_finish(layer, fused, hidden, attended)
                decoded, _ = fused.fast_add_rms_norm(*pending, self.norm.weight,
                                                     self.norm.variance_epsilon)
                decoded = decoded.reshape(-1, rows, self.hidden_size)[0]
                return decoded, self._chain_lm_head(decoded)

            current = torch.cuda.current_stream(like.device)
            stream = torch.cuda.Stream(device=like.device)
            stream.wait_stream(current)
            with torch.cuda.stream(stream):
                for _ in range(3):
                    step()
            current.wait_stream(stream)
            expected = tuple(value.clone() for value in step())
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                result = step()
            graph.replay()
            if not all(torch.equal(a, b) for a, b in zip(expected, result)):
                raise RuntimeError(f'decoder window graph replay changed bucket {bucket}')
            state['graphs'][bucket] = (graph, result)
            state['steps'][bucket] = step
        return state

    def _window_graph_predict(self, hidden, rope, mask, prefix, context_length):
        """``decoder_layer_lean`` over both trained layers, the final add+norm and the
        lm_head, replayed at the bucket of ``context_length + window rows`` keys."""
        end = context_length + hidden.shape[1]
        bucket = next(b for b in self._window_buckets() if b >= end)
        state = self._window_graph(hidden, bucket)
        if setting(self.run_config, 'engine', 'engine_settings.keylen_bucket', '0') == '1':
            self._stage_window_prefix(state, prefix, context_length)
        else:
            for layer in self.layers:
                index = layer.self_attn.layer_idx
                if context_length and index in prefix.kv:
                    for staged, stored in zip(state['kv'][index], prefix.kv[index]):
                        staged[..., :context_length, :].copy_(stored[..., :context_length, :])
        state['hidden'].copy_(hidden)
        state['cos'].copy_(rope[0])
        state['sin'].copy_(rope[1])
        torch.add(state['lanes'], context_length, out=state['slots'])
        buffer = state['masks'][bucket]
        buffer.fill_(torch.finfo(buffer.dtype).min)
        buffer[..., :end].copy_(mask[..., :end])
        graph, result = state['graphs'][bucket]
        graph.replay()
        return result[0].clone(), result[1].clone()

    def decoder_chain_body(self, rp, tokens: TokenBuffers, front: FrontInputs, bucket):
        """Append the fixed prefix lanes, decode the window, and select on device."""
        if getattr(self, '_uses_shipped_state', False):
            return self._stateful_decoder_chain_body(rp, tokens, front, bucket)
        from jetspec.engines.jetspec_plus_plus.profiles import kernel_enabled, kernel_value
        if setting(self.run_config, 'engine', 'engine_settings.front_static', '0') != '1':
            raise ValueError('decoder chain body requires JETSPEC_FRONT_STATIC=1')
        use_len_attention = (rp.is_cuda and
                             (((bucket.head.P <= 2304 and
                               kernel_enabled('decoder_attn_len', '1')))))
        del bucket
        lanes = front.lanes
        length, count = rp[Offset.DEC_LEN], rp[Offset.C1_COUNT]
        prefix_hidden = front.decoder_prefix_taps[None]
        state = (self._window_graph_states[front.decoder_bucket]
                 if setting(self.run_config, 'engine', 'engine_settings.chain_decoder_key_bucket', '0') == '1'
                 else self._window_graph_state)
        fused_glue = kernel_enabled('decoder_glue_fused', '1')
        combined_glue = (fused_glue and kernel_enabled('draft_fuse_t12', '1')
                         and kernel_enabled('decoder_prefix_merge', '1'))
        if fused_glue:
            from jetspec.engines.jetspec_plus_plus.kernels.decoder_glue import prefix_glue
            table = self._rope_table
            if table is None or not all(value.is_contiguous() for value in table):
                raise RuntimeError('decoder glue requires a prewarmed contiguous rotary table')
            if combined_glue:
                from jetspec.engines.jetspec_plus_plus.kernels.decoder_staging import combined_glue as stage_combined
                stage_combined(
                    rp, tokens.pending_draft_ids, prefix_hidden, front.staged_hidden, *table,
                    state['combined_hidden'], state['combined_cos'], state['combined_sin'],
                    state['prefix_masks'][front.decoder_bucket], state['masks'][front.decoder_bucket],
                    state['prefix_positions'], state['prefix_slots'], state['slots'], front.rejected_ids,
                    length_offset=Offset.DEC_LEN, count_offset=Offset.C1_COUNT,
                    correction_offset=Offset.CORRECTION, miss_offset=Offset.AFTER_MISS,
                    wire=setting(self.run_config, 'engine', 'engine_settings.decoder_frame', None) == 'wire',
                    zero_miss=setting(self.run_config, 'method', 'method_settings.kset_after_miss', None) == 'zero')
            else:
                prefix_glue(rp, tokens.pending_draft_ids, *table,
                        state['prefix_masks'][front.decoder_bucket], state['prefix_positions'],
                        state['prefix_slots'], state['prefix_cos'], state['prefix_sin'],
                        front.rejected_ids, length_offset=Offset.DEC_LEN,
                        count_offset=Offset.C1_COUNT,
                        correction_offset=Offset.CORRECTION, miss_offset=Offset.AFTER_MISS,
                        wire=setting(self.run_config, 'engine', 'engine_settings.decoder_frame', None) == 'wire',
                        zero_miss=setting(self.run_config, 'method', 'method_settings.kset_after_miss', None) == 'zero')
            positions = state['prefix_positions']
            slots = state['prefix_slots']
            rope = state['prefix_cos'], state['prefix_sin']
            mask = state['prefix_masks'][front.decoder_bucket]
        else:
            front.rejected_ids.copy_(chain_rejected_tokens(rp, tokens, run_config=self.run_config))
            lane = torch.arange(lanes, device=rp.device, dtype=torch.long)
            positions = (length - count + lane)[None]
            slots = chain_decoder_prefix_slots(length, count, front.decoder_bucket, lanes)
            rope = self._rope(prefix_hidden, positions)
            mask = (prefix_hidden.new_empty(1, 1, lanes, front.decoder_bucket)
                    if use_len_attention
                    else chain_decoder_prefix_mask(
                        length, count, front.decoder_bucket, lanes, prefix_hidden.dtype))
        if kernel_enabled('decoder_prefix_merge', '1'):
            glue_level = int(kernel_value('chain_sega_glue', '0'))
            from jetspec.engines.jetspec_plus_plus.draft.glue import available
            from jetspec.engines.jetspec_plus_plus.draft.glue import window_mask_rope
            fused_window = (glue_level == 2 and rp.is_cuda and available()
                            and self._rope_table is not None
                            and all(table.is_contiguous() for table in self._rope_table))
            if combined_glue:
                pass  # Both sets of inputs already share the combined staging launch.
            elif fused_glue:
                from jetspec.engines.jetspec_plus_plus.kernels.decoder_glue import window_glue
                window_glue(length, front.staged_hidden,
                            *self._rope_table, state['hidden'],
                            state['masks'][front.decoder_bucket], state['slots'],
                            state['cos'], state['sin'])
            else:
                window_positions = stage_chain_window_state(
                    state, front, length, glue_level=glue_level, fused_mask=fused_window)
            if fused_window and not fused_glue:
                window_mask_rope(length, *self._rope_table,
                                 state['masks'][front.decoder_bucket],
                                 state['cos'], state['sin'])
            elif not fused_glue:
                cos, sin = self._rope(state['hidden'], window_positions)
                state['cos'].copy_(cos)
                state['sin'].copy_(sin)
            logits = self._merged_chain_layers(prefix_hidden, rope, mask, slots, state,
                                                front, length, count, use_len_attention,
                                                fused_kernels())
            from jetspec.engines.jetspec_plus_plus.draft.heads.topk_select import GraphedSelect
            ids, p64, pack = GraphedSelect.chain_body(
                logits, front.rejected_ids, front.selection_thresholds, front.select_cap)
            return logits, ids, p64, pack
        fused = fused_kernels()
        pending = None
        for index, layer in enumerate(self.prefix_layers):
            if index == len(self.prefix_layers) - 1:
                key, value = decoder_prefix_kv_lean(
                    layer, prefix_hidden if pending is None else None, rope, fused,
                    pending=pending)
            else:
                query, key, value, hidden = decoder_lean_project(
                    layer, fused, prefix_hidden if pending is None else None, pending, rope)
            stored_key, stored_value = front.decoder_prefix_kv[layer.self_attn.layer_idx]
            stored_key.index_copy_(2, slots, key)
            stored_value.index_copy_(2, slots, value)
            if index != len(self.prefix_layers) - 1:
                attended = decoder_lean_attention(
                    layer, fused, query, stored_key[..., :front.decoder_bucket, :],
                    stored_value[..., :front.decoder_bucket, :], mask,
                    length=length if use_len_attention else None,
                    count=count if use_len_attention else None)
                pending = decoder_lean_finish(layer, fused, hidden, attended)

        glue_level = int(kernel_value('chain_sega_glue', '0'))
        from jetspec.engines.jetspec_plus_plus.draft.glue import available
        from jetspec.engines.jetspec_plus_plus.draft.glue import window_mask_rope
        fused_window = (glue_level == 2 and rp.is_cuda and available()
                        and self._rope_table is not None
                        and all(table.is_contiguous() for table in self._rope_table))
        if fused_glue:
            from jetspec.engines.jetspec_plus_plus.kernels.decoder_glue import window_glue
            window_glue(length, front.staged_hidden,
                        *self._rope_table, state['hidden'],
                        state['masks'][front.decoder_bucket], state['slots'],
                        state['cos'], state['sin'])
        else:
            window_positions = stage_chain_window_state(
                state, front, length, glue_level=glue_level, fused_mask=fused_window)
        if fused_window and not fused_glue:
            window_mask_rope(length, *self._rope_table,
                             state['masks'][front.decoder_bucket],
                             state['cos'], state['sin'])
        elif not fused_glue:
            cos, sin = self._rope(state['hidden'], window_positions)
            state['cos'].copy_(cos)
            state['sin'].copy_(sin)
        _, logits = state['steps'][front.decoder_bucket]()
        from jetspec.engines.jetspec_plus_plus.draft.heads.topk_select import GraphedSelect
        ids, p64, pack = GraphedSelect.chain_body(
            logits, front.rejected_ids, front.selection_thresholds, front.select_cap)
        return logits, ids, p64, pack

    def refresh_window_state(self, prefix_kv, bucket, *, rows, lanes, like, glue=None):
        """Static buffers of ``refresh_window_logits`` for one chain key (serving JETSPEC_REFRESH_SWAP).

        ``prefix_kv`` is the key's decoder prefix K/V (``FrontInputs.decoder_prefix_kv``) and
        ``bucket`` its key bucket.  The read's private K/V holds the bucket's columns, then the
        read's own tokens in a fixed order: the 4 true prefix positions, the ``lanes`` staged
        tail slots and ``rows`` x (lanes - 1) block slots, padded to a multiple of 16 columns.
        Allocated before capture; the bucket's prefix K/V is copied in, never written."""
        if getattr(self, '_uses_shipped_state', False):
            return self._stateful_refresh_window_state(prefix_kv, bucket, rows=rows, lanes=lanes, like=like, glue=glue)
        from jetspec.engines.jetspec_plus_plus.profiles import kernel_enabled
        if self.prefix_layers is not self.layers:
            raise ValueError('JETSPEC_REFRESH_SWAP requires the target_slice decoder '
                             '(one set of shared prefix and window layers)')
        from jetspec.engines.jetspec_plus_plus.kernels import refresh_glue
        if refresh_glue.enabled(run_config=self.run_config) if glue is None else glue:
            return refresh_glue.allocate(self, prefix_kv, bucket, rows, lanes, like)
        slots = lanes - 1
        tokens = 4 + lanes + rows * slots
        width = bucket + (tokens + 15) // 16 * 16
        kv = {}
        for layer in self.layers:
            index = layer.self_attn.layer_idx
            key, value = prefix_kv[index]
            if key.shape[-2] < bucket:
                raise ValueError('decoder prefix K/V is shorter than its key bucket')
            shape = (*key.shape[:2], width, key.shape[-1])
            kv[index] = (key.new_zeros(shape), value.new_zeros(shape))
        device = like.device
        lane = torch.arange(tokens, device=device)
        block = lane - 4 - lanes
        # kind 0: true prefix position, 1: tail slot, 2: block slot of row ``row``
        kind = torch.where(lane < 4, 0, torch.where(lane < 4 + lanes, 1, 2))
        index = torch.where(kind == 0, lane, torch.where(kind == 1, lane - 4,
                                                         block.clamp_min(0) % slots))
        row = torch.where(kind == 2, block.clamp_min(0) // slots, -1)
        column = torch.arange(width, device=device)
        local = column - bucket
        inside = (local >= 0) & (local < tokens)
        key_lane = local.clamp(0, tokens - 1)
        state = dict(prefix_kv=prefix_kv, kv=kv, bucket=bucket, width=width, tokens=tokens,
                    rows=rows, lanes=lanes, slots=slots, kind=kind, index=index, row=row,
                    column=column, key_kind=torch.where(inside, kind[key_lane], -1),
                    key_index=index[key_lane], key_row=row[key_lane])
        if kernel_enabled('refresh_swap_lean', '1') and like.is_cuda and like.dtype == torch.bfloat16:
            from jetspec.engines.jetspec_plus_plus.draft.chain.refresh_swap import allocate_compact
            state['compact'] = allocate_compact(rows, slots, self.lm_head.weight.shape[0], device)
        return state

    def refresh_window_compact(self, state, *, ids, **kwargs):
        """The opt-in refresh read returns compact swap inputs, never BF16 vocab logits."""
        if getattr(self, '_uses_shipped_state', False):
            return self._stateful_refresh_window_compact(state, ids=ids, **kwargs)
        return self.refresh_window_logits(state, _compact_ids=ids, **kwargs)

    def refresh_window_logits(self, state, *, length, true_taps, tail, tail_counts, block,
                              slot_counts, _compact_ids=None):
        """The anchor decoder over each row's refreshed block, in the refresh context, with fixed
        shapes and no host read.

        * prefix: the decoder K/V of the TRUE tap-33 states of committed positions 0..length-1:
          the key's cached prefix (positions below length - 4) and the 4
          positions length-4..length-1 run through the layers from ``true_taps`` [4, H];
        * tail: ``tail`` [lanes, H] (the staged block's tap-33 rows) at positions length..,
          causal, shared by every row;
        * row r: ``block[r, s, -H:]`` for s < slot_counts[r] (zero past them) at position
          length + tail_counts[r] + s, reading the prefix, tail slots < tail_counts[r] and its
          own slots <= s (the diagonal included); no row reads another.

        Returns logits [rows, lanes - 1, V]: decoder row j of a block is the distribution for
        block slot j + 1 (rows past slot_counts[r] are undefined).
        """
        if getattr(self, '_uses_shipped_state', False):
            return self._stateful_refresh_window_logits(state, length=length, true_taps=true_taps, tail=tail, tail_counts=tail_counts, block=block, slot_counts=slot_counts, _compact_ids=_compact_ids)
        bucket, width, tokens = state['bucket'], state['width'], state['tokens']
        rows, lanes, slots = state['rows'], state['lanes'], state['slots']
        hidden_size = self.hidden_size
        if state.get('glue'):
            from jetspec.engines.jetspec_plus_plus.kernels.refresh_glue import layers as refresh_glue_layers
            hidden, decoded = refresh_glue_layers(
                self, state, length=length, true_taps=true_taps, tail=tail,
                tail_counts=tail_counts, block=block, slot_counts=slot_counts, run_config=self.run_config)
            compact_cuda = _compact_ids is not None and 'compact' in state
        else:
            for index, (key, value) in state['kv'].items():
                prefix_key, prefix_value = state['prefix_kv'][index]
                key[..., :bucket, :].copy_(prefix_key[..., :bucket, :])
                value[..., :bucket, :].copy_(prefix_value[..., :bucket, :])
            counts = slot_counts.to(torch.long)
            tails = tail_counts.to(torch.long)
            valid = torch.arange(slots, device=counts.device)[None, :] < counts[:, None]
            block_rows = block[:, :slots, -hidden_size:]
            block_rows = torch.where(valid[..., None], block_rows, torch.zeros_like(block_rows))
            hidden = torch.cat((true_taps.to(block_rows.dtype), tail[:lanes].to(block_rows.dtype),
                                block_rows.reshape(rows * slots, hidden_size)), dim=0)[None]
            kind, index, row = state['kind'], state['index'], state['row']
            row_tail = tails[row.clamp_min(0)]
            positions = torch.where(kind == 0, length - 4 + index,
                                    torch.where(kind == 1, length + index, length + row_tail + index))
            positions = positions.clamp_min(0)
            key_kind, key_index, key_row = state['key_kind'], state['key_index'], state['key_row']
            query_kind, query_index = kind[:, None], index[:, None]
            column = state['column'][None, :]
            allowed = (column < bucket) & (column < length - 4)
            allowed = allowed | (key_kind == 0)[None, :] & (
                (query_kind != 0) | (key_index[None, :] <= query_index))
            allowed = allowed | (key_kind == 1)[None, :] & (
                ((query_kind == 1) & (key_index[None, :] <= query_index))
                | ((query_kind == 2) & (key_index[None, :] < row_tail[:, None])))
            allowed = allowed | (key_kind == 2)[None, :] & (query_kind == 2) & (
                key_row[None, :] == row[:, None]) & (key_index[None, :] <= query_index)
            mask = hidden.new_zeros(1, 1, tokens, width).masked_fill_(
                ~allowed[None, None], torch.finfo(hidden.dtype).min)
            rope = self._rope(hidden, positions[None])
            if hidden.is_cuda:
                fused = fused_kernels()
                pending = None
                compact_cuda = _compact_ids is not None and 'compact' in state
                decoded = None
                if decoded is None:
                    for layer in self.layers:
                        query, key, value, residual = decoder_lean_project(
                            layer, fused, hidden if pending is None else None, pending, rope)
                        stored_key, stored_value = state['kv'][layer.self_attn.layer_idx]
                        stored_key[..., bucket:bucket + tokens, :].copy_(key)
                        stored_value[..., bucket:bucket + tokens, :].copy_(value)
                        attended = decoder_lean_attention(layer, fused, query, stored_key, stored_value,
                                                          mask)
                        pending = decoder_lean_finish(layer, fused, residual, attended)
                    decoded, _ = fused.fast_add_rms_norm(*pending, self.norm.weight,
                                                         self.norm.variance_epsilon)
            else:
                # The reference arithmetic (CPU): the vendored layer's projections, sdpa and MLP.
                from transformers.integrations.sdpa_attention import repeat_kv
                current = hidden
                for layer in self.layers:
                    query, key, value = decoder_window_project(layer, current, rope)
                    stored_key, stored_value = state['kv'][layer.self_attn.layer_idx]
                    stored_key[..., bucket:bucket + tokens, :].copy_(key)
                    stored_value[..., bucket:bucket + tokens, :].copy_(value)
                    groups = layer.self_attn.num_key_value_groups
                    attended = torch.nn.functional.scaled_dot_product_attention(
                        query, repeat_kv(stored_key, groups), repeat_kv(stored_value, groups),
                        attn_mask=mask, scale=layer.self_attn.scaling).transpose(1, 2)
                    current = decoder_window_finish(layer, current, attended)
                decoded = self.norm(current)
        from jetspec.engines.jetspec_plus_plus.profiles import kernel_enabled
        trimmed = state.get('glue') and kernel_enabled('draft_fuse_t12', '1')
        decoded = decoded[0] if trimmed else decoded[0, 4 + lanes:]
        if hidden.is_cuda and compact_cuda:
            from jetspec.engines.jetspec_plus_plus.draft.chain.refresh_swap import compact_head
            # This kernel consumes native [N,K] tiles. Seg A's [K,N] cuBLAS layout
            # is already prepared, but makes this tiled load strided on SM80.
            return compact_head(decoded.contiguous(), self.lm_head.weight,
                                _compact_ids, state['compact'])
        if _compact_ids is not None:
            # FP32 validation and CPU retain the reference head/softmax arithmetic.
            from jetspec.engines.jetspec_plus_plus.draft.chain.refresh_swap import compact_logits
            return compact_logits(self._chain_lm_head(decoded).reshape(rows, slots, -1), _compact_ids)
        return self._chain_lm_head(decoded).reshape(rows, slots, -1)

    @staticmethod
    def _stage_window_prefix(state, prefix, context_length):
        """Append new sequence K/V rows to the graph's persistent prefix buffers."""
        start = (state.get('prefix_length', 0)
                 if state.get('prefix_owner') is prefix else 0)
        if start > context_length:
            start = 0
        if start < context_length:
            for index, pair in prefix.kv.items():
                if index in state['kv']:
                    for staged, stored in zip(state['kv'][index], pair):
                        staged[..., start:context_length, :].copy_(
                            stored[..., start:context_length, :])
        state['prefix_owner'] = prefix
        state['prefix_length'] = context_length

    def _lean_project(self, layer, hidden, pending, rope):
        fused = fused_kernels()
        if pending is None:
            return self._replay_window_part(
                (id(layer), 'lean_project', tuple(hidden.shape), hidden.dtype),
                lambda state, cos, sin: decoder_lean_project(layer, fused, state, None, (cos, sin)),
                (hidden, *rope))
        return self._replay_window_part(
            (id(layer), 'lean_project_pending', tuple(pending[0].shape), pending[0].dtype),
            lambda update, residual, cos, sin: decoder_lean_project(
                layer, fused, None, (update, residual), (cos, sin)),
            (*pending, *rope))

    def _lean_finish(self, layer, hidden, attended):
        fused = fused_kernels()
        return self._replay_window_part(
            (id(layer), 'lean_finish', tuple(hidden.shape), hidden.dtype),
            lambda state, output: decoder_lean_finish(layer, fused, state, output),
            (hidden, attended))

    def _lean_prefix_kv(self, layer, hidden, pending, rope):
        fused = fused_kernels()
        if pending is None:
            return self._replay_window_part(
                (id(layer), 'lean_prefix_kv', tuple(hidden.shape), hidden.dtype),
                lambda state, cos, sin: decoder_prefix_kv_lean(layer, state, (cos, sin), fused),
                (hidden, *rope))
        return self._replay_window_part(
            (id(layer), 'lean_prefix_kv_pending', tuple(pending[0].shape), pending[0].dtype),
            lambda update, residual, cos, sin: decoder_prefix_kv_lean(
                layer, None, (cos, sin), fused, pending=(update, residual)),
            (*pending, *rope))

    def _window_layer_lean(self, layer, hidden, pending, mask, rope, frozen, cache_position=None):
        """``decoder_layer_lean`` as graph parts around the eager attention; returns
        the (mlp_output, residual) pair for the next part to sum."""
        query, key, value, state = self._lean_project(layer, hidden, pending, rope)
        attn = layer.self_attn
        if frozen is not None:
            key, value = frozen.update(key, value, attn.layer_idx,
                                       {'sin': rope[1], 'cos': rope[0],
                                        'cache_position': cache_position})
        attended = decoder_lean_attention(layer, fused_kernels(), query, key, value, mask)
        return self._lean_finish(layer, state, attended)







    @torch.inference_mode()
    def prefetch_prefix(self, sequence, *, run_config=None):
        """``predict``'s prefix append, issued early: the same guards (a
        staged block, a non-negative context) and the same ``_prefix`` call on the same
        taps, so the append chunks and the bits are those ``predict`` would produce;
        ``predict`` then finds the prefix at its length."""
        if getattr(self, '_uses_shipped_state', False):
            return self._stateful_prefetch_prefix(sequence, run_config=run_config)
        staged = getattr(getattr(sequence, 'staged', None), 'predicted_hidden', None)
        from jetspec.engines.jetspec_plus_plus.wire.protocol import draft_width
        if staged is None or int(staged.shape[1]) < draft_width(run_config=run_config if run_config is not None else getattr(self, 'run_config', None)) + 1:
            return
        context_length = int(sequence.taps.shape[0])
        if context_length >= 0:
            self._prefix(sequence, context_length)

    @torch.inference_mode()
    def prewarm_prefix_appends(self, make_sequence, counts):
        """Run ``_prefix`` once per append size in ``counts`` on throwaway sequences from
        ``make_sequence(length)``, so the Triton entries each size compiles (its row count
        is a constexpr: ~0.6 s per new size) are built before any request is admitted."""
        if getattr(self, '_uses_shipped_state', False):
            return self._stateful_prewarm_prefix_appends(make_sequence, counts)
        capacity = int(setting(self.run_config, 'engine', 'engine_settings.decoder_cache_capacity', '800'))
        counts = sorted({int(count) for count in counts if 1 <= int(count) < capacity})
        sequence, length = None, 0
        for count in counts:
            if sequence is None or length + count > capacity:
                if sequence is not None:
                    self.drop_sequence(sequence)
                sequence = make_sequence(capacity)
                length = 1
                self._prefix(sequence, length)
            length += count
            self._prefix(sequence, length)
        if sequence is not None:
            self.drop_sequence(sequence)
        return counts

    @torch.inference_mode()
    def prefetch_batch(self, sequences, *, run_config=None):
        """Opt-in packed append over the sum of independent request deltas."""
        if getattr(self, '_uses_shipped_state', False):
            return self._stateful_prefetch_batch(sequences, run_config=run_config)
        from jetspec.engines.jetspec_plus_plus.wire.protocol import draft_width
        ready = []
        for sequence in sequences:
            staged = getattr(getattr(sequence, 'staged', None), 'predicted_hidden', None)
            desired = int(sequence.taps.shape[0])
            if (staged is None
                    or int(staged.shape[1]) < draft_width(run_config=run_config if run_config is not None else getattr(self, 'run_config', None)) + 1 or desired < 0):
                continue
            prefix = self._prefixes.get(id(sequence))
            if prefix is None:
                prefix = self._new_prefix(sequence)
            if not isinstance(prefix, StaticPrefix) or prefix.length > desired:
                self._prefix(sequence, desired)
                continue
            if prefix.length < desired:
                ready.append((sequence, prefix, prefix.length, desired))
        if len(ready) < 2:
            for sequence, _, _, desired in ready:
                self._prefix(sequence, desired)
            return
        self._pool_prefixes([prefix for _, prefix, _, _ in ready])
        parameter = next(self.layers.parameters())
        chunks = [sequence.taps[start:stop, self.tap_position, :].to(parameter)
                  for sequence, _, start, stop in ready]
        hidden = torch.cat(chunks, dim=0)[None]
        positions = torch.cat([torch.arange(start, stop, device=hidden.device)
                               for _, _, start, stop in ready])
        rope = self._rope(hidden, positions[None])
        counts = [stop - start for _, _, start, stop in ready]
        prefix_counts = [start for _, _, start, _ in ready]
        query_owner = torch.repeat_interleave(torch.arange(len(ready), device=hidden.device),
                                               torch.tensor(counts, device=hidden.device))
        key_owner = torch.cat((torch.repeat_interleave(
            torch.arange(len(ready), device=hidden.device),
            torch.tensor(prefix_counts, device=hidden.device)), query_owner))
        key_positions = torch.cat((torch.cat([
            torch.arange(start, device=hidden.device) for start in prefix_counts]), positions))
        allowed = ((query_owner[:, None] == key_owner[None, :])
                   & (key_positions[None, :] <= positions[:, None]))
        mask = hidden.new_zeros((1, 1, len(positions), len(key_owner))).masked_fill_(
            ~allowed[None, None], torch.finfo(hidden.dtype).min)
        lean = decoder_graph_lean() and hidden.device.type == 'cuda'
        fused = fused_kernels() if rows_fused(hidden, run_config=run_config) or lean else None
        pending = None
        for index, layer in enumerate(self.prefix_layers):
            layer_idx = layer.self_attn.layer_idx
            if index == len(self.prefix_layers) - 1:
                if fused is not None:
                    keys, values = decoder_prefix_kv_lean(layer, hidden, rope, fused,
                                                          pending=pending)
                else:
                    keys, values = decoder_prefix_kv(layer, hidden, rope)
            else:
                attn = layer.self_attn
                empty = hidden.new_empty((1, attn.k_proj.out_features // attn.head_dim,
                                          0, attn.head_dim))
                pair = tuple(torch.cat([
                    prefix.kv[layer_idx][component][..., :start, :]
                    if start else empty for _, prefix, start, _ in ready], dim=-2)
                    for component in range(2))
                packed = PackedPrefix({layer_idx: pair})
                if fused is not None:
                    pending = decoder_layer_lean(layer, hidden, rope, mask, packed, None,
                                                 fused, pending=pending, run_config=self.run_config)
                else:
                    hidden = layer(hidden, attention_mask=mask, position_ids=positions[None],
                                   position_embeddings=rope, past_key_values=packed,
                                   use_cache=False)
                keys, values = packed.appended[layer_idx]
            offset = 0
            for _, prefix, start, stop in ready:
                count = stop - start
                prefix.update(keys[..., offset:offset + count, :],
                              values[..., offset:offset + count, :], layer_idx,
                              {'cache_position': torch.arange(start, stop, device=hidden.device)})
                offset += count
        for _, prefix, _, stop in ready:
            prefix.length = stop

    def drop_sequence(self, sequence):
        """Drop replay/live state explicitly; live sequences also have a finalizer."""
        key = id(sequence)
        prefix = self._prefixes.pop(key, None)
        if isinstance(prefix, StaticPrefix) and prefix.pool is not None:
            prefix.pool.release(prefix.slot)
        finalizer = self._prefix_finalizers.pop(key, None)
        if finalizer is not None:
            finalizer.detach()

    def _pool_prefixes(self, prefixes):
        capacity = prefixes[0].capacity
        pool = self._static_prefix_pool
        if pool is None or pool.capacity != capacity:
            pool = self._static_prefix_pool = StaticPrefixPool(capacity)
        for prefix in prefixes:
            if isinstance(prefix, StaticPrefix) and prefix.pool is None:
                pool.adopt(prefix)

    def _new_prefix(self, sequence):
        key = id(sequence)
        fast_path = setting(self.run_config, 'engine', 'engine_settings.heads_fast', None) == '1'
        if fast_path:
            capacity = int(setting(self.run_config, 'engine', 'engine_settings.decoder_cache_capacity', '800'))
            prefix = StaticPrefix(capacity)
        else:
            prefix = IncrementalPrefix()
        self._prefixes[key] = prefix
        if fast_path:
            def drop():
                self._prefixes.pop(key, None)
                self._prefix_finalizers.pop(key, None)
                if prefix.pool is not None:
                    prefix.pool.release(prefix.slot)
            self._prefix_finalizers[key] = weakref.finalize(sequence, drop)
        return prefix

    def _prefix(self, sequence, desired_length, *, run_config=None):
        if getattr(self, '_uses_shipped_state', False):
            return self._stateful_prefix(sequence, desired_length)
        if self.state_policy['full_attention'] == 'ship':
            return SimpleNamespace(length=desired_length, kv=sequence.decoder_prefix_kv)
        key = id(sequence)
        prefix = self._prefixes.get(key)
        if prefix is None or prefix.length > desired_length:
            self.drop_sequence(sequence)
            prefix = self._new_prefix(sequence)
        tap33 = sequence.taps[:, self.tap_position, :].to(device=next(self.layers.parameters()).device,
                                          dtype=next(self.layers.parameters()).dtype)
        if isinstance(prefix, StaticPrefix) and prefix.length < desired_length:
            start = prefix.length
            hidden = tap33[start:desired_length][None]
            positions = torch.arange(start, desired_length, device=hidden.device,
                                     dtype=torch.long)
            position_ids = positions[None]
            rope = self._rope(hidden, position_ids)
            keys = torch.arange(desired_length, device=hidden.device)
            allowed = keys[None, :] <= positions[:, None]
            mask = hidden.new_zeros((1, 1, len(positions), desired_length)).masked_fill_(
                ~allowed[None, None], torch.finfo(hidden.dtype).min)
            lean = decoder_graph_lean() and hidden.device.type == 'cuda'
            # a rebuild longer than the captured shapes runs the same lean kernels eagerly
            fused = fused_kernels() if rows_fused(hidden, run_config=run_config) or lean else None
            pending = None
            lean_graphs = lean and hidden.shape[1] <= 16
            graphed = (hidden.device.type == 'cuda' and hidden.shape[1] <= 16 and
                       setting(self.run_config, 'engine', 'engine_settings.rows_graphs', None) == '1')
            for index, layer in enumerate(self.prefix_layers):
                if lean_graphs:
                    if index == len(self.prefix_layers) - 1:
                        key_states, value_states = self._lean_prefix_kv(layer, hidden, pending, rope)
                        cos, sin = rope
                        prefix.update(key_states, value_states, layer.self_attn.layer_idx,
                                      {'sin': sin, 'cos': cos, 'cache_position': positions})
                    else:
                        pending = self._window_layer_lean(layer, hidden, pending, mask, rope,
                                                          prefix, cache_position=positions)
                    continue
                if index == len(self.prefix_layers) - 1 and hasattr(layer, 'self_attn'):
                    # Only layer-35 K/V is consumed by the trained window.  Its
                    # attention output and MLP would be dead work.
                    graphed = (hidden.device.type == 'cuda' and hidden.shape[1] <= 16 and
                               setting(self.run_config, 'engine', 'engine_settings.rows_graphs', None) == '1')
                    if graphed:
                        key_states, value_states = self._replay_window_part(
                            (id(layer), 'prefix_kv', tuple(hidden.shape), hidden.dtype),
                            lambda state, cos, sin: decoder_prefix_kv(layer, state, (cos, sin)),
                            (hidden, *rope))
                    elif fused is not None:
                        key_states, value_states = decoder_prefix_kv_lean(
                            layer, hidden, rope, fused, pending=pending)
                    else:
                        key_states, value_states = decoder_prefix_kv(layer, hidden, rope)
                    attn = layer.self_attn
                    cos, sin = rope
                    prefix.update(key_states, value_states, attn.layer_idx,
                                  {'sin': sin, 'cos': cos, 'cache_position': positions})
                elif (hidden.device.type == 'cuda' and hidden.shape[1] <= 16 and
                      setting(self.run_config, 'engine', 'engine_settings.rows_graphs', None) == '1'):
                    hidden = self._window_layer(layer, hidden, mask, rope, prefix)
                elif fused is not None:
                    pending = decoder_layer_lean(layer, hidden, rope, mask, prefix, positions, fused,
                                                 pending=pending, run_config=self.run_config)
                else:
                    hidden = layer(hidden, attention_mask=mask, position_ids=position_ids,
                                   position_embeddings=rope, past_key_values=prefix,
                                   cache_position=positions, use_cache=True)
            prefix.length = desired_length
            return prefix
        for position in range(prefix.length, desired_length):
            hidden = tap33[position:position + 1][None]
            position_ids = torch.tensor([[position]], device=hidden.device, dtype=torch.long)
            rope = self.rotary(hidden, position_ids)
            for layer in self.prefix_layers:
                hidden = layer(hidden, attention_mask=None, position_ids=position_ids,
                               position_embeddings=rope, past_key_values=prefix,
                               cache_position=position_ids.flatten(), use_cache=True)
            prefix.length += 1
        return prefix

    def _replay_window_part(self, name, step, values):
        entry = self._window_graphs.get(name)
        if entry is None:
            if self._graphs_warmed:
                raise RuntimeError(f'decoder graph missing after warm-up: {name}')
            inputs = tuple(value.clone() for value in values)
            current = torch.cuda.current_stream(values[0].device)
            stream = torch.cuda.Stream(device=values[0].device)
            stream.wait_stream(current)
            with torch.cuda.stream(stream):
                for _ in range(3):
                    step(*inputs)
            current.wait_stream(stream)
            expected = tuple(value.clone() for value in step(*inputs))
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                result = step(*inputs)
            graph.replay()
            if not all(torch.equal(a, b) for a, b in zip(expected, result)):
                raise RuntimeError(f'decoder graph replay changed {name}')
            self._window_graphs[name] = (inputs, graph, result)
        else:
            inputs, graph, result = entry
            for destination, value in zip(inputs, values):
                destination.copy_(value)
            graph.replay()
        return result

    def _window_layer(self, layer, hidden, mask, rope, frozen):
        from transformers.integrations.sdpa_attention import sdpa_attention_forward
        attn = layer.self_attn
        query, key, value = self._replay_window_part(
            (id(layer), 'project', tuple(hidden.shape), hidden.dtype),
            lambda state, cos, sin: decoder_window_project(layer, state, (cos, sin)),
            (hidden, *rope))
        if frozen is not None:
            key, value = frozen.update(key, value, attn.layer_idx,
                                       {'sin': rope[1], 'cos': rope[0],
                                        'cache_position': None})
        attended = sdpa_attention_forward(
            attn, query, key, value, mask, dropout=0.0, scaling=attn.scaling,
            sliding_window=attn.sliding_window)[0]
        result, = self._replay_window_part(
            (id(layer), 'finish', tuple(hidden.shape), hidden.dtype),
            lambda state, output: (decoder_window_finish(layer, state, output),),
            (hidden, attended))
        return result

    def _cached_window_mask(self, context_length, hidden, size=15):
        mask = self._window_mask_cache
        if (mask is None or mask.shape[-1] < context_length + size or mask.shape[-2] != size or
                mask.device != hidden.device or mask.dtype != hidden.dtype):
            capacity = ((context_length + 255) // 256) * 256
            mask = window_mask(torch.tensor([capacity]), capacity, hidden.dtype,
                               size).to(hidden.device)
            self._window_mask_cache = mask
        return mask[..., mask.shape[-1] - context_length - size:].contiguous()

    @torch.inference_mode()
    def presize_batch_pool(self, slots):
        """JETSPEC_BATCH_DRAFT_GRAPH: create or grow the N>1 static prefix pool to ``slots``
        once, before any batch graph is captured, so its storage (which the captured decoder
        and prefix-append graphs read) does not move during a read."""
        capacity = int(setting(self.run_config, 'engine', 'engine_settings.decoder_cache_capacity', '800'))
        pool = self._static_prefix_pool
        if pool is None or pool.capacity != capacity:
            pool = self._static_prefix_pool = StaticPrefixPool(capacity)
        first = len(pool.owners)
        if slots <= first:
            return pool
        pool.owners.extend([None] * (slots - first))
        pool.free = list(range(slots - 1, first - 1, -1)) + pool.free
        for index, pair in list(pool.kv.items()):
            pool.kv[index] = tuple(torch.cat((tensor, tensor.new_zeros(
                (slots - first, *tensor.shape[1:])))) for tensor in pair)
            pool._rebind(index)
        return pool

    @torch.inference_mode()
    def retire_stale_batch_prefixes(self, sequences):
        """JETSPEC_BATCH_DRAFT_GRAPH, N>1: the drafter keeps finished sequences, so their
        pooled prefixes are released only by a finalizer that does not fire during a read, and
        the pool would double (moving its storage) once per slot count. Before this round pools
        its new prefixes, release the least recently used prefixes that no N>1 round used in
        the last two rounds, until the pool has a free slot for each. A released prefix that
        is used again is rebuilt from its taps (``_prefix``), so the values stay exact."""
        tick = self.__dict__.get('_batch_tick', 0) + 1
        self._batch_tick = tick
        seen = self.__dict__.setdefault('_batch_seen', {})
        for sequence in sequences:
            seen[id(sequence)] = tick
        pool = self._static_prefix_pool
        if pool is None:
            return 0
        need = sum(1 for sequence in sequences
                   if getattr(self._prefixes.get(id(sequence)), 'pool', None) is not pool)
        released = 0
        if len(pool.free) >= need:
            return released
        stale = sorted((used, key) for key, used in seen.items()
                       if used <= tick - 2 and getattr(self._prefixes.get(key), 'pool', None) is pool)
        for _, key in stale:
            if len(pool.free) >= need:
                break
            prefix = self._prefixes.pop(key)
            pool.release(prefix.slot)
            finalizer = self._prefix_finalizers.pop(key, None)
            if finalizer is not None:
                finalizer.detach()
            seen.pop(key, None)
            released += 1
        return released

    @torch.inference_mode()
    def prefetch_batch_graph(self, sequences, *, run_config=None):
        """JETSPEC_BATCH_DRAFT_GRAPH_FRONT=1, N>1: every pooled request's prefix append of at most one
        window (up to 16 rows) as one captured call per (batch bucket, 128-row prefix bucket),
        writing both prefix layers' K/V rows into the static pool; a new, stale or unpooled
        prefix takes ``_prefix``. Not bitwise to the per-request append (other GEMM row counts;
        masked padded keys contribute exact zeros)."""
        from jetspec.engines.jetspec_plus_plus.wire.protocol import draft_width
        parameter = next(self.layers.parameters())
        ready = []
        for sequence in sequences:
            staged = getattr(getattr(sequence, 'staged', None), 'predicted_hidden', None)
            if (staged is None
                    or int(staged.shape[1]) < draft_width(run_config=run_config if run_config is not None else getattr(self, 'run_config', None)) + 1):
                continue
            desired = int(sequence.taps.shape[0])
            if desired < 0:
                continue
            prefix = self._prefixes.get(id(sequence))
            if isinstance(prefix, StaticPrefix) and prefix.pool is None and prefix.length:
                self._pool_prefixes([prefix])
            if (not isinstance(prefix, StaticPrefix) or prefix.pool is None or not prefix.length
                    or prefix.length > desired or desired - prefix.length > 16
                    or not self._batch_graph_ready([prefix], parameter)):
                self._prefix(sequence, desired)
                continue
            if prefix.length < desired:
                ready.append((sequence, prefix, prefix.length, desired))
        if not ready:
            return
        pool = ready[0][1].pool
        indices = [layer.self_attn.layer_idx for layer in self.prefix_layers]
        if any(index not in pool.kv for index in indices) or any(
                prefix.pool is not pool for _, prefix, _, _ in ready):
            for sequence, _, _, desired in ready:
                self._prefix(sequence, desired)
            return
        trash = getattr(self, '_prefetch_trash', None)
        if trash is None or trash.pool is not pool:
            # a pool slot of its own takes the padded rows' writes
            trash = self._prefetch_trash = StaticPrefix(pool.capacity, pool=pool)
        count, window = len(ready), 16
        bucket = max(2, 1 << (count - 1).bit_length())
        capacity = min(pool.capacity - window,
                       max(128, ((max(start for _, _, start, _ in ready) + 127) // 128) * 128))
        if max(start for _, _, start, _ in ready) > capacity:
            for sequence, _, _, desired in ready:
                self._prefix(sequence, desired)
            return
        storage = tuple(tensor.data_ptr() for index in sorted(pool.kv) for tensor in pool.kv[index])
        graphs = self.__dict__.setdefault('_prefetch_graphs', {})
        key = (bucket, capacity, parameter.dtype, storage)
        entry = graphs.get(key)
        if entry is None:
            for stale in [name for name in graphs if name[3] != storage]:
                graphs.pop(stale)  # the pool grew: its storage moved
            frozen = {}
            for layer in self.prefix_layers[:-1]:
                index = layer.self_attn.layer_idx
                for component, source in enumerate(pool.kv[index]):
                    frozen[(index, component)] = source.new_empty(
                        (bucket, source.shape[1], capacity + window, source.shape[-1]))
            entry = dict(hidden=parameter.new_zeros(bucket * window, self.hidden_size),
                         ints=torch.zeros(2 * bucket + 2 * bucket * window, dtype=torch.long,
                                          device=parameter.device),
                         host_ints=torch.empty(2 * bucket + 2 * bucket * window,
                                               dtype=torch.long, pin_memory=True),
                         host_index=torch.empty(bucket * window, dtype=torch.long, pin_memory=True),
                         index=torch.empty(bucket * window, dtype=torch.long, device=parameter.device),
                         capacity=capacity, pool=pool, frozen=frozen, graph=None)
            ints = entry['ints']
            entry['starts'], entry['slots'] = ints[:bucket], ints[bucket:2 * bucket]
            entry['write_slots'] = ints[2 * bucket:2 * bucket + bucket * window]
            entry['write_positions'] = ints[2 * bucket + bucket * window:]
            graphs[key] = entry
        starts = [start for _, _, start, _ in ready] + [0] * (bucket - count)
        slots = [prefix.slot for _, prefix, _, _ in ready] + [trash.slot] * (bucket - count)
        if entry.get('copy_done') is not None:
            entry['copy_done'].synchronize()
        parts = [sequence.taps[start:desired, self.tap_position, :]
                 for sequence, _, start, desired in ready]
        used = _fill_prefetch_indices(entry['host_ints'], entry['host_index'],
                                      starts[:count], slots[:count],
                                      [desired - start for _, _, start, desired in ready],
                                      trash.slot, bucket, window)
        entry['ints'].copy_(entry['host_ints'], non_blocking=True)
        entry['index'][:used].copy_(entry['host_index'][:used], non_blocking=True)
        if entry.get('copy_done') is None:
            entry['copy_done'] = torch.cuda.Event()
        entry['copy_done'].record()
        entry['hidden'].index_copy_(0, entry['index'][:used], torch.cat(parts).to(parameter))
        if entry['graph'] is None:
            stream = torch.cuda.Stream(device=parameter.device)
            stream.wait_stream(torch.cuda.current_stream(parameter.device))
            with torch.cuda.stream(stream):
                for _ in range(2):
                    self._prefetch_graph_body(entry, bucket, window)
            torch.cuda.current_stream(parameter.device).wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            if self.__dict__.get('_batch_graph_mempool') is None:
                self._batch_graph_mempool = torch.cuda.graph_pool_handle()
            with torch.cuda.graph(graph, pool=self._batch_graph_mempool):
                self._prefetch_graph_body(entry, bucket, window)
            entry['graph'] = graph
            print(f'BATCH_DRAFT_GRAPH_CAPTURE prefetch bucket={bucket} prefix={capacity}', flush=True)
        entry['graph'].replay()
        for _, prefix, _, desired in ready:
            prefix.length = desired

    def _prefetch_graph_body(self, entry, bucket, window):
        hidden = entry['hidden'].view(bucket, window, self.hidden_size)
        starts, slots, capacity, pool = (entry['starts'], entry['slots'], entry['capacity'],
                                         entry['pool'])
        positions = starts[:, None] + torch.arange(window, device=hidden.device)[None, :]
        rope = self._rope(hidden, positions)
        fused = fused_kernels()
        pending = None
        for position, layer in enumerate(self.prefix_layers):
            index = layer.self_attn.layer_idx
            if position == len(self.prefix_layers) - 1:
                key, value = decoder_prefix_kv_lean(layer, hidden, rope, fused, pending=pending)
            else:
                slot_prefix = SlotBatchPrefix(pool, slots, starts, capacity)
                pending = decoder_layer_lean(layer, hidden, rope, None,
                                             slot_prefix,
                                             None, fused, pending=pending, run_config=self.run_config)
                key, value = slot_prefix.latest
            for component, rows in enumerate((key, value)):
                pool.kv[index][component][entry['write_slots'], :, entry['write_positions']] = (
                    rows.transpose(1, 2).reshape(bucket * window, rows.shape[1], rows.shape[-1]))

    def _batch_graph_ready(self, prefixes, parameter, *, run_config=None):
        """JETSPEC_BATCH_DRAFT_GRAPH_FRONT=1 on the fused bf16 CUDA path with every request's prefix
        in the one static pool (captured attention reads that pool's storage by slot)."""
        if (setting(self.run_config, 'engine', 'engine_settings.batch_draft_graph', '0') != '1'
                or setting(self.run_config, 'engine', 'engine_settings.batch_draft_graph_front', '0') != '1'
                or parameter.device.type != 'cuda' or not rows_fused(parameter, run_config=self.run_config)):
            return False
        pool = getattr(prefixes[0], 'pool', None)
        return pool is not None and all(
            isinstance(prefix, StaticPrefix) and prefix.pool is pool for prefix in prefixes)

    def _batch_graph_body(self, entry):
        hidden, lengths, slots = entry['hidden'], entry['lengths'], entry['slots']
        capacity, pool = entry['capacity'], entry['pool']
        count, size = hidden.shape[:2]
        positions = lengths[:, None] + torch.arange(size, device=hidden.device)[None, :]
        rope = self._rope(hidden, positions)
        frozen = SlotBatchPrefix(pool, slots, lengths, capacity)
        fused = fused_kernels()
        pending = None
        for layer in self.layers:
            pending = decoder_layer_lean(layer, hidden, rope, None, frozen, None, fused,
                                         pending=pending, run_config=self.run_config)
        decoded = fused.fast_add_rms_norm(*pending, self.norm.weight,
                                          self.norm.variance_epsilon)[0]
        decoded = decoded.reshape(count, size, self.hidden_size)
        if entry['compact']:
            from jetspec.engines.jetspec_plus_plus.kernels.drafter_lmhead import head_topk_logits
            logits = self.lm_head(decoded.reshape(-1, self.hidden_size))
            return decoded, head_topk_logits(
                logits, entry['rejected'], entry['cap'],
                score_rejected=entry['score_rejected'],
                prefilter='0' == '1')
        logits = self.lm_head(decoded.reshape(-1, self.hidden_size)).view(count, size, -1)
        return decoded, logits

    def _predict_batch_graph(self, hidden, lengths, prefixes, rejected=None, cap=32,
                             score_rejected=None):
        """[N, draft_width() + 1, hidden] windows -> (decoded, logits) for the N live requests, from the
        graph of the padded batch bucket and the 128-row prefix bucket (padded requests read
        pool slot 0 with no visible prefix; their rows are dropped)."""
        count, size = hidden.shape[:2]
        bucket = max(2, 1 << (count - 1).bit_length())
        pool = prefixes[0].pool
        capacity = min(pool.capacity, max(128, ((max(lengths) + 127) // 128) * 128))
        storage = tuple(tensor.data_ptr() for index in sorted(pool.kv) for tensor in pool.kv[index])
        graphs = self.__dict__.setdefault('_batch_graphs', {})
        compact = (rejected is not None and
                   batch_head_epilogue_enabled(count, hidden.dtype, cap, run_config=self.run_config))
        key = (bucket, capacity, size, hidden.dtype, storage, compact,
               cap if compact else None)
        entry = graphs.get(key)
        if entry is None:
            for stale in [name for name in graphs if name[4] != storage]:
                graphs.pop(stale)  # the pool grew: its storage moved
            frozen = {}
            for layer in self.layers:
                index = layer.self_attn.layer_idx
                for component, source in enumerate(pool.kv[index]):
                    frozen[(index, component)] = source.new_empty(
                        (bucket, source.shape[1], capacity + size, source.shape[-1]))
            entry = dict(hidden=hidden.new_zeros(bucket, size, hidden.shape[-1]),
                         ints=torch.zeros(2 * bucket, dtype=torch.long, device=hidden.device),
                         capacity=capacity, pool=pool, frozen=frozen, graph=None,
                         compact=compact, cap=cap)
            if compact:
                entry['rejected'] = torch.full((bucket, size), -1, dtype=torch.long,
                                               device=hidden.device)
                entry['score_rejected'] = torch.full_like(entry['rejected'], -1)
            entry['lengths'], entry['slots'] = entry['ints'][:bucket], entry['ints'][bucket:]
            graphs[key] = entry
        host = torch.tensor(list(lengths) + [0] * (bucket - count)
                            + [prefix.slot for prefix in prefixes] + [0] * (bucket - count),
                            dtype=torch.long).pin_memory()
        entry['ints'].copy_(host, non_blocking=True)
        entry['hidden'][:count].copy_(hidden)
        if compact:
            entry['rejected'][:count].copy_(rejected)
            entry['score_rejected'][:count].copy_(
                rejected if score_rejected is None else score_rejected)
        if entry['graph'] is None:
            stream = torch.cuda.Stream(device=hidden.device)
            stream.wait_stream(torch.cuda.current_stream(hidden.device))
            with torch.cuda.stream(stream):
                for _ in range(2):
                    self._batch_graph_body(entry)
            torch.cuda.current_stream(hidden.device).wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            if self.__dict__.get('_batch_graph_mempool') is None:
                self._batch_graph_mempool = torch.cuda.graph_pool_handle()
            with torch.cuda.graph(graph, pool=self._batch_graph_mempool):
                entry['output'] = self._batch_graph_body(entry)
            entry['graph'] = graph
            print(f'BATCH_DRAFT_GRAPH_CAPTURE decoder bucket={bucket} prefix={capacity}', flush=True)
        entry['graph'].replay()
        decoded, logits = entry['output']
        capture = None
        if capture and compact and not entry.get('w632_captured'):
            torch.save(decoded[:count].detach().cpu(), capture + '-decoder.pt')
            torch.save(entry['rejected'][:count].detach().cpu(), capture + '-rejected.pt')
            torch.save(entry['score_rejected'][:count].detach().cpu(),
                       capture + '-score-rejected.pt')
            entry['w632_captured'] = True
        # the captured outputs are overwritten by the next replay: the round keeps copies
        if compact:
            from jetspec.engines.jetspec_plus_plus.kernels.drafter_lmhead import DecoderEpilogue
            live = count * size
            epilogue = DecoderEpilogue(logits.ids[:live].clone(), logits.p64[:live].clone(),
                                        logits.pack[:, :live].clone(),
                                        logits.rejected_probability[:live].clone(),
                                        logits.rejected_was_top1[:live].clone())
            return decoded[:count].clone(), epilogue
        return decoded[:count].clone(), logits[:count].clone()

    @torch.inference_mode()
    def _predict_batch_graphed(self, sequences, pending_blocks, lengths, prefixes, *, dataset,
                               coverage, grouped, cap, after_misses, batch_started, prefix_ms, run_config=None):
        """JETSPEC_BATCH_DRAFT_GRAPH_FRONT=1: ``predict_batch`` with its window layers, norm and
        vocabulary projection as one captured call per (batch bucket, 128-row prefix bucket);
        the offer selection below is ``predict_batch``'s own."""
        from jetspec.engines.jetspec_plus_plus.wire.protocol import draft_width
        count = draft_width(run_config=run_config if run_config is not None else getattr(self, 'run_config', None)) + 1
        parameter = next(self.layers.parameters())
        currents = torch.cat([self._tap_hidden(sequence.staged.predicted_hidden[:, :count])
                              for sequence in sequences], dim=0).to(parameter)
        started = time.perf_counter()
        rejected_base = torch.stack([rejected_tokens(block, after_miss=False, run_config=self.run_config)
                                     for block in pending_blocks])
        rejected = rejected_base
        if setting(self.run_config, 'method', 'method_settings.kset_after_miss', None) == 'zero' and any(after_misses):
            rejected = rejected_base.clone()
            rejected[torch.as_tensor(after_misses, device=rejected.device), 0] = -1
        decoded, logits = self._predict_batch_graph(currents,
                                                    lengths, prefixes, rejected, cap,
                                                    score_rejected=rejected_base)
        selection_started = time.perf_counter()
        from jetspec.engines.jetspec_plus_plus.kernels.drafter_lmhead import DecoderEpilogue
        if isinstance(logits, DecoderEpilogue):
            offers = [DeviceAnchorOffers(
                logits.ids[row * count:(row + 1) * count],
                logits.p64[row * count:(row + 1) * count],
                logits.pack[:, row * count:(row + 1) * count],
                None, cap, cap + 1)
                for row in range(len(sequences))]
            elapsed = (time.perf_counter() - started) * 1000
            total_ms = (time.perf_counter() - batch_started) * 1000
            return [DecoderPrediction(
                hidden=decoded[row], logits=None, offers=offers[row], elapsed_ms=elapsed,
                total_ms=total_ms, prefix_ms=prefix_ms,
                select_ms=(time.perf_counter() - selection_started) * 1000,
                stage_ms={'prefix': prefix_ms, 'selection': 0.0},
                rejected=rejected_base[row],
                compact_q=logits.rejected_probability.reshape(-1, count)[row],
                compact_argmax_eq=logits.rejected_was_top1.reshape(-1, count)[row])
                for row in range(len(sequences))]
        if (setting(self.run_config, 'engine', 'engine_settings.rows_assembly', '0') == '1'
                and setting(self.run_config, 'method', 'method_settings.anchor_select', None) == 'topk'
                and setting(self.run_config, 'method', 'method_settings.anchor_select_home', None) == 'one'
                and setting(self.run_config, 'engine', 'engine_settings.heads_fast', None) == '1'):
            combined = anchor_offers(logits.reshape(-1, logits.shape[-1]),
                                     rejected.reshape(-1).to(logits.device), root=self.root,
                                     dataset=dataset, coverage=coverage, grouped=grouped, cap=cap, run_config=self.run_config)
            if isinstance(combined, DeviceAnchorOffers):
                ids, probabilities, pack = combined.device
                offers = [DeviceAnchorOffers(
                    ids[row * count:(row + 1) * count],
                    probabilities[row * count:(row + 1) * count],
                    pack[:, row * count:(row + 1) * count],
                    combined._probabilities[row * count:(row + 1) * count],
                    combined._cap, combined._width)
                    for row in range(len(sequences))]
            else:
                offers = None
        else:
            offers = None
        if offers is None:
            offers = [anchor_offers(logits[row], rejected[row].to(logits.device),
                      root=self.root, dataset=dataset, coverage=coverage,
                      grouped=grouped, cap=cap, run_config=self.run_config)
                      for row, (block, miss) in enumerate(zip(pending_blocks, after_misses))]
        select_ms = (time.perf_counter() - selection_started) * 1000
        elapsed = (time.perf_counter() - started) * 1000
        total_ms = (time.perf_counter() - batch_started) * 1000
        return [DecoderPrediction(
            hidden=decoded[row], logits=logits[row],
            offers=offers[row], elapsed_ms=elapsed,
            total_ms=total_ms, prefix_ms=prefix_ms, select_ms=select_ms,
            stage_ms={'prefix': prefix_ms, 'selection': select_ms},
            rejected=rejected_base[row])
            for row, (block, miss) in enumerate(zip(pending_blocks, after_misses))]

    @torch.inference_mode()
    def predict_batch(self, sequences, pending_blocks, *, dataset, coverage, grouped, cap,
                      after_misses, run_config=None):
        """Run the trained decoder window for independent requests in one batch."""
        if getattr(self, '_uses_shipped_state', False):
            return self._stateful_predict_batch(sequences, pending_blocks, dataset=dataset, coverage=coverage, grouped=grouped, cap=cap, after_misses=after_misses, run_config=run_config)
        if len(sequences) != len(pending_blocks) or len(sequences) != len(after_misses):
            raise ValueError('decoder batch inputs must align')
        if len(sequences) == 1:
            return [self.predict(sequences[0], pending_blocks[0], dataset=dataset,
                                 coverage=coverage, grouped=grouped, cap=cap,
                                 after_miss=after_misses[0], run_config=run_config if run_config is not None else getattr(self, 'run_config', None))]
        from jetspec.engines.jetspec_plus_plus.wire.protocol import draft_width
        count = draft_width(run_config=run_config if run_config is not None else getattr(self, 'run_config', None)) + 1
        size = count
        batch_started = time.perf_counter()
        parameter = next(self.layers.parameters())
        lengths, prefixes, windows = [], [], []
        vectorized = getattr(self, '_batch_vectorized', True)
        for sequence in sequences:
            staged = sequence.staged.predicted_hidden
            if staged is None or staged.shape[1] < count:
                if setting(self.run_config, 'engine', 'engine_settings.device_split', '0') == '1':
                    print('B22_DECODER_FALLBACK batch_unready=1', flush=True)
                return [self.predict(seq, block, dataset=dataset, coverage=coverage,
                                     grouped=grouped, cap=cap, after_miss=miss, run_config=run_config if run_config is not None else getattr(self, 'run_config', None))
                        for seq, block, miss in zip(sequences, pending_blocks, after_misses)]
            length = int(sequence.taps.shape[0])
            if length < 0:
                if setting(self.run_config, 'engine', 'engine_settings.device_split', '0') == '1':
                    print('B22_DECODER_FALLBACK prefix_unready=1', flush=True)
                return [self.predict(seq, block, dataset=dataset, coverage=coverage,
                                     grouped=grouped, cap=cap, after_miss=miss, run_config=run_config if run_config is not None else getattr(self, 'run_config', None))
                        for seq, block, miss in zip(sequences, pending_blocks, after_misses)]
            lengths.append(length)
            prefixes.append(self._prefix(sequence, length))
            if not vectorized:
                current = self._tap_hidden(staged[0, :count]).to(parameter)
                windows.append(current)
        prefix_ms = (time.perf_counter() - batch_started) * 1000
        if all(isinstance(prefix, StaticPrefix) for prefix in prefixes):
            self._pool_prefixes(prefixes)
        if vectorized and self._batch_graph_ready(prefixes, parameter):
            return self._predict_batch_graphed(
                sequences, pending_blocks, lengths, prefixes, dataset=dataset, coverage=coverage,
                grouped=grouped, cap=cap, after_misses=after_misses, batch_started=batch_started,
                prefix_ms=prefix_ms, run_config=run_config if run_config is not None else getattr(self, 'run_config', None))
        if vectorized:
            currents = torch.cat([self._tap_hidden(sequence.staged.predicted_hidden[:, :count])
                                  for sequence in sequences], dim=0).to(parameter)
            hidden = currents
            positions = (torch.as_tensor(lengths, device=hidden.device)[:, None]
                         + torch.arange(size, device=hidden.device)[None, :])
        else:
            hidden = torch.stack(windows)
            positions = torch.stack([torch.arange(length, length + size,
                                   device=hidden.device) for length in lengths])
        rope = self._rope(hidden, positions)
        capacity = max(lengths)
        if vectorized:
            keys = torch.arange(capacity + size, device=hidden.device)
            queries = torch.arange(size, device=hidden.device)
            allowed = ((keys[None, None, :] < torch.as_tensor(lengths, device=hidden.device)[:, None, None])
                       | ((keys[None, None, :] >= capacity)
                          & (keys[None, None, :] - capacity <= queries[None, :, None])))
            mask = hidden.new_zeros((len(sequences), 1, size, capacity + size)).masked_fill_(
                ~allowed[:, None], torch.finfo(hidden.dtype).min)
        else:
            mask = hidden.new_full((len(sequences), 1, size, capacity + size),
                                   torch.finfo(hidden.dtype).min)
            private = torch.arange(size, device=hidden.device)
            for row, length in enumerate(lengths):
                mask[row, :, :, :length] = 0
                mask[row, 0, :, capacity:] = torch.zeros((size, size), device=hidden.device,
                                                          dtype=hidden.dtype).masked_fill(
                    private[None, :] > private[:, None], torch.finfo(hidden.dtype).min)
        frozen = None
        if capacity:
            kv = {}
            pool = prefixes[0].pool if isinstance(prefixes[0], StaticPrefix) else None
            pooled = vectorized and pool is not None and all(
                isinstance(prefix, StaticPrefix) and prefix.pool is pool for prefix in prefixes)
            if pooled:
                slots = torch.tensor([prefix.slot for prefix in prefixes],
                                     device=hidden.device, dtype=torch.long)
            for layer in self.layers:
                index = layer.self_attn.layer_idx
                if pooled:
                    pair = []
                    for component, source in enumerate(pool.kv[index]):
                        name = (index, component)
                        stored = self._batch_frozen.get(name)
                        shape = (len(sequences), source.shape[1], pool.capacity + size, source.shape[-1])
                        if (stored is None or stored.shape != shape or stored.device != source.device
                                or stored.dtype != source.dtype):
                            stored = self._batch_frozen[name] = torch.empty(
                                shape, device=source.device, dtype=source.dtype)
                        torch.index_select(source[..., :capacity, :], 0, slots,
                                           out=stored[..., :capacity, :])
                        pair.append(stored[..., :capacity + size, :])
                    kv[index] = tuple(pair)
                else:
                    example = next(prefix.kv[index] for prefix in prefixes if index in prefix.kv)
                    pair = []
                    for component in range(2):
                        rows = []
                        for prefix, length in zip(prefixes, lengths):
                            value = (prefix.kv[index][component][..., :length, :]
                                     if length else example[component][..., :0, :])
                            rows.append(torch.nn.functional.pad(value, (0, 0, 0, capacity - length)))
                        pair.append(torch.cat(rows, dim=0))
                    kv[index] = tuple(pair)
            frozen = FrozenBatchPrefix(kv, capacity) if pooled else FrozenPrefix(kv)
        started = time.perf_counter()
        fused = fused_kernels() if rows_fused(hidden, run_config=run_config) else None
        pending = None
        for layer in self.layers:
            if fused is not None:
                pending = decoder_layer_lean(layer, hidden, rope, mask, frozen, None, fused,
                                             pending=pending, run_config=self.run_config)
            else:
                hidden = decode_layer(layer, hidden, mask, positions, rope, frozen)
        decoded = (fused.fast_add_rms_norm(*pending, self.norm.weight,
                   self.norm.variance_epsilon)[0] if fused is not None else self.norm(hidden))
        decoded = decoded.reshape(len(sequences), size, self.hidden_size)
        if setting(self.run_config, 'engine', 'engine_settings.batch_draft_graph', '0') == '1':
            # The [N, 16, hidden] input sends F.linear to a batched GEMM over the
            # broadcast vocabulary weight (N GEMMs of 16 rows); one [N*16, hidden] GEMM instead
            logits = self.lm_head(decoded.reshape(-1, self.hidden_size)).view(
                len(sequences), count, -1)
        else:
            logits = self.lm_head(decoded)
        selection_started = time.perf_counter()
        rejected_base = torch.stack([rejected_tokens(block, after_miss=False, run_config=self.run_config)
                                     for block in pending_blocks])
        rejected = rejected_base
        if setting(self.run_config, 'method', 'method_settings.kset_after_miss', None) == 'zero' and any(after_misses):
            rejected = rejected_base.clone()
            rejected[torch.as_tensor(after_misses, device=rejected.device), 0] = -1
        if (setting(self.run_config, 'engine', 'engine_settings.rows_assembly', '0') == '1'
                and setting(self.run_config, 'method', 'method_settings.anchor_select', None) == 'topk'
                and setting(self.run_config, 'method', 'method_settings.anchor_select_home', None) == 'one'
                and setting(self.run_config, 'engine', 'engine_settings.heads_fast', None) == '1'):
            combined = anchor_offers(logits.reshape(-1, logits.shape[-1]),
                                     rejected.reshape(-1).to(logits.device), root=self.root,
                                     dataset=dataset, coverage=coverage, grouped=grouped, cap=cap, run_config=self.run_config)
            if isinstance(combined, DeviceAnchorOffers):
                ids, probabilities, pack = combined.device
                offers = [DeviceAnchorOffers(
                    ids[row * count:(row + 1) * count],
                    probabilities[row * count:(row + 1) * count],
                    pack[:, row * count:(row + 1) * count],
                    combined._probabilities[row * count:(row + 1) * count],
                    combined._cap, combined._width)
                    for row in range(len(sequences))]
            else:
                offers = None
        else:
            offers = None
        if offers is None:
            offers = [anchor_offers(logits[row], rejected[row].to(logits.device),
                      root=self.root, dataset=dataset, coverage=coverage,
                      grouped=grouped, cap=cap, run_config=self.run_config)
                      for row, (block, miss) in enumerate(zip(pending_blocks, after_misses))]
        select_ms = (time.perf_counter() - selection_started) * 1000
        elapsed = (time.perf_counter() - started) * 1000
        total_ms = (time.perf_counter() - batch_started) * 1000
        return [DecoderPrediction(
            hidden=decoded[row], logits=logits[row],
            offers=offers[row], elapsed_ms=elapsed,
            total_ms=total_ms, prefix_ms=prefix_ms, select_ms=select_ms,
            stage_ms={'prefix': prefix_ms, 'selection': select_ms},
            rejected=rejected_base[row])
            for row, (block, miss) in enumerate(zip(pending_blocks, after_misses))]

    @torch.inference_mode()
    def predict(self, sequence, pending_block, *, dataset, coverage, grouped, cap,
                after_miss=False, run_config=None):
        if getattr(self, '_uses_shipped_state', False):
            return self._stateful_predict(sequence, pending_block, dataset=dataset, coverage=coverage, grouped=grouped, cap=cap, after_miss=after_miss, run_config=run_config)
        staged = sequence.staged.predicted_hidden
        if staged is None:
            return None
        from jetspec.engines.jetspec_plus_plus.wire.protocol import draft_width
        # serving: 15 current slots (c, d1..d14), 16 with 15 drafts (k = 0..15)
        count = draft_width(run_config=run_config if run_config is not None else getattr(self, 'run_config', None)) + 1
        size = count
        if int(staged.shape[1]) < count:
            raise RuntimeError(f'decoder requires the staged {count}-position adapter block')
        parameter = next(self.layers.parameters())
        live_round = hasattr(sequence, '_jetspec_seq_id') and \
            hasattr(sequence, '_jetspec_candidate_round')
        profile = None == '1' and live_round
        if profile and parameter.device.type == 'cuda':
            torch.cuda.synchronize(parameter.device)
        profile_total_started = time.perf_counter()
        window_started = profile_total_started
        total_started = time.perf_counter()
        # Event timing ends in a device synchronize; only the profile asks for it.
        cuda_timing = profile and parameter.device.type == 'cuda'
        if cuda_timing:
            (total_begin, window_end, prefix_begin, prefix_end, mask_begin,
             mask_end, layers_begin, layers_end, norm_begin, norm_end,
             select_begin, select_end, total_end) = (
                torch.cuda.Event(enable_timing=True) for _ in range(13))
            total_begin.record()
        current = self._tap_hidden(staged[0, :count]).to(parameter)
        hidden = decoder_window(current).to(
            device=parameter.device, dtype=parameter.dtype)
        anchor = int(sequence.taps.shape[0])
        context_length = anchor
        if context_length < 0:
            return None
        record_root = None
        record_limit = int('240')
        if record_root and live_round and self._recorded_inputs < record_limit:
            destination = Path(record_root) / 'decoder'
            destination.mkdir(parents=True, exist_ok=True)
            torch.save(dict(
                tap33=sequence.taps[:, self.tap_position, :].detach().cpu(),
                current=current.detach().cpu(),
                draft_ids=pending_block.draft_ids.detach().cpu(),
                correction_guess=int(pending_block.correction_guess),
                k_probabilities=(None if pending_block.k_probabilities is None else
                                 pending_block.k_probabilities.detach().cpu()),
                kpred_pass=pending_block.kpred_pass,
                dataset=dataset, coverage=str(coverage), grouped=bool(grouped),
                cap=int(cap), after_miss=bool(after_miss),
                seq_id=getattr(sequence, '_jetspec_seq_id', None),
                round_id=getattr(sequence, '_jetspec_candidate_round', None)),
                destination / f'{os.getpid()}-{self._recorded_inputs:04d}.pt')
            self._recorded_inputs += 1
        if profile and parameter.device.type == 'cuda':
            torch.cuda.synchronize(parameter.device)
        window_assembly_ms = (time.perf_counter() - window_started) * 1000.0
        prefix_started = time.perf_counter()
        if cuda_timing:
            window_end.record()
            prefix_begin.record()
        prefix = self._prefix(sequence, context_length)
        if cuda_timing:
            prefix_end.record()
        prefix_cpu_ms = (time.perf_counter() - prefix_started) * 1000.0
        window_mask_started = time.perf_counter()
        if cuda_timing:
            mask_begin.record()
        if setting(self.run_config, 'engine', 'engine_settings.decoder_tables', None) == '1' and rows_fused(hidden, run_config=self.run_config):
            # the fused layers read only the rotary rows, never the position ids
            position_ids = None
            rope = self._rope_range(hidden, anchor, anchor + count)
        else:
            position_ids = torch.arange(
                anchor, anchor + count, device=hidden.device, dtype=torch.long)[None]
            rope = self._rope(hidden, position_ids)
        mask = self._cached_window_mask(context_length, hidden, size)
        if profile and parameter.device.type == 'cuda':
            torch.cuda.synchronize(parameter.device)
        window_mask_ms = (time.perf_counter() - window_mask_started) * 1000.0
        started = time.perf_counter()
        if isinstance(prefix, StaticPrefix):
            frozen = StaticWindowPrefix(prefix, context_length)
        else:
            frozen = FrozenPrefix(prefix.kv) if context_length else None
        layers_started = time.perf_counter()
        if cuda_timing:
            mask_end.record()
            layers_begin.record()
        fused = fused_kernels() if rows_fused(hidden, run_config=run_config) else None
        pending = None
        lean_graphs = decoder_graph_lean() and hidden.device.type == 'cuda'
        if lean_graphs:
            fused = fused_kernels()
        window_graph = (lean_graphs and decoder_graph_window(run_config=self.run_config) and isinstance(prefix, StaticPrefix)
                        and size == self._window_rows(run_config=getattr(self, 'run_config', None))
                        and context_length + size <= self._window_buckets()[-1])
        if window_graph:
            decoded, logits = self._window_graph_predict(hidden, rope, mask, prefix, context_length)
        looped = None
        if fused is None and not lean_graphs and not window_graph and hidden.device.type == 'cpu':
            decoded, logits = self(hidden, mask=mask, positions=position_ids,
                                  rope=rope, prefix=frozen)
            decoded, logits = decoded[0], logits[0]
            looped = decoded, logits
        for layer in (() if window_graph or looped is not None else self.layers):
            if lean_graphs:
                pending = self._window_layer_lean(layer, hidden, pending, mask, rope, frozen)
            elif hidden.device.type == 'cuda' and setting(self.run_config, 'engine', 'engine_settings.rows_graphs', None) == '1':
                hidden = self._window_layer(layer, hidden, mask, rope, frozen)
            elif fused is not None:
                pending = decoder_layer_lean(layer, hidden, rope, mask, frozen, None, fused,
                                             pending=pending, run_config=self.run_config)
            else:
                hidden = decode_layer(layer, hidden, mask, position_ids, rope, frozen)
        if profile and parameter.device.type == 'cuda':
            torch.cuda.synchronize(parameter.device)
        layers_ms = (time.perf_counter() - layers_started) * 1000.0
        norm_lm_started = time.perf_counter()
        if cuda_timing:
            layers_end.record()
            norm_begin.record()
        if window_graph or looped is not None:
            pass
        elif fused is not None:
            decoded, _ = fused.fast_add_rms_norm(*pending, self.norm.weight, self.norm.variance_epsilon)
        else:
            decoded = self.norm(hidden)
        if not (window_graph or looped is not None):
            decoded = decoded.reshape(-1, size, self.hidden_size)[0]
            logits = self.lm_head(decoded)
        if profile and parameter.device.type == 'cuda':
            torch.cuda.synchronize(parameter.device)
        norm_lm_ms = (time.perf_counter() - norm_lm_started) * 1000.0
        if cuda_timing:
            norm_end.record()
        if None == '1' and live_round:
            drafts = pending_block.draft_ids.reshape(-1).to(logits.device)
            reference_rejected = torch.cat((
                drafts.new_tensor([pending_block.correction_guess]), drafts))
            wire_rejected = torch.cat((drafts, drafts.new_tensor([-1])))
            print('W278_OFFER ' + json.dumps(offer_dump_record(
                logits, reference_rejected, wire_rejected,
                seq_id=sequence._jetspec_seq_id,
                round_id=sequence._jetspec_candidate_round)), flush=True)
        rejected = rejected_tokens(pending_block, after_miss=after_miss, run_config=self.run_config).to(logits.device)
        select_started = time.perf_counter()
        if cuda_timing:
            select_begin.record()
        offers = anchor_offers(logits, rejected, root=self.root, dataset=dataset,
                               coverage=coverage, grouped=grouped, cap=cap, run_config=self.run_config)
        if cuda_timing:
            select_end.record()
            total_end.record()
            torch.cuda.synchronize(logits.device)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            prefix_ms = prefix_begin.elapsed_time(prefix_end)
            select_ms = select_begin.elapsed_time(select_end)
            total_ms = total_begin.elapsed_time(total_end)
            stage_ms = dict(
                window_assembly=total_begin.elapsed_time(window_end),
                prefix=prefix_ms, window_mask=mask_begin.elapsed_time(mask_end),
                trained_layers=layers_begin.elapsed_time(layers_end),
                norm_lm_head=norm_begin.elapsed_time(norm_end),
                selection=select_ms)
        else:
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            prefix_ms = prefix_cpu_ms
            select_ms = (time.perf_counter() - select_started) * 1000.0
            total_ms = (time.perf_counter() - total_started) * 1000.0
            stage_ms = dict(window_assembly=window_assembly_ms, prefix=prefix_ms,
                            window_mask=window_mask_ms, trained_layers=layers_ms,
                            norm_lm_head=norm_lm_ms, selection=select_ms)
        if profile:
            print('W278_DECODER_PROFILE ' + json.dumps(dict(
                seq_id=getattr(sequence, '_jetspec_seq_id', None),
                round_id=getattr(sequence, '_jetspec_candidate_round', None),
                prefix_ms=prefix_ms, window_assembly_ms=window_assembly_ms,
                window_mask_ms=window_mask_ms, trained_layers_ms=layers_ms,
                norm_lm_head_ms=norm_lm_ms, select_ms=select_ms,
                total_wall_ms=(time.perf_counter() - profile_total_started) * 1000.0,
                fast=setting(self.run_config, 'engine', 'engine_settings.heads_fast', None) == '1')), flush=True)
        return DecoderPrediction(hidden=decoded, logits=logits, offers=offers,
                                 elapsed_ms=elapsed_ms,
                                 total_ms=total_ms, prefix_ms=prefix_ms,
                                 select_ms=select_ms, stage_ms=stage_ms)


class FrozenPrefix:
    def __init__(self, kv):
        self.kv = kv

    def update(self, key, value, layer_idx, cache_kwargs=None):
        prefix_key, prefix_value = self.kv[layer_idx]
        return torch.cat((prefix_key, key), dim=-2), torch.cat((prefix_value, value), dim=-2)


class FrozenBatchPrefix:
    """Reuse the gathered prefix allocation for both private window layers."""
    def __init__(self, kv, capacity):
        self.kv = kv
        self.capacity = capacity

    def update(self, key, value, layer_idx, cache_kwargs=None):
        stored_key, stored_value = self.kv[layer_idx]
        end = self.capacity + key.shape[-2]
        stored_key[..., self.capacity:end, :].copy_(key)
        stored_value[..., self.capacity:end, :].copy_(value)
        return stored_key[..., :end, :], stored_value[..., :end, :]


class SlotBatchPrefix:
    """Read pooled prefix K/V by slot in the N>1 captured decoder paths."""

    def __init__(self, pool, slots, lengths, capacity):
        self.pool, self.slots, self.lengths, self.capacity = pool, slots, lengths, capacity

    def update(self, key, value, layer_idx, cache_kwargs=None):
        self.latest = (key, value)
        return key, value


class PackedPrefix:
    """Independent prefixes followed by packed append rows for one layer call."""
    def __init__(self, kv):
        self.kv = kv
        self.appended = {}

    def update(self, key, value, layer_idx, cache_kwargs=None):
        self.appended[layer_idx] = (key, value)
        prefix_key, prefix_value = self.kv[layer_idx]
        return torch.cat((prefix_key, key), dim=-2), torch.cat((prefix_value, value), dim=-2)


class StaticPrefixPool:
    """Stable sequence slots in zero-initialized per-layer K/V tensors."""
    def __init__(self, capacity):
        self.capacity = capacity
        self.owners = [None] * 16
        self.free = list(range(15, -1, -1))
        self.kv = {}

    def acquire(self, prefix):
        if not self.free:
            first = len(self.owners)
            self.owners.extend([None] * first)
            self.free.extend(range(2 * first - 1, first - 1, -1))
            for index, pair in list(self.kv.items()):
                grown = tuple(torch.cat((tensor, torch.zeros_like(tensor)), dim=0)
                              for tensor in pair)
                self.kv[index] = grown
                self._rebind(index)
        slot = self.free.pop()
        self.owners[slot] = prefix
        return slot

    def release(self, slot):
        if self.owners[slot] is not None:
            self.owners[slot] = None
            self.free.append(slot)

    def adopt(self, prefix):
        old = prefix.kv
        prefix.kv = {}
        prefix.pool = self
        prefix.slot = self.acquire(prefix)
        for index, (old_key, old_value) in old.items():
            key, value = self.storage(prefix, index, old_key, old_value)
            key[..., :prefix.length, :].copy_(old_key[..., :prefix.length, :])
            value[..., :prefix.length, :].copy_(old_value[..., :prefix.length, :])

    def _rebind(self, index):
        pair = self.kv[index]
        for slot, owner in enumerate(self.owners):
            if owner is not None and index in owner.kv:
                owner.kv[index] = tuple(tensor[slot:slot + 1] for tensor in pair)

    def storage(self, prefix, index, key, value):
        if index not in self.kv:
            shape = (len(self.owners), *key.shape[1:-2], self.capacity, key.shape[-1])
            self.kv[index] = (
                torch.zeros(shape, device=key.device, dtype=key.dtype),
                torch.zeros(shape, device=value.device, dtype=value.dtype),
            )
        pair = self.kv[index]
        if pair[0].shape[1:-2] != key.shape[1:-2] or pair[0].shape[-1] != key.shape[-1]:
            raise ValueError('static decoder cache shape changed')
        prefix.kv[index] = tuple(tensor[prefix.slot:prefix.slot + 1] for tensor in pair)
        return prefix.kv[index]


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


class StaticPrefix:
    """Sequence-owned fixed-capacity K/V storage for target layers 34-35."""
    def __init__(self, capacity, pool=None):
        self.capacity = int(capacity)
        self.length = 0
        self.kv = {}
        self.pool = pool
        self.slot = pool.acquire(self) if pool is not None else None

    def _storage(self, layer_idx, key, value):
        if self.pool is not None:
            return self.pool.storage(self, layer_idx, key, value)
        if layer_idx not in self.kv:
            shape = (*key.shape[:-2], self.capacity, key.shape[-1])
            self.kv[layer_idx] = (
                torch.empty(shape, device=key.device, dtype=key.dtype),
                torch.empty(shape, device=value.device, dtype=value.dtype),
            )
        stored_key, stored_value = self.kv[layer_idx]
        if stored_key.shape[:-2] != key.shape[:-2] or stored_key.shape[-1] != key.shape[-1]:
            raise ValueError('static decoder cache shape changed')
        return stored_key, stored_value

    def update(self, key, value, layer_idx, cache_kwargs=None):
        cache_kwargs = {} if cache_kwargs is None else cache_kwargs
        positions = cache_kwargs.get('cache_position')
        if positions is None:
            positions = torch.arange(
                self.length, self.length + key.shape[-2], device=key.device)
        positions = torch.as_tensor(positions, device=key.device, dtype=torch.long).reshape(-1)
        end = self.length + key.shape[-2]
        if positions.numel() != key.shape[-2] or end > self.capacity:
            raise ValueError('decoder prefix exceeds its static cache capacity')
        stored_key, stored_value = self._storage(layer_idx, key, value)
        stored_key.index_copy_(-2, positions, key)
        stored_value.index_copy_(-2, positions, value)
        return stored_key[..., :end, :], stored_value[..., :end, :]


class StaticWindowPrefix:
    """Append one private window into the sequence prefix's spare storage."""
    def __init__(self, prefix, context_length):
        self.prefix = prefix
        self.context_length = int(context_length)

    def update(self, key, value, layer_idx, cache_kwargs=None):
        start = self.context_length
        end = start + key.shape[-2]
        if end > self.prefix.capacity:
            raise ValueError('decoder window exceeds its static cache capacity')
        stored_key, stored_value = self.prefix._storage(layer_idx, key, value)
        stored_key[..., start:end, :].copy_(key)
        stored_value[..., start:end, :].copy_(value)
        return stored_key[..., :end, :], stored_value[..., :end, :]
