"""Served two-layer anchor decoder and exact flat anchor offers.

``decoder/manifest.json`` selects ``target_slice`` with ``target_files`` mapping
target-relative index/shard names to sha256 digests, or ``file`` with the pinned
``decoder/checkpoint-0.pt`` path and sha256. Bundles without it retain the jd01
``decoder/checkpoint-final-model.pt`` load.
"""
from __future__ import annotations

import json
import copy
import hashlib
import os
import time
import weakref
import numpy as np
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors import SafetensorError, safe_open
from torch import nn
from jetspec.engines.jetspec_plus_plus.draft.method.decoder import AnchorDecoder as MethodAnchorDecoder; from jetspec.engines.jetspec_plus_plus.draft.method.decoder import decode_layer
from transformers import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Attention,
    Qwen3DecoderLayer,
    Qwen3RotaryEmbedding,
    rotate_half,
)

import jetspec.engines.jetspec_plus_plus.draft.run_manifest as run_manifest
from jetspec.engines.jetspec_plus_plus.models.heads.lac import keep_device; from jetspec.engines.jetspec_plus_plus.models.heads.lac import lac_candidates
from jetspec.engines.jetspec_plus_plus.draft.rows import fused_kernels; from jetspec.engines.jetspec_plus_plus.draft.rows import rows_fused
from jetspec.engines.jetspec_plus_plus.models.heads.fast import dataset_entry
from jetspec.engines.jetspec_plus_plus.wire.chain.contract import FrontInputs; from jetspec.engines.jetspec_plus_plus.wire.chain.contract import Offset; from jetspec.engines.jetspec_plus_plus.wire.chain.contract import TokenBuffers




REUSED_DECODER_SHA256 = '41c85c871acd7bce2904a8f16d4f2beb0a4261044fc30bfd1aa605edbc1c90b7'


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _required_env(var_name: str, note: str) -> str:
    value = os.environ.get(var_name)
    if value:
        return value
    raise RuntimeError(f"Missing required env var {var_name}. {note}")


from jetspec.engines.jetspec_plus_plus.draft.method.decoder.window import window_mask


from jetspec.engines.jetspec_plus_plus.draft.method.decoder.window import chain_decoder_window_mask


from jetspec.engines.jetspec_plus_plus.draft.method.decoder.window import stage_chain_window_state


from jetspec.engines.jetspec_plus_plus.draft.method.decoder.window import chain_decoder_prefix_slots


from jetspec.engines.jetspec_plus_plus.draft.method.decoder.window import chain_decoder_prefix_mask




from jetspec.engines.jetspec_plus_plus.draft.method.decoder.window import decoder_window




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


def decoder_layer_lean(layer, hidden, rope, mask, cache, cache_position, fused, *, pending=None):
    """One eager Qwen3 decoder layer on the certified fused kernels.

    Same projections, cache update and sdpa call as the vendored layer; the
    norms, the rotary and the residual add come from ``fused`` (whose CPU
    and fp32 fallbacks spell the reference arithmetic, so the layer is bitwise
    the vendored one there).  ``pending=(update, residual)`` hands in the
    previous layer's unsummed residual pair: the add lands in this layer's
    input add+norm kernel.  Returns the same ``(mlp_output, residual)`` pair
    for the caller to sum or to hand on.
    """
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
    lean = (os.environ.get('JETSPEC_C1_LEAN') == '1'
            and (mask is not None or isinstance(cache, SlotBatchPrefix))
            and query.device.type == 'cuda' and query.dtype == torch.bfloat16)
    if isinstance(cache, SlotBatchPrefix):
        groups = attn.num_key_value_groups
        cache_key, cache_value = cache.pool.kv[attn.layer_idx]
        attended = attend_slots(
            query.reshape(query.shape[0], kv_heads, groups * query.shape[-2], head_dim),
            cache_key, cache_value, key, value, cache.slots, cache.lengths,
            cache.capacity, attn.scaling, inclusive=True)
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
    singleton = (os.environ.get('JETSPEC_DECODER_SINGLE_GATEUP_PACKED', '0') == '1'
                 and source.ndim == 3 and source.shape[0] == 1)
    if not singleton and (os.environ.get('JETSPEC_DECODER_GATEUP_PACKED', '0') != '1'
                         or source.ndim != 3 or source.shape[0] <= 1):
        return mlp.gate_proj(source), mlp.up_proj(source)
    _prepare_decoder_gate_up(mlp)
    weight, bias, width = mlp._decoder_gate_up_packed
    output = torch.nn.functional.linear(source.reshape(-1, source.shape[-1]), weight, bias)
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
    if (os.environ.get('JETSPEC_DECODER_QKV_FUSED', '0') == '1'
            and type(attn) is Qwen3Attention and hasattr(attn, '_decoder_prefix_kv')):
        _, key, value = _decoder_packed_project(attn, fused, source, rope, prefix=True)
        return key, value
    return fused.fast_kv_postprocess(attn.k_proj(source), attn.v_proj(source),
                                     attn.k_norm.weight, cos, sin,
                                     attn.k_norm.variance_epsilon, kv_heads, 1, head_dim)


def decoder_graph_lean():
    """JETSPEC_DECODER_GRAPH_LEAN=1: the graphed regime's decoder parts on the
    certified kernels (``decoder_layer_lean`` with JETSPEC_C1_LEAN=1, split around
    its attention so each side is one shape-only graph)."""
    return os.environ.get('JETSPEC_DECODER_GRAPH_LEAN') == '1'


def decoder_graph_window():
    """JETSPEC_DECODER_GRAPH_WINDOW=1 (with JETSPEC_DECODER_GRAPH_LEAN=1): the predict
    window (both trained layers, their attention, the norm and the lm_head) as one graph
    per key bucket, attending over a decoder-owned copy of the sequence's K/V."""
    return os.environ.get('JETSPEC_DECODER_GRAPH_WINDOW') == '1'




def check_decoder_graph_lean_regime():
    if decoder_graph_window() and not decoder_graph_lean():
        raise ValueError('JETSPEC_DECODER_GRAPH_WINDOW=1 requires JETSPEC_DECODER_GRAPH_LEAN=1')
    if not decoder_graph_lean():
        return
    if (os.environ.get('JETSPEC_ROWS_GRAPHS') != '1' or os.environ.get('JETSPEC_HEADS_FAST') != '1'
            or os.environ.get('JETSPEC_C1_LEAN') != '1'):
        raise ValueError('JETSPEC_DECODER_GRAPH_LEAN=1 is the graphed rows regime on the lean '
                         'kernels: it requires JETSPEC_ROWS_GRAPHS=1, JETSPEC_HEADS_FAST=1 and '
                         'JETSPEC_C1_LEAN=1')


def decoder_lean_project(layer, fused, hidden, pending, rope):
    """``decoder_layer_lean`` up to its attention: the input (add+)norm, the q and
    k/v projections with their norm + rotary.  Returns (query, key, value, hidden)."""
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
    if (os.environ.get('JETSPEC_DECODER_QKV_FUSED', '0') == '1'
            and type(attn) is Qwen3Attention and hasattr(attn, '_decoder_qkv')):
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
    weight, bias, widths = (attn._decoder_prefix_kv if prefix else attn._decoder_qkv)
    parts = torch.nn.functional.linear(source, weight, bias).split(widths, dim=-1)
    cos, sin = rope
    head_dim = attn.head_dim
    kv_heads = attn.k_proj.out_features // head_dim
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
    if (length is not None and query.is_cuda
            and (os.environ.get('JETSPEC_DECODER_ATTN_LEN', '0') == '1'
                 or os.environ.get('JETSPEC_DECODER_ATTN_LEN_LONG', '0') == '1')):
        from jetspec.engines.jetspec_plus_plus.kernels.chain.len_attention import attend as attend_len
        return attend_len(query, key, value, length, layer.self_attn.scaling,
                          mode='decoder', count=count,
                          long=os.environ.get('JETSPEC_DECODER_ATTN_LEN_LONG', '0') == '1').transpose(1, 2).contiguous()
    from transformers.integrations.sdpa_attention import repeat_kv
    attn = layer.self_attn
    groups = attn.num_key_value_groups
    if (os.environ.get('JETSPEC_DECODER_GQA_VIEW', '0') == '1'
            and query.is_cuda and query.dtype == torch.bfloat16
            and query.shape[0] == 1 and groups > 1):
        return decoder_grouped_attention(fused, query, key, value, mask,
                                          attn.scaling).transpose(1, 2).contiguous()
    key, value = repeat_kv(key, groups), repeat_kv(value, groups)
    return fused.efficient_attention(query, key, value, mask[:, :, :, :key.shape[-2]],
                                     attn.scaling).transpose(1, 2).contiguous()


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


def rejected_tokens(pending_block, *, after_miss=False):
    """Return the drafter token at every decoder k offset (0 through 14)."""
    drafts = pending_block.draft_ids.reshape(-1)
    from jetspec.engines.jetspec_plus_plus.wire.protocol import draft_width
    if drafts.numel() != draft_width():
        raise ValueError('decoder pending block requires JETSPEC_DRAFTS (14) draft ids')
    # Device fills, not host tensors: a host list would drain the stream.
    if os.environ.get('JETSPEC_DECODER_FRAME') == 'wire':
        rejected = torch.cat((drafts, drafts.new_full((1,), -1)))
    else:
        rejected = torch.cat((drafts.new_full((1,), int(pending_block.correction_guess)), drafts))
    if after_miss and os.environ.get('JETSPEC_KSET_AFTER_MISS') == 'zero':
        rejected = rejected.clone()
        rejected[0] = -1
    return rejected




from jetspec.engines.jetspec_plus_plus.draft.method.decoder.window import AnchorOffers


_GRAPHED_SELECT = None


class DeviceAnchorOffers:
    """JETSPEC_ROWS_ASSEMBLY=1: the eager top-k offers left on the device.

    ``device`` holds (ids, fp64 probabilities, pack) as _select_body returns them; the
    assembly stage reads them there and brings home only what the round needs.  Every
    AnchorOffers field is still served: the first read of a host field makes the same one-copy
    trip (topk_select._home) the HOME=one path makes, or takes the stage's host values."""

    def __init__(self, ids, p64, pack, probabilities, cap, width):
        self.device = (ids, p64, pack)
        self._probabilities, self._cap, self._width = probabilities, cap, width
        self._host = None
        self._summary = None


    def _materialize(self):
        if self._host is None:
            from jetspec.engines.jetspec_plus_plus.models.heads.topk_select import _home
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
    def lac_fallbacks(self):
        return self._summary[2] if self._summary else self._materialize()[4]


def _topk_offers(logits, rejected, q_hats, cap):
    """JETSPEC_ANCHOR_SELECT=topk: the exact top-k selection (heads/topk_select.py), eager, or
    one CUDA graph per shape under the graphed rows regime; offers come home in one copy."""
    global _GRAPHED_SELECT
    from jetspec.engines.jetspec_plus_plus.models.heads.topk_select import GraphedSelect; from jetspec.engines.jetspec_plus_plus.models.heads.topk_select import fast_offers_topk; from jetspec.engines.jetspec_plus_plus.models.heads.topk_select import fast_offers_topk_home
    if (os.environ.get('JETSPEC_ANCHOR_SELECT_EAGER_FUSED', '0') == '1'
            and os.environ.get('JETSPEC_ANCHOR_SELECT_GRAPH', '0') != '1'
            and os.environ.get('JETSPEC_ROWS_GRAPHS', '0') != '1'
            and isinstance(q_hats, (tuple, int, float))):
        from jetspec.engines.jetspec_plus_plus.models.heads.topk_select import EagerSelect
        eager = EagerSelect()
        if (os.environ.get('JETSPEC_ANCHOR_SELECT_HOME') == 'one'
                and os.environ.get('JETSPEC_ROWS_ASSEMBLY', '0') == '1'):
            body, cap, width = eager.device(logits, rejected, q_hats, cap)
            return DeviceAnchorOffers(*body, cap, width)
        ids, probabilities, sizes, top1, fallbacks = eager.home(logits, rejected, q_hats, cap)
        return AnchorOffers(dict(enumerate(ids)), dict(enumerate(probabilities)),
                            [int(size) for size in sizes], list(top1), fallbacks)
    if (os.environ.get('JETSPEC_ANCHOR_SELECT_HOME') == 'one'
            and not (logits.device.type == 'cuda' and os.environ.get('JETSPEC_ROWS_GRAPHS') == '1')
            and isinstance(q_hats, (tuple, int, float))):
        if (os.environ.get('JETSPEC_ROWS_ASSEMBLY', '0') == '1'
                and os.environ.get('JETSPEC_ANCHOR_SELECT_GRAPH', '0') == '1'
                and torch.as_tensor(logits).device.type == 'cuda'):
            # The same selection body replayed as one CUDA graph (the graphed regime's
            # GraphedSelect), left on the device for the assembly stage
            if _GRAPHED_SELECT is None:
                _GRAPHED_SELECT = GraphedSelect()
            body, cap, width = _GRAPHED_SELECT.device(logits, rejected, q_hats, cap)
            return DeviceAnchorOffers(*body, cap, width)
        if os.environ.get('JETSPEC_ROWS_ASSEMBLY', '0') == '1':
            # The same eager selection body, left on the device for the assembly stage
            from jetspec.engines.jetspec_plus_plus.models.heads.topk_select import SLACK; from jetspec.engines.jetspec_plus_plus.models.heads.topk_select import _check; from jetspec.engines.jetspec_plus_plus.models.heads.topk_select import _q_pair; from jetspec.engines.jetspec_plus_plus.models.heads.topk_select import _select_body
            logits = torch.as_tensor(logits).float()
            rejected = torch.as_tensor(rejected, device=logits.device, dtype=torch.long)
            _check(logits, rejected, cap)
            cap = int(cap)
            width = min(logits.shape[1], cap + 1 + int(SLACK))
            q_a, q_b = _q_pair(q_hats)
            return DeviceAnchorOffers(*_select_body(logits, rejected, q_a, q_b, cap, width), cap, width)
        # the eager selection, its offers home in one device-to-host copy
        ids, probabilities, sizes, top1, fallbacks = fast_offers_topk_home(
            logits, rejected, q_hats, cap)
        return AnchorOffers(dict(enumerate(ids)), dict(enumerate(probabilities)),
                            [int(size) for size in sizes], list(top1), fallbacks)
    if logits.device.type == 'cuda' and os.environ.get('JETSPEC_ROWS_GRAPHS') == '1':
        if _GRAPHED_SELECT is None:
            _GRAPHED_SELECT = GraphedSelect()
        if (os.environ.get('JETSPEC_ANCHOR_SELECT_HOME') == 'one'
                and os.environ.get('JETSPEC_ROWS_ASSEMBLY', '0') == '1'
                and isinstance(q_hats, (tuple, int, float))):
            # The graphed regime's same selection graph, its offers left on the device
            # for the assembly stage, as the eager branch above does
            body, cap, width = _GRAPHED_SELECT.device(logits, rejected, q_hats, cap)
            return DeviceAnchorOffers(*body, cap, width)
        if os.environ.get('JETSPEC_ANCHOR_SELECT_HOME') == 'one':
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


def _threshold_dataset(dataset, coverage):
    """Serve an uncalibrated benchmark from the gsm8k anchor entry only where that is behaviour-identical.

    The e6v2 bundle carries anchor thresholds for gsm8k and math500 only. Under E04_ROWRULE=joint the anchor
    q_hat at coverage 0.99 is exactly 1.0 for every dataset (the joint rule needs all 32 offers), so reading the
    gsm8k entry cannot change served behaviour. Any other combination still refuses below.
    """
    if dataset in ('gsm8k', 'math500') or str(coverage) != '0.99' or os.environ.get('E04_ROWRULE') != 'joint':
        return dataset
    if dataset not in _THRESHOLD_FALLBACK_LOGGED:
        _THRESHOLD_FALLBACK_LOGGED.add(dataset)
        print(f"DECODER_THRESHOLD_FALLBACK dataset={dataset} -> gsm8k coverage=0.99 rowrule=joint", flush=True)
    return 'gsm8k'




def anchor_offers(logits, rejected, *, root, dataset, coverage, grouped, cap):
    """Remove d_rej, then apply the pinned dataset/group LAC threshold."""
    if os.environ.get('E04_ROWRULE') and os.environ.get('E04_JOINT_OFFERS') == '1':
        # Joint offers behind E04_JOINT_OFFERS=1 (unset = the per-row offers below)
        offers = e04_joint_offers(logits, rejected, root=root, dataset=dataset, coverage=coverage,
                                  grouped=grouped, cap=cap)
        if offers is not None:
            return offers
    logits = keep_device(logits).float()
    rejected = torch.as_tensor(rejected, device=logits.device, dtype=torch.long)
    if logits.ndim != 2 or rejected.shape != logits.shape[:1]:
        raise ValueError('decoder logits and rejected ids must have shapes [15,V] and [15]')
    dataset = _threshold_dataset(dataset, coverage)
    if not dataset or str(coverage) not in ('0.94', '0.99'):
        raise ValueError('decoder selection requires a known dataset and coverage 0.94 or 0.99')
    profile = os.environ.get('W278_HEAD_PROFILE') == '1'
    if profile and logits.device.type == 'cuda':
        torch.cuda.synchronize(logits.device)
    profile_started = time.perf_counter()
    fast_path = os.environ.get('JETSPEC_HEADS_FAST') == '1'
    budget = os.environ.get('JETSPEC_ROW_SELECT') == 'budget'
    if budget:
        pass  # Budget offers do not read calibrated coverage thresholds.
    elif fast_path:
        from jetspec.engines.jetspec_plus_plus.models.heads.fast import fast_lac_candidates; from jetspec.engines.jetspec_plus_plus.models.heads.fast import read_thresholds
        thresholds = read_thresholds(root)['anchor_predictor']
    else:
        run_manifest.record_file_once('heads/thresholds', Path(root) / 'thresholds.json')
        thresholds = json.loads((Path(root) / 'thresholds.json').read_text())['anchor_predictor']
    if profile and logits.device.type == 'cuda':
        torch.cuda.synchronize(logits.device)
    threshold_ms = (time.perf_counter() - profile_started) * 1000.0
    if os.environ.get('JETSPEC_ANCHOR_SELECT') == 'topk' and (budget or fast_path):
        if budget:
            q_hats = 1.0
        elif grouped:
            values = dataset_entry(thresholds['drej_top1_grouped'], dataset)[str(coverage)]
            q_hats = (float(values['drej-top1']['value']),
                      float(values['not-drej-top1']['value']))
        else:
            q_hats = float(dataset_entry(thresholds['pooled'], dataset)[str(coverage)]['value'])
        lac_started = time.perf_counter()
        result = _topk_offers(logits, rejected, q_hats, cap)
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
        from jetspec.engines.jetspec_plus_plus.models.heads.fast import fast_lac_candidates
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


from jetspec.engines.jetspec_plus_plus.draft.method.decoder.window import FrozenPrefix


from jetspec.engines.jetspec_plus_plus.draft.method.decoder.window import FrozenBatchPrefix


from jetspec.engines.jetspec_plus_plus.draft.method.decoder.window import SlotBatchPrefix


from jetspec.engines.jetspec_plus_plus.draft.method.decoder.window import PackedPrefix


from jetspec.engines.jetspec_plus_plus.draft.method.decoder.window import StaticPrefixPool


from jetspec.engines.jetspec_plus_plus.draft.method.decoder.window import IncrementalPrefix


from jetspec.engines.jetspec_plus_plus.draft.method.decoder.window import StaticPrefix


from jetspec.engines.jetspec_plus_plus.draft.method.decoder.window import StaticWindowPrefix


from jetspec.engines.jetspec_plus_plus.draft.method.decoder.window import DecoderPrediction


def _load_target_prefix_layers(root, config, *, device, dtype, indices, key_prefix,
                               file_pins=None):
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
    layers = nn.ModuleList([Qwen3DecoderLayer(config, target_index)
                            for target_index in indices])
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
    run_manifest.record_file('decoder_prefix/index', root / 'model.safetensors.index.json')
    for shard in sorted(by_shard):
        run_manifest.record_file(f'decoder_prefix/{shard}', root / shard)
    return layers.to(device=device, dtype=dtype).eval().requires_grad_(False)




def _decoder_indices(config, tap_ids):
    if not tap_ids:
        raise ValueError('target-slice decoder requires adapter target_layer_ids')
    deepest = max(int(index) for index in tap_ids)
    indices = tuple(range(deepest + 1, int(config.num_hidden_layers)))
    if len(indices) != 2:
        raise ValueError(f'target-slice decoder requires two layers after tap {deepest}; got {indices}')
    return indices


def _decoder_config(target_path, drafter):
    root = Path(target_path)
    raw = json.loads((root / 'config.json').read_text())
    raw = raw.get('text_config', raw)
    tap_ids = getattr(getattr(drafter.adapter, 'head', None), 'target_layer_ids', None)
    indices = _decoder_indices(Qwen3Config(num_hidden_layers=raw['num_hidden_layers']), tap_ids)
    layer_types = raw.get('layer_types') or ['full_attention'] * raw['num_hidden_layers']
    model_type = raw.get('model_type')
    for index in indices:
        layer_type = layer_types[index]
        if model_type != 'qwen3' or layer_type != 'full_attention':
            raise ValueError(f'unsupported decoder layer {index}: type={layer_type} model_type={model_type}')
    config = Qwen3Config(**raw)
    config._attn_implementation = 'sdpa'
    index_path = root / 'model.safetensors.index.json'
    key_prefix = None
    if index_path.is_file():
        weight_map = json.loads(index_path.read_text())['weight_map']
        for prefix in ('model.', 'model.language_model.'):
            if f'{prefix}layers.{indices[0]}.input_layernorm.weight' in weight_map:
                key_prefix = prefix
                break
    return config, indices, key_prefix


def _target_slice_state(root, config, indices, key_prefix, file_pins):
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
        run_manifest.record_file(f'heads/decoder_target/{name}', path)
    verify(index_name)
    if 'config.json' in file_pins:
        verify('config.json')
    index = json.loads((root / index_name).read_text())['weight_map']
    layers = nn.ModuleList([Qwen3DecoderLayer(config, index) for index in indices])
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
    if len(layers.state_dict()) != 22:
        raise ValueError(f'target-slice decoder expected 22 layer tensors; got {len(layers.state_dict())}')
    layers.load_state_dict({key: state[key] for key in layers.state_dict()}, strict=True, assign=True)
    return layers, state


def _check_reused_fallback(layers, root, expected_sha256):
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
    run_manifest.record_file('heads/decoder_fallback', fallback)
    print(f'REUSED_DECODER_LOAD_EQUALITY=PASS fallback=decoder/checkpoint-0.pt bitwise_tensors={len(reference)}', flush=True)


class AnchorDecoder(MethodAnchorDecoder):
    """Two trained target layers over one private adapter window per round."""
    def __init__(self, *, layers, prefix_layers, norm, lm_head, rotary, root, hidden_size):
        super().__init__(layers=layers, prefix_layers=prefix_layers, norm=norm,
                         lm_head=lm_head, rotary=rotary, root=root, hidden_size=hidden_size)
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

    def _rope(self, hidden, position_ids):
        """Rotary rows gathered from one table on the fused path; each position's
        cos/sin is a single product, so the gather is bitwise the rotary call.
        The graphed lean decoder (JETSPEC_DECODER_GRAPH_LEAN) gathers too."""
        if not (rows_fused(hidden) or (decoder_graph_lean() and hidden.device.type == 'cuda')):
            return self.rotary(hidden, position_ids)
        capacity = int(os.environ.get('JETSPEC_DECODER_CACHE_CAPACITY', '800')) + 32
        table = self._rope_table
        if table is None or table[0].shape[0] < capacity or table[0].device != hidden.device:
            positions = torch.arange(capacity, device=hidden.device)[None]
            cos, sin = self.rotary(hidden, positions)
            table = self._rope_table = (cos[0], sin[0])
        flat = position_ids.reshape(-1)
        return table[0].index_select(0, flat)[None], table[1].index_select(0, flat)[None]



    @classmethod
    def load(cls, root, drafter):
        from jetspec.engines.jetspec_plus_plus.draft.tau_lab import check_regime as e04_lab_check_regime
        e04_lab_check_regime()  # E04_LAB_*: a no-op while every lab name is unset
        if os.environ.get('JETSPEC_HYBRID_PAIR', '0') == '1':
            from jetspec.engines.jetspec_plus_plus.models.heads.hybrid_decoder import HybridAnchorDecoder
            target_model_path = _required_env('JETSPEC_TARGET_MODEL_PATH',
                                              'Set it to the target model directory.')
            manifest_path = Path(root) / 'decoder/manifest.json'
            manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else None
            return HybridAnchorDecoder.load(root, drafter, target_model_path, manifest)
        from jetspec.engines.jetspec_plus_plus.draft.rows import check_rows_fused_regime
        check_rows_fused_regime()
        check_decoder_graph_lean_regime()
        root = Path(root)
        dtype = drafter.dtype
        device = drafter.device
        manifest_path = root / 'decoder/manifest.json'
        manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else None
        source = None if manifest is None else manifest.get('source')
        if manifest is not None:
            run_manifest.record_file('heads/decoder_manifest', manifest_path)
            if source not in ('target_slice', 'file'):
                raise ValueError(f'unknown decoder source: {source}')
        target_model_path = _required_env('JETSPEC_TARGET_MODEL_PATH',
                                          'Set it to the target model directory.')
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
                                                manifest.get('target_files', {}))
            _check_reused_fallback(layers, root,
                                   manifest.get('fallback_sha256', REUSED_DECODER_SHA256))
            for name, tensor in state.items():
                if tensor.dtype != dtype:
                    raise ValueError(f'target-slice decoder dtype mismatch: {name} target={tensor.dtype} runner={dtype}')
            prefix_layers = layers
            norm = copy.deepcopy(drafter.head.norm)
            norm.load_state_dict({'weight': state['norm.weight']}, strict=True)
            lm_head = copy.deepcopy(drafter.lm_head)
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
            layers = nn.ModuleList([Qwen3DecoderLayer(config, index) for index in indices])
            layers.load_state_dict(state, strict=True)
            run_manifest.record_file('heads/decoder', path)
            if draft_prefix:
                prefix_layers = layers
            else:
                prefix_layers = _load_target_prefix_layers(
                    target_model_path, config, device=device, dtype=dtype,
                    indices=indices, key_prefix=key_prefix,
                    file_pins=manifest.get('target_files') if source == 'file' else None)
            if source is None:
                # The trained CE and KL decoders learned against the target's final norm,
                # not the drafter head's norm.
                norm = copy.deepcopy(drafter.head.norm)
                norm.load_state_dict({'weight': _load_target_norm_weight(
                    target_model_path, key_prefix)}, strict=True)
            else:
                norm = drafter.head.norm
            norm = norm.to(device=device, dtype=dtype).eval().requires_grad_(False)
            lm_head = drafter.lm_head
            layers = layers.to(device=device, dtype=dtype).eval().requires_grad_(False)
        # Construct RoPE on the runner device.  The serving process installs a
        # torch default-device context; constructing on CPU and moving later can
        # leave Qwen3's non-persistent ``original_inv_freq`` alias on CPU.
        rotary = Qwen3RotaryEmbedding(config, device=device).to(device=device)
        if any(buffer.device != device for buffer in rotary.buffers()):
            raise RuntimeError('decoder rotary buffers are not on the runner device')
        if os.environ.get('JETSPEC_HEADS_FAST') == '1':
            from jetspec.engines.jetspec_plus_plus.models.heads.fast import read_thresholds
            read_thresholds(root)
        decoder = cls(layers=layers, prefix_layers=prefix_layers,
                      norm=norm, lm_head=lm_head,
                      rotary=rotary, root=root, hidden_size=config.hidden_size).eval()
        if os.environ.get('JETSPEC_DECODER_LMHEAD_KN', '0') == '1':
            if source != 'target_slice' or lm_head.bias is not None:
                raise ValueError('JETSPEC_DECODER_LMHEAD_KN requires a bias-free target_slice lm_head')
            decoder._lm_head_kn = lm_head.weight.detach().T.contiguous()
        if os.environ.get('JETSPEC_DECODER_QKV_FUSED', '0') == '1':
            from jetspec.engines.jetspec_plus_plus.draft.projections import prepare_decoder_qkv
            for layer in (*layers, *prefix_layers):
                if type(layer.self_attn) is Qwen3Attention:
                    prepare_decoder_qkv(layer.self_attn)
        if os.environ.get('JETSPEC_DECODER_SINGLE_GATEUP_PACKED', '0') == '1':
            for layer in (*layers, *prefix_layers):
                _prepare_decoder_gate_up(layer.mlp)
        if (device.type == 'cuda' and os.environ.get('JETSPEC_HEADS_FAST') == '1'
                and os.environ.get('JETSPEC_ROWS_GRAPHS') == '1'):
            decoder.warm_graphs()
        elif rows_fused(next(layers.parameters())):
            decoder.warm_kernels()
        return decoder


    @torch.inference_mode()
    def warm_graphs(self):
        """Capture every projection shape before any sequence is admitted.

        Prefix appends use 1..16 positions; the decoder window uses draft_width() + 1 rows.
        Attention and cache updates stay outside these shape-only graphs.
        """
        parameter = next(self.layers.parameters())
        if decoder_graph_lean():
            self._warm_lean_graphs(parameter)
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
        hidden = parameter.new_zeros(1, self._window_rows(), self.hidden_size)
        rope = self.rotary(hidden, torch.arange(self._window_rows(), device=parameter.device)[None])
        for layer in self.layers:
            hidden = self._window_layer(layer, hidden, None, rope, None)
        self._graphs_warmed = True

    def _warm_lean_graphs(self, parameter):
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
        layer_parts(self.layers, self._window_rows(), False)
        if decoder_graph_window():
            hidden = parameter.new_zeros(1, self._window_rows(), self.hidden_size)
            for bucket in self._window_buckets():
                self._window_graph(hidden, bucket)

    @staticmethod
    def _window_rows():
        """The current decoder slots [c, d1..d_width], following every true tap."""
        from jetspec.engines.jetspec_plus_plus.wire.protocol import draft_width
        return draft_width() + 1

    def _chain_lm_head(self, decoded):
        if self._lm_head_kn is not None:
            return torch.matmul(decoded, self._lm_head_kn)
        return self.lm_head(decoded)

    def _check_prefix_merge(self, state, front):
        if self.prefix_layers is not self.layers or len(self.layers) != 2:
            raise ValueError('JETSPEC_DECODER_PREFIX_MERGE requires two shared target_slice layers')
        if front.decoder_prefix_kv is not state['kv']:
            raise ValueError('decoder prefix and window K/V must share storage')
        if any(type(layer.self_attn) is not Qwen3Attention or
               not hasattr(layer.self_attn, '_decoder_qkv') for layer in self.layers):
            raise ValueError('JETSPEC_DECODER_PREFIX_MERGE requires fused decoder QKV')

    def _merged_chain_layers(self, prefix_hidden, rope, mask, slots, state, front,
                             length, count, use_len_attention, fused):
        """Share the two layers' projection reads across prefix and window rows."""
        self._check_prefix_merge(state, front)
        lanes = prefix_hidden.shape[1]
        combined_rope = tuple(torch.cat((p, w), dim=1)
                              for p, w in zip(rope, (state['cos'], state['sin'])))
        source = torch.cat((prefix_hidden, state['hidden']), dim=1)
        pending = None
        for index, layer in enumerate(self.layers):
            stored_key, stored_value = state['kv'][layer.self_attn.layer_idx]
            direct = os.environ.get('JETSPEC_DECODER_SLOT_WRITE', '0') == '1'
            if direct:
                from jetspec.engines.jetspec_plus_plus.models.heads.decoder_cache import project_into_cache
                query, hidden = project_into_cache(
                    layer, fused, source if pending is None else None, pending, combined_rope,
                    (stored_key, stored_value), slots, state['slots'])
            else:
                query, key, value, hidden = decoder_lean_project(
                    layer, fused, source if pending is None else None, pending, combined_rope)
                stored_key.index_copy_(2, slots, key[..., :lanes, :])
                stored_value.index_copy_(2, slots, value[..., :lanes, :])
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
        capacity = int(os.environ.get('JETSPEC_DECODER_CACHE_CAPACITY', '800'))
        if (capacity > 800 and
                os.environ.get('JETSPEC_CHAIN_DECODER_KEY_BUCKET', '0') != '1'):
            return [capacity]  # one graph for the selected run capacity
        return [*range(128, capacity, 128), capacity]

    def _window_state(self, like, bucket):
        """Keep each opted-in decoder key's static inputs and K/V distinct."""
        bucketed = os.environ.get('JETSPEC_CHAIN_DECODER_KEY_BUCKET', '0') == '1'
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
        if os.environ.get('JETSPEC_DECODER_GLUE_FUSED', '0') == '1':
            prefix_lanes = like.shape[1]
            state.update(prefix_positions=torch.empty((1, prefix_lanes), device=like.device,
                                                      dtype=torch.long),
                         prefix_slots=torch.empty(prefix_lanes, device=like.device,
                                                  dtype=torch.long),
                         prefix_masks={},
                         prefix_cos=like.new_empty(1, prefix_lanes, head_dim),
                         prefix_sin=like.new_empty(1, prefix_lanes, head_dim))
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
            if self._graphs_warmed and os.environ.get('JETSPEC_FRONT_STATIC', '0') != '1':
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
        if os.environ.get('JETSPEC_KEYLEN_BUCKET', '0') == '1':
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
        if os.environ.get('JETSPEC_FRONT_STATIC', '0') != '1':
            raise ValueError('decoder chain body requires JETSPEC_FRONT_STATIC=1')
        use_len_attention = (rp.is_cuda and
                             ((bucket.head.P <= 2304 and
                               os.environ.get('JETSPEC_DECODER_ATTN_LEN', '0') == '1') or
                              (2304 < bucket.head.P <= 3328 and
                               os.environ.get('JETSPEC_DECODER_ATTN_LEN_LONG', '0') == '1')))
        del bucket
        lanes = front.lanes
        length, count = rp[Offset.DEC_LEN], rp[Offset.C1_COUNT]
        prefix_hidden = front.decoder_prefix_taps[None]
        state = (self._window_graph_states[front.decoder_bucket]
                 if os.environ.get('JETSPEC_CHAIN_DECODER_KEY_BUCKET', '0') == '1'
                 else self._window_graph_state)
        fused_glue = os.environ.get('JETSPEC_DECODER_GLUE_FUSED', '0') == '1'
        if fused_glue:
            from jetspec.engines.jetspec_plus_plus.models.heads.decoder_glue import prefix_glue
            table = self._rope_table
            if table is None or not all(value.is_contiguous() for value in table):
                raise RuntimeError('decoder glue requires a prewarmed contiguous rotary table')
            prefix_glue(rp, tokens.pending_draft_ids, *table,
                        state['prefix_masks'][front.decoder_bucket], state['prefix_positions'],
                        state['prefix_slots'], state['prefix_cos'], state['prefix_sin'],
                        front.rejected_ids, length_offset=Offset.DEC_LEN,
                        count_offset=Offset.C1_COUNT,
                        correction_offset=Offset.CORRECTION, miss_offset=Offset.AFTER_MISS,
                        wire=os.environ.get('JETSPEC_DECODER_FRAME') == 'wire',
                        zero_miss=os.environ.get('JETSPEC_KSET_AFTER_MISS') == 'zero')
            positions = state['prefix_positions']
            slots = state['prefix_slots']
            rope = state['prefix_cos'], state['prefix_sin']
            mask = state['prefix_masks'][front.decoder_bucket]
        else:
            front.rejected_ids.copy_(chain_rejected_tokens(rp, tokens))
            lane = torch.arange(lanes, device=rp.device, dtype=torch.long)
            positions = (length - count + lane)[None]
            slots = chain_decoder_prefix_slots(length, count, front.decoder_bucket, lanes)
            rope = self._rope(prefix_hidden, positions)
            mask = (prefix_hidden.new_empty(1, 1, lanes, front.decoder_bucket)
                    if use_len_attention
                    else chain_decoder_prefix_mask(
                        length, count, front.decoder_bucket, lanes, prefix_hidden.dtype))
        if os.environ.get('JETSPEC_DECODER_PREFIX_MERGE', '0') == '1':
            glue_level = int(os.environ.get('JETSPEC_CHAIN_SEGA_GLUE', '0'))
            from jetspec.engines.jetspec_plus_plus.kernels.chain.glue2 import available; from jetspec.engines.jetspec_plus_plus.kernels.chain.glue2 import window_mask_rope
            fused_window = (glue_level == 2 and rp.is_cuda and available()
                            and self._rope_table is not None
                            and all(table.is_contiguous() for table in self._rope_table))
            if fused_glue:
                from jetspec.engines.jetspec_plus_plus.models.heads.decoder_glue import window_glue
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
            from jetspec.engines.jetspec_plus_plus.models.heads.topk_select import GraphedSelect
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

        glue_level = int(os.environ.get('JETSPEC_CHAIN_SEGA_GLUE', '0'))
        from jetspec.engines.jetspec_plus_plus.kernels.chain.glue2 import available; from jetspec.engines.jetspec_plus_plus.kernels.chain.glue2 import window_mask_rope
        fused_window = (glue_level == 2 and rp.is_cuda and available()
                        and self._rope_table is not None
                        and all(table.is_contiguous() for table in self._rope_table))
        if fused_glue:
            from jetspec.engines.jetspec_plus_plus.models.heads.decoder_glue import window_glue
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
        from jetspec.engines.jetspec_plus_plus.models.heads.topk_select import GraphedSelect
        ids, p64, pack = GraphedSelect.chain_body(
            logits, front.rejected_ids, front.selection_thresholds, front.select_cap)
        return logits, ids, p64, pack

    def refresh_window_state(self, prefix_kv, bucket, *, rows, lanes, like, glue=None):
        """Static buffers of ``refresh_window_logits`` for one chain key (JETSPEC_REFRESH_SWAP).

        ``prefix_kv`` is the key's decoder prefix K/V (``FrontInputs.decoder_prefix_kv``) and
        ``bucket`` its key bucket.  The read's private K/V holds the bucket's columns, then the
        read's own tokens in a fixed order: the 4 true prefix positions, the ``lanes`` staged
        tail slots and ``rows`` x (lanes - 1) block slots, padded to a multiple of 16 columns.
        Allocated before capture; the bucket's prefix K/V is copied in, never written."""
        if self.prefix_layers is not self.layers:
            raise ValueError('JETSPEC_REFRESH_SWAP requires the target_slice decoder '
                             '(one set of shared prefix and window layers)')
        import jetspec.engines.jetspec_plus_plus.models.heads.refresh_glue as refresh_glue
        if refresh_glue.enabled() if glue is None else glue:
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
        if os.environ.get('JETSPEC_REFRESH_SWAP_LEAN', '0') == '1' and like.is_cuda and like.dtype == torch.bfloat16:
            from jetspec.engines.jetspec_plus_plus.draft.chain.refresh_swap import allocate_compact
            state['compact'] = allocate_compact(rows, slots, self.lm_head.weight.shape[0], device)
        return state

    def refresh_window_compact(self, state, *, ids, **kwargs):
        """The opt-in refresh read returns compact swap inputs, never BF16 vocab logits."""
        return self.refresh_window_logits(state, _compact_ids=ids, **kwargs)

    def refresh_window_logits(self, state, *, length, true_taps, tail, tail_counts, block,
                              slot_counts, _compact_ids=None):
        """The anchor decoder over each row's refreshed block, in the refresh context, with fixed
        shapes and no host read: the graphed port of ``tau_lab.refresh_decoder_logits``.


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
        bucket, width, tokens = state['bucket'], state['width'], state['tokens']
        rows, lanes, slots = state['rows'], state['lanes'], state['slots']
        hidden_size = self.hidden_size
        if state.get('glue'):
            from jetspec.engines.jetspec_plus_plus.models.heads.refresh_glue import layers as refresh_glue_layers
            hidden, decoded = refresh_glue_layers(
                self, state, length=length, true_taps=true_taps, tail=tail,
                tail_counts=tail_counts, block=block, slot_counts=slot_counts)
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
                if os.environ.get('JETSPEC_REFRESH_DECODER_LOOP', '0') == '1':
                    decoded = self._refresh_loop(fused, hidden, rope, mask, state)
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
        decoded = decoded[0, 4 + lanes:]
        if hidden.is_cuda and compact_cuda:
            from jetspec.engines.jetspec_plus_plus.draft.chain.refresh_swap import compact_head
            # This kernel consumes native [N,K] tiles. Seg A's [K,N] cuBLAS layout
            # is already prepared, but makes this tiled load strided on SM80.
            return compact_head(decoded.contiguous(), self.lm_head.weight,
                                _compact_ids, state['compact'])
        if _compact_ids is not None:
            # FP32 validation and CPU retain the reference head/softmax arithmetic.
            return compact_logits(self._chain_lm_head(decoded).reshape(rows, slots, -1), _compact_ids)
        return self._chain_lm_head(decoded).reshape(rows, slots, -1)


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
    def prewarm_prefix_appends(self, make_sequence, counts):
        """Run ``_prefix`` once per append size in ``counts`` on throwaway sequences from
        ``make_sequence(length)``, so the Triton entries each size compiles (its row count
        is a constexpr: ~0.6 s per new size) are built before any request is admitted."""
        capacity = int(os.environ.get('JETSPEC_DECODER_CACHE_CAPACITY', '800'))
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



    def drop_sequence(self, sequence):
        """Drop replay/live state explicitly; live sequences also have a finalizer."""
        key = id(sequence)
        prefix = self._prefixes.pop(key, None)
        if isinstance(prefix, StaticPrefix) and prefix.pool is not None:
            prefix.pool.release(prefix.slot)
        finalizer = self._prefix_finalizers.pop(key, None)
        if finalizer is not None:
            finalizer.detach()


    def _new_prefix(self, sequence):
        key = id(sequence)
        fast_path = os.environ.get('JETSPEC_HEADS_FAST') == '1'
        if fast_path:
            capacity = int(os.environ.get('JETSPEC_DECODER_CACHE_CAPACITY', '800'))
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

    def _prefix(self, sequence, desired_length):
        key = id(sequence)
        prefix = self._prefixes.get(key)
        if prefix is None or prefix.length > desired_length:
            self.drop_sequence(sequence)
            prefix = self._new_prefix(sequence)
        tap33 = sequence.taps[:, 4, :].to(device=next(self.layers.parameters()).device,
                                          dtype=next(self.layers.parameters()).dtype)
        if isinstance(prefix, StaticPrefix) and prefix.length < desired_length:
            start = prefix.length
            hidden = tap33[start:desired_length][None]
            if (os.environ.get('JETSPEC_DECODER_TABLES') == '1' and rows_fused(hidden)
                    and not decoder_graph_lean()
                    and not (hidden.shape[1] <= 16 and os.environ.get('JETSPEC_ROWS_GRAPHS') == '1')
                    and self._loop_prefix(fused_kernels(), hidden,
                                          self._rope_range(hidden, start, desired_length),
                                          self._causal_rows(hidden, start, desired_length),
                                          prefix, start)):
                prefix.length = desired_length
                return prefix
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
            fused = fused_kernels() if rows_fused(hidden) or lean else None
            pending = None
            lean_graphs = lean and hidden.shape[1] <= 16
            graphed = (hidden.device.type == 'cuda' and hidden.shape[1] <= 16 and
                       os.environ.get('JETSPEC_ROWS_GRAPHS') == '1')
            if (not lean and not graphed and fused is not None
                    and self._loop_prefix(fused, hidden, rope, mask, prefix, start)):
                prefix.length = desired_length
                return prefix
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
                               os.environ.get('JETSPEC_ROWS_GRAPHS') == '1')
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
                      os.environ.get('JETSPEC_ROWS_GRAPHS') == '1'):
                    hidden = self._window_layer(layer, hidden, mask, rope, prefix)
                elif fused is not None:
                    pending = decoder_layer_lean(layer, hidden, rope, mask, prefix, positions, fused,
                                                 pending=pending)
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
    def predict(self, sequence, pending_block, *, dataset, coverage, grouped, cap,
                after_miss=False):
        staged = sequence.staged.predicted_hidden
        if staged is None:
            return None
        from jetspec.engines.jetspec_plus_plus.wire.protocol import draft_width
        # 15 current slots (c, d1..d14), 16 with 15 drafts (k = 0..15)
        count = draft_width() + 1
        size = count
        if int(staged.shape[1]) < count:
            raise RuntimeError(f'decoder requires the staged {count}-position adapter block')
        parameter = next(self.layers.parameters())
        live_round = hasattr(sequence, '_jetspec_seq_id') and \
            hasattr(sequence, '_jetspec_candidate_round')
        profile = os.environ.get('W278_HEAD_PROFILE') == '1' and live_round
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
        current = staged[0, :count, -self.hidden_size:].to(parameter)
        hidden = decoder_window(current).to(
            device=parameter.device, dtype=parameter.dtype)
        anchor = int(sequence.taps.shape[0])
        context_length = anchor
        if context_length < 0:
            return None
        record_root = os.environ.get('W278_RECORD_INPUTS_DIR')
        record_limit = int(os.environ.get('W278_RECORD_INPUTS_LIMIT', '240'))
        if record_root and live_round and self._recorded_inputs < record_limit:
            destination = Path(record_root) / 'decoder'
            destination.mkdir(parents=True, exist_ok=True)
            torch.save(dict(
                tap33=sequence.taps[:, 4, :].detach().cpu(),
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
        if os.environ.get('JETSPEC_DECODER_TABLES') == '1' and rows_fused(hidden):
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
        fused = fused_kernels() if rows_fused(hidden) else None
        pending = None
        lean_graphs = decoder_graph_lean() and hidden.device.type == 'cuda'
        if lean_graphs:
            fused = fused_kernels()
        window_graph = (lean_graphs and decoder_graph_window() and isinstance(prefix, StaticPrefix)
                        and size == self._window_rows()
                        and context_length + size <= self._window_buckets()[-1])
        if window_graph:
            decoded, logits = self._window_graph_predict(hidden, rope, mask, prefix, context_length)
        looped = None
        if not (lean_graphs or window_graph) and fused is not None:
            looped = self._loop_window(fused, hidden, rope, mask, frozen)
            if looped is not None:
                decoded, logits = looped
        for layer in (() if window_graph or looped is not None else self.layers):
            if lean_graphs:
                pending = self._window_layer_lean(layer, hidden, pending, mask, rope, frozen)
            elif hidden.device.type == 'cuda' and os.environ.get('JETSPEC_ROWS_GRAPHS') == '1':
                hidden = self._window_layer(layer, hidden, mask, rope, frozen)
            elif fused is not None:
                pending = decoder_layer_lean(layer, hidden, rope, mask, frozen, None, fused,
                                             pending=pending)
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
        if os.environ.get('W278_OFFER_DUMP') == '1' and live_round:
            drafts = pending_block.draft_ids.reshape(-1).to(logits.device)
            reference_rejected = torch.cat((
                drafts.new_tensor([pending_block.correction_guess]), drafts))
            wire_rejected = torch.cat((drafts, drafts.new_tensor([-1])))
            print('W278_OFFER ' + json.dumps(offer_dump_record(
                logits, reference_rejected, wire_rejected,
                seq_id=sequence._jetspec_seq_id,
                round_id=sequence._jetspec_candidate_round)), flush=True)
        rejected = rejected_tokens(pending_block, after_miss=after_miss).to(logits.device)
        select_started = time.perf_counter()
        if cuda_timing:
            select_begin.record()
        offers = anchor_offers(logits, rejected, root=self.root, dataset=dataset,
                               coverage=coverage, grouped=grouped, cap=cap)
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
                fast=os.environ.get('JETSPEC_HEADS_FAST') == '1')), flush=True)
        return DecoderPrediction(hidden=decoded, logits=logits, offers=offers,
                                 elapsed_ms=elapsed_ms,
                                 total_ms=total_ms, prefix_ms=prefix_ms,
                                 select_ms=select_ms, stage_ms=stage_ms)
