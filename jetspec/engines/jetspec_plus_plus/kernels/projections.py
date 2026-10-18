"""Load-time packed projections for the opt-in J and C1 paths."""

import torch
from torch.nn import functional as F


class GroupedProjection:
    """Independent block projections in one GEMM, with persistent row buffers.

    A block-diagonal weight preserves distinct normalized inputs and their MX
    scale groups. It trades zero weight tiles for fewer launches. Each M is
    warmed eagerly, and returned block outputs are views consumed before reuse.
    """

    def __init__(self, weights):
        self.groups, self.n, self.k = weights.shape
        if self.k % 128 or self.n % 128:
            raise ValueError('grouped projections require 128-aligned weights')
        self.weight = weights.new_zeros(self.groups * self.n, self.groups * self.k)
        for index, weight in enumerate(weights):
            self.weight[index * self.n:(index + 1) * self.n,
                        index * self.k:(index + 1) * self.k].copy_(weight)
        self.buffers = {}
        if weights.is_cuda:
            from jetspec.engines.jetspec_plus_plus.models.target_linear import _w4a8_registry
            _w4a8_registry.pack(self.weight)

    def __call__(self, sources):
        groups, m, k = sources.shape
        if groups != self.groups or k != self.k or not 1 <= m <= 128:
            raise ValueError('grouped projection input shape or M is unsupported')
        if m not in self.buffers:
            if sources.is_cuda and torch.cuda.is_current_stream_capturing():
                raise RuntimeError('warm the grouped projection M before graph capture')
            self.buffers[m] = (sources.new_empty(m, groups, k),
                               sources.new_empty(m, groups * self.n))
        source, output = self.buffers[m]
        source.copy_(sources.permute(1, 0, 2))
        from jetspec.engines.jetspec_plus_plus.models.target_linear import w4a8_draft_linear
        result = w4a8_draft_linear(source.view(m, -1), self.weight, out=output)
        if result is None:
            torch.mm(source.view(m, -1), self.weight.t(), out=output)
        return output.view(m, groups, self.n).permute(1, 0, 2)


def prepare_grouped_kv(head):
    """Pack independent K/V blocks in each adapter stream after weights load."""
    if hasattr(head, '_gm_span_kv'):
        return
    plan = head._d1_lean_plan()
    weights = head._c1_span_kv[0]
    packs = {}
    first = 0
    while first < len(plan):
        stop = first + 1
        while stop < len(plan) and plan[stop][0] == plan[first][0] and plan[stop][7] == plan[first][7]:
            stop += 1
        if stop - first > 1:
            packs[first, stop] = GroupedProjection(weights[first:stop])
        first = stop
    head._gm_span_kv = packs


def _pack(module, names, weight):
    widths = [getattr(module, name).weight.shape[0] for name in names]
    biases = [getattr(module, name).bias for name in names]
    if any((item is None) != (biases[0] is None) for item in biases):
        raise ValueError("packed projections require matching bias policy")
    bias = torch.cat(biases).detach() if biases[0] is not None else None
    for name, view, bias_view in zip(names, weight.split(widths),
                                     bias.split(widths) if bias is not None else [None] * len(names)):
        linear = getattr(module, name)
        linear.weight.data = view
        if bias_view is not None:
            linear.bias.data = bias_view
    return weight, bias, widths


def prepare_qkv(attn):
    if hasattr(attn, "_fused_qkv"):
        return
    names = ("q_proj", "k_proj", "v_proj")
    weight = torch.cat([getattr(attn, name).weight for name in names]).detach()
    attn._fused_qkv = _pack(attn, names, weight)


def prepare_j0_gate_up(mlp):
    """Cache a draft-row gate/up pack without rebinding original parameters."""
    packed = getattr(mlp, '_j0_gate_up_packed', None)
    if packed is None:
        biases = (mlp.gate_proj.bias, mlp.up_proj.bias)
        if (biases[0] is None) != (biases[1] is None):
            raise ValueError('J0 gate/up requires matching bias policy')
        packed = (torch.cat((mlp.gate_proj.weight, mlp.up_proj.weight)).detach(),
                  torch.cat(biases).detach() if biases[0] is not None else None,
                  (mlp.gate_proj.out_features, mlp.up_proj.out_features))
        mlp._j0_gate_up_packed = packed
    return packed


def prepare_j0_lmhead_kn(lm_head):
    """Keep one immutable K-by-vocab copy for the opt-in draft projection."""
    packed = getattr(lm_head, '_j0_lmhead_kn', None)
    if packed is None:
        if lm_head.bias is not None:
            raise ValueError('J0 vocab packing requires a bias-free head')
        packed = lm_head.weight.t().contiguous().detach()
        lm_head._j0_lmhead_kn = packed
    return packed


def prepare_decoder_qkv(attn):
    """Pack the decoder's window QKV and prefix KV once after checkpoint load."""
    if hasattr(attn, "_decoder_qkv"):
        return
    qkv = (attn.q_proj, attn.k_proj, attn.v_proj)
    widths = tuple(linear.out_features for linear in qkv)
    biases = tuple(linear.bias for linear in qkv)
    if any((bias is None) != (biases[0] is None) for bias in biases):
        raise ValueError("decoder QKV requires matching bias policy")
    attn._decoder_qkv = (
        torch.cat([linear.weight for linear in qkv]).detach(),
        torch.cat(biases).detach() if biases[0] is not None else None,
        widths,
    )
    attn._decoder_prefix_kv = (
        torch.cat([linear.weight for linear in qkv[1:]]).detach(),
        torch.cat(biases[1:]).detach() if biases[0] is not None else None,
        widths[1:],
    )


def prepare_context_kv(head):
    """Pack every row-head layer's K/V from their common hidden input."""
    if hasattr(head, "_context_kv"):
        return head._context_kv
    attns = [layer.self_attn for layer in head.layers]
    widths = [(attn.k_proj.out_features, attn.v_proj.out_features) for attn in attns]
    if not widths or any(pair != widths[0] for pair in widths):
        raise ValueError("context K/V layers require matching projection widths")
    linears = [linear for attn in attns for linear in (attn.k_proj, attn.v_proj)]
    biases = [linear.bias for linear in linears]
    if any((bias is None) != (biases[0] is None) for bias in biases):
        raise ValueError("context K/V requires matching bias policy")
    head._context_kv = (
        torch.cat([linear.weight for linear in linears]).detach(),
        torch.cat(biases).detach() if biases[0] is not None else None,
        torch.stack([attn.k_norm.weight for attn in attns]).detach(),
        widths[0][0],
    )
    return head._context_kv


def prepare_c1_span_kv(head, *, reuse_append=False):
    """Pack C1 K/V weights by block for span-wise batched GEMM."""
    if hasattr(head, "_c1_span_kv"):
        return head._c1_span_kv
    layers = [layer for index, stream in enumerate(head.streams)
              for layer in (stream, *head.extra_stream_layers[index])]
    widths = [(layer.k_proj.out_features, layer.v_proj.out_features) for layer in layers]
    if not widths or any(pair != widths[0] for pair in widths):
        raise ValueError("C1 span K/V layers require matching projection widths")
    if any(layer.k_proj.bias is not None or layer.v_proj.bias is not None for layer in layers):
        raise ValueError("C1 span K/V requires bias-free projections")
    if reuse_append:
        weights = layers[0]._fused_kv_stack
        if weights.shape[0] != len(layers):
            raise ValueError("C1 append and span K/V block counts differ")
    else:
        weights = torch.stack([
            torch.cat((layer.k_proj.weight, layer.v_proj.weight)) for layer in layers
        ]).detach()
    head._c1_span_kv = (weights, widths[0])
    return head._c1_span_kv


def prepare_kv_group(layers):
    """The stacked C1 extension weight also backs each layer's K and V views."""
    if not layers:
        return None
    if hasattr(layers[0], "_fused_kv"):
        return layers[0]._fused_kv_stack
    weights = torch.stack([torch.cat((layer.k_proj.weight, layer.v_proj.weight))
                           for layer in layers]).detach()
    for layer, weight in zip(layers, weights):
        layer._fused_kv = _pack(layer, ("k_proj", "v_proj"), weight)
        layer._fused_kv_stack = weights
    return weights


def qkv(attn, source):
    packed = getattr(attn, "_fused_qkv", None)
    if packed is None:
        return attn.q_proj(source), attn.k_proj(source), attn.v_proj(source)
    weight, bias, widths = packed
    from jetspec.engines.jetspec_plus_plus.models.target_linear import packed_draft_linear
    return packed_draft_linear(source, weight, bias).split(widths, dim=-1)


def kv(layer, source):
    packed = getattr(layer, "_fused_kv", None)
    if packed is None:
        return layer.k_proj(source), layer.v_proj(source)
    weight, bias, widths = packed
    from jetspec.engines.jetspec_plus_plus.models.target_linear import packed_draft_linear
    return packed_draft_linear(source, weight, bias).split(widths, dim=-1)
