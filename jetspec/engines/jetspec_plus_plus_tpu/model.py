"""JAX models for JetSpec++ on one TPU chip: Qwen3 target, drafter, adapter, anchor decoder.

Every module mirrors the torch reference in ``jetspec/engines/jetspec_plus_plus/draft``:

- target: Qwen3 with a persistent KV cache; verify returns the five taps (layers 1, 9, 17, 25, 33).
- drafter (the checkpoint's ``dflash.py``): five layers whose keys are [context | refreshed block | noise block].
- adapter (``method/adapter.py``, C1 tap-aligned H2H): twelve stream layers, window causal.
- anchor decoder (``reference/models/decoder.py``): target layers 34 and 35, final norm and LM head.

The reference recomputes every context projection on each pass. Here every projection that
depends only on committed positions is cached once per position: the drafter's context K/V,
the adapter's per-layer prefix K/V and the decoder's layer-34/35 prefix K/V (which equal the
target's own KV cache for those layers, since the decoder is that slice of the target).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

BF16 = jnp.bfloat16
F32 = jnp.float32
NEG = -1e30


# --------------------------------------------------------------------------- config

@dataclass(frozen=True)
class Dims:
    hidden: int = 4096
    heads: int = 32
    kv_heads: int = 8
    head_dim: int = 128
    vocab: int = 151936
    target_layers: int = 36
    eps: float = 1e-6
    theta: float = 1_000_000.0
    taps: tuple = (1, 9, 17, 25, 33)
    draft_layers: int = 5
    stream_depths: tuple = (1, 1, 1, 4, 5)
    block: int = 16

    @property
    def tap_width(self):
        return len(self.taps) * self.hidden

    @property
    def adapter_layers(self):
        return sum(self.stream_depths)

    @property
    def decoder_layers(self):
        return tuple(range(max(self.taps) + 1, self.target_layers))


# --------------------------------------------------------------------------- primitives

def rms(x, w, eps):
    """HF Qwen3RMSNorm: fp32 normalisation, cast back, then the weight multiply."""
    xf = x.astype(F32)
    xf = xf * jax.lax.rsqrt(jnp.mean(xf * xf, axis=-1, keepdims=True) + eps)
    return w * xf.astype(x.dtype)


def mm(x, w):
    """bf16 x [.., in] @ w [in, out] with fp32 accumulation, rounded to bf16 like torch.

    ``w`` may be a weight-only int8 dict {'q': int8 [in, out], 's': fp32 [out]} (draft side only).
    """
    return dot32(x, w).astype(x.dtype)


def dot32(x, w):
    if isinstance(w, dict):
        return jnp.dot(x, w['q'].astype(x.dtype), preferred_element_type=F32) * w['s']
    return jnp.dot(x, w, preferred_element_type=F32)


def quantize(w):
    """Symmetric per-output-channel int8 for a [.., in, out] weight."""
    wf = w.astype(F32)
    s = jnp.maximum(jnp.max(jnp.abs(wf), axis=-2, keepdims=True), 1e-8) / 127.0
    return dict(q=jnp.clip(jnp.round(wf / s), -127, 127).astype(jnp.int8), s=jnp.squeeze(s, -2))


MATRICES = ('qkv', 'q', 'kv', 'o', 'gate_up', 'down', 'fc', 'lm_head')


def quantize_draft(weights, dims):
    """Weight-only int8 for everything that only proposes tokens: the drafter, the adapter, and
    draft-side copies of the anchor decoder layers and the LM head. The target used for
    verification keeps its bf16 weights, so the output is unchanged; only acceptance can move."""
    qz = jax.jit(quantize)

    def per_layer(stacked):
        parts = [qz(stacked[i]) for i in range(stacked.shape[0])]
        return dict(q=jnp.stack([p['q'] for p in parts]), s=jnp.stack([p['s'] for p in parts]))

    def tree(w):
        out = {}
        for name, value in w.items():
            if name in MATRICES:
                out[name] = per_layer(value) if value.ndim == 3 else qz(value)
            elif isinstance(value, dict):
                out[name] = tree(value)
            else:
                out[name] = value
        return out

    tw = weights['target']
    lo, hi = dims.decoder_layers[0], dims.decoder_layers[-1] + 1
    decoder = dict(layers=tree({k: v[lo:hi] for k, v in tw['layers'].items()}), norm=tw['norm'])
    head = qz(tw['lm_head'])
    drafter = dict(tree(weights['drafter']), lm_head=head)
    adapter = tree(weights['adapter'])
    decoder['lm_head'] = head
    return dict(target=tw, drafter=drafter, adapter=adapter, decoder=decoder)


def rope_tables(positions, head_dim, theta):
    inv = 1.0 / (theta ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))
    freqs = positions.astype(F32)[:, None] * jnp.asarray(inv)[None, :]
    emb = jnp.concatenate([freqs, freqs], axis=-1)
    return jnp.cos(emb).astype(BF16), jnp.sin(emb).astype(BF16)


def rope(x, cos, sin):
    """x [T, H, D]; cos/sin [T, D]."""
    half = x.shape[-1] // 2
    rot = jnp.concatenate([-x[..., half:], x[..., :half]], axis=-1)
    return x * cos[:, None, :] + rot * sin[:, None, :]


def attend(q, segments, scale):
    """GQA attention of q [T, Hq, D] over several key segments with one joint softmax.

    ``segments`` is a list of (k [Hk, S, D], v [Hk, S, D], allowed [T, S] bool). Keys are
    head-major, the physical layout the TPU attention wants, so caches never change layout.
    """
    t, hq, d = q.shape
    hk = segments[0][0].shape[0]
    qg = q.reshape(t, hk, hq // hk, d)
    scores = []
    for k, _, allowed in segments:
        s = jnp.einsum('tkgd,ksd->kgts', qg, k, preferred_element_type=F32) * scale
        scores.append(jnp.where(allowed[None, None], s, NEG))
    s = jnp.concatenate(scores, axis=-1)
    p = jax.nn.softmax(s, axis=-1).astype(q.dtype)
    out, start = 0.0, 0
    for k, v, _ in segments:
        width = k.shape[1]
        out = out + jnp.einsum('kgts,ksd->tkgd', p[..., start:start + width], v,
                               preferred_element_type=F32)
        start += width
    return out.astype(q.dtype).reshape(t, hq * d)


def hm(x):
    """[T, H, D] -> head-major [H, T, D]."""
    return jnp.swapaxes(x, 0, 1)


def mlp(x, w_gate_up, w_down):
    gu = mm(x, w_gate_up)
    inter = gu.shape[-1] // 2
    return mm(jax.nn.silu(gu[..., :inter]) * gu[..., inter:], w_down)


# --------------------------------------------------------------------------- weights

def _t(a):
    return np.ascontiguousarray(a.T)


def load_target(root, dims: Dims):
    """Stacked, pre-transposed Qwen3 weights: [L, in, out] with fused qkv and gate|up."""
    root = Path(root)
    index = json.loads((root / 'model.safetensors.index.json').read_text())['weight_map']
    from safetensors.numpy import load_file
    by_file = {}
    for name, f in index.items():
        by_file.setdefault(f, []).append(name)
    raw = {}
    for f in sorted(by_file):
        raw.update(load_file(str(root / f)))
    n = dims.target_layers
    p = 'model.layers.{}.'

    def stack(fn):
        return jax.device_put(np.stack([fn(p.format(i)) for i in range(n)]))

    w = dict(
        in_norm=stack(lambda q: raw[q + 'input_layernorm.weight']),
        post_norm=stack(lambda q: raw[q + 'post_attention_layernorm.weight']),
        q_norm=stack(lambda q: raw[q + 'self_attn.q_norm.weight']),
        k_norm=stack(lambda q: raw[q + 'self_attn.k_norm.weight']),
        qkv=stack(lambda q: _t(np.concatenate([raw[q + 'self_attn.q_proj.weight'],
                                                raw[q + 'self_attn.k_proj.weight'],
                                                raw[q + 'self_attn.v_proj.weight']], 0))),
        o=stack(lambda q: _t(raw[q + 'self_attn.o_proj.weight'])),
        gate_up=stack(lambda q: _t(np.concatenate([raw[q + 'mlp.gate_proj.weight'],
                                                    raw[q + 'mlp.up_proj.weight']], 0))),
        down=stack(lambda q: _t(raw[q + 'mlp.down_proj.weight'])),
    )
    top = dict(embed=jax.device_put(raw['model.embed_tokens.weight']),
               norm=jax.device_put(raw['model.norm.weight']),
               lm_head=jax.device_put(_t(raw['lm_head.weight'])))
    raw.clear()
    return dict(layers=w, **top)


def load_drafter(root, dims: Dims):
    from safetensors.numpy import load_file
    raw = load_file(str(Path(root) / 'model.safetensors'))
    n = dims.draft_layers
    p = 'layers.{}.'

    def stack(fn):
        return jax.device_put(np.stack([fn(p.format(i)) for i in range(n)]))

    w = dict(
        in_norm=stack(lambda q: raw[q + 'input_layernorm.weight']),
        post_norm=stack(lambda q: raw[q + 'post_attention_layernorm.weight']),
        q_norm=stack(lambda q: raw[q + 'self_attn.q_norm.weight']),
        k_norm=stack(lambda q: raw[q + 'self_attn.k_norm.weight']),
        q=stack(lambda q: _t(raw[q + 'self_attn.q_proj.weight'])),
        kv=stack(lambda q: _t(np.concatenate([raw[q + 'self_attn.k_proj.weight'],
                                               raw[q + 'self_attn.v_proj.weight']], 0))),
        o=stack(lambda q: _t(raw[q + 'self_attn.o_proj.weight'])),
        gate_up=stack(lambda q: _t(np.concatenate([raw[q + 'mlp.gate_proj.weight'],
                                                    raw[q + 'mlp.up_proj.weight']], 0))),
        down=stack(lambda q: _t(raw[q + 'mlp.down_proj.weight'])),
    )
    return dict(layers=w, fc=jax.device_put(_t(raw['fc.weight'])),
                hidden_norm=jax.device_put(raw['hidden_norm.weight']),
                norm=jax.device_put(raw['norm.weight']))


def adapter_layer_names(dims: Dims):
    """Stream-ordered layer names and, per layer, its tap index and whether it opens a stream."""
    names = []
    for tap, depth in enumerate(dims.stream_depths):
        names.append((f'streams.{tap}.', tap, True))
        names.extend((f'extra_stream_layers.{tap}.{d}.', tap, False) for d in range(depth - 1))
    return names


def load_adapter(root, dims: Dims):
    from safetensors.numpy import load_file
    raw = load_file(str(Path(root) / 'adapter' / 'head.safetensors'))
    names = adapter_layer_names(dims)

    def stack(fn):
        return jax.device_put(np.stack([fn(q) for q, _, _ in names]))

    return dict(
        in_norm=stack(lambda q: raw[q + 'input_norm.weight']),
        kv_norm=stack(lambda q: raw[q + 'kv_norm.weight']),
        post_norm=stack(lambda q: raw[q + 'post_attention_norm.weight']),
        q_norm=stack(lambda q: raw[q + 'q_norm.weight']),
        k_norm=stack(lambda q: raw[q + 'k_norm.weight']),
        q=stack(lambda q: _t(raw[q + 'q_proj.weight'])),
        kv=stack(lambda q: _t(np.concatenate([raw[q + 'k_proj.weight'], raw[q + 'v_proj.weight']], 0))),
        o=stack(lambda q: _t(raw[q + 'o_proj.weight'])),
        gate_up=stack(lambda q: _t(np.concatenate([raw[q + 'mlp.gate_proj.weight'],
                                                    raw[q + 'mlp.up_proj.weight']], 0))),
        down=stack(lambda q: _t(raw[q + 'mlp.down_proj.weight'])),
    )


# --------------------------------------------------------------------------- caches

def new_cache(dims: Dims, capacity: int, *, adapter=True, draft=True):
    shape = lambda n: (n, dims.kv_heads, capacity, dims.head_dim)
    z = lambda n: jnp.zeros(shape(n), BF16)
    cache = dict(tk=z(dims.target_layers), tv=z(dims.target_layers))
    if draft:
        cache.update(dk=z(dims.draft_layers), dv=z(dims.draft_layers))
    if adapter:
        cache.update(ak=z(dims.adapter_layers), av=z(dims.adapter_layers))
    return cache


# --------------------------------------------------------------------------- target

def target_forward(tw, cache, ids, pos, dims: Dims, span: int, *, logit_row=None):
    """Run ids [T] at positions pos..pos+T-1 against the cache, writing their K/V.

    Attention reads cache rows [0, span); span is a static bucket >= pos + T.
    Returns (logits fp32 [T, V], or [1, V] for logit_row, taps [T, 5H], cache).
    """
    t = ids.shape[0]
    positions = pos + jnp.arange(t)
    cos, sin = rope_tables(positions, dims.head_dim, dims.theta)
    # Cached rows [0, pos) are read-only here; the new rows attend to each other causally.
    prefix_allowed = jnp.broadcast_to(jnp.arange(span)[None, :] < pos, (t, span))
    causal = jnp.arange(t)[None, :] <= jnp.arange(t)[:, None]
    x = dims_embed(tw['embed'], ids)
    qd, kd = dims.heads * dims.head_dim, dims.kv_heads * dims.head_dim
    scale = dims.head_dim ** -0.5
    tk_all, tv_all = cache['tk'], cache['tv']

    def layer(x, xs):
        w, li = xs
        h = rms(x, w['in_norm'], dims.eps)
        qkv = mm(h, w['qkv'])
        q = qkv[:, :qd].reshape(t, dims.heads, dims.head_dim)
        k = qkv[:, qd:qd + kd].reshape(t, dims.kv_heads, dims.head_dim)
        v = qkv[:, qd + kd:].reshape(t, dims.kv_heads, dims.head_dim)
        q = rope(rms(q, w['q_norm'], dims.eps), cos, sin)
        k = rope(rms(k, w['k_norm'], dims.eps), cos, sin)
        k, v = hm(k), hm(v)
        ks = jax.lax.dynamic_slice(tk_all, (li, 0, 0, 0), (1, dims.kv_heads, span, dims.head_dim))[0]
        vs = jax.lax.dynamic_slice(tv_all, (li, 0, 0, 0), (1, dims.kv_heads, span, dims.head_dim))[0]
        a = attend(q, [(ks, vs, prefix_allowed), (k, v, causal)], scale)
        x = x + mm(a, w['o'])
        x = x + mlp(rms(x, w['post_norm'], dims.eps), w['gate_up'], w['down'])
        return x, (x, k, v)

    x, (outs, new_k, new_v) = jax.lax.scan(layer, x, (tw['layers'], jnp.arange(dims.target_layers)))
    tk = jax.lax.dynamic_update_slice(tk_all, new_k, (0, 0, pos, 0))
    tv = jax.lax.dynamic_update_slice(tv_all, new_v, (0, 0, pos, 0))
    taps = jnp.concatenate([outs[i] for i in dims.taps], axis=-1)
    if logit_row is not None:
        x = jax.lax.dynamic_slice_in_dim(x, logit_row, 1)
    x = rms(x, tw['norm'], dims.eps)
    logits = jnp.dot(x, tw['lm_head'], preferred_element_type=F32)
    return logits, taps, dict(cache, tk=tk, tv=tv)


def dims_embed(table, ids):
    return jnp.take(table, ids, axis=0)


# --------------------------------------------------------------------------- context caches

def write_context(dw, aw, cache, taps, pos, dims: Dims):
    """Cache the drafter's context K/V and the adapter's prefix K/V for taps [T, 5H] at pos."""
    t = taps.shape[0]
    cos, sin = rope_tables(pos + jnp.arange(t), dims.head_dim, dims.theta)
    kd = dims.kv_heads * dims.head_dim
    shape = (t, dims.kv_heads, dims.head_dim)
    ctx = rms(mm(taps, dw['fc']), dw['hidden_norm'], dims.eps)
    dk, dv = [], []
    for i in range(dims.draft_layers):
        w = jax.tree.map(lambda a: a[i], dw['layers'])
        kv = mm(ctx, w['kv'])
        dk.append(hm(rope(rms(kv[:, :kd].reshape(shape), w['k_norm'], dims.eps), cos, sin)))
        dv.append(hm(kv[:, kd:].reshape(shape)))
    put = lambda buf, rows: jax.lax.dynamic_update_slice(buf, jnp.stack(rows), (0, 0, pos, 0))
    cache = dict(cache, dk=put(cache['dk'], dk), dv=put(cache['dv'], dv))
    if aw is None:                      # DFlash: drafter context only
        return cache
    ak, av = [], []
    for i, (_, tap, _) in enumerate(adapter_layer_names(dims)):
        w = jax.tree.map(lambda a: a[i], aw)
        src = rms(taps[:, tap * dims.hidden:(tap + 1) * dims.hidden], w['kv_norm'], dims.eps)
        kv = mm(src, w['kv'])
        ak.append(hm(rope(rms(kv[:, :kd].reshape(shape), w['k_norm'], dims.eps), cos, sin)))
        av.append(hm(kv[:, kd:].reshape(shape)))
    return dict(cache, ak=put(cache['ak'], ak), av=put(cache['av'], av))


# --------------------------------------------------------------------------- drafter

def drafter_pass(dw, tw, cache, ids, length, dims: Dims, span: int, *, extra=None, extra_khat=None,
                 causal_block=True):
    """One uncached-equivalent drafter pass over block ids [B] at positions length..length+B-1.

    ``extra`` [B, 5H] is the refreshed block (``method/refresh.inject``): extra entry j is
    visible to block query i iff j < i and j <= extra_khat. Returns bf16 logits [B, V].
    ``causal_block=False`` is DFlash: the block attends to itself bidirectionally.
    """
    b = ids.shape[0]
    positions = length + jnp.arange(b)
    cos, sin = rope_tables(positions, dims.head_dim, dims.theta)
    ctx_allowed = jnp.broadcast_to(jnp.arange(span)[None, :] < length, (b, span))
    slots = jnp.arange(b)
    causal = (slots[None, :] <= slots[:, None]) if causal_block else jnp.ones((b, b), bool)
    kd = dims.kv_heads * dims.head_dim
    shape = (b, dims.kv_heads, dims.head_dim)
    scale = dims.head_dim ** -0.5
    if extra is not None:
        ectx = rms(mm(extra, dw['fc']), dw['hidden_norm'], dims.eps)
        e_allowed = (slots[None, :] < slots[:, None]) & (slots[None, :] <= extra_khat)
    x = dims_embed(tw['embed'], ids)
    for i in range(dims.draft_layers):
        w = jax.tree.map(lambda a: a[i], dw['layers'])
        h = rms(x, w['in_norm'], dims.eps)
        q = rope(rms(mm(h, w['q']).reshape(b, dims.heads, dims.head_dim), w['q_norm'], dims.eps), cos, sin)
        kv = mm(h, w['kv'])
        nk = hm(rope(rms(kv[:, :kd].reshape(shape), w['k_norm'], dims.eps), cos, sin))
        nv = hm(kv[:, kd:].reshape(shape))
        segments = [(cache['dk'][i, :, :span], cache['dv'][i, :, :span], ctx_allowed)]
        if extra is not None:
            ekv = mm(ectx, w['kv'])
            ek = hm(rope(rms(ekv[:, :kd].reshape(shape), w['k_norm'], dims.eps), cos, sin))
            segments.append((ek, hm(ekv[:, kd:].reshape(shape)), e_allowed))
        segments.append((nk, nv, causal))
        x = x + mm(attend(q, segments, scale), w['o'])
        x = x + mlp(rms(x, w['post_norm'], dims.eps), w['gate_up'], w['down'])
    x = rms(x, dw['norm'], dims.eps)
    return dot32(x, dw.get('lm_head', tw['lm_head'])).astype(BF16)


# --------------------------------------------------------------------------- adapter

def adapter_predict(aw, tw, cache, tokens, length, dims: Dims, span: int):
    """C1 head: predicted taps [N, 5H] for tokens [N] at positions length..length+N-1."""
    n = tokens.shape[0]
    positions = length + jnp.arange(n)
    cos, sin = rope_tables(positions, dims.head_dim, dims.theta)
    prefix_allowed = jnp.broadcast_to(jnp.arange(span)[None, :] < length, (n, span))
    slots = jnp.arange(n)
    window_allowed = slots[None, :] < slots[:, None]
    kd = dims.kv_heads * dims.head_dim
    shape = (n, dims.kv_heads, dims.head_dim)
    scale = dims.head_dim ** -0.5
    u = dims_embed(tw['embed'], tokens)
    outputs, state = [], None
    names = adapter_layer_names(dims)
    for i, (_, tap, opens) in enumerate(names):
        w = jax.tree.map(lambda a: a[i], aw)
        if opens:
            if state is not None:
                outputs.append(state)
                u = state
            x = u
        else:
            x = state
        h = rms(x, w['in_norm'], dims.eps)
        q = rope(rms(mm(h, w['q']).reshape(n, dims.heads, dims.head_dim), w['q_norm'], dims.eps), cos, sin)
        kv = mm(rms(u, w['kv_norm'], dims.eps), w['kv'])
        wk = hm(rope(rms(kv[:, :kd].reshape(shape), w['k_norm'], dims.eps), cos, sin))
        wv = hm(kv[:, kd:].reshape(shape))
        a = attend(q, [(cache['ak'][i, :, :span], cache['av'][i, :, :span], prefix_allowed),
                       (wk, wv, window_allowed)], scale)
        s = x + mm(a, w['o'])
        state = s + mlp(rms(s, w['post_norm'], dims.eps), w['gate_up'], w['down'])
    outputs.append(state)
    return jnp.concatenate(outputs, axis=-1)


# --------------------------------------------------------------------------- anchor decoder

def decoder_logits(tw, cache, hidden, length, dims: Dims, span: int, dec=None):
    """Target layers 34-35 + norm + LM head over isolated causal block slots.

    ``dec`` optionally holds draft-side (int8) copies of those layers and the LM head.

    hidden [W, H] are predicted deepest-tap states at positions length..length+W-1; the
    committed prefix K/V are the target cache rows of those two layers. Returns fp32 logits
    of the bf16 LM head, [W, V]; output j predicts slot j + 1.
    """
    w_count = hidden.shape[0]
    positions = length + jnp.arange(w_count)
    cos, sin = rope_tables(positions, dims.head_dim, dims.theta)
    prefix_allowed = jnp.broadcast_to(jnp.arange(span)[None, :] < length, (w_count, span))
    slots = jnp.arange(w_count)
    causal = slots[None, :] <= slots[:, None]
    qd, kd = dims.heads * dims.head_dim, dims.kv_heads * dims.head_dim
    scale = dims.head_dim ** -0.5
    x = hidden
    for local, li in enumerate(dims.decoder_layers):
        w = (jax.tree.map(lambda a: a[li], tw['layers']) if dec is None
             else jax.tree.map(lambda a: a[local], dec['layers']))
        h = rms(x, w['in_norm'], dims.eps)
        qkv = mm(h, w['qkv'])
        q = rope(rms(qkv[:, :qd].reshape(w_count, dims.heads, dims.head_dim), w['q_norm'], dims.eps), cos, sin)
        k = rope(rms(qkv[:, qd:qd + kd].reshape(w_count, dims.kv_heads, dims.head_dim), w['k_norm'], dims.eps),
                 cos, sin)
        v = qkv[:, qd + kd:].reshape(w_count, dims.kv_heads, dims.head_dim)
        a = attend(q, [(cache['tk'][li, :, :span], cache['tv'][li, :, :span], prefix_allowed),
                       (hm(k), hm(v), causal)], scale)
        x = x + mm(a, w['o'])
        x = x + mlp(rms(x, w['post_norm'], dims.eps), w['gate_up'], w['down'])
    x = rms(x, tw['norm'], dims.eps)
    return dot32(x, tw['lm_head'] if dec is None else dec['lm_head']).astype(BF16).astype(F32)
