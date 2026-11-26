"""JetSpec++ foresight drafting on two TPU chips (B = 1, any J/R schedule).

Chip 0 holds the target and verifies block t. At the same time chip 1, which holds the drafter,
adapter, anchor-decoder copies, embedding and a mirror of the target K/V of the decoder layers,
pre-drafts block t+1 for 16 (accepted length k, anchor a) outcomes of that verification:

  1. decoder over the adapter-predicted states P_t of block t -> logits for the token after k accepts
  2. anchor offers (top-32 after excluding the drafted token), BLEND mass over k, joint score
     P(k) * P_T(a | k), top-16 rows                    (method/drafter.py ``_prepare``, paper_eval)
  3. the 16 predicted taps of block t are written once into the drafter context cache at L..L+15;
     row (k, a) drafts with length L+k+1, so its context is exact[<L] + predicted[L..L+k]
     (``context.build_protocol_hypothesis_contexts``; the adapted-position mark is -0.0)
  4. the adapter predicts each row's block from the exact prefix with the row's tail ids[:k+1]

After verification chip 1 writes the exact taps and decoder K/V, looks up (acc, correction) among
the rows, and on a miss redrafts from the exact context. Lookup and redraft stay on chip 1, so a
round needs no host round trip. Output is the target's greedy decode: every block is verified.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from .engine import _sample_init, _swap, schedule_ops
from .model import (BF16, F32, Dims, adapter_layer_names, attend, dims_embed, drafter_pass, hm, mlp,
                    mm, dot32, rms, rope, rope_tables, target_forward, write_context)

ROWS, POOL, POSITION_SCALE, EPS, GRID = 16, 32, 14.0, 1e-6, 2.0 ** 32


# --------------------------------------------------------------------------- drafting (J passes)

def draft_j(dw, tw, cache, anchor, init, length, dims: Dims, span: int, passes: int):
    """J-only RoundDriver.run: J0 then Jacobi passes, frozen once converged.
    Returns (ids [16], p_top1 [16] of the pass that produced ids)."""
    ids = jnp.concatenate([anchor[None], init]).astype(jnp.int32)
    final, p_final = ids, jnp.zeros(dims.block, F32)
    finished = jnp.array(False)
    for _ in range(passes):
        logits = drafter_pass(dw, tw, cache, ids, length, dims, span)
        tokens = jnp.argmax(logits, axis=-1).astype(jnp.int32).at[0].set(anchor)
        p_top1 = jnp.max(jax.nn.softmax(logits.astype(F32), axis=-1), axis=-1)
        fixed = jnp.all(tokens == ids)
        final = jnp.where(finished, final, tokens)
        p_final = jnp.where(finished, p_final, p_top1)
        ids = tokens
        finished = finished | fixed
    return final, p_final


def draft_jr(dw, tw, anchor, init, length, dims: Dims, span: int, cache, ops, refresh):
    """RoundDriver.run(admission=True) for any J/R schedule (engine.draft_block). ``refresh(ids)`` returns
    (predicted taps [16, 5H], decoder logits [15, V]) for the current ids; rows pass one that sees the
    tail of the block being verified. Returns (ids [16], p_top1 [16] of the J pass that produced ids)."""
    width = dims.block
    ids = jnp.concatenate([anchor[None], init]).astype(jnp.int32)
    pin_mask, pin_values = jnp.zeros(width, bool), jnp.zeros(width, jnp.int32)
    final, p_final = ids, jnp.zeros(width, F32)
    finished, refreshed = jnp.array(False), None
    for op in ops:
        if op.startswith('J'):
            if refreshed is None:
                logits = drafter_pass(dw, tw, cache, ids, length, dims, span)
            else:
                logits = drafter_pass(dw, tw, cache, ids, length, dims, span,
                                      extra=refreshed[0], extra_khat=refreshed[1])
            tokens = jnp.argmax(logits, axis=-1).astype(jnp.int32).at[0].set(anchor)
            tokens = jnp.where(pin_mask, pin_values, tokens)
            p_top1 = jnp.max(jax.nn.softmax(logits.astype(F32), axis=-1), axis=-1)
            fixed = jnp.all(tokens == ids)
            final = jnp.where(finished, final, tokens)
            p_final = jnp.where(finished, p_final, p_top1)
            ids = tokens
            finished = finished | fixed
        else:
            predicted, dec = refresh(ids)
            ids, pin_mask, pin_values, finished, khat_used = _swap(dec, ids, pin_mask, pin_values, finished, width)
            refreshed = (predicted, khat_used)
    return final, p_final


# --------------------------------------------------------------------------- adapter with a tail

def adapter_predict_seq(aw, tw, cache, tokens, positions, valid, length, dims: Dims, span: int):
    """adapter_predict over an explicit window: tokens [N] at positions [N]; window entry j is
    visible to i iff valid[j] and j < i. Prefix = adapter cache rows < length."""
    n = tokens.shape[0]
    cos, sin = rope_tables(positions, dims.head_dim, dims.theta)
    prefix_allowed = jnp.broadcast_to(jnp.arange(span)[None, :] < length, (n, span))
    slots = jnp.arange(n)
    window_allowed = (slots[None, :] < slots[:, None]) & valid[None, :]
    kd = dims.kv_heads * dims.head_dim
    shape = (n, dims.kv_heads, dims.head_dim)
    scale = dims.head_dim ** -0.5
    u = dims_embed(tw['embed'], tokens)
    outputs, state = [], None
    for i, (_, tap, opens) in enumerate(adapter_layer_names(dims)):
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


def predict_row(aw, tw, cache, prev_ids, k, block, length, dims, span):
    """Adapter states of ``block`` drafted after tail prev_ids[:k+1] (positions length..length+k)."""
    n = dims.block
    lanes = jnp.arange(n)
    tokens = jnp.concatenate([prev_ids, block])
    positions = jnp.concatenate([length + lanes, length + k + 1 + lanes])
    valid = jnp.concatenate([lanes <= k, jnp.ones(n, bool)])
    return adapter_predict_seq(aw, tw, cache, tokens, positions, valid, length, dims, span)[n:]


def predict_block(aw, tw, cache, block, length, dims, span):
    n = dims.block
    lanes = jnp.arange(n)
    return adapter_predict_seq(aw, tw, cache, block, length + lanes, jnp.ones(n, bool), length, dims, span)


# --------------------------------------------------------------------------- anchor decoder (mirror K/V)

def decoder_mirror(dec, norm, mk, mv, hidden, length, dims: Dims, span: int):
    """model.decoder_logits with the decoder layers' target K/V held in a 2-layer mirror."""
    w_count = hidden.shape[0]
    return decoder_seq(dec, norm, mk, mv, hidden, length + jnp.arange(w_count), jnp.ones(w_count, bool),
                       length, dims, span)


def decoder_seq(dec, norm, mk, mv, hidden, positions, valid, length, dims: Dims, span: int):
    """decoder_mirror over an explicit window: hidden [W, H] at positions [W]; window entry j is
    visible to i iff valid[j] and j <= i. Prefix = mirror rows < length."""
    w_count = hidden.shape[0]
    cos, sin = rope_tables(positions, dims.head_dim, dims.theta)
    prefix_allowed = jnp.broadcast_to(jnp.arange(span)[None, :] < length, (w_count, span))
    slots = jnp.arange(w_count)
    causal = (slots[None, :] <= slots[:, None]) & valid[None, :]
    qd, kd = dims.heads * dims.head_dim, dims.kv_heads * dims.head_dim
    scale = dims.head_dim ** -0.5
    x = hidden
    for local in range(len(dims.decoder_layers)):
        w = jax.tree.map(lambda a: a[local], dec['layers'])
        h = rms(x, w['in_norm'], dims.eps)
        qkv = mm(h, w['qkv'])
        q = rope(rms(qkv[:, :qd].reshape(w_count, dims.heads, dims.head_dim), w['q_norm'], dims.eps), cos, sin)
        k = rope(rms(qkv[:, qd:qd + kd].reshape(w_count, dims.kv_heads, dims.head_dim), w['k_norm'], dims.eps),
                 cos, sin)
        v = qkv[:, qd + kd:].reshape(w_count, dims.kv_heads, dims.head_dim)
        a = attend(q, [(mk[local, :, :span], mv[local, :, :span], prefix_allowed), (hm(k), hm(v), causal)], scale)
        x = x + mm(a, w['o'])
        x = x + mlp(rms(x, w['post_norm'], dims.eps), w['gate_up'], w['down'])
    x = rms(x, norm, dims.eps)
    return dot32(x, dec['lm_head']).astype(BF16).astype(F32)


# --------------------------------------------------------------------------- row selection

def select_rows(dec_logits, block, p_top1, coef, temperature):
    """anchor_offers (q_hat = 1, cap 32) + blend + joint_rule(floor 16, fill_zero) + select_rows."""
    n = block.shape[0] - 1                                    # 15 drafts
    drafts = block[1:]
    rejected = jnp.concatenate([drafts, jnp.array([-1], jnp.int32)])          # [16]
    lanes = jnp.arange(n + 1)
    vocab = dec_logits.shape[-1]
    excl = jax.nn.one_hot(jnp.maximum(rejected, 0), vocab, dtype=bool) & (rejected >= 0)[:, None]
    probs = jax.nn.softmax(jnp.where(excl, -jnp.inf, dec_logits), axis=-1)  # post-exclusion
    top_p, top_id = jax.lax.top_k(probs, POOL)                                # [16, 32]
    # BLEND over k = 0..14 hazards
    q = jnp.take_along_axis(jax.nn.softmax(dec_logits, axis=-1), jnp.maximum(rejected, 0)[:, None], 1)[:, 0]
    d = jnp.clip(p_top1[1:n + 1], EPS, 1 - EPS)
    dq = jnp.clip(q[:n], EPS, 1 - EPS)
    logit = lambda v: jnp.log(v) - jnp.log1p(-v)
    z = coef['a'] * logit(d) + coef['b'] * logit(dq) + coef['c']
    z = z + coef['argmax'] * (jnp.argmax(dec_logits, -1)[:n] == drafts).astype(F32)
    z = z + coef['top'] * logit(jnp.clip(top_p[:n, 0], EPS, 1 - EPS))
    z = z + coef['pos'] * (jnp.arange(1, n + 1, dtype=F32) / POSITION_SCALE)
    h = jax.nn.sigmoid(z)
    surv = jnp.concatenate([jnp.ones(1, F32), jnp.cumprod(h)])               # [16]
    mass = jnp.concatenate([surv[:-1] * (1 - h), surv[-1:]])
    mass = jnp.round(mass * GRID) / GRID
    scaled = jnp.maximum(top_p, 1e-30) ** (1.0 / temperature)
    anchor_p = scaled / jnp.sum(scaled, -1, keepdims=True)
    score = jnp.round(mass[:, None] * anchor_p * GRID) / GRID                 # [16, 32]
    _, flat = jax.lax.top_k(score.reshape(-1), ROWS)
    ks = (flat // POOL).astype(jnp.int32)
    anchors = top_id.reshape(-1)[flat].astype(jnp.int32)
    return ks, anchors, lanes


def select_rows_fast(dec_logits, block, p1, coef, temperature, rows):
    """select_rows without a full-vocabulary softmax + exact top_k: logsumexp normalisation and the TPU
    approximate top-k of the masked logits. Same scores; only a near-tie at the pool edge can differ."""
    n = block.shape[0] - 1
    drafts = block[1:]
    rejected = jnp.concatenate([drafts, jnp.array([-1], jnp.int32)])
    vocab = dec_logits.shape[-1]
    excl = (jnp.arange(vocab)[None, :] == rejected[:, None])
    masked = jnp.where(excl, -jnp.inf, dec_logits)
    lse_m = jax.nn.logsumexp(masked, axis=-1)
    lse_f = jax.nn.logsumexp(dec_logits, axis=-1)
    top_l, top_id = jax.lax.approx_max_k(masked, POOL, recall_target=0.95)
    top_p = jnp.exp(top_l - lse_m[:, None])
    q = jnp.exp(jnp.take_along_axis(dec_logits, jnp.maximum(rejected, 0)[:, None], 1)[:, 0] - lse_f)
    d = jnp.clip(p1[1:n + 1], EPS, 1 - EPS)
    dq = jnp.clip(q[:n], EPS, 1 - EPS)
    logit = lambda v: jnp.log(v) - jnp.log1p(-v)
    z = coef['a'] * logit(d) + coef['b'] * logit(dq) + coef['c']
    z = z + coef['argmax'] * (jnp.argmax(dec_logits, -1)[:n] == drafts).astype(F32)
    z = z + coef['top'] * logit(jnp.clip(top_p[:n, 0], EPS, 1 - EPS))
    z = z + coef['pos'] * (jnp.arange(1, n + 1, dtype=F32) / POSITION_SCALE)
    h = jax.nn.sigmoid(z)
    surv = jnp.concatenate([jnp.ones(1, F32), jnp.cumprod(h)])
    mass = jnp.round(jnp.concatenate([surv[:-1] * (1 - h), surv[-1:]]) * GRID) / GRID
    scaled = jnp.maximum(top_p, 1e-30) ** (1.0 / temperature)
    score = jnp.round(mass[:, None] * scaled / jnp.sum(scaled, -1, keepdims=True) * GRID) / GRID
    _, flat = jax.lax.top_k(score.reshape(-1), rows)
    return (flat // POOL).astype(jnp.int32), top_id.reshape(-1)[flat].astype(jnp.int32)


# --------------------------------------------------------------------------- the engine

@dataclass
class TwoChipJetSpecPP:
    """Per round on chip 1: ``resolve`` (hit lookup, or redraft on a miss) hands the next block to chip 0
    at once; ``foresight`` then writes the exact taps of the round just verified, predicts the chosen
    block's states and pre-drafts the rows, while chip 0 verifies. Nothing waits on the host."""
    w0: dict                      # chip 0: {'target'}
    w1: dict                      # chip 1: {'drafter','adapter','decoder','embed','norm'}
    coef: dict
    temperature: float
    dims: Dims = field(default_factory=Dims)
    schedule: str = 'J1-R0'
    rows: int = 8
    capacity: int = 2048 + 512
    seed: int = 9504
    eos_ids: tuple = (151645, 151643)
    timing: bool = False

    def __post_init__(self):
        self.ops = schedule_ops(self.schedule)
        self.d0, self.d1 = jax.devices()[0], jax.devices()[1]
        self.stage_ms = []
        self._jit = {}

    def span_for(self, need):
        span = 512
        while span < need:
            span *= 2
        return min(span, self.capacity)

    def _c(self, name, *key):
        k = (name,) + key
        if k not in self._jit:
            self._jit[k] = getattr(self, '_build_' + name)(*key)
        return self._jit[k]

    def _tw1(self, w1):
        # drafter_pass evaluates tw['lm_head'] as a default; the drafter's own int8 head is the one used
        return dict(embed=w1['embed'], lm_head=w1['drafter']['lm_head'])

    def _exact_refresh(self, w1, cache, length, span):
        """R from the exact context: adapter over the block, decoder over its first 15 states."""
        dims, tw = self.dims, self._tw1(w1)

        def refresh(ids):
            pr = predict_block(w1['adapter'], tw, cache, ids, length, dims, span)
            dec = decoder_mirror(w1['decoder'], w1['norm'], cache['mk'], cache['mv'], pr[:dims.block - 1, -dims.hidden:],
                                 length, dims, span)
            return pr, dec
        return refresh

    def _row_refresh(self, w1, cache, tail_ids, tail_hidden, k, length, span):
        """R for row (k, a): the block sits at length+k+1 after the predicted tail tail_ids[:k+1]."""
        dims, tw, n = self.dims, self._tw1(w1), self.dims.block
        lanes = jnp.arange(n)

        def refresh(ids):
            pr = predict_row(w1['adapter'], tw, cache, tail_ids, k, ids, length, dims, span)
            hidden = jnp.concatenate([tail_hidden, pr[:n - 1, -dims.hidden:]])
            positions = jnp.concatenate([length + lanes, length + k + 1 + lanes[:n - 1]])
            valid = jnp.concatenate([lanes <= k, jnp.ones(n - 1, bool)])
            dec = decoder_seq(w1['decoder'], w1['norm'], cache['mk'], cache['mv'], hidden, positions, valid,
                              length, dims, span)[n:]
            return pr, dec
        return refresh

    # ---- chip 0
    def _build_verify(self, span):
        dims, n = self.dims, self.dims.block
        lo, nd = dims.decoder_layers[0], len(dims.decoder_layers)

        @partial(jax.jit, donate_argnums=(1,))
        def verify(w0, cache, block, length):
            logits, taps, cache = target_forward(w0['target'], cache, block, length, dims, span)
            pred = jnp.argmax(logits, axis=-1).astype(jnp.int32)
            acc = jnp.sum(jnp.cumprod((pred[:-1] == block[1:]).astype(jnp.int32)))
            corr = pred[acc]
            mk = jax.lax.dynamic_slice(cache['tk'], (lo, 0, length, 0), (nd, dims.kv_heads, n, dims.head_dim))
            mv = jax.lax.dynamic_slice(cache['tv'], (lo, 0, length, 0), (nd, dims.kv_heads, n, dims.head_dim))
            return cache, acc, corr, taps, mk, mv
        return verify

    def _build_prefill(self, span, padded):
        dims = self.dims
        lo = dims.decoder_layers[0]

        @partial(jax.jit, donate_argnums=(1,))
        def prefill(w0, cache, ids, plen):
            logits, taps, cache = target_forward(w0['target'], cache, ids, 0, dims, span, logit_row=plen - 1)
            first = jnp.argmax(logits[0]).astype(jnp.int32)
            return cache, first, taps, cache['tk'][lo:], cache['tv'][lo:]
        return prefill

    # ---- chip 1
    def _build_admit(self, span):
        dims = self.dims

        @partial(jax.jit, donate_argnums=(1,))
        def admit(w1, c1, taps, mk, mv, plen, first, init):
            c1 = write_context(w1['drafter'], w1['adapter'], c1, taps, 0, dims)
            c1 = dict(c1, mk=mk, mv=mv)
            block, p1 = draft_jr(w1['drafter'], self._tw1(w1), first, init, plen, dims, span, c1, self.ops,
                                 self._exact_refresh(w1, c1, plen, span))
            return c1, block, p1
        return admit

    def _build_resolve(self, span):
        """Critical path on a hit: pick the row. A miss drafts from the exact context in a scratch copy
        of the drafter cache (foresight writes the exact taps into the real caches right after)."""
        dims, n = self.dims, self.dims.block
        has_r = any(op.startswith('R') for op in self.ops)

        @jax.jit
        def resolve(w1, c1, tokens, block_t, length, acc, corr, taps, mk, mv, ks, anchors, rows, rp1, key, step):
            key = jax.random.fold_in(key, step)
            lanes = jnp.arange(n)
            committed = jnp.where(lanes <= acc, block_t, corr)
            old = jax.lax.dynamic_slice(tokens, (length,), (n,))
            tokens = jax.lax.dynamic_update_slice(tokens, jnp.where(lanes <= acc + 1, committed, old), (length,))
            new_length = length + acc + 1
            match = (ks == acc) & (anchors == corr)
            hit = jnp.any(match)
            idx = jnp.argmax(match)

            def take(_):
                return rows[idx], rp1[idx]

            def redraft(_):
                if has_r:
                    scratch = write_context(w1['drafter'], w1['adapter'], c1, taps, length, dims)
                    scratch = dict(scratch, mk=jax.lax.dynamic_update_slice(c1['mk'], mk, (0, 0, length, 0)),
                                   mv=jax.lax.dynamic_update_slice(c1['mv'], mv, (0, 0, length, 0)))
                else:
                    scratch = write_context(w1['drafter'], None, c1, taps, length, dims)
                init = _sample_init(key, tokens, new_length + 1, n - 1)
                return draft_jr(w1['drafter'], self._tw1(w1), corr, init, new_length, dims, span, scratch, self.ops,
                                self._exact_refresh(w1, scratch, new_length, span))
            block, p1 = jax.lax.cond(hit, take, redraft, None)
            emitted = jnp.where(lanes < acc, jnp.roll(block_t, -1), jnp.where(lanes == acc, corr, -1))
            return tokens, block, p1, new_length, hit, emitted
        return resolve

    def _build_foresight(self, span, first_round):
        dims, n, rows = self.dims, self.dims.block, self.rows

        @partial(jax.jit, donate_argnums=(1,))
        def foresight(w1, c1, block, p1, length, tokens, key, step, prev, prev_k, prev_length, hit, taps, mk, mv):
            key = jax.random.fold_in(key, step)
            tw = self._tw1(w1)
            if first_round:
                pred = predict_block(w1['adapter'], tw, c1, block, length, dims, span)
            else:
                # exact taps and decoder K/V of the round just verified (deferred off the critical path)
                c1 = write_context(w1['drafter'], w1['adapter'], c1, taps, prev_length, dims)
                c1 = dict(c1, mk=jax.lax.dynamic_update_slice(c1['mk'], mk, (0, 0, prev_length, 0)),
                          mv=jax.lax.dynamic_update_slice(c1['mv'], mv, (0, 0, prev_length, 0)))
                pred = jax.lax.cond(
                    hit,
                    lambda _: predict_row(w1['adapter'], tw, c1, prev, prev_k, block, prev_length, dims, span),
                    lambda _: predict_block(w1['adapter'], tw, c1, block, length, dims, span), None)
            dec = decoder_mirror(w1['decoder'], w1['norm'], c1['mk'], c1['mv'], pred[:, -dims.hidden:],
                                 length, dims, span)
            ks, anchors = select_rows_fast(dec, block, p1, self.coef, self.temperature, rows)
            cpred = write_context(w1['drafter'], None, c1, pred, length, dims)
            inits = jax.vmap(lambda kk: _sample_init(kk, tokens, length + 1, n - 1))(jax.random.split(key, rows))
            tail_hidden = pred[:, -dims.hidden:]
            drafted, dp1 = jax.vmap(lambda k, a, i: draft_jr(
                w1['drafter'], tw, a, i, length + k + 1, dims, span, cpred, self.ops,
                self._row_refresh(w1, c1, block, tail_hidden, k, length, span)))(ks, anchors, inits)
            return c1, ks, anchors, drafted, dp1
        return foresight

    # ---- host loop
    def new_caches(self, capacity):
        dims = self.dims
        shp = lambda k: (k, dims.kv_heads, capacity, dims.head_dim)
        c0 = jax.device_put(dict(tk=jnp.zeros(shp(dims.target_layers), BF16),
                                 tv=jnp.zeros(shp(dims.target_layers), BF16)), self.d0)
        nd = len(dims.decoder_layers)
        c1 = jax.device_put(dict(dk=jnp.zeros(shp(dims.draft_layers), BF16), dv=jnp.zeros(shp(dims.draft_layers), BF16),
                                 ak=jnp.zeros(shp(dims.adapter_layers), BF16), av=jnp.zeros(shp(dims.adapter_layers), BF16),
                                 mk=jnp.zeros(shp(nd), BF16), mv=jnp.zeros(shp(nd), BF16)), self.d1)
        return c0, c1

    def generate(self, prompt, max_new_tokens=256, *, prompt_index=0):
        n, p, cap = self.dims.block, len(prompt), self.capacity
        if p + max_new_tokens + 4 * n > cap:
            raise ValueError('prompt plus max_new_tokens exceeds the cache capacity')
        to0 = lambda x: jax.device_put(x, self.d0)
        to1 = lambda x: jax.device_put(x, self.d1)
        c0, c1 = self.new_caches(cap)
        padded = max(64, -(-p // 64) * 64)
        ids = to0(jnp.asarray(np.pad(np.asarray(prompt, np.int32), (0, padded - p))))
        c0, first, ptaps, pmk, pmv = self._c('prefill', self.span_for(padded), padded)(self.w0, c0, ids, p)
        tokens = np.zeros(cap, np.int32)
        tokens[:p] = prompt
        tokens[p] = int(first)
        tokens = to1(jnp.asarray(tokens))
        key = jax.random.key(self.seed ^ prompt_index)
        init = _sample_init(jax.random.fold_in(key, 0), tokens, p + 1, n - 1)
        c1, block, p1 = self._c('admit', self.span_for(p + 3 * n))(
            self.w1, c1, to1(ptaps), to1(pmk), to1(pmv), to1(jnp.int32(p)), to1(first), init)
        length1 = to1(jnp.int32(p))
        out, hits, accepts = [int(first)], [], []
        host_len = p
        z = to1(jnp.int32(0))
        start = time.perf_counter()
        # round 0: verify the admission block while chip 1 pre-drafts its rows
        span = self.span_for(host_len + 4 * n)
        c0, acc, corr, vtaps, vmk, vmv = self._c('verify', span)(self.w0, c0, to0(block), to0(length1))
        key1 = to1(key)
        c1, *fs = self._c('foresight', span, True)(self.w1, c1, block, p1, length1, tokens, key1, 1,
                                                   block, z, length1, False, None, None, None)
        r, pending, done = 1, None, False
        while not done:
            span = self.span_for(host_len + 4 * n)
            t0 = time.perf_counter()
            vt1, acc1, mk1, mv1 = to1(vtaps), to1(acc), to1(vmk), to1(vmv)
            tokens, nblock, p1, nlength, hit, emitted = self._c('resolve', span)(
                self.w1, c1, tokens, block, length1, acc1, to1(corr), vt1, mk1, mv1, *fs, key1, 2 * r)
            if self.timing:
                jax.block_until_ready(nblock); t1 = time.perf_counter()
            c0, acc_n, corr_n, vtaps_n, vmk_n, vmv_n = self._c('verify', span)(
                self.w0, c0, to0(nblock), to0(nlength))
            if self.timing == 'split':
                jax.block_until_ready(acc_n); tv = time.perf_counter()
            c1, *fs = self._c('foresight', span, False)(
                self.w1, c1, nblock, p1, nlength, tokens, key1, 2 * r + 1,
                block, acc1, length1, hit, vt1, mk1, mv1)
            if self.timing:
                jax.block_until_ready((acc_n, fs)); t2 = time.perf_counter()
                if self.timing == 'split':
                    self.stage_ms.append((1e3 * (t1 - t0), 1e3 * (tv - t1), 1e3 * (t2 - tv)))
                else:
                    self.stage_ms.append((1e3 * (t1 - t0), 1e3 * (t2 - t1)))
            block, length1 = nblock, nlength
            acc, corr, vtaps, vmk, vmv = acc_n, corr_n, vtaps_n, vmk_n, vmv_n
            r += 1
            # Read the previous round's tokens while this one runs; the host length lags by one round.
            if pending is not None:
                done = self._emit(pending, out, accepts, hits, max_new_tokens)
                host_len = p + sum(accepts)
            pending = (emitted, hit)
        seconds = time.perf_counter() - start
        return out[:max_new_tokens], dict(rounds=len(accepts), accepts=accepts, hits=hits, seconds=seconds)

    def _emit(self, pending, out, accepts, hits, max_new_tokens):
        emitted, hit = pending
        row = np.asarray(emitted)
        new = [int(t) for t in row[row >= 0]]
        accepts.append(len(new))
        hits.append(bool(hit))
        for t in new:
            out.append(t)
            if t in self.eos_ids:
                return True
        return len(out) >= max_new_tokens
