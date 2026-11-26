"""Two-chip foresight JetSpec++ with continuous batching (B slots in lockstep).

Same split as ``foresight.TwoChipJetSpecPP``, every program ``vmap``-ed over the slots as in ``serve.py``:

  chip 0: batched verify of the B current blocks
  chip 1: batched resolve (row lookup; a batch-level redraft only when some slot missed) and batched
          foresight (deferred exact writes, adapter prediction, anchor decoder, row selection, row drafts)

A slot admitted mid-run is marked ``fresh``: its next resolve keeps the admission block (drafted from the
exact prefill context) instead of looking up rows, and its next foresight takes the miss path. Idle
slots stay fresh, so they never trigger a redraft. Writes at positions >= a slot's length are scratch:
every such row is overwritten by an exact write before that slot's length passes it.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from .engine import _sample_init, schedule_ops
from .foresight import TwoChipJetSpecPP, decoder_mirror, draft_jr, predict_block, predict_row, select_rows_fast
from .model import BF16, Dims, new_cache, target_forward, write_context


@dataclass
class TwoChipServer:
    w0: dict
    w1: dict
    coef: dict
    temperature: float
    slots: int
    capacity: int
    dims: Dims = field(default_factory=Dims)
    schedule: str = 'J1-R0'
    rows: int = 8
    seed: int = 9504
    eos_ids: tuple = (151645, 151643)
    group: int = 0          # redraft group size (0: whole batch, as before)
    exact_tail: bool = False  # predict the block over the exact cache prefix (16 tokens) instead of re-running
                              # the adapter over the drafted tail as well (32 tokens)

    def __post_init__(self):
        self.ops = schedule_ops(self.schedule)
        self.has_r = any(op.startswith('R') for op in self.ops)
        self.one = TwoChipJetSpecPP(self.w0, self.w1, self.coef, self.temperature, dims=self.dims,
                                    schedule=self.schedule, rows=self.rows, capacity=self.capacity)
        self.d0, self.d1 = jax.devices()[0], jax.devices()[1]
        self.limit = self.capacity - 3 * self.dims.block
        self._jit = {}

    def _c(self, name, *key):
        k = (name,) + key
        if k not in self._jit:
            self._jit[k] = getattr(self, '_build_' + name)(*key)
        return self._jit[k]

    def span_for(self, need):
        span = 512
        while span < need:
            span *= 2
        return min(span, self.capacity)

    @staticmethod
    def pad_len(p):
        return -(-p // 128) * 128

    # ------------------------------------------------------------------ state
    def _c1_one(self):
        dims, cap = self.dims, self.capacity
        shp = lambda k: (k, dims.kv_heads, cap, dims.head_dim)
        nd = len(dims.decoder_layers)
        return dict(dk=jnp.zeros(shp(dims.draft_layers), BF16), dv=jnp.zeros(shp(dims.draft_layers), BF16),
                    ak=jnp.zeros(shp(dims.adapter_layers), BF16), av=jnp.zeros(shp(dims.adapter_layers), BF16),
                    mk=jnp.zeros(shp(nd), BF16), mv=jnp.zeros(shp(nd), BF16))

    def empty_state(self):
        b, n, dims, cap, r = self.slots, self.dims.block, self.dims, self.capacity, self.rows
        batch = lambda tree: jax.tree.map(lambda a: jnp.zeros((b,) + a.shape, a.dtype), tree)
        with jax.default_device(self.d0):
            c0 = batch(new_cache(dims, cap, adapter=False, draft=False))
        with jax.default_device(self.d1):
            s1 = dict(c1=batch(self._c1_one()), tokens=jnp.zeros((b, cap), jnp.int32),
                      block=jnp.zeros((b, n), jnp.int32), p1=jnp.zeros((b, n), jnp.float32),
                      length=jnp.zeros((b,), jnp.int32), fresh=jnp.ones((b,), bool),
                      key=jnp.zeros((b, 2), jnp.uint32),
                      ks=jnp.zeros((b, r), jnp.int32), anchors=jnp.full((b, r), -1, jnp.int32),
                      drafted=jnp.zeros((b, r, n), jnp.int32), dp1=jnp.zeros((b, r, n), jnp.float32))
            v1 = dict(acc=jnp.zeros((b,), jnp.int32), corr=jnp.zeros((b,), jnp.int32),
                      taps=jnp.zeros((b, n, dims.tap_width), BF16),
                      mk=jnp.zeros((b, len(dims.decoder_layers), dims.kv_heads, n, dims.head_dim), BF16),
                      mv=jnp.zeros((b, len(dims.decoder_layers), dims.kv_heads, n, dims.head_dim), BF16))
        # commit to the chips: jit outputs are committed, and a first call on uncommitted inputs would
        # compile a second variant mid-run
        return jax.device_put(c0, self.d0), jax.device_put(s1, self.d1), jax.device_put(v1, self.d1)

    # ------------------------------------------------------------------ chip 0
    def _build_prefill(self, padded):
        dims, span = self.dims, self.span_for(padded + 2 * self.dims.block)
        lo = dims.decoder_layers[0]

        @partial(jax.jit, donate_argnums=(1,))
        def prefill(w0, c0, slot, ids, plen):
            one = new_cache(dims, self.capacity, adapter=False, draft=False)
            logits, taps, one = target_forward(w0['target'], one, ids, 0, dims, span, logit_row=plen - 1)
            first = jnp.argmax(logits[0]).astype(jnp.int32)
            c0 = jax.tree.map(lambda big, x: jax.lax.dynamic_update_index_in_dim(big, x, slot, 0), c0, one)
            return c0, first, taps, one['tk'][lo:], one['tv'][lo:]
        return prefill

    def _build_verify(self, span):
        dims, n = self.dims, self.dims.block
        lo, nd = dims.decoder_layers[0], len(dims.decoder_layers)

        def one(w0, cache, block, length):
            logits, taps, cache = target_forward(w0['target'], cache, block, length, dims, span)
            pred = jnp.argmax(logits, axis=-1).astype(jnp.int32)
            acc = jnp.sum(jnp.cumprod((pred[:-1] == block[1:]).astype(jnp.int32)))
            mk = jax.lax.dynamic_slice(cache['tk'], (lo, 0, length, 0), (nd, dims.kv_heads, n, dims.head_dim))
            mv = jax.lax.dynamic_slice(cache['tv'], (lo, 0, length, 0), (nd, dims.kv_heads, n, dims.head_dim))
            return cache, dict(acc=acc, corr=pred[acc], taps=taps, mk=mk, mv=mv)

        @partial(jax.jit, donate_argnums=(1,))
        def verify(w0, c0, block, length):
            return jax.vmap(one, in_axes=(None, 0, 0, 0))(w0, c0, block, length)
        return verify

    # ------------------------------------------------------------------ chip 1
    def _build_admit(self, padded):
        dims, n, span = self.dims, self.dims.block, self.span_for(padded + 3 * self.dims.block)

        @partial(jax.jit, donate_argnums=(1,))
        def admit(w1, s1, slot, ids, plen, first, taps, mk, mv, key):
            c = write_context(w1['drafter'], w1['adapter'], self._c1_one(), taps, 0, dims)
            c = dict(c, mk=mk, mv=mv)
            tokens = jax.lax.dynamic_update_slice(jnp.zeros(self.capacity, jnp.int32), ids, (0,)).at[plen].set(first)
            init = _sample_init(jax.random.fold_in(key, 0), tokens, plen + 1, n - 1)
            block, p1 = draft_jr(w1['drafter'], self.one._tw1(w1), first, init, plen, dims, span, c, self.ops,
                                 self.one._exact_refresh(w1, c, plen, span))
            put = lambda big, x: jax.lax.dynamic_update_index_in_dim(big, x, slot, 0)
            return dict(s1, c1=jax.tree.map(put, s1['c1'], c), tokens=put(s1['tokens'], tokens),
                        block=put(s1['block'], block), p1=put(s1['p1'], p1), length=put(s1['length'], plen),
                        fresh=put(s1['fresh'], True), key=put(s1['key'], key))
        return admit

    def _build_retire(self):
        @partial(jax.jit, donate_argnums=(0,))
        def retire(s1, slot):
            return dict(s1, fresh=jax.lax.dynamic_update_index_in_dim(s1['fresh'], True, slot, 0))
        return retire

    def _build_resolve(self, span):
        dims, n, limit = self.dims, self.dims.block, self.limit
        lanes = jnp.arange(n)

        def lookup(tokens, block_t, length, acc, corr, ks, anchors, rows, rp1, fresh):
            committed = jnp.where(lanes <= acc, block_t, corr)
            old = jax.lax.dynamic_slice(tokens, (length,), (n,))
            new_tokens = jax.lax.dynamic_update_slice(tokens, jnp.where(lanes <= acc + 1, committed, old), (length,))
            match = (ks == acc) & (anchors == corr)
            hit = jnp.any(match)
            idx = jnp.argmax(match)
            emitted = jnp.where(lanes < acc, jnp.roll(block_t, -1), jnp.where(lanes == acc, corr, -1))
            return (jnp.where(fresh, tokens, new_tokens), rows[idx], rp1[idx],
                    jnp.where(fresh, length, jnp.minimum(length + acc + 1, limit)),
                    hit & ~fresh, jnp.where(fresh, -1, emitted))

        def redraft(w1, c1, tokens, length, new_length, corr, taps, mk, mv, key):
            if self.has_r:
                scratch = write_context(w1['drafter'], w1['adapter'], c1, taps, length, dims)
                scratch = dict(scratch, mk=jax.lax.dynamic_update_slice(c1['mk'], mk, (0, 0, length, 0)),
                               mv=jax.lax.dynamic_update_slice(c1['mv'], mv, (0, 0, length, 0)))
            else:
                scratch = write_context(w1['drafter'], None, c1, taps, length, dims)
            init = _sample_init(key, tokens, new_length + 1, n - 1)
            return draft_jr(w1['drafter'], self.one._tw1(w1), corr, init, new_length, dims, span, scratch, self.ops,
                            self.one._exact_refresh(w1, scratch, new_length, span))

        @jax.jit
        def resolve(w1, s1, v, step):
            keys = jax.vmap(lambda k: jax.random.fold_in(k, step))(s1['key'])
            tokens, block, p1, length, hit, emitted = jax.vmap(lookup)(
                s1['tokens'], s1['block'], s1['length'], v['acc'], v['corr'], s1['ks'], s1['anchors'],
                s1['drafted'], s1['dp1'], s1['fresh'])
            miss = ~hit & ~s1['fresh']

            vred = jax.vmap(redraft, in_axes=(None, 0, 0, 0, 0, 0, 0, 0, 0, 0))
            G = self.group if 0 < self.group < self.slots else 0
            if not G:
                def with_redraft(_):
                    rb, rp = vred(w1, s1['c1'], tokens, s1['length'], length, v['corr'], v['taps'], v['mk'],
                                  v['mv'], keys)
                    return jnp.where(miss[:, None], rb, block), jnp.where(miss[:, None], rp, p1)
                block, p1 = jax.lax.cond(jnp.any(miss), with_redraft, lambda _: (block, p1), None)
            else:
                # redraft only the missed slots, G at a time: cost scales with misses, not with B
                m = jnp.sum(miss)
                order = jnp.argsort(~miss, stable=True)          # missed slots first
                order = jnp.concatenate([order, order[:G]])      # so every G-slice is in range
                take = lambda x, sel: jax.tree.map(lambda a: a[sel], x)

                def body(carry):
                    c, block, p1 = carry
                    sel = jax.lax.dynamic_slice(order, (c * G,), (G,))
                    ok = (c * G + jnp.arange(G)) < m
                    rb, rp = vred(w1, take(s1['c1'], sel), tokens[sel], s1['length'][sel], length[sel],
                                  v['corr'][sel], v['taps'][sel], v['mk'][sel], v['mv'][sel], keys[sel])
                    block = block.at[sel].set(jnp.where(ok[:, None], rb, block[sel]))
                    p1 = p1.at[sel].set(jnp.where(ok[:, None], rp, p1[sel]))
                    return c + 1, block, p1
                _, block, p1 = jax.lax.while_loop(lambda cr: cr[0] * G < m, body, (0, block, p1))
            block = jnp.where(s1['fresh'][:, None], s1['block'], block)
            p1 = jnp.where(s1['fresh'][:, None], s1['p1'], p1)
            return tokens, block, p1, length, hit, emitted, jnp.sum(miss)
        return resolve

    def _build_foresight(self, span):
        dims, n, rows = self.dims, self.dims.block, self.rows
        coef, temperature = self.coef, self.temperature

        def one(w1, c1, block, p1, length, tokens, key, prev, prev_k, prev_length, hit, taps, mk, mv):
            tw = self.one._tw1(w1)
            # exact taps and decoder K/V of the round just verified (scratch for a fresh slot)
            c1 = write_context(w1['drafter'], w1['adapter'], c1, taps, prev_length, dims)
            c1 = dict(c1, mk=jax.lax.dynamic_update_slice(c1['mk'], mk, (0, 0, prev_length, 0)),
                      mv=jax.lax.dynamic_update_slice(c1['mv'], mv, (0, 0, prev_length, 0)))
            # hit: the row's tail view (as drafted); miss/fresh: exact context (an invisible tail at k = -1)
            k_eff = jnp.where(hit, prev_k, -1)
            base = jnp.where(hit, prev_length, length)
            if self.exact_tail:
                # rows < length are exact after the deferred write above (hit: tail = verified prev[:acc+1])
                pred = predict_block(w1['adapter'], tw, c1, block, length, dims, span)
            else:
                pred = predict_row(w1['adapter'], tw, c1, prev, k_eff, block, base, dims, span)
            dec = decoder_mirror(w1['decoder'], w1['norm'], c1['mk'], c1['mv'], pred[:, -dims.hidden:],
                                 length, dims, span)
            ks, anchors = select_rows_fast(dec, block, p1, coef, temperature, rows)
            cpred = write_context(w1['drafter'], None, c1, pred, length, dims)
            inits = jax.vmap(lambda kk: _sample_init(kk, tokens, length + 1, n - 1))(jax.random.split(key, rows))
            tail_hidden = pred[:, -dims.hidden:]
            drafted, dp1 = jax.vmap(lambda k, a, i: draft_jr(
                w1['drafter'], tw, a, i, length + k + 1, dims, span, cpred, self.ops,
                self.one._row_refresh(w1, c1, block, tail_hidden, k, length, span)))(ks, anchors, inits)
            return c1, ks, anchors, drafted, dp1

        @partial(jax.jit, donate_argnums=(1,))
        def foresight(w1, s1, tokens, block, p1, length, hit, v, step):
            keys = jax.vmap(lambda k: jax.random.fold_in(k, step))(s1['key'])
            c1, ks, anchors, drafted, dp1 = jax.vmap(
                one, in_axes=(None, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0))(
                w1, s1['c1'], block, p1, length, tokens, keys, s1['block'], v['acc'], s1['length'], hit,
                v['taps'], v['mk'], v['mv'])
            return dict(s1, c1=c1, tokens=tokens, block=block, p1=p1, length=length,
                        fresh=jnp.zeros_like(s1['fresh']), ks=ks, anchors=anchors, drafted=drafted, dp1=dp1)
        return foresight

    # ------------------------------------------------------------------ one lockstep round
    def _round(self, c0, s1, v0, step, span):
        to1 = lambda x: jax.device_put(x, self.d1)
        to0 = lambda x: jax.device_put(x, self.d0)
        v1 = jax.tree.map(to1, v0)
        tokens, block, p1, length, hit, emitted, misses = self._c('resolve', span)(self.w1, s1, v1, 2 * step)
        c0, v0 = self._c('verify', span)(self.w0, c0, to0(block), to0(length))
        s1 = self._c('foresight', span)(self.w1, s1, tokens, block, p1, length, hit, v1, 2 * step + 1)
        return c0, s1, v0, (emitted, hit, misses)

    # ------------------------------------------------------------------ warm-up and serving
    def warmup(self, prompt_lens, max_new_tokens):
        c0, s1, v0 = self.empty_state()
        key = jax.random.PRNGKey(0)
        to1 = lambda x: jax.device_put(x, self.d1)
        for padded in sorted({self.pad_len(p) for p in prompt_lens}):
            ids = jnp.zeros(padded, jnp.int32)
            for slot in range(min(2, self.slots)):
                c0, first, taps, mk, mv = self._c('prefill', padded)(self.w0, c0, slot,
                                                                     jax.device_put(ids, self.d0), 8)
                s1 = self._c('admit', padded)(self.w1, s1, slot, to1(ids), 8, to1(first), to1(taps), to1(mk),
                                              to1(mv), to1(key))
        need = max(prompt_lens) + max_new_tokens + 4 * self.dims.block
        span = 512
        while True:
            s = min(span, self.capacity)
            for step in range(2):        # a fresh round and a lookup round
                c0, s1, v0, out = self._round(c0, s1, v0, step, s)
            # force the redraft branch once as well: a slot that is neither fresh nor hit
            s1 = dict(s1, fresh=jnp.zeros_like(s1['fresh']), anchors=jnp.full_like(s1['anchors'], -1))
            c0, s1, v0, out = self._round(c0, s1, v0, 2, s)
            if s >= min(need, self.capacity):
                break
            span *= 2
        self._c('retire')
        jax.block_until_ready((c0, s1, v0))

    def run(self, requests, max_new_tokens):
        if max(len(ids) for _, ids in requests) + max_new_tokens + 4 * self.dims.block > self.capacity:
            raise ValueError('capacity too small for the longest request')
        queue = deque(requests)
        c0, s1, v0 = self.empty_state()
        owner = [None] * self.slots
        host_len = [0] * self.slots
        records, misses_total = [], []
        to1 = lambda x: jax.device_put(x, self.d1)
        start = time.perf_counter()

        def admit(slot):
            nonlocal c0, s1
            index, ids = queue.popleft()
            p = len(ids)
            padded = self.pad_len(p)
            buf = np.zeros(padded, np.int32)
            buf[:p] = ids
            ids_dev = jnp.asarray(buf)
            c0, first, taps, mk, mv = self._c('prefill', padded)(self.w0, c0, slot, jax.device_put(ids_dev, self.d0), p)
            key = jax.random.PRNGKey(self.seed ^ index)
            s1 = self._c('admit', padded)(self.w1, s1, slot, to1(ids_dev), p, to1(first), to1(taps), to1(mk),
                                          to1(mv), to1(key))
            rec = dict(index=index, prompt_len=p, out=[], first=first, t_admit=time.perf_counter(),
                       t_first=None, t_end=None, rounds=0, accepts=[], hits=[])
            records.append(rec)
            owner[slot] = rec
            host_len[slot] = p

        for slot in range(self.slots):
            if queue:
                admit(slot)
        pending, step = None, 0
        while any(o is not None for o in owner) or pending is not None:
            if any(o is not None for o in owner):
                span = self.span_for(max(host_len[s] for s in range(self.slots) if owner[s] is not None)
                                     + 4 * self.dims.block)
                c0, s1, v0, out = self._round(c0, s1, v0, step, span)
                dispatched = (out, list(owner))
                step += 1
            else:
                dispatched = None
            if pending is not None:
                (emitted, hit, misses), owners = pending
                emitted_host, hit_host = np.asarray(emitted), np.asarray(hit)
                misses_total.append(int(misses))
                now = time.perf_counter()
                for slot, rec in enumerate(owners):
                    if rec is None or rec['t_end'] is not None:
                        continue
                    new = [int(t) for t in emitted_host[slot] if t >= 0]
                    if rec['t_first'] is None:
                        rec['out'].append(int(rec.pop('first')))
                        rec['t_first'] = now
                        if not new:              # the fresh round: nothing verified yet
                            continue
                    rec['rounds'] += 1
                    rec['accepts'].append(len(new))
                    rec['hits'].append(bool(hit_host[slot]))
                    done = False
                    for t in new:
                        rec['out'].append(t)
                        if t in self.eos_ids or len(rec['out']) >= max_new_tokens:
                            done = True
                            break
                    host_len[slot] = rec['prompt_len'] + len(rec['out'])
                    if done:
                        rec['t_end'] = now
                        if owner[slot] is rec:
                            owner[slot] = None
                            if queue:
                                admit(slot)
                            else:
                                s1 = self._c('retire')(s1, slot)
            pending = dispatched
        wall = time.perf_counter() - start
        jax.block_until_ready((c0, s1))
        for rec in records:
            rec['out'] = rec['out'][:max_new_tokens]
        return records, wall, dict(rounds=step, redraft_rounds=sum(m > 0 for m in misses_total))


def summarize(records, wall):
    tokens = sum(len(r['out']) for r in records)
    per_user = [len(r['out']) / (r['t_end'] - r['t_admit']) for r in records]
    accepts = [a for r in records for a in r['accepts']]
    hits = [h for r in records for h in r['hits']]
    return dict(requests=len(records), tokens=tokens, wall=wall, total_tok_s=tokens / wall,
                per_user_tok_s_mean=float(np.mean(per_user)), per_user_tok_s_p50=float(np.median(per_user)),
                tau=float(np.mean(accepts)) if accepts else None, hit_rate=float(np.mean(hits)) if hits else None)
