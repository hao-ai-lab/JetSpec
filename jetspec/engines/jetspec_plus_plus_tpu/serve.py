"""Batched serving on one TPU chip with continuous batching: AR, DFlash and JetSpec++.

B slots decode in lockstep. Each round is one compiled program ``vmap``-ed over the slots, so
each weight sweep serves all B sequences. When a request finishes, the next queued request is
prefilled into its slot. Rounds are dispatched one ahead of the host reads, as in ``engine.py``.

Methods (all greedy, verified by the same bf16 target code):
- ``ar``: ``chunk`` single-token target steps per round.
- ``dflash``: block [anchor, mask x 15], one bidirectional drafter pass (z-lab/Qwen3-8B-DFlash-b16).
- ``jetspec``: the J/R self-refinement draft of ``engine.draft_block``.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from .engine import _sample_init, draft_block, schedule_ops
from .model import Dims, drafter_pass, new_cache, target_forward, write_context

MASK_TOKEN = 151669


def _dflash_block(weights, cache, anchor, length, dims, span):
    n = dims.block
    ids = jnp.concatenate([anchor[None], jnp.full((n - 1,), MASK_TOKEN, jnp.int32)]).astype(jnp.int32)
    logits = drafter_pass(weights['drafter'], weights['target'], cache, ids, length, dims, span,
                          causal_block=False)
    return jnp.argmax(logits, axis=-1).astype(jnp.int32).at[0].set(anchor)


@dataclass
class Server:
    weights: dict
    method: str                      # 'ar' | 'dflash' | 'jetspec'
    slots: int
    capacity: int
    dims: Dims = field(default_factory=Dims)
    schedule: str = 'J1-R1'
    chunk: int = 8                   # AR tokens per round
    seed: int = 9504
    eos_ids: tuple = (151645, 151643)

    def __post_init__(self):
        if self.method not in ('ar', 'dflash', 'jetspec'):
            raise ValueError(self.method)
        self.ops = schedule_ops(self.schedule)
        self.width = self.chunk if self.method == 'ar' else self.dims.block
        self._jit = {}

    # ------------------------------------------------------------------ per-sequence programs

    def _draft(self, weights, cache, anchor, tokens, count, length, key, span):
        dims, n = self.dims, self.dims.block
        if self.method == 'dflash':
            return _dflash_block(weights, cache, anchor, length, dims, span)
        init = _sample_init(key, tokens, count, n - 1)
        return draft_block(weights, cache, anchor, init, length, dims, span, self.ops)

    def _round_one(self, weights, cache, tokens, block, length, key, span):
        dims, n, limit = self.dims, self.dims.block, self.capacity - 3 * self.dims.block
        if self.method == 'ar':
            def step(carry, _):
                cache, tok, pos = carry
                logits, _, cache = target_forward(weights['target'], cache, tok[None], pos, dims, span)
                nxt = jnp.argmax(logits[0]).astype(jnp.int32)
                return (cache, nxt, jnp.minimum(pos + 1, limit)), nxt
            (cache, tok, length), emitted = jax.lax.scan(step, (cache, block[0], length), None,
                                                         length=self.chunk)
            return cache, tokens, block.at[0].set(tok), length, emitted
        logits, taps, cache = target_forward(weights['target'], cache, block, length, dims, span)
        pred = jnp.argmax(logits, axis=-1).astype(jnp.int32)
        acc = jnp.sum(jnp.cumprod((pred[:-1] == block[1:]).astype(jnp.int32)))
        correction = pred[acc]
        lanes = jnp.arange(n)
        committed = jnp.where(lanes <= acc, block, jnp.where(lanes == acc + 1, correction, 0))
        old = jax.lax.dynamic_slice(tokens, (length,), (n,))
        tokens = jax.lax.dynamic_update_slice(tokens, jnp.where(lanes <= acc + 1, committed, old), (length,))
        emitted = jnp.where(lanes < acc, jnp.roll(block, -1), jnp.where(lanes == acc, correction, -1))
        cache = write_context(weights['drafter'], weights.get('adapter'), cache, taps, length, dims)
        # Idle slots keep running on garbage; clamp so their writes stay inside their own cache.
        new_length = jnp.minimum(length + acc + 1, limit)
        nxt = self._draft(weights, cache, correction, tokens, new_length + 1, new_length, key, span)
        return cache, tokens, nxt, new_length, emitted

    def _prefill_one(self, weights, ids, plen, key, span):
        dims = self.dims
        cache = new_cache(dims, self.capacity, adapter=self.method == 'jetspec', draft=self.method != 'ar')
        logits, taps, cache = target_forward(weights['target'], cache, ids, 0, dims, span, logit_row=plen - 1)
        first = jnp.argmax(logits[0]).astype(jnp.int32)
        tokens = jnp.zeros(self.capacity, jnp.int32)
        tokens = jax.lax.dynamic_update_slice(tokens, ids, (0,))
        tokens = tokens.at[plen].set(first)
        block = jnp.zeros(dims.block, jnp.int32).at[0].set(first)
        if self.method != 'ar':
            cache = write_context(weights['drafter'], weights.get('adapter'), cache, taps, 0, dims)
            block = self._draft(weights, cache, first, tokens, plen + 1, plen, key, span)
        return cache, tokens, block, plen, first

    # ------------------------------------------------------------------ compiled batched programs

    def _compiled(self, key, build):
        if key not in self._jit:
            self._jit[key] = build()
        return self._jit[key]

    def round_fn(self, span):
        def build():
            one = partial(self._round_one, span=span)

            @partial(jax.jit, donate_argnums=(1,))
            def batched(weights, state, keys):
                cache, tokens, block, length, emitted = jax.vmap(
                    one, in_axes=(None, 0, 0, 0, 0, 0))(weights, state['cache'], state['tokens'],
                                                         state['block'], state['length'], keys)
                return dict(cache=cache, tokens=tokens, block=block, length=length), emitted
            return batched
        return self._compiled(('round', span), build)

    def prefill_fn(self, padded):
        span = self.span_for(padded + 2 * self.dims.block)

        def build():
            @partial(jax.jit, donate_argnums=(1,))
            def prefill(weights, state, slot, ids, plen, key):
                cache, tokens, block, length, first = self._prefill_one(weights, ids, plen, key, span)
                put = lambda big, one: jax.lax.dynamic_update_index_in_dim(big, one, slot, 0)
                state = dict(cache=jax.tree.map(put, state['cache'], cache),
                             tokens=put(state['tokens'], tokens), block=put(state['block'], block),
                             length=put(state['length'], length))
                return state, first
            return prefill
        return self._compiled(('prefill', padded), build)

    def span_for(self, need):
        span = 512
        while span < need:
            span *= 2
        return min(span, self.capacity)

    def empty_state(self):
        b, dims = self.slots, self.dims
        one = new_cache(dims, self.capacity, adapter=self.method == 'jetspec', draft=self.method != 'ar')
        cache = jax.tree.map(lambda a: jnp.zeros((b,) + a.shape, a.dtype), one)
        return dict(cache=cache, tokens=jnp.zeros((b, self.capacity), jnp.int32),
                    block=jnp.zeros((b, dims.block), jnp.int32), length=jnp.zeros((b,), jnp.int32))

    @staticmethod
    def pad_len(p):
        return -(-p // 128) * 128

    def warmup(self, prompt_lens, max_new_tokens):
        """Compile every prefill and round shape a run can reach, outside timing."""
        state = self.empty_state()
        key = jax.random.key(0)
        for padded in sorted({self.pad_len(p) for p in prompt_lens}):
            state, _ = self.prefill_fn(padded)(self.weights, state, 0, jnp.zeros(padded, jnp.int32),
                                               jnp.int32(min(padded, 8)), key)
        need = max(prompt_lens) + max_new_tokens + 4 * self.dims.block
        span = 512
        while True:
            s = min(span, self.capacity)
            state, _ = self.round_fn(s)(self.weights, state, jax.random.split(key, self.slots))
            if s >= min(need, self.capacity):
                break
            span *= 2
        jax.block_until_ready(state)

    # ------------------------------------------------------------------ continuous batching

    def run(self, requests, max_new_tokens):
        """requests: list of (index, token_ids). Returns per-request records and totals."""
        if max(len(ids) for _, ids in requests) + max_new_tokens + 4 * self.dims.block > self.capacity:
            raise ValueError('capacity too small for the longest request')
        queue = deque(requests)
        state = self.empty_state()
        base = jax.random.key(self.seed)
        owner = [None] * self.slots            # request record per slot
        host_len = [0] * self.slots
        records = []
        start = time.perf_counter()

        def admit(slot, state):
            index, ids = queue.popleft()
            p = len(ids)
            padded = self.pad_len(p)
            buf = np.zeros(padded, np.int32)
            buf[:p] = ids
            state, first = self.prefill_fn(padded)(
                self.weights, state, slot, jnp.asarray(buf), jnp.int32(p), jax.random.fold_in(base, index))
            rec = dict(index=index, prompt_len=p, out=[], first=first, t_admit=time.perf_counter(),
                       t_first=None, t_end=None, rounds=0, accepts=[])
            records.append(rec)
            owner[slot] = rec
            host_len[slot] = p
            return state

        for slot in range(self.slots):
            if queue:
                state = admit(slot, state)
        pending, rnd = None, 0
        while any(o is not None for o in owner) or pending is not None:
            if any(o is not None for o in owner):
                span = self.span_for(max(host_len[s] for s in range(self.slots) if owner[s] is not None)
                                     + 4 * self.dims.block)
                keys = jax.random.split(jax.random.fold_in(base, 10_000_000 + rnd), self.slots)
                state, emitted = self.round_fn(span)(self.weights, state, keys)
                dispatched = (emitted, list(owner))
                rnd += 1
            else:
                dispatched = None
            if pending is not None:
                emitted_host = np.asarray(pending[0])
                now = time.perf_counter()
                for slot, rec in enumerate(pending[1]):
                    if rec is None or rec['t_end'] is not None:
                        continue
                    if rec['t_first'] is None:
                        rec['out'].append(int(rec.pop('first')))
                        rec['t_first'] = now
                    new = [int(t) for t in emitted_host[slot] if t >= 0]
                    rec['rounds'] += 1
                    rec['accepts'].append(len(new))
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
                                state = admit(slot, state)
            pending = dispatched
        wall = time.perf_counter() - start
        jax.block_until_ready(state)
        for rec in records:
            rec['out'] = rec['out'][:max_new_tokens]
        return records, wall


def summarize(records, wall):
    tokens = sum(len(r['out']) for r in records)
    # End-to-end speed each user sees: its tokens over admission (prefill) to last token.
    per_user = [len(r['out']) / (r['t_end'] - r['t_admit']) for r in records]
    accepts = [a for r in records for a in r['accepts']]
    return dict(requests=len(records), tokens=tokens, wall=wall, total_tok_s=tokens / wall,
                per_user_tok_s_mean=float(np.mean(per_user)), per_user_tok_s_p50=float(np.median(per_user)),
                tau=float(np.mean(accepts)) if accepts else None)
