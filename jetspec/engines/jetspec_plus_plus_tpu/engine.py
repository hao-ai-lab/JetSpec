"""JetSpec++ on one TPU chip.

Two GPUs let the reference hide drafting behind verification: while block t verifies, the
draft GPU pre-drafts block t+1 for up to sixteen (accepted length, anchor) outcomes. A single
chip runs one program at a time, so those rows cannot hide anywhere, and a hit row is drafted
from adapter-predicted context while the reference's miss path drafts from the exact verified
context. This engine therefore runs, per round, in one compiled program:

    verify block t -> accept -> cache the new taps -> draft block t+1 from the exact context
    with the J/R self-refinement schedule (``RoundDriver.run(admission=True)``).

The draft follows ``round.py`` with the paper-eval policy (``JetSpecPPConfig.paper_eval``):
khat = 15 at every refresh, dense agreed pins, margin-0 anchor swap and frozen converged passes.
Verification accepts the longest draft prefix equal to the target's greedy tokens and appends
the target's own next token, so the output is the target's greedy decode.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from .model import (Dims, adapter_predict, decoder_logits, drafter_pass, new_cache,
                    target_forward, write_context)


def schedule_ops(schedule: str):
    m = re.fullmatch(r'[Jj]([0-9]+)-[Rr]([0-9]+)', schedule)
    if m is None:
        raise ValueError(f'schedule={schedule!r}: expected J<m>-R<n>')
    j, r = map(int, m.groups())
    ops = ['J0']
    for i in range(1, max(j, r) + 1):
        if i <= r:
            ops.append(f'R{i}')
        if i <= j:
            ops.append(f'J{i}')
    return tuple(ops)


# --------------------------------------------------------------------------- one draft

def _swap(dec_logits, ids, pin_mask, pin_values, finished, width):
    """RefreshSwap.apply for one row at khat=15 with slot_counts=15 (``method/swap.py``)."""
    n = width - 1                                    # 15 decoder rows, slots 1..15
    dec_arg = jnp.argmax(dec_logits, axis=-1)        # [15]
    lanes = jnp.arange(1, width)
    agree = (dec_arg == ids[1:]) & (lanes < n)       # slot_counts = 15: slots 1..14 valid
    f = jnp.sum(jnp.cumprod(agree.astype(jnp.int32)))
    khat_used = jnp.minimum(15, f)
    target = khat_used + 1
    slot = jnp.minimum(target, width - 1)
    eligible = target <= width - 1
    row = dec_logits[jnp.clip(slot - 1, 0, n - 1)]
    current = ids[slot]
    probs = jax.nn.softmax(row)
    new = jnp.argmax(row.at[current].set(-jnp.inf))
    apply = (probs[new] - probs[current] >= 0) & (new != current) & eligible & ~pin_mask[slot]
    all_lanes = jnp.arange(width)
    agreed = (all_lanes >= 1) & (all_lanes <= khat_used) & ~pin_mask
    pin_values = jnp.where(agreed, ids, pin_values)
    pin_mask = pin_mask | agreed
    swapped = (all_lanes == slot) & apply
    pin_values = jnp.where(swapped, new, pin_values)
    pin_mask = pin_mask | swapped
    ids = ids.at[slot].set(jnp.where(apply, new, current))
    finished = finished & ~apply
    return ids, pin_mask, pin_values, finished, khat_used


def draft_block(weights, cache, anchor, init, length, dims: Dims, span: int, ops):
    """``RoundDriver.run(admission=True)``: the next block [16] from the exact context."""
    tw, dw, aw = weights['target'], weights['drafter'], weights['adapter']
    width = dims.block
    ids = jnp.concatenate([anchor[None], init]).astype(jnp.int32)
    pin_mask = jnp.zeros(width, bool)
    pin_values = jnp.zeros(width, jnp.int32)
    finished = jnp.array(False)
    final = ids
    refreshed = None
    for op in ops:
        if op.startswith('J'):
            if refreshed is None:
                logits = drafter_pass(dw, tw, cache, ids, length, dims, span)
            else:
                logits = drafter_pass(dw, tw, cache, ids, length, dims, span,
                                      extra=refreshed[0], extra_khat=refreshed[1])
            tokens = jnp.argmax(logits, axis=-1).astype(jnp.int32).at[0].set(anchor)
            tokens = jnp.where(pin_mask, pin_values, tokens)
            fixed = jnp.all(tokens == ids)
            final = jnp.where(finished, final, tokens)
            ids = tokens
            finished = finished | fixed
        else:
            predicted = adapter_predict(aw, tw, cache, ids, length, dims, span)     # [16, 5H]
            dec = decoder_logits(tw, cache, predicted[:width - 1, -dims.hidden:], length, dims, span,
                                 dec=weights.get('decoder'))
            ids, pin_mask, pin_values, finished, khat_used = _swap(
                dec, ids, pin_mask, pin_values, finished, width)
            refreshed = (predicted, khat_used)
    return final


# --------------------------------------------------------------------------- compiled steps

def _sample_init(key, tokens, count, n):
    """J0 slots 1..15: tokens drawn uniformly from the committed context (``kth`` mode)."""
    idx = jax.random.randint(key, (n,), 0, count)
    return tokens[idx]


@dataclass
class TPUJetSpecPP:
    weights: dict
    dims: Dims = field(default_factory=Dims)
    schedule: str = 'J1-R1'
    capacity: int = 4096 + 512
    seed: int = 9504
    eos_ids: tuple = (151645, 151643)

    def __post_init__(self):
        self.ops = schedule_ops(self.schedule)
        self._jit = {}

    # Each compiled program takes a static attention span bucket.
    def span_for(self, need):
        span = 512
        while span < need:
            span *= 2
        return min(span, self.capacity)

    def spans(self, need):
        """Every span bucket a request reaching ``need`` cache rows can use."""
        out, span = [], 512
        while True:
            out.append(min(span, self.capacity))
            if span >= min(need, self.capacity):
                return out
            span *= 2

    def warmup(self, prompt_len, max_new_tokens, *, ar=False, chunk=32):
        """Compile every program a request of this size can reach, outside any timed region."""
        n = self.dims.block
        need = prompt_len + max_new_tokens + 4 * n
        cache = new_cache(self.dims, self.capacity)
        tokens = jnp.zeros(self.capacity, jnp.int32)
        key = jax.random.key(0)
        for span in self.spans(need):
            length = jnp.asarray(min(span, self.capacity) - 3 * n, jnp.int32)
            if ar:
                cache, _ = self._compiled('ar', span, (chunk,))(self.weights, cache, jnp.int32(0), length)
                continue
            block = self._compiled('draft', span)(self.weights, cache, jnp.int32(0),
                                                  jnp.zeros(n - 1, jnp.int32), length)
            cache, tokens, block, _, _ = self._compiled('round', span)(
                self.weights, cache, tokens, block, length, key)
        jax.block_until_ready(cache)

    def _compiled(self, name, span, extra=()):
        key = (name, span) + tuple(extra)
        if key not in self._jit:
            self._jit[key] = getattr(self, '_build_' + name)(span, *extra)
        return self._jit[key]

    def _build_prefill(self, span, padded):
        dims = self.dims

        @partial(jax.jit, donate_argnums=(1,))
        def prefill(weights, cache, ids, prompt_len):
            logits, taps, cache = target_forward(weights['target'], cache, ids, 0, dims, span,
                                                 logit_row=prompt_len - 1)
            cache = write_context(weights['drafter'], weights['adapter'], cache, taps, 0, dims)
            first = jnp.argmax(logits[0]).astype(jnp.int32)
            return first, cache
        return prefill

    def _build_draft(self, span):
        dims, ops = self.dims, self.ops

        @jax.jit
        def draft(weights, cache, anchor, init, length):
            return draft_block(weights, cache, anchor, init, length, dims, span, ops)
        return draft

    def _build_round(self, span):
        """verify(block) + accept + cache the accepted taps + draft the next block."""
        dims, ops, n = self.dims, self.ops, self.dims.block

        @partial(jax.jit, donate_argnums=(1, 2))
        def round_(weights, cache, tokens, block, length, key):
            logits, taps, cache = target_forward(weights['target'], cache, block, length, dims, span)
            pred = jnp.argmax(logits, axis=-1).astype(jnp.int32)
            match = (pred[:-1] == block[1:]).astype(jnp.int32)
            acc = jnp.sum(jnp.cumprod(match))
            correction = pred[acc]
            lanes = jnp.arange(n)
            # committed: block[0..acc] at length.., then the correction
            committed = jnp.where(lanes <= acc, block, jnp.where(lanes == acc + 1, correction, 0))
            old = jax.lax.dynamic_slice(tokens, (length,), (n,))
            tokens = jax.lax.dynamic_update_slice(
                tokens, jnp.where(lanes <= acc + 1, committed, old), (length,))
            # emitted this round: block[1..acc] then the correction, -1 padded
            emitted = jnp.where(lanes < acc, jnp.roll(block, -1),
                                jnp.where(lanes == acc, correction, -1))
            cache = write_context(weights['drafter'], weights['adapter'], cache, taps, length, dims)
            new_length = length + acc + 1
            init = _sample_init(key, tokens, new_length + 1, n - 1)
            nxt = draft_block(weights, cache, correction, init, new_length, dims, span, ops)
            return cache, tokens, nxt, new_length, emitted
        return round_

    def _build_ar(self, span, steps):
        dims = self.dims

        @partial(jax.jit, donate_argnums=(1,))
        def ar(weights, cache, token, length):
            def body(carry, _):
                cache, tok, pos = carry
                logits, _, cache = target_forward(weights['target'], cache, tok[None], pos, dims, span)
                nxt = jnp.argmax(logits[0]).astype(jnp.int32)
                return (cache, nxt, pos + 1), nxt
            (cache, tok, pos), out = jax.lax.scan(body, (cache, token, length), None, length=steps)
            return cache, out
        return ar

    # ------------------------------------------------------------------- host loops

    def _prefill(self, prompt):
        p = len(prompt)
        padded = max(64, -(-p // 64) * 64)
        cache = new_cache(self.dims, self.capacity)
        ids = jnp.asarray(np.pad(np.asarray(prompt, np.int32), (0, padded - p)))
        fn = self._compiled('prefill', self.span_for(padded), (padded,))
        first, cache = fn(self.weights, cache, ids, p)
        return first, cache

    def generate(self, prompt, max_new_tokens=2048, *, prompt_index=0, stats=False):
        """Greedy JetSpec++ decode of prompt token ids; returns emitted ids (and stats)."""
        n = self.dims.block
        p = len(prompt)
        if p + max_new_tokens + 4 * n > self.capacity:
            raise ValueError('prompt plus max_new_tokens exceeds the cache capacity')
        first, cache = self._prefill(prompt)
        out = [int(first)]
        accepts = []
        if out[0] in self.eos_ids or max_new_tokens <= 1:
            info = dict(rounds=0, accept_lengths=[], decode_seconds=0.0)
            return (out[:max_new_tokens], info) if stats else out[:max_new_tokens]
        tokens = np.zeros(self.capacity, np.int32)
        tokens[:p] = prompt
        tokens[p] = out[0]
        tokens = jnp.asarray(tokens)
        key = jax.random.key(self.seed ^ prompt_index)
        length = p
        init = _sample_init(jax.random.fold_in(key, 0), tokens, p + 1, n - 1)
        block = self._compiled('draft', self.span_for(p + n))(
            self.weights, cache, first, init, jnp.asarray(length, jnp.int32))
        length_dev = jnp.asarray(length, jnp.int32)
        start = time.perf_counter()
        pending, rounds, done = None, 0, False
        # Dispatch round r+1 before reading round r, so the host never idles the chip.
        while not done:
            # The host length lags the device by at most one round of 16 tokens.
            fn = self._compiled('round', self.span_for(length + 3 * n))
            cache, tokens, block, length_dev, emitted = fn(
                self.weights, cache, tokens, block, length_dev, jax.random.fold_in(key, rounds + 1))
            rounds += 1
            if pending is not None:
                done = self._emit(pending, out, accepts, max_new_tokens)
                length = p + sum(accepts)
            pending = emitted
        # The clock stops when the last needed round is read; the one in flight is not counted.
        decode_seconds = time.perf_counter() - start
        jax.block_until_ready(block)
        out = out[:max_new_tokens]
        if stats:
            return out, dict(rounds=len(accepts), accept_lengths=accepts, decode_seconds=decode_seconds)
        return out

    def _emit(self, emitted, out, accepts, max_new_tokens):
        row = np.asarray(emitted)
        new = [int(t) for t in row[row >= 0]]
        accepts.append(len(new))
        for t in new:
            out.append(t)
            if t in self.eos_ids:
                return True
        return len(out) >= max_new_tokens

    def generate_ar(self, prompt, max_new_tokens=2048, *, chunk=32, stats=False):
        """Plain greedy decode with the same target code, chunk tokens per compiled call."""
        p = len(prompt)
        first, cache = self._prefill(prompt)
        out = [int(first)]
        token, length = first, p
        start = time.perf_counter()
        pending, done = None, out[-1] in self.eos_ids or len(out) >= max_new_tokens
        # Dispatch chunk c+1 before reading chunk c, as the spec loop does.
        while not done:
            fn = self._compiled('ar', self.span_for(length + chunk), (chunk,))
            cache, nxt = fn(self.weights, cache, token, jnp.asarray(length, jnp.int32))
            token, length = nxt[-1], length + chunk
            if pending is not None:
                host = [int(x) for x in np.asarray(pending)]
                out.extend(host)
                done = any(t in self.eos_ids for t in host) or len(out) >= max_new_tokens
            pending = nxt
        # The clock stops when the last needed chunk is read; the one in flight is not counted.
        seconds = time.perf_counter() - start
        jax.block_until_ready(cache)
        out = _truncate(out, self.eos_ids, max_new_tokens)
        return (out, dict(decode_seconds=seconds)) if stats else out


def _truncate(ids, eos_ids, limit):
    for i, t in enumerate(ids):
        if t in eos_ids:
            ids = ids[:i + 1]
            break
    return ids[:limit]
