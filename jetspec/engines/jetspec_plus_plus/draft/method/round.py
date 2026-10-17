"""Executable account of admission, served segments, and miss recovery."""
from copy import copy
from dataclasses import InitVar, dataclass, field
from types import SimpleNamespace
from typing import Optional, Protocol

import torch

from .adapter import CtxKVC1Head
from .batching import prefix_lengths, sequence_slice, sequence_stack, sequencewise
from .cache_rows import after_miss, cache_rows_chain, hit_lookup
from .decoder.offers import lac_candidates
from .geometry import Geometry
from .ops import reference_op
from .passes import PassValues, j0_fill, reference_pass
from .refresh import refresh_bookkeeping, swap_write


class Target(Protocol):
    def tap(self, prefix: torch.Tensor) -> torch.Tensor: ...
    def verify(self, tokens: torch.Tensor) -> torch.Tensor: ...


class Adapter(Protocol):
    def tap(self, hidden: torch.Tensor) -> torch.Tensor: ...
    def block(self, ids: torch.Tensor, context: torch.Tensor) -> torch.Tensor: ...


class Drafter(Protocol):
    def __call__(self, ids: torch.Tensor, adapted: torch.Tensor) -> torch.Tensor: ...


@dataclass
class MethodState:
    prefix: torch.Tensor
    anchor: torch.Tensor
    width: int = 16
    seed: int = 9504
    context: Optional[torch.Tensor] = None
    ids: Optional[torch.Tensor] = None
    adapted: Optional[torch.Tensor] = None
    values: Optional[PassValues] = None
    finished: Optional[torch.Tensor] = None
    pins: Optional[tuple] = None
    passes_run: Optional[torch.Tensor] = None
    keys: Optional[torch.Tensor] = None
    requested: Optional[torch.Tensor] = None
    steps: list = field(default_factory=list)
    active: Optional[torch.Tensor] = None
    previous_top1: Optional[torch.Tensor] = None
    pending: Optional[torch.Tensor] = None
    prior_top1: Optional[torch.Tensor] = None
    offers: Optional[tuple] = None
    row_home: Optional[torch.Tensor] = None
    err: Optional[torch.Tensor] = None
    refresh_decisions: list = field(default_factory=list)
    round_id: int = 0
    candidate: Optional['Candidate'] = None
    redraft: Optional['Candidate'] = None
    missed: bool = False
    staged: Optional[torch.Tensor] = None
    prefix_lengths: InitVar[Optional[torch.Tensor]] = None

    def __post_init__(self, prefix_lengths):
        # InitVar keeps legacy state snapshots/serialization unchanged.
        self.prefix_lengths = prefix_lengths

    @classmethod
    def batch(cls, states):
        """Group independently admitted states, retaining their caches and round IDs.

        Prefix padding is storage only; prefix_lengths is always explicit.
        The supplied states remain the per-sequence owners and are advanced by run.
        """
        states = tuple(states)
        if not states or any(s.width != states[0].width or s.prefix_lengths is not None
                             for s in states):
            raise ValueError('batch requires unbatched states with a common width')
        result = cls(torch.nn.utils.rnn.pad_sequence([s.prefix.reshape(-1) for s in states],
                                                     batch_first=True),
                     torch.stack([s.anchor for s in states]), width=states[0].width,
                     seed=torch.tensor([int(s.seed) for s in states]),
                     prefix_lengths=torch.tensor([s.prefix.numel() for s in states]))
        result.sequence_states = list(states)
        if all(s.context is not None for s in states):
            result.context = tuple(s.context for s in states)
        return result


@dataclass(frozen=True)
class Candidate:
    tokens: torch.Tensor
    keys: torch.Tensor
    requested: torch.Tensor
    passes_run: torch.Tensor
    pins: Optional[tuple] = None
    refresh_decisions: tuple = ()
    p_top1: Optional[torch.Tensor] = None


@reference_op("adapter_tap")
def adapter_tap(adapter, target, prefix, *, lengths=None):
    if lengths is not None:
        valid = prefix_lengths(prefix, lengths)
        return tuple(adapter_tap(adapter, target, prefix[c, :length])
                     for c, length in enumerate(valid))
    if isinstance(adapter, CtxKVC1Head):
        return target.tap(prefix)
    return adapter.tap(target.tap(prefix))


@reference_op("adapter_block")
@sequencewise('ids', 3, shared=('adapter', 'embedding'))
def adapter_block(adapter, ids, context, *, embedding=None, tail=None, tail_counts=None,
                  context_lengths=None):
    if context_lengths is not None:
        context = context[:int(context_lengths)]
    if not isinstance(adapter, CtxKVC1Head):
        return adapter.block(ids, context)
    prefix = context.reshape(1, -1, context.shape[-1]).expand(ids.shape[0], -1, -1)
    length = prefix.shape[1]
    mask = None
    if tail is not None:
        prefix = torch.cat((prefix, tail.expand(ids.shape[0], -1, -1)), dim=1)
        visible = torch.arange(prefix.shape[1], device=ids.device)[None] < length + tail_counts[:, None]
        mask = torch.where(visible[:, None, None, :], 0., float('-inf'))
    positions = torch.arange(prefix.shape[1], device=ids.device)[None]
    starts = length if tail_counts is None else length + tail_counts[:, None]
    query = starts + torch.arange(ids.shape[1], device=ids.device)[None]
    return adapter(token_embeddings=embedding(ids), prefix_h=prefix,
                   prefix_position_ids=positions, query_position_ids=query,
                   attention_mask=mask)


@reference_op("draft_forward")
@sequencewise('ids', 3, shared=('drafter',))
def draft_forward(drafter, ids, adapted):
    return drafter(ids, adapted)


@reference_op("inject_hidden")
@sequencewise('previous', 4)
def inject_hidden(previous, refreshed, cutoff):
    slots = torch.arange(previous.shape[1], device=previous.device)
    return torch.where((slots[None, :] <= cutoff[:, None])[..., None], refreshed, previous)


class MethodRound:
    """Caller-supplied models; method homes own all step math.

    The first call admits a seed block. Subsequent calls take the verifier's
    true_k and correction for the previously published keys. A miss publishes
    a J0-only redraft before constructing the next hypotheses.

    Batched states carry prefix [C, L], prefix_lengths [C], anchor [C], and
    optionally seed [C]. Candidate tensors have C before their legacy axes.
    ``sequence_states`` retains ragged context, steps, redrafts and refresh
    telemetry per sequence. A schedule may be shared or a C-element tuple.
    Use MethodState.batch to combine states that have already been admitted.
    """
    def __init__(self, *, adapter, target, drafter, decoder, schedule, threshold=.1,
                 tau=0., temperature=1., cap=16, offer_width=32,
                 coefficients=(1., 0., 0.), method=None, knockout_mean=None, chain_binding=None):
        self.adapter, self.target, self.drafter, self.decoder = adapter, target, drafter, decoder
        self.schedule = schedule
        self.threshold, self.tau, self.temperature = threshold, tau, temperature
        self.cap, self.offer_width = cap, offer_width
        self.coefficients = coefficients
        self.chain = None
        if method is not None:
            from .config import _validate_fields
            from .config_schemas import METHOD_SCHEMA
            _validate_fields(method, METHOD_SCHEMA.fields, METHOD_SCHEMA.required, 'method')
            if method.get('round_protocol', 'served') == 'standalone-chain':
                from .chain import StandaloneChain
                self.chain = chain_binding if chain_binding is not None else StandaloneChain(
                    head=drafter, target=target, adapter=adapter, schedule=schedule,
                    method=method, knockout_mean=knockout_mean)

    def steps(self):
        return ('offers', 'cache_rows', 'context', *self.schedule.operations, 'publish')

    def _block(self, state, ids, *, hypotheses=False):
        native = getattr(self.adapter, 'head', self.adapter)
        adapter = native if isinstance(native, CtxKVC1Head) else self.adapter
        return adapter_block(adapter, ids, state.context,
                             embedding=getattr(self.drafter, 'embedding', None),
                             tail=state.staged if hypotheses else None,
                             tail_counts=state.keys[:, 0].clamp_min(0) + 1 if hypotheses else None)

    def _draw_indices(self, state):
        context = torch.arange(state.prefix.numel(), device=state.prefix.device)[None]
        return j0_fill(context, block_size=Geometry(state.width).block,
                       seed=state.seed)[0, 1:]

    def _seed(self, state):
        ids = j0_fill(state.prefix, block_size=state.width, indices=self._draw_indices(state),
                      pinned_first=state.anchor.reshape(1, 1))
        values = reference_pass(draft_forward(self.drafter, ids, self._block(state, ids)),
                                ids, state.anchor.reshape(1))
        return Candidate(values.tokens.clone(),
                         torch.stack((state.anchor.new_zeros(()), state.anchor)).reshape(1, 2),
                         state.anchor.new_ones(1), state.anchor.new_ones(1),
                         p_top1=values.p_top1.clone())

    def _admit(self, state, true_k, correction):
        state.context = adapter_tap(self.adapter, self.target, state.prefix)
        state.missed = False
        state.redraft = None
        if state.candidate is None:
            selected = self._seed(state)
            row = 0
        else:
            snapshot = SimpleNamespace(round_id=state.round_id,
                                       live_keys=state.candidate.keys,
                                       legacy_pins=None, redraft=False)
            state.missed = bool(after_miss(snapshot, round_id=state.round_id,
                                           true_k=true_k, recovery=correction))
            row = hit_lookup(state.candidate.keys, true_k, correction)
            if state.missed:
                state.anchor = state.anchor.new_tensor(correction)
                selected = self._seed(state)
                selected.keys[0, 0] = true_k
                state.redraft = selected
                row = 0
                state.steps.extend(('miss_redraft', 'redraft_J0', 'redraft_publish',
                                    'round_after_miss'))
            else:
                selected = state.candidate
                state.anchor = state.anchor.new_tensor(correction)
        state.pending = selected.tokens[row, 1:].clone()
        state.prior_top1 = selected.p_top1[row].clone()

    def _offers(self, state):
        ids = torch.cat((state.anchor.reshape(1), state.pending)).reshape(1, -1)
        state.staged = self._block(state, ids)
        _, logits = self.decoder(state.staged)
        logits = logits[0]
        rejected = torch.cat((state.pending, state.pending.new_full((1,), -1)))
        exclude = torch.zeros_like(logits, dtype=torch.bool).scatter_(
            1, rejected.clamp_min(0)[:, None], (rejected >= 0)[:, None])
        probabilities = logits.float().masked_fill(exclude, float('-inf')).softmax(-1).double()
        pool = 32
        offers_ids = torch.zeros(state.width, pool, dtype=torch.long, device=ids.device)
        offers_p = probabilities.new_zeros(state.width, pool)
        pack = torch.zeros(5, state.width, dtype=torch.long, device=ids.device)
        pack[3] = 1
        for k in range(state.width):
            excluded = (int(state.pending[k]),) if k < state.width - 1 else ()
            ranked, p = lac_candidates(probabilities[k], q_hat=1.,
                                       cap=min(pool, self.offer_width), excluded=excluded)
            offers_ids[k, :ranked.numel()] = ranked
            offers_p[k, :ranked.numel()] = p
            pack[0, k] = ranked.numel()
        state.offers = logits, offers_ids, offers_p, pack

    def _rows(self, state):
        logits, ids, probabilities, pack = state.offers
        device = state.prefix.device
        out = dict(keys=torch.empty(self.cap, 2, dtype=torch.long, device=device),
                   active=torch.empty(self.cap, dtype=torch.bool, device=device),
                   requested=torch.empty(1, dtype=torch.long, device=device),
                   row_home=torch.empty(3 + 2 * self.cap + 6 * state.width,
                                        dtype=torch.long, device=device),
                   err=torch.empty(1, dtype=torch.long, device=device))
        cache_rows_chain(state.pending, ids, probabilities, pack, logits, state.prior_top1,
                         self.cap, coefficients=self.coefficients,
                         miss=state.anchor.new_zeros(()), tau_idx=state.anchor.new_zeros(()),
                         correction=state.anchor, tau_table=logits.new_tensor([self.tau]).double(),
                         temperature=self.temperature, out=out)
        state.keys, state.active = out['keys'], out['active']
        state.requested, state.row_home, state.err = out['requested'], out['row_home'], out['err']

    def _initial(self, state):
        state.ids = torch.empty(self.cap, state.width, dtype=torch.long, device=state.prefix.device)
        state.finished = torch.empty_like(state.active)
        state.passes_run = torch.empty_like(state.active, dtype=torch.long)
        j0_fill(state.prefix, block_size=state.width, indices=self._draw_indices(state),
                pinned_first=state.keys[:, 1:2], out=state.ids, active=state.active,
                finished=state.finished, passes=state.passes_run)
        state.pins = (torch.zeros_like(state.ids, dtype=torch.bool), torch.zeros_like(state.ids))
        state.values = None

    def _jacobi(self, state, index):
        values = reference_pass(draft_forward(self.drafter, state.ids, state.adapted),
                                state.ids, state.keys[:, 1].clamp_min(0), pins=state.pins)
        was_finished = state.finished.clone()
        tokens, p = values.tokens, values.p_top1
        if index:
            tokens = torch.where(was_finished[:, None], state.values.tokens, tokens)
            p = torch.where(was_finished[:, None], state.values.p_top1, p)
        state.values = PassValues(tokens.masked_fill(~state.active[:, None], -1),
                                  values.fixed, p.masked_fill(~state.active[:, None], 0))
        state.passes_run.masked_fill_(~was_finished, index + 1)
        state.finished |= values.fixed
        state.previous_top1 = values.p_top1[:, 1:].clone()
        state.ids.copy_(values.tokens)

    def _refresh(self, state):
        cutoff, counts, _, _ = refresh_bookkeeping(
            state.previous_top1.double().cumprod(1), state.active, state.keys,
            state.anchor.new_tensor(state.prefix.numel()), Geometry(state.width).block,
            self.threshold, None)
        refreshed = self._block(state, state.ids, hypotheses=True)
        _, logits = self.decoder(refreshed)
        eligible = state.active & ~state.pins[0].any(1)
        decision = swap_write(logits[:, :-1], state.ids, cutoff, counts, eligible)
        used, slot, _, applied = decision
        state.pins[0].scatter_(1, slot[:, None], applied[:, None] |
                              state.pins[0].gather(1, slot[:, None]))
        state.pins[1].copy_(torch.where(state.pins[0], state.ids, state.pins[1]))
        state.finished &= ~applied
        state.adapted = inject_hidden(state.adapted, refreshed, used)
        state.values.tokens.copy_(state.ids.masked_fill(~state.active[:, None], -1))
        state.refresh_decisions.append(tuple(value.clone() for value in decision))

    @torch.inference_mode()
    def run(self, state, *, true_k=None, correction=None):
        if state.prefix_lengths is not None:
            return self._run_batched(state, true_k=true_k, correction=correction)
        if self.chain is not None:
            state.steps.clear()
            state.steps.append('context')
            state.chain_result = self.chain.run(state)
            result = state.chain_result
            tokens = torch.cat((state.anchor.reshape(1), result.final_tokens)).reshape(1, -1)
            state.candidate = Candidate(tokens, torch.stack((state.anchor.new_zeros(()),
                                                            state.anchor)).reshape(1, 2),
                                        state.anchor.new_ones(1),
                                        state.anchor.new_tensor([result.passes_run]))
            state.steps.append('publish')
            state.round_id += 1
            return state.candidate
        if state.candidate is not None and (true_k is None or correction is None):
            raise ValueError('the next round requires true_k and correction from verification')
        state.steps.clear()
        state.refresh_decisions.clear()
        self._admit(state, true_k, correction)
        state.steps.insert(0, 'admission')
        for step in self.steps():
            if step == 'offers':
                self._offers(state)
            elif step == 'cache_rows':
                self._rows(state)
            elif step == 'context':
                self._initial(state)
                state.adapted = self._block(state, state.ids, hypotheses=True)
            elif step.startswith('J'):
                self._jacobi(state, int(step[1:]))
            elif step.startswith('R'):
                self._refresh(state)
            elif step == 'publish':
                state.candidate = Candidate(
                    state.values.tokens.clone(), state.keys.clone(), state.requested.clone(),
                    state.passes_run.clone(), tuple(value.clone() for value in state.pins),
                    tuple(state.refresh_decisions), state.values.p_top1.clone())
            state.steps.append(step)
        state.round_id += 1
        return state.candidate

    def _run_batched(self, state, *, true_k, correction):
        lengths = prefix_lengths(state.prefix, state.prefix_lengths)
        count = len(lengths)
        if state.anchor.shape != (count,):
            raise ValueError('batched anchor must have shape [C]')
        schedules = self.schedule if isinstance(self.schedule, (tuple, list)) else (self.schedule,) * count
        if len(schedules) != count:
            raise ValueError('one schedule is required per sequence')
        if not hasattr(state, 'sequence_states'):
            state.sequence_states = [MethodState(state.prefix[c, :length], state.anchor[c],
                                                 width=state.width,
                                                 seed=sequence_slice(state.seed, c, count))
                                     for c, length in enumerate(lengths)]
        if len(state.sequence_states) != count:
            raise ValueError('C cannot change on an admitted state')
        if not hasattr(state, 'sequence_methods'):
            state.sequence_methods = []
            for schedule in schedules:
                method = copy(self)
                method.schedule = schedule
                if self.chain is not None:
                    method.chain = copy(self.chain)
                    method.chain.schedule = schedule
                    method.chain.reset_cache()
                    method.chain._fidelity = {}
                state.sequence_methods.append(method)
        candidates = []
        for c, (length, single, method) in enumerate(zip(lengths, state.sequence_states,
                                                       state.sequence_methods)):
            single.prefix, single.anchor = state.prefix[c, :length], state.anchor[c]
            single.seed = sequence_slice(state.seed, c, count)
            method.schedule = schedules[c]
            if method.chain is not None:
                method.chain.schedule = schedules[c]
                if state.context is None:
                    raise ValueError('standalone chain requires exact target hidden states')
                context = state.context[c]
                single.context = context.reshape(1, -1, context.shape[-1])[:, :length]
            candidates.append(method.run(single,
                                          true_k=None if true_k is None else int(sequence_slice(true_k, c, count)),
                                          correction=None if correction is None else int(sequence_slice(correction, c, count))))
        state.candidate = Candidate(
            *(torch.stack([getattr(candidate, name) for candidate in candidates])
              for name in ('tokens', 'keys', 'requested', 'passes_run')),
            pins=None if candidates[0].pins is None else sequence_stack([v.pins for v in candidates]),
            refresh_decisions=tuple(v.refresh_decisions for v in candidates),
            p_top1=None if candidates[0].p_top1 is None else torch.stack([v.p_top1 for v in candidates]))
        for name in ('anchor', 'ids', 'adapted', 'values', 'finished', 'pins', 'passes_run',
                     'keys', 'requested', 'active', 'previous_top1', 'pending', 'prior_top1',
                     'offers', 'row_home', 'err', 'staged'):
            values = [getattr(single, name) for single in state.sequence_states]
            if values[0] is not None:
                setattr(state, name, sequence_stack(values))
        state.missed = torch.tensor([s.missed for s in state.sequence_states], device=state.prefix.device)
        state.round_id = torch.tensor([s.round_id for s in state.sequence_states], device=state.prefix.device)
        for name in ('steps', 'refresh_decisions', 'redraft'):
            setattr(state, name, tuple(getattr(s, name) for s in state.sequence_states))
        if self.chain is None:
            state.context = tuple(s.context for s in state.sequence_states)
        return state.candidate
