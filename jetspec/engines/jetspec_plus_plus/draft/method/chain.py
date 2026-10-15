"""Standalone Jacobi policy used by MethodRound's offline model binding."""
from typing import NamedTuple
import math

import torch

from .passes import j0_fill, jacobi_tokens, jacobi_converged
from .adapter import build_adapter_query_mask
from .refresh import refresh_cutoff


class PassTelemetry(NamedTuple):
    pass_index: int
    input_tokens: tuple
    edit_rate: float


class DraftForwardPass(NamedTuple):
    logits: torch.Tensor
    input_tokens: torch.Tensor


class RefinementResult(NamedTuple):
    logits: torch.Tensor
    final_tokens: torch.Tensor
    passes_run: int
    converged: bool
    telemetry: tuple


def refine(forward, *, depth, max_passes, convergence_limit=64):
    """Stop on the first unchanged block, retaining the actual forward count."""
    if depth < 1:
        raise ValueError('depth must be positive')
    if max_passes is not None and max_passes < 1:
        raise ValueError('max_passes must be positive or None')
    limit = convergence_limit if max_passes is None else max_passes
    if limit < 1:
        raise ValueError('convergence_limit must be positive')
    telemetry = []
    current = None
    for index in range(limit):
        result = forward(fill_tokens=current, pass_index=index)
        if not isinstance(result, DraftForwardPass):
            raise TypeError('forward must return DraftForwardPass')
        if result.logits.shape[:2] != (1, depth):
            raise ValueError(f'forward must return [1, {depth}, vocab]')
        inputs = result.input_tokens.reshape(-1)
        if inputs.shape != (depth,):
            raise ValueError(f'forward input tokens must have shape [{depth}]')
        current = jacobi_tokens(result.logits.squeeze(0))
        telemetry.append(PassTelemetry(index, tuple(int(t) for t in inputs.detach().cpu().tolist()),
                                       float((current != inputs).float().mean().item())))
        fixed = bool(jacobi_converged(current, inputs))
        if fixed:
            break
    return RefinementResult(result.logits, current, len(telemetry), fixed, tuple(telemetry))


class StandaloneChain:
    """One model binding; context and refresh variants are method policies."""
    def __init__(self, *, head, target, adapter, schedule, method, knockout_mean=None):
        self.head, self.target, self.adapter = head, target, adapter
        self.method, self.schedule = method, schedule
        self.device = next(head.parameters()).device
        self.dtype = next(head.parameters()).dtype
        options = head.config.dflash_config
        self.causal = options.get('causal_head')
        if not isinstance(self.causal, bool):
            raise ValueError('dflash_config.causal_head must be a boolean')
        if any('clean_prefix_embedding' in name for name, _ in head.named_parameters()):
            raise AssertionError('p-free Jacobi serving forbids clean_prefix_embedding')
        self.noise_source = method.get('noise', {}).get('source', options.get('noise_source', 'context'))
        if self.noise_source not in ('context', 'mask'):
            raise ValueError('noise source must be context or mask')
        self.block_size = method['block_size']
        self.prompt_index = 0
        self.round_results = []
        self.round_context_lengths = []
        context = method.get('context', {})
        self.context_mode = context.get('mode', 'fresh')
        self.stale_fill = context.get('fill', 'adapter')
        self.staged = None
        self.adapted_counts = []
        self._fidelity = {}
        self.tap_ids = list(options['target_layer_ids'])
        self.tap_width = int(getattr(getattr(target, 'config', None), 'hidden_size', head.config.hidden_size))
        self.refresh_mode = method.get('refresh', {}).get('mode', 'off')
        if self.refresh_mode not in ('off', 'after_j0', 'after_j0_j1'):
            raise ValueError('standalone refresh mode must be off, after_j0, or after_j0_j1')
        if self.refresh_mode != 'off' and self.context_mode != 'one_round':
            raise ValueError('context refresh requires one-round stale context')
        if self.context_mode == 'one_round' and self.stale_fill == 'adapter' and adapter is None:
            raise ValueError('one-round adapter context requires an adapter')
        if self.stale_fill == 'copy_previous' and self.refresh_mode != 'off':
            raise ValueError('copy_previous requires refresh off')
        needed = {'off': 0, 'after_j0': 1, 'after_j0_j1': 2}[self.refresh_mode]
        if schedule.j < needed:
            raise ValueError('context refresh requires enough Jacobi passes')
        if schedule.r != needed:
            raise ValueError('refresh mode and schedule must request the same refresh count')
        if self.block_size < 2:
            raise ValueError('standalone chain requires a block of at least two tokens')
        self.refresh_khats = []
        knockout = method.get('tap_knockout', {})
        layers = knockout.get('layers', [])
        if len(set(layers)) != len(layers) or any(tap not in self.tap_ids for tap in layers):
            raise ValueError(f'tap knockout must name distinct target taps: {self.tap_ids}')
        self.knockout_slices = tuple(self.tap_ids.index(tap) for tap in layers)
        self.knockout_value = knockout.get('value', 'mean')
        self.knockout_mean = knockout_mean
        if layers and self.knockout_value == 'mean' and (
                knockout_mean is None or knockout_mean.shape != (len(self.tap_ids), self.tap_width)
                or knockout_mean.dtype != torch.float32):
            raise ValueError('knockout mean must have one fp32 vector per target tap')

    def reset_cache(self):
        self.round_results = []
        self.round_context_lengths = []
        self.staged = None
        self.adapted_counts = []
        self.refresh_khats = []

    def _fresh_context(self, exact):
        ids = self.context_ids
        exact = exact.to(device=self.device, dtype=self.dtype)
        indicator = torch.zeros(exact.shape[:2], dtype=torch.long, device=exact.device)
        if self.context_mode == 'one_round' and self.staged is not None:
            start, tokens, predicted = self.staged
            delta = exact.shape[1] - start
            if not 1 <= delta <= tokens.shape[1]:
                raise RuntimeError('one-round adapter tail outside staged range')
            if not torch.equal(ids[:, start:start + delta].to(tokens.device), tokens[:, :delta]):
                raise RuntimeError('one-round adapter token mismatch after verification')
            self.observe_fidelity(predicted, exact[:, start:start + delta, :], delta)
            exact = exact.clone()
            exact[:, -delta:, :] = predicted[:, :delta, :].to(exact)
            indicator[:, -delta:] = 1
        self.adapted_counts.append(int(indicator.sum().item()))
        if self.knockout_slices:
            exact = exact.clone()
            for index in self.knockout_slices:
                value = (self.knockout_mean[index].to(device=self.device, dtype=self.dtype)
                         if self.knockout_value == 'mean' else 0)
                exact[..., index * self.tap_width:(index + 1) * self.tap_width] = value
        embedding = getattr(self.head, 'adapted_position_embedding', None)
        if self.context_mode == 'one_round' and embedding is not None:
            exact = exact + embedding(indicator).to(exact.dtype)
        return exact

    def observe_fidelity(self, predicted, exact, delta):
        for position in range(1, delta):
            for index, tap in enumerate(self.tap_ids):
                start, end = index * self.tap_width, (index + 1) * self.tap_width
                actual = exact[0, position, start:end].float()
                estimate = predicted[0, position, start:end].float()
                cosine = float(torch.nn.functional.cosine_similarity(estimate[None], actual[None]).item())
                relative = float((torch.linalg.vector_norm(estimate - actual) /
                                  torch.linalg.vector_norm(actual)).item())
                if not math.isfinite(relative):
                    raise ValueError('target hidden tap has zero norm; relative L2 is undefined')
                old = self._fidelity.get((position, tap), (0., 0., 0))
                self._fidelity[(position, tap)] = old[0] + cosine, old[1] + relative, old[2] + 1


    def _predict(self, *, prefix, tokens):
        length, count = prefix.shape[1], tokens.shape[1]
        if self.stale_fill == 'copy_previous':
            return prefix[:, -1:, :].expand(-1, count, -1).detach().clone()
        positions = torch.arange(length + count, dtype=torch.long, device=self.device)
        window = bool(getattr(self.adapter, 'resolved_config', {}).get('window_causal_over_tokens', False))
        return self.adapter(token_embeddings=self.target.model.embed_tokens(tokens),
                            h_prev=prefix[:, -1:, :].expand(-1, count, -1), prefix_h=prefix,
                            prefix_position_ids=positions[:length][None],
                            query_position_ids=positions[length:][None],
                            attention_mask=build_adapter_query_mask(
                                prefix_count=length, query_count=count, dtype=prefix.dtype,
                                device=self.device, window_causal=window)).detach()

    def block_ids(self, context_ids, fill_tokens):
        anchor = context_ids[:, -1:].to(self.device)
        if fill_tokens is not None:
            fill = fill_tokens.reshape(1, -1).to(device=self.device, dtype=anchor.dtype)
            if fill.shape[1] != self.block_size - 1:
                raise ValueError('later Jacobi passes must feed all draft slots')
            ids = torch.cat((anchor, fill), 1)
        elif self.noise_source == 'mask':
            ids = torch.cat((anchor, anchor.new_full((1, self.block_size - 1),
                                                     int(self.head.mask_token_id))), 1)
        else:
            ids = j0_fill(context_ids, block_size=self.block_size,
                          seed=self.method.get('noise', {}).get('seed', 9504),
                          prompt_index=self.prompt_index,
                          init_window=self.method.get('noise', {}).get('init_window')).to(self.device)
        if (fill_tokens is not None or self.noise_source == 'context') and bool(
                (ids == int(self.head.mask_token_id)).any()):
            raise ValueError('Jacobi serving input must be mask-free')
        return ids

    def forward(self, ids, context, *, positions=None, mask=None):
        if positions is None:
            positions = torch.arange(context.shape[1] + self.block_size,
                                     dtype=torch.long, device=self.device)[None]
        kwargs = {} if mask is None else {'attention_mask': mask}
        hidden = self.head(target_hidden=context, noise_embedding=self.target.model.embed_tokens(ids),
                           position_ids=positions, past_key_values=None, use_cache=False,
                           is_causal=self.causal, attention_backend='sdpa', **kwargs)
        return DraftForwardPass(self.target.lm_head(hidden[:, 1:, :]), ids[:, 1:])

    def _refresh_block(self, *, target_hidden, context_ids, fill_tokens, logits, again):
        options = self.method.get('refresh', {})
        first = options.get('threshold', .1)
        threshold = options.get('second_threshold', first) if again else first
        probabilities = torch.softmax(logits, -1, dtype=torch.float32).amax(-1)
        khat = int(refresh_cutoff(probabilities, threshold)[0].item())
        cap = options.get('second_cap') if again else None
        if cap is not None:
            khat = min(khat, cap)
        self.refresh_khats.append(khat)
        block = torch.cat((context_ids[:, -1:].to(self.device),
                           fill_tokens.reshape(1, -1).to(self.device)), 1)
        exact = target_hidden.to(device=self.device, dtype=self.dtype)
        if self.staged is None:
            prefix, tail = exact, block[:, :0]
        else:
            start, staged_tokens, _ = self.staged
            delta = exact.shape[1] - start
            prefix, tail = exact[:, :start], staged_tokens[:, :delta].to(self.device)
        predicted = self._predict(prefix=prefix, tokens=torch.cat((tail, block), 1))
        return predicted[:, tail.shape[1]:tail.shape[1] + khat + 1, :], khat

    def forward_refreshed(self, ids, context, refreshed):
        base, count = context.shape[1], refreshed.shape[1]
        positions = torch.cat((torch.arange(base + count, device=self.device),
                               torch.arange(base, base + self.block_size, device=self.device)))[None]
        allowed = torch.ones(self.block_size, base + count + self.block_size,
                             dtype=torch.bool, device=self.device)
        slots = torch.arange(self.block_size, device=self.device)
        allowed[:, base:base + count] = slots[:, None] > torch.arange(count, device=self.device)[None]
        mask = torch.zeros((1, 1, self.block_size, allowed.shape[1]), dtype=context.dtype,
                           device=self.device).masked_fill_(~allowed[None, None],
                                                           torch.finfo(context.dtype).min)
        return self.forward(ids, torch.cat((context, refreshed), 1), positions=positions, mask=mask)

    def run(self, state):
        if state.context is None:
            raise ValueError('standalone chain requires exact target hidden states')
        context_ids = state.prefix.reshape(1, -1)
        self.context_ids = context_ids
        context = self._fresh_context(state.context)
        last_logits, refreshed = None, None

        def forward(*, fill_tokens, pass_index):
            nonlocal last_logits, refreshed
            if pass_index == 1 and self.refresh_mode != 'off':
                refreshed, _ = self._refresh_block(target_hidden=state.context, context_ids=context_ids,
                                                   fill_tokens=fill_tokens, logits=last_logits, again=False)
                state.steps.append('R1')
            elif pass_index == 2 and self.refresh_mode == 'after_j0_j1':
                refreshed, _ = self._refresh_block(target_hidden=state.context, context_ids=context_ids,
                                                   fill_tokens=fill_tokens, logits=last_logits, again=True)
                state.steps.append('R2')
            state.steps.append(f'J{pass_index}')
            ids = self.block_ids(context_ids, fill_tokens)
            result = (self.forward(ids, context) if refreshed is None
                      else self.forward_refreshed(ids, context, refreshed))
            last_logits = result.logits
            return result

        result = refine(forward, depth=self.block_size - 1,
                        max_passes=None if self.method.get('converge_until_fixed', False)
                        else self.schedule.j + 1,
                        convergence_limit=self.method.get('convergence_limit', 64))
        if self.context_mode == 'one_round':
            prefix = state.context.to(device=self.device, dtype=self.dtype)
            tokens = torch.cat((context_ids[:, -1:].to(self.device),
                                result.final_tokens.reshape(1, -1).to(self.device)), 1)
            self.staged = prefix.shape[1], tokens.detach().clone(), self._predict(prefix=prefix, tokens=tokens)
        self.round_results.append(result)
        self.round_context_lengths.append(context_ids.shape[1])
        return result
