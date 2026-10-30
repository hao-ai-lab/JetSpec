"""Ported from the JetSpec++ reference implementation.

Public API:
    OneRoundAdaptedContext(max_adapted=16): stage(...), consume(hidden, ids), clear().
    build_hypothesis_contexts(exact_hidden, *, predicted_hidden, k_hypotheses,
                              adapted_position_embedding) -> (contexts, indicators).
    build_protocol_hypothesis_contexts(exact_hidden, *, predicted_hidden,
        k_hypotheses, adapted_position_embedding, block_size) -> (contexts, mask, positions).
    build_random_context_block_token_ids(context_ids, *, block_size, seed,
        prompt_index, mode='token', device_gather=False) -> [1, block_size] ids.

J0's kth mode draws block_size-1 indices with a private CPU generator seeded by
seed XOR prompt_index, then prepends committed[-2]. Staged adapter predictions
remain subject to the existing one-round lag and token check.
"""
from collections.abc import Sequence
from typing import Literal
import torch
from torch import nn

__all__ = ['OneRoundAdaptedContext', 'build_hypothesis_contexts',
           'build_protocol_hypothesis_contexts', 'build_random_context_block_token_ids']


def build_random_context_block_token_ids(
    context_ids: torch.Tensor,
    *,
    block_size: int,
    seed: int,
    prompt_index: int,
    mode: Literal['token', 'noise', 'kth'] = 'token',
    device_gather: bool = False,
) -> torch.Tensor:
    """Build the anchor plus its paired random-context draft slots."""
    if mode not in {'token', 'noise', 'kth'}:
        raise ValueError('mode must be token, noise, or kth')
    context = context_ids.reshape(-1)
    if int(context.numel()) == 0:
        raise ValueError('context_ids must contain at least one token')
    generator = torch.Generator(device='cpu')
    generator.manual_seed(int(seed) ^ int(prompt_index))
    sample_indices = torch.randint(int(context.numel()), (int(block_size) - 1,),
                                   generator=generator)
    if device_gather:
        if context_ids.device.type == 'cuda':
            sample_indices = sample_indices.pin_memory().to(context_ids.device, non_blocking=True)
        else:
            sample_indices = sample_indices.to(context_ids.device)
        sampled = context.detach().index_select(0, sample_indices).reshape(1, -1)
    else:
        sampled = context.detach().cpu()[sample_indices].to(context_ids.device).reshape(1, -1)
    if mode == 'noise':
        slot_zero_index = torch.randint(int(context.numel()), (1,), generator=generator)
        if device_gather:
            slot_zero = context.detach().index_select(0, slot_zero_index.to(context_ids.device)).reshape(1, 1)
        else:
            slot_zero = context.detach().cpu()[slot_zero_index].to(context_ids.device).reshape(1, 1)
        return torch.cat([slot_zero, sampled], dim=1)
    if mode == 'kth':
        if int(context_ids.shape[1]) < 2:
            raise ValueError('kth mode requires a token before the correction')
        return torch.cat([context_ids[:, -2:-1], sampled], dim=1)
    return torch.cat([context_ids[0, -1:].reshape(1, 1), sampled], dim=1)


class OneRoundAdaptedContext:
    """Carry the adapter-predicted hidden tail into exactly the next round."""

    def __init__(self, max_adapted: int=16) -> None:
        self.max_adapted = int(max_adapted)
        self.clear()

    def clear(self) -> None:
        self.start_position: int | None = None
        self.token_ids: torch.Tensor | None = None
        self.predicted_hidden: torch.Tensor | None = None

    def stage(self, *, start_position: int, token_ids: torch.Tensor, predicted_hidden: torch.Tensor) -> None:
        if token_ids.ndim != 2 or predicted_hidden.ndim != 3:
            raise ValueError('staged tokens/hidden must have shapes [1, q] and [1, q, h]')
        if token_ids.shape[:2] != predicted_hidden.shape[:2]:
            raise ValueError('staged token and hidden lengths differ')
        self.start_position = int(start_position)
        self.token_ids = token_ids.detach().clone()
        self.predicted_hidden = predicted_hidden.detach().clone()

    def consume(self, exact_hidden: torch.Tensor, context_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int]:
        full_length = int(exact_hidden.shape[1])
        indicator = torch.zeros(exact_hidden.shape[:2], dtype=torch.long, device=exact_hidden.device)
        if self.start_position is None:
            return (exact_hidden, indicator, 0)
        assert self.token_ids is not None and self.predicted_hidden is not None
        delta = full_length - self.start_position
        if delta < 1 or delta > int(self.token_ids.shape[1]):
            raise RuntimeError(f'one-round adapter tail length {delta} is outside staged range 1..{self.token_ids.shape[1]}')
        actual = context_ids[:, self.start_position:full_length].to(self.token_ids.device)
        expected = self.token_ids[:, :delta]
        if not torch.equal(actual, expected):
            raise RuntimeError('one-round adapter token mismatch after target verification')
        adapted_count = min(delta, self.max_adapted)
        staged_start = delta - adapted_count
        adapted = exact_hidden.clone()
        adapted[:, -adapted_count:, :] = self.predicted_hidden[:, staged_start:delta, :].to(device=exact_hidden.device, dtype=exact_hidden.dtype)
        indicator[:, -adapted_count:] = 1
        return (adapted, indicator, adapted_count)

def build_hypothesis_contexts(exact_hidden: torch.Tensor, *, predicted_hidden: torch.Tensor | None, k_hypotheses: Sequence[int], adapted_position_embedding: nn.Module) -> tuple[torch.Tensor, torch.Tensor]:
    """Build fixed-length stale contexts for every possible accepted-draft count."""
    if exact_hidden.ndim != 3 or int(exact_hidden.shape[0]) != 1:
        raise ValueError('exact_hidden must have shape [1, positions, hidden]')
    hypotheses = [int(value) for value in k_hypotheses]
    contexts = exact_hidden.expand(len(hypotheses), -1, -1).clone()
    indicators = torch.zeros(contexts.shape[:2], dtype=torch.long, device=contexts.device)
    if predicted_hidden is not None:
        predicted = predicted_hidden.to(device=contexts.device, dtype=contexts.dtype)
        for (row, hypothesis) in enumerate(hypotheses):
            count = hypothesis + 1
            if count < 1 or count > int(predicted.shape[1]):
                raise ValueError('hypothesis exceeds the staged adapter window')
            if count > int(contexts.shape[1]):
                raise ValueError('hypothesis exceeds the available target context')
            contexts[row, -count:, :] = predicted[0, :count, :]
            indicators[row, -count:] = 1
    marked = contexts + adapted_position_embedding(indicators).to(dtype=contexts.dtype)
    return (marked, indicators)

def build_protocol_hypothesis_contexts(exact_hidden: torch.Tensor, *, predicted_hidden: torch.Tensor, k_hypotheses: Sequence[int], adapted_position_embedding: nn.Module, block_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Batch the rail-equivalent variable-length context for each next-round k."""
    if exact_hidden.ndim != 3 or int(exact_hidden.shape[0]) != 1:
        raise ValueError('exact_hidden must have shape [1, positions, hidden]')
    if predicted_hidden.ndim != 3 or int(predicted_hidden.shape[0]) != 1:
        raise ValueError('predicted_hidden must have shape [1, positions, hidden]')
    hypotheses = [int(value) for value in k_hypotheses]
    if not hypotheses or any((value < 0 for value in hypotheses)):
        raise ValueError('k_hypotheses must be non-empty and non-negative')
    max_hypothesis = max(hypotheses)
    if max_hypothesis > int(predicted_hidden.shape[1]):
        raise ValueError('hypothesis exceeds the staged adapter window')
    predicted = predicted_hidden.to(device=exact_hidden.device, dtype=exact_hidden.dtype)
    exact_length = int(exact_hidden.shape[1])
    max_context_length = exact_length + max_hypothesis
    hidden_size = int(exact_hidden.shape[2])
    contexts = exact_hidden.new_zeros((len(hypotheses), max_context_length, hidden_size))
    valid_context = torch.zeros((len(hypotheses), max_context_length), dtype=torch.bool, device=exact_hidden.device)
    position_ids = torch.zeros((len(hypotheses), max_context_length + int(block_size)), dtype=torch.long, device=exact_hidden.device)
    for (row, hypothesis) in enumerate(hypotheses):
        if hypothesis:
            context = torch.cat([exact_hidden[0], predicted[0, :hypothesis]], dim=0)
            indicator = torch.cat([torch.zeros(exact_length, dtype=torch.long, device=exact_hidden.device), torch.ones(hypothesis, dtype=torch.long, device=exact_hidden.device)])
        else:
            context = exact_hidden[0]
            indicator = torch.zeros(exact_length, dtype=torch.long, device=exact_hidden.device)
        marked = context + adapted_position_embedding(indicator).to(dtype=exact_hidden.dtype)
        context_length = int(marked.shape[0])
        padding = max_context_length - context_length
        contexts[row, padding:] = marked
        valid_context[row, padding:] = True
        position_ids[row, padding:max_context_length] = torch.arange(context_length, dtype=torch.long, device=exact_hidden.device)
        position_ids[row, max_context_length:] = torch.arange(context_length, context_length + int(block_size), dtype=torch.long, device=exact_hidden.device)
    valid_noise = torch.ones((len(hypotheses), int(block_size)), dtype=torch.bool, device=exact_hidden.device)
    attention_mask = torch.cat([valid_context, valid_noise], dim=1)
    return (contexts, attention_mask[:, None, None, :], position_ids)
