"""Jacobi pass math and completed-depth labels."""
from dataclasses import dataclass
import hashlib
from typing import NamedTuple

import torch

from .ops import reference_op
from .batching import prefix_lengths, sequence_slice, sequencewise
from .schedule import normalized_label, completed_label, resolve_tau


@dataclass
class PassValues:
    tokens: torch.Tensor
    fixed: torch.Tensor
    p_top1: torch.Tensor


@reference_op("jacobi_tokens")
@sequencewise('logits', 4)
def jacobi_tokens(logits, pinned_first=None, pinned_second=None, pins=None):
    """Argmax and token constraints, independent of pass bookkeeping."""
    tokens = logits.argmax(dim=-1)
    if pinned_first is not None:
        tokens[..., 0] = pinned_first
    if pinned_second is not None:
        tokens[..., 1] = pinned_second
    if pins is not None:
        tokens = apply_pins(tokens, pins)
    return tokens


@reference_op("apply_pins")
@sequencewise('tokens', 3)
def apply_pins(tokens, pins):
    if pins[0].ndim == 2:
        return torch.where(pins[0], pins[1], tokens)
    lane = torch.arange(tokens.shape[1], device=tokens.device)
    return torch.where(lane[None, :] == pins[0][:, None], pins[1][:, None], tokens)


@reference_op("jacobi_converged")
def jacobi_converged(tokens, inputs):
    return tokens.eq(inputs).all(dim=-1)


@reference_op("jacobi_pass")
@sequencewise('input_tokens', 3, shared=('head_weight',))
def reference_pass(logits, input_tokens, pinned_first=None, pinned_second=None, pins=None,
                   *, p_top1=None, head_weight=None, partials=None):
    """Argmax, pins, convergence and float32 softmax in served order."""
    if head_weight is not None:
        logits = torch.nn.functional.linear(logits, head_weight)
    tokens = jacobi_tokens(logits, pinned_first, pinned_second, pins)
    fixed = jacobi_converged(tokens, input_tokens)
    if p_top1 is None:
        p_top1 = logits.float().softmax(-1).amax(-1)
    return PassValues(tokens, fixed, p_top1)


@reference_op("j0_fill")
def j0_fill(context_ids=None, *, block_size, seed=None, prompt_index=0,
            init_window=None, indices=None, pinned_first=None, gather_vec=False,
            noise_key=None, rows=1, vocab=None, mask_token=None,
            out=None, active=None, finished=None, passes=None, err=None, context_lengths=None):
    """J0 draws: reader CPU RNG, captured context indices, or keyed lab noise.

    These draw policies retain their original generators and draw geometry.
    The captured path receives already drawn indices and never reads a device scalar.
    """
    if context_lengths is not None:
        lengths = prefix_lengths(context_ids, context_lengths)
        count = len(lengths)
        outputs = []
        for c, length in enumerate(lengths):
            context = context_ids[c, :length]
            if indices is None:
                context = context[None]
            outputs.append(j0_fill(
                context, block_size=block_size, seed=sequence_slice(seed, c, count),
                prompt_index=sequence_slice(prompt_index, c, count), init_window=init_window,
                indices=indices if indices is None or indices.ndim == 1 else indices[c],
                pinned_first=sequence_slice(pinned_first, c, count), gather_vec=gather_vec,
                noise_key=noise_key, rows=rows, vocab=vocab, mask_token=mask_token,
                out=sequence_slice(out, c, count), active=sequence_slice(active, c, count),
                finished=sequence_slice(finished, c, count), passes=sequence_slice(passes, c, count),
                err=sequence_slice(err, c, count)))
        return out if out is not None else torch.stack(outputs)
    if noise_key is not None:
        draws = []
        inside = 0 <= mask_token < vocab
        for row in range(rows):
            parts = (*noise_key, 0, 1, row)
            digest = hashlib.sha256(':'.join(str(int(part)) for part in parts).encode()).digest()
            generator = torch.Generator().manual_seed(
                int.from_bytes(digest[:8], 'little') & ((1 << 63) - 1))
            values = torch.randint(0, vocab - int(inside), (block_size,), generator=generator)
            if inside:
                values += (values >= mask_token).long()
            draws.append(values)
        return torch.stack(draws)
    if indices is not None:
        drawn = context_ids[indices] if gather_vec else context_ids.index_select(0, indices)
        values = torch.cat((pinned_first.clamp_min(0), drawn.expand(pinned_first.shape[0], -1)), dim=1)
        if out is not None:
            out.copy_(values)
            finished.copy_(~active)
            passes.zero_()
            if err is not None:
                err.zero_()
            return out
        return values
    context = context_ids.reshape(-1)
    if int(context.numel()) == 0:
        raise ValueError("context_ids must contain at least one token")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) ^ int(prompt_index))
    length = int(context.numel())
    sample_length = min(length, init_window) if init_window is not None else length
    sample_indices = torch.randint(sample_length, (int(block_size) - 1,), generator=generator)
    if sample_length < length:
        sample_indices += length - sample_length
    sampled = context.detach().cpu()[sample_indices].to(context_ids.device).reshape(1, -1)
    return torch.cat([context_ids[0, -1:].reshape(1, 1), sampled], dim=1)


class JacobiForcingBatch(NamedTuple):
    noise_input_ids: torch.Tensor
    target_ids: torch.Tensor
    anchor_positions: torch.Tensor
    block_keep_mask: torch.Tensor
    clean_prefix_lengths: torch.Tensor


@reference_op("jacobi_forcing_batch")
def jacobi_forcing_batch(
    *,
    input_ids: torch.Tensor,
    anchor_positions: torch.Tensor,
    block_keep_mask: torch.Tensor,
    clean_prefix_lengths: torch.Tensor,
    block_size: int,
    mask_token_id: int,
    generator: torch.Generator,
    noise_source: str = "context",
    past_end_noise: bool = False,
) -> JacobiForcingBatch:
    """Build one ordered suffix-noise copy for every sampled block.

    ``past_end_noise`` (anchor_rule=upstream) pads a block's slots past the
    sequence end with the noise source even inside the clean prefix: no token
    exists there, and upstream SpecForge 7757409 feeds its unknown-slot input
    (MASK) at those slots while the loss masks their labels.
    """
    if input_ids.ndim != 2:
        raise ValueError("input_ids must have shape [batch, sequence].")
    if anchor_positions.shape != block_keep_mask.shape:
        raise ValueError("anchor positions and block keep mask must have equal shape.")
    if clean_prefix_lengths.shape != anchor_positions.shape:
        raise ValueError("clean prefix lengths must contain one value per block.")
    if isinstance(block_size, bool) or not isinstance(block_size, int) or block_size < 2:
        raise ValueError("Jacobi forcing requires block_size >= 2.")
    if torch.any((clean_prefix_lengths < 0) | (clean_prefix_lengths >= block_size)):
        raise ValueError("clean prefix lengths must be in 0..block_size-1.")
    if torch.any(input_ids == mask_token_id):
        raise ValueError("Jacobi forcing requires mask-free committed context.")
    if not isinstance(noise_source, str) or noise_source not in {"context", "mask"}:
        raise ValueError("noise_source must be context or mask.")

    batch_size, sequence_length = input_ids.shape
    block_count = anchor_positions.shape[1]
    offsets = torch.arange(block_size, device=input_ids.device).view(1, 1, -1)
    target_indices = (anchor_positions.unsqueeze(-1) + offsets).clamp(
        max=sequence_length - 1
    )
    expanded_input = input_ids.unsqueeze(1).expand(-1, block_count, -1)
    targets = torch.gather(expanded_input, 2, target_indices)

    draft_slots = block_size - 1
    if noise_source == "context":
        context_sizes = (anchor_positions.clamp(max=sequence_length - 1) + 1).unsqueeze(-1)
        randoms = torch.rand(
            (batch_size, block_count, draft_slots),
            generator=generator,
            device=input_ids.device,
        )
        sampled_indices = (randoms * context_sizes.float()).long().clamp(
            max=sequence_length - 1
        )
        sampled_suffix = torch.gather(expanded_input, 2, sampled_indices)
    else:
        sampled_suffix = torch.full_like(targets[:, :, 1:], mask_token_id)
    noisy = targets.clone()
    draft_offsets = torch.arange(1, block_size, device=input_ids.device).view(1, 1, -1)
    suffix_mask = draft_offsets > clean_prefix_lengths.unsqueeze(-1)
    if past_end_noise:
        suffix_mask = suffix_mask | (
            anchor_positions.unsqueeze(-1) + draft_offsets >= sequence_length
        )
    noisy[:, :, 1:] = torch.where(
        suffix_mask,
        sampled_suffix,
        targets[:, :, 1:],
    )
    if noise_source == "context" and torch.any(noisy == mask_token_id):
        raise ValueError("Jacobi forcing produced a mask token; mask-free input is required.")

    return JacobiForcingBatch(
        noise_input_ids=noisy.reshape(batch_size, -1),
        target_ids=targets,
        anchor_positions=anchor_positions,
        block_keep_mask=block_keep_mask,
        clean_prefix_lengths=clean_prefix_lengths,
    )
