"""Ported from the JetSpec++ reference implementation."""
from dataclasses import dataclass
from collections import Counter
from types import SimpleNamespace
import re

import torch

from .method.schedule import Schedule
from .method.passes import reference_pass, draft_forward
from .method.refresh import refresh_cutoff, inject
from .method.swap import RefreshSwap
from .context import build_random_context_block_token_ids
from .models.adapter import build_head_attention_mask
from .models.decoder import refresh_decoder_logits
from .models.drafter import (
    resolve_stock_drafter, resolve_causal_head, check_stock_drafter_keys, StockAdaptedMark,
)


def completed_label(passes_run):
    """Normalize the computed pass count using the reference's calibration labels."""
    if passes_run < 1:
        raise ValueError('a served block must have at least one computed pass')
    return 'J0d0' if passes_run == 1 else f'J{passes_run - 1}'


def resolve_tau(table, label):
    """Resolve the explicit pass label, then the reference's pooled fallback."""
    if re.fullmatch(r'J0d0|J[1-9][0-9]*', label) is None:
        raise ValueError(f'invalid normalized pass label {label!r}')
    key = label if label in table else 'pooled'
    if key not in table:
        raise ValueError(f'missing tau for {label}; no explicit label or pooled key')
    value = table[key]
    return float(value['value'] if isinstance(value, dict) else value)


@dataclass(frozen=True)
class JetSpecPPConfig:
    schedule: str = 'J1-R1'
    block_size: int = 16
    seed: int = 9504
    refresh_threshold: float = 0.1
    second_refresh_threshold: float = 0.1
    second_refresh_cap: int | None = None
    rows_min: int = 8
    rows_cap: int = 16
    anchor_cap: int = 32
    anchor_coverage: float = 0.99
    pin_agreed: bool = True
    refresh_full: bool = False
    score_all_anchors: bool = False
    reject_after_miss: bool = True
    rows_fill_zero: bool = False
    freeze_converged: bool = False

    @classmethod
    def paper_eval(cls, *, schedule='J1-R1', seed=9504):
        """The uncut B16 pipelined eval profile, with sixteen keyed rows."""
        return cls(schedule=schedule, seed=seed, rows_min=16, refresh_full=True,
                   score_all_anchors=True, reject_after_miss=False, rows_fill_zero=True,
                   freeze_converged=True)

    def __post_init__(self):
        Schedule.parse(self.schedule)
        if self.block_size != 16:
            raise ValueError('the Qwen3-8B method requires B16')
        if not 0 < self.refresh_threshold < 1 or not 0 < self.second_refresh_threshold < 1:
            raise ValueError('refresh thresholds must be in (0, 1)')
        if not 0 <= self.rows_min <= self.rows_cap:
            raise ValueError('rows_min must be between zero and rows_cap')
        if self.second_refresh_cap is not None and not 0 <= self.second_refresh_cap <= 15:
            raise ValueError('second_refresh_cap must be in 0..15')

    @property
    def operations(self):
        return Schedule.parse(self.schedule).operations


@dataclass
class RoundResult:
    ids: torch.Tensor
    logits: torch.Tensor
    p_top1: torch.Tensor
    predicted: torch.Tensor
    label: str
    operations: tuple
    draft_cutoffs: tuple
    converged: bool = False
    effective_passes: int | None = None
    pinned: torch.Tensor | None = None

    @property
    def passes_run(self):
        if self.effective_passes is not None:
            return self.effective_passes
        return sum(operation.startswith('J') for operation in self.operations)


class RoundDriver:
    """Connect the served tensor steps without caches in the draft model."""
    def __init__(self, target, head, adapter, decoder, config):
        options = head.config.dflash_config
        stock = resolve_stock_drafter(options, 'drafter config')
        resolve_causal_head(options, 'drafter config')
        if not stock or not getattr(head, 'jacobi_forcing', False):
            raise ValueError('expected a stock Jacobi-forcing drafter')
        check_stock_drafter_keys(head, head.state_dict(), 'drafter model')
        parameter = next(head.parameters())
        head.adapted_position_embedding = StockAdaptedMark(
            head.fc.in_features, dtype=parameter.dtype, device=parameter.device)
        self.target, self.head = target, head
        self.embed_tokens, self.lm_head = target.model.embed_tokens, target.lm_head
        self.adapter, self.decoder, self.config = adapter, decoder, config
        self.counts = Counter()

    @torch.inference_mode()
    def predict(self, prefix, tokens):
        # The reference casts taps before adapter use.
        prefix = prefix.to(next(self.adapter.parameters()))
        length, count = prefix.shape[1], tokens.shape[1]
        device = prefix.device
        positions = torch.arange(length + count, device=device)[None]
        mask = build_head_attention_mask(
            head=self.adapter, prefix_position_ids=positions[:, :length],
            query_position_ids=positions[:, length:],
            block_anchors=torch.full((1, count), length, device=device),
            dtype=prefix.dtype, device=device)
        return self.adapter(token_embeddings=self.embed_tokens(tokens),
                            prefix_h=prefix, h_prev=prefix[:, -1:],
                            prefix_position_ids=positions[:, :length],
                            query_position_ids=positions[:, length:], attention_mask=mask).detach()

    @torch.inference_mode()
    def run(self, context_ids, exact, *, prompt_index=0, anchor=None,
            target_context=None, tail_tokens=None, tail_hidden=None, admission=False, sequence=None):
        config = self.config
        device = exact.device
        if sequence is None:
            sequence = SimpleNamespace(taps=exact[0].reshape(-1, 5, self.target.config.hidden_size))
        # The reference casts before draft context construction.
        exact = exact.to(next(self.head.parameters()))
        anchor = context_ids[0, -1] if anchor is None else torch.as_tensor(anchor, device=device)
        ids = build_random_context_block_token_ids(
            context_ids, block_size=16, seed=config.seed, prompt_index=prompt_index, mode='kth')
        # Anchor0 rows replace the kth predecessor with the known correction, as in admission.
        ids[:, 0] = anchor
        if target_context is None:
            indicator = torch.zeros(exact.shape[:2], dtype=torch.long, device=device)
            target_context = exact + self.head.adapted_position_embedding(indicator).to(exact.dtype)
        tail_tokens = context_ids[:, :0] if tail_tokens is None else tail_tokens
        tail_count = tail_tokens.shape[1]
        self.decoder._prefix(sequence, max(0, exact.shape[1] - 4))
        swap = RefreshSwap(rows=1, lanes=16, device=device, pin_agreed=config.pin_agreed)
        swap.configure_pins([op for op in config.operations if op.startswith('R')], rows=1, lanes=16)
        swap.reset()
        finished = torch.zeros(1, dtype=torch.bool, device=device)
        refreshed, logits, p_top1 = None, None, None
        final_ids, final_pins, effective_passes = None, None, 0
        cutoffs, passes = [], 0
        for operation in config.operations:
            if operation.startswith('J'):
                if refreshed is None:
                    context = target_context
                    positions = torch.arange(context.shape[1] + 16, device=device)[None]
                    mask = None
                else:
                    context, positions, mask = inject(target_context, refreshed[0], refreshed[1], block_size=16)
                forward = draft_forward(self.head, self.embed_tokens,
                                        self.lm_head, ids, context,
                                        position_ids=positions, attention_mask=mask)
                values = reference_pass(forward.logits, ids, anchor, pins=swap.pins())
                # Fixed launches keep the last effective output until a swap reopens it.
                if not config.freeze_converged or not bool(finished[0]):
                    final_ids, logits, p_top1 = values.tokens.clone(), forward.logits, values.p_top1
                    effective_passes = passes + 1
                    mask = swap.pins()[0]
                    final_pins = (mask.clone() if mask.ndim == 2 else
                                  torch.arange(16, device=device)[None] == mask[:, None])
                    final_pins[:, 0] = True
                ids = values.tokens
                if config.freeze_converged:
                    finished.logical_or_(values.fixed)
                else:
                    finished.copy_(values.fixed)
                passes += 1
                self.counts['J'] += 1
                self.counts['converged'] += int(values.fixed.sum())
                continue
            again = operation != 'R1'
            threshold = config.second_refresh_threshold if again else config.refresh_threshold
            # Uncut eval projects the full window before the decoder lowers the cutoff.
            khat = (torch.full((1,), 15, dtype=torch.long, device=device) if admission or config.refresh_full else
                    refresh_cutoff(p_top1[:, 1:], threshold, final=15))
            if again and config.second_refresh_cap is not None and not admission and not config.refresh_full:
                khat = khat.clamp(max=config.second_refresh_cap)
            cutoffs.append(int(khat[0]))
            predicted = self.predict(exact, torch.cat((tail_tokens, ids), dim=1))
            block_hidden = predicted[:, tail_count:]
            slot_counts = khat.clamp(max=14) + 1
            decoded, rows, slots = refresh_decoder_logits(
                self.decoder, sequence, length=exact.shape[1],
                tail_hidden=None if tail_hidden is None else tail_hidden[0, :, -self.decoder.hidden_size:],
                tail_counts=[tail_count], block_hidden=block_hidden, slot_counts=slot_counts)
            decoder_logits = decoded.new_zeros(1, 15, decoded.shape[-1])
            decoder_logits[rows, slots] = decoded
            before = ids.clone()
            old_pins = swap.pin_mask.clone() if swap.dense_pins else None
            entries = swap.apply(decoder_logits, ids=ids, khat=khat, slot_counts=slot_counts,
                                 active=torch.ones(1, dtype=torch.bool, device=device), finished=finished)
            if config.pin_agreed:
                agreed = swap.pin_mask & ~old_pins & ids.eq(before)
                self.counts['agreed_pins'] += int(agreed.sum())
            # The swap returns used+1 entries; inclusive injection reads entries 0..used.
            refreshed = block_hidden, entries - 1
            self.counts['swaps'] += int(ids.ne(before).any(dim=1).sum())
            self.counts['R'] += 1
        predicted = self.predict(exact, torch.cat((tail_tokens, final_ids), dim=1))[:, tail_count:]
        self.counts['admission' if admission else 'rounds'] += 1
        return RoundResult(final_ids, logits, p_top1, predicted, completed_label(effective_passes),
                           config.operations, tuple(cutoffs), bool(finished.all()),
                           effective_passes, final_pins)
