"""The tried refresh state machine, loaded only by an exploration method."""
import torch
from jetspec.engines.jetspec_plus_plus.jetspec.chain.refresh_swap import (
    RefreshSwap as ServingRefreshSwap, NAME, refresh_swap_lean_enabled,
    noise_fill_tokens, rule_f, swap_decision, compact_decision, decision_errors)
from jetspec.engines.jetspec_plus_plus.jetspec.config_values import config_values
from .swap_rule import policy_swap_decision
from .after_swap_fill import fill_after_swap
from . import method_value


def selected(name, legacy, run_config):
    value = method_value(run_config, name)
    return legacy if value is None else value

class RefreshSwap(ServingRefreshSwap):
    """One chain key's swap state and decoder read buffers. The legacy pin is
    (slot, token) per row; pin_agreed uses a dense (mask, tokens) per slot.
    Allocated once per key, before capture; prepare_noise runs on the host
    before replay and the other methods are capture-safe."""

    def __init__(self, decoder, front, tap_store, *, rows, lanes, run_config=None):
        self.run_config = run_config
        from jetspec.engines.jetspec_plus_plus.engine import current_engine
        if decoder is None:
            raise ValueError(f'{NAME}=1 needs the anchor decoder')
        self.decoder = decoder
        self.front = front
        self.tap_store = tap_store
        from jetspec.engines.jetspec_plus_plus.jetspec.profile import states_only
        self.states_only = states_only(run_config=self.run_config)
        self.lean = refresh_swap_lean_enabled(run_config=self.run_config)
        from jetspec.engines.jetspec_plus_plus.draft.serve_policy import section
        params = section('swap_rule', run_config=run_config)
        self.rule = params if selected('swap_rule', params['enabled'], run_config) else None
        if self.rule is not None:
            self.lean = False
        fill = section('after_swap_fill', run_config=run_config)
        self.fill_mode = fill['mode'] if selected('after_swap_fill', fill['enabled'], run_config) else 'keep'
        self.fill_seed = fill['seed']
        device = front.decoder_prefix_taps.device
        self.eval_lanes = (torch.arange(lanes, device=device)[None, :]
                           if current_engine().execution in ("pipelined", "critical-path", "on") else None)
        self.pin_agreed = False
        if 'engine.engine_settings.serve_policy' in config_values(self.run_config):
            from jetspec.engines.jetspec_plus_plus.draft.serve_policy import section
            self.pin_agreed = section('pin_agreed', run_config=run_config)['enabled']
        self.dense_pins = self.pin_agreed
        self.multi_refresh = False
        if self.pin_agreed:
            self.pin_mask = torch.zeros((rows, lanes), dtype=torch.bool, device=device)
            self.pin_values = torch.zeros((rows, lanes), dtype=torch.long, device=device)
            self.pin_lanes = torch.arange(lanes, device=device)[None, :]
        self.pin_slot = torch.full((rows,), -1, dtype=torch.long, device=device)
        self.pin_token = torch.zeros(rows, dtype=torch.long, device=device)
        if self.states_only:
            self.eval_effects = torch.zeros(2, dtype=torch.long, device=device)
        if self.rule is not None:
            n = min(self.rule['n'], decoder.lm_head.weight.shape[0])
            from jetspec.engines.jetspec_plus_plus.draft.serve_policy import SLOT_BUCKETS
            from jetspec.engines.jetspec_plus_plus.draft.serve_policy import KHAT_BUCKETS
            self.rule_bias = torch.tensor([[params['bias'][s][k] for k in KHAT_BUCKETS]
                                           for s in SLOT_BUCKETS], dtype=torch.float64, device=device)
            self.draft_top = (torch.zeros(rows, lanes, n, dtype=torch.long, device=device),
                              torch.zeros(rows, lanes, n, dtype=torch.float32, device=device))
        if self.fill_mode != 'keep':
            self.fill_filled = torch.zeros((rows, lanes), dtype=torch.bool, device=device)
        if self.fill_mode == 'context':
            self.fill_context = torch.empty((rows, lanes), dtype=torch.long, device=device)
        if self.fill_mode == 'noise':
            self.fill_noise = torch.zeros((rows, lanes), dtype=torch.long, device=device)
            self.fill_noise_cpu = torch.empty((rows, lanes), dtype=torch.long,
                                              pin_memory=device.type == 'cuda')
        self.true_lane = torch.arange(4, device=device, dtype=torch.long) - 4
        self.window = decoder.refresh_window_state(
            front.decoder_prefix_kv, front.decoder_bucket, rows=rows, lanes=lanes,
            like=front.decoder_prefix_taps)
        self.check = self.lean and config_values(self.run_config).get('engine.engine_settings.refresh_swap_lean_check', '0') == '1'
        if self.check:
            if self.window.get('glue'):
                self.reference_window = decoder.refresh_window_state(
                    front.decoder_prefix_kv, front.decoder_bucket, rows=rows, lanes=lanes,
                    like=front.decoder_prefix_taps, glue=False)
            self.check_rounds = torch.zeros(1, device=device, dtype=torch.long)
            self.check_errors = torch.zeros(5, device=device, dtype=torch.long)
            self.check_records = torch.full((512, rows, 4), -1, device=device, dtype=torch.long)
            self.check_expected_records = torch.full_like(self.check_records, -1)
            import atexit
            atexit.register(self._report_check)

    def prepare_noise(self, *, seq_id, candidate_round, vocab, mask_token):
        """Stage the lab's per-request/round CPU draws outside CUDA graph capture."""
        self.fill_noise_cpu.copy_(noise_fill_tokens(
            seed=self.fill_seed, seq_id=seq_id, candidate_round=candidate_round,
            rows=self.fill_noise.shape[0], lanes=self.fill_noise.shape[1],
            vocab=vocab, mask_token=mask_token))
        self.fill_noise.copy_(self.fill_noise_cpu, non_blocking=True)

    def _report_check(self):
        """Default-off per-key audit, emitted only when the serving worker exits."""
        import json
        count = int(self.check_rounds.item())
        print('SWAP_LEAN_CHECK ' + json.dumps(dict(
            bucket=self.window['bucket'], rows=self.pin_slot.numel(), rounds=count,
            mismatches=self.check_errors.cpu().tolist(),
            fields=['khat_used', 'slot', 'swapped_token', 'apply', 'unused_alternative'],
            records=self.check_records[:min(count, 512)].cpu().tolist())), flush=True)

    def tensors(self):
        """The mutable state, for the chain's buffers (T0 shadow checkpoint and recording)."""
        state = {'refresh_swap_pin_slot': self.pin_slot,
                 'refresh_swap_pin_token': self.pin_token}
        if self.states_only:
            state['eval_refresh_effects'] = self.eval_effects
        if self.dense_pins:
            state.update(refresh_swap_pin_mask=self.pin_mask,
                         refresh_swap_pin_values=self.pin_values)
        if self.rule is not None:
            state.update(refresh_swap_draft_ids=self.draft_top[0],
                         refresh_swap_draft_probabilities=self.draft_top[1])
        if self.fill_mode != 'keep':
            state['after_swap_fill_filled'] = self.fill_filled
        if self.fill_mode == 'context':
            state['after_swap_fill_context'] = self.fill_context
        if self.fill_mode == 'noise':
            state['after_swap_fill_noise'] = self.fill_noise
        if self.check:
            state.update(refresh_swap_check_rounds=self.check_rounds,
                         refresh_swap_check_errors=self.check_errors,
                         refresh_swap_check_records=self.check_records,
                         refresh_swap_check_expected_records=self.check_expected_records)
        return state

    def configure_pins(self, refresh_at, *, rows, lanes):
        """Allocate accumulated swap pins for a multi-refresh key, before capture."""
        self.multi_refresh = len(refresh_at) > 1
        if len(refresh_at) > 1 and not self.dense_pins:
            device = self.pin_slot.device
            self.pin_mask = torch.zeros((rows, lanes), dtype=torch.bool, device=device)
            self.pin_values = torch.zeros((rows, lanes), dtype=torch.long, device=device)
            self.pin_lanes = torch.arange(lanes, device=device)[None, :]
            self.dense_pins = True

    def reset(self):
        """J0: no pin survives from the previous round."""
        self.pin_slot.fill_(-1)
        if self.states_only:
            self.eval_effects.zero_()
        if self.dense_pins:
            self.pin_mask.zero_()
        if self.fill_mode != 'keep':
            self.fill_filled.zero_()

    def pins(self):
        if self.dense_pins:
            return self.pin_mask, self.pin_values
        return self.pin_slot, self.pin_token

    def retain_draft(self, logits, weight=None):
        """Save the distribution of the pass whose block the refresh will read."""
        if weight is not None:
            logits = torch.nn.functional.linear(logits, weight)
        ranked = logits.float().softmax(-1).topk(self.draft_top[0].shape[-1], -1)
        self.draft_top[0].copy_(ranked.indices)
        self.draft_top[1].copy_(ranked.values)

    def apply(self, *, length, ids, khat, slot_counts, tail_counts, active, block, finished):
        """At the refresh, after the adapter's refreshed rows ``block`` [R, W, 5H] exist and before
        they are injected: swap ``ids`` in place, pin the swaps, reopen swapped rows in
        ``finished``, and return the fixed-layout inject count (entries 0..khat_used; 0 on an
        inactive row)."""
        from jetspec.engines.jetspec_plus_plus.profiles import kernel_enabled
        hidden = self.decoder.hidden_size
        legacy_write = self.rule is None and not self.dense_pins
        positions = (length + self.true_lane).clamp_min(0)
        true_taps = self.tap_store.storage.index_select(0, positions)[:, -1]
        read = (self.decoder.refresh_window_compact if self.lean
                else self.decoder.refresh_window_logits)
        result = read(
            self.window, length=length, true_taps=true_taps,
            tail=self.front.staged_hidden[0, :, -hidden:], tail_counts=tail_counts,
            block=block, slot_counts=slot_counts, **({'ids': ids} if self.lean else {}))
        if self.states_only:
            # Preserve the decoder cutoff and state injection, with no draft
            # write, swapped/agreed pin, filler, or convergence reopening.
            top = result[0] if self.lean else result.argmax(dim=-1)
            used = torch.minimum(khat, rule_f(top, ids, slot_counts))
            pins = self.pin_mask.sum() if self.dense_pins else (self.pin_slot >= 0).sum()
            self.eval_effects[1].add_(pins)
            return torch.where(active, used + 1, torch.zeros_like(used))
        wrote_ids = (legacy_write and self.lean and not self.check and
                     kernel_enabled('refresh_swap_write_fused', '1'))
        if self.rule is not None:
            khat_used, slot, new, apply = policy_swap_decision(
                result, self.draft_top, ids, khat, slot_counts, active, self.rule,
                None if self.dense_pins else self.pin_slot, self.rule_bias,
                self.pin_mask if self.dense_pins else None)
        elif wrote_ids:
            from jetspec.engines.jetspec_plus_plus.draft.swap_write import decide_and_write
            khat_used, slot, new, apply = decide_and_write(result, ids, khat, slot_counts, active)
        else:
            decision = compact_decision if self.lean else swap_decision
            khat_used, slot, new, apply = decision(result, ids, khat, slot_counts, active)
            if self.dense_pins:
                apply &= ~self.pin_mask.gather(1, slot[:, None])[:, 0]
        if self.check:
            baseline = self.decoder.refresh_window_logits(
                getattr(self, 'reference_window', self.window), length=length, true_taps=true_taps,
                tail=self.front.staged_hidden[0, :, -hidden:], tail_counts=tail_counts,
                block=block, slot_counts=slot_counts)
            expected = swap_decision(baseline, ids, khat, slot_counts, active)
            if self.dense_pins:
                expected = (*expected[:3], expected[3] &
                            ~self.pin_mask.gather(1, expected[1][:, None])[:, 0])
            observed = (khat_used, slot, new, apply)
            errors = decision_errors(observed, expected)
            self.check_errors.add_(errors)
            record = (khat_used, slot, torch.where(apply, new, torch.full_like(new, -1)), apply)
            self.check_records.index_copy_(0, self.check_rounds.clamp(max=511),
                                           torch.stack(record, -1).long()[None])
            expected_record = (expected[0], expected[1],
                               torch.where(expected[3], expected[2], torch.full_like(expected[2], -1)),
                               expected[3])
            self.check_expected_records.index_copy_(0, self.check_rounds.clamp(max=511),
                                                    torch.stack(expected_record, -1).long()[None])
            self.check_rounds.add_(1)
        if self.pin_agreed:
            # tau_lab._pin_agreed (6908a729): preserve the refresh's input at
            # 1..khat_used even if no swap applies, leaving existing pins intact.
            from jetspec.engines.jetspec_plus_plus.draft.method.refresh import agreed_pins
            agreed_pins(ids, khat_used, active, self.pin_mask, self.pin_values, lanes=self.pin_lanes)
        if self.dense_pins and (self.rule is None or self.rule['pin']
                                or (self.pin_agreed and not self.multi_refresh)):
            swapped = (self.pin_lanes == slot[:, None]) & apply[:, None]
            self.pin_values.copy_(torch.where(swapped, new[:, None], self.pin_values))
            self.pin_mask.logical_or_(swapped)
        if not wrote_ids:
            if self.eval_lanes is not None:
                # Deterministic CUDA scatter is not capture-safe in the pinned runtime.
                # Each row has one unique slot; the mask writes the same IDs.
                swapped = (self.eval_lanes == slot[:, None]) & apply[:, None]
                ids.copy_(torch.where(swapped, new[:, None], ids))
            else:
                ids.scatter_(1, slot[:, None], torch.where(apply, new, ids.gather(1, slot[:, None])[:, 0])[:, None])
        if self.rule is None:
            self.pin_slot.copy_(torch.where(apply, slot, self.pin_slot))
            # Keep the legacy R1 unused alternatives byte-identical.
            self.pin_token.copy_(torch.where(apply, new, self.pin_token) if self.multi_refresh else new)
        elif self.rule['pin']:
            self.pin_slot.copy_(torch.where(apply, slot, self.pin_slot))
            self.pin_token.copy_(torch.where(apply, new, self.pin_token))
        if self.fill_mode != 'keep':
            top = None
            if self.fill_mode == 'decoder':
                top = result[0] if self.lean else result.argmax(dim=-1)
            self.fill_filled.copy_(fill_after_swap(
                self.fill_mode, ids, slot, apply, mask_token=self.fill_mask_token,
                context=getattr(self, 'fill_context', None),
                noise=getattr(self, 'fill_noise', None),
                decoder_argmax=top, slot_counts=slot_counts, pins=self.pins()))
        finished.logical_and_(~apply)
        return torch.where(active, khat_used + 1, torch.zeros_like(khat_used))
