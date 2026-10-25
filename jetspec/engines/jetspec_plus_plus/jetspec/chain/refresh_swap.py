"""JETSPEC_REFRESH_SWAP: shared N=1 eager and graphed refresh-swap decisions.

A fixed-shape, host-sync-free port of the eager tau lab at the adapter refresh with
``E04_LAB_REFRESH_CUTOFF=decoder_f`` and ``E04_LAB_SWAP=first_wrong_refresh`` at margin 0
(``dev-test/serving-tau-lab`` 464e69ce, ``engine/jetspec.engines.jetspec_plus_plus/draft/tau_lab.py``: ``refresh_hook``, ``rule_f``,
``margin_swap``, ``refresh_decoder_logits``; spec: serving4's 2026-09-29 refresh-decoder-cutoff
note, section 8).  At the B1r refresh, per row:

1. the anchor decoder reads the refreshed states of block slots 0..m, m = min(khat_draft, 14), in
   the refresh-context window (``AnchorDecoder.refresh_window_logits``);
2. F = ``rule_f``: the NUMBER of leading slots j >= 1 whose decoder argmax (decoder row j - 1)
   equals the draft d_j;
3. khat_used = min(khat_draft, F) and the swap slot is s = khat_used + 1 (never past khat_draft + 1);
4. the new token is the decoder's best token at s excluding d_s; it replaces d_s iff
   p(new) - p(d_s) >= 0 (``margin_swap``), i.e. iff the decoder argmax at s differs from d_s, so
   khat_draft < F swaps nothing;
5. the swapped token is pinned: every later pass's output at s is overwritten with it before
   publish (the K3 ``_finish`` pin), and a converged row is reopened. The serving policy's
   default-off ``pin_agreed`` section also pins the refresh input at slots 1..khat_used;
6. the refreshed entries 0..khat_used (inclusive, so the anchor always) are injected.

The chain has no swap on the first round after prefill (admission rows never pass through B1r)
nor on miss-redraft rows (the J0-only redraft is an R0-only segment).

The knob is default off; with it unset nothing here runs and ``passes_chain_segments`` receives
no swap object.
"""
from __future__ import annotations

from jetspec.engines.jetspec_plus_plus.profiles import kernel_enabled, kernel_value

import os

import torch
from jetspec.engines.jetspec_plus_plus.draft.method.passes import j0_fill

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None


NAME = 'engine.engine_settings.refresh_swap'
R1_CONFIGS = ('J1-R1', 'J2-R1', 'J3-R1', 'J2-R2', 'J3-R3')


def refresh_swap_lean_enabled(*, run_config=None):
    from jetspec.engines.jetspec_plus_plus.profiles import kernel_value
    value = kernel_value('refresh_swap_lean', '0')
    if value not in ('0', '1'):
        raise ValueError('JETSPEC_REFRESH_SWAP_LEAN must be 0 or 1')
    if value == '1' and not refresh_swap_enabled(run_config=run_config):
        raise ValueError('JETSPEC_REFRESH_SWAP_LEAN=1 requires JETSPEC_REFRESH_SWAP=1')
    return value == '1'


def refresh_swap_enabled(environ=None, *, run_config=None):
    env = environ
    value = config_values((run_config if environ is None else environ)).get('engine.engine_settings.refresh_swap', '0')
    if value not in ('0', '1'):
        raise ValueError(f'{NAME} must be 0 or 1')
    return value == '1'


def check_regime(*, max_num_seqs=1, environ=None, run_config=None):
    """Check the fixed N=1 refresh contract at either executor's construction."""
    env = environ
    if not refresh_swap_enabled(env, run_config=run_config):
        return
    if max_num_seqs != 1:
        raise ValueError(f'{NAME}=1 requires N=1')
    if config_values((run_config if environ is None else environ)).get('engine.engine_settings.chain_graph', '0') not in ('0', 'seg'):
        raise ValueError(f'{NAME}=1 requires an eager engine or segmented chain')
    config = config_values((run_config if environ is None else environ)).get('method.method_settings.chain_fixed_config')
    from jetspec.engines.jetspec_plus_plus.draft.method.schedule import Schedule
    from jetspec.engines.jetspec_plus_plus.jetspec.profile import FIXED_CONFIGS
    from jetspec.engines.jetspec_plus_plus.jetspec.profile import enabled as profile_enabled
    eval_fixed = profile_enabled(env) and config in FIXED_CONFIGS
    eval_r0_buffers = eval_fixed and config_values((run_config if environ is None else environ)).get('engine.engine_settings.chain_graph', '0') == 'seg'
    if config is None or (Schedule.parse(config).r < 1 and not eval_r0_buffers):
        raise ValueError(f'{NAME}=1 requires a fixed refresh chain config: '
                         f'{config}: an R0 config has no refresh')
    # Eval R0 entries may allocate the same buffers, but their empty refresh
    # schedule never replays a swap. Serving retains its existing schedule list.
    if config_values((run_config if environ is None else environ)).get('method.method_settings.adapter_refresh', 'off') != 'after_j0':
        raise ValueError(f'{NAME}=1 requires JETSPEC_ADAPTER_REFRESH=after_j0')
    if config_values((run_config if environ is None else environ)).get('method.method_settings.anchor', 'guessed') != 'decoder':
        raise ValueError(f'{NAME}=1 reads the anchor decoder (JETSPEC_ANCHOR=decoder)')


# --- the lab's device rules, ported unchanged (tau_lab.py at 464e69ce) -------------------------

from jetspec.engines.jetspec_plus_plus.draft.method.refresh import rule_f


from jetspec.engines.jetspec_plus_plus.draft.method.refresh import margin_swap


from jetspec.engines.jetspec_plus_plus.draft.method.refresh import swap_decision


from jetspec.engines.jetspec_plus_plus.draft.method.refresh import compact_logits


def compact_decision(compact, ids, khat, slot_counts, active, pin_slot=None):
    """Consume only the top token, excluded top token and margin comparison per slot."""
    return swap_decision(compact, ids, khat, slot_counts, active, pin_slot)


def decision_errors(observed, expected):
    """Compare consumed decisions; an unapplied alternative is never a swapped token.

    The final counter retains differences in those unused alternatives as information.
    """
    used = observed[3] | expected[3]
    token_diff = observed[2] != expected[2]
    return torch.stack(((observed[0] != expected[0]).sum(),
                        (observed[1] != expected[1]).sum(),
                        (token_diff & used).sum(),
                        (observed[3] != expected[3]).sum(),
                        (token_diff & ~used).sum()))


if triton is not None:
    from jetspec.engines.jetspec_plus_plus.kernels.compact_head import (_compact_head, _compact_merge)


def allocate_compact(rows, slots, vocab, device):
    chunks = (vocab + 127) // 128
    shape = (rows * slots, chunks)
    partials = tuple(torch.empty(shape, device=device, dtype=dtype)
                     for dtype in (torch.float32, torch.float32, torch.int32,
                                   torch.float32, torch.int32, torch.float32))
    outputs = tuple(torch.empty((rows, slots), device=device, dtype=dtype)
                    for dtype in (torch.long, torch.long, torch.bool))
    return partials, outputs


def compact_head(hidden, weight, ids, storage=None, *, persistent=False):
    """Emit six partial scalars per tile, with W4A8 scratch logits in head mode."""
    if not hidden.is_cuda or storage is None:
        if getattr(weight, '_draft_vocab', None) is not None:
            from jetspec.engines.jetspec_plus_plus.models.target_linear import draft_linear
            logits = draft_linear(hidden, weight).reshape(ids.shape[0], ids.shape[1] - 1, -1)
            return compact_logits(logits, ids)
        from jetspec.engines.jetspec_plus_plus.draft.method.ops import reference
        return reference('refresh_compact_head')(hidden, weight, ids, storage, persistent=persistent)
    if triton is None or hidden.dtype != torch.bfloat16:
        raise ValueError('compact refresh head requires CUDA BF16')
    partials, outputs = storage
    rows, width = hidden.shape
    vocab = weight.shape[0]
    slots = ids.shape[1] - 1
    chunks = partials[0].shape[1]
    grid = (triton.cdiv(rows, 64), chunks)
    if persistent:
        grid = (min(grid[0] * chunks, 2 * torch.cuda.get_device_properties(hidden.device).multi_processor_count),)
    from jetspec.engines.jetspec_plus_plus.models.target_linear import w4a8_head_logits
    logits = w4a8_head_logits(hidden, weight)
    bm = 64
    if logits is not None and kernel_value('draft_fuse_chgeom', '0') == '1':
        bm = 16
        grid = (triton.cdiv(rows, bm), chunks)
        if persistent:
            grid = (min(grid[0] * chunks, 2 * torch.cuda.get_device_properties(hidden.device).multi_processor_count),)
    bn, warps = 128, 4
    if (logits is not None and vocab == 151936 and slots == 15 and rows == 60
            and not persistent and kernel_value('draft_fuse_geo8', '0') == '1'):
        bm = 16
        grid = (triton.cdiv(rows, bm), chunks)
    if (logits is not None and vocab == 248320 and slots == 15 and rows in (60, 240)
            and kernel_value('draft_fuse_stats27', '0') == '1'):
        bm, bn, warps = 4, 512, 2
        chunks = triton.cdiv(vocab, bn)
        # The existing scratch has capacity for 128-column tiles. Both kernels
        # consume its packed prefix with the same, smaller chunk stride.
        persistent = False
        grid = (triton.cdiv(rows, bm), chunks)
    _compact_head[grid](
        hidden if logits is None else logits, weight, ids, *partials, rows, vocab, width, *weight.stride(),
        slots, chunks, bm, bn, 64, PERSISTENT=persistent,
        FROM_LOGITS=logits is not None, num_warps=warps, num_stages=4)
    _compact_merge[(rows,)](*partials, *outputs, chunks,
                           triton.next_power_of_2(chunks), num_warps=4)
    return outputs


from jetspec.engines.jetspec_plus_plus.kernels import register_optimization
from jetspec.engines.jetspec_plus_plus.jetspec.config_values import config_values
register_optimization('refresh_compact_head', compact_head)


def noise_fill_tokens(*, seed, seq_id, candidate_round, rows, lanes, vocab, mask_token):
    """Lab noise draws on CPU, before replay; exact=0 and refresh pass=1 on this path."""
    return j0_fill(block_size=lanes, noise_key=(seed, seq_id, candidate_round),
                   rows=rows, vocab=vocab, mask_token=mask_token)


class RefreshSwap:
    """One chain key's swap state and decoder read buffers. The legacy pin is
    (slot, token) per row; pin_agreed uses a dense (mask, tokens) per slot.
    Allocated once per key, before capture; its methods are capture-safe."""

    def __new__(cls, *args, run_config=None, **kwargs):
        if cls is RefreshSwap:
            from jetspec.engines.jetspec_plus_plus.draft.explore import refresh_type
            variant = refresh_type(run_config)
            if variant is not None:
                return object.__new__(variant)
        return object.__new__(cls)

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
        self.rule = None
        self.fill_mode = 'keep'
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
        self.true_lane = torch.arange(4, device=device, dtype=torch.long) - 4
        self.true_taps = None
        if (config_values(run_config).get('engine.engine_settings.draft_fuse_t10', '0') == '1'
                and front.decoder_prefix_taps.is_cuda and kernel_enabled('chain_intake_fused', '1')):
            self.true_taps = front.decoder_prefix_taps.new_empty(4, decoder.hidden_size)
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
        if getattr(self, 'true_taps', None) is not None:
            state['refresh_swap_true_taps'] = self.true_taps
        if self.states_only:
            state['eval_refresh_effects'] = self.eval_effects
        if self.dense_pins:
            state.update(refresh_swap_pin_mask=self.pin_mask,
                         refresh_swap_pin_values=self.pin_values)
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

    def pins(self):
        if self.dense_pins:
            return self.pin_mask, self.pin_values
        return self.pin_slot, self.pin_token

    def apply(self, *, length, ids, khat, slot_counts, tail_counts, active, block, finished,
              inclusive=False):
        """At the refresh, after the adapter's refreshed rows ``block`` [R, W, 5H] exist and before
        they are injected: swap ``ids`` in place, pin the swaps, reopen swapped rows in
        ``finished``, and return the fixed-layout inject count (entries 0..khat_used; 0 on an
        inactive row)."""
        from jetspec.engines.jetspec_plus_plus.profiles import kernel_enabled
        hidden = self.decoder.hidden_size
        legacy_write = not self.dense_pins
        if getattr(self, 'true_taps', None) is not None:
            true_taps = self.true_taps
        else:
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
            count = torch.where(active, used + 1, torch.zeros_like(used))
            return count - 1 if inclusive else count
        if (config_values(getattr(self, 'run_config', None)).get(
                'engine.engine_settings.draft_fuse_t10', '0') == '1'
                and legacy_write and self.lean and not self.check and not self.multi_refresh):
            from jetspec.engines.jetspec_plus_plus.draft.swap_write import decide_and_finish
            return decide_and_finish(result, ids, khat, slot_counts, active,
                                     self.pin_slot, self.pin_token, finished, inclusive=inclusive)
        wrote_ids = (legacy_write and self.lean and not self.check and
                     kernel_enabled('refresh_swap_write_fused', '1'))
        if wrote_ids:
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
        if self.dense_pins:
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
        self.pin_slot.copy_(torch.where(apply, slot, self.pin_slot))
        # Keep the legacy R1 unused alternatives byte-identical.
        self.pin_token.copy_(torch.where(apply, new, self.pin_token) if self.multi_refresh else new)
        finished.logical_and_(~apply)
        count = torch.where(active, khat_used + 1, torch.zeros_like(khat_used))
        return count - 1 if inclusive else count
