"""JETSPEC_REFRESH_SWAP: shared N=1 eager and graphed refresh-swap decisions.

A fixed-shape, host-sync-free port of the eager tau lab at the adapter refresh with
``E04_LAB_REFRESH_CUTOFF=decoder_f`` and ``E04_LAB_SWAP=first_wrong_refresh`` at margin 0
(``draft/tau_lab.py``: ``refresh_hook``, ``rule_f``,
``margin_swap``, ``refresh_decoder_logits``).  At the B1r refresh, per row:


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

import hashlib
import os

import torch

try:
    from jetspec.engines.jetspec_plus_plus._optional import triton
    from jetspec.engines.jetspec_plus_plus._optional import tl
except ImportError:
    triton = None
    tl = None


NAME = 'JETSPEC_REFRESH_SWAP'
R1_CONFIGS = ('J1-R1', 'J2-R1', 'J3-R1', 'J2-R2', 'J3-R3')


def refresh_swap_lean_enabled():
    value = os.environ.get('JETSPEC_REFRESH_SWAP_LEAN', '0')
    if value not in ('0', '1'):
        raise ValueError('JETSPEC_REFRESH_SWAP_LEAN must be 0 or 1')
    if value == '1' and not refresh_swap_enabled():
        raise ValueError('JETSPEC_REFRESH_SWAP_LEAN=1 requires JETSPEC_REFRESH_SWAP=1')
    return value == '1'


def refresh_swap_enabled(environ=None):
    env = os.environ if environ is None else environ
    value = env.get(NAME, '0')
    if value not in ('0', '1'):
        raise ValueError(f'{NAME} must be 0 or 1')
    return value == '1'


def check_regime(*, max_num_seqs=1, environ=None):
    """Check the fixed N=1 refresh contract at either executor's construction."""
    env = os.environ if environ is None else environ
    if not refresh_swap_enabled(env):
        return
    if max_num_seqs != 1:
        raise ValueError(f'{NAME}=1 requires N=1')
    if env.get('JETSPEC_CHAIN_GRAPH', '0') not in ('0', 'seg'):
        raise ValueError(f'{NAME}=1 requires an eager engine or segmented chain')
    config = env.get('JETSPEC_CHAIN_FIXED_CONFIG')
    from jetspec.engines.jetspec_plus_plus.draft.method.schedule import Schedule
    from jetspec.engines.jetspec_plus_plus.draft.eval_profile import FIXED_CONFIGS; from jetspec.engines.jetspec_plus_plus.draft.eval_profile import enabled as eval_enabled
    eval_fixed = eval_enabled(env) and config in FIXED_CONFIGS
    eval_r0_buffers = eval_fixed and env.get('JETSPEC_CHAIN_GRAPH', '0') == 'seg'
    if config is None or (Schedule.parse(config).r < 1 and not eval_r0_buffers):
        raise ValueError(f'{NAME}=1 requires a fixed refresh chain config: '
                         f'{config}: an R0 config has no refresh')
    # Eval R0 entries may allocate the same buffers, but their empty refresh
    # schedule never replays a swap. Serving retains its existing schedule list.
    if env.get('JETSPEC_ADAPTER_REFRESH', 'off') != 'after_j0':
        raise ValueError(f'{NAME}=1 requires JETSPEC_ADAPTER_REFRESH=after_j0')
    if env.get('JETSPEC_ANCHOR', 'guessed') != 'decoder':
        raise ValueError(f'{NAME}=1 reads the anchor decoder (JETSPEC_ANCHOR=decoder)')


# --- the lab's device rules, ported unchanged (tau_lab.py at 464e69ce) -------------------------

from jetspec.engines.jetspec_plus_plus.draft.method.refresh import rule_f


from jetspec.engines.jetspec_plus_plus.draft.method.refresh import margin_swap


from jetspec.engines.jetspec_plus_plus.draft.method.refresh import swap_decision


from jetspec.engines.jetspec_plus_plus.draft.method.refresh import policy_swap_decision




def compact_decision(compact, ids, khat, slot_counts, active, pin_slot=None):
    """Consume only the top token, excluded top token and margin comparison per slot."""
    top, alternative, margin = compact
    f = rule_f(top, ids, slot_counts)
    khat_used = torch.minimum(khat, f.to(khat.device))
    target = khat_used.to(ids.device) + 1
    slot = target.clamp(max=ids.shape[1] - 1)
    index = (slot - 1).clamp(0, top.shape[1] - 1)[:, None]
    new = alternative.gather(1, index)[:, 0]
    apply = margin.gather(1, index)[:, 0] & active & (target < ids.shape[1])
    apply = apply & (new != ids.gather(1, slot[:, None])[:, 0])
    if pin_slot is not None:
        apply = apply & (pin_slot < 0)
    return khat_used, slot, new, apply


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
    @triton.jit
    def _compact_head(H, W, IDS, PMAX, PSUM, PTOP, PALT, PALTID, PCURRENT,
                      M: tl.constexpr, V: tl.constexpr, K: tl.constexpr,
                      WS0: tl.constexpr, WS1: tl.constexpr,
                      SLOTS: tl.constexpr, C: tl.constexpr,
                      BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                      PERSISTENT: tl.constexpr = False):
        depth = tl.arange(0, BK)
        if PERSISTENT:
            first = tl.program_id(0)
            end = tl.cdiv(M, BM) * C
            step = tl.num_programs(0)
        else:
            first = tl.program_id(0) * C + tl.program_id(1)
            end = first + 1
            step = 1
        for tile in range(first, end, step):
            row = tile // C * BM + tl.arange(0, BM)
            col = tile % C * BN + tl.arange(0, BN)
            current = tl.load(IDS + row // SLOTS * (SLOTS + 1) + row % SLOTS + 1,
                              row < M, other=0)
            acc = tl.full((BM, BN), 0, tl.float32)
            for block in range(tl.cdiv(K, BK)):
                k = block * BK + depth
                h = tl.load(H + row[:, None] * K + k[None, :],
                            (row[:, None] < M) & (k[None, :] < K), other=0)
                w = tl.load(W + col[:, None] * WS0 + k[None, :] * WS1,
                            (col[:, None] < V) & (k[None, :] < K), other=0)
                acc = tl.dot(h, tl.trans(w), acc)
            value = acc.to(tl.bfloat16).to(tl.float32)
            value = tl.where(col[None, :] < V, value, -float('inf'))
            maximum = tl.max(value, 1)
            top = tl.min(tl.where(value == maximum[:, None], col[None, :], 2147483647), 1)
            excluded = tl.where(col[None, :] == current[:, None], -float('inf'), value)
            alternative = tl.max(excluded, 1)
            alternate_id = tl.min(tl.where(excluded == alternative[:, None],
                                           col[None, :], 2147483647), 1)
            draft = tl.max(tl.where(col[None, :] == current[:, None], value, -float('inf')), 1)
            total = tl.sum(tl.exp(value - maximum[:, None]), 1)
            out = row * C + tile % C
            tl.store(PMAX + out, maximum, row < M)
            tl.store(PSUM + out, total, row < M)
            tl.store(PTOP + out, top, row < M)
            tl.store(PALT + out, alternative, row < M)
            tl.store(PALTID + out, alternate_id, row < M)
            tl.store(PCURRENT + out, draft, row < M)

    @triton.jit
    def _compact_merge(PMAX, PSUM, PTOP, PALT, PALTID, PCURRENT, TOP, ALT, MARGIN,
                       C: tl.constexpr, PC: tl.constexpr):
        row = tl.program_id(0)
        chunk = tl.arange(0, PC)
        maximum_part = tl.load(PMAX + row * C + chunk, chunk < C, other=-float('inf'))
        total_part = tl.load(PSUM + row * C + chunk, chunk < C, other=0)
        top_part = tl.load(PTOP + row * C + chunk, chunk < C, other=2147483647)
        alternative_part = tl.load(PALT + row * C + chunk, chunk < C, other=-float('inf'))
        alternate_id = tl.load(PALTID + row * C + chunk, chunk < C, other=2147483647)
        draft_part = tl.load(PCURRENT + row * C + chunk, chunk < C, other=-float('inf'))
        maximum = tl.max(maximum_part, 0)
        top = tl.min(tl.where(maximum_part == maximum, top_part, 2147483647), 0)
        alternative = tl.max(alternative_part, 0)
        alternate = tl.min(tl.where(alternative_part == alternative, alternate_id, 2147483647), 0)
        draft = tl.max(draft_part, 0)
        denominator = tl.sum(total_part * tl.exp(maximum_part - maximum), 0)
        # Compare rounded probabilities, preserving margin-zero ties and underflow.
        p_new = tl.exp(alternative - maximum) / denominator
        p_draft = tl.exp(draft - maximum) / denominator
        tl.store(TOP + row, top)
        tl.store(ALT + row, alternate)
        tl.store(MARGIN + row, p_new >= p_draft)


def allocate_compact(rows, slots, vocab, device):
    chunks = (vocab + 127) // 128
    shape = (rows * slots, chunks)
    partials = tuple(torch.empty(shape, device=device, dtype=dtype)
                     for dtype in (torch.float32, torch.float32, torch.int32,
                                   torch.float32, torch.int32, torch.float32))
    outputs = tuple(torch.empty((rows, slots), device=device, dtype=dtype)
                    for dtype in (torch.long, torch.long, torch.bool))
    return partials, outputs


def compact_head(hidden, weight, ids, storage, *, persistent=False):
    """BF16 vocab tiles emit only six partial scalars; no vocab-logit tensor exists."""
    if triton is None or not hidden.is_cuda or hidden.dtype != torch.bfloat16:
        raise ValueError('compact refresh head requires CUDA BF16')
    partials, outputs = storage
    rows, width = hidden.shape
    vocab = weight.shape[0]
    slots = ids.shape[1] - 1
    chunks = partials[0].shape[1]
    grid = (triton.cdiv(rows, 64), chunks)
    if persistent:
        grid = (min(grid[0] * chunks, 2 * torch.cuda.get_device_properties(hidden.device).multi_processor_count),)
    _compact_head[grid](
        hidden, weight, ids, *partials, rows, vocab, width, *weight.stride(),
        slots, chunks, 64, 128, 64, PERSISTENT=persistent, num_warps=4, num_stages=4)
    _compact_merge[(rows,)](*partials, *outputs, chunks,
                           triton.next_power_of_2(chunks), num_warps=4)
    return outputs


def fill_after_swap(kind, ids, slots, applied, *, mask_token, context=None, noise=None,
                    decoder_argmax=None, slot_counts=None, pins=None):
    """Fill unpinned slots after applied swaps in place; return this pass's guard exemptions.

    Decoder row j-1 predicts block slot j. Only covered decoder rows are filled.
    The filled slots remain unpinned and the next drafter pass can replace them.
    """
    lane = torch.arange(ids.shape[1], device=ids.device)[None]
    free = applied[:, None] & (lane > slots[:, None])
    if pins is not None:
        if isinstance(pins, tuple):
            mask = (pins[0] if pins[0].ndim == 2 else lane == pins[0][:, None])
            free &= ~mask
        else:
            free &= pins.to(ids.device) < 0
    if kind == 'decoder':
        top = decoder_argmax.to(ids.device)
        values = torch.cat((top[:, :1], top), dim=1)
        free &= (lane - 1 < slot_counts[:, None]) & (values >= 0)
    elif kind == 'mask':
        values = torch.full_like(ids, mask_token)
    else:
        values = context if kind == 'context' else noise
    ids.copy_(torch.where(free, values, ids))
    return free




class RefreshSwap:
    """One chain key's swap state and decoder read buffers. The legacy pin is
    (slot, token) per row; pin_agreed uses a dense (mask, tokens) per slot.
    Allocated once per key, before capture; prepare_noise runs on the host
    before replay and the other methods are capture-safe."""

    def __init__(self, decoder, front, tap_store, *, rows, lanes):
        if decoder is None:
            raise ValueError(f'{NAME}=1 needs the anchor decoder')
        self.decoder = decoder
        self.front = front
        self.tap_store = tap_store
        from jetspec.engines.jetspec_plus_plus.draft.eval_profile import states_only
        self.states_only = states_only()
        self.lean = refresh_swap_lean_enabled()
        from jetspec.engines.jetspec_plus_plus.draft.serve_policy import section
        params = section('swap_rule')
        self.rule = params if params['enabled'] else None
        if self.rule is not None:
            self.lean = False
        fill = section('after_swap_fill')
        self.fill_mode = fill['mode'] if fill['enabled'] else 'keep'
        self.fill_seed = fill['seed']
        device = front.decoder_stale.device
        self.eval_lanes = (torch.arange(lanes, device=device)[None, :]
                           if os.environ.get('JETSPEC_EVAL_MODE', 'off') != 'off' else None)
        self.pin_agreed = False
        if 'JETSPEC_SERVE_POLICY' in os.environ:
            from jetspec.engines.jetspec_plus_plus.draft.serve_policy import section
            self.pin_agreed = section('pin_agreed')['enabled']
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
            from jetspec.engines.jetspec_plus_plus.draft.serve_policy import SLOT_BUCKETS; from jetspec.engines.jetspec_plus_plus.draft.serve_policy import KHAT_BUCKETS
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
            like=front.decoder_stale)
        self.check = self.lean and os.environ.get('JETSPEC_REFRESH_SWAP_LEAN_CHECK', '0') == '1'
        if self.check:
            if self.window.get('glue'):
                self.reference_window = decoder.refresh_window_state(
                    front.decoder_prefix_kv, front.decoder_bucket, rows=rows, lanes=lanes,
                    like=front.decoder_stale, glue=False)
            self.check_rounds = torch.zeros(1, device=device, dtype=torch.long)
            self.check_errors = torch.zeros(5, device=device, dtype=torch.long)
            self.check_records = torch.full((512, rows, 4), -1, device=device, dtype=torch.long)
            self.check_expected_records = torch.full_like(self.check_records, -1)
            import atexit
            atexit.register(self._report_check)



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
        hidden = self.decoder.hidden_size
        legacy_write = self.rule is None and not self.dense_pins
        fuse_glue = (legacy_write and self.fill_mode == 'keep'
                     and self.lean and not self.check and block.dtype == torch.bfloat16
                     and os.environ.get('JETSPEC_REFRESH_SWAP_GLUE_FUSED', '0') == '1'
                     and os.environ.get('JETSPEC_REFRESH_SWAP_WRITE_FUSED', '0') == '1')
        if fuse_glue:
            from jetspec.engines.jetspec_plus_plus.kernels.chain.swap_write import read_true_taps
            true_taps = read_true_taps(self.tap_store.storage, length, self.true_lane)
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
            return torch.where(active, used + 1, torch.zeros_like(used))
        if fuse_glue:
            from jetspec.engines.jetspec_plus_plus.kernels.chain.swap_write import decide_and_finish
            return decide_and_finish(result, ids, khat, slot_counts, active,
                                     self.pin_slot, self.pin_token, finished)
        wrote_ids = (legacy_write and self.lean and not self.check and
                     os.environ.get('JETSPEC_REFRESH_SWAP_WRITE_FUSED', '0') == '1')
        if self.rule is not None:
            khat_used, slot, new, apply = policy_swap_decision(
                result, self.draft_top, ids, khat, slot_counts, active, self.rule,
                None if self.dense_pins else self.pin_slot, self.rule_bias,
                self.pin_mask if self.dense_pins else None)
        elif wrote_ids:
            from jetspec.engines.jetspec_plus_plus.kernels.chain.swap_write import decide_and_write
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
            agreed = ((self.pin_lanes >= 1) & (self.pin_lanes <= khat_used[:, None])
                      & active[:, None] & ~self.pin_mask)
            self.pin_values.copy_(torch.where(agreed, ids, self.pin_values))
            self.pin_mask.logical_or_(agreed)
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
