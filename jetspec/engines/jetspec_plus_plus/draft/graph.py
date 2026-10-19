"""Lazy, segmented CUDA graph capture for the N=1 draft chain.

The caller supplies capture-safe segment bodies over fixed buffers. A chain key
owns its graphs and buffers; each segment gets a private pool so later launches
may omit B2 or C without violating CUDA graph pool replay order.
"""
from __future__ import annotations

from jetspec.engines.jetspec_plus_plus.profiles import kernel_enabled, kernel_value

from collections import OrderedDict
from dataclasses import dataclass
import json
import os
import re
import time

import torch
from jetspec.engines.jetspec_plus_plus.draft.method.schedule import Schedule, schedule_segments, CHAIN_SEGMENTS
from jetspec.engines.jetspec_plus_plus.jetspec.env import context_append_enabled

from jetspec.engines.jetspec_plus_plus.wire.contract import ChainKey
from jetspec.engines.jetspec_plus_plus.wire.contract import Offset
from jetspec.engines.jetspec_plus_plus.wire.contract import OutcomeOffset
from jetspec.engines.jetspec_plus_plus.wire.contract import patch_round_params_from_outcome
from jetspec.engines.jetspec_plus_plus.jetspec.chain.intake import CHAIN_ERR_CAPACITY
from jetspec.engines.jetspec_plus_plus.jetspec.chain.intake import CHAIN_ERR_DRIFT
from jetspec.engines.jetspec_plus_plus.jetspec.chain.intake import CHAIN_ERR_PACKET
from jetspec.engines.jetspec_plus_plus.jetspec.chain.intake import CHAIN_ERR_PENDING
from jetspec.engines.jetspec_plus_plus.jetspec.chain.intake import CHAIN_ERR_CONTROL
from jetspec.engines.jetspec_plus_plus.jetspec.chain.intake import CHAIN_ERR_AFTER_MISS
from jetspec.engines.jetspec_plus_plus.jetspec.chain.intake import CHAIN_ERR_ROW_MISSING
from jetspec.engines.jetspec_plus_plus.wire.contract import HASH_WEIGHTS
from jetspec.engines.jetspec_plus_plus.jetspec.chain.intake import append_outcome
from jetspec.engines.jetspec_plus_plus.jetspec.chain.intake import fill_front_from_outcome


SEGMENTS = CHAIN_SEGMENTS
OUTPUTS = ("keys", "final_tokens", "final_p_top1", "passes_run", "finished", "err")


def tail_segments(refresh=True, run_config=None):
    from jetspec.engines.jetspec_plus_plus.draft.serve_policy import section
    tail = section('tail_passes', run_config=run_config)
    return tuple(f'T{i + 1}' for i in range(tail['count'])) if refresh and tail['enabled'] else ()


def reachable_chain_lengths(prompt_lengths, tokens, lanes, max_live):
    """Startup keys covering the read's host tracker bound plus one spare bucket.

    The host key is ChainLengthTracker.key_length = L_known + (age + 1) * lanes, and
    the packet mirror is read after run + publish, so age reaches 2 in steady state
    (bound = L + 3 lanes). One more 128-token bucket absorbs a late mirror.
    """
    if not prompt_lengths or any(type(length) is not int or length < 2
                                 for length in prompt_lengths):
        raise ValueError("reachable chain warm-up needs prompt token lengths")
    if type(tokens) is not int or tokens < 0 or max_live < 128:
        raise ValueError("invalid chain token budget or cache capacity")
    first = max(128, ((min(prompt_lengths) + 127) // 128) * 128)
    last = min((max_live // 128) * 128,
               ((max(prompt_lengths) + tokens + 3 * lanes + 127) // 128) * 128 + 128)
    if first > last:
        raise ValueError("chain read has no safe reachable length key")
    return tuple(range(first, last + 1, 128))


def extended_segment(name):
    return bool(re.fullmatch(r'J(?:[4-9]|[1-9][0-9]+)|R[1-9][0-9]*', name))


def reachable_chain_capture(fixed_config, row_cap, tau_enabled, tau_cap, run_config=None):
    """Keys and segment bodies that the launch policy can actually select."""
    from jetspec.engines.jetspec_plus_plus.draft.serve_policy import rows_min_capacity
    row_cap = rows_min_capacity(row_cap, run_config=run_config)
    if fixed_config is not None:
        schedule = Schedule.parse(fixed_config)
        segments = schedule_segments(schedule)
        return ((row_cap if row_cap in (4, 8, 16) else 4),), segments + tail_segments(schedule.r, run_config=run_config) + ("P",)
    if row_cap not in (4, 8, 16):
        raise ValueError("unsupported chain row cap")
    caps = (4, 8) if tau_cap else (row_cap,)
    segments = SEGMENTS if tau_enabled else ("A", "B1n", "B2", "P")
    return caps, segments


def chain_fixed_choice(fixed_config, length_bucket, switch_L):
    """Choose the fixed replay depth from the already selected chain key bucket."""
    from jetspec.engines.jetspec_plus_plus.engine import current_engine
    switch_L = current_engine().budget.switch_length(switch_L)
    if switch_L is not None and length_bucket >= switch_L:
        return 1, False
    schedule = Schedule.parse(fixed_config)
    return schedule.j, (schedule.r if schedule.r > 1 else bool(schedule.r))


class ChainPublishedError(RuntimeError):
    """Seg P completed before the host detected this error."""


class ChainDriftError(ChainPublishedError):
    """Draft and verifier committed-token mirrors differ."""


class ChainCapacityFallback(ChainPublishedError):
    """The device length exceeds this captured prefix capacity."""


def chain_graph_mode(run_config=None):
    from jetspec.engines.jetspec_plus_plus.draft.config import setting, required_setting, as_run_config
    mode = setting(run_config, 'engine.engine_settings.chain_graph', "0")
    if mode not in ("0", "seg"):
        raise ValueError("JETSPEC_CHAIN_GRAPH must be 0 or seg")
    return mode


@dataclass
class _Entry:
    buffers: dict
    bodies: dict
    graphs: dict


class ChainGraph:
    """Capture on first key sight, outside timing; replay only the chosen passes."""

    def __init__(self, build, *, capacity=8, check=None, capture=None,
                 event_factory=None, shadow_build=None, shadow_state=None,
                 capture_names=SEGMENTS, run_config=None):
        from jetspec.engines.jetspec_plus_plus.draft.config import setting, required_setting, as_run_config
        self.run_config = as_run_config(run_config)
        self.build = build
        self.capacity = capacity
        self.check = (setting(self.run_config, 'engine.engine_settings.chain_graph_check', "0") == "1"
                      if check is None else check)
        self.ctx_kv_check = setting(self.run_config, 'engine.engine_settings.ctx_kv_append_check', "0") == "1"
        self.ctx_kv_checked = 0
        self.ctx_kv_tolerated = 0
        self.capture = capture or self._cuda_capture
        self.event_factory = event_factory or (lambda: torch.cuda.Event(enable_timing=True))
        self.shadow_build = shadow_build
        self.shadow_state = shadow_state
        self.capture_names = tuple(capture_names)
        if self.capture_names == SEGMENTS and setting(self.run_config, 'method.method_settings.chain_fixed_config'):
            schedule = Schedule.parse(required_setting(self.run_config, 'method.method_settings.chain_fixed_config'))
            extra = tuple(name for name in schedule_segments(schedule) if extended_segment(name))
            self.capture_names = self.capture_names[:-1] + extra + self.capture_names[-1:]
        tails = tail_segments(run_config=self.run_config)
        if tails and not any(name in tails for name in self.capture_names):
            self.capture_names = self.capture_names[:-1] + tails + self.capture_names[-1:]
        if not self.capture_names or any(name not in SEGMENTS + tails and not extended_segment(name)
                                        for name in self.capture_names):
            raise ValueError("invalid chain capture segment set")
        self.entries: OrderedDict[ChainKey, _Entry] = OrderedDict()
        self.last_ms = {}
        self.last_pre_a0_gap_ms = {}
        self.last_a0_to_a1_gap_ms = {}
        self.last_span_ms = {}
        self.mix = {"J3-R1": 0, "J2-R1": 0, "J1-R1": 0,
                    "J2-R0": 0, "J1-R0": 0}
        self.cap_mix = {"r4": 0, "r8": 0}
        self.fallback_counts = {"capacity": 0}
        self.round_syncs = 0
        self._prewarmed = False
        from jetspec.engines.jetspec_plus_plus.trace.kprofile import ChainKernelProfile
        self.kernel_profile = ChainKernelProfile(run_config=run_config)

    @staticmethod
    def _cuda_capture(body):
        # No pool= argument: every graph owns its private CUDA memory pool.
        body()  # compile/warm outside capture
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            body()
        return graph.replay

    def _shared_segment_pools(self):
        # Eval only: one pool per segment name across keys. A key's segments keep
        # separate pools, so omitting B2 or C still keeps replay order valid.
        from jetspec.engines.jetspec_plus_plus.jetspec.profile import enabled as profile_enabled
        return profile_enabled() and self.capture == self._cuda_capture

    def _segment_pool_capture(self, name, body):
        pools = self.__dict__.setdefault("_segment_pools", {})
        pool = pools.setdefault(name, torch.cuda.graph_pool_handle())
        body()  # compile/warm outside capture
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=pool):
            body()
        return graph.replay

    @staticmethod
    def _wrap_intake(buffers, bodies, *, max_length=None, run_config=None):
        from jetspec.engines.jetspec_plus_plus.profiles import kernel_enabled
        from jetspec.engines.jetspec_plus_plus.draft.config import setting, required_setting, as_run_config
        if not {"packet", "tokens", "row_ring", "intake_err"} <= set(buffers):
            return bodies
        if "published" in buffers and "final_p_top1" not in buffers:
            raise ValueError("chain publication requires frozen final_p_top1")
        if "front" in buffers and "tap_store" not in buffers:
            raise ValueError("chain front intake requires exact tap storage")
        buffers.setdefault("hash_weights", torch.tensor(
            HASH_WEIGHTS, dtype=torch.int64, device=buffers["packet"].device))
        buffers.setdefault("err_host", torch.empty(
            1, dtype=torch.int64, pin_memory=buffers["err"].is_cuda))
        buffers.setdefault("expected_round", buffers["packet"][OutcomeOffset.ROUND].clone())
        if (((setting(run_config, 'engine.engine_settings.chain_miss_redraft', "0") == "1") or (context_append_enabled(run_config=run_config)))):
            buffers.setdefault("control_round", torch.zeros(
                (), dtype=torch.int64, device=buffers["packet"].device))
            buffers.setdefault("control_L", torch.zeros(
                (), dtype=torch.int64, device=buffers["packet"].device))
        if setting(run_config, 'engine.engine_settings.chain_miss_redraft', "0") == "1":
            buffers.setdefault("control_after_miss", torch.zeros(
                (), dtype=torch.int64, device=buffers["packet"].device))
        buffers.setdefault("safe_length", torch.tensor(
            min(16, buffers["tokens"].committed_ids.numel() - 1,
                max_length if max_length is not None else 16),
            dtype=buffers["packet"].dtype, device=buffers["packet"].device))
        from jetspec.engines.jetspec_plus_plus.jetspec.profile import enabled as profile_enabled
        fused = (kernel_enabled('chain_intake_fused', '1')
                 and buffers["packet"].is_cuda
                 and buffers["row_ring"].rows.shape[-1] in ((16,32) if profile_enabled() else (16,)))
        tau_depth = 3 if profile_enabled() else 2
        if fused:
            from jetspec.engines.jetspec_plus_plus.draft.intake import fused_intake
            buffers.setdefault("intake_selected", torch.empty(
                (1,), dtype=torch.int64, device=buffers["packet"].device))
        if (not kernel_enabled('chain_sega_glue', '0')
                and buffers["tokens"].committed_ids.numel() > buffers["row_ring"].rows.shape[-1]):
            buffers.setdefault("append_lanes", torch.arange(
                buffers["row_ring"].rows.shape[-1], dtype=torch.int64,
                device=buffers["packet"].device))
        if "passes_run" in buffers:
            # Per-key labels have R entries; the published ring spans the wire.
            # Allocate padding before capture so Seg P only copies device data.
            buffers.setdefault("wire_passes_run", torch.zeros(
                buffers["row_ring"].rows.shape[1], dtype=torch.int64,
                device=buffers["passes_run"].device))
        first, publish = bodies.get("A", bodies.get("A0")), bodies["P"]
        fuse_intake_error = (setting(run_config, 'engine.engine_settings.draft_fuse_t14', '0') == '1'
                             and kernel_enabled('rows_k3', '1')
                             and buffers['err'].is_cuda)
        if fuse_intake_error:
            buffers['_t14_intake_err'] = buffers['intake_err']
        if (fuse_intake_error and kernel_enabled('j0_lmhead_kn', '1')
                and kernel_enabled('lmhead_fused', '1') and kernel_enabled('refine_k3', '1')
                and kernel_enabled('chain_publish_fused', '1') and 'publisher' in buffers
                and setting(run_config, 'method.method_settings.chain_fixed_config') in ('J1-R0', 'J1-R1')
                and not tail_segments(run_config=run_config)
                and 'final_draft_top16_ids' not in buffers):
            import triton
            publisher, ring = buffers['publisher'], buffers['row_ring']
            buffers['_t14_publication'] = dict(PUBLISH=True, Packet=buffers['packet'],
                Published=buffers['published'], Wire=publisher.wire, Scores=publisher.scores,
                WirePasses=buffers['wire_passes_run'], RingRows=ring.rows, RingKeys=ring.keys,
                RingTop1=ring.p_top1, RingPasses=ring.passes_run, RingRounds=ring.rounds,
                ROWS=publisher.rows, CAP=publisher.row_cap, RING=ring.rows.shape[1],
                WORDS=publisher.wire.numel(), BLOCK_OFF=publisher.views[2].storage_offset(),
                KEY_OFF=publisher.views[5].storage_offset(),
                PAD_BLOCK=triton.next_power_of_2(max(publisher.wire.numel(), publisher.scores.numel())))

        def after_miss_control():
            if "control_after_miss" in buffers:
                active = ((buffers["packet"][OutcomeOffset.ROUND] != 0)
                          & (buffers["packet"][OutcomeOffset.EOS_FLAG] == 0))
                wrong_miss = (active & (buffers["rp"][Offset.AFTER_MISS]
                                        != buffers["control_after_miss"]))
                buffers["intake_err"][0].bitwise_or_(
                    wrong_miss.to(torch.int64) * (CHAIN_ERR_AFTER_MISS | CHAIN_ERR_DRIFT))

        def intake_common():
            from jetspec.engines.jetspec_plus_plus.profiles import kernel_enabled
            if 'prior_draft_top16_ids' in buffers:
                values = buffers['row_ring'].select_pending_candidates(buffers['packet'])
                buffers['prior_draft_top16_ids'].copy_(values[0])
                buffers['prior_draft_top16_probabilities'].copy_(values[1])
                buffers['prior_draft_token_probabilities'].copy_(values[2])
                buffers['prior_draft_pinned'].copy_(values[3])
                buffers['prior_draft_token_ids'].copy_(buffers['row_ring'].select_pending(buffers['packet'])[0])
            if fused:
                fused_intake(buffers, max_length=max_length)
                if not kernel_enabled('chain_mr_glue_lean', '1'):
                    after_miss_control()  # the fused kernel writes rp[AFTER_MISS]; serving's command check follows it
                return
            buffers["intake_err"].zero_()
            invalid = ((buffers["packet"][OutcomeOffset.CMD] != 1)
                       | (buffers["packet"][OutcomeOffset.ROUND]
                          != buffers["expected_round"]))
            buffers["intake_err"][0].bitwise_or_(
                invalid.to(torch.int64) * (CHAIN_ERR_PACKET | CHAIN_ERR_DRIFT))
            if "control_round" in buffers:
                control_bad = ((buffers["packet"][OutcomeOffset.ROUND]
                                != buffers["control_round"])
                               | (buffers["packet"][OutcomeOffset.L]
                                  != buffers["control_L"]))
                buffers["intake_err"][0].bitwise_or_(
                    control_bad.to(torch.int64) * (CHAIN_ERR_CONTROL | CHAIN_ERR_DRIFT))
            # Invalid legacy envelopes must reach Seg P's named error without
            # feeding negative C1/decoder indices to a captured kernel.
            buffers["packet"][OutcomeOffset.L].copy_(torch.where(
                invalid, buffers["safe_length"],
                buffers["packet"][OutcomeOffset.L]))
            patch_round_params_from_outcome(buffers["rp"], buffers["packet"])
            if "rng_table" in buffers:
                buffers["rng_table"].fill(buffers["rp"])
            if "tap_store" in buffers and "taps" in buffers:
                buffers["tap_store"].append(buffers["packet"], buffers["taps"])
            if "front" in buffers and "taps" in buffers:
                if "admission_hidden" in buffers:
                    index, _, _ = buffers["row_ring"].index_for_pending(buffers["packet"])
                    previous = buffers["admission_hidden"].index_select(0, index)
                    first_round = buffers["packet"][OutcomeOffset.ROUND] == 0
                    buffers["front"].staged_hidden.copy_(torch.where(
                        first_round, previous, buffers["front"].staged_hidden))
                fill_front_from_outcome(buffers["front"], buffers["packet"],
                                        buffers["taps"], buffers["tap_store"])
            append_outcome(buffers["tokens"], buffers["row_ring"],
                           buffers["packet"], buffers["intake_err"],
                           hash_weights=buffers["hash_weights"],
                           max_length=max_length,
                           lane_positions=buffers.get("append_lanes"))
            pending, top1, passes, exact, valid = buffers["row_ring"].select_pending(
                buffers["packet"])
            if "append_lanes" in buffers:
                torch.where(valid, pending[1:], torch.zeros_like(pending[1:]),
                            out=buffers["tokens"].pending_draft_ids)
            else:
                buffers["tokens"].pending_draft_ids.copy_(torch.where(
                    valid, pending[1:], torch.zeros_like(pending[1:])))
            if "prior_p_top1" in buffers:
                if "append_lanes" in buffers:
                    torch.where(valid, top1, torch.zeros_like(top1),
                                out=buffers["prior_p_top1"])
                else:
                    buffers["prior_p_top1"].copy_(torch.where(
                        valid, top1, torch.zeros_like(top1)))
            buffers["rp"][Offset.AFTER_MISS].copy_((~exact).to(torch.int64))
            after_miss_control()
            buffers["rp"][Offset.TAU_IDX].copy_(torch.where(
                valid, (passes - 1).clamp(0, tau_depth), torch.zeros_like(passes)))
            correction = buffers["tokens"].committed_ids.index_select(
                0, buffers["rp"][Offset.L].clamp(
                    0, buffers["tokens"].committed_ids.numel() - 1).reshape(1))[0]
            wrong_anchor = exact & (pending[0] != correction)
            buffers["intake_err"][0].bitwise_or_(
                wrong_anchor.to(torch.int64) * (CHAIN_ERR_PENDING | CHAIN_ERR_DRIFT))
        def with_intake():
            intake_common()
            first()
            buffers["err"].bitwise_or_(buffers["intake_err"])

        def intake_only():
            intake_common()
            bodies["A0"]()
            buffers["err"].bitwise_or_(buffers["intake_err"])

        def with_error_home():
            from jetspec.engines.jetspec_plus_plus.profiles import kernel_enabled
            if '_t14_publication' in buffers:
                buffers['publisher'].home.copy_(buffers['publisher'].wire, non_blocking=True)
                buffers['err_host'].copy_(buffers['err'], non_blocking=True)
                return
            if 'final_draft_top16_ids' in buffers:
                count = buffers['final_draft_top16_ids'].shape[0]
                buffers['wire_draft_top16_ids'].fill_(-1)
                buffers['wire_draft_top16_probabilities'].zero_()
                buffers['wire_draft_top16_ids'][:count].copy_(buffers['final_draft_top16_ids'])
                buffers['wire_draft_top16_probabilities'][:count].copy_(buffers['final_draft_top16_probabilities'])
                buffers['wire_draft_token_probabilities'].zero_()
                buffers['wire_draft_pinned'].zero_()
                buffers['wire_draft_token_probabilities'][:count].copy_(buffers['final_draft_token_probabilities'])
                buffers['wire_draft_pinned'][:count].copy_(buffers['final_draft_pinned'])
                buffers['row_ring'].publish_candidates(buffers['packet'][OutcomeOffset.ROUND] + 1,
                    buffers['wire_draft_top16_ids'], buffers['wire_draft_top16_probabilities'],
                    buffers['wire_draft_token_probabilities'], buffers['wire_draft_pinned'])
            if (kernel_enabled('chain_publish_fused', '1')
                    and "publisher" in buffers and buffers["packet"].is_cuda):
                buffers["publisher"].fused_body(
                    buffers["packet"], buffers["final_tokens"], buffers["keys"],
                    buffers["final_p_top1"], buffers["passes_run"],
                    buffers["published"], buffers["wire_passes_run"],
                    buffers["row_ring"])
                buffers["err_host"].copy_(buffers["err"], non_blocking=True)
                return
            eos = buffers["packet"][OutcomeOffset.EOS_FLAG].to(torch.bool)
            buffers["final_tokens"].masked_fill_(eos, -1)
            if "final_p_top1" in buffers:
                buffers["final_p_top1"].masked_fill_(eos, 0)
            buffers["keys"].masked_fill_(eos, -1)
            if "published" in buffers:
                buffers["published"].logical_and_(~eos)
            publish()
            if "published" in buffers:
                row_source = buffers.get("wire_rows", buffers["final_tokens"])
                key_source = buffers.get("wire_keys", buffers["keys"])
                score_source = buffers.get("wire_p_top1", buffers["final_p_top1"])
                pass_source = buffers["wire_passes_run"]
                pass_source.zero_()
                pass_source[:buffers["passes_run"].shape[0]].copy_(buffers["passes_run"])
                rows = buffers["row_ring"].rows.shape[1]
                buffers["row_ring"].publish_masked(
                    buffers["packet"][OutcomeOffset.ROUND] + 1,
                    row_source[:rows], key_source[:rows], buffers["published"],
                    score_source[:rows], pass_source[:rows])
            buffers["err_host"].copy_(buffers["err"], non_blocking=True)

        wrapped = {**bodies, "P": with_error_home}
        if "A0" in bodies:
            wrapped["A0"] = intake_only
        for name in ("A", "A_inc"):
            if name in bodies:
                first = bodies[name]
                def with_selected_intake(first=first):
                    intake_common()
                    first()
                    if not fuse_intake_error:
                        buffers["err"].bitwise_or_(buffers["intake_err"])
                wrapped[name] = with_selected_intake
        return wrapped

    def prepare(self, key: ChainKey, run_config=None) -> _Entry:
        from jetspec.engines.jetspec_plus_plus.profiles import kernel_enabled
        run_config = run_config if run_config is not None else getattr(self, 'run_config', None)
        from jetspec.engines.jetspec_plus_plus.draft.config import setting, required_setting, as_run_config
        if key in self.entries:
            self.entries.move_to_end(key)
            return self.entries[key]
        if self._prewarmed:
            raise RuntimeError(f"chain key {key} was not captured at startup")
        buffers, bodies = self.build(key)
        redraft = setting(run_config, 'engine.engine_settings.chain_miss_redraft', "0") == "1"
        names = ((SEGMENTS) + tail_segments(run_config=run_config)
                 + tuple(name for name in bodies if extended_segment(name)))
        append = context_append_enabled(run_config=run_config) and "bucket" in buffers
        if append:
            names += ("A_inc",)
        if redraft:
            names += ("A0", "A1_miss", "A1_hyp")
        if set(bodies) != set(names):
            raise ValueError(f"chain bodies must be {names}")
        bodies = self._wrap_intake(buffers, bodies, max_length=key.L_bucket, run_config=run_config)
        graphs = {}
        # The launch policy lists A once; capture all selected A variants per key.
        a_parts = (("A",))
        if append:
            a_parts += ("A_inc",)
        if redraft:
            a_parts += (("A0", "A1_miss", "A1_hyp"))
        capture = tuple(part for name in self.capture_names
                        for part in (a_parts if name == "A" else (name,)))
        for name in capture:
            started = time.perf_counter()
            graphs[name] = (self._segment_pool_capture(name, bodies[name])
                            if self._shared_segment_pools() else self.capture(bodies[name]))
            if setting(run_config, 'engine.engine_settings.chain_capture_stamps', "0") == "1":
                print("CHAIN_CAPTURE_WALL " + json.dumps(dict(
                    key=str(key), segment=name,
                    seconds=round(time.perf_counter() - started, 6))), flush=True)
        if "c1_state" in buffers:
            # A1 captures pointers produced by A0's private pool. Keep that
            # capture's tensors alive even when the T0 eager shadow replaces
            # c1_state with its own temporary first-block state.
            buffers["c1_capture_state"] = buffers["c1_state"]
        entry = _Entry(buffers, bodies, graphs)
        self.entries[key] = entry
        if len(self.entries) > self.capacity:
            self.entries.popitem(last=False)
        return entry

    def prewarm(self, keys, *, after_capture=None, run_config=None) -> None:
        """Capture the complete served key set before measured requests."""
        run_config = run_config if run_config is not None else getattr(self, 'run_config', None)
        unique = tuple(dict.fromkeys(keys))
        if self._prewarmed:
            raise RuntimeError("chain keys were already prewarmed")
        if len(unique) > self.capacity:
            raise ValueError("chain cache cannot hold every startup key")
        for key in unique:
            entry = self.prepare(key, run_config=run_config)
            if after_capture is not None:
                after_capture(key, entry.buffers)
        self._prewarmed = True

    @staticmethod
    def chosen(J_round: int, refresh: bool, *, ctx_append=False, run_config=None) -> tuple[str, ...]:
        from jetspec.engines.jetspec_plus_plus.profiles import kernel_enabled
        schedule = Schedule(J_round, int(refresh))
        names = (("A_inc" if ctx_append else "A",))
        return names + schedule_segments(schedule)[1:] + tail_segments(refresh, run_config=run_config) + ("P",)

    def run(self, key: ChainKey, *, J_round: int, refresh: bool,
            on_replay=None, on_first_launch=None,
            before_a1=None, pre_a0_event=None, miss=False, after_redraft=None,
            ctx_append=False, run_config=None) -> dict:
        # prepare is deliberately outside the timed region, including its warm-up.
        from jetspec.engines.jetspec_plus_plus.trace.anatomy_lite import enabled as anatomy_enabled
        from jetspec.engines.jetspec_plus_plus.profiles import kernel_value
        run_config = run_config if run_config is not None else getattr(self, 'run_config', None)
        from jetspec.engines.jetspec_plus_plus.draft.config import setting, required_setting, as_run_config
        entry = self.prepare(key, run_config=run_config)
        if miss and after_redraft is None:
            raise ValueError("J0-only redraft requires a second publisher")
        # The same-round row stays J0-only. The following hypothesis set uses
        # the fixed cell's refresh (and swap), just as a normal hit round does.
        names = (("A0", "A1_miss", "A1_hyp")
                 + schedule_segments(Schedule(J_round, int(refresh)))[1:]
                 + tail_segments(refresh, run_config=run_config) + ("P",)
                 if miss else self.chosen(J_round, refresh, ctx_append=ctx_append, run_config=run_config))
        missing = set(names) - set(entry.graphs)
        if missing:
            raise RuntimeError(f"chain key {key} lacks captured segments {sorted(missing)}")
        shadow_buffers = shadow_bodies = shadow_expected = None
        if self.check:
            if self.shadow_state is not None:
                # Diagnostic served T0: execute the fused bodies from an exact
                # checkpoint, restore it, then replay the graph on the same
                # packet/prefix. The two executions cannot observe each other's
                # writes. This is intentionally outside the timed round path.
                state = tuple(self.shadow_state(key, entry.buffers))
                if not state or any(not isinstance(value, torch.Tensor) for value in state):
                    raise RuntimeError("served chain shadow needs tensor state")
                saved = tuple(value.clone() for value in state)
                for name in names:
                    entry.bodies[name]()
                compared = OUTPUTS + ("wire_rows", "wire_keys", "wire_p_top1",
                                      "redraft_wire")
                shadow_expected = {name: entry.buffers[name].clone()
                                   for name in compared if name in entry.buffers}
                if "tokens" in entry.buffers:
                    shadow_expected["committed_ids"] = entry.buffers["tokens"].committed_ids.clone()
                for value, original in zip(state, saved):
                    value.copy_(original)
            else:
                if "tokens" in entry.buffers and self.shadow_build is None:
                    raise RuntimeError("served chain shadow requires an independent state builder")
                shadow_buffers, shadow_bodies = (
                    self.shadow_build(key, entry.buffers) if self.shadow_build is not None
                    else self.build(key))
                if ("tokens" in entry.buffers and
                        shadow_buffers["tokens"].committed_ids.data_ptr() ==
                        entry.buffers["tokens"].committed_ids.data_ptr()):
                    raise RuntimeError("served chain shadow aliases committed IDs")
                shadow_bodies = self._wrap_intake(shadow_buffers, shadow_bodies,
                                                  max_length=key.L_bucket, run_config=run_config)
        cuda = entry.buffers["rp"].device.type == "cuda"
        profiled = cuda and not self.check and self.kernel_profile.begin()
        events = []
        launch_name = ("A1_miss" if miss else
                       "A1_inc" if "A1_inc" in names else
                       "A1" if "A1" in names else
                       "A_inc" if "A_inc" in names else "A")
        for name in names:
            if cuda:
                start, end = self.event_factory(), self.event_factory()
                start.record()
            if name == launch_name and on_first_launch is not None:
                on_first_launch()
            with self.kernel_profile.segment(name):
                entry.graphs[name]()
            if cuda:
                end.record()
                events.append((name, start, end))
            if name == "A1_miss":
                if cuda:
                    end.synchronize()
                    self.round_syncs += 1
                if "redraft_err_host" in entry.buffers:
                    redraft_error = int(entry.buffers["redraft_err_host"][0])
                    if redraft_error & CHAIN_ERR_AFTER_MISS:
                        raise RuntimeError("chain host AFTER_MISS differs from device lookup")
                    if redraft_error:
                        raise RuntimeError("chain miss redraft failed before delivery")
                after_redraft(entry.buffers, entry.buffers["packet"])
        if on_replay is not None and not self.check:
            # Enqueued replays run while the host pre-stages the next round.
            on_replay()
        if self.check and shadow_bodies is not None:
            for name in names:
                shadow_bodies[name]()
        if cuda:
            # The publish event is the only round synchronization.
            events[-1][2].synchronize()
            self.round_syncs += 1
            self.last_ms[key] = {name: start.elapsed_time(end) for name, start, end in events}
            if (setting(run_config, 'engine.engine_settings.chain_tail_trace', "0") == "1"
                    or anatomy_enabled(run_config)):
                # First segment start to publish end, launch gaps included.
                self.last_span_ms[key] = events[0][1].elapsed_time(events[-1][2])
            if setting(run_config, 'engine.engine_settings.chain_glue2_profile', "0") == "1":
                print("CHAIN_SEGMENT_MS " + json.dumps(dict(
                    glue=kernel_value('chain_sega_glue', "0"),
                    key=str(key), segments=self.last_ms[key])), flush=True)
            if profiled:
                from jetspec.engines.jetspec_plus_plus.wire.contract import OutcomeOffset as _Outcome
                self.kernel_profile.finish(
                    names, key=key.L_bucket, J=J_round, miss=bool(miss),
                    L=int(entry.buffers["packet"][_Outcome.L]),
                    segments_ms=self.last_ms[key])
            if names[:2] in (("A0", "A1"), ("A0", "A1_miss")):
                self.last_a0_to_a1_gap_ms[key] = events[0][2].elapsed_time(events[1][1])
        if "err_host" in entry.buffers and int(entry.buffers["err_host"][0]) & CHAIN_ERR_PACKET:
            raise ChainPublishedError("chain outcome packet CMD/ROUND mismatch")
        if "err_host" in entry.buffers and int(entry.buffers["err_host"][0]) & CHAIN_ERR_CONTROL:
            raise ChainDriftError("chain ATTN3 host control round/L differs from device packet")
        if "err_host" in entry.buffers and int(entry.buffers["err_host"][0]) & CHAIN_ERR_AFTER_MISS:
            raise ChainDriftError("chain host AFTER_MISS differs from device lookup")
        if "err_host" in entry.buffers and int(entry.buffers["err_host"][0]) & CHAIN_ERR_PENDING:
            raise ChainDriftError("chain pending row anchor differs from committed correction")
        if self.ctx_kv_check:
            report = entry.buffers["bucket"].head.check_chain_context_kv(
                incremental=ctx_append)
            self.ctx_kv_checked += 1
            exact = all(layer["exact"] for layer in report)
            accepted = all(layer["accepted"] for layer in report)
            if not exact:
                self.ctx_kv_tolerated += 1
            print("CTX_KV_APPEND_CHECK " + json.dumps(dict(
                round=self.ctx_kv_checked, key=str(key), exact=exact,
                accepted=accepted, tolerance="2x measured full-M/dirty-M noise max abs; row L2 relative <= 1e-2",
                recompute_exact=all(layer["recompute_exact"] for layer in report),
                unchanged_exact=all(layer["unchanged_exact"] for layer in report),
                selection_exact=all(layer["selection_exact"] for layer in report),
                noise_max_abs=[layer["noise_max_abs"] for layer in report],
                noise_p99_abs=[layer["noise_p99_abs"] for layer in report],
                noise_max_ulp=[layer["noise_max_ulp"] for layer in report],
                noise_p99_ulp=[layer["noise_p99_ulp"] for layer in report],
                within_floor=all(layer["within_floor"] for layer in report),
                relative_count=[layer["relative_count"] for layer in report],
                max_relative=[layer["max_relative"] for layer in report],
                offending_count=[layer["offending_count"] for layer in report],
                offending_row=[layer["offending_row"] for layer in report],
                write_path="append" if ctx_append else "full",
                max_abs=[layer["max_abs"] for layer in report],
                max_row=[layer["max_row"] for layer in report],
                max_ulp=[layer["max_ulp"] for layer in report],
                ulp_row=[layer["ulp_row"] for layer in report],
                ulp_values=[layer["ulp_values"] for layer in report])), flush=True)
            if not accepted:
                raise ChainDriftError("JETSPEC_CTX_KV_APPEND_CHECK cache exceeds measured noise floor")
        if self.check:
            reference = (shadow_expected if shadow_expected is not None else shadow_buffers)
            for name in OUTPUTS + ("wire_rows", "wire_keys", "wire_p_top1",
                                   "redraft_wire"):
                if name not in entry.buffers or name not in reference:
                    continue
                if not torch.equal(entry.buffers[name], reference[name]):
                    entry.buffers["err"].bitwise_or_(CHAIN_ERR_DRIFT)
                    raise ChainDriftError(f"JETSPEC_CHAIN_GRAPH_CHECK mismatch: {name}")
            expected_ids = (reference.get("committed_ids") if shadow_expected is not None
                            else (reference["tokens"].committed_ids
                                  if "tokens" in reference else None))
            if expected_ids is not None and not torch.equal(
                    entry.buffers["tokens"].committed_ids, expected_ids):
                entry.buffers["err"].bitwise_or_(CHAIN_ERR_DRIFT)
                raise ChainDriftError("JETSPEC_CHAIN_GRAPH_CHECK mismatch: committed_ids")
        if "err_host" in entry.buffers:
            error = int(entry.buffers["err_host"][0])
            if error & CHAIN_ERR_CAPACITY:
                self.fallback_counts["capacity"] += 1
                raise ChainCapacityFallback("chain L exceeded captured prefix capacity")
            if error & CHAIN_ERR_DRIFT:
                raise ChainDriftError("chain committed-token hash16 mismatch after publish")
            if error & CHAIN_ERR_ROW_MISSING:
                raise ChainPublishedError("chain outcome did not match a published row")
            if error:
                raise ChainPublishedError(f"chain device error after publish: {error}")
        if on_replay is not None and self.check:
            # Check mode reads graph outputs on the host. A preposted NCCL
            # receive can block those reads behind the next outcome packet.
            on_replay()
        label = f"J{J_round}-R{int(refresh)}"
        self.mix[label] = self.mix.get(label, 0) + 1
        cap_label = f"r{key.R}"
        self.cap_mix[cap_label] = self.cap_mix.get(cap_label, 0) + 1
        return entry.buffers
