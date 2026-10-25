"""Host command and fixed-envelope dispatch for one served chain sequence."""
from __future__ import annotations

from jetspec.engines.jetspec_plus_plus.trace.anatomy_lite import enabled as anatomy_enabled

from jetspec.engines.jetspec_plus_plus.profiles import kernel_enabled, kernel_value

import json
import os
import time
import atexit
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

import torch

from jetspec.engines.jetspec_plus_plus.wire.transport import ChainCommandInbox
from jetspec.engines.jetspec_plus_plus.draft.state import ChainLengthTracker
from jetspec.engines.jetspec_plus_plus.wire.transport import OutcomeReceiveRing
from jetspec.engines.jetspec_plus_plus.wire.protocol import ChainKeySnapshot
from jetspec.engines.jetspec_plus_plus.wire.protocol import chain_after_miss
from jetspec.engines.jetspec_plus_plus.draft.graph import ChainPublishedError
from jetspec.engines.jetspec_plus_plus.draft.graph import hostpath_setting
from jetspec.engines.jetspec_plus_plus.wire.contract import Offset
from jetspec.engines.jetspec_plus_plus.jetspec.env import context_append_enabled


RECORD_OUTPUTS = ("wire_rows", "wire_keys", "wire_p_top1", "wire_home",
                  "keys", "final_tokens", "final_p_top1", "passes_run",
                  "finished", "err", "published", "redraft_wire")


def chain_ctx_append_needs_full(*, previous_key, key, previous_length, length,
                                block, prior_refresh, prior_miss, miss,
                                refresh_isolated=False):
    """Select the full context graph without reading a device scalar."""
    return (previous_key != key or previous_length is None or
            length < previous_length or length - previous_length > block or
            (prior_refresh and not refresh_isolated) or prior_miss or miss)


def configure_host_lean_entry(buffers, bodies, run_config=None):
    """Bind one graph-owned control upload before the first segment is captured."""
    from jetspec.engines.jetspec_plus_plus.profiles import kernel_enabled
    from jetspec.engines.jetspec_plus_plus.draft.config import setting, required_setting, as_run_config
    packet = buffers["packet"]
    host = torch.empty(5, dtype=torch.int64, pin_memory=packet.is_cuda)
    device = torch.zeros(5, dtype=torch.int64, device=packet.device)
    if "control_L" in buffers:
        device[2].copy_(buffers["control_L"])
    buffers["host_control_cpu"] = host
    buffers["host_control_view"] = host.numpy()
    buffers["host_control_device"] = device
    buffers["expected_round"] = device[0]
    if (((setting(run_config, 'engine.engine_settings.chain_miss_redraft', "0") == "1") or (context_append_enabled(run_config=run_config)))):
        buffers["control_round"] = device[1]
        buffers["control_L"] = device[2]
    if setting(run_config, 'engine.engine_settings.chain_miss_redraft', "0") == "1":
        buffers["control_after_miss"] = device[3]
    first_names = ("A0",) if "A0" in bodies else ("A", "A_inc")
    wrapped = dict(bodies)
    for name in first_names:
        if name not in bodies:
            continue
        first = bodies[name]
        def with_round_choice(first=first):
            buffers["rp"][Offset.J_ROUND].copy_(device[4])
            first()
        wrapped[name] = with_round_choice
    return wrapped


def _host_copy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _host_copy(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_host_copy(item) for item in value]
    return value


class ChainDraftRecorder:
    """Write a bounded, self-contained draft trace after each publish sync."""

    def __init__(self, root, *, limit=300, admission=None):
        if limit < 1:
            raise ValueError("chain record limit must be positive")
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        if (self.root / "manifest.json").exists():
            raise FileExistsError(f"chain fixture already exists: {self.root}")
        self.limit = limit
        self.rounds = []
        self.states = []
        self.admissions = []
        self.last_key = None
        self.closed = False
        self.writer = ThreadPoolExecutor(max_workers=1, thread_name_prefix="chain-record")
        self.pending = []
        self.mark_request(admission)
        atexit.register(self.close)

    def mark_request(self, admission):
        if self.closed:
            return
        index = len(self.admissions)
        name = f"admission-{index:02d}.pt"
        self.pending.append(self.writer.submit(torch.save, _host_copy(admission),
                                               self.root / name))
        self.admissions.append({"request": index, "at_round": len(self.rounds),
                                "file": name})
        self.last_key = None

    def write_state(self, round_id, key, values):
        if self.closed:
            return
        name = f"state-{round_id:04d}.pt"
        self.pending.append(self.writer.submit(torch.save, _host_copy(values),
                                               self.root / name))
        self.states.append({"round": round_id, "key": asdict(key), "file": name})
        self.last_key = key

    def write_round(self, round_id, record):
        if self.closed:
            return
        name = f"round-{round_id:04d}.pt"
        self.pending.append(self.writer.submit(torch.save, _host_copy(record),
                                               self.root / name))
        self.rounds.append({"round": round_id, "file": name})
        if len(self.rounds) >= self.limit:
            self.close()

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.writer.shutdown(wait=True)
        for pending in self.pending:
            pending.result()
        manifest = {"schema": "jetspec.chain_draft_fixture/1", "limit": self.limit,
                    "rounds": self.rounds, "states": self.states,
                    "admissions": self.admissions}
        (self.root / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        print(f"CHAIN_DRAFT_RECORD rounds={len(self.rounds)} root={self.root}", flush=True)


class ChainRoundDriver:
    """Post the receive before acknowledgement, then replay and publish once."""

    def __init__(self, *, store, request_group, graph, receiver: OutcomeReceiveRing,
                 tracker: ChainLengthTracker, seq_id: int, key_for, select, publish,
                 event_factory=None, inbox=None, fallback=None, publish_redraft=None,
                 start_round=1, pre_stage=None, recorder=None, rows=None, run_config=None):
        from jetspec.engines.jetspec_plus_plus.profiles import kernel_enabled
        from jetspec.engines.jetspec_plus_plus.draft.config import setting, required_setting, as_run_config
        self.run_config = as_run_config(run_config)
        self._hostpath_setting = hostpath_setting(self.run_config)
        from jetspec.engines.jetspec_plus_plus.engine import current_engine
        self.store = store
        self.request_group = request_group
        self.graph = graph
        self.receiver = receiver
        self.tracker = tracker
        self.seq_id = seq_id
        self.key_for = key_for
        self.select = select
        self.publish = publish
        self.publish_redraft = publish_redraft
        self.fallback = fallback
        self.inbox = inbox or ChainCommandInbox(store, run_config=run_config)
        self.next_round = start_round
        self.event_factory = event_factory or (lambda: torch.cuda.Event())
        self.pre_stage = pre_stage
        self.recorder = recorder
        self.rows = rows
        self._eval_draft_delay_seconds = 0.0
        if current_engine().execution in ("pipelined", "critical-path", "on"):
            from jetspec.engines.jetspec_plus_plus.jetspec.profile import draft_delay_seconds
            self._eval_draft_delay_seconds = draft_delay_seconds(run_config=run_config)
        self._staged = None
        self._timing = setting(self.run_config, 'engine.engine_settings.chain_handoff_timing', "0") == "1"
        self._prepost = setting(self.run_config, 'engine.engine_settings.chain_prepost', "0") == "1"
        self._prepost_timing = setting(self.run_config, 'engine.engine_settings.chain_prepost_timing', "0") == "1"
        self._tail_trace = setting(self.run_config, 'engine.engine_settings.chain_tail_trace', "0") == "1"
        self._anatomy_lite = anatomy_enabled(self.run_config)
        self._host_lean = kernel_enabled('chain_host_lean', '1')
        self._post_lean = kernel_enabled('chain_post_lean', '1')
        self._preposted = None
        self._deferred = []
        self._last_p_sync_event = None
        self._release_events = ([self.event_factory() for _ in range(
            len(getattr(receiver, "buffers", (None, None))))]
                                if pre_stage is not None else None)
        # 5f: one host key snapshot per ring slot, tagged with its published round.
        self._key_snapshots = [None, None, None]
        self._home_views = {}
        self._ctx_last_key = None
        self._ctx_last_length = None
        self._ctx_prior_refresh = False
        self._ctx_prior_miss = False
        self._wire_recorder = None
        if setting(self.run_config, 'engine.engine_settings.chain_wire_record', "0") == "1" and seq_id >= 0:
            root = setting(self.run_config, 'engine.engine_settings.w280_record')
            if (not root or setting(self.run_config, 'engine.engine_settings.record_ptop1') != "1"
                    or setting(self.run_config, 'engine.engine_settings.record_qshadow') != "1"):
                raise ValueError("JETSPEC_CHAIN_WIRE_RECORD requires W280_RECORD and both E04 record flags")
            from jetspec.engines.jetspec_plus_plus.draft.records import ChainWireRecorder
            self._wire_recorder = ChainWireRecorder(root, seq_id=seq_id, run_config=self.run_config)

    def _slot(self, round_id: int) -> int:
        slot_for_round = getattr(self.receiver, "slot_for_round", None)
        return slot_for_round(round_id) if slot_for_round is not None else round_id % 2

    def snapshot_published_keys(self, home, *, redraft=False, run_config=None) -> None:
        """Keep the keys of a set this rank just placed on the wire (5f).

        ``home`` is the pinned host wire the store publish just read, so its
        single D2H already completed at the publish sync; no copy is added. The
        tag is the wire's own round word, and the slot is that round mod 3.
        """
        views = self._home_views.get(id(home))
        if views is None:
            from jetspec.engines.jetspec_plus_plus.jetspec.live_rows import candidate_wire_views
            views = candidate_wire_views(home, 1, 1, 1, run_config=run_config if run_config is not None else getattr(self, 'run_config', None))
            if len(views) != 6:
                raise RuntimeError("chain key snapshot requires the keyed candidate wire")
            self._home_views[id(home)] = views
        published_round = int(views[1][0, 1])
        # An unpublished (EOS) set enters the ring with no live key (publish_masked).
        keys = views[5][0] if int(views[0][0]) else ()
        self._key_snapshots[published_round % 3] = ChainKeySnapshot.from_row_keys(
            published_round, keys, redraft=redraft)

    def snapshot_blank_round(self, published_round: int) -> None:
        """A capacity-fallback round publishes an empty keyed set."""
        self._key_snapshots[published_round % 3] = ChainKeySnapshot(
            int(published_round), frozenset())

    @torch.inference_mode()
    def next(self, run_config=None) -> str:
        from jetspec.engines.jetspec_plus_plus.profiles import kernel_enabled
        run_config = run_config if run_config is not None else getattr(self, 'run_config', None)
        from jetspec.engines.jetspec_plus_plus.draft.config import setting, required_setting, as_run_config
        if self._hostpath_setting is not None:
            setting = self._hostpath_setting
        from jetspec.engines.jetspec_plus_plus.engine import current_engine
        tail = {"enter": time.perf_counter_ns()} if self._tail_trace else None
        command_id, command = self.inbox.next()
        lite_command_ns = time.monotonic_ns() if self._anatomy_lite else None
        command_at = time.perf_counter()
        if tail is not None:
            tail["command"] = time.perf_counter_ns()
        if command != "outcome":
            if self._preposted is not None:
                raise RuntimeError("non-outcome command followed a non-EOS prepost")
            self._flush_deferred()
            return command
        command_at = time.perf_counter()
        ctx_append_enabled = (setting(run_config, 'engine.engine_settings.ctx_kv_append', "0") == "1"
                              or context_append_enabled(run_config=run_config))
        control = self.inbox.last_outcome if (((ctx_append_enabled) or (setting(run_config, 'engine.engine_settings.chain_miss_redraft', "0") == "1"))) else None
        if (((setting(run_config, 'engine.engine_settings.chain_miss_redraft', "0") == "1"))) and control is None:
            raise RuntimeError("ATTN3 outcome command omitted exact L")
        round_id = self.next_round
        if self._prepost and (self.inbox.last_outcome is None or
                              self.inbox.last_outcome[0] != round_id or
                              self.inbox.last_eos not in (0, 1)):
            raise RuntimeError("prepost command omitted sequential round or EOS")
        redraft = False
        after_miss = None
        if setting(run_config, 'engine.engine_settings.chain_miss_redraft', "0") == "1":
            if self.inbox.last_outcome[0] != round_id:
                raise RuntimeError("miss-redraft command round differs from receiver")
            # 5f: AFTER_MISS is the shared exact-key rule over this rank's own
            # snapshot of the set it published for this round (ring slot mod 3).
            after_miss = chain_after_miss(
                self._key_snapshots[round_id % 3], round_id=round_id,
                true_k=control[2], recovery=control[3], eos=bool(self.inbox.last_eos))
            redraft = after_miss == 1
            from jetspec.engines.jetspec_plus_plus.jetspec.profile import critical_path
            redraft = redraft or (critical_path() and round_id > 0 and not self.inbox.last_eos)
        if redraft and self.publish_redraft is None:
            raise RuntimeError("miss-redraft has no second publisher")
        if self._tail_trace:
            print(f"MAILBOX_ID side=draft event=branch command_id={command_id} round={round_id} control={control} branch={'eos' if self.inbox.last_eos else 'redraft' if redraft else 'hit'}", flush=True)
        stamps = {"command": time.perf_counter_ns()} if self._timing else None
        if self._preposted is None:
            self.receiver.post(round_id, group=self.request_group,
                               store=self.store, command_id=command_id)
        else:
            if self._preposted != (round_id, command_id) or not self.receiver.is_posted(
                    round_id, command_id):
                raise RuntimeError("preposted receive differs from command counter")
            self._preposted = None
        if stamps is not None:
            stamps["posted"] = time.perf_counter_ns()
        packet, taps = self.receiver.take(round_id)
        if self._wire_recorder is not None:
            # Resolve the ORIGINAL set before a miss recovery replaces its ring slot.
            self._wire_recorder.resolve(packet)
        if stamps is not None:
            stamps["receive"] = time.perf_counter_ns()
        if tail is not None:
            tail["receive"] = time.perf_counter_ns()
        self.receiver.stage_mirror(round_id)
        if stamps is not None:
            stamps["mirror"] = time.perf_counter_ns()
        staged = self._staged if (self._staged is not None and
                                  self._staged[0] == round_id) else None
        self._staged = None
        key = staged[1] if staged is not None else self.key_for(self.tracker, round_id)
        if stamps is not None:
            stamps["key"] = time.perf_counter_ns()
        failure = None
        post_finished = False
        if key is None:
            if self._tail_trace:
                print(f"MAILBOX_ID side=draft event=branch command_id={command_id} round={round_id} branch=capacity_fallback", flush=True)
            if self.fallback is None:
                raise RuntimeError("chain capacity fallback requires a host publisher")
            key = self.fallback(packet, taps, round_id)
        if key is None:
            if hasattr(self.receiver, "mark_copy_done"):
                self.receiver.mark_copy_done(round_id)
            if after_miss is not None:
                self.snapshot_blank_round(round_id + 1)
        else:
            entry = staged[2] if staged is not None else self.graph.prepare(key)
            if self._anatomy_lite:
                entry.buffers['_lite_command_ns'] = lite_command_ns
            if stamps is not None:
                stamps["prepare"] = time.perf_counter_ns()
            if self._host_lean and "envelope" in entry.buffers and hasattr(self.receiver, "buffers"):
                entry.buffers["envelope"].copy_(
                    self.receiver.buffers[self._slot(round_id)],
                    non_blocking=True)
            else:
                entry.buffers["packet"].copy_(packet, non_blocking=True)
                entry.buffers["taps"].copy_(taps, non_blocking=True)
                if "hybrid_recurrent" in entry.buffers:
                    source = self.receiver.hybrid_views[self._slot(round_id)]
                    for name, value in zip(("hybrid_recurrent", "hybrid_conv",
                                            "hybrid_keys", "hybrid_values"), source):
                        entry.buffers[name].copy_(value, non_blocking=True)
            if "hybrid_recurrent" in entry.buffers:
                from jetspec.engines.jetspec_plus_plus.wire.hybrid import append_hybrid_kv
                front = entry.buffers["front"]
                front.hybrid_recurrent.copy_(entry.buffers["hybrid_recurrent"])
                front.hybrid_conv.copy_(entry.buffers["hybrid_conv"])
                if setting(run_config, 'engine.engine_settings.hybrid_state_audit', "0") == "1" and self.seq_id >= 0:
                    from jetspec.engines.jetspec_plus_plus.wire.hybrid import state_digest
                    print("HYBRID_STATE_DRAFT " +
                          f"round={int(entry.buffers['packet'][4].item())} " +
                          f"sha256={state_digest(front.hybrid_recurrent, front.hybrid_conv)}",
                          flush=True)
                append_hybrid_kv(
                    front.hybrid_prefix_key, front.hybrid_prefix_value,
                    entry.buffers["hybrid_keys"], entry.buffers["hybrid_values"],
                    entry.buffers["packet"][8], entry.buffers["packet"][5])
            if hasattr(self.receiver, "mark_copy_done"):
                self.receiver.mark_copy_done(round_id)
            if not self._host_lean:
                if "expected_round" in entry.buffers:
                    entry.buffers["expected_round"].fill_(round_id)
                if control is not None and "control_round" in entry.buffers:
                    entry.buffers["control_round"].fill_(control[0])
                    entry.buffers["control_L"].fill_(control[1])
                    if after_miss is not None and "control_after_miss" in entry.buffers:
                        entry.buffers["control_after_miss"].fill_(after_miss)
            if stamps is not None:
                stamps["copy"] = time.perf_counter_ns()
            J_round, refresh = self.select(key)
            ctx_append = False
            if ctx_append_enabled:
                if control is None:
                    raise RuntimeError("context KV append requires exact host L")
                ctx_append = not chain_ctx_append_needs_full(
                    previous_key=self._ctx_last_key, key=key,
                    previous_length=self._ctx_last_length, length=control[1],
                    block=key.lanes, prior_refresh=self._ctx_prior_refresh,
                    prior_miss=self._ctx_prior_miss, miss=redraft,
                    refresh_isolated=setting(run_config, 'method.method_settings.chain_fixed_config') in
                    ("J1-R1", "J2-R1", "J3-R1", "J2-R2", "J3-R3") or
                    (current_engine().execution in ("pipelined", "critical-path", "on") and bool(refresh)))
            if self._host_lean:
                values = entry.buffers["host_control_view"]
                values[:] = (round_id, control[0] if control is not None else 0,
                             control[1] if control is not None else 0,
                             after_miss if after_miss is not None else 0, J_round)
                entry.buffers["host_control_device"].copy_(
                    entry.buffers["host_control_cpu"], non_blocking=True)
            if 'after_swap_fill_prepare' in entry.buffers:
                entry.buffers['after_swap_fill_prepare'](
                    seq_id=self.seq_id, candidate_round=round_id + 1)
            recording = self.recorder is not None and not self.recorder.closed
            record_index = len(self.recorder.rounds) if recording else None
            if recording and self.recorder.last_key != key:
                from jetspec.engines.jetspec_plus_plus.draft.segments import named_chain_state_tensors

                self.recorder.write_state(record_index, key,
                                          named_chain_state_tensors(entry.buffers, self.rows))
            if stamps is not None:
                stamps["select"] = time.perf_counter_ns()
            scored_passes = None
            if self._wire_recorder is not None:
                scored_passes = (1 if redraft else int(
                    entry.buffers["row_ring"].select_pending(packet)[2].item()))
            a0_enqueue_at = time.perf_counter()
            plan_ms = 0.0
            a1_launch_at = None


            try:
                redraft_done = None
                eval_started = time.perf_counter() if 'eval_audit' in entry.buffers else None
                if self._eval_draft_delay_seconds:
                    delay_started = time.perf_counter()
                    time.sleep(self._eval_draft_delay_seconds)
                    self._defer_print('EVAL_DRAFT_DELAY ', dict(
                        seq_id=self.seq_id, round=round_id,
                        requested_ms=self._eval_draft_delay_seconds * 1000,
                        elapsed_ms=(time.perf_counter() - delay_started) * 1000))
                start_event = None
                prior_p_sync_event = self._last_p_sync_event
                if (((self.pre_stage is None) and (stamps is None) and (not redraft) and (not ctx_append_enabled) and (not self._prepost) and (not self._prepost_timing) and (not self._post_lean) and (tail is None))):
                    self.graph.run(key, J_round=J_round, refresh=refresh)
                else:
                    options = {}
                    if self.pre_stage is not None or self._prepost or self._post_lean:
                        def after_replay():
                            nonlocal post_finished
                            if self._prepost and not self.inbox.last_eos:
                                self.receiver.post(round_id + 1, group=self.request_group,
                                                   store=self.store, command_id=command_id + 1)
                                self._preposted = (round_id + 1, command_id + 1)
                            if self.pre_stage is not None:
                                self._stage_following(round_id, stamps)
                            if self._post_lean:
                                self._finish_post(round_id)
                                post_finished = True
                            self._flush_deferred()
                        options["on_replay"] = after_replay
                    if stamps is not None or self._prepost_timing or tail is not None:
                        # With ATTN3 the graph stamps launch at A1, after the exact-L plan.
                        def on_first_launch():
                            nonlocal start_event
                            if tail is not None:
                                tail["launch"] = time.perf_counter_ns()
                            if stamps is not None:
                                stamps["launch"] = time.perf_counter_ns()
                            if self._prepost_timing and entry.buffers["packet"].is_cuda:
                                start_event = torch.cuda.Event(enable_timing=True)
                                start_event.record()
                        options["on_first_launch"] = on_first_launch
                        if stamps is not None:
                            stamps["run_call"] = time.perf_counter_ns()
                    if redraft:
                        def second_delivery(buffers, packet):
                            nonlocal redraft_done
                            if tail is not None:
                                tail["redraft_sync"] = time.perf_counter_ns()
                            self.publish_redraft(buffers, round_id)
                            if "eval_redraft_audit" in buffers and not self.inbox.last_eos:
                                from jetspec.engines.jetspec_plus_plus.jetspec.profile import round_audit
                                self._defer_print("EVAL_ROUND ", round_audit(
                                    buffers, seq_id=self.seq_id, round_id=round_id,
                                    j=J_round, r=int(refresh), rows=key.R, redraft=True, run_config=run_config))
                            redraft_done = time.perf_counter()
                            if tail is not None:
                                tail["redraft_published"] = time.perf_counter_ns()
                            if "redraft_home" in buffers:
                                self.snapshot_published_keys(buffers["redraft_home"],
                                                             redraft=True, run_config=run_config if run_config is not None else getattr(self, 'run_config', None))
                        options.update(miss=True, after_redraft=second_delivery)
                    if ctx_append_enabled:
                        options["ctx_append"] = ctx_append
                    self.graph.run(key, J_round=J_round, refresh=refresh, **options)
                if self._anatomy_lite:
                    entry.buffers['_lite_chain_end_ns'] = time.monotonic_ns()
                if ctx_append_enabled:
                    self._ctx_last_key = key
                    self._ctx_last_length = control[1]
                    self._ctx_prior_refresh = refresh
                    self._ctx_prior_miss = redraft
                if tail is not None:
                    tail["sync"] = time.perf_counter_ns()
                if self._prepost_timing and entry.buffers["packet"].is_cuda:
                    if prior_p_sync_event is not None and start_event is not None:
                        print("CHAIN_PREPOST_TIMING " + json.dumps(dict(
                            round=round_id, p_sync_to_next_a_ms=round(
                                prior_p_sync_event.elapsed_time(start_event), 3))),
                              flush=True)
                    self._last_p_sync_event = torch.cuda.Event(enable_timing=True)
                    self._last_p_sync_event.record()
            except ChainPublishedError as error:
                # Graph errors are raised after Seg P, so its immutable wire still
                # has to reach the verifier before the draft rank stops.
                failure = error
                self._ctx_last_key = None
            if recording:
                self.recorder.write_round(record_index, dict(
                    round=round_id, key=asdict(key), J_round=J_round,
                    request=len(self.recorder.admissions) - 1,
                    command=getattr(self.inbox, "last_raw_command", None),
                    control=control, redraft=redraft,
                    reserved_bytes=torch.cuda.memory_reserved(entry.buffers["packet"].device),
                    captured_segments=sum(len(item.graphs)
                                          for item in self.graph.entries.values()),
                    refresh=refresh, packet=packet, taps=taps,
                    outputs={name: entry.buffers[name] for name in RECORD_OUTPUTS
                             if name in entry.buffers}))
            if setting(run_config, 'engine.engine_settings.chain_miss_redraft', "0") == "1":
                entry.buffers["host_redrafted"] = redraft
            deferred_publish = self.publish(entry.buffers, round_id)
            if eval_started is not None and not self.inbox.last_eos:
                eval_done = time.perf_counter()
                self._defer_print('EVAL_DRAFT_TIMING ', dict(
                    seq_id=self.seq_id, round=round_id,
                    total_ms=(eval_done - eval_started) * 1000,
                    redraft_ms=(None if redraft_done is None else
                                (redraft_done - eval_started) * 1000)))
            if "eval_audit" in entry.buffers and not self.inbox.last_eos:
                from jetspec.engines.jetspec_plus_plus.jetspec.profile import round_audit
                self._defer_print("EVAL_ROUND ", round_audit(
                    entry.buffers, seq_id=self.seq_id, round_id=round_id + 1,
                    j=J_round, r=int(refresh), rows=key.R, run_config=run_config))
            if self._wire_recorder is not None:
                if redraft:
                    self._wire_recorder.redraft(entry.buffers, round_id=round_id, run_config=run_config)
                self._wire_recorder.capture(entry.buffers, round_id=round_id + 1,
                                            scored_passes=scored_passes, run_config=run_config if run_config is not None else getattr(self, 'run_config', None))
            if tail is not None:
                tail["published"] = time.perf_counter_ns()
                self._defer_print("CHAIN_TAIL_TRACE ", dict(
                    seq_id=self.seq_id, round=round_id, key=getattr(key, "L_bucket", key),
                    L=(None if self.inbox.last_outcome is None
                       else self.inbox.last_outcome[1]),
                    staged=staged is not None, redraft=redraft, stamps_ns=tail,
                    segments_ms=getattr(self.graph, "last_ms", {}).get(key, {}),
                    device_span_ms=getattr(self.graph, "last_span_ms", {}).get(key)))
            if after_miss is not None and "wire_home" in entry.buffers:
                self.snapshot_published_keys(entry.buffers["wire_home"], run_config=run_config if run_config is not None else getattr(self, 'run_config', None))
            if (self._prepost or self._post_lean) and deferred_publish is not None:
                self._deferred.append(deferred_publish)
            if redraft and setting(run_config, 'engine.engine_settings.chain_miss_redraft_timing', "0") == "1":
                parts = self.graph.last_ms.get(key, {})
                self._defer_print("CHAIN_MISS_REDRAFT_TIMING ", dict(
                    round=round_id, redraft_ms=(redraft_done - command_at) * 1000,
                    redraft_device_ms=parts.get("A0", 0.0) + parts.get("A1_miss", 0.0),
                    round_ms=(time.perf_counter() - command_at) * 1000))
            if stamps is not None:
                stamps["published"] = time.perf_counter_ns()
                us = lambda start, end: round((stamps[end] - stamps[start]) / 1000, 1)
                self._defer_print("CHAIN_HANDOFF_DRAFT ", dict(
                    seq_id=self.seq_id, round=round_id, staged=staged is not None,
                    post_us=us("command", "posted"),
                    receive_wait_us=us("posted", "receive"),
                    mirror_us=us("receive", "mirror"),
                    key_us=us("mirror", "key"),
                    prepare_us=us("key", "prepare"),
                    copy_us=us("prepare", "copy"),
                    select_us=us("copy", "select"),
                    run_preamble_us=us("run_call", "launch"),
                    receive_to_launch_us=us("receive", "launch"),
                    replay_stage_us=round(stamps.get("stage_us", 0), 1)))
        if not post_finished:
            self._finish_post(round_id)
        self.next_round += 1
        if self._prepost and self.inbox.last_eos:
            self._flush_deferred()
        if failure is not None:
            raise failure
        return "outcome"

    def _finish_post(self, round_id: int) -> None:
        slot = self._slot(round_id)
        event = (self._release_events[slot] if self._release_events is not None
                 else self.event_factory())
        event.record()
        self.receiver.release(round_id, event)
        if self._post_lean:
            mirror_event = self.receiver.mirror_events[slot]
            if mirror_event is not None and hasattr(mirror_event, "synchronize"):
                mirror_event.synchronize()
        for outcome in self.receiver.poll_mirrors():
            if outcome.seq_id != self.seq_id:
                raise RuntimeError("chain outcome changed sequence ID")
            self.tracker.mirror(round_id=outcome.round, L=outcome.L,
                                eos=bool(outcome.eos_flag))
            if self._prepost and outcome.round == round_id and bool(outcome.eos_flag) != bool(
                    self.inbox.last_eos):
                raise RuntimeError("prepost host EOS differs from outcome mirror")

    def _stage_following(self, round_id: int, stamps=None) -> None:
        start = time.perf_counter_ns() if stamps is not None else None
        self._staged = self.pre_stage(self.tracker, round_id + 1)
        if stamps is not None:
            stamps["stage_us"] = (time.perf_counter_ns() - start) / 1000

    def _defer_print(self, prefix: str, payload: dict) -> None:
        def emit():
            print(prefix + json.dumps(payload), flush=True)
        if self._prepost or self._post_lean:
            self._deferred.append(emit)
        else:
            emit()

    def _flush_deferred(self) -> None:
        deferred, self._deferred = self._deferred, []
        for emit in deferred:
            emit()
