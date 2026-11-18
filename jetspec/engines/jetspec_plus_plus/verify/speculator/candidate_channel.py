from __future__ import annotations
import time
from functools import partial
import torch
import torch.distributed as dist
from jetspec.engines.jetspec_plus_plus.wire.outcome_packet import announce_chain_command
from jetspec.engines.jetspec_plus_plus.serving.sequence import Sequence
from jetspec.engines.jetspec_plus_plus.wire.protocol import CandidateReceiver
from jetspec.engines.jetspec_plus_plus.wire.protocol import CandidateDelivery
from jetspec.engines.jetspec_plus_plus.wire.protocol import unpack_candidate_delivery
from jetspec.engines.jetspec_plus_plus.wire.protocol import PackedCandidateBuffers
from jetspec.engines.jetspec_plus_plus.wire.protocol import anchor_mode
from jetspec.engines.jetspec_plus_plus.wire.protocol import candidate_wire_shape
from jetspec.engines.jetspec_plus_plus.wire.protocol import branch_count
from jetspec.engines.jetspec_plus_plus.wire.protocol import ChainKeySnapshot
from jetspec.engines.jetspec_plus_plus.wire.pair_transport import PairOutcomeTransport
from jetspec.engines.jetspec_plus_plus.wire.pair_transport import start_command_ids

class CandidateChannel:
    def _chain_note_delivery(self, candidate_set) -> None:
        """Count first/second deliveries; a set arriving after its round was taken
        and not waited for is a redraft the verifier did not use (REDRAFT_UNUSED)."""
        key = (int(candidate_set.seq_id), int(candidate_set.round_id))
        if key in self._chain_expected:
            return
        if key[1] <= self._chain_taken_round.get(key[0], -1):
            if key in self._chain_late_ok:
                self._chain_late_ok.discard(key)  # the late tail of a counted REDRAFT_MISSING
            else:
                self.metrics["jetspec_chain_redraft_unused"] = (
                    self.metrics.get("jetspec_chain_redraft_unused", 0) + 1)
            return
        deliveries = self._chain_deliveries.get(key, 0)
        from jetspec.engines.jetspec_plus_plus.runtime.profile import enabled as profile_enabled
        if profile_enabled() and deliveries == 0:
            # The mailbox can replace this set with its exact redraft before
            # speculate takes it. Copy keys now; receive storage is borrowed.
            snapshots = getattr(self, "_chain_first_key_snapshots", None)
            if snapshots is None:
                self._chain_first_key_snapshots = snapshots = {}
            snapshots[key] = ChainKeySnapshot.from_candidate(candidate_set, round_id=key[1])
        self._chain_deliveries[key] = deliveries + 1

    def _consume_jetspec_candidate_receive(self, works, buffers) -> None:
        from jetspec.engines.jetspec_plus_plus.serving.settings import setting
        receive_key = getattr(buffers, 'key', None)
        # Transport completion orders every field before host reads.
        for work in works:
            work.wait()
        if hasattr(buffers, "cpu"):
            timing = getattr(self, '_anatomy_lite', None)
            if timing is not None:
                timing.mark(3)
            message, buffers = buffers, buffers.cpu()
            if timing is not None:
                timing.mark(4)
        delivery = (buffers if isinstance(buffers, CandidateDelivery) else
                    unpack_candidate_delivery(buffers, run_config=getattr(self, 'run_config', None)))
        if setting(self, 'engine.engine_settings.chain_tail_trace', '0') == '1':
            print(f"MAILBOX_ID side=verify event=candidate_receive key={receive_key} sets={[(c.seq_id, c.round_id) if c is not None else None for c in delivery.candidates]}", flush=True)
        self.metrics["jetspec_draft_round_ms"].extend(delivery.draft_ms)
        self.metrics["jetspec_coalesced_count"] += delivery.coalesced
        self.metrics["jetspec_gap_histogram"].extend(gap for gap in delivery.gaps if gap > 0)
        for row, candidate_set in enumerate(delivery.candidates):
            if candidate_set is not None:
                self.jetspec_mailbox.put(candidate_set)
                self._jetspec_candidate_gaps[
                    (candidate_set.seq_id, candidate_set.round_id)
                ] = delivery.gaps[row]
                if getattr(self, "_chain_miss_redraft", False):
                    self._chain_note_delivery(candidate_set)
        timing = getattr(self, '_anatomy_lite', None)
        if timing is not None:
            timing.stamp('candidate_decode_end')

    def _poll_jetspec_candidate_receives(self) -> None:
        pending = []
        completed = 0
        for receive_id, works, buffers in self._jetspec_candidate_receives:
            ready_key = f"candidate_{receive_id}"
            # A store-fetched packet (CandidateReceiver) never gets a ready key;
            # its own future marks completion (the chain miss-redraft valve polls it).
            fetched = getattr(buffers, "ready", None) if not works else None
            if not (fetched.done() if fetched is not None
                    else self.jetspec_mailbox_store.check([ready_key])):
                pending.append((receive_id, works, buffers))
                continue
            self._consume_jetspec_candidate_receive(works, buffers)
            completed += 1
        self._jetspec_candidate_receives = pending
        if not self.oracle_anchor:
            for _ in range(completed):
                self._post_jetspec_candidate_receive()

    def _post_jetspec_candidate_receive(self) -> None:
        from jetspec.engines.jetspec_plus_plus.serving.settings import setting
        from jetspec.engines.jetspec_plus_plus.runtime.kernel_profiles import kernel_enabled
        from jetspec.engines.jetspec_plus_plus.runtime.engine import current_engine
        if self.jetspec_mailbox_store is None:
            raise RuntimeError("JetSpec mailbox requires its readiness store")
        if self._jetspec_wire_capacity is None:
            raise RuntimeError("JetSpec candidate capacity is unknown before prefill")
        capacity = self._jetspec_wire_capacity
        receive_id = self._jetspec_next_receive_id
        if setting(self, 'engine.engine_settings.rows', '0')=='1' and setting(self, 'engine.engine_settings.rows_fast', '0')=='1':
            from jetspec.engines.jetspec_plus_plus.draft.live_rows import candidate_wire_views
            from jetspec.engines.jetspec_plus_plus.draft.live_rows import candidate_wire_numel
            if self.jetspec_wait_ms == float("inf"):
                if self._jetspec_candidate_receiver is None:
                    decode = partial(candidate_wire_views, capacity=capacity,
                        hypotheses=self.jetspec_hyp, branches=branch_count(run_config=getattr(self, 'run_config', None)),
                        run_config=getattr(self, 'run_config', None))
                    lean_decode = getattr(self, "_candidate_decode_lean_enabled", None)
                    if lean_decode is None:
                        lean_decode = kernel_enabled('candidate_decode_lean', '1')
                    host_select = getattr(self, "_verify_select_host", False)
                    unpack = (lambda wire: unpack_candidate_delivery(
                        decode(wire), private_wire=True, host_select=host_select, run_config=getattr(self, 'run_config', None))) if lean_decode else (
                        lambda wire: unpack_candidate_delivery(decode(wire), run_config=getattr(self, 'run_config', None)))
                    receiver = CandidateReceiver
                    if (current_engine().kind == 'reference'
                            and setting(self, 'engine.engine_settings.reference_candidate_socket', '0') == "1"
                            and anchor_mode(run_config=getattr(self, 'run_config', None)) == "decoder" and capacity == 1):
                        from jetspec.engines.jetspec_plus_plus.wire.candidate_transport import decode_reference_candidate
                        unpack = partial(decode_reference_candidate,
                            rows=int(setting(self, 'engine.engine_settings.rows_cap', '128')), lanes=self.K + 1, run_config=getattr(self, 'run_config', None))
                        if setting(self, 'engine.engine_settings.reference_candidate_main', '0') == "1":
                            from jetspec.engines.jetspec_plus_plus.wire.candidate_transport import MainThreadCandidateReceiver
                            receiver = MainThreadCandidateReceiver
                    self._jetspec_candidate_receiver = receiver(
                        self.jetspec_mailbox_store, unpack, run_config=self.run_config)
                buffers = self._jetspec_candidate_receiver.post(f"candidate_payload_{receive_id}")
                works = []
            else:
                wire=torch.empty(candidate_wire_numel(capacity,self.jetspec_hyp,branch_count(run_config=getattr(self, 'run_config', None)), run_config=getattr(self, 'run_config', None)),dtype=torch.int64,device=self.device)
                buffers=candidate_wire_views(wire,capacity,self.jetspec_hyp,branch_count(run_config=getattr(self, 'run_config', None)), run_config=getattr(self, 'run_config', None))
                buffers=PackedCandidateBuffers(wire, buffers)
                works=[dist.irecv(wire,src=self.draft_runner_rank,group=self.jetspec_candidate_pg)]
        else:
            available = torch.empty(capacity, dtype=torch.int64, device=self.device)
            keys = torch.empty(capacity, 2, dtype=torch.int64, device=self.device)
            blocks = torch.empty(
                *candidate_wire_shape(capacity,self.jetspec_hyp, run_config=getattr(self, 'run_config', None)),
                dtype=torch.int64,
                device=self.device,
            )
            draft_ms = torch.empty(capacity, dtype=torch.float32, device=self.device)
            telemetry = torch.empty(capacity, 2, dtype=torch.int64, device=self.device)
            buffers = (available, keys, blocks, draft_ms, telemetry)
            works = [
                dist.irecv(
                    buffer,
                    src=self.draft_runner_rank,
                    group=self.jetspec_candidate_pg,
                )
                for buffer in buffers
            ]
        self._jetspec_next_receive_id += 1
        self._jetspec_candidate_receives.append(
            (receive_id, works, buffers)
        )

    def _wait_for_chain_candidate_round(self, seq_id, round_id, *, deadline=None):
        """Receive the exact round; the redraft valve never uses the draft deadline."""
        while not self.jetspec_mailbox.has(seq_id=seq_id, round_id=round_id):
            if deadline is not None:
                self._poll_jetspec_candidate_receives()
                if self.jetspec_mailbox.has(seq_id=seq_id, round_id=round_id):
                    break
                if time.perf_counter() >= deadline:
                    return None
                time.sleep(0.00005)
                continue
            if not self._jetspec_candidate_receives:
                self._post_jetspec_candidate_receive()
            _, works, buffers = self._jetspec_candidate_receives.pop(0)
            self._consume_jetspec_candidate_receive(works, buffers)
            self._post_jetspec_candidate_receive()
        return self.jetspec_mailbox.take(seq_id=seq_id, round_id=round_id)

    def _wait_for_jetspec_candidates(self, seqs: list[Sequence]) -> None:
        """Join exact candidates, polling only when a finite budget requires it."""
        from jetspec.engines.jetspec_plus_plus.runtime.profile import enabled as profile_enabled
        if len(seqs) > 1 and getattr(self, "_verify_prestage_runner", None) is not None:
            self._verify_prestage_runner.stage_jetspec_verify(seqs)
        from jetspec.engines.jetspec_plus_plus.runtime.engine import current_engine
        wait_ms = current_engine().budget.wait_ms(self.jetspec_wait_ms)
        if wait_ms == float("inf"):
            while not all(self.jetspec_mailbox.has(
                    seq_id=seq.seq_id, round_id=seq.jetspec_round) for seq in seqs):
                _, works, buffers = self._jetspec_candidate_receives.pop(0)
                self._consume_jetspec_candidate_receive(works, buffers)
                if not self.oracle_anchor:
                    self._post_jetspec_candidate_receive()
            return
        started = time.perf_counter()
        deadline = started + wait_ms / 1000.0
        while True:
            self._poll_jetspec_candidate_receives()
            if all(
                self.jetspec_mailbox.has(
                    seq_id=seq.seq_id,
                    round_id=seq.jetspec_round,
                )
                for seq in seqs
            ):
                return
            if wait_ms == 0.0:
                return
            if wait_ms != float("inf") and time.perf_counter() >= deadline:
                return
            time.sleep(0.00005)

    def adopt_candidate_stream(self, previous):
        """Continue the draft's process-lifetime candidate ids across generate calls.

        The draft numbers candidate_payload keys once per process. A fresh receiver
        restarting at id 0 would re-read every earlier payload at the next request's
        first round and leave the old reader thread blocked on a store get.
        """
        self._jetspec_candidate_receives = previous._jetspec_candidate_receives
        self._jetspec_candidate_receiver = previous._jetspec_candidate_receiver
        self._jetspec_next_receive_id = previous._jetspec_next_receive_id
        previous._jetspec_candidate_receives = []
        # The legacy request wire has the same shape: the draft reads request_<id>
        # with a process-lifetime counter, so a restart at 0 would never be read.
        self._jetspec_request_sends = previous._jetspec_request_sends
        self._jetspec_request_publisher = previous._jetspec_request_publisher
        self._jetspec_next_request_send_id = previous._jetspec_next_request_send_id
        self._pair_transport = getattr(previous, "_pair_transport", None)
        if self._pair_transport is not None:
            previous._pair_transport = None
        previous._jetspec_request_sends = []

    def warm_jetspec_session(self):
        """Measurement-only barrier and real T-module warm-up before prefill."""
        from jetspec.engines.jetspec_plus_plus.serving.settings import setting
        from jetspec.engines.jetspec_plus_plus.runtime.engine import current_engine
        if self._jetspec_next_receive_id or setting(self, 'benchmark.benchmark_settings.w240_warmup', '0')!='1':
            raise RuntimeError('warm-up must precede all mailbox receives')
        verify_device = getattr(self, "_verify_device_enabled", None)
        if verify_device is None:
            verify_device = setting(self, 'engine.engine_settings.jetspec_verify_device', '0') == "1"
        if ((verify_device or current_engine().kind == 'reference')
                and not self._verify_device_legacy):
            warmup_id = announce_chain_command(self.jetspec_mailbox_store, "warmup", run_config=getattr(self, 'run_config', None))
            if getattr(self, "_pair_outcome_push", False):
                start_command_ids(self.jetspec_mailbox_store, warmup_id)
                from jetspec.engines.jetspec_plus_plus.wire.outcome_packet import outcome_layout
                layout = outcome_layout(self)
                self._pair_transport = PairOutcomeTransport(
                    store=self.jetspec_mailbox_store, group=self.jetspec_request_pg,
                    device=self.device, lanes=self.K + 1,
                    hidden=layout.hidden, tap_dtype=layout.tap_dtype,
                    envelope_bytes=layout.outcome_bytes)
        else:
            dist.send(torch.tensor([3],dtype=torch.int64,device=self.device),
                      dst=self.draft_runner_rank,group=self.jetspec_request_pg)
        done=torch.empty(1,dtype=torch.int64,device=self.device)
        dist.recv(done,src=self.draft_runner_rank,group=self.jetspec_request_pg)
        if int(done.item())!=1:
            raise RuntimeError('draft warm-up acknowledgement mismatch')
