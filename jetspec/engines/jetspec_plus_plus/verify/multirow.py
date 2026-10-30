from __future__ import annotations

from jetspec.engines.jetspec_plus_plus.profiles import kernel_enabled, kernel_value

import json
import os
import time
import warnings
from functools import partial

import torch
import torch.distributed as dist
from transformers import AutoTokenizer

from jetspec.engines.jetspec_plus_plus.serving.speculate_types import SpeculateResult
from jetspec.engines.jetspec_plus_plus.serving.speculate_types import VerifyResult
from jetspec.engines.jetspec_plus_plus.serving.speculate_types import SpeculatorBase
from jetspec.engines.jetspec_plus_plus.serving.runner_helpers import prepare_prefill_payload
from jetspec.engines.jetspec_plus_plus.wire.outcome_packet import admission_outcome
from jetspec.engines.jetspec_plus_plus.wire.outcome_packet import announce_chain_command
from jetspec.engines.jetspec_plus_plus.wire.outcome_packet import await_chain_receive
from jetspec.engines.jetspec_plus_plus.wire.outcome_packet import make_hash16_weights
from jetspec.engines.jetspec_plus_plus.serving.sequence import Sequence
from jetspec.engines.jetspec_plus_plus.serving.misc import decode_tokens
from jetspec.engines.jetspec_plus_plus.wire.nccl_pack import concat_int64
from jetspec.engines.jetspec_plus_plus.wire.nccl_pack import pack_jetspec_taps
from jetspec.engines.jetspec_plus_plus.wire.nccl_pack import send_int64
from jetspec.engines.jetspec_plus_plus.wire.protocol import candidate_row_coverage
from jetspec.engines.jetspec_plus_plus.wire.protocol import CandidateMailbox
from jetspec.engines.jetspec_plus_plus.wire.protocol import CandidateReceiver
from jetspec.engines.jetspec_plus_plus.wire.protocol import RequestPublisher
from jetspec.engines.jetspec_plus_plus.wire.protocol import CandidateDelivery
from jetspec.engines.jetspec_plus_plus.wire.protocol import unpack_candidate_delivery
from jetspec.engines.jetspec_plus_plus.wire.protocol import PackedCandidateBuffers
from jetspec.engines.jetspec_plus_plus.wire.protocol import pack_request_wire
from jetspec.engines.jetspec_plus_plus.wire.protocol import anchor_mode
from jetspec.engines.jetspec_plus_plus.wire.protocol import oracle_request_predictions
from jetspec.engines.jetspec_plus_plus.wire.protocol import candidate_wire_shape
from jetspec.engines.jetspec_plus_plus.wire.protocol import branch_count
from jetspec.engines.jetspec_plus_plus.wire.protocol import select_candidate_block
from jetspec.engines.jetspec_plus_plus.wire.protocol import select_candidate_block_indexed
from jetspec.engines.jetspec_plus_plus.wire.protocol import ProtocolCounters
from jetspec.engines.jetspec_plus_plus.wire.protocol import pack_prompt_indices
from jetspec.engines.jetspec_plus_plus.wire.protocol import select_candidate_drafts
from jetspec.engines.jetspec_plus_plus.wire.protocol import select_candidate_drafts_host
from jetspec.engines.jetspec_plus_plus.wire.protocol import select_candidate_drafts_list
from jetspec.engines.jetspec_plus_plus.wire.protocol import unpack_candidate_sets
from jetspec.engines.jetspec_plus_plus.wire.protocol import ChainKeySnapshot
from jetspec.engines.jetspec_plus_plus.wire.protocol import chain_exact_hit
from jetspec.engines.jetspec_plus_plus.wire.protocol import chain_after_miss
from jetspec.engines.jetspec_plus_plus.runtime.env import require_verify_device_batch
from jetspec.engines.jetspec_plus_plus.wire.pair_transport import PairOutcomeTransport
from jetspec.engines.jetspec_plus_plus.wire.pair_transport import announce_check_outcome
from jetspec.engines.jetspec_plus_plus.wire.pair_transport import announce_tcp
from jetspec.engines.jetspec_plus_plus.wire.pair_transport import claim_command_id
from jetspec.engines.jetspec_plus_plus.wire.pair_transport import start_command_ids


class SpeculatorAsync(SpeculatorBase):
    _round_env = {
        "_outcome_pack_presync": 'engine.engine_settings.outcome_pack_presync',
        "_outcome_pack_presync_check": 'engine.engine_settings.outcome_pack_presync_check',
        "_outcome_prepost_no_ack": 'engine.engine_settings.outcome_prepost_no_ack',
        "_pair_outcome_push": 'pair_outcome_push',
        "_pair_outcome_push_check": 'engine.engine_settings.pair_outcome_push_check',
        "_verify_select_host": 'engine.engine_settings.verify_select_host',
        "_verify_record_late": 'engine.engine_settings.verify_record_late',
        "_verify_post_lean": 'verify_post_lean',
        "_candidate_recv_ahead": 'engine.engine_settings.candidate_recv_ahead',
        "_verify_select_list": 'engine.engine_settings.verify_select_list',
        "_reference_host_select": 'reference_host_select',
        "_reference_tap_prepare": 'reference_tap_prepare',
        "_verify_publish_first": 'engine.engine_settings.verify_publish_first',
        "_verify_device_enabled": 'engine.engine_settings.jetspec_verify_device',
        "_verify_stage_host_enabled": 'engine.engine_settings.verify_stage_host',
        "_verify_stage_early_enabled": 'engine.engine_settings.verify_stage_early',
        "_verify_input_direct_enabled": 'engine.engine_settings.verify_input_direct',
        "_verify_input_view_enabled": 'engine.engine_settings.verify_input_view',
        "_verify_select_lean_enabled": 'verify_select_lean',
        "_candidate_decode_lean_enabled": 'candidate_decode_lean',
        "_outcome_redraft": 'engine.engine_settings.chain_miss_redraft',
        "_outcome_prepost": 'engine.engine_settings.chain_prepost',
        "_outcome_sync_audit": 'engine.engine_settings.jetspec_verify_sync_audit',
        "_chain_cmd_local_counter": 'engine.engine_settings.chain_cmd_local_counter',
        "_chain_graph_seg": 'engine.engine_settings.chain_graph',
        "_chain_handoff_timing": 'engine.engine_settings.chain_handoff_timing',
        "_chain_prepost_timing": 'engine.engine_settings.chain_prepost_timing',
        "_chain_redraft_max_ms": 'engine.engine_settings.chain_miss_redraft_max_ms',
        "_w258b_trace": 'engine.engine_settings.w258b_trace',
    }

    def __getattr__(self, name):
        # The first access can follow serve_read's environment setup. Later
        # rounds use the instance attribute, including __new__ test instances.
        from jetspec.engines.jetspec_plus_plus.profiles import transport_enabled, TRANSPORT_SWITCHES
        from jetspec.engines.jetspec_plus_plus.profiles import kernel_enabled
        env_name = type(self)._round_env.get(name)
        if env_name is None:
            raise AttributeError(name)
        if "." in env_name:
            from jetspec.engines.jetspec_plus_plus.serving.settings import setting
            raw = setting(self, env_name, "10.0" if name == "_chain_redraft_max_ms" else "0")
        elif env_name in TRANSPORT_SWITCHES:
            raw = "1" if transport_enabled(env_name, run_config=self) else "0"
        else:
            raw = "1" if kernel_enabled(env_name) else "0"
            if raw == "0":
                from jetspec.engines.jetspec_plus_plus.profiles import verify_cut_lean27b
                raw = "1" if verify_cut_lean27b(
                    env_name, self, getattr(self, '_verify_stage_runner', None)) else "0"
        if name == "_chain_redraft_max_ms":
            value = float(raw)
        else:
            value = raw == ("seg" if name == "_chain_graph_seg" else "1")
        setattr(self, name, value)
        return value

    def __init__(
        self,
        lookahead: int,
        device: torch.device,
        async_fan_out: int,
        max_blocks: int,
        vocab_size: int,
        draft_dtype: torch.dtype,
        kvcache_block_size: int,
        max_model_len: int,
        max_num_seqs: int,
        async_pg: dist.ProcessGroup,
        jetspec_request_pg,
        jetspec_candidate_pg,
        jetspec_mailbox_store,
        draft_runner_rank: int,
        tokenizer: AutoTokenizer,
        verbose: bool,
        jetspec_taps: bool,
        jetspec_tap_hidden_size: int | None,
        jetspec: bool,
        jetspec_sync: bool,
        jetspec_wait_ms: float,
        jetspec_hyp: int,
        jetspec_miss: str,
        metrics: dict,
        target_model_runner=None,
        run_config=None,
    ):
        from jetspec.engines.jetspec_plus_plus.serving.settings import setting
        from jetspec.engines.jetspec_plus_plus.profiles import kernel_enabled
        from jetspec.engines.jetspec_plus_plus.engine import current_engine
        super().__init__(lookahead, device)
        self.run_config = run_config if run_config is not None else getattr(target_model_runner, 'run_config', None)
        self.async_fan_out = async_fan_out
        self.max_blocks = max_blocks
        self.vocab_size = vocab_size
        self.draft_dtype = draft_dtype
        self.kvcache_block_size = kvcache_block_size
        self.max_model_len = max_model_len
        self.max_num_seqs = int(max_num_seqs)
        require_verify_device_batch(self.max_num_seqs, run_config=self.run_config)
        self.async_pg = async_pg
        self.jetspec_request_pg = jetspec_request_pg
        self.jetspec_candidate_pg = jetspec_candidate_pg
        self.jetspec_mailbox_store = jetspec_mailbox_store
        self.draft_runner_rank = draft_runner_rank
        self.tokenizer = tokenizer
        self.verbose = verbose
        self.K = lookahead
        self.jetspec_taps = jetspec_taps
        self.jetspec_tap_hidden_size = jetspec_tap_hidden_size
        self.jetspec = jetspec
        self._verify_device_legacy = (
            jetspec and setting(self, 'engine.engine_settings.jetspec_verify_device', '0') == "1"
            and setting(self, 'engine.engine_settings.verify_device_transport', 'chain') == "legacy")
        self.oracle_anchor = jetspec and anchor_mode(run_config=run_config) == 'oracle'
        if self.oracle_anchor and (jetspec_sync or float(jetspec_wait_ms) != float('inf')):
            raise ValueError('oracle anchor requires asynchronous rows with infinite wait')
        self.jetspec_sync = jetspec_sync
        self.jetspec_wait_ms = float(jetspec_wait_ms)
        if jetspec and (setting(self, 'engine.engine_settings.jetspec_verify_device', '0') == "1"
                        or current_engine().kind == 'reference'):
            if jetspec_sync or not jetspec_taps or jetspec_mailbox_store is None:
                raise ValueError("device verify requires async JetSpec taps and a host mailbox")
            if (((setting(self, 'engine.engine_settings.rows', '0') != "1") or (setting(self, 'engine.engine_settings.rows_fast', '0') != "1") or (self.jetspec_wait_ms != float("inf")))):
                raise ValueError(
                    "device verify requires CPU fast candidate delivery with infinite wait")
        from jetspec.engines.jetspec_plus_plus.draft.live_rows import resolve_admission_count
        from jetspec.engines.jetspec_plus_plus.draft.live_rows import miss_redraft
        from jetspec.engines.jetspec_plus_plus.draft.live_rows import miss_redraft_force
        # serving JETSPEC_MISS_REDRAFT=1: a missed round verifies the draft rank's fallback row.
        self.miss_redraft = bool(jetspec) and not jetspec_sync and miss_redraft(run_config=getattr(target_model_runner, "run_config", None))
        self.miss_redraft_force = self.miss_redraft and miss_redraft_force(run_config=getattr(target_model_runner, "run_config", None))
        if self.miss_redraft and (((float(jetspec_wait_ms) != float('inf')))):
            raise ValueError('JETSPEC_MISS_REDRAFT requires infinite wait and no tree budget')
        if jetspec and setting(self, 'engine.engine_settings.auto', '0') == '1':
            raise ValueError('retired variant fuse: the automatic route requests all fusion')
        self.jetspec_hyp = resolve_admission_count(jetspec_hyp, run_config=run_config) if jetspec else jetspec_hyp
        self.jetspec_miss = jetspec_miss
        if (setting(self, 'engine.engine_settings.chain_miss_redraft', '0') == "1"
                and jetspec_miss != "ar"):
            raise ValueError("chain miss redraft requires the lossless AR timeout path")
        self.jetspec_counters = ProtocolCounters()
        self.jetspec_mailbox = CandidateMailbox()
        self._jetspec_candidate_receives = []
        self._jetspec_candidate_receiver = None
        self._jetspec_next_receive_id = 0
        self._jetspec_request_sends = []
        self._jetspec_request_publisher = None
        self._jetspec_next_request_send_id = 0
        self._jetspec_candidate_gaps = {}
        self._jetspec_previous_round_ar = {}
        self._jetspec_wire_capacity = None
        self._jetspec_receive_window = 2 if self._candidate_recv_ahead else 1
        self._jetspec_outcome_wire = None
        self._jetspec_outcome_wire_ring = None
        self._presync_packed_outcome = None
        self._reference_prepared_outcome = None
        self._pair_transport = None
        self._pair_pending_id = None
        self._presync_pack_rounds = 0
        self._presync_pack_mismatches = 0
        self._outcome_wire_ring_enabled = (
            setting(self, 'engine.engine_settings.outcome_wire_ring', '0') == "1")
        if self._outcome_wire_ring_enabled:
            self._outcome_redraft = setting(self, 'engine.engine_settings.chain_miss_redraft', '0') == "1"
            self._outcome_prepost = setting(self, 'engine.engine_settings.chain_prepost', '0') == "1"
            self._outcome_sync_audit = setting(self, 'engine.engine_settings.jetspec_verify_sync_audit', '0') == "1"
        self.metrics = metrics
        self._verify_stage_runner = target_model_runner
        self._outcome_layout = None
        if jetspec_taps and target_model_runner is not None:
            from jetspec.engines.jetspec_plus_plus.wire.contract import OutcomeLayout
            from jetspec.engines.jetspec_plus_plus.wire.hybrid import target_wire_spec
            spec = target_wire_spec(target_model_runner)
            self._outcome_layout = OutcomeLayout.from_spec(spec, lanes=self.K + 1,
                tap_dtype=target_model_runner.hf_config.torch_dtype, sequence_count=1)
            self.jetspec_tap_hidden_size = spec.hidden_size
        from jetspec.engines.jetspec_plus_plus.models.state import has_state, pair_enabled
        self._pair_layout = None
        if pair_enabled(self):
            from jetspec.engines.jetspec_plus_plus.wire.hybrid import HybridPairLayout
            if (target_model_runner is None or not has_state(target_model_runner)
                    or self.K not in (15, 31)):
                raise ValueError("hybrid pair transfer requires a hybrid b16/b32 target")
            self._pair_layout = HybridPairLayout.from_config(
                target_model_runner.hf_config, lanes=self.K + 1)
        if self._outcome_pack_presync and not self._outcome_wire_ring_enabled:
            raise ValueError("JETSPEC_OUTCOME_PACK_PRESYNC requires JETSPEC_OUTCOME_WIRE_RING")
        if self._outcome_pack_presync and self._outcome_sync_audit:
            raise ValueError("JETSPEC_OUTCOME_PACK_PRESYNC refuses JETSPEC_VERIFY_SYNC_AUDIT")
        if self._pair_outcome_push and max_num_seqs != 1:
            raise ValueError("JETSPEC_PAIR_OUTCOME_PUSH requires N=1")
        if self._outcome_prepost_no_ack:
            from jetspec.engines.jetspec_plus_plus.engine import current_engine
            if not ((self._verify_device_enabled or current_engine().kind == 'reference')
                    and not self._verify_device_legacy
                    and self._outcome_prepost and self._chain_cmd_local_counter
                    and self._outcome_wire_ring_enabled and not self._pair_outcome_push):
                raise ValueError("JETSPEC_OUTCOME_PREPOST_NO_ACK requires preposted chain "
                                 "outcomes, a local command counter and wire ring")
        if self._verify_select_host and not (self._candidate_decode_lean_enabled
                                             and self._verify_select_lean_enabled):
            raise ValueError("JETSPEC_VERIFY_SELECT_HOST requires candidate decode and select lean")
        if self._candidate_recv_ahead and not (
                jetspec and not jetspec_sync and self.max_num_seqs == 1
                and self._verify_device_enabled and not self._verify_device_legacy
                and setting(self, 'engine.engine_settings.rows', '0') == "1"
                and setting(self, 'engine.engine_settings.rows_fast', '0') == "1"
                and self.jetspec_wait_ms == float("inf") and not self.oracle_anchor
                and True):
            raise ValueError("JETSPEC_CANDIDATE_RECV_AHEAD requires the N=1 infinite-wait store receiver")
        if self._verify_select_list and not (self.max_num_seqs == 1 and self._verify_select_host):
            raise ValueError("JETSPEC_VERIFY_SELECT_LIST requires N=1 host selection")
        if self._verify_select_list:
            self._verify_decision_metric_keys = {
                kind: f"jetspec_{kind}_count" for kind in ("hit", "miss", "late")}
        if self._verify_publish_first and not (
                self.max_num_seqs == 1 and self._verify_device_enabled
                and self._outcome_pack_presync and self._verify_post_lean
                and self._verify_record_late):
            raise ValueError("JETSPEC_VERIFY_PUBLISH_FIRST requires N=1 pre-sync pack, post lean and late record")
        if self._verify_publish_first:
            self._publish_command_builder = self._make_publish_command_builder()

        # Pre-allocate handshake send/recv buffers (reused every step)
        self._alloc_handshake_bufs(1)

        # Pre-allocate speculate() output buffers (avoid torch.tensor(device=cuda) sync)
        self._recovery_buf = torch.empty(1, dtype=torch.int64, device=device)
        self._speculations_buf = torch.empty(1, lookahead + 1, dtype=torch.int64, device=device)
        self._chain_redraft_info = None
        # 5f: the verifier decides a miss at speculate(r) and counts deliveries
        # per (seq, round) so a redraft it did not wait for is visible.
        self._chain_miss_redraft = setting(self, 'engine.engine_settings.chain_miss_redraft', '0') == "1"
        self._chain_deliveries = {}
        self._chain_taken_round = {}
        self._chain_expected = set()
        self._chain_late_ok = set()
        if self._chain_miss_redraft:
            self.metrics.setdefault("jetspec_chain_redraft_missing", 0)
            self.metrics.setdefault("jetspec_chain_redraft_unused", 0)

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
        from jetspec.engines.jetspec_plus_plus.engine import current_engine
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

    def _post_jetspec_candidate_receive(self) -> None:
        from jetspec.engines.jetspec_plus_plus.serving.settings import setting
        from jetspec.engines.jetspec_plus_plus.profiles import kernel_enabled
        from jetspec.engines.jetspec_plus_plus.engine import current_engine
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
            from jetspec.engines.jetspec_plus_plus.trace.host_census import note_candidate
            note_candidate(message, run_config=getattr(self, 'run_config', None))
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

    def _wait_for_jetspec_candidates(self, seqs: list[Sequence]) -> None:
        """Join exact candidates, polling only when a finite budget requires it."""
        from jetspec.engines.jetspec_plus_plus.runtime.profile import enabled as profile_enabled
        if len(seqs) > 1 and getattr(self, "_verify_prestage_runner", None) is not None:
            self._verify_prestage_runner.stage_jetspec_verify(seqs)
        from jetspec.engines.jetspec_plus_plus.engine import current_engine
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

    def _chain_speculate_redraft(self, seq, candidate_set, true_k):
        """5f: decide the miss at speculate(r), where knob-off already selects.

        The probe is the shared exact-key rule over the delivered set, the same
        function the draft host applied to its own snapshot of that set. Only a
        miss waits. Reference requires the full redraft; other modes wait up to
        the valve cap, then keep the plain miss on timeout.
        """
        from jetspec.engines.jetspec_plus_plus.serving.settings import setting
        key = (int(seq.seq_id), int(seq.jetspec_round))
        correction = int(seq.recovery_token_id)
        command = getattr(seq, "_jetspec_chain_command_key", None)
        if (key[1] > 0 and command is not None
                and command != (key[1], int(true_k), correction)):
            raise RuntimeError("chain command key differs from the verifier's speculate key")
        self._chain_taken_round[key[0]] = key[1]
        deliveries = self._chain_deliveries.pop(key, 1)
        from jetspec.engines.jetspec_plus_plus.runtime.profile import enabled as profile_enabled
        from jetspec.engines.jetspec_plus_plus.runtime.profile import critical_path
        if profile_enabled():
            first = getattr(self, "_chain_first_key_snapshots", {}).pop(key, None)
            if first is None:
                if deliveries > 1:
                    raise RuntimeError("eval redraft lacks its first-set key snapshot")
                first = ChainKeySnapshot.from_candidate(candidate_set, round_id=key[1])
            seq._jetspec_first_set_anchor_hit = chain_exact_hit(first, true_k, correction)
        if deliveries > 1:
            # Both deliveries landed before the take: the mailbox kept the redraft.
            chain_after_miss(ChainKeySnapshot.from_candidate(
                candidate_set, round_id=key[1], redraft=True),
                round_id=key[1], true_k=true_k, recovery=correction)
            self._chain_redraft_info = (*key, True, 0.0)
            self.metrics.setdefault("jetspec_chain_redraft_wait_ms", []).append(0.0)
            return candidate_set
        terminal = bool(getattr(seq, '_jetspec_last_packet_eos', 0))
        miss = chain_after_miss(ChainKeySnapshot.from_candidate(candidate_set, round_id=key[1]),
                                round_id=key[1], true_k=true_k, recovery=correction,
                                eos=terminal)
        if setting(self, 'engine.engine_settings.chain_tail_trace', '0') == '1':
            print(f"MAILBOX_ID side=verify event=branch seq={key[0]} round={key[1]} deliveries={deliveries} true_k={true_k} recovery={correction} branch={'eos' if terminal else 'redraft' if miss else 'hit'}", flush=True)
        if not miss and not (critical_path() and key[1] > 0 and not terminal):
            return candidate_set
        valve = getattr(self, "_chain_redraft_max_ms", None)
        if valve is None:
            valve = float(setting(self, 'engine.engine_settings.chain_miss_redraft_max_ms', '10.0'))
        if not valve > 0:
            raise ValueError("miss redraft valve must be positive (inf disables the cutoff)")
        from jetspec.engines.jetspec_plus_plus.engine import current_engine
        budget = current_engine().budget
        if not budget.uncut:
            budget.redraft_deadline(0, valve)
        self._chain_expected.add(key)
        waited = time.perf_counter()
        try:
            if current_engine().kind == 'reference':
                # Reference verification requires the full tap block and cannot
                # use the finite valve's AR fallback. Join the exact redraft.
                delivered = self._wait_for_chain_candidate_round(*key)
            else:
                delivered = self._wait_for_chain_candidate_round(
                    *key, deadline=budget.redraft_deadline(waited, valve))
        finally:
            self._chain_expected.discard(key)
        wait_ms = (time.perf_counter() - waited) * 1000
        self.metrics.setdefault("jetspec_chain_redraft_wait_ms", []).append(wait_ms)
        self._chain_redraft_info = (*key, delivered is not None, wait_ms)
        if delivered is None:
            self.metrics["jetspec_chain_redraft_missing"] = (
                self.metrics.get("jetspec_chain_redraft_missing", 0) + 1)
            self._chain_late_ok.add(key)
            return candidate_set
        chain_after_miss(ChainKeySnapshot.from_candidate(delivered, round_id=key[1],
                                                         redraft=True),
                         round_id=key[1], true_k=true_k, recovery=correction)
        return delivered

    def _queue_jetspec_request(self, tensors: list[torch.Tensor], *, control=None,
                               signal_store=True, chain_command=None,
                               chain_round=None, chain_length=None, chain_after_miss=None,
                               chain_eos=None, chain_sync_at=None, chain_true_k=None,
                               chain_recovery=None) -> None:
        from jetspec.engines.jetspec_plus_plus.profiles import kernel_enabled
        from jetspec.engines.jetspec_plus_plus.trace.host_census import mark as _census_mark
        pair_outcome = chain_command == "outcome" and getattr(self, "_pair_outcome_push", False)
        pair_command_id = None
        if pair_outcome:
            pair_command_id = self._pair_pending_id
            if pair_command_id is None:
                raise RuntimeError("pair outcome was not pushed before publication")
            self._pair_pending_id = None
            if not getattr(self, "_pair_outcome_push_check", False):
                # Keep the legacy request counter's lifetime without a host
                # clone, TCP announcement, ack, or NCCL send on this path.
                if self._jetspec_outcome_wire_ring is not None:
                    self._jetspec_outcome_wire_ring.sent([])
                self._jetspec_next_request_send_id += 1
                return
        # Compact packets retain immutable tap storage and an immutable command.
        # Retain them until completion; legacy packets include reused buffers.
        borrowed_wire = (chain_command == "outcome" and
                         getattr(self, "_outcome_wire_ring_enabled", False) and
                         self._jetspec_outcome_wire_ring is not None)
        from jetspec.engines.jetspec_plus_plus.wire.hybrid import eager_pair
        eager_outcome = (eager_pair(getattr(self, '_verify_stage_runner', None))
                         and len(tensors) == 2 and tensors[1].dtype == torch.uint8)
        snapshots = (tensors if control is not None or borrowed_wire else
                     ([tensors[0], tensors[1].detach().clone().contiguous()]
                      if eager_outcome else
                      [tensor.detach().clone().contiguous() for tensor in tensors]))
        _census_mark("clone", run_config=getattr(self, 'run_config', None))
        request_id = self._jetspec_next_request_send_id
        self._jetspec_next_request_send_id += 1
        if chain_command is not None:
            if pair_outcome:
                chain_command_id = pair_command_id
                announce_check_outcome(
                    self.jetspec_mailbox_store, chain_command_id,
                    round_id=chain_round, L=chain_length,
                    true_k=chain_true_k, recovery=chain_recovery, eos=chain_eos,
                    redraft=self._outcome_redraft, prepost=self._outcome_prepost,
                    attn3=False)
            else:
                chain_command_id = announce_chain_command(
                    self.jetspec_mailbox_store, chain_command,
                    round_id=chain_round, L=chain_length, after_miss=chain_after_miss,
                    eos=chain_eos, true_k=chain_true_k, recovery=chain_recovery,
                    redraft=getattr(self, "_outcome_redraft", None),
                    prepost=getattr(self, "_outcome_prepost", None),
                    attn3=False,
                    local_counter=getattr(self, "_chain_cmd_local_counter", None), run_config=getattr(self, 'run_config', None))
            _census_mark("announce", run_config=getattr(self, 'run_config', None))
            if (chain_command == "outcome"
                    and not getattr(self, "_outcome_prepost_no_ack", False)):
                timing = getattr(self, "_chain_prepost_timing", None)
                if timing is None:
                    await_chain_receive(self.jetspec_mailbox_store, chain_command_id, run_config=getattr(self, 'run_config', None))
                else:
                    await_chain_receive(self.jetspec_mailbox_store, chain_command_id,
                                        timing=timing, run_config=getattr(self, 'run_config', None))
            _census_mark("ack", run_config=getattr(self, 'run_config', None))
        publication = None
        if control is not None:
            if self._jetspec_request_publisher is None:
                self._jetspec_request_publisher = RequestPublisher(self.jetspec_mailbox_store)
            publication = self._jetspec_request_publisher.post(
                f"request_{request_id}", control.numpy().tobytes())
        kernel_profile = getattr(self, "_verify_kernel_profile", None)
        if kernel_profile is not None and kernel_profile.active is not None:
            works = []
            for index, tensor in enumerate(snapshots):
                kernel_profile.step(f"send_{index}")
                works.append(dist.isend(
                    tensor, dst=self.draft_runner_rank,
                    group=self.jetspec_request_pg))
        else:
            works = [
                dist.isend(
                    tensor,
                    dst=self.draft_runner_rank,
                    group=self.jetspec_request_pg,
                )
                for tensor in snapshots
            ]
        if borrowed_wire:
            self._jetspec_outcome_wire_ring.sent(works)
        # Gloo CPU staging (one-GPU placements) copies each payload to the host inside isend,
        # and a Gloo send never reports is_completed() before wait(): retaining the CUDA
        # snapshots here would pin every round's committed-tap storage for the whole read.
        if getattr(dist, "_ssd_cpu_staging_installed", False):
            snapshots = []
        self._jetspec_request_sends.append((works, snapshots))
        if publication is not None:
            works.append(publication)
        elif signal_store:
            self.jetspec_mailbox_store.set(f"request_{request_id}", "1")
        _census_mark("isend", run_config=getattr(self, 'run_config', None))

    def _prune_jetspec_request_sends(self) -> None:
        self._jetspec_request_sends = [
            item
            for item in self._jetspec_request_sends
            if not all(work.is_completed() for work in item[0])
        ]

    def _alloc_handshake_bufs(self, B):
        self._hs_B = B
        if self.jetspec:
            self._jetspec_proposal_host = torch.empty(
                B, self.K + 1, dtype=torch.int64, device="cpu",
                pin_memory=torch.device(self.device).type == "cuda")
        d = self.device
        self._cmd = torch.zeros(1, dtype=torch.int64, device=d)
        self._meta = torch.tensor([B, self.K, self.async_fan_out], dtype=torch.int64, device=d)
        self._cache_keys = torch.empty(B, 3, dtype=torch.int64, device=d)
        self._num_tokens_buf = torch.empty(B, dtype=torch.int64, device=d)
        self._temps_buf = torch.empty(B, dtype=torch.float32, device=d)
        self._block_tables_buf = torch.full((B, self.max_blocks), -1, dtype=torch.int32, device=d)
        self._fused_response = torch.empty(B + B * self.K, dtype=torch.int64, device=d)
        self._logits_q = torch.empty(B, self.K, self.vocab_size, dtype=self.draft_dtype, device=d)
        self._extend_counts = torch.zeros(B, dtype=torch.int64, device=d)
        self._jetspec_rounds = torch.empty(B, dtype=torch.int64, device=d)
        self._jetspec_available = torch.empty(B, dtype=torch.int64, device=d)
        self._jetspec_keys = torch.empty(B, 2, dtype=torch.int64, device=d)
        self._jetspec_blocks = torch.empty(
            *candidate_wire_shape(B, self.jetspec_hyp, run_config=getattr(self, 'run_config', None)), dtype=torch.int64, device=d
        )
        self._jetspec_draft_ms = torch.empty(B, dtype=torch.float32, device=d)

    def prefill(self, seqs: list[Sequence], verify_result: VerifyResult) -> SpeculateResult:
        from jetspec.engines.jetspec_plus_plus.serving.settings import setting
        from jetspec.engines.jetspec_plus_plus.engine import current_engine
        eagle_acts = verify_result.eagle_acts
        input_id_list = [seq.token_ids for seq in seqs]

        # EAGLE token-conditioning shift: token at position j gets conditioning
        # from target act at position j-1. Skip first token per seq and drop
        # last eagle_act per seq so they align correctly.
        if eagle_acts is not None:
            sliced = []
            offset = 0
            for ids in input_id_list:
                seq_len = len(ids)
                sliced.append(eagle_acts[offset:offset + seq_len - 1])
                offset += seq_len
            eagle_acts = torch.cat(sliced, dim=0)
            input_id_list = [ids[1:] for ids in input_id_list]

        max_blocks = (self.max_model_len + self.kvcache_block_size - 1) // self.kvcache_block_size
        cmd, metadata, input_ids, num_tokens, draft_block_table, eagle_acts = prepare_prefill_payload(
            input_id_list, eagle_acts, self.device, max_blocks,
            [seq.draft_block_table for seq in seqs],
        )
        if self.jetspec and not self.jetspec_sync:
            if self._jetspec_wire_capacity is None:
                self._jetspec_wire_capacity = max(self.max_num_seqs, len(seqs))
            if len(seqs) > self._jetspec_wire_capacity:
                raise RuntimeError("JetSpec prefill batch exceeds wire capacity")
        request_pg = self.jetspec_request_pg if self.jetspec else self.async_pg
        if self.jetspec:
            verify_device = getattr(self, "_verify_device_enabled", None)
            if verify_device is None:
                verify_device = setting(self, 'engine.engine_settings.jetspec_verify_device', '0') == "1"
        else:
            verify_device = False
        if ((verify_device or (self.jetspec and current_engine().kind == 'reference'))
                and not getattr(self, "_verify_device_legacy", False)):
            if getattr(self, "_pair_outcome_push", False):
                announce_tcp(self.jetspec_mailbox_store, "prefill")
            else:
                announce_chain_command(self.jetspec_mailbox_store, "prefill", run_config=getattr(self, 'run_config', None))
        else:
            dist.send(cmd, dst=self.draft_runner_rank, group=request_pg)
        dist.send(metadata, dst=self.draft_runner_rank, group=request_pg)
        send_int64(request_pg, self.draft_runner_rank,
                   input_ids, num_tokens, draft_block_table.to(torch.int64))
        if eagle_acts is not None:
            # The draft rank allocates its receive buffer in draft_hf_config.torch_dtype.
            # Match that dtype when an fp32 target supplies EAGLE activations.
            dist.send(eagle_acts.to(dtype=self.draft_dtype),
                      dst=self.draft_runner_rank, group=request_pg)
        if self.jetspec:
            seq_ids = torch.tensor(
                [seq.seq_id for seq in seqs], dtype=torch.int64, device=self.device
            )
            prompt_indices = pack_prompt_indices(seqs, device=self.device)
            recovery_tokens = torch.tensor(
                verify_result.recovery_tokens,
                dtype=torch.int64,
                device=self.device,
            )
            send_int64(
                request_pg,
                self.draft_runner_rank,
                seq_ids,
                prompt_indices,
                recovery_tokens,
            )
            from jetspec.engines.jetspec_plus_plus.wire.outcome_packet import outcome_layout
            layout = outcome_layout(self)
            tap_lengths, tap_payload = pack_jetspec_taps(
                seqs,
                device=self.device,
                tap_shape=(layout.tap_count, layout.hidden),
                dtype=layout.tap_dtype,
            )
            dist.send(tap_lengths, dst=self.draft_runner_rank, group=request_pg)
            if tap_payload.shape[0] > 0:
                # The draft rank receives taps in its own dtype, as for EAGLE activations above.
                dist.send(tap_payload.to(dtype=self.draft_dtype), dst=self.draft_runner_rank,
                          group=request_pg)
            if layout.sections:
                from jetspec.engines.jetspec_plus_plus.wire.hybrid import target_prefill_tensors
                for seq, length in zip(seqs, tap_lengths.tolist()):
                    positions = torch.arange(int(length), device=self.device)
                    for payload in target_prefill_tensors(
                            self._verify_stage_runner, seq, positions, layout):
                        dist.send(payload.contiguous(), dst=self.draft_runner_rank,
                                  group=request_pg)
            for seq, length in zip(seqs, tap_lengths.tolist()):
                seq.jetspec_taps_sent += int(length)
            if not self.jetspec_sync and self._jetspec_next_receive_id == 0:
                for _ in range(self._jetspec_receive_window):
                    self._post_jetspec_candidate_receive()
        return SpeculateResult([], [])

    def speculate(self, seqs: list[Sequence], verify_result: VerifyResult) -> SpeculateResult:
        from jetspec.engines.jetspec_plus_plus.serving.settings import setting
        from jetspec.engines.jetspec_plus_plus.trace.host_census import mark as _census_mark
        if self.jetspec and any(
            getattr(seq, "draft_temperature", None) not in (None, 0) for seq in seqs
        ):
            raise ValueError("JetSpec requires a greedy draft")
        # Handshake tensors have explicit devices; avoid the process-wide CPU
        # default-device Python hook on every small operation. Restore it before
        # the verifier runs so forward execution keeps its original context.
        with torch._C.DisableTorchFunction():
            for seq in seqs:
                assert seq.recovery_token_id is not None
                seq.append_token(seq.recovery_token_id)
            _census_mark("append", run_config=getattr(self, 'run_config', None))

            if self.verbose:
                sep = '=' * 80
                print(f"\n{sep}", flush=True)
                print(f"[TARGET SEQUENCE TRUNK] Batch size: {len(seqs)}", flush=True)
                for i, seq in enumerate(seqs):
                    trunk = seq.token_ids[-20:] if len(seq.token_ids) > 20 else seq.token_ids
                    print(f"  Seq {seq.seq_id} (len={len(seq.token_ids)}):", flush=True)
                    print(f"    Trunk: ...{decode_tokens(trunk, self.tokenizer)}", flush=True)
                    print(f"    Recovery: {seq.recovery_token_id} ({decode_tokens([seq.recovery_token_id], self.tokenizer)})", flush=True)
                print(f"{sep}\n", flush=True)

            eagle = verify_result.eagle_acts is not None
            (
                speculations_tokens,
                logits_q,
                cache_hits,
                jetspec_verify_mask,
                jetspec_decisions,
            ) = self._speculation_request(seqs, eagle)
            _census_mark("request_tail", run_config=getattr(self, 'run_config', None))

            # Build speculations using pre-allocated buffers (avoids torch.tensor(device=cuda) sync)
            B = len(seqs)
            if B != self._recovery_buf.shape[0]:
                self._recovery_buf = torch.empty(B, dtype=torch.int64, device=self.device)
                self._speculations_buf = torch.empty(B, self.K + 1, dtype=torch.int64, device=self.device)
            lean_select = self.jetspec and isinstance(speculations_tokens, list)
            if not lean_select:
                _rec_cpu = torch.tensor([seq.recovery_token_id for seq in seqs], dtype=torch.int64)
            if self.jetspec:
                if lean_select:
                    row = self._jetspec_proposal_host[0].numpy()
                    row[0] = seqs[0].recovery_token_id
                    row[1:] = speculations_tokens[0]
                else:
                    self._jetspec_proposal_host[:, 0].copy_(_rec_cpu)
                    self._jetspec_proposal_host[:, 1:].copy_(speculations_tokens)
                direct = (B == 1 and self._verify_input_direct_enabled and
                          self._verify_device_enabled)
                if direct:
                    if self._verify_stage_runner is None:
                        raise RuntimeError("direct verify input lacks target runner")
                    if self._verify_input_view_enabled:
                        speculations = getattr(self, "_jetspec_n1_cached_input_view", None)
                        if speculations is None:
                            speculations = self._verify_stage_runner.call("jetspec_n1_input_view")
                            self._jetspec_n1_cached_input_view = speculations
                    else:
                        speculations = self._verify_stage_runner.call("jetspec_n1_input_view")
                    speculations.copy_(self._jetspec_proposal_host, non_blocking=True)
                else:
                    self._speculations_buf.copy_(self._jetspec_proposal_host, non_blocking=True)
                if B == 1:
                    stage_host = getattr(self, "_verify_stage_host_enabled", None)
                    if stage_host is None:
                        stage_host = setting(self, 'engine.engine_settings.verify_stage_host', '0') == "1"
                    if stage_host:
                        verify_device = getattr(self, "_verify_device_enabled", None)
                        if verify_device is None:
                            verify_device = setting(self, 'engine.engine_settings.jetspec_verify_device', '0') == "1"
                        if verify_device:
                            seqs[0]._jetspec_verify_proposal_host = self._jetspec_proposal_host[0]
            else:
                self._recovery_buf.copy_(_rec_cpu, non_blocking=True)
                self._speculations_buf[:, 0] = self._recovery_buf
                self._speculations_buf[:, 1:] = speculations_tokens
            if not (self.jetspec and B == 1 and self._verify_input_direct_enabled and
                    self._verify_device_enabled):
                speculations = self._speculations_buf
            _census_mark("proposal", run_config=getattr(self, 'run_config', None))

            for i, seq in enumerate(seqs):
                verified = (jetspec_verify_mask is None or
                            bool(jetspec_verify_mask[i] if lean_select else
                                 jetspec_verify_mask[i].item())) if self.jetspec else True
                if verified:
                    seq.token_ids.extend(speculations_tokens[i] if lean_select else
                                         speculations_tokens[i].tolist())
                seq.num_tokens = len(seq.token_ids)
                seq.last_token = seq.token_ids[-1]
                seq.num_draft_cached_tokens += (
                    len(speculations_tokens[i]) + 1
                    if verified
                    else 1
                )
            _census_mark("spec_tail", run_config=getattr(self, 'run_config', None))

            return SpeculateResult(
                speculations,
                logits_q,
                cache_hits,
                jetspec_verify_mask,
                jetspec_decisions,
            )

    def _new_outcome_wire(self):
        from jetspec.engines.jetspec_plus_plus.wire.outcome_packet import make_outcome_wire
        return make_outcome_wire(self)

    def _new_outcome_wire_ring(self):
        from jetspec.engines.jetspec_plus_plus.wire.outcome_packet import make_outcome_wire
        return make_outcome_wire(self, ring=True)

    def _pack_outcome_wire(self, wire, seq, *, skip_full_zero=False, prepared_taps=False):
        from jetspec.engines.jetspec_plus_plus.wire.hybrid import pack_target_outcome
        return pack_target_outcome(getattr(self, '_verify_stage_runner', None), seq, wire,
            seq._jetspec_device_outcome, seq._jetspec_tap_candidate_block,
            skip_full_zero=skip_full_zero, prepared_taps=prepared_taps)

    def _pack_jetspec_request(self, seqs):
        from jetspec.engines.jetspec_plus_plus.serving.settings import setting
        from jetspec.engines.jetspec_plus_plus.engine import current_engine
        ring_enabled = getattr(self, "_outcome_wire_ring_enabled", False)
        verify_device = getattr(self, "_verify_device_enabled", None)
        if verify_device is None:
            verify_device = setting(self, 'engine.engine_settings.jetspec_verify_device', '0') == "1"
        from jetspec.engines.jetspec_plus_plus.wire.hybrid import eager_pair
        if ((verify_device or current_engine().kind == 'reference'
             or (getattr(self, 'max_num_seqs', 1) == 1
                 and eager_pair(getattr(self, '_verify_stage_runner', None))))
                and not getattr(self, "_verify_device_legacy", False) and len(seqs) == 1
                and hasattr(seqs[0], "_jetspec_device_outcome")):
            seq = seqs[0]
            if self.jetspec_sync or not self.jetspec_taps:
                raise RuntimeError("device outcome requires async JetSpec taps")
            if self._jetspec_outcome_wire is None:
                if not ring_enabled:
                    self._jetspec_outcome_wire = self._new_outcome_wire()
            if getattr(self, "_reference_tap_prepare", False):
                prepared = self._reference_prepared_outcome
                if (prepared is None or prepared[0] is not seq or prepared[1] != seq.jetspec_round
                        or prepared[3] != seq._jetspec_tap_candidate_block.data_ptr()):
                    raise RuntimeError("reference outcome taps were not prepared for this round")
                message = self._pack_outcome_wire(prepared[2], seq, prepared_taps=True)
                self._reference_prepared_outcome = None
            elif ring_enabled and getattr(self, "_outcome_pack_presync", False):
                packed = self._presync_packed_outcome
                if packed is None or packed[0] is not seq or packed[1] is not seq._jetspec_device_outcome:
                    raise RuntimeError("pre-sync outcome pack was not staged for this round")
                message = packed[2]
                self._presync_packed_outcome = None
            else:
                message = None
            if ring_enabled and message is None:
                if self._jetspec_outcome_wire_ring is None:
                    self._jetspec_outcome_wire_ring = self._new_outcome_wire_ring()
                wire = self._jetspec_outcome_wire_ring.acquire()
            elif not ring_enabled:
                wire = self._jetspec_outcome_wire
            if message is None:
                message = self._pack_outcome_wire(wire, seq, skip_full_zero=ring_enabled)
            if (getattr(self, "_pair_outcome_push", False)
                    and not getattr(self, "_outcome_pack_presync", False)
                    and current_engine().kind == 'reference'):
                if self._pair_transport is None or self._pair_pending_id is not None:
                    raise RuntimeError("pair outcome transport is not ready for reference push")
                self._pair_pending_id = claim_command_id(self.jetspec_mailbox_store)
                self._pair_transport.push(self._pair_pending_id, message)
            seq.jetspec_taps_sent += seq._jetspec_device_tap_count
            self.metrics["jetspec_tap_payload_bytes"].append(message.numel())
            # The eager command loop still consumes its existing int64 command
            # before receiving the fixed outcome envelope. Chain intake does not.
            return None, ([self._cmd, message] if eager_pair(
                getattr(self, '_verify_stage_runner', None)) else [message])
        if current_engine().kind == 'reference':
            raise RuntimeError("reference chain request requires a published outcome envelope")
        tap_started = time.perf_counter()
        lengths, pending = [], []
        from jetspec.engines.jetspec_plus_plus.wire.outcome_packet import outcome_layout
        layout = outcome_layout(self)
        shape = (layout.tap_count, layout.hidden)
        for seq in seqs:
            suffix = None if seq.jetspec_taps is None else seq.jetspec_taps[seq.jetspec_taps_sent:]
            length = 0 if suffix is None else len(suffix)
            lengths.append(length)
            if length:
                if tuple(suffix.shape[1:]) != shape:
                    raise ValueError("tap suffix does not match the configured tap shape")
                pending.append(suffix.to(device=self.device, dtype=layout.tap_dtype))
        # Committed taps are append-only via fresh storage. Keep a sole suffix
        # alive through the request queue instead of copying it into a new cat.
        if len(pending) == 1:
            taps = pending[0].contiguous()
        elif pending:
            taps = torch.cat(pending, dim=0).contiguous()
        else:
            taps = torch.empty((0, *shape), dtype=layout.tap_dtype, device=self.device)
        self.metrics["jetspec_tap_send_ms"].append((time.perf_counter() - tap_started) * 1000)
        self.metrics["jetspec_tap_payload_bytes"].append(
            len(seqs) * 8 + taps.numel() * taps.element_size())
        wire = pack_request_wire(seqs, capacity=self._jetspec_wire_capacity,
            tap_lengths=lengths, tree=bool(False),
            oracle_records=self.metrics["jetspec_round_records"] if self.oracle_anchor else None)
        tensors = [self._cmd]
        if taps.shape[0]:
            # The draft rank receives taps in its own dtype (an fp32 target packs fp32 taps).
            tensors.append(taps.to(dtype=self.draft_dtype))
        if layout.sections and getattr(self, 'max_num_seqs', 1) > 1:
            from jetspec.engines.jetspec_plus_plus.wire.hybrid import pack_batch_hybrid_outcomes
            tensors.append(pack_batch_hybrid_outcomes(
                self._verify_stage_runner, seqs, lengths, layout))
        for seq, length in zip(seqs, lengths):
            seq.jetspec_taps_sent += length
        return wire, tensors

    def prepare_reference_taps(self, seq, taps):
        """Prepare immutable taps while the target forward is in flight."""
        if self._reference_prepared_outcome is not None:
            raise RuntimeError("previous reference tap payload was not published")
        if self._jetspec_outcome_wire_ring is None:
            self._jetspec_outcome_wire_ring = self._new_outcome_wire_ring()
        wire = self._jetspec_outcome_wire_ring.acquire()
        wire.prepare_taps(taps)
        self._reference_prepared_outcome = (seq, seq.jetspec_round, wire, taps.data_ptr())
        self._prune_jetspec_request_sends()

    def pack_device_outcome_presync(self, seq):
        """Enqueue the final packet and tap copies after the mirror event."""
        from jetspec.engines.jetspec_plus_plus.trace.host_census import mark as _census_mark
        if not self._outcome_pack_presync:
            return
        if self._presync_packed_outcome is not None:
            raise RuntimeError("previous pre-sync outcome was not published")
        if self._jetspec_outcome_wire_ring is None:
            self._jetspec_outcome_wire_ring = self._new_outcome_wire_ring()
        # A future local command counter belongs to announce, not this early pack:
        # read it without consuming it if the 6c command-id condition lands.
        wire = self._jetspec_outcome_wire_ring.acquire()
        message = self._pack_outcome_wire(wire, seq, skip_full_zero=True)
        self._presync_packed_outcome = (seq, seq._jetspec_device_outcome, message)
        if getattr(self, "_pair_outcome_push", False):
            if self._pair_transport is None or self._pair_pending_id is not None:
                raise RuntimeError("pair outcome transport is not ready for pre-sync push")
            self._pair_pending_id = claim_command_id(self.jetspec_mailbox_store)
            self._pair_transport.push(self._pair_pending_id, message)
        _census_mark("pack", run_config=getattr(self, 'run_config', None))
        self._prune_jetspec_request_sends()
        _census_mark("prune", run_config=getattr(self, 'run_config', None))

    def check_device_outcome_presync(self, seq):
        if not (self._outcome_pack_presync and self._outcome_pack_presync_check):
            return
        packed = self._presync_packed_outcome
        if packed is None or packed[0] is not seq:
            raise RuntimeError("pre-sync outcome check has no staged packet")
        scratch = self._new_outcome_wire()
        expected = self._pack_outcome_wire(scratch, seq)
        self._presync_pack_rounds += 1
        self._presync_pack_mismatches += int(not torch.equal(packed[2], expected))
        print("OUTCOME_PACK_PRESYNC_CHECK "
              f"rounds={self._presync_pack_rounds} mismatches={self._presync_pack_mismatches}",
              flush=True)

    def _append_jetspec_round_record(self, seq, candidate_set, true_k, decision_kind,
                                     selected_guess, selected_candidate_ids, candidate_kind,
                                     previous_round_ar, candidate_gap, fallback, trace):
        from jetspec.engines.jetspec_plus_plus.trace.host_census import mark as _census_mark
        coverage = {}
        if trace:
            served_ks = []
            rows_published = None
            anchor_hit = decision_kind == "hit"
            if candidate_set is not None:
                if candidate_set.row_keys is not None:
                    served_ks, rows_published, anchor_hit = candidate_row_coverage(
                        (getattr(candidate_set, "_row_keys_host", candidate_set.row_keys)
                         if self._verify_select_host else candidate_set.row_keys), true_k=true_k,
                        correction=seq.recovery_token_id)
                else:
                    pins = candidate_set.blocks[..., 0]
                    valid = (pins >= 0) if pins.ndim == 1 else (pins >= 0).any(-1)
                    served_ks = valid.nonzero().reshape(-1).tolist()
            coverage = {
                "k_true": true_k,
                "k_set": served_ks,
                "kset_served_size": len(served_ks),
                "k_hit": true_k in served_ks,
                "hit_at_f": decision_kind == "hit",
                "rows_published": len(served_ks) if rows_published is None else rows_published,
                "anchor_hit": anchor_hit,
                "oracle_masked_k15": self.oracle_anchor and true_k == 15,
            }
        self.metrics["jetspec_round_records"].append({
            "seq_id": seq.seq_id,
            "prompt_index": int(seq.seq_id if seq.jetspec_prompt_index is None
                                else seq.jetspec_prompt_index),
            "round": seq.jetspec_round,
            "candidate_round": candidate_set.round_id if candidate_set is not None else None,
            "k": true_k,
            **coverage,
            "correction": int(seq.recovery_token_id),
            "guess": selected_guess,
            "candidate_draft_ids": (selected_candidate_ids[1:]
                                    if selected_candidate_ids is not None else None),
            "decision": decision_kind,
            "candidate_kind": candidate_kind,
            "previous_round_ar": previous_round_ar,
            "gap": candidate_gap,
            "wait_ms": self.jetspec_wait_ms,
            **({} if fallback is None else fallback),
        })
        _census_mark("record", run_config=getattr(self, 'run_config', None))

    def finish_jetspec_late_record(self, seq):
        pending = getattr(self, "_pending_jetspec_record", None)
        if pending is None or pending[0] is not seq:
            raise RuntimeError("late JetSpec record does not match the replay row")
        self._pending_jetspec_record = None
        self._append_jetspec_round_record(*pending)

    def _make_publish_command_builder(self):
        """Bind the recipe's command shape once, before timed rounds."""
        def base(seq):
            return dict(chain_round=seq.jetspec_round + 1,
                        chain_length=seq._jetspec_last_packet_l,
                        chain_sync_at=getattr(seq, "_jetspec_packet_sync_at", None))
        if self._outcome_redraft:
            def build(seq, terminal, true_k, recovery):
                return dict(base(seq), chain_true_k=true_k,
                            chain_recovery=recovery, chain_eos=int(terminal))
        elif self._outcome_prepost:
            def build(seq, terminal, _true_k, _recovery):
                return dict(base(seq), chain_after_miss=0, chain_eos=int(terminal))
        else:
            def build(_seq, _terminal, _true_k, _recovery):
                return {}
        return build

    def publish_device_outcome(self, seqs, outcome):
        """Send the next-round correction and fixed tap block before host postprocess."""
        from jetspec.engines.jetspec_plus_plus.serving.settings import setting
        from jetspec.engines.jetspec_plus_plus.profiles import kernel_enabled
        from jetspec.engines.jetspec_plus_plus.trace.host_census import mark as _census_mark
        ring_enabled = getattr(self, "_outcome_wire_ring_enabled", False)
        from jetspec.engines.jetspec_plus_plus.wire.hybrid import eager_pair
        if (self._verify_device_legacy or
                eager_pair(getattr(self, '_verify_stage_runner', None))):
            # The legacy request wire is packed after postprocess updates the
            # sequence metadata, just as it is with verify-device disabled.
            return
        started = time.perf_counter()
        if len(seqs) != 1 or outcome is not seqs[0]._jetspec_device_outcome:
            raise ValueError("device outcome must belong to the N=1 verify row")
        if getattr(self, "_verify_post_lean", False):
            redraft, prepost = self._outcome_redraft, self._outcome_prepost
        else:
            redraft = getattr(self, "_outcome_redraft", None)
            if redraft is None:
                redraft = setting(self, 'engine.engine_settings.chain_miss_redraft', '0') == "1"
            prepost = getattr(self, "_outcome_prepost", None)
            if prepost is None:
                prepost = setting(self, 'engine.engine_settings.chain_prepost', '0') == "1"
        seq = seqs[0]
        terminal = bool(getattr(seq, "_jetspec_last_packet_eos", 0))
        if redraft:
            # 5f: send the synced TRUE_K (= TAP_COUNT - 1) and RECOVERY host words;
            # no probe and no wait for set r+1 here. The draft host derives AFTER_MISS.
            true_k = int(seq._jetspec_device_tap_count) - 1
            recovery = (int(seq._jetspec_last_recovery_id)
                        if getattr(self, "_verify_publish_first", False) else
                        int(self.metrics["jetspec_round_records"][-1]["next_correction"]))
            if true_k < 0:
                raise RuntimeError("chain outcome has no committed tap")
            seq._jetspec_chain_command_key = (seq.jetspec_round + 1, true_k, recovery)
        command_fields = (self._publish_command_builder(
            seq, terminal, true_k if redraft else None, recovery if redraft else None)
            if getattr(self, "_verify_publish_first", False) else
            (dict(chain_round=seq.jetspec_round + 1,
                               chain_length=seq._jetspec_last_packet_l,
                               chain_sync_at=getattr(seq, "_jetspec_packet_sync_at", None),
                               **({"chain_true_k": true_k, "chain_recovery": recovery}
                                  if redraft else {"chain_after_miss": 0} if prepost else {}),
                               **({"chain_eos": int(terminal)} if prepost or redraft else {}))
                          if ((redraft) or (prepost))
                          else {}))
        sync_audit = getattr(self, "_outcome_sync_audit", None)
        if sync_audit is None:
            sync_audit = setting(self, 'engine.engine_settings.jetspec_verify_sync_audit', '0') == "1"
        if sync_audit:
            with warnings.catch_warnings(record=True) as seen:
                warnings.simplefilter("always")
                _, tensors = self._pack_jetspec_request(seqs)
                self._prune_jetspec_request_sends()
                self._queue_jetspec_request(tensors, signal_store=False,
                                            chain_command="outcome", **command_fields)
            self.metrics["jetspec_device_sync_warnings"][-1] += sum(
                "synchroniz" in str(warning.message).lower() for warning in seen)
        else:
            _census_mark("publish_head", run_config=getattr(self, 'run_config', None))
            if getattr(self, "_outcome_pack_presync", False):
                self.check_device_outcome_presync(seq)
            _, tensors = self._pack_jetspec_request(seqs)
            if not getattr(self, "_outcome_pack_presync", False):
                _census_mark("pack", run_config=getattr(self, 'run_config', None))
                if not getattr(self, "_reference_tap_prepare", False):
                    self._prune_jetspec_request_sends()
                _census_mark("prune", run_config=getattr(self, 'run_config', None))
            self._queue_jetspec_request(tensors, signal_store=False,
                                        chain_command="outcome", **command_fields)
        seqs[0]._jetspec_device_outcome_published = True
        if getattr(self, "_pair_outcome_push", False):
            self.metrics.setdefault("jetspec_pair_outcome_rounds", []).append(dict(
                seq_id=seq.seq_id, round=seq.jetspec_round + 1,
                release_waits=self._pair_transport.release_waits,
                ahead_max=self._pair_transport.ahead_max))
        self.metrics.setdefault("jetspec_device_publish_host_ms", []).append(
            (time.perf_counter() - started) * 1000)
        _census_mark("publish_tail", run_config=getattr(self, 'run_config', None))

    def _publish_chain_admission(self, seq):
        """Replace the suppressed legacy first request with a device packet."""
        from jetspec.engines.jetspec_plus_plus.serving.settings import setting
        from jetspec.engines.jetspec_plus_plus.profiles import kernel_enabled
        ring_enabled = getattr(self, "_outcome_wire_ring_enabled", False)
        if ring_enabled:
            if self._jetspec_outcome_wire_ring is None:
                self._jetspec_outcome_wire_ring = self._new_outcome_wire_ring()
            wire = self._jetspec_outcome_wire_ring.acquire()
        else:
            if self._jetspec_outcome_wire is None:
                self._jetspec_outcome_wire = self._new_outcome_wire()
            wire = self._jetspec_outcome_wire
        weights = getattr(self, "_jetspec_hash16_weights", None)
        if weights is None:
            weights = make_hash16_weights(self.device)
            self._jetspec_hash16_weights = weights
        packet = admission_outcome(
            seq, capacity=self.max_model_len + self.K + 2,
            device=self.device, weights=weights)
        empty_taps = wire.taps[:0]
        from jetspec.engines.jetspec_plus_plus.wire.hybrid import pack_target_outcome
        message = pack_target_outcome(getattr(self, '_verify_stage_runner', None), seq,
                                      wire, packet, empty_taps, admission=True)
        if getattr(self, "_pair_outcome_push", False):
            if self._pair_transport is None or self._pair_pending_id is not None:
                raise RuntimeError("pair outcome transport is not ready for admission")
            self._pair_pending_id = claim_command_id(self.jetspec_mailbox_store)
            self._pair_transport.push(self._pair_pending_id, message)
        self._prune_jetspec_request_sends()
        prepost = getattr(self, "_outcome_prepost", None)
        if prepost is None:
            prepost = setting(self, 'engine.engine_settings.chain_prepost', '0') == "1"
        redraft = getattr(self, "_outcome_redraft", None)
        if redraft is None:
            redraft = setting(self, 'engine.engine_settings.chain_miss_redraft', '0') == "1"
        if redraft:
            # 5c-0: TRUE_K = 0 and RECOVERY = c_0, the untapped token after the prompt.
            recovery = int(seq.token_ids[int(seq.num_cached_tokens)])
            seq._jetspec_chain_command_key = (0, 0, recovery)
        command_fields = (dict(chain_round=0, chain_length=int(seq.num_cached_tokens),
                               **({"chain_true_k": 0, "chain_recovery": recovery}
                                  if redraft else {"chain_after_miss": 0} if prepost else {}),
                               **({"chain_eos": 0} if (prepost or redraft) else {}))
                          if (((redraft) or (prepost)))
                          else {})
        self._queue_jetspec_request([message], signal_store=False,
                                    chain_command="outcome", **command_fields)
        if getattr(self, "_pair_outcome_push", False):
            self.metrics.setdefault("jetspec_pair_outcome_rounds", []).append(dict(
                seq_id=seq.seq_id, round=0,
                release_waits=self._pair_transport.release_waits,
                ahead_max=self._pair_transport.ahead_max))

    def _speculation_request(self, seqs: list[Sequence], eagle: bool):
        from jetspec.engines.jetspec_plus_plus.serving.settings import setting
        from jetspec.engines.jetspec_plus_plus.profiles import kernel_enabled
        from jetspec.engines.jetspec_plus_plus.engine import current_engine
        from jetspec.engines.jetspec_plus_plus.trace.host_census import mark as _census_mark
        B = len(seqs)
        chain_graph_seg = getattr(self, "_chain_graph_seg", None)
        if chain_graph_seg is None:
            chain_graph_seg = setting(self, 'engine.engine_settings.chain_graph', '0') == "seg"
        verify_device = getattr(self, "_verify_device_enabled", None)
        if verify_device is None:
            verify_device = setting(self, 'engine.engine_settings.jetspec_verify_device', '0') == "1"
        redraft = getattr(self, "_outcome_redraft", None)
        if redraft is None:
            redraft = setting(self, 'engine.engine_settings.chain_miss_redraft', '0') == "1"
        handoff_timing = getattr(self, "_chain_handoff_timing", None)
        if handoff_timing is None:
            handoff_timing = setting(self, 'engine.engine_settings.chain_handoff_timing', '0') == "1"
        select_lean = getattr(self, "_verify_select_lean_enabled", None)
        if select_lean is None:
            select_lean = kernel_enabled('verify_select_lean', '1')
        sync_audit = getattr(self, "_outcome_sync_audit", None)
        if sync_audit is None:
            sync_audit = setting(self, 'engine.engine_settings.jetspec_verify_sync_audit', '0') == "1"
        trace = getattr(self, "_w258b_trace", None)
        if trace is None:
            trace = setting(self, 'engine.engine_settings.w258b_trace') == "1"
        if B != self._hs_B:
            self._alloc_handshake_bufs(B)

        request_control = None
        outcome_already_sent = (self.jetspec and not self.jetspec_sync and not eagle
                                and len(seqs) == 1
                                and getattr(seqs[0], "_jetspec_device_outcome_published", False))
        # Prefill already published round zero. The first chain request has no
        # verifier outcome, so it must not leave a legacy NCCL message in the group.
        chain_first_round = (self.jetspec and not self.jetspec_sync and not eagle
                             and len(seqs) == 1
                             and chain_graph_seg
                             and (verify_device or current_engine().kind == 'reference')
                             and not self._verify_device_legacy
                             and not hasattr(seqs[0], "_jetspec_device_outcome"))
        if self.jetspec and not self.jetspec_sync and not eagle:
            if outcome_already_sent or chain_first_round:
                request_control, request_tensors = None, []
                seqs[0]._jetspec_device_outcome_published = False
                if chain_first_round:
                    self._publish_chain_admission(seqs[0])
            else:
                legacy_audit = (getattr(self, "_verify_device_legacy", False)
                                and sync_audit)
                if legacy_audit:
                    with warnings.catch_warnings(record=True) as seen:
                        warnings.simplefilter("always")
                        request_control, request_tensors = self._pack_jetspec_request(seqs)
                    self.metrics.setdefault("jetspec_device_legacy_transport_sync_warnings", []).append(
                        sum("synchroniz" in str(warning.message).lower()
                            for warning in seen))
                else:
                    request_control, request_tensors = self._pack_jetspec_request(seqs)
        else:
            # Fill send buffers in-place (avoids torch.tensor from Python lists)
            for i, seq in enumerate(seqs):
                self._cache_keys[i, 0] = seq.seq_id
                self._cache_keys[i, 1] = (
                    max(0, seq.last_spec_step_accepted_len - 1)
                    if self.jetspec
                    else seq.last_spec_step_accepted_len - 1
                )
                self._cache_keys[i, 2] = seq.recovery_token_id
                self._jetspec_rounds[i] = seq.jetspec_round
                self._num_tokens_buf[i] = seq.num_tokens
                self._temps_buf[i] = seq.draft_temperature if seq.draft_temperature is not None else seq.temperature
                bt = seq.draft_block_table
                bt_len = len(bt)
                if bt_len > 0:
                    self._block_tables_buf[i, :bt_len] = torch.tensor(bt, dtype=torch.int32, device=self.device)
                self._block_tables_buf[i, bt_len:] = -1

            # Assemble one immutable async request. The draft worker consumes these
            # in order while target-side verification proceeds without waiting.
            temps_as_int64 = self._temps_buf.view(torch.int32).to(torch.int64)
            payloads = [
                self._cache_keys,
                self._num_tokens_buf,
                self._block_tables_buf.to(torch.int64),
                temps_as_int64,
            ]
            if self.jetspec:
                payloads.append(self._jetspec_rounds)
                if self.oracle_anchor:
                    exported = [oracle_request_predictions(self.metrics['jetspec_round_records'],
                        seq_id=int(seq.seq_id), round_id=int(seq.jetspec_round)) for seq in seqs]
                    payloads.append(torch.tensor(exported, dtype=torch.int64, device=self.device))
            request_tensors = [self._cmd, self._meta, concat_int64(*payloads)]

            if eagle:
                recovery_activations = torch.stack(
                    [seq.last_target_hidden_state for seq in seqs], dim=0,
                ).to(self.device)
                request_tensors.append(recovery_activations.to(self.draft_dtype))

                # Send extend data for glue decode with fused extend
                K = self.K
                act_dim = recovery_activations.shape[-1]
                for i, seq in enumerate(seqs):
                    self._extend_counts[i] = seq.extend_count
                extend_eagle_acts = torch.zeros(B, K, act_dim, dtype=self.draft_dtype, device=self.device)
                extend_token_ids = torch.zeros(B, K, dtype=torch.int64, device=self.device)
                for i, seq in enumerate(seqs):
                    n = seq.extend_count
                    if n > 0 and seq.extend_eagle_acts is not None:
                        extend_eagle_acts[i, :n] = seq.extend_eagle_acts[:n].to(self.draft_dtype)
                        extend_token_ids[i, :n] = seq.extend_token_ids[:n]
                request_tensors.extend(
                    [self._extend_counts, extend_eagle_acts, extend_token_ids]
                )

            if self.jetspec_taps:
                tap_started = time.perf_counter()
                from jetspec.engines.jetspec_plus_plus.wire.outcome_packet import outcome_layout
                layout = outcome_layout(self)
                tap_lengths, tap_payload = pack_jetspec_taps(
                    seqs,
                    device=self.device,
                    tap_shape=(layout.tap_count, layout.hidden),
                    dtype=layout.tap_dtype,
                )
                request_tensors.append(tap_lengths)
                if tap_payload.shape[0] > 0:
                    # The draft rank receives round taps in its own dtype (an fp32 target sends
                    # fp32 taps; a bf16 buffer would read their bytes as garbage).
                    request_tensors.append(tap_payload.to(dtype=self.draft_dtype))
                self.metrics["jetspec_tap_send_ms"].append(
                    (time.perf_counter() - tap_started) * 1000
                )
                self.metrics["jetspec_tap_payload_bytes"].append(
                    tap_lengths.numel() * tap_lengths.element_size()
                    + tap_payload.numel() * tap_payload.element_size()
                )
                for seq, length in zip(seqs, tap_lengths.tolist()):
                    seq.jetspec_taps_sent += int(length)

        if self.jetspec and not self.jetspec_sync:
            if not outcome_already_sent and not chain_first_round:
                self._prune_jetspec_request_sends()
                outcome_request = (len(request_tensors) == 1
                                   and request_tensors[0].dtype == torch.uint8)
                if (getattr(self, "_verify_device_legacy", False)
                        and sync_audit):
                    with warnings.catch_warnings(record=True) as seen:
                        warnings.simplefilter("always")
                        self._queue_jetspec_request(request_tensors, control=request_control,
                                                    signal_store=True)
                    self.metrics["jetspec_device_legacy_transport_sync_warnings"][-1] += sum(
                        "synchroniz" in str(warning.message).lower()
                        for warning in seen)
                else:
                    self._queue_jetspec_request(request_tensors, control=request_control,
                                                signal_store=not outcome_request)
            if self.oracle_anchor and not self._jetspec_candidate_receives:
                # Posting a future NCCL receive here would block the next verify.
                # Only the just-requested candidate is outstanding in serial mode.
                self._post_jetspec_candidate_receive()
        else:
            request_pg = self.jetspec_request_pg if self.jetspec else self.async_pg
            for tensor in request_tensors:
                dist.send(
                    tensor,
                    dst=self.draft_runner_rank,
                    group=request_pg,
                )

        _census_mark("request", run_config=getattr(self, 'run_config', None))
        if self.jetspec:
            if self.jetspec_sync:
                dist.recv(
                    self._jetspec_available,
                    src=self.draft_runner_rank,
                    group=self.jetspec_candidate_pg,
                )
                dist.recv(
                    self._jetspec_keys,
                    src=self.draft_runner_rank,
                    group=self.jetspec_candidate_pg,
                )
                dist.recv(
                    self._jetspec_blocks,
                    src=self.draft_runner_rank,
                    group=self.jetspec_candidate_pg,
                )
                dist.recv(
                    self._jetspec_draft_ms,
                    src=self.draft_runner_rank,
                    group=self.jetspec_candidate_pg,
                )
                draft_ms = self._jetspec_draft_ms.tolist()
                self.metrics["jetspec_draft_round_ms"].extend(draft_ms)
                candidate_sets = unpack_candidate_sets(
                    self._jetspec_available,
                    self._jetspec_keys,
                    self._jetspec_blocks,
                 run_config=getattr(self, 'run_config', None))
            else:
                stage_early = getattr(self, "_verify_stage_early_enabled", None)
                if stage_early is None:
                    stage_early = setting(self, 'engine.engine_settings.verify_stage_early', '0') == "1"
                if len(seqs) == 1 and stage_early and verify_device:
                    if self._verify_stage_runner is None:
                        raise RuntimeError("early verify stage lacks target runner")
                    self._verify_stage_runner.call("prepare_jetspec_mixed_device", seqs[0], True)
                self._wait_for_jetspec_candidates(seqs)
                _census_mark("cand_wait", run_config=getattr(self, 'run_config', None))
                candidate_sets = [
                    self.jetspec_mailbox.take(
                        seq_id=seq.seq_id,
                        round_id=seq.jetspec_round,
                    )
                    for seq in seqs
                ]
                if (redraft
                        and len(seqs) == 1):
                    candidate_sets = [self._chain_speculate_redraft(
                        seqs[0], candidate_sets[0],
                        max(0, seqs[0].last_spec_step_accepted_len - 1))]
                if handoff_timing:
                    consumed_ns = time.perf_counter_ns()
                    for seq, candidate_set in zip(seqs, candidate_sets):
                        if candidate_set is not None:
                            print("CHAIN_HANDOFF_CONSUME " + json.dumps(dict(
                                seq_id=seq.seq_id, round=candidate_set.round_id,
                                consume_ns=consumed_ns)), flush=True)
                self.metrics["jetspec_stale_count"] = self.jetspec_mailbox.stale
                _census_mark("cand_take", run_config=getattr(self, 'run_config', None))
            reference_select = B == 1 and getattr(self, "_reference_host_select", False)
            lean_select = B == 1 and select_lean and verify_device or reference_select
            host_select = lean_select and (getattr(self, "_verify_select_host", False) or reference_select)
            list_select = host_select and (self._verify_select_list or reference_select)
            tokens = [[0] * self.K] if lean_select else torch.zeros(
                B, self.K, dtype=torch.int64, device="cpu"
            )
            verify_mask = [0] if lean_select else torch.zeros(B, dtype=torch.int64, device="cpu")
            if reference_select:
                # Reference acceptance still receives its original int64 CPU mask.
                # The prior verify has finished before this storage is reused.
                verify_mask = getattr(self, '_reference_verify_mask', None)
                if verify_mask is None:
                    verify_mask = self._reference_verify_mask = torch.zeros(1, dtype=torch.int64, device='cpu')
                else:
                    verify_mask.zero_()
            decisions = []
            for i, (seq, candidate_set) in enumerate(zip(seqs, candidate_sets)):
                true_k = max(0, seq.last_spec_step_accepted_len - 1)
                previous_round_ar = self._jetspec_previous_round_ar.get(
                    int(seq.seq_id), False
                )
                candidate_gap = self._jetspec_candidate_gaps.pop(
                    (seq.seq_id, seq.jetspec_round), 0
                )
                candidate_kind = None
                if candidate_set is not None:
                    candidate_kind = (
                        "catchup" if candidate_gap > 0 else "speculative"
                    )
                selected_guess = None
                selected_candidate_ids = None
                selected_block = None
                if host_select and (candidate_set is None or hasattr(candidate_set, "_blocks_host")):
                    selector = (select_candidate_drafts_list if list_select
                                else select_candidate_drafts_host)
                    decision, selected_candidate_ids = selector(
                        candidate_set, seq_id=seq.seq_id, round_id=seq.jetspec_round,
                        true_k=true_k, correction=seq.recovery_token_id,
                        miss_mode=self.jetspec_miss, counters=self.jetspec_counters)
                    if selected_candidate_ids is not None:
                        selected_guess = selected_candidate_ids[0]
                else:
                    if (
                        candidate_set is not None
                        and (candidate_set.row_keys is not None or
                             0 <= true_k < candidate_set.blocks.shape[0])
                    ):
                        selected_block = (select_candidate_block_indexed(
                            candidate_set, true_k=true_k, correction=seq.recovery_token_id)
                            if lean_select else select_candidate_block(candidate_set.blocks,
                                true_k=true_k, correction=seq.recovery_token_id,
                                row_keys=candidate_set.row_keys))
                        selected_candidate_ids = selected_block.tolist()
                        selected_guess = selected_candidate_ids[0]
                    decision = select_candidate_drafts(
                        candidate_set,
                        seq_id=seq.seq_id,
                        round_id=seq.jetspec_round,
                        true_k=true_k,
                        correction=seq.recovery_token_id,
                        miss_mode=self.jetspec_miss,
                        counters=self.jetspec_counters,
                        selected_block=selected_block,
                    )
                _census_mark("select", run_config=getattr(self, 'run_config', None))
                if setting(self, 'engine.engine_settings.chain_tail_trace', '0') == '1':
                    print(f"MAILBOX_ID side=verify event=select seq={seq.seq_id} round={seq.jetspec_round} branch={decision.kind} ar={len(decision.drafts) == 0}", flush=True)
                fallback = None
                if self.miss_redraft and seq.jetspec_round > 0 and (
                        decision.kind != 'hit' or self.miss_redraft_force):
                    # The draft rank saw the same miss in this round's request and publishes one
                    # fresh row for this round; it is an ordinary keyed candidate.
                    waited = time.perf_counter()
                    self._wait_for_jetspec_candidates([seq])
                    fallback_set = self.jetspec_mailbox.take(
                        seq_id=seq.seq_id, round_id=seq.jetspec_round)
                    if host_select and hasattr(fallback_set, "_blocks_host"):
                        fallback_selector = select_candidate_drafts_list if reference_select else select_candidate_drafts_host
                        fallback_decision, fallback_ids = fallback_selector(
                            fallback_set, seq_id=seq.seq_id, round_id=seq.jetspec_round,
                            true_k=true_k, correction=seq.recovery_token_id,
                            miss_mode=self.jetspec_miss, counters=ProtocolCounters())
                    else:
                        fallback_block = (select_candidate_block_indexed(
                            fallback_set, true_k=true_k, correction=seq.recovery_token_id)
                            if lean_select else select_candidate_block(fallback_set.blocks,
                                true_k=true_k, correction=seq.recovery_token_id,
                                row_keys=fallback_set.row_keys))
                        fallback_decision = select_candidate_drafts(
                            fallback_set, seq_id=seq.seq_id, round_id=seq.jetspec_round,
                            true_k=true_k, correction=seq.recovery_token_id,
                            miss_mode=self.jetspec_miss, counters=ProtocolCounters(),
                            selected_block=fallback_block)
                        fallback_ids = fallback_block.tolist()
                    if fallback_decision.kind != 'hit':
                        raise RuntimeError('fallback row does not match the committed correction')
                    fallback = dict(fallback=True, fallback_wait_ms=(time.perf_counter() - waited) * 1000,
                                    fallback_first_kind=decision.kind)
                    selected_candidate_ids = fallback_ids
                    selected_guess = selected_candidate_ids[0]
                    decision = type(decision)(decision.kind, fallback_decision.drafts)
                redraft_info = getattr(self, "_chain_redraft_info", None)
                if (redraft_info is not None and
                        redraft_info[:2] == (seq.seq_id, seq.jetspec_round)):
                    self._chain_redraft_info = None
                    _, _, delivered, redraft_wait_ms = redraft_info
                    if delivered:
                        if decision.kind != "hit":
                            raise RuntimeError("same-round redraft key was not verified")
                        self.jetspec_counters.hit -= 1
                        self.jetspec_counters.miss += 1
                        decision = type(decision)("miss", decision.drafts)
                        fallback = dict(fallback=True, chain_redraft=True,
                                        fallback_first_kind="miss",
                                        fallback_wait_ms=redraft_wait_ms)
                    else:
                        fallback = dict(chain_redraft_timeout=True,
                                        fallback_wait_ms=redraft_wait_ms)
                from jetspec.engines.jetspec_plus_plus.runtime.profile import enabled as profile_enabled
                if trace and profile_enabled():
                    used_redraft = bool(fallback and fallback.get("chain_redraft"))
                    first_hit = getattr(seq, "_jetspec_first_set_anchor_hit", None)
                    if first_hit is None:
                        first_hit = chain_exact_hit(ChainKeySnapshot.from_candidate(
                            candidate_set, round_id=seq.jetspec_round),
                            true_k, int(seq.recovery_token_id))
                    fallback = dict(fallback or {}, redraft=used_redraft,
                                    first_set_anchor_hit=first_hit,
                                    anchor_hit_set="redraft" if used_redraft else "first")
                draft_count = (len(decision.drafts) if list_select and isinstance(decision.drafts, list)
                               else decision.drafts.numel())
                if verify_device or current_engine().kind == 'reference':
                    seq._jetspec_verified_row_key = (
                        (true_k, int(seq.recovery_token_id))
                        if draft_count else (-1, -1))
                decisions.append(decision.kind)
                if draft_count:
                    tokens[i] = (decision.drafts if isinstance(decision.drafts, list)
                                 else decision.drafts.tolist() if lean_select else decision.drafts)
                    verify_mask[i] = 1
                self._jetspec_previous_round_ar[int(seq.seq_id)] = not bool(
                    draft_count
                )
                self.metrics[(self._verify_decision_metric_keys[decision.kind]
                              if list_select and not reference_select else f"jetspec_{decision.kind}_count")] += 1
                rounds = (
                    self.jetspec_counters.hit
                    + self.jetspec_counters.miss
                    + self.jetspec_counters.late
                )
                self.metrics["jetspec_late_fraction"] = (
                    self.jetspec_counters.late / rounds if rounds else 0.0
                )
                if B == 1 and verify_device and getattr(self, "_verify_record_late", False):
                    record_args = (seq, candidate_set, true_k, decision.kind, selected_guess,
                                   selected_candidate_ids, candidate_kind, previous_round_ar,
                                   candidate_gap, fallback, trace)
                    if getattr(self, "_pending_jetspec_record", None) is not None:
                        raise RuntimeError("previous JetSpec record was not flushed")
                    self._pending_jetspec_record = record_args
                else:
                    coverage = {}
                    if trace:
                        served_ks = []
                        rows_published = None
                        anchor_hit = decision.kind == 'hit'
                        if candidate_set is not None:
                            if candidate_set.row_keys is not None:
                                served_ks, rows_published, anchor_hit = candidate_row_coverage(
                                    (candidate_set._row_keys_host if reference_select else candidate_set.row_keys), true_k=true_k,
                                    correction=seq.recovery_token_id)
                            else:
                                pins = candidate_set.blocks[..., 0]
                                valid = (pins >= 0) if pins.ndim == 1 else (pins >= 0).any(-1)
                                served_ks = valid.nonzero().reshape(-1).tolist()
                        coverage = {
                            "k_true": true_k,
                            "k_set": served_ks,
                            "kset_served_size": len(served_ks),
                            "k_hit": true_k in served_ks,
                            "hit_at_f": decision.kind == 'hit',
                            "rows_published": len(served_ks) if rows_published is None else rows_published,
                            "anchor_hit": anchor_hit,
                            "oracle_masked_k15": self.oracle_anchor and true_k == 15,
                        }
                    self.metrics["jetspec_round_records"].append(
                        {
                            "seq_id": seq.seq_id,
                            "prompt_index": int(
                                seq.seq_id
                                if seq.jetspec_prompt_index is None
                                else seq.jetspec_prompt_index
                            ),
                            "round": seq.jetspec_round,
                            "candidate_round": (
                                candidate_set.round_id
                                if candidate_set is not None
                                else None
                            ),
                            "k": true_k,
                            **coverage,
                            "correction": int(seq.recovery_token_id),
                            "guess": selected_guess,
                            "candidate_draft_ids": (
                                selected_candidate_ids[1:]
                                if selected_candidate_ids is not None
                                else None
                            ),
                            "decision": decision.kind,
                            "candidate_kind": candidate_kind,
                            "previous_round_ar": previous_round_ar,
                            "gap": candidate_gap,
                            "wait_ms": self.jetspec_wait_ms,
                            **({} if fallback is None else fallback),
                        }
                    )
                    _census_mark("record", run_config=getattr(self, 'run_config', None))
            return tokens, self._logits_q, verify_mask, verify_mask, decisions

        # Recv into pre-allocated buffers
        dist.recv(self._fused_response, src=self.draft_runner_rank, group=self.async_pg)
        cache_hits = self._fused_response[:B]
        speculations = self._fused_response[B:].view(B, self.K)
        dist.recv(self._logits_q, src=self.draft_runner_rank, group=self.async_pg)

        return speculations, self._logits_q, cache_hits, None, None
