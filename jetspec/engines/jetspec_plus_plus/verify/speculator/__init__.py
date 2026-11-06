from __future__ import annotations

from jetspec.engines.jetspec_plus_plus.runtime.profiles import kernel_enabled, kernel_value

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
from jetspec.engines.jetspec_plus_plus.serving.token_text import decode_tokens
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


from jetspec.engines.jetspec_plus_plus.verify.speculator.request_channel import RequestChannel

from jetspec.engines.jetspec_plus_plus.verify.speculator.round_records import RoundRecords

from jetspec.engines.jetspec_plus_plus.verify.speculator.candidate_channel import CandidateChannel

from jetspec.engines.jetspec_plus_plus.verify.speculator.chain_redraft import ChainRedraft

from jetspec.engines.jetspec_plus_plus.verify.speculator.outcome_publisher import OutcomePublisher

class SpeculatorAsync(RequestChannel, RoundRecords, CandidateChannel, ChainRedraft, OutcomePublisher, SpeculatorBase):
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
        from jetspec.engines.jetspec_plus_plus.runtime.profiles import transport_enabled, TRANSPORT_SWITCHES
        from jetspec.engines.jetspec_plus_plus.runtime.profiles import kernel_enabled
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
                from jetspec.engines.jetspec_plus_plus.runtime.profiles import verify_cut_lean27b
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
        from jetspec.engines.jetspec_plus_plus.runtime.profiles import kernel_enabled
        from jetspec.engines.jetspec_plus_plus.runtime.engine import current_engine
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
            from jetspec.engines.jetspec_plus_plus.runtime.engine import current_engine
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

    

    

    

    

    

    

    

    

    

    

    

    

    def prefill(self, seqs: list[Sequence], verify_result: VerifyResult) -> SpeculateResult:
        from jetspec.engines.jetspec_plus_plus.serving.settings import setting
        from jetspec.engines.jetspec_plus_plus.runtime.engine import current_engine
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

            return SpeculateResult(
                speculations,
                logits_q,
                cache_hits,
                jetspec_verify_mask,
                jetspec_decisions,
            )

    

    

    

    

    

    

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

    

    

    

    

    

    def _speculation_request(self, seqs: list[Sequence], eagle: bool):
        from jetspec.engines.jetspec_plus_plus.serving.settings import setting
        from jetspec.engines.jetspec_plus_plus.runtime.profiles import kernel_enabled
        from jetspec.engines.jetspec_plus_plus.runtime.engine import current_engine
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
            return tokens, self._logits_q, verify_mask, verify_mask, decisions

        # Recv into pre-allocated buffers
        dist.recv(self._fused_response, src=self.draft_runner_rank, group=self.async_pg)
        cache_hits = self._fused_response[:B]
        speculations = self._fused_response[B:].view(B, self.K)
        dist.recv(self._logits_q, src=self.draft_runner_rank, group=self.async_pg)

        return speculations, self._logits_q, cache_hits, None, None
