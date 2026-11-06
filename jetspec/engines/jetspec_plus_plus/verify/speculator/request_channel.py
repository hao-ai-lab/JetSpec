from __future__ import annotations
import time
import torch
import torch.distributed as dist
from jetspec.engines.jetspec_plus_plus.wire.outcome_packet import announce_chain_command
from jetspec.engines.jetspec_plus_plus.wire.outcome_packet import await_chain_receive
from jetspec.engines.jetspec_plus_plus.wire.protocol import RequestPublisher
from jetspec.engines.jetspec_plus_plus.wire.protocol import pack_request_wire
from jetspec.engines.jetspec_plus_plus.wire.protocol import candidate_wire_shape
from jetspec.engines.jetspec_plus_plus.wire.pair_transport import announce_check_outcome
from jetspec.engines.jetspec_plus_plus.wire.pair_transport import claim_command_id

class RequestChannel:
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

    def _pack_jetspec_request(self, seqs):
        from jetspec.engines.jetspec_plus_plus.serving.settings import setting
        from jetspec.engines.jetspec_plus_plus.runtime.engine import current_engine
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

    def _prune_jetspec_request_sends(self) -> None:
        self._jetspec_request_sends = [
            item
            for item in self._jetspec_request_sends
            if not all(work.is_completed() for work in item[0])
        ]

    def _queue_jetspec_request(self, tensors: list[torch.Tensor], *, control=None,
                               signal_store=True, chain_command=None,
                               chain_round=None, chain_length=None, chain_after_miss=None,
                               chain_eos=None, chain_sync_at=None, chain_true_k=None,
                               chain_recovery=None) -> None:
        from jetspec.engines.jetspec_plus_plus.runtime.profiles import kernel_enabled
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
            if (chain_command == "outcome"
                    and not getattr(self, "_outcome_prepost_no_ack", False)):
                timing = getattr(self, "_chain_prepost_timing", None)
                if timing is None:
                    await_chain_receive(self.jetspec_mailbox_store, chain_command_id, run_config=getattr(self, 'run_config', None))
                else:
                    await_chain_receive(self.jetspec_mailbox_store, chain_command_id,
                                        timing=timing, run_config=getattr(self, 'run_config', None))
        publication = None
        if control is not None:
            if self._jetspec_request_publisher is None:
                self._jetspec_request_publisher = RequestPublisher(self.jetspec_mailbox_store)
            publication = self._jetspec_request_publisher.post(
                f"request_{request_id}", control.numpy().tobytes())
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
