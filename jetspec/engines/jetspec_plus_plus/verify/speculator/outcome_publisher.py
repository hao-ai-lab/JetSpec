from __future__ import annotations
import time
import warnings
from jetspec.engines.jetspec_plus_plus.wire.outcome_packet import admission_outcome
from jetspec.engines.jetspec_plus_plus.wire.outcome_packet import make_hash16_weights
from jetspec.engines.jetspec_plus_plus.wire.pair_transport import claim_command_id

class OutcomePublisher:
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

    def _publish_chain_admission(self, seq):
        """Replace the suppressed legacy first request with a device packet."""
        from jetspec.engines.jetspec_plus_plus.serving.settings import setting
        from jetspec.engines.jetspec_plus_plus.runtime.kernel_profiles import kernel_enabled
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

    def pack_device_outcome_presync(self, seq):
        """Enqueue the final packet and tap copies after the mirror event."""
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
        self._prune_jetspec_request_sends()

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

    def publish_device_outcome(self, seqs, outcome):
        """Send the next-round correction and fixed tap block before host postprocess."""
        from jetspec.engines.jetspec_plus_plus.serving.settings import setting
        from jetspec.engines.jetspec_plus_plus.runtime.kernel_profiles import kernel_enabled
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
            if getattr(self, "_outcome_pack_presync", False):
                self.check_device_outcome_presync(seq)
            _, tensors = self._pack_jetspec_request(seqs)
            if not getattr(self, "_outcome_pack_presync", False):
                if not getattr(self, "_reference_tap_prepare", False):
                    self._prune_jetspec_request_sends()
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
