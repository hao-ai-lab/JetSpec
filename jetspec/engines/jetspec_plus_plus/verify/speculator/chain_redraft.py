from __future__ import annotations
import time
from jetspec.engines.jetspec_plus_plus.wire.protocol import ChainKeySnapshot
from jetspec.engines.jetspec_plus_plus.wire.protocol import chain_exact_hit
from jetspec.engines.jetspec_plus_plus.wire.protocol import chain_after_miss

class ChainRedraft:
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
        from jetspec.engines.jetspec_plus_plus.runtime.engine import current_engine
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
