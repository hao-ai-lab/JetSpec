from __future__ import annotations
from jetspec.engines.jetspec_plus_plus.wire.protocol import candidate_row_coverage

class RoundRecords:
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
