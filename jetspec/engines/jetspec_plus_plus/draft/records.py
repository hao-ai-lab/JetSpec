"""Default-off calibration capture after chain replay, with delayed verify truth."""
from __future__ import annotations

import torch
from jetspec.engines.jetspec_plus_plus.draft.method.cache_rows import DeviceOffers, blend_features, blend_mass, joint_scores, select_rows

from jetspec.engines.jetspec_plus_plus.wire.contract import Offset
from jetspec.engines.jetspec_plus_plus.wire.contract import OutcomeOffset
from jetspec.engines.jetspec_plus_plus.jetspec.pass_labels import chain_labels


def candidate_snapshot(ids, probabilities):
    """JSON arrays: block slot (including anchor slot zero), then candidate rank."""
    return dict(token_ids=ids.detach().cpu().tolist(),
                probabilities=probabilities.detach().cpu().tolist())


class ChainWireRecorder:
    """Own host snapshots until the verifier returns the matching round's truth."""

    def __init__(self, root, *, seq_id, run_config=None):
        from jetspec.engines.jetspec_plus_plus.draft.config import as_run_config
        self.run_config = as_run_config(run_config)
        self.root = root
        self.seq_id = seq_id
        self.pending = {}
        self.taps_written = 0

    def resolve(self, packet, run_config=None):
        run_config = run_config if run_config is not None else getattr(self, 'run_config', None)
        words = packet.detach().cpu().tolist()
        round_id = int(words[OutcomeOffset.ROUND])
        record = self.pending.pop(round_id, None)
        if record is None:
            return  # round zero is the synthetic admission outcome
        if int(words[OutcomeOffset.SEQ_ID]) != self.seq_id:
            raise RuntimeError('chain calibration outcome changed sequence ID')
        k, anchor = int(words[OutcomeOffset.TRUE_K]), int(words[OutcomeOffset.RECOVERY])
        keys = record['published_keys']
        record.update(k_true=k, anchor_true=anchor, true_anchor=anchor, correction=anchor,
                      k_hit=any(row[0] == k for row in keys),
                      anchor_hit=[k, anchor] in keys,
                      outcome_miss=[k, anchor] not in keys)
        append_proposal_record(self.root, record, run_config=run_config)

    def capture(self, buffers, *, round_id, scored_passes, run_config=None):
        if not bool(buffers['published'].item()):
            return  # EOS publishes an empty envelope, not a proposal
        rp = buffers['rp'].detach().cpu().tolist()
        after_miss = bool(rp[Offset.AFTER_MISS])
        label = chain_labels(run_config=run_config)[int(rp[Offset.TAU_IDX])]
        keys = buffers['wire_keys'].detach().cpu()
        keys = keys[(keys[:, 0] >= 0) & (keys[:, 1] >= 0)]
        logits = buffers['dec_logits'].float()
        drafts = buffers['tokens'].pending_draft_ids
        rejected = torch.cat((drafts, drafts.new_full((1,), -1)))
        if after_miss:
            rejected[0] = -1
        q = logits.softmax(-1).gather(1, rejected.clamp_min(0)[:, None])[:, 0]
        q, argmax, rejected_ids = (q.detach().cpu().tolist(),
                                   logits.argmax(-1).detach().cpu().tolist(),
                                   rejected.detach().cpu().tolist())
        offers = dict(d_rej=rejected_ids,
                      top_ids=buffers['offers_ids'][:, :32].detach().cpu().tolist(),
                      top_probabilities=buffers['offers_p64'][:, :32].detach().cpu().tolist(),
                      pre_exclusion_logsumexp=logits.logsumexp(-1).detach().cpu().tolist())
        n, R = logits.shape[0], buffers['keys'].shape[0]
        home = buffers['row_home'].detach().cpu()
        mass = home[1 + 2*R + 5*n:1 + 2*R + 5*n + n + 1].view(torch.float64)
        requested = int(buffers['requested'].item())
        record = proposal_record(
            seq_id=self.seq_id, candidate_round=round_id, pass_label=label,
            predictor_probabilities=mass, anchor_logits=None, rejected=None,
            after_miss=after_miss, k_set=sorted(set(keys[:, 0].tolist())),
            rows_requested=requested, rows_published=len(keys),
            rows_truncated=max(0, requested - R), fallback_anchor=rp[Offset.CORRECTION],
            published_keys=keys, batched_offers=offers, run_config=run_config if run_config is not None else getattr(self, 'run_config', None))
        record.update(schema='jetspec.chain_wire_record/1', record_kind='proposal',
                      passes_run=int(scored_passes), raw_pass_label=f'J{int(scored_passes)-1}',
                      served_pass_label=label, row_cap=R,
                      p_top1=buffers['prior_p_top1'].detach().cpu().double().tolist(),
                      q_shadow=[value if token >= 0 else None for value, token in zip(q, rejected_ids)],
                      shadow_argmax=argmax, slot_index=list(range(1, n)),
                      published_blocks=buffers['wire_rows'].detach().cpu().tolist(),
                      published_p_top1=buffers['final_p_top1'].detach().cpu().double().tolist(),
                      published_passes_run=buffers['passes_run'].detach().cpu().tolist())
        if 'prior_draft_top16_ids' in buffers:
            record.update(schema='jetspec.eval_round_candidates/1',
                          draft_top16=candidate_snapshot(buffers['prior_draft_top16_ids'],
                                                         buffers['prior_draft_top16_probabilities']),
                          published_draft_top16=candidate_snapshot(buffers['wire_draft_top16_ids'],
                                                                   buffers['wire_draft_top16_probabilities']),
                          draft_token_ids=buffers['prior_draft_token_ids'].cpu().tolist(),
                          draft_token_probabilities=buffers['prior_draft_token_probabilities'].cpu().tolist(),
                          draft_pinned=buffers['prior_draft_pinned'].cpu().tolist(),
                          published_draft_token_probabilities=buffers['wire_draft_token_probabilities'].cpu().tolist(),
                          published_draft_pinned=buffers['wire_draft_pinned'].cpu().tolist())
        if 'capture_dec_window' in buffers:
            self._write_decoder_inputs(buffers, round_id)
        if round_id in self.pending:
            raise RuntimeError('chain calibration published a round twice')
        self.pending[round_id] = record

    def _write_decoder_inputs(self, buffers, round_id):
        """Append the decoder window and the newly committed prefix taps as raw BF16."""
        import json
        from pathlib import Path
        dec_len, c1_count = (int(value) for value in buffers['capture_dec_lengths'].cpu().tolist())
        window = buffers['capture_dec_window'].detach().cpu().contiguous()
        start = self.taps_written
        taps = buffers['tap_store'].storage[start:dec_len, 4].detach().cpu().contiguous()
        # A subdirectory keeps the sidecar out of the candidate glob over <root>/*.jsonl.
        root = Path(self.root) / 'decoder_inputs'
        root.mkdir(parents=True, exist_ok=True)
        data = root / 'decoder_inputs.bin'
        with data.open('ab') as sink:
            window_offset = sink.tell()
            sink.write(window.view(torch.int16).numpy().tobytes())
            taps_offset = sink.tell()
            sink.write(taps.view(torch.int16).numpy().tobytes())
        row = dict(schema='jetspec.eval_decoder_inputs/1', seq_id=self.seq_id,
                   candidate_round=int(round_id), record_kind='proposal', dtype='bfloat16',
                   window_offset=window_offset, window_shape=list(window.shape),
                   dec_len=dec_len, c1_count=c1_count, taps_layer=4,
                   taps_from=start, taps_to=max(start, dec_len), taps_offset=taps_offset,
                   taps_shape=list(taps.shape))
        with (root / 'decoder_inputs.jsonl').open('a') as index:
            index.write(json.dumps(row) + '\n')
        self.taps_written = max(start, dec_len)

    def redraft(self, buffers, *, round_id, run_config=None):
        """A miss recovery is separate from the original scored proposal."""
        run_config = run_config if run_config is not None else getattr(self, 'run_config', None)
        packet = buffers['packet'].detach().cpu().tolist()
        keys = buffers['redraft_keys'].detach().cpu()
        keys = keys[(keys[:, 0] >= 0) & (keys[:, 1] >= 0)]
        k, anchor = int(packet[OutcomeOffset.TRUE_K]), int(packet[OutcomeOffset.RECOVERY])
        record = proposal_record(
            seq_id=self.seq_id, candidate_round=round_id, pass_label='redraft',
            predictor_probabilities=None, anchor_logits=None, rejected=None,
            after_miss=True, k_set=sorted(set(keys[:, 0].tolist())),
            rows_requested=len(keys), rows_published=len(keys), rows_truncated=0,
            fallback_anchor=anchor, published_keys=keys, run_config=run_config if run_config is not None else getattr(self, 'run_config', None))
        record.update(schema='jetspec.chain_wire_record/1', record_kind='redraft',
                      passes_run=1, raw_pass_label='J0', served_pass_label='redraft',
                      k_true=k, anchor_true=anchor, true_anchor=anchor, correction=anchor,
                      p_top1=buffers['redraft_scores'][0].detach().cpu().double().tolist(),
                      q_shadow=None, shadow_argmax=None,
                      published_blocks=buffers['redraft_rows'].detach().cpu().tolist())
        if 'redraft_draft_top16_ids' in buffers:
            from jetspec.engines.jetspec_plus_plus.jetspec.pass_labels import completed_label
            label = completed_label(buffers['redraft_passes'][0])
            record.update(schema='jetspec.eval_round_candidates/1',
                          pass_label=label, served_pass_label=label,
                          passes_run=int(buffers['redraft_passes'][0]), raw_pass_label=label.replace('d0', ''),
                          draft_top16=candidate_snapshot(buffers['redraft_draft_top16_ids'][0],
                                                         buffers['redraft_draft_top16_probabilities'][0]),
                          published_draft_top16=candidate_snapshot(buffers['redraft_draft_top16_ids'],
                                                                   buffers['redraft_draft_top16_probabilities']),
                          draft_token_ids=buffers['redraft_rows'][0].cpu().tolist(),
                          draft_token_probabilities=buffers['redraft_draft_token_probabilities'][0].cpu().tolist(),
                          draft_pinned=buffers['redraft_draft_pinned'][0].cpu().tolist(),
                          published_draft_token_probabilities=buffers['redraft_draft_token_probabilities'].cpu().tolist(),
                          published_draft_pinned=buffers['redraft_draft_pinned'].cpu().tolist())
        append_proposal_record(self.root, record, run_config=run_config)




from jetspec.engines.jetspec_plus_plus.draft.config import legacy_name, named_setting
import json
import os
from pathlib import Path

E04_RECORDED_ENV = (
    legacy_name('method.method_settings.kpred_renorm'), legacy_name('method.method_settings.kpred_posterior'), legacy_name('engine.engine_settings.record_ptop1'),
    legacy_name('engine.engine_settings.record_qshadow'), legacy_name('method.method_settings.kscore'), legacy_name('method.method_settings.rowrule'), legacy_name('method.method_settings.kpred_coverage'),
    legacy_name('method.method_settings.anchor_coverage'), legacy_name('method.method_settings.kset_cap'), legacy_name('method.method_settings.anchor_cap'), legacy_name('engine.engine_settings.rows_cap'),
    legacy_name('method.method_settings.anchor_drej'), legacy_name('method.method_settings.kset_after_miss'), legacy_name('method.method_settings.kpred_frame'),
    legacy_name('engine.engine_settings.legacy_kpred_forward'))



def _anchor_evidence(logits, rejected, top_n):
    logits = torch.as_tensor(logits).float()
    rejected = torch.as_tensor(rejected, device=logits.device, dtype=torch.long)
    if logits.ndim != 2 or rejected.shape != logits.shape[:1]:
        raise ValueError("anchor logits and rejected ids have incompatible shapes")
    pre_exclusion_logsumexp = logits.logsumexp(-1)
    filtered = logits.clone()
    valid = rejected >= 0
    filtered[valid, rejected[valid]] = float("-inf")
    probabilities = filtered.softmax(-1).double()
    ranked = torch.argsort(probabilities, dim=-1, descending=True, stable=True)
    ranked = ranked[:, :min(int(top_n), logits.shape[1])]
    return {
        "top_ids": ranked.tolist(),
        "top_probabilities": probabilities.gather(1, ranked).tolist(),
        "pre_exclusion_logsumexp": pre_exclusion_logsumexp.tolist(),
        "d_rej": rejected.tolist(),
    }




def proposal_record(*, seq_id, candidate_round, pass_label,
                    predictor_probabilities, anchor_logits, rejected,
                    after_miss, k_set, rows_requested, rows_published,
                    rows_truncated, fallback_anchor, published_keys, top_n=32,
                    batched_offers=None, run_config=None):
    """Build one JSON-safe proposal-time record without changing decisions."""
    from jetspec.engines.jetspec_plus_plus.wire.protocol import draft_width

    prediction_width = draft_width(run_config=run_config) + 2
    prediction = None
    if predictor_probabilities is not None:
        prediction = torch.as_tensor(predictor_probabilities).double().reshape(-1)
        if (prediction.shape != (prediction_width,)
                or not bool(torch.isfinite(prediction).all())):
            raise ValueError(
                f"wire recorder requires {prediction_width} finite predictor probabilities")
    if prediction is None:
        wire = None
    elif after_miss:
        wire = prediction.new_zeros(draft_width(run_config=run_config) + 1)
        wire[0] = 1.0
    else:
        wire = prediction[1:]
    offers = (batched_offers if batched_offers is not None else
              None if anchor_logits is None else
              _anchor_evidence(anchor_logits, rejected, top_n))
    key_rows = torch.as_tensor(
        published_keys, dtype=torch.long).reshape(-1, 2).tolist()
    return {
        "seq_id": int(seq_id),
        "candidate_round": int(candidate_round),
        "pass_label": str(pass_label),
        "p_pred": None if prediction is None else prediction.tolist(),
        "p_wire": None if wire is None else wire.tolist(),
        "offers": offers,
        "fallback_anchor": int(fallback_anchor),
        "k_set": [int(value) for value in k_set],
        "rows_requested": int(rows_requested),
        "rows_published": int(rows_published),
        "rows_truncated": int(rows_truncated),
        "published_keys": key_rows,
        "after_miss": bool(after_miss),
    }


def e04_record_flags(record, run_config=None):
    """e04 opt-in (E04_RECORD_FLAGS=1): the policy flags in force, logged on every round."""
    from jetspec.engines.jetspec_plus_plus.draft.config import setting, required_setting, as_run_config
    if required_setting(run_config, 'engine.engine_settings.record_flags') != "1":
        raise ValueError("E04_RECORD_FLAGS must be unset or 1")
    return dict(record, e04_flags={name: named_setting(run_config, name) for name in E04_RECORDED_ENV})


def append_proposal_record(directory, record, run_config=None):
    """Append one compact JSON line to a process-owned recorder shard."""
    from jetspec.engines.jetspec_plus_plus.draft.config import setting, required_setting, as_run_config
    if setting(run_config, 'engine.engine_settings.record_flags'):
        record = e04_record_flags(record, run_config=run_config)
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{os.getpid()}.jsonl"
    with path.open("a") as stream:
        stream.write(json.dumps(record, separators=(",", ":")) + "\n")
    return path
