"""Ported from the JetSpec++ reference implementation."""
import torch

SCHEMA = 'jetspec.eval_round_candidates/1'


def block_snapshot(result):
    """Read the final effective pass, including probabilities of forced tokens."""
    probabilities = result.logits[0].float().softmax(-1)
    ranked = probabilities.topk(16, -1)
    return dict(top16=dict(token_ids=ranked.indices.cpu().tolist(),
                           probabilities=ranked.values.cpu().tolist()),
                token_ids=result.ids[0].cpu().tolist(),
                probabilities=probabilities.gather(-1, result.ids[0, :, None])[:, 0].cpu().tolist(),
                pinned=result.pinned[0].cpu().tolist())


def candidate_record(result, published, keys, *, seq_id, round_id, kind,
                     requested, after_miss=False, decoder_logits=None, prediction=None):
    """A publication for round r; its incoming truth and verifier are joined later."""
    snapshots = {}

    def snapshot(block):
        if id(block) not in snapshots:
            snapshots[id(block)] = block_snapshot(block)
        return snapshots[id(block)]

    prior = snapshot(result)
    blocks = [snapshot(block) for block in published]
    record = dict(schema=SCHEMA, seq_id=int(seq_id), candidate_round=round_id, record_kind=kind,
                  pass_label=result.label, p_top1=result.p_top1[0].double().cpu().tolist(),
                  draft_token_ids=prior['token_ids'], draft_top16=prior['top16'],
                  draft_token_probabilities=prior['probabilities'], draft_pinned=prior['pinned'],
                  published_keys=keys, published_blocks=[block['token_ids'] for block in blocks],
                  published_draft_top16={name: [block['top16'][name] for block in blocks]
                                        for name in ('token_ids', 'probabilities')},
                  published_draft_token_probabilities=[block['probabilities'] for block in blocks],
                  published_draft_pinned=[block['pinned'] for block in blocks],
                  p_pred=None, p_wire=None, offers=None, q_shadow=None, shadow_argmax=None,
                  fallback_anchor=prior['token_ids'][0], k_set=sorted({key[0] for key in keys}),
                  rows_requested=requested, rows_published=len(keys),
                  rows_truncated=max(0, requested - len(keys)), after_miss=after_miss)
    if kind == 'admission':
        record['published_pass_labels'] = [block.label for block in published]
    else:
        passes = (sum(op.startswith('J') for op in result.operations)
                  if kind == 'proposal' else result.passes_run)
        record.update(passes_run=passes, raw_pass_label=f'J{passes - 1}', served_pass_label=result.label)
    if kind == 'proposal':
        logits = decoder_logits.float()
        rejected = torch.cat((result.ids[0, 1:], result.ids.new_tensor([-1])))
        if after_miss:
            rejected[0] = -1
        filtered = logits.clone()
        valid = rejected >= 0
        filtered[valid, rejected[valid]] = float('-inf')
        probabilities = filtered.softmax(-1).double()
        ids = torch.argsort(probabilities, descending=True, stable=True)[:, :32]
        q = logits.softmax(-1).gather(-1, rejected.clamp_min(0)[:, None])[:, 0].cpu().tolist()
        record.update(p_pred=prediction.double().cpu().tolist(),
                      p_wire=([1.0] + [0.0] * 15 if after_miss else prediction[1:].double().cpu().tolist()),
                      offers=dict(d_rej=rejected.cpu().tolist(), top_ids=ids.cpu().tolist(),
                                  top_probabilities=probabilities.gather(-1, ids).cpu().tolist(),
                                  pre_exclusion_logsumexp=logits.logsumexp(-1).cpu().tolist()),
                      q_shadow=[value if token >= 0 else None for value, token in zip(q, rejected.cpu().tolist())],
                      shadow_argmax=logits.argmax(-1).cpu().tolist(), slot_index=list(range(1, 16)),
                      row_cap=len(keys), published_p_top1=[block.p_top1[0].double().cpu().tolist()
                                                         for block in published],
                      published_passes_run=[block.passes_run for block in published])
    return record


def verified_records(drafter, accept_lengths, *, prompt_index):
    """Export measured rounds only, with incoming selection and outgoing acceptance separate."""
    records = []
    for snapshot in drafter.candidate_records:
        r = snapshot['candidate_round']
        if r >= len(accept_lengths):
            continue  # The last verification can leave an unverified future publication.
        row = dict(snapshot)
        if row['record_kind'] == 'redraft':
            records.append(row)
            continue
        current = drafter.round_results[r]
        outcome = drafter.records[r]
        hit = r == 0 or drafter.records[r - 1]['hit']
        k = 0 if r == 0 else drafter.records[r - 1]['k_true']
        anchor = int(current.ids[0, 0])
        served_ks = row['k_set'] if hit else [k]
        row.update(k_true=k, anchor_true=anchor)
        row['verifier'] = dict(seq_id=row['seq_id'], prompt_index=prompt_index,
                               round=r, candidate_round=r, k=k, k_true=k, correction=anchor,
                               guess=anchor, k_set=served_ks, kset_served_size=len(served_ks),
                               k_hit=k in served_ks, hit_at_f=hit, anchor_hit=True,
                               rows_published=row['rows_published'] if hit else 1, oracle_masked_k15=False,
                               candidate_draft_ids=current.ids[0, 1:].cpu().tolist(),
                               decision='hit' if hit else 'miss', candidate_kind='speculative',
                               previous_round_ar=False, gap=0, wait_ms='inf', redraft=not hit,
                               first_set_anchor_hit=hit, anchor_hit_set='first' if hit else 'redraft',
                               accepted_len=accept_lengths[r],
                               accepted_ids=current.ids[0, 1:outcome['k_true'] + 1].cpu().tolist(),
                               next_correction=outcome['correction'])
        if not hit:
            row['verifier'].update(fallback=True, chain_redraft=True, fallback_first_kind='miss')
        records.append(row)
    if sum(row['record_kind'] != 'redraft' for row in records) != len(accept_lengths):
        raise ValueError('capture lacks a verified round')
    return records
