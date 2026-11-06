"""Reduce Table 3 reads with the original EOS-stopped yardstick."""
from statistics import mean
from jetspec.engines.jetspec_plus_plus.launch.eos_stop import compute_eos_stopped, stop_lengths, round_is_miss
import base64
import json
import zlib


def unpack(payload, width=None):
    if payload['schema'] not in ('jetspec.anatomy_lite/1', 'jetspec.anatomy_lite/2') or payload['clock'] != 'CLOCK_MONOTONIC':
        raise ValueError('ANATOMY_LITE_SCHEMA')
    if width is None:
        width = 12 if payload['schema'].endswith('/2') else 11
    rows = json.loads(zlib.decompress(base64.b64decode(payload['rows_zlib'])))
    if len(rows) != payload['rounds'] or any(len(row) != width for row in rows):
        raise ValueError('ANATOMY_LITE_ALIGNMENT')
    return rows



def lite_targets(metrics):
    result = {}
    for payload in metrics.get('anatomy_lite', []):
        for row in unpack(payload):
            key = tuple(row[:2])
            assert key not in result, 'ANATOMY_LITE_DUPLICATE'
            assert row[10] is not None, 'ANATOMY_LITE_GPU_MISSING'
            assert row[2] <= row[5] <= row[6] <= row[9], 'ANATOMY_LITE_CLOCK_ORDER'
            assert 0 <= row[10] <= (row[9]-row[2])/1e6 + .1, 'ANATOMY_LITE_GPU_RANGE'
            result[key] = row
    return result


def percentile(values, q):
    if not values:
        return None
    values = sorted(values)
    pos = (len(values)-1)*q
    lo = int(pos)
    return values[lo] + (values[min(lo+1, len(values)-1)]-values[lo])*(pos-lo)


def target_host_segments(t, following=None):
    """Disjoint step spans; cycle spans join only adjacent rounds of one request."""
    if len(t) == 11 or 'schedule_end' not in t[11]:
        return {}, {}, {}
    s = t[11]
    device = 'native_verify_end' not in s
    verify_start = t[5] if device else s['native_verify_start']
    def span(a,b):
        assert b >= a, 'TARGET_HOST_CLOCK_ORDER'
        return (b-a)/1e6
    step_start = s.get('native_step_start',t[2]) if device else t[2]
    step_end = s.get('native_step_end',s['step_return']) if device else s['step_return']
    pre = dict(scheduler=span(step_start,s['schedule_end']),
        step_bookkeeping=span(s['schedule_end'],s['speculate_start'])+
                         span(s['speculate_end'],verify_start),
        speculate_take=span(s['speculate_start'],t[3]),
        intake_wait=span(t[3],t[4]),
        candidate_decode=span(t[4],s['candidate_decode_end']),
        candidate_select=span(s['candidate_decode_end'],s['speculate_end']))
    if device:
        # Device packet/state work belongs to the unchanged verify bracket.
        # Observe only its caller; never add stamps inside the DSpark verifier.
        post = dict(outcome_dispatch=span(t[6],t[7]),
            outcome_publish=span(t[7],t[8]),
            restore=span(t[8],s['restore_end']),
            postprocess=span(s['restore_end'],s['postprocess_end']),
            step_end=span(s['postprocess_end'],step_end))
        if 'device_verify_start' in s:
            pre['verify_entry'] = span(t[5],s['device_verify_start'])
            post['verify_tail'] = span(s['device_verify_end'],t[6])
        verify = {}
    else:
        post = dict(outcome_record=span(s['native_verify_end'],s['envelope_start']),
        outcome_envelope=span(s['envelope_start'],s['envelope_end']),
        outcome_dispatch=span(s['envelope_end'],t[7]),
        outcome_publish=span(t[7],t[8]),
        restore=span(t[8],s['restore_end']),
        postprocess=span(s['restore_end'],s['postprocess_end']),
        step_end=span(s['postprocess_end'],s['step_return']))
        verify = dict(input_prep=span(s['native_verify_start'],s['plan_start']),
        plan=span(s['plan_start'],s['replay_start']),
        replay_enqueue=span(s['replay_start'],s['logits_end']),
        accept=span(s['logits_end'],s['native_verify_end']))
    cycle = {}
    if following is not None:
        assert following[:2] == [t[0],t[1]+1], 'TARGET_HOST_CYCLE_ALIGNMENT'
        next_step, _, _ = target_host_segments(following)
        next_pre = {name:next_step[name] for name in pre}
        next_start = following[11].get('native_step_start',following[2]) if device else following[2]
        cycle = dict(post, **next_pre, scheduler_loop=span(step_end,next_start))
    return dict(pre, **post), verify, cycle


def target_host_report(rows, steady=False):
    """Same window for per-read and per-cell tables; never fill missing stamps."""
    start = 1 if steady else 0
    result = {}
    bound = min((r.get('target_host_sum_bound', .02) for r in rows), default=.02)
    for field,label in (('target_segment_ms_values','target'),
                        ('verify_segment_ms_values','verify_host'),
                        ('target_cycle_segment_ms_values','target_cycle')):
        names = sorted({n for r in rows for n in r.get(field,{})})
        for name in names:
            values = [v for r in rows for v in r.get(field,{}).get(name,[])[start:] if v is not None]
            result[f'{label}_{name}_mean_ms'] = mean(values) if values else None
            result[f'{label}_{name}_p50_ms'] = percentile(values,.5)
    pairs = [(sum(r['target_segment_ms_values'][n][i] for n in r['target_segment_ms_values']),
              r['round_ms_values'][i]-r['paper_verify_host_ms_values'][i])
             for r in rows if r.get('target_segment_ms_values')
             for i in range(start,len(r['round_ms_values']))]
    if pairs:
        total, expected = map(mean,zip(*pairs))
        error = abs(total-expected)/abs(expected) if expected else abs(total-expected)
        result.update(target_sum_mean_ms=total,target_round_minus_verify_mean_ms=expected,
            target_sum_relative_error=error,target_sum_pass=error<=bound,target_sum_bound=bound,
            target_sum_check=f'{total:.6f} ms / {expected:.6f} ms; error {error:.2%}; '+('PASS' if error<=bound else 'FAIL'),
            target_sum_rounds=len(pairs),
            target_cycle_basis='verify end to next verify start, same request; includes outside-step scheduler/loop; terminal round excluded',
            verify_host_basis='monotonic host brackets; device observes native duration at append, reference accept includes existing GPU waits; do not add GPU time')
    cycles = [(sum(r['target_cycle_segment_ms_values'][n][i] for n in r['target_cycle_segment_ms_values']),v)
              for r in rows for i,v in enumerate(r.get('target_cycle_exposed_ms_values',[]))
              if i>=start and v is not None]
    if cycles:
        total, expected = map(mean,zip(*cycles))
        error = abs(total-expected)/abs(expected) if expected else abs(total-expected)
        result.update(target_cycle_sum_mean_ms=total,target_cycle_round_minus_verify_mean_ms=expected,
            target_cycle_sum_relative_error=error,target_cycle_sum_pass=error<=bound,
            target_cycle_sum_check=f'{total:.6f} ms / {expected:.6f} ms; error {error:.2%}; '+('PASS' if error<=bound else 'FAIL'),
            target_cycle_sum_rounds=len(cycles))
    return result


def per_prompt(method, outputs, metrics, populations, original_indices, tokens, eos, *, chain_lite=None):
    """Exclude each prompt's prefill; retain full cost of the EOS-containing round."""
    source = 'jetspec_round_records' if method == 'jetspecpp' else 'blockhead_round_records'
    records = metrics.get(source, []) if method != 'ar' else []
    targets = lite_targets(metrics)
    target_seqs = list(dict.fromkeys(key[0] for key in targets))
    draft_rows = unpack(chain_lite, width=7) if chain_lite else []
    drafts = {tuple(row[:2]): row for row in draft_rows}
    assert len(drafts)==len(draft_rows), 'ANATOMY_LITE_DRAFT_DUPLICATE'
    assert all(d[2]<=d[3]<=d[4] and d[5]>0 and
               all(value>=0 for value in d[6].values()) for d in draft_rows), 'ANATOMY_LITE_DRAFT_INVALID'
    stops = stop_lengths(outputs, eos, tokens=tokens)
    steps = metrics.get('target_step_times', [])
    cursor, record_cursor = 0, 0
    rows = []
    for index, (output, stop, original_index) in enumerate(zip(outputs, stops, original_indices)):
        selected = [dict(r, prompt_index=0) for r in records if int(r['prompt_index']) == index]
        local = dict(metrics)
        if method == 'ar':
            length = len(output['token_ids'])
            local['target_step_times'] = steps[cursor:cursor+length]
            durations = [1000*float(s) for s in local['target_step_times'][1:stop]]
            verify = durations
            paper_verify = verify
            draft = [0.0]*len(durations)
            raw_accepted = raw_rounds = max(0, length-1)
            credited = [1]*len(durations)
            cursor += length
        else:
            local[source] = selected
            if method == 'jetspecpp':
                local['target_step_times'] = steps[cursor:cursor+len(selected)+1]
                assert len(local['target_step_times']) == len(selected)+1, 'ROUND_TIME_ALIGNMENT'
                all_durations = [1000*float(s) for s in local['target_step_times'][1:]]
                all_verify = metrics['jetspec_verify_round_ms'][record_cursor:record_cursor+len(selected)]
                assert len(all_verify) == len(selected)
                tv = metrics.get('target_verify_times', [])
                all_paper_verify = [1000*float(v) for v in tv[record_cursor:record_cursor+len(selected)]] if tv else []
                assert not tv or len(all_paper_verify)==len(selected), 'PAPER_VERIFY_ALIGNMENT'
                accepted_key = 'accepted_len'
                cursor += len(selected)+1
                record_cursor += len(selected)
            else:
                all_durations = [float(r['round_ms']) for r in selected]
                all_verify = [float(r['verify_ms']) for r in selected]
                all_paper_verify = all_verify
                accepted_key = 'accepted'
            raw_accepted = sum(int(r[accepted_key]) for r in selected)
            raw_rounds = len(selected)
            durations, verify, paper_verify, draft, credited = [], [], [], [], []
            progress = 0
            for pos, record in enumerate(selected):
                accepted = int(record[accepted_key])
                before = progress if method == 'jetspecpp' else int(record['completion_before'])
                progress += accepted
                left = stop-1-before
                if left <= 0:
                    continue
                credited.append(min(accepted, left))
                durations.append(all_durations[pos])
                verify.append(float(all_verify[pos]))
                if all_paper_verify:
                    paper_verify.append(float(all_paper_verify[pos]))
                if method != 'jetspecpp':
                    draft.append(sum(float(record.get(k, 0)) for k in ('head_ms', 'markov_ms', 'domino_ms')))
        seq_ids = {r['seq_id'] for r in selected} if method == 'jetspecpp' else set()
        pops = [p for p in populations if p['seq_id'] in seq_ids] if method == 'jetspecpp' else []
        computed = compute_eos_stopped(method, [output], local, tokens=tokens,
            eos_token_ids=eos, populations=pops, batch=1)
        assert computed['decode_tokens_eos_stopped'] == sum(credited)
        assert computed['rounds_eos_stopped'] == len(durations)
        assert abs(computed['round_ms_eos_stopped']-sum(durations)) < 1e-7
        eos_present = any(t in set(eos) for t in output['token_ids'][:tokens])
        rows.append(dict(original_index=original_index, decode_tokens=sum(credited),
            rounds=len(durations), decode_ms=sum(durations),
            tps=computed['tps_eos_stopped'], tau=computed['tau_eos_stopped'],
            accepted_raw=raw_accepted, raw_rounds=raw_rounds,
            tau_raw=raw_accepted/raw_rounds if raw_rounds else None,
            stop='eos' if eos_present else ('cap' if stop == tokens else 'short_without_eos'),
            stopped_tokens=stop, output_token_ids=output['token_ids'],
            round_ms_values=durations, verify_host_ms_values=verify, draft_host_ms_values=draft,
            paper_verify_host_ms_values=paper_verify,
            cache_hit_values=[r['decision']=='hit' for r in selected[:len(durations)]] if method=='jetspecpp' else [],
            steady_tokens=sum(credited[1:]), steady_rounds=max(0,len(durations)-1),
            steady_ms=sum(durations[1:]),
            hit_rate=computed.get('hit_rate_eos_stopped'),
            cutoff_rate=computed.get('cutoff_rate_eos_stopped'),
            fallback_rounds_eos=sum(bool(r.get('fallback')) for r in selected[:len(durations)])
                if method=='jetspecpp' else 0,
            fallback_rounds_raw=sum(bool(r.get('fallback')) for r in selected)
                if method=='jetspecpp' else 0))
        row = rows[-1]
        row['token_cap'] = tokens
        row['credited_tokens_values'] = credited
        row['fallback_values'] = [bool(r.get('fallback')) for r in selected[:len(durations)]] if method=='jetspecpp' else []
        if method == 'jetspecpp':
            population_index = {(p['seq_id'],p['candidate_round']):p for p in pops}
            hits = [not round_is_miss(r,population_index[(r['seq_id'],r['candidate_round'])])
                    for r in selected[:len(durations)]]
        else:
            hits = []
        row['tau_hit_values'] = hits
        row.update(hit_rounds=sum(hits), miss_rounds=len(hits)-sum(hits),
                   hit_tokens=sum(c for c,h in zip(credited,hits) if h),
                   miss_tokens=sum(c for c,h in zip(credited,hits) if not h))
        if targets:
            # Counters are per-request decode positions, independent of warmups
            # and the engine's speculative candidate-round numbering.
            seq_id = target_seqs[index]
            timed = [targets[(seq_id,pos)] for pos in range(len(durations))]
            if method=='ar':
                row['verify_host_ms_values'] = [(t[6]-t[5])/1e6 for t in timed]
                row['paper_verify_host_ms_values'] = list(row['verify_host_ms_values'])
            row['verify_gpu_ms_values'] = [t[10] for t in timed]
            row['verify_host_minus_gpu_ms_values'] = [v-t[10] for v,t in zip(paper_verify,timed)]
            row['intake_wait_ms_values'] = [(t[4]-t[3])/1e6 for t in timed]
            row['outcome_publish_ms_values'] = [(t[8]-t[7])/1e6 for t in timed]
            joined = [drafts.get((seq_id, int(selected[pos]['round'])+1))
                      for pos in range(len(durations))] if method=='jetspecpp' else []
            row['chain_span_ms_values'] = [d[5] if d else None for d in joined]
            if method=='jetspecpp':
                row['draft_host_ms_values'] = [(d[4]-d[2])/1e6 if d else None for d in joined]
            verify_bounds = [(t[11].get('device_verify_start',t[5]),
                              t[11].get('device_verify_end',t[6])) if len(t) == 12
                             else (t[5],t[6]) for t in timed]
            row['publish_slack_ms_values'] = [(end-d[4])/1e6 if d else None
                                             for (_,end),d in zip(verify_bounds,joined)]
            row['draft_hidden_share_values'] = [
                max(0,min(end,d[4])-max(start,d[2]))/(d[4]-d[2])
                if d and d[4]>d[2] else None for (start,end),d in zip(verify_bounds,joined)]
            names = sorted({name for d in joined if d for name in d[6]})
            row['chain_segment_ms_values'] = {name:[d[6].get(name) if d else None for d in joined]
                                             for name in names}
            segments = [target_host_segments(t,targets.get((seq_id,t[1]+1))) for t in timed]
            device_stamps = any(len(t) == 12 and 'schedule_end' in t[11]
                                and 'native_verify_end' not in t[11] for t in timed)
            if device_stamps:
                row['target_host_sum_bound'] = .005
            row['target_cycle_exposed_ms_values'] = [
                ((targets[(seq_id,t[1]+1)][11].get('device_verify_start',targets[(seq_id,t[1]+1)][5])
                  -t[11].get('device_verify_end',t[6])) if device_stamps else
                 (targets[(seq_id,t[1]+1)][11]['native_verify_start']-t[11]['native_verify_end']))/1e6
                if parts[2] else None for t,parts in zip(timed,segments)]
            for part,field in enumerate(('target_segment_ms_values','verify_segment_ms_values',
                                         'target_cycle_segment_ms_values')):
                names = sorted({name for parts in segments for name in parts[part]})
                row[field] = {name:[parts[part].get(name) for parts in segments] for name in names}
    if method in ('jetspecpp', 'ar'):
        assert cursor == len(steps), 'UNASSIGNED_STEP_TIMES'
    if targets:
        expected = sum(len(o['token_ids'])-1 for o in outputs) if method=='ar' else len(records)
        assert len(targets)==expected, 'ANATOMY_LITE_ROUND_ALIGNMENT'
    return rows


def aggregate(rows):
    tok = sum(r['decode_tokens'] for r in rows)
    duration = sum(r['decode_ms'] for r in rows)
    rounds = sum(r['rounds'] for r in rows)
    total_raw_rounds = sum(r['raw_rounds'] for r in rows)
    steady_rounds = sum(r.get('steady_rounds',0) for r in rows)
    steady_ms = sum(r.get('steady_ms',0) for r in rows)
    prompt_tps = [r['tps'] for r in rows if r['tps'] is not None]
    result = dict(n=len(rows), tps_pooled=1000*tok/duration if duration else None,
        tps_prompt_mean=mean(prompt_tps) if prompt_tps else None,
        tau_eos=tok/rounds if rounds else None,
        tau_raw=sum(r['accepted_raw'] for r in rows)/total_raw_rounds if total_raw_rounds else None,
        decode_tokens=tok, decode_ms=duration, rounds=rounds,
        eos_stopped=sum(r['stop']=='eos' for r in rows), capped=sum(r['stop']=='cap' for r in rows),
        capped_share=sum(r['stop']=='cap' for r in rows)/len(rows),
        short_without_eos=sum(r['stop']=='short_without_eos' for r in rows),
        T_eff_ms=duration/rounds if rounds else None,
        round_mean_ms=duration/rounds if rounds else None,
        T_round_inloop_ms=steady_ms/steady_rounds if steady_rounds else None,
        tps_steady_pooled=1000*sum(r.get('steady_tokens',0) for r in rows)/steady_ms if steady_ms else None)
    for label, key in [('round','round_ms_values'), ('verify_host','verify_host_ms_values'),
                       ('verify','paper_verify_host_ms_values'), ('draft_host','draft_host_ms_values'),
                       ('verify_gpu','verify_gpu_ms_values'), ('intake_wait','intake_wait_ms_values'),
                       ('verify_host_minus_gpu','verify_host_minus_gpu_ms_values'),
                       ('chain_span','chain_span_ms_values'), ('outcome_publish','outcome_publish_ms_values'),
                       ('publish_slack','publish_slack_ms_values')]:
        values = [v for r in rows for v in r.get(key, []) if v is not None]
        if label != 'round':
            result[f'{label}_mean_ms'] = mean(values) if values else None
        result[f'{label}_p50_ms'] = percentile(values,.5)
        result[f'{label}_p90_ms'] = percentile(values,.9)
    result['exposed_beyond_verify_mean_ms'] = (result['round_mean_ms']-result['verify_mean_ms']
        if result['round_mean_ms'] is not None and result['verify_mean_ms'] is not None else None)
    slacks = [v for r in rows for v in r.get('publish_slack_ms_values',[]) if v is not None]
    result['draft_matched_rounds'] = len(slacks)
    result['draft_late_share'] = mean(v<0 for v in slacks) if slacks else None
    hidden = [v for r in rows for v in r.get('draft_hidden_share_values',[]) if v is not None]
    result['draft_hidden_share'] = mean(hidden) if hidden else None
    result['draft_hidden_basis'] = 'command-to-publish host interval overlapping target verify; matched next-round candidate'
    for kind in ('hit','miss'):
        count = sum(r.get(f'{kind}_rounds',0) for r in rows)
        result[f'{kind}_rounds'] = count
        result[f'tau_{kind}'] = sum(r.get(f'{kind}_tokens',0) for r in rows)/count if count else None
    for key in ('hit_rate','cutoff_rate'):
        values = [(r[key],r['rounds']) for r in rows if r.get(key) is not None]
        result[key] = sum(v*n for v,n in values)/sum(n for _,n in values) if values else None
    for key in ('ar_steps_inside_eos', 'ar_steps_after_eos',
                'capacity_fallback_inside_eos', 'capacity_fallback_after_eos',
                'fallback_rounds_eos', 'fallback_rounds_raw'):
        result[key] = sum(r.get(key, 0) for r in rows)
    result['overhead'] = result['T_eff_ms']/result['T_round_inloop_ms'] if steady_rounds else None
    result.update(target_host_report(rows))
    return result
