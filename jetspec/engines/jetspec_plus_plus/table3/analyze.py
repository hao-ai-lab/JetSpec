"""Reduce validated saved reads with the repository's EOS-stopped yardstick."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from statistics import mean
import sys

# Adaptation: imports resolve inside the public package.
from jetspec.engines.jetspec_plus_plus.table3.recipe_record import recipe_record
from jetspec.engines.jetspec_plus_plus.table3.eos_stop import compute_eos_stopped
from jetspec.engines.jetspec_plus_plus.table3.eos_stop import stop_lengths
from jetspec.engines.jetspec_plus_plus.table3.eos_stop import round_is_miss
from jetspec.engines.jetspec_plus_plus.table3.protocol import ARMS
from jetspec.engines.jetspec_plus_plus.table3.protocol import HASHES
from jetspec.engines.jetspec_plus_plus.table3.protocol import validate_coverage
from jetspec.engines.jetspec_plus_plus.trace.anatomy_lite import unpack


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
    if method == 'ssd':
        return ssd_per_prompt(outputs,metrics,original_indices,tokens,eos)
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


def ssd_per_prompt(outputs, metrics, indices, tokens, eos):
    """t2kit amendment-4 alignment: first SSD suffix includes prefill recovery."""
    lens = metrics['accepted_suffix_lens_with_recovery']
    steps = metrics['target_step_times']
    verify = metrics['target_verify_times']
    hits = metrics['cache_hits']
    assert len(steps)==len(lens)+len(outputs), 'SSD_STEP_ALIGNMENT'
    assert len(verify)==len(hits)==len(lens), 'SSD_VERIFY_HIT_ALIGNMENT'
    assert [v for v,h in zip(lens,hits) if h]==metrics['accepted_suffix_lens_on_hit'], 'SSD_CACHE_HIT_ALIGNMENT'
    assert [v for v,h in zip(lens,hits) if not h]==metrics['accepted_suffix_lens_on_miss'], 'SSD_CACHE_HIT_ALIGNMENT'
    rows, cursor, prefill_ms = [], 0, []
    for i,(output,stop,index) in enumerate(zip(outputs,stop_lengths(outputs,eos,tokens=tokens),indices)):
        progress, durations, verification, credits, kept_hits = 0, [], [], [], []
        prefill_ms.append(1000*float(steps[cursor+i]))
        raw_accepted, raw_rounds = 0, 0
        while progress < len(output['token_ids']):
            assert cursor < len(lens), 'SSD_SUFFIX_COVERAGE'
            accepted = int(lens[cursor]); assert accepted>0, 'SSD_SUFFIX_WIDTH'
            credit = min(accepted,max(0,stop-progress))
            if raw_rounds==0:
                credit = max(0,credit-1)
            if progress < stop:
                duration = 1000*float(steps[cursor+i+1])
                assert duration>0, 'SSD_ROUND_TIME'
                durations.append(duration); credits.append(credit)
                verification.append(1000*float(verify[cursor])); kept_hits.append(float(hits[cursor]))
            raw_accepted += accepted-(1 if raw_rounds==0 else 0)
            raw_rounds += 1; progress += accepted; cursor += 1
        assert sum(credits)==max(0,stop-1), 'SSD_EOS_CREDIT'
        elapsed, rounds = sum(durations), len(durations)
        rows.append(dict(original_index=index,decode_tokens=sum(credits),rounds=rounds,
            decode_ms=elapsed,tps=1000*sum(credits)/elapsed if elapsed else None,
            tau=sum(credits)/rounds if rounds else None,accepted_raw=raw_accepted,
            raw_rounds=raw_rounds,tau_raw=raw_accepted/raw_rounds,
            stop='eos' if any(t in set(eos) for t in output['token_ids'][:tokens]) else ('cap' if stop==tokens else 'short_without_eos'),
            stopped_tokens=stop,token_cap=tokens,output_token_ids=output['token_ids'],round_ms_values=durations,
            credited_tokens_values=credits,tau_hit_values=[bool(h) for h in kept_hits],
            hit_rounds=sum(bool(h) for h in kept_hits),
            miss_rounds=sum(not h for h in kept_hits),
            hit_tokens=sum(c for c,h in zip(credits,kept_hits) if h),
            miss_tokens=sum(c for c,h in zip(credits,kept_hits) if not h),
            verify_host_ms_values=verification,draft_host_ms_values=[],
            paper_verify_host_ms_values=verification,cache_hit_values=kept_hits,
            steady_tokens=sum(credits[1:]),steady_rounds=max(0,rounds-1),steady_ms=sum(durations[1:]),
            hit_rate=mean(kept_hits) if kept_hits else None,cutoff_rate=None,
            fallback_rounds_eos=0,fallback_rounds_raw=0))
    assert cursor==len(lens), 'SSD_UNASSIGNED_SUFFIXES'
    from statistics import median
    timed = [v for r in rows for v in r['round_ms_values']]
    assert timed and all(v >= 2*median(timed) for v in prefill_ms), 'SSD_PREFILL_SIGNATURE'
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


def paired_ratios(ours, baseline):
    import numpy as np
    ours = sorted(ours,key=lambda r:r['original_index'])
    baseline = sorted(baseline,key=lambda r:r['original_index'])
    assert [r['original_index'] for r in ours] == [r['original_index'] for r in baseline]
    assert len({r['original_index'] for r in ours}) == len(ours)
    draws = np.random.default_rng(0).integers(0,len(ours),size=(10000,len(ours)))
    def sums(rows,key):
        arr=np.array([r[key] for r in rows],dtype=float)
        return arr.sum(),arr[draws].sum(axis=1)
    ot,otb=sums(ours,'decode_tokens');om,omb=sums(ours,'decode_ms');orr,orrb=sums(ours,'rounds')
    bt,btb=sums(baseline,'decode_tokens');bm,bmb=sums(baseline,'decode_ms');br,brb=sums(baseline,'rounds')
    tps_boot=(otb/omb)/(btb/bmb);tau_boot=(otb/orrb)/(btb/brb)
    prompt_mean_boot=np.array([r['tps'] for r in ours])[draws].mean(axis=1)/np.array([r['tps'] for r in baseline])[draws].mean(axis=1)
    return dict(n=len(ours),resamples=10000,seed=0,tps_ratio_pooled=(ot/om)/(bt/bm),
        round_mean_ms=om/orr, dse_round_mean_ms=bm/br,
        serving_gain_G=(bm/br)/(om/orr),
        tps_ratio_prompt_mean=mean(r['tps'] for r in ours)/mean(r['tps'] for r in baseline),
        tau_ratio=(ot/orr)/(bt/br),tps_ratio_ci95=np.quantile(tps_boot,[.025,.975]).tolist(),
        tps_prompt_mean_ratio_ci95=np.quantile(prompt_mean_boot,[.025,.975]).tolist(),
        tau_ratio_ci95=np.quantile(tau_boot,[.025,.975]).tolist())


GB200_BARS = {'8b': 4.82, '27b': 23.67}
DSE_SHA = {'8b': '0470b2be9459a422de83920c19611820ebfa854bcc9c65481d68611ab7ab990c',
           '27b': 'bdf456907b0f3023b69af577a88c428e3384a5cc7141b0ea1c234b58594ccb12'}


def matched_baseline(ours, baseline, *, same_session=False):
    by_index = {r['original_index']: r for r in baseline}
    assert len(by_index) == len(baseline), 'DUPLICATE_DSE_PROMPTS'
    if any(r['original_index'] not in by_index for r in ours):
        return None
    selected = [by_index[r['original_index']] for r in ours]
    if same_session:
        assert all(a['rid'].split('--j-')[0] == b['rid'].split('--j-')[0]
                   for a,b in zip(ours,selected)), 'A79_DSE_SAME_SESSION'
    return selected


def analyze(board):
    from jetspec.engines.jetspec_plus_plus.table3.protocol import validate_read
    from jetspec.engines.jetspec_plus_plus.table3.protocol import sha256
    from jetspec.engines.jetspec_plus_plus.table3.command import sealed_source
    from jetspec.engines.jetspec_plus_plus.table3.rule_a import rule_a_summary
    state = json.loads((board/'progress.json').read_text())
    populations = {}
    receipts = []
    for entry in state['completed']:
        directory = Path(entry['evidence_dir'])
        model, arm, dataset = (entry[k] for k in ('model','arm','dataset'))
        saved = entry['receipt']
        assert sha256(directory/'read.json') == saved['output_sha256'], 'OUTPUT_RECEIPT_SHA'
        assert sha256(directory/'read.json.campaign.json') == saved['campaign_sha256'], 'CAMPAIGN_RECEIPT_SHA'
        receipt = validate_read(directory/'read.json', directory/'read.log', arm, dataset,
            directory/'prompts.json', sealed_source(model,dataset), entry['offset'],
            entry['audit'], model=model)
        assert receipt['original_indices'] == saved['original_indices']
        receipt.update(rid=entry['rid'], source_head=saved['source_head'])
        receipts.append(receipt)
        if entry['audit']:
            continue
        if ARMS[model][arm][0] == 'jetspecpp' and arm not in ('J1R1REF','J1R1REF_F4'):
            assert any(e['model']==model and e['arm']==arm and e['audit']
                       for e in state['completed']), 'AUDIT_REQUIRED'
        raw = json.loads((directory/'read.json').read_text())
        campaign = json.loads((directory/'read.json.campaign.json').read_text())
        measured = campaign['reads'][0]['measured']
        condition = campaign['resolved_condition']
        pops = json.loads((directory/'read.json.populations.json').read_text())['populations']
        rows = per_prompt(ARMS[model][arm][0],raw['outputs'],measured['metrics'],pops,
            receipt['original_indices'],condition['tokens'],condition['eos_token_ids'],
            chain_lite=measured.get('chain',{}).get('anatomy_lite'))
        computed = aggregate(rows)
        receipt.update(eos=computed, rule_a=rule_a_summary(rows))
        native = saved['ssd_eos'] if ARMS[model][arm][0]=='ssd' else measured
        assert abs(computed['tps_pooled']/native.get('tps_eos_stopped',native.get('tps_pooled'))-1) < 1e-10
        assert abs(computed['tau_eos']/native.get('tau_eos_stopped',native.get('tau_eos'))-1) < 1e-10
        for row in rows:
            row.update(model=model,arm=arm,dataset=dataset,rid=entry['rid'],
                source_sha256=receipt['source_sha256'], source_head=saved['source_head'],
                node=receipt['node'], host_load1=receipt['host_load1'],
                receipt=str(directory/'validated-reads.json'))
            events = [e for e in receipt.get('eos_region_events', [])
                      if e['original_index']==row['original_index']]
            for kind in ('ar_steps','capacity_fallback'):
                for region in ('inside_eos','after_eos'):
                    row[f'{kind}_{region}'] = sum(e['kind']==kind and e['region']==region for e in events)
        populations.setdefault((model,dataset,arm), []).extend(rows)
    summaries = {}
    for (model,dataset,arm), rows in populations.items():
        assert len({r['original_index'] for r in rows}) == len(rows), 'DUPLICATE_PROMPTS'
        if len(rows)==128:
            validate_coverage(receipts,arm,dataset,model=model)
        recipe_name = ARMS[model][arm][1]
        recipe = json.loads((recipe_record(recipe_name)).read_text())
        loads = [r['host_load1'] for r in rows if r['host_load1'] is not None]
        planned_n = 32 if model=='8b' and arm in ('A79','DSE_A79','DS','DFlash','SSD','AR_PAPER_GB1','J1R1REF','J1R1REF_F4') else 128
        reference = arm in ('AR','AR_PAPER_GB1','DS','DFlash','SSD','J1R1REF','J1R1REF_F4')
        summary = dict(model=model, recipe=recipe_name, passes=recipe['passes'],
            population='EOS-stopped decode; prefill excluded; full closing round cost',
            planned_n=planned_n, dse_checkpoint_sha256=DSE_SHA[model], venue='GB200',
            baseline_checkpoint_sha256=DSE_SHA[model],engine='reference' if reference else 'fast',
            paper_eligible=reference,
            node_ids=sorted({r['node'] for r in rows if r['node']}),
            host_load1_min=min(loads) if loads else None, host_load1_max=max(loads) if loads else None,
            dataset=dataset,arm=arm,partial=len(rows)!=planned_n, **aggregate(rows))
        summary.update(original_indices=sorted(r['original_index'] for r in rows),
            run_ids=sorted({r['rid'] for r in rows}))
        summary['rule_a'] = rule_a_summary(rows)
        if state.get('reference_engine_ruling') and reference:
            summary['baseline_arm']='DS'
            baseline = populations.get((model,dataset,'DS'))
            baseline = matched_baseline(rows,baseline) if baseline else None
            if baseline:
                ratios = paired_ratios(rows,baseline)
                summary['vs_DS'] = {k.replace('dse_','baseline_'):v for k,v in ratios.items()}
                summary.update(baseline_round_mean_ms=ratios['dse_round_mean_ms'],
                               serving_gain_G=ratios['serving_gain_G'])
            summaries[f'{model}/{dataset}/{arm}'] = summary
            continue
        baseline_arm = 'DSE_A79' if arm in ('A79','DSE_A79') else 'DSE'
        summary['baseline_arm'] = baseline_arm
        baseline = populations.get((model,dataset,baseline_arm))
        if baseline:
            baseline = matched_baseline(rows,baseline,same_session=arm=='A79')
        if baseline:
            ratios = paired_ratios(rows,baseline)
            summary['vs_DSE'] = ratios
            base = aggregate(baseline)
            bar = GB200_BARS[model]
            summary.update(dse_round_mean_ms=ratios['dse_round_mean_ms'],
                serving_gain_G=ratios['serving_gain_G'],
                dse_T_round_inloop_ms=base['T_round_inloop_ms'])
            G = ratios['serving_gain_G']
            theory = ratios['tau_ratio']*G
            summary['yardstick'] = dict(tau_ratio=ratios['tau_ratio'], G=G,
                S_theory=theory, S_actual=ratios['tps_ratio_pooled'],
                eta=ratios['tps_ratio_pooled']/theory,
                historical_verify_G=base['round_mean_ms']/bar,
                historical_S_theory=ratios['tau_ratio']*base['round_mean_ms']/bar,
                historical_eta=ratios['tps_ratio_pooled']/(ratios['tau_ratio']*base['round_mean_ms']/bar),
                hiding=summary['T_round_inloop_ms']/bar if summary['T_round_inloop_ms'] else None,
                hiding_basis='in-loop host round / historical GB200 verify-through-handoff bar',
                overhead=summary['overhead'],verify_bar_ms=bar,
                verify_bar_source='experiment-747' if model=='8b' else 'experiment-762',
                device_verify_this_read=summary['verify_gpu_mean_ms'])
            summary['output_identical_to_DSE'] = sum(a['output_token_ids']==b['output_token_ids']
                for a,b in zip(sorted(rows,key=lambda r:r['original_index']),
                               sorted(baseline,key=lambda r:r['original_index'])))
        summaries[f'{model}/{dataset}/{arm}'] = summary
    payload = dict(inputs=HASHES,summaries=summaries,
        groups_completed=sum(not e['audit'] for e in state['completed']),
        audits_completed=sum(e['audit'] for e in state['completed']),group_receipts=receipts,
        timing_basis='EOS reducer; mean round includes full first/closing round cost; G=DSE mean round/arm mean round; historical GB200 bars separate; physical hidden duration untraced')
    (board/'analysis.json').write_text(json.dumps(payload,indent=2)+'\n')
    (board/'per-prompt.json').write_text(json.dumps([r for rows in populations.values() for r in rows],indent=2)+'\n')
    for model in ('27b','8b'):
        selected = [s for s in summaries.values() if s['model']==model]
        if state.get('reference_engine_ruling'):
            selected = [s for s in selected if s['paper_eligible'] and s['arm']!='AR_PAPER_GB1']
        if not selected:
            continue
        basis = ['model','recipe','passes','population','planned_n','n','dse_checkpoint_sha256',
                 'venue','node_ids','host_load1_min','host_load1_max','partial']
        if state.get('reference_engine_ruling'):
            basis[basis.index('dse_checkpoint_sha256')]='baseline_checkpoint_sha256'
        timing = ['tps_pooled','tps_prompt_mean','tau_eos','tau_raw','round_mean_ms',
                  'dse_round_mean_ms','serving_gain_G','T_round_inloop_ms','dse_T_round_inloop_ms']
        columns = basis+timing+sorted(set().union(*(s.keys() for s in selected))-set(basis+timing))
        name = f't2-{model}-1007_GB200-{model.upper()}-table2-n128.csv'
        with (board/name).open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=columns)
            writer.writeheader()
            writer.writerows({k:json.dumps(v,sort_keys=True) if isinstance(v,(dict,list)) else v
                             for k,v in s.items()} for s in selected)
    return payload


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--board',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(analyze(args.board),indent=2))
