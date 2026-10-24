"""Analysis-only paper repetition cut, matching Table 1 rule A."""
from copy import deepcopy


def repetition_onset(ids, *, span=64):
    """Count non-overlapping final-span matches, as subset_recompute.loop_stats.

    Return the first occurrence only when there are at least three matches.
    The caller restricts classification to capped outputs without EOS.
    """
    if len(ids) < span:
        return None, 0
    last = ids[-span:]
    matches, pos = [], 0
    while pos + span <= len(ids):
        if ids[pos:pos+span] == last:
            matches.append(pos)
            pos += span
        else:
            pos += 1
    return (matches[0] if len(matches) >= 3 else None), len(matches)


def rule_a(outputs, rows, *, tokens=2048):
    """Cut aligned EOS per-prompt records, retaining the closing round's cost.

    Prefill's token receives no decode credit. Tau uses all retained rounds;
    paper tok/s and round anatomy exclude each prompt's first decode round.
    Raw EOS records and outputs are left untouched.
    """
    from jetspec.engines.jetspec_plus_plus.table3.analyze import aggregate
    from jetspec.engines.jetspec_plus_plus.table3.analyze import percentile
    assert len(outputs) == len(rows), 'RULE_A_PROMPT_ALIGNMENT'
    cut_rows, cut_outputs, capped = [], [], []
    for output, source in zip(outputs, rows):
        row = deepcopy(source)
        ids = output['token_ids'][:tokens]
        assert row['output_token_ids'] == output['token_ids'], 'RULE_A_OUTPUT_ALIGNMENT'
        onset, occurrences = repetition_onset(ids) if source['stop']=='cap' else (None, 0)
        if source['stop']=='cap':
            capped.append(dict(original_index=row['original_index'], onset=onset,
                               loop_found=onset is not None, occurrences=occurrences))
        credits = row['credited_tokens_values']
        assert len(credits)==row['rounds']==len(row['round_ms_values']), 'RULE_A_ROUND_ALIGNMENT'
        assert sum(credits)==row['decode_tokens'], 'RULE_A_CREDIT_ALIGNMENT'
        stop = source['stopped_tokens'] if onset is None else onset
        remaining = max(0, stop-1)
        kept = []
        for credit in credits:
            if remaining <= 0:
                break
            kept.append(min(credit, remaining))
            remaining -= kept[-1]
        assert remaining==0, 'RULE_A_ROUND_COVERAGE'
        count = len(kept)
        for key, value in row.items():
            if key.endswith('_values') and isinstance(value, list):
                assert not value or len(value)==len(credits), 'RULE_A_ANATOMY_ALIGNMENT'
                row[key] = value[:count]
        row['chain_segment_ms_values'] = {k:v[:count] for k,v in row.get('chain_segment_ms_values',{}).items()}
        row['credited_tokens_values'] = kept
        row.update(decode_tokens=sum(kept), rounds=count,
                   decode_ms=sum(row['round_ms_values']), stopped_tokens=stop,
                   output_token_ids=output['token_ids'][:stop],
                   rule_a_onset=onset, rule_a_loop_found=onset is not None,
                   steady_tokens=sum(kept[1:]), steady_rounds=max(0,count-1),
                   steady_ms=sum(row['round_ms_values'][1:]))
        row['tps'] = 1000*row['decode_tokens']/row['decode_ms'] if row['decode_ms'] else None
        row['tau'] = row['decode_tokens']/count if count else None
        split = row.get('tau_hit_values', [])
        for kind, hit in [('hit',True), ('miss',False)]:
            row[f'{kind}_rounds'] = sum(h==hit for h in split)
            row[f'{kind}_tokens'] = sum(c for c,h in zip(kept,split) if h==hit)
        cache = row.get('cache_hit_values', [])
        row['hit_rate'] = sum(cache)/len(cache) if cache else None
        row['fallback_rounds_eos'] = sum(row.get('fallback_values',[]))
        cut_rows.append(row)
        # Text cannot be reconstructed without a tokenizer; return tokens only.
        cut_outputs.append(dict(token_ids=row['output_token_ids']))
    stats = aggregate(cut_rows)
    steady = [v for r in cut_rows for v in r['round_ms_values'][1:]]
    verify = [v for r in cut_rows for v in r.get('paper_verify_host_ms_values',r['verify_host_ms_values'])[1:] if v is not None]
    cache = [v for r in cut_rows for v in r.get('cache_hit_values',[])[1:]]
    return dict(outputs=cut_outputs, rows=cut_rows, tau=stats['tau_eos'],
        tok_s=stats['tps_steady_pooled'], stats=stats, capped_prompts=capped,
        loop_count=sum(p['loop_found'] for p in capped),
        round_p50_ms=percentile(steady,.5),
        round_mean_ms=sum(steady)/len(steady) if steady else None,
        verify_p50_ms=percentile(verify,.5),
        verify_mean_ms=sum(verify)/len(verify) if verify else None,
        cache_hit=sum(cache)/len(cache) if cache else None)


def rule_a_summary(rows):
    """Small side-by-side report for kit analysis and future 8B/27B reads."""
    result = rule_a([dict(token_ids=r['output_token_ids']) for r in rows], rows,
                    tokens=rows[0]['token_cap'])
    return {k:v for k,v in result.items() if k not in ('rows','outputs')}
