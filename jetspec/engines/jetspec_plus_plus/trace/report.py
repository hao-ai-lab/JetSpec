"""Measured intervals, exclusive accounting, temporal matching, and trace export."""

# Locate the component from either its category or its stable forwarding path.
from pathlib import Path as _ToolPath
import sys as _tool_sys
_TOOL_ROOT = next((p for p in _ToolPath(__file__).resolve().parents
                   if (p / "artifacts/manifest.json").is_file() or (p / "artifacts/manifest.json").is_file()), _ToolPath.cwd())
_tool_sys.path[:0] = [str(_TOOL_ROOT), str(_TOOL_ROOT / 'engine'), str(_ToolPath(__file__).resolve().parent)]
if __name__ == "__main__" and any(a in ("-h", "--help") for a in _tool_sys.argv[1:]):
    from jetspec.engines.jetspec_plus_plus.serving.entrypoints import cpu_help as _cpu_help
    _cpu_help(__file__)

import argparse
from collections import defaultdict
import csv
import hashlib
from html import escape
import json
from pathlib import Path
from statistics import mean, median


def percentile(values, q):
    values = sorted(values)
    if not values:
        return 0.
    position = (len(values)-1)*q
    lo = int(position)
    return values[lo] + (values[min(lo+1, len(values)-1)]-values[lo])*(position-lo)


def union(intervals):
    merged = []
    for a, b in sorted(intervals):
        if b <= a:
            continue
        if merged and a <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(b, merged[-1][1]))
        else:
            merged.append((a, b))
    return merged


def subtract(a, b, intervals):
    pieces = []
    cursor = a
    for lo, hi in union(intervals):
        lo, hi = max(a, lo), min(b, hi)
        if hi <= lo or hi <= cursor:
            continue
        if lo > cursor:
            pieces.append((cursor, lo))
        cursor = max(cursor, hi)
    if cursor < b:
        pieces.append((cursor, b))
    return pieces


def match_rounds(verify, draft):
    """One-to-one maximum-overlap pairing on the host clock, never rank IDs.

    No overlap means unmatched. This is temporal anatomy, not a claim that the
    concurrent draft packet was consumed by this verifier round.
    """
    candidates = []
    for i, v in enumerate(verify):
        for j, d in enumerate(draft):
            overlap = min(v['stop'], d['stop'])-max(v['start'], d['start'])
            if overlap > 0:
                candidates.append((-overlap, abs(v['start']-d['start']), i, j))
    used_v, used_d, pairs = set(), set(), []
    for _, _, i, j in sorted(candidates):
        if i not in used_v and j not in used_d:
            pairs.append((verify[i], draft[j]))
            used_v.add(i)
            used_d.add(j)
    return sorted(pairs, key=lambda p:p[0]['start']), [r for i,r in enumerate(draft) if i not in used_d]


def exclusive_spans(spans):
    children = defaultdict(list)
    by_id = {(s['rank'],s['id']):s for s in spans}
    if len(by_id) != len(spans):
        raise ValueError('duplicate span id on one rank: the trace directory holds more than one run')
    for row in spans:
        if row.get('parent') is not None:
            parent = by_id.get((row['rank'],row['parent']))
            if parent is not None and not (parent['start']<=row['start']<=row['stop']<=parent['stop']):
                raise ValueError('span lies outside its recorded parent')
            children[(row['rank'], row['parent'])].append((row['start'], row['stop']))
    out = []
    for row in spans:
        nested = children[(row['rank'], row['id'])]
        label = row['label'] + ('.self' if nested else '')
        for a,b in subtract(row['start'], row['stop'], nested):
            out.append(dict(row, label=label, start=a, stop=b))
    return out


def clipped(spans, side, a, b, rank=None):
    return [dict(s, start=max(a, s['start']), stop=min(b, s['stop'])) for s in spans
            if s['side']==side and (rank is None or s['rank']==rank)
            and s['stop']>a and s['start']<b]


def accounting(spans, a, b):
    summed = sum(s['stop']-s['start'] for s in spans)
    covered = sum(hi-lo for lo,hi in union((s['start'],s['stop']) for s in spans))
    return dict(window_ms=(b-a)*1000, named_ms=summed*1000,
                covered_ms=covered*1000, unattributed_ms=max(0.,b-a-covered)*1000,
                overlap_ms=max(0.,summed-covered)*1000)


WAIT_FOR_DRAFT = {'candidate_wait', 'candidate_receive_wait'}
COMPONENT_NAMES = {
    'anchor_decoder': 'anchor decoder', 'anchor_prefix': 'anchor decoder',
    'anchor_window': 'anchor decoder', 'row_selection': 'row selection',
    'row_assembly': 'row assembly', 'accept_decision': 'accept decision',
    'kv_commit': 'KV commit', 'result_publish': 'result publish',
}


def _base_label(label):
    return label[:-5] if label.endswith('.self') else label


def _component_name(label):
    label = _base_label(label)
    if label == 'bucket_pass':
        return 'draft pass'
    if label.startswith('refine_'):
        return 'draft pass ' + label.split('_', 1)[1]
    return COMPONENT_NAMES.get(label, label.replace('_', ' '))


def attributed_lane(spans, a, b, *, device_labels=(), extra=()):
    """Partition one host-clock lane window without overlap or hidden whitespace."""
    device_labels = {_base_label(label) for label in device_labels}
    intervals = []
    for row in spans:
        label = _base_label(row['label'])
        if label in WAIT_FOR_DRAFT:
            category, component = 'wait for draft rows', 'wait for draft rows'
        elif label in device_labels:
            category, component = 'named', _component_name(label)
        else:
            category, component = 'host-only', _component_name(label)
        intervals.append((max(a, row['start']), min(b, row['stop']),
                          category, component, label))
    intervals.extend(extra)
    boundaries = sorted({a, b, *(point for row in intervals for point in row[:2])})
    priority = {'wait for verify result': 0, 'wait for draft rows': 0,
                'named': 1, 'host-only': 2}
    segments = []
    for lo, hi in zip(boundaries, boundaries[1:]):
        if hi <= lo:
            continue
        middle = (lo + hi) / 2
        candidates = [row for row in intervals if row[0] <= middle < row[1]]
        if candidates:
            chosen = min(candidates, key=lambda row: (priority[row[2]], row[1] - row[0], row[4]))
            category, component, label = chosen[2:]
        else:
            category = component = label = 'unattributed'
        if segments and segments[-1]['category'] == category and segments[-1]['component'] == component:
            segments[-1]['stop'] = hi
        else:
            segments.append(dict(start=lo, stop=hi, category=category,
                                 component=component, label=label))
    totals = {name: 0. for name in ('named', 'wait for verify result',
                                     'wait for draft rows', 'host-only', 'unattributed')}
    exported = []
    for row in segments:
        duration = (row['stop'] - row['start']) * 1000
        totals[row['category']] += duration
        exported.append(dict(row, start_ms=(row['start'] - a) * 1000,
                             stop_ms=(row['stop'] - a) * 1000, ms=duration))
        exported[-1].pop('start'); exported[-1].pop('stop')
    summary = dict(window_ms=(b - a) * 1000, named_ms=totals['named'],
                   wait_for_verify_result_ms=totals['wait for verify result'],
                   wait_for_draft_rows_ms=totals['wait for draft rows'],
                   host_only_ms=totals['host-only'],
                   unattributed_ms=totals['unattributed'])
    summary['accounted_ms'] = sum(value for key, value in summary.items()
                                  if key != 'window_ms')
    return summary, exported


def analyze(records):
    rounds = [r for r in records if r['type']=='round']
    raw = [r for r in records if r['type']=='span' and r.get('kind','host')!='cuda']
    device_raw = [r for r in records if r['type']=='span' and 'cuda_ms' in r]
    not_ready_raw = [r for r in records if r['type']=='span'
                     and r.get('cuda_status') == 'NOT_READY']
    for r in rounds+raw:
        if r['stop'] < r['start']:
            raise ValueError('negative interval')
    verify = sorted((r for r in rounds if r['side']=='verify'), key=lambda r:r['start'])
    draft = sorted((r for r in rounds if r['side']=='draft'), key=lambda r:r['start'])
    if not verify:
        raise ValueError('no verify rounds')
    pairs, unmatched = match_rounds(verify, draft)
    spans = exclusive_spans(raw)
    per_component = defaultdict(dict)
    offsets = defaultdict(list)
    lane_samples = defaultdict(list)
    device_per_component = defaultdict(dict)
    not_ready_per_component = defaultdict(int)
    for row in not_ready_raw:
        not_ready_per_component[(row['side'], row['label'], row['cuda_stream'])] += 1
    samples = []
    def lane_attribution(side, v, w):
        a = v['start']
        b = w['stop']
        selected = clipped(spans, side, a, b, w['rank'])
        labels = {row['label'] for row in device_raw
                  if row['side'] == side and row['rank'] == w['rank']
                  and row.get('round') == w['id']}
        extra = []
        if side == 'draft' and w['start'] > a:
            extra.append((a, min(w['start'], b), 'wait for verify result',
                          'wait for verify result', 'wait_for_verify_result'))
        summary, lane_segments = attributed_lane(
            selected, a, b, device_labels=labels, extra=extra)
        legacy = accounting(clipped(spans, side, w['start'], w['stop'], w['rank']),
                            w['start'], w['stop'])
        summary['overlap_ms'] = legacy['overlap_ms']
        return summary, lane_segments

    for side, windows in [('verify', [(v,v) for v in verify]), ('draft', pairs)]:
        for i,(v,w) in enumerate(windows):
            selected = clipped(spans, side, w['start'], w['stop'], w['rank'])
            lane, _ = lane_attribution(side, v, w)
            lane_samples[side].append(lane)
            samples.append(dict(side=side, round=w['id'], verify_round=v['id'], **lane))
            grouped = defaultdict(list)
            for s in sorted(selected, key=lambda s:(s['start'], s['stop'])):
                grouped[s['label']].append(s)
            for label, group in grouped.items():
                # Fragments of a parent's self time count as one invocation.
                per_component[(side,label)][i] = (sum(s['stop']-s['start'] for s in group)*1000,
                                                  len({s['id'] for s in group}))
                for occurrence,s in enumerate(group):
                    offsets[(side,label,occurrence)].append(((s['start']-v['start'])*1000,
                                                            (s['stop']-v['start'])*1000))
            device_grouped = defaultdict(list)
            for row in device_raw:
                if (row['side'] == side and row['rank'] == w['rank']
                        and row.get('round') == w['id'] and 'cuda_ms' in row):
                    device_grouped[(row['label'], row['cuda_stream'])].append(row)
            for (label, stream), group in device_grouped.items():
                device_per_component[(side, label, stream)][i] = (
                    sum(row['cuda_ms'] for row in group), len(group))
    device_components = []
    for side, label, stream in sorted(device_per_component.keys() | not_ready_per_component.keys()):
        observations = device_per_component.get((side, label, stream), {})
        values = [value[0] for value in observations.values()]
        n = len(lane_samples[side])
        device_components.append(dict(
            side=side, component=label, stream=stream,
            median_ms=median(values) if values else None,
            p10_ms=percentile(values, .1) if values else None,
            p90_ms=percentile(values, .9) if values else None,
            calls_per_round=(sum(value[1] for value in observations.values()) / len(values)
                             if values else None),
            rounds=n, present_rounds=len(values),
            not_ready=not_ready_per_component[(side, label, stream)]))
    device_by_component = defaultdict(list)
    for row in device_components:
        device_by_component[(row['side'], row['component'])].append(row)
    components = []
    for (side,label), observations in sorted(per_component.items()):
        n = len(lane_samples[side])
        values = [observations.get(i,(0.,0))[0] for i in range(n)]
        component = dict(side=side, component=label, median_ms=median(values),
                         p10_ms=percentile(values,.1), p90_ms=percentile(values,.9),
                         calls_per_round=sum(v[1] for v in observations.values())/n,
                         rounds=n, present_rounds=len(observations),
                         device_stream=None, device_median_ms=None,
                         device_p10_ms=None, device_p90_ms=None)
        device = device_by_component.get((side, label), [])
        if len(device) == 1:
            component.update(device_stream=device[0]['stream'],
                             device_median_ms=device[0]['median_ms'],
                             device_p10_ms=device[0]['p10_ms'],
                             device_p90_ms=device[0]['p90_ms'])
        elif device:
            component['device_stream'] = 'MULTIPLE_NOT_SUMMED'
        components.append(component)
    timeline = [dict(side=side,label=label,occurrence=occurrence,
                     start_ms=median(v[0] for v in values), stop_ms=median(v[1] for v in values),
                     present_rounds=len(values))
                for (side,label,occurrence),values in sorted(offsets.items())]
    # Explicit end-of-forward -> next-forward window, including inter-step gaps.
    forward = sorted((s for s in raw if s['label']=='verify_forward' and any(
        v['rank']==s['rank'] and v['start']<=s['start']<v['stop'] for v in verify)),
        key=lambda s:s['start'])
    between = []
    for previous, following in zip(forward,forward[1:]):
        a,b = previous['stop'],following['start']
        owners = [v for v in verify if v['start']<=previous['start']<v['stop'] or
                  v['start']<=following['start']<v['stop']]
        if len(owners)==2 and owners[0].get('sequences') and owners[1].get('sequences'):
            if [s[0] for s in owners[0]['sequences']] != [s[0] for s in owners[1]['sequences']]:
                continue
        if b>=a:
            between.append(accounting(clipped(spans,'verify',a,b,previous['rank']),a,b))
    summarize = lambda values: {key:median(v[key] for v in values) for key in values[0]} if values else {}
    dual = device_raw
    pairing = dict(
        rows=len(dual) + len(not_ready_raw), resolved=len(dual),
        not_ready=len(not_ready_raw),
        device_exceeds_host=sum(row['cuda_ms'] > (row['stop'] - row['start']) * 1000
                                for row in dual),
        max_device_minus_host_ms=max(
            [row['cuda_ms'] - (row['stop'] - row['start']) * 1000 for row in dual] or [0.]))
    representative = {}
    if pairs:
        typical = median((v['stop'] - v['start']) for v, _ in pairs)
        rep_v, rep_d = min(pairs, key=lambda pair: abs(
            (pair[0]['stop'] - pair[0]['start']) - typical))
        for side, window in (('verify', rep_v), ('draft', rep_d)):
            summary, lane_segments = lane_attribution(side, rep_v, window)
            representative[side] = dict(summary, round=window['id'], segments=lane_segments)
        publishes = [row for row in representative['verify']['segments']
                     if row['component'] == 'result publish'
                     and row['stop_ms'] <= (rep_d['start'] - rep_v['start']) * 1000 + 1e-9]
        representative['dependency'] = {
            'from_ms': publishes[-1]['stop_ms'] if publishes else 0.,
            'to_ms': max(0., (rep_d['start'] - rep_v['start']) * 1000),
            'label': 'wait for verify result',
        }
        representative['origin_host_s'] = rep_v['start']
    else:
        typical = median((v['stop'] - v['start']) for v in verify)
        rep_v = min(verify, key=lambda v: abs((v['stop'] - v['start']) - typical))
        summary, lane_segments = lane_attribution('verify', rep_v, rep_v)
        representative = {'verify': dict(summary, round=rep_v['id'],
                                         segments=lane_segments),
                          'origin_host_s': rep_v['start']}
    lanes = {side: {key: value for key, value in representative[side].items()
                    if key.endswith('_ms')}
             for side in ('verify', 'draft') if side in representative}
    ranking = []
    for side in ('verify', 'draft'):
        grouped = defaultdict(float)
        for row in representative.get(side, {}).get('segments', []):
            if row['category'] in ('named', 'wait for verify result', 'wait for draft rows'):
                grouped[(row['component'], row['category'])] += row['ms']
        ranking.extend(dict(side=side, component=component, category=category, ms=value)
                       for (component, category), value in grouped.items())
    ranking.sort(key=lambda row: row['ms'], reverse=True)
    span_totals = []
    for (side, label), group in sorted(((key, list(values)) for key, values in
            ((key, (row for row in raw if (row['side'], row['label']) == key))
             for key in {(row['side'], row['label']) for row in raw})), key=lambda item: item[0]):
        values = [(row['stop'] - row['start']) * 1000 for row in group]
        if values:
            span_totals.append(dict(side=side, component=label, median_ms=median(values)))
    return dict(components=components, device_components=device_components,
                device_pairing=pairing,
                lanes=lanes,
                lane_medians={side:summarize(v) for side,v in lane_samples.items()},
                representative=representative, where_round_goes=ranking,
                span_totals=span_totals,
                between_forwards=summarize(between), between_forward_samples=between,
                round_samples=samples, timeline=timeline,
                round_cost_ms=median((v['stop']-v['start'])*1000 for v in verify),
                matching=dict(policy='one-to-one maximum timestamp overlap; not packet identity',
                              verify_rounds=len(verify), draft_rounds=len(draft), paired=len(pairs),
                              unmatched_draft=len(unmatched), unmatched_verify=len(verify)-len(pairs),
                              pairs=[dict(verify_id=v['id'],draft_id=d['id'],verify_start=v['start'],
                                          draft_start=d['start']) for v,d in pairs]),
                cuda=[r for r in records if 'cuda_ms' in r])


def table(result):
    lines = ['side | component | median ms | p10 | p90 | calls/round',
             '--- | --- | ---: | ---: | ---: | ---:']
    for row in result['components']:
        lines.append('{side} | {component} | {median_ms:.4f} | {p10_ms:.4f} | {p90_ms:.4f} | {calls_per_round:.2f}'.format(**row))
    lines.extend(['', 'side | window ms | named ms | wait verify result ms | wait draft rows ms | host-only ms | unattributed ms',
                  '--- | ---: | ---: | ---: | ---: | ---: | ---:'])
    for side,lane in result['lanes'].items():
        lines.append(f"{side} | {lane['window_ms']:.4f} | {lane['named_ms']:.4f} | "
                     f"{lane['wait_for_verify_result_ms']:.4f} | {lane['wait_for_draft_rows_ms']:.4f} | "
                     f"{lane['host_only_ms']:.4f} | {lane['unattributed_ms']:.4f}")
    lines.extend(['', '## Where the round goes', '', 'rank | side | component | category | ms',
                  '---: | --- | --- | --- | ---:'])
    for rank, row in enumerate(result['where_round_goes'], 1):
        lines.append(f"{rank} | {row['side']} | {row['component']} | {row['category']} | {row['ms']:.4f}")
    lines.extend(['', 'side | component | stream | device median ms | p10 | p90 | calls/round | not ready',
                  '--- | --- | --- | ---: | ---: | ---: | ---: | ---:'])
    for row in result['device_components']:
        fmt = lambda value, digits: f'{value:.{digits}f}' if value is not None else 'NOT_MEASURED'
        lines.append(f"{row['side']} | {row['component']} | {row['stream']} | "
                     f"{fmt(row['median_ms'], 4)} | {fmt(row['p10_ms'], 4)} | "
                     f"{fmt(row['p90_ms'], 4)} | {fmt(row['calls_per_round'], 2)} | "
                     f"{row['not_ready']}")
    lines.append('Device/host pairing: ' + json.dumps(result['device_pairing'], sort_keys=True))
    lines.append('Between verify forwards: '+json.dumps(result['between_forwards'],sort_keys=True))
    lines.append('Matching: '+json.dumps(result['matching'],sort_keys=True))
    return '\n'.join(lines)


def write_report(result, directory):
    root = Path(directory)
    root.mkdir(parents=True,exist_ok=True)
    (root/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    (root/'report.md').write_text(table(result)+'\n')
    with (root/'components.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['side','component','median_ms','p10_ms','p90_ms','calls_per_round','rounds','present_rounds','device_stream','device_median_ms','device_p10_ms','device_p90_ms'])
        writer.writeheader();writer.writerows(result['components'])
    with (root/'device_components.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['side','component','stream','median_ms','p10_ms','p90_ms','calls_per_round','rounds','present_rounds','not_ready'])
        writer.writeheader();writer.writerows(result['device_components'])
    events = [dict(ph='M',name='thread_name',pid=1,tid=i,args=dict(name=side))
              for i,side in enumerate(('verify','draft'))]
    for row in result['timeline']:
        events.append(dict(name=row['label'],cat=row['side'],ph='X',pid=1,
                           tid=0 if row['side']=='verify' else 1,
                           ts=row['start_ms']*1000,dur=(row['stop_ms']-row['start_ms'])*1000,
                           args=dict(occurrence=row['occurrence'],present_rounds=row['present_rounds'])))
    events.append(dict(name='median round cost',ph='i',s='g',pid=1,tid=0,ts=result['round_cost_ms']*1000))
    (root/'timeline.json').write_text(json.dumps(dict(traceEvents=events,displayTimeUnit='ms',
        metadata=dict(description='Median offsets, not a single observed execution. Raw clocks remain in rounds JSONL.')))+'\n')
    _svg(result,root/'timeline.svg')


def served_method_summary(result, measured_path, method):
    """Method-local summary; JetSpec++ reports are left in their existing format."""
    if method not in ('ar', 'blockhead', 'dflash', 'jetspec_sync'):
        raise ValueError(f'unsupported served method summary: {method}')
    measured = json.loads(Path(measured_path).read_text())['metrics']['serve_read']['measured']
    records = measured['metrics'].get('blockhead_round_records', [])
    accepted = [row['accepted'] for row in records if row.get('accepted') is not None]
    if method != 'ar' and not accepted:
        raise ValueError(f'{method} has no accepted round lengths for tau')
    samples = result['round_samples']
    verify = [row['window_ms'] for row in samples if row['side'] == 'verify']
    draft = [row['window_ms'] for row in samples if row['side'] == 'draft']
    if method != 'ar' and not draft:
        raise ValueError(f'{method} has no draft round samples')
    return dict(method=method, verify_p50_ms=median(verify),
                draft_p50_ms=median(draft) if draft else None,
                draft_p90_ms=percentile(draft, .9) if draft else None,
                component_medians_ms={f"{row['side']}/{row['component']}": row['median_ms']
                                      for row in result['components']},
                tau=1.0 if method == 'ar' else mean(accepted),
                round_wall_p50_ms=result['round_cost_ms'])


def _svg(result,path):
    rows = result['timeline']
    low=min([0.]+[r['start_ms'] for r in rows])
    high=max([result['round_cost_ms']]+[r['stop_ms'] for r in rows])
    scale=1100/max(high-low,.001)
    x=lambda value:150+(value-low)*scale
    lanes={}
    for side in ('verify','draft'):
        ends=[]; placed=[]
        for r in sorted((r for r in rows if r['side']==side),key=lambda r:r['start_ms']):
            level=next((i for i,end in enumerate(ends) if end<=r['start_ms']),len(ends))
            if level==len(ends):ends.append(r['stop_ms'])
            else:ends[level]=r['stop_ms']
            placed.append((level,r))
        lanes[side]=(max(1,len(ends)),placed)
    height=170+sum(levels*30+45 for levels,_ in lanes.values())+len(result['components'])*18
    svg=[f'<svg xmlns="http://www.w3.org/2000/svg" width="1300" height="{height}" viewBox="0 0 1300 {height}">',
         '<rect width="100%" height="100%" fill="white"/>',
         '<g font-family="sans-serif" font-size="12" fill="#172033">',
         '<text x="20" y="24" font-size="18">Round timeline: median measured offsets</text>',
         '<text x="20" y="44">Host call spans; parent self intervals; empty gaps remain empty. Concurrent tracks stay separate.</text>']
    y=85
    for side,(levels,placed) in lanes.items():
        svg.append(f'<text x="20" y="{y+18}">{side}</text>')
        for level,r in placed:
            colour='#'+hashlib.sha256(r['label'].encode()).hexdigest()[:6]
            left=x(r['start_ms']);width=max(.4,(r['stop_ms']-r['start_ms'])*scale)
            rgb=[int(colour[j:j+2],16) for j in (1,3,5)]
            ink='#111' if .2126*rgb[0]+.7152*rgb[1]+.0722*rgb[2]>150 else 'white'
            top=y+level*30
            title=escape(f"{r['label']} [{r['start_ms']:.4f}, {r['stop_ms']:.4f}] ms; n={r['present_rounds']}")
            svg.append(f'<rect x="{left:.3f}" y="{top}" width="{width:.3f}" height="24" fill="{colour}"><title>{title}</title></rect>')
            if width>48:
                svg.append(f'<text x="{left+4:.3f}" y="{top+16}" fill="{ink}">{r["stop_ms"]-r["start_ms"]:.2f} ms</text>')
        y+=levels*30+45
    marker=x(result['round_cost_ms'])
    svg.append(f'<path d="M {marker:.3f} 64 V {y-15}" stroke="#111" stroke-dasharray="4 3"/>')
    svg.append(f'<text x="{min(marker,1100):.3f}" y="65">round cost {result["round_cost_ms"]:.3f} ms</text>')
    for i in range(11):
        value=low+(high-low)*i/10
        svg.append(f'<text x="{x(value):.3f}" y="{y}">{value:.2f}</text>')
    y+=30
    for row in result['components']:
        colour='#'+hashlib.sha256(row['component'].encode()).hexdigest()[:6]
        svg.append(f'<rect x="20" y="{y-10}" width="10" height="10" fill="{colour}"/><text x="40" y="{y}">{escape(row["side"]+" / "+row["component"])}: {row["median_ms"]:.4f} ms</text>')
        y+=18
    svg.append('</g></svg>')
    path.write_text('\n'.join(svg)+'\n')


def load(directory):
    rows=[]
    for path in sorted(Path(directory).glob('rounds-*.jsonl')):
        rows.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory');parser.add_argument('--compare')
    parser.add_argument('--measured');parser.add_argument('--method')
    args=parser.parse_args()
    result=analyze(load(args.directory))
    if bool(args.measured) != bool(args.method):
        raise ValueError('--measured and --method must be supplied together')
    if args.measured:
        result['served_method'] = served_method_summary(result,args.measured,args.method)
    write_report(result,args.directory)
    print(table(result))
    if args.compare:
        other=analyze(load(args.compare))
        left={(r['side'],r['component']):r for r in result['components']}
        right={(r['side'],r['component']):r for r in other['components']}
        print('\nside | component | delta median ms (compare - first)')
        delta=[]
        for key in sorted(left.keys()|right.keys()):
            value=right.get(key,{}).get('median_ms',0)-left.get(key,{}).get('median_ms',0)
            delta.append(dict(side=key[0],component=key[1],delta_median_ms=value,
                              first_present=key in left,compare_present=key in right))
            print(f'{key[0]} | {key[1]} | {value:+.4f}')
        (Path(args.directory)/'compare.json').write_text(json.dumps(delta,indent=2)+'\n')


if __name__=='__main__':
    main()
