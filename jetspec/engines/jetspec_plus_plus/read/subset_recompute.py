"""Recompute served Table 1 cells over a prompt subset (loop-free LCB etc.), read-only.

Two modes:
  collect (run ON a host, stdlib only; eos_stop.py source, minus its __main__ guard, is prepended so
  compute_eos_stopped is in scope):
      { sed '/^if __name__ == "__main__":/,$d' eos_stop.py; cat subset_recompute.py; } |
          timeout 400 ssh host python3 - collect '<complete.json glob>' ... > cells.jsonl
    prints one JSON line per cell: the reference values Table 1 uses, eos_stop's own result, and per-request
    arrays (stop, credited tokens, credited rounds, prefill step s, decode step s) plus loop statistics for every
    output that reaches the cap without EOS.
  analyze (local): python3 subset_recompute.py analyze cells.jsonl
    runs the empty-set gate, classifies cap hits (LOOP vs LONG), and prints full-set vs loop-free tables.

Definitions reproduced (rehearsal_fill.py / paper_t1_fill.py / table1_export.py / eval/serving/reduce/eos_stop.py):
  tokens  = EOS-stopped decode tokens = sum over requests of (stop_i - 1), stop_i = first EOS + 1 within 2048.
  served  = prefill_total_time + decode_total_time of the timed read.
  tau     = tau_eos_stopped (credited tokens / rounds with credited > 0), except SSD whose exported tau is the
            anatomy's to-cap value decode_tokens / (len(target_step_times) - 1).
  speedup = (tokens / served) / (same for AR on the same benchmark and temperature).
Per-request served seconds: the engine accumulates prefill_total_time / decode_total_time from an inner timer and
target_step_times from an outer one (llm_engine.step vs generate), so per-step inner times are not saved. A subset's
served time is P * sum_kept(prefill steps)/sum_all(prefill steps) + D * sum_kept(decode steps)/sum_all(decode steps),
which is exact for the empty exclusion; the raw outer-step sum is reported alongside as a sensitivity check.
"""
import glob
import json
import os
import sys

EOS_DEFAULT = (151643, 151645)
TOKENS = 2048
TABLE1_HEAD = '376fa1a1'
DOMINO_HEAD = '879f3abf'  # Domino cells: dev-test/serving-domino-old = 54af5104 + the Domino port (the 1b643bcc wave is void)


# ----------------------------------------------------------------------------- collect (remote)

def timed_read_path(receipt, campaign):
    out = receipt.get('output') or ''
    m0 = out.replace('.json', '.measured-0.json')
    if out and os.path.exists(m0):
        return m0, 'measured-0'
    reads = [r for r in campaign['reads'] if r.get('included_in_timing')]
    if len(reads) == 1:
        return reads[0]['output'], 'included_in_timing'
    if len(campaign['reads']) == 1:
        return campaign['reads'][0]['output'], 'single-read'
    raise ValueError('no unique timed read')


def stops_of(outputs, eos, tokens):
    stops = []
    for row in outputs:
        ids = row['token_ids'][:tokens]
        first = next((i for i, t in enumerate(ids) if t in eos), None)
        stops.append(len(ids) if first is None else first + 1)
    return stops


def segment(method, outputs, metrics, eos, tokens, cut=None):
    """Per request: prefill step s, decode step s (outer timer), credited tokens, credited rounds, all rounds, and the
    decode step s of the credited rounds only (through the round that crosses the stop).

    cut = {prompt index: onset}: stop that request at its loop onset exactly like an EOS at onset - 1, i.e. keep
    tokens [0, onset); the crossing round is credited min(accepted, left) and later rounds are dropped."""
    n = len(outputs)
    steps = metrics['target_step_times']
    stops = stops_of(outputs, eos, tokens)
    for i, onset in (cut or {}).items():
        stops[i] = min(stops[i], onset)
    pre = [0.0] * n
    dec = [0.0] * n
    decc = [0.0] * n
    cred = [0] * n
    rnd = [0] * n
    allr = [0] * n
    if method == 'ar':
        if len(steps) != sum(len(r['token_ids']) for r in outputs):
            raise ValueError('ar steps misaligned')
        off = 0
        for i, r in enumerate(outputs):
            L = len(r['token_ids'])
            pre[i] = steps[off]
            dec[i] = sum(steps[off + 1:off + L])
            cred[i] = rnd[i] = max(0, stops[i] - 1)
            decc[i] = sum(steps[off + 1:off + stops[i]])
            allr[i] = L - 1
            off += L
        return stops, pre, dec, cred, rnd, allr, decc
    if method in ('blockhead', 'dflash', 'jetspec_sync', 'jetspecpp') or metrics.get('blockhead_round_records'):
        recs = metrics['jetspec_round_records'] if method == 'jetspecpp' else metrics['blockhead_round_records']
        if len(steps) != len(recs) + n:
            raise ValueError('steps != records + n')
        idx = [int(r['prompt_index']) for r in recs]
        if idx != sorted(idx):
            raise ValueError('interleaved records')
        cursor = 0
        pos = 0
        progress = [0] * n
        for i in range(n):
            pre[i] = steps[cursor]
            cursor += 1
            while pos < len(recs) and int(recs[pos]['prompt_index']) == i:
                rec = recs[pos]
                if method != 'jetspecpp' and 'step_index' in rec and int(rec['step_index']) != cursor:
                    raise ValueError(f'step_index {rec["step_index"]} != cursor {cursor}')
                dec[i] += steps[cursor]
                allr[i] += 1
                if method == 'jetspecpp':
                    before = progress[i]
                    acc = int(rec['accepted_len'])
                    progress[i] += acc
                else:
                    before = int(rec['completion_before'])
                    acc = int(rec['accepted'])
                left = stops[i] - 1 - before
                if left > 0:
                    c = min(acc, left)
                    if c:
                        cred[i] += c
                        rnd[i] += 1
                        decc[i] += steps[cursor]
                cursor += 1
                pos += 1
        if pos != len(recs) or cursor != len(steps):
            raise ValueError('records not exhausted')
        return stops, pre, dec, cred, rnd, allr, decc
    # spec_sync / ssd: accepted suffix lengths walked in serial prompt order (eos_stop._spec_sync_records)
    lengths = metrics['accepted_suffix_lens_with_recovery']
    if len(steps) != len(lengths) + n:
        raise ValueError('steps != suffixes + n')
    suffix = 0
    step = 0
    for i, r in enumerate(outputs):
        pre[i] = steps[step]
        step += 1
        progress = 0
        # SSD's suffix lengths sum to >= len(token_ids) per request (its first round re-counts the prefill token);
        # walking to len - 1 (the spec_sync rule) stops one round early and misaligns every later request.
        target = len(r['token_ids']) if method == 'ssd' else max(0, len(r['token_ids']) - 1)
        while progress < target and suffix < len(lengths):
            acc = int(lengths[suffix])
            # SSD's progress already counts the prefill token (see target above).
            left = stops[i] - progress - (0 if method == 'ssd' else 1)
            if left > 0:
                c = min(acc, left)
                cred[i] += c
                rnd[i] += 1
                decc[i] += steps[step]
            dec[i] += steps[step]
            allr[i] += 1
            progress += acc
            suffix += 1
            step += 1
        if progress < target:
            raise ValueError(f'suffixes do not cover output {i}')
    if suffix != len(lengths):
        raise ValueError(f'{len(lengths) - suffix} unassigned suffixes')
    # Diagnostic: a request boundary should land on a prefill step, which is shorter than the median decode step.
    decode_median = sorted(steps)[len(steps) // 2]
    metrics['_boundary_suspect'] = [i for i in range(1, n) if pre[i] > 0.8 * decode_median]
    return stops, pre, dec, cred, rnd, allr, decc


def loop_stats(ids, window=512, max_period=256):
    w = ids[-window:]
    n = len(w)
    best = (0.0, None)
    first = None
    for p in range(1, max_period + 1):
        if n - p <= 0:
            break
        match = sum(1 for i in range(p, n) if w[i] == w[i - p]) / (n - p)
        if match > best[0]:
            best = (match, p)
        if first is None and match >= 0.9:
            first = p
    grams = [tuple(w[i:i + 4]) for i in range(n - 3)]
    distinct4 = len(set(grams)) / len(grams) if grams else None
    # Verbatim recurrence: how many times the final 64-token span occurs (non-overlapping) anywhere in the output.
    last = ids[-64:]
    rep64, i = 0, 0
    while i + 64 <= len(ids):
        if ids[i:i + 64] == last:
            rep64 += 1
            i += 64
        else:
            i += 1
    # Same periodicity statistic on the last 256 tokens (period <= 128).
    w2 = ids[-256:]
    cover256 = max((sum(1 for j in range(p, len(w2)) if w2[j] == w2[j - p]) / (len(w2) - p)
                    for p in range(1, 129)), default=0.0)
    return {'period': first, 'best_cover': best[0], 'best_period': best[1], 'distinct4': distinct4,
            'rep64': rep64, 'cover256': cover256}


def onsets(ids, span=64):
    """Loop onset of a LOOP output (final `span` tokens recur verbatim >= 3 times).
    A: start of the FIRST occurrence of the final span.
    B: smallest p with ids[-span-p:-p] == ids[-span:], then walk exact periodicity back while ids[k] == ids[k + p];
       onset = first index of the exactly p-periodic suffix."""
    last = ids[-span:]
    a = next(i for i in range(len(ids) - span + 1) if ids[i:i + span] == last)
    L = len(ids)
    p = next(p for p in range(1, L - span + 1) if ids[L - span - p:L - p] == last)
    k = L - p - 1
    while k >= 0 and ids[k] == ids[k + p]:
        k -= 1
    return a, k + 1, p


def cell_key(receipt_path, receipt, campaign):
    name = os.path.basename(os.path.dirname(receipt_path))
    rc = campaign['resolved_condition']
    temp = str(int(float(receipt.get('temperature', rc.get('temperature')))))
    if name.startswith('eager-'):
        parts = name.split('-')
        label = {'ar': 'AR', 'dspark': 'DSpark', 'dflash': 'DFlash', 'jetspec': 'JetSpec', 'ssd': 'SSD',
                 'domino': 'Domino'}[parts[1]]
        if label == 'Domino' and not str(receipt.get('backend', '')).startswith(DOMINO_HEAD):
            return None
        return temp, label, int(parts[2][1:]), '-'.join(parts[3:-2]), name
    recipe = rc.get('recipe') or ''
    head = str((receipt.get('source') or {}).get('head', ''))
    if not head.startswith(TABLE1_HEAD):
        return None
    if recipe.startswith('jetspecpp-eager-1gpu-e6v3-'):
        label = 'JetSpec++'
    elif recipe.startswith('jetspecpp-eager-pair-e6v3-'):
        label = 'JetSpec++ pair'
    else:
        return None
    budget = '-b32-' if '-b32-' in name else '-b16-'
    bench = name.split(budget, 1)[1].split(f'-{recipe}', 1)[0]
    return temp, label, int(budget.strip('-b')), bench, name


def collect(globs):
    exports = {}
    for g in globs:
        for cj in glob.glob(os.path.join(os.path.dirname(os.path.dirname(g)), 'cells.jsonl')):
            for line in open(cj):
                if line.startswith('{'):
                    row = json.loads(line)
                    exports[row['receipt']] = row
    for path in sorted(p for g in globs for p in glob.glob(g)):
        if any(s in path for s in ('/prime/', '/filler', '/validation', '/mps/', 'warmup')):
            continue
        rec = {'receipt': path}
        try:
            receipt = json.load(open(path))
            if receipt.get('stage') not in (None, 'cell'):
                continue
            campaign = json.load(open(receipt['campaign']))
            key = cell_key(path, receipt, campaign)
            if key is None:
                continue
            temp, label, budget, bench, name = key
            rc = campaign['resolved_condition']
            method = rc['method']
            eos = set(rc.get('eos_token_ids') or EOS_DEFAULT)
            tokens = int(rc.get('tokens', TOKENS))
            read_path, how = timed_read_path(receipt, campaign)
            raw = json.load(open(read_path))
            m = raw['metrics']
            outputs = raw['outputs']
            rec.update(temp=temp, label=label, budget=budget, bench=bench, cell=name, method=method,
                       read=read_path, read_rule=how, n=len(outputs), batch=raw.get('batch', 1),
                       P=m['prefill_total_time'], D=m['decode_total_time'], steps_sum=sum(m['target_step_times']))
            # Reference values exactly as the Table 1 fill uses them.
            if label.startswith('JetSpec++'):
                ex = exports.get(path)
                if ex:
                    rec['ref'] = {'tokens': ex['eos_tokens'], 'served': ex['prefill_plus_decode_s'], 'tau': ex['tau_eos']}
                    rec['ref_source'] = 'cells.jsonl'
                else:  # export not refreshed yet: rebuild the row the way table1_export.py does (read-only)
                    eos_rows = json.load(open(receipt['eos']))
                    ev = next(r for r in eos_rows if os.path.basename(r['read']) == os.path.basename(read_path))
                    rec['ref'] = {'tokens': ev['decode_tokens_eos_stopped'], 'served': m['prefill_total_time'] +
                                  m['decode_total_time'], 'tau': ev['tau_eos_stopped']}
                    rec['ref_source'] = 'eos.json (no cells.jsonl row)'
            else:
                out = json.load(open(receipt['output'])) if receipt['output'] != read_path else raw
                last = campaign['reads'][-1]
                block = last.get('measured') or last.get('summary')
                rec['ref'] = {'tokens': block.get('decode_tokens_eos_stopped') or block.get('decode_tokens'),
                              'served': out['metrics']['prefill_total_time'] + out['metrics']['decode_total_time'],
                              'tau': receipt['metrics']['tau']}
            # eos_stop's own computation (eval/serving/reduce/eos_stop.py prepended by the driver), where it supports the method.
            try:
                pops_path = read_path + '.populations.json'
                pops = json.load(open(pops_path))['populations'] if os.path.exists(pops_path) else []
                es = compute_eos_stopped(method, outputs, m, tokens=tokens, eos_token_ids=sorted(eos),  # noqa: F821
                                         populations=pops, batch=int(raw.get('batch', 1)))
                rec['eos_stop'] = {'tokens': es['decode_tokens_eos_stopped'], 'tau': es['tau_eos_stopped'],
                                   'rounds': es['rounds_eos_stopped']}
            except Exception as error:  # ssd is not an eos_stop method
                rec['eos_stop'] = {'error': str(error)}
            stops, pre, dec, cred, rnd, allr, _ = segment(method, outputs, m, eos, tokens)
            rec.update(stop=stops, len=[len(r['token_ids']) for r in outputs], pre=pre, dec=dec, cred=cred, rnd=rnd,
                       allr=allr, nsteps=len(m['target_step_times']), boundary_suspect=m.get('_boundary_suspect'))
            caps = []
            for i, r in enumerate(outputs):
                ids = r['token_ids'][:tokens]
                if len(ids) >= tokens - 8 and not any(t in eos for t in ids):
                    st = loop_stats(ids)
                    st.update(i=i, tail=(r.get('text') or '')[-1200:])
                    if st['rep64'] >= 3:
                        a, b, p = onsets(ids)
                        st.update(onsetA=a, onsetB=b, periodB=p, winA=ids[max(0, a - 100):a + 100], winB=ids[max(0, b - 100):b + 100])
                    caps.append(st)
            rec['caps'] = caps
            # Loop-cut views: every LOOP output of this row cut at its own onset (rule A and rule B).
            for rule in ('A', 'B'):
                cut = {cap['i']: cap['onset' + rule] for cap in caps if 'onset' + rule in cap}
                if not cut:
                    continue
                c_stops, _, _, c_cred, c_rnd, _, c_decc = segment(method, outputs, m, eos, tokens, cut=cut)
                for cap in caps:
                    if cap['i'] in cut:
                        i = cap['i']
                        cap['cut' + rule] = {'stop': c_stops[i], 'cred': c_cred[i], 'rnd': c_rnd[i], 'dec': c_decc[i]}
        except Exception as error:
            rec['error'] = f'{type(error).__name__}: {error}'
        print(json.dumps(rec))


# ----------------------------------------------------------------------------- analyze (local)

def subset(c, excluded):
    keep = [i for i in range(c['n']) if i not in excluded]
    pre_all, dec_all = sum(c['pre']), sum(c['dec'])
    pre_k, dec_k = sum(c['pre'][i] for i in keep), sum(c['dec'][i] for i in keep)
    served = c['P'] * (pre_k / pre_all) + c['D'] * (dec_k / dec_all)
    raw = pre_k + dec_k
    if c['label'] == 'SSD':
        tokens = sum(c['len'][i] - 1 for i in keep)
        tau = tokens / sum(c['allr'][i] for i in keep)   # per-round, as view()
    else:
        tokens = sum(c['stop'][i] - 1 for i in keep)
        rounds = sum(c['rnd'][i] for i in keep)
        cred = sum(c['cred'][i] for i in keep)
        tau = 1.0 if c['label'] == 'AR' else cred / rounds
    return {'n': len(keep), 'tokens': tokens, 'served': served, 'raw': raw, 'tau': tau}


def rel(a, b):
    return abs(a - b) / max(abs(b), 1e-300)


def is_loop(cap):
    # LOOP: the final 64 output tokens recur verbatim (non-overlapping) at least 3 times in the capped output.
    return cap['rep64'] >= 3


def is_loop_periodic(cap):
    # Alternative rule, reported for robustness: >= 90 % exact period-p self-match (p <= 256) in the last 512 tokens.
    return cap['period'] is not None


def analyze(path):
    cells = [json.loads(line) for line in open(path) if line.strip()]
    errors = [c for c in cells if 'error' in c]
    cells = [c for c in cells if 'error' not in c]
    table = {}
    for c in cells:
        k = (c['temp'], c['label'], c['budget'], c['bench'])
        if k in table:
            print('DUPLICATE', k, table[k]['receipt'], c['receipt'])
        table[k] = c
    print(f'cells {len(cells)}, collect errors {len(errors)}')
    for e in errors:
        print('  ERROR', e['receipt'], e['error'])

    # ---- gate
    print('\n== GATE (empty excluded set) ==')
    worst = {'tokens': 0, 'served': 0, 'tau': 0, 'eos_tokens': 0, 'eos_tau': 0, 'partition': 0}
    fails = []
    for k, c in sorted(table.items()):
        s = subset(c, set())
        v0 = view(c)  # the loop-cut code path with nothing cut must be the full set too
        if (v0['tokens'], v0['served'], v0['tau']) != (s['tokens'], s['served'], s['tau']):
            fails.append((k, 'view() with no cut differs from full set'))
        ref = c.get('ref')
        if not ref:
            fails.append((k, 'no reference export'))
            continue
        d = {'tokens': rel(s['tokens'], ref['tokens']), 'served': rel(s['served'], ref['served']),
             'tau': rel(s['tau'], ref['tau']), 'partition': rel(s['raw'], c['steps_sum'])}
        es = c.get('eos_stop') or {}
        if 'tokens' in es:
            d['eos_tokens'] = rel(s['tokens'], es['tokens'])
            if c['label'] != 'AR':
                d['eos_tau'] = rel(s['tau'], es['tau'])
        for name, v in d.items():
            worst[name] = max(worst[name], v)
            if v > 1e-9:
                fails.append((k, name, v))
    print('max relative error:', {k: f'{v:.1e}' for k, v in worst.items()})
    print('GATE', 'PASS' if not fails else 'FAIL', f'over {len(table)} cells')
    for f in fails[:40]:
        print('  FAIL', f)

    # ---- loop classification
    print('\n== CAP HITS: LOOP vs LONG ==')
    print('rule: LOOP iff the final 64 tokens of a capped (no-EOS) output recur verbatim >= 3 times in that output; '
          'else LONG')
    alt = {}
    for k, c in table.items():
        for cap in c['caps']:
            if is_loop_periodic(cap):
                alt.setdefault((k[0], k[3]), set()).add(cap['i'])
    benches = sorted({k[3] for k in table})
    loops, longs, evidence = {}, {}, {'LOOP': [], 'LONG': []}
    covers = []
    for k, c in sorted(table.items()):
        for cap in c['caps']:
            kind = 'LOOP' if is_loop(cap) else 'LONG'
            covers.append((cap['best_cover'], cap['distinct4'], kind, k, cap['i']))
            (loops if kind == 'LOOP' else longs).setdefault((k[0], k[3]), set()).add(cap['i'])
            evidence[kind].append((k, cap))
    # distribution of the statistic, to show the gap
    hist = {}
    for cov, d4, kind, *_ in covers:
        b = min(int(cov * 10), 9)
        hist[b] = hist.get(b, 0) + 1
    print('best periodic cover histogram (bin = floor(10*cover)):', dict(sorted(hist.items())))
    reps = {}
    for k, c in table.items():
        for cap in c['caps']:
            reps[cap['rep64']] = reps.get(cap['rep64'], 0) + 1
    print('rep64 histogram (recurrences of the final 64 tokens):', dict(sorted(reps.items())))
    near = sorted(cv for cv in covers if 0.6 <= cv[0] < 0.95)
    print('cap hits with best cover in [0.6, 0.95) (cover, distinct4, class, T, row, budget, bench, prompt):')
    for a, b, kd, kk, i in near:
        print(f'  {a:.3f} {b:.3f} {kd} T{kk[0]} {kk[1]} b{kk[2]} {kk[3]} #{i}')
    for t in ('0', '1'):
        for b in benches:
            lp, lg = sorted(loops.get((t, b), set())), sorted(longs.get((t, b), set()) - loops.get((t, b), set()))
            if lp or lg:
                print(f'T{t} {b:9s} LOOP union ({len(lp)}): {lp}')
                print(f'T{t} {b:9s} LONG union, not LOOP anywhere ({len(lg)}): {lg}')
                ap = sorted(alt.get((t, b), set()))
                if ap != lp:
                    print(f'T{t} {b:9s}   periodic-rule LOOP union differs: {ap}')
    # per-row loop counts on lcb
    print('\nper-row LOOP / LONG cap hits on LCB:')
    for k, c in sorted(table.items()):
        if k[3] == 'lcb':
            lp = [cap['i'] for cap in c['caps'] if is_loop(cap)]
            lg = [cap['i'] for cap in c['caps'] if not is_loop(cap)]
            print(f'  T{k[0]} {k[1]:15s} b{k[2]:<3d} LOOP {lp}  LONG {lg}')

    result = {}
    # ---- onsets
    print('\n== LOOP ONSETS ==')
    print('A = start of the first occurrence of the final 64-token span; B = start of the exactly p-periodic suffix '
          '(smallest p whose shift reproduces the final 64 tokens, walked back while tok[k] == tok[k+p])')
    diffs = []
    for k, c in sorted(table.items()):
        for cap in c['caps']:
            if 'onsetA' in cap:
                diffs.append((cap['onsetB'] - cap['onsetA'], k, cap))
    same = sum(1 for d, *_ in diffs if d == 0)
    ds = sorted(d for d, *_ in diffs)
    print(f'loop outputs {len(diffs)}: A == B {same}; B - A (tokens) min {ds[0]} median {ds[len(ds) // 2]} max {ds[-1]}; '
          f'B later than A in {sum(d > 0 for d in ds)}, earlier in {sum(d < 0 for d in ds)}')
    print('  |B - A| > 64:', [(k[0], k[3], k[1], k[2], cap['i'], cap['onsetA'], cap['onsetB'], cap['periodB'])
                              for d, k, cap in diffs if abs(d) > 64])

    rows = [('AR', 0), ('DFlash', 16), ('DFlash', 32), ('JetSpec', 16), ('JetSpec', 32), ('DSpark', 16),
            ('DSpark', 32), ('SSD', 4), ('JetSpec++', 16), ('JetSpec++', 32), ('JetSpec++ pair', 16),
            ('JetSpec++ pair', 32)]
    views = ('full', 'drop', 'cutA', 'cutB')
    bad = 0
    for c in table.values():
        for cap in c['caps']:
            for rule in 'AB':
                cut = cap.get('cut' + rule)
                if cut and c['label'] not in ('AR', 'SSD') and cut['cred'] != cut['stop'] - 1:
                    bad += 1
    print(f'cut credit check (credited tokens == onset - 1 on every cut output): {"PASS" if not bad else f"FAIL {bad}"}')
    for (t, b), excl in sorted(loops.items()):
        base = table.get((t, 'AR', 0, b))
        if not base:
            continue
        ar = {v: view(base, excl if v == 'drop' else set(), v[3] if v.startswith('cut') else None) for v in views}
        print(f'\n== {b} T{t}: speedup× (tau) · full / loop-drop (drop {sorted(excl)}) / loop-cut A / loop-cut B ==')
        print('| row | full | loop-drop | loop-cut A | loop-cut B | tokens cut A (outputs) | tokens cut B |')
        print('|---|---|---|---|---|---|---|')
        for label, budget in rows:
            c = table.get((t, label, budget, b))
            if not c:
                print(f'| {label} b{budget} | pending | | | | | |')
                continue
            cells = []
            for v in views:
                x = view(c, excl if v == 'drop' else set(), v[3] if v.startswith('cut') else None)
                sp = (x['tokens'] / x['served']) / (ar[v]['tokens'] / ar[v]['served'])
                result[(t, b, label, budget, v)] = sp
                cells.append(f'{sp:.2f}× ({x["tau"]:.2f})')
            xa, xb = view(c, set(), 'A'), view(c, set(), 'B')
            print(f'| {label} b{budget} | ' + ' | '.join(cells) +
                  f' | {xa["removed"]} ({xa["ncut"]}) | {xb["removed"]} |')
        print(f'| AR tok/s | ' + ' | '.join(f'{ar[v]["tokens"] / ar[v]["served"]:.2f}' for v in views) + ' | | |')
        print('JetSpec++ vs DSpark-15, same budget (full / drop / cut A / cut B):')
        for label in ('JetSpec++', 'JetSpec++ pair'):
            for budget in (16, 32):
                vals = [(result.get((t, b, label, budget, v)), result.get((t, b, 'DSpark', budget, v))) for v in views]
                if all(a and d for a, d in vals):
                    print(f'  {"MPS" if label == "JetSpec++" else "2 GPUs"} b{budget}: ' +
                          ' / '.join(f'{100 * (a / d - 1):+.1f} %' for a, d in vals))
                else:
                    print(f'  {label} b{budget}: pending')
    print('\nBenchmarks/temperatures not listed have no LOOP outputs: all four views equal the full set.')


def view(c, excluded=frozenset(), rule=None):
    """One cell under a view: drop `excluded` prompts, and/or cut every LOOP output at its onset (rule 'A' or 'B')."""
    over = {cap['i']: cap['cut' + rule] for cap in c['caps'] if rule and ('cut' + rule) in cap}
    keep = [i for i in range(c['n']) if i not in excluded]
    pre_all, dec_all = sum(c['pre']), sum(c['dec'])
    pre_k = sum(c['pre'][i] for i in keep)
    dec_k = sum(over[i]['dec'] if i in over else c['dec'][i] for i in keep)
    served = c['P'] * (pre_k / pre_all) + c['D'] * (dec_k / dec_all)
    removed = sum(c['stop'][i] - over[i]['stop'] for i in keep if i in over)
    if c['label'] == 'SSD':
        tokens = sum(over[i]['stop'] - 1 if i in over else c['len'][i] - 1 for i in keep)
        # Per-round SSD tau (Aaron, 09-25 ~19:20 PDT): decode rounds exclude the one prefill step per request, the
        # convention every other row uses. SSD's anatomy export divides by (steps - 1), i.e. R + n - 1.
        rounds = sum(over[i]['rnd'] if i in over else c['allr'][i] for i in keep)
        tau = tokens / rounds
    else:
        tokens = sum((over[i]['stop'] if i in over else c['stop'][i]) - 1 for i in keep)
        rounds = sum(over[i]['rnd'] if i in over else c['rnd'][i] for i in keep)
        cred = sum(over[i]['cred'] if i in over else c['cred'][i] for i in keep)
        tau = 1.0 if c['label'] == 'AR' else cred / rounds
    return {'n': len(keep), 'tokens': tokens, 'served': served, 'tau': tau, 'removed': removed, 'ncut': len(over)}


def examples(path, picks):
    """Decode 100 tokens either side of each onset (needs the Qwen3-8B tokenizer locally)."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained('Qwen/Qwen3-8B', local_files_only=True)
    for line in open(path):
        c = json.loads(line)
        for cap in c.get('caps', []):
            key = (c.get('temp'), c.get('bench'), c.get('label'), c.get('budget'), cap['i'])
            if key in picks and 'onsetA' in cap:
                print(f'===== T{key[0]} {key[1]} {key[2]} b{key[3]} #{key[4]}: onset A {cap["onsetA"]}, '
                      f'onset B {cap["onsetB"]} (period {cap["periodB"]})')
                for rule in 'AB':
                    w, o = cap['win' + rule], cap['onset' + rule]
                    cut = min(100, o)
                    print(f'--- rule {rule}: BEFORE |CUT| AFTER')
                    print(repr(tok.decode(w[:cut])) + '\n   |CUT|\n' + repr(tok.decode(w[cut:])))


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'collect':
        collect(sys.argv[2:])
    elif len(sys.argv) > 1 and sys.argv[1] == 'analyze':
        analyze(sys.argv[2])
    elif len(sys.argv) > 1 and sys.argv[1] == 'examples':
        examples(sys.argv[2], {tuple(json.loads(x)) for x in sys.argv[3:]})
    else:
        print(__doc__)
