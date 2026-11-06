"""Per-request stop lengths and served-time segments for the Table 1 cells."""


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


# ----------------------------------------------------------------------------- analyze (local)


