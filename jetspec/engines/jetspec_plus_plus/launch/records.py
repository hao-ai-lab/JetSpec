"""Structured copies of the markers still printed by legacy serving reads."""
import json
import math
from pathlib import Path
from statistics import mean, median

from jetspec.engines.jetspec_plus_plus.runtime.launch_env import environment_name

MARKERS = ('CHAIN_ROUND', 'CHAIN_FALLBACK', 'EVAL_ROUND', 'RUN_MANIFEST',
           'SERVE_READ_PHASE_START', 'SERVE_READ_PHASE_END')


def read_fields(condition, stdout, config_read=None):
    fields = {marker: [] for marker in MARKERS}
    fields['RESOLVED_CONDITION'] = condition
    for line in stdout.splitlines():
        for marker in MARKERS:
            position = line.find(marker + ' ')
            if position < 0:
                continue
            payload = line[position + len(marker) + 1:]
            try:
                value, _ = json.JSONDecoder().raw_decode(payload)
            except ValueError:
                value = payload
            fields[marker].append(value)
            break
    if config_read is not None:
        fields.update(config=config_read['config'], canonical_json=config_read['canonical_json'],
                      config_sha256=config_read['config_sha256'])
    return fields


ROUND_METHODS = {"blockhead", "dflash", "jetspecpp", "jetspec_sync", "spec_sync"}


def round_is_miss(record, population):
    """A miss can emit a correction without taking the redraft fallback path."""
    return bool(population.get("fallback") or record.get("decision") == "miss")


def _need(method, obj, fields, source):
    missing = [name for name in fields if name not in obj or obj[name] is None]
    if missing:
        raise ValueError(f"{method}: missing {source}: {', '.join(missing)}")


def stop_lengths(outputs, eos_token_ids, *, tokens):
    """Count tokens through the first target EOS, subject to the served cap."""
    if not eos_token_ids:
        raise ValueError("missing target_eos_token_ids (use --eos-token-ids for old campaigns)")
    if not isinstance(tokens, int) or tokens < 1:
        raise ValueError("missing positive token budget")
    eos = set(map(int, eos_token_ids))
    stops = []
    for index, row in enumerate(outputs):
        _need("outputs", row, ("token_ids",), f"outputs[{index}]")
        ids = row["token_ids"][:tokens]
        first = next((i for i, token in enumerate(ids) if token in eos), None)
        stops.append(len(ids) if first is None else first + 1)
    return stops


def _spec_sync_records(outputs, metrics, *, batch):
    """Align stock synchronous suffix lengths and step times in serial prompt order."""
    method = "spec_sync"
    _need(method, metrics, ("accepted_suffix_lens_with_recovery", "target_step_times"),
          "metrics")
    lengths = metrics["accepted_suffix_lens_with_recovery"]
    steps = metrics["target_step_times"]
    if batch != 1 or len(steps) != len(lengths) + len(outputs):
        raise ValueError(f"{method}: target_step_times cannot be aligned to accepted suffixes")
    records = []
    suffix = 0
    step = 0
    for index, output in enumerate(outputs):
        step += 1  # prefill
        progress = 0
        target = max(0, len(output["token_ids"]) - 1)
        while progress < target and suffix < len(lengths):
            accepted = int(lengths[suffix])
            records.append({"prompt_index": index, "completion_before": progress,
                            "accepted": accepted, "round_ms": float(steps[step]) * 1000})
            progress += accepted
            suffix += 1
            step += 1
        if progress < target:
            raise ValueError(f"{method}: accepted suffixes do not cover output {index}")
    if suffix != len(lengths):
        raise ValueError(f"{method}: unassigned accepted suffixes after final output")
    return records


def compute_eos_stopped(method, outputs, metrics, *, tokens, eos_token_ids,
                        populations=(), batch=1, prompt_indices=None):
    """Count decode tokens and complete rounds up to EOS for one saved read.

    The first output token is produced by prefill. It is in the EOS search but
    excluded from decode tokens and round timing, matching the served receipt.
    """
    if method not in ROUND_METHODS | {"ar"}:
        raise ValueError(f"{method}: unsupported EOS-stopped method")
    stops = stop_lengths(outputs, eos_token_ids, tokens=tokens)
    native_indices = (list(range(len(outputs))) if prompt_indices is None
                      else list(map(int, prompt_indices)))
    if len(native_indices) != len(outputs) or len(set(native_indices)) != len(outputs):
        raise ValueError(f"{method}: prompt indices must identify each output once")
    positions = {index: position for position, index in enumerate(native_indices)}
    kept = []  # (record, credited tokens, complete round milliseconds)
    if method == "ar":
        if batch != 1:
            # At batch > 1 one target step decodes a token for every live request: no per-token time.
            for stop in stops:
                kept.extend((None, 1, None) for _ in range(max(0, stop - 1)))
        else:
            _need(method, metrics, ("target_step_times",), "metrics")
            steps = metrics["target_step_times"]
            if len(steps) != sum(len(row["token_ids"]) for row in outputs):
                raise ValueError(f"{method}: target_step_times cannot be aligned to outputs")
            offset = 0
            for row, stop in zip(outputs, stops):
                length = len(row["token_ids"])
                kept.extend((None, 1, float(step) * 1000)
                            for step in steps[offset + 1:offset + stop])
                offset += length
    else:
        source = ("jetspec_round_records" if method == "jetspecpp" else
                  "blockhead_round_records")
        if method == "spec_sync" and not metrics.get(source):
            records = _spec_sync_records(outputs, metrics, batch=batch)
        else:
            _need(method, metrics, (source,), "metrics")
            records = metrics[source]
        if not records and any(stop > 1 for stop in stops):
            raise ValueError(f"{method}: empty {source} for nonempty decode")
        for record in records:
            _need(method, record, ("prompt_index",), source)
            index = int(record["prompt_index"])
            if index not in positions:
                raise ValueError(f"{method}: invalid prompt_index {index}")
        step_ms = None
        if method == "jetspecpp" and batch == 1:
            _need(method, metrics, ("target_step_times",), "metrics")
            if len(metrics["target_step_times"]) != len(records) + len(outputs):
                raise ValueError(f"{method}: target_step_times cannot be aligned to {source}")
            indices = [positions[int(record["prompt_index"])] for record in records]
            if indices != sorted(indices):
                raise ValueError(f"{method}: interleaved {source} cannot align step times")
            step_ms = []
            cursor = 0
            for index in native_indices:
                cursor += 1  # prefill/admission step
                for record in records:
                    if int(record["prompt_index"]) == index:
                        step_ms.append(float(metrics["target_step_times"][cursor]) * 1000)
                        cursor += 1
        progress = [0] * len(outputs)
        for pos, record in enumerate(records):
            index = positions[int(record["prompt_index"])]
            if method == "jetspecpp":
                _need(method, record, ("accepted_len", "seq_id", "candidate_round"), source)
                before = progress[index]
                accepted = int(record["accepted_len"])
                progress[index] += accepted
                # At batch > 1 one target step serves many requests: no per-round time.
                duration = step_ms[pos] if step_ms is not None else None
            else:
                _need(method, record, ("completion_before", "accepted", "round_ms"), source)
                before = int(record["completion_before"])
                accepted = int(record["accepted"])
                duration = float(record["round_ms"])
            left = stops[index] - 1 - before
            if left <= 0:
                continue
            if accepted < 1 or (duration is not None and duration <= 0):
                raise ValueError(f"{method}: invalid accepted tokens or round_ms")
            credited = min(accepted, left)
            if credited:
                kept.append((record, credited, duration))
    expected = sum(max(0, stop - 1) for stop in stops)
    count = sum(tokens_kept for _, tokens_kept, _ in kept)
    if count != expected:
        raise ValueError(f"{method}: round records credit {count} decode tokens; expected {expected}")
    rounds = len(kept)
    timed = all(duration is not None for _, _, duration in kept)
    elapsed_ms = sum(duration for _, _, duration in kept) if timed else None
    result = {"decode_tokens_eos_stopped": count, "rounds_eos_stopped": rounds,
              "round_ms_eos_stopped": elapsed_ms,
              "tau_eos_stopped": count / rounds if rounds else None,
              "tps_eos_stopped": count * 1000 / elapsed_ms if elapsed_ms else None}
    if not timed:
        result["timing_eos_stopped"] = ("NOT_MEASURED: batch > 1 target steps are shared "
                                        "across requests")
    if method == "jetspecpp":
        if not populations:
            raise ValueError(f"{method}: missing populations for hit/cutoff metrics")
        population_index = {(row["seq_id"], row["candidate_round"]): row
                            for row in populations}
        if len(population_index) != len(populations):
            raise ValueError(f"{method}: duplicate population round key")
        hit_tokens = miss_tokens = hits = misses = cutoffs = 0
        for record, credited, _ in kept:
            key = (record["seq_id"], record["candidate_round"])
            row = population_index.get(key)
            if row is None or row.get("join_status") not in ("admission", "joined"):
                raise ValueError(f"{method}: missing joined/admission population for {key}")
            if not round_is_miss(record, row):
                hits += 1
                hit_tokens += credited
            else:
                misses += 1
                miss_tokens += credited
            deadline = (row or {}).get("deadline") or {}
            cutoffs += bool(deadline.get("truncated") or deadline.get("deadline_miss"))
        result.update(hit_rate_eos_stopped=hits / rounds if rounds else None,
                      tau_hit_eos_stopped=hit_tokens / hits if hits else None,
                      tau_miss_eos_stopped=miss_tokens / misses if misses else None,
                      cutoff_rate_eos_stopped=cutoffs / rounds if rounds else None)
    return result


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




def quantile(values, q):
    ordered = sorted(values)
    at = (len(ordered) - 1) * q
    lo = int(at)
    return ordered[lo] + (ordered[min(lo + 1, len(ordered) - 1)] - ordered[lo]) * (at - lo)


def stats(values):
    values = list(values)
    if not values:
        raise ValueError('round anatomy requires timing samples')
    return dict(count=len(values), mean=mean(values), median=median(values),
                p90=quantile(values, .9))


def aligned_round_ms(raw):
    """Walk serial N=1 prompts, skipping each prompt's admission/prefill step."""
    metrics = raw['metrics']
    records = metrics['jetspec_round_records']
    steps = metrics['target_step_times']
    n = len(raw['outputs'])
    indices = [int(row['prompt_index']) for row in records]
    if (not records or len(steps) != len(records) + n or indices != sorted(indices)
            or any(i < 0 or i >= n for i in indices)):
        raise ValueError('round steps are not aligned to serial prompt records')
    cursor = pos = 0
    values = []
    for prompt in range(n):
        cursor += 1
        while pos < len(records) and indices[pos] == prompt:
            values.append(steps[cursor] * 1000)
            cursor += 1
            pos += 1
    if cursor != len(steps) or pos != len(records):
        raise ValueError('round steps were not exhausted')
    return values


def anatomy(raw, *, log=None):
    metrics = raw['metrics']
    rounds = aligned_round_ms(raw)
    verify = metrics['jetspec_verify_round_ms']
    draft = metrics['jetspec_draft_round_ms']
    if len(verify) != len(rounds):
        raise ValueError('verify timings are not aligned to verifier records')
    result = dict(round_ms=stats(rounds), verify_ms=stats(verify), draft_ms=stats(draft),
                inner_round_ms_mean=1000 * metrics['decode_total_time'] / len(rounds),
                draft_work_ms_per_verified_round=sum(draft) / len(rounds),
                draft_source='candidate wire draft_ms',
                source='existing host counters; target_step_times excluding every prompt prefill',
                population='all measured physical rounds, before EOS/rule-A credit reduction',
                limitation='draft publications include unused proposals and redrafts; draft and verify overlap, so their durations do not add to round wall time')
    if raw.get('run_manifest', {}).get('env', {}).get(environment_name('engine', 'engine_settings.chain_graph')) == 'seg':
        result.update(draft_ms=None, draft_work_ms_per_verified_round=None,
                      draft_source='unmeasured: graphed candidate wire timer is zero-filled')
        records, active = [], None
        log = log or ''
        phased = 'SERVE_READ_PHASE_START ' in log
        for line in log.splitlines():
            if line.startswith('SERVE_READ_PHASE_START '):
                active = line.split(' ', 1)[1]
            elif line.startswith('SERVE_READ_PHASE_END '):
                active = None
            if phased and active != 'measured':
                continue
            if 'EVAL_DRAFT_TIMING ' in line:
                row, _ = json.JSONDecoder().raw_decode(line.split('EVAL_DRAFT_TIMING ', 1)[1])
                if row['seq_id'] >= 0:
                    records.append(row)
        if records:
            total = [r['total_ms'] for r in records]
            redraft = [r['redraft_ms'] for r in records if r['redraft_ms'] is not None]
            proposal = [r['total_ms'] - (r['redraft_ms'] or 0) for r in records]
            if any(v < 0 for v in proposal + redraft):
                raise ValueError('draft timing spans are negative')
            result.update(draft_ms=stats(proposal), redraft_ms=stats(redraft) if redraft else None,
                          draft_graph_call_ms=stats(total),
                          draft_work_ms_per_verified_round=sum(total) / len(rounds),
                          draft_source='eval host spans from graph.run entry through synchronized publication, split at exact-redraft publication; admission excluded')
    return result


def reduce_paper_read(raw, method, loop_cut=True, *, eos_token_ids):
    outputs, metrics = raw['outputs'], raw['metrics']
    cuts = {}
    for i, row in enumerate(outputs):
        ids = row['token_ids'][:2048]
        if loop_cut and len(ids) == 2048 and not any(t in eos_token_ids for t in ids):
            last = ids[-64:]
            occurrences, j = [], 0
            while j + 64 <= len(ids):
                if ids[j:j+64] == last:
                    occurrences.append(j)
                    j += 64
                else:
                    j += 1
            if len(occurrences) >= 3:
                cuts[i] = occurrences[0]
    full = segment(method, outputs, metrics, eos_token_ids, 2048)
    cut = segment(method, outputs, metrics, eos_token_ids, 2048, cut=cuts)
    stops, pre, dec, cred, rounds, allr, credited_time = cut
    # Scale outer step attribution to the measured inner prefill/decode clocks,
    # exactly as the paper baseline bank does.
    P, D = metrics['prefill_total_time'], metrics['decode_total_time']
    pre_scale = P / sum(full[1])
    dec_scale = D / sum(full[2])
    times = [pre_scale*p + dec_scale*(c if i in cuts else d)
             for i, (p, d, c) in enumerate(zip(pre, dec, credited_time))]
    tokens = [max(0, stop-1) for stop in stops]
    assert tokens == cred, 'CREDIT_NOT_EQUAL_STOP_MINUS_PREFILL'
    total, R, seconds = sum(tokens), sum(rounds), sum(times)
    return dict(tokens=total, rounds=R, tau=total/R if R else 0,
                served_seconds=seconds, tok_s=total/seconds,
                cut_prompts=sorted(cuts), cut_onsets=cuts,
                prompt_tokens=tokens, prompt_rounds=rounds, prompt_seconds=times)


def export_candidates(output_json, metrics):
    """Keep only verified measured rounds; refuse missing or incomplete captures."""
    decisions = {(row['seq_id'], row['candidate_round']): row
                 for row in metrics['jetspec_round_records']}
    seen = set()
    records = 0
    sidecar = Path(str(output_json) + '.candidates.jsonl')
    with sidecar.open('w') as output:
        for source in sorted(Path(str(output_json) + '.rows').glob('*.jsonl')):
            with source.open() as lines:
                for line in lines:
                    row = json.loads(line)
                    key = row['seq_id'], row['candidate_round']
                    if key not in decisions:
                        continue
                    if row.get('schema') != 'jetspec.eval_round_candidates/1':
                        raise ValueError(f'eval capture lacks candidate schema: {key}')
                    for name in ('draft_top16', 'published_draft_top16'):
                        top = row[name]
                        ids, probabilities = top['token_ids'], top['probabilities']
                        if len(ids) != len(probabilities):
                            raise ValueError(f'eval capture has invalid {name} shape: {key}')
                        if name == 'published_draft_top16':
                            if any(len(one) != len(row['p_top1']) or len(one) != len(other)
                                   for one, other in zip(ids, probabilities)):
                                raise ValueError(f'eval capture has invalid {name} slots: {key}')
                            pairs = [(a, b) for one, other in zip(ids, probabilities)
                                     for a, b in zip(one, other)]
                        else:
                            if len(ids) != len(row['p_top1']):
                                raise ValueError(f'eval capture has invalid {name} slots: {key}')
                            pairs = list(zip(ids, probabilities))
                        if not pairs or any(len(a) != 16 or len(b) != 16 or
                                            any(not math.isfinite(p) or not 0 <= p <= 1 for p in b)
                                            for a, b in pairs):
                            raise ValueError(f'eval capture has invalid {name}: {key}')
                    for name in ('p_top1', 'q_shadow', 'shadow_argmax', 'offers', 'published_keys'):
                        if name not in row:
                            raise ValueError(f'eval capture lacks {name}: {key}')
                    for name in ('draft_token_ids', 'draft_token_probabilities', 'draft_pinned',
                                 'published_draft_token_probabilities', 'published_draft_pinned'):
                        if name not in row:
                            raise ValueError(f'eval capture lacks {name}: {key}')
                    width = len(row['p_top1'])
                    if len(row['draft_token_ids']) != width:
                        raise ValueError(f'eval capture has invalid draft_token_ids: {key}')
                    for prefix in ('', 'published_'):
                        probabilities = row[prefix+'draft_token_probabilities']
                        pinned = row[prefix+'draft_pinned']
                        if prefix:
                            if len(probabilities) != len(row['published_draft_top16']['token_ids']):
                                raise ValueError(f'eval capture has invalid published rows: {key}')
                        else:
                            probabilities, pinned = [probabilities], [pinned]
                        if (len(probabilities) != len(pinned) or
                                any(len(values) != width or len(flags) != width or
                                    any(not math.isfinite(p) or not 0 <= p <= 1 for p in values) or
                                    any(type(flag) is not bool for flag in flags)
                                    for values, flags in zip(probabilities, pinned))):
                            raise ValueError(f'eval capture has invalid {prefix}draft_token values: {key}')
                    from jetspec.engines.jetspec_plus_plus.runtime.pass_labels import normalized_label
                    if not normalized_label(row['pass_label']):
                        raise ValueError(f'eval capture lacks derived pass label: {key}')
                    if row['record_kind'] != 'redraft':
                        if key in seen:
                            raise ValueError(f'eval capture duplicates verified round: {key}')
                        seen.add(key)
                        row['verifier'] = dict(decisions[key])
                        # The verifier uses +inf for an unlimited wait budget.
                        # Preserve that sentinel without nonstandard JSON numbers.
                        if row['verifier'].get('wait_ms') == math.inf:
                            row['verifier']['wait_ms'] = 'inf'
                        row.setdefault('k_true', decisions[key]['k'])
                        row.setdefault('anchor_true', decisions[key]['correction'])
                    output.write(json.dumps(row, allow_nan=False) + '\n')
                    records += 1
    missing = decisions.keys() - seen
    if missing:
        raise ValueError(f'eval capture lacks verified rounds: {sorted(missing)[:8]}')
    return dict(schema='jetspec.eval_round_candidates/1', path=str(sidecar),
                verified_rounds=len(seen), records=records,
                fields=['draft_top16', 'published_draft_top16', 'draft_token_ids',
                        'draft_token_probabilities', 'draft_pinned',
                        'published_draft_token_probabilities', 'published_draft_pinned', 'pass_label'])


def export_decoder_inputs(output_json, metrics):
    """Move the decoder-input sidecar beside the read; refuse a verified round without its window."""
    import hashlib
    rows_dir = Path(str(output_json) + '.rows') / 'decoder_inputs'
    data, index = rows_dir / 'decoder_inputs.bin', rows_dir / 'decoder_inputs.jsonl'
    if not data.is_file() or not index.is_file():
        raise ValueError('eval decoder-input capture wrote no sidecar')
    target_data = Path(str(output_json) + '.decoder_inputs.bin')
    target_index = Path(str(output_json) + '.decoder_inputs.jsonl')
    data.replace(target_data)
    index.replace(target_index)
    verified = {(row['seq_id'], row['candidate_round']) for row in metrics['jetspec_round_records']}
    # Only proposals run the decoder; admissions and redrafts carry an exact correction.
    proposals = set()
    with Path(str(output_json) + '.candidates.jsonl').open() as lines:
        for line in lines:
            row = json.loads(line)
            if row['record_kind'] == 'proposal':
                proposals.add((row['seq_id'], row['candidate_round']))
    size = target_data.stat().st_size
    covered, rows, tap_rows = set(), 0, {}
    with target_index.open() as lines:
        for line in lines:
            row = json.loads(line)
            if row.get('schema') != 'jetspec.eval_decoder_inputs/1':
                raise ValueError('eval decoder-input index has an unknown schema')
            for offset, shape in ((row['window_offset'], row['window_shape']),
                                  (row['taps_offset'], row['taps_shape'])):
                if offset + 2 * math.prod(shape) > size:
                    raise ValueError(f'eval decoder-input index points past its data: {row}')
            seq = row['seq_id']
            if row['taps_from'] != tap_rows.get(seq, 0):
                raise ValueError(f'eval decoder-input taps are not contiguous: {row}')
            tap_rows[seq] = row['taps_to']
            covered.add((seq, row['candidate_round']))
            rows += 1
    missing = proposals - covered
    if missing:
        raise ValueError(f'eval decoder-input capture lacks verified proposals: {sorted(missing)[:8]}')
    digest = hashlib.sha256()
    with target_data.open('rb') as source:
        for block in iter(lambda: source.read(1 << 24), b''):
            digest.update(block)
    return dict(schema='jetspec.eval_decoder_inputs/1', path=str(target_data), index=str(target_index),
                rows=rows, verified_rounds=len(verified), verified_proposals=len(proposals),
                bytes=size, sha256=digest.hexdigest())


def write_eval_records(config, *, recipe_method, output, eval_values, process_wall_seconds):
    """Use the existing wrapper's reducers and the launcher's recorded evidence."""
    from jetspec.engines.jetspec_plus_plus.runtime.profile import read_counters
    method, model = config['method'], config['model']
    output = Path(output)
    campaign = json.loads(Path(str(output) + '.campaign.json').read_text())
    condition = campaign['resolved_condition']
    for read in campaign['reads']:
        path = Path(read['output'])
        raw = json.loads(path.read_text())
        schedule = method.get('schedule', dict(j=0, r=0))
        raw['eval'] = dict(model='27B' if model['target']['family'] == 'hybrid' else '8B',
            j=schedule['j'], r=schedule['r'], budget=method['block_size'],
            rows=int(eval_values['rows']),
            drafting_mode=eval_values['drafting_mode'],
            j3_label=eval_values['j3_label'],
            dtype=model['target']['dtype'], placement=config['hardware']['placement'],
            process_wall_seconds=process_wall_seconds,
            served_wall_seconds=raw['metrics']['serve_read']['measured']['wall_seconds'],
            draft_delay_ms=float(eval_values['draft_delay_ms']),
            host_fast=eval_values['host_fast'], eos_token_ids=condition['eos_token_ids'])
        if recipe_method == 'jetspecpp':
            log = '\n'.join(marker + ' ' + json.dumps(value)
                            for marker in ('EVAL_ROUND', 'CHAIN_FALLBACK')
                            for value in read.get(marker, []))
            raw['eval']['counters'] = read_counters(log, raw['metrics'])
            raw['eval']['anatomy'] = anatomy(raw, log=log)
            raw['eval']['paper_metrics'] = reduce_paper_read(raw, 'jetspecpp', eos_token_ids=condition['eos_token_ids'])
            refresh = eval_values['refresh_mode']
            if refresh != 'full':
                raw['eval']['refresh_mode'] = refresh
            capture = eval_values['capture_round_candidates']
            if capture != '0':
                raw['eval']['round_candidates'] = export_candidates(path, raw['metrics'])
                if capture == 'decoder':
                    raw['eval']['decoder_inputs'] = export_decoder_inputs(path, raw['metrics'])
        else:
            raw['eval']['paper_metrics'] = reduce_paper_read(
                raw, recipe_method, eos_token_ids=condition['eos_token_ids'])
        path.write_text(json.dumps(raw, indent=2) + '\n')
