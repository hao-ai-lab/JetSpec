"""Sealed Table 3 groups and public serving configurations."""
import hashlib
import json
from pathlib import Path
import re
import sys

from bench.jetspec_plus_plus.dataloader import PINNED_PROMPTS
from jetspec.engines.jetspec_plus_plus.launch import entry, runner
from jetspec.engines.jetspec_plus_plus.launch.eos_stop import stop_lengths

ROOT = Path(__file__).resolve().parents[3] / 'jetspec/engines/jetspec_plus_plus'
INPUTS = Path(__file__).resolve().parents[1] / 'prompts'
SEALS = json.loads((INPUTS / 'seals.json').read_text())
SOURCES = {'8b': '8b', '27b': '27b-sys'}
PROMPT_KEYS = {model: {dataset: f'{source}/prompts_{dataset}128.json'
                      for dataset in ('gsm8k', 'math500')}
              for model, source in SOURCES.items()}
HASHES = {model: {dataset: SEALS[key] for dataset, key in sources.items()}
          for model, sources in PROMPT_KEYS.items()}
ARMS = {
    '8b': {'AR': ('ar', 'ar-graphed', 'one_gpu'),
           'DS': ('blockhead', 'blockhead15-800k-graphed', 'one_gpu'),
           'JetSpec++': ('jetspecpp', None, 'pair')},
    '27b': {'AR': ('ar', 'ar-hybrid27b-graphed', 'one_gpu'),
            'DS': ('blockhead', 'blockhead27b-800k-e6-graphed-sm100', 'one_gpu'),
            'JetSpec++': ('jetspecpp', None, 'pair')}}
STOCK_DS_KNOBS = ('JETSPEC_BLOCKHEAD_ATTN_LEN', 'JETSPEC_BLOCKHEAD_GLUE_LEAN',
                  'JETSPEC_BLOCKHEAD_QKV_FUSED', 'JETSPEC_BLOCKHEAD_LMHEAD_KN')


def read_record(job, output, cache_root):
    """Resolve public fast compositions for the method, sealed recipes for baselines."""
    method, recipe, placement = ARMS[job['model']][job['arm']]
    if method == 'jetspecpp':
        names = dict(method='jetspecpp-j1r1-b16-floor4',
                     model='qwen3-8b-release' if job['model'] == '8b' else 'qwen3.8-27b-p2-cal28',
                     engine='fast-sm100', hardware='gb200-pair', benchmark='gsm8k-128')
        launch, overrides = None, []
        if job['model'] == '27b':
            overrides += ['method.schedule.j=3', 'method.schedule.r=3']
    else:
        rows = json.loads((ROOT / 'launch/recipe-index.json').read_text())['rows']
        row = next(r for r in rows if r.get('launch', {}).get('name') == recipe)
        names, launch, overrides = row['names'], row['launch'], list(row['overrides'])
        # A sealed baseline replays its named recipe, so the engine never looks for
        # composed metadata, which baseline families do not have.
        overrides.append('engine.recipe=' + json.dumps(recipe))
    overrides += ['hardware.placement=' + json.dumps(placement),
                  'hardware.devices=' + json.dumps(list(range(2 if placement == 'pair' else 1))),
                  'hardware.cpu_pin=' + json.dumps('auto' if method == 'jetspecpp' else 'off'),
                  'benchmark.count=32', 'benchmark.cap=2048', 'benchmark.stop_at_eos=true',
                  'benchmark.temperature=0', 'benchmark.seed=0', 'benchmark.repeats=1']
    overrides += ['benchmark.dataset=' + json.dumps(job['dataset']),
                  'benchmark.offset=' + json.dumps(job['offset']),
                  'benchmark.prompt_files=' + json.dumps([dict(
                      file=PINNED_PROMPTS[SOURCES[job['model']], job['dataset']],
                      sha256=HASHES[job['model']][job['dataset']],
                      count=128, turn=1, temperature=0)])]
    record = entry.resolve_read(names, overrides, root=ROOT / 'config', launch=launch,
                                output=str(output), sequential_prompts=True,
                                cache_root=str(cache_root), wire_records='off', anatomy='off')
    if method != 'jetspecpp':
        # Baselines replay their recorded environments, including thread caps.
        record['recipe']['environment'] = dict(row['environment'])
        record['environment'] = dict(row['environment'])
    return record


def build_argv(job, run, prompts, capacity, record, args):
    recipe = record['recipe']
    argv = [sys.executable, '-m', 'jetspec.engines.jetspec_plus_plus.launch.runner',
            '--method', recipe['method'], '--recipe', recipe['name'], '--regime', 'graphed',
            '--placement', ARMS[job['model']][job['arm']][2], '--devices', job['devices'],
            '--dataset', job['dataset'], '--prompts-json', str(prompts), '--count', '32',
            '--batch', '1', '--sequential-prompts', '--tokens', '2048', '--stop-at-eos', 'on',
            '--temperature', '0', '--seed', '0', '--repeats', '1',
            '--cache-root', str((args.output_dir / 'compiler-cache').resolve()),
            '--wire-records', 'off', '--anatomy', 'off',
            '--cpu-pin', 'auto' if recipe['method'] == 'jetspecpp' else 'off',
            '--max-model-len', str(capacity), '--child-wall-seconds', '2400',
            '--output-json', str(run / 'read.json'), '--knob', 'SSD_SHARED_GPU_KV_UTILIZATION=0.10']
    if args.host_values:
        argv += ['--host-values', str(args.host_values.resolve())]
    if args.artifacts_dir:
        argv += ['--artifacts-dir', str(args.artifacts_dir.resolve())]
    if recipe['method'] == 'jetspecpp':
        argv += ['--draft-deadline', 'on']
    if job['model'] == '27b':
        argv += ['--knob', 'JETSPEC_ANATOMY_LITE=1']
        if job['arm'] in ('DS', 'AR'):
            argv += ['--knob', 'JETSPEC_HYBRID_VERIFY_FLASHINFER=1']
    return argv


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def select_group(original, offset, count, source_sha):
    if len(original['prompts']) != 128 or offset < 0 or offset + count > 128:
        raise ValueError('invalid sealed population or group bounds')
    rows = [dict(row, index=i, original_index=row['index'])
            for i, row in enumerate(original['prompts'][offset:offset + count])]
    group = dict(original, count=count, prompts=rows, source_sha256=source_sha,
                 original_prompt_offset=offset)
    validate_group(group, original, offset, count)
    return group


def validate_group(group, original, offset, count):
    assert len(group['prompts']) == count and group['count'] == count
    indices = []
    for i, (row, source) in enumerate(zip(group['prompts'], original['prompts'][offset:offset + count])):
        assert row['index'] == i and row['original_index'] == source['index'] == offset + i
        assert row['text'] == source['text'] and row['token_ids'] == source['token_ids'], 'SOURCE_IDENTITY'
        indices.append(row['original_index'])
    assert indices == list(range(offset, offset + count))
    return indices


def validate_coverage(receipts, arm, dataset, model=None):
    indices = [i for r in receipts if r['arm'] == arm and r['dataset'] == dataset
               and (model is None or r.get('model') == model)
               and not r['audit'] for i in r['original_indices']]
    assert sorted(indices) == list(range(128)), 'NON_BIJECTIVE_COVERAGE'
    return sorted(indices)


def eos_region_events(outputs, metrics, text, tokens, eos, indices, capacity_total):
    """Use the EOS reducer's prefill-excluded credit frontier for every event."""
    stops = [max(0, n-1) for n in stop_lengths(outputs, eos, tokens=tokens)]
    records = metrics.get('jetspec_round_records', [])
    progress = [0] * len(outputs)
    locations, sequences, events = {}, {}, []
    counts = dict(ar_steps_inside_eos=0, ar_steps_after_eos=0,
                  capacity_fallback_inside_eos=0, capacity_fallback_after_eos=0)

    def event(kind, record, index, before):
        region = 'inside_eos' if before < stops[index] else 'after_eos'
        counts[f'{kind}_{region}'] += 1
        events.append(dict(kind=kind, region=region, original_index=indices[index],
            seq_id=record['seq_id'], round=record.get('round'),
            candidate_round=record.get('candidate_round'), completion_before=before,
            eos_stopped_decode_tokens=stops[index],
            chain_redraft_timeout=bool(record.get('chain_redraft_timeout')),
            fallback_wait_ms=record.get('fallback_wait_ms')))

    for record in records:
        index = int(record['prompt_index'])
        assert 0 <= index < len(outputs), 'EVENT_PROMPT_ALIGNMENT'
        before = progress[index]
        round_id = record.get('candidate_round')
        if round_id is None:
            round_id = record['round']
        key = (record['seq_id'], round_id)
        assert key not in locations, 'DUPLICATE_EVENT_ROUND'
        locations[key] = (record, index, before)
        sequences[record['seq_id']] = (index, int(record['round']))
        # Pinned chain: a delivered redraft is full verification even though its
        # decision remains "miss". A miss/late without delivery is the AR valve.
        assert record['decision'] in ('hit', 'miss', 'late'), 'UNKNOWN_VERIFY_DECISION'
        # A timed-out round can still commit more than the correction. The engine then counts
        # it as full verification with no AR step, so it is not the AR valve.
        late_full = bool(record.get('chain_redraft_timeout')) and int(record['accepted_len']) > 1
        is_ar = record['decision'] != 'hit' and not record.get('fallback') and not late_full
        if is_ar:
            assert int(record['accepted_len']) == 1, 'AR_EVENT_WIDTH'
            event('ar_steps', record, index, before)
        progress[index] += int(record['accepted_len'])
    assert counts['ar_steps_inside_eos'] + counts['ar_steps_after_eos'] == metrics['jetspec_ar_steps'], 'AR_EVENT_ALIGNMENT'
    assert sum(bool(r.get('chain_redraft_timeout')) for r in records) == metrics.get('jetspec_chain_redraft_missing', 0), 'REDRAFT_EVENT_ALIGNMENT'
    active = None
    phased = 'SERVE_READ_PHASE_START ' in text
    capacity = []
    for line in text.splitlines():
        if line.startswith('SERVE_READ_PHASE_START '):
            active = line.split(' ', 1)[1]
        elif line.startswith('SERVE_READ_PHASE_END '):
            active = None
        elif 'CHAIN_FALLBACK ' in line and (not phased or active == 'measured'):
            row, _ = json.JSONDecoder().raw_decode(line.split('CHAIN_FALLBACK ', 1)[1])
            if row['reason'] == 'capacity':
                capacity.append(row)
    assert len(capacity) == capacity_total, 'CAPACITY_EVENT_ALIGNMENT'
    for row in capacity:
        location = locations.get((row['seq_id'], row['round']))
        if location:
            record, index, before = location
        else:
            # A publication can outlive its final verify round. Only credit an
            # unmatched later publication after the observed EOS frontier.
            assert row['seq_id'] in sequences, 'CAPACITY_EVENT_UNASSIGNED'
            index, last_round = sequences[row['seq_id']]
            before = progress[index]
            assert row['round'] > last_round and before >= stops[index], 'CAPACITY_EVENT_UNASSIGNED'
            record = dict(row, candidate_round=row['round'])
        event('capacity_fallback', record, index, before)
    return dict(counts, ar_steps=int(metrics['jetspec_ar_steps']),
                capacity_fallback=capacity_total, eos_region_events=events,
                gate='EOS_REGION_2026-10-07')


def validate_read(output, log, job, prompts, original, record):
    group = json.loads(prompts.read_text())
    indices = validate_group(group, original, job['offset'], 32)
    assert group['source_sha256'] == HASHES[job['model']][job['dataset']]
    benchmark = record['config']['benchmark']
    assert benchmark['dataset'] == job['dataset'], 'RECORD_DATASET'
    assert benchmark['offset'] == job['offset'], 'RECORD_OFFSET'
    assert benchmark['count'] == 32, 'RECORD_COUNT'
    prompt = next(p for p in benchmark['prompt_files']
                  if p['temperature'] == benchmark['temperature'])
    assert prompt['sha256'] == HASHES[job['model']][job['dataset']], 'RECORD_PROMPT_SHA'
    prompt_file = ROOT / 'config/prompts' / prompt['file']
    assert prompt_file.is_file(), 'RECORD_PROMPT_FILE'
    assert sha256(prompt_file) == HASHES[job['model']][job['dataset']], 'RECORD_PROMPT_FILE_SHA'
    raw = json.loads(output.read_text())
    campaign = json.loads(Path(str(output) + '.campaign.json').read_text())
    condition = campaign['resolved_condition']
    recipe = record['recipe']
    for key, expected in dict(count=32, tokens=2048, temperature=0, seed=0,
                              stop_at_eos='on', sequential_prompts=True,
                              prompts_sha256=sha256(prompts), method=recipe['method'],
                              recipe=recipe['name'], regime='graphed',
                              placement=ARMS[job['model']][job['arm']][2]).items():
        assert condition[key] == expected, (key, condition[key], expected)
    eos = {151643, 151645} if job['model'] == '8b' else {248044, 248046}
    assert set(condition['eos_token_ids']) == eos, 'EOS_IDENTITY'
    env = condition['environment']
    assert env['SSD_SHARED_GPU_KV_UTILIZATION'] == '0.10'
    assert len(raw['outputs']) == 32
    assert all(1 <= len(row['token_ids']) <= 2048 for row in raw['outputs'])
    text = log.read_text()
    assert 'SERVE_READ_COMPLETE' in text and 'Traceback (most recent call last)' not in text
    assert not re.search(r'HYBRID_VERIFY_LOGIT_AUDIT .*violations=[1-9]', text), 'AUDIT_VIOLATION'
    measured = campaign['reads'][0]['measured']
    metrics = measured['metrics']
    receipt = dict(**job, audit=False, count=32, tokens=2048, original_indices=indices,
                   source_sha256=HASHES[job['model']][job['dataset']],
                   output=str(output), output_sha256=sha256(output),
                   campaign_sha256=sha256(Path(str(output) + '.campaign.json')))
    if job['arm'] == 'DS':
        assert all(env.get(k, '0') == '0' for k in STOCK_DS_KNOBS), 'STOCK_DS_ENHANCED'
        if job['model'] == '27b':
            assert env['SSD_VERIFY_DEVICE'] == '1' and env['SSD_ATTENTION_BACKEND'] == 'flashinfer'
    if recipe['method'] == 'jetspecpp':
        assert condition.get('bundle_artifact') == recipe['bundle_artifact'], 'ARTIFACT_IDENTITY'
        assert campaign['reads'][0]['manifest']['thresholds_sha256'] == recipe['thresholds_sha256']
        assert env['JETSPEC_CHAIN_MISS_REDRAFT'] == '1'
        assert env['JETSPEC_ROWS_RULE_CAP'] == '4'
        assert env['JETSPEC_CHAIN_FIXED_CONFIG'] == ('J1-R1' if job['model'] == '8b' else 'J3-R3')
        assert env.get('JETSPEC_VERIFY_REFERENCE', '0') == '0', 'PUBLIC_FAST_ENGINE_REQUIRED'
        receipt.update(eos_region_events(raw['outputs'], metrics, text, 2048, eos, indices,
                                         measured['chain']['capacity_fallback']))
        assert receipt['ar_steps_inside_eos'] == 0, 'NONZERO_AR_STEPS_INSIDE_EOS'
        assert receipt['capacity_fallback_inside_eos'] == 0, 'CAPACITY_FALLBACK_INSIDE_EOS'
        if job['model'] == '27b':
            errors = [int(v) for v in re.findall(r'W645_CHAIN_AUDIT .*"error_bits": (\d+)', text)]
            assert errors and max(errors) == 0, 'CHAIN_ERR'
            receipt.update(chain_err=max(errors), chain_audit_rounds=len(errors))
    return receipt
