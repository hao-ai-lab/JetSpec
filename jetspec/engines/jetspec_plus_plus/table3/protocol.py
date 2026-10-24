"""Sealed Table 2 groups and EOS-region gate, ported from serving."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile

# Adaptation: bundled package paths, with no sys.path mutation.
ROOT = Path(__file__).resolve().parents[1]
from jetspec.engines.jetspec_plus_plus.table3.recipe_record import recipe_record
from jetspec.engines.jetspec_plus_plus.table3.eos_stop import stop_lengths

INPUTS = Path(__file__).resolve().parent / 'inputs'
FLOOR4_POLICY = (INPUTS / 'policy-rows-min-4.json')
FLOOR4_SHA = '30958c605091bfa444dbce404188093514678a2fdfc9a349a9e594c646a6bd90'
FLOOR16_POLICY = (INPUTS / 'policy-rows-min-16.json')
FLOOR16_SHA = '60275f8a02e5a6cd7fed7b36cfc726821a62a7e083bea04aba780d8aa36887be'
# The cal28 bundle bytes the Table 2 eval recipe serves, sealed in the kit so the
# rx-synced tree carries them to NVL (host_values maps the manifest key here).
CAL28_BUNDLE = (INPUTS / 'cal28')
CAL28_THRESHOLDS_SHA = 'd08f118243db1a6755fbd64bdf4387bcb32802c78d343431355409d25f501994'
CAL28_DECODER_MANIFEST_SHA = '43eb014293a5bace28116f6289ca3b893507ccbe2104a102bb4d370b4a0c308b'
HASHES_8B = {
    'gsm8k': 'f95ef100190abdb8810c5e8b12c0a5efc1525d31a6f272861c158f453cddf2b7',
    'math500': '3c2fd08be240ad6a907ebaf5aeb47c7bcff581e63dac06289884ae3dd2767be7',
}
ARMS_8B = {
    'J1R1REF': ('jetspecpp', 'jetspecpp-reference-pair-e6v3-b16-chain-r4-j1r1-lean-swap-fast6-table3', 'pair'),
    'J1R1REF_F4': ('jetspecpp', 'jetspecpp-reference-pair-e6v3-b16-chain-r4-j1r1-lean-swap-fast6-table3', 'pair'),
    'J1R1': ('jetspecpp', 'jetspecpp-graphed-pair-e6v3-b16-chain-r4-j1r1-lean-swap-fast6', 'pair'),
    'J2R0': ('jetspecpp', 'jetspecpp-graphed-pair-e6v3-b16-chain-r4-j2r0-lean-fast6', 'pair'),
    'DSE': ('blockhead', 'blockhead15-800k-graphed', 'one_gpu'),
    'AR': ('ar', 'ar-graphed', 'one_gpu'),
    'A79': ('jetspecpp', 'jetspecpp-graphed-pair-e6-b16-chain-r4-jswitch1024-lean', 'pair'),
    'DSE_A79': ('blockhead', 'blockhead15-800k-graphed', 'one_gpu'),
    'DS': ('blockhead', 'blockhead15-800k-graphed', 'one_gpu'),
    'DFlash': ('dflash', 'dflash-ours-graphed', 'one_gpu'),
    'SSD': ('ssd', 'ssd-graphed', 'pair'),
    'AR_PAPER_GB1': ('ar', 'ar-graphed', 'one_gpu'),
}
ORIGINAL_HASHES_27B = {
    'gsm8k': '2ac6748d460da9c8fa501161e1843aa856fda5bb1e66d9f8287c8b369422a648',
    'math500': 'd93fe259357949bcd046ee9b477b632437ea3156998bbdb7a641e2972b7d9f79'}
TABLE1_RENDERED_HASHES_27B = {
    'gsm8k': '85fbc2c736add59a734acacf06dbc25f09d4dab464f591628f5e79f0c9542493',
    'math500': 'c81232e3e2eda8b509cbcad29e9f7cf480b9a810bd880a922716857b327d85ae'}
# The named sysrender-served renderer has different metadata from the Table 1
# renderer. Seal the bytes actually staged; preserve the original seals above.
HASHES = {'8b': HASHES_8B, '27b': {
    'gsm8k': '9262b74cd32a010df76b9f669c4d182ae8c3f29d01a57fabb8e016cdb53f2a47',
    'math500': '02f837f24228ca135559a941abd8df73af2b78afeaa0c7a864eaedefa5861d2c'}}
ARMS = {'8b': ARMS_8B, '27b': {
    'J1R1REF_F4': ('jetspecpp', 'jetspecpp-reference-pair-27b-drafter6ep-base800k-j1r1-swap-command', 'pair'),
    'J3R3REF_F4': ('jetspecpp', 'jetspecpp-reference-pair-27b-drafter6ep-base800k-j3r3-swap-command', 'pair'),
    'J3R3DEV_F4': ('jetspecpp', 'jetspecpp-device-pair-27b-drafter6ep-base800k-j3r3-swap-dspark', 'pair'),
    # Measurement-only J3R3DEV_F4 variants for the Table 3 basis check against
    # the Table 2 eval reads: rows 16 (rows_min 16, rule cap 16), cal28 thresholds, both.
    'J3R3DEV_R16': ('jetspecpp', 'jetspecpp-device-pair-27b-drafter6ep-base800k-j3r3-swap-dspark-rows16', 'pair'),
    'J3R3DEV_CAL28': ('jetspecpp', 'jetspecpp-device-pair-27b-drafter6ep-base800k-j3r3-swap-dspark-cal28', 'pair'),
    'J3R3DEV_R16_CAL28': ('jetspecpp', 'jetspecpp-device-pair-27b-drafter6ep-base800k-j3r3-swap-dspark-rows16-cal28', 'pair'),
    'J3R3': ('jetspecpp', 'jetspecpp-graphed-pair-27b-drafter6ep-base800k-j3r3-swap-fast3-refresh-key-bucket', 'pair'),
    'J2R2': ('jetspecpp', 'jetspecpp-graphed-pair-27b-drafter6ep-base800k-j2r2-swap-fast3', 'pair'),
    'DSE': ('blockhead', 'blockhead27b-800k-e6-fast-graphed-sm100', 'one_gpu'),
    'AR': ('ar', 'ar-hybrid27b-graphed', 'one_gpu'),
    'DS': ('blockhead', 'blockhead27b-800k-e6-graphed-sm100', 'one_gpu')}}
DEV_VARIANTS_27B = ('J3R3DEV_R16', 'J3R3DEV_CAL28', 'J3R3DEV_R16_CAL28')
DEV_ARMS_27B = ('J3R3DEV_F4', *DEV_VARIANTS_27B)
ROWS16_ARMS = ('J3R3DEV_R16', 'J3R3DEV_R16_CAL28')
CAL28_ARMS = ('J3R3DEV_CAL28', 'J3R3DEV_R16_CAL28')
# Adaptation: these GB200 pair arms pin each rank to CPUs on its GPU's NUMA node, as the
# backend Table 3 hardware entry gb200-pair does (cpu_pin auto).
GB200_PINNED_ARMS = {('8b', 'J1R1REF_F4'), ('27b', 'J3R3DEV_F4')}


def arm_policy(arm):
    """The sealed row-floor policy, its sha and its failure label for a floored arm."""
    if arm in ROWS16_ARMS:
        return FLOOR16_POLICY, FLOOR16_SHA, 'FLOOR16'
    return FLOOR4_POLICY, FLOOR4_SHA, 'FLOOR4'


# a79 DSE: its device verify and our four shared head kernels, not stock DSpark.
DSE_KNOBS = {'SSD_VERIFY_DEVICE': '1', 'SSD_VERIFY_SYNC_AUDIT': '0',
             **{f'JETSPEC_BLOCKHEAD_{name}': '1'
                for name in ('ATTN_LEN', 'GLUE_LEAN', 'QKV_FUSED', 'LMHEAD_KN')}}


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
        is_ar = record['decision'] != 'hit' and not record.get('fallback')
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


def build_argv(arm, dataset, devices, run, prompts, max_model_len, audit=False, model='8b', *,
               allowed_devices=('0','1','2','3'), smoke=False, qualification=False, ar_qualification=False):
    method, recipe, placement = ARMS[model][arm]
    assert len(devices.split(',')) == (2 if placement == 'pair' else 1)
    assert len(set(devices.split(','))) == len(devices.split(','))
    assert all(i in allowed_devices for i in devices.split(',')), 'UNAUTHORIZED_CARDS'
    count, tokens = (8, 256) if audit else (32, 2048)
    if smoke:
        assert model == '27b' and not audit and arm in ('J1R1REF_F4','J3R3REF_F4',*DEV_ARMS_27B,'DS','AR'), 'SMOKE_CONDITION'
        count, tokens = 8, 512
    if qualification:
        assert model == '27b' and not audit and not smoke and arm in ('J1R1REF_F4','J3R3REF_F4',*DEV_ARMS_27B,'DS'), 'QUALIFICATION_CONDITION'
        count, tokens = 16, 512
    if ar_qualification:
        assert arm == 'AR' and model == '8b' and not audit
        count, tokens = 8, 512
    if audit:
        assert method == 'jetspecpp' and dataset == 'gsm8k'
    argv = [sys.executable, '-m', 'jetspec.engines.jetspec_plus_plus.read.legacy', '--method', method,
            '--recipe', recipe, '--regime', 'graphed', '--placement', placement,
            '--devices', devices, '--dataset', dataset, '--host-values', str(run / 'host-values.json'),
            '--prompts-json', str(prompts), '--count', str(count), '--batch', '1',
            '--sequential-prompts', '--tokens', str(tokens), '--stop-at-eos', 'on',
            '--temperature', '0', '--seed', '0', '--repeats', '1',
            '--cache-root', str(run / 'compiler-cache'), '--wire-records', 'off',
            '--anatomy', 'off', '--cpu-pin', 'auto' if (model, arm) in GB200_PINNED_ARMS else 'off',
            '--max-model-len', str(max_model_len),
            '--child-wall-seconds', '2400', '--output-json', str(run / 'read.json'),
            '--knob', 'SSD_SHARED_GPU_KV_UTILIZATION=0.10']
    if method == 'jetspecpp':
        argv.extend(['--draft-deadline', 'on'])
    if model == '27b' and arm in ('J1R1REF_F4','J3R3REF_F4',*DEV_ARMS_27B,'DS','AR'):
        argv.extend(['--knob', 'JETSPEC_ANATOMY_LITE=1'])
    if model == '27b' and arm in ('DS', 'AR'):
        # Adaptation: the paper's 27B AR and DSpark rows ran FlashInfer on sm100; name its opt-in.
        argv.extend(['--knob', 'JETSPEC_HYBRID_VERIFY_FLASHINFER=1'])
    for key, value in (DSE_KNOBS.items() if arm in ('DSE','DSE_A79') and model == '8b' else ()):
        argv.extend(['--knob', f'{key}={value}'])
    if audit:
        argv.extend(['--knob', 'JETSPEC_HYBRID_VERIFY_AUDIT=1'])
    if arm in ('J1R1REF_F4','J3R3REF_F4',*DEV_ARMS_27B):
        policy, policy_sha, label = arm_policy(arm)
        assert sha256(policy) == policy_sha, f'{label}_POLICY_SHA'
        (run / 'policy-receipt.json').write_text(json.dumps(dict(
            policy=str(policy), policy_sha256=policy_sha)) + '\n')
        argv.extend(['--knob', f'JETSPEC_SERVE_POLICY={policy}'])
    return argv


def validate_read(output, log, arm, dataset, prompts, original, offset, audit=False, model='8b', *,
                  smoke=False, qualification=False, write_receipt=True, draft_deadline='on', ar_qualification=False):
    # Table 3 arms (J1R1REF_F4, DS, DFlash, AR, SSD) read any sealed group of 32.
    if model == '8b' and arm in ('A79','DSE_A79','J1R1REF'):
        assert offset == 0, 'A79_FIRST32_ONLY'
    count, tokens = (8, 256) if audit else (32, 2048)
    # Table 3 arms (J3R3DEV_F4, DS, AR) read any sealed group of 32.
    if model == '27b' and arm in ('J1R1REF_F4','J3R3REF_F4',*DEV_VARIANTS_27B):
        assert offset == 0, 'REFERENCE_FIRST32_ONLY'
    if smoke:
        assert model == '27b' and offset == 0 and not audit and arm in ('J1R1REF_F4','J3R3REF_F4',*DEV_ARMS_27B,'DS','AR'), 'SMOKE_CONDITION'
        count, tokens = 8, 512
    if qualification:
        assert model == '27b' and offset == 0 and not audit and not smoke and arm in ('J1R1REF_F4','J3R3REF_F4',*DEV_ARMS_27B,'DS'), 'QUALIFICATION_CONDITION'
        count, tokens = 16, 512
    if ar_qualification:
        assert arm == 'AR' and model == '8b' and not audit
        count, tokens = 8, 512
    group = json.loads(prompts.read_text())
    indices = validate_group(group, original, offset, count)
    assert group['source_sha256'] == HASHES[model][dataset]
    raw = json.loads(output.read_text())
    campaign = json.loads(Path(str(output) + '.campaign.json').read_text())
    condition = campaign['resolved_condition']
    method, recipe, placement = ARMS[model][arm]
    for key, expected in dict(count=count, tokens=tokens, temperature=0, seed=0,
                              stop_at_eos='on', sequential_prompts=True,
                              prompts_sha256=sha256(prompts), method=method,
                              recipe=recipe, regime='graphed', placement=placement).items():
        assert condition[key] == expected, (key, condition[key], expected)
    assert set(condition['eos_token_ids']) == ({151643, 151645} if model == '8b' else {248044, 248046})
    env = condition['environment']
    assert env['SSD_SHARED_GPU_KV_UTILIZATION'] == '0.10'
    assert len(raw['outputs']) == count
    assert all(1 <= len(row['token_ids']) <= tokens for row in raw['outputs'])
    text = log.read_text()
    assert 'SERVE_READ_COMPLETE' in text and 'Traceback (most recent call last)' not in text
    assert not re.search(r'HYBRID_VERIFY_LOGIT_AUDIT .*violations=[1-9]', text), 'AUDIT_VIOLATION'
    measured = campaign['reads'][0]['measured']
    metrics = measured['metrics']
    receipt = dict(model=model, arm=arm, dataset=dataset, audit=audit, count=count, tokens=tokens,
                   original_indices=indices, source_sha256=HASHES[model][dataset],
                   output=str(output), output_sha256=sha256(output),
                   campaign_sha256=sha256(Path(str(output) + '.campaign.json')))
    if model == '27b':
        receipt.update(prompt_rendering='system', original_source_sha256=ORIGINAL_HASHES_27B[dataset],
            table1_rendered_sha256=TABLE1_RENDERED_HASHES_27B[dataset],
            table1_file_identity=HASHES[model][dataset] == TABLE1_RENDERED_HASHES_27B[dataset])
    receipt['compile_pin_sha256'] = env.get('JETSPEC_C1_COMPILE_PIN_SHA')
    receipt['compiler_cache'] = condition.get('compiler_cache')
    receipt['compiler_cache_audit'] = metrics.get('compiler_cache_audit')
    from jetspec.engines.jetspec_plus_plus.read.legacy import startup_summary
    receipt['phase_split'] = startup_summary(text)
    host_path = output.parent / 'host-load.json'
    host = json.loads(host_path.read_text()) if host_path.exists() else {}
    receipt['host_load1'] = host.get('loadavg', [None])[0]
    receipt['node'] = host.get('node')
    receipt.update(ar_steps_inside_eos=0, ar_steps_after_eos=0,
                   capacity_fallback_inside_eos=0, capacity_fallback_after_eos=0)
    if env.get('JETSPEC_ANATOMY_LITE') == '1' and arm in ('J1R1REF','J1R1REF_F4','J3R3REF_F4',*DEV_ARMS_27B,'DS','AR'):
        from jetspec.engines.jetspec_plus_plus.table3.analyze import lite_targets
        targets = lite_targets(metrics)
        assert targets, 'ANATOMY_LITE_MISSING'
        expected = (sum(len(o['token_ids'])-1 for o in raw['outputs']) if method=='ar' else
                    len(metrics['jetspec_round_records' if method=='jetspecpp' else 'blockhead_round_records']))
        assert len(targets)==expected, 'ANATOMY_LITE_ROUND_ALIGNMENT'
        chain_lite = measured.get('chain',{}).get('anatomy_lite')
        if method=='jetspecpp':
            assert chain_lite, 'ANATOMY_LITE_DRAFT_MISSING'
        timing_bytes = len(json.dumps(metrics['anatomy_lite']))+len(json.dumps(chain_lite))
        assert timing_bytes<2_000_000, 'ANATOMY_LITE_BUDGET'
        receipt['anatomy_lite'] = dict(rounds=len(targets),bytes=timing_bytes,
            sha256=hashlib.sha256(json.dumps([metrics['anatomy_lite'],chain_lite],sort_keys=True).encode()).hexdigest())
    if arm in ('J1R1REF_F4','J3R3REF_F4',*DEV_ARMS_27B):
        expected_policy, expected_sha, label = arm_policy(arm)
        policy = json.loads((output.parent / 'policy-receipt.json').read_text())
        assert policy['policy_sha256'] == expected_sha, f'{label}_POLICY_SHA'
        assert env.get('JETSPEC_SERVE_POLICY') == policy['policy'], f'{label}_POLICY_PATH'
        assert Path(policy['policy']).name == expected_policy.name, f'{label}_POLICY_PATH'
        receipt.update(policy)
    if model == '27b' and arm in DEV_VARIANTS_27B:
        served = json.loads((recipe_record(recipe)).read_text())
        assert served['thresholds_sha256'] == (
            CAL28_THRESHOLDS_SHA if arm in CAL28_ARMS else
            '01eaebfbad1d2c7f447e4bf24a55f03b5d55142b128f74f68c63e537c05c4b29'), 'THRESHOLDS_IDENTITY'
        assert condition.get('bundle_artifact') == served['bundle_artifact'], 'THRESHOLDS_IDENTITY'
        assert campaign['reads'][0]['manifest']['thresholds_sha256'] == served['thresholds_sha256'], 'THRESHOLDS_IDENTITY'
        receipt.update(bundle_artifact=served['bundle_artifact'],
                       thresholds_sha256=served['thresholds_sha256'])
    if method == 'jetspecpp':
        assert draft_deadline in ('on', 'off')
        assert condition['draft_deadline'] == ('1' if draft_deadline == 'on' else '0')
        assert env['JETSPEC_CHAIN_MISS_REDRAFT'] == '1'
        assert env['JETSPEC_ROWS_RULE_CAP'] == ('16' if arm in ROWS16_ARMS else '4')
        assert env['JETSPEC_CHAIN_FIXED_CONFIG'] == {'J1R1': 'J1-R1', 'J1R1REF': 'J1-R1', 'J1R1REF_F4': 'J1-R1', 'J3R3REF_F4': 'J3-R3', 'J3R3DEV_F4': 'J3-R3', **dict.fromkeys(DEV_VARIANTS_27B, 'J3-R3'), 'J2R0': 'J2-R0', 'J3R3': 'J3-R3', 'J2R2': 'J2-R2', 'A79': 'J2-R0'}[arm]
        if arm in ('J1R1REF','J1R1REF_F4','J3R3REF_F4'):
            assert env.get('JETSPEC_VERIFY_REFERENCE') == '1', 'REFERENCE_VERIFY_REQUIRED'
        if model == '27b' and arm in ('J1R1REF_F4','J3R3REF_F4'):
            assert env.get('JETSPEC_VERIFY_DEVICE') == '0' and env.get('SSD_ATTENTION_BACKEND') == 'flashinfer', 'REFERENCE_VERIFY_REQUIRED'
        if model == '27b' and arm in DEV_ARMS_27B:
            assert env.get('JETSPEC_VERIFY_DEVICE') == '1', 'DEVICE_VERIFY_REQUIRED'
            assert env.get('JETSPEC_VERIFY_REFERENCE', '0') == '0', 'DEVICE_VERIFY_REQUIRED'
            assert env.get('JETSPEC_HYBRID_VERIFY_FLASHINFER') == '1', 'DEVICE_FLASHINFER_REQUIRED'
            assert env.get('SSD_ATTENTION_BACKEND') == 'flashinfer', 'DEVICE_FLASHINFER_REQUIRED'
            assert env.get('JETSPEC_HYBRID_FUSED_NORM', '0') == '0', 'DS_TARGET_NORM_REQUIRED'
            assert env.get('JETSPEC_HYBRID_FAST_SNAPSHOT') == '1', 'DS_TARGET_SNAPSHOT_REQUIRED'
        if model == '27b' and arm in ('J1R1REF_F4','J3R3REF_F4',*DEV_ARMS_27B):
            assert env.get('JETSPEC_ANATOMY_LITE') == '1', 'ANATOMY_LITE_REQUIRED'
            assert metrics['jetspec_ar_steps'] == metrics.get('jetspec_chain_redraft_missing', 0) == 0, 'REFERENCE_CHAIN_FALLBACK'
            errors = [int(v) for v in re.findall(r'W645_CHAIN_AUDIT .*"error_bits": (\d+)', text)]
            assert errors and max(errors) == 0, 'CHAIN_ERR'
            receipt.update(chain_err=max(errors), chain_audit_rounds=len(errors), redraft_missing=0)
        assert env.get('JETSPEC_REFRESH_SWAP', '0') == ('0' if arm in ('J2R0','A79') else '1')
        if arm == 'A79':
            assert offset == 0, 'A79_FIRST32_ONLY'
            assert env['JETSPEC_CHAIN_J_SWITCH_L'] == '1024'
        receipt.update(eos_region_events(raw['outputs'], metrics, text, tokens,
            condition['eos_token_ids'], indices, measured['chain']['capacity_fallback']))
        if write_receipt:
            (output.parent / 'gate-receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
        for kind, label in [('ar_steps', 'NONZERO_AR_STEPS_INSIDE_EOS'),
                            ('capacity_fallback', 'CAPACITY_FALLBACK_INSIDE_EOS')]:
            if receipt[f'{kind}_inside_eos']:
                first = next(e for e in receipt['eos_region_events'] if e['kind'] == kind and e['region'] == 'inside_eos')
                raise AssertionError(f'{label} original_index={first["original_index"]} seq_id={first["seq_id"]} round={first["round"]}')
        if audit:
            audits = re.findall(r'HYBRID_VERIFY_LOGIT_AUDIT prefill=(\d+) rounds=(\d+) tokens=(\d+) violations=(\d+)', text)
            assert len(audits) == count, 'BF16_AUDIT_MISSING'
            for row, output_row in zip(audits, raw['outputs']):
                prefill, rounds, audited, violations = map(int, row)
                assert prefill == 1 and rounds > 0 and violations == 0
                assert prefill + audited >= len(output_row['token_ids'])
            receipt['bf16_audit'] = audits
        else:
            assert env.get('JETSPEC_HYBRID_VERIFY_AUDIT', '0') == '0'
    elif arm in ('DSE','DSE_A79') and model == '8b':
        assert all(env[key] == value for key, value in DSE_KNOBS.items())
    elif arm == 'DS' and model == '8b':
        assert all(env.get(key, '0') == '0' for key in DSE_KNOBS if key != 'SSD_VERIFY_SYNC_AUDIT'), 'STOCK_DS_ENHANCED'
    elif arm == 'DS' and model == '27b':
        assert env['SSD_VERIFY_DEVICE'] == '1' and env['SSD_ATTENTION_BACKEND'] == 'flashinfer'
        assert all(env.get(key, '0') == '0' for key in DSE_KNOBS
                   if key.startswith('JETSPEC_BLOCKHEAD_')), 'REFERENCE_DS_ENHANCED'
    elif arm == 'AR' and model == '27b':
        assert env.get('SSD_ATTENTION_BACKEND') == 'flashinfer', 'REFERENCE_AR_BACKEND'
    elif method == 'ssd':
        assert condition['draft_repo']=='Qwen/Qwen3-0.6B'
        assert condition['draft_revision']=='c1899de289a04d12100db370d81485cdf75e47ca'
        assert condition['k']==4 and condition['async'] is True
        assert condition['stock_lookahead']==4 and condition['stock_fanout']==(1 if dataset=='math500' else 3)
        from jetspec.engines.jetspec_plus_plus.table3.analyze import per_prompt
        from jetspec.engines.jetspec_plus_plus.table3.analyze import aggregate
        rows = per_prompt('ssd',raw['outputs'],metrics,[],indices,tokens,condition['eos_token_ids'])
        receipt['ssd_eos'] = aggregate(rows)
    return receipt
