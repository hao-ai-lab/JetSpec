"""One offline read through the fast graphed serving engine."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

ROOT = (Path(__file__).resolve().parent.parent.parent.parent / '.')
sys.path[:0] = [str(ROOT), str(ROOT / 'engine'), str(ROOT / 'method')]
from jetspec.engines.jetspec_plus_plus.jetspec.profile import read_counters
from jetspec.engines.jetspec_plus_plus.legacy_env import environment_name


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', default='8B', choices=('8B', '27B'))
    p.add_argument('--j', type=int, choices=range(4), default=1)
    p.add_argument('--r', type=int, choices=range(4), default=1)
    p.add_argument('--budget', type=int, choices=(16, 32), default=16)
    p.add_argument('--rows', type=int, choices=(4, 8, 16), default=16)
    p.add_argument('--drafting-mode', choices=('pipelined', 'critical-path'), default='pipelined')
    p.add_argument('--dtype', choices=('bf16', 'fp32'), default='bf16')
    p.add_argument('--temperature', type=float, default=0)
    p.add_argument('--prompts-json', type=Path, required=True)
    p.add_argument('--dataset', default='gsm8k')
    p.add_argument('--count', type=int, default=128)
    p.add_argument('--cap', type=int, default=2048)
    p.add_argument('--host-values', type=Path, required=True)
    p.add_argument('--artifacts-dir', type=Path, required=True)
    p.add_argument('--cache-root', type=Path, required=True)
    p.add_argument('--compile-pin', type=Path)
    p.add_argument('--compile-pin-sha')
    p.add_argument('--devices', default=os.environ.get('CUDA_VISIBLE_DEVICES') or '0')
    p.add_argument('--placement', choices=('one_gpu_mps', 'pair'), default='one_gpu_mps',
                   help='pair uses --devices VERIFY,DRAFT on two claimed cards')
    p.add_argument('--mps-root', type=Path, help='private MPS directory; defaults beside read.json')
    p.add_argument('--output-json', type=Path, required=True)
    p.add_argument('--metrics-kit', type=Path)
    p.add_argument('--resolve-only', action='store_true')
    p.add_argument('--ar', action='store_true', help='same target AR control')
    p.add_argument('--host-fast', choices=('0', '1'), help='host staging control')
    p.add_argument('--draft-delay-ms', type=float, default=0,
                   help='eval-only delay before chain replay; timing-independence control')
    p.add_argument('--refresh-mode', choices=('full', 'states_only'), default='full')
    p.add_argument('--capture-round-candidates', action='store_true')
    p.add_argument('--capture-decoder-inputs', action='store_true',
                   help='also record each proposal decoder window and its prefix taps (needs --capture-round-candidates)')
    return p


def command(args, *, run_config=None):
    from jetspec.engines.jetspec_plus_plus.jetspec.config_values import config_values
    if bool(args.compile_pin) != bool(args.compile_pin_sha):
        raise ValueError('--compile-pin and --compile-pin-sha must be paired')
    if not math.isfinite(args.draft_delay_ms) or args.draft_delay_ms < 0:
        raise ValueError('eval draft delay must be finite nonnegative milliseconds')
    if args.ar and (args.host_fast is not None or args.draft_delay_ms):
        raise ValueError('AR does not accept draft controls')
    if args.capture_decoder_inputs and not args.capture_round_candidates:
        raise ValueError('--capture-decoder-inputs requires --capture-round-candidates')
    if args.ar and (args.refresh_mode != 'full' or args.capture_round_candidates):
        raise ValueError('eval refresh and capture flags are unavailable for AR')
    if not 0 <= args.r <= args.j <= 3:
        raise ValueError('eval requires J0..3 with R0..J')
    hybrid = args.model == '27B'
    if hybrid and not args.ar:
        if args.placement != 'pair':
            raise ValueError('27B eval requires pair placement')
        if (args.j, args.r) not in ((1, 1), (3, 3)):
            raise ValueError('27B eval requires J1-R1 or J3-R3')
        if args.rows != 16:
            raise ValueError('27B eval requires rows 16')
    suffix = '-fp32' if args.dtype == 'fp32' else ''
    if hybrid:
        recipe = ('ar-hybrid27b-fast-graphed-sm80' if args.ar
                  else f'eval27b-j{args.j}r{args.r}-b{args.budget}')
    else:
        recipe = 'ar-graphed' if args.ar else f'eval8b-j{args.j}r{args.r}-b{args.budget}{suffix}'
    devices = [device.strip() for device in args.devices.split(',')]
    count = 2 if args.placement == 'pair' else 1
    if len(devices) != count or not all(devices) or len(set(devices)) != count:
        number = 'two distinct' if count == 2 else 'one'
        raise ValueError(f'the eval read needs exactly {number} claimed device' + ('s' if count == 2 else ''))
    if args.ar and args.placement == 'pair':
        raise ValueError('AR needs one verify card; use --devices VERIFY without --placement pair')
    argv = [sys.executable, '-B', '-u', str((ROOT / 'engine/jetspec.engines.jetspec_plus_plus/read/legacy.py')),
            '--method', 'ar' if args.ar else 'jetspecpp', '--regime', 'graphed',
            '--recipe', recipe, '--placement', 'one_gpu' if args.ar else args.placement,
            '--devices', ','.join(devices),
            '--dataset', args.dataset, '--prompts-json', str(args.prompts_json),
            '--count', str(args.count), '--tokens', str(args.cap), '--batch', '1',
            '--temperature', str(args.temperature), '--seed', '0', '--stop-at-eos', 'on',
            '--host-values', str(args.host_values), '--artifacts-dir', str(args.artifacts_dir),
            '--cache-root', str(args.cache_root), '--cpu-pin', 'off',
            '--child-wall-seconds', '14400', '--output-json', str(args.output_json)]
    if args.compile_pin:
        argv += ['--compile-pin', str(args.compile_pin), '--compile-pin-sha', args.compile_pin_sha]
    if hybrid and args.dtype == 'fp32':
        argv += ['--knob', 'JETSPEC_TARGET_DTYPE=fp32',
                 '--knob', 'JETSPEC_HYBRID_FP32_LINEAR_BF16_STORAGE=1']
    if args.ar:
        if args.dtype == 'fp32' and not hybrid:
            argv += ['--knob', 'JETSPEC_TARGET_DTYPE=fp32',
                     '--knob', 'SSD_ATTENTION_BACKEND=sdpa']
    else:
        if args.placement == 'one_gpu_mps':
            argv += ['--mps-root', str(args.mps_root or args.output_json.parent/'mps')]
        argv += ['--causal-n1', '--wire-records', 'on' if args.capture_round_candidates else 'off', '--draft-deadline', 'off',
                 '--chain-warm-reachable', 'on']
        knobs = dict(JETSPEC_EVAL_ROWS=str(args.rows),
                     JETSPEC_EVAL_J3_LABEL=config_values(run_config).get('method.method_settings.eval_j3_label', 'J3'),
                     JETSPEC_ROWS_RULE_CAP=str(args.rows), JETSPEC_CHAIN_GRAPH_CHECK='0',
                     JETSPEC_EVAL_DRAFT_DELAY_MS=str(args.draft_delay_ms))
        from jetspec.engines.jetspec_plus_plus.engine import Engine
        Engine.from_environment(knobs, profile=args.drafting_mode)
        if args.host_fast is not None:
            knobs[environment_name('engine', 'engine_settings.host_fast')] = args.host_fast
        if args.refresh_mode != 'full':
            knobs[environment_name('method', 'method_settings.eval_refresh_mode')] = args.refresh_mode
        if args.capture_round_candidates:
            knobs.update(JETSPEC_EVAL_CAPTURE_ROUND_CANDIDATES='decoder' if args.capture_decoder_inputs else '1',
                         JETSPEC_CHAIN_WIRE_RECORD='1', E04_RECORD_PTOP1='1',
                         E04_RECORD_QSHADOW='1')
        for key, value in knobs.items():
            argv += ['--knob', key + '=' + value]
    if args.resolve_only:
        argv += ['--resolve-only']
    return argv


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
                    from jetspec.engines.jetspec_plus_plus.jetspec.pass_labels import normalized_label
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


def main():
    from jetspec.engines.jetspec_plus_plus.jetspec.config_values import launch_config
    from jetspec.engines.jetspec_plus_plus.jetspec.config_values import config_values
    run_config = launch_config(os.environ)
    args = parser().parse_args()
    argv = command(args, run_config=run_config)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    Path(str(args.output_json)+'.command.json').write_text(json.dumps(argv, indent=2)+'\n')
    print('EVAL_COMMAND '+shlex.join(argv), flush=True)
    log_path = Path(str(args.output_json)+'.log')
    started = time.perf_counter()
    environment = dict(os.environ, CUBLAS_WORKSPACE_CONFIG=':4096:8')
    with log_path.open('w') as log:
        done = subprocess.run(argv, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
    wall = time.perf_counter() - started
    if done.returncode:
        print(log_path.read_text()[-18000:], flush=True)
        raise SystemExit(done.returncode)
    if args.resolve_only:
        print(log_path.read_text(), flush=True)
        return
    raw = json.loads(args.output_json.read_text())
    condition = next(json.loads(line.split(' ', 1)[1]) for line in log_path.read_text().splitlines()
                     if line.startswith('RESOLVED_CONDITION '))
    raw['eval'] = dict(model=args.model, j=args.j, r=args.r, budget=args.budget,
                       rows=args.rows, drafting_mode=args.drafting_mode,
                       j3_label=config_values(run_config).get('method.method_settings.eval_j3_label', 'J3'),
                       dtype=args.dtype, placement=args.placement, process_wall_seconds=wall,
                       served_wall_seconds=raw['metrics']['serve_read']['measured']['wall_seconds'],
                       draft_delay_ms=args.draft_delay_ms, host_fast=args.host_fast,
                       eos_token_ids=condition['eos_token_ids'])
    if not args.ar:
        raw['eval']['counters'] = read_counters(log_path.read_text(), raw['metrics'])
        print('EVAL_COUNTERS '+json.dumps(raw['eval']['counters'], sort_keys=True), flush=True)
        from jetspec.engines.jetspec_plus_plus.read.analysis import anatomy
        raw['eval']['anatomy'] = anatomy(raw, log=log_path.read_text())
        print('EVAL_ANATOMY '+json.dumps(raw['eval']['anatomy'], sort_keys=True), flush=True)
        if args.refresh_mode != 'full':
            raw['eval']['refresh_mode'] = args.refresh_mode
        if args.capture_round_candidates:
            raw['eval']['round_candidates'] = export_candidates(args.output_json, raw['metrics'])
        if args.capture_decoder_inputs:
            raw['eval']['decoder_inputs'] = export_decoder_inputs(args.output_json, raw['metrics'])
    if args.metrics_kit:
        sys.path.insert(0, str(args.metrics_kit))
        import paper_metrics
        paper_metrics.EOS = raw['eval']['eos_token_ids']
        raw['eval']['paper_metrics'] = paper_metrics.reduce_read(
            raw, 'ar' if args.ar else 'jetspecpp', loop_cut=True)
        print('EVAL_PAPER_METRICS '+json.dumps(raw['eval']['paper_metrics']), flush=True)
    args.output_json.write_text(json.dumps(raw, indent=2)+'\n')
    print('EVAL_READ '+str(args.output_json), flush=True)


if __name__ == '__main__':
    main()
