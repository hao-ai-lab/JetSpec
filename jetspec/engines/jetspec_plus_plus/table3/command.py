"""Public orchestration for the bundled Table 3 protocol and EOS/rule-A reducer.

Each benchmark has 128 sealed prompts, run as four groups of 32. Throughput and
timing omit each request's first decode round; tau retains it. Hit and conditional
tau use the pre-drafted outcome flag on steady rounds. Repeated reads of a group
are averaged before pooling groups, so repeats do not increase its weight.
"""
import argparse
import contextlib
import csv
import json
import os
from pathlib import Path
from statistics import mean
import subprocess
import sys

from . import protocol
from .analyze import aggregate, per_prompt
from .rule_a import rule_a
from ..read import entry, legacy

ROOT = Path(__file__).resolve().parents[1]
MODELS = {'qwen3-8b': '8b', 'qwen3.8-27b': '27b'}
PAPER_ARMS = {'8b': ('AR', 'DS', 'DFlash', 'SSD', 'J1R1REF_F4'),
              '27b': ('AR', 'DS', 'J3R3DEV_F4')}
NAMES = {'DS': 'DSpark', 'J1R1REF_F4': 'JetSpec++', 'J3R3DEV_F4': 'JetSpec++'}
# Baselines by name. AR (the speedup denominator) and the JetSpec++ arm always run; DSpark is the default
# comparison and all restores the paper population.
BASELINES = {'8b': {'dspark': 'DS', 'dflash': 'DFlash', 'ssd': 'SSD'}, '27b': {'dspark': 'DS'}}
METHOD_ARMS = {'8b': 'J1R1REF_F4', '27b': 'J3R3DEV_F4'}
OFFSETS = (0, 32, 64, 96)
WEIGHTS = dict(tok_s='steady_ms', tau='rounds', round='steady_rounds',
               verify='verify_rounds', hit='cache_rounds',
               tau_hit='hit_rounds', tau_miss='miss_rounds')
# The 27B pair reads run the chain-graph error audit; the protocol requires its zero word every round.
AUDIT = Path(__file__).resolve().parent / 'audit'
AUDITED_27B = ('J1R1REF_F4', 'J3R3REF_F4', *protocol.DEV_ARMS_27B)
# The public Hugging Face release swaps only the JetSpec++ recipes: same serving environment,
# release drafter, adapter and calibration (release_recipes.py derives them from the paper ones).
RELEASE_RECIPES = {
    ('8b', 'J1R1REF_F4'): 'jetspecpp-reference-pair-e6v3-b16-chain-r4-j1r1-lean-swap-fast6-table3-release',
    ('27b', 'J3R3DEV_F4'): 'jetspecpp-device-pair-27b-drafter6ep-p2-cal28-j3r3-swap-dspark-release'}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', choices=MODELS, required=True)
    p.add_argument('--host-values', type=Path)
    p.add_argument('--artifacts-dir', type=Path)
    p.add_argument('--devices', default='0,1')
    p.add_argument('--benchmarks', default='gsm8k,math500')
    p.add_argument('--baselines', default='dspark',
                   help='comma-separated baselines beside AR and JetSpec++: dspark, dflash, ssd (8B only), '
                        'or all for the paper population')
    p.add_argument('--arms', help='comma-separated protocol arm names, overriding --baselines; AR is always included')
    p.add_argument('--offsets', default='0,32,64,96',
                   help='prompt groups to read; fewer than four prints per-group values without pooling')
    p.add_argument('--assets', choices=('paper', 'release'), default='paper',
                   help='paper: the artifacts the paper measured; release: the public Hugging Face '
                        'JetSpec++ release (baselines keep their checkpoints)')
    p.add_argument('--output-dir', type=Path, default=Path('bench/outputs/jetspec_plus_plus_table3'))
    p.add_argument('--resolve-only', action='store_true', help='print config and launch plans on CPU, without weights')
    p.add_argument('--reduce-only', action='store_true', help='validate and reduce existing group outputs')
    p.add_argument('--resume', action='store_true',
                   help='keep groups under --output-dir that already validate; run only the rest')
    return p


def jobs(args):
    model = MODELS[args.model]
    if args.arms:
        arms = args.arms.split(',')
    else:
        names = BASELINES[model]
        chosen = list(names) if args.baselines == 'all' else args.baselines.split(',')
        if set(chosen) - set(names):
            raise ValueError('baselines must be selected from ' + ','.join(names) + ' or all')
        arms = ['AR', *(names[name] for name in chosen), METHOD_ARMS[model]]
    if set(arms) - set(PAPER_ARMS[model]):
        raise ValueError('arms must be selected from ' + ','.join(PAPER_ARMS[model]))
    benchmarks = args.benchmarks.split(',')
    if not benchmarks or set(benchmarks) - {'gsm8k', 'math500'} or len(set(benchmarks)) != len(benchmarks):
        raise ValueError('benchmarks must be distinct gsm8k,math500 names')
    devices = args.devices.split(',')
    if len(devices) != 2 or len(set(devices)) != 2 or not all(d.isdigit() for d in devices):
        raise ValueError('--devices requires two distinct GPU indices')
    offsets = [int(o) for o in args.offsets.split(',')]
    if set(offsets) - set(OFFSETS) or len(set(offsets)) != len(offsets):
        raise ValueError('offsets must be distinct values from 0,32,64,96')
    for dataset in benchmarks:
        for arm in PAPER_ARMS[model]:
            if arm not in arms and arm != 'AR':
                continue
            for offset in offsets:
                yield dict(model=model, dataset=dataset, arm=arm, offset=offset,
                           devices=args.devices if protocol.ARMS[model][arm][2] == 'pair' else devices[0])


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def sealed_source(model, dataset, tokenizer=None):
    name = ('27b-sys' if model == '27b' else model) + f'/prompts_{dataset}128.json'
    path = protocol.INPUTS / name
    seals = json.loads((protocol.INPUTS / 'seals.json').read_text())
    assert protocol.sha256(path) == seals[name], 'SEALED_SOURCE_SHA'
    source = json.loads(path.read_text())
    assert [r['index'] for r in source['prompts']] == list(range(128)), 'SEALED_SOURCE_INDICES'
    if tokenizer is not None:
        for row in source['prompts']:
            assert tokenizer.encode(row['text'], add_special_tokens=False) == row['token_ids'], 'TOKEN_IDENTITY'
    return source


def prepare(job, directory, args):
    directory.mkdir(parents=True, exist_ok=True)
    source = sealed_source(job['model'], job['dataset'])
    group = protocol.select_group(source, job['offset'], 32, protocol.HASHES[job['model']][job['dataset']])
    prompts = directory / 'prompts.json'
    if not args.reduce_only:
        write_json(prompts, group)
    method, name, _ = protocol.ARMS[job['model']][job['arm']]
    recipe = legacy.load_recipe(ROOT, method, 'graphed', name)
    # All paper heads have budget 16; SSD and AR use the same minimum reservation.
    # Execution additionally checks the actual local model/head configs below.
    lookahead = max(legacy.MIN_LOOKAHEAD, int(recipe['environment'].get('JETSPEC_BLOCK_SIZE_READ', '16')))
    needed = max(len(p['token_ids']) for p in group['prompts']) + 2048 + lookahead
    capacity = max(legacy.MAX_MODEL_LEN_FLOOR, -(-needed // legacy.KV_BLOCK_SIZE) * legacy.KV_BLOCK_SIZE)
    if not args.resolve_only and not args.reduce_only:
        host = legacy.load_host_values(args.host_values)
        blockhead = host['artifacts'].get(recipe.get('blockhead_artifact'))
        if blockhead is None and args.artifacts_dir and recipe.get('blockhead_artifact'):
            blockhead, _ = legacy.artifact_directory(recipe['blockhead_artifact'], legacy._manifest_artifacts(),
                                                     host, args.artifacts_dir)
        capacity = legacy.resolve_max_model_len(prompts, 32, 2048, method, recipe['environment'],
                    host['environment']['JETSPEC_TARGET_MODEL_PATH'], blockhead,
                    forced=recipe.get('max_model_len'))['value']
    argv = protocol.build_argv(job['arm'], job['dataset'], job['devices'], directory, prompts,
                               recipe.get('max_model_len', capacity), model=job['model'],
                               allowed_devices=tuple(args.devices.split(',')))
    # One persistent compiler cache root for the whole run, which the reader's --cache-root expects
    # ("shared by serial reads"). A fresh root per group recompiles every group, and a pinned pair's
    # cold warmup can then exceed the 10-minute NCCL timeout of the first pair receive.
    argv[argv.index('--cache-root') + 1] = str((args.output_dir / 'compiler-cache').resolve())
    if args.host_values:
        argv[argv.index('--host-values') + 1] = str(args.host_values.resolve())
    if args.artifacts_dir:
        argv += ['--artifacts-dir', str(args.artifacts_dir.resolve())]
    return source, argv


def set_aside(directory):
    """Move a group's earlier reader outputs to invalid/<n>/; the reader refuses to overwrite them."""
    paths = sorted(directory.glob('read.json*'))
    if not paths:
        return None
    n = 1
    while (directory / 'invalid' / str(n)).exists():
        n += 1
    target = directory / 'invalid' / str(n)
    target.mkdir(parents=True)
    for path in paths + [p for p in (directory / 'read.log', directory / 'validated-reads.json') if p.exists()]:
        path.replace(target / path.name)
    return target


@contextlib.contextmanager
def assets(name):
    """Point the JetSpec++ arms at the chosen assets; every other condition and check stays the paper's."""
    saved = {key: protocol.ARMS[key[0]][key[1]] for key in RELEASE_RECIPES}
    try:
        if name == 'release':
            for (model, arm), recipe in RELEASE_RECIPES.items():
                method, _, placement = saved[model, arm]
                protocol.ARMS[model][arm] = (method, recipe, placement)
        yield
    finally:
        for (model, arm), value in saved.items():
            protocol.ARMS[model][arm] = value


def read_environment(job, env):
    if job['model'] == '27b' and job['arm'] in AUDITED_27B:
        return dict(env, TABLE3_CHAIN_AUDIT='1',
                    PYTHONPATH=os.pathsep.join(filter(None, [str(AUDIT), env.get('PYTHONPATH')])))
    return env


def launch_plan(job, argv):
    """Resolve the same named composition and protocol arguments without artifacts."""
    args = legacy._parser().parse_args(argv[3:])
    rows = json.loads((ROOT / 'read/recipe-index.json').read_text())['rows']
    row = next(r for r in rows if r.get('launch', {}).get('name') == args.recipe)
    overrides = list(row['overrides']) + [
        'hardware.placement=' + json.dumps(args.placement),
        'hardware.devices=' + json.dumps(list(range(len(args.devices.split(','))))),
        'hardware.cpu_pin=' + json.dumps(args.cpu_pin), 'benchmark.count=32', 'benchmark.cap=2048',
        'benchmark.stop_at_eos=true', 'benchmark.temperature=0', 'benchmark.seed=0', 'benchmark.repeats=1']
    record = entry.resolve_read(row['names'], overrides, root=ROOT / 'config', launch=row['launch'],
                                output=args.output_json, sequential_prompts=True,
                                cache_root=args.cache_root, wire_records='off', anatomy='off')
    # The protocol supplies the exact runtime knobs, policy and deadline flags.
    runtime_recipe = legacy.load_recipe(ROOT, args.method, args.regime, args.recipe)
    return dict(**job, composition=record, runtime_recipe=runtime_recipe, argv=argv)


def reduce_rows(rows):
    """Apply the backend's rule A, then retain the paper's sufficient statistics."""
    cut = rule_a([dict(token_ids=r['output_token_ids']) for r in rows], rows,
                 tokens=rows[0]['token_cap'])
    rows, stats = cut['rows'], cut['stats']
    steady = [v for r in rows for v in r['round_ms_values'][1:]]
    verify = [v for r in rows for v in r.get('paper_verify_host_ms_values', r['verify_host_ms_values'])[1:]
              if v is not None]
    cache = [v for r in rows for v in r.get('cache_hit_values', [])[1:]]
    split = [(h, c) for r in rows for h, c in zip(r.get('cache_hit_values', [])[1:],
                                                 r['credited_tokens_values'][1:])]
    hits = sum(bool(h) for h, _ in split)
    misses = len(split) - hits
    values = dict(tok_s=cut['tok_s'], tau=cut['tau'], round=cut['round_mean_ms'],
                  verify=cut['verify_mean_ms'], hit=cut['cache_hit'],
                  tau_hit=sum(c for h, c in split if h)/hits if hits else None,
                  tau_miss=sum(c for h, c in split if not h)/misses if misses else None)
    weights = dict(rounds=stats['rounds'], steady_ms=sum(r['steady_ms'] for r in rows),
                   steady_rounds=len(steady), verify_rounds=len(verify), cache_rounds=len(cache),
                   hit_rounds=hits, miss_rounds=misses)
    return dict(values=values, weights=weights, loop_count=cut['loop_count'])


def reduce_directory(directory, job, source):
    output = directory / 'read.json'
    saved_path = directory / 'validated-reads.json'
    if saved_path.exists():
        saved = json.loads(saved_path.read_text())[0]
        assert protocol.sha256(output) == saved['output_sha256'], 'OUTPUT_RECEIPT_SHA'
        assert protocol.sha256(Path(str(output) + '.campaign.json')) == saved['campaign_sha256'], 'CAMPAIGN_RECEIPT_SHA'
    receipt = protocol.validate_read(output, directory/'read.log', job['arm'], job['dataset'],
                                     directory/'prompts.json', source, job['offset'], model=job['model'])
    campaign = json.loads(Path(str(output) + '.campaign.json').read_text())
    raw = json.loads(output.read_text())
    measured = campaign['reads'][0]['measured']
    populations = json.loads(Path(str(output) + '.populations.json').read_text())['populations']
    rows = per_prompt(campaign['resolved_condition']['method'], raw['outputs'], measured['metrics'],
                       populations, receipt['original_indices'], 2048,
                       campaign['resolved_condition']['eos_token_ids'],
                       chain_lite=measured.get('chain', {}).get('anatomy_lite'))
    eos = aggregate(rows)
    native = receipt['ssd_eos'] if job['arm'] == 'SSD' else measured
    assert abs(eos['tps_pooled']/native.get('tps_eos_stopped',native.get('tps_pooled'))-1) < 1e-10, 'NATIVE_TPS_MISMATCH'
    assert abs(eos['tau_eos']/native.get('tau_eos_stopped',native.get('tau_eos'))-1) < 1e-10, 'NATIVE_TAU_MISMATCH'
    result = dict(**job, **reduce_rows(rows), receipt=receipt)
    write_json(directory/'validated-reads.json', [receipt])
    return result


def pool(reads):
    result = []
    for model, dataset, arm in dict.fromkeys((r['model'], r['dataset'], r['arm']) for r in reads):
        offsets = {r['offset'] for r in reads if (r['model'],r['dataset'],r['arm']) == (model,dataset,arm)}
        if offsets != set(OFFSETS):
            raise ValueError(f'incomplete population: {model}/{dataset}/{arm}: {sorted(offsets)}')
        groups = []
        for offset in OFFSETS:
            runs = [r for r in reads if (r['model'], r['dataset'], r['arm'], r['offset']) == (model,dataset,arm,offset)]
            if not runs:
                raise ValueError(f'incomplete population: {model}/{dataset}/{arm}/{offset}')
            groups.append(dict(values={k: mean(vs) if (vs := [r['values'][k] for r in runs if r['values'][k] is not None]) else None
                                        for k in WEIGHTS},
                               weights={k: mean(r['weights'][k] for r in runs) for k in set(WEIGHTS.values())}))
        values = {}
        for key, weight in WEIGHTS.items():
            pairs = [(g['values'][key], g['weights'][weight]) for g in groups
                     if g['values'][key] is not None and g['weights'][weight]]
            values[key] = sum(v*w for v,w in pairs)/sum(w for _,w in pairs) if pairs else None
        values['exposed'] = values['round'] - values['verify'] if values['verify'] is not None else None
        result.append(dict(model=model, dataset=dataset, arm=arm,
                           gpus=2 if protocol.ARMS[model][arm][2] == 'pair' else 1, **values))
    for row in result:
        ar = next(r for r in result if (r['model'],r['dataset'],r['arm']) == (row['model'],row['dataset'],'AR'))
        row['speedup'] = row['tok_s']/ar['tok_s']
    return result


def formatted(row):
    def cell(key, digits=2, scale=1):
        return 'N/A' if row[key] is None else f'{row[key]*scale:.{digits}f}'
    return [row['model'], row['dataset'], NAMES.get(row['arm'], row['arm']), str(row['gpus']),
            f"{cell('tok_s',1)} ({cell('speedup')}x)", cell('tau'),
            cell('hit',1,100) + ('%' if row['hit'] is not None else ''),
            cell('tau_hit'), cell('tau_miss'), cell('round'), cell('verify'), cell('exposed')]


def main(argv=None):
    args = parser().parse_args(argv)
    if not args.resolve_only and not args.reduce_only and args.host_values is None:
        parser().error('--host-values is required for execution')
    with assets(args.assets):
        return run(args)


def run(args):
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    plans, reads = [], []
    repo = ROOT.parents[2]
    env = dict(os.environ, HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1',
               PYTHONPATH=os.pathsep.join(filter(None, [str(repo), os.environ.get('PYTHONPATH')])))
    for job in jobs(args):
        directory = output / job['dataset'] / job['arm'] / f"p{job['offset']:03d}"
        source, command = prepare(job, directory, args)
        plan = launch_plan(job, command)
        plans.append(plan)
        if args.resolve_only:
            print(json.dumps(plan, sort_keys=True))
            continue
        done = None
        if args.resume and (directory/'read.json').exists():
            try:
                done = reduce_directory(directory, job, source)
            except (AssertionError, KeyError, OSError, ValueError) as error:
                print(f"RERUN {job['dataset']} {job['arm']} p{job['offset']:03d}: {error}", flush=True)
        if done is None and not args.reduce_only:
            if args.resume:
                set_aside(directory)
            for attempt in (1, 2):
                with (directory/'read.log').open('w') as log:
                    rc = subprocess.run(command, cwd=repo, stdout=log, stderr=subprocess.STDOUT,
                                        env=read_environment(job, env)).returncode
                if rc == 0:
                    break
                if attempt == 2:
                    raise subprocess.CalledProcessError(rc, command)
                # A cold compiler cache can push a pinned pair's warmup past the NCCL first-receive
                # timeout. The kernels it compiled stay in the shared cache, so one rerun starts warm.
                (directory/'read.log').replace(directory/'read.failed.log')
                set_aside(directory)
                print(f"RETRY {job['dataset']} {job['arm']} p{job['offset']:03d}: exit {rc}", flush=True)
        reads.append(done or reduce_directory(directory, job, source))
    write_json(output/'plans.json', plans)
    if args.resolve_only:
        return 0
    if sorted(int(o) for o in args.offsets.split(',')) != list(OFFSETS):
        # A subset check: per-group values only, since the paper pools all four groups.
        write_json(output/'groups.json', reads)
        for read in reads:
            print(read['dataset'], NAMES.get(read['arm'], read['arm']), f"p{read['offset']:03d}",
                  json.dumps({k: round(v, 3) for k, v in read['values'].items() if v is not None}))
        return 0
    table = pool(reads)
    write_json(output/'table3.json', dict(reads=reads, rows=table))
    with (output/'table3.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    print('model | benchmark | method | GPUs | tok/s (speedup) | tau | hit | tau_hit | tau_miss | round | verify | exposed')
    for row in table:
        print(' | '.join(formatted(row)))
    return 0
