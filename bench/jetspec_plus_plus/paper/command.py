"""Public orchestration for the bundled Table 3 protocol and EOS/rule-A reducer.

Each benchmark has 128 sealed prompts, run as four groups of 32. Throughput and
timing omit each request's first decode round; tau retains it. Hit and conditional
tau use the pre-drafted outcome flag on steady rounds. Repeated reads of a group
are averaged before pooling groups, so repeats do not increase its weight.
"""
import argparse
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
from jetspec.engines.jetspec_plus_plus.launch import runner

ROOT = Path(__file__).resolve().parents[3] / 'jetspec/engines/jetspec_plus_plus'
MODELS = {'qwen3-8b': '8b', 'qwen3.8-27b': '27b'}
PAPER_ARMS = {'8b': ('AR', 'DS', 'JetSpec++'), '27b': ('AR', 'DS', 'JetSpec++')}
NAMES = {'DS': 'DSpark'}
BASELINES = {'8b': {'dspark': 'DS'}, '27b': {'dspark': 'DS'}}
METHOD_ARMS = {'8b': 'JetSpec++', '27b': 'JetSpec++'}
OFFSETS = (0, 32, 64, 96)
WEIGHTS = dict(tok_s='steady_ms', tau='rounds', round='steady_rounds',
               verify='verify_rounds', hit='cache_rounds',
               tau_hit='hit_rounds', tau_miss='miss_rounds')
# The 27B pair reads run the chain-graph error audit; the protocol requires its zero word every round.
AUDIT = Path(__file__).resolve().parent / 'audit'
AUDITED_27B = ('JetSpec++',)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', choices=MODELS, required=True)
    p.add_argument('--host-values', type=Path)
    p.add_argument('--artifacts-dir', type=Path)
    p.add_argument('--devices', default='0,1')
    p.add_argument('--benchmarks', default='gsm8k,math500')
    p.add_argument('--baselines', default='dspark',
                   help='comma-separated baselines beside AR and JetSpec++: dspark, '
                        'or all for the paper population')
    p.add_argument('--arms', help='comma-separated protocol arm names, overriding --baselines; AR is always included')
    p.add_argument('--offsets', default='0,32,64,96',
                   help='prompt groups to read; fewer than four prints per-group values without pooling')
    p.add_argument('--assets', choices=('release',), default='release', help='public release checkpoints')
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
    from bench.jetspec_plus_plus.dataloader import pinned_prompts
    path = pinned_prompts(protocol.SOURCES[model], dataset)
    assert protocol.sha256(path) == protocol.HASHES[model][dataset], 'SEALED_SOURCE_SHA'
    source = json.loads(path.read_text())
    assert [r['index'] for r in source['prompts']] == list(range(128)), 'SEALED_SOURCE_INDICES'
    if tokenizer is not None:
        assert all(tokenizer.encode(r['text'], add_special_tokens=False) == r['token_ids']
                   for r in source['prompts']), 'TOKEN_IDENTITY'
    return source


def prepare(job, directory, args):
    directory.mkdir(parents=True, exist_ok=True)
    source = sealed_source(job['model'], job['dataset'])
    group = protocol.select_group(source, job['offset'], 32, protocol.HASHES[job['model']][job['dataset']])
    prompts = directory / 'prompts.json'
    if not args.reduce_only:
        write_json(prompts, group)
    record = protocol.read_record(job, directory / 'read.json', args.output_dir / 'compiler-cache')
    recipe = record['recipe']
    lookahead = max(runner.MIN_LOOKAHEAD, int(recipe['environment'].get('JETSPEC_BLOCK_SIZE_READ', '16')))
    needed = max(len(p['token_ids']) for p in group['prompts']) + 2048 + lookahead
    capacity = max(runner.MAX_MODEL_LEN_FLOOR, -(-needed // runner.KV_BLOCK_SIZE) * runner.KV_BLOCK_SIZE)
    argv = protocol.build_argv(job, directory, prompts, capacity, record, args)
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





def read_environment(job, env):
    if job['model'] == '27b' and job['arm'] in AUDITED_27B:
        return dict(env, TABLE3_CHAIN_AUDIT='1',
                    PYTHONPATH=os.pathsep.join(filter(None, [str(AUDIT), env.get('PYTHONPATH')])))
    return env


def launch_plan(job, argv):
    args = runner._parser().parse_args(argv[3:])
    record = protocol.read_record(job, args.output_json, args.cache_root)
    return dict(**job, composition=record, runtime_recipe=record['recipe'], argv=argv)


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
    record = json.loads((directory/'plan.json').read_text())['composition']
    receipt = protocol.validate_read(output, directory/'read.log', job, directory/'prompts.json', source, record)
    campaign = json.loads(Path(str(output) + '.campaign.json').read_text())
    raw = json.loads(output.read_text())
    measured = campaign['reads'][0]['measured']
    populations = json.loads(Path(str(output) + '.populations.json').read_text())['populations']
    rows = per_prompt(campaign['resolved_condition']['method'], raw['outputs'], measured['metrics'],
                       populations, receipt['original_indices'], 2048,
                       campaign['resolved_condition']['eos_token_ids'],
                       chain_lite=measured.get('chain', {}).get('anatomy_lite'))
    eos = aggregate(rows)
    native = measured
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
        if not args.reduce_only:
            write_json(directory/'plan.json', plan)
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
                    rc = subprocess.run([sys.executable, '-m', 'bench.jetspec_plus_plus.paper.read',
                                         str(directory/'plan.json')], cwd=repo, stdout=log, stderr=subprocess.STDOUT,
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
