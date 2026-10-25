"""CPU contracts for paper Table 3 orchestration."""
import gzip
import json
from pathlib import Path
import re

import pytest

from jetspec.engines.jetspec_plus_plus.table3 import command as c
from jetspec.engines.jetspec_plus_plus.table3.command import jobs, parser, sealed_source


def test_paper_population_and_prompt_seals():
    for model, expected in [('qwen3-8b', 40), ('qwen3.8-27b', 24)]:
        args = parser().parse_args(['--model', model, '--resolve-only', '--baselines', 'all'])
        work = list(jobs(args))
        assert len(work) == expected
        assert {r['offset'] for r in work} == {0, 32, 64, 96}
        for dataset in ('gsm8k', 'math500'):
            source = sealed_source(work[0]['model'], dataset)
            assert len(source['prompts']) == 128


def test_default_compares_with_dspark_and_all_restores_the_paper_population():
    for model, method in [('qwen3-8b', 'J1R1REF_F4'), ('qwen3.8-27b', 'J3R3DEV_F4')]:
        work = list(jobs(parser().parse_args(['--model', model, '--resolve-only'])))
        assert sorted({job['arm'] for job in work}) == sorted(['AR', 'DS', method])
        assert len(work) == 24
    work = list(jobs(parser().parse_args(['--model', 'qwen3-8b', '--resolve-only', '--baselines', 'dflash,ssd'])))
    assert {job['arm'] for job in work} == {'AR', 'DFlash', 'SSD', 'J1R1REF_F4'}
    with pytest.raises(ValueError, match='baselines must be selected from dspark or all'):
        list(jobs(parser().parse_args(['--model', 'qwen3.8-27b', '--resolve-only', '--baselines', 'ssd'])))


def test_all_paper_arms_resolve_without_weights(tmp_path):
    for model in c.MODELS:
        args = parser().parse_args(['--model', model, '--resolve-only', '--baselines', 'all'])
        for job in jobs(args):
            directory = tmp_path / model / job['dataset'] / job['arm'] / str(job['offset'])
            _, argv = c.prepare(job, directory, args)
            plan = c.launch_plan(job, argv)
            assert plan['composition']['recipe']['name'] == c.protocol.ARMS[job['model']][job['arm']][1]
            assert plan['runtime_recipe']['name'] == plan['composition']['recipe']['name']
            assert argv[argv.index('--count')+1] == '32'
            assert argv[argv.index('--tokens')+1] == '2048'


def test_gb200_pinned_pair_arms(tmp_path):
    # The JetSpec++ GB200 pair arms pin each rank to its GPU's NUMA CPUs; the other arms stay unpinned.
    pins = {}
    for model in c.MODELS:
        args = parser().parse_args(['--model', model, '--resolve-only', '--baselines', 'all'])
        for job in jobs(args):
            directory = tmp_path / model / job['dataset'] / job['arm'] / str(job['offset'])
            _, argv = c.prepare(job, directory, args)
            pin = argv[argv.index('--cpu-pin') + 1]
            assert pin == ('auto' if (job['model'], job['arm']) in c.protocol.GB200_PINNED_ARMS else 'off')
            assert c.launch_plan(job, argv)['composition']['config']['hardware']['cpu_pin'] == pin
            pins[job['model'], job['arm']] = pin
    assert pins[('8b', 'J1R1REF_F4')] == pins[('27b', 'J3R3DEV_F4')] == 'auto'
    assert pins[('8b', 'SSD')] == 'off'


def test_27b_one_gpu_rows_name_the_flashinfer_opt_in(tmp_path):
    # The paper's 27B AR and DSpark rows ran FlashInfer on sm100; their D256 hybrid route needs the opt-in.
    from jetspec.engines.jetspec_plus_plus.jetspec import env
    args = parser().parse_args(['--model', 'qwen3.8-27b', '--resolve-only'])
    for job in jobs(args):
        _, argv = c.prepare(job, tmp_path / job['dataset'] / job['arm'] / str(job['offset']), args)
        knobs = [argv[i + 1] for i, a in enumerate(argv) if a == '--knob']
        assert ('JETSPEC_HYBRID_VERIFY_FLASHINFER=1' in knobs) == (job['arm'] in ('AR', 'DS'))
    name = 'JETSPEC_HYBRID_VERIFY_FLASHINFER'
    env.validate_environment({name: '1'}, jetspec=False, max_num_seqs=1)
    with pytest.raises(env.EnvironmentContractError, match='requires N=1 JetSpec'):
        env.validate_environment({name: '1'}, jetspec=False, max_num_seqs=2)
    with pytest.raises(env.EnvironmentContractError, match='requires'):
        env.validate_environment({name: '1'}, jetspec=True, max_num_seqs=1)


def test_27b_pair_reads_run_the_chain_audit_that_validation_requires():
    import os
    base = {'PYTHONPATH': 'repo'}
    audited = c.read_environment(dict(model='27b', arm='J3R3DEV_F4'), base)
    assert audited['TABLE3_CHAIN_AUDIT'] == '1'
    assert audited['PYTHONPATH'].split(os.pathsep) == [str(c.AUDIT), 'repo']
    assert (c.AUDIT / 'sitecustomize.py').is_file()
    # The reader's strict launch contract must pass the switch through to the rank processes.
    from jetspec.engines.jetspec_plus_plus.run_config import child_environment, launch_overlay
    launch_overlay(audited)
    assert child_environment(audited)['TABLE3_CHAIN_AUDIT'] == '1'
    for model, arm in [('27b', 'AR'), ('27b', 'DS'), ('8b', 'J1R1REF_F4'), ('8b', 'SSD')]:
        assert c.read_environment(dict(model=model, arm=arm), base) is base


def test_release_assets_swap_only_the_jetspec_plus_plus_recipes(tmp_path):
    from jetspec.engines.jetspec_plus_plus.read.entry import profile_environment, resolve_read
    paper = {key: c.protocol.ARMS[key[0]][key[1]][1] for key in c.RELEASE_RECIPES}
    rows = {r['launch']['name']: r for r in json.loads((c.ROOT / 'read/recipe-index.json').read_text())['rows']}
    for model in c.MODELS:
        args = parser().parse_args(['--model', model, '--resolve-only', '--assets', 'release'])
        with c.assets('release'):
            for job in jobs(args):
                _, argv = c.prepare(job, tmp_path / model / job['dataset'] / job['arm'] / str(job['offset']), args)
                recipe = argv[argv.index('--recipe') + 1]
                assert recipe == c.RELEASE_RECIPES.get((job['model'], job['arm']), paper.get((job['model'], job['arm']), recipe))
                assert c.launch_plan(job, argv)['runtime_recipe']['name'] == recipe
    assert {key: c.protocol.ARMS[key[0]][key[1]][1] for key in c.RELEASE_RECIPES} == paper
    for key, release in c.RELEASE_RECIPES.items():
        launches = [rows[paper[key]], rows[release]]
        environments = [profile_environment(resolve_read(r['names'], list(r['overrides']), root=c.ROOT / 'config',
                                                         launch=r['launch'], output='read.json')['config'])
                        for r in launches]
        assert environments[0] == environments[1]
        assert launches[1]['launch']['drafter_artifact'].endswith('_release')
        assert launches[1]['launch']['target_release_slice']['source'] == 'target_slice'


def test_offsets_select_groups_and_refuse_unknown_ones():
    args = parser().parse_args(['--model', 'qwen3-8b', '--resolve-only', '--offsets', '0', '--arms', 'J1R1REF_F4'])
    assert {(j['arm'], j['offset']) for j in jobs(args)} == {('AR', 0), ('J1R1REF_F4', 0)}
    with pytest.raises(ValueError, match='offsets'):
        list(jobs(parser().parse_args(['--model', 'qwen3-8b', '--offsets', '0,16'])))


def test_chain_audit_prints_every_error_word_and_fails_a_nonzero_word(capsys):
    import importlib.util
    from types import SimpleNamespace
    spec = importlib.util.spec_from_file_location('table3_chain_audit', c.AUDIT / 'sitecustomize.py')
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    assert audit.TARGET == 'jetspec.engines.jetspec_plus_plus.draft.graph'
    words = [0, 4]
    class ChainGraph:
        def prepare(self, key):
            return SimpleNamespace(buffers={'err_host': [words.pop(0)]})
        def run(self, key, **kwargs):
            return 'ran'
    audit.patch(SimpleNamespace(ChainGraph=ChainGraph))
    assert ChainGraph().run('key', J_round=1, refresh=False) == 'ran'
    with pytest.raises(AssertionError, match='CHAIN_ERR'):
        ChainGraph().run('key', J_round=1, refresh=False)
    text = capsys.readouterr().out
    assert [int(v) for v in re.findall(r'W645_CHAIN_AUDIT .*"error_bits": (\d+)', text)] == [0, 4]


def test_all_paper_arms_pass_the_native_reader_model_lookup():
    # The native reader resolves each named recipe's model through recipe-index; launch_plan does not.
    from jetspec.engines.jetspec_plus_plus.read import legacy
    indexed = {row['launch']['name'] for row in json.loads((c.ROOT / 'read/recipe-index.json').read_text())['rows']}
    for model, arms in c.protocol.ARMS.items():
        for arm, (_, recipe, _) in arms.items():
            if recipe in indexed:
                assert isinstance(legacy.recipe_model_entry(recipe), dict), (model, arm)


def expand(row):
    row = dict(row)
    if isinstance(row['output_token_ids'], int):
        row['output_token_ids'] = list(range(row['output_token_ids']))
    for key in ('credited_tokens_values','cache_hit_values','tau_hit_values'):
        row[key] = [v for n,v in row[key] for _ in range(n)]
    for key in ('round_ms_values','paper_verify_host_ms_values'):
        row[key] = [total/n for n,total in row[key] for _ in range(n)]
    row['verify_host_ms_values'] = []
    return row


def test_recorded_paper_cells():
    path = Path(__file__).with_name('fixtures') / 'table3-golden.json.gz'
    with gzip.open(path, 'rt') as stream:
        fixture = json.load(stream)
    reads = [{**{k:r[k] for k in ('model','dataset','arm','offset')},
              **c.reduce_rows([expand(row) for row in r['rows']])} for r in fixture['reads']]
    table = c.pool(reads)
    checked = 0
    for expected in fixture['expected']:
        row = next(r for r in table if (r['model'],r['dataset'],c.NAMES.get(r['arm'],r['arm'])) ==
                   (expected['model'],expected['dataset'],expected['method']))
        assert c.formatted(row)[3] == ('2' if c.protocol.ARMS[row['model']][row['arm']][2] == 'pair' else '1')
        actual = [v for part in c.formatted(row)[4:] for v in (re.findall(r'\d+\.\d+',part) or ['N/A'])]
        assert actual == expected['values']
        checked += sum(v != 'N/A' for v in actual)
    assert checked == 114  # 98 measured cells plus 16 derived speedups.


def test_pool_refuses_an_incomplete_population():
    args = parser().parse_args(['--model','qwen3-8b','--resolve-only'])
    job = next(jobs(args))
    with pytest.raises(ValueError, match='incomplete population'):
        c.pool([dict(**job, values={}, weights={})])


@pytest.mark.parametrize('onset,tau,tok_s,round_ms,hit,tau_miss',
                         [(65,32,1600,20,1,None),(70,23,740,25,.5,5)])
def test_rule_a_keeps_the_closing_round_cost_and_steady_hit_basis(onset,tau,tok_s,round_ms,hit,tau_miss):
    ids = list(range(onset)) + list(range(1000,1064))*3
    credits = [32,32,64,64,len(ids)-193]
    row = dict(original_index=0,token_cap=len(ids),stop='cap',stopped_tokens=len(ids),
               output_token_ids=ids,rounds=5,decode_tokens=len(ids)-1,raw_rounds=5,
               accepted_raw=len(ids)-1,credited_tokens_values=credits,
               round_ms_values=[100,20,30,40,50],verify_host_ms_values=[10,5,6,7,8],
               paper_verify_host_ms_values=[10,5,6,7,8],
               cache_hit_values=[False,True,False,True,True],tau_hit_values=[False,True,False,True,True])
    result = c.reduce_rows([row])
    assert result['loop_count'] == 1
    assert result['values'] == dict(tau=tau,tok_s=tok_s,round=round_ms,verify=5 if onset==65 else 5.5,
                                    hit=hit,tau_hit=32,tau_miss=tau_miss)


def test_read_validation_rejects_a_native_traceback_even_after_completion(tmp_path):
    source = sealed_source('8b','gsm8k')
    group = c.protocol.select_group(source,0,32,c.protocol.HASHES['8b']['gsm8k'])
    prompts = tmp_path/'prompts.json'
    c.write_json(prompts,group)
    output = tmp_path/'read.json'
    c.write_json(output,dict(outputs=[dict(token_ids=[151643]) for _ in range(32)]))
    condition = dict(count=32,tokens=2048,temperature=0,seed=0,stop_at_eos='on',sequential_prompts=True,
                     prompts_sha256=c.protocol.sha256(prompts),method='ar',recipe='ar-graphed',
                     regime='graphed',placement='one_gpu',eos_token_ids=[151643,151645],
                     environment={'SSD_SHARED_GPU_KV_UTILIZATION':'0.10'})
    c.write_json(tmp_path/'read.json.campaign.json',dict(resolved_condition=condition,reads=[dict(measured={'metrics':{}})]))
    log = tmp_path/'read.log'
    log.write_text('SERVE_READ_COMPLETE\n')
    receipt = c.protocol.validate_read(output,log,'AR','gsm8k',prompts,source,0,write_receipt=False)
    assert receipt['original_indices'] == list(range(32))
    log.write_text('SERVE_READ_COMPLETE\nTraceback (most recent call last)\n')
    with pytest.raises(AssertionError):
        c.protocol.validate_read(output,log,'AR','gsm8k',prompts,source,0,write_receipt=False)


def test_resume_keeps_validated_groups_and_runs_only_the_rest(tmp_path, monkeypatch):
    kept, ran = [], []
    work = [dict(model='8b', dataset='gsm8k', arm='AR', offset=offset) for offset in (0, 32)]
    monkeypatch.setattr(c, 'jobs', lambda args: iter(work))
    def prepare(job, directory, args):
        directory.mkdir(parents=True, exist_ok=True)
        return None, ['run', str(job['offset'])]
    monkeypatch.setattr(c, 'prepare', prepare)
    monkeypatch.setattr(c, 'launch_plan', lambda job, argv: dict(job))
    (tmp_path/'gsm8k'/'AR'/'p000').mkdir(parents=True)
    (tmp_path/'gsm8k'/'AR'/'p000'/'read.json').write_text('{}')
    def reduce(directory, job, source):
        kept.append(job['offset'])
        return dict(job)
    monkeypatch.setattr(c, 'reduce_directory', reduce)
    def run(command, **kwargs):
        ran.append(command[1])
        return c.subprocess.CompletedProcess(command, 0)
    monkeypatch.setattr(c.subprocess, 'run', run)
    monkeypatch.setattr(c, 'pool', lambda reads: [dict(reads[0], tok_s=1.0)])
    monkeypatch.setattr(c, 'formatted', lambda row: [])
    c.main(['--model', 'qwen3-8b', '--host-values', str(tmp_path/'host.json'), '--resume',
            '--output-dir', str(tmp_path)])
    assert ran == ['32']
    assert kept == [0, 32]


def test_resume_sets_an_invalid_read_aside_before_its_rerun(tmp_path, monkeypatch):
    # The reader refuses to overwrite its artifacts, so a rerun needs the invalid read moved away.
    work = [dict(model='27b', dataset='gsm8k', arm='J3R3DEV_F4', offset=0)]
    monkeypatch.setattr(c, 'jobs', lambda args: iter(work))
    def prepare(job, directory, args):
        directory.mkdir(parents=True, exist_ok=True)
        return None, ['run', str(job['offset'])]
    monkeypatch.setattr(c, 'prepare', prepare)
    monkeypatch.setattr(c, 'launch_plan', lambda job, argv: dict(job))
    group = tmp_path/'gsm8k'/'J3R3DEV_F4'/'p000'
    group.mkdir(parents=True)
    for name in ('read.json', 'read.json.campaign.json', 'read.json.manifest.json', 'read.log'):
        (group/name).write_text(name)
    checks = []
    def reduce(directory, job, source):
        checks.append(job['offset'])
        if len(checks) == 1:
            raise AssertionError('CHAIN_ERR')
        return dict(job)
    monkeypatch.setattr(c, 'reduce_directory', reduce)
    def run(command, stdout, **kwargs):
        assert not list(group.glob('read.json*'))
        (group/'read.json').write_text('fresh')
        return c.subprocess.CompletedProcess(command, 0)
    monkeypatch.setattr(c.subprocess, 'run', run)
    monkeypatch.setattr(c, 'pool', lambda reads: [dict(reads[0], tok_s=1.0)])
    monkeypatch.setattr(c, 'formatted', lambda row: [])
    c.main(['--model', 'qwen3.8-27b', '--host-values', str(tmp_path/'host.json'), '--resume',
            '--output-dir', str(tmp_path)])
    assert checks == [0, 0]
    assert sorted(p.name for p in (group/'invalid'/'1').iterdir()) == [
        'read.json', 'read.json.campaign.json', 'read.json.manifest.json', 'read.log']
    assert (group/'read.json').read_text() == 'fresh'
    assert c.set_aside(group).name == '2' and not (group/'read.json').exists()


def test_a_failed_group_reruns_once_and_keeps_its_log(tmp_path, monkeypatch):
    # A cold compiler cache can time out a pinned pair's first receive; one warm rerun follows.
    work = [dict(model='8b', dataset='gsm8k', arm='J1R1REF_F4', offset=0)]
    monkeypatch.setattr(c, 'jobs', lambda args: iter(work))
    def prepare(job, directory, args):
        directory.mkdir(parents=True, exist_ok=True)
        return None, ['run', str(job['offset'])]
    monkeypatch.setattr(c, 'prepare', prepare)
    monkeypatch.setattr(c, 'launch_plan', lambda job, argv: dict(job))
    monkeypatch.setattr(c, 'reduce_directory', lambda directory, job, source: dict(job))
    codes = [-6, 0]
    def run(command, stdout, **kwargs):
        stdout.write(f'attempt exit {codes[0]}\n')
        return c.subprocess.CompletedProcess(command, codes.pop(0))
    monkeypatch.setattr(c.subprocess, 'run', run)
    monkeypatch.setattr(c, 'pool', lambda reads: [dict(reads[0], tok_s=1.0)])
    monkeypatch.setattr(c, 'formatted', lambda row: [])
    c.main(['--model', 'qwen3-8b', '--host-values', str(tmp_path/'host.json'), '--output-dir', str(tmp_path)])
    group = tmp_path/'gsm8k'/'J1R1REF_F4'/'p000'
    assert codes == []
    assert (group/'read.failed.log').read_text() == 'attempt exit -6\n'
    assert (group/'read.log').read_text() == 'attempt exit 0\n'
    codes[:] = [1, 1]
    with pytest.raises(c.subprocess.CalledProcessError):
        c.main(['--model', 'qwen3-8b', '--host-values', str(tmp_path/'host.json'), '--output-dir', str(tmp_path)])


def test_block_head_kit_ships_byte_exact_inside_the_package():
    # DS and DFlash rows load the kit by path; its manifest pins every file.
    from jetspec.engines.jetspec_plus_plus.jetspec import blockhead_serve
    assert blockhead_serve.KIT_DIR.resolve().is_relative_to(c.ROOT.resolve())
    blockhead_serve.validate_vendor_manifest()
