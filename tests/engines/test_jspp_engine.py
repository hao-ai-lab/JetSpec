"""The serving entry resolves artifact paths without loading accelerator models."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


def test_launch_overlay_preserves_launcher_precedence_and_rejects_missing_fields():
    from jetspec.engines.jetspec_plus_plus import entry
    from jetspec.engines.jetspec_plus_plus.run_config import RunConfig, launch_overlay
    args = entry.parser().parse_args(['--schedule', 'j1-r1', '--engine', 'reference',
                                     '--prompts-json', '/prompts.json', '--output-json', '/read.json'])
    entry.resolve(args)
    overlay = launch_overlay({'JETSPEC_ROWS_CAP': '8'})
    config = RunConfig.from_overlay(overlay, args._record['config'])
    assert config.provided_values['engine.engine_settings.rows_cap'] == '8'
    assert overlay['method.method_settings.adapter_refresh'] is None
    assert 'method.method_settings.adapter_refresh' not in config.provided_values
    overlay.pop('engine.engine_settings.rows_cap')
    with pytest.raises(ValueError, match='launch_overlay missing keys'):
        RunConfig.from_overlay(overlay, args._record['config'])


def test_entry_maps_user_paths_without_loading_models():
    from jetspec.engines.jetspec_plus_plus import entry
    args = entry.parser().parse_args([
        '--schedule', 'j1-r1', '--engine', 'reference', '--target', '/target', '--drafter', '/draft',
        '--adapter', '/adapter', '--heads-bundle', '/heads',
        '--prompts-json', '/prompts.json', '--tokens', '2048', '--devices', '0,1',
        '--output-json', '/read.json'])
    recipe, argv, host = entry.resolve(args)
    assert recipe['environment']['JETSPEC_VERIFY_REFERENCE'] == '1'
    assert host['environment']['JETSPEC_TARGET_MODEL_PATH'] == '/target'
    assert host['artifacts'][recipe['drafter_artifact']] == '/draft'
    assert host['artifacts'][recipe['bundle_artifact']] == '/heads'
    assert argv[argv.index('--tokens') + 1] == '2048'


def _host(tmp_path, family):
    path = tmp_path / f'host-{family}.json'
    host = {'cuda_arch': '8.0' if family == 'sm80' else '10.0', 'environment': {}, 'artifacts': {}}
    if family == 'sm100-aarch64':
        host['machine'] = 'aarch64'
    path.write_text(json.dumps(host))
    return str(path)


def _entry_keys():
    from jetspec.engines.jetspec_plus_plus import entry
    return sorted(entry.RECIPES)


@pytest.mark.parametrize('key', _entry_keys())
def test_every_supported_combination_resolves_with_persistent_cache_and_knobs(key, tmp_path, monkeypatch):
    from jetspec.engines.jetspec_plus_plus import entry
    model, schedule, engine, family = key
    monkeypatch.setenv('JSPP_CACHE_ROOT', '/persistent-cache')
    args = entry.parser().parse_args([
        '--model', model, '--schedule', schedule, '--engine', engine,
        '--host-values', _host(tmp_path, family),
        '--prompts-json', '/prompts.json', '--output-json', '/read.json',
        '--knob', 'SSD_SHARED_GPU_KV_UTILIZATION=0.1'])
    recipe, argv, _ = entry.resolve(args)
    expected = entry.RECIPES[key]
    if isinstance(expected, dict):
        # A release composition resolves through its own config names, with passes J + 1.
        assert args._record['config']['method']['schedule']['j'] == 1
        assert recipe['passes'] == 2
    else:
        assert argv[argv.index('--recipe') + 1] == expected
    assert args._record['config']['engine']['cutoffs'] is True
    assert argv[argv.index('--cache-root') + 1] == '/persistent-cache'
    assert argv[argv.index('--knob') + 1] == 'SSD_SHARED_GPU_KV_UTILIZATION=0.1'
    if recipe['method'] == 'jetspecpp':
        assert args._record['config']['model']['thresholds_sha256'] == recipe['thresholds_sha256']
    else:
        assert argv[argv.index('--placement') + 1] == 'one_gpu'
        assert argv[argv.index('--devices') + 1] == '0'


def test_sm80_default_keeps_the_pinned_schedule_and_thresholds():
    from jetspec.engines.jetspec_plus_plus import entry
    args = entry.parser().parse_args(['--prompts-json', '/prompts.json', '--output-json', '/read.json'])
    recipe, _, _ = entry.resolve(args)
    assert args._record['config']['method']['schedule'] == {'j': 1, 'r': 0}
    assert args._record['config']['model']['thresholds_sha256'] == recipe['thresholds_sha256']


def test_full_read_without_eval_consumer_names_the_eval_entry():
    from jetspec.engines.jetspec_plus_plus.read.entry import write_eval_records
    with pytest.raises(ValueError, match=r'run full reads through python -m engines\.eval\.entry'):
        write_eval_records({}, process_wall_seconds=0)


def test_host_values_do_not_gain_an_empty_release_override(tmp_path):
    from jetspec.engines.jetspec_plus_plus import entry
    path = tmp_path / 'host.json'
    path.write_text(json.dumps({'cuda_arch': '8.0', 'environment': {}, 'artifacts': {}}))
    args = entry.parser().parse_args(['--model', 'qwen3.8-27b', '--host-values', str(path),
                                     '--prompts-json', '/prompts.json', '--output-json', '/read.json'])
    recipe, _, host = entry.resolve(args)
    # The 27B release takes its embedding and LM head from the target, so it names no release artifact.
    assert recipe.get('target_release_artifact') is None
    assert 'target_release_slice' in recipe
    assert all(name in {recipe['drafter_artifact'], recipe['adapter_artifact'], recipe['bundle_artifact']}
               for name in host['artifacts'])


def test_manifest_preserves_trained_adapter_serving_layout():
    from jetspec.engines.jetspec_plus_plus import entry
    manifest = json.loads((entry.ROOT / 'artifacts/manifest.json').read_text())
    row = next(row for row in manifest['artifacts'] if row['name'] == 'qwen3p8_27b_adapter_base_800k')
    assert row['serving_layout'] == 'trained_c1'


def test_default_cpu_pin_queries_gpu_and_partitions_local_cpus(tmp_path, monkeypatch):
    from jetspec.engines.jetspec_plus_plus.serving import cpu_affinity
    node = tmp_path / 'devices/system/node/node0'
    node.mkdir(parents=True)
    (node / 'cpulist').write_text('0-47')
    bus = tmp_path / 'bus/pci/devices/0000:01:00.0'
    bus.mkdir(parents=True)
    (bus / 'numa_node').write_text('0')
    queried = []

    def query(command, **kwargs):
        queried.append(command[2])
        return '00000000:01:00.0\n'

    monkeypatch.setattr(cpu_affinity.subprocess, 'check_output', query)
    plan = cpu_affinity.plan_cpu_pin('auto', ['target-gpu', 'draft-gpu'], 'pair', 3, sysfs=tmp_path)
    assert queried == ['target-gpu', 'draft-gpu']
    assert plan['spec'] == '12-23;24-35;36-47'
    assert [row['gpu'] for row in plan['ranks']] == ['target-gpu', 'draft-gpu', 'draft-gpu']


def test_entry_selects_architecture_from_the_granted_gpu(monkeypatch):
    import torch
    from jetspec.engines.jetspec_plus_plus import entry
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '3,4')
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)

    def capability(device):
        assert device == 0
        return 10, 0

    monkeypatch.setattr(torch.cuda, 'get_device_capability', capability)
    args = entry.parser().parse_args(['--devices', '3,4',
                                     '--prompts-json', '/prompts.json', '--output-json', '/read.json'])
    _, argv, host = entry.resolve(args)
    assert (args.model, args.schedule, args.engine) == ('qwen3-8b', 'j1-r1', 'fast')
    assert argv[argv.index('--recipe') + 1] == 'jetspecpp-graphed-pair-e6v3-b16-chain-r4-j1r1-lean-swap-fast6'
    assert host['cuda_arch'] == '10.0'


def test_unsupported_combination_names_the_supported_ones(tmp_path):
    from jetspec.engines.jetspec_plus_plus import entry
    args = entry.parser().parse_args(['--model', 'qwen3.8-27b', '--schedule', 'j1-r0',
                                     '--host-values', _host(tmp_path, 'sm80'),
                                     '--prompts-json', '/prompts.json', '--output-json', '/read.json'])
    with pytest.raises(SystemExit, match='supported here:.*--schedule j1-r1'):
        entry.resolve(args)


def test_no_entry_names_a_machine():
    from jetspec.engines.jetspec_plus_plus import entry
    import re
    text = ' '.join(' '.join(key) for key in entry.RECIPES)
    assert not re.search(r'a100|b200|gb200|h100|h200', text)


def test_source_revision_is_null_outside_git_without_receipt(tmp_path, monkeypatch):
    from jetspec.engines.jetspec_plus_plus.read import legacy
    monkeypatch.delenv('RX_SOURCE_PROVENANCE', raising=False)
    package = tmp_path / 'site/engine'
    package.mkdir(parents=True)
    assert legacy.source_revision(package) == (None, None)


def test_all_shipped_modules_import_on_cpu():
    root = Path(__file__).resolve().parents[2]
    code = '''
import importlib, os, sys
from pathlib import Path
import torch
os.environ.setdefault('SSD_HF_CACHE', '/tmp/unused-import-hf')
os.environ.setdefault('SSD_DATASET_DIR', '/tmp/unused-import-datasets')
root = Path(sys.argv[1])
# CuTe DSL and Triton kernels that their callers import lazily, on CUDA only.
cuda_only = {'kernels/nvfp4/_k1_kernel.py', 'kernels/nvfp4/_w4a16_kernel.py', 'kernels/nvfp4/_w4a8_kernel.py',
             'kernels/nvfp4/_w4a8_quant.py', 'kernels/nvfp4/_w4a8_triton.py'}
package = root / 'jetspec/engines/jetspec_plus_plus'
# The byte-pinned block-head kit loads by path under neutral names, never as package modules.
paths = [path for path in sorted(package.rglob('*.py'))
         if str(path.relative_to(package)) not in cuda_only
         and path.relative_to(package).parts[0] != 'third_party']
for path in paths:
    name = '.'.join(path.relative_to(root).with_suffix('').parts).removesuffix('.__init__')
    importlib.import_module(name)
assert not torch.cuda.is_initialized()
print('IMPORTS_PASS', len(paths))
'''
    result = subprocess.run([sys.executable, '-c', code, str(root)], cwd=root,
                            env=dict(os.environ, PYTHONPATH=str(root)),
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'IMPORTS_PASS' in result.stdout


@pytest.mark.parametrize('is_draft', [False, True])
def test_long_decode_allocates_and_releases_additional_blocks(monkeypatch, is_draft):
    from jetspec.engines.jetspec_plus_plus.serving.block_manager import BlockManager
    from jetspec.engines.jetspec_plus_plus.serving.sequence import Sequence
    monkeypatch.setattr(Sequence, 'block_size', 256, raising=False)
    manager = BlockManager(num_blocks=16, block_size=256, is_draft=is_draft, max_model_len=4096)
    sequence = Sequence([1] * 255)
    manager.allocate(sequence)
    for _ in range(2048):
        assert manager.can_append(sequence)
        manager.may_append(sequence)
        sequence.append_token(2)
    table = sequence.draft_block_table if is_draft else sequence.block_table
    assert len(table) == 9
    assert len(manager.free_block_ids) == 7
    manager.deallocate(sequence)
    assert table == []
    assert len(manager.free_block_ids) == 16
    assert manager.used_block_ids == set()


@pytest.mark.parametrize('model,schedule,engine', [('qwen3-8b', 'j1-r1', 'reference'),
                                                   ('qwen3.8-27b', 'j1-r1', 'fast')])
def test_speedup_uses_matching_ar_baseline_and_stops_at_eos(model, schedule, engine, tmp_path):
    from bench.jetspec_plus_plus import speedup
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        output = Path(command[command.index('--output-json') + 1])
        selected = command[command.index('--schedule') + 1]
        rate = 50.0 if selected == 'ar' else 200.0
        output.write_text(json.dumps({'outputs': [{'token_ids': [1, 2]}] * 2,
            'metrics': {'serve_read': {'measured': {'decode_tok_s': rate}}}}))

    assert speedup.main(['--model', model, '--schedule', schedule, '--engine', engine, '--target', '/target', '--drafter', '/draft',
                         '--prompts-json', '/prompts.json', '--output-dir', str(tmp_path),
                         '--knob', 'SSD_SHARED_GPU_KV_UTILIZATION=0.1'], run=run) == 0
    ar, method = calls
    assert ar[ar.index('--schedule') + 1] == 'ar'
    assert ar[ar.index('--model') + 1] == method[method.index('--model') + 1] == model
    assert '--engine' not in ar and method[method.index('--engine') + 1] == engine
    assert '--drafter' not in ar and '--knob' not in ar
    assert '--drafter' in method and '--knob' in method
    for command in calls:
        assert command[command.index('--stop-at-eos') + 1] == 'on'
    assert json.loads((tmp_path / 'speedup.json').read_text())['speedup'] == 4.0


def test_speedup_auto_measures_every_candidate_and_keeps_the_fastest(tmp_path):
    import subprocess
    from bench.jetspec_plus_plus import speedup
    calls = []
    rates = {'j1-r0': 310.0, 'j1-r1': 420.0, 'ar': 100.0}

    def run(command, **kwargs):
        calls.append(command)
        output = Path(command[command.index('--output-json') + 1])
        selected = command[command.index('--schedule') + 1]
        if selected == 'j1-r0' and 'tune-' in output.name and len(calls) == 1:
            return subprocess.CompletedProcess(command, 1)
        output.write_text(json.dumps({'outputs': [{'token_ids': [1]}] * 4,
            'metrics': {'serve_read': {'measured': {'decode_tok_s': rates[selected]}}}}))
        return subprocess.CompletedProcess(command, 0)

    host = tmp_path / 'host.json'
    host.write_text(json.dumps({'cuda_arch': '10.0', 'machine': 'aarch64'}))
    assert speedup.main(['--model', 'qwen3-8b', '--host-values', str(host), '--prompts-json', '/prompts.json',
                         '--count', '32', '--output-dir', str(tmp_path / 'out')], run=run) == 0
    tuned = [c for c in calls if Path(c[c.index('--output-json') + 1]).name.startswith('tune-')]
    assert sorted(c[c.index('--schedule') + 1] for c in tuned) == ['j1-r0', 'j1-r1']
    assert all(c[c.index('--count') + 1] == '4' and c[c.index('--tokens') + 1] == '2048' for c in tuned)
    result = json.loads((tmp_path / 'out' / 'speedup.json').read_text())
    assert result['schedule'] == 'j1-r1' and result['speedup'] == 4.2
    assert result['tuning']['family'] == 'sm100-aarch64' and result['tuning']['failed'] == {'j1-r0': 1}


@pytest.mark.parametrize('model,target,checkpoint', [
    ('qwen3-8b', ('Qwen/Qwen3-8B', 'b968826d9c46dd6066d109eabc6255188de91218'),
     ('JetSpec/jetspec-plus-plus-qwen3-8b', '9bd8b1578e3d14d74d68e82f2fc5decbee001b28')),
    ('qwen3.8-27b', ('Qwen/Qwen3.8-27B', '1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0'),
     ('JetSpec/jetspec-plus-plus-qwen3p8-27b', '3147a3513f7cc7d81bc82e40e4ce3f28ad7d8047'))])
def test_release_paths_download_the_pinned_target_and_checkpoint(model, target, checkpoint, monkeypatch):
    import huggingface_hub
    from jetspec.engines.jetspec_plus_plus import entry
    calls = []
    monkeypatch.setattr(huggingface_hub, 'snapshot_download',
                        lambda repo, revision: calls.append((repo, revision)) or f'/hub/{repo}')
    assert entry.release_paths(model) == (Path('/hub') / target[0], Path('/hub') / checkpoint[0])
    assert calls == [target, checkpoint]
    calls.clear()
    assert entry.release_paths(model, '/local/target', '/local/checkpoint') == (
        Path('/local/target'), Path('/local/checkpoint'))
    assert calls == []


def test_generate_answers_one_prompt_with_the_release_checkpoint(tmp_path, monkeypatch, capsys):
    from jetspec.engines.jetspec_plus_plus import entry
    checkpoint = tmp_path / 'checkpoint'
    monkeypatch.setattr(entry, 'release_paths', lambda model, target, checkpoint_: (tmp_path / 'target', checkpoint))
    monkeypatch.setattr(entry, 'prompt_rows', lambda target, text: {'count': 1, 'prompts': [
        {'index': 0, 'text': text, 'token_ids': [1, 2, 3]}]})
    seen = []

    def run(argv):
        seen.append(argv)
        prompts = json.loads(Path(argv[argv.index('--prompts-json') + 1]).read_text())
        assert prompts['prompts'][0]['token_ids'] == [1, 2, 3]
        Path(argv[argv.index('--output-json') + 1]).write_text(json.dumps(
            {'throughput_tok_s': 9999.0, 'metrics': {'serve_read': {'measured': {'decode_tok_s': 300.0}}},
             'outputs': [{'text': 'Nine.', 'token_ids': [7, 8]}]}))
        return 0

    monkeypatch.setattr(entry, 'main', run)
    assert entry.generate(['How many divisors does 196 have?', '--cache-root', str(tmp_path / 'cache')]) == 0
    argv, = seen
    assert argv[argv.index('--drafter') + 1] == str(checkpoint)
    assert argv[argv.index('--adapter') + 1] == str(checkpoint / 'adapter')
    assert argv[argv.index('--heads-bundle') + 1] == str(checkpoint / 'calibration')
    assert argv[argv.index('--count') + 1] == '1' and argv[argv.index('--stop-at-eos') + 1] == 'on'
    assert '--schedule' not in argv
    printed = capsys.readouterr().out
    assert 'Nine.' in printed and 'decode 300.0 tok/s' in printed and '9999' not in printed


def test_published_heads_bundle_is_found_under_its_registration_name(tmp_path):
    from jetspec.engines.jetspec_plus_plus import entry
    calibration = tmp_path / 'checkpoint' / 'calibration'
    calibration.mkdir(parents=True)
    key = 'jetspecpp_e6_calibration_release'
    host = {'artifacts': {key: str(calibration)}}
    entry.alias_heads_bundle(host, {'bundle_artifact': key, 'bundle_name': key}, tmp_path / 'run')
    assert Path(host['artifacts'][key]).name == key
    assert Path(host['artifacts'][key]).resolve() == calibration.resolve()
    # A recipe serving a sibling operating point keeps the declared path: the sibling must exist.
    sibling = {'artifacts': {key: str(calibration)}}
    entry.alias_heads_bundle(sibling, {'bundle_artifact': key, 'bundle_name': 'other'}, tmp_path / 'run2')
    assert sibling['artifacts'][key] == str(calibration)


def test_generic_draft_is_downloaded_only_when_absent(tmp_path, monkeypatch):
    import huggingface_hub
    from jetspec.engines.jetspec_plus_plus import entry
    calls = []
    monkeypatch.setattr(huggingface_hub, 'snapshot_download',
                        lambda repo, revision, cache_dir: calls.append((repo, revision, cache_dir)))
    entry.ensure_generic_draft(str(tmp_path))
    assert calls == [(*entry.GENERIC_DRAFT, str(tmp_path))]
    snapshot = tmp_path / 'models--Qwen--Qwen3-0.6B' / 'snapshots' / entry.GENERIC_DRAFT[1]
    snapshot.mkdir(parents=True)
    (snapshot / 'config.json').write_text('{}')
    entry.ensure_generic_draft(str(tmp_path))
    assert len(calls) == 1


def test_entry_reads_the_generic_draft_from_the_hub_cache(monkeypatch):
    from jetspec.engines.jetspec_plus_plus import entry
    monkeypatch.delenv('SSD_HF_CACHE', raising=False)
    monkeypatch.setattr(entry, 'hub_cache', lambda: '/hub-cache')
    args = entry.parser().parse_args(['--prompts-json', '/prompts/p.json', '--output-json', '/read.json'])
    _, _, host = entry.resolve(args)
    assert host['environment']['SSD_HF_CACHE'] == '/hub-cache'


def test_speedup_defaults_to_bundled_prompts_and_the_release_checkpoint(tmp_path, monkeypatch):
    from bench.jetspec_plus_plus import speedup
    from jetspec.engines.jetspec_plus_plus import entry
    monkeypatch.setattr(entry, 'release_paths', lambda model: (Path('/target'), Path('/checkpoint')))
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        rate = 50.0 if command[command.index('--schedule') + 1] == 'ar' else 300.0
        Path(command[command.index('--output-json') + 1]).write_text(json.dumps(
            {'outputs': [{'token_ids': [1]}], 'metrics': {'serve_read': {'measured': {'decode_tok_s': rate}}}}))
        return subprocess.CompletedProcess(command, 0)

    assert speedup.main(['--dataset', 'math500', '--schedule', 'j1-r1', '--output-dir', str(tmp_path)], run=run) == 0
    ar, method = calls
    assert method[method.index('--prompts-json') + 1].endswith('inputs/8b/prompts_math500128.json')
    assert Path(method[method.index('--prompts-json') + 1]).is_file()
    assert method[method.index('--heads-bundle') + 1] == '/checkpoint/calibration'
    assert ar[ar.index('--target') + 1] == '/target' and '--drafter' not in ar
    assert method[method.index('--count') + 1] == '32' and method[method.index('--tokens') + 1] == '2048'
    with pytest.raises(SystemExit, match='no bundled prompts for --dataset squad'):
        speedup.main(['--dataset', 'squad', '--output-dir', str(tmp_path)], run=run)


@pytest.mark.parametrize('dataset,count', [('aime25', 30), ('humaneval', 128), ('livecodebench', 128),
                                           ('mbpp', 128), ('mt-bench', 80)])
def test_every_evaluation_benchmark_has_bundled_first_turn_prompts(dataset, count, tmp_path):
    from bench.jetspec_plus_plus import speedup
    path = Path(speedup.bundled_prompts('qwen3-8b', dataset, None, tmp_path))
    rows = json.loads(path.read_text())['prompts']
    assert len(rows) == count and all(row['token_ids'] and '<|im_start|>user\n' in row['text'] for row in rows)
    assert '-t2-' not in path.name


def test_27b_prompts_are_rendered_with_the_target_template(tmp_path, monkeypatch):
    import transformers
    from bench.jetspec_plus_plus import speedup

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            assert kwargs['add_generation_prompt'] and not kwargs['enable_thinking']
            return '<user>' + messages[0]['content'] + '</user>'

        def __call__(self, text, add_special_tokens):
            return type('Encoded', (), {'input_ids': [len(text)]})()

    monkeypatch.setattr(transformers.AutoTokenizer, 'from_pretrained', lambda target: Tokenizer())
    path = Path(speedup.bundled_prompts('qwen3.8-27b', 'mbpp', '/target-27b', tmp_path))
    rows = json.loads(path.read_text())['prompts']
    assert len(rows) == 128 and all(row['text'].startswith('<user>') for row in rows)
    assert '<|im_start|>' not in rows[0]['text'] and rows[0]['token_ids'] == [len(rows[0]['text'])]
    with pytest.raises(SystemExit, match='pass --target'):
        speedup.bundled_prompts('qwen3.8-27b', 'mbpp', None, tmp_path)
