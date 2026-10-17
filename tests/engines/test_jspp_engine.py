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
    assert recipe['target_release_artifact'] not in host['artifacts']


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
    args = entry.parser().parse_args(['--schedule', 'j1-r1', '--host-values', _host(tmp_path, 'sm80'),
                                     '--prompts-json', '/prompts.json', '--output-json', '/read.json'])
    with pytest.raises(SystemExit, match='supported here:.*--schedule j1-r0'):
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
# CuTe DSL kernels that their callers import lazily, on CUDA only.
cuda_only = {'kernels/nvfp4/_k1_kernel.py'}
package = root / 'jetspec/engines/jetspec_plus_plus'
paths = [path for path in sorted(package.rglob('*.py'))
         if str(path.relative_to(package)) not in cuda_only]
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
