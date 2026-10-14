"""The JetSpec++ serving engine resolves its configurations and imports on CPU."""
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
ENGINE = ROOT / 'jetspec/engines/jetspec_plus_plus'


@pytest.mark.parametrize('is_draft', [False, True])
def test_long_decode_allocates_and_releases_additional_blocks(monkeypatch, is_draft):
    from jetspec.engines.jetspec_plus_plus.serving.block_manager import BlockManager
    from jetspec.engines.jetspec_plus_plus.serving.sequence import Sequence

    monkeypatch.setattr(Sequence, 'block_size', 256, raising=False)
    manager = BlockManager(num_blocks=16, block_size=256, is_draft=is_draft,
                           max_model_len=4096)
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


def test_speculative_decode_finalizes_completed_target_and_draft_blocks(monkeypatch, tmp_path):
    monkeypatch.setenv('SSD_HF_CACHE', str(tmp_path))
    monkeypatch.setenv('SSD_DATASET_DIR', str(tmp_path))
    from jetspec.engines.jetspec_plus_plus.serving.block_manager import BlockManager
    from jetspec.engines.jetspec_plus_plus.serving.scheduler import Scheduler
    from jetspec.engines.jetspec_plus_plus.serving.sequence import Sequence

    monkeypatch.setattr(Sequence, 'block_size', 256, raising=False)
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.block_size = 256
    scheduler.max_model_len = 4096
    scheduler.eos = -1
    scheduler.jetspec = True
    scheduler.block_manager = BlockManager(16, 256, max_model_len=4096)
    scheduler.draft_block_manager = BlockManager(16, 256, is_draft=True, max_model_len=4096)
    sequence = Sequence([1] * 254)
    for manager in (scheduler.block_manager, scheduler.draft_block_manager):
        manager.allocate(sequence)
    sequence.num_cached_tokens = sequence.num_draft_cached_tokens = 254
    scheduler.postprocess_speculate([sequence], [[2, 3]], [4])
    for manager in (scheduler.block_manager, scheduler.draft_block_manager):
        block = manager.blocks[0]
        assert block.token_ids == sequence.token_ids
        assert manager.hash_to_block_id[block.hash] == block.block_id

IMPORT_CHECK = '''
import importlib, os, sys
from pathlib import Path
import torch
os.environ.setdefault('SSD_HF_CACHE', '/tmp/unused-import-hf')
os.environ.setdefault('SSD_DATASET_DIR', '/tmp/unused-import-datasets')
root = Path(sys.argv[1])
count = 0
for path in sorted((root / 'jetspec/engines/jetspec_plus_plus').rglob('*.py')):
    name = '.'.join(path.relative_to(root).with_suffix('').parts).removesuffix('.__init__')
    importlib.import_module(name)
    count += 1
print(f'IMPORTS_PASS {count} modules')
'''


def test_entry_maps_user_paths_without_loading_models():
    from jetspec.engines.jetspec_plus_plus import entry
    args = entry.parser().parse_args([
        '--config', 'a100-j1-r0', '--target', '/target', '--drafter', '/draft',
        '--adapter', '/adapter', '--heads-bundle', '/heads',
        '--prompts-json', '/prompts.json', '--tokens', '64', '--placement', 'pair',
        '--devices', '0,1', '--output-json', '/read.json'])
    recipe, argv, host = entry.resolve(args)
    assert recipe['environment']['JETSPEC_ATTN_SM80'] == '1'
    assert host['environment']['JETSPEC_TARGET_MODEL_PATH'] == '/target'
    assert host['artifacts'][recipe['drafter_artifact']] == '/draft'
    assert host['artifacts'][recipe['bundle_artifact']] == '/heads'
    assert argv[argv.index('--temperature') + 1] == '0'
    assert argv[argv.index('--tokens') + 1] == '64'


def test_all_engine_modules_import_on_cpu():
    completed = subprocess.run([sys.executable, '-c', IMPORT_CHECK, str(ROOT)], cwd=ROOT,
                               env=dict(os.environ, PYTHONPATH=str(ROOT)),
                               text=True, capture_output=True)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    expected = sum(1 for _ in ENGINE.rglob('*.py'))
    assert f'IMPORTS_PASS {expected} modules' in completed.stdout


def test_entry_uses_persistent_cache_environment_and_flag_precedence(monkeypatch):
    from jetspec.engines.jetspec_plus_plus import entry
    monkeypatch.setenv('JSPP_CACHE_ROOT', '/persistent-cache')
    common = ['--config', 'reference-j1-r1', '--prompts-json', '/prompts.json',
              '--output-json', '/read.json']
    _, argv, _ = entry.resolve(entry.parser().parse_args(common))
    assert argv[argv.index('--cache-root') + 1] == '/persistent-cache'
    _, argv, _ = entry.resolve(entry.parser().parse_args(common + ['--cache-root', '/explicit']))
    assert argv[argv.index('--cache-root') + 1] == '/explicit'


@pytest.mark.parametrize('config', [None, 'b200-j1-r1'])
def test_entry_selects_b200_fast6_recipe_and_host_architecture(monkeypatch, config):
    import torch
    from jetspec.engines.jetspec_plus_plus import entry
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'get_device_capability', lambda device: (10, 0))
    argv = ['--prompts-json', '/prompts.json', '--output-json', '/read.json']
    if config:
        argv += ['--config', config]
    recipe, _, host = entry.resolve(entry.parser().parse_args(argv))
    assert recipe['name'] == 'jetspecpp-graphed-pair-e6v3-b16-chain-r4-j1r1-lean-swap-fast6'
    assert recipe['environment']['JETSPEC_CHAIN_FIXED_CONFIG'] == 'J1-R1'
    assert recipe['environment']['JETSPEC_VERIFY_DEVICE'] == '1'
    assert host['cuda_arch'] == '10.0'


def test_manifest_lists_only_the_supported_artifacts():
    import json
    manifest = json.loads((ENGINE / 'artifacts/manifest.json').read_text())
    names = {row['name'] for row in manifest['artifacts']}
    assert names == {'jetspecpp_e6_drafter', 'jetspecpp_e6_adapter', 'mask_original_drafter',
                     'served_heads_reused_e6v2_b16', 'served_heads_reused_e6v3_b16'}


def test_entry_passes_explicit_memory_and_capture_knobs_to_the_reader():
    from jetspec.engines.jetspec_plus_plus import entry
    knobs = ['SSD_SHARED_GPU_KV_UTILIZATION=0.1', 'JETSPEC_CHAIN_WARM_REACHABLE=1',
             'JETSPEC_CHAIN_WARM_SINGLE_KEY=1']
    args = entry.parser().parse_args([
        '--config', 'a100-j1-r0', '--prompts-json', '/prompts.json',
        '--output-json', '/read.json', *[item for knob in knobs for item in ('--knob', knob)]])
    _, argv, _ = entry.resolve(args)
    assert [argv[i + 1] for i, value in enumerate(argv) if value == '--knob'] == knobs


def test_entry_ar_baseline_runs_one_gpu_without_draft_artifacts():
    from jetspec.engines.jetspec_plus_plus import entry
    args = entry.parser().parse_args([
        '--config', 'ar', '--target', '/target', '--prompts-json', '/prompts.json',
        '--devices', '2,3', '--output-json', '/ar.json'])
    recipe, argv, host = entry.resolve(args)
    assert recipe['method'] == 'ar'
    assert argv[argv.index('--placement') + 1] == 'one_gpu'
    assert argv[argv.index('--devices') + 1] == '2'
    assert host['artifacts'] == {}
    assert host['environment']['JETSPEC_TARGET_MODEL_PATH'] == '/target'


def test_speedup_runs_ar_then_jetspec_plus_plus_and_divides_decode_rates(tmp_path):
    import json
    sys.path.insert(0, str(ROOT))
    from bench.jetspec_plus_plus import speedup
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        output = Path(command[command.index('--output-json') + 1])
        rate = 50.0 if '--config' in command and command[command.index('--config') + 1] == 'ar' else 200.0
        output.write_text(json.dumps({
            'outputs': [{'token_ids': [1, 2]}] * 2,
            'metrics': {'serve_read': {'measured': {'decode_tok_s': rate}}}}))

    assert speedup.main(['--config', 'a100-j1-r0', '--target', '/t', '--drafter', '/d',
                         '--adapter', '/a', '--heads-bundle', '/h',
                         '--prompts-json', '/p.json', '--output-dir', str(tmp_path),
                         '--knob', 'SSD_SHARED_GPU_KV_UTILIZATION=0.1'], run=run) == 0
    ar, method = calls
    assert '--drafter' not in ar and ar[ar.index('--config') + 1] == 'ar'
    assert method[method.index('--config') + 1] == 'a100-j1-r0' and '--drafter' in method
    assert ar[ar.index('--stop-at-eos') + 1] == method[method.index('--stop-at-eos') + 1] == 'on'
    assert '--knob' not in ar
    assert method[method.index('--knob') + 1] == 'SSD_SHARED_GPU_KV_UTILIZATION=0.1'
    result = json.loads((tmp_path / 'speedup.json').read_text())
    assert result['speedup'] == 4.0 and result['prompts'] == 2


def test_source_revision_is_null_outside_git_without_a_receipt(tmp_path, monkeypatch):
    from jetspec.engines.jetspec_plus_plus.serving import read
    monkeypatch.delenv('RX_SOURCE_PROVENANCE', raising=False)
    package = tmp_path / 'site/jetspec_plus_plus'
    package.mkdir(parents=True)
    assert read.source_revision(package) == (None, None)
