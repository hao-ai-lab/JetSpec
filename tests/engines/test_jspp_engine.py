"""The JetSpec++ serving engine resolves its configurations and imports on CPU."""
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
ENGINE = ROOT / 'jetspec/engines/jetspec_plus_plus'

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
