"""Narrow replay consumes the actual golden byte writer and exact candidate schema."""
from golden_values import golden
import copy
import json
import subprocess
import sys

import pytest
import torch

from conftest import tiny_target
from jetspec.methods.jetspec_plus_plus.models.decoder import AnchorDecoder


def capture(tmp_path, seed=0, target=None):
    target = tiny_target(seed) if target is None else target
    taps_ids = [0, 0, 0, 0, 1] if target.config.num_hidden_layers == 4 else [1, 9, 17, 25, 33]
    decoder = AnchorDecoder.from_target(target, tap_ids=taps_ids)
    thresholds = dict(e04_kscore_blend={'tiny': {k: {'value': v} for k, v in
        zip(('a', 'b', 'c', 'argmax', 'top', 'pos'), (.7, .2, -.5, .1, .2, .3))}},
        e04_rowrule_joint={'tiny': {'tau': {'pooled': {'value': .001}},
                                   'temperature': {'value': 1.1}}})
    records = golden('capture_records')
    candidates = tmp_path / 'read.json.candidates.jsonl'
    candidates.write_text(''.join(json.dumps(row) + '\n' for row in records))
    index = golden('capture_index')
    (tmp_path / 'read.json.decoder_inputs.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in index))
    (tmp_path / 'read.json.decoder_inputs.bin').write_bytes(golden('capture_bytes').numpy().tobytes())
    return decoder, thresholds, candidates, records


@pytest.mark.parametrize('seed', [0, 1, 2])
def test_replay_matching_and_corrupt_capture(tmp_path, seed):
    from tests.jetspec_plus_plus.parity.replay_engine_rounds import replay
    decoder, thresholds, candidates, records = capture(tmp_path, seed)
    index = tmp_path / 'read.json.decoder_inputs.jsonl'
    data = tmp_path / 'read.json.decoder_inputs.bin'
    result = replay(candidates, index, data, decoder=decoder, thresholds=thresholds, dataset='tiny')
    assert result['equal'], result
    assert result['fields']['published_keys']['rounds_equal'] == 2
    assert result['fields']['q_shadow']['rounds_compared'] == 2
    assert result['fields']['p_pred']['rounds_equal'] == 2
    assert result['fields']['p_wire']['rounds_equal'] == 2
    corrupt = copy.deepcopy(records)
    corrupt[1]['published_keys'][0][1] += 1
    candidates.write_text(''.join(json.dumps(row) + '\n' for row in corrupt))
    result = replay(candidates, index, data, decoder=decoder, thresholds=thresholds, dataset='tiny')
    assert not result['equal']
    first = result['fields']['published_keys']['first_difference']
    assert first['candidate_round'] == 1
    assert first['reference'] == corrupt[1]['published_keys']
    assert first['candidate'] == records[1]['published_keys']


def test_replay_refuses_missing_and_truncated_inputs(tmp_path):
    from tests.jetspec_plus_plus.parity.replay_engine_rounds import replay
    decoder, thresholds, candidates, records = capture(tmp_path)
    index = tmp_path / 'read.json.decoder_inputs.jsonl'
    data = tmp_path / 'read.json.decoder_inputs.bin'
    rows = index.read_text().splitlines()
    index.write_text(rows[0] + '\n')
    with pytest.raises(ValueError, match='missing decoder'):
        replay(candidates, index, data, decoder=decoder, thresholds=thresholds, dataset='tiny')
    index.write_text('\n'.join(rows) + '\n')
    data.write_bytes(data.read_bytes()[:-2])
    with pytest.raises(ValueError, match='truncated'):
        replay(candidates, index, data, decoder=decoder, thresholds=thresholds, dataset='tiny')


def test_replay_help_states_narrow_coverage():
    result = subprocess.run([sys.executable, 'tests/jetspec_plus_plus/parity/replay_engine_rounds.py', '--help'],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert 'does not cover J/R passes, swaps or pins' in result.stdout
    assert '2a' in result.stdout


def test_replay_cli_reports_corruption_and_imports_no_backend(tmp_path):
    from test_artifacts import local_artifacts
    target, _, _, bundle, _ = local_artifacts(tmp_path, 0)
    decoder, thresholds, candidates, records = capture(tmp_path, target=target)
    (bundle / 'thresholds.json').write_text(json.dumps(thresholds))
    output = tmp_path / 'report.json'
    script = ('import runpy, sys, torch\ntorch.set_num_threads(1)\n'
              'try:\n'
              '    runpy.run_path("tests/jetspec_plus_plus/parity/replay_engine_rounds.py", run_name="__main__")\n'
              'except SystemExit as result:\n'
              '    from pathlib import Path\n'
              '    from importlib.metadata import packages_distributions\n'
              '    import sysconfig\n'
              "    known_roots = set(sys.stdlib_module_names) | set(packages_distributions()) | {'__main__', '__mp_main__', 'jetspec', 'tests'}\n"
              "    allowed = [Path(sysconfig.get_path(key)).resolve() for key in ('stdlib', 'platstdlib', 'purelib', 'platlib')]\n"
              "    allowed += [Path.cwd() / 'jetspec', Path.cwd() / 'tests']\n"
              "    templates = sys.modules.get('torch.distributed.nn.jit.instantiator')\n"
              '    if templates:\n'
              '        allowed.append(Path(templates.INSTANTIATED_TEMPLATE_DIR_PATH).resolve())\n'
              '    for name, module in list(sys.modules.items()):\n'
              "        origin = getattr(module, '__file__', None)\n"
              '        if origin and (Path(origin).is_absolute() or Path(origin).exists()):\n'
              '            assert any(Path(origin).resolve().is_relative_to(root) for root in allowed), origin\n'
              '        else:\n'
              "            assert name.split('.')[0] in known_roots or name.startswith(('_cython_', 'cython_runtime')), name\n"
              '    raise result\n')
    command = [sys.executable, '-c', script, '--candidates', str(candidates),
        '--target', target.config._name_or_path, '--heads-bundle', str(bundle), '--dataset', 'tiny',
        '--device', 'cpu', '--dtype', 'fp32', '--output-json', str(output)]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(output.read_text())['equal']
    records[0]['q_shadow'][0] = 0.0
    candidates.write_text(''.join(json.dumps(row) + '\n' for row in records))
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 1, result.stderr
    first = json.loads(output.read_text())['fields']['q_shadow']['first_difference']
    assert first['candidate_round'] == 0
