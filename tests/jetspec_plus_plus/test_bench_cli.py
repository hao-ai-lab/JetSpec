"""User benchmark entry points preserve argv and run on local CPU artifacts."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = [
    'engine/tps_walltime.py', 'reference/benchmark.py',
    'reference/hf_fa2_decode_baseline.py', 'reference/dflash_baseline.py',
    'profiling/collect_tree_diagnostics.py', 'profiling/depth_rank_profile.py',
    'profiling/compare_conditioned_draft_logits.py',
    'profiling/compare_engine_with_vllm_integration.py',
    'profiling/collect_depth_rank_stats.py',
]


def cli(path, *args, env=None):
    script = ('import runpy, sys, torch; torch.set_num_threads(1); '
              'sys.argv = sys.argv[1:]; runpy.run_path(sys.argv[0], run_name="__main__")')
    command = ([sys.executable, str(ROOT / path), *args] if args == ('--help',) else
               [sys.executable, '-c', script, str(ROOT / path), *args])
    environment = dict(os.environ)
    environment.pop('PYTHONPATH', None)
    environment.update(env or {})
    return subprocess.run(command, cwd=ROOT, env=environment,
                          capture_output=True, text=True)


@pytest.mark.parametrize('script', SCRIPTS)
def test_old_bench_help_matches_moved_script(script):
    old = cli('bench/' + script, '--help')
    new = cli('bench/jetspec/' + script, '--help')
    assert old.returncode == new.returncode == 0, old.stderr + new.stderr
    assert old.stdout == new.stdout


def local_cli_inputs(tmp_path):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast
    from test_artifacts import local_artifacts

    target, head, adapter, bundle, _ = local_artifacts(tmp_path, 0)
    path = target.config._name_or_path
    PreTrainedTokenizerFast(tokenizer_object=Tokenizer(
        WordLevel({str(i): i for i in range(41)}, unk_token='0')),
        unk_token='0').save_pretrained(path)
    prompts = tmp_path / 'prompts.json'
    prompts.write_text(json.dumps([{'token_ids': [1, 2, 3, 4, 5, 6]},
                                  {'token_ids': [2, 3, 4, 5, 6, 7]}]))
    return ['--model', path, '--drafter', str(head), '--adapter', str(adapter),
            '--heads-bundle', str(bundle), '--dataset', 'tiny',
            '--prompts-file', str(prompts), '--samples', '2', '--max-tokens', '16',
            '--device', 'cpu', '--dtype', 'fp32']


def test_tau_cli_reads_local_artifacts_and_writes_summary(tmp_path):
    output = tmp_path / 'tau.json'
    result = cli('bench/jetspec_plus_plus/tau.py', *local_cli_inputs(tmp_path),
                 '--output', str(output))
    assert result.returncode == 0, result.stderr
    read = json.loads(output.read_text())
    summary = read['summary']
    assert summary['n_prompts'] == 2
    assert summary['tokens'] == 32
    assert summary['rounds'] > 0
    assert summary['tau'] == summary['accepted_sum'] / summary['rounds']
    assert summary['wall_seconds'] > 0
    assert summary['tokens_per_second'] == summary['tokens'] / summary['wall_seconds']
    assert read['manifest']['config']['rows_min'] == 16
    assert 'tau=' in result.stdout and 'tok/s' in result.stdout


def test_verify_cli_matches_ar_on_local_artifacts(tmp_path):
    output = tmp_path / 'verify.json'
    result = cli('bench/jetspec_plus_plus/verify.py', *local_cli_inputs(tmp_path),
                 '--output', str(output))
    assert result.returncode == 0, result.stderr
    assert 'PASS' in result.stdout and 'speedup=' in result.stdout
    read = json.loads(output.read_text())
    assert read['equal'] is True
    assert read['ar_wall_seconds'] > 0 and read['summary']['wall_seconds'] > 0
    assert read['speedup'] == read['ar_wall_seconds'] / read['summary']['wall_seconds']
    assert [row['token_ids'] for row in read['outputs']] == [row['token_ids'] for row in read['ar_outputs']]


def test_verify_exits_one_for_wrong_emitted_token_source(tmp_path):
    """Fault injection bypasses target verification; bad proposals alone are lossless."""
    args = local_cli_inputs(tmp_path) + ['--output', str(tmp_path / 'wrong.json')]
    script = '''
import sys, torch
torch.set_num_threads(1)
from jetspec.core.llm import LLM
from bench.jetspec_plus_plus.verify import main

class WrongTokenSource:
    def generate(self, count):
        return [40] * count

def target_factory(**kwargs):
    llm = LLM(**kwargs)
    # A deliberately broken method emits its draft without target correction.
    def unchecked(prompt, source, **options):
        ids = source.generate(options['sampling_params'].max_new_tokens)
        return dict(token_ids=ids, accept_lengths=[len(ids) - 1])
    llm.generate_tree = unchecked
    return llm

raise SystemExit(main(sys.argv[1:], llm_factory=target_factory,
                      drafter_factory=lambda **kwargs: WrongTokenSource()))
'''
    result = subprocess.run([sys.executable, '-c', script, *args], cwd=ROOT,
                            env=dict(os.environ, PYTHONPATH=str(ROOT)),
                            capture_output=True, text=True)
    assert result.returncode == 1, result.stderr
    assert 'FAIL' in result.stdout and 'PASS' not in result.stdout
    assert json.loads((tmp_path / 'wrong.json').read_text())['equal'] is False


@pytest.mark.parametrize('script', ['tau', 'verify'])
def test_user_command_help(script):
    result = cli(f'bench/jetspec_plus_plus/{script}.py', '--help')
    assert result.returncode == 0, result.stderr
    for flag in ('--prompt-set', '--samples', '--max-tokens', '--model', '--drafter',
                 '--adapter', '--heads-bundle', '--output'):
        assert flag in result.stdout


def test_tau_model_paths_from_environment(tmp_path):
    args = local_cli_inputs(tmp_path)
    env = {}
    for flag, name in [('--model', 'JETSPEC_MODEL'), ('--drafter', 'JETSPEC_DRAFTER'),
                       ('--adapter', 'JETSPEC_ADAPTER'),
                       ('--heads-bundle', 'JETSPEC_HEADS_BUNDLE')]:
        at = args.index(flag)
        env[name] = args[at + 1]
        del args[at:at + 2]
    output = tmp_path / 'env.json'
    result = cli('bench/jetspec_plus_plus/tau.py', *args, '--output', str(output), env=env)
    assert result.returncode == 0, result.stderr
    read = json.loads(output.read_text())
    assert read['manifest']['target'] == env['JETSPEC_MODEL']
    assert read['summary']['tokens'] == 32
