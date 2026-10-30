"""Reader options preserve fp32 reads and share the AR target and prompt bank."""
import importlib.util
import hashlib
import json
from pathlib import Path
import sys

import pytest
import torch

from test_artifacts import local_artifacts
from test_lossless import llm_for
from jetspec.core.llm import SamplingParams
from jetspec.methods.jetspec_plus_plus import reader
from jetspec.methods.jetspec_plus_plus.drafter import build_drafter


def baseline_reader(tmp_path):
    path = tmp_path / 'baseline_reader.py'
    path.write_bytes((Path(__file__).parent / 'fixtures/reference/reader_initial.py').read_bytes())
    spec = importlib.util.spec_from_file_location(
        'jetspec.methods.jetspec_plus_plus._baseline_reader', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('seed', [0, 1, 2])
def test_local_reader_fp32_matches_baseline_and_ar(tmp_path, seed):
    target, head, adapter, bundle, _ = local_artifacts(tmp_path, seed)
    prompts = tmp_path / 'prompts.json'
    rows = [{'token_ids': [1, 2, 3, 4, 5, 6]},
            {'token_ids': [2, 3, 4, 5, 6, 7]},
            {'index': 27, 'token_ids': [3, 4, 5, 6, 7, 8]}]
    prompts.write_text(json.dumps(rows))
    args = ['--target', target.config._name_or_path, '--drafter', str(head),
            '--adapter', str(adapter), '--heads-bundle', str(bundle),
            '--prompts-file', str(prompts), '--max-new-tokens', '16',
            '--device', 'cpu', '--dtype', 'fp32']
    factory = lambda **kwargs: build_drafter(**kwargs, dataset='tiny')
    llm_factory = lambda **kwargs: llm_for(target)
    old = baseline_reader(tmp_path).main(
        args + ['--output', str(tmp_path / 'old.json')],
        drafter_factory=factory, llm_factory=llm_factory)
    new = reader.main(args + ['--output', str(tmp_path / 'new.json')],
                      drafter_factory=factory, llm_factory=llm_factory)
    assert [r['token_ids'] for r in new['outputs']] == [r['token_ids'] for r in old['outputs']]
    assert [r['accept_lengths'] for r in new['outputs']] == [r['accept_lengths'] for r in old['outputs']]
    assert all('accepted_len' in r for r in new['metrics']['jetspec_round_records'])
    assert new['manifest']['prompt_inputs_sha256'] == hashlib.sha256(json.dumps(
        [row['token_ids'] for row in rows], sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    ar = reader.main(args + ['--ar', '--prompt-start', '1', '--prompt-count', '2',
                            '--output', str(tmp_path / 'ar.json')], llm_factory=llm_factory)
    assert [r['prompt_index'] for r in ar['outputs']] == [1, 27]
    for row, actual in zip(rows[1:], ar['outputs']):
        expected = llm_for(target).generate(torch.tensor([row['token_ids']]),
                                           SamplingParams(0, 16))['token_ids']
        assert actual['token_ids'] == expected
        assert actual['accept_lengths'] == [1] * (len(expected) - 1)
    assert ar['manifest']['method'] == 'ar'
    assert ar['manifest']['prompt_start'] == 1
    assert ar['rule_a']['tokens'] == sum(len(r['token_ids']) - 1 for r in ar['outputs'])


def test_reader_default_factory_and_placement(tmp_path):
    target, head, adapter, bundle, _ = local_artifacts(tmp_path, 4)
    prompts = tmp_path / 'prompts.json'
    prompts.write_text(json.dumps([{'token_ids': [1, 2, 3, 4, 5, 6]}]))
    seen = {}
    def llm_factory(**kwargs):
        seen.update(kwargs)
        return llm_for(target)
    read = reader.main(['--target', target.config._name_or_path, '--drafter', str(head),
        '--adapter', str(adapter), '--heads-bundle', str(bundle), '--dataset', 'tiny',
        '--prompts-file', str(prompts), '--output', str(tmp_path / 'read.json'),
        '--max-new-tokens', '2', '--device', 'cuda', '--dtype', 'bf16'],
        llm_factory=llm_factory)
    assert seen['device'] == 'cuda'
    assert seen['dtype'] == torch.bfloat16
    assert read['manifest']['device'] == 'cuda'


def test_reader_loads_bf16_local_models_on_cpu(tmp_path):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast
    from jetspec.core.llm import LLM
    target, head, adapter, bundle, _ = local_artifacts(tmp_path, 4)
    path = target.config._name_or_path
    PreTrainedTokenizerFast(tokenizer_object=Tokenizer(
        WordLevel({str(i): i for i in range(41)}, unk_token='0')), unk_token='0').save_pretrained(path)
    prompts = tmp_path / 'prompts.json'
    prompts.write_text(json.dumps([{'token_ids': [1, 2, 3, 4, 5, 6]}]))
    seen = {}
    def llm_factory(**kwargs):
        llm = LLM(**kwargs)
        seen['target_dtype'] = next(llm.model.parameters()).dtype
        return llm
    def factory(**kwargs):
        drafter = build_drafter(**kwargs, dataset='tiny')
        seen['head_dtype'] = next(drafter.driver.head.parameters()).dtype
        seen['adapter_dtype'] = next(drafter.driver.adapter.parameters()).dtype
        return drafter
    read = reader.main(['--target', path, '--drafter', str(head), '--adapter', str(adapter),
        '--heads-bundle', str(bundle), '--prompts-file', str(prompts),
        '--output', str(tmp_path / 'bf16.json'), '--max-new-tokens', '2',
        '--device', 'cpu', '--dtype', 'bf16'], drafter_factory=factory, llm_factory=llm_factory)
    assert set(seen.values()) == {torch.bfloat16}
    assert len(read['outputs'][0]['token_ids']) == 2


@pytest.mark.parametrize('options', [['--prompt-start', '-1'], ['--prompt-count', '0'],
                                    ['--prompt-start', '8']])
def test_reader_rejects_invalid_slice_before_loading(tmp_path, options):
    prompts = tmp_path / 'prompts.json'
    prompts.write_text(json.dumps(['one prompt']))
    with pytest.raises(SystemExit):
        reader.main(['--target', 'unused', '--ar', '--prompts-file', str(prompts),
                     '--output', str(tmp_path / 'read.json')] + options,
                    llm_factory=lambda **kwargs: pytest.fail('loaded invalid selection'))
