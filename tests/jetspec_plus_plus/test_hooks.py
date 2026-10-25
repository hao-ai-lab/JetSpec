"""Core hooks preserve baseline tokens and the frozen reference's emitted round credit."""
# Ported from the JetSpec++ reference implementation.
from golden_values import golden
import importlib.util
import json
from pathlib import Path
import sys

import pytest
import torch
from transformers import Qwen3Config

from conftest import tiny_target
from jetspec.core import llm as core
from jetspec.core.model_runner import ModelRunner
from jetspec.methods.jetspec.draft_head_adapter import DraftHeadTreeDrafter
from jetspec.methods.jetspec.draft_head import DFlashDraftModel
from real_artifacts import bundled_model, real_checkpoint


def make_llm(module, target, eos=()):
    llm = module.LLM.__new__(module.LLM)
    llm.model = target
    llm.runner = ModelRunner(target)
    llm.device = 'cpu'
    llm.dtype = torch.float32
    llm.eos_token_ids = set(eos)
    return llm


def draft_config():
    config = Qwen3Config(vocab_size=41, hidden_size=16, intermediate_size=32,
                         num_hidden_layers=5, num_attention_heads=2,
                         num_key_value_heads=1, head_dim=8, num_target_layers=36,
                         block_size=16, dflash_config=dict(causal_head=True,
                         jacobi_forcing=True, stock_drafter=True, fresh_fraction=1.0,
                         target_layer_ids=[1, 9, 17, 25, 33],
                         mask_token_id=0))
    config._attn_implementation = 'sdpa'
    return config


@pytest.mark.parametrize('seed', [0, 1, 2])
def test_emit_verified_block_matches_golden(seed):
    torch.manual_seed(seed)
    block = torch.randint(0, 7, (12,)).tolist()
    for cap in [1, 5, 20]:
        for eos in [set(), {block[3]}]:
            actual = [8]
            count = core.emit_verified_block(actual, block, eos, cap)
            assert count == golden('value_1')
            assert torch.equal(torch.tensor(actual), torch.tensor(golden('emitted_tokens')))


@pytest.mark.parametrize('seed', [0, 1, 2])
def test_tree_tokens_match_original_and_golden(seed, tmp_path):
    path = tmp_path / 'original_llm.py'
    path.write_bytes((Path(__file__).parent / 'fixtures/reference/llm.py').read_bytes())
    spec = importlib.util.spec_from_file_location('original_llm', path)
    original = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = original
    spec.loader.exec_module(original)
    target = tiny_target(seed)
    config = draft_config()
    config.num_hidden_layers = 2
    config.layer_types = ['full_attention'] * 2
    config.dflash_config['target_layer_ids'] = [0, 2]
    head = DFlashDraftModel(config).eval()
    drafter = DraftHeadTreeDrafter(head, target, 16, [0, 2])
    prompt = torch.randint(0, 41, (1, 5))
    results = []
    for module in [original, core]:
        results.append(make_llm(module, target).generate_tree(
            prompt, drafter, block_size=16, tree_width=1, budget=16,
            target_layer_ids=[0, 2], sampling_params=module.SamplingParams(max_new_tokens=64),
            return_stats=True, return_text=False))
    for result in results[1:]:
        assert torch.equal(torch.tensor(result['token_ids']), torch.tensor(results[0]['token_ids']))
    expected = golden('engine_tree')
    assert torch.equal(torch.tensor(results[1]['token_ids']), torch.tensor(expected['token_ids']))
    assert results[1]['accept_lengths'] == expected['accept_lengths']
    assert sum(results[1]['accept_lengths']) == 63


def test_prompt_metadata_and_observe_reach_drafter():
    from jetspec.methods.jetspec.draft import TargetEchoTreeDrafter

    class ObservedDrafter(TargetEchoTreeDrafter):
        def __init__(self, model):
            super().__init__(model)
            self.proposals, self.observations = [], []

        def propose_logits(self, *args, **kwargs):
            self.proposals.append((kwargs.pop('prompt_index'), kwargs.pop('seed')))
            return super().propose_logits(*args, **kwargs)

        def observe(self, acc, correction):
            self.observations.append((int(acc), int(correction)))

    target = tiny_target(0)
    drafter = ObservedDrafter(target)
    result = make_llm(core, target).generate_tree(
        torch.tensor([[1, 2, 3]]), drafter, block_size=4, tree_width=1, budget=4,
        sampling_params=core.SamplingParams(max_new_tokens=8), return_stats=True,
        return_text=False, prompt_index=7, seed=9504)
    assert drafter.proposals == [(7, 9504)] * result['rounds']
    assert len(drafter.observations) == result['rounds']
    assert all(acc == 3 for acc, _ in drafter.observations)
    assert result['accept_lengths'] == [4, 3]


@pytest.mark.parametrize('seed', [0, 1, 2])
def test_public_dflash_is_not_bundled_checkpoint_model(seed):
    """One cached fp32 batch differs despite identical stock state keys.

    The bundled code queries layer zero's cache length at every layer. Retain
    that checkpoint behavior rather than changing its attention in the loader.
    """
    from transformers import DynamicCache

    torch.manual_seed(seed)
    bundled = bundled_model(draft_config())
    public = DFlashDraftModel(draft_config()).eval()
    public.load_state_dict(bundled.state_dict(), strict=True)
    inputs = dict(target_hidden=torch.randn(1, 3, 80), noise_embedding=torch.randn(1, 4, 16),
                  position_ids=torch.arange(7)[None], use_cache=True)
    with torch.inference_mode():
        expected = bundled(**inputs, past_key_values=DynamicCache())
        actual = public(**inputs, past_key_values=DynamicCache())
    assert not torch.equal(actual, expected)


def checkpoint(tmp_path, seed):
    return real_checkpoint(tmp_path, seed)


@pytest.mark.parametrize('seed', [0, 1, 2])
def test_drafter_loader_matches_bundled_forward(seed, tmp_path):
    """Load the served stock family and preserve its bundled cached forward."""
    from jetspec.methods.jetspec_plus_plus.models.drafter import load_drafter
    from transformers import DynamicCache

    expected = checkpoint(tmp_path, seed)
    actual = load_drafter(tmp_path, device='cpu', dtype=torch.float32)
    inputs = dict(target_hidden=torch.randn(1, 3, 80), noise_embedding=torch.randn(1, 4, 16),
                  position_ids=torch.arange(7)[None], use_cache=True)
    with torch.inference_mode():
        assert torch.equal(actual(**inputs, past_key_values=DynamicCache()),
                           expected(**inputs, past_key_values=DynamicCache()))
    assert actual.target_layer_ids == [1, 9, 17, 25, 33]
    assert len(actual.layers) == 5


@pytest.mark.parametrize('change', ['missing', 'unexpected', 'shape', 'layers', 'taps',
                                    'auto_map', 'revision', 'missing_code'])
def test_drafter_loader_fails_closed(change, tmp_path):
    from jetspec.methods.jetspec_plus_plus.models.drafter import load_drafter
    from safetensors.torch import load_file, save_file

    checkpoint(tmp_path, 0)
    if change in ['missing', 'unexpected', 'shape']:
        state = load_file(tmp_path / 'model.safetensors')
        if change == 'missing':
            state.pop('fc.weight')
        elif change == 'unexpected':
            state['unknown.weight'] = torch.zeros(1)
        else:
            state['fc.weight'] = torch.zeros(1)
        save_file(state, tmp_path / 'model.safetensors')
    elif change == 'missing_code':
        (tmp_path / 'dflash.py').rename(tmp_path / 'unused.py')
    else:
        config = json.loads((tmp_path / 'config.json').read_text())
        if change == 'layers':
            config['num_hidden_layers'] = 4
        elif change == 'taps':
            config['dflash_config']['target_layer_ids'] = [0, 1, 2, 3, 4]
        elif change == 'auto_map':
            config['auto_map']['AutoModel'] = 'other.Model'
        (tmp_path / 'config.json').write_text(json.dumps(config))
    options = {'revision': 'main'} if change == 'revision' else {}
    with pytest.raises((RuntimeError, ValueError, FileNotFoundError)):
        load_drafter(tmp_path, device='cpu', dtype=torch.float32, **options)


@pytest.mark.parametrize('seed', [0, 1, 2])
def test_loader_leaf_functions_match_golden(seed, tmp_path):
    from jetspec.methods.jetspec_plus_plus.models import drafter

    torch.manual_seed(seed)
    keys = {f'weight.{n}' for n in torch.randint(0, 20, (5,)).tolist()}
    for module in [drafter]:
        module.check_state_keys(keys, keys)
        with pytest.raises(RuntimeError):
            module.check_state_keys(keys, keys | {'unexpected'})
        with pytest.raises(RuntimeError):
            module.check_state_keys(keys, set())
    assert golden('key_contract') == {'equal': True, 'unexpected': 'RuntimeError', 'missing': 'RuntimeError'}
    raw = draft_config().to_dict()
    raw.pop('rope_theta', None)
    raw['rope_parameters'] = {'rope_theta': float(10000 + seed), 'rope_type': 'default'}
    (tmp_path / 'config.json').write_text(json.dumps(raw))
    configs = [module.load_qwen3_config_with_rope_compat(str(tmp_path))
               for module in [drafter]]
    for module, config in zip([drafter], configs):
        del config.rope_theta
        module.restore_rope_theta(config)
    assert torch.equal(golden('restored_rope'), torch.tensor(configs[0].rope_theta))
