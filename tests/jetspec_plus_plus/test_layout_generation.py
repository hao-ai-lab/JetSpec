"""The method layout preserves tree generation against the frozen public core."""
import importlib
from pathlib import Path
import sys
from types import ModuleType

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM
from jetspec.core.model_runner import ModelRunner


@pytest.fixture(autouse=True)
def seed_and_single_thread():
    torch.set_num_threads(1)
    torch.manual_seed(0)


def tiny_target(seed):
    torch.manual_seed(seed)
    config = Qwen3Config(vocab_size=41, hidden_size=16, intermediate_size=32,
                        num_hidden_layers=4, num_attention_heads=2,
                        num_key_value_heads=1, head_dim=8)
    config._attn_implementation = 'sdpa'
    return Qwen3ForCausalLM(config).eval()


def draft_config():
    config = Qwen3Config(vocab_size=41, hidden_size=16, intermediate_size=32,
                         num_hidden_layers=5, num_attention_heads=2,
                         num_key_value_heads=1, head_dim=8, num_target_layers=36,
                         block_size=16, dflash_config=dict(causal_head=True,
                         jacobi_forcing=True, stock_drafter=True, fresh_fraction=1.0,
                         target_layer_ids=[1, 9, 17, 25, 33], mask_token_id=0))
    config._attn_implementation = 'sdpa'
    return config


def make_llm(module, target):
    llm = module.LLM.__new__(module.LLM)
    llm.model = target
    llm.runner = ModelRunner(target)
    llm.device = 'cpu'
    llm.dtype = torch.float32
    llm.eos_token_ids = set()
    return llm


@pytest.mark.parametrize('seed', [0, 1, 2])
def test_generate_tree_tokens_equal_public_baseline(seed):
    """Reuse the trained-head hook pattern on 64 fp32 tokens without downloads."""
    source = (Path(__file__).parent / 'fixtures/reference/llm.py').read_bytes()
    baseline = ModuleType('layout_baseline_llm')
    sys.modules[baseline.__name__] = baseline
    exec(compile(source, '2c7b3fa/llm.py', 'exec'), baseline.__dict__)
    core = importlib.import_module('jetspec.core.llm')
    from jetspec.methods.jetspec.draft_head import DFlashDraftModel
    from jetspec.methods.jetspec.draft_head_adapter import DraftHeadTreeDrafter
    target = tiny_target(seed)
    config = draft_config()
    config.num_hidden_layers = 2
    config.layer_types = ['full_attention'] * 2
    config.dflash_config['target_layer_ids'] = [0, 2]
    head = DFlashDraftModel(config).eval()
    drafter = DraftHeadTreeDrafter(head, target, 16, [0, 2])
    prompt = torch.randint(0, 41, (1, 5))
    results = [make_llm(module, target).generate_tree(
        prompt, drafter, block_size=16, tree_width=1, budget=16,
        target_layer_ids=[0, 2], sampling_params=module.SamplingParams(max_new_tokens=64),
        return_stats=True, return_text=False) for module in (baseline, core)]
    assert results[0]['token_ids'] == results[1]['token_ids']
