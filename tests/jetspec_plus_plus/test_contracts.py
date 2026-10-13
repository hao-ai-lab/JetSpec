"""The torch hooks preserve the lazy package and the existing DFlash tree token contract."""
# Ported from the JetSpec++ reference implementation.
from golden_values import golden
from types import ModuleType
from pathlib import Path
import subprocess
import sys
import pytest
import torch
from conftest import tiny_target
from test_lossless import llm_for
from test_pass import draft_config
from jetspec.methods.jetspec.draft_head import DFlashDraftModel
from jetspec.core.llm import LLM, SamplingParams, emit_verified_block


def test_lazy_imports():
    for imported in ('jetspec', 'jetspec.methods.jetspec.tree', 'jetspec.methods.jetspec_plus_plus'):
        script = f"""
import sys
import {imported}
assert not any(name.startswith(('jetspec.models', 'jetspec.core', 'transformers')) for name in sys.modules)
"""
        result = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True)
        assert result.returncode == 0, result.stderr


@pytest.mark.parametrize('seed', [0, 1, 2])
def test_existing_dflash_tree(seed):
    source = (Path(__file__).parent / 'fixtures/reference/llm.py').read_text()
    original = ModuleType('baseline_llm')
    sys.modules[original.__name__] = original
    exec(compile(source, '2c7b3fa/llm.py', 'exec'), original.__dict__)
    target = tiny_target(seed)
    head = DFlashDraftModel(draft_config()).eval()
    class TreeHead:
        def propose_logits(self, ids, depth, target_hidden=None, **kwargs):
            noise = target.model.embed_tokens(ids[:, -1:].expand(1, depth + 1))
            positions = torch.arange(target_hidden.shape[1] + depth + 1)[None]
            hidden = head(target_hidden=target_hidden, noise_embedding=noise, position_ids=positions,
                          past_key_values=None, use_cache=False, is_causal=True)
            return target.lm_head(hidden[:, 1:])
    args = dict(block_size=16, tree_width=1, budget=16, target_layer_ids=[0, 1],
                sampling_params=SamplingParams(max_new_tokens=64), return_stats=True)
    prompt = torch.tensor([[1, 3, 4, 5, 8, 2]])
    expected = llm_for(target, original.LLM).generate_tree(prompt, TreeHead(), **args)
    actual = llm_for(target).generate_tree(prompt, TreeHead(), **args)
    assert actual['token_ids'] == expected['token_ids']


@pytest.mark.parametrize('seed', [0, 1, 2])
def test_emit_block(seed):
    torch.manual_seed(seed)
    block = torch.randint(0, 7, (16,)).tolist()
    for cap in [0, 3, 40]:
        left = [1, 2]
        assert emit_verified_block(left, block, {4}, cap) == golden('value_1')
        assert left == golden('emitted_tokens')
