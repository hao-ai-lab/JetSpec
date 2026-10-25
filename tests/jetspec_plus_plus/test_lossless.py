"""Tiny random fp32 Qwen3 decoding remains lossless with real J/R, swap and keyed rows."""
from collections import Counter
from types import SimpleNamespace
import pytest
import torch
from jetspec.core.llm import LLM, SamplingParams
from jetspec.core.model_runner import ModelRunner
from jetspec.methods.jetspec_plus_plus.drafter import JetSpecPPDrafter
from test_drafter import thresholds
from test_round import components


def llm_for(target, cls=LLM):
    llm = cls.__new__(cls)
    llm.model = target
    llm.tokenizer = SimpleNamespace(decode=lambda ids, **kwargs: ' '.join(map(str, ids)))
    llm.runner = ModelRunner(target)
    llm.device, llm.dtype, llm.eos_token_ids = 'cpu', torch.float32, set()
    return llm


@pytest.mark.parametrize('seed', [0, 1, 2])
def test_lossless(seed):
    target, driver, taps = components(seed)
    assert driver.head.stock_drafter
    assert 'adapted_position_embedding.weight' not in driver.head.state_dict()
    assert list(driver.head.adapted_position_embedding.parameters()) == []
    assert torch.signbit(driver.head.adapted_position_embedding.weight).all()
    llm = llm_for(target)
    drafter = JetSpecPPDrafter(target=target, head=driver.head, adapter=driver.adapter,
                             decoder=driver.decoder, config=driver.config, thresholds=thresholds())
    totals = Counter()
    prompts = [torch.tensor([[1, 4, 7, 2, 8, 3]]), torch.tensor([[8, 3, 2, 9, 1, 7, 6]]),
               torch.tensor([[2, 5, 4, 3, 8, 9, 6, 1]])]
    for index, prompt in enumerate(prompts):
        expected = llm.generate(prompt, SamplingParams(max_new_tokens=64))['token_ids']
        actual = llm.generate_tree(prompt, drafter, block_size=16, tree_width=1, budget=16,
                                   target_layer_ids=taps, sampling_params=SamplingParams(max_new_tokens=64),
                                   return_stats=True, prompt_index=index, seed=9504)
        assert actual['token_ids'] == expected
        assert sum(actual['accept_lengths']) == 63
        totals.update(drafter.counts)
        totals.update(drafter.driver.counts)
    assert all(totals[key] > 0 for key in ('J', 'R', 'swaps', 'hits', 'misses', 'redrafts'))
