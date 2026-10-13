"""The public target seam captures precisely the frozen reference's committed taps."""
from golden_values import golden
import pytest
import torch
from transformers import DynamicCache

from jetspec.core.model_runner import ModelRunner
from jetspec.methods.jetspec.draft_head import build_target_layer_ids
from conftest import tiny_target


@pytest.mark.parametrize('seed', [0, 1, 2])
def test_prefill_taps(seed):
    target = tiny_target(seed)
    ids = torch.randint(0, 41, (1, 7))
    positions = torch.arange(7)[None]
    options = dict(output_hidden_states=True, target_layer_ids=[0, 1],
                   last_position_logits_only=True)
    actual = ModelRunner(target).forward(ids, DynamicCache(), positions, **options)
    assert torch.equal(actual[0], golden('value_1'))
    assert torch.equal(actual[2], golden('value_2'))
    assert build_target_layer_ids(36, 5) == [1, 9, 17, 25, 33]
