"""J0 uses CPU seed-XOR draws and the token before the correction."""
from golden_values import golden
import pytest
import torch
from jetspec.methods.jetspec_plus_plus.context import build_random_context_block_token_ids


@pytest.mark.parametrize('seed', [0, 1, 2])
def test_kth_init(seed):
    ids = torch.arange(7)[None]
    for prompt_index in [0, 1, 9]:
        options = dict(block_size=16, seed=seed, prompt_index=prompt_index, mode='kth')
        actual = build_random_context_block_token_ids(ids, **options)
        assert torch.equal(actual, golden('value_1'))
        assert actual[0, 0] == ids[0, -2]
