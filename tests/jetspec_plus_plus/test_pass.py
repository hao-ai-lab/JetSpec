"""Cache-free public DFlash forward and constrained Jacobi outputs match the golden."""
from golden_values import golden
import pytest
import torch
from conftest import target_config
from jetspec.methods.jetspec.draft_head import DFlashDraftModel
from jetspec.methods.jetspec_plus_plus.method.passes import reference_pass


def draft_config():
    config = target_config()
    config.num_hidden_layers = 2
    config.layer_types = ['full_attention'] * 2
    config.num_target_layers = 4
    config.block_size = 16
    config.dflash_config = dict(target_layer_ids=[0, 1], causal_head=True,
                               jacobi_forcing=True, stock_drafter=True, noise_source='context')
    return config


@pytest.mark.parametrize('seed', [0, 1, 2])
def test_j_pass(seed):
    torch.manual_seed(seed)
    head = DFlashDraftModel(draft_config())
    torch.set_rng_state(golden('input_rng'))
    args = dict(target_hidden=torch.randn(1, 7, 32), noise_embedding=torch.randn(1, 16, 16),
                position_ids=torch.arange(23)[None], past_key_values=None, use_cache=False,
                is_causal=True)
    assert torch.equal(head(**args), golden('value_1'))
    logits, ids = torch.randn(2, 16, 41), torch.randint(0, 41, (2, 16))
    mask, values = torch.rand(2, 16) > 0.7, torch.randint(0, 41, (2, 16))
    actual = reference_pass(logits, ids, 3, pins=(mask, values))
    for field in ('tokens', 'fixed', 'p_top1'):
        assert torch.equal(getattr(actual, field), golden('value_2'))
