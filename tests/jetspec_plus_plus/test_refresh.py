"""Refresh cutoff and strictly-earlier injected visibility reproduce the CPU reader."""
from golden_values import golden
import pytest
import torch
from jetspec.methods.jetspec_plus_plus.method.refresh import refresh_cutoff, inject


@pytest.mark.parametrize('seed', [0, 1, 2])
def test_refresh(seed):
    torch.manual_seed(seed)
    logits = torch.randn(2, 16, 41)
    p = logits.float().softmax(-1).amax(-1)[:, 1:]
    assert torch.equal(refresh_cutoff(p, 0.1, final=15),
                       golden('value_1'))
    prefix, hidden = torch.randn(1, 7, 32), torch.randn(1, 5, 32)
    full = torch.cat((hidden, torch.randn(1, 11, 32)), 1)
    context, positions, mask = inject(prefix, full, torch.tensor([4]), block_size=16)
    context = context[:, :12]
    positions = torch.cat((positions[:, :12], positions[:, 23:]), 1)
    mask = torch.cat((mask[..., :12], mask[..., 23:]), -1)
    for key, actual in [('target_hidden', context), ('position_ids', positions), ('attention_mask', mask)]:
        assert torch.equal(actual, golden('value_2'))
    assert not (mask[0, 0, 4, 7:12] == 0)[4]
