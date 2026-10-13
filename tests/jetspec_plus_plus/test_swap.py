"""Swap decisions, dense pins and agreed pins preserve the served ordering."""
from golden_values import golden
from types import SimpleNamespace
import pytest
import torch
from jetspec.methods.jetspec_plus_plus.method.swap import swap_decision
from jetspec.methods.jetspec_plus_plus.method.pins import _pin_agreed


@pytest.mark.parametrize('seed', [0, 1, 2])
def test_swap_pins(seed):
    torch.manual_seed(seed)
    logits, ids = torch.randn(3, 15, 41), torch.randint(0, 41, (3, 16))
    args = (logits, ids, torch.tensor([3, 6, 15]), torch.tensor([4, 7, 15]),
            torch.tensor([True, True, False]))
    actual = swap_decision(*args)
    expected = golden('value_1')
    for x, y in zip(actual, expected):
        assert torch.equal(x, y)
    mask = torch.rand(3, 16) > 0.8
    values = torch.randint(0, 41, (3, 16))
    left = SimpleNamespace(pins=torch.where(mask, values, -1), agreed=None)
    _pin_agreed(left, ids, actual[0])
    assert torch.equal(left.pins, golden('pins'))
    assert torch.equal(left.agreed, golden('value_3'))
