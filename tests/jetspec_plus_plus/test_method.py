"""Registry dispatch and retained decoder leaves match their original torch modules."""
from golden_values import golden
import pytest
import torch
from conftest import tiny_target
from jetspec.methods.jetspec_plus_plus.method import ops
from jetspec.methods.jetspec_plus_plus.method.decoder import core, window, lac
from jetspec.methods.jetspec_plus_plus.method.calibration import dataset_entry


@pytest.mark.parametrize('size', [15, 16])
def test_window_reads_all_verified_rows_and_isolates_blocks(size):
    anchors = [3, 7]
    mask = window.window_mask(anchors, 7, torch.float32, size=size)[0, 0]
    allowed = mask == 0
    for row, anchor in enumerate(anchors):
        for slot in range(size):
            query = row * size + slot
            expected = list(range(anchor)) + list(range(7 + row * size, 8 + row * size + slot))
            assert allowed[query].nonzero().flatten().tolist() == expected
    assert window.window_mask(anchors, 7, torch.float32).shape == (1, 1, 30, 37)


@pytest.mark.parametrize('seed', [0, 1, 2])
def test_registry_equal_golden(seed):
    torch.manual_seed(seed)
    x = torch.randn(3, 4)
    name = f'port_equivalence_{seed}'
    actual = ops.reference_op(name)(torch.relu)
    for module in (ops,):
        module.register_impl(name, torch.neg, engine='port_test')
    assert torch.equal(actual(x), golden('value_1'))
    with ops.use_engine('port_test'):
        assert torch.equal(actual(x), golden('value_1'))
        assert torch.equal(ops.reference(name)(x), golden('value_2'))
        with ops.use_engine('missing'):
            assert torch.equal(actual(x), golden('value_1'))
    for module in (ops,):
        module.unregister_impl(name, engine='port_test')
    assert torch.equal(actual(x), golden('value_1'))


@pytest.mark.parametrize('seed', [0, 1, 2])
def test_retained_decoder_leaves_equal_golden(seed):
    model = tiny_target(seed)
    kwargs = dict(layers=model.model.layers[-2:], norm=model.model.norm,
                  lm_head=model.lm_head, hidden_size=16, rotary=model.model.rotary_emb)
    actual = core.AnchorDecoder(**kwargs)
    hidden = torch.randn(1, 5, 16)
    for left, right in zip(actual(hidden), golden('value_3')):
        assert torch.equal(left, right)
    assert torch.equal(window.window_mask([3, 5], 5, torch.float32, size=16),
                       golden('value_4'))
    left = window.IncrementalPrefix()
    for _ in range(2):
        key, value = torch.randn(1, 1, 3, 8), torch.randn(1, 1, 3, 8)
        assert all(torch.equal(a, b) for a, b in zip(left.update(key, value, 0), golden('value_5')))
    probabilities = torch.rand(41)
    assert torch.equal(lac.keep_device(probabilities), golden('value_6'))
    assert all(torch.equal(a, b) for a, b in zip(
        lac.lac_candidates(probabilities, q_hat=0.5, cap=8, excluded=[3]),
        golden('value_7')))
    entries = {'tiny': {'value': torch.rand(()).item()}}
    assert dataset_entry(entries, 'tiny') == golden('value_8')
