"""Every published key is drafted before verification in its private stale context."""
from golden_values import golden
import pytest
import torch
from types import SimpleNamespace
from test_round import components
from jetspec.methods.jetspec_plus_plus.drafter import JetSpecPPDrafter, anchor_offers
from jetspec.methods.jetspec_plus_plus.models.decoder import refresh_decoder_logits


def thresholds():
    return dict(blend=dict(a=0.0, b=0.0, c=0.0, argmax=0.0, top=0.0, pos=0.0),
                anchor_q_hat=1.0, row_tau={'pooled': {'value': 0.0}}, row_temperature=1.0)


@pytest.mark.parametrize('seed', [0, 1, 2])
def test_lac_offers_equal_golden(seed):
    torch.manual_seed(seed)
    logits, drafts = torch.randn(16, 41), torch.randint(0, 41, (15,))
    actual = anchor_offers(logits, drafts, q_hat=0.99, cap=8)
    expected = golden('value_1')
    ids, probabilities, sizes, _, _ = expected
    for row, count in enumerate(sizes):
        assert torch.equal(actual.ids[row, :count], ids[row])
        assert torch.equal(actual.probabilities[row, :count], probabilities[row])
        assert int(actual.valid[row].sum()) == count


@pytest.mark.parametrize('seed', [0, 1, 2])
def test_ordinary_window_equal_golden(seed):
    _, driver, _ = components(seed)
    decoder = driver.decoder
    sequence = SimpleNamespace(taps=torch.randn(9, 5, 16))
    tail, block = torch.randn(4, 16), torch.randn(1, 16, 80)
    prefix = decoder._prefix(sequence, 5)
    actual, _, _ = refresh_decoder_logits(decoder, sequence, length=5, tail_hidden=tail,
                                          tail_counts=[4], block_hidden=block, slot_counts=[16])
    assert torch.equal(actual, golden('value_2'))


@pytest.mark.parametrize('seed', [0, 1, 2])
def test_keyed_rows_drafted(seed):
    target, driver, taps = components(seed)
    drafter = JetSpecPPDrafter(target=target, head=driver.head, adapter=driver.adapter,
                             decoder=driver.decoder, config=driver.config, thresholds=thresholds())
    context = torch.randint(0, 41, (1, 8))
    exact = torch.randn(1, 7, 80)
    result = drafter.propose_logits(context, 15, target_hidden=exact)
    assert result.shape == (1, 15, 41)
    assert len(drafter.row_cache) == 16
    for key, row in drafter.row_cache.items():
        assert int(row.ids[0, 0]) == key[1]
        assert row.operations == ('J0', 'R1', 'J1')
    drafter.reset_cache()
    assert torch.equal(drafter.propose(context, 15, target_hidden=exact), result.argmax(-1)[0])


@pytest.mark.parametrize('seed', [0, 1, 2])
def test_row_hit_and_miss_equal_golden(seed):
    target, driver, _ = components(seed)
    drafter = JetSpecPPDrafter(target=target, head=driver.head, adapter=driver.adapter,
                             decoder=driver.decoder, config=driver.config, thresholds=thresholds())
    drafter.propose_logits(torch.randint(0, 41, (1, 8)), 15, target_hidden=torch.randn(1, 7, 80))
    blocks = [SimpleNamespace(hypothesis=k, correction_guess=a, ids=row.ids)
              for (k, a), row in drafter.row_cache.items()]
    first = next(iter(drafter.row_cache))
    for k, a in (first, (first[0], 1000), (15, 1000)):
        miss = golden('miss')
        drafter.observe(k, a)
        assert drafter.records[-1]['hit'] == (not miss)
        if not miss:
            assert torch.equal(drafter.selected.ids, golden('selected_ids'))
        else:
            assert drafter.selected is None
