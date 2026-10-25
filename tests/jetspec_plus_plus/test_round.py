"""The evaluated fixed schedule runs J0, every refresh and every later J pass."""
from golden_values import golden
import pytest
import torch
from conftest import tiny_target
from real_artifacts import bundled_model
from test_adapter import adapter_config
from test_pass import draft_config
from jetspec.methods.jetspec_plus_plus.models.adapter import CtxKVC1Head
from jetspec.methods.jetspec_plus_plus.models.decoder import AnchorDecoder
from jetspec.methods.jetspec_plus_plus.models.drafter import StockAdaptedMark
from jetspec.methods.jetspec_plus_plus.round import JetSpecPPConfig, RoundDriver, completed_label, resolve_tau


def components(seed, schedule='J1-R1'):
    target = tiny_target(seed)
    taps = [0, 0, 0, 0, 1]
    dc, ac = draft_config(), adapter_config()
    dc.dflash_config['target_layer_ids'] = taps
    dc.dflash_config['fresh_fraction'] = 1.0
    ac.dflash_config['target_layer_ids'] = taps
    head, adapter = bundled_model(dc), CtxKVC1Head(ac).eval()
    head.adapted_position_embedding = StockAdaptedMark(
        head.fc.in_features, dtype=torch.float32, device=torch.device('cpu'))
    decoder = AnchorDecoder.from_target(target, tap_ids=taps)
    driver = RoundDriver(target, head, adapter, decoder, JetSpecPPConfig(schedule=schedule))
    return target, driver, taps


@pytest.mark.parametrize('seed', [0, 1, 2])
def test_all_schedule_operations(seed):
    _, driver, _ = components(seed, 'J3-R3')
    ids, exact = torch.randint(0, 41, (1, 8)), torch.randn(1, 7, 80)
    result = driver.run(ids, exact, admission=True)
    assert result.operations == ('J0', 'R1', 'J1', 'R2', 'J2', 'R3', 'J3')
    assert result.ids.shape == (1, 16)
    assert result.ids[0, 0] == ids[0, -1]
    assert driver.counts['J'] == 4
    assert driver.counts['R'] == 3
    assert result.label == 'J3'
    assert all(value == 15 for value in result.draft_cutoffs)


@pytest.mark.parametrize('retired', ['config', 'learned_mark'])
def test_round_refuses_retired_drafter(retired):
    target, driver, _ = components(0)
    if retired == 'config':
        driver.head.config.dflash_config['stock_drafter'] = False
    else:
        driver.head.adapted_position_embedding = torch.nn.Embedding(2, 80)
    with pytest.raises(ValueError, match='retired'):
        RoundDriver(target, driver.head, driver.adapter, driver.decoder, driver.config)


@pytest.mark.parametrize('seed', [0, 1, 2])
def test_pass_labels_and_tau_equal_golden(seed):
    # Ported from the JetSpec++ reference implementation.
    torch.manual_seed(seed)
    table = {'pooled': {'value': torch.rand(()).item()}, 'J3': {'value': torch.rand(()).item()}}
    for passes in (1, 2, 4):
        label = completed_label(passes)
        assert label == golden('value_1')
        assert torch.equal(torch.tensor(resolve_tau(table, label)),
                           torch.tensor(golden('value_2')))
