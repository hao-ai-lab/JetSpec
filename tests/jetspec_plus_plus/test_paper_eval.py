"""The paper reader uses the uncut golden eval profile, not the serving rows8 policy."""
from golden_values import golden
import json
from types import SimpleNamespace

import pytest
import torch

from test_round import components
from test_drafter import thresholds
from jetspec.methods.jetspec_plus_plus.round import JetSpecPPConfig
from jetspec.methods.jetspec_plus_plus.drafter import JetSpecPPDrafter, anchor_offers



def test_paper_rows_floor_and_cap_match_golden():
    # Ported from the JetSpec++ reference implementation.
    config = JetSpecPPConfig.paper_eval()
    env = golden('recipe_environment')
    assert config.rows_min == golden('value_2') == 16
    assert config.rows_cap == int(env['JETSPEC_ROWS_RULE_CAP']) == int(env['JETSPEC_ROWS_CAP'])
    assert JetSpecPPConfig().rows_min == 8


def test_paper_seed_and_j3_label_are_already_equal():
    # Ported from the JetSpec++ reference implementation.
    config = JetSpecPPConfig.paper_eval(schedule='J3-R3')
    assert config.seed == 9504
    assert golden('completed_label') == 'J3'
    assert config.operations == ('J0', 'R1', 'J1', 'R2', 'J2', 'R3', 'J3')


@pytest.mark.parametrize('schedule', ['J1-R1', 'J3-R3'])
def test_paper_refresh_projects_full_golden_window(schedule):
    _, driver, _ = components(0, schedule)
    driver.config = JetSpecPPConfig.paper_eval(schedule=schedule)
    result = driver.run(torch.randint(0, 41, (1, 8)), torch.randn(1, 7, 80))
    full = golden('value_3')
    assert result.draft_cutoffs == (int(full[0]),) * int(schedule[-1]) == (15,) * int(schedule[-1])


@pytest.mark.parametrize('seed', [0, 1, 2])
def test_paper_joint_pool_keeps_all_32_golden_offers(seed):
    config = JetSpecPPConfig.paper_eval()
    torch.manual_seed(seed)
    logits, drafts = torch.randn(16, 41), torch.randint(0, 41, (15,))
    actual = anchor_offers(logits, drafts, q_hat=1.0 if config.score_all_anchors else 0.01,
                           cap=config.anchor_cap)
    rejected = torch.cat((drafts, drafts.new_tensor([-1])))
    ids, probabilities, sizes, _, _ = golden('value_4')
    assert torch.equal(actual.ids, torch.stack(ids))
    assert torch.equal(actual.probabilities, torch.stack(probabilities))
    assert actual.valid.all() and all(size == 32 for size in sizes)


def test_paper_after_miss_keeps_the_old_draft_at_k_zero():
    config = JetSpecPPConfig.paper_eval()
    logits = torch.arange(41).float().expand(16, -1).clone()
    drafts = torch.full((15,), 40)
    actual = anchor_offers(logits, drafts, q_hat=1.0, cap=32,
                           after_miss=not config.reject_after_miss)
    rejected = torch.cat((drafts, drafts.new_tensor([-1])))
    rejected[0] = -1
    ids, probabilities, _, _, _ = golden('value_5')
    assert torch.equal(actual.ids, torch.stack(ids))
    assert torch.equal(actual.probabilities, torch.stack(probabilities))
    assert actual.ids[0, 0] == 40 and actual.ids[1, 0] == 39


def test_paper_zero_score_floor_matches_uncut_golden(monkeypatch):
    from jetspec.methods.jetspec_plus_plus.method.rows import DeviceOffers, joint_rule, select_rows

    config = JetSpecPPConfig.paper_eval()
    for key, value in golden('recipe_environment').items():
        monkeypatch.setenv(key, value)
    offers = DeviceOffers(torch.arange(32).expand(16, -1), torch.ones(16, 32).double(),
                          torch.ones(16, 32, dtype=torch.bool), torch.arange(16))
    mass = torch.zeros(16, dtype=torch.float64)
    actual = joint_rule(1.0, 1.0, floor=config.rows_min,
                        fill_zero=config.rows_fill_zero)(mass, offers)
    expected = golden('value_6')
    assert all(torch.equal(a, b) for a, b in zip(actual, expected))
    requested, keys = select_rows(*actual, offers, config.rows_cap, config.rows_cap)
    assert int(requested) == 16
    assert keys.tolist() == [[0, i] for i in range(16)]


@pytest.mark.parametrize('seed', [0, 1, 2])
def test_paper_keeps_converged_outputs_and_effective_label(monkeypatch, seed):
    from jetspec.methods.jetspec_plus_plus import round as rounds
    from jetspec.methods.jetspec_plus_plus.context import build_random_context_block_token_ids
    from jetspec.methods.jetspec_plus_plus.method.passes import DraftForwardPass

    _, driver, _ = components(seed, 'J3-R0')
    driver.config = JetSpecPPConfig.paper_eval(schedule='J3-R0')
    context, exact = torch.randint(0, 41, (1, 8)), torch.randn(1, 7, 80)
    initial = build_random_context_block_token_ids(context, block_size=16, seed=9504,
                                                  prompt_index=0, mode='kth')
    initial[:, 0] = context[0, -1]
    logits = [torch.zeros(1, 16, 41).scatter_(2, initial[..., None], 8.0)]
    logits += [torch.zeros(1, 16, 41).scatter_(2, ((initial + i) % 41)[..., None], 2.0)
               for i in range(1, 4)]
    calls = []

    def forward(head, embed, lm_head, ids, hidden, **kwargs):
        value = logits[len(calls)]
        calls.append(value)
        return DraftForwardPass(value, ids)

    monkeypatch.setattr(rounds, 'draft_forward', forward)
    actual = driver.run(context, exact)
    assert len(calls) == 4  # Fixed launches still execute after convergence.
    assert torch.equal(actual.ids, golden('value_7'))
    assert torch.equal(actual.p_top1, golden('value_8'))
    assert actual.passes_run == int(golden('value_9')) == 1
    assert actual.label == 'J0d0' and actual.converged


def test_paper_swap_reopens_a_converged_row(monkeypatch):
    from jetspec.methods.jetspec_plus_plus import round as rounds
    from jetspec.methods.jetspec_plus_plus.method.passes import DraftForwardPass

    _, driver, _ = components(0)
    driver.config = JetSpecPPConfig.paper_eval()
    inputs, scores = [], []

    def forward(head, embed, lm_head, ids, hidden, **kwargs):
        inputs.append(ids.clone())
        logits = torch.zeros(1, 16, 41).scatter_(2, ids[..., None], float(len(inputs) + 3))
        scores.append(logits)
        return DraftForwardPass(logits, ids)

    def decoder(*args, **kwargs):
        tokens = (inputs[0][0, 1:] + 1) % 41
        logits = torch.zeros(15, 41).scatter_(1, tokens[:, None], 10.0)
        return logits, torch.zeros(15, dtype=torch.long), torch.arange(15)

    monkeypatch.setattr(rounds, 'draft_forward', forward)
    monkeypatch.setattr(rounds, 'refresh_decoder_logits', decoder)
    actual = driver.run(torch.randint(0, 41, (1, 8)), torch.randn(1, 7, 80))
    expected = golden('value_10')
    assert bool(expected[3][0])
    assert actual.ids[0, int(expected[1][0])] == int(expected[2][0])
    assert actual.passes_run == 2 and actual.label == 'J1'
    assert torch.equal(actual.p_top1, scores[-1].softmax(-1).amax(-1))


def test_reader_defaults_to_paper_preset():
    from jetspec.methods.jetspec_plus_plus.reader import read_passes
    from jetspec.methods.jetspec.draft import TargetEchoTreeDrafter
    from test_lossless import llm_for

    target, _, _ = components(0)
    seen = []

    def factory(*, llm, config):
        seen.append(config)
        return TargetEchoTreeDrafter(llm.model)

    read_passes(llm_for(target), [{'token_ids': [1, 2, 3, 4, 5]}],
                drafter_factory=factory, tokens=2)
    assert seen == [JetSpecPPConfig.paper_eval()]


@pytest.mark.parametrize('schedule', ['J1-R1', 'J3-R3'])
def test_paper_preset_drives_lossless_rows(schedule):
    from jetspec.core.llm import SamplingParams
    from test_lossless import llm_for

    target, driver, taps = components(0, schedule)
    calibration = thresholds()
    calibration.update(anchor_q_hat=0.001, row_tau={'pooled': {'value': 1.0}})
    drafter = JetSpecPPDrafter(target=target, head=driver.head, adapter=driver.adapter,
                             decoder=driver.decoder, config=JetSpecPPConfig.paper_eval(schedule=schedule),
                             thresholds=calibration)
    llm = llm_for(target)
    prompt = torch.tensor([[1, 2, 3, 4, 5, 6]])
    actual = llm.generate_tree(prompt, drafter, block_size=16, tree_width=1, budget=16,
                               target_layer_ids=taps, return_stats=True, return_text=False,
                               sampling_params=SamplingParams(0.0, 20), prompt_index=0, seed=9504)
    expected = llm.generate(prompt, SamplingParams(0.0, 20))
    assert actual['token_ids'] == expected['token_ids']
    assert all(len(record['keys']) == 16 for record in drafter.records)
    assert all(cutoff == 15 for result in drafter.round_results for cutoff in result.draft_cutoffs)
