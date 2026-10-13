"""CPU equivalence for the target decoder, refresh window and swap pins."""
from golden_values import golden
from types import SimpleNamespace

import pytest
import torch

from conftest import target_config, tiny_target


@pytest.mark.parametrize('seed', [0, 1, 2])
@pytest.mark.parametrize('tail_tokens', [0, 4])
def test_refresh_window_layout(seed, tail_tokens):
    from jetspec.methods.jetspec_plus_plus.models.decoder import refresh_window_layout
    torch.manual_seed(seed)
    tails = torch.randint(1, tail_tokens + 1, (3,)) if tail_tokens else torch.zeros(3).long()
    counts = torch.randint(1, 5, (3,))
    actual = refresh_window_layout(tail_tokens, tails, counts, length=7)
    expected = golden('value_1')
    for observed, wanted in zip(actual, expected):
        assert torch.equal(observed, wanted)
    allowed, _, rows, _ = actual
    assert allowed.diagonal().all()
    cross_row = (rows[:, None] >= 0) & (rows[None, :] >= 0) & (rows[:, None] != rows[None, :])
    assert not allowed[cross_row].any()


@pytest.mark.parametrize('seed', [0, 1, 2])
def test_anchor_decoder_view_and_forward(seed):
    from jetspec.methods.jetspec_plus_plus.models.decoder import AnchorDecoder, decode_hidden, decode_layer
    target = tiny_target(seed)
    decoder = AnchorDecoder.from_target(target, tap_ids=[0, 1])
    assert decoder.layers[0] is target.model.layers[2]
    assert decoder.layers[1] is target.model.layers[3]
    assert decoder.prefix_layers is decoder.layers
    assert decoder.norm is target.model.norm
    assert decoder.lm_head is target.lm_head
    assert decoder.rotary is target.model.rotary_emb
    assert {p.data_ptr() for p in decoder.parameters()} <= {p.data_ptr() for p in target.parameters()}
    hidden = torch.randn(1, 4, 16)
    positions = torch.arange(4)[None]
    rope = decoder.rotary(hidden, positions)
    with torch.inference_mode():
        for actual, expected in zip(decoder(hidden), golden('value_2')):
            assert torch.equal(actual, expected)
        assert torch.equal(decode_layer(decoder.layers[0], hidden, None, positions, rope, None),
                           golden('value_3'))
        assert torch.equal(decode_hidden(decoder.layers, hidden, positions=positions, rope=rope),
                           golden('value_4'))
    # Ported from the JetSpec++ reference implementation.
    assert golden('value_5') == (34, 35)
    with pytest.raises(ValueError, match='two layers'):
        AnchorDecoder.from_target(target, tap_ids=[0])
    with pytest.raises(ValueError, match='target_layer_ids'):
        AnchorDecoder.from_target(target, tap_ids=[])


@pytest.mark.parametrize('seed', [0, 1, 2])
def test_committed_prefix(seed, monkeypatch):
    """The frozen reference's original static-prefix CPU branch needs no CUDA kernels."""
    from jetspec.methods.jetspec_plus_plus.models.decoder import AnchorDecoder, decoder_prefix_kv
    monkeypatch.delenv('JETSPEC_DECODER_TABLES', raising=False)
    target = tiny_target(seed)
    decoder = AnchorDecoder.from_target(target, tap_ids=[0, 1])
    sequence = SimpleNamespace(taps=torch.randn(9, 5, 16))
    with torch.inference_mode():
        for length in [3, 5, 5, 2, 9]:
            actual = decoder._prefix(sequence, length)
            assert actual.length == golden('value_6') == length
            for index, pair in actual.kv.items():
                for observed, wanted in zip(pair, golden('value_7')):
                    assert torch.equal(observed[..., :length, :], wanted[..., :length, :])
        hidden = sequence.taps[None, :, 4]
        rope = decoder.rotary(hidden, torch.arange(9)[None])
        for actual, expected in zip(decoder_prefix_kv(decoder.layers[1], hidden, rope),
                                    golden('value_8')):
            assert torch.equal(actual, expected)
    decoder.drop_sequence(sequence)
    assert id(sequence) not in decoder._prefixes


@pytest.mark.parametrize('seed', [0, 1, 2])
@pytest.mark.parametrize('use_tail', [False, True])
@pytest.mark.parametrize('cached', [False, True])
def test_refresh_decoder_logits(seed, use_tail, cached, monkeypatch):
    from jetspec.methods.jetspec_plus_plus.models.decoder import AnchorDecoder, refresh_decoder_logits, dense_rows
    monkeypatch.delenv('JETSPEC_DECODER_TABLES', raising=False)
    target = tiny_target(seed)
    decoder = AnchorDecoder.from_target(target, tap_ids=[0, 1])
    sequence = SimpleNamespace(taps=torch.randn(7, 5, 16))
    if cached:
        decoder._prefix(sequence, 3)
    snapshots = {index: tuple(tensor.clone() for tensor in pair)
                 for index, pair in decoder._prefixes.get(id(sequence), SimpleNamespace(kv={})).kv.items()}
    options = dict(length=7, tail_hidden=torch.randn(4, 80) if use_tail else None,
                   tail_counts=torch.tensor([1, 4, 2]) if use_tail else torch.zeros(3).long(),
                   block_hidden=torch.randn(3, 5, 80), slot_counts=torch.tensor([1, 3, 4]))
    actual = refresh_decoder_logits(decoder, sequence, **options)
    expected = golden('value_9')
    for observed, wanted in zip(actual, expected):
        assert torch.equal(observed, wanted)
    for index, pair in snapshots.items():
        for observed, wanted in zip(decoder._prefixes[id(sequence)].kv[index], pair):
            assert torch.equal(observed, wanted)
    values = actual[0].argmax(-1)
    assert torch.equal(dense_rows(values, actual[1], actual[2], 3, 5, -1),
                       golden('value_10'))


@pytest.mark.parametrize('seed', [0, 1, 2])
def test_prefix_cache_helpers(seed):
    from jetspec.methods.jetspec_plus_plus.models.decoder import IncrementalPrefix, _WindowCache, _additive_mask
    torch.manual_seed(seed)
    actual = IncrementalPrefix()
    for count in [3, 2]:
        pair = (torch.randn(1, 1, count, 8), torch.randn(1, 1, count, 8))
        for observed, wanted in zip(actual.update(*pair, 2), golden('value_11')):
            assert torch.equal(observed, wanted)
    for append in [True, False]:
        frozen = _WindowCache(actual.kv, append)
        for index in [2, 3]:
            pair = (torch.randn(1, 1, 2, 8), torch.randn(1, 1, 2, 8))
            for observed, wanted in zip(frozen.update(*pair, index), golden('value_12')):
                assert torch.equal(observed, wanted)
            for layer, stored in frozen.kv.items():
                for observed, wanted in zip(stored, golden('value_13')):
                    assert torch.equal(observed, wanted)
    allowed = torch.rand(3, 6) > 0.5
    like = torch.randn(1, 3, 16)
    assert torch.equal(_additive_mask(allowed, like), golden('value_14'))


def test_8b_layer_indices_without_weight_copy():
    from jetspec.methods.jetspec_plus_plus.models.decoder import AnchorDecoder
    from transformers import Qwen3ForCausalLM
    config = target_config()
    config.num_hidden_layers = 36
    config.layer_types = ['full_attention'] * 36
    target = Qwen3ForCausalLM(config).eval()
    decoder = AnchorDecoder.from_target(target)
    assert decoder.layers[0] is target.model.layers[34]
    assert decoder.layers[1] is target.model.layers[35]
    assert decoder.norm is target.model.norm
    assert decoder.lm_head is target.lm_head


@pytest.mark.parametrize('seed', [0, 1, 2])
def test_swap_decision(seed):
    from jetspec.methods.jetspec_plus_plus.method.swap import rule_f, margin_swap, swap_decision
    torch.manual_seed(seed)
    logits = torch.randn(5, 4, 11)
    ids = torch.randint(0, 11, (5, 5))
    ids[0, 1:] = logits[0].argmax(-1)
    logits[1].zero_()  # Equal probabilities must satisfy the margin-0 comparison.
    khat = torch.tensor([4, 2, 3, 0, 1])
    counts = torch.tensor([5, 3, 4, 1, 2])
    active = torch.tensor([True, True, False, True, True])
    assert torch.equal(rule_f(logits.argmax(-1), ids, counts),
                       golden('value_15'))
    for exclude in [False, True]:
        for margin in [0.0, 0.1]:
            for actual, expected in zip(margin_swap(logits[:, 0], ids[:, 1], margin, exclude_current=exclude),
                                        golden('value_16')):
                assert torch.equal(actual, expected)
    for pins in [None, torch.tensor([-1, 2, -1, 1, -1])]:
        actual = swap_decision(logits, ids, khat, counts, active, pins)
        expected = golden('value_17')
        for observed, wanted in zip(actual, expected):
            assert torch.equal(observed, wanted)
        assert not actual[-1][0]  # No room beyond the full agreement run.
        assert not actual[-1][2]  # Inactive row.


@pytest.mark.parametrize('seed', [0, 1, 2])
def test_apply_pins_and_agreed_pins(seed):
    from jetspec.methods.jetspec_plus_plus.method.pins import apply_pins, _pin_agreed
    torch.manual_seed(seed)
    tokens = torch.randint(0, 11, (4, 5))
    for pins in [(torch.tensor([-1, 1, 3, 4]), torch.tensor([2, 3, 4, 5])),
                 (torch.rand(4, 5) > 0.5, torch.randint(0, 11, (4, 5)))]:
        assert torch.equal(apply_pins(tokens, pins), golden('value_18'))
    original = torch.full_like(tokens, -1)
    original[1, 1] = 10
    state = SimpleNamespace(pins=original.clone(), agreed=None)
    for used in [torch.tensor([0, 2, 4, 1]), torch.tensor([3, 4, 2, 0])]:
        assert _pin_agreed(state, tokens, used) == golden('value_19')
        assert torch.equal(state.pins, golden('agreed_pins'))
        assert torch.equal(state.agreed, golden('value_20'))
    assert state.pins[1, 1] == 10
    assert (state.pins[:, 0] == -1).all()


@pytest.mark.parametrize('seed', [0, 1, 2])
@pytest.mark.parametrize('pin_agreed,refresh_at', [(True, (1,)), (True, (1, 2, 3)),
                                                 (False, (1,)), (False, (1, 2, 3))])
def test_refresh_swap_state(seed, pin_agreed, refresh_at, monkeypatch):
    """Compare the frozen reference's CPU epilogue with identical decoder logits supplied.

    The decoder read is covered separately above. Its graph-buffer allocation
    is replaced here by empty buffers so this gate isolates swaps and pins.
    """
    from jetspec.methods.jetspec_plus_plus.method.swap import RefreshSwap
    torch.manual_seed(seed)
    monkeypatch.delenv('JETSPEC_REFRESH_SWAP_LEAN', raising=False)
    monkeypatch.delenv('JETSPEC_SERVE_POLICY', raising=False)
    logits = torch.randn(4, 4, 11)
    actual = RefreshSwap(rows=4, lanes=5, pin_agreed=pin_agreed)
    for swap in [actual]:
        swap.configure_pins(refresh_at, rows=4, lanes=5)
        swap.reset()
    ids = torch.randint(0, 11, (4, 5))
    ids[0, 1] = logits[0, 0].argmax()
    ids[0, 2] = (logits[0, 1].argmax() + 1) % 11
    ids[1, 1] = (logits[1, 0].argmax() + 1) % 11
    if actual.dense_pins:
        for swap in [actual]:
            swap.pin_mask[1, 1] = True
            swap.pin_values[1, 1] = 10
    active = torch.tensor([True, True, False, True])
    khat = torch.tensor([3, 3, 2, 1])
    counts = torch.tensor([4, 4, 3, 2])
    finished = torch.ones(4).bool()
    for _ in refresh_at:
        inject = actual.apply(logits, ids=ids, khat=khat, slot_counts=counts,
                              active=active, finished=finished)
        assert torch.equal(inject, golden('swap_inject'))
        assert torch.equal(ids, golden('expected_ids'))
        assert torch.equal(finished, golden('expected_finished'))
        assert torch.equal(actual.pin_slot, golden('value_21'))
        assert torch.equal(actual.pin_token, golden('value_22'))
        for observed, wanted in zip(actual.pins(), golden('value_23')):
            assert torch.equal(observed, wanted)
        if actual.dense_pins:
            assert ids[1, 1] == golden('preserved_pin')
            assert actual.pin_values[1, 1] == 10
            assert finished[1]
    for swap in [actual]:
        swap.reset()
    for observed, wanted in zip(actual.pins(), golden('value_23')):
        assert torch.equal(observed, wanted)


def test_decoder_rejects_unsupported_layer():
    from jetspec.methods.jetspec_plus_plus.models.decoder import AnchorDecoder
    target = tiny_target(0)
    target.config.layer_types[3] = 'sliding_attention'
    with pytest.raises(ValueError, match='unsupported decoder layer 3'):
        AnchorDecoder.from_target(target, tap_ids=[0, 1])
