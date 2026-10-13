"""Published metadata loads on CPU with only tensor dimensions reduced."""
# Ported from the JetSpec++ reference implementation.
from golden_values import golden
import json
import pytest
import torch
from safetensors.torch import load_file, save_file
from transformers import DynamicCache, Qwen3Config

from real_artifacts import REAL_8B, real_checkpoint, tiny_config
from jetspec.methods.jetspec_plus_plus.artifacts import dataset_entry, load_heads_bundle
from jetspec.methods.jetspec_plus_plus.models.adapter import CtxKVC1Head, load_adapter
from jetspec.methods.jetspec_plus_plus.models.drafter import load_drafter, StockAdaptedMark


@pytest.mark.parametrize('seed', [0, 1, 2])
def test_real_stock_checkpoint_loads(seed, tmp_path):
    expected = real_checkpoint(tmp_path, seed)
    actual = load_drafter(tmp_path, dtype=torch.float32)
    assert len(actual.state_dict()) == 58
    assert 'adapted_position_embedding.weight' not in actual.state_dict()
    inputs = dict(target_hidden=torch.randn(1, 3, 80), noise_embedding=torch.randn(1, 4, 16),
                  position_ids=torch.arange(7)[None], use_cache=True)
    with torch.inference_mode():
        assert torch.equal(actual(**inputs, past_key_values=DynamicCache()),
                           expected(**inputs, past_key_values=DynamicCache()))


@pytest.mark.parametrize('variant', [
    'stock', 'explicit_context', 'not_stock', 'missing_stock', 'invalid_stock',
    'invalid_jacobi', 'not_causal', 'missing_causal', 'invalid_causal',
    'not_fresh', 'mask_noise', 'missing_key', 'unexpected_key', 'adapted_mark',
])
def test_loader_acceptance_matches_golden(variant, tmp_path):
    expected = real_checkpoint(tmp_path, 0)
    raw = json.loads((tmp_path / 'config.json').read_text())
    options = raw['dflash_config']
    changes = dict(explicit_context=('noise_source', 'context'), not_stock=('stock_drafter', False),
                   invalid_stock=('stock_drafter', 1), invalid_jacobi=('jacobi_forcing', 1),
                   not_causal=('causal_head', False), invalid_causal=('causal_head', 1),
                   not_fresh=('fresh_fraction', 0.5), mask_noise=('noise_source', 'mask'))
    if variant in changes:
        key, value = changes[variant]
        options[key] = value
    elif variant == 'missing_stock':
        options.pop('stock_drafter')
    elif variant == 'missing_causal':
        options.pop('causal_head')
    (tmp_path / 'config.json').write_text(json.dumps(raw))
    state = load_file(tmp_path / 'model.safetensors')
    if variant == 'missing_key':
        state.pop('fc.weight')
    elif variant == 'unexpected_key':
        state['unknown.weight'] = torch.zeros(1)
    elif variant == 'adapted_mark':
        state['adapted_position_embedding.weight'] = torch.randn(2, 80)
    save_file(state, tmp_path / 'model.safetensors')

    if variant in ('stock', 'explicit_context'):
        assert golden('accepted')
        assert load_drafter(tmp_path, dtype=torch.float32).stock_drafter
    else:
        assert not golden('accepted')
        match = 'retired' if variant in ('not_stock', 'missing_stock', 'adapted_mark') else None
        with pytest.raises(ValueError, match=match):
            load_drafter(tmp_path, dtype=torch.float32)


def test_loader_refuses_bundled_model_that_builds_a_learned_mark(tmp_path):
    head = real_checkpoint(tmp_path, 0)
    head.adapted_position_embedding = torch.nn.Embedding(2, 80)
    state = load_file(tmp_path / 'model.safetensors')
    assert golden('retired_error') == 'ValueError'
    model_file = tmp_path / 'dflash.py'
    model_file.write_text(model_file.read_text() + '''
_StockModel = DFlashDraftModel
class DFlashDraftModel(_StockModel):
    def __init__(self, config):
        super().__init__(config)
        self.adapted_position_embedding = nn.Embedding(2, self.fc.in_features)
''')
    with pytest.raises(ValueError, match='retired mark'):
        load_drafter(tmp_path, dtype=torch.float32)


@pytest.mark.parametrize('seed', [0, 1, 2])
@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_stock_mark_preserves_context_bits(seed, dtype):
    torch.manual_seed(seed)
    actual = StockAdaptedMark(16, dtype=dtype, device=torch.device('cpu'))
    indicators = torch.randint(0, 2, (2, 4))
    context = torch.randn(2, 4, 16).to(dtype)
    context[..., 0] = -0.0
    context[..., 1] = 0.0
    context[..., 2] = float('inf')
    context[..., 3] = -float('inf')
    assert torch.equal(actual(indicators).view(torch.uint8), golden('value_1'))
    assert torch.equal((context + actual(indicators)).view(torch.uint8), context.view(torch.uint8))
    assert actual.state_dict() == {}
    assert list(actual.parameters()) == []


@pytest.mark.parametrize('seed', [0, 1, 2])
def test_real_adapter_checkpoint_loads(seed, tmp_path):
    raw = tiny_config('adapter')
    config = Qwen3Config.from_dict(raw)
    config._attn_implementation = 'sdpa'
    torch.manual_seed(seed)
    keys = json.loads((REAL_8B / 'adapter/keys.json').read_text())
    assert set(golden('adapter_state')) == set(keys['head.safetensors'])
    assert len(golden('adapter_state')) == 144
    (tmp_path / 'config.json').write_text(json.dumps(raw))
    save_file(golden('adapter_state'), tmp_path / 'head.safetensors')
    torch.set_rng_state(golden('input_rng'))
    converter = {'fc.weight': torch.randn(16, 80), 'hidden_norm.weight': torch.randn(16)}
    assert set(converter) == set(keys['converter.safetensors'])
    save_file(converter, tmp_path / 'converter.safetensors')
    actual = load_adapter(tmp_path, 'cpu', torch.float32)
    assert isinstance(actual, CtxKVC1Head)
    assert actual.stream_depths == [1, 1, 1, 4, 5]
    assert not actual.training
    assert all(not parameter.requires_grad for parameter in actual.parameters())
    args = dict(token_embeddings=torch.randn(1, 4, 16), prefix_h=torch.randn(1, 3, 80),
                prefix_position_ids=torch.arange(3)[None], query_position_ids=torch.arange(3, 7)[None])
    with torch.inference_mode():
        assert torch.equal(actual(**args), golden('adapter_forward'))


def test_real_heads_bundle_resolves_gsm8k():
    bundle = load_heads_bundle(REAL_8B / 'heads_b16')
    assert bundle.thresholds == json.loads((REAL_8B / 'heads_b16/thresholds.json').read_text())
    blend = dataset_entry(bundle.thresholds['e04_kscore_blend'], 'gsm8k')
    rule = dataset_entry(bundle.thresholds['e04_rowrule_joint'], 'gsm8k')
    assert set(blend) == {'a', 'b', 'c', 'argmax', 'top', 'pos'}
    assert set(rule['tau']) == {'J0d0', 'J1', 'J2', 'J3'}
    assert bundle.thresholds['anchor_predictor']['pooled']['gsm8k']['0.99']['value'] == 1.0
    assert len(bundle.decoder_manifest['target_files']) == 3
    pins = json.loads((REAL_8B / 'heads_b16/bundle-pin.json').read_text())
    assert bundle.decoder_manifest['target_files'] == pins['target_files']
