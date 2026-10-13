"""CPU golden equality and local checkpoint contracts for the C1 adapter.

Public test helpers: adapter_config(**options) -> Qwen3Config.
"""
from golden_values import golden
import copy
import hashlib
import json
import os
import shutil
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file, save_file

from conftest import target_config
from jetspec.methods.jetspec_plus_plus.models.adapter import (
    CtxKVC1Head, CtxKVC1Stream, HybridC1RMSNorm, build_head_attention_mask,
    load_adapter, resolve_ctxkv_head_config,
)

__all__ = ['adapter_config']


def adapter_config(**options):
    config = target_config()
    config.dflash_config = dict(target_layer_ids=[0, 2], ctxkv_head=dict(
        mode='c1_tap_aligned_h2h', smooth_l1_weight=1.0, cosine_lambda=0.5,
        projection_weight=0.0, g_distribution={'1': 0.5, '4': 0.5},
        window_causal_over_tokens=True, **options))
    return config


@pytest.mark.parametrize('seed', [0, 1, 2])
def test_resolve_config_matches_golden(seed):
    # Ported from the JetSpec++ reference implementation.
    torch.manual_seed(seed)
    config = adapter_config(per_tap_width_multiplier=[1, 2], stream_depths=[1, 2])
    config.dflash_config['ctxkv_head']['cosine_lambda'] = torch.rand(()).item()
    before = copy.deepcopy(config.to_dict())
    actual = resolve_ctxkv_head_config(config)
    expected = golden('value_1')
    assert actual == expected
    numeric = ['smooth_l1_weight', 'cosine_lambda', 'projection_weight']
    assert torch.equal(torch.tensor([actual[k] for k in numeric]),
                       torch.tensor([expected[k] for k in numeric]))
    assert config.to_dict() == before


@pytest.mark.parametrize('seed', [0, 1, 2])
def test_hybrid_norm_matches_golden(seed):
    torch.manual_seed(seed)
    actual = HybridC1RMSNorm(8, eps=1e-6)
    with torch.no_grad():
        actual.weight.copy_(torch.randn(8))
    x = torch.randn(2, 3, 8)
    assert torch.equal(actual(x), golden('value_2'))


@pytest.mark.parametrize('seed', [0, 1, 2])
@pytest.mark.parametrize('window_causal', [False, True])
def test_attention_mask_matches_golden(seed, window_causal):
    torch.manual_seed(seed)
    kwargs = dict(head=SimpleNamespace(resolved_config={
        'window_causal_over_tokens': window_causal}),
        prefix_position_ids=torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]]),
        query_position_ids=torch.tensor([[4, 5, 6, 7], [5, 4, 7, 6]]),
        block_anchors=torch.tensor([[4, 4, 6, 6], [4, 4, 6, 6]]),
        dtype=torch.float32, device=torch.device('cpu'),
        prefix_valid=torch.rand(2, 4) > 0.2,
        query_key_valid=torch.rand(2, 4) > 0.2,
        query_row_valid=torch.rand(2, 4) > 0.2)
    actual = build_head_attention_mask(**kwargs)
    assert torch.equal(actual, golden('value_3'))
    window = actual[..., 4:]
    if window_causal:
        assert torch.all(window.diagonal(dim1=-2, dim2=-1) == torch.finfo(torch.float32).min)
        assert window[0, 0, 2, 1] == torch.finfo(torch.float32).min
    else:
        assert torch.all(window.diagonal(dim1=-2, dim2=-1) == 0)


def test_d1_mask_reads_only_same_block_earlier_slots():
    mask = build_head_attention_mask(
        head=SimpleNamespace(resolved_config={'window_causal_over_tokens': True}),
        prefix_position_ids=torch.tensor([[0, 1, 2, 3]]),
        query_position_ids=torch.tensor([[2, 3, 4, 5]]),
        block_anchors=torch.tensor([[2, 2, 4, 4]]),
        dtype=torch.float32, device=torch.device('cpu'))
    expected = torch.tensor([
        [1, 1, 0, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 1, 0, 0, 0],
        [1, 1, 1, 1, 0, 0, 0, 0],
        [1, 1, 1, 1, 0, 0, 1, 0]], dtype=torch.bool)
    assert torch.equal(mask[0, 0] == 0, expected)


@pytest.mark.parametrize('seed', [0, 1, 2])
@pytest.mark.parametrize('window_causal', [False, True])
@pytest.mark.parametrize('hybrid', [False, True])
def test_stream_matches_golden(seed, window_causal, hybrid):
    torch.manual_seed(seed)
    config = adapter_config()
    target_text_config = None
    if hybrid:
        target_text_config = dict(
            hidden_size=16, intermediate_size=32, num_attention_heads=2,
            num_key_value_heads=1, head_dim=8, rms_norm_eps=1e-6,
            hidden_act='silu', attention_bias=False, attn_output_gate=True,
            rope_parameters={'rope_type': 'default', 'rope_theta': 10000.0,
                             'partial_rotary_factor': 0.5})
    actual = CtxKVC1Stream(config, mlp_intermediate=48,
                          target_text_config=target_text_config).eval()
    torch.set_rng_state(golden('input_rng'))
    prefix_positions = torch.arange(4).expand(2, -1)
    query_positions = torch.arange(4, 7).expand(2, -1)
    mask = build_head_attention_mask(
        head=SimpleNamespace(resolved_config={'window_causal_over_tokens': window_causal}),
        prefix_position_ids=prefix_positions, query_position_ids=query_positions,
        block_anchors=torch.full_like(query_positions, 4),
        dtype=torch.float32, device=torch.device('cpu'))
    kwargs = dict(u=torch.randn(2, 3, 16), prefix_tap=torch.randn(2, 4, 16),
                  prefix_position_ids=prefix_positions, query_position_ids=query_positions,
                  window_keys=torch.randn(2, 3, 16) if window_causal else None,
                  window_position_ids=query_positions if window_causal else None,
                  attention_mask=mask if window_causal else mask[..., :4])
    assert torch.equal(actual(**kwargs), golden('value_4'))
    source = torch.randn(2, 4, 16)
    for observed, wanted in zip(actual.project_kv(source), golden('value_5')):
        assert torch.equal(observed, wanted)
    # The golden can pre-pack K/V projections; preserve that pure-torch fast path.
    for stream in (actual,):
        stream._fused_kv = (torch.cat((stream.k_proj.weight, stream.v_proj.weight)),
                            None, [8, 8])
    for observed, wanted in zip(actual.project_kv(source), golden('value_5')):
        assert torch.equal(observed, wanted)
    assert torch.equal(actual(**kwargs), golden('value_4'))


@pytest.mark.parametrize('filename', ['config.json', 'head.safetensors', 'converter.safetensors'])
def test_loader_requires_all_local_files(checkpoint, filename):
    incomplete = checkpoint / 'incomplete'
    incomplete.mkdir()
    for name in ('config.json', 'head.safetensors', 'converter.safetensors'):
        if name != filename:
            shutil.copyfile(checkpoint / name, incomplete / name)
    with pytest.raises(FileNotFoundError, match=filename):
        load_adapter(incomplete, 'cpu', torch.float32)


@pytest.mark.parametrize('bank', ['head', 'converter'])
@pytest.mark.parametrize('corruption', ['missing', 'unexpected', 'shape', 'integer'])
def test_loader_rejects_incompatible_tensors(checkpoint, bank, corruption):
    path = checkpoint / f'{bank}.safetensors'
    state = load_file(path)
    name = next(iter(state))
    if corruption == 'missing':
        state.pop(name)
    elif corruption == 'unexpected':
        state['unexpected.weight'] = torch.ones(1)
    elif corruption == 'shape':
        state[name] = torch.zeros(1)
    else:
        state[name] = state[name].long()
    save_file(state, path)
    with pytest.raises((ValueError, RuntimeError)):
        load_adapter(checkpoint, 'cpu', torch.float32)


def test_loader_rejects_target_cache_metadata(checkpoint):
    path = checkpoint / 'head.safetensors'
    save_file(load_file(path), path, metadata={'PREFIX_KV_SOURCE': 'target'})
    with pytest.raises(ValueError, match='PREFIX_KV_SOURCE'):
        load_adapter(checkpoint, 'cpu', torch.float32)


@pytest.mark.parametrize('options,match', [
    ({'mode': 'unknown'}, 'mode'),
    ({'prefix_kv_source': 'target'}, 'target cache'),
    ({'unread': True}, 'keys'),
    ({'projection_weight': 0.1}, 'projection_weight'),
    ({'attention_init': 'unknown'}, 'attention_init'),
    ({'g_distribution': {'1': 0.1}}, 'sum to one'),
])
def test_loader_rejects_unsupported_config(checkpoint, options, match):
    path = checkpoint / 'config.json'
    config = json.loads(path.read_text())
    config['dflash_config']['ctxkv_head'].update(options)
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match=match):
        load_adapter(checkpoint, 'cpu', torch.float32)


@pytest.mark.parametrize('corruption', ['type', 'rope', 'partial', 'theta'])
def test_loader_rejects_unsupported_architecture(checkpoint, corruption):
    path = checkpoint / 'config.json'
    config = json.loads(path.read_text())
    if corruption == 'type':
        config['model_type'] = 'other'
    elif corruption == 'rope':
        config['rope_parameters'] = {'rope_type': 'linear'}
    elif corruption == 'partial':
        config['partial_rotary_factor'] = 0.5
    else:
        config['rope_theta'] = -1
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError):
        load_adapter(checkpoint, 'cpu', torch.float32)


@pytest.mark.parametrize('seed', [0, 1, 2])
def test_loader_preserves_nested_rope_theta(seed, checkpoint):
    torch.manual_seed(seed)
    path = checkpoint / 'config.json'
    config = json.loads(path.read_text())
    theta = float(torch.randint(1000, 100000, ()).item())
    config.pop('rope_theta', None)
    config['rope_parameters'] = {'rope_type': 'default', 'rope_theta': theta}
    path.write_text(json.dumps(config))
    actual = load_adapter(checkpoint, 'cpu', torch.float32)
    for stream in actual.streams:
        assert torch.equal(stream.rotary_emb.inv_freq, golden('value_6'))


def test_loader_casts_all_parameters(checkpoint):
    head = load_adapter(checkpoint, 'cpu', torch.bfloat16)
    assert all(parameter.dtype == torch.bfloat16 and parameter.device.type == 'cpu'
               for parameter in head.parameters())


@pytest.mark.parametrize('seed', [0, 1, 2])
@pytest.mark.parametrize('depths', [None, [1, 2], [2, 3]])
@pytest.mark.parametrize('window_causal', [False, True])
def test_head_matches_golden(seed, depths, window_causal):
    torch.manual_seed(seed)
    config = adapter_config(per_tap_width_multiplier=[1, 2], layers_per_stream=2,
                            ffn_source='target_init')
    config.dflash_config['ctxkv_head']['window_causal_over_tokens'] = window_causal
    if depths is not None:
        config.dflash_config['ctxkv_head']['stream_depths'] = depths
    actual = CtxKVC1Head(config).eval()
    torch.set_rng_state(golden('input_rng'))
    prefix_positions = torch.arange(4).expand(2, -1)
    query_positions = torch.arange(4, 7).expand(2, -1)
    mask = build_head_attention_mask(
        head=actual, prefix_position_ids=prefix_positions,
        query_position_ids=query_positions, block_anchors=torch.full_like(query_positions, 4),
        dtype=torch.float32, device=torch.device('cpu'))
    kwargs = dict(token_embeddings=torch.randn(2, 3, 16),
                  prefix_h=torch.randn(2, 4, 32),
                  h_prev=torch.randn(2, 3, 32),
                  prefix_position_ids=prefix_positions, query_position_ids=query_positions,
                  attention_mask=mask)
    assert torch.equal(actual(**kwargs), golden('value_4'))
    assert torch.equal(actual(**{**kwargs, 'attention_mask': None}),
                       golden('value_7'))
    assert actual.parameter_count_split() == golden('value_8')
    actual_final = actual.final_output_parameters()
    expected_final = golden('value_9')
    assert actual_final.keys() == expected_final.keys()
    for name in actual_final:
        assert torch.equal(actual_final[name], expected_final[name])


@pytest.mark.parametrize('seed', [0, 1, 2])
def test_rotary_helpers_match_golden(seed):
    from jetspec.methods.jetspec_plus_plus.models import adapter
    torch.manual_seed(seed)
    q = torch.randn(2, 2, 3, 8)
    k = torch.randn(2, 2, 5, 8)
    cos, sin = torch.randn(2, 5, 8), torch.randn(2, 5, 8)
    for observed, wanted in zip(adapter._apply_rotary_pos_emb(q, k, cos, sin),
                                golden('value_10')):
        assert torch.equal(observed, wanted)
    for width in (4, 8):
        assert torch.equal(adapter._c1_rotary(k, cos[..., :width], sin[..., :width]),
                           golden('value_11'))


@pytest.fixture
def checkpoint(tmp_path):
    config = adapter_config(stream_depths=[1, 2], ffn_source='target_init')
    (tmp_path / 'config.json').write_text(json.dumps(config.to_dict()))
    save_file(CtxKVC1Head(config).state_dict(), tmp_path / 'head.safetensors')
    save_file({'fc.weight': torch.randn(16, 32), 'hidden_norm.weight': torch.randn(16)},
              tmp_path / 'converter.safetensors')
    return tmp_path


@pytest.mark.parametrize('seed', [0, 1, 2])
def test_local_loader_matches_golden(seed, checkpoint, monkeypatch):
    """Compare the frozen reference's strict eager loader with all optional engine paths disabled."""
    for name in list(os.environ):
        if name.startswith('JETSPEC_'):
            monkeypatch.delenv(name)
    pins = dict(checkpoints={'adapter': dict(directory=str(checkpoint), files={
        name: {'sha256': hashlib.sha256((checkpoint / name).read_bytes()).hexdigest()}
        for name in ('config.json', 'head.safetensors')})})
    pins_file = checkpoint / 'pins.json'
    pins_file.write_text(json.dumps(pins))
    torch.manual_seed(seed)
    actual = load_adapter(checkpoint, device='cpu', dtype=torch.float32)
    assert isinstance(actual, CtxKVC1Head)
    assert not actual.training
    assert all(not parameter.requires_grad for parameter in actual.parameters())
    assert actual.state_dict().keys() == golden('loaded_state').keys()
    for name, value in actual.state_dict().items():
        assert torch.equal(value, golden('loaded_state')[name])
    torch.set_rng_state(golden('input_rng'))
    prefix_positions = torch.arange(4)[None]
    query_positions = torch.arange(4, 7)[None]
    mask = build_head_attention_mask(
        head=actual, prefix_position_ids=prefix_positions,
        query_position_ids=query_positions, block_anchors=torch.full_like(query_positions, 4),
        dtype=torch.float32, device=torch.device('cpu'))
    kwargs = dict(token_embeddings=torch.randn(1, 3, 16), prefix_h=torch.randn(1, 4, 32),
                  prefix_position_ids=prefix_positions, query_position_ids=query_positions,
                  attention_mask=mask)
    assert torch.equal(actual(**kwargs), golden('value_4'))
