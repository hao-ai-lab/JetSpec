"""The hybrid target and pinned tail use standalone, frozen reference outputs."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from jetspec.methods.jetspec_plus_plus.models.hybrid import (
    HybridTarget, HybridAnchorDecoder, text_config,
)


def hybrid_config():
    return SimpleNamespace(model_type='qwen3_5_text', vocab_size=41, hidden_size=16,
        intermediate_size=32, num_hidden_layers=8, num_attention_heads=2,
        num_key_value_heads=1, head_dim=8, rms_norm_eps=1e-6,
        attention_bias=False, hidden_act='silu', linear_num_key_heads=1,
        linear_num_value_heads=2, linear_key_head_dim=4, linear_value_head_dim=4,
        linear_conv_kernel_dim=4, layer_types=['linear_attention', 'full_attention'] * 4,
        rope_parameters={'rope_type': 'default', 'rope_theta': 10000000.,
                         'partial_rotary_factor': 0.5}, eos_token_id=None,
        _name_or_path='', _attn_implementation='sdpa')


@pytest.mark.parametrize('seed', [0, 1, 2])
def test_hybrid_verify_rolls_back_rejected_recurrence(seed):
    from test_lossless import llm_for
    from jetspec.core.llm import SamplingParams
    from jetspec.methods.jetspec.draft import TargetEchoTreeDrafter

    torch.manual_seed(seed)
    target = HybridTarget(hybrid_config()).eval()
    llm = llm_for(target)
    prompt = torch.tensor([[1, 4, 7, 2, 8, 3]])
    expected = llm.generate(prompt, SamplingParams(max_new_tokens=32))['token_ids']
    # A deliberately wrong causal chain exercises rejection and GDN rollback.
    class WrongDraft(TargetEchoTreeDrafter):
        def propose_logits(self, context_ids, depth, **kwargs):
            return torch.zeros(1, depth, 41).scatter_(2,
                torch.zeros(1, depth, 1, dtype=torch.long), 1.)
    actual = llm.generate_tree(prompt, WrongDraft(target), block_size=16, tree_width=1,
        budget=16, return_stats=True, return_text=False,
        sampling_params=SamplingParams(max_new_tokens=32))
    assert actual['token_ids'] == expected
    assert min(actual['accept_lengths']) < 16


@pytest.mark.parametrize('seed', [0, 1, 2])
def test_hybrid_tail_matches_frozen_reference(seed, tmp_path):
    import hashlib
    from safetensors.torch import load_file, save_file
    root = Path(__file__).parent / 'fixtures/golden_27b'
    tensors = load_file(root / f'tail-{seed}.safetensors')
    target = HybridTarget(hybrid_config()).eval()
    decoder = HybridAnchorDecoder(target, tap_ids=[0, 1, 2, 3, 5])
    state = {name.removeprefix('weight.'): value for name, value in tensors.items()
             if name.startswith('weight.')}
    decoder.load_state_dict(state, strict=True)
    # Exercise the actual local checkpoint and pinned decoder loaders against
    # frozen reference outputs, rather than comparing two copies of public code.
    weights = {name.replace('model.', 'model.language_model.', 1): value
               for name, value in target.state_dict().items()}
    save_file(weights, tmp_path / 'model.safetensors')
    (tmp_path / 'config.json').write_text(json.dumps({'text_config': vars(target.config)}))
    (tmp_path / 'model.safetensors.index.json').write_text(json.dumps({'weight_map':
        {name: 'model.safetensors' for name in weights}}))
    pins = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in tmp_path.iterdir()}
    target = HybridTarget.from_checkpoint(tmp_path, device='cpu', dtype=torch.float32)
    decoder = HybridAnchorDecoder.from_checkpoint(target, tap_ids=[0, 1, 2, 3, 5], file_pins=pins)
    target._cache = target.new_cache()
    target._cache.length = 3
    target._cache.states[6] = tensors['recurrent'], tensors['conv']
    target._cache.states[7] = tensors['key'], tensors['value']
    sequence = SimpleNamespace(taps=torch.zeros(3, 5, 16))
    logits, rows, slots = decoder.refresh_logits(sequence, length=3,
        tail_hidden=None, tail_counts=[0], block_hidden=tensors['hidden'][None],
        slot_counts=[4])
    assert torch.equal(logits, tensors['logits'])
    assert rows.tolist() == [0] * 4 and slots.tolist() == list(range(4))
    assert torch.equal(target._cache.states[6][0], tensors['recurrent'])
    # Two branches use distinct tails and never share speculative state.
    branch, rows, slots = decoder.refresh_logits(sequence, length=3,
        tail_hidden=tensors['tail'], tail_counts=[1, 2],
        block_hidden=tensors['blocks'], slot_counts=[3, 2])
    assert torch.equal(branch, tensors['branch_logits'])
    assert rows.tolist() == [0, 0, 0, 1, 1] and slots.tolist() == [0, 1, 2, 0, 1]


def test_hybrid_checkpoint_loads_exact_text_slice(tmp_path):
    import hashlib
    from safetensors.torch import save_file
    target = HybridTarget(hybrid_config()).eval()
    source = {name.replace('model.', 'model.language_model.', 1): value
              for name, value in target.state_dict().items()}
    save_file(source, tmp_path / 'model.safetensors')
    raw = dict(vars(target.config))
    (tmp_path / 'config.json').write_text(json.dumps({'text_config': raw}))
    (tmp_path / 'model.safetensors.index.json').write_text(json.dumps({
        'weight_map': {name: 'model.safetensors' for name in source}}))
    pins = {name: hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()
            for name in ('config.json', 'model.safetensors.index.json', 'model.safetensors')}
    actual = HybridTarget.from_checkpoint(tmp_path, device='cpu', dtype=torch.float32)
    assert all(torch.equal(value, actual.state_dict()[name]) for name, value in target.state_dict().items())
    decoder = HybridAnchorDecoder.from_checkpoint(actual, tap_ids=[0, 1, 2, 3, 5], file_pins=pins)
    assert decoder.layers[0] is actual.model.layers[6]
    with pytest.raises(ValueError, match='config and index'):
        HybridAnchorDecoder.from_checkpoint(actual, tap_ids=[0, 1, 2, 3, 5],
            file_pins={name: value for name, value in pins.items() if name != 'config.json'})
    source['model.language_model.layers.6.unexpected'] = torch.zeros(1)
    save_file(source, tmp_path / 'model.safetensors')
    index = {'weight_map': {name: 'model.safetensors' for name in source}}
    (tmp_path / 'model.safetensors.index.json').write_text(json.dumps(index))
    with pytest.raises(ValueError, match='keys differ'):
        HybridTarget.from_checkpoint(tmp_path, device='cpu', dtype=torch.float32)


@pytest.mark.parametrize('target_layers,taps', [(64, [5, 19, 33, 47, 61]), (8, [0, 1, 2, 3, 5])])
def test_drafter_reads_target_contract_from_checkpoint(tmp_path, target_layers, taps):
    from real_artifacts import real_checkpoint
    from jetspec.methods.jetspec_plus_plus.models.drafter import load_drafter
    real_checkpoint(tmp_path, 0)
    path = tmp_path / 'config.json'
    raw = json.loads(path.read_text())
    raw['num_target_layers'] = target_layers
    raw['dflash_config']['target_layer_ids'] = taps
    path.write_text(json.dumps(raw))
    head = load_drafter(tmp_path, dtype=torch.float32)
    assert head.target_layer_ids == taps
    with pytest.raises(ValueError):
        load_drafter(tmp_path, dtype=torch.float32, model_family='8b')
    raw['dflash_config']['target_layer_ids'] = [taps[0]] * 5
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match='ordered target taps'):
        load_drafter(tmp_path, dtype=torch.float32, model_family='27b')


def hybrid_artifacts(tmp_path, seed=0):
    import hashlib
    from safetensors.torch import save_file
    from transformers import Qwen3Config
    from real_artifacts import real_checkpoint, tiny_config
    from jetspec.methods.jetspec_plus_plus.models.adapter import CtxKVC1Head
    taps = [0, 1, 2, 3, 5]
    torch.manual_seed(seed)
    config = hybrid_config()
    target = HybridTarget(config).eval()
    root = tmp_path / 'target'
    root.mkdir()
    weights = {name.replace('model.', 'model.language_model.', 1): tensor
               for name, tensor in target.state_dict().items()}
    save_file(weights, root / 'model.safetensors')
    (root / 'config.json').write_text(json.dumps({'text_config': vars(config)}))
    (root / 'model.safetensors.index.json').write_text(json.dumps({'weight_map':
        {name: 'model.safetensors' for name in weights}}))
    target.config._name_or_path = str(root)
    drafter = tmp_path / 'drafter'
    drafter.mkdir()
    head = real_checkpoint(drafter, seed)
    raw = json.loads((drafter / 'config.json').read_text())
    raw['num_target_layers'] = 8
    raw['rope_theta'] = 10000000.
    if 'rope_parameters' in raw:
        raw['rope_parameters']['rope_theta'] = 10000000.
    raw['dflash_config']['target_layer_ids'] = taps
    (drafter / 'config.json').write_text(json.dumps(raw))
    adapter = tmp_path / 'adapter'
    adapter.mkdir()
    raw = tiny_config('adapter')
    raw['rope_theta'] = 10000000.
    if 'rope_parameters' in raw:
        raw['rope_parameters']['rope_theta'] = 10000000.
    raw['num_target_layers'] = 8
    raw['dflash_config']['target_layer_ids'] = taps
    target_text = dict(vars(config), full_attention_interval=2, attn_output_gate=True,
                       partial_rotary_factor=0.5)
    raw['dflash_config']['ctxkv_head'].update(target_text_config=target_text,
        hybrid_attention_source='nearest_full_at_or_below', stream_depths=[1] * 5)
    (adapter / 'config.json').write_text(json.dumps(raw))
    ac = Qwen3Config.from_dict(raw)
    ac.rope_theta = 10000000.
    ah = CtxKVC1Head(ac).eval()
    save_file(ah.state_dict(), adapter / 'head.safetensors')
    save_file({key: head.state_dict()[key] for key in ('fc.weight', 'hidden_norm.weight')},
              adapter / 'converter.safetensors')
    bundle = tmp_path / 'heads'
    (bundle / 'decoder').mkdir(parents=True)
    (bundle / 'thresholds.json').write_text(json.dumps(dict(
        e04_kscore_blend={'tiny': {key: {'value': 0.0} for key in ('a', 'b', 'c', 'argmax', 'top', 'pos')}},
        e04_rowrule_joint={'tiny': {'tau': {'pooled': {'value': 1.0}}, 'temperature': {'value': 1.0}}},
        anchor_predictor={'pooled': {'tiny': {'0.99': {'value': 1.0}}}})))
    pins = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.iterdir()}
    (bundle / 'decoder/manifest.json').write_text(json.dumps(dict(source='target_slice', target_files=pins)))
    return target, drafter, adapter, bundle


@pytest.mark.parametrize('schedule', ['J1-R1', 'J3-R3'])
def test_hybrid_loader_drives_lossless_method(tmp_path, schedule):
    from test_lossless import llm_for
    from jetspec.core.llm import SamplingParams
    from jetspec.methods.jetspec_plus_plus.drafter import build_drafter
    from jetspec.methods.jetspec_plus_plus.round import JetSpecPPConfig
    from jetspec.methods.jetspec_plus_plus.reader import read_passes
    target, head, adapter, bundle = hybrid_artifacts(tmp_path)
    llm = llm_for(target)
    config = JetSpecPPConfig.paper_eval(schedule=schedule, model_family='27b')
    def factory(**kwargs):
        return build_drafter(**kwargs, drafter_checkpoint=head, adapter_artifact=adapter,
                             heads_bundle=bundle, dataset='tiny')
    prompt = torch.tensor([[1, 4, 7, 2, 8, 3]])
    expected = llm.generate(prompt, SamplingParams(max_new_tokens=24))['token_ids']
    actual, = read_passes(llm, [{'token_ids': prompt[0].tolist()}], drafter_factory=factory,
                          config=config, tokens=24)
    assert actual['token_ids'] == expected
    assert sum(actual['accept_lengths']) == 23
    assert actual['passes_run']


@pytest.mark.parametrize('family', ['auto', '27b'])
def test_reader_selects_hybrid_family(tmp_path, family):
    from test_lossless import llm_for
    from jetspec.methods.jetspec_plus_plus.reader import main
    from jetspec.methods.jetspec_plus_plus.drafter import build_drafter
    target, head, adapter, bundle = hybrid_artifacts(tmp_path)
    prompts = tmp_path / 'prompts.json'
    prompts.write_text(json.dumps([{'token_ids': [1, 2, 3, 4, 5, 6]}]))
    output = tmp_path / 'read.json'
    def factory(**kwargs):
        return build_drafter(**kwargs, dataset='tiny')
    result = main(['--model-family', family, '--target', target.config._name_or_path,
        '--drafter', str(head), '--adapter', str(adapter), '--heads-bundle', str(bundle),
        '--prompts-file', str(prompts), '--max-new-tokens', '2', '--output', str(output)],
        llm_factory=lambda **kwargs: llm_for(target), drafter_factory=factory)
    assert result['manifest']['model_family'] == '27b'
    assert result['manifest']['drafter_revision'] is None
    assert result['outputs'][0]['n_tokens'] == 2
