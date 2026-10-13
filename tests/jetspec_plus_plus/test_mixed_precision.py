"""The reader reproduces the served fp32-target/bf16-draft precision layout."""
from golden_values import golden
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from test_artifacts import local_artifacts
from test_lossless import llm_for
from jetspec.core.llm import SamplingParams
from jetspec.methods.jetspec_plus_plus import reader
from jetspec.methods.jetspec_plus_plus.drafter import build_drafter
from jetspec.methods.jetspec_plus_plus.models.decoder import AnchorDecoder, refresh_decoder_logits
from jetspec.methods.jetspec_plus_plus.round import JetSpecPPConfig


def artifacts(tmp_path, seed):
    target, head, adapter, bundle, _ = local_artifacts(tmp_path, seed)
    root = Path(target.config._name_or_path)
    # The real target checkpoint is bf16, including the separately loaded slice.
    for path in root.glob('*.safetensors'):
        save_file({k: v.bfloat16() for k, v in load_file(path).items()}, path,
                  metadata={'format': 'pt'})
    manifest_path = bundle / 'decoder/manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['target_files'] = {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                                for name in manifest['target_files']}
    manifest_path.write_text(json.dumps(manifest))
    target.load_state_dict({key: value for path in root.glob('*.safetensors')
                            for key, value in load_file(path).items()}, strict=True)
    return target, head, adapter, bundle


def arguments(tmp_path, paths):
    target, head, adapter, bundle = paths
    prompts = tmp_path / 'prompts.json'
    prompts.write_text(json.dumps([{'token_ids': ids} for ids in
        ([1, 2, 3, 4, 5, 6], [2, 3, 4, 5, 6, 7], [3, 4, 5, 6, 7, 8])]))
    return ['--target', target.config._name_or_path, '--drafter', str(head),
            '--adapter', str(adapter), '--heads-bundle', str(bundle), '--dataset', 'tiny',
            '--prompts-file', str(prompts), '--max-new-tokens', '16', '--dtype', 'fp32',
            '--capture-rounds']


@pytest.mark.parametrize('seed', [0, 1, 2])
def test_reader_mixed_layout_and_lossless(tmp_path, seed):
    """The reference draft runner and decoder select draft precision."""
    paths = artifacts(tmp_path, seed)
    target = paths[0]
    seen = {}
    def factory(**kwargs):
        drafter = build_drafter(**kwargs, dataset='tiny')
        seen['drafter'] = drafter
        return drafter
    read = reader.main(arguments(tmp_path, paths) + ['--draft-dtype', 'bf16',
                       '--output', str(tmp_path / 'mixed.json')],
                       llm_factory=lambda **kwargs: llm_for(target), drafter_factory=factory)
    driver = seen['drafter'].driver
    assert {p.dtype for p in target.parameters()} == {torch.float32}
    for module in (driver.head, driver.adapter, driver.decoder, driver.embed_tokens, driver.lm_head):
        assert {p.dtype for p in module.parameters()} == {torch.bfloat16}
    assert driver.decoder.norm is not target.model.norm
    # The reference keeps RoPE frequencies fp32 in a bf16 decoder.
    assert driver.decoder.rotary.inv_freq.dtype == torch.float32
    assert driver.embed_tokens is not target.model.embed_tokens
    assert read['manifest']['dtype'] == 'fp32'
    assert read['manifest']['draft_dtype'] == 'bf16'
    for row, prompt in zip(read['outputs'], json.loads((tmp_path / 'prompts.json').read_text())):
        expected = llm_for(target).generate(torch.tensor([prompt['token_ids']]),
                                            SamplingParams(0, 16))['token_ids']
        assert row['token_ids'] == expected
    root = Path(target.config._name_or_path)
    weight_map = json.loads((root / 'model.safetensors.index.json').read_text())['weight_map']
    modules = [(f'model.layers.{i}.', driver.decoder.layers[j]) for j, i in enumerate((34, 35))]
    modules += [('model.norm.', driver.decoder.norm), ('lm_head.', driver.decoder.lm_head),
                ('model.embed_tokens.', driver.embed_tokens), ('lm_head.', driver.lm_head)]
    for prefix, module in modules:
        for key, value in module.state_dict().items():
            name = prefix + key
            with safe_open(root / weight_map[name], framework='pt') as checkpoint:
                assert torch.equal(value.view(torch.uint8), checkpoint.get_tensor(name).view(torch.uint8))
    # Compare the frozen reference's strict checkpoint loader as well as the file bytes.
    for key, value in driver.decoder.layers.state_dict().items():
        assert torch.equal(value.view(torch.uint8), golden('value_1'))
    assert torch.equal(driver.decoder.norm.weight, golden('value_2'))
    assert torch.equal(driver.decoder.lm_head.weight, golden('value_3'))


@pytest.mark.parametrize('selection', [[], ['--draft-dtype', 'same'], ['--draft-dtype', 'fp32']])
def test_same_matches_old_reader_tokens_and_captures(tmp_path, selection):
    paths = artifacts(tmp_path, 4)
    old_path = tmp_path / 'old_reader.py'
    old_path.write_bytes((Path(__file__).parent / 'fixtures/reference/reader_fp32.py').read_bytes())
    spec = importlib.util.spec_from_file_location('jetspec.methods.jetspec_plus_plus._old_reader', old_path)
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    args = arguments(tmp_path, paths)
    kwargs = dict(llm_factory=lambda **kwargs: llm_for(paths[0]))
    baseline = old.main(args + ['--output', str(tmp_path / 'old.json')], **kwargs)
    current = reader.main(args + selection + ['--output', str(tmp_path / 'new.json')], **kwargs)
    assert current['outputs'] == baseline['outputs']
    assert (tmp_path / 'new.json.candidates.jsonl').read_bytes() == (
        tmp_path / 'old.json.candidates.jsonl').read_bytes()
    assert current['manifest']['draft_dtype'] == 'fp32'


@pytest.fixture
def mixed(tmp_path):
    target, head, adapter, bundle = artifacts(tmp_path, 0)
    return build_drafter(llm=llm_for(target), config=JetSpecPPConfig.paper_eval(),
                        drafter_checkpoint=head, adapter_artifact=adapter, heads_bundle=bundle,
                        dataset='tiny', draft_dtype=torch.bfloat16)


def test_default_factory_loads_real_mixed_models(tmp_path):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast
    paths = artifacts(tmp_path, 4)
    root = paths[0].config._name_or_path
    PreTrainedTokenizerFast(tokenizer_object=Tokenizer(
        WordLevel({str(i): i for i in range(41)}, unk_token='0')), unk_token='0').save_pretrained(root)
    read = reader.main(arguments(tmp_path, paths) + ['--draft-dtype', 'bf16',
                       '--output', str(tmp_path / 'real.json')])
    assert all(len(row['token_ids']) == 16 for row in read['outputs'])
    assert read['manifest']['draft_dtype'] == 'bf16'


def test_decoder_reads_checkpoint_instead_of_casting_target(mixed):
    """The reference loads raw checkpoint tensors, independent of target modules."""
    driver = mixed.driver
    target = driver.target
    expected = driver.decoder.norm.weight.detach().clone()
    with torch.no_grad():
        target.model.norm.weight.add_(1)
    root = Path(target.config._name_or_path)
    manifest = json.loads((root.parent / 'heads/decoder/manifest.json').read_text())
    actual = AnchorDecoder.from_checkpoint(target, tap_ids=driver.head.target_layer_ids,
        device='cpu', dtype=torch.bfloat16, file_pins=manifest['target_files'])
    assert torch.equal(actual.norm.weight.view(torch.uint8), expected.view(torch.uint8))
    assert not torch.equal(actual.norm.weight, target.model.norm.weight.bfloat16())


@pytest.mark.parametrize('change', ['dtype', 'missing', 'unexpected', 'unpinned'])
def test_checkpoint_slice_fails_closed(tmp_path, change):
    """The reference rejects absent, unpinned or wrong-dtype slice tensors."""
    target, _, _, bundle = artifacts(tmp_path, 0)
    root = Path(target.config._name_or_path)
    index_path = root / 'model.safetensors.index.json'
    index = json.loads(index_path.read_text())
    name = 'model.layers.34.self_attn.q_proj.weight'
    manifest = json.loads((bundle / 'decoder/manifest.json').read_text())
    if change == 'missing':
        del index['weight_map'][name]
    elif change == 'unexpected':
        index['weight_map']['model.layers.34.extra.weight'] = index['weight_map'][name]
    elif change == 'unpinned':
        del manifest['target_files'][index['weight_map'][name]]
    index_path.write_text(json.dumps(index))
    manifest['target_files'][index_path.name] = hashlib.sha256(index_path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match='dtype mismatch|missing tensors|unexpected tensors|lacks pin'):
        AnchorDecoder.from_checkpoint(target, tap_ids=[1, 9, 17, 25, 33], device='cpu',
            dtype=torch.float32 if change == 'dtype' else torch.bfloat16,
            file_pins=manifest['target_files'])


def test_prepare_casts_before_hypothesis_assembly(mixed, monkeypatch):
    """The reference casts exact taps before tail/marker assembly."""
    from jetspec.methods.jetspec_plus_plus import drafter as module
    exact = torch.randn(1, 7, 80)
    seen = []
    build = module.build_protocol_hypothesis_contexts
    def observe(prefix, **kwargs):
        assert prefix.dtype == torch.bfloat16
        assert torch.equal(prefix, golden('value_4'))
        actual = build(prefix, **kwargs)
        expected = golden('value_5')
        for value, wanted in zip(actual, expected):
            assert torch.equal(value, wanted)
        seen.append(actual[0].dtype)
        return actual
    monkeypatch.setattr(module, 'build_protocol_hypothesis_contexts', observe)
    mixed.propose_logits(torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]]), 15, target_hidden=exact)
    assert seen and set(seen) == {torch.bfloat16}
    assert mixed.sequence.taps.dtype == torch.float32
    assert mixed.staged.predicted_hidden.dtype == torch.bfloat16


def test_predict_casts_before_adapter(mixed):
    """The reference uses draft-dtype adapter taps/embeddings."""
    driver = mixed.driver
    prefix = torch.randn(1, 7, 80)
    tokens = torch.tensor([[1, 2, 3]])
    seen = []
    hook = driver.adapter.register_forward_pre_hook(
        lambda module, args, kwargs: seen.append(kwargs), with_kwargs=True)
    try:
        predicted = driver.predict(prefix, tokens)
    finally:
        hook.remove()
    call, = seen
    assert torch.equal(call['prefix_h'], prefix.bfloat16())
    assert call['prefix_h'].dtype == call['h_prev'].dtype == torch.bfloat16
    assert call['token_embeddings'].dtype == call['attention_mask'].dtype == torch.bfloat16
    assert torch.equal(call['token_embeddings'], driver.embed_tokens(tokens))
    assert predicted.dtype == torch.bfloat16


def test_round_casts_before_draft_context(mixed):
    """The reference keeps J/R contexts bf16."""
    driver = mixed.driver
    exact = torch.randn(1, 7, 80)
    seen = []
    hook = driver.head.register_forward_pre_hook(
        lambda module, args, kwargs: seen.append(kwargs), with_kwargs=True)
    try:
        driver.run(torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]]), exact, admission=True)
    finally:
        hook.remove()
    assert len(seen) == 2
    assert torch.equal(seen[0]['target_hidden'], exact.bfloat16())
    for call in seen:
        assert call['target_hidden'].dtype == call['noise_embedding'].dtype == torch.bfloat16
    assert seen[1]['attention_mask'].dtype == torch.bfloat16


def test_decoder_committed_prefix_cast(mixed, monkeypatch):
    """The reference casts true fp32 tap33 to bf16 before prefix K/V."""
    monkeypatch.delenv('JETSPEC_DECODER_TABLES', raising=False)
    decoder = mixed.driver.decoder
    # The golden helper needs a tiny target-shaped owner of these same slice modules.
    target = SimpleNamespace(model=SimpleNamespace(layers=[None, None, *decoder.layers],
        norm=decoder.norm, rotary_emb=decoder.rotary), lm_head=decoder.lm_head)
    sequence = SimpleNamespace(taps=torch.randn(7, 5, 16))
    seen = []
    hook = decoder.layers[0].register_forward_pre_hook(lambda module, args: seen.append(args[0]))
    try:
        actual = decoder._prefix(sequence, 3)
    finally:
        hook.remove()
    assert seen[0].dtype == torch.bfloat16
    assert torch.equal(seen[0], sequence.taps[None, :3, 4].bfloat16())
    for index, pair in actual.kv.items():
        for value, wanted in zip(pair, golden('value_6')):
            assert value.dtype == torch.bfloat16
            assert torch.equal(value, wanted[..., :3, :])


def test_refresh_prefix_cast(mixed):
    """The reference casts committed fp32 taps to decoder bf16."""
    decoder = mixed.driver.decoder
    sequence = SimpleNamespace(taps=torch.randn(7, 5, 16))
    decoder._prefix(sequence, 3)
    seen = []
    hook = decoder.layers[0].register_forward_pre_hook(lambda module, args: seen.append(args[0]))
    try:
        refresh_decoder_logits(decoder, sequence, length=7, tail_hidden=None,
            tail_counts=[0], block_hidden=torch.randn(1, 2, 80).bfloat16(), slot_counts=[2])
    finally:
        hook.remove()
    assert seen[0].dtype == torch.bfloat16
    assert torch.equal(seen[0], sequence.taps[None, 3:7, 4].bfloat16())


@pytest.mark.parametrize('use_tail', [False, True])
def test_refresh_window_cast(mixed, use_tail):
    """The reference casts staged predictions to decoder dtype."""
    decoder = mixed.driver.decoder
    sequence = SimpleNamespace(taps=torch.randn(7, 5, 16))
    decoder._prefix(sequence, 7)
    tail, block = torch.randn(2, 80), torch.randn(1, 2, 80)
    seen = []
    hook = decoder.layers[0].register_forward_pre_hook(lambda module, args: seen.append(args[0]))
    try:
        logits, _, _ = refresh_decoder_logits(decoder, sequence, length=7,
            tail_hidden=tail if use_tail else None, tail_counts=[2 if use_tail else 0],
            block_hidden=block, slot_counts=[2])
    finally:
        hook.remove()
    expected = torch.cat((tail[:, -16:], block[0, :, -16:])).bfloat16() if use_tail else block[0, :, -16:].bfloat16()
    assert torch.equal(seen[0], expected[None])
    assert seen[0].dtype == logits.dtype == torch.bfloat16
