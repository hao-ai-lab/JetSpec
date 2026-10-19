"""The reader factory verifies target pins and connects strict local artifacts."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path
import pytest
import torch
from safetensors.torch import save_file
from transformers import Qwen3ForCausalLM
from conftest import target_config
from test_adapter import adapter_config
from test_hooks import checkpoint
from test_lossless import llm_for
from real_artifacts import REAL_8B, tiny_config
from jetspec.methods.jetspec_plus_plus.models.adapter import CtxKVC1Head
from jetspec.methods.jetspec_plus_plus.round import JetSpecPPConfig


def local_artifacts(tmp_path, seed):
    config = target_config()
    config.rope_theta = tiny_config('drafter')['rope_theta']
    config.num_hidden_layers = 36
    config.layer_types = ['full_attention'] * 36
    torch.manual_seed(seed)
    target = Qwen3ForCausalLM(config).eval()
    target_path = tmp_path / 'target'
    target.save_pretrained(target_path, max_shard_size='50KB')
    target.config._name_or_path = str(target_path)
    head_path = tmp_path / 'drafter'
    head_path.mkdir()
    head = checkpoint(head_path, seed)
    adapter_path = tmp_path / 'adapter'
    adapter_path.mkdir()
    ac = adapter_config()
    ac.rope_theta = config.rope_theta
    ac.dflash_config['target_layer_ids'] = [1, 9, 17, 25, 33]
    ac.save_pretrained(adapter_path)
    adapter = CtxKVC1Head(ac).eval()
    save_file(adapter.state_dict(), adapter_path / 'head.safetensors')
    save_file({key: head.state_dict()[key] for key in ('fc.weight', 'hidden_norm.weight')},
              adapter_path / 'converter.safetensors')
    bundle_path = tmp_path / 'heads'
    (bundle_path / 'decoder').mkdir(parents=True)
    thresholds = dict(
        e04_kscore_blend={'tiny': {key: {'value': 0.0} for key in
                                 ('a', 'b', 'c', 'argmax', 'top', 'pos')}},
        e04_rowrule_joint={'tiny': {'tau': {'pooled': {'value': 0.0}},
                                   'temperature': {'value': 1.0}}},
        anchor_predictor={'pooled': {'tiny': {'0.99': {'value': 1.0}}}})
    (bundle_path / 'thresholds.json').write_text(json.dumps(thresholds))
    files = sorted(target_path.glob('*.safetensors'))[-2:] + [target_path / 'model.safetensors.index.json']
    pins = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
    manifest = dict(source='target_slice', target_files=pins)
    (bundle_path / 'decoder/manifest.json').write_text(json.dumps(manifest))
    return target, head_path, adapter_path, bundle_path, pins


@pytest.mark.parametrize('seed', [0, 1, 2])
def test_factory_reads_strict_local_checkpoints(tmp_path, seed):
    from jetspec.methods.jetspec_plus_plus.drafter import build_drafter
    from jetspec.methods.jetspec_plus_plus.reader import read_passes
    from jetspec.core.llm import SamplingParams
    target, head, adapter, bundle, pins = local_artifacts(tmp_path, seed)
    llm = llm_for(target)
    def factory(**kwargs):
        return build_drafter(**kwargs, drafter_checkpoint=head, adapter_artifact=adapter,
                             heads_bundle=bundle, dataset='tiny')
    drafter = factory(llm=llm, config=JetSpecPPConfig())
    assert drafter.driver.decoder.norm is target.model.norm
    assert drafter.artifact_pins == pins
    prompt = torch.tensor([[1, 2, 3, 4, 5, 6]])
    expected = llm.generate(prompt, SamplingParams(max_new_tokens=16))['token_ids']
    actual, = read_passes(llm, [{'token_ids': prompt[0].tolist()}], drafter_factory=factory, tokens=16)
    assert actual['token_ids'] == expected
    assert actual['accept_lengths']
    assert actual['passes_run']


@pytest.mark.parametrize('change', ['pin0', 'pin1', 'pin2', 'converter'])
def test_factory_refuses_mismatched_artifacts(tmp_path, change):
    from jetspec.methods.jetspec_plus_plus.drafter import build_drafter
    target, head, adapter, bundle, pins = local_artifacts(tmp_path, 0)
    if change.startswith('pin'):
        name = list(pins)[int(change[-1])]
        with (Path(target.config._name_or_path) / name).open('ab') as stream:
            stream.write(b'x')
    else:
        save_file({'fc.weight': torch.randn(16, 79), 'hidden_norm.weight': torch.randn(16)},
                  adapter / 'converter.safetensors')
    with pytest.raises(ValueError, match='SHA256|converter'):
        build_drafter(llm=llm_for(target), config=JetSpecPPConfig(), drafter_checkpoint=head,
                      adapter_artifact=adapter, heads_bundle=bundle, dataset='tiny')


def test_factory_reads_real_metadata_and_provenance_only_converter(tmp_path):
    from transformers import Qwen3Config
    from jetspec.methods.jetspec_plus_plus.drafter import build_drafter

    target, head_path, adapter_path, bundle_path, pins = local_artifacts(tmp_path, 0)
    raw = tiny_config('adapter')
    (adapter_path / 'config.json').write_text(json.dumps(raw))
    adapter = CtxKVC1Head(Qwen3Config.from_dict(raw)).eval()
    save_file(adapter.state_dict(), adapter_path / 'head.safetensors')
    # A projection_weight=0 converter carries provenance rather than served weights.
    converter = {'fc.weight': torch.randn(16, 80), 'hidden_norm.weight': torch.randn(16)}
    save_file(converter, adapter_path / 'converter.safetensors')
    (bundle_path / 'thresholds.json').write_bytes((REAL_8B / 'heads_b16/thresholds.json').read_bytes())
    manifest = json.loads((REAL_8B / 'heads_b16/decoder/manifest.json').read_text())
    manifest['target_files'] = pins  # Tiny target files have their own three SHA256 digests.
    (bundle_path / 'decoder/manifest.json').write_text(json.dumps(manifest))
    drafter = build_drafter(llm=llm_for(target), config=JetSpecPPConfig(),
        drafter_checkpoint=head_path, adapter_artifact=adapter_path,
        heads_bundle=bundle_path, dataset='gsm8k')
    assert len(drafter.artifact_pins) == 3
    assert drafter.artifact_pins == pins
    assert drafter.driver.head.stock_drafter
    assert drafter.driver.adapter.stream_depths == [1, 1, 1, 4, 5]


def test_factory_validates_unchanged_real_target_slice_pins(tmp_path):
    from jetspec.methods.jetspec_plus_plus.drafter import build_drafter

    target, head, adapter, bundle, _ = local_artifacts(tmp_path, 0)
    manifest_bytes = (REAL_8B / 'heads_b16/decoder/manifest.json').read_bytes()
    (bundle / 'decoder/manifest.json').write_bytes(manifest_bytes)
    pins = json.loads(manifest_bytes)['target_files']
    target_root = Path(target.config._name_or_path)
    for name in pins:
        (target_root / name).write_bytes(b'not the pinned target artifact')
    with pytest.raises(ValueError, match='target-file SHA256 mismatch'):
        build_drafter(llm=llm_for(target), config=JetSpecPPConfig(), drafter_checkpoint=head,
                      adapter_artifact=adapter, heads_bundle=bundle, dataset='tiny')


def test_cli_reads_local_artifacts_with_public_factory(tmp_path):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast
    target, head, adapter, bundle, _ = local_artifacts(tmp_path, 4)
    target_path = Path(target.config._name_or_path)
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(
        WordLevel({str(i): i for i in range(41)}, unk_token='0')), unk_token='0')
    tokenizer.save_pretrained(target_path)
    prompts = tmp_path / 'prompts.json'
    prompts.write_text(json.dumps([{'token_ids': [1, 2, 3, 4, 5, 6]}]))
    output = tmp_path / 'read.json'
    script = 'import torch; torch.set_num_threads(1); from jetspec.methods.jetspec_plus_plus.reader import main; main()'
    result = subprocess.run([sys.executable, '-c', script, '--target', str(target_path),
        '--drafter', str(head), '--adapter', str(adapter), '--heads-bundle', str(bundle),
        '--prompts-file', str(prompts), '--output', str(output), '--max-new-tokens', '8',
        '--drafter-factory', 'jetspec.methods.jetspec_plus_plus.drafter:build_drafter'],
        capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    read = json.loads(output.read_text())
    assert len(read['outputs'][0]['token_ids']) == 8
    assert read['rule_a']['tokens'] == 7
    assert Path(str(output) + '.manifest.json').is_file()
