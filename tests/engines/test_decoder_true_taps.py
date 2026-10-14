"""The live window follows every verified true tap, at the anchor itself."""
import copy
from types import SimpleNamespace

import pytest
import torch
from transformers import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3DecoderLayer, Qwen3RMSNorm, Qwen3RotaryEmbedding,
)

from jetspec.engines.jetspec_plus_plus.models.heads import decoder


class Sequence(SimpleNamespace):
    pass


def _case(tmp_path, monkeypatch, *, width, prefix_rows=9):
    monkeypatch.setenv('JETSPEC_DRAFTS', str(width))
    monkeypatch.setenv('JETSPEC_ROW_LAYOUT', 'anchor0')
    monkeypatch.delenv('JETSPEC_ROWS_FUSED', raising=False)
    monkeypatch.setenv('JETSPEC_HEADS_FAST', '0')
    monkeypatch.setattr(decoder, 'anchor_offers', lambda *args, **kwargs: object())
    cfg = Qwen3Config(hidden_size=32, intermediate_size=64, num_attention_heads=4,
                      num_key_value_heads=2, head_dim=8, vocab_size=64,
                      num_hidden_layers=36, max_position_embeddings=128)
    cfg.layer_types = ['full_attention'] * 36
    cfg._attn_implementation = 'sdpa'
    torch.manual_seed(404)
    layers = torch.nn.ModuleList([Qwen3DecoderLayer(cfg, i) for i in (34, 35)])
    model = decoder.AnchorDecoder(
        layers=layers, prefix_layers=layers,
        norm=Qwen3RMSNorm(32, eps=cfg.rms_norm_eps),
        lm_head=torch.nn.Linear(32, 64, bias=False),
        rotary=Qwen3RotaryEmbedding(cfg, device='cpu'), root=tmp_path,
        hidden_size=32).eval()
    sequence = Sequence(taps=torch.randn(prefix_rows, 5, 32),
                        staged=SimpleNamespace(predicted_hidden=torch.randn(1, width + 1, 32)))
    block = SimpleNamespace(draft_ids=torch.randint(0, 64, (width,)), correction_guess=2)
    return model, sequence, block


def _predict(model, sequence, block):
    return model.predict(sequence, block, dataset='gsm8k', coverage='0.99',
                         grouped=False, cap=16)


def _reference(model, sequence):
    """Frozen layers over every true prefix row and the current block."""
    n = len(sequence.taps)
    current = sequence.staged.predicted_hidden
    hidden = torch.cat((sequence.taps[:, 4][None], current), dim=1)
    length = hidden.shape[1]
    positions = torch.arange(length)[None]
    allowed = torch.arange(length)[None, :] <= torch.arange(length)[:, None]
    mask = hidden.new_zeros((1, 1, length, length)).masked_fill_(
        ~allowed[None, None], torch.finfo(hidden.dtype).min)
    rope = model.rotary(hidden, positions)
    for layer in model.layers:
        hidden = layer(hidden, attention_mask=mask, position_ids=positions,
                       position_embeddings=rope, use_cache=False)
    decoded = model.norm(hidden)[:, n:][0]
    return decoded, model.lm_head(decoded), positions[:, n:], mask[:, :, n:, :]


@pytest.mark.parametrize('width', [14, 15])
@pytest.mark.parametrize('prefix_rows', [1, 3, 9])
@torch.inference_mode()
def test_predict_matches_full_true_prefix(tmp_path, monkeypatch, width, prefix_rows):
    model, sequence, block = _case(tmp_path, monkeypatch, width=width, prefix_rows=prefix_rows)
    reference = copy.deepcopy(model)
    captured = []
    hook = model.layers[0].register_forward_pre_hook(
        lambda module, args, kwargs: captured.append((args[0].clone(), kwargs)), with_kwargs=True)
    actual = _predict(model, sequence, block)
    expected_hidden, expected_logits, positions, mask = _reference(reference, sequence)
    assert model._prefixes[id(sequence)].length == len(sequence.taps)
    assert torch.equal(captured[-1][0], sequence.staged.predicted_hidden)
    assert torch.equal(captured[-1][1]['position_ids'], positions)
    assert torch.equal(captured[-1][1]['attention_mask'], mask)
    torch.testing.assert_close(actual.hidden, expected_hidden, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(actual.logits, expected_logits, atol=1e-5, rtol=1e-5)
    hook.remove()


@pytest.mark.parametrize('width', [14, 15])
@pytest.mark.parametrize('glue_level', [0, 1])
@torch.inference_mode()
def test_chain_inputs_and_fused_glue_match_eager(tmp_path, monkeypatch, width, glue_level):
    from jetspec.engines.jetspec_plus_plus.models.heads.decoder_glue import window_glue, prefix_glue
    from jetspec.engines.jetspec_plus_plus.wire.chain.contract import Offset
    model, sequence, _ = _case(tmp_path, monkeypatch, width=width)
    _, _, positions, mask = _reference(model, sequence)
    rows = width + 1
    length = torch.tensor(len(sequence.taps))
    bucket = len(sequence.taps) + rows
    front = SimpleNamespace(staged_hidden=sequence.staged.predicted_hidden,
                            decoder_bucket=bucket)
    state = dict(hidden=torch.empty(1, rows, 32), lanes=torch.arange(rows),
                 slots=torch.empty(rows, dtype=torch.long),
                 masks={bucket: torch.empty_like(mask)})
    actual_positions = decoder.stage_chain_window_state(state, front, length,
                                                        glue_level=glue_level)
    assert torch.equal(state['hidden'], sequence.staged.predicted_hidden)
    assert torch.equal(actual_positions, positions)
    assert torch.equal(state['masks'][bucket], mask)
    cos_table = torch.randn(64, 8)
    sin_table = torch.randn_like(cos_table)
    cos, sin = torch.empty(1, rows, 8), torch.empty(1, rows, 8)
    window_glue(length, front.staged_hidden, cos_table, sin_table,
                state['hidden'], state['masks'][bucket], state['slots'], cos, sin)
    assert torch.equal(state['hidden'], sequence.staged.predicted_hidden)
    assert torch.equal(state['slots'], positions[0])
    assert torch.equal(state['masks'][bucket], mask)
    assert torch.equal(cos, cos_table[positions])
    assert torch.equal(sin, sin_table[positions])
    # Prefix appends include the last verified row and never read a padded lane.
    for take in (1, rows // 2, rows):
        prefix_length = torch.tensor(24)
        count = torch.tensor(take)
        expected_slots = torch.cat((torch.arange(24 - take, 24),
                                    torch.full((rows - take,), bucket - 1)))
        slots = decoder.chain_decoder_prefix_slots(prefix_length, count, bucket, rows)
        assert torch.equal(slots, expected_slots)
        prefix_mask = decoder.chain_decoder_prefix_mask(prefix_length, count, bucket, rows,
                                                        torch.float32)
        for lane in range(rows):
            visible = torch.nonzero(prefix_mask[0, 0, lane] == 0).flatten()
            expected = torch.arange(min(24 - take + lane + 1, bucket)) if lane < take else torch.tensor([0])
            assert torch.equal(visible, expected)
        rp = torch.zeros(64, dtype=torch.long)
        rp[Offset.DEC_LEN], rp[Offset.C1_COUNT] = prefix_length, count
        prefix_positions = torch.empty(1, rows, dtype=torch.long)
        prefix_slots = torch.empty(rows, dtype=torch.long)
        fused_mask = torch.empty_like(prefix_mask)
        prefix_glue(rp, torch.arange(width), cos_table, sin_table, fused_mask,
                    prefix_positions, prefix_slots, cos, sin, torch.empty(rows, dtype=torch.long),
                    length_offset=Offset.DEC_LEN, count_offset=Offset.C1_COUNT,
                    correction_offset=Offset.CORRECTION, miss_offset=Offset.AFTER_MISS,
                    wire=False, zero_miss=False)
        assert torch.equal(prefix_slots, slots)
        assert torch.equal(fused_mask, prefix_mask)
        assert torch.equal(prefix_positions, (24 - take + torch.arange(rows))[None])
