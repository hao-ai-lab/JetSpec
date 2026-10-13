"""BLEND decoder features and quantized hazard mass equal the served CPU arithmetic."""
from golden_values import golden
from types import SimpleNamespace
import pytest
import torch
from jetspec.methods.jetspec_plus_plus.method.blend import blend_features


@pytest.mark.parametrize('seed', [0, 1, 2])
def test_blend(monkeypatch, seed):
    torch.manual_seed(seed)
    logits = torch.randn(16, 41)
    drafts, p_top1 = torch.randint(0, 41, (15,)), torch.rand(16)
    offers, sizes = torch.rand(16, 8).double(), torch.full((16,), 8)
    coefficients = dict(zip(('a', 'b', 'c', 'argmax', 'top', 'pos'), torch.randn(6).tolist()))
    actual = blend_features(logits, drafts, p_top1, offers, sizes, coefficients)
    for key, value in {'E04_KSCORE': 'blend_dec', 'JETSPEC_DRAFTS': '15',
                       'JETSPEC_ROW_LAYOUT': 'anchor0',
                       'JETSPEC_DECODER_FRAME': 'wire'}.items():
        monkeypatch.setenv(key, value)
    block = SimpleNamespace(draft_ids=drafts, e04_p_top1=p_top1)
    thresholds = {'e04_kscore_blend': {'tiny': {
        name: {'value': value} for name, value in coefficients.items()}}}
    expected, checks = golden('value_1')
    assert not checks.any()
    assert torch.equal(actual, expected[1:])
