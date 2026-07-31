from types import SimpleNamespace

import pytest
import torch
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

pytest.importorskip("triton")

from jetflow.core import tree_attention_kernel as kernel


def test_tree_attention_context_uses_global_hf_registry_and_restores_it():
    original = ALL_ATTENTION_FUNCTIONS.get("sdpa")
    ancestor = torch.eye(3, dtype=torch.uint8)
    depths = torch.tensor([0, 1, 1])

    with kernel.use_tree_attention(ancestor, depths, prefix_len=5):
        assert ALL_ATTENTION_FUNCTIONS.get("sdpa") is kernel._triton_attention_forward
        assert torch.equal(kernel._tree_attn_state["depths"], depths.to(torch.int32))

    assert ALL_ATTENTION_FUNCTIONS.get("sdpa") is original
    assert kernel._tree_attn_state["ancestor"] is None
    assert kernel._tree_attn_state["depths"] is None


def test_triton_dispatch_forwards_sliding_window_and_gpt_oss_sinks(monkeypatch):
    captured = {}

    def fake_tree_sdpa(query, key, value, ancestor, depths, prefix_len, **kwargs):
        captured.update(
            ancestor=ancestor,
            depths=depths,
            prefix_len=prefix_len,
            **kwargs,
        )
        return torch.zeros_like(query)

    monkeypatch.setattr(kernel, "_tree_sdpa", fake_tree_sdpa)
    kernel._tree_attn_state.update(
        ancestor=torch.eye(2, dtype=torch.uint8),
        depths=torch.tensor([0, 1], dtype=torch.int32),
        prefix_len=7,
    )
    sinks = torch.tensor([0.1, 0.2, 0.3])
    module = SimpleNamespace(sliding_window=128)
    q = torch.randn(1, 3, 2, 4)
    kv = torch.randn(1, 1, 9, 4)

    output, weights = kernel._triton_attention_forward(
        module, q, kv, kv, None, scaling=0.5, s_aux=sinks
    )

    assert output.shape == (1, 2, 3, 4)
    assert weights is None
    assert captured["prefix_len"] == 7
    assert captured["sliding_window"] == 128
    assert captured["sinks"] is sinks
    assert captured["sm_scale"] == 0.5
    kernel._clear_tree_mask()
