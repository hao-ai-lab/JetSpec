from types import SimpleNamespace

import torch
from torch import nn
from transformers import Qwen3Config

from ptd.draft_head_drafter import CompiledDraftHead, _DraftHeadForward
from ptd.models.draft_head import DFlashDraftModel


class _TinyTarget(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int):
        super().__init__()
        self.model = SimpleNamespace(embed_tokens=nn.Embedding(vocab_size, hidden_size))
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)


def _tiny_head_and_target(seed: int = 0):
    torch.manual_seed(seed)
    cfg = Qwen3Config(
        vocab_size=31,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
        tie_word_embeddings=False,
        block_size=4,
        num_target_layers=1,
    )
    cfg.dflash_config = {
        "target_layer_ids": [0],
        "mask_token_id": 0,
        "causal_head": False,
    }
    cfg._attn_implementation = "eager"
    head = DFlashDraftModel(cfg).eval().to(torch.float32)
    target = _TinyTarget(cfg.vocab_size, cfg.hidden_size).eval().to(torch.float32)
    return head, target


def _inputs(ctx_len: int, vocab_size: int, hidden_size: int):
    context_ids = torch.arange(1, ctx_len + 2, dtype=torch.long).remainder(vocab_size).view(1, -1)
    target_hidden = torch.randn(1, ctx_len, hidden_size)
    return context_ids, target_hidden


@torch.inference_mode()
def test_compiled_draft_head_matches_eager_across_context_buckets():
    head, target = _tiny_head_and_target()
    eager = _DraftHeadForward(head, target, block_size=4, target_layer_ids=[0])
    compiled = CompiledDraftHead(
        head,
        target,
        block_size=4,
        target_layer_ids=[0],
        ctx_buckets=(4, 8),
    )

    for ctx_len in (2, 4, 5):
        context_ids, target_hidden = _inputs(ctx_len, head.config.vocab_size, head.config.hidden_size)
        expected = eager._forward_head(context_ids, target_hidden, depth=3)
        actual = compiled(context_ids, target_hidden, depth=3)

        assert compiled.bucket_for_ctx_len(ctx_len) in (4, 8)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


def test_draft_head_context_bucket_math():
    head, target = _tiny_head_and_target()
    compiled = CompiledDraftHead(
        head,
        target,
        block_size=4,
        target_layer_ids=[0],
        ctx_buckets=(4, 8),
        compile=False,
    )

    assert compiled.bucket_for_ctx_len(1) == 4
    assert compiled.bucket_for_ctx_len(4) == 4
    assert compiled.bucket_for_ctx_len(5) == 8
    assert compiled.bucket_for_ctx_len(8) == 8
    assert compiled.bucket_for_ctx_len(9) == 16


@torch.inference_mode()
def test_draft_head_padded_context_tail_is_masked():
    head, target = _tiny_head_and_target()
    compiled = CompiledDraftHead(
        head,
        target,
        block_size=4,
        target_layer_ids=[0],
        ctx_buckets=(4, 8),
        compile=False,
    )
    context_ids, target_hidden = _inputs(3, head.config.vocab_size, head.config.hidden_size)
    padded = compiled.pad_target_hidden(target_hidden, capacity=4)
    changed_tail = padded.clone()
    changed_tail[:, 3:, :] = torch.randn_like(changed_tail[:, 3:, :]) * 1000.0

    baseline = compiled.forward_padded(context_ids, padded, real_ctx_len=3, depth=3)
    changed = compiled.forward_padded(context_ids, changed_tail, real_ctx_len=3, depth=3)

    torch.testing.assert_close(changed, baseline, rtol=1e-5, atol=1e-5)
