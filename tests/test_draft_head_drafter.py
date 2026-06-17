from types import SimpleNamespace

import torch
from torch import nn
from transformers import Qwen3Config

from bench import reseed_probe
from jetflow.draft_head_drafter import DraftHeadTreeDrafter
from jetflow.models.draft_head import DFlashDraftModel


class _TinyTarget(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int):
        super().__init__()
        self.model = SimpleNamespace(embed_tokens=nn.Embedding(vocab_size, hidden_size))
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)


def _tiny_head_and_target(seed: int = 0, *, causal_head: bool = False):
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
        "causal_head": causal_head,
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
def test_eager_conditioned_logits_match_reseed_reference_exactly():
    head, target = _tiny_head_and_target()
    drafter = DraftHeadTreeDrafter(head, target, block_size=4, target_layer_ids=[0])
    context_ids, target_hidden = _inputs(3, head.config.vocab_size, head.config.hidden_size)

    actual = drafter.propose_logits_conditioned(context_ids, [5], target_hidden=target_hidden)
    expected = reseed_probe.conditioned_logits(drafter._fwd, context_ids, [5], target_hidden)

    assert actual.shape == (1, 3, head.config.vocab_size)
    assert torch.equal(actual, expected)


@torch.inference_mode()
def test_eager_draft_cache_crops_to_target_context_and_matches_full_forward():
    head, target = _tiny_head_and_target()
    cached = DraftHeadTreeDrafter(head, target, block_size=4, target_layer_ids=[0])
    fresh = DraftHeadTreeDrafter(head, target, block_size=4, target_layer_ids=[0])

    context_ids, target_hidden = _inputs(3, head.config.vocab_size, head.config.hidden_size)
    first = cached.propose_logits(context_ids, 3, target_hidden=target_hidden)
    torch.testing.assert_close(first, fresh.propose_logits(context_ids, 3, target_hidden=target_hidden))
    assert cached._fwd.cache.get_seq_length() == target_hidden.shape[1]

    next_context_ids = torch.cat(
        [context_ids, torch.tensor([[7]], dtype=context_ids.dtype)],
        dim=1,
    )
    next_hidden = torch.cat(
        [target_hidden, torch.randn(1, 1, head.config.hidden_size)],
        dim=1,
    )
    cached_next = cached.propose_logits(next_context_ids, 3, target_hidden=next_hidden)

    fresh.reset_cache()
    fresh_next = fresh.propose_logits(next_context_ids, 3, target_hidden=next_hidden)
    torch.testing.assert_close(cached_next, fresh_next, rtol=1e-5, atol=1e-5)
    assert cached._fwd.cache.get_seq_length() == next_hidden.shape[1]


@torch.inference_mode()
def test_eager_conditioned_logits_can_reuse_cached_context():
    head, target = _tiny_head_and_target()
    cached = DraftHeadTreeDrafter(head, target, block_size=4, target_layer_ids=[0])
    fresh = DraftHeadTreeDrafter(head, target, block_size=4, target_layer_ids=[0])
    context_ids, target_hidden = _inputs(3, head.config.vocab_size, head.config.hidden_size)

    cached.propose_logits(context_ids, 3, target_hidden=target_hidden)
    actual = cached.propose_logits_conditioned(context_ids, [5], target_hidden=target_hidden)
    expected = fresh.propose_logits_conditioned(context_ids, [5], target_hidden=target_hidden)

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    assert cached._fwd.cache.get_seq_length() == target_hidden.shape[1]


@torch.inference_mode()
def test_causal_eager_draft_cache_matches_fresh_full_context():
    head, target = _tiny_head_and_target(causal_head=True)
    cached = DraftHeadTreeDrafter(head, target, block_size=4, target_layer_ids=[0])
    fresh = DraftHeadTreeDrafter(head, target, block_size=4, target_layer_ids=[0])

    context_ids, target_hidden = _inputs(3, head.config.vocab_size, head.config.hidden_size)
    first = cached.propose_logits(context_ids, 3, target_hidden=target_hidden)
    fresh_first = fresh.propose_logits(context_ids, 3, target_hidden=target_hidden)
    torch.testing.assert_close(first, fresh_first, rtol=1e-5, atol=1e-5)

    next_context_ids = torch.cat(
        [context_ids, torch.tensor([[7]], dtype=context_ids.dtype)],
        dim=1,
    )
    next_hidden = torch.cat(
        [target_hidden, torch.randn(1, 1, head.config.hidden_size)],
        dim=1,
    )

    cached_next = cached.propose_logits(next_context_ids, 3, target_hidden=next_hidden)
    fresh.reset_cache()
    fresh_next = fresh.propose_logits(next_context_ids, 3, target_hidden=next_hidden)

    torch.testing.assert_close(cached_next, fresh_next, rtol=1e-5, atol=1e-5)
    assert cached._fwd.cache.get_seq_length() == next_hidden.shape[1]
