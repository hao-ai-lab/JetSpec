from types import SimpleNamespace

import torch
from torch import nn
from transformers import Qwen3Config

from bench import reseed_probe
from ptd.draft_head_drafter import CompiledDraftHead, DraftHeadTreeDrafter, GraphedDraftHead, _DraftHeadForward
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


def _cpu_graphed_draft_head(head, target, ctx_buckets=(4, 8)):
    compiled = CompiledDraftHead(
        head,
        target,
        block_size=4,
        target_layer_ids=[0],
        ctx_buckets=ctx_buckets,
        compile=False,
    )
    graphed = object.__new__(GraphedDraftHead)
    graphed.compiled = compiled
    graphed.device = compiled.device
    graphed.dtype = compiled.dtype
    graphed.block_size = compiled.block_size
    graphed._buffers = {}
    graphed.graphs = {}
    graphed.outputs = {}
    graphed._pool = None
    return graphed


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
        path_tokens = torch.tensor([5, 7], dtype=torch.long)
        expected_conditioned = eager._forward_head(
            context_ids,
            target_hidden,
            depth=3,
            fill_tokens=path_tokens,
        )
        actual_conditioned = compiled.propose_logits_conditioned(
            context_ids,
            path_tokens,
            target_hidden=target_hidden,
        )

        assert compiled.bucket_for_ctx_len(ctx_len) in (4, 8)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(actual_conditioned, expected_conditioned, rtol=1e-5, atol=1e-5)


@torch.inference_mode()
def test_compiled_draft_head_reuses_one_bucket_graph_across_context_lengths():
    head, target = _tiny_head_and_target()
    compiled = CompiledDraftHead(
        head,
        target,
        block_size=4,
        target_layer_ids=[0],
        ctx_buckets=(16,),
    )

    old_error_on_recompile = torch._dynamo.config.error_on_recompile
    torch._dynamo.reset()
    torch._dynamo.config.error_on_recompile = True
    try:
        for ctx_len in range(1, 13):
            context_ids, target_hidden = _inputs(ctx_len, head.config.vocab_size, head.config.hidden_size)
            logits = compiled.propose_logits(context_ids, depth=3, target_hidden=target_hidden)
            path_tokens = torch.tensor(
                [(ctx_len + 3) % head.config.vocab_size, (ctx_len + 4) % head.config.vocab_size],
                dtype=torch.long,
            )
            conditioned = compiled.propose_logits_conditioned(
                context_ids,
                path_tokens,
                target_hidden=target_hidden,
            )

            assert compiled.bucket_for_ctx_len(ctx_len) == 16
            assert logits.shape == (1, 3, head.config.vocab_size)
            assert conditioned.shape == (1, 3, head.config.vocab_size)
    finally:
        torch._dynamo.config.error_on_recompile = old_error_on_recompile
        torch._dynamo.reset()


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


@torch.inference_mode()
def test_graphed_draft_head_stages_every_head_input_before_replay():
    head, target = _tiny_head_and_target()
    graphed = _cpu_graphed_draft_head(head, target)
    context_ids, target_hidden = _inputs(3, head.config.vocab_size, head.config.hidden_size)

    capacity, buffers = graphed._stage_inputs(context_ids, target_hidden)

    assert capacity == 4
    assert set(buffers) == {"noise_embedding", "target_hidden", "position_ids", "attention_mask"}
    torch.testing.assert_close(buffers["noise_embedding"], graphed.compiled.noise_embedding(context_ids))
    torch.testing.assert_close(
        buffers["target_hidden"],
        graphed.compiled.pad_target_hidden(target_hidden, capacity),
    )
    torch.testing.assert_close(
        buffers["position_ids"],
        graphed.compiled._position_ids(real_ctx_len=3, capacity=capacity),
    )
    torch.testing.assert_close(
        buffers["attention_mask"],
        graphed.compiled._attention_mask(real_ctx_len=3, capacity=capacity),
    )


@torch.inference_mode()
def test_graphed_draft_head_restages_anchor_embedding_each_round():
    head, target = _tiny_head_and_target()
    graphed = _cpu_graphed_draft_head(head, target)
    context_ids, target_hidden = _inputs(3, head.config.vocab_size, head.config.hidden_size)
    graphed._stage_inputs(context_ids, target_hidden)
    first_noise = graphed._buffers[4]["noise_embedding"].clone()

    next_context_ids = context_ids.clone()
    next_context_ids[:, -1] = (next_context_ids[:, -1] + 7).remainder(head.config.vocab_size)
    graphed._stage_inputs(next_context_ids, target_hidden)
    expected_noise = graphed.compiled.noise_embedding(next_context_ids)

    assert not torch.equal(first_noise, expected_noise)
    torch.testing.assert_close(graphed._buffers[4]["noise_embedding"], expected_noise)


@torch.inference_mode()
def test_graphed_draft_head_stages_conditioned_path_embedding_before_replay():
    head, target = _tiny_head_and_target()
    graphed = _cpu_graphed_draft_head(head, target)
    context_ids, target_hidden = _inputs(3, head.config.vocab_size, head.config.hidden_size)

    capacity, buffers = graphed._stage_inputs(context_ids, target_hidden, fill_tokens=[10, 20])
    expected_noise = graphed.compiled.noise_embedding(context_ids, fill_tokens=[10, 20])
    default_noise = graphed.compiled.noise_embedding(context_ids)

    assert capacity == 4
    assert not torch.equal(default_noise, expected_noise)
    torch.testing.assert_close(buffers["noise_embedding"], expected_noise)
