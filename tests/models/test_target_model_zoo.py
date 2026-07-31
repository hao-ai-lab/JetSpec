import pytest
import torch
from transformers import (
    DynamicCache,
    Gemma4Config,
    Gemma4ForConditionalGeneration,
    Gemma4TextConfig,
    GptOssConfig,
    GptOssForCausalLM,
    Qwen3Config,
)

from jetflow.core.model_runner import ModelRunner
from jetflow.draft_head_adapter import DraftHeadTreeDrafter
from jetflow.models.draft_head import DFlashDraftModel
from jetflow.models.target_adapter import (
    build_tree_attention_mask,
    require_backend_support,
    validate_draft_target_compatibility,
)


VOCAB_SIZE = 64
HIDDEN_SIZE = 32


def _draft_head() -> DFlashDraftModel:
    config = Qwen3Config(
        vocab_size=VOCAB_SIZE,
        hidden_size=HIDDEN_SIZE,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=128,
        tie_word_embeddings=False,
        block_size=4,
        num_target_layers=2,
    )
    config.dflash_config = {
        "target_layer_ids": [0],
        "mask_token_id": 0,
        "causal_head": True,
    }
    config._attn_implementation = "eager"
    return DFlashDraftModel(config).eval().to(torch.float32)


def _gpt_oss_target() -> GptOssForCausalLM:
    config = GptOssConfig(
        vocab_size=VOCAB_SIZE,
        hidden_size=HIDDEN_SIZE,
        intermediate_size=HIDDEN_SIZE,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        num_local_experts=4,
        num_experts_per_tok=2,
        experts_per_token=2,
        layer_types=["sliding_attention", "full_attention"],
        sliding_window=4,
        max_position_embeddings=131072,
    )
    return GptOssForCausalLM(config).eval().to(torch.float32)


def _gemma4_target() -> Gemma4ForConditionalGeneration:
    text_config = Gemma4TextConfig(
        vocab_size=VOCAB_SIZE,
        vocab_size_per_layer_input=VOCAB_SIZE,
        hidden_size_per_layer_input=4,
        hidden_size=HIDDEN_SIZE,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_global_key_value_heads=2,
        head_dim=8,
        global_head_dim=8,
        max_position_embeddings=128,
        sliding_window=4,
        layer_types=["sliding_attention", "full_attention"],
        enable_moe_block=True,
        num_experts=4,
        top_k_experts=2,
        moe_intermediate_size=16,
    )
    config = Gemma4Config(
        text_config=text_config,
        vision_config=None,
        audio_config=None,
    )
    return Gemma4ForConditionalGeneration(config).eval().to(torch.float32)


@pytest.mark.parametrize(
    ("factory", "model_type"),
    [(_gpt_oss_target, "gpt_oss"), (_gemma4_target, "gemma4")],
)
@torch.inference_mode()
def test_model_zoo_target_runs_draft_head(factory, model_type):
    torch.manual_seed(0)
    target = factory()
    head = _draft_head()
    compatibility = validate_draft_target_compatibility(head, target)
    assert compatibility.model_type == model_type

    runner = ModelRunner(target)
    prompt = torch.tensor([[1, 2, 3, 4]])
    logits, _, target_hidden = runner.forward(
        prompt,
        DynamicCache(),
        torch.arange(prompt.shape[1]).unsqueeze(0),
        output_hidden_states=True,
        target_layer_ids=head.target_layer_ids,
    )
    assert logits.shape == (1, 4, VOCAB_SIZE)
    assert target_hidden.shape == (1, 4, HIDDEN_SIZE)

    drafter = DraftHeadTreeDrafter(
        head,
        target,
        block_size=head.block_size,
        target_layer_ids=head.target_layer_ids,
    )
    draft_logits = drafter.propose_logits(
        prompt, head.block_size - 1, target_hidden=target_hidden
    )
    assert draft_logits.shape == (1, head.block_size - 1, VOCAB_SIZE)


@pytest.mark.parametrize("factory", [_gpt_oss_target, _gemma4_target])
def test_hybrid_tree_mask_preserves_sliding_window(factory):
    target = factory()
    ancestor = torch.tensor(
        [
            [1, 0, 0, 0],
            [1, 1, 0, 0],
            [1, 0, 1, 0],
            [1, 1, 0, 1],
        ],
        dtype=torch.bool,
    )
    masks = build_tree_attention_mask(
        target,
        ancestor,
        past_len=5,
        depths=torch.tensor([0, 1, 1, 2]),
        dtype=torch.float32,
    )
    assert set(masks) == {"full_attention", "sliding_attention"}
    full = masks["full_attention"][0, 0]
    sliding = masks["sliding_attention"][0, 0]
    assert torch.all(full[:, :5] == 0)
    # Window=4 and root query position=5: prefix positions 0 and 1 are outside.
    assert sliding[0, 0] < -1e20
    assert sliding[0, 1] < -1e20
    assert sliding[0, 2] == 0
    assert sliding[0, 5] == 0
    assert sliding[0, 6] < -1e20


@pytest.mark.parametrize("factory", [_gpt_oss_target, _gemma4_target])
def test_non_qwen_reference_triton_backend_is_supported(factory):
    require_backend_support(factory(), "triton")


@pytest.mark.parametrize("factory", [_gpt_oss_target, _gemma4_target])
def test_non_qwen_paged_backend_fails_loudly(factory):
    with pytest.raises(ValueError, match="Qwen3/Qwen3-MoE-only"):
        require_backend_support(factory(), "triton_paged_tree")
