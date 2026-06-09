import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from ptd.engine import SamplingParams
from ptd.engine.model_runner import ModelRunner
from ptd.nano_vllm.engine import NanoEngine


class _StubTokenizer:
    def decode(self, ids, skip_special_tokens=True):
        return " ".join(str(int(t)) for t in ids)


class _StubTreeDrafter:
    def __init__(self, vocab_size: int):
        self.vocab_size = int(vocab_size)

    def propose_logits(self, committed, D, target_hidden=None):
        logits = torch.full(
            (1, D, self.vocab_size),
            -1000.0,
            dtype=torch.float32,
            device=committed.device,
        )
        base = int(committed[0, -1].item())
        for depth in range(D):
            logits[0, depth, (base + depth + 1) % self.vocab_size] = 0.0
        return logits


def _tiny_model(seed: int = 0) -> Qwen3ForCausalLM:
    torch.manual_seed(seed)
    cfg = Qwen3Config(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=512,
        tie_word_embeddings=False,
    )
    return Qwen3ForCausalLM(cfg).eval().to(torch.float32)


def _tiny_nano(model, block_size: int) -> NanoEngine:
    eng = object.__new__(NanoEngine)
    eng.model = model
    eng.tokenizer = _StubTokenizer()
    eng.runner = ModelRunner(model)
    eng.device = "cpu"
    eng.dtype = torch.float32
    eng.block_size = block_size
    eng.eos_token_ids = set()
    return eng


def _tree(engine, prompt, drafter, *, tree_block_size: int, max_new: int = 12, **kwargs):
    return engine.generate_tree(
        prompt,
        drafter,
        block_size=tree_block_size,
        tree_width=2,
        budget=15,
        sampling_params=SamplingParams(0.0, max_new),
        **kwargs,
    )["token_ids"]


def test_tree_session_matches_fresh_calls_across_prompts_and_cache_blocks():
    prompts = [
        torch.tensor([[3, 14, 15, 92]]),
        torch.tensor([[65, 35, 89, 7, 9, 3]]),
        torch.tensor([[2, 71, 82]]),
    ]

    for block_size in (4, 16):
        model = _tiny_model(0)
        drafter = _StubTreeDrafter(model.config.vocab_size)
        fresh = _tiny_nano(model, block_size)
        session = _tiny_nano(model, block_size)

        expected = [
            _tree(fresh, prompt, drafter, tree_block_size=block_size)
            for prompt in prompts
        ]
        got = []
        cache_ids = []
        for prompt in prompts:
            got.append(_tree(
                session,
                prompt,
                drafter,
                tree_block_size=block_size,
                session=True,
                session_prompt_capacity=8,
            ))
            cache_ids.append(id(session._tree_session["cache"]))

        assert got == expected
        assert len(set(cache_ids)) == 1


def test_tree_session_resets_stale_long_prompt_state_before_short_prompt():
    model = _tiny_model(1)
    drafter = _StubTreeDrafter(model.config.vocab_size)
    long_prompt = torch.tensor([[3, 14, 15, 92, 65, 35, 89, 7, 9, 3, 2, 71]])
    short_prompt = torch.tensor([[8, 6, 7]])
    fresh_short = _tiny_nano(model, 4)
    session = _tiny_nano(model, 4)

    _tree(
        session,
        long_prompt,
        drafter,
        tree_block_size=4,
        session=True,
        session_prompt_capacity=16,
    )

    expected = _tree(fresh_short, short_prompt, drafter, tree_block_size=4)
    got = _tree(
        session,
        short_prompt,
        drafter,
        tree_block_size=4,
        session=True,
        session_prompt_capacity=16,
    )

    assert got == expected


def test_tree_session_rejects_prompt_beyond_session_capacity():
    model = _tiny_model(2)
    drafter = _StubTreeDrafter(model.config.vocab_size)
    session = _tiny_nano(model, 4)

    _tree(
        session,
        torch.tensor([[3, 14, 15, 92]]),
        drafter,
        tree_block_size=4,
        session=True,
        session_prompt_capacity=4,
    )

    with pytest.raises(ValueError, match="prompt_len.*session_prompt_capacity"):
        _tree(
            session,
            torch.tensor([[3, 14, 15, 92, 65]]),
            drafter,
            tree_block_size=4,
            session=True,
            session_prompt_capacity=4,
        )


def test_tree_session_cache_reset_hygiene_returns_all_blocks():
    model = _tiny_model(3)
    drafter = _StubTreeDrafter(model.config.vocab_size)
    session = _tiny_nano(model, 4)

    _tree(
        session,
        torch.tensor([[3, 14, 15, 92, 65, 35]]),
        drafter,
        tree_block_size=4,
        session=True,
        session_prompt_capacity=8,
    )
    cache = session._tree_session["cache"]
    assert cache.get_seq_length() > 0

    cache.reset()

    assert cache.get_seq_length() == 0
    assert cache.num_free_blocks == cache._num_blocks
