import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from jetflow.draft import RandomTreeDrafter, TargetEchoTreeDrafter
from jetflow.core.llm import LLM, SamplingParams
from jetflow.core.model_runner import ModelRunner
from jetflow.inference_engine.engine import JetFlowEngine


class _StubTokenizer:
    def decode(self, ids, skip_special_tokens=True):
        return " ".join(str(int(t)) for t in ids)


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
        max_position_embeddings=256,
        tie_word_embeddings=False,
    )
    return Qwen3ForCausalLM(cfg).eval().to(torch.float32)


def _tiny_llm(model) -> LLM:
    llm = object.__new__(LLM)
    llm.model = model
    llm.tokenizer = _StubTokenizer()
    llm.runner = ModelRunner(model)
    llm.device = "cpu"
    llm.eos_token_ids = set()
    return llm


def _tiny_jetflow(model, block_size: int = 16) -> JetFlowEngine:
    eng = object.__new__(JetFlowEngine)
    eng.model = model
    eng.tokenizer = _StubTokenizer()
    eng.runner = ModelRunner(model)
    eng.device = "cpu"
    eng.dtype = torch.float32
    eng.block_size = block_size
    eng.eos_token_ids = set()
    return eng


PROMPT = torch.tensor([[3, 14, 15, 92, 65, 35, 89, 7]])
SP = SamplingParams(0.0, 24)


def _run_ref(llm, drafter, *, seed: int, prompt=PROMPT, sp=SP, tree_block_size=4):
    torch.manual_seed(seed)
    return llm.generate_tree(
        prompt,
        drafter,
        block_size=tree_block_size,
        tree_width=2,
        budget=15,
        sampling_params=sp,
        return_stats=True,
    )


def _run_device(engine, drafter, *, seed: int, prompt=PROMPT, sp=SP, tree_block_size=4):
    torch.manual_seed(seed)
    return engine.generate_tree(
        prompt,
        drafter,
        block_size=tree_block_size,
        tree_width=2,
        budget=15,
        sampling_params=sp,
        return_stats=True,
    )


def _assert_same_tokens_and_accept_lengths(ref, got):
    assert got["token_ids"] == ref["token_ids"]
    assert got["accept_lengths"] == ref["accept_lengths"]


def test_device_commit_matches_current_reference_across_random_seeds():
    for seed in (1, 7, 13):
        model = _tiny_model(0)
        ref = _run_ref(_tiny_llm(model), RandomTreeDrafter(model.config.vocab_size), seed=seed)
        got = _run_device(
            _tiny_jetflow(model),
            RandomTreeDrafter(model.config.vocab_size),
            seed=seed,
        )
        _assert_same_tokens_and_accept_lengths(ref, got)


def test_device_commit_matches_current_reference_with_echo_drafter():
    model = _tiny_model(0)
    ref = _run_ref(_tiny_llm(model), TargetEchoTreeDrafter(model), seed=1)
    got = _run_device(_tiny_jetflow(model), TargetEchoTreeDrafter(model), seed=1)
    _assert_same_tokens_and_accept_lengths(ref, got)


def _run_long_echo(engine, model, *, seed: int = 1):
    torch.manual_seed(seed)
    return engine.generate_tree(
        PROMPT,
        TargetEchoTreeDrafter(model),
        block_size=4,
        tree_width=1,
        budget=4,
        sampling_params=SamplingParams(0.0, 80),
        return_stats=True,
    )


def _run_long_echo_ref(llm, model, *, seed: int = 1):
    torch.manual_seed(seed)
    return llm.generate_tree(
        PROMPT,
        TargetEchoTreeDrafter(model),
        block_size=4,
        tree_width=1,
        budget=4,
        sampling_params=SamplingParams(0.0, 80),
        return_stats=True,
    )


def _first_unique_midblock_token(tokens, tree_block_size: int):
    seen = set()
    for idx, token in enumerate(tokens):
        if token in seen:
            continue
        seen.add(token)
        if idx == 0:
            continue
        round_idx = ((idx - 1) // tree_block_size) + 1
        offset = (idx - 1) % tree_block_size
        if round_idx >= 3 and 0 < offset < tree_block_size - 1:
            return token
    raise AssertionError("no unique mid-block EOS candidate")


def test_device_eos_stops_at_same_midround_token_as_current_reference():
    model = _tiny_model(0)
    full = _run_long_echo(_tiny_jetflow(model), model)
    ref_full = _run_long_echo_ref(_tiny_llm(model), model)
    assert full["token_ids"] == ref_full["token_ids"]

    eos_token = _first_unique_midblock_token(full["token_ids"], tree_block_size=4)
    ref = _tiny_llm(model)
    got = _tiny_jetflow(model)
    ref.eos_token_ids = {eos_token}
    got.eos_token_ids = {eos_token}

    ref_eos = _run_long_echo_ref(ref, model)
    got_eos = _run_long_echo(got, model)

    _assert_same_tokens_and_accept_lengths(ref_eos, got_eos)
    assert got_eos["token_ids"][-1] == eos_token
