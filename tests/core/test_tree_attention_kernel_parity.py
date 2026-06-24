import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from jetspec.draft import RandomTreeDrafter, TargetEchoTreeDrafter
from jetspec.core.llm import LLM, SamplingParams
from jetspec.core.model_runner import ModelRunner

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="triton tree-attention parity needs CUDA"
)


class _StubTokenizer:
    def decode(self, ids, skip_special_tokens=True):
        return " ".join(str(int(t)) for t in ids)


def _tiny_llm(seed: int = 0) -> LLM:
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
    model = Qwen3ForCausalLM(cfg).eval().to(torch.float32).to("cuda")
    model.config._attn_implementation = "sdpa"
    llm = object.__new__(LLM)
    llm.model = model
    llm.tokenizer = _StubTokenizer()
    llm.runner = ModelRunner(model)
    llm.device = "cuda"
    llm.eos_token_ids = set()
    return llm


PROMPT = torch.tensor([[3, 14, 15, 92, 65, 35, 89, 7]], device="cuda")
SP = SamplingParams(0.0, 24)


def _run(llm, drafter, *, tree_attn: str, seed: int = 1):
    torch.manual_seed(seed)
    reset = getattr(drafter, "reset_cache", None)
    if reset is not None:
        reset()
    return llm.generate_tree(
        PROMPT,
        drafter,
        block_size=4,
        tree_width=2,
        budget=15,
        tree_attn=tree_attn,
        sampling_params=SP,
    )["token_ids"]


@pytest.mark.parametrize("seed", [1, 7])
def test_triton_tree_attention_matches_sdpa_random(seed):
    llm = _tiny_llm(0)
    sdpa = _run(llm, RandomTreeDrafter(128), tree_attn="sdpa", seed=seed)
    triton = _run(llm, RandomTreeDrafter(128), tree_attn="triton", seed=seed)
    assert triton == sdpa


def test_triton_tree_attention_matches_sdpa_echo():
    llm = _tiny_llm(0)
    sdpa = _run(llm, TargetEchoTreeDrafter(llm.model), tree_attn="sdpa")
    triton = _run(llm, TargetEchoTreeDrafter(llm.model), tree_attn="triton")
    assert triton == sdpa
