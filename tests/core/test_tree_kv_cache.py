"""Tree KV-cache verify gate: `generate_tree` uses persistent-cache tree verify
by default and must produce the same tokens as plain greedy. Losslessness is
preserved by the select_kv_cache gather that keeps only the accepted root-to-leaf
path's KV.

Runs on CPU with a tiny randomly-initialized fp32 Qwen3 (no network, no GPU): in
fp32 the cached prefix and gathered accepted path are deterministic, so this gates
the gather / mask / position arithmetic directly.
"""
import os

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from jetflow.core.llm import LLM, SamplingParams
from jetflow.core.model_runner import ModelRunner
from jetflow.draft import RandomTreeDrafter, TargetEchoTreeDrafter


class _StubTokenizer:
    """Only `.decode` is exercised when prompts are passed as input_ids tensors."""

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(str(int(t)) for t in ids)


def _tiny_llm(seed: int = 0) -> LLM:
    """A tiny fp32 Qwen3 wired into an LLM without touching the network."""
    torch.manual_seed(seed)
    cfg = Qwen3Config(
        vocab_size=128, hidden_size=64, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        head_dim=16, max_position_embeddings=256, tie_word_embeddings=False,
    )
    model = Qwen3ForCausalLM(cfg).eval().to(torch.float32)
    llm = object.__new__(LLM)            # bypass load_target (no download)
    llm.model = model
    llm.tokenizer = _StubTokenizer()
    llm.runner = ModelRunner(model)
    llm.device = "cpu"
    llm.eos_token_ids = set()            # no EOS -> deterministic length
    return llm


PROMPT = torch.tensor([[3, 14, 15, 92, 65, 35, 89, 7]])   # arbitrary fixed input_ids
SP = SamplingParams(0.0, 24)


def _greedy(llm):
    return llm.generate(PROMPT, SP)["token_ids"]


def _tree(llm, drafter, *, seed=1):
    # seed before each call so the random drafter builds identical trees.
    torch.manual_seed(seed)
    return llm.generate_tree(
        PROMPT, drafter, block_size=4, tree_width=2, budget=15,
        sampling_params=SP,
    )


def test_kv_cache_tree_lossless_random():
    """Random drafter (accepts ~0/round) exercises keep-root-only gather."""
    llm = _tiny_llm()
    greedy = _greedy(llm)
    cached = _tree(llm, RandomTreeDrafter(128))["token_ids"]
    n = min(len(greedy), len(cached))
    assert cached[:n] == greedy[:n], "kv-cache tree diverged from greedy (KV-cache gather bug)"


def test_kv_cache_tree_lossless_echo_and_accepts():
    """Echo tree's top-1 path is the greedy chain -> full-depth accept, exercising
    the gather's deep non-contiguous keep set (acc > 0)."""
    llm = _tiny_llm()
    greedy = _greedy(llm)
    cached = _tree(llm, TargetEchoTreeDrafter(llm.model))
    n = min(len(greedy), len(cached["token_ids"]))
    assert cached["token_ids"][:n] == greedy[:n], "kv-cache (echo) diverged from greedy"
    assert cached["tpf"] >= 2.0, f"echo should accept multiple tokens/round, got tpf={cached['tpf']:.2f}"


def test_kv_cache_tree_stats_shape():
    """return_stats exposes per-round accept lengths / tree sizes on the cached path,
    and sum(accept_lengths) accounts for every committed token after the first."""
    llm = _tiny_llm()
    out = _tree(llm, RandomTreeDrafter(128))
    full = llm.generate_tree(
        PROMPT, RandomTreeDrafter(128), block_size=4, tree_width=2, budget=15,
        return_stats=True, sampling_params=SP,
    )
    assert len(full["accept_lengths"]) == full["rounds"]
    assert len(full["tree_sizes"]) == full["rounds"]
    assert all(a >= 1 for a in full["accept_lengths"])   # each round commits >= the correction


# --- b200 gate: real model in bf16 (skips locally; runs on b200) ---------------
_REAL_MODEL = os.environ.get("JETFLOW_TEST_MODEL")


@pytest.mark.skipif(
    not (torch.cuda.is_available() and _REAL_MODEL),
    reason="bf16 lossless gate needs CUDA + a real checkpoint; run on b200 with "
           "JETFLOW_TEST_MODEL=Qwen/Qwen3-8B",
)
def test_kv_cache_tree_real_model_bf16_lossless_and_accepts():
    """On b200/bf16 the cached tree path stays lossless vs greedy and accepts the
    full depth with the echo drafter — exercising the deep gather on the real model.
    (bf16 may flip a borderline argmax after ~tens of tokens, the same as
    generate_chain; the 40-token horizon here stays exact in practice.)"""
    llm = LLM(_REAL_MODEL)
    prompt = "Solve: what is 17 times 23? Answer:"
    greedy = llm.generate(prompt, SamplingParams(0.0, 40))["token_ids"]
    out = llm.generate_tree(
        prompt, TargetEchoTreeDrafter(llm.model),
        block_size=4, tree_width=2, budget=15,
        sampling_params=SamplingParams(0.0, 40),
    )
    n = min(len(greedy), len(out["token_ids"]))
    assert out["token_ids"][:n] == greedy[:n], "kv-cache tree (bf16, echo) diverged from greedy"
    assert out["tpf"] >= 3.5, f"echo tree should accept full depth, got tpf={out['tpf']:.2f}"
