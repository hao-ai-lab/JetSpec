"""Tree spec-decode gate: tree speculative decoding is lossless — output equals plain
greedy for any tree drafter — and the echo tree drafter (whose top-1 path is the
greedy chain) accepts the full depth. Needs CUDA + Qwen3-8B; run on b200.

    JETFLOW_TEST_MODEL=Qwen/Qwen3-8B pytest tests/tree/test_tree_lossless.py -x
"""
import os

import pytest
import torch

from jetflow.core.llm import LLM, SamplingParams
from jetflow.draft import RandomTreeDrafter, TargetEchoTreeDrafter

MODEL = os.environ.get("JETFLOW_TEST_MODEL")

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and MODEL),
    reason="needs CUDA + a real checkpoint; set JETFLOW_TEST_MODEL to run",
)

PROMPTS = [
    "The capital of France is",
    "Solve: what is 17 times 23? Answer:",
    "def fibonacci(n):",
]


@pytest.fixture(scope="module")
def llm():
    return LLM(MODEL)


@pytest.mark.parametrize("prompt", PROMPTS)
def test_tree_lossless_random(llm, prompt):
    """Any tree → verify accepts only the greedy-agreeing path → output == greedy."""
    greedy = llm.generate(prompt, SamplingParams(0.0, 40))["token_ids"]
    out = llm.generate_tree(
        prompt, RandomTreeDrafter(llm.model.config.vocab_size),
        block_size=4, tree_width=2, budget=15, sampling_params=SamplingParams(0.0, 40),
    )["token_ids"]
    n = min(len(greedy), len(out))
    assert out[:n] == greedy[:n], f"tree (random) diverged from greedy on {prompt!r}"


@pytest.mark.parametrize("prompt", PROMPTS)
def test_tree_lossless_echo_and_accepts(llm, prompt):
    """Echo tree's top-1 path is the greedy chain → lossless AND full-depth accept."""
    greedy = llm.generate(prompt, SamplingParams(0.0, 40))["token_ids"]
    out = llm.generate_tree(
        prompt, TargetEchoTreeDrafter(llm.model),
        block_size=4, tree_width=2, budget=15, sampling_params=SamplingParams(0.0, 40),
    )
    n = min(len(greedy), len(out["token_ids"]))
    assert out["token_ids"][:n] == greedy[:n], f"tree (echo) diverged from greedy on {prompt!r}"
    assert out["tpf"] >= 3.5, f"echo tree should accept full depth (tpf≈4), got {out['tpf']:.2f}"
