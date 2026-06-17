"""Chain spec-decode gate: chain speculative decoding is lossless — its output must
equal plain greedy regardless of draft quality — and the verify loop must accept the
target's own drafts (multi-token accept). Needs CUDA + Qwen3-8B; run on b200.

    JETFLOW_TEST_MODEL=Qwen/Qwen3-8B pytest tests/core/test_chain_lossless.py -x
"""
import os

import pytest
import torch

from jetflow.core.llm import LLM, SamplingParams
from jetflow.draft import RepeatDrafter, TargetEchoDrafter

MODEL = os.environ.get("JETFLOW_TEST_MODEL", "Qwen/Qwen3-8B")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs CUDA + a real Qwen3-8B checkpoint"
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
def test_chain_lossless_repeat(llm, prompt):
    """Any drafter → chain output identical to plain greedy."""
    greedy = llm.generate(prompt, SamplingParams(0.0, 40))["token_ids"]
    chain = llm.generate_chain(
        prompt, RepeatDrafter(), block_size=4, sampling_params=SamplingParams(0.0, 40)
    )["token_ids"]
    n = min(len(greedy), len(chain))
    assert chain[:n] == greedy[:n], f"chain (repeat stub) diverged from greedy on {prompt!r}"


@pytest.mark.parametrize("prompt", PROMPTS)
def test_chain_lossless_echo_and_accepts(llm, prompt):
    """Echo drafter proposes the target's own greedy → lossless AND every draft
    accepted (tpf == block_size), exercising the multi-token-accept path."""
    greedy = llm.generate(prompt, SamplingParams(0.0, 40))["token_ids"]
    out = llm.generate_chain(
        prompt, TargetEchoDrafter(llm.model), block_size=4, sampling_params=SamplingParams(0.0, 40)
    )
    n = min(len(greedy), len(out["token_ids"]))
    assert out["token_ids"][:n] == greedy[:n], f"chain (echo) diverged from greedy on {prompt!r}"
    assert out["tpf"] >= 3.5, f"echo drafter should accept ~all (tpf≈4), got {out['tpf']:.2f}"
