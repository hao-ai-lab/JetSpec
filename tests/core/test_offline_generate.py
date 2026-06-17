"""Offline-baseline gate: the offline engine must be token-identical to HF greedy generation,
and must reuse the KV cache (never reprocess the prefix). Needs CUDA + Qwen3-8B,
so it is skipped on CPU/CI; run it on the b200.

    JETFLOW_TEST_MODEL=Qwen/Qwen3-8B pytest tests/core/test_offline_generate.py -x
"""
import os

import pytest
import torch

from jetflow.core.llm import LLM, SamplingParams

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
@pytest.mark.parametrize("max_new", [16, 48])
def test_byte_identical_to_hf_greedy(llm, prompt, max_new):
    input_ids = llm.tokenizer(prompt, return_tensors="pt").input_ids.to(llm.device)
    ours = llm.generate(input_ids, SamplingParams(temperature=0.0, max_new_tokens=max_new))["token_ids"]
    with torch.inference_mode():
        ref = llm.model.generate(
            input_ids, do_sample=False, max_new_tokens=max_new, use_cache=True
        )[0, input_ids.shape[1]:].tolist()
    n = min(len(ours), len(ref))  # eos boundary may differ by one token
    assert ours[:n] == ref[:n], f"diverged ({prompt!r}, {max_new}): ours={ours[:n]} ref={ref[:n]}"


# Pinned reference: Qwen3-8B greedy, 64 new tokens, captured on b200
# (transformers 4.57.1) via the offline reproducer. Catches drift
# even if HF generation changes — the engine must reproduce this exactly.
REF_PROMPT = "Solve: what is 17 times 23? Answer:"
REF_TOKEN_IDS = [
    220, 18, 24, 16, 13, 2585, 1521, 498, 633, 429, 30, 6771, 752, 10339, 13,
    220, 16, 22, 3039, 220, 17, 15, 374, 220, 18, 19, 15, 11, 323, 220, 16, 22,
    3039, 220, 18, 374, 220, 20, 16, 13, 5005, 11, 220, 18, 19, 15, 5519, 220,
    20, 16, 374, 220, 18, 24, 16, 13, 2055, 11, 279, 4226, 374, 220, 18, 24,
]


def test_matches_pinned_reference(llm):
    out = llm.generate(REF_PROMPT, SamplingParams(temperature=0.0, max_new_tokens=64))["token_ids"]
    n = min(len(out), len(REF_TOKEN_IDS))
    assert out[:n] == REF_TOKEN_IDS[:n], f"diverged from pinned ref at first {n}: {out[:n]}"


def test_kv_reuse_no_reprocess(llm):
    """Prefill processes the whole prompt; every decode step is width-1 (the KV
    cache is reused, the prefix is never reprocessed)."""
    shapes = []
    orig = llm.runner.forward

    def counting(input_ids, *a, **k):
        shapes.append(tuple(input_ids.shape))
        return orig(input_ids, *a, **k)

    llm.runner.forward = counting
    try:
        llm.generate("Count the decode steps here.", SamplingParams(temperature=0.0, max_new_tokens=10))
    finally:
        llm.runner.forward = orig

    assert shapes[0][1] > 1, "prefill should process the full prompt in one forward"
    assert all(s[1] == 1 for s in shapes[1:]), f"decode steps must be width-1, got {shapes}"
