"""Every registered tree algorithm is lossless — output equals plain greedy —
at an ACTIVE (non-identity) knob setting that exercises its fanout/score logic,
not just the crossproduct-recovering default. Tree speculative decoding is
lossless by construction for any tree shape; this gate proves each algorithm's
build path produces a valid tree the verifier accepts correctly.

Needs CUDA + Qwen3-8B; run on b200:

    PTD_TEST_MODEL=Qwen/Qwen3-8B pytest tests/test_tree_algos_lossless.py -x
"""
import os

import pytest
import torch

from ptd.engine.llm import LLM, SamplingParams
from ptd.draft import RandomTreeDrafter
from ptd.tree import list_algorithms

MODEL = os.environ.get("PTD_TEST_MODEL", "Qwen/Qwen3-8B")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs CUDA + a real Qwen3-8B checkpoint"
)

# An ACTIVE knob setting per algorithm — chosen so the fanout/score logic
# actually fires (not the crossproduct-identity default). Losslessness must
# hold regardless of the knob, so any in-range value works; these just
# guarantee the non-trivial branch is covered.
ACTIVE_KWARGS = {
    "crossproduct": {},
    "top2gap_fanout": {"beta": 2.0, "g_0": 1.0},
    "task_router": {},          # routes via logit-fingerprint fallback (no prompt_info)
    "reasoning_router": {},     # routes via logit-fingerprint fallback
    "class_histogram": {},      # per-class template stand-in
}

PROMPT = "Solve: what is 17 times 23? Answer:"


def test_active_kwargs_cover_every_registered_algo():
    """The test config must be exhaustive — a newly-registered algorithm with
    no active-knob entry should fail here rather than go silently untested."""
    assert set(list_algorithms()) == set(ACTIVE_KWARGS), (
        f"registry {sorted(list_algorithms())} != "
        f"test config {sorted(ACTIVE_KWARGS)}"
    )


@pytest.fixture(scope="module")
def llm():
    return LLM(MODEL)


@pytest.mark.parametrize("algo", sorted(ACTIVE_KWARGS))
def test_tree_algo_lossless(llm, algo):
    """Any tree (random drafter) → verify accepts only the greedy-agreeing path
    → output == greedy, for every algorithm at an active knob setting."""
    greedy = llm.generate(PROMPT, SamplingParams(0.0, 40))["token_ids"]
    out = llm.generate_tree(
        PROMPT, RandomTreeDrafter(llm.model.config.vocab_size),
        block_size=4, tree_width=2, budget=15,
        algo=algo, algo_kwargs=ACTIVE_KWARGS[algo],
        sampling_params=SamplingParams(0.0, 40),
    )["token_ids"]
    n = min(len(greedy), len(out))
    assert out[:n] == greedy[:n], f"{algo} diverged from greedy on {PROMPT!r}"
