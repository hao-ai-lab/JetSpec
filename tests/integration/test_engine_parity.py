"""Engine parity / regression guard: on aligned gsm8k samples (identical
shuffle+formatting to the reference `causal_parallel_drafting/benchmark.py`),
the jetflow tree engine must reproduce the reference's acceptance behavior. This is
the guard that catches drafting/verify regressions — e.g. an off-distribution
prompt or a broken target_hidden update collapses accept_len from ~9 to ~3.

Recorded reference (Snyhlxde/...-epoch6-no-gamma, accum_logp, width 7,
budget 255, sdpa, single-pass, gsm8k shuffle seed=0): accept_len 9.48,
per-position d0/d1/d2/d3 = 1.00/0.96/0.92/0.84. Our engine matches within a few
percent (the residual is the bf16 recompute-vs-KV-cache tail divergence).

    CUDA_VISIBLE_DEVICES=0 JETFLOW_TEST_MODEL=Qwen/Qwen3-8B \
      JETFLOW_DRAFT_HEAD=Snyhlxde/jetflow-qwen3-8b-distill-epoch6-3e-4-no-gamma \
      HF_HOME=/path/to/hf_cache HF_DATASETS_CACHE=/path/to/hf_cache/datasets \
      pytest tests/integration/test_engine_parity.py -x -s
"""
import os

import pytest
import torch

from jetflow.core.llm import LLM, SamplingParams
from jetflow.models.draft_head import load_draft_head
from jetflow.draft_head_adapter import DraftHeadTreeDrafter

MODEL = os.environ.get("JETFLOW_TEST_MODEL", "Qwen/Qwen3-8B")
DRAFT_HEAD = os.environ.get("JETFLOW_DRAFT_HEAD")
N_SAMPLES, WIDTH, BUDGET, MAX_NEW = 3, 7, 255, 192
# Regression floor: a healthy engine on this config gets ~9; raw-prompt or broken
# target_hidden drops it to ~3. Floor at 6 catches that class without flaking on
# sample/numeric noise. Reference is 9.48; we observe ~9.1.
ACCEPT_LEN_FLOOR = 6.0

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not DRAFT_HEAD,
    reason="needs CUDA + Qwen3-8B target + JETFLOW_DRAFT_HEAD checkpoint",
)


def _gsm8k_prompts(tokenizer, n):
    try:
        from datasets import load_dataset
    except Exception:
        pytest.skip("datasets not installed")
    ds = load_dataset("openai/gsm8k", "main", split="test")
    ds = ds.shuffle(seed=0).select(range(n))  # identical selection to the reference
    fmt = "{q}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
    return [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": fmt.format(q=ds[i]["question"])}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
        for i in range(n)
    ]


def test_tree_engine_accept_len_parity():
    """accum_logp accept_len on aligned gsm8k stays near the reference (>= floor),
    and per-position acceptance is monotone non-increasing starting at d0=1.0."""
    llm = LLM(MODEL)
    head = load_draft_head(DRAFT_HEAD)
    tli = head.target_layer_ids
    drafter = DraftHeadTreeDrafter(head, target=llm.model, block_size=head.block_size,
                                   target_layer_ids=tli, draft_shift=False)
    prompts = _gsm8k_prompts(llm.tokenizer, N_SAMPLES)
    sp = SamplingParams(0.0, MAX_NEW)
    all_acc = []
    for p in prompts:
        out = llm.generate_tree(
            p, drafter, block_size=head.block_size, tree_width=WIDTH, budget=BUDGET,
            algo="accum_logp", target_layer_ids=tli, sampling_params=sp, return_stats=True)
        all_acc += out["accept_lengths"]
    tau = sum(all_acc) / len(all_acc)
    per_pos = [sum(1 for al in all_acc if al >= k + 2) / len(all_acc) for k in range(4)]
    print(f"\naccept_len={tau:.2f} per_pos={[round(r, 3) for r in per_pos]} "
          f"(ref 9.48, d=[1.0,0.96,0.92,0.84])")
    assert tau >= ACCEPT_LEN_FLOOR, (
        f"accept_len {tau:.2f} < floor {ACCEPT_LEN_FLOOR} — drafting/verify regression "
        f"(check prompt formatting + target_hidden threading)")
    assert per_pos[0] >= 0.99, f"d0 should be ~1.0 (anchor always accepts), got {per_pos[0]:.3f}"
    assert per_pos == sorted(per_pos, reverse=True), f"per-position must be monotone, got {per_pos}"
