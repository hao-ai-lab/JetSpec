"""Tokens-per-forward sweep across the bundled tree algorithms.

Runs every registered tree algorithm through the real DraftHead engine at a
range of node budgets and reports tokens-per-forward (TPF) plus the exact-prefix
agreement with recompute-greedy. The point is to show each algorithm's
budget-dependent niche — e.g. the top-2-gap caps win at low budget by collapsing
to a near-chain, while accum_logp catches up once the budget is large enough
to expand everything.

Needs CUDA + a real Qwen3-8B target + a trained DFlash head; run on b200:

    CUDA_VISIBLE_DEVICES=0 JETFLOW_TEST_MODEL=Qwen/Qwen3-8B \
      JETFLOW_DRAFT_HEAD="Snyhlxde/jetflow-qwen3-8b-distill-epoch6-3e-4-no-gamma" \
      HF_HOME=/path/to/hf_cache \
      python examples/tree/tree_algo_sweep.py

Env knobs: JETFLOW_SWEEP_BUDGETS (comma list, default "15,63,127"),
JETFLOW_SWEEP_WIDTH (tree_width, default 7), JETFLOW_SWEEP_MAXNEW (default 128).
"""
import os

import torch
from transformers import DynamicCache

from jetflow.core.llm import LLM, SamplingParams
from jetflow.models.draft_head import load_draft_head
from jetflow.draft_head_adapter import DraftHeadTreeDrafter
from jetflow.tree import list_algorithms

MODEL = os.environ.get("JETFLOW_TEST_MODEL", "Qwen/Qwen3-8B")
DRAFT_HEAD = os.environ.get("JETFLOW_DRAFT_HEAD")
BUDGETS = [int(b) for b in os.environ.get("JETFLOW_SWEEP_BUDGETS", "15,63,127").split(",")]
WIDTH = int(os.environ.get("JETFLOW_SWEEP_WIDTH", "7"))
MAX_NEW = int(os.environ.get("JETFLOW_SWEEP_MAXNEW", "128"))

# The drafter conditions on the target's hidden states in the *chat-formatted*
# distribution it was trained on; a raw, untemplated prompt gives off-distribution
# conditioning and halves accept-len. Always chat-template (see main()).
QUESTION = ("Natalia sold clips to 48 of her friends in April, and then she sold "
            "half as many clips in May. How many clips did Natalia sell altogether "
            "in April and May?")

# Active knob per algorithm — the setting that exercises its niche (not the
# accum_logp-identity default). Losslessness holds for any knob; these make
# the TPF comparison meaningful.
ALGO_KWARGS = {
    "accum_logp": {},
    "top2gap_fanout": {"beta": 2.0, "g_0": 1.0},
    "task_router": {},
    "reasoning_router": {},
    "class_histogram": {},
}


def _recompute_greedy(llm, prompt, n):
    ids = llm.tokenizer(prompt, return_tensors="pt").input_ids.to(llm.device)
    out, cur = [], ids
    for _ in range(n):
        pos = torch.arange(cur.shape[1], device=llm.device).unsqueeze(0)
        logits, _, _ = llm.runner.forward(cur, DynamicCache(), pos)
        t = int(logits[0, -1].argmax())
        out.append(t)
        if t in llm.eos_token_ids:
            break
        cur = torch.cat([cur, torch.tensor([[t]], device=llm.device)], dim=1)
    return out


def _exact_prefix(a, b):
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


def main():
    if not DRAFT_HEAD:
        raise SystemExit("set JETFLOW_DRAFT_HEAD to a trained DFlash head checkpoint")
    llm = LLM(MODEL)
    head = load_draft_head(DRAFT_HEAD)
    tli = head.target_layer_ids
    prompt = llm.tokenizer.apply_chat_template(
        [{"role": "user", "content": QUESTION}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    ref = _recompute_greedy(llm, prompt, MAX_NEW)
    sp = SamplingParams(0.0, MAX_NEW)
    tree_drafter = DraftHeadTreeDrafter(
        head, target=llm.model, block_size=head.block_size,
        target_layer_ids=tli, draft_shift=False,
    )

    algos = sorted(list_algorithms())
    missing = [a for a in algos if a not in ALGO_KWARGS]
    if missing:
        raise SystemExit(f"no sweep knobs configured for: {missing}")

    print(f"\nmodel={MODEL} head={DRAFT_HEAD}")
    print(f"block_size={head.block_size} tree_width={WIDTH} max_new={MAX_NEW} "
          f"ref_len={len(ref)}\n")
    header = "algorithm".ljust(24) + "".join(f"B={b:<10}" for b in BUDGETS)
    print(header)
    print("-" * len(header))
    for algo in algos:
        cells = []
        for b in BUDGETS:
            out = llm.generate_tree(
                prompt, tree_drafter, block_size=head.block_size, tree_width=WIDTH,
                budget=b, algo=algo, algo_kwargs=ALGO_KWARGS[algo],
                target_layer_ids=tli, sampling_params=sp,
            )
            exact = _exact_prefix(ref, out["token_ids"])
            cells.append(f"{out['tpf']:.2f} (p{exact})")
        print(algo.ljust(24) + "".join(c.ljust(12) for c in cells))
    print("\nTPF = tokens accepted per target forward; pN = exact-prefix length vs "
          "recompute-greedy (bf16 SDPA flips after the prefix; lossless by construction).")


if __name__ == "__main__":
    main()
