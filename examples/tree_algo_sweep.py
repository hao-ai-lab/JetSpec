"""Tokens-per-forward sweep across the bundled tree algorithms.

Runs every registered tree algorithm through the real DraftHead engine at a
range of node budgets and reports tokens-per-forward (TPF) plus the exact-prefix
agreement with recompute-greedy. The point is to show each algorithm's
budget-dependent niche — e.g. the top-2-gap caps win at low budget by collapsing
to a near-chain, while crossproduct catches up once the budget is large enough
to expand everything.

Needs CUDA + a real Qwen3-8B target + a trained DFlash head; run on b200:

    CUDA_VISIBLE_DEVICES=5 PTD_TEST_MODEL=Qwen/Qwen3-8B \
      PTD_DRAFT_HEAD="Snyhlxde/ptd-qwen3-8b-distill-epoch6-3e-4-no-gamma" \
      HF_HOME=/raid/zhf004/hf_cache \
      python examples/tree_algo_sweep.py

Env knobs: PTD_SWEEP_BUDGETS (comma list, default "15,63,127"),
PTD_SWEEP_WIDTH (tree_width, default 7), PTD_SWEEP_MAXNEW (default 128).
"""
import os

import torch
from transformers import DynamicCache

from ptd.engine.llm import LLM, SamplingParams
from ptd.models.draft_head import load_draft_head
from ptd.draft_head_drafter import DraftHeadTreeDrafter
from ptd.tree import list_algorithms

MODEL = os.environ.get("PTD_TEST_MODEL", "Qwen/Qwen3-8B")
DRAFT_HEAD = os.environ.get("PTD_DRAFT_HEAD")
BUDGETS = [int(b) for b in os.environ.get("PTD_SWEEP_BUDGETS", "15,63,127").split(",")]
WIDTH = int(os.environ.get("PTD_SWEEP_WIDTH", "7"))
MAX_NEW = int(os.environ.get("PTD_SWEEP_MAXNEW", "128"))

PROMPT = "What is 127 times 384? Reason step by step, then give the final number."

# Active knob per algorithm — the setting that exercises its niche (not the
# crossproduct-identity default). Losslessness holds for any knob; these make
# the TPF comparison meaningful.
ALGO_KWARGS = {
    "crossproduct": {},
    "top2gap_fanout": {"beta": 2.0, "g_0": 1.0},
    "top2gap_budget_gated": {"beta": 2.0, "g_0": 1.0, "B_0": 16.0},
    "entropy_gate": {"tau_high": 1.5, "tau_low": 0.2},
    "entropy_soft": {"alpha": 1.0},
    "entropy_topk": {"tau_high": 1.5, "tau_low": 0.2},
    "prob_mass": {"m_0": 0.5},
    "entropy_score": {"lambda_": 0.5},
    "budget_blend": {"B_0": 16.0, "lambda_max": 2.0},
    "drift_brake": {"delta": 2.0},
    "rank_decay": {"gamma": 0.5},
}


def _recompute_greedy(llm, n):
    ids = llm.tokenizer(PROMPT, return_tensors="pt").input_ids.to(llm.device)
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
        raise SystemExit("set PTD_DRAFT_HEAD to a trained DFlash head checkpoint")
    llm = LLM(MODEL)
    head = load_draft_head(DRAFT_HEAD)
    tli = head.target_layer_ids
    ref = _recompute_greedy(llm, MAX_NEW)
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
                PROMPT, tree_drafter, block_size=head.block_size, tree_width=WIDTH,
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
