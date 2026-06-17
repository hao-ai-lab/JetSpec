"""Per-prompt fixed-overhead vs steady-round split for nano tree decode.

THROWAWAY scratch. Times each generate_tree call separately and regresses
time_i = a + b * rounds_i over the prompt set: the intercept `a` is the
per-prompt fixed cost (prefill + pool alloc + CUDA-graph recapture + buffer
setup), the slope `b` is the steady per-round time. Decides whether the
663->739 gap should be attacked in the round loop (slope) or in cross-prompt
reuse (intercept).

    JETFLOW_BACKEND=triton_paged_tree_cudagraph_nogather HF_DATASETS_CACHE=... \
      CUDA_VISIBLE_DEVICES=4 PYTHONPATH=. PTD_DRAFT_HEAD=... \
      python bench/prompt_overhead.py --samples 64 --budget 127 --drafter graphed
"""
import argparse
import os
import time

import torch

from ptd.engine.llm import SamplingParams
from ptd.jetflow.engine import JetFlowEngine
from ptd.models.draft_head import load_draft_head
from ptd.draft_head_drafter import DraftHeadTreeDrafter

GSM8K_FMT = ("{question}\n"
             "Please reason step by step, and put your final answer within \\boxed{{}}.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=64)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--budget", type=int, default=127)
    ap.add_argument("--tree-width", type=int, default=7)
    ap.add_argument("--algo", default="crossproduct")
    ap.add_argument("--drafter", default="graphed", choices=["eager", "compiled", "graphed"])
    ap.add_argument("--session", action="store_true",
                    help="W11: reuse the tree session (pool + captured graphs) across prompts")
    ap.add_argument("--prompt-set", default="gsm8k",
                    choices=["gsm8k", "math500", "humaneval", "aime"])
    args = ap.parse_args()

    backend = os.environ.get("JETFLOW_BACKEND", "triton_paged_tree_cudagraph")
    head_id = os.environ["PTD_DRAFT_HEAD"]
    eng = JetFlowEngine("Qwen/Qwen3-8B", device="cuda", dtype=torch.bfloat16,
                     attn_backend=backend, block_size=16)
    head = load_draft_head(head_id)
    tli, bs = head.target_layer_ids, head.block_size
    if args.drafter == "compiled":
        from ptd.draft_head_drafter import CompiledDraftHead
        drafter = CompiledDraftHead(head, target=eng.model, block_size=bs,
                                    target_layer_ids=tli, draft_shift=False)
    elif args.drafter == "graphed":
        from ptd.draft_head_drafter import GraphedDraftHead
        drafter = GraphedDraftHead(head, target=eng.model, block_size=bs,
                                   target_layer_ids=tli, draft_shift=False)
    else:
        drafter = DraftHeadTreeDrafter(head, target=eng.model, block_size=bs,
                                       target_layer_ids=tli, draft_shift=False)

    from bench.tree_diag import build_prompts
    prompts = build_prompts(eng.tokenizer, args.samples, prompt_set=args.prompt_set)

    sp = SamplingParams(0.0, args.max_tokens)
    tkw = dict(block_size=bs, tree_width=args.tree_width, budget=args.budget,
               algo=args.algo, target_layer_ids=tli, return_stats=True)
    if args.session:
        tkw["session"] = True
        # capacity = the longest prompt in the set (session guard is loud, not growing)
        max_len = max(eng.tokenizer(p, return_tensors="pt").input_ids.shape[1] for p in prompts)
        tkw["session_prompt_capacity"] = ((max_len + 255) // 256) * 256

    eng.generate_tree(prompts[0], drafter, sampling_params=sp, **tkw)
    eng.generate_tree(prompts[0], drafter, sampling_params=sp, **tkw)
    torch.cuda.synchronize()

    times, rounds_l, toks = [], [], []
    for p in prompts:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        o = eng.generate_tree(p, drafter, sampling_params=sp, **tkw)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
        rounds_l.append(o["rounds"])
        toks.append(len(o["token_ids"]))

    n = len(times)
    total_t, total_r, total_tok = sum(times), sum(rounds_l), sum(toks)
    # least-squares time = a + b*rounds
    mr = total_r / n
    mt = total_t / n
    sxx = sum((r - mr) ** 2 for r in rounds_l)
    sxy = sum((r - mr) * (t - mt) for r, t in zip(rounds_l, times))
    b = sxy / sxx if sxx else float("nan")
    a = mt - b * mr
    print(f"samples={n}  total={total_t:.3f}s  rounds={total_r}  tokens={total_tok}")
    print(f"TPS (this harness) = {total_tok/total_t:.1f}")
    print(f"regression: time_i = a + b*rounds_i")
    print(f"  intercept a (per-prompt fixed) = {a*1e3:.1f} ms")
    print(f"  slope b (steady round)         = {b*1e3:.2f} ms/round")
    print(f"  fixed total = {a*n:.2f}s = {100*a*n/total_t:.1f}% of leg")
    print(f"  TPS if fixed cost were 0       = {total_tok/(total_t - a*n):.1f}")
    # distribution tails to sanity-check the regression
    pairs = sorted(zip(rounds_l, times))
    lo = pairs[:3]
    hi = pairs[-3:]
    print(f"  3 fewest-round prompts (rounds, s): {[(r, round(t, 3)) for r, t in lo]}")
    print(f"  3 most-round prompts  (rounds, s): {[(r, round(t, 3)) for r, t in hi]}")


if __name__ == "__main__":
    main()
