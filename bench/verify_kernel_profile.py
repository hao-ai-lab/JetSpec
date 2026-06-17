"""Per-kernel CUDA attribution of the verify forward (T010: 6.9ms vs fork ~4.2).

THROWAWAY scratch. Runs a short tree decode on the COMPILED (non-graph)
backend so torch.profiler can attribute kernels (graph replays are opaque),
plus a plain AR leg on the same stack. Prints top kernels by CUDA time with
call counts, the GEMM share, and the per-round totals. The fusion gap =
non-GEMM, non-attention kernel time; GEMM efficiency = GEMM time vs the
weight-streaming floor (~2.3ms @ 8TB/s for 16GB bf16 weights).

    JETFLOW_BACKEND=triton_paged_tree_compiled_nogather HF_DATASETS_CACHE=... \
      CUDA_VISIBLE_DEVICES=4 PYTHONPATH=. JETFLOW_DRAFT_HEAD=... \
      python bench/verify_kernel_profile.py --max-tokens 256 --budget 127
"""
import argparse
import os

import torch
from torch.profiler import ProfilerActivity, profile

from jetflow.core.llm import SamplingParams
from jetflow.inference_engine.engine import JetFlowEngine
from jetflow.models.draft_head import load_draft_head
from jetflow.draft_head_drafter import DraftHeadTreeDrafter

GSM8K_FMT = ("{question}\n"
             "Please reason step by step, and put your final answer within \\boxed{{}}.")


def _report(prof, label, rounds):
    ka = prof.key_averages()
    rows = [(e.key, e.device_time_total, e.count) for e in ka if e.device_time_total > 0]
    rows.sort(key=lambda r: -r[1])
    total = sum(r[1] for r in rows)
    gemm = sum(r[1] for r in rows if any(s in r[0].lower() for s in
                                         ("gemm", "cutlass", "nvjet", "matmul", "mm_")))
    attn = sum(r[1] for r in rows if "paged_tree" in r[0].lower() or "attention" in r[0].lower())
    print(f"\n### {label}: total CUDA {total/1e3:.2f}ms over {rounds} rounds "
          f"= {total/1e3/max(rounds,1):.3f} ms/round")
    print(f"    GEMM {gemm/1e3:.2f}ms ({100*gemm/total:.0f}%)  "
          f"attn {attn/1e3:.2f}ms ({100*attn/total:.0f}%)  "
          f"other {((total-gemm-attn))/1e3:.2f}ms ({100*(total-gemm-attn)/total:.0f}%)")
    print(f"{'kernel':<72}{'ms':>9}{'calls':>8}")
    print("-" * 89)
    for k, t, c in rows[:25]:
        print(f"{k[:72]:<72}{t/1e3:>9.2f}{c:>8}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--budget", type=int, default=127)
    ap.add_argument("--tree-width", type=int, default=7)
    args = ap.parse_args()

    backend = os.environ.get("JETFLOW_BACKEND", "triton_paged_tree_compiled_nogather")
    head_id = os.environ["JETFLOW_DRAFT_HEAD"]
    eng = JetFlowEngine("Qwen/Qwen3-8B", device="cuda", dtype=torch.bfloat16,
                     attn_backend=backend, block_size=16)
    head = load_draft_head(head_id)
    tli, bs = head.target_layer_ids, head.block_size
    drafter = DraftHeadTreeDrafter(head, target=eng.model, block_size=bs,
                                   target_layer_ids=tli, draft_shift=False)

    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="test")
    prompt = eng.tokenizer.apply_chat_template(
        [{"role": "user", "content": GSM8K_FMT.format(question=ds[0]["question"])}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)

    sp = SamplingParams(0.0, args.max_tokens)
    tkw = dict(block_size=bs, tree_width=args.tree_width, budget=args.budget,
               algo="crossproduct", target_layer_ids=tli, return_stats=True)

    # warmup: compile + autotune absorbed before profiling
    eng.generate_tree(prompt, drafter, sampling_params=sp, **tkw)
    eng.generate_tree(prompt, drafter, sampling_params=sp, **tkw)
    torch.cuda.synchronize()

    with profile(activities=[ProfilerActivity.CUDA], record_shapes=False) as prof:
        o = eng.generate_tree(prompt, drafter, sampling_params=sp, **tkw)
        torch.cuda.synchronize()
    _report(prof, f"TREE decode (verify+drafter, compiled, budget {args.budget})", o["rounds"])

    eng.generate(prompt, sp)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA], record_shapes=False) as prof2:
        out = eng.generate(prompt, sp)
        torch.cuda.synchronize()
    n_ar = len(out["token_ids"]) if isinstance(out, dict) else args.max_tokens
    _report(prof2, "AR decode (same stack, N=1)", n_ar)


if __name__ == "__main__":
    main()
