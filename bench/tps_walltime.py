"""Wall-clock TPS for the optimized JetFlow engine — AR baseline vs tree-spec.

Reports REAL wall-clock tokens/sec (time.perf_counter), NOT GPU-self-time —
i.e. what a user actually sees, including host/Python overhead. Complements
bench/identical_fork_compare.py (which reports decode_cuda_speedup =
GPU-self-time, drafter-excluded). The production configuration behind the
README Results table:

    JETFLOW_FUSE_GEMMS=1 JETFLOW_BACKEND=triton_paged_tree_cudagraph_nogather \
      PYTHONPATH=. PTD_DRAFT_HEAD=Snyhlxde/ptd-qwen3-8b-distill-epoch6-3e-4-no-gamma \
      python bench/tps_walltime.py --samples 64 --max-tokens 2048 --budget 127 \
        --drafter graphed --session --prompt-set gsm8k
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


def _load_prompt_bank(prompt_set: str, n: int) -> list:
    """Raw prompt texts, formatted exactly like the fork's dflash_profiling bank."""
    from datasets import load_dataset
    if prompt_set == "gsm8k":
        ds = load_dataset("openai/gsm8k", "main", split="test")
        return [GSM8K_FMT.format(question=ds[i]["question"]) for i in range(min(n, len(ds)))]
    if prompt_set == "math500":
        ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
        fmt = ("{problem}\n"
               "Please reason step by step, and put your final answer within \\boxed{{}}.")
        return [fmt.format(problem=ds[i]["problem"]) for i in range(min(n, len(ds)))]
    if prompt_set == "aime":
        ds = load_dataset("HuggingFaceH4/aime_2024", split="train")
        fmt = ("{problem}\n"
               "Please reason step by step, and put your final answer within \\boxed{{}}.")
        return [fmt.format(problem=ds[i]["problem"]) for i in range(min(n, len(ds)))]
    if prompt_set == "humaneval":
        ds = load_dataset("openai/openai_humaneval", split="test")
        fmt = ("Write a solution to the following problem and make sure that it "
               "passes the tests:\n```python\n{prompt}\n```")
        return [fmt.format(prompt=ds[i]["prompt"]) for i in range(min(n, len(ds)))]
    raise ValueError(f"unknown prompt set: {prompt_set}")


def _walltime(fn, prompts):
    """Return (total_tokens, wall_seconds, per_prompt_outputs)."""
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    outs = [fn(p) for p in prompts]
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    ntok = sum(len(o["token_ids"]) for o in outs)
    return ntok, dt, outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=5)
    ap.add_argument("--max-tokens", type=int, default=210)
    ap.add_argument("--budget", type=int, default=255)
    ap.add_argument("--tree-width", type=int, default=7)
    ap.add_argument("--algo", default="crossproduct")
    ap.add_argument("--drafter", default="eager", choices=["eager", "compiled", "graphed"],
                    help="L4: route propose_logits through the W2 wrappers (accept_len-gated)")
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

    bank = _load_prompt_bank(args.prompt_set, args.samples)
    prompts = [eng.tokenizer.apply_chat_template(
        [{"role": "user", "content": p}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
        for p in bank]

    sp = SamplingParams(0.0, args.max_tokens)
    tkw = dict(block_size=bs, tree_width=args.tree_width, budget=args.budget,
               algo=args.algo, target_layer_ids=tli, return_stats=True)
    if args.session:
        tkw["session"] = True
        # capacity = the longest prompt in the set (session guard is loud, not growing)
        max_len = max(eng.tokenizer(p, return_tensors="pt").input_ids.shape[1] for p in prompts)
        tkw["session_prompt_capacity"] = ((max_len + 255) // 256) * 256

    # warmup (excluded): warm at the timed sp, twice, so compile + Triton autotune
    # + the first cudagraph capture are absorbed before timing.
    eng.generate(prompts[0], sp)
    eng.generate(prompts[0], sp)
    eng.generate_tree(prompts[0], drafter, sampling_params=sp, **tkw)
    eng.generate_tree(prompts[0], drafter, sampling_params=sp, **tkw)
    torch.cuda.synchronize()

    ar_tok, ar_t, _ = _walltime(lambda p: eng.generate(p, sp), prompts)
    ar_tps = ar_tok / ar_t

    tree_tok, tree_t, touts = _walltime(
        lambda p: eng.generate_tree(p, drafter, sampling_params=sp, **tkw), prompts)
    spec_tps = tree_tok / tree_t
    rounds = sum(o["rounds"] for o in touts)
    acc_sum = sum(sum(o["accept_lengths"]) for o in touts)
    accept_len = acc_sum / rounds if rounds else 0.0

    print(f"\nbackend={backend}  head={head_id}  algo={args.algo}")
    print(f"samples={args.samples} budget={args.budget} width={args.tree_width} "
          f"max_tokens={args.max_tokens}")
    print(f"AR    : {ar_tok:5d} tok  {ar_t:7.3f}s  ->  {ar_tps:8.1f} tok/s   (1x baseline)")
    print(f"tree  : {tree_tok:5d} tok  {tree_t:7.3f}s  ->  {spec_tps:8.1f} tok/s   "
          f"accept_len={accept_len:.2f}")
    print(f"\nWALL-CLOCK spec speedup = {spec_tps / ar_tps:.2f}x   "
          f"(spec {spec_tps:.0f} tok/s vs AR {ar_tps:.0f} tok/s)")


if __name__ == "__main__":
    main()
