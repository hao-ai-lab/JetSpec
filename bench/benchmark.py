"""Aligned, multi-metric benchmark for the ptd tree engine.

Reports the SAME metrics as the reference `causal_parallel_drafting/benchmark.py`,
computed identically, on the SAME dataset with the SAME chat-template formatting,
so the two engines can be compared row-for-row (and our engine guarded against
regressions). For each tree algorithm it reports:

  accept_len   mean tokens committed per target forward (= reference "Average
               Acceptance length"; reference def is acceptance_length+1, ours too)
  d0..d3       per-position acceptance rate (fraction of steps with accept_len >= k+2)
  tree         mean tree node count per step
  ar_tps       autoregressive-greedy tokens/sec (the 1x wall-clock baseline)
  spec_tps     speculative tokens/sec
  speedup      wall-clock = ar_time_per_token / spec_time_per_token
  lossless     mean exact-prefix (tokens matching AR greedy before the first
               divergence) as a fraction of generated length; 1.00 = byte-identical

WALL-CLOCK CAVEAT: ptd's tree verify is recompute-based (no KV reuse yet — see
the tree-KV-cache task), so spec_tps / speedup UNDERSTATE what the engine will do
with cached verify. accept_len / d_k / tree / lossless are cache-independent and
ARE the apples-to-apples parity metrics vs the reference.

    CUDA_VISIBLE_DEVICES=0 PTD_TEST_MODEL=Qwen/Qwen3-8B \
      PTD_DRAFT_HEAD=Snyhlxde/ptd-qwen3-8b-distill-epoch6-3e-4-no-gamma \
      HF_HOME=/path/to/hf_cache HF_DATASETS_CACHE=/path/to/hf_cache/datasets \
      PYTHONPATH=. python bench/benchmark.py --dataset gsm8k --samples 5 \
        --algos crossproduct,top2gap_fanout,task_router,reasoning_router,class_histogram --width 7 --budget 255
"""
import argparse
import os
import time

import torch

from transformers import DynamicCache

from ptd.engine.llm import LLM, SamplingParams
from ptd.models.draft_head import load_draft_head
from ptd.draft_head_drafter import DraftHeadTreeDrafter

# Same prompt formatting as the reference (model/utils.load_and_process_dataset)
# then chat-templated with enable_thinking=False (benchmark.py:834).
PROMPT_FMT = {
    "gsm8k": ("openai/gsm8k", "main", "test", "question",
              "{q}\nPlease reason step by step, and put your final answer within \\boxed{{}}."),
    "math500": ("HuggingFaceH4/MATH-500", None, "test", "problem",
                "{q}\nPlease reason step by step, and put your final answer within \\boxed{{}}."),
}

ALGO_KWARGS = {
    "crossproduct": {},
    "top2gap_fanout": {"beta": 2.0, "g_0": 1.0},
    "task_router": {},          # prompt-adaptive; routes via fallback w/o prompt_info
    "reasoning_router": {},
    "class_histogram": {},
    "depth_rank_histogram": {"tau": 0.02},  # needs --profile to shape; else == crossproduct
}


def build_prompts(tokenizer, dataset, n):
    from datasets import load_dataset
    repo, cfg, split, field, fmt = PROMPT_FMT[dataset]
    ds = load_dataset(repo, cfg, split=split) if cfg else load_dataset(repo, split=split)
    if n < len(ds):
        ds = ds.shuffle(seed=0).select(range(n))   # MATCH reference benchmark.py:770 exactly
    prompts = []
    for i in range(min(n, len(ds))):
        user = fmt.format(q=ds[i][field])
        prompts.append(tokenizer.apply_chat_template(
            [{"role": "user", "content": user}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False))
    return prompts


def _exact_prefix(a, b):
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


@torch.inference_mode()
def _recompute_greedy(llm, prompt, n):
    """Greedy via full recompute — the SAME block-forward numerics the spec verify
    uses, so the only residual mismatch is the bf16 reduction-order flip. This is
    the honest losslessness reference (vs KV-cache AR greedy, which differs from
    the recompute path in bf16 regardless of spec correctness)."""
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


@torch.inference_mode()
def _timed(fn):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    return out, time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--draft-head", default=None)
    ap.add_argument("--dataset", default="gsm8k", choices=list(PROMPT_FMT))
    ap.add_argument("--samples", type=int, default=5)
    ap.add_argument("--algos", default="crossproduct,top2gap_fanout,task_router,reasoning_router,class_histogram")
    ap.add_argument("--width", type=int, default=7)
    ap.add_argument("--budget", type=int, default=255)
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--kv-cache-verify", action="store_true",
                    help="persistent-cache tree verify (real wall-clock) instead of recompute")
    ap.add_argument("--profile", default=None,
                    help="JSON profile_table (bench/collect_profile.py) for depth_rank_histogram")
    ap.add_argument("--b2-tau", type=float, default=None,
                    help="override depth_rank_histogram tau (per-(depth,rank) accept cutoff)")
    args = ap.parse_args()
    profile_table = None
    if args.profile:
        import json
        with open(args.profile) as f:
            profile_table = json.load(f)
    if args.b2_tau is not None:
        ALGO_KWARGS["depth_rank_histogram"] = {"tau": args.b2_tau}
    head_path = args.draft_head or os.environ.get("PTD_DRAFT_HEAD")
    if not head_path:
        raise SystemExit("set --draft-head or PTD_DRAFT_HEAD")
    algos = args.algos.split(",")
    for a in algos:
        if a not in ALGO_KWARGS:
            raise SystemExit(f"unknown algo {a}; known: {sorted(ALGO_KWARGS)}")

    llm = LLM(args.model)
    head = load_draft_head(head_path)
    tli = head.target_layer_ids
    bs = head.block_size
    drafter = DraftHeadTreeDrafter(head, target=llm.model, block_size=bs,
                                   target_layer_ids=tli, draft_shift=False)
    prompts = build_prompts(llm.tokenizer, args.dataset, args.samples)
    sp = SamplingParams(0.0, args.max_new)

    # AR-greedy (KV-cache) baseline = the 1x wall-clock denominator.
    ar_tokens, ar_time = [], 0.0
    for p in prompts:
        out, dt = _timed(lambda: llm.generate(p, sp))
        ar_tokens.append(out["token_ids"]); ar_time += dt
    ar_ntok = sum(len(t) for t in ar_tokens)
    ar_tps = ar_ntok / ar_time
    # Recompute-greedy = the losslessness reference (matching block-forward numerics).
    rg_tokens = [_recompute_greedy(llm, p, args.max_new) for p in prompts]

    print(f"\nmodel={args.model} head={head_path}")
    print(f"dataset={args.dataset} samples={len(prompts)} block_size={bs} "
          f"width={args.width} budget={args.budget} max_new={args.max_new}")
    print(f"AR-greedy baseline: {ar_tps:.1f} tok/s  ({ar_ntok} tok)\n")
    hdr = (f"{'algorithm':<22}{'accept_len':>11}{'d0':>7}{'d1':>7}{'d2':>7}{'d3':>7}"
           f"{'tree':>7}{'spec_tps':>10}{'speedup':>9}{'lossless':>10}")
    print(hdr); print("-" * len(hdr))

    for algo in algos:
        all_acc, all_tree, spec_ntok, spec_time, loss_fracs = [], [], 0, 0.0, []
        for p, rg in zip(prompts, rg_tokens):
            out, dt = _timed(lambda: llm.generate_tree(
                p, drafter, block_size=bs, tree_width=args.width, budget=args.budget,
                algo=algo, algo_kwargs=ALGO_KWARGS[algo], target_layer_ids=tli,
                sampling_params=sp, return_stats=True, kv_cache_verify=args.kv_cache_verify,
                profile_table=profile_table))
            all_acc += out["accept_lengths"]; all_tree += out["tree_sizes"]
            spec_ntok += len(out["token_ids"]); spec_time += dt
            ep = _exact_prefix(rg, out["token_ids"])
            loss_fracs.append(ep / max(1, len(out["token_ids"])))
        tau = sum(all_acc) / len(all_acc)
        per_pos = [sum(1 for al in all_acc if al >= k + 2) / len(all_acc) for k in range(4)]
        tree_avg = sum(all_tree) / len(all_tree)
        spec_tps = spec_ntok / spec_time
        speedup = (ar_time / ar_ntok) / (spec_time / spec_ntok)
        lossless = sum(loss_fracs) / len(loss_fracs)
        print(f"{algo:<22}{tau:>11.2f}" + "".join(f"{r:>7.2f}" for r in per_pos) +
              f"{tree_avg:>7.0f}{spec_tps:>10.1f}{speedup:>9.2f}{lossless:>10.3f}")
    verify_note = ("persistent KV-cache verify (real wall-clock)" if args.kv_cache_verify
                   else "recompute verify -> spec_tps/speedup UNDERSTATE; pass --kv-cache-verify for real wall-clock")
    print("\naccept_len = tokens/forward (= reference Average Acceptance length). "
          "d_k = per-position accept rate. lossless = mean exact-prefix fraction vs "
          "recompute-greedy (matching numerics; <1.0 = a late bf16 reduction-order flip, "
          f"not a spec error). verify mode: {verify_note}.")


if __name__ == "__main__":
    main()
