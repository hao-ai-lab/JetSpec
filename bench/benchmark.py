"""Aligned, multi-metric benchmark for the ptd tree engine.

Reports the SAME metrics as the reference `causal_parallel_drafting/benchmark.py`,
computed identically, on the SAME dataset with the SAME chat-template formatting,
so the two engines can be compared row-for-row (and our engine guarded against
regressions). For each tree algorithm it reports:

  accept_len   mean tokens committed per target forward (= reference "Average
               Acceptance length"; reference def is acceptance_length+1, ours too)
  d0..d3       per-position acceptance rate (fraction of steps with accept_len >= k+2)
  tree         mean tree node count per step
  ar_tps       autoregressive-greedy decode tokens/sec (prefill excluded)
  spec_tps     speculative decode tokens/sec (prefill/first draft prefill excluded)
  speedup      wall-clock = ar_time_per_token / spec_time_per_token
  lossless     mean exact-prefix (tokens matching measured AR KV-cache greedy
               before the first divergence) as a fraction of generated length;
               1.00 = byte-identical

The reference-style wall-clock path uses persistent KV tree verify
(`--kv-cache-verify`). AR TPS is measured with the KV-cache greedy baseline,
and `lossless` compares speculative tokens against that same serving-style
baseline. Full-recompute comparison is available via `--debug-lossless`.
Like the reference benchmark, TPS excludes prefill/setup from per-output-token
decode timing.

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
import torch.distributed as dist

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


def _dist_info():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1 and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
    return rank, local_rank, world_size


def _barrier(world_size: int):
    if world_size > 1:
        dist.barrier()


def _all_reduce_sum(value: float, device, world_size: int) -> float:
    tensor = torch.tensor(float(value), dtype=torch.float64, device=device)
    if world_size > 1:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return float(tensor.item())


def _all_reduce_max(value: float, device, world_size: int) -> float:
    tensor = torch.tensor(float(value), dtype=torch.float64, device=device)
    if world_size > 1:
        dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return float(tensor.item())


@torch.inference_mode()
def _timed_samples(fn, prompts, world_size: int):
    """Run prompts one by one and return summed sample latency.

    Under torchrun this intentionally does NOT compute data-parallel throughput
    (`sum(tokens) / max(rank wall time)`). Summing per-sample latency across ranks
    lets rank 0 report a latency-derived per-GPU TPS:

        per_gpu_tps = global_tokens / global_sample_latency_sum

    which is comparable to single-GPU request latency instead of scaling with the
    number of GPUs used for dataset sharding.
    """
    torch.cuda.synchronize()
    _barrier(world_size)
    outs, total_latency = [], 0.0
    for prompt in prompts:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        outs.append(fn(prompt))
        torch.cuda.synchronize()
        total_latency += time.perf_counter() - t0
    return outs, total_latency


def _exact_prefix(a, b):
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


def _divergence_summary(name_a: str, a: list[int], name_b: str, b: list[int]) -> str:
    ep = _exact_prefix(a, b)
    a_tok = a[ep] if ep < len(a) else "<end>"
    b_tok = b[ep] if ep < len(b) else "<end>"
    return (
        f"{name_a} vs {name_b}: prefix={ep}/{max(1, min(len(a), len(b)))} "
        f"next=({a_tok}, {b_tok}) lens=({len(a)}, {len(b)})"
    )


def _build_drafter(head, target, block_size: int, target_layer_ids):
    return DraftHeadTreeDrafter(
        head, target=target, block_size=block_size,
        target_layer_ids=target_layer_ids, draft_shift=False,
    )


@torch.inference_mode()
def _recompute_greedy(llm, prompt, n):
    """Greedy by full-context recompute every token.

    This is intentionally separate from the measured AR KV-cache baseline and is
    used only by `--debug-lossless` to expose bf16/FA2 differences between
    serving-style KV-cache decoding and full recompute.
    """
    ids = llm.tokenizer(prompt, return_tensors="pt").input_ids.to(llm.device)
    out, cur = [], ids
    for _ in range(n):
        pos = torch.arange(cur.shape[1], device=llm.device).unsqueeze(0)
        logits, _, _ = llm.runner.forward(cur, DynamicCache(), pos, last_position_logits_only=True)
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
    ap.add_argument("--attn-implementation", default="auto",
                    choices=["auto", "sdpa", "flash_attention_2"])
    ap.add_argument("--torch-compile", action="store_true", default=False,
                    help="Apply torch.compile(dynamic=True) to the target model")
    ap.add_argument("--no-torch-compile", action="store_false", dest="torch_compile")
    ap.add_argument("--fused-moe", action="store_true",
                    help="Patch compatible Qwen3-MoE blocks with grouped-mm experts")
    ap.add_argument("--warmup-samples-per-rank", type=int, default=0)
    ap.add_argument("--dataset", default="gsm8k", choices=list(PROMPT_FMT))
    ap.add_argument("--samples", type=int, default=5)
    ap.add_argument("--algos", default="crossproduct,top2gap_fanout,task_router,reasoning_router,class_histogram")
    ap.add_argument("--width", type=int, default=7)
    ap.add_argument("--budget", type=int, default=255)
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--kv-cache-verify", action="store_true",
                    help="persistent-cache tree verify (real wall-clock) instead of recompute")
    ap.add_argument("--tree-attn", default="triton", choices=["sdpa", "triton"],
                    help="tree verify attention backend; triton requires --kv-cache-verify")
    ap.add_argument("--profile", default=None,
                    help="JSON profile_table (bench/collect_profile.py) for depth_rank_histogram")
    ap.add_argument("--b2-tau", type=float, default=None,
                    help="override depth_rank_histogram tau (per-(depth,rank) accept cutoff)")
    ap.add_argument("--debug-lossless", type=int, default=0,
                    help="on rank 0, print first-divergence diagnostics for N local prompts")
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
    if args.tree_attn != "sdpa" and not args.kv_cache_verify:
        raise SystemExit("--tree-attn triton requires --kv-cache-verify")

    rank, local_rank, world_size = _dist_info()
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    llm = LLM(args.model, device=device, attn_implementation=args.attn_implementation)
    resolved_attn = getattr(llm.model, "_ptd_attn_implementation", args.attn_implementation)
    fused_moe_blocks = 0
    if args.fused_moe:
        from ptd.models.moe_fused import patch_qwen3_moe_with_grouped_mm

        fused_moe_blocks = patch_qwen3_moe_with_grouped_mm(llm.model)
    if args.torch_compile:
        llm.model = torch.compile(llm.model, dynamic=True)
        llm.model._ptd_attn_implementation = resolved_attn
        llm.runner.model = llm.model

    head = load_draft_head(head_path, device=device, attn_implementation=resolved_attn)
    tli = head.target_layer_ids
    bs = head.block_size
    drafter = _build_drafter(head, llm.model, bs, tli)
    all_prompts = build_prompts(llm.tokenizer, args.dataset, args.samples)
    prompts = all_prompts[rank::world_size]
    sp = SamplingParams(0.0, args.max_new)

    if prompts:
        for i in range(max(0, args.warmup_samples_per_rank)):
            prompt = prompts[i % len(prompts)]
            llm.generate(prompt, sp)
            # Warm each selected algorithm shape once; this absorbs FA2/compile and
            # persistent-KV setup outside the timed window.
            for algo in algos:
                llm.generate_tree(
                    prompt, drafter, block_size=bs, tree_width=args.width,
                    budget=args.budget, algo=algo, algo_kwargs=ALGO_KWARGS[algo],
                    target_layer_ids=tli, sampling_params=sp, return_stats=True,
                    kv_cache_verify=args.kv_cache_verify,
                    tree_attn=args.tree_attn,
                    profile_table=profile_table)
        torch.cuda.synchronize()

    # AR-greedy (KV-cache) baseline = the 1x wall-clock denominator.
    ar_outs, ar_wall_latency_local = _timed_samples(lambda p: llm.generate(p, sp), prompts, world_size)
    ar_tokens = [out["token_ids"] for out in ar_outs]
    ar_ntok_local = sum(len(t) for t in ar_tokens)
    ar_ntok = int(_all_reduce_sum(ar_ntok_local, device, world_size))
    ar_decode_latency_local = sum(float(out.get("decode_time", 0.0)) for out in ar_outs)
    ar_latency = _all_reduce_sum(ar_decode_latency_local, device, world_size)
    ar_wall_latency = _all_reduce_sum(ar_wall_latency_local, device, world_size)
    ar_tps = ar_ntok / ar_latency if ar_latency > 0 else 0.0
    recompute_tokens = (
        [_recompute_greedy(llm, p, args.max_new) for p in prompts]
        if args.debug_lossless > 0 else None
    )
    ar_recompute_lossless = None
    if recompute_tokens is not None:
        ar_recompute_fracs = [
            _exact_prefix(recompute, ar) / max(1, len(ar))
            for recompute, ar in zip(recompute_tokens, ar_tokens)
        ]
        ar_recompute_sum = _all_reduce_sum(sum(ar_recompute_fracs), device, world_size)
        ar_recompute_count = _all_reduce_sum(len(ar_recompute_fracs), device, world_size)
        ar_recompute_lossless = (
            ar_recompute_sum / ar_recompute_count if ar_recompute_count else 0.0
        )
    if rank == 0:
        print(f"\nmodel={args.model} head={head_path}")
        print(f"dataset={args.dataset} samples={len(all_prompts)} world_size={world_size} "
              f"block_size={bs} width={args.width} budget={args.budget} max_new={args.max_new}")
        print(f"attn_implementation={resolved_attn} torch_compile={args.torch_compile} "
              f"fused_moe_blocks={fused_moe_blocks} tree_attn={args.tree_attn} "
              f"draft_attn={resolved_attn}")
        ar_avg_ms = 1000.0 * ar_latency / len(all_prompts) if all_prompts else 0.0
        ar_wall_avg_ms = 1000.0 * ar_wall_latency / len(all_prompts) if all_prompts else 0.0
        print(f"AR-greedy baseline: {ar_tps:.1f} tok/s/gpu  "
              f"({ar_ntok} tok, avg_decode_latency={ar_avg_ms:.1f} ms/sample, "
              f"avg_wall_latency={ar_wall_avg_ms:.1f} ms/sample)\n")
        if ar_recompute_lossless is not None:
            print(f"AR KV-cache vs full-recompute exact-prefix: {ar_recompute_lossless:.3f}\n")
        hdr = (f"{'algorithm':<22}{'accept_len':>11}{'d0':>7}{'d1':>7}{'d2':>7}{'d3':>7}"
               f"{'tree':>7}{'spec_tps/gpu':>13}{'speedup':>9}{'lossless':>10}")
        print(hdr); print("-" * len(hdr))

    for algo in algos:
        outs, spec_wall_latency_local = _timed_samples(
            lambda p: llm.generate_tree(
                p, drafter, block_size=bs, tree_width=args.width, budget=args.budget,
                algo=algo, algo_kwargs=ALGO_KWARGS[algo], target_layer_ids=tli,
                sampling_params=sp, return_stats=True, kv_cache_verify=args.kv_cache_verify,
                tree_attn=args.tree_attn,
                profile_table=profile_table),
            prompts,
            world_size)

        all_acc, all_tree, loss_fracs = [], [], []
        spec_ntok_local = 0
        for out, ar in zip(outs, ar_tokens):
            all_acc += out["accept_lengths"]; all_tree += out["tree_sizes"]
            spec_ntok_local += len(out["token_ids"])
            ep = _exact_prefix(ar, out["token_ids"])
            loss_fracs.append(ep / max(1, len(out["token_ids"])))

        acc_sum = _all_reduce_sum(sum(all_acc), device, world_size)
        acc_count = _all_reduce_sum(len(all_acc), device, world_size)
        tree_sum = _all_reduce_sum(sum(all_tree), device, world_size)
        tree_count = _all_reduce_sum(len(all_tree), device, world_size)
        loss_sum = _all_reduce_sum(sum(loss_fracs), device, world_size)
        loss_count = _all_reduce_sum(len(loss_fracs), device, world_size)
        pos_counts = [
            _all_reduce_sum(sum(1 for al in all_acc if al >= k + 2), device, world_size)
            for k in range(4)
        ]
        spec_ntok = int(_all_reduce_sum(spec_ntok_local, device, world_size))
        spec_decode_latency_local = sum(float(out.get("decode_time", 0.0)) for out in outs)
        spec_latency = _all_reduce_sum(spec_decode_latency_local, device, world_size)

        tau = acc_sum / acc_count if acc_count else 0.0
        per_pos = [count / acc_count if acc_count else 0.0 for count in pos_counts]
        tree_avg = tree_sum / tree_count if tree_count else 0.0
        spec_tps = spec_ntok / spec_latency if spec_latency > 0 else 0.0
        speedup = ((ar_latency / ar_ntok) / (spec_latency / spec_ntok)
                   if ar_ntok > 0 and spec_ntok > 0 and spec_latency > 0 else 0.0)
        lossless = loss_sum / loss_count if loss_count else 0.0
        if rank == 0:
            print(f"{algo:<22}{tau:>11.2f}" + "".join(f"{r:>7.2f}" for r in per_pos) +
                  f"{tree_avg:>7.0f}{spec_tps:>13.1f}{speedup:>9.2f}{lossless:>10.3f}")

        if rank == 0 and args.debug_lossless > 0:
            n_debug = min(args.debug_lossless, len(prompts), len(outs))
            if n_debug:
                print(f"\nlossless debug ({algo}, first {n_debug} rank-0 prompts):")
            for i in range(n_debug):
                prompt = prompts[i]
                spec_kv = outs[i]["token_ids"]
                ar = ar_tokens[i]
                recompute = recompute_tokens[i]
                spec_recompute = llm.generate_tree(
                    prompt, drafter, block_size=bs, tree_width=args.width,
                    budget=args.budget, algo=algo, algo_kwargs=ALGO_KWARGS[algo],
                    target_layer_ids=tli, sampling_params=sp, return_stats=True,
                    kv_cache_verify=False,
                    tree_attn="sdpa",
                    profile_table=profile_table)["token_ids"]
                print(f"  prompt[{i}]:")
                print("    " + _divergence_summary("ar_kv", ar, "recompute", recompute))
                print("    " + _divergence_summary("spec_kv", spec_kv, "ar_kv", ar))
                print("    " + _divergence_summary("spec_kv", spec_kv, "recompute", recompute))
                print("    " + _divergence_summary("spec_recompute", spec_recompute, "recompute", recompute))
                print("    " + _divergence_summary("spec_recompute", spec_recompute, "ar_kv", ar))
    if rank == 0:
        verify_note = ("persistent KV-cache verify (real wall-clock)" if args.kv_cache_verify
                       else "recompute verify -> spec_tps/speedup UNDERSTATE; pass --kv-cache-verify for real wall-clock")
        print("\naccept_len = tokens/forward (= reference Average Acceptance length). "
              "d_k = per-position accept rate. lossless = mean exact-prefix fraction vs "
              f"the measured AR KV-cache greedy baseline. verify mode: {verify_note}. "
              "tok/s/gpu is latency-derived from summed per-sample decode time "
              "(prefill/setup excluded, matching reference), not data-parallel "
              "aggregate throughput.")

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
