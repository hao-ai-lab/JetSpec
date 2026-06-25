"""Wall-clock TPS for the optimized JetSpec engine — AR baseline vs tree-spec.

Reports REAL wall-clock tokens/sec (time.perf_counter), NOT GPU-self-time —
i.e. what a user actually sees, including host/Python overhead. Complements
bench/profiling/compare_engine_with_vllm_integration.py (which reports decode_cuda_speedup =
GPU-self-time, drafter-excluded). The production configuration behind the
README Results table:

    JETSPEC_FUSE_GEMMS=1 JETSPEC_BACKEND=triton_paged_tree_cudagraph_nogather \
      PYTHONPATH=. JETSPEC_DRAFT_HEAD=JetSpec/jetspec-qwen3-8b \
      python bench/engine/tps_walltime.py --samples 64 --max-tokens 2048 \
        --tree-depth 15 --budget 127 --session --prompt-set gsm8k

The same script also runs under torchrun; each rank owns a disjoint prompt shard
and rank 0 reports aggregate throughput using total tokens / slowest-rank wall
time, matching data-parallel benchmark accounting.
"""
import argparse
import os
import time

import torch
import torch.distributed as dist

from jetspec.core.llm import SamplingParams
from jetspec.inference_engine.engine import JetSpecEngine
from jetspec.models.draft_head import load_draft_head
from jetspec.draft_head_adapter import (
    CompiledDraftHead,
    DraftHeadTreeDrafter,
    GraphedDraftHead,
)

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
    if prompt_set == "aime25":
        ds = load_dataset("yentinglin/aime_2025", split="train")
        fmt = ("{problem}\n"
               "Please reason step by step, and put your final answer within \\boxed{{}}.")
        return [fmt.format(problem=ds[i]["problem"]) for i in range(min(n, len(ds)))]
    if prompt_set == "mbpp":
        ds = load_dataset("google-research-datasets/mbpp", "full", split="test")
        fmt = ("You are an expert Python programmer. Write a solution to the following "
               "task and make sure it passes the tests:\n{text}\nYour code should pass "
               "these tests:\n{tests}")
        return [fmt.format(text=ds[i]["text"], tests="\n".join(ds[i]["test_list"]))
                for i in range(min(n, len(ds)))]
    if prompt_set == "livecodebench":
        # script-based dataset -> needs datasets<4 + trust_remote_code (pinned in the image)
        ds = load_dataset("livecodebench/code_generation_lite", version_tag="release_v5",
                          split="test", trust_remote_code=True)
        fmt = ("Write a Python solution to the following competitive-programming "
               "problem:\n```\n{q}\n```")
        return [fmt.format(q=ds[i]["question_content"]) for i in range(min(n, len(ds)))]
    if prompt_set == "mt_bench":
        ds = load_dataset("HuggingFaceH4/mt_bench_prompts", split="train")
        # single-turn: the first user turn of each MT-Bench item
        return [ds[i]["prompt"][0] for i in range(min(n, len(ds)))]
    raise ValueError(f"unknown prompt set: {prompt_set}")


def _dist_info():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    return rank, local_rank, world_size


def _barrier(world_size: int):
    if world_size > 1:
        dist.barrier()


def _walltime(fn, prompts, world_size: int):
    """Return (total_tokens, wall_seconds, per_prompt_outputs)."""
    torch.cuda.synchronize()
    _barrier(world_size)
    t0 = time.perf_counter()
    outs = [fn(p) for p in prompts]
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    ntok = sum(len(o["token_ids"]) for o in outs)
    return ntok, dt, outs


def _sum_and_max(local_tokens: int, local_seconds: float, device: str, world_size: int):
    stats = torch.tensor(
        [float(local_tokens), float(local_seconds)],
        dtype=torch.float64,
        device=device,
    )
    if world_size > 1:
        token_stats = stats[:1].clone()
        time_stats = stats[1:].clone()
        dist.all_reduce(token_stats, op=dist.ReduceOp.SUM)
        dist.all_reduce(time_stats, op=dist.ReduceOp.MAX)
        return int(token_stats.item()), float(time_stats.item())
    return local_tokens, local_seconds


def _sum_pair(local_a: float, local_b: float, device: str, world_size: int):
    stats = torch.tensor([local_a, local_b], dtype=torch.float64, device=device)
    if world_size > 1:
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    return float(stats[0].item()), float(stats[1].item())


def _rank_details(values: list[float], device: str, world_size: int):
    local = torch.tensor(values, dtype=torch.float64, device=device)
    if world_size <= 1:
        return [local.cpu().tolist()]
    gathered = [torch.empty_like(local) for _ in range(world_size)]
    dist.all_gather(gathered, local)
    return [item.cpu().tolist() for item in gathered]


def _build_drafter(head, eng: JetSpecEngine, block_size: int, target_layer_ids,
                   drafter: str = "eager"):
    """eager = per-round head forward; compiled = torch.compile per ctx bucket;
    graphed = CUDA-graph replay per ctx bucket. graphed is the production headline
    config — at bs=1 the tree round is host-launch-bound, so replaying the draft-head
    forward as a CUDA graph (instead of an eager per-round forward) is ~2x."""
    if drafter == "graphed":
        return GraphedDraftHead(
            head, target=eng.model, block_size=block_size,
            target_layer_ids=target_layer_ids, draft_shift=False,
        )
    if drafter == "compiled":
        return CompiledDraftHead(
            head, target=eng.model, block_size=block_size,
            target_layer_ids=target_layer_ids, draft_shift=False,
        )
    return DraftHeadTreeDrafter(
        head, target=eng.model, block_size=block_size,
        target_layer_ids=target_layer_ids, draft_shift=False,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--draft-head", default=None)
    ap.add_argument("--attn-implementation", default="sdpa",
                    choices=["auto", "sdpa", "flash_attention_2"])
    ap.add_argument("--torch-compile", action="store_true", default=False,
                    help="Apply torch.compile(dynamic=True) to the target model")
    ap.add_argument("--no-torch-compile", action="store_false", dest="torch_compile")
    ap.add_argument("--fused-moe", action="store_true",
                    help="Patch compatible Qwen3-MoE blocks with grouped-mm experts")
    ap.add_argument("--warmup-samples-per-rank", type=int, default=1)
    ap.add_argument("--warm-all", action="store_true",
                    help="warm the tree path over EVERY timed prompt first, so the graphed "
                         "draft head captures all context buckets up front (steady-state TPS; "
                         "avoids mid-timing re-capture that understates long/varied benchmarks)")
    ap.add_argument("--samples", type=int, default=5)
    ap.add_argument("--max-tokens", type=int, default=210)
    ap.add_argument("--budget", type=int, default=255)
    ap.add_argument("--tree-depth", type=int, default=None,
                    help="Maximum draft tree depth excluding root. Defaults to draft-head block_size - 1.")
    ap.add_argument("--tree-width", type=int, default=7)
    ap.add_argument("--algo", default="accum_logp")
    ap.add_argument("--drafter", default="eager", choices=["eager", "compiled", "graphed"],
                    help="graphed = CUDA-graph draft head (the production headline config); "
                         "eager is correct but ~2x slower at bs=1 (host-launch-bound)")
    ap.add_argument("--session", action="store_true",
                    help="W11: reuse the tree session (pool + captured graphs) across prompts")
    ap.add_argument("--prompt-set", default="gsm8k",
                    choices=["gsm8k", "math500", "humaneval", "aime"])
    args = ap.parse_args()

    rank, local_rank, world_size = _dist_info()
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    backend = os.environ.get("JETSPEC_BACKEND", "triton_paged_tree_cudagraph")
    head_id = args.draft_head or os.environ["JETSPEC_DRAFT_HEAD"]
    eng = JetSpecEngine(
        args.model,
        device=device,
        dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation,
        attn_backend=backend,
        block_size=16,
        torch_compile=args.torch_compile,
        fused_moe=args.fused_moe,
    )
    resolved_attn = eng.resolved_attn_implementation
    head = load_draft_head(
        head_id,
        device=device,
        dtype=torch.bfloat16,
        attn_implementation=resolved_attn,
    )
    tli = head.target_layer_ids
    tree_block_size = (
        head.block_size if args.tree_depth is None else int(args.tree_depth) + 1
    )
    if tree_block_size < 2:
        raise ValueError(f"--tree-depth must be >= 1; got {args.tree_depth}")
    drafter = _build_drafter(head, eng, tree_block_size, tli, args.drafter)

    bank = _load_prompt_bank(args.prompt_set, args.samples)
    shard_bank = bank[rank::world_size]
    prompts = [eng.tokenizer.apply_chat_template(
        [{"role": "user", "content": p}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
        for p in shard_bank]

    sp = SamplingParams(0.0, args.max_tokens)
    tkw = dict(block_size=tree_block_size, tree_width=args.tree_width,
               budget=args.budget, tree_depth=tree_block_size - 1,
               algo=args.algo, target_layer_ids=tli, return_stats=True)
    if args.session and prompts:
        tkw["session"] = True
        # capacity = the longest prompt in the set (session guard is loud, not growing)
        max_len = max(eng.tokenizer(p, return_tensors="pt").input_ids.shape[1] for p in prompts)
        tkw["session_prompt_capacity"] = ((max_len + 255) // 256) * 256

    # Warmup (excluded): absorb HF compile, torch.compile, Triton autotune, and
    # first CUDA graph captures before the measured windows.
    if prompts:
        for i in range(max(0, args.warmup_samples_per_rank)):
            p = prompts[i % len(prompts)]
            eng.generate(p, sp)
            eng.generate_tree(p, drafter, sampling_params=sp, **tkw)
        if args.warm_all:
            # Capture every context bucket the timed run will hit, so the graphed
            # draft head never re-captures mid-timing (which understates long/varied
            # benchmarks like MATH-500). Measures steady-state, post-warmup throughput.
            for p in prompts:
                eng.generate_tree(p, drafter, sampling_params=sp, **tkw)
    torch.cuda.synchronize()

    ar_tok, ar_t, _ = _walltime(lambda p: eng.generate(p, sp), prompts, world_size)
    ar_tok_total, ar_t_max = _sum_and_max(ar_tok, ar_t, device, world_size)
    ar_tps = ar_tok_total / ar_t_max if ar_t_max > 0 else 0.0

    tree_tok, tree_t, touts = _walltime(
        lambda p: eng.generate_tree(p, drafter, sampling_params=sp, **tkw),
        prompts,
        world_size,
    )
    tree_tok_total, tree_t_max = _sum_and_max(tree_tok, tree_t, device, world_size)
    spec_tps = tree_tok_total / tree_t_max if tree_t_max > 0 else 0.0
    rounds = sum(o["rounds"] for o in touts)
    acc_sum = sum(sum(o["accept_lengths"]) for o in touts)
    acc_total, rounds_total = _sum_pair(acc_sum, rounds, device, world_size)
    accept_len = acc_total / rounds_total if rounds_total else 0.0

    details = _rank_details(
        [len(prompts), ar_tok, ar_t, tree_tok, tree_t, rounds, acc_sum],
        device,
        world_size,
    )

    if rank == 0:
        print(f"\nbackend={backend}  head={head_id}  algo={args.algo}")
        print(f"model={args.model}  attn_implementation={resolved_attn}  "
              f"draft_attn={resolved_attn}  torch_compile={args.torch_compile}  "
              f"fused_moe_blocks={eng.fused_moe_blocks}")
        print(f"samples={args.samples} world_size={world_size} budget={args.budget} "
              f"tree_depth={tree_block_size - 1} width={args.tree_width} "
              f"max_tokens={args.max_tokens}")
        print(f"AR    : {ar_tok_total:5d} tok  {ar_t_max:7.3f}s  ->  {ar_tps:8.1f} tok/s   (1x baseline)")
        print(f"tree  : {tree_tok_total:5d} tok  {tree_t_max:7.3f}s  ->  {spec_tps:8.1f} tok/s   "
              f"accept_len={accept_len:.2f}")
        speedup = spec_tps / ar_tps if ar_tps > 0 else 0.0
        print(f"\nWALL-CLOCK spec speedup = {speedup:.2f}x   "
              f"(spec {spec_tps:.0f} tok/s vs AR {ar_tps:.0f} tok/s)")
        if world_size > 1:
            print("\nper-rank: rank samples ar_tok ar_s tree_tok tree_s rounds accept_sum")
            for i, row in enumerate(details):
                print(
                    f"  {i:>2d} {int(row[0]):>7d} {int(row[1]):>6d} {row[2]:>7.3f} "
                    f"{int(row[3]):>8d} {row[4]:>7.3f} {int(row[5]):>6d} {row[6]:>10.1f}"
                )

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
