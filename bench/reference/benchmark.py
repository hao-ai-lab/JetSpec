"""Aligned, multi-metric benchmark for the jetflow tree engine.

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

The wall-clock path always uses persistent KV tree verify. AR TPS is measured
with the KV-cache greedy baseline.
Like the reference benchmark, TPS excludes prefill/setup from per-output-token
decode timing.

    CUDA_VISIBLE_DEVICES=0 JETFLOW_TEST_MODEL=Qwen/Qwen3-8B \
      JETFLOW_DRAFT_HEAD=Snyhlxde/jetflow-qwen3-8b-distill-epoch6-3e-4-no-gamma \
      HF_HOME=/path/to/hf_cache HF_DATASETS_CACHE=/path/to/hf_cache/datasets \
      PYTHONPATH=. python bench/reference/benchmark.py --dataset gsm8k --samples 5 \
        --algos accum_logp,top2gap_fanout,task_router,reasoning_router,class_histogram --width 7 --budget 255
"""
import argparse
import json
import os
import time

import torch
import torch.distributed as dist
from tqdm import tqdm

from bench.reference.dflash_baseline import dflash_generate
from jetflow.core.llm import LLM, SamplingParams, make_cache
from jetflow.core.sampler import sample
from jetflow.models.draft_head import load_draft_head
from jetflow.draft_head_adapter import DraftHeadTreeDrafter

# Same prompt formatting as the reference (model/utils.load_and_process_dataset)
# then chat-templated with enable_thinking=False (benchmark.py:834).
PROMPT_FMT = {
    "gsm8k": ("openai/gsm8k", "main", "test", "question",
              "{q}\nPlease reason step by step, and put your final answer within \\boxed{{}}."),
    "math500": ("HuggingFaceH4/MATH-500", None, "test", "problem",
                "{q}\nPlease reason step by step, and put your final answer within \\boxed{{}}."),
}

ALGO_KWARGS = {
    "accum_logp": {},
    "top2gap_fanout": {"beta": 2.0, "g_0": 1.0},
    "task_router": {},          # prompt-adaptive; routes via fallback w/o prompt_info
    "reasoning_router": {},
    "class_histogram": {},
    "depth_rank_histogram": {"tau": 0.02},  # needs --profile to shape; else == accum_logp
}


def build_prompts(tokenizer, dataset, n, *, disable_progress: bool = False):
    from datasets import load_dataset
    repo, cfg, split, field, fmt = PROMPT_FMT[dataset]
    ds = load_dataset(repo, cfg, split=split) if cfg else load_dataset(repo, split=split)
    if n is not None and n < len(ds):
        ds = ds.shuffle(seed=0).select(range(n))   # MATCH reference benchmark.py:770 exactly
    prompts = []
    for i in tqdm(range(len(ds)), desc=f"format {dataset} prompts", disable=disable_progress):
        user = fmt.format(q=ds[i][field])
        prompts.append(tokenizer.apply_chat_template(
            [{"role": "user", "content": user}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False))
    return prompts


def tokenize_prompts(tokenizer, prompts, device, *, disable_progress: bool = False):
    return [
        tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        for prompt in tqdm(prompts, desc="tokenize prompts", disable=disable_progress)
    ]


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


@torch.inference_mode()
def _timed_samples(fn, prompts, world_size: int, *, desc: str, disable_progress: bool = False):
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
    for prompt in tqdm(prompts, desc=desc, disable=disable_progress):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        outs.append(fn(prompt))
        torch.cuda.synchronize()
        total_latency += time.perf_counter() - t0
    return outs, total_latency


def _build_drafter(head, target, block_size: int, target_layer_ids):
    return DraftHeadTreeDrafter(
        head, target=target, block_size=block_size,
        target_layer_ids=target_layer_ids, draft_shift=False,
    )


def _gpu_clock_warmup(device: str, seconds: float = 2.0):
    """Burst matmuls to ramp GPU clocks to boost before timed benchmarks."""
    a = torch.randn(2048, 2048, device=device, dtype=torch.bfloat16)
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        torch.mm(a, a)
    torch.cuda.synchronize()
    del a


@torch.inference_mode()
def _direct_target_ar_decode(llm: LLM, input_ids: torch.Tensor, sp: SamplingParams,
                              **kwargs) -> dict:
    """Reference-style AR path: direct target calls plus preallocated ids/positions."""
    input_ids = input_ids.to(llm.device)
    prompt_len = input_ids.shape[1]
    max_new = int(sp.max_new_tokens)
    if max_new <= 0:
        return {"token_ids": [], "decode_time": 0.0}

    max_length = prompt_len + max_new
    output_ids = torch.empty((1, max_length + 1), dtype=torch.long, device=llm.device)
    position_ids = torch.arange(max_length + 1, device=llm.device).unsqueeze(0)
    cache = make_cache(llm.model)
    _fwd = llm._model_fwd
    greedy = sp.temperature == 0.0

    prefill_out = _fwd(
        input_ids=input_ids,
        position_ids=position_ids[:, :prompt_len],
        past_key_values=cache,
        cache_position=position_ids[0, :prompt_len],
        use_cache=True,
        output_hidden_states=False,
        logits_to_keep=1,
    )
    output_ids[:, :prompt_len] = input_ids
    if greedy:
        next_tok = prefill_out.logits[:, -1:, :].argmax(dim=-1)
    else:
        next_tok = sample(prefill_out.logits, sp.temperature).reshape(1, 1)
    output_ids[:, prompt_len:prompt_len + 1] = next_tok

    if not llm.eos_token_ids:
        decode_start = llm._timer()
        for step in range(1, max_new):
            start = prompt_len + step - 1
            out = _fwd(
                input_ids=output_ids[:, start:start + 1],
                position_ids=position_ids[:, start:start + 1],
                past_key_values=cache,
                cache_position=position_ids[0, start:start + 1],
                use_cache=True,
                output_hidden_states=False,
                logits_to_keep=1,
            )
            if greedy:
                next_tok = out.logits[:, -1:, :].argmax(dim=-1)
            else:
                next_tok = sample(out.logits, sp.temperature).reshape(1, 1)
            output_ids[:, start + 1:start + 2] = next_tok
        decode_time = llm._timer() - decode_start
        token_ids = output_ids[0, prompt_len:max_length].cpu().tolist()
        return {"token_ids": token_ids, "decode_time": decode_time}

    eos_ids = llm.eos_token_ids
    out_ids = [int(next_tok.item())]
    decode_start = llm._timer()
    for step in range(1, max_new):
        if out_ids[-1] in eos_ids:
            break
        start = prompt_len + step - 1
        out = _fwd(
            input_ids=output_ids[:, start:start + 1],
            position_ids=position_ids[:, start:start + 1],
            past_key_values=cache,
            cache_position=position_ids[0, start:start + 1],
            use_cache=True,
            output_hidden_states=False,
            logits_to_keep=1,
        )
        if greedy:
            next_tok = out.logits[:, -1:, :].argmax(dim=-1)
        else:
            next_tok = sample(out.logits, sp.temperature).reshape(1, 1)
        output_ids[:, start + 1:start + 2] = next_tok
        out_ids.append(int(next_tok.item()))

    return {"token_ids": out_ids, "decode_time": llm._timer() - decode_start}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--draft-head", default=None)
    ap.add_argument("--attn-implementation", default="auto",
                    choices=["auto", "sdpa", "flash_attention_2"])
    ap.add_argument("--torch-compile", action="store_true", default=False,
                    help="torch.compile the target model (dynamic=True); mirrors the reference benchmark")
    ap.add_argument("--no-torch-compile", action="store_false", dest="torch_compile")
    ap.add_argument("--ignore-eos", action="store_true",
                    help="Ignore EOS during generation; useful for short performance probes")
    ap.add_argument("--include-dflash-baseline", action="store_true",
                    help="Also run the linear DFlash block baseline")
    ap.add_argument("--fused-moe", action="store_true",
                    help="Patch compatible Qwen3-MoE blocks with grouped-mm experts")
    ap.add_argument("--warmup-rounds", type=int, default=3,
                    help="Warmup rounds per rank; each round runs AR and all selected tree configs")
    ap.add_argument("--profile-phases", action="store_true",
                    help="Report draft/tree-build/verify/accept/KV-select timing; adds sync overhead")
    ap.add_argument("--output-dir", default=None,
                    help="Directory to save per-sample generation outputs as JSONL "
                         "(auto-named under bench/outputs/ if not given)")
    ap.add_argument("--dataset", default="gsm8k", choices=list(PROMPT_FMT))
    ap.add_argument("--samples", type=int, default=None,
                    help="Number of dataset examples to run; omit to use the full split")
    ap.add_argument("--algos", default="accum_logp,top2gap_fanout,task_router,reasoning_router,class_histogram")
    ap.add_argument("--width", type=int, default=7)
    ap.add_argument("--depth", type=int, default=16,
                    help="Tree block_size (effective depth = depth-1). "
                         "Clamped to the draft head's native block_size.")
    ap.add_argument("--budget", type=int, default=255)
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--tree-attn-implementation", default="triton", choices=["sdpa", "triton"],
                    help="tree verify attention implementation")
    ap.add_argument("--profile", default=None,
                    help="JSON profile_table (bench/profiling/collect_depth_rank_stats.py) for depth_rank_histogram")
    ap.add_argument("--b2-tau", type=float, default=None,
                    help="override depth_rank_histogram tau (per-(depth,rank) accept cutoff)")
    args = ap.parse_args()
    profile_table = None
    if args.profile:
        with open(args.profile) as f:
            profile_table = json.load(f)
    if args.b2_tau is not None:
        ALGO_KWARGS["depth_rank_histogram"] = {"tau": args.b2_tau}
    head_path = args.draft_head or os.environ.get("JETFLOW_DRAFT_HEAD")
    if not head_path:
        raise SystemExit("set --draft-head or JETFLOW_DRAFT_HEAD")
    algos = args.algos.split(",")
    for a in algos:
        if a not in ALGO_KWARGS:
            raise SystemExit(f"unknown algo {a}; known: {sorted(ALGO_KWARGS)}")
    # Persist Inductor and Triton compiled kernels to a stable location so that
    # subsequent runs reuse the compiled graphs without recompilation.
    # Both env vars are read lazily (at first compile call), so setting them here
    # takes effect even though torch is already imported.
    _cache_root = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        ".cache",
    )
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", os.path.join(_cache_root, "torchinductor"))
    os.environ.setdefault("TRITON_CACHE_DIR", os.path.join(_cache_root, "triton"))

    torch.set_float32_matmul_precision("high")
    rank, local_rank, world_size = _dist_info()
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    llm = LLM(args.model, device=device, attn_implementation=args.attn_implementation)
    resolved_attn = getattr(llm.model, "_jetflow_attn_implementation", args.attn_implementation)
    if args.ignore_eos:
        llm.eos_token_ids = set()
    fused_moe_blocks = 0
    if args.fused_moe:
        from jetflow.models.moe_fused import patch_qwen3_moe_with_grouped_mm

        fused_moe_blocks = patch_qwen3_moe_with_grouped_mm(llm.model)
    if args.torch_compile:
        # Per-layer Dynamo guards that fire once during warmup then stabilise:
        #   - layer_idx (static int on nn.Module) -> allow_unspec_int_on_nn_module
        #   - DynamicLayer.is_initialized (False→True on first KV write, per layer)
        #   - wrapped_forward hidden_states list length (grows per layer)
        # Raise limits so warmup absorbs all of them without falling back to eager.
        torch._dynamo.config.allow_unspec_int_on_nn_module = True
        n_layers = getattr(llm.model.config, "num_hidden_layers", 64)
        limit = n_layers * 8
        torch._dynamo.config.recompile_limit = max(torch._dynamo.config.recompile_limit, limit)
        torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, limit)
        torch._dynamo.config.accumulated_recompile_limit = max(
            torch._dynamo.config.accumulated_recompile_limit, limit * 4,
        )
        llm.model = torch.compile(llm.model, dynamic=True)
        llm.runner.model = llm.model

    head = load_draft_head(head_path, device=device, attn_implementation=resolved_attn)
    tli = head.target_layer_ids
    bs = head.block_size
    tree_bs = min(args.depth, bs)   # tree depth, clamped to head's native block_size
    drafter = _build_drafter(head, llm.model, bs, tli)
    show_progress = rank == 0
    all_prompts = build_prompts(
        llm.tokenizer, args.dataset, args.samples,
        disable_progress=not show_progress,
    )
    all_input_ids = tokenize_prompts(
        llm.tokenizer, all_prompts, device,
        disable_progress=not show_progress,
    )
    prompts = all_input_ids[rank::world_size]
    prompt_texts = all_prompts[rank::world_size]   # parallel to prompts
    sp = SamplingParams(0.0, args.max_new)

    n_warmup = max(0, args.warmup_rounds)
    if prompts and n_warmup > 0:
        _gpu_clock_warmup(device)
        for i in tqdm(range(n_warmup), desc="warmup", disable=not show_progress):
            prompt = prompts[i % len(prompts)]
            _direct_target_ar_decode(llm, prompt, sp)
            if args.include_dflash_baseline:
                dflash_generate(
                    target=llm.model,
                    input_ids=prompt,
                    max_new_tokens=args.max_new,
                    block_size=bs,
                    stop_token_ids=[] if args.ignore_eos else list(llm.eos_token_ids),
                    temperature=0.0,
                    drafter=drafter,
                    target_layer_ids=tli,
                )
            for algo in algos:
                llm.generate_tree(
                    prompt, drafter, block_size=tree_bs, tree_width=args.width,
                    budget=args.budget, algo=algo, algo_kwargs=ALGO_KWARGS[algo],
                    target_layer_ids=tli, sampling_params=sp, return_stats=True,
                    tree_attn=args.tree_attn_implementation,
                    profile_table=profile_table)
        torch.cuda.synchronize()

    # AR-greedy (raw HF KV-cache loop) = the 1x target-forward denominator.
    ar_outs, ar_wall_latency_local = _timed_samples(
        lambda p: _direct_target_ar_decode(llm, p, sp),
        prompts, world_size, desc="AR decode", disable_progress=not show_progress,
    )
    ar_tokens = [out["token_ids"] for out in ar_outs]
    ar_ntok_local = sum(len(t) for t in ar_tokens)
    ar_ntok = int(_all_reduce_sum(ar_ntok_local, device, world_size))
    ar_decode_latency_local = sum(float(out.get("decode_time", 0.0)) for out in ar_outs)
    ar_latency = _all_reduce_sum(ar_decode_latency_local, device, world_size)
    ar_wall_latency = _all_reduce_sum(ar_wall_latency_local, device, world_size)
    ar_tps = ar_ntok / ar_latency if ar_latency > 0 else 0.0
    dflash_block = None
    dflash_outs = []
    if args.include_dflash_baseline:
        dflash_outs, dflash_wall_latency_local = _timed_samples(
            lambda p: dflash_generate(
                target=llm.model,
                input_ids=p,
                max_new_tokens=args.max_new,
                block_size=bs,
                stop_token_ids=[] if args.ignore_eos else list(llm.eos_token_ids),
                temperature=0.0,
                drafter=drafter,
                target_layer_ids=tli,
            ),
            prompts,
            world_size,
            desc=f"DFlash blocksize={bs}",
            disable_progress=not show_progress,
        )
        dflash_ntok = int(_all_reduce_sum(
            sum(int(out.num_output_tokens) for out in dflash_outs),
            device,
            world_size,
        ))
        dflash_latency = _all_reduce_sum(
            sum(float(out.decode_time) for out in dflash_outs),
            device,
            world_size,
        )
        dflash_acc_local = []
        for out in dflash_outs:
            dflash_acc_local += list(out.acceptance_lengths)
        dflash_acc_sum = _all_reduce_sum(sum(dflash_acc_local), device, world_size)
        dflash_acc_count = _all_reduce_sum(len(dflash_acc_local), device, world_size)
        dflash_wall_latency = _all_reduce_sum(dflash_wall_latency_local, device, world_size)
        dflash_tps = dflash_ntok / dflash_latency if dflash_latency > 0 else 0.0
        dflash_e2e_tps = dflash_ntok / dflash_wall_latency if dflash_wall_latency > 0 else 0.0
        dflash_speedup = (
            (ar_latency / ar_ntok) / (dflash_latency / dflash_ntok)
            if ar_ntok > 0 and dflash_ntok > 0 and dflash_latency > 0 else 0.0
        )
        dflash_e2e_speedup = (
            (ar_wall_latency / ar_ntok) / (dflash_wall_latency / dflash_ntok)
            if ar_ntok > 0 and dflash_ntok > 0 and dflash_wall_latency > 0 else 0.0
        )
        dflash_block = {
            "accept_len": dflash_acc_sum / dflash_acc_count if dflash_acc_count else 0.0,
            "tokens": dflash_ntok,
            "tps": dflash_tps,
            "e2e_tps": dflash_e2e_tps,
            "speedup": dflash_speedup,
            "e2e_speedup": dflash_e2e_speedup,
        }
    rows = []
    _algo_outs_map: dict = {}   # algo -> per-sample output list (for output saving)
    for algo in algos:
        outs, spec_wall_latency_local = _timed_samples(
            lambda p: llm.generate_tree(
                p, drafter, block_size=tree_bs, tree_width=args.width, budget=args.budget,
                algo=algo, algo_kwargs=ALGO_KWARGS[algo], target_layer_ids=tli,
                sampling_params=sp, return_stats=True,
                profile_phases=args.profile_phases,
                tree_attn=args.tree_attn_implementation,
                profile_table=profile_table,
                ),
            prompts,
            world_size,
            desc=f"{algo} tree decode",
            disable_progress=not show_progress,
        )

        _algo_outs_map[algo] = outs
        all_acc, all_tree = [], []
        spec_ntok_local = 0
        for out in outs:
            all_acc += out["accept_lengths"]; all_tree += out["tree_sizes"]
            spec_ntok_local += len(out["token_ids"])

        acc_sum = _all_reduce_sum(sum(all_acc), device, world_size)
        acc_count = _all_reduce_sum(len(all_acc), device, world_size)
        tree_sum = _all_reduce_sum(sum(all_tree), device, world_size)
        tree_count = _all_reduce_sum(len(all_tree), device, world_size)
        pos_counts = [
            _all_reduce_sum(sum(1 for al in all_acc if al >= k + 2), device, world_size)
            for k in range(4)
        ]
        spec_ntok = int(_all_reduce_sum(spec_ntok_local, device, world_size))
        spec_decode_latency_local = sum(float(out.get("decode_time", 0.0)) for out in outs)
        spec_latency = _all_reduce_sum(spec_decode_latency_local, device, world_size)
        spec_wall_latency = _all_reduce_sum(spec_wall_latency_local, device, world_size)
        phase_totals = {}
        if args.profile_phases:
            for name in ("draft", "tree_build", "verify", "accept", "kv_select"):
                local_total = sum(
                    float(out.get("phase_times", {}).get(name, 0.0))
                    for out in outs
                )
                phase_totals[name] = _all_reduce_sum(local_total, device, world_size)

        tau = acc_sum / acc_count if acc_count else 0.0
        per_pos = [count / acc_count if acc_count else 0.0 for count in pos_counts]
        tree_avg = tree_sum / tree_count if tree_count else 0.0
        spec_tps = spec_ntok / spec_latency if spec_latency > 0 else 0.0
        spec_e2e_tps = spec_ntok / spec_wall_latency if spec_wall_latency > 0 else 0.0
        speedup = ((ar_latency / ar_ntok) / (spec_latency / spec_ntok)
                   if ar_ntok > 0 and spec_ntok > 0 and spec_latency > 0 else 0.0)
        e2e_speedup = ((ar_wall_latency / ar_ntok) / (spec_wall_latency / spec_ntok)
                       if ar_ntok > 0 and spec_ntok > 0 and spec_wall_latency > 0 else 0.0)
        phase_ms = None
        if args.profile_phases:
            denom = acc_count if acc_count else 1.0
            phase_ms = {
                name: 1000.0 * phase_totals[name] / denom
                for name in ("draft", "tree_build", "verify", "accept", "kv_select")
            }
        rows.append((algo, tau, per_pos, tree_avg, spec_tps, spec_e2e_tps, speedup, e2e_speedup, phase_ms))

    if rank == 0 and args.output_dir is not False:
        # Auto-name if not given: bench/outputs/<timestamp>_<model_short>/
        if args.output_dir:
            out_dir = args.output_dir
        else:
            _ts = time.strftime("%Y%m%d_%H%M%S")
            _mshort = args.model.split("/")[-1]
            out_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "outputs", f"{_ts}_{_mshort}",
            )
        os.makedirs(out_dir, exist_ok=True)

        def _decode(token_ids):
            return llm.tokenizer.decode(token_ids, skip_special_tokens=True)

        # AR outputs
        with open(os.path.join(out_dir, "ar.jsonl"), "w") as f:
            for i, (out, ptext) in enumerate(zip(ar_outs, prompt_texts)):
                f.write(json.dumps({
                    "sample": i,
                    "prompt": ptext,
                    "output": _decode(out["token_ids"]),
                    "token_ids": out["token_ids"],
                    "num_tokens": len(out["token_ids"]),
                    "decode_time_s": round(float(out.get("decode_time", 0.0)), 4),
                }, ensure_ascii=False) + "\n")

        # DFlash outputs
        if args.include_dflash_baseline:
            with open(os.path.join(out_dir, "dflash.jsonl"), "w") as f:
                for i, (out, ptext) in enumerate(zip(dflash_outs, prompt_texts)):
                    gen_ids = out.output_ids[0, out.num_input_tokens:].tolist()
                    f.write(json.dumps({
                        "sample": i,
                        "prompt": ptext,
                        "output": _decode(gen_ids),
                        "token_ids": gen_ids,
                        "num_tokens": out.num_output_tokens,
                        "acceptance_lengths": list(out.acceptance_lengths),
                        "accept_len_mean": (sum(out.acceptance_lengths) / len(out.acceptance_lengths)
                                            if out.acceptance_lengths else 0.0),
                        "decode_time_s": round(float(out.decode_time), 4),
                    }, ensure_ascii=False) + "\n")

        # Tree algo outputs
        for algo, _tau, _per_pos, _tree_avg, *_rest in rows:
            algo_outs = _algo_outs_map[algo]
            with open(os.path.join(out_dir, f"{algo}.jsonl"), "w") as f:
                for i, (out, ptext) in enumerate(zip(algo_outs, prompt_texts)):
                    f.write(json.dumps({
                        "sample": i,
                        "prompt": ptext,
                        "output": out.get("text", _decode(out["token_ids"])),
                        "token_ids": out["token_ids"],
                        "num_tokens": len(out["token_ids"]),
                        "accept_lengths": out.get("accept_lengths", []),
                        "accept_len_mean": (sum(out["accept_lengths"]) / len(out["accept_lengths"])
                                            if out.get("accept_lengths") else 0.0),
                        "tree_sizes": out.get("tree_sizes", []),
                        "rounds": out.get("rounds", 0),
                        "decode_time_s": round(float(out.get("decode_time", 0.0)), 4),
                    }, ensure_ascii=False) + "\n")

        print(f"outputs saved → {out_dir}/")

    if rank == 0:
        print(f"\nmodel={args.model} head={head_path}")
        print(f"dataset={args.dataset} samples={len(all_prompts)} world_size={world_size} "
              f"block_size={bs} tree_depth={tree_bs} width={args.width} budget={args.budget} max_new={args.max_new}")
        print(f"attn_implementation={resolved_attn} torch_compile={args.torch_compile} "
              f"fused_moe_blocks={fused_moe_blocks} tree_attn_implementation={args.tree_attn_implementation} "
              f"draft_attn={resolved_attn}")
        ar_avg_ms = 1000.0 * ar_latency / len(all_prompts) if all_prompts else 0.0
        ar_wall_avg_ms = 1000.0 * ar_wall_latency / len(all_prompts) if all_prompts else 0.0
        ar_e2e_tps = ar_ntok / ar_wall_latency if ar_wall_latency > 0 else 0.0
        print(f"AR-greedy baseline: decode={ar_tps:.1f} tok/s/gpu  e2e={ar_e2e_tps:.1f} tok/s/gpu  "
              f"({ar_ntok} tok, avg_decode_ms={ar_avg_ms:.1f}, avg_wall_ms={ar_wall_avg_ms:.1f})\n")
        if dflash_block is not None:
            print(f"DFlash blocksize={bs}: decode={dflash_block['tps']:.1f} tok/s/gpu  "
                  f"e2e={dflash_block['e2e_tps']:.1f} tok/s/gpu  "
                  f"({dflash_block['tokens']} tok, accept_len={dflash_block['accept_len']:.2f}, "
                  f"speedup={dflash_block['speedup']:.2f}  e2e_speedup={dflash_block['e2e_speedup']:.2f})\n")
        hdr = (f"{'algorithm':<22}{'accept_len':>11}{'d0':>7}{'d1':>7}{'d2':>7}{'d3':>7}"
               f"{'tree':>7}{'decode_tps':>11}{'e2e_tps':>9}{'speedup':>9}{'e2e_spdup':>10}")
        print(hdr); print("-" * len(hdr))
        for algo, tau, per_pos, tree_avg, spec_tps, spec_e2e_tps, speedup, e2e_speedup, phase_ms in rows:
            print(f"{algo:<22}{tau:>11.2f}" + "".join(f"{r:>7.2f}" for r in per_pos) +
                  f"{tree_avg:>7.0f}{spec_tps:>11.1f}{spec_e2e_tps:>9.1f}{speedup:>9.2f}{e2e_speedup:>10.2f}")
            if phase_ms is not None:
                phase_text = " ".join(
                    f"{name}={phase_ms[name]:.2f}ms/round"
                    for name in ("draft", "tree_build", "verify", "accept", "kv_select")
                )
                print(f"  phase_profile: {phase_text}")
        print(
            "\naccept_len = tokens/forward (= reference Average Acceptance length)."
            "\nd_k = per-position accept rate."
            "\ndecode_tps = latency-derived from decode-only time (prefill excluded)."
            "\ne2e_tps = latency-derived from total wall time (prefill included)."
            "\nspeedup = decode_tps / AR decode_tps."
            "\ne2e_spdup = e2e_tps / AR e2e_tps."
        )

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
