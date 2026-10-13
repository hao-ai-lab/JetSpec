
# Locate the component from either its category or its stable forwarding path.
from pathlib import Path as _ToolPath
import sys as _tool_sys
_TOOL_ROOT = next((p for p in _ToolPath(__file__).resolve().parents
                   if (p / "artifacts/manifest.json").is_file() or (p / "serve/README.md").is_file()), _ToolPath.cwd())
_tool_sys.path[:0] = [str(_TOOL_ROOT), str(_TOOL_ROOT / 'serve'), str(_ToolPath(__file__).resolve().parent)]
if __name__ == "__main__" and any(a in ("-h", "--help") for a in _tool_sys.argv[1:]):
    from jetspec.engines.jetspec_plus_plus.serving.tools.lib.entrypoints import cpu_help as _cpu_help
    _cpu_help(__file__)

import os

_cute_cache_root = os.environ.get("SSD_CUTE_DSL_CACHE_ROOT")
if _cute_cache_root:
    os.environ["CUTE_DSL_CACHE_DIR"] = f"{_cute_cache_root}/{os.getpid()}"

import jetspec.engines.jetspec_plus_plus.serving.paths as ssd  # noqa: F401 — sets TORCH_CUDA_ARCH_LIST before flashinfer import
import sys
import time
import argparse
import hashlib
import json
import tempfile
from pathlib import Path
from random import randint, seed
from jetspec.engines.jetspec_plus_plus.serving.api import LLM; from jetspec.engines.jetspec_plus_plus.serving.api import SamplingParams
from jetspec.engines.jetspec_plus_plus.serving.llm_engine import METRICS
from transformers import AutoTokenizer
from jetspec.engines.jetspec_plus_plus._optional import wandb
import jetspec.engines.jetspec_plus_plus.serving.bench.bench_helpers as bench_helpers
from jetspec.engines.jetspec_plus_plus.serving.bench.bench_helpers import get_model_paths
from jetspec.engines.jetspec_plus_plus.serving.bench.bench_guards import exported_prompt_ids; from jetspec.engines.jetspec_plus_plus.serving.bench.bench_guards import generate_inputs_fail_closed
import jetspec.engines.jetspec_plus_plus.draft.eagle3 as eagle3; import jetspec.engines.jetspec_plus_plus.draft.run_manifest as run_manifest
from jetspec.engines.jetspec_plus_plus.draft.env import validate_environment


def parse_arguments():
    """Parse command line arguments for benchmarking."""
    parser = argparse.ArgumentParser(description="Benchmark SSD performance (API similar to example.py)")

    # Model configuration
    parser.add_argument("--size", type=str, choices=["0.6", "1.7", "4", "8", "14", "32", "1", "3", "70"], default="70",
                        help="Model size in billions of parameters (0.6, 1.7, 4, 8, 14, 32, 1, 3, 70)")
    parser.add_argument("--llama", action="store_true", default=True, help="Use Llama models (default)")
    parser.add_argument("--qwen", action="store_true", help="Use Qwen models instead of Llama")
    parser.add_argument("--draft", type=str, default=None,
                        help="Draft model size (0.6 for Qwen-0.6B, 1 for Llama-1B) or path to draft model")

    # Execution configuration
    parser.add_argument("--eager", action="store_true", help="Use eager execution (disable CUDA graphs)")
    parser.add_argument("--gpus", type=int, default=1, help="Total number of gpus")

    # Speculative decoding configuration
    parser.add_argument("--spec", action="store_true", help="Enable speculative decoding")
    parser.add_argument("--eagle", action="store_true", help="Enable eagle speculative decoding (implies --spec, uses default eagle draft for model)")
    parser.add_argument("--k", type=int, default=6, help="Speculative decoding k value")
    parser.add_argument("--async", action="store_true", help="Enable async speculative decoding")
    parser.add_argument("--f", type=int, default=3, help="Async fan out value")
    parser.add_argument("--fl", type=int, nargs='+', default=None, help="Fan out list (e.g., --fl 1 3 4 becomes [1, 3, 4])")
    parser.add_argument("--flh", type=int, nargs='+', default=None, help="Fan out list (e.g., --flh 1 3 4 becomes [1, 3, 4])")
    parser.add_argument("--flm", type=int, nargs='+', default=None, help="Fan out list miss (e.g., --flm 1 3 4 becomes [1, 3, 4])")
    parser.add_argument("--backup", type=str, choices=["jit", "fast"], default="jit", help="Backup strategy (jit or fast)")

    # Memory and batching configuration
    parser.add_argument("--block_sz", type=int, default=256, help="KV cache block size (see config.py: kvcache_block_size)")
    parser.add_argument("--b", type=int, default=1, help="Maximum number of sequences in batch")
    parser.add_argument("--max_model_len", type=int, default=8192, help="Maximum model length")

    # Generation configuration
    parser.add_argument("--input_len", type=int, default=128, help="Maximum input length")
    parser.add_argument("--output_len", type=int, default=512, help="Maximum output length")
    parser.add_argument("--stop_at_eos", action="store_true", help="Stop each sequence at end-of-sequence (default: ignore EOS and run to --output_len)")
    parser.add_argument("--numseqs", type=int, default=128, help="Number of sequences to generate")
    parser.add_argument("--temp", type=float, default=0.0, help="Temperature for generation")
    parser.add_argument("--dtemp", type=float, default=None, help="Draft async temperature (overrides --temp)")
    parser.add_argument("--seed", type=int, default=0, help="Sampling seed")
    parser.add_argument("--x", type=float, default=None, help="Sampler x for generation (Saguaro sampling coefficient)")

    # Example mode
    parser.add_argument("--example", action="store_true", help="Use real prompts like in example.py and print generations (supports up to batch size 8)")
    parser.add_argument("--humaneval", action="store_true", help="Use HumanEval prompts")
    parser.add_argument("--alpaca", action="store_true", help="Use Alpaca prompts")
    parser.add_argument("--c4", action="store_true", help="Use C4 prompts")
    parser.add_argument("--ultrafeedback", action="store_true", help="Use UltraFeedback prompts")
    parser.add_argument("--random", action="store_true", help="Use random tokens instead of dataset prompts")
    parser.add_argument("--prompt_offset", type=int, default=0, help="Skip first N prompts per dataset (for variance testing)")
    parser.add_argument("--all", action="store_true", help="Use numseqs from each dataset (union dataset with numseqs*4 total)")
    parser.add_argument("--chat_template", action="store_true", help="Wrap dataset prompts in chat template before tokenizing")

    # Debugging and logging
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode (saves draft inputs during prefill)")
    parser.add_argument("--max-steps", type=int, default=None, help="Maximum number of steps to run")
    parser.add_argument("--wandb", action="store_true", help="Log metrics to wandb")
    parser.add_argument("--group", type=str, default=None, help="Wandb group name")
    parser.add_argument("--name", type=str, default=None, help="Wandb run name")
    parser.add_argument("--output-json", type=str, default=None,
                        help="Write outputs, token ids, timing, and metrics as JSON; supports {run_name}, {b}, {temp}")
    parser.add_argument("--jetspec-taps", action="store_true",
                        help="Capture and ship target layers 1, 9, 17, 25, and 33")
    parser.add_argument("--blockhead", type=str, default=None,
                        help="Serve a linear block-head checkpoint in the serial slot")
    parser.add_argument("--jetspec-sync-drafter", type=str, default=None,
                        help="Serve a DFlash-class drafter checkpoint in the serial slot, fresh context")
    parser.add_argument("--taps-dump", type=str, default=None,
                        help="Write completed per-sequence JetSpec taps as a torch file")
    parser.add_argument("--jetspec", action="store_true",
                        help="Use the JetSpec try-all-k protocol")
    parser.add_argument("--jetspec-sync", action="store_true",
                        help="Test-only sequential proof: draft after each actual (k, f)")
    parser.add_argument("--jetspec-hyp", type=int, default=16,
                        help="Number of k hypotheses, starting at k=0")
    parser.add_argument("--jetspec-passes", type=int, default=3,
                        help="Jacobi refinement passes per candidate set")
    parser.add_argument("--jetspec-miss", choices=["ar", "verify"], default="ar",
                        help="AR fallback or verify drafts after a correction miss")
    parser.add_argument(
        "--jetspec-verify-backend",
        choices=["flashinfer_paged", "sdpa"],
        default="flashinfer_paged",
        help="Target block-verification attention backend",
    )
    parser.add_argument(
        "--jetspec-draft-graphs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Capture the JetSpec block head by (H, context bucket)",
    )
    parser.add_argument(
        "--jetspec-wait-ms",
        type=float,
        default=float("inf"),
        help="Candidate wait budget in ms (0=never wait, inf=unbounded)",
    )
    parser.add_argument("--jetspec-exclude-rejected", action="store_true",
                        help="Exclude the preceding rejected token at correction slot 1")
    parser.add_argument("--prompts-json", type=str, default=None,
                        help="Use exact prompt token ids from an exported JSON file")

    # Sweep mode: load engine once, run multiple configs
    parser.add_argument("--sweep", type=str, default=None,
                        help="JSON list of override dicts. Sweepable keys: temp, b. "
                             "Each dict also supports 'name' for wandb run name.")

    args = parser.parse_args()
    if args.jetspec_exclude_rejected:
        raise ValueError("--jetspec-exclude-rejected is unsupported: the engine has no reader")
    for name, flag in (("JETSPEC_BLOCKHEAD", args.blockhead),
                       ("JETSPEC_SYNC_DRAFTER", args.jetspec_sync_drafter)):
        if name in os.environ and os.environ[name] != flag:
            raise ValueError(f"{name}={os.environ[name]!r} disagrees with its bench flag {flag!r}")
    if args.jetspec and args.dtemp not in (None, 0):
        raise ValueError("JetSpec requires a greedy draft")
    if args.qwen and '--llama' in sys.argv:
        raise ValueError("--llama and --qwen are mutually exclusive")
    if args.qwen:
        args.llama = False
    if args.eagle:
        args.spec = True
        # A sampled target needs --dtemp 0: the verifier's point-mass q is lossless only for a greedy draft.
        if args.dtemp not in (None, 0) or (args.temp != 0.0 and args.dtemp is None):
            raise ValueError("Eagle requires a greedy draft (--dtemp 0); a sampled target needs it explicitly")
        if not getattr(args, 'async', False):
            raise ValueError("Eagle currently only supports async speculative decoding")
    if args.jetspec:
        args.spec = True
        setattr(args, "async", True)
        from jetspec.engines.jetspec_plus_plus.wire.protocol import draft_width
        args.k = draft_width()
        args.jetspec_taps = True
        if not args.qwen:
            raise ValueError("JetSpec currently requires the pinned Qwen3 target")
        split = os.environ.get("JETSPEC_DRAFT_SPLIT", "") == "verify_gpu"
        if args.gpus != (3 if split else 2):
            raise ValueError("JetSpec requires one target and " +
                             ("two draft ranks for the split" if split else "one draft GPU"))
    if args.blockhead:
        if args.jetspec or args.eagle:
            raise ValueError("Block-head mode is exclusive")
        if getattr(args, "async", False):
            raise ValueError("Block-head mode uses the sync slot")
        if not args.qwen:
            raise ValueError("Block-head mode requires the pinned Qwen3 target")
        # --gpus 2 without async runs the target tensor-parallel over both ranks (tp2); the
        # head stays in rank 0's sync slot.
        if (args.gpus not in (1, 2) or (args.b > 1 and not args.eager
                and os.environ.get("JETSPEC_BLOCKHEAD_GRAPH", "0") == "1"
                and os.environ.get("JETSPEC_BLOCKHEAD_BATCH", "0") != "1")):
            raise ValueError("Block-head graph batching requires one GPU (or two for target TP) and the batch flag")
        if args.dtemp not in (None, 0):
            raise ValueError("Block-head mode requires a greedy draft")
        with open(os.path.join(args.blockhead, "config.json"), encoding="utf-8") as handle:
            config = json.load(handle)
            args.k = int(config["block_size"]) - ("target_layer_ids" not in config)
            if os.environ.get("JETSPEC_BLOCKHEAD_BLOCK"):
                args.k = int(os.environ["JETSPEC_BLOCKHEAD_BLOCK"]) - 1
        os.environ["JETSPEC_BLOCKHEAD"] = args.blockhead
        args.spec = True
        args.jetspec_taps = True
    if args.jetspec_sync_drafter:
        if args.jetspec or args.eagle or args.blockhead:
            raise ValueError("Sync drafter mode is exclusive")
        if getattr(args, "async", False):
            raise ValueError("Sync drafter mode uses the sync slot")
        if not args.qwen:
            raise ValueError("Sync drafter mode requires the pinned Qwen3 target")
        if args.gpus != 1 or (args.b > 1 and not args.eager):
            raise ValueError("Sync drafter batching requires one GPU and eager mode")
        if args.dtemp not in (None, 0):
            raise ValueError("Sync drafter mode requires a greedy draft")
        args.k = 15
        os.environ["JETSPEC_SYNC_DRAFTER"] = args.jetspec_sync_drafter
        args.spec = True
        args.jetspec_taps = True
    return args


def resolve_model_paths(args):
    """get_model_paths, handed the EAGLE-3 draft the fork's table lacks as if --draft named it."""
    if os.environ.get(eagle3.DRAFT_PATH_ENV) and not args.eagle:
        raise ValueError(f"{eagle3.DRAFT_PATH_ENV} is set but --eagle is not: nothing would load it")
    base = eagle3.eagle_draft_base(args, bench_helpers.HF_CACHE_DIR) if args.eagle else None
    if base is None:
        return get_model_paths(args)
    return get_model_paths(argparse.Namespace(**{**vars(args), "draft": base}))


def create_run_name(args):
    """Create a descriptive run name for wandb logging."""
    spec_mode_str = "spec" if args.spec else "normal"
    async_mode_str = "_async" if getattr(args, 'async', False) else ""
    jit_mode_str = "_jit" if args.backup == "jit" else ""
    model_type = "llama" if args.llama else "qwen"
    example_str = "_example" if args.example else ""
    humaneval_str = "_humaneval" if args.humaneval else ""
    alpaca_str = "_alpaca" if args.alpaca else ""
    c4_str = "_c4" if args.c4 else ""
    ultrafeedback_str = "_ultrafeedback" if args.ultrafeedback else ""
    random_str = "_random" if args.random else ""
    all_str = "_all" if args.all else ""
    gsm_str = "_gsm" if not args.example and not args.humaneval and not args.alpaca and not args.c4 and not args.ultrafeedback and not args.random and not args.all else ""
    sampler_x_str = f"_sampler_x{args.x}" if args.x else ""

    temp_str = f"_temp{args.temp}"
    if args.dtemp is not None:
        temp_str += f"_dtemp{args.dtemp}"

    draft_str = f"_draft{args.draft}" if args.draft is not None else "_nodraft"
    k_str = f"_k{args.k}"
    f_str = f"_f{args.f}"

    return args.name if args.name else f"{model_type}_size{args.size}_{spec_mode_str}{async_mode_str}{jit_mode_str}_b{args.b}{k_str}{f_str}{draft_str}{temp_str}{sampler_x_str}{example_str}{humaneval_str}{alpaca_str}{c4_str}{ultrafeedback_str}{random_str}{all_str}{gsm_str}"


def initialize_wandb(args, run_name):
    """Initialize wandb logging if requested."""
    if not args.wandb:
        return

    wandb.init(
        project="jetspec.engines.jetspec_plus_plus.serving.api",
        name=run_name,
        group=args.group,
        config={
            "model_size": args.size,
            "gpus": args.gpus,
            "speculative_decoding": args.spec,
            "async_speculative": getattr(args, 'async', False),
            "jit_speculative": args.backup == "jit",
            "k": args.k if args.spec else None,
            "f": args.f,
            "fan_out_list": args.flh,
            "fan_out_list_miss": args.flm,
            "llama": args.llama,
            "max_model_len": args.max_model_len,
            "input_len": args.input_len,
            "output_len": args.output_len,
            "numseqs": args.numseqs,
            "draft_model": args.draft,
            "b": args.b,
            "block_size": args.block_sz,
            "eager": args.eager,
            "example_mode": args.example,
            "humaneval_mode": args.humaneval,
            "alpaca_mode": args.alpaca,
            "c4_mode": args.c4,
            "ultrafeedback_mode": args.ultrafeedback,
            "random_mode": args.random,
            "all_mode": args.all,
            "sampler_x": args.x,
            "implementation": "jetspec.engines.jetspec_plus_plus.serving.api",
            "max_steps": args.max_steps,
        }
    )


def create_llm_kwargs(args, draft_path):
    """Create LLM initialization arguments."""
    llm_kwargs = dict(
        enforce_eager=args.eager,
        num_gpus=args.gpus,
        speculate=args.spec,
        speculate_k=args.k,
        draft_async=getattr(args, 'async', False),
        async_fan_out=args.f,
        verbose=args.verbose,
        draft=draft_path,
        kvcache_block_size=args.block_sz,
        max_num_seqs=args.b,
        max_model_len=args.max_model_len,
        sampler_x=args.x,
        jit_speculate=(args.backup == "jit"),
        max_steps=args.max_steps,
        jetspec_taps=args.jetspec_taps,
        jetspec_taps_dump=args.taps_dump,
        jetspec=args.jetspec,
        jetspec_sync=args.jetspec_sync,
        jetspec_hyp=args.jetspec_hyp,
        jetspec_passes=args.jetspec_passes,
        jetspec_miss=args.jetspec_miss,
        jetspec_verify_backend=args.jetspec_verify_backend,
        jetspec_draft_graphs=args.jetspec_draft_graphs,
        jetspec_wait_ms=args.jetspec_wait_ms,
        jetspec_checkpoint_dir=os.environ.get("JETSPEC_CHECKPOINT_DIR"),
        jetspec_adapter_config=os.environ.get("JETSPEC_ADAPTER_CONFIG"),
        jetspec_adapter_head=os.environ.get("JETSPEC_ADAPTER_HEAD"),
        jetspec_target_release_weights=os.environ.get(
            "JETSPEC_TARGET_RELEASE_WEIGHTS"
        ),
        target_dtype=os.environ.get("JETSPEC_TARGET_DTYPE"),
    )

    if args.flh is not None:
        llm_kwargs["fan_out_list"] = args.flh
    if args.flm is not None:
        llm_kwargs["fan_out_list_miss"] = args.flm

    return llm_kwargs


def log_wandb_metrics(args, metrics, total_tokens, total_time, throughput, model_name, mode, run_name):
    """Log metrics to wandb if enabled."""
    if not args.wandb:
        return

    wandb_metrics = {
        "official_total_tokens": total_tokens,
        "official_total_time": total_time,
        "official_end_to_end_throughput": throughput,
        "model_name": model_name,
        "mode": mode,
        "run_name": run_name,
    }

    if metrics:
        if "prefill_total_time" in metrics and "prefill_total_tokens" in metrics:
            if metrics["prefill_total_time"] > 0:
                wandb_metrics["metrics_prefill_throughput"] = metrics["prefill_total_tokens"] / metrics["prefill_total_time"]

        if "decode_total_time" in metrics and "decode_total_tokens" in metrics:
            if metrics["decode_total_time"] > 0:
                wandb_metrics["metrics_decode_throughput"] = metrics["decode_total_tokens"] / metrics["decode_total_time"]

        if "target_step_times" in metrics and metrics["target_step_times"]:
            avg_target_step_time_ms = sum(metrics["target_step_times"]) * 1000 / len(metrics["target_step_times"])
            wandb_metrics["metrics_avg_target_step_time_ms"] = avg_target_step_time_ms

        if "cache_hits" in metrics and metrics["cache_hits"]:
            wandb_metrics["metrics_avg_cache_hits"] = sum(metrics["cache_hits"]) / len(metrics["cache_hits"])

        if "accepted_suffix_lens_with_recovery" in metrics and metrics["accepted_suffix_lens_with_recovery"]:
            wandb_metrics["metrics_avg_accepted_suffix_lens_with_recovery"] = sum(metrics["accepted_suffix_lens_with_recovery"]) / len(metrics["accepted_suffix_lens_with_recovery"])
            wandb_metrics["metrics_accepted_suffix_lens_with_recovery_histogram"] = wandb.Histogram(metrics["accepted_suffix_lens_with_recovery"])

        if "accepted_suffix_lens_on_hit" in metrics and metrics["accepted_suffix_lens_on_hit"]:
            wandb_metrics["metrics_avg_accepted_suffix_lens_on_hit"] = sum(metrics["accepted_suffix_lens_on_hit"]) / len(metrics["accepted_suffix_lens_on_hit"])
            wandb_metrics["metrics_accepted_suffix_lens_on_hit_histogram"] = wandb.Histogram(metrics["accepted_suffix_lens_on_hit"])

    wandb.log(wandb_metrics)


def run_benchmark(args, llm, prompts, sampling_params):
    """Run the actual benchmark and return results."""
    if args.wandb:
        wandb.log({"sequences_processed": 0, "total_sequences": len(prompts)})

    start_time = time.time()
    outputs, metrics = llm.generate(prompts, sampling_params)
    served_end = getattr(llm, "_generate_served_end_time", None)
    total_time = (served_end if served_end is not None else time.time()) - start_time

    if args.wandb:
        wandb.log({"sequences_processed": len(prompts), "total_sequences": len(prompts)})

    return outputs, total_time, metrics


def reset_metrics():
    """Reset the global METRICS dict between sweep runs."""
    for k, v in METRICS.items():
        if k == "jetspec_split_draft_ms_by_rank":
            METRICS[k] = {}
        elif isinstance(v, list):
            v.clear()
        else:
            METRICS[k] = 0


def reconfigure_engine(llm, b=None):
    """Reconfigure a live engine for a new sweep run without reloading weights."""
    if b is not None:
        assert b <= llm.config.max_num_seqs, f"b={b} > initial max_num_seqs={llm.config.max_num_seqs}"
        llm.config.max_num_seqs = b
        llm.scheduler.max_num_seqs = b


def dataset_name(args):
    """The prompt source, by the same precedence create_run_name uses for its suffix."""
    if args.prompts_json:
        return "prompts_json"
    for name in ("example", "humaneval", "alpaca", "c4", "ultrafeedback", "random", "all"):
        if getattr(args, name):
            return name
    return "gsm"


def output_label(device, backend, *, jetspec=False, sync=False):
    parts = [device or "unknown device", backend or "sdpa"]
    if jetspec:
        parts.append("jetspec")
    if sync:
        parts.append("sync-proof")
    return " · ".join(parts)


def run_mode(args):
    """ar / spec / spec_async / eagle3 / jetspec / jetspec_sync / blockhead, after parse_arguments has resolved the implied flags."""
    if args.blockhead:
        return "blockhead"
    if args.jetspec_sync_drafter:
        return "jetspec_sync"
    if args.jetspec:
        return "jetspec"
    if args.eagle:
        return "eagle3"
    if args.spec:
        return "spec_async" if getattr(args, "async", False) else "spec"
    return "ar"


def record_run_inputs(args, model_path, draft_path, prompts, exported):
    """Bind the run to what it was given, before the engine (and its draft process) starts."""
    if not os.environ.get(run_manifest.PARTS_ENV):
        os.environ[run_manifest.PARTS_ENV] = tempfile.mkdtemp(prefix="jetspec-run-manifest-")
    run_manifest.snapshot_env()
    if 'JETSPEC_SERVE_POLICY' in os.environ:
        record_policy()
    run_manifest.git_revision(Path(__file__).resolve().parents[3])
    run_manifest.record_value("args", vars(args))
    run_manifest.record_value("dataset", dataset_name(args))
    run_manifest.record_value("jetspec_dataset", os.environ.get("JETSPEC_DATASET"))
    run_manifest.record_value("target_model_path", str(model_path))
    run_manifest.record_value("draft_path", None if draft_path is None else str(draft_path))
    run_manifest.record_value("prompts_sha256", hashlib.sha256(json.dumps(
        prompts, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest())
    if args.prompts_json:
        run_manifest.record_file("prompts", args.prompts_json)
        if isinstance(exported, dict):
            run_manifest.record_value("prompts_json_meta", {
                key: value for key, value in exported.items()
                if isinstance(value, (str, int, float, bool, type(None)))})
    if os.path.isdir(model_path):
        run_manifest.record_model_dir("target", model_path)
    else:
        run_manifest.record_value("target_model_files", "absent: not a local directory")
    # A value, not a file: content_sha256 stays the digest of the bytes loaded and nothing else.
    mode = run_mode(args)
    run_manifest.record_value("mode", mode)
    # get_model_paths resolves a draft for every run; only the generic spec modes load it.
    # JetSpec and block-head record their own drafter under drafter/ and blockhead/.
    if mode in ("spec", "spec_async", "eagle3") and draft_path is not None:
        if os.path.isdir(draft_path):
            run_manifest.record_model_dir("draft", draft_path)
            if mode == "eagle3":
                # SpecForge/AngelSlim heads ship pytorch_model.bin, which record_model_dir does not list
                for name in sorted(os.listdir(draft_path)):
                    if name.startswith("pytorch_model") and name.endswith(".bin"):
                        run_manifest.record_file(f"draft/{name}", os.path.join(draft_path, name))
        else:
            run_manifest.record_value("draft_model_files", "absent: not a local directory")
    if args.jetspec:
        from jetspec.engines.jetspec_plus_plus.models.heads.kpred import pass_name
        # The label saturates at J2: a run asking for more passes reads J2 calibration.
        run_manifest.record_value("jetspec_passes_requested", args.jetspec_passes)
        run_manifest.record_value("jetspec_pass_label", pass_name(args.jetspec_passes))




def seal_run_manifest(output_json):
    """Fold in the other processes' records and write the manifest beside the output."""
    import torch
    run_manifest.merge_parts(os.environ[run_manifest.PARTS_ENV])
    run_manifest.package_versions()
    device_name = torch.cuda.get_device_name() if torch.cuda.is_available() else None
    run_manifest.record_value("device_name", device_name)
    manifest = run_manifest.write(f"{output_json}.manifest.json")
    print(f"RUN_MANIFEST sha256={manifest['manifest_sha256']} "
          f"content_sha256={manifest['content_sha256']} "
          f"thresholds_sha256={run_manifest.file_sha256('heads/thresholds', manifest) or 'absent'}",
          flush=True)
    return manifest


def main():
    args = parse_arguments()
    validate_environment(jetspec=args.jetspec, max_num_seqs=args.b,
                         regime='eager' if args.eager else 'graphed')
    import torch
    seed(args.seed)
    torch.manual_seed(args.seed)

    if args.example and args.numseqs > 8:
        print("Warning: --example mode supports up to 8 sequences, reducing numseqs from {} to 8".format(args.numseqs))
        args.numseqs = 8

    model_name, model_path, draft_path = resolve_model_paths(args)

    exported = None
    if args.prompts_json:
        with open(args.prompts_json, encoding="utf-8") as handle:
            exported = json.load(handle)
        rows = exported["prompts"] if isinstance(exported, dict) else exported
        rows = rows[args.prompt_offset : args.prompt_offset + args.numseqs]
        prompt_token_ids = exported_prompt_ids(rows, model_path)
        prompt_indices = [
            int(row.get("index", args.prompt_offset + i))
            for i, row in enumerate(rows)
        ]
        string_prompts = None
        original_prompts = [row.get("text", "") for row in rows]
    else:
        string_prompts, prompt_token_ids, original_prompts = generate_inputs_fail_closed(
            args, model_path, bench_helpers)
        prompt_count = len(
            string_prompts if string_prompts is not None else prompt_token_ids
        )
        prompt_indices = list(
            range(args.prompt_offset, args.prompt_offset + prompt_count)
        )
    prompts = string_prompts if string_prompts is not None else prompt_token_ids

    if prompts:
        num_reqs = len(prompts)
    else:
        num_reqs = args.numseqs
    sampling_params = [SamplingParams(
        temperature=args.temp,
        draft_temperature=args.dtemp,
        ignore_eos=not args.stop_at_eos,
        max_new_tokens=args.output_len,
        jetspec_prompt_index=(
            prompt_indices[i] if args.jetspec or args.blockhead else None
        ) if not args.jetspec_sync_drafter else prompt_indices[i],
    ) for i in range(num_reqs)]

    if prompts:
        for i, prompt in enumerate(prompts):
            if isinstance(prompt, str):
                print(f'Prompt: {prompt}')
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                num_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
            elif isinstance(prompt, list):
                num_tokens = len(prompt)
            else:
                raise ValueError(f"Invalid prompt type: {type(prompt)}")

    # Create LLM (once, reused across sweep configs)
    llm_kwargs = create_llm_kwargs(args, draft_path)
    if args.eagle:
        llm_kwargs['use_eagle'] = True
    if args.debug:
        llm_kwargs['debug_mode'] = True

    record_run_inputs(args, model_path, draft_path, prompts, exported)
    llm = LLM(model_path, **llm_kwargs)
    if args.eagle:
        record_eagle_engine(llm.config)

    # Build sweep configs
    if args.sweep:
        sweep_configs = json.loads(args.sweep)
        assert isinstance(sweep_configs, list), "--sweep must be a JSON list of dicts"
    else:
        sweep_configs = [{}]

    for si, sweep_cfg in enumerate(sweep_configs):
        bad_keys = {"backup", "flh", "flm"} & set(sweep_cfg.keys())
        assert not bad_keys, f"Cannot sweep {bad_keys} — draft process won't see changes."

        temp = sweep_cfg.get("temp", args.temp)
        b = sweep_cfg.get("b", args.b)
        run_name_override = sweep_cfg.get("name", None)

        reconfigure_engine(llm, b=b)

        cur_sampling_params = [SamplingParams(
            temperature=temp,
            draft_temperature=args.dtemp,
            ignore_eos=not args.stop_at_eos,
            max_new_tokens=args.output_len,
            jetspec_prompt_index=(
                prompt_indices[i] if args.jetspec or args.blockhead else None
            ) if not args.jetspec_sync_drafter else prompt_indices[i],
        ) for i in range(num_reqs)]

        reset_metrics()

        if run_name_override:
            cur_run_name = run_name_override
        elif args.sweep:
            cur_run_name = f"{create_run_name(args)}_sweep{si}"
        else:
            cur_run_name = create_run_name(args)

        orig_temp, orig_name, orig_b = args.temp, args.name, args.b
        args.temp, args.b = temp, b
        if run_name_override:
            args.name = run_name_override
        initialize_wandb(args, cur_run_name)
        args.temp, args.name, args.b = orig_temp, orig_name, orig_b

        try:
            print(f"\n{'='*60}")
            print(f"SWEEP [{si+1}/{len(sweep_configs)}] temp={temp} b={b}")
            print(f"{'='*60}")

            outputs, total_time, metrics = run_benchmark(args, llm, prompts, cur_sampling_params)

            total_tokens = sum(sp.max_new_tokens for sp in cur_sampling_params)
            throughput = total_tokens / total_time

            mode = "Eager" if args.eager else "CUDA Graphs"
            spec_mode = f" + Speculative(k={args.k})" if args.spec else ""
            async_mode = " + Async" if getattr(args, 'async', False) else ""
            jit_mode = " + JIT" if args.backup == "jit" else ""
            x_mode = f" + X({args.x})" if args.x else ""
            full_mode = mode + spec_mode + async_mode + jit_mode + x_mode

            if args.output_json:
                result = {
                    "run_name": cur_run_name,
                    "model": model_name,
                    "mode": full_mode,
                    "batch": b,
                    "prompts": len(prompts),
                    "max_new": args.output_len,
                    "taps": args.jetspec_taps,
                    "wall_seconds": total_time,
                    "throughput_tok_s": throughput,
                    "outputs": outputs,
                    "metrics": metrics,
                    "jetspec_wait_ms": args.jetspec_wait_ms,
                }
                output_json = args.output_json.format(run_name=cur_run_name, b=b, temp=temp)
                result["run_manifest"] = seal_run_manifest(output_json)
                result["device_name"] = result["run_manifest"]["values"]["device_name"]
                from jetspec.engines.jetspec_plus_plus.models.layers.attention import ATTENTION_BACKEND
                result["label"] = output_label(result["device_name"], ATTENTION_BACKEND,
                                               jetspec=args.jetspec, sync=args.jetspec_sync)
                with open(output_json, "w", encoding="utf-8") as f:
                    json.dump(result, f, indent=2, default=lambda value: (
                        value.item() if hasattr(value, "item")
                        else value.tolist() if hasattr(value, "tolist")
                        else str(value)))
                print(f"Output JSON: {output_json}", flush=True)

            print(f"Model: {model_name}, Mode: {full_mode}, Total: {total_tokens}tok, Time: {total_time:.2f}s, Total Throughput: {throughput:.2f}tok/s")

            if not args.random and si == 0:
                print("\n" + "="*80)
                print("GENERATIONS:")
                print("="*80)

                tokenizer = AutoTokenizer.from_pretrained(model_path)

                for i, (prompt, output) in enumerate(zip(prompts, outputs)):
                    if i >= 10:
                        break
                    if isinstance(prompt, list):
                        decoded_prompt = tokenizer.decode(prompt, skip_special_tokens=True)
                    else:
                        decoded_prompt = prompt
                    if original_prompts and i < len(original_prompts):
                        display_prompt = original_prompts[i]
                    else:
                        display_prompt = decoded_prompt
                    print(f"\nPrompt {i+1}: {display_prompt!r}")
                    print(f"Generation: {output['text']!r}")
                    print("-" * 40)

            log_wandb_metrics(args, metrics, total_tokens, total_time, throughput, model_name, full_mode, cur_run_name)

            if args.wandb:
                wandb.finish()
        except KeyboardInterrupt:
            print("\nBenchmark interrupted by user")
            break

    print(f'Engine exited!')
    sys.exit(0)


if __name__ == "__main__":
    main()
