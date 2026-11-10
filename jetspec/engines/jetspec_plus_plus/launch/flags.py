"""Command-line definitions for serving, launch children and calibration.

Entry-specific defaults remain explicit here; execution modules only parse and
validate the resulting namespace. Recipe contents remain sealed data.
"""
import argparse
from copy import deepcopy
import os
from pathlib import Path


# Shared spellings and common semantics. Each entry point supplies only its
# differing defaults, requirements, choices or help text.
_COMMON = {
    'model': ('--model', dict(default='qwen3-8b', help='target model')),
    'schedule': ('--schedule', dict()),
    'engine': ('--engine', dict()),
    'prompts-json': ('--prompts-json', dict()),
    'tokens': ('--tokens', dict(type=int)),
    'placement': ('--placement', dict()),
    'devices': ('--devices', dict()),
    'output-json': ('--output-json', dict()),
    'dataset': ('--dataset', dict()),
    'count': ('--count', dict(type=int)),
    'cache-root': ('--cache-root', dict()),
    'cpu-pin': ('--cpu-pin', dict(default='auto')),
    'knob': ('--knob', dict(action='append', default=[], metavar='NAME=VALUE')),
    'stop-at-eos': ('--stop-at-eos', dict(choices=('on', 'off'), default='off')),
    'resolve-only': ('--resolve-only', dict(action='store_true')),
    'output': ('--output', dict()),
    'host-values': ('--host-values', dict()),
    'artifacts-dir': ('--artifacts-dir', dict()),
    'sequential-prompts': ('--sequential-prompts', dict(action='store_true')),
    'wire-records': ('--wire-records', dict(choices=('on', 'off'))),
    'child-wall-seconds': ('--child-wall-seconds', dict(type=float)),
    'seed': ('--seed', dict(type=int, default=0)),
    'fit': ('--fit', dict(type=Path, required=True)),
    'capture': ('--capture', dict(type=Path, required=True)),
    'target': ('--target', dict()),
    'drafter': ('--drafter', dict()),
    'adapter': ('--adapter', dict()),
    'heads-bundle': ('--heads-bundle', dict()),
    'target-release': ('--target-release', dict()),
}


def _argument(parser, name, **overrides):
    flag, defaults = _COMMON[name]
    return parser.add_argument(flag, **{**deepcopy(defaults), **overrides})


def _parser(**kwargs):
    return argparse.ArgumentParser(**kwargs)


def serve_parser():
    from .cli import MODELS, SCHEDULES, ENGINES, __doc__
    result = _parser(description=__doc__)
    _argument(result, 'model', choices=MODELS)
    _argument(result, 'schedule', choices=SCHEDULES, help='J/R schedule, or ar for the autoregressive baseline; defaults to the fastest measured one')
    _argument(result, 'engine', choices=ENGINES, default='fast', help='fast (CUDA graphs)')
    for role in ('target', 'drafter', 'adapter', 'heads-bundle', 'target-release', 'host-values'):
        _argument(result, role, default=os.environ.get('JSPP_' + role.upper().replace('-', '_')))
    _argument(result, 'prompts-json', required=True)
    _argument(result, 'tokens', default=64)
    _argument(result, 'placement', choices=('pair', 'one_gpu'))
    _argument(result, 'devices', default='0,1')
    _argument(result, 'output-json', required=True)
    _argument(result, 'dataset', default='gsm8k')
    _argument(result, 'count', default=2)
    _argument(result, 'cache-root', default=os.environ.get('JSPP_CACHE_ROOT'))
    _argument(result, 'cpu-pin')
    _argument(result, 'knob')
    _argument(result, 'stop-at-eos')
    _argument(result, 'resolve-only')
    return result


def generate_parser():
    from .cli import MODELS, SCHEDULES, generate
    result = _parser(prog='python -m jetspec.engines.jetspec_plus_plus generate',
                                     description=generate.__doc__)
    result.add_argument('prompt')
    _argument(result, 'model', choices=MODELS)
    _argument(result, 'schedule', choices=tuple((schedule for schedule in SCHEDULES if schedule != 'ar')), help='defaults to the fastest measured one')
    result.add_argument('--max-tokens', type=int, default=2048)
    _argument(result, 'devices', default='0,1', help='two GPUs, as nvidia-smi numbers them')
    _argument(result, 'dataset', default='gsm8k', help='calibration entry of the heads')
    _argument(result, 'cache-root', default=os.environ.get('JSPP_CACHE_ROOT') or str(Path(os.environ.get('XDG_CACHE_HOME') or Path.home() / '.cache') / 'jetspec-plus-plus'), help='persistent compiler cache; the first run compiles for several minutes')
    _argument(result, 'target', help='local target directory (default: the pinned Hugging Face snapshot)')
    result.add_argument('--checkpoint', help='local JetSpec++ checkpoint (default: the pinned release)')
    return result


def entry_parser():
    from .entry import __doc__
    parser = _parser(description=__doc__)
    parser.add_argument('settings', nargs='+', help='category=entry or category.key=JSON')
    _argument(parser, 'resolve-only')
    _argument(parser, 'output', default='read.json')
    _argument(parser, 'host-values')
    _argument(parser, 'artifacts-dir')
    _argument(parser, 'sequential-prompts', help='serve C=1 prompts sequentially in one loaded process')
    _argument(parser, 'cache-root', help='root for the serving child compiler caches')
    _argument(parser, 'wire-records', help='write proposal wire records (serving default: on)')
    _argument(parser, 'child-wall-seconds', help="wall limit of the serve child, compile included (default: the launcher's)")
    return parser


def runner_parser():
    from .runner import (METHODS, REGIMES, PLACEMENTS, DEFAULT_ARTIFACTS_DIR,
                         HOST_VALUES_SCHEMA, HOST_VARIABLES, __doc__)
    parser = _parser(description=__doc__)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--regime", choices=REGIMES, required=True)
    _argument(parser, 'engine', choices=('eager', 'reference', 'eval', 'fast'))
    _argument(parser, 'placement', choices=PLACEMENTS, required=True)
    parser.add_argument("--mps-root", help="private per-GPU MPS control and log directory; "
                        "required with one_gpu_mps")
    parser.add_argument("--mps-draft-priority", choices=("0", "1"),
                        help="opt-in MPS priority for the spawned draft client (0 normal, 1 low)")
    parser.add_argument("--mps-draft-active-thread-percentage", type=int,
                        help="opt-in MPS thread cap for the spawned draft client (1..100)")
    _argument(parser, 'devices', required=True)
    _argument(parser, 'dataset', required=True)
    _argument(parser, 'prompts-json', required=True)
    _argument(parser, 'count', default=32)
    parser.add_argument("--batch", type=int, default=1,
                        help="maximum concurrent requests (default: 1)")
    _argument(parser, 'sequential-prompts', help='serve C=1 prompts one at a time in one loaded process')
    _argument(parser, 'tokens', default=256)
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="target sampling temperature (draft remains greedy)")
    _argument(parser, 'seed', help='sampling seed')
    parser.add_argument("--repeats", type=int, default=1)
    _argument(parser, 'child-wall-seconds', default=18 * 60, help='wall-clock cap on one serve child before SIGTERM (default 18 min)')
    parser.add_argument("--passes", type=int,
                        help="drafter passes; when supplied, must match the named recipe")
    parser.add_argument("--sweep-rows-cap", type=int, choices=(2, 4, 8, 16),
                        help="override JetSpec++ eager pair row cap for draft-work sweep")
    parser.add_argument("--sweep-refresh", choices=("after_j0", "off"),
                        help="override JetSpec++ eager pair refresh for draft-work sweep")
    parser.add_argument("--sweep-j", type=int, choices=(1, 2),
                        help="override JetSpec++ eager pair J for draft-work sweep")
    parser.add_argument("--policy", help="C-indexed draft-budget policy JSON")
    parser.add_argument("--draft-split", choices=("verify_gpu",),
                        help="place a second draft executor beside the verify under MPS")
    parser.add_argument("--draft-split-frac", type=float,
                        help="fraction of requests assigned to the co-resident executor")
    parser.add_argument("--draft-deadline", choices=("on", "off"),
                 help="explicitly override the recipe draft deadline")
    parser.add_argument("--miss-redraft", choices=("on", "off"),
                 help="override miss redrafting for a paired after-miss read")
    parser.add_argument("--chain-miss-redraft", choices=("on", "off"),
                        help="enable the J0-only same-round chain miss redraft")
    parser.add_argument("--chain-miss-redraft-max-ms", type=float,
                        help="dedicated same-round redraft wait valve")
    parser.add_argument("--chain-check", choices=("on", "off"),
                        help="run same-round fused eager shadow and enforce the 300-round T0 gate")
    parser.add_argument("--chain-warm-reachable", choices=("on", "off"),
                        help="capture only chain keys reachable by this read")
    parser.add_argument("--chain-capture-stamps", choices=("on", "off"),
                        help="print wall seconds for every chain key and segment capture")
    parser.add_argument("--chain-decoder-key-bucket", choices=("on", "off"),
                        help="override decoder-window key bucketing for a chain T0 read")
    parser.add_argument("--chain-sega-glue", choices=("0", "1", "2"),
                        help="override segmented chain A glue for a T0 or timing read")
    parser.add_argument("--chain-refine-k3", choices=("on", "off"),
                        help="override K3 refinement bookkeeping for a chain timing pair")
    parser.add_argument("--chain-handoff", choices=("on", "off"),
                        help="pre-stage the next supported fixed R0 chain key")
    parser.add_argument("--chain-handoff-timing", action="store_true",
                        help="print per-round draft and verify handoff timestamps")
    _argument(parser, 'knob', help='set one registered JETSPEC_* knob for this read (repeatable)')
    parser.add_argument("--verify-lean", choices=("on", "off"),
                        help="set the six N=1 verifier host cuts (JETSPEC_VERIFY_STAGE_HOST ... STAGE_EARLY)")
    parser.add_argument("--verify-lean-stage-early", choices=("on", "off"),
                        help="override JETSPEC_VERIFY_STAGE_EARLY alone (after --verify-lean)")
    parser.add_argument("--chain-prepost", choices=("on", "off"),
                        help="post the next chain outcome receive before Seg P sync")
    parser.add_argument("--chain-prepost-timing", action="store_true",
                        help="print P-sync to next Seg A and verifier ready-key wait")
    parser.add_argument("--chain-tail-trace", action="store_true",
                        help="print absolute host stamps for every draft chain round")
    parser.add_argument("--chain-k3", choices=("on", "off"),
                        help="override both segmented-chain K3 kernels")
    parser.add_argument("--chain-attn3", choices=("on", "off"),
                        help="override split Seg A FlashInfer planning for a chain read")
    parser.add_argument("--recipe", required=True)
    parser.add_argument("--max-model-len", type=int,
                        help="serve this max_model_len instead of the one derived from the prompts "
                             "(still refused if a prompt + --tokens + lookahead does not fit)")
    _argument(parser, 'output-json', required=True)
    _argument(parser, 'artifacts-dir', default=str(DEFAULT_ARTIFACTS_DIR), help='root that holds the manifest destination directories')
    _argument(parser, 'cache-root', help='persistent compiler cache root shared by serial reads')
    parser.add_argument("--compile-pin", type=Path, help="sealed machine-local compile pin directory")
    parser.add_argument("--compile-pin-sha", help="SHA256 of --compile-pin (both flags required)")
    parser.add_argument("--bundle-override", type=Path,
                        help="pinned task-local reused decoder bundle")
    parser.add_argument("--bundle-pin", type=Path,
                        help="hash receipt for --bundle-override")
    _argument(parser, 'host-values', help='per-host JSON (schema %s): artifact directories, CUDA arch, compile pin SHA, rendezvous mode, and %s' % (HOST_VALUES_SCHEMA, ', '.join(HOST_VARIABLES)))
    parser.add_argument("--host-artifact", action="append", metavar="KEY=ABS_DIR",
                        help="override one manifest artifact directory on this host")
    _argument(parser, 'cpu-pin', metavar='auto|off|A-B[;C-D]', help='NUMA-local CPU pinning (default: auto); off or explicit ranges')
    parser.add_argument("--rank-cpus", metavar="VERIFY_RANGE;DRAFT_RANGE[;SPLIT_RANGE]",
                        help="alias for --cpu-pin with explicit ranges")
    _argument(parser, 'stop-at-eos', help='end each sequence at end-of-sequence instead of running to --tokens (default: off)')
    _argument(parser, 'wire-records', default='on', help='write JetSpec proposal wire records (default: on)')
    parser.add_argument("--causal-n1", action="store_true",
                         help="use the FlashInfer causal kernel for a single verify chain")
    _argument(parser, 'resolve-only', help='print RESOLVED_CONDITION and exit 0 before any model load')
    parser.add_argument("--internal-plan", help=argparse.SUPPRESS)
    return parser


def bench_parser():
    parser = _parser(description="Serving benchmark")

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
    _argument(parser, 'seed', help='Sampling seed')
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
    _argument(parser, 'output-json', type=str, default=None, help='Write outputs, token ids, timing, and metrics as JSON; supports {run_name}, {b}, {temp}')
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
    _argument(parser, 'prompts-json', type=str, default=None, help='Use exact prompt token ids from an exported JSON file')

    # Sweep mode: load engine once, run multiple configs
    parser.add_argument("--sweep", type=str, default=None,
                        help="JSON list of override dicts. Sweepable keys: temp, b. "
                             "Each dict also supports 'name' for wandb run name.")

    return parser


def stage_cal28_parser():
    parser = _parser()
    parser.add_argument('--root', type=Path, required=True)
    _argument(parser, 'fit')
    parser.add_argument('--legacy', type=Path, required=True)
    parser.add_argument('--host', type=Path, required=True)
    parser.add_argument('--cache-source', type=Path, required=True)
    return parser


def calibration_parser(*, description):
    parser = _parser(description=description)
    commands = parser.add_subparsers(dest='command', required=True)
    fit_parser = commands.add_parser('fit')
    _argument(fit_parser, 'capture')
    fit_parser.add_argument('--decoder-manifest', type=Path, required=True)
    fit_parser.add_argument('--decoder-checkpoint', type=Path)
    fit_parser.add_argument('--mode', choices=('generic-v1', 'historical'), default='generic-v1')
    _argument(fit_parser, 'output', type=Path, required=True)
    replay_parser = commands.add_parser('replay')
    _argument(replay_parser, 'capture')
    _argument(replay_parser, 'fit')
    replay_parser.add_argument('--served-floor', type=int, default=0)
    _argument(replay_parser, 'output', type=Path, required=True)
    return parser
