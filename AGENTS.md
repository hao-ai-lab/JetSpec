# AGENTS.md

This file tells coding agents how this repository is organized and how to run each path. The README keeps only the shortest commands. Everything else is here.

## Repository map

- `jetspec/methods/jetspec_plus_plus/`: JetSpec++ torch reference (adapter, decoder, refinement passes, rounds).
- `jetspec/engines/jetspec_plus_plus/`: JetSpec++ serving engine (pipelined draft and verify on two GPUs).
- `jetspec/methods/jetspec/`, `jetspec/engines/jetspec/`: JetSpec tree drafting and its engine.
- `bench/jetspec_plus_plus/`: `verify.py`, `tau.py`, `speedup.py`, `table3.py`.
- `tests/`: CPU tests, golden fixtures for Table 3.

## Environments

- Install the exact lock file for the platform with `pip install --no-deps`: `requirements-x86_64.lock` (A100, B200) or `requirements-gb200-aarch64.lock` (GB200). A resolved install can change the output tokens.
- The lock alone serves both Qwen3-8B and Qwen3.8-27B.
- `pytest tests/` runs on CPU. GPU checks are the commands below.

## JetSpec++ serving engine

- `python -m jetspec.engines.jetspec_plus_plus generate PROMPT`: one prompt. It downloads the pinned target and checkpoint, renders the chat template without thinking, and prints the answer with its throughput. `--target` and `--checkpoint` take local directories instead.
- `python -m jetspec.engines.jetspec_plus_plus` without `generate` reads a prompts file. Its rows need `token_ids`; `generate` and `speedup.py` build them.
- `--devices` names two GPUs as `nvidia-smi` numbers them. `CUDA_VISIBLE_DEVICES` does not renumber them. The engine refuses a GPU that another process uses.
- `--schedule` picks the J/R schedule; without it the engine takes the fastest measured one for the GPU: A100 `j1-r0` for 8B, `j1-r1` elsewhere.
- `--cache-root` keeps compiled kernels (default `~/.cache/jetspec-plus-plus`). The first read compiles for several minutes. Reads that are compared must share one cache.
- The engine's generic draft slot reads the pinned `Qwen/Qwen3-0.6B` snapshot from `SSD_HF_CACHE` (default: the Hugging Face hub cache). The entry downloads it when it is missing.
- `--heads-bundle` may name a checkpoint's `calibration/` directory. The entry links it under its registration name, which is where the reader looks.
- Checkpoint layout: the draft model at the repository root, the adapter in `adapter/`, the calibration in `calibration/`. The manifest pins every file by size and SHA-256.
- Context limit on A100 80GB: the 8B draft rank precomputes attention mask banks for the longest prompt plus `--tokens`. A context of 2,304 tokens fits; 2,560 runs out of memory on the draft GPU (HumanEval prompts with a 2,048-token cap reach it). `SSD_SHARED_GPU_KV_UTILIZATION` does not change this. On A100, lower `--tokens` for longer prompts. The limit was measured on A100 only.

## JetSpec++ torch reference

- `examples/jetspec_plus_plus/simple_generate.py`: one prompt, AR equality check.
- `bench/jetspec_plus_plus/verify.py`: token equality with greedy AR over a prompt set.
- `bench/jetspec_plus_plus/tau.py`: acceptance length.

## Benchmarks

- `speedup.py`: JetSpec++ against graphed AR on the same GPU type. Defaults: 32 prompts, at most 2,048 tokens, stop at EOS, `--schedule auto`, which reads 4 prompts with each supported schedule and keeps the fastest.
- `speedup.py --dataset`: `gsm8k` and `math500` use the paper's Table 3 prompts; `aime25`, `humaneval`, `mbpp`, `livecodebench` and `mt-bench` (first turn) use the evaluation exports. For Qwen3.8-27B the exports are rendered again with the target's chat template.
- `table3.py`: the paper's Table 3 protocol on a GB200 pair. Default arms: AR, DSpark and JetSpec++. `--baselines all` adds DFlash and SSD for 8B. `--assets release` uses the published checkpoints. See `bench/jetspec_plus_plus/TABLE3.md`.

## JetSpec

- HF reference, the JetSpec engine, benchmark scripts and the vLLM integration: see the JetSpec section of the README.
- The HF reference benchmark needs `pip install -e '.[bench,flash-attn]'` for `--attn-implementation flash_attention_2`; `sdpa` runs without it.
- The engine benchmark (`bench/jetspec/engine/tps_walltime.py`) was measured on B200. With the README flags it runs out of memory on an A100 80GB, with or without `--warm-all`.

## Conventions

- Speedup is measured against AR on one GPU of the same type.
- Temperature 0, batch size 1 and EOS stopping unless a flag says otherwise.
