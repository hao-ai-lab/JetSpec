# AGENTS.md

This file tells coding agents how this repository is organized and how to run each path. The README keeps only the shortest commands. Everything else is here.

## Repository map

- `jetspec/methods/jetspec_plus_plus/`: JetSpec++ torch reference (adapter, decoder, refinement passes, rounds).
- `jetspec/engines/jetspec_plus_plus/`: JetSpec++ serving engine (pipelined draft and verify on two GPUs).
- `jetspec/methods/jetspec/`, `jetspec/engines/jetspec/`: JetSpec tree drafting and its engine.
- `bench/jetspec_plus_plus/`: `verify.py`, `tau.py`, `speedup.py`, `reproduce_paper_results.py`.

## How JetSpec++ works

The paper is [arXiv 2610.19792](https://arxiv.org/abs/2610.19792). It has three parts. The torch reference in `jetspec/methods/jetspec_plus_plus/` implements each part in readable form, and the serving engine runs the same method with CUDA graphs and fused kernels.

- **Pipelined drafting and verification.** The target model runs on one GPU and the draft model on a second GPU. While the target verifies block t, the draft model drafts block t+1, so the draft time hides under the verify time. `round.py` (`RoundDriver`) runs one round.
- **Foresight drafting.** While block t is in verification, two facts are unknown: how many draft tokens the target accepts, and the anchor token (the target's own token after the accepted prefix, which starts block t+1).
  - A hidden-to-hidden adapter (`models/adapter.py`) predicts the target hidden states of block t.
  - The anchor decoder (`models/decoder.py`) is the target's last layers, final norm and LM head, untrained. It reads the predicted states and scores the possible anchor tokens.
  - The draft model drafts block t+1 for several (accepted length, anchor token) outcomes in one pass, one row per outcome (`context.py` builds the hypothesis contexts).
  - The rows to draft are selected with calibrated thresholds on the blended outcome probabilities (`method/rows.py`, `method/blend.py`, `method/kmass.py`). The thresholds ship in each checkpoint's `calibration/`.
  - After verification, a row that matches the real outcome becomes the next draft (a hit). On a miss, the draft model drafts block t+1 again from the verified context (a fallback round).
- **Self refinement.** The draft model is trained to correct its own parallel draft. Jacobi passes (`method/passes.py`) redraft the block from its previous version.
  - A context refresh (`method/refresh.py`) predicts the target states of the current draft again between passes.
  - An anchor replacement (`method/swap.py`) replaces the first draft token that the anchor decoder disagrees with.
  - Pins (`method/pins.py`) keep the tokens that agree.
- **Schedules.** `jJ-rR` means J refinement passes after the first draft and R context refreshes. The release defaults are `j1-r0` for Qwen3-8B on A100 and `j1-r1` elsewhere. The paper's Table 3 runs J1-R1 for Qwen3-8B and J3-R3 for Qwen3.8-27B.
- **Lossless output.** At temperature 0, verification accepts the longest draft prefix that equals the target's greedy tokens (`method/accept.py`), then appends the target's own next token.

## Environments

- Install the exact lock file for the platform with `pip install --no-deps`: `requirements-x86_64.lock` (A100, B200) or `requirements-gb200-aarch64.lock` (GB200). A resolved install can change the output tokens.
- The lock alone serves both Qwen3-8B and Qwen3.8-27B.

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
- `speedup.py --dataset`: on Qwen3-8B, `gsm8k` and `math500` use the paper's Table 3 prompts. On Qwen3.8-27B they use user-only exports, because the Table 3 prompts add a system turn. `aime25`, `humaneval`, `mbpp`, `livecodebench` and `mt-bench` (first turn) use the evaluation exports. For Qwen3.8-27B these exports are rendered again with the target's chat template.
- `reproduce_paper_results.py`: the paper's Table 3 protocol on a GB200 pair. Default arms: AR, DSpark and JetSpec++. `--baselines all` adds DFlash and SSD for 8B. `--assets release` uses the published checkpoints. See `bench/jetspec_plus_plus/reproduce_paper_results.md`.

## Measured results

- **Paper, Table 3:** GB200 pair, batch size 1, temperature 0, 2048-token cap with EOS stopping, 128 prompts per benchmark. Throughput counts the steady rounds after each request's first decode round.
  - Qwen3-8B: 1,148.8 tok/s on GSM8K (6.30x over AR) and 1,361.1 tok/s on MATH-500 (7.45x). DSpark: 744.3 and 902.1 tok/s.
  - Qwen3.8-27B: 342.4 tok/s on GSM8K (5.68x) and 367.2 tok/s on MATH-500 (6.20x). DSpark: 249.6 and 279.8 tok/s.
  - Per round, the time beyond verification is 0.61 ms for JetSpec++ and 3.76 ms for DSpark (Qwen3-8B, GSM8K).
- **Paper, Table 1:** acceptance length on seven benchmarks at verification budgets 16 and 32, temperatures 0 and 1. At budget 16, JetSpec++ averages 8.94 tokens per round at temperature 0, against 6.89 for DSpark.
- **Reproduction:** `reproduce_paper_results.py` on a GB200 pair gives every tok/s cell within -3.1% to +3.4% of the paper, with tau and hit within 1.6%.
- **Release checkpoints, `speedup.py`:** Qwen3-8B on an A100 pair gives 7.56x on MATH-500 (AR 83.2, JetSpec++ 629.3 tok/s). On GSM8K, Qwen3-8B gives 5.78x on A100 and 5.86x on B200; Qwen3.8-27B gives 4.18x on A100 and 3.44x on B200.
- **Token streams:** greedy JetSpec++ output can differ from greedy AR output in bf16. Every divergence checked on MATH-500 (Qwen3-8B and Qwen3.8-27B) is a pair of tokens that are the target's fp32 top 2. The one-token AR kernels and the 16-token verify kernels round differently.

## JetSpec

- HF reference, the JetSpec engine, benchmark scripts and the vLLM integration: see the JetSpec section of the README.
- The HF reference benchmark needs `pip install -e '.[bench,flash-attn]'` for `--attn-implementation flash_attention_2`; `sdpa` runs without it.
- The engine benchmark (`bench/jetspec/engine/tps_walltime.py`) was measured on B200. With the README flags it runs out of memory on an A100 80GB, with or without `--warm-all`.

## Conventions

- Speedup is measured against AR on one GPU of the same type.
- Temperature 0, batch size 1 and EOS stopping unless a flag says otherwise.
