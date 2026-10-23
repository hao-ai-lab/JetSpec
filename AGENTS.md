# AGENTS.md

This file tells coding agents how this repository is organized and how to run each path. The README keeps only the shortest commands. Everything else is here.

> Draft outline. Each section will list the exact commands, flags and defaults.

## Repository map

- `jetspec/methods/jetspec_plus_plus/`: JetSpec++ torch reference (adapter, decoder, refinement passes, rounds).
- `jetspec/engines/jetspec_plus_plus/`: JetSpec++ serving engine (pipelined draft and verify on two GPUs).
- `jetspec/methods/jetspec/`, `jetspec/engines/jetspec/`: JetSpec tree drafting and its engine.
- `bench/jetspec_plus_plus/`: `verify.py`, `tau.py`, `speedup.py`, `table3.py`.
- `tests/`: CPU tests, golden fixtures for Table 3.

## Environments

- Exact lock files per platform: x86_64 (A100, B200) and aarch64 (GB200).
- Install with `--no-deps`. A resolved install can change the output tokens.
- One persistent compiler cache per comparison (`--cache-root`).

## JetSpec++ serving engine

- `generate`: one prompt, Hugging Face ids, schedule picked from the GPU.
- Full flag set: `--model`, `--schedule`, `--engine fast|reference`, artifact paths, `--host-values`, `--devices`, `--stop-at-eos`.
- Supported compositions per GPU (A100, B200, GB200) and their default J/R schedules.

## JetSpec++ torch reference

- `examples/jetspec_plus_plus/simple_generate.py`: one prompt, AR equality check.
- `bench/jetspec_plus_plus/verify.py`: token equality with greedy AR over a prompt set.
- `bench/jetspec_plus_plus/tau.py`: acceptance length.

## Benchmarks

- `speedup.py`: JetSpec++ against graphed AR on the same GPU. `--schedule auto` tries each supported schedule. Seven bundled prompt sets.
- `table3.py`: the paper's Table 3 protocol on a GB200 pair. Default arms AR, DSpark and JetSpec++. DFlash and SSD through `--arms`. `--assets paper|release`, `--offsets`, `--resume`, `--reduce-only`, `--resolve-only`.

## Checkpoints and artifacts

- Hugging Face repositories, their layout (`adapter/`, `calibration/`) and pinned revisions.
- The manifest pins every artifact by size and hash. Loaders refuse a mismatch.
- Baseline checkpoints for DSpark, DFlash and SSD.

## JetSpec

- HF reference, the JetSpec engine, benchmark scripts and the vLLM integration: see the JetSpec section of the README.

## Tests

- `pytest tests/` runs on CPU. GPU checks are the commands above.

## Conventions

- Speedup is measured against AR on one GPU of the same type.
- Temperature 0, batch size 1 and EOS stopping unless a flag says otherwise.
