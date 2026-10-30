# AGENTS.md

This folder is the JetSpec++ serving engine. It runs the method on two GPUs with CUDA graphs and fused kernels. The [root `AGENTS.md`](../../../AGENTS.md) has the commands and the results, and the [README](README.md) has the code map.

## Read the engine

1. Read "From the command to the first forward pass" in the README. It follows one launch.
2. `Scheduler` in `serving/scheduler.py` admits requests.
3. `SpecDecodeStep` in `verify/step.py` drives verification on the target GPU.
4. `ChainRoundDriver` in `draft/driver.py` runs the draft passes on the draft GPU.
5. `wire/` carries packets between the two GPUs.

To read the method without the engine, use `jetspec/methods/jetspec_plus_plus/`.

## Rules

- `config/` and `artifacts/` hold data, not code. Keep their hashes equal to the published checkpoints.
- Do not edit `third_party/`. The engine checks that code by hash.
- Share one `--cache-root` between reads that you compare.

## Check a change

- Run `python -m jetspec.engines.jetspec_plus_plus generate --model qwen3-8b "PROMPT"` on two GPUs.
- Run `bench/jetspec_plus_plus/speedup.py` to compare with graphed AR on the same GPU type.
