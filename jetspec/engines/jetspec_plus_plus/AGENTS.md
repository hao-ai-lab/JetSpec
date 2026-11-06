# AGENTS.md

This folder holds the JetSpec++ method, serving engine and eager reference route. Read the [README](README.md) first: it maps the code and follows a command to the first forward pass. The [root `AGENTS.md`](../../../AGENTS.md) has the commands and the results.

## Rules

- `config/` and `artifacts/` hold data, not code. Keep their hashes equal to the published checkpoints.
- Do not edit `bench/jetspec_plus_plus/third_party/`. The comparison code is available from a source checkout and is checked by hash when selected.
- Share one `--cache-root` between reads that you compare.

## Check a change

- Run `python -m jetspec.engines.jetspec_plus_plus generate --model qwen3-8b "PROMPT"` on two GPUs.
- Run `bench/jetspec_plus_plus/speedup.py` to compare with graphed AR on the same GPU type.

## Eager reference route

`draft/reference/` contains the one-GPU round loop, local artifact loaders and reader. It uses `draft/method/` for adapter, decoder, pass, schedule, refresh and row-selection computations. See its [guide](draft/reference/README.md) for token verification and acceptance-length commands.
