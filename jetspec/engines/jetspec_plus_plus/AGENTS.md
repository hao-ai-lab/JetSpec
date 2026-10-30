# AGENTS.md

This folder is the JetSpec++ serving engine. Read the [README](README.md) first: it maps the code and follows a command to the first forward pass. The [root `AGENTS.md`](../../../AGENTS.md) has the commands and the results.

## Rules

- `config/` and `artifacts/` hold data, not code. Keep their hashes equal to the published checkpoints.
- Do not edit `third_party/`. The engine checks that code by hash.
- Share one `--cache-root` between reads that you compare.

## Check a change

- Run `python -m jetspec.engines.jetspec_plus_plus generate --model qwen3-8b "PROMPT"` on two GPUs.
- Run `bench/jetspec_plus_plus/speedup.py` to compare with graphed AR on the same GPU type.
