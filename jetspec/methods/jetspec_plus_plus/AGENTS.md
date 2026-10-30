# AGENTS.md

This folder is the JetSpec++ torch reference. Read the [README](README.md) first: it walks through one round and maps the files. The [root `AGENTS.md`](../../../AGENTS.md) has the commands and the results.

## Check a change

- Run `bench/jetspec_plus_plus/verify.py` with `--dtype fp32`. It compares every output token with plain AR greedy generation and exits nonzero on a mismatch.
- Use `--device cpu` with full precision for a local check.
- Make the same change in `jetspec/engines/jetspec_plus_plus/`. The engine runs the same method with CUDA graphs and fused kernels.
