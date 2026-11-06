# AGENTS.md

This is the eager one-GPU JetSpec++ route. Read the [README](README.md) for the round flow and commands. The [engine guide](../../AGENTS.md) maps the shared method and serving paths.

## Check a change

- Run `bench/jetspec_plus_plus/verify.py` with `--dtype fp32`. It compares every output token with plain AR greedy generation and exits nonzero on a mismatch.
- Use `--device cpu` with full precision for small local checks.
- Keep shared computations in `../method/`; this folder owns eager orchestration and local artifact loading. Check both eager and serving callers when changing the shared method.
