# AGENTS.md

This folder is the JetSpec++ torch reference. It holds the method in readable form, without the serving engine. The [root `AGENTS.md`](../../../AGENTS.md) has the commands and the results.

## Read one round

Start at `_prepare` in `drafter.py`. Then read `RoundDriver.run` in `round.py`, and `observe` back in `drafter.py`.

## Code map

| Path | Holds |
| --- | --- |
| `drafter.py` | The drafter that the shared torch core calls |
| `round.py` | One round, in `RoundDriver` |
| `context.py` | The hypothesis contexts |
| `method/` | Row selection, Jacobi passes, context refresh, anchor replacement, pins, acceptance |
| `models/` | The adapter, the anchor decoder and the draft model |
| `artifacts.py` | Artifact loaders and hash checks |
| `reader.py` | The acceptance-length reader |

## Check a change

- Run `bench/jetspec_plus_plus/verify.py` with `--dtype fp32`. It compares every output token with plain AR greedy generation and exits nonzero on a mismatch.
- Use `--device cpu` with full precision for a local check.
- Make the same change in `jetspec/engines/jetspec_plus_plus/`. The engine runs the same method with CUDA graphs and fused kernels.
