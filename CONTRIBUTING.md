# Contributing to jetflow

Thanks for your interest in contributing. This guide covers local setup, how to
run the tests, and the PR process.

## Dev setup

Single clone, single install — no submodules.

```bash
git clone https://github.com/snyhlxde1/jetflow
cd jetflow
pip install -e '.[test]'      # base deps + pytest (CPU test subset)
```

Optional extras (install only what you need):

- `pip install -e '.[kernel]'` — `triton`, for the `JetFlow` tree-attention
  kernel (GPU only; imported lazily, so the rest of the package works without it).
- `pip install -e '.[bench]'` — `datasets`, for the benchmark scripts under `bench/`.

You can combine them: `pip install -e '.[test,kernel,bench]'`.

## Repository layout

The code is organized as two decode engines plus an engine-agnostic tree-method
layer, with a strict one-way dependency (engine → tree; the tree never imports an
engine):

- **`jetflow/core/`** — the SDPA reference engine. HF `transformers` + SDPA; the
  single-clone correctness oracle. Its plain offline Qwen3-8B decode is the 1×
  denominator that speedups are measured against.
- **`jetflow/inference_engine/`** — the high-throughput engine: paged KV cache, a triton
  tree-attention kernel, and a `torch.compile` + CUDA-graph verify path.
- **`jetflow/tree/`** — the engine-agnostic tree-construction *method*. Turns
  per-depth draft logits into a verification tree and selects the accepted path.
  Pure torch/numpy; imports nothing from an engine. **Import it only through the
  public API `jetflow.tree` — never `jetflow.tree._core`** (the `_core` package is
  internal and may change without notice).

## Running tests

### CPU subset (no GPU, no triton)

These are the tests CI runs. They build a tiny fp32 Qwen3 locally (no network)
or operate on plain tensors, so they pass on any CPU-only box:

```bash
pytest -q \
  tests/tree/test_build_from_topk.py \
  tests/tree/test_depth_rank_histogram.py \
  tests/inference_engine/test_jetflow_attn_metadata.py \
  tests/inference_engine/test_jetflow_batch.py \
  tests/inference_engine/test_jetflow_paged_multiseq.py \
  tests/inference_engine/test_jetflow_tree.py \
  tests/inference_engine/test_jetflow_tree_batch.py
```

### Full GPU gate

The remaining tests load a real model (offline decode, engine parity, draft-head
and compiled-verify losslessness) and need a CUDA GPU. Point them at the target
model via `JETFLOW_TEST_MODEL` and run the whole suite:

```bash
JETFLOW_TEST_MODEL=Qwen/Qwen3-8B pytest tests/
```

The losslessness tests are exact in fp32; in bf16 a block/compiled verify can flip
a borderline argmax after tens of exact tokens (a known bf16 caveat), so those
gates are run on the GPU box, not in CPU CI.

## Pull request process

1. Fork and branch off `main`.
2. Make your change. Keep the diff focused — one logical change per PR.
3. Add or update a test for any behavior you change. CPU-testable behavior should
   have a CPU test; GPU-only behavior should be covered by a GPU-gated test with a
   note on how to run it.
4. Run the CPU subset locally (and the GPU gate if you have a GPU and touched
   model-path code).
5. Open the PR against `main` with a clear description of what changed and why.
   CI runs the CPU subset on every PR.

## Code style

- Match the surrounding code — naming, comment density, and import ordering follow
  the existing files. The codebase favors descriptive module/test docstrings that
  explain *why* a unit exists and what property it gates; mirror that.
- Keep the `engine → tree` dependency one-way; the tree layer must not import from
  `jetflow.core`, `jetflow.inference_engine`, or `jetflow.draft`.
- Commit messages are single-line and prefixed by a tag (`[FIX]`, `[FEAT]`,
  `[DOCS]`, `[CHORE]`, etc.) describing the change.
