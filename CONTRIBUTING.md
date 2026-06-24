# Contributing to JetSpec

Thanks for your interest in contributing. This guide covers local setup, how to
run the tests, and the PR process.

## Dev setup

Single clone, single install — no submodules.

```bash
git clone https://github.com/hao-ai-lab/JetSpec
cd parallel-tree-decoding
pip install -e '.[test]'      # base deps + pytest (CPU test subset)
```

Optional extras (install only what you need):

- `pip install -e '.[kernel]'` — `triton`, for the JetSpec tree-attention
  kernel (GPU only; imported lazily, so the rest of the package works without it).
- `pip install -e '.[bench]'` — `datasets`, `psutil`, `ninja`, and `packaging`,
  for benchmark scripts and profiling helpers under `bench/`.
- `pip install -e '.[flash-attn]'` — `flash-attn`, for FA2 reference benchmarks.

You can combine them: `pip install -e '.[test,kernel,bench,flash-attn]'`.
The project also declares `uv` build dependency metadata for `flash-attn`
(`psutil`, `packaging`, `ninja`).

## Repository layout

The code is organized as a HuggingFace reference core, an optimized inference
engine, and an engine-agnostic tree layer, with a strict one-way dependency
(engine -> tree; the tree never imports an engine):

- **`jetspec/core/`** — the lightweight HF reference core: `LLM`, `ModelRunner`,
  sampling, and the tree-attention hook used by reference benchmarks.
- **`jetspec/inference_engine/`** — the optimized serving engine: paged KV cache,
  scheduler, Triton tree attention, and CUDA graph paths.
- **`jetspec/tree/`** — the engine-agnostic tree-construction *method*. Turns
  per-depth draft logits into a verification tree and selects the accepted path.
  Pure torch/numpy; imports nothing from an engine. **Import it only through the
  public API `jetspec.tree` — never `jetspec.tree._core`** (the `_core` package is
  internal and may change without notice).
- **`jetspec/models/`** — target/draft-head model loading and model utilities.
- **`jetspec/draft.py`** and **`jetspec/draft_head_adapter.py`** — simple test
  drafters and trained draft-head adapters.

Top-level scripts are grouped by purpose:

- **`bench/reference/`** — HF/reference benchmarks and raw HF FA2 sanity checks.
- **`bench/engine/`** — optimized JetSpec engine throughput scripts.
- **`bench/profiling/`** — profiling, diagnostics, probes, and profile-table builders.
- **`examples/basic/`**, **`examples/tree/`**, **`examples/engine/`** — runnable
  examples grouped by scope.
- **`tests/manual/profiling/`** — scratch/manual profiling utilities that are not
  part of normal benchmark entry points.

## Running tests

### CPU Subset

These are the tests CI runs. They build a tiny fp32 Qwen3 locally (no network)
or operate on plain tensors, so they pass on any CPU-only box:

```bash
pytest -q \
  tests/tree/test_build_from_topk.py \
  tests/tree/test_depth_rank_histogram.py \
  tests/inference_engine/test_jetspec_attn_metadata.py \
  tests/inference_engine/test_jetspec_batch.py \
  tests/inference_engine/test_jetspec_paged_multiseq.py \
  tests/inference_engine/test_jetspec_tree.py \
  tests/inference_engine/test_jetspec_tree_batch.py
```

### Full Suite

Run everything locally with:

```bash
PYTHONPATH=. pytest tests/
```

Most tests use tiny local models. Real-model gates are opt-in and require CUDA
plus explicit model/checkpoint environment variables:

```bash
CUDA_VISIBLE_DEVICES=0 \
JETSPEC_TEST_MODEL=Qwen/Qwen3-8B \
JETSPEC_DRAFT_HEAD=/path/to/draft-head-or-hf-repo \
PYTHONPATH=. pytest tests/
```

The fp32 gates are token-identical. In bf16, a block/tree verify can flip a
borderline argmax after many exact tokens due to kernel reduction order; GPU
gates account for that caveat.

### Test Buckets

- `tests/core/` — HF core generation, draft-head adapter, and KV tree verify.
- `tests/tree/` — tree algorithms, registry, top-k construction, and acceptance.
- `tests/inference_engine/` — paged KV, batching, kernels, CUDA graph helpers,
  and JetSpec engine parity against the HF core.
- `tests/bench/` — benchmark/profile helper unit tests.
- `tests/integration/` — real-model or end-to-end parity checks.

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
  `jetspec.core`, `jetspec.inference_engine`, or `jetspec.draft`.
- Commit messages are single-line and prefixed by a tag (`[FIX]`, `[FEAT]`,
  `[DOCS]`, `[CHORE]`, etc.) describing the change.
