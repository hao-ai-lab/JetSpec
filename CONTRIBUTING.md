# Contributing to JetSpec

Thanks for your interest in contributing. This guide covers local setup, how to
run the tests, and the PR process.

## Dev setup

Single clone, single install — no submodules.

```bash
git clone https://github.com/hao-ai-lab/JetSpec
cd JetSpec
pip install -e '.[test,jetspec-plus-plus]'  # base deps + method loaders + pytest
```

Optional extras (install only what you need):

- `pip install -e '.[kernel]'` — `triton`, for the JetSpec tree-attention
  kernel (GPU only; imported lazily, so the rest of the package works without it).
- `pip install -e '.[bench]'` — `datasets`, `psutil`, `ninja`, and `packaging`,
  for benchmark scripts and profiling helpers under `bench/jetspec/` and
  `bench/jetspec_plus_plus/` (local prompts do not need this extra).
- `pip install -e '.[jetspec-plus-plus]'` — `safetensors` and `huggingface_hub`,
  for the JetSpec++ artifact loaders.
- `pip install -e '.[flash-attn]'` — `flash-attn`, for FA2 reference benchmarks.

You can combine them: `pip install -e '.[test,kernel,bench,flash-attn]'`.
The project also declares `uv` build dependency metadata for `flash-attn`
(`psutil`, `packaging`, `ninja`).

## Repository layout

The code is organized by method over a shared HuggingFace reference core,
optimized inference engine, and target loaders, with a strict one-way dependency
(engine -> tree; the tree never imports an engine):

- **`jetspec/core/`** — the lightweight HF reference core: `LLM`, `ModelRunner`,
  sampling, and the tree-attention hook used by reference benchmarks.
- **`jetspec/engines/jetspec/`** — the optimized serving engine, organized into
  `serving/`, `kernels/`, `verify/`, `draft/`, `models/`, and `wire/` roles.
  `jetspec/inference_engine/` preserves the old public imports as identity shims.
- **`jetspec/methods/jetspec/`** — causal parallel tree drafting: tree algorithms,
  simple drafters, the draft head and trained draft-head adapter.
- **`jetspec/methods/jetspec/tree/`** — the engine-agnostic tree-construction *method*. Turns
  per-depth draft logits into a verification tree and selects the accepted path.
  Pure torch/numpy; imports nothing from an engine. **Import it only through the
  public API `jetspec.tree` — never `jetspec.methods.jetspec.tree._core`** (the `_core` package is
  internal and may change without notice).
- **`jetspec/models/`** — target/draft-head model loading and model utilities.
- **`jetspec/methods/jetspec_plus_plus/`** — the Qwen3-8B, B16 greedy torch
  method: J/R rounds, adapter, anchor decoder, refresh swap, pins, rows and reader.
- **`jetspec/tree/`**, **`jetspec/draft.py`**, **`jetspec/draft_head_adapter.py`**
  and **`jetspec/models/draft_head.py`** — compatibility shims to the JetSpec
  method package. Existing public imports stay supported; add method code in
  the by-method packages. The shared models package holds target loaders.

Top-level scripts are grouped by purpose:

- **`bench/jetspec/reference/`** — HF/reference benchmarks and raw HF FA2 sanity checks.
- **`bench/jetspec/engine/`** — optimized JetSpec engine throughput scripts.
- **`bench/jetspec/profiling/`** — profiling, diagnostics, probes, and profile-table builders.
- **`bench/jetspec_plus_plus/`** — user commands: `verify.py` checks AR equality
  and `tau.py` reads acceptance length. Legacy JetSpec benchmark locations remain
  forwarding scripts; document the by-method location first.
- **`examples/basic/`**, **`examples/tree/`**, **`examples/engine/`** — runnable
  examples grouped by scope.
- **`examples/jetspec_plus_plus/`** — one-prompt generation with local artifacts.
- **`tests/jetspec_plus_plus/parity/`** — developer-only torch/engine comparison
  and replay tools, separate from the user benchmark entry points.

## Running tests

### CPU Subset

The JetSpec++ CPU tests build tiny fp32 Qwen3 models locally (no model downloads)
or operate on plain tensors:

```bash
pytest tests/jetspec_plus_plus -q
```

Reference-equivalence tests read frozen JSON and safetensors goldens and need
no reference export. The maintainers generated these frozen values from the
JetSpec++ reference implementation; regeneration is not part of this repository.

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

- `tests/jetspec_plus_plus/` — reference equivalence, artifact loaders, J/R rounds,
  lossless tiny-model generation, import compatibility and benchmark commands.
- `tests/jetspec_plus_plus/parity/` — developer-only capture comparisons and
  engine replay tools; real-model runs need explicitly supplied artifacts.

## Pull request process

1. Fork and branch off `master`.
2. Make your change. Keep the diff focused — one logical change per PR.
3. Add or update a test for any behavior you change. CPU-testable behavior should
   have a CPU test; GPU-only behavior should be covered by a GPU-gated test with a
   note on how to run it.
4. Run the CPU subset locally (and the GPU gate if you have a GPU and touched
   model-path code).
5. Open the PR against `master` with a clear description of what changed and why.
   CI runs the CPU subset on every PR.

## Code style

- Match the surrounding code — naming, comment density, and import ordering follow
  the existing files. The codebase favors descriptive module/test docstrings that
  explain *why* a unit exists and what property it gates; mirror that.
- Keep the `engine → tree` dependency one-way; the tree layer must not import from
  `jetspec.core`, `jetspec.engines.jetspec`, or `jetspec.methods.jetspec.draft`.
- Commit messages are single-line and prefixed by a tag (`[FIX]`, `[FEAT]`,
  `[DOCS]`, `[CHORE]`, etc.) describing the change.
