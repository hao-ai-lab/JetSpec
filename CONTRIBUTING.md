# Contributing to JetSpec

Thanks for your interest in contributing. This guide covers local setup and the PR process.

## Dev setup

Single clone, single install — no submodules.

```bash
git clone https://github.com/hao-ai-lab/JetSpec
cd JetSpec
pip install -e '.[jetspec-plus-plus]'  # base deps + method loaders
```

Optional extras (install only what you need):

- `pip install -e '.[kernel]'` — `triton`, for the JetSpec tree-attention
  kernel (GPU only; imported lazily, so the rest of the package works without it).
- `pip install -e '.[bench]'` — `datasets`, `psutil`, `ninja`, and `packaging`,
  for benchmark scripts under `bench/jetspec/` and
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
- **`jetspec/methods/jetspec/`** — causal parallel tree drafting: tree algorithms,
  simple drafters, the draft head and trained draft-head adapter.
- **`jetspec/methods/jetspec/tree/`** — the engine-agnostic tree-construction *method*. Turns
  per-depth draft logits into a verification tree and selects the accepted path.
  Pure torch/numpy; imports nothing from an engine. **Import it only through the
  public API `jetspec.tree` — never `jetspec.methods.jetspec.tree._core`** (the `_core` package is
  internal and may change without notice).
- **`jetspec/models/`** — target/draft-head model loading and model utilities.
- **`jetspec/methods/jetspec_plus_plus/`** — the JetSpec++ torch reference: J/R
  rounds, adapter, anchor decoder, refresh swap, pins, rows and reader.
- **`jetspec/engines/jetspec_plus_plus/`** — the JetSpec++ serving engine. Its
  [README](jetspec/engines/jetspec_plus_plus/README.md) maps the code and lists
  the steps to add a model.
- **`jetspec/methods/baselines/`** — the baseline methods used for comparison.
- **`jetspec/tree/`**, **`jetspec/draft.py`**, **`jetspec/draft_head_adapter.py`**
  and **`jetspec/models/draft_head.py`** — compatibility shims to the JetSpec
  method package. Existing public imports stay supported; add method code in
  the by-method packages. The shared models package holds target loaders.

Top-level scripts are grouped by purpose:

- **`bench/jetspec/reference/`** — HF/reference benchmarks and raw HF FA2 sanity checks.
- **`bench/jetspec/engine/`** — optimized JetSpec engine throughput scripts.
- **`bench/jetspec_plus_plus/`** — user commands: `verify.py` checks AR equality,
  `tau.py` reads acceptance length, `speedup.py` measures speedup over AR and
  `reproduce_paper_results.py` reproduces the paper's Table 3.
- **`examples/basic/`**, **`examples/tree/`**, **`examples/engine/`** — runnable
  examples grouped by scope.
- **`examples/jetspec_plus_plus/`** — one-prompt generation and launch scripts.

## Pull request process

1. Fork and branch off `master`.
2. Make your change. Keep the diff focused — one logical change per PR.
3. Open the PR against `master` with a clear description of what changed and why.

## Code style

- Match the surrounding code — naming, comment density, and import ordering follow
  the existing files. The codebase favors descriptive module/test docstrings that
  explain *why* a unit exists and what property it gates; mirror that.
- Keep the `engine → tree` dependency one-way; the tree layer must not import from
  `jetspec.core`, `jetspec.engines.jetspec`, or `jetspec.methods.jetspec.draft`.
- Commit messages are single-line and prefixed by a tag (`[FIX]`, `[FEAT]`,
  `[DOCS]`, `[CHORE]`, etc.) describing the change.
