# parallel-tree-decoding

Lightweight **offline tree-speculative decoding** for LLM inference — a small drafter proposes multi-token, tree-structured drafts; the target verifies them in one batched forward, accepting the longest path consistent with its own logits. Single clone, single install, no submodules.

Built on top of HF `transformers` (the target is a standard `AutoModelForCausalLM`; the draft head subclasses the HF per-architecture model and shares the target's embedding + LM head). We supply the offline spec-decode loop, the tree construction + verify, and (later) a dedicated tree-attention kernel.

## Status

**Offline autoregressive baseline.** Plain offline Qwen3-8B greedy/temperature decode over an HF `DynamicCache` — the 1× denominator the speedup is measured against. The chain + tree speculative-decode paths build on it; the trained draft head + tree-attention kernel land next (see roadmap).

## Quickstart

```bash
git clone https://github.com/snyhlxde1/parallel-tree-decoding
cd parallel-tree-decoding
pip install -e .
python examples/simple_generate.py            # offline greedy on Qwen3-8B (needs a CUDA GPU)
```

```python
from ptd import LLM, SamplingParams

llm = LLM("Qwen/Qwen3-8B")
out = llm.generate("The three primary colors are", SamplingParams(temperature=0.0, max_new_tokens=64))
print(out["text"])
```

## Architecture

Two layers with a strict one-way dependency, so they can be owned and evolved independently:

- **`ptd/tree/` — the tree-drafting *method*** (engine-agnostic). Turns per-depth draft logits into a verification tree and selects the accepted path. Pure torch/numpy; imports nothing from the engine. Public contract: `get_algorithm(name).build(...) → DraftTree`, `build_ancestor_matrix(tree)`, `tree_accept(tree, target_logits, temperature)`.
- **`ptd/engine/` — the decode *substrate*** (`LLM`, KV cache, verify forward). Consumes `ptd.tree` one-way (engine → tree); the tree never imports the engine.

The tree is decoupled from the backend on purpose: the same `ptd.tree` plugs into this HF engine today and a serving-engine (vLLM / SGLang) integration later. Import the tree only through its public API (`ptd.tree`), never `ptd.tree._core`.

## Roadmap

| stage | what | status |
|---|---|---|
| **Offline baseline** | offline Qwen3-8B autoregressive decode | ✅ validated, byte-identical to HF greedy |
| **Chain spec decode** | `Drafter` + `LLM.generate_chain` (accept-longest-prefix) | ✅ validated, lossless |
| **Tree spec decode** | crossproduct + 4D ancestor mask + `tree_accept` | ✅ validated, lossless |
| **Trained draft head** | JF-trained `DraftHead` (`draft.py`, `draft_shift`/I-DLM param) → tokens-per-forward | planned (needs checkpoint) |
| **Tree-attention kernel** | `optimus_cutedsl` CuTe-DSL kernel → wall-clock TPS | vendored (Hopper / SM90 only) |
| **Fanout + bench + recipes** | per-depth fanout cap, benchmarks, recipes, HF checkpoint | planned |

> The baseline + chain + tree verify loops are **recompute-based** (correctness-first; KV-reuse is a later optimization). Speculative decoding is lossless, so the chain + tree paths were validated with stub drafters — output byte-identical to plain greedy — before the trained drafter checkpoint.

## Tests

```bash
PTD_TEST_MODEL=Qwen/Qwen3-8B pytest tests/   # the validation gate; needs CUDA + the model
```
The gate asserts the offline engine is token-identical to HF greedy generation and reuses the KV cache (the prefix is never reprocessed). On CPU/CI it is skipped.
