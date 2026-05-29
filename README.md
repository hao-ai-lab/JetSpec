# parallel-tree-decoding

Lightweight **offline tree-speculative decoding** for LLM inference — a small drafter proposes multi-token, tree-structured drafts; the target verifies them in one batched forward, accepting the longest path consistent with its own logits. Single clone, single install, no submodules.

Built on top of HF `transformers` (the target is a standard `AutoModelForCausalLM`; the draft head subclasses the HF per-architecture model and shares the target's embedding + LM head). We supply the offline spec-decode loop, the tree construction + verify, and (later) a Triton tree-attention kernel.

## Status

**M0 — the 1× autoregressive baseline.** Plain offline Qwen3-8B greedy/temperature decode over an HF `DynamicCache`. This is the denominator the speedup is measured against; the draft head + tree verify land next (see roadmap).

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

## Roadmap

| milestone | what | status |
|---|---|---|
| **M0** | offline Qwen3-8B autoregressive baseline (this) | ✅ |
| **M1** | draft head (`draft.py`, JF-trained causal head, `draft_shift`/I-DLM param) + crossproduct tree + verify → tokens-per-forward | next |
| **M2** | Triton tree-attention kernel → wall-clock TPS | — |
| **M3** | V5 (per-depth top-2-gap fanout) + bench + recipes + checkpoint | — |

## Tests

```bash
PTD_TEST_MODEL=Qwen/Qwen3-8B pytest tests/   # the M0 gate; needs CUDA + the model
```
The gate asserts the offline engine is token-identical to HF greedy generation and reuses the KV cache (the prefix is never reprocessed). On CPU/CI it is skipped.
