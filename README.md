# parallel-tree-decoding

Lightweight **offline tree-speculative decoding** for LLM inference — a small drafter proposes multi-token, tree-structured drafts; the target verifies them in one batched forward, accepting the longest path consistent with its own logits. Single clone, single install, no submodules.

Built on top of HF `transformers` (the target is a standard `AutoModelForCausalLM`; the draft head subclasses the HF per-architecture model and shares the target's embedding + LM head). We supply the offline spec-decode loop, the tree construction + verify, and (later) a dedicated tree-attention kernel.

## Status

**Shipped: paged, continuous-batched, lossless tree-speculative decode.** The `nano_vllm` engine does paged-KV, continuous-batched, lossless tree-spec decode via `torch.compile` + CUDA-graph verify and drafter, fused GEMMs, cross-prompt session reuse, and our own triton tree-attention kernel. The trained draft head is published at HF `Snyhlxde/ptd-qwen3-8b-distill-epoch6-3e-4-no-gamma`. On Qwen3-8B (B200, bf16, single-stream) it reaches **96–103% of the reference vLLM-based fork's wall-clock TPS** — 738.6 vs 718.9 tok/s on humaneval (above it), 791.0 vs 820.7 on gsm8k, 910.3 vs 930.3 on math500 — from an engine core of ~3.8k lines vs vLLM's ~560k. See [Results](#results). The HF + SDPA `ptd/engine` substrate remains the single-clone correctness reference.

## Quickstart

```bash
git clone https://github.com/snyhlxde1/parallel-tree-decoding
cd parallel-tree-decoding
pip install -e '.[kernel]'                    # [kernel] pulls triton for the tree-spec path
python examples/tree_spec_generate.py         # trained-head tree-spec on Qwen3-8B (needs a CUDA GPU)
python examples/simple_generate.py            # offline greedy baseline
```

```python
from ptd import load_draft_head, DraftHeadTreeDrafter
from ptd.nano_vllm import NanoEngine, SamplingParams

# Compiled tree-attention verify path (the contribution); "triton_paged_tree" runs it un-compiled.
engine = NanoEngine("Qwen/Qwen3-8B", attn_backend="triton_paged_tree_compiled")
head = load_draft_head("Snyhlxde/ptd-qwen3-8b-distill-epoch6-3e-4-no-gamma")
drafter = DraftHeadTreeDrafter(head, target=engine.model, block_size=head.block_size,
                               target_layer_ids=head.target_layer_ids)
out = engine.generate_tree("The three primary colors are", drafter,
                           block_size=head.block_size, tree_width=7, budget=63,
                           target_layer_ids=head.target_layer_ids,
                           sampling_params=SamplingParams(temperature=0.0, max_new_tokens=64))
print(out["text"], "\ntokens-per-forward:", out["tpf"])
```

```python
from ptd import LLM, SamplingParams           # the SDPA reference baseline

llm = LLM("Qwen/Qwen3-8B")
out = llm.generate("The three primary colors are", SamplingParams(temperature=0.0, max_new_tokens=64))
print(out["text"])
```

## Architecture

Two layers with a strict one-way dependency, so they can be owned and evolved independently:

- **`ptd/tree/` — the tree-drafting *method*** (engine-agnostic). Turns per-depth draft logits into a verification tree and selects the accepted path. Pure torch/numpy; imports nothing from the engine. Public contract: `get_algorithm(name).build(...) → DraftTree`, `build_ancestor_matrix(tree)`, `tree_accept(tree, target_logits, temperature)`.
- **`ptd/engine/` — the decode *substrate*** (`LLM`, KV cache, verify forward). Consumes `ptd.tree` one-way (engine → tree); the tree never imports the engine.

The tree is decoupled from the backend on purpose: the same `ptd.tree` plugs into this HF engine today and a serving-engine (vLLM / SGLang) integration later. Import the tree only through its public API (`ptd.tree`), never `ptd.tree._core`.

## Results

The `nano_vllm` engine ships paged, lossless tree-spec decode via `torch.compile` + CUDA-graph verify and drafter, fused qkv/gate-up GEMMs, cross-prompt session reuse, and our own triton tree-attention kernel. Headline numbers — **wall-clock tokens/sec**, the number a user actually sees (Qwen3-8B, B200, bf16, single-stream, tree budget 127, width 7, trained epoch6 distill head, 2048-token generation window, full-dataset sample counts):

| dataset | this engine | reference fork (full vLLM) | ratio | accept_len (ours / fork) |
|---|---|---|---|---|
| humaneval (164) | **738.6** | 718.9 | **1.03×** | 7.25 / 7.23 |
| math500 (100) | **910.3** | 930.3 | 0.98× | 9.60 / 9.76 |
| gsm8k (64) | **791.0** | 820.7 | 0.96× | 7.72 / 8.01 |

- Both engines run the same published draft head, the same prompts, the same budget, on the same GPU; the fork rows are our own measurements of its production configuration (triton kernel + logical KV layout + CUDA graphs), not paper claims.
- **~5.8× wall-clock speedup** over the same stack's autoregressive decode.
- The engine core is **~3.8k lines** (vs ~560k lines of Python in vLLM): the condensation is the point — fork-class throughput from a codebase you can read in an afternoon.
- **Lossless:** fp32 token-identical to an SDPA oracle; bf16 is lossless-by-construction (each accepted token is target-greedy) but not bitwise-equal to AR greedy — borderline-argmax flips move with kernel reduction order. fp32 is exact.

Production configuration: `NANO_FUSE_GEMMS=1`, `attn_backend="triton_paged_tree_cudagraph_nogather"`, graphed drafter, `session=True`. Reproduce the table with `bench/tps_walltime.py` (per-dataset fingerprints: `bench/tree_diag.py`); verify-only GPU-time comparison: `bench/identical_fork_compare.py`.

## Roadmap

| stage | what | status |
|---|---|---|
| **Offline baseline** | offline Qwen3-8B autoregressive decode | ✅ validated, byte-identical to HF greedy |
| **Chain spec decode** | `Drafter` + `LLM.generate_chain` (accept-longest-prefix) | ✅ validated, lossless |
| **Tree spec decode** | crossproduct + 4D ancestor mask + `tree_accept` | ✅ validated, lossless |
| **Trained draft head** | JF-trained `DraftHead` (`draft.py`, `draft_shift`/I-DLM param) → tokens-per-forward | ✅ shipped (head at HF `Snyhlxde/ptd-qwen3-8b-distill-epoch6-3e-4-no-gamma`) |
| **Tree-attention kernel** | owned paged triton tree-attention kernel → wall-clock TPS | ✅ shipped, lossless-verified |
| **Fanout + bench + recipes** | per-depth fanout cap, benchmarks, recipes, HF checkpoint | ✅ shipped |

> The baseline + chain + tree verify loops are **recompute-based** (correctness-first; KV-reuse is a later optimization). Speculative decoding is lossless, so the chain + tree paths were validated with stub drafters — output byte-identical to plain greedy — before the trained drafter checkpoint.

## Tests

```bash
PTD_TEST_MODEL=Qwen/Qwen3-8B pytest tests/   # the validation gate; needs CUDA + the model
```
The gate asserts the offline engine is token-identical to HF greedy generation and reuses the KV cache (the prefix is never reprocessed). On CPU/CI it is skipped.
