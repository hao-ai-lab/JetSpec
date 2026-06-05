# nano_vllm — design + milestone ladder

An owned, minimal high-throughput engine substrate, sitting beside `ptd/engine`
(the HF/SDPA reference). Same one-way contract — it consumes `ptd.tree` and never
the reverse — so every tree algorithm runs unchanged; the engine choice changes
*throughput*, not what the tree builds or whether decoding stays lossless.

## Why a second engine

`ptd/engine` (HF + `DynamicCache` + SDPA 4D mask) is for **clarity, single-clone
reproducibility, and correctness**. It is single-stream and tops out where the HF
substrate does. The collaborator's vLLM fork gives the **throughput upper bound**
(measured: `ptd_crossproduct` ≡ the fork's native tree, **~7.8× decode** on b200) —
but it's an external, heavy dependency we don't own. `nano_vllm` is the **owned**
substrate that aims at that ceiling with no external serving dependency.

> Target to beat / approach: the fork's **7.8× decode_cuda_speedup** (gsm8k, B=255,
> Qwen3-8B + epoch6 head). nano is measured *against that ceiling*.

## What it reuses (don't rebuild)

- **Tree method** — `ptd.tree.build_from_topk(name, root, topk_tokens, topk_logprobs,
  budget, device, …)` (already public). The proposer hands per-depth top-k; the
  tree contract turns it into a `DraftTree`. Identical to the fork's adapter.
- **Verify accept** — `ptd.tree.tree_accept`, ancestor mask via
  `ptd.tree.build_ancestor_matrix`.
- **The #58 persistent-cache verify pattern** — `ptd/engine/llm.py
  `_generate_tree_kv_cached` + `_select_kv_cache`: forward only the tree nodes
  against a cached prefix, then gather the accepted root-to-leaf path's KV back to
  a linear prefix. nano's paged cache generalises exactly this gather to block
  storage. Port the *logic*, swap the storage.

## The distinguisher vs `ptd/engine`

| | `ptd/engine` | `nano_vllm` |
|---|---|---|
| KV store | HF `DynamicCache` (contiguous) | **paged** (fixed blocks + block table) |
| tree attn | SDPA 4D additive mask | 4D mask now; **tree-attn kernel** later |
| streams | single (batch=1) | **continuous batching** |
| target fwd | HF `model(...)` | HF `model(...)` initially, then own |

## Milestone ladder (mirrors how `ptd/engine` was built: M0→M1→M2)

- **N0 — single-stream AR over a paged KV cache.** A `PagedKVCache` (block pool +
  per-seq block table; `allocate / append / gather(positions) / free`) and a decode
  loop that reproduces `ptd/engine`'s plain `generate()` token-for-token. Gate:
  byte-identical greedy vs `ptd/engine` on CPU (tiny fp32 model) + b200.
- **N1 — single-stream tree spec on the paged cache.** Build via `build_from_topk`,
  verify the tree against the cached prefix, `tree_accept`, then **gather the
  accepted path's blocks** (the paged analogue of `_select_kv_cache`). Gate:
  lossless == greedy; accept_len matches `ptd/engine`'s `_generate_tree_kv_cached`.
- **N2 — continuous batching.** Multiple sequences sharing the block pool; a
  scheduler that admits/evicts; per-seq tree spec in the same step. This is the
  throughput unlock single-stream can't give. Gate: aggregate tok/s vs the fork at
  matched batch.
- **N3 — tree-attention kernel.** Replace the SDPA 4D mask with a paged tree-attn
  kernel (the fork uses a triton tree kernel; ours can start from that shape).
  Gate: decode_cuda_speedup approaching the fork's ~7.8×.

## Module layout (to be created at N0)

```
ptd/nano_vllm/
  paged_kv_cache.py   # block pool + block table; allocate/append/gather/free
  engine.py           # NanoEngine: prefill + decode loop (consumes ptd.tree)
  scheduler.py        # (N2) admit/evict across sequences
  README.md / DESIGN.md
```

## Open decisions (resolve at N0)

1. **Target forward** — wrap HF `model(...)` with a paged cache adapter (fast to
   N1, reuses the validated forward), or a from-scratch attention (more control,
   much more work). Recommend HF-wrapped through N2; own-attention only at N3.
2. **Block size** — 16 (matches the head's `block_size`; convenient) vs tuned.
3. **Paged gather for trees** — store tree nodes in scratch blocks during verify,
   then copy the accepted path into the sequence's blocks (mirrors `_select_kv_cache`
   but block-granular). Confirm against N1's lossless gate.

Status: design fixed; **N0 is the next implementation step** (greenfield, in-repo,
CPU-testable first). Not yet implemented.
