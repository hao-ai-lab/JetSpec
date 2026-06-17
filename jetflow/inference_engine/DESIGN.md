# JetFlow — design + milestone ladder

An owned, minimal high-throughput engine substrate, sitting beside `jetflow/core`
(the HF/SDPA reference). Same one-way contract — it consumes `jetflow.tree` and never
the reverse — so every tree algorithm runs unchanged; the engine choice changes
*throughput*, not what the tree builds or whether decoding stays lossless.

## Why a second engine

`jetflow/core` (HF + `DynamicCache` + SDPA 4D mask) is for **clarity, single-clone
reproducibility, and correctness**. It is single-stream and tops out where the HF
substrate does. The collaborator's vLLM fork gives the **throughput upper bound**
(measured: `jetflow_accum_logp` ≡ the fork's native tree, **7.55× decode**) —
but it's an external, heavy dependency we don't own. `JetFlow` is the **owned**
substrate that aims at that ceiling with no external serving dependency, and it now
reaches it: verify-only `decode_cuda_speedup` **7.31× (cudagraph) / 6.27× (compiled)**
vs the fork's **7.55×** (see "N3 result" below).
JetFlow follows the nano-vllm doctrine: the least code that reproduces
big-engine performance, with the JetFlow tree path kept local and lossless.

> Target to beat / approach: the fork's **7.55× decode_cuda_speedup** (gsm8k, B=255,
> Qwen3-8B + epoch6 head). JetFlow reaches parity: **7.31×** verify-only, cudagraph backend.

## What it reuses (don't rebuild)

- **Tree method** — `jetflow.tree.build_from_topk(name, root, topk_tokens, topk_logprobs,
  budget, device, …)` (already public). The proposer hands per-depth top-k; the
  tree contract turns it into a `DraftTree`. Identical to the fork's adapter.
- **Verify accept** — `jetflow.tree.tree_accept`, ancestor mask via
  `jetflow.tree.build_ancestor_matrix`.
- **The persistent-cache verify pattern** — `jetflow/core/llm.py
  `_generate_tree_kv_cached` + `_select_kv_cache`: forward only the tree nodes
  against a cached prefix, then gather the accepted root-to-leaf path's KV back to
  a linear prefix. JetFlow's paged cache generalises exactly this gather to block
  storage. Port the *logic*, swap the storage.

## The distinguisher vs `jetflow/core`

| | `jetflow/core` | `JetFlow` |
|---|---|---|
| KV store | HF `DynamicCache` (contiguous) | **paged** (fixed blocks + block table) |
| tree attn | SDPA 4D additive mask | 4D mask now; **tree-attn kernel** later |
| streams | single (batch=1) | **continuous batching** |
| target fwd | HF `model(...)` | HF `model(...)` initially, then own |

## Milestone ladder (mirrors how `jetflow/core` was built: M0→M1→M2)

- **N0 — single-stream AR over a paged KV cache.** A `PagedKVCache` (block pool +
  per-seq block table; `allocate / append / gather(positions) / free`) and a decode
  loop that reproduces `jetflow/core`'s plain `generate()` token-for-token. Gate:
  byte-identical greedy vs `jetflow/core` on CPU (tiny fp32 model) + b200.
- **N1 — single-stream tree spec on the paged cache.** Build via `build_from_topk`,
  verify the tree against the cached prefix, `tree_accept`, then **gather the
  accepted path's blocks** (the paged analogue of `_select_kv_cache`). Gate:
  lossless == greedy; accept_len matches `jetflow/core`'s `_generate_tree_kv_cached`.
- **N2 — continuous batching.** Multiple sequences sharing the block pool; a
  scheduler that admits/evicts; per-seq tree spec in the same step. This is the
  throughput unlock single-stream can't give. Gate: aggregate tok/s vs the fork at
  matched batch.
- **N3 — tree-attention kernel.** Replace the SDPA 4D mask with a paged tree-attn
  kernel (the fork uses a triton tree kernel; ours can start from that shape).
  Gate: decode_cuda_speedup approaching the fork's 7.55×. **Reached: 7.31×
  (cudagraph) verify-only.**

**Status: N0–N2b shipped + merged to `master`** (`9fc9123` / `498ebd0` / `bce70c5`
/ `8e00dc0`); **N3 kernel shipped on `feat/draft-head`** (`457df8e` metadata builder
· `b6158d5` triton kernel · `bdc665e` engine integration). JetFlow does paged,
continuous-batched, lossless tree-spec decode via `torch.compile` + CUDA-graph verify
with our own triton tree-attention kernel.

**N3 result (`attn_backend="triton_paged_tree"`):** the paged tree-attn kernel is
**correct + lossless** — `kernel == SDPA` on a random pool (30/30, fp32 2e-6 / bf16
8e-3) and token-identical end-to-end (`test_jetflow_kernel_e2e.py`, 13/13). It **reaches
parity** with the fork: with `torch.compile` + a CUDA-graph verify, the **verify-only
`decode_cuda_speedup` is 7.31× (cudagraph backend) / 6.27× (compiled backend) vs the
fork's 7.55×** (Qwen3-8B, gsm8k, tree budget 255, width 7, epoch6 distill head, bf16,
4-sample, single-stream). The residual gap to the fork is **accept_len** (6.96 vs
7.16), **not** engine efficiency — per-round verify ≈ per-token AR forward, ratio
≈ 0.95 (the fork's is ≈ 0.98). Losslessness is **by construction** (each verify row
is target-greedy; fp32 token-identical, lossless gate 21/21) but **not bitwise-equal
to AR greedy in bf16** — one borderline-argmax flip; fp32 is exact. The
`decode_cuda_speedup` is **verify-only**: the drafter is excluded from both legs via
a CUDA-event split, matching the fork's `decode_cuda_s` accounting (which wraps only
the target forward, not the drafter). Reproduce via `bench/engine/identical_fork_compare.py`.

> **Superseded:** an earlier eager-mode measurement (no `torch.compile` / CUDA graph)
> framed the kernel as a batch-scaling crossover — kernel/SDPA = 0.59× @ B8 · 0.80× @
> B16 · 1.04× @ B32, crossing over ~B=32, overhead-limited at small batch (per-layer
> block-table rebuild + H2D transfer + many small triton launches). That regime is
> superseded by the compile + CUDA-graph verify result above, which removes the
> per-step host overhead and reaches fork parity. SDPA stays the default + the
> correctness oracle.

## N3 — implementation plan (resumption)

**Goal:** replace the per-step dense-KV-reconstruct + SDPA 4D mask (in
`engine._batched_decode_forward` / `_batched_tree_verify_forward`) with a **paged
tree-attention kernel** that reads K/V straight from the block pool (via block
table / slot mapping) and applies the per-node ancestor mask + fused softmax — so
no padding waste, no dense copy. This is the lever toward
the fork's **7.55× decode** (reached: verify-only 7.31× cudagraph; see "N3 result" above).

**Why it's tractable despite being a custom kernel:**
1. **Correctness oracle, for free.** The SDPA path (N2) is already lossless-validated.
   The kernel must produce the *same* attention output → unit-test `kernel(q,k,v,
   block_table, ancestor_mask)` vs the SDPA result on small random tensors on b200,
   iterate until bitwise-close (fp32) / within-tolerance (bf16), THEN wire it in. No
   guessing — every iteration has a ground-truth check.
2. **Reference implementation.** The fork's working triton tree kernel —
   `refs`/GPU `/an/external/vllm/fork/vllm/v1/attention/backends/tree_attn.py` +
   `vllm/v1/attention/ops/triton_unified_attention.py` — is the shape to adapt &
   simplify (drop CUDA-graph capture / debug instrumentation), not invent.
3. **Cheap failure.** Work is pushed + on `master`; failed kernel attempts cost
   nothing — iterate/reset freely. (This is exactly why we shipped N2 first.)

**Approach (scout → implement → verify):**
- Component 1: a paged tree-attn kernel (triton) — signature ~`(q[B,Hq,N,D],
  paged_k, paged_v, block_table, ancestor_mask[B,N,N], past_lens) -> out[B,Hq,N,D]`.
- Component 2: a metadata builder (per-seq block table + slot mapping + packed
  ancestor matrix → kernel inputs) — pure compute, CPU-testable.
- Component 3: an attention backend hook so `ModelRunner.forward` routes to the
  kernel instead of SDPA when enabled (opt-in flag; SDPA path stays the default +
  the oracle).

**Gates:**
1. **Kernel == SDPA** on random inputs (b200) — the correctness gate. Then the full
   JetFlow suite (`test_jetflow_*`) must stay green with the kernel enabled (reuses the
   existing per-seq lossless gates as end-to-end correctness).
2. **Throughput** — `decode_cuda_speedup` / tok/s vs the SDPA path and vs the
   fork's 7.55×, at matched batch + budget.

**Caveats / constraints:**
- **GPU-bound.** Triton kernels can't be CPU-validated like the rest of JetFlow — N3
  (Triton's interpret mode helps for small debugging only.)
- **Hardest piece so far** — budget a focused multi-attempt session; correctness
  first (match SDPA), perf tuning (block sizes, memory coalescing) second.
- Keep it **opt-in** (a flag on the engine) so the validated SDPA path remains the
  default + the correctness oracle; flip the default to the kernel only once it
  matches SDPA + beats it on throughput.

## Module layout (to be created at N0)

```
jetflow/inference_engine/
  paged_kv_cache.py   # block pool + block table; allocate/append/gather/free
  engine.py           # JetFlowEngine: prefill + decode loop (consumes jetflow.tree)
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

Status: **N0→N2 merged to `master`; N3 triton tree-attn kernel shipped on
`feat/draft-head`** (correct + lossless). With `torch.compile` + CUDA-graph verify,
verify-only `decode_cuda_speedup` reaches **7.31× (cudagraph) / 6.27× (compiled)** vs
the fork's **7.55×** — parity; the residual gap is accept_len, not engine efficiency.
See the milestone-ladder status + N3 result above.
