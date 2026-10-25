# JetSpec — high-throughput engine substrate

A second decode **substrate**, sitting alongside [`jetspec/core`](../engine) in the
same swappable-engine design. N0–N2b plus the N3 triton tree-attention kernel are
**shipped and lossless-verified** — paged KV-cache, continuous batching, and an owned
paged tree-attn kernel.

| substrate | optimizes for | status |
|---|---|---|
| `jetspec/core` | clarity, single-clone reproducibility (HF + SDPA) | ✅ implemented |
| `jetspec/inference_engine` | throughput (paged KV-cache, tree-attention kernel, batching) | ✅ shipped (N0–N2b + N3 kernel) |

Both consume the **same** engine-agnostic tree contract — `get_algorithm(...)`,
`build_ancestor_matrix(...)`, `tree_accept(...)` from [`jetspec.tree`](../tree) — with
a strict one-way dependency (engine → tree). So every tree algorithm runs
unchanged on either engine: the engine choice changes **throughput**, not what
the tree builds or whether decoding is lossless.

The goal is an owned, minimal high-throughput engine (single clone, no heavy
external dependency) that reaches serving-class numbers, complementing the
reference HF engine used for correctness and demos.
JetSpec follows the nano-vllm doctrine: the least code that reproduces
big-engine performance, adapted here to PTD's tree-speculative decode path.

See [`DESIGN.md`](./DESIGN.md) for the architecture, what it reuses (the
`jetspec.tree.build_from_topk` contract + the persistent-cache verify pattern),
the throughput target (the vLLM fork's measured 7.55× decode), and the N0→N3
milestone ladder. **N0–N2b plus the N3 triton tree-attention kernel are shipped
and lossless-verified;** with `torch.compile` + CUDA-graph verify the verify-only
`decode_cuda_speedup` reaches 7.31× (cudagraph) vs the fork's 7.55×.
