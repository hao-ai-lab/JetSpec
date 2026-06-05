# nano_vllm — high-throughput engine substrate (placeholder)

This package reserves the structure for a second decode **substrate**, sitting
alongside [`ptd/engine`](../engine) in the same swappable-engine design:

| substrate | optimizes for | status |
|---|---|---|
| `ptd/engine` | clarity, single-clone reproducibility (HF + SDPA) | ✅ implemented |
| `ptd/nano_vllm` | throughput (paged KV-cache, tree-attention kernel, batching) | 🚧 reserved |

Both consume the **same** engine-agnostic tree contract — `get_algorithm(...)`,
`build_ancestor_matrix(...)`, `tree_accept(...)` from [`ptd.tree`](../tree) — with
a strict one-way dependency (engine → tree). So every tree algorithm runs
unchanged on either engine: the engine choice changes **throughput**, not what
the tree builds or whether decoding is lossless.

The goal is an owned, minimal high-throughput engine (single clone, no heavy
external dependency) that reaches serving-class numbers, complementing the
reference HF engine used for correctness and demos.

See [`DESIGN.md`](./DESIGN.md) for the architecture, what it reuses (the
`ptd.tree.build_from_topk` contract + the #58 persistent-cache verify pattern),
the throughput target (the vLLM fork's measured ~7.8× decode), and the N0→N3
milestone ladder. **N0 (single-stream AR over a paged KV cache) is the next
implementation step.**
