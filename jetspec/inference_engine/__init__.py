"""JetSpec — a minimal, self-contained high-throughput decode substrate.

A second decode *substrate* alongside ``jetspec.core`` (the reference HF/SDPA
engine). Where ``jetspec.core`` favors clarity and single-clone reproducibility,
``JetSpec`` favors throughput: a paged KV-cache, a triton tree-attention
kernel, continuous batching, and a ``torch.compile``-fused + CUDA-graphed
tree-verify path — a minimal engine this repo owns, rather than depending on an
external serving fork.

It consumes the SAME engine-agnostic tree contract as ``jetspec.core``; the
one-way dependency stays (engine -> tree, never the reverse)::

    from jetspec.tree import get_algorithm, build_ancestor_matrix, tree_accept

so every algorithm in ``jetspec.tree`` runs unchanged on either substrate. The
choice of engine changes throughput — not what the tree builds, nor whether
decoding stays lossless.

Status: shipped (N0–N3 + the compiled / CUDA-graph tree-verify path). The SDPA
path is the default + lossless oracle; the kernel / compiled / cudagraph
backends are opt-in via ``JetSpecEngine(attn_backend=...)``. See ``DESIGN.md``.
"""
from jetspec.core.llm import SamplingParams
from jetspec.inference_engine.engine import JetSpecEngine

__all__ = ["JetSpecEngine", "SamplingParams"]
