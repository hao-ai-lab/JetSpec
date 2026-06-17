"""JetFlow — a minimal, self-contained high-throughput decode substrate.

A second decode *substrate* alongside ``jetflow.core`` (the reference HF/SDPA
engine). Where ``jetflow.core`` favors clarity and single-clone reproducibility,
``JetFlow`` favors throughput: a paged KV-cache, a triton tree-attention
kernel, continuous batching, and a ``torch.compile``-fused + CUDA-graphed
tree-verify path — a minimal engine this repo owns, rather than depending on an
external serving fork.

It consumes the SAME engine-agnostic tree contract as ``jetflow.core``; the
one-way dependency stays (engine -> tree, never the reverse)::

    from jetflow.tree import get_algorithm, build_ancestor_matrix, tree_accept

so every algorithm in ``jetflow.tree`` runs unchanged on either substrate. The
choice of engine changes throughput — not what the tree builds, nor whether
decoding stays lossless.

Status: shipped (N0–N3 + the compiled / CUDA-graph tree-verify path). The SDPA
path is the default + lossless oracle; the kernel / compiled / cudagraph
backends are opt-in via ``JetFlowEngine(attn_backend=...)``. See ``DESIGN.md``.
"""
from jetflow.core.llm import SamplingParams
from jetflow.inference_engine.engine import JetFlowEngine

__all__ = ["JetFlowEngine", "SamplingParams"]
