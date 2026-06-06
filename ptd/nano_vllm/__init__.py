"""nano_vllm — a minimal, self-contained high-throughput decode substrate.

A second decode *substrate* alongside ``ptd.engine`` (the reference HF/SDPA
engine). Where ``ptd.engine`` favors clarity and single-clone reproducibility,
``nano_vllm`` favors throughput: a paged KV-cache, a triton tree-attention
kernel, continuous batching, and a ``torch.compile``-fused + CUDA-graphed
tree-verify path — a minimal engine this repo owns, rather than depending on an
external serving fork.

It consumes the SAME engine-agnostic tree contract as ``ptd.engine``; the
one-way dependency stays (engine -> tree, never the reverse)::

    from ptd.tree import get_algorithm, build_ancestor_matrix, tree_accept

so every algorithm in ``ptd.tree`` runs unchanged on either substrate. The
choice of engine changes throughput — not what the tree builds, nor whether
decoding stays lossless.

Status: shipped (N0–N3 + the compiled / CUDA-graph tree-verify path). The SDPA
path is the default + lossless oracle; the kernel / compiled / cudagraph
backends are opt-in via ``NanoEngine(attn_backend=...)``. See ``DESIGN.md``.
"""
from ptd.engine.llm import SamplingParams
from ptd.nano_vllm.engine import NanoEngine

__all__ = ["NanoEngine", "SamplingParams"]
