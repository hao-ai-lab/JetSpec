"""nano_vllm — planned high-throughput engine substrate (placeholder).

A second decode *substrate* alongside ``ptd.engine`` (the reference HF/SDPA
engine). Where ``ptd.engine`` favors clarity and single-clone reproducibility,
``nano_vllm`` will favor throughput: paged KV-cache, a tree-attention kernel,
and continuous batching — a minimal, self-contained engine this repo owns,
rather than depending on an external serving fork.

It will consume the SAME engine-agnostic tree contract as ``ptd.engine``; the
one-way dependency stays (engine -> tree, never the reverse)::

    from ptd.tree import get_algorithm, build_ancestor_matrix, tree_accept

so every algorithm in ``ptd.tree`` runs unchanged on either substrate. The
choice of engine changes throughput — not what the tree builds, nor whether
decoding stays lossless.

Status: reserved. Not implemented yet.
"""

__all__: list[str] = []
