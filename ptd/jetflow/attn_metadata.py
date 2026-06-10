"""Paged tree-attention metadata builder (JetFlow N3, Unit 2).

Pure-compute layer that turns JetFlow's per-sequence engine state into the exact
input tensors the paged tree-attention triton kernel (`paged_tree_attn`, Unit 1)
consumes. The kernel reads K/V straight from `PagedKVCache`'s block pool via a
batched block table and folds the per-tree ancestor relation in as an additive
bias; this module packs that block table + the cumulative query/key lengths +
the block-diagonal `qq_bias` and nothing else (no triton, no CUDA, no model).

The packing is the dense-mask construction of `engine.py` re-expressed for the
kernel:

  - `block_table[s]` indexes the SAME logical KV positions that
    `PagedKVCache._logical_kv(layer, seq_id=s)` reconstructs — the kernel's slot
    math (`pos // block_size`, `pos % block_size`) walks this row for
    `pos in range(seq_lens_k[s])`.
  - `qq_bias` is the tree-node sub-block of the engine's 4D mask
    (`build_ancestor_matrix(tree)` mapped to 0 / -inf), made block-diagonal so a
    seq never attends across to another seq's query nodes. Prefix keys are always
    visible and handled by the kernel, so they are NOT encoded here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import torch

from ptd.tree import DraftTree, build_ancestor_matrix


@dataclass
class AttnMeta:
    """The four tensors the paged tree-attention kernel consumes.

    block_table   : (num_seqs, max_blocks) int32 — per-seq physical block ids in
                    logical order, right-padded to the batch-max block count.
    cu_seqlens_q  : (num_seqs + 1,) int32 — cumulative query-row counts; cu[0]=0,
                    cu[-1] == total_q.
    seq_lens_k    : (num_seqs,) int32 — total key length per seq (past + this
                    step's query nodes).
    qq_bias       : (total_q, total_q) fp32 additive bias (0 allowed / -inf
                    disallowed), block-diagonal per seq, or None when every seq
                    is a pure decode step (the kernel's no-bias fast path).
    """
    block_table: torch.Tensor
    cu_seqlens_q: torch.Tensor
    seq_lens_k: torch.Tensor
    qq_bias: Optional[torch.Tensor]


def _query_count(tree: Optional[DraftTree], num_query_nodes: Optional[int]) -> int:
    """Query rows a seq contributes: tree.num_nodes, or the decode override / 1."""
    if tree is not None:
        return int(tree.num_nodes)
    return 1 if num_query_nodes is None else int(num_query_nodes)


def build_attn_metadata(
    seq_ids: Sequence[int],
    block_tables_per_seq: Sequence[Sequence[int]],
    past_lens: Sequence[int],
    trees: Sequence[Optional[DraftTree]],
    *,
    num_query_nodes: Optional[Sequence[Optional[int]]] = None,
    device: torch.device = torch.device("cpu"),
) -> AttnMeta:
    """Pack per-sequence engine state into the kernel's input tensors.

    Args:
        seq_ids: batch-ordered sequence ids (used only for length validation; the
            block tables are passed in explicitly so this stays cache-agnostic).
        block_tables_per_seq: per seq, the physical block-id list for the target
            layer (i.e. `cache._seq_block_tables[seq_id][layer]`), in logical
            order. Row s of `block_table` is this list, right-padded.
        past_lens: per seq, the cached prefix length (`cache.get_seq_length`).
        trees: per seq, the `DraftTree` for a tree-verify step, or None for a pure
            decode step (one query row).
        num_query_nodes: optional per-seq override for the decode query count
            (None entries fall back to 1); ignored where `trees[s]` is not None.
        device: device for the produced tensors (CPU by default).

    Returns:
        AttnMeta with `block_table`, `cu_seqlens_q`, `seq_lens_k`, and `qq_bias`
        (None when every seq is pure decode).
    """
    num_seqs = len(seq_ids)
    if not (len(block_tables_per_seq) == len(past_lens) == len(trees) == num_seqs):
        raise ValueError("seq_ids, block_tables_per_seq, past_lens, trees must align")
    if num_query_nodes is not None and len(num_query_nodes) != num_seqs:
        raise ValueError("num_query_nodes must align with seq_ids when given")

    nq = [
        _query_count(trees[s], None if num_query_nodes is None else num_query_nodes[s])
        for s in range(num_seqs)
    ]

    # cu_seqlens_q: cumulative query rows (cu[0] = 0).
    cu = torch.zeros(num_seqs + 1, dtype=torch.int32, device=device)
    if num_seqs:
        cu[1:] = torch.tensor(nq, dtype=torch.int32, device=device).cumsum(0)
    total_q = int(cu[-1].item())

    # seq_lens_k: total key length = prefix + this step's query nodes.
    seq_lens_k = torch.tensor(
        [int(past_lens[s]) + nq[s] for s in range(num_seqs)],
        dtype=torch.int32, device=device,
    )

    # block_table: per-seq block ids, right-padded to the batch-max block count
    # (pad value 0 — the kernel never reads past seq_lens_k[s]).
    max_blocks = max((len(t) for t in block_tables_per_seq), default=0)
    block_table = torch.zeros((num_seqs, max_blocks), dtype=torch.int32, device=device)
    for s, table in enumerate(block_tables_per_seq):
        if table:
            block_table[s, : len(table)] = torch.tensor(table, dtype=torch.int32, device=device)

    # qq_bias: block-diagonal (total_q, total_q); per-seq block = ancestor matrix
    # mapped to 0 / -inf, off-diagonal blocks all -inf. None if every seq decodes.
    if any(trees[s] is not None for s in range(num_seqs)):
        qq_bias = torch.full((total_q, total_q), float("-inf"), dtype=torch.float32, device=device)
        for s in range(num_seqs):
            lo = int(cu[s].item())
            hi = int(cu[s + 1].item())
            if trees[s] is not None:
                anc = build_ancestor_matrix(trees[s]).to(device=device, dtype=torch.bool)
                block = torch.where(
                    anc,
                    torch.zeros((), dtype=torch.float32, device=device),
                    torch.full((), float("-inf"), dtype=torch.float32, device=device),
                )
                qq_bias[lo:hi, lo:hi] = block
            else:
                # Single decode query attends to itself (no ancestor structure).
                qq_bias[lo:hi, lo:hi] = 0.0
    else:
        qq_bias = None

    return AttnMeta(
        block_table=block_table,
        cu_seqlens_q=cu,
        seq_lens_k=seq_lens_k,
        qq_bias=qq_bias,
    )
