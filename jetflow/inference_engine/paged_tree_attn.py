"""Paged tree-attention triton kernel (JetFlow N3, opt-in).

Reads K/V straight from the block pool via per-sequence block tables and folds
the per-node ancestor mask in as an additive bias on the scores — no dense KV
reconstruction, no padding waste. This is the throughput path; the SDPA fallback
in ``engine.py`` stays the default and the correctness oracle (see
``tests/test_jetflow_kernel.py``).

Pool layout matches ``PagedKVCache``: ``(num_blocks, block_size, Hkv, head_dim)``
for both K and V. K is **already post-RoPE** (the engine applies RoPE before
storing), so this kernel applies no RoPE — it reads the exact bytes SDPA reads,
which makes "kernel == SDPA" the correctness statement.

Adapted from vLLM's ``kernel_unified_attention_2d`` (2D online-softmax path) by
Ringlein/van Lunteren/Yang/Parnell — stripped of the 3D segmented kernel,
CUDA-graph capture, FP8/descale, alibi, sinks, sliding-window, and multimodal
paths. The ``qq_bias`` (additive -inf/0 over query-vs-query positions) IS our
tree ancestor mask and is kept verbatim.
"""
from typing import Optional

import torch
import triton
import triton.language as tl


@triton.jit
def _cdiv_fn(x, y):
    return (x + y - 1) // y


@triton.jit
def _find_seq_idx(
    query_start_len_ptr,
    target_idx,
    num_seqs,
    BLOCK_Q: tl.constexpr,
):
    # Binary search for the seq whose Q-block range contains target_idx, using the
    # same per-seq Q-block offset convention as the launch grid (val // BLOCK_Q + s).
    left: tl.int32 = 0
    right = num_seqs
    while left < right:
        mid = (left + right) // 2
        val = tl.load(query_start_len_ptr + mid)
        mid_val = val // BLOCK_Q + mid
        if mid_val <= target_idx:
            left = mid + 1
        else:
            right = mid
    return left - 1


@triton.jit
def _kernel_paged_tree_attn(
    output_ptr,           # [total_q, Hq, D]
    query_ptr,            # [total_q, Hq, D]
    key_cache_ptr,        # [num_blocks, block_size, Hkv, D]
    value_cache_ptr,      # [num_blocks, block_size, Hkv, D]
    block_tables_ptr,     # [num_seqs, max_blocks]
    logical_kv_slots_ptr,  # [num_seqs, max_logical_slots] or unused
    logical_kv_starts_ptr,  # [num_seqs] or unused
    logical_kv_lens_ptr,  # [num_seqs] or unused
    seq_lens_ptr,         # [num_seqs]
    qq_bias_ptr,          # [total_q, total_q] or unused
    scale,                # float32
    num_query_heads: tl.constexpr,
    num_queries_per_kv: tl.constexpr,
    block_table_stride: tl.int64,
    logical_kv_slots_stride: tl.int64,
    query_stride_0: tl.int64,
    query_stride_1: tl.int64,
    output_stride_0: tl.int64,
    output_stride_1: tl.int64,
    qq_bias_stride_0: tl.int64,
    BLOCK_SIZE: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    HEAD_SIZE_PADDED: tl.constexpr,
    USE_QQ_BIAS: tl.constexpr,
    USE_LOGICAL_KV_SLOTS: tl.constexpr,
    stride_k_cache_0: tl.int64,
    stride_k_cache_1: tl.int64,
    stride_k_cache_2: tl.int64,
    stride_k_cache_3: tl.constexpr,
    stride_v_cache_0: tl.int64,
    stride_v_cache_1: tl.int64,
    stride_v_cache_2: tl.int64,
    stride_v_cache_3: tl.constexpr,
    query_start_len_ptr,  # [num_seqs+1]
    BLOCK_Q: tl.constexpr,
    num_seqs: tl.int32,
    BLOCK_M: tl.constexpr,
):
    q_block_global_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)

    seq_idx = _find_seq_idx(query_start_len_ptr, q_block_global_idx, num_seqs, BLOCK_Q)

    q_block_start_idx = tl.load(query_start_len_ptr + seq_idx) // BLOCK_Q + seq_idx
    q_block_local_idx = q_block_global_idx - q_block_start_idx

    cur_batch_in_all_start_index = tl.load(query_start_len_ptr + seq_idx)
    cur_batch_in_all_stop_index = tl.load(query_start_len_ptr + seq_idx + 1)
    cur_batch_query_len = cur_batch_in_all_stop_index - cur_batch_in_all_start_index

    if q_block_local_idx * BLOCK_Q >= cur_batch_query_len:
        return

    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_SIZE_PADDED)
    offs_t = tl.arange(0, TILE_SIZE)
    # Each q-block packs num_queries_per_kv query heads per query row (GQA in-kernel).
    query_pos = q_block_local_idx * BLOCK_Q + offs_m // num_queries_per_kv

    query_offset_0 = cur_batch_in_all_start_index + query_pos
    query_offset_1 = kv_head_idx * num_queries_per_kv + offs_m % num_queries_per_kv
    query_offset = (
        query_offset_0[:, None] * query_stride_0
        + query_offset_1[:, None] * query_stride_1
        + offs_d[None, :]
    )

    dim_mask = tl.where(offs_d < HEAD_SIZE, 1, 0).to(tl.int1)
    query_mask_0 = tl.where(query_pos < cur_batch_query_len, 1, 0).to(tl.int1)
    query_mask_1 = tl.where(query_offset_1 < num_query_heads, 1, 0).to(tl.int1)

    # Q : (BLOCK_M, HEAD_SIZE_PADDED)
    Q = tl.load(
        query_ptr + query_offset,
        mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
        other=0.0,
    )

    block_table_offset = seq_idx * block_table_stride
    logical_kv_slots_offset = seq_idx * logical_kv_slots_stride

    M = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    L = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_SIZE_PADDED], dtype=tl.float32)

    seq_len = tl.load(seq_lens_ptr + seq_idx)
    # The query section starts at this key position; keys [context_len, seq_len)
    # are this seq's query nodes, keys [0, context_len) are the visible prefix.
    context_len = seq_len - cur_batch_query_len

    if USE_QQ_BIAS:
        qq_bias_row_ptrs = qq_bias_ptr + query_offset_0[:, None] * qq_bias_stride_0

    # Longest prefix any query row in this q-block reaches (causal tile pruning).
    max_seq_prefix_len = (
        context_len
        + q_block_local_idx * BLOCK_Q
        + (BLOCK_M - 1) // num_queries_per_kv
        + 1
    )
    max_seq_prefix_len = tl.minimum(max_seq_prefix_len, seq_len)
    num_tiles = _cdiv_fn(max_seq_prefix_len, TILE_SIZE)

    for j in range(0, num_tiles):
        seq_offset = j * TILE_SIZE + offs_t
        tile_mask = seq_offset < max_seq_prefix_len
        physical_offset = seq_offset % BLOCK_SIZE

        if USE_LOGICAL_KV_SLOTS:
            logical_kv_len = tl.load(logical_kv_lens_ptr + seq_idx)
            logical_kv_start = tl.load(logical_kv_starts_ptr + seq_idx)
            logical_kv_end = logical_kv_start + logical_kv_len
            use_logical_kv = (seq_offset >= logical_kv_start) & (
                seq_offset < logical_kv_end
            )
            logical_kv_idx = seq_offset - logical_kv_start
            logical_kv_slot = tl.load(
                logical_kv_slots_ptr + logical_kv_slots_offset + logical_kv_idx,
                mask=tile_mask & use_logical_kv,
                other=0,
            ).to(tl.int64)
            physical_offset = tl.where(
                use_logical_kv,
                logical_kv_slot % BLOCK_SIZE,
                physical_offset,
            )

        block_table_block_idx = tl.load(
            block_tables_ptr + block_table_offset + seq_offset // BLOCK_SIZE
        ).to(tl.int64)
        physical_block_idx = block_table_block_idx
        if USE_LOGICAL_KV_SLOTS:
            physical_block_idx = tl.where(
                use_logical_kv,
                logical_kv_slot // BLOCK_SIZE,
                block_table_block_idx,
            )

        v_offset = (
            physical_block_idx[:, None] * stride_v_cache_0
            + kv_head_idx * stride_v_cache_2
            + offs_d[None, :] * stride_v_cache_3
            + physical_offset[:, None] * stride_v_cache_1
        )
        k_offset = (
            physical_block_idx[None, :] * stride_k_cache_0
            + kv_head_idx * stride_k_cache_2
            + offs_d[:, None] * stride_k_cache_3
            + physical_offset[None, :] * stride_k_cache_1
        )

        # K : (HEAD_SIZE, TILE_SIZE) — other=0.0 guards the tail past seq_len so a
        # partially filled last block never reads stale pool slots.
        K = tl.load(
            key_cache_ptr + k_offset,
            mask=dim_mask[:, None] & tile_mask[None, :],
            other=0.0,
        )
        # V : (TILE_SIZE, HEAD_SIZE)
        V = tl.load(
            value_cache_ptr + v_offset,
            mask=dim_mask[None, :] & tile_mask[:, None],
            other=0.0,
        )

        # Causal mask: key position must not exceed the query's absolute position.
        query_abs_pos = context_len + query_pos[:, None]
        seq_mask = seq_offset[None, :] <= query_abs_pos

        # S : (BLOCK_M, TILE_SIZE), fp32 accumulate (no tf32) for the matmul.
        S = tl.zeros(shape=(BLOCK_M, TILE_SIZE), dtype=tl.float32)
        S += scale * tl.dot(Q, K, allow_tf32=False)

        S = tl.where(
            query_mask_1[:, None] & query_mask_0[:, None] & seq_mask, S, float("-inf")
        )

        if USE_QQ_BIAS:
            # Key position relative to this seq's query section: a node key when >= 0.
            # qq_bias is (total_q, total_q) block-diagonal, so the column is the
            # GLOBAL index cu[s] + key_rel_pos (the row base is already global).
            key_rel_pos = seq_offset - context_len
            key_col = cur_batch_in_all_start_index + key_rel_pos
            is_query_key = (key_rel_pos >= 0) & (key_rel_pos < cur_batch_query_len)
            qq_bias = tl.load(
                qq_bias_row_ptrs + key_col[None, :],
                mask=is_query_key[None, :],  # prefix / OOB keys read bias 0
                other=0.0,
            )
            S += qq_bias

        m_j = tl.maximum(M, tl.max(S, axis=1))
        m_j = tl.where(m_j > float("-inf"), m_j, 0.0)

        P = tl.exp(S - m_j[:, None])
        l_j = tl.sum(P, axis=1)
        alpha = tl.exp(M - m_j)

        acc = acc * alpha[:, None]
        L = L * alpha + l_j
        M = m_j

        acc += tl.dot(P.to(V.dtype), V, allow_tf32=False)

    acc = acc / L[:, None]

    output_offset = (
        query_offset_0[:, None] * output_stride_0
        + query_offset_1[:, None] * output_stride_1
        + offs_d[None, :]
    )
    tl.store(
        output_ptr + output_offset,
        acc,
        mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
    )


def paged_tree_attn(
    q: torch.Tensor,            # (total_q, Hq, D)            post-RoPE queries, ragged-batched
    k_pool: torch.Tensor,       # (num_blocks, block_size, Hkv, D)  post-RoPE keys, paged
    v_pool: torch.Tensor,       # (num_blocks, block_size, Hkv, D)
    block_table: torch.Tensor,  # (num_seqs, max_blocks) int32
    cu_seqlens_q: torch.Tensor,  # (num_seqs+1,) int32
    seq_lens_k: torch.Tensor,   # (num_seqs,) int32           TOTAL key length per seq
    qq_bias: Optional[torch.Tensor],  # (total_q, total_q) fp32 additive (-inf/0), block-diagonal; None for decode
    scale: float,               # head_dim ** -0.5
    num_queries_per_kv: int,    # Hq // Hkv
    block_size: int,
    logical_kv_slots: Optional[torch.Tensor] = None,  # (num_seqs, max_logical_slots) physical slot ids
    logical_kv_starts: Optional[torch.Tensor] = None,  # (num_seqs,) first logical key pos to remap
    logical_kv_lens: Optional[torch.Tensor] = None,  # (num_seqs,) number of logical key positions
) -> torch.Tensor:
    """Tree attention over a paged K/V pool; returns (total_q, Hq, D), q's dtype.

    For each seq s, its ``Nq_s = cu[s+1]-cu[s]`` query rows attend over keys
    ``[0, seq_lens_k[s])`` pulled from the pool via ``block_table[s]``. Prefix
    keys are always visible; the per-node tree mask over the query section comes
    from ``qq_bias`` (additive -inf/0). Decode is the ``Nq_s == 1`` / ``qq_bias
    is None`` case (single query attends to its full prefix + self).

    ``logical_kv_slots`` optionally remaps a contiguous per-seq key range
    ``[logical_kv_starts[s], logical_kv_starts[s] + logical_kv_lens[s])`` to
    absolute physical pool slots; ``None`` keeps the legacy block-table path."""
    total_q, num_query_heads, head_size = q.shape
    num_seqs = seq_lens_k.shape[0]

    out = torch.empty_like(q)
    use_qq_bias = qq_bias is not None
    use_logical_kv_slots = logical_kv_slots is not None
    if use_logical_kv_slots:
        assert logical_kv_starts is not None
        assert logical_kv_lens is not None
        assert logical_kv_slots.ndim == 2
        assert logical_kv_starts.ndim == 1
        assert logical_kv_lens.ndim == 1
        assert logical_kv_slots.shape[0] == num_seqs
        assert logical_kv_starts.shape[0] == num_seqs
        assert logical_kv_lens.shape[0] == num_seqs

    BLOCK_M = 16 if num_queries_per_kv <= 16 else triton.next_power_of_2(num_queries_per_kv)
    BLOCK_Q = BLOCK_M // num_queries_per_kv
    # Upper-bound launch grid: sum_s ceil(Nq_s / BLOCK_Q) <= total_q // BLOCK_Q + num_seqs.
    total_num_q_blocks = total_q // BLOCK_Q + num_seqs
    num_kv_heads = k_pool.shape[2]
    # bf16/fp16 pools tile at 16; fp32 needs 32+ for a valid tl.dot tile.
    # B2 lever: bf16 bumped 16->32 (16 was too small for the B200 tensor cores; the
    # tree's N-scaling attn cost rides this tile). fp32 stays 32. Gate token+accept-len.
    TILE_SIZE = 32 if q.element_size() >= 2 else 32

    _kernel_paged_tree_attn[(total_num_q_blocks, num_kv_heads)](
        output_ptr=out,
        query_ptr=q,
        key_cache_ptr=k_pool,
        value_cache_ptr=v_pool,
        block_tables_ptr=block_table,
        logical_kv_slots_ptr=logical_kv_slots,
        logical_kv_starts_ptr=logical_kv_starts,
        logical_kv_lens_ptr=logical_kv_lens,
        seq_lens_ptr=seq_lens_k,
        qq_bias_ptr=qq_bias,
        scale=scale,
        num_query_heads=num_query_heads,
        num_queries_per_kv=num_queries_per_kv,
        block_table_stride=block_table.stride(0),
        logical_kv_slots_stride=logical_kv_slots.stride(0) if use_logical_kv_slots else 0,
        query_stride_0=q.stride(0),
        query_stride_1=q.stride(1),
        output_stride_0=out.stride(0),
        output_stride_1=out.stride(1),
        qq_bias_stride_0=qq_bias.stride(0) if use_qq_bias else 0,
        BLOCK_SIZE=block_size,
        TILE_SIZE=TILE_SIZE,
        HEAD_SIZE=head_size,
        HEAD_SIZE_PADDED=triton.next_power_of_2(head_size),
        USE_QQ_BIAS=use_qq_bias,
        USE_LOGICAL_KV_SLOTS=use_logical_kv_slots,
        stride_k_cache_0=k_pool.stride(0),
        stride_k_cache_1=k_pool.stride(1),
        stride_k_cache_2=k_pool.stride(2),
        stride_k_cache_3=k_pool.stride(3),
        stride_v_cache_0=v_pool.stride(0),
        stride_v_cache_1=v_pool.stride(1),
        stride_v_cache_2=v_pool.stride(2),
        stride_v_cache_3=v_pool.stride(3),
        query_start_len_ptr=cu_seqlens_q,
        BLOCK_Q=BLOCK_Q,
        num_seqs=num_seqs,
        BLOCK_M=BLOCK_M,
    )
    return out
