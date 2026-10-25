# Adaptation: package relocation and deferred optional dependencies.
"""Graph-safe per-row refresh suffix state for FlashInfer cascade merge."""

import torch
from jetspec.engines.jetspec_plus_plus._optional import triton
from jetspec.engines.jetspec_plus_plus._optional import tl


@triton.jit
def _suffix_state_kernel(Q, K, V, START, ROW_INDEX, ROW_STARTS, COUNTS, FIRST,
                         OUT, LSE, q_stride_h: tl.constexpr, q_stride_t: tl.constexpr,
                         k_stride_h: tl.constexpr, k_stride_s: tl.constexpr,
                         v_stride_h: tl.constexpr, v_stride_s: tl.constexpr,
                         HEADS_PER_KV: tl.constexpr, DIM: tl.constexpr,
                         WINDOW: tl.constexpr, ROWS: tl.constexpr,
                         SCALE: tl.constexpr, BLOCK_KEYS: tl.constexpr,
                         BLOCK_DIM: tl.constexpr):
    token = tl.program_id(0)
    head = tl.program_id(1)
    row = tl.load(ROW_INDEX + token)
    row_start = tl.load(ROW_STARTS + row)
    count = tl.load(COUNTS + row)
    first = tl.load(FIRST + row)
    local = token - row_start
    start = tl.load(START)
    keys = tl.arange(0, BLOCK_KEYS)
    dims = tl.arange(0, BLOCK_DIM)
    tail = keys < WINDOW
    visible = (row < ROWS) & ((tail & (keys < count + first)) |
                              (~tail & (keys - WINDOW < local)))
    slots = tl.where(tail, start - WINDOW + keys,
                     start + row_start + keys - WINDOW)
    kv_head = head // HEADS_PER_KV
    query = tl.load(Q + head * q_stride_h + token * q_stride_t + dims,
                    dims < DIM, other=0).to(tl.float32)
    key = tl.load(K + kv_head * k_stride_h + slots[:, None] * k_stride_s
                  + dims[None, :], visible[:, None] & (dims[None, :] < DIM),
                  other=0).to(tl.float32)
    score = tl.sum(key * query[None, :], 1) * SCALE
    score = tl.where(visible, score, -float("inf"))
    maximum = tl.max(score, 0)
    safe_maximum = tl.where(maximum == -float("inf"), 0.0, maximum)
    weight = tl.exp2((score - safe_maximum) * 1.4426950408889634)
    denominator = tl.sum(weight, 0)
    value = tl.load(V + kv_head * v_stride_h + slots[:, None] * v_stride_s
                    + dims[None, :], visible[:, None] & (dims[None, :] < DIM),
                    other=0).to(tl.float32)
    output = tl.sum(weight[:, None] * value, 0) / tl.where(denominator > 0,
                                                          denominator, 1.0)
    lse = tl.where(denominator > 0,
                   tl.log2(denominator) + safe_maximum * 1.4426950408889634,
                   -float("inf"))
    tl.store(OUT + (token * tl.num_programs(1) + head) * DIM + dims,
             output, dims < DIM)
    tl.store(LSE + token * tl.num_programs(1) + head, lse)


def suffix_state(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                 start: torch.Tensor, window: int, scale: float,
                 meta: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]):
    row_index, row_starts, counts, first = meta
    query = q[0]
    keys, values = k[0], v[0]
    heads, tokens, dim = query.shape
    kv_heads = keys.shape[0]
    output = torch.empty((tokens, heads, dim), device=q.device, dtype=q.dtype)
    lse = torch.empty((tokens, heads), device=q.device, dtype=torch.float32)
    _suffix_state_kernel[(tokens, heads)](
        query, keys, values, start, row_index, row_starts, counts, first,
        output, lse, query.stride(0), query.stride(1), keys.stride(0),
        keys.stride(1), values.stride(0), values.stride(1), heads // kv_heads,
        dim, window, counts.numel() - 1, scale,
        triton.next_power_of_2(2 * window), triton.next_power_of_2(dim))
    return output, lse
