# Adaptation: package relocation and deferred optional dependencies.
"""Paged causal GQA attention for the hybrid target's 256-wide heads."""

import torch
from jetspec.engines.jetspec_plus_plus._optional import triton
from jetspec.engines.jetspec_plus_plus._optional import tl


from jetspec.engines.jetspec_plus_plus.kernels.paged_attention import _partial


from jetspec.engines.jetspec_plus_plus.kernels.paged_attention import _merge


def attend(q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor,
           context_lens: torch.Tensor, block_tables: torch.Tensor,
           scale: float) -> torch.Tensor:
    """Match Attention._sdpa_paged's mask without materializing the KV gather."""
    batch = context_lens.shape[0]
    if (q.device.type != "cuda" or q.dtype not in (torch.bfloat16, torch.float32)
            or k_cache.dtype != q.dtype or v_cache.dtype != q.dtype
            or q.ndim != 3 or k_cache.ndim != 4 or k_cache.shape != v_cache.shape
            or q.shape[-1] != 256 or k_cache.shape[-1] != 256
            or q.shape[1] != 24 or k_cache.shape[2] != 4
            or batch < 1 or q.shape[0] % batch
            or q.shape[0] // batch not in (1, 8, 16, 17, 32, 33)
            or block_tables.shape[0] != batch):
        raise ValueError("triton paged attention requires CUDA bf16/fp32, 24:4 GQA, D=256, width 1, 8, 16, 17, 32 or 33")
    width = q.shape[0] // batch
    heads = q.shape[1]
    page_size = k_cache.shape[1]
    splits = triton.cdiv(block_tables.shape[1] * page_size, 64)
    partial = torch.empty((batch, heads, splits, width, 256),
                          device=q.device, dtype=torch.float32)
    lse = torch.empty((batch, heads, splits, width), device=q.device,
                      dtype=torch.float32)
    output = torch.empty_like(q)
    # A 64-row fp32 tile exceeds sm80's shared memory at recovery width 33.
    # Separate query tiles retain the same per-row dot products and causal mask.
    rows = min(32, max(16, triton.next_power_of_2(width)))
    _partial[(batch, heads * triton.cdiv(width, rows), splits)](
        q, k_cache, v_cache, block_tables, context_lens, partial, lse,
        *q.stride(), *k_cache.stride(), *v_cache.stride(), *block_tables.stride(),
        width, rows, heads, heads // k_cache.shape[2], page_size, block_tables.shape[1],
        splits, scale, num_warps=8)
    _merge[(batch, heads, width)](
        partial, lse, output, *output.stride(), width, heads, splits,
        triton.next_power_of_2(splits), num_warps=4)
    return output
