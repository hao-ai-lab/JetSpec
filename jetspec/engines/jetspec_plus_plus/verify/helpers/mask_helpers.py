import torch

# Global cache for precomputed mask components
_mask_cache = {
    'glue_and_rec_mask': None,
    'diag_components': None,
    'ones_tensor': None,
    'cached_params': None
}








@torch.inference_mode()
def flat_blocks_after_cat(L: torch.Tensor, M: torch.Tensor):
    """
    L: 1D long/int tensor [k] with positive lengths
    M: [N, y] tensor (constant block appended to each [N, L[i]] along dim=1)
    Returns:
      1D tensor equal to cat_i( flatten( [ones(N, L[i]) cat M] ) ).
    """
    assert L.ndim == 1 and L.numel() > 0
    N, y = M.shape
    k = L.numel()
    device, dtype = M.device, M.dtype

    cols_per_block = L.to(torch.long) + y                 # [k]
    total_cols = int(cols_per_block.sum().item())

    # one allocation (all ones for the variable-length left parts)
    T = torch.ones((N, total_cols), device=device, dtype=dtype)

    # mark the last y columns of each block (vectorized) and write M everywhere at once
    offs = torch.cat([torch.zeros(1, device=device, dtype=torch.long),
                      cols_per_block.cumsum(0)[:-1]])     # [k] start col per block
    pos = torch.arange(total_cols, device=device)         # [total_cols]
    blk = torch.repeat_interleave(torch.arange(
        k, device=device), cols_per_block)  # [total_cols]
    # 0..(L[i]+y-1) per block
    within = pos - offs[blk]
    # last y columns per block
    mask = within.ge(L[blk])
    # single strided write
    T[:, mask] = M.repeat(1, k)

    # views per block, then flatten each view and cat once
    # list of [N, L[i]+y] VIEWS
    blocks = T.split(cols_per_block.tolist(), dim=1)
    out = torch.cat([b.reshape(-1) for b in blocks], dim=0)
    return out


@torch.inference_mode()
def get_verify_chain_mask(
    context_lens: torch.Tensor,
    query_len: int,
) -> torch.Tensor:
    # Pack per-request prefix-plus-causal masks for target verification.
    if query_len <= 0:
        raise ValueError(f"query_len must be positive, got {query_len}")
    if context_lens.ndim != 1 or context_lens.numel() == 0:
        raise ValueError("context_lens must be a non-empty 1D tensor")

    prefix_lens = context_lens.to(torch.long) - query_len
    if int(prefix_lens.min().item()) < 0:
        raise ValueError(
            f"context length must be at least query_len={query_len}, "
            f"got {context_lens.tolist()}"
        )

    causal_suffix = torch.tril(torch.ones(
        query_len,
        query_len,
        dtype=torch.bool,
        device=context_lens.device,
    ))
    return flat_blocks_after_cat(prefix_lens, causal_suffix)


@torch.inference_mode()
def get_verify_chain_mask_cached(model_runner, context_lens, query_len):
    """Reuse one exact mask; CPU sequence lengths avoid device scalar reads."""
    lengths = getattr(model_runner, "_verify_context_lens_cpu", None)
    if lengths is None:
        return get_verify_chain_mask(context_lens, query_len)
    if query_len <= 0 or not lengths or min(lengths) < query_len:
        raise ValueError("context lengths must be nonempty and at least query_len > 0")
    padded_count = context_lens.numel()
    if padded_count < len(lengths):
        raise ValueError("graph bucket cannot be smaller than the sequence count")
    lengths = lengths + (lengths[-1],) * (padded_count - len(lengths))
    key = (lengths, query_len, context_lens.device)
    cached = getattr(model_runner, "_verify_chain_mask_cache", None)
    if cached is not None and cached[0] == key:
        return cached[1]
    device = context_lens.device
    suffix = torch.tril(torch.ones(query_len, query_len, dtype=torch.bool, device=device))
    mask = torch.cat([
        torch.cat([torch.ones(query_len, length - query_len, dtype=torch.bool, device=device), suffix], dim=1).reshape(-1)
        for length in lengths
    ])
    model_runner._verify_chain_mask_cache = (key, mask)
    return mask


# Cache for vectorized path precomputed components
_vec_cache = {}



@torch.inference_mode()
def get_custom_mask(config, context_lens, step: int, K: int, F: int, B: int, device: torch.device, cache_hits: torch.Tensor):
    if B <= 8:
        return get_custom_mask_cached(config, context_lens, step, K, F, B, device, fan_out_list=config.fan_out_list, fan_out_list_miss=config.fan_out_list_miss, cache_hits=cache_hits)
    else:
        return get_custom_mask_vectorized(config, context_lens, step, K, B, device, cache_hits)


