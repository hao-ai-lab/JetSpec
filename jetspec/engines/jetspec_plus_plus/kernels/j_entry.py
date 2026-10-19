"""T14 embedding gather, first-layer RMSNorm and per-token guard."""
import torch

try:
    import triton
    import triton.language as tl
    from jetspec.engines.jetspec_plus_plus.kernels.nvfp4.producers import store_mx8
except ImportError:
    triton = None


if triton is not None:
    @triton.jit
    def _embedding_norm(Ids, Embedding, Gamma, Hidden, Normalized, Active, Err,
                        WIDTH: tl.constexpr, LANES: tl.constexpr, EPS: tl.constexpr,
                        MASK_TOKEN: tl.constexpr, GUARD: tl.constexpr, BLOCK: tl.constexpr,
                        Q=None, SF=(), TILES: tl.constexpr = (), STRIDES: tl.constexpr = ()):
        row = tl.program_id(0)
        col = tl.arange(0, BLOCK)
        token = tl.load(Ids + row)
        value = tl.load(Embedding + token * WIDTH + col, col < WIDTH, other=0.)
        tl.store(Hidden + row * WIDTH + col, value, col < WIDTH)
        values = value.to(tl.float32)
        scale = tl.rsqrt(tl.sum(values * values, 0) / WIDTH + EPS)
        normed = (values * scale).to(tl.bfloat16)
        weight = tl.load(Gamma + col, col < WIDTH, other=0.)
        tl.store(Normalized + row * WIDTH + col,
                 (normed * weight).to(tl.bfloat16), col < WIDTH)
        if len(TILES):
            store_mx8((normed * weight).to(tl.bfloat16), Q, SF, row, 0, 0,
                      WIDTH, BLOCK, TILES, STRIDES)
        if GUARD:
            if (token == MASK_TOKEN) & tl.load(Active + row // LANES):
                # J0's reset is an earlier launch; no CTA in this grid clears Err.
                tl.atomic_or(Err, 1 << 7, sem='relaxed')


def embedding_norm(ids, embedding, gamma, eps, active=None, err=None, mask_token=-1):
    """Return embedding and its certified BF16 norm; optionally OR the guard bit."""
    width = embedding.shape[1]
    if not ids.is_cuda or embedding.dtype != torch.bfloat16 or triton is None:
        from jetspec.engines.jetspec_plus_plus.jetspec.c1_fused import fast_rms_norm
        hidden = embedding[ids].reshape(-1, width)
        if err is not None:
            err.bitwise_or_((ids.eq(mask_token) & active[:, None]).any().long().reshape(1) << 7)
        return hidden, fast_rms_norm(hidden, gamma, eps)
    hidden = torch.empty((ids.numel(), width), device=ids.device, dtype=embedding.dtype)
    normalized = torch.empty_like(hidden)
    block = triton.next_power_of_2(width)
    from jetspec.engines.jetspec_plus_plus.profiles import kernel_enabled
    if kernel_enabled('draft_fuse_w1', '1'):
        from jetspec.engines.jetspec_plus_plus.kernels.nvfp4.producers import enabled, allocate_activation, _arguments
        if enabled(normalized):
            produced = allocate_activation(normalized)
            _embedding_norm[(ids.numel(),)](
                ids, embedding, gamma, hidden, normalized, active, err, width, ids.shape[1],
                eps, mask_token, err is not None, block, *_arguments(produced),
                num_warps=8 if block >= 4096 else 4)
            return hidden, normalized
    _embedding_norm[(ids.numel(),)](
        ids, embedding, gamma, hidden, normalized, active, err, width, ids.shape[1],
        eps, mask_token, err is not None, block,
        num_warps=8 if block >= 4096 else 4)
    return hidden, normalized
