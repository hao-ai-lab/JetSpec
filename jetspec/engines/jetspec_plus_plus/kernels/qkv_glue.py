"""Draft-row Q/K norm, RoPE and direct KV writes in one BF16 launch."""
import torch


_postprocess = None


def qkv_norm_rotary_cache(query, key, value, q_weight, k_weight, cos, sin,
                          cache, slots, span, q_eps, k_eps, q_heads, kv_heads, dim):
    """Preserve the lean row path's BF16 roundings and strided packed input."""
    import triton

    rows = query.shape[0]
    if (not query.is_cuda or query.dtype != torch.bfloat16
            or query.shape != (rows, q_heads * dim)
            or key.shape != (rows, kv_heads * dim) or value.shape != key.shape
            or any(x.stride(-1) != 1 for x in (query, key, value, cos, sin))
            or cache.shape[1:] != (2, 1, kv_heads, dim) or cache.stride(-1) != 1):
        raise ValueError('QKV glue requires dense BF16 CUDA head columns and paged KV')
    output = query.new_empty(rows, q_heads, dim)
    from jetspec.engines.jetspec_plus_plus.profiles import kernel_enabled
    if kernel_enabled('draft_fuse_t6'):
        from jetspec.engines.jetspec_plus_plus.kernels.draft_glue_tune import qkv_into
        qkv_into(query, key, value, q_weight, k_weight, cos, sin, output,
                 cache, slots, span, q_eps, k_eps, q_heads, kv_heads, dim)
        return output
    global _postprocess
    if _postprocess is None:
        _postprocess = _kernel()
    _postprocess[(rows, q_heads + kv_heads)](
        query, key, value, q_weight, k_weight, cos, sin, output, cache, slots,
        q_heads, kv_heads, dim, query.stride(0), key.stride(0), value.stride(0),
        cos.stride(0), sin.stride(0), cache.stride(0), cache.stride(1), cache.stride(3),
        span[0] if span is not None else 0, span is not None, q_eps, k_eps,
        BLOCK=triton.next_power_of_2(dim), num_warps=4, enable_fp_fusion=False)
    return output


def _kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def postprocess(Q, K, V, QW, KW, Cos, Sin, Out, Cache, Slots,
                    QH: tl.constexpr, KH: tl.constexpr, D: tl.constexpr,
                    QS: tl.constexpr, KS: tl.constexpr, VS: tl.constexpr,
                    CS: tl.constexpr, SS: tl.constexpr,
                    CACHE_ROW: tl.constexpr, CACHE_KV: tl.constexpr, CACHE_HEAD: tl.constexpr,
                    START: tl.constexpr, SPAN: tl.constexpr,
                    QE: tl.constexpr, KE: tl.constexpr, BLOCK: tl.constexpr):
        row, group = tl.program_id(0), tl.program_id(1)
        col = tl.arange(0, BLOCK)
        valid = col < D
        partner = tl.where(col < D // 2, col + D // 2, col - D // 2)
        cosine = tl.load(Cos + row * CS + col, valid, other=0)
        sine = tl.load(Sin + row * SS + col, valid, other=0)
        if group < QH:
            source = tl.load(Q + row * QS + group * D + col, valid, other=0).to(tl.float32)
            rms = tl.rsqrt(tl.sum(source * source, 0) / D + QE)
            normed = (source * rms).to(tl.bfloat16)
            weighted = (normed * tl.load(QW + col, valid, other=0)).to(tl.bfloat16)
            rotated = tl.gather(weighted, partner, 0)
            rotated = tl.where(col < D // 2, -rotated, rotated)
            first = (weighted * cosine).to(tl.bfloat16)
            second = (rotated * sine).to(tl.bfloat16)
            tl.store(Out + (row * QH + group) * D + col,
                     (first + second).to(tl.bfloat16), valid)
        else:
            head = group - QH
            source = tl.load(K + row * KS + head * D + col, valid, other=0).to(tl.float32)
            rms = tl.rsqrt(tl.sum(source * source, 0) / D + KE)
            normed = (source * rms).to(tl.bfloat16)
            weighted = (normed * tl.load(KW + col, valid, other=0)).to(tl.bfloat16)
            rotated = tl.gather(weighted, partner, 0)
            rotated = tl.where(col < D // 2, -rotated, rotated)
            first = (weighted * cosine).to(tl.bfloat16)
            second = (rotated * sine).to(tl.bfloat16)
            if SPAN:
                slot = START + row
            else:
                slot = tl.load(Slots + row)
            target = slot * CACHE_ROW + head * CACHE_HEAD + col
            tl.store(Cache + target, (first + second).to(tl.bfloat16), valid)
            copied = tl.load(V + row * VS + head * D + col, valid, other=0)
            tl.store(Cache + target + CACHE_KV, copied, valid)

    return postprocess
