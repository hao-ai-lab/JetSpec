"""Device kernels for draft/refresh_inject.py."""


def _kernels():
    import triton
    import triton.language as tl

    @triton.jit
    def front(source, khat, marker, output, width: tl.constexpr, block: tl.constexpr,
              mark: tl.constexpr, TILE: tl.constexpr):
        token = tl.program_id(0)
        col = tl.program_id(1) * TILE + tl.arange(0, TILE)
        visible = (token % block <= tl.load(khat + token // block)) & (token % block < block - 1)
        value = tl.load(source + token * width + col, (col < width) & visible, other=0)
        dtype = output.dtype.element_ty
        value = value.to(dtype)
        if mark:
            value = (value + tl.load(marker + width + col, col < width, other=0)).to(dtype)
        tl.store(output + token * width + col, value, col < width)

    @triton.jit
    def postprocess(projected, weights, cos, sin, cache, slots, khat,
                    tokens: tl.constexpr, kv_heads: tl.constexpr, head_dim: tl.constexpr,
                    block: tl.constexpr, projected_row_stride: tl.constexpr,
                    cos_row_stride: tl.constexpr, cache_layer_stride: tl.constexpr,
                    cache_row_stride: tl.constexpr, cache_kv_stride: tl.constexpr,
                    eps: tl.constexpr, BLOCK: tl.constexpr,
                    MASK=None, MASK_START: tl.constexpr = 0,
                    MASK_BYTES: tl.constexpr = 0, CACHE_PROGRAMS: tl.constexpr = 0,
                    FUSED_MASK: tl.constexpr = False, RESET: tl.constexpr = False):
        program = tl.program_id(0)
        if not FUSED_MASK or program < CACHE_PROGRAMS:
            head = program % kv_heads
            token = (program // kv_heads) % tokens
            layer = program // (kv_heads * tokens)
            visible = (token % block <= tl.load(khat + token // block)) & (token % block < block - 1)
            if visible:
                # Seg A's arithmetic, including every BF16 rounding boundary.
                col = tl.arange(0, BLOCK)
                mask = col < head_dim
                width = kv_heads * head_dim
                base = token * projected_row_stride + layer * 2 * width + head * head_dim
                value = tl.load(projected + base + col, mask=mask, other=0).to(tl.float32)
                scale = tl.rsqrt(tl.sum(value * value, 0) / head_dim + eps)
                normed = (value * scale).to(tl.bfloat16)
                weight = tl.load(weights + layer * head_dim + col, mask=mask, other=0)
                normed = (normed * weight).to(tl.bfloat16)
                half = head_dim // 2
                partner = tl.where(col < half, col + half, col - half)
                rotated = tl.gather(normed, partner, 0)
                rotated = tl.where(col < half, -rotated, rotated)
                rope = token * cos_row_stride + col
                cosine = tl.load(cos + rope, mask=mask, other=0)
                sine = tl.load(sin + rope, mask=mask, other=0)
                first = (normed * cosine).to(tl.bfloat16)
                second = (rotated * sine).to(tl.bfloat16)
                cache_row = tl.load(slots + token)
                target = (layer * cache_layer_stride + cache_row * cache_row_stride + head * head_dim + col)
                tl.store(cache + target, (first + second).to(tl.bfloat16), mask=mask)
                copied = tl.load(projected + base + width + col, mask=mask, other=0)
                tl.store(cache + target + cache_kv_stride, copied, mask=mask)
        else:
            index = (program - CACHE_PROGRAMS) * 256 + tl.arange(0, 256)
            valid = index < MASK_BYTES
            row_bytes = (MASK_START + tokens) // 8
            query = index // row_bytes
            entry = (index % row_bytes) * 8 - MASK_START
            region = (entry >= 0) & (entry < tokens)
            own = region & (entry // block == query // block)
            cutoff = tl.load(khat + query // block, valid, other=0)
            limit = tl.minimum(tl.minimum(tl.maximum(cutoff, 0), block - 1),
                               query % block - 1)
            count = tl.minimum(tl.maximum(limit - entry % block + 1, 0), 8)
            packed = tl.where(own, (1 << count) - 1, 0).to(tl.uint8)
            prior = tl.load(MASK + index, valid, other=0)
            if RESET:
                prior = tl.where(region, 0, prior).to(tl.uint8)
            tl.store(MASK + index, prior | packed, valid)

    return front, postprocess

