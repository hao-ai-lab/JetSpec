"""Device kernels for draft/refresh_inject.py."""


def _kernels():
    import triton
    import triton.language as tl

    @triton.jit
    def front(source, khat, marker, output, width: tl.constexpr, block: tl.constexpr,
              mark: tl.constexpr, TILE: tl.constexpr,
              STRIDED: tl.constexpr = False, S0: tl.constexpr = 0,
              S1: tl.constexpr = 0, S2: tl.constexpr = 1,
              INCLUSIVE: tl.constexpr = True):
        token = tl.program_id(0)
        col = tl.program_id(1) * TILE + tl.arange(0, TILE)
        visible = (token % block < tl.load(khat + token // block) + int(INCLUSIVE)) & (token % block < block - 1)
        if STRIDED:
            offset = (token // block) * S0 + (token % block) * S1 + col * S2
        else:
            offset = token * width + col
        value = tl.load(source + offset, (col < width) & visible, other=0)
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
                    FUSED_MASK: tl.constexpr = False, RESET: tl.constexpr = False,
                    IDS=None, ACTIVE=None, ERR=None, GUARD: tl.constexpr = False,
                    GUARD_ROWS: tl.constexpr = 0, MASK_TOKEN: tl.constexpr = -1,
                    GUARD_BLOCK: tl.constexpr = 1,
                    KEYS=(), VALUES=(), CACHES=(), INCLUSIVE: tl.constexpr = True,
                    NORMALIZED: tl.constexpr = False):
        program = tl.program_id(0)
        guard_program: tl.constexpr = CACHE_PROGRAMS + (MASK_BYTES + 255) // 256
        if GUARD:
            if program == guard_program:
                flat = tl.arange(0, GUARD_BLOCK)
                ids = tl.load(IDS + flat, flat < GUARD_ROWS * block, other=-1)
                active = tl.load(ACTIVE + flat // block, flat < GUARD_ROWS * block, other=0)
                found = tl.sum(((ids == MASK_TOKEN) & active &
                                (flat < GUARD_ROWS * block)).to(tl.int32), 0) > 0
                tl.store(ERR, tl.load(ERR) | tl.where(found, 1 << 7, 0).to(tl.int64))
        if (not GUARD or program < guard_program) and (not FUSED_MASK or program < CACHE_PROGRAMS):
            head = program % kv_heads
            token = (program // kv_heads) % tokens
            layer = program // (kv_heads * tokens)
            visible = (token % block < tl.load(khat + token // block) + int(INCLUSIVE)) & (token % block < block - 1)
            if visible:
                # Seg A's arithmetic, including every BF16 rounding boundary.
                col = tl.arange(0, BLOCK)
                mask = col < head_dim
                width = kv_heads * head_dim
                base = token * projected_row_stride + layer * 2 * width + head * head_dim
                if len(KEYS):
                    key_ptr, value_ptr, cache_ptr = KEYS[0], VALUES[0], CACHES[0]
                    for i in tl.static_range(1, len(KEYS)):
                        key_ptr = tl.where(layer == i, KEYS[i], key_ptr)
                        value_ptr = tl.where(layer == i, VALUES[i], value_ptr)
                        cache_ptr = tl.where(layer == i, CACHES[i], cache_ptr)
                    base = token * width + head * head_dim
                    value = tl.load(key_ptr + base + col, mask=mask, other=0).to(tl.float32)
                else:
                    value = tl.load(projected + base + col, mask=mask, other=0).to(tl.float32)
                if NORMALIZED:
                    normed = value.to(tl.bfloat16)
                else:
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
                if len(KEYS):
                    target = cache_row * cache_row_stride + head * head_dim + col
                    tl.store(cache_ptr + target, (first + second).to(tl.bfloat16), mask=mask)
                    copied = tl.load(value_ptr + base + col, mask=mask, other=0)
                    tl.store(cache_ptr + target + cache_kv_stride, copied, mask=mask)
                else:
                    tl.store(cache + target, (first + second).to(tl.bfloat16), mask=mask)
                    copied = tl.load(projected + base + width + col, mask=mask, other=0)
                    tl.store(cache + target + cache_kv_stride, copied, mask=mask)
        elif FUSED_MASK and program < guard_program and len(KEYS):
            index = (program - CACHE_PROGRAMS) * 256 + tl.arange(0, 256)
            bit = index[:, None] * 8 + tl.arange(0, 8)[None, :]
            query = bit // (MASK_START + tokens)
            entry = bit % (MASK_START + tokens) - MASK_START
            valid = (index[:, None] < MASK_BYTES) & (query < tokens)
            region = valid & (entry >= 0)
            cutoff = tl.load(khat + query // block, valid, other=0) - (1 - int(INCLUSIVE))
            cutoff = tl.minimum(tl.maximum(cutoff, 0), block - 1)
            allowed = region & (entry // block == query // block)
            allowed &= (entry % block < query % block) & (entry % block <= cutoff)
            shifts = tl.arange(0, 8)[None, :]
            packed = tl.sum(allowed.to(tl.uint32) << shifts, 1).to(tl.uint8)
            prior = tl.load(MASK + index, index < MASK_BYTES, other=0)
            if RESET:
                clear = tl.sum(region.to(tl.uint32) << shifts, 1).to(tl.uint8)
                prior &= ~clear
            tl.store(MASK + index, prior | packed, index < MASK_BYTES)
        elif FUSED_MASK and program < guard_program:
            index = (program - CACHE_PROGRAMS) * 256 + tl.arange(0, 256)
            valid = index < MASK_BYTES
            row_bytes = (MASK_START + tokens) // 8
            query = index // row_bytes
            entry = (index % row_bytes) * 8 - MASK_START
            region = (entry >= 0) & (entry < tokens)
            own = region & (entry // block == query // block)
            cutoff = tl.load(khat + query // block, valid, other=0)
            if not INCLUSIVE:
                cutoff -= 1
            limit = tl.minimum(tl.minimum(tl.maximum(cutoff, 0), block - 1),
                               query % block - 1)
            count = tl.minimum(tl.maximum(limit - entry % block + 1, 0), 8)
            packed = tl.where(own, (1 << count) - 1, 0).to(tl.uint8)
            prior = tl.load(MASK + index, valid, other=0)
            if RESET:
                prior = tl.where(region, 0, prior).to(tl.uint8)
            tl.store(MASK + index, prior | packed, valid)

    return front, postprocess
