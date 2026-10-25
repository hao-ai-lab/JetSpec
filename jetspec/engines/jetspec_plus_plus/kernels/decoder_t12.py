"""Default-off decoder staging and Q/K postprocessing without intermediate cats."""
import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = tl = None


def combined_glue(rp, pending, prefix, staged, cos_table, sin_table,
                  hidden, cos, sin, prefix_mask, window_mask, positions,
                  prefix_slots, window_slots, rejected, *, length_offset,
                  count_offset, correction_offset, miss_offset, wire, zero_miss):
    """Write concatenated hidden/rotary buffers and both masks in one launch."""
    lanes, rows = prefix.shape[1], staged.shape[1]
    if not rp.is_cuda:
        from jetspec.engines.jetspec_plus_plus.kernels.decoder_glue import prefix_glue, window_glue
        prefix_glue(rp, pending, cos_table, sin_table, prefix_mask, positions,
                    prefix_slots, cos[:, :lanes], sin[:, :lanes], rejected,
                    length_offset=length_offset, count_offset=count_offset,
                    correction_offset=correction_offset, miss_offset=miss_offset,
                    wire=wire, zero_miss=zero_miss)
        window_glue(rp[length_offset], staged, cos_table, sin_table,
                    hidden[:, lanes:], window_mask, window_slots,
                    cos[:, lanes:], sin[:, lanes:])
        hidden[:, :lanes].copy_(prefix)
        return
    bucket, width, dim = prefix_mask.shape[-1], hidden.shape[-1], cos.shape[-1]
    mb = triton.cdiv((lanes + rows) * bucket, 256)
    rb = triton.cdiv((lanes + rows) * dim, 256)
    hb = triton.cdiv((lanes + rows) * width, 256)
    from jetspec.engines.jetspec_plus_plus.kernels.nvfp4.entry import stage_entry
    if stage_entry(hidden, staged, prefix, None, _stage,
                   (rp, pending, prefix, staged, cos_table, sin_table, hidden, cos, sin,
                    prefix_mask, window_mask, positions, prefix_slots, window_slots, rejected),
                   (length_offset, count_offset, correction_offset, miss_offset, wire, zero_miss,
                    lanes, rows, bucket, width, staged.shape[-1], dim, cos_table.shape[0],
                    float(torch.finfo(prefix_mask.dtype).min), mb, rb, 0, 256), mb + rb + 1):
        return
    _stage[(mb + rb + hb + 1,)](
        rp, pending, prefix, staged, cos_table, sin_table, hidden, cos, sin,
        prefix_mask, window_mask, positions, prefix_slots, window_slots, rejected,
        length_offset, count_offset, correction_offset, miss_offset, wire, zero_miss,
        lanes, rows, bucket, width, staged.shape[-1], dim, cos_table.shape[0],
        float(torch.finfo(prefix_mask.dtype).min), mb, rb, hb, 256)


def qkv_postprocess(fused, q, k, v, qw, kw, cos, sin, qe, ke, heads, kv_heads, dim,
                    *, cache=None, prefix_slots=None, window_slots=None):
    """Normalize/rotate Q and K and copy V in one launch, optionally into slots."""
    tokens = q.shape[1]
    if not q.is_cuda:
        query = fused.fast_q_postprocess(q, qw, cos, sin, qe, heads, dim)
        key, value = fused.fast_kv_postprocess(k, v, kw, cos, sin, ke, kv_heads, 1, dim)
        if cache is not None:
            n = prefix_slots.numel()
            for dest, src in zip(cache, (key, value)):
                dest.index_copy_(2, prefix_slots, src[..., :n, :])
                dest.index_copy_(2, window_slots, src[..., n:, :])
            key, value = cache
        return query, key, value
    query = q.new_empty(tokens, heads, dim)
    key, value = (cache if cache is not None else
                  (q.new_empty(1, kv_heads, tokens, dim),
                   q.new_empty(1, kv_heads, tokens, dim)))
    _qkv[(tokens, heads + kv_heads)](
        q, k, v, qw, kw, cos, sin, query, key, value, prefix_slots, window_slots,
        heads, dim, q.stride(1), k.stride(1), v.stride(1), cos.stride(1),
        key.stride(1), key.stride(2), value.stride(1), value.stride(2),
        qe, ke, cache is not None, prefix_slots.numel() if cache is not None else 0,
        triton.next_power_of_2(dim), num_warps=4, enable_fp_fusion=False)
    return query.transpose(0, 1)[None], key, value


if triton is not None:
    @triton.jit
    def _qkv(Q, K, V, QW, KW, COS, SIN, QO, KO, VO, PS, WS,
             HEADS: tl.constexpr, D: tl.constexpr,
             QS: tl.constexpr, KS: tl.constexpr, VS: tl.constexpr, RS: tl.constexpr,
             KH: tl.constexpr, KR: tl.constexpr, VH: tl.constexpr, VR: tl.constexpr,
             QE: tl.constexpr, KE: tl.constexpr, DIRECT: tl.constexpr,
             PREFIX: tl.constexpr, BLOCK: tl.constexpr):
        row, h = tl.program_id(0), tl.program_id(1)
        d = tl.arange(0, BLOCK)
        mask = d < D
        if h < HEADS:
            source = tl.load(Q + row * QS + h * D + d, mask, 0).to(tl.float32)
            weight = tl.load(QW + d, mask, 0)
            eps = QE
        else:
            source = tl.load(K + row * KS + (h - HEADS) * D + d, mask, 0).to(tl.float32)
            weight = tl.load(KW + d, mask, 0)
            eps = KE
        rms = tl.rsqrt(tl.sum(source * source, 0) / D + eps)
        normed = (source * rms).to(tl.bfloat16)
        weighted = (normed * weight).to(tl.bfloat16)
        partner = tl.where(d < D // 2, d + D // 2, d - D // 2)
        rotated = tl.gather(weighted, partner, 0)
        rotated = tl.where(d < D // 2, -rotated, rotated)
        first = (weighted * tl.load(COS + row * RS + d, mask, 0)).to(tl.bfloat16)
        second = (rotated * tl.load(SIN + row * RS + d, mask, 0)).to(tl.bfloat16)
        result = (first + second).to(tl.bfloat16)
        if h < HEADS:
            tl.store(QO + (row * HEADS + h) * D + d, result, mask)
        else:
            slot = row.to(tl.int64)
            if DIRECT:
                if row < PREFIX:
                    slot = tl.load(PS + row).to(tl.int64)
                else:
                    slot = tl.load(WS + row - PREFIX).to(tl.int64)
            head = h - HEADS
            tl.store(KO + head * KH + slot * KR + d, result, mask)
            value = tl.load(V + row * VS + head * D + d, mask, 0)
            tl.store(VO + head * VH + slot * VR + d, value, mask)

    @triton.jit
    def _stage(RP, PENDING, PREFIX, STAGED, CT, ST, HIDDEN, COS, SIN,
               PM, WM, POS, PS, WS, REJECTED,
               LO: tl.constexpr, CO: tl.constexpr, COR: tl.constexpr, MO: tl.constexpr,
               WIRE: tl.constexpr, ZERO: tl.constexpr,
               LANES: tl.constexpr, ROWS: tl.constexpr, BUCKET: tl.constexpr,
               H: tl.constexpr, SH: tl.constexpr, D: tl.constexpr, TABLE: tl.constexpr,
               NEG: tl.constexpr, MB: tl.constexpr, RB: tl.constexpr,
               HB: tl.constexpr, BLOCK: tl.constexpr, OFFSET: tl.constexpr = 0):
        p = tl.program_id(0) - OFFSET
        x = tl.arange(0, BLOCK)
        length, count = tl.load(RP + LO), tl.load(RP + CO)
        if p < MB:
            flat = p * BLOCK + x
            r, k = flat // BUCKET, flat % BUCKET
            pos = length - count + r
            prefix_ok = ((r < count) & (k <= pos) & (k < length)) | ((r >= count) & (k == 0))
            window_ok = k <= length + r - LANES
            tl.store(PM + flat, tl.where(prefix_ok, 0., NEG), r < LANES)
            tl.store(WM + flat - LANES * BUCKET, tl.where(window_ok, 0., NEG),
                     (r >= LANES) & (r < LANES + ROWS))
        elif p < MB + RB:
            flat = (p - MB) * BLOCK + x
            r, d = flat // D, flat % D
            pos = tl.where(r < LANES, length - count + r, length + r - LANES)
            valid = (r < LANES + ROWS) & (pos >= 0) & (pos < TABLE)
            c, s = tl.load(CT + pos * D + d, valid, 0), tl.load(ST + pos * D + d, valid, 0)
            tl.store(COS + flat, c, r < LANES + ROWS)
            tl.store(SIN + flat, s, r < LANES + ROWS)
        elif p < MB + RB + HB:
            flat = (p - MB - RB) * BLOCK + x
            r, d = flat // H, flat % H
            a = tl.load(PREFIX + flat, r < LANES, 0)
            b = tl.load(STAGED + (r - LANES) * SH + SH - H + d,
                        (r >= LANES) & (r < LANES + ROWS), 0)
            tl.store(HIDDEN + flat, tl.where(r < LANES, a, b), r < LANES + ROWS)
        else:
            pos = length - count + x
            tl.store(POS + x, pos, x < LANES)
            tl.store(PS + x, tl.where(x < count, pos, BUCKET - 1), x < LANES)
            tl.store(WS + x, length + x, x < ROWS)
            value = tl.load(PENDING + tl.where(WIRE, x, x - 1),
                            (x < LANES - 1) if WIRE else ((x > 0) & (x < LANES)), 0)
            if WIRE:
                value = tl.where(x == LANES - 1, -1, value)
            else:
                value = tl.where(x == 0, tl.load(RP + COR), value)
            if ZERO:
                value = tl.where((x == 0) & (tl.load(RP + MO) != 0), -1, value)
            tl.store(REJECTED + x, value, x < LANES)
