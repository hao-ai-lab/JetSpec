"""Dense (8B) target verify glue: Q/K head norm, RoPE and the K/V cache write in one launch.

The eager chain is two reshape copies, the compiled q_norm and k_norm, the compiled
RoPE and store_kvcache. This kernel keeps the compiled chain's BF16 rounding points:
one after the norm's weight product and one after the FP32 rotation. The FP32 sum order of
the norm can differ from Inductor's, so results are near the BF16 rounding error,
not bitwise (class B).
"""
import torch

from jetspec.engines.jetspec_plus_plus.kernels.verify_w1 import activation_buffers, w1_linear

try:
    import triton
    import triton.language as tl
    from jetspec.engines.jetspec_plus_plus.kernels.nvfp4._w4a8_quant import store_activation_row, quantize_activation_block
    from jetspec.engines.jetspec_plus_plus.kernels.verify_w1 import store_bf16_vector
except ImportError:  # The CPU suite runs without Triton.
    triton = None


if triton is not None:
    @triton.jit
    def _qkv_glue(QKV, qkv_row, POS, CS, cs_row, QW, KW, QOUT, KC, VC, SLOT, q_eps, k_eps,
                  QH: tl.constexpr, KH: tl.constexpr, D: tl.constexpr, PDL: tl.constexpr,
                  PREWAIT: tl.constexpr = False):
        # One program per (token, head); heads 0..QH-1 are queries, QH..QH+KH-1 keys (plus values).
        if PDL and not PREWAIT:
            gdc_wait()
            gdc_launch()
        row = tl.program_id(0).to(tl.int64)
        head = tl.program_id(1)
        col = tl.arange(0, D)
        half: tl.constexpr = D // 2
        partner = tl.where(col < half, col + half, col - half)
        position = tl.load(POS + row).to(tl.int64)
        cosine = tl.load(CS + position * cs_row + col % half)
        sine = tl.load(CS + position * cs_row + half + col % half)
        sign = tl.where(col < half, -1.0, 1.0)
        # Query heads then key heads are adjacent in the packed qkv row.
        source = QKV + row * qkv_row + head * D
        if head < QH:
            weight = QW
            eps = q_eps
        else:
            weight = KW
            eps = k_eps
        if PREWAIT:
            norm_weight = tl.load(weight + col).to(tl.float32)
            partner_weight = tl.load(weight + partner).to(tl.float32)
            slot = tl.load(SLOT + row).to(tl.int64)
            if PDL:
                gdc_wait()
                gdc_launch()
        x = tl.load(source + col).to(tl.float32)
        x_partner = tl.load(source + partner).to(tl.float32)
        if PREWAIT:
            value = tl.load(QKV + row * qkv_row + (head + KH) * D + col, head >= QH, 0.)
        else:
            norm_weight = tl.load(weight + col).to(tl.float32)
            partner_weight = tl.load(weight + partner).to(tl.float32)
        rms = tl.rsqrt(tl.sum(x * x, 0) / D + eps)
        # Inductor drops the norm's intermediate BF16 cast: one rounding after the weight product.
        y = (x * rms * norm_weight).to(tl.bfloat16).to(tl.float32)
        y_partner = (x_partner * rms
                     * partner_weight).to(tl.bfloat16).to(tl.float32)
        rotated = (y * cosine + sign * (y_partner * sine)).to(tl.bfloat16)
        if head < QH:
            tl.store(QOUT + (row * QH + head) * D + col, rotated)
        else:
            kv_head = head - QH
            if not PREWAIT:
                slot = tl.load(SLOT + row).to(tl.int64)
            live = (col < D) & (slot >= 0)
            target = slot * (KH * D) + kv_head * D + col
            tl.store(KC + target, rotated, live)
            if not PREWAIT:
                value = tl.load(QKV + row * qkv_row + (QH + KH + kv_head) * D + col)
            tl.store(VC + target, value, live)


def supported(x, k_cache, v_cache, q_heads, kv_heads, dim):
    """True when the fused glue can serve this call (CUDA BF16 rows, dense paged caches)."""
    return (triton is not None and x.is_cuda and x.dtype == torch.bfloat16
            and dim in (64, 128, 256) and k_cache.dtype == v_cache.dtype == torch.bfloat16
            and k_cache.is_cuda and k_cache.is_contiguous() and v_cache.is_contiguous()
            and tuple(k_cache.shape[-2:]) == (kv_heads, dim) and v_cache.shape == k_cache.shape)


def qkv_glue(qkv, positions, cos_sin_cache, q_weight, k_weight, q_eps, k_eps,
             k_cache, v_cache, slot_mapping, q_heads, kv_heads, dim, pdl=False, *, prewait=False):
    """Normed, rotated queries [rows, q_heads * dim]; this step's keys and values go to the cache."""
    rows = qkv.shape[0]
    if (qkv.dim() != 2 or qkv.stride(-1) != 1 or qkv.shape[1] != (q_heads + 2 * kv_heads) * dim
            or cos_sin_cache.dtype != torch.float32 or cos_sin_cache.stride(-1) != 1
            or cos_sin_cache.shape[-1] != dim or slot_mapping.shape[0] < rows):
        raise ValueError('verify_layer8b glue needs dense qkv rows and an FP32 [positions, dim] cos/sin cache')
    query = qkv.new_empty(rows, q_heads * dim)
    pdl = pdl and USE_PDL
    _qkv_glue[(rows, q_heads + kv_heads)](
        qkv, qkv.stride(0), positions, cos_sin_cache, cos_sin_cache.stride(0),
        q_weight, k_weight, query, k_cache, v_cache, slot_mapping, float(q_eps), float(k_eps),
        QH=q_heads, KH=kv_heads, D=dim, PDL=pdl,
        **({'PREWAIT': True} if prewait else {}),
        num_warps=1, **(dict(launch_pdl=True) if pdl else {}))
    return query


# Knob value '2': the add+norm and SiLU-mul glue between GEMMs as Triton launches with
# programmatic dependent launch (PDL), so each starts while the GEMM before it drains.
if triton is not None:
    @triton.jit
    def store_producer_blocks(values, Q, SF, row, column, N: tl.constexpr,
                              BLOCK: tl.constexpr, TILE: tl.constexpr):
        """Quantize owned K blocks once and write four adjacent scale bytes per word."""
        col = column + tl.arange(0, BLOCK)
        q, byte = quantize_activation_block(tl.where(col < N, values, 0.), BLOCK)
        tl.store(Q + row * N + col, q, col < N)
        shifts = tl.arange(0, 4) * 8
        packed = tl.sum(tl.reshape(byte.to(tl.uint32), (BLOCK // 128, 4)) << shifts[None, :], 1)
        group = column // 128 + tl.arange(0, BLOCK // 128)
        atom: tl.constexpr = min(TILE, 128)
        within = row % atom
        physical = ((row // atom) * (N // 128) + group) * 128
        physical += (within % 32) * 4 + within // 32
        tl.store(SF.to(tl.pointer_type(tl.uint32)) + physical, packed, group < N // 128)

    @triton.jit
    def gdc_wait():
        # griddepcontrol.wait: block until the preceding grid's memory is visible (PDL secondary).
        # Triton 3.4's extra.cuda.gdc_wait does not compile under its own semantic, so inline PTX.
        tl.inline_asm_elementwise('griddepcontrol.wait; mov.b32 $0, $1;', '=r,r',
                                  [tl.full([1], 0, tl.int32)], dtype=tl.int32, is_pure=False, pack=1)

    @triton.jit
    def gdc_launch():
        # griddepcontrol.launch_dependents: let the next PDL grid launch; its wait still sees all our writes.
        tl.inline_asm_elementwise('griddepcontrol.launch_dependents; mov.b32 $0, $1;', '=r,r',
                                  [tl.full([1], 0, tl.int32)], dtype=tl.int32, is_pure=False, pack=1)

    @triton.jit
    def _add_norm(H, R, W, OUT, ROUT, eps, N: tl.constexpr, BLOCK: tl.constexpr, PDL: tl.constexpr,
                  Q, SF, TILE: tl.constexpr, EARLY: tl.constexpr, FP8STORE: tl.constexpr = False,
                  PRODBATCH: tl.constexpr = False, PREWAIT: tl.constexpr = False):
        # hidden + residual in FP32, residual rounded once, the norm of the unrounded sum.
        row = tl.program_id(0).to(tl.int64)
        col = tl.arange(0, BLOCK)
        live = col < N
        if PREWAIT:
            weight = tl.load(W + col, live, 0.).to(tl.float32)
            residual = tl.load(R + row * N + col, live, 0.).to(tl.float32)
        if PDL:
            if EARLY:
                gdc_launch()
            gdc_wait()
        row = tl.program_id(0).to(tl.int64)
        col = tl.arange(0, BLOCK)
        live = col < N
        if not PREWAIT:
            residual = tl.load(R + row * N + col, live, 0.).to(tl.float32)
        x = tl.load(H + row * N + col, live, 0.).to(tl.float32) + residual
        if FP8STORE:
            store_bf16_vector(ROUT + row * N + col, x, live)
        else:
            tl.store(ROUT + row * N + col, x.to(tl.bfloat16), live)
        rms = tl.rsqrt(tl.sum(x * x, 0) / N + eps)
        if not PREWAIT:
            weight = tl.load(W + col, live, 0.).to(tl.float32)
        output = (x * rms * weight).to(tl.bfloat16)
        if FP8STORE:
            store_bf16_vector(OUT + row * N + col, output, live)
        else:
            tl.store(OUT + row * N + col, output, live)
        if TILE:
            if PRODBATCH:
                store_producer_blocks(output, Q, SF, row, 0, N, BLOCK, TILE)
            else:
                store_activation_row(output, Q, SF, row, 0, N, BLOCK, TILE)

    @triton.jit
    def _silu_mul(X, OUT, N: tl.constexpr, BLOCK: tl.constexpr, PDL: tl.constexpr,
                  Q, SF, TILE: tl.constexpr, EARLY: tl.constexpr, FP8STORE: tl.constexpr = False,
                  PRODBATCH: tl.constexpr = False):
        if PDL:
            if EARLY:
                gdc_launch()
            gdc_wait()
        row = tl.program_id(0).to(tl.int64)
        col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        live = col < N
        gate = tl.load(X + row * 2 * N + col, live, 0.).to(tl.float32)
        up = tl.load(X + row * 2 * N + N + col, live, 0.).to(tl.float32)
        output = (gate / (1. + tl.exp(-gate)) * up).to(tl.bfloat16)
        if FP8STORE:
            store_bf16_vector(OUT + row * N + col, output, live)
        else:
            tl.store(OUT + row * N + col, output, live)
        if TILE:
            if PRODBATCH:
                store_producer_blocks(output, Q, SF, row, tl.program_id(1) * BLOCK, N, BLOCK, TILE)
            else:
                store_activation_row(output, Q, SF, row, tl.program_id(1) * BLOCK, N, BLOCK, TILE)

    @triton.jit
    def store_attention_rows(output, Q, SF, head, start, W: tl.constexpr, G: tl.constexpr,
                             KH: tl.constexpr, D: tl.constexpr, MR: tl.constexpr, TILE: tl.constexpr):
        # Each row owns complete 32-column quantization blocks. Gather selects
        # already-rounded BF16 values without introducing another reduction.
        for i in tl.static_range(MR):
            values = tl.reshape(tl.gather(output, tl.full((1, D), 0, tl.int32) + i, axis=0), (D,))
            row = start + i
            column = (head * G + row // W) * D
            store_activation_row(values, Q, SF, row % W, column, KH * G * D, D, TILE)


def pdl_available():
    return triton is not None and torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 9


def add_norm(hidden, residual, weight, eps, pdl=True, *, activation=None, early=False, fp8store=False, prodbatch=False, prewait=False):
    """BF16 add+norm; early PDL changes scheduling while retaining the original reduction and stores."""
    rows, width = hidden.shape
    out, rout = torch.empty_like(hidden), torch.empty_like(residual)
    pdl = pdl and USE_PDL
    extra = dict(launch_pdl=True) if pdl else {}
    layout, q, sf = activation if activation is not None else (None, None, None)
    _add_norm[(rows,)](hidden, residual, weight, out, rout, float(eps), N=width,
                       BLOCK=triton.next_power_of_2(width), PDL=pdl, Q=q, SF=sf,
                       TILE=0 if layout is None else layout.token_tile, EARLY=early, FP8STORE=fp8store,
                       PRODBATCH=prodbatch and width % 128 == 0,
                       **({'PREWAIT': True} if prewait else {}),
                       num_warps=8, **extra)
    if activation is not None:
        out._verify_w1prod8b = activation
    return out, rout


def silu_mul(gate_up, pdl=True, block=2048, *, activation=None, early=False, fp8store=False, prodbatch=False):
    """SiluAndMul of BF16 [rows, 2 * width] CUDA rows."""
    rows, width = gate_up.shape[0], gate_up.shape[1] // 2
    out = gate_up.new_empty(rows, width)
    pdl = pdl and USE_PDL
    extra = dict(launch_pdl=True) if pdl else {}
    layout, q, sf = activation if activation is not None else (None, None, None)
    _silu_mul[(rows, triton.cdiv(width, block))](gate_up, out, N=width, BLOCK=block, PDL=pdl,
                                              Q=q, SF=sf, TILE=0 if layout is None else layout.token_tile,
                                              EARLY=early, FP8STORE=fp8store,
                                              PRODBATCH=prodbatch and width % 128 == 0 and block % 128 == 0,
                                              num_warps=8, **extra)
    if activation is not None:
        out._verify_w1prod8b = activation
    return out


# Knob value '3': value '2' plus the paged causal attention of the N=1 verify row as a
# split-KV Triton launch and a merge launch. Each program attends one KV head's grouped
# query rows over one KV split; the merge combines the splits per query row.
SPLITS = 16
ATTENTION_TILE = 64  # keys per split-attention step
ATTENTION_WARPS = 4
_scratch = {}
USE_PDL = True

if triton is not None:
    @triton.jit
    def _split_attention(Q, KC, VC, TABLE, LENS, ACC, ML, scale_log2,
                         W: tl.constexpr, G: tl.constexpr, KH: tl.constexpr, D: tl.constexpr,
                         PAGE: tl.constexpr, S: tl.constexpr, R: tl.constexpr, RP: tl.constexpr,
                         BN: tl.constexpr, PDL: tl.constexpr):
        if PDL:
            gdc_wait()
            gdc_launch()
        head = tl.program_id(0)
        split = tl.program_id(1)
        rows = tl.arange(0, RP)
        dims = tl.arange(0, D)
        live_row = rows < R
        # Row r is query head head * G + r // W at token r % W.
        token = rows % W
        q_head = head * G + rows // W
        q = tl.load(Q + (token * (KH * G) + q_head)[:, None] * D + dims[None, :],
                    live_row[:, None], 0.)
        kv_len = tl.load(LENS)
        position = kv_len - W + token
        chunk = tl.cdiv(tl.cdiv(kv_len, S), BN) * BN
        start = split * chunk
        end = tl.minimum(start + chunk, kv_len)
        m = tl.full([RP], float('-inf'), tl.float32)
        l = tl.zeros([RP], tl.float32)
        acc = tl.zeros([RP, D], tl.float32)
        for tile in range(start, end, BN):
            keys = tile + tl.arange(0, BN)
            live_key = keys < end
            page = tl.load(TABLE + keys // PAGE, live_key, 0).to(tl.int64)
            slot = page * PAGE + keys % PAGE
            offsets = (slot * KH + head)[:, None] * D + dims[None, :]
            k = tl.load(KC + offsets, live_key[:, None], 0.)
            v = tl.load(VC + offsets, live_key[:, None], 0.)
            scores = tl.dot(q, tl.trans(k)) * scale_log2
            visible = live_key[None, :] & (keys[None, :] <= position[:, None])
            scores = tl.where(visible, scores, float('-inf'))
            m_new = tl.maximum(m, tl.max(scores, 1))
            m_safe = tl.where(m_new == float('-inf'), 0., m_new)
            p = tl.math.exp2(scores - m_safe[:, None])
            alpha = tl.math.exp2(m - m_safe)
            l = l * alpha + tl.sum(p, 1)
            acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v)
            m = m_new
        base = (head * S + split) * RP
        tl.store(ACC + (base + rows)[:, None] * D + dims[None, :], acc)
        tl.store(ML + (base + rows) * 2, m)
        tl.store(ML + (base + rows) * 2 + 1, l)

    @triton.jit
    def _merge_splits(ACC, ML, OUT, W: tl.constexpr, G: tl.constexpr, KH: tl.constexpr,
                      D: tl.constexpr, S: tl.constexpr, RP: tl.constexpr, PDL: tl.constexpr,
                      Q, SF, TILE: tl.constexpr):
        # One program per (KV head, grouped query row): combine the S split partials.
        if PDL:
            if TILE:
                gdc_launch()
            gdc_wait()
        head = tl.program_id(0)
        row = tl.program_id(1)
        splits = tl.arange(0, S)
        dims = tl.arange(0, D)
        at = (head * S + splits) * RP + row
        m = tl.load(ML + at * 2)
        l = tl.load(ML + at * 2 + 1)
        top = tl.max(m, 0)
        weight = tl.where(m == float('-inf'), 0., tl.math.exp2(m - top))
        acc = tl.load(ACC + at[:, None] * D + dims[None, :])
        out = tl.sum(weight[:, None] * acc, 0) / tl.sum(weight * l, 0)
        token = row % W
        q_head = head * G + row // W
        tl.store(OUT + (token * (KH * G) + q_head) * D + dims, out.to(tl.bfloat16))
        if TILE:
            store_activation_row(out.to(tl.bfloat16), Q, SF, token, q_head * D, KH * G * D, D, TILE)


def _attention_scratch(device, kv_heads, rows_padded, dim):
    key = (device, kv_heads, rows_padded, dim, SPLITS)
    if key not in _scratch:
        _scratch[key] = (torch.empty(kv_heads * SPLITS * rows_padded, dim, dtype=torch.float32, device=device),
                         torch.empty(kv_heads * SPLITS * rows_padded, 2, dtype=torch.float32, device=device))
    return _scratch[key]


def split_attention(query, k_cache, v_cache, block_table, context_lens, scale, q_heads, kv_heads, dim,
                    pdl=True, activation=None, mergefp8=False):
    """Causal paged attention of one verify row [width, q_heads * dim] (queries end at context_lens[0])."""
    width = query.shape[0]
    group = q_heads // kv_heads
    rows = group * width
    rows_padded = max(16, triton.next_power_of_2(rows))
    if (query.dtype != torch.bfloat16 or not query.is_contiguous() or block_table.dim() != 1
            or context_lens.numel() != 1 or q_heads % kv_heads or k_cache.shape[-2:] != (kv_heads, dim)):
        raise ValueError('split_attention serves one BF16 verify row over a dense paged cache')
    out = torch.empty_like(query)
    acc, ml = _attention_scratch(query.device, kv_heads, rows_padded, dim)
    pdl = pdl and USE_PDL
    extra = dict(launch_pdl=True) if pdl else {}
    layout, q, sf = activation if activation is not None else (None, None, None)
    _split_attention[(kv_heads, SPLITS)](
        query, k_cache, v_cache, block_table, context_lens, acc, ml,
        float(scale) * 1.4426950408889634, W=width, G=group, KH=kv_heads, D=dim,
        PAGE=k_cache.shape[1], S=SPLITS, R=rows, RP=rows_padded, BN=ATTENTION_TILE, PDL=pdl,
        num_warps=ATTENTION_WARPS, num_stages=2, **extra)
    merge_kernel = _merge_splits
    if mergefp8 and activation is not None and dim == 128:
        from jetspec.engines.jetspec_plus_plus.kernels.qwen3_8b.verify_mergefp8_8b import merge_decode as merge_kernel
    merge_kernel[(kv_heads, rows)](acc, ml, out, W=width, G=group, KH=kv_heads, D=dim, S=SPLITS,
                                    RP=rows_padded, PDL=pdl,
                                    Q=q, SF=sf, TILE=0 if layout is None else layout.token_tile,
                                    num_warps=4, **extra)
    if activation is not None:
        out._verify_w1prod8b = activation
    return out
