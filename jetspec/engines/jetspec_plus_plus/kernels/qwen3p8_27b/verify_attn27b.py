"""27B gated-attention N=1 verify: split-KV paged causal attention at head_dim 256, gate in the merge.

The eager chain is the FlashInfer paged prefill of the verify row (its own kernels for the
causal pass and the merge of its KV chunks) and the gate multiply. Here one launch attends
each KV head's grouped query rows over one KV split, in row blocks, and a second launch
merges the splits, rounds to BF16 and applies the output gate with the arithmetic of
kernels/verify_mlp_glue.py gate_mul (sigmoid rounded to BF16, one rounded product). The
softmax order differs from FlashInfer, so results are near the BF16 rounding error, not
bitwise (class B). The gate product is bitwise to gate_mul on the same attention output.
"""
import torch

try:
    import triton
    import triton.language as tl
    from triton.language.extra.cuda import libdevice
except ImportError:  # The CPU suite runs without Triton.
    triton = None

SPLITS = 16  # KV splits per (KV head, row block)
BLOCK_ROWS = 32  # grouped query rows per program
KEY_TILE = 32  # keys per split-attention step
WARPS = 4
STAGES = 2
_scratch = {}


if triton is not None:
    from jetspec.engines.jetspec_plus_plus.kernels.verify_mlp_glue import _product, _ratio
    from jetspec.engines.jetspec_plus_plus.kernels.nvfp4._w4a8_quant import store_activation_row
    from jetspec.engines.jetspec_plus_plus.kernels.verify_w1 import store_activation_packed
    from jetspec.engines.jetspec_plus_plus.kernels.qwen3p8_27b.verify_pdl27b import producer_start

    @triton.jit
    def _split_attention(Q, KC, VC, TABLE, LENS, ACC, ML, scale_log2, q_token,
                         W: tl.constexpr, G: tl.constexpr, KH: tl.constexpr, D: tl.constexpr,
                         PAGE: tl.constexpr, S: tl.constexpr, R: tl.constexpr, RP: tl.constexpr,
                         BR: tl.constexpr, BN: tl.constexpr):
        # Program (KV head x row block, split). Row r of a KV head is query head head * G + r // W
        # at token r % W; RP pads the R = G * W rows to a multiple of BR.
        blocks: tl.constexpr = RP // BR
        head = tl.program_id(0) // blocks
        block = tl.program_id(0) % blocks
        split = tl.program_id(1)
        rows = block * BR + tl.arange(0, BR)
        dims = tl.arange(0, D)
        live_row = rows < R
        token = rows % W
        q_head = head * G + rows // W
        q = tl.load(Q + token[:, None] * q_token + q_head[:, None] * D + dims[None, :],
                    live_row[:, None], 0.)
        kv_len = tl.load(LENS)
        position = kv_len - W + token
        chunk = tl.cdiv(tl.cdiv(kv_len, S), BN) * BN
        start = split * chunk
        end = tl.minimum(start + chunk, kv_len)
        m = tl.full([BR], float('-inf'), tl.float32)
        l = tl.zeros([BR], tl.float32)
        acc = tl.zeros([BR, D], tl.float32)
        for tile in range(start, end, BN):
            keys = tile + tl.arange(0, BN)
            live_key = keys < end
            page = tl.load(TABLE + keys // PAGE, live_key, 0).to(tl.int64)
            slot = page * PAGE + keys % PAGE
            offsets = (slot * KH + head)[:, None] * D + dims[None, :]
            k_tile = tl.load(KC + offsets, live_key[:, None], 0.)
            v_tile = tl.load(VC + offsets, live_key[:, None], 0.)
            scores = tl.dot(q, tl.trans(k_tile)) * scale_log2
            visible = live_key[None, :] & (keys[None, :] <= position[:, None])
            scores = tl.where(visible, scores, float('-inf'))
            m_new = tl.maximum(m, tl.max(scores, 1))
            m_safe = tl.where(m_new == float('-inf'), 0., m_new)
            p = tl.math.exp2(scores - m_safe[:, None])
            alpha = tl.math.exp2(m - m_safe)
            l = l * alpha + tl.sum(p, 1)
            acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v_tile)
            m = m_new
        base = (head * S + split) * RP
        tl.store(ACC + (base + rows)[:, None] * D + dims[None, :], acc)
        tl.store(ML + (base + rows) * 2, m)
        tl.store(ML + (base + rows) * 2 + 1, l)

    @triton.jit
    def _merge_gate(ACC, ML, GATE, OUT, g_token, g_head, W: tl.constexpr, G: tl.constexpr,
                    KH: tl.constexpr, D: tl.constexpr, S: tl.constexpr, RP: tl.constexpr,
                    GATED: tl.constexpr, Q, SF, TILE: tl.constexpr, PDL: tl.constexpr,
                    FP8STORE: tl.constexpr = False):
        if PDL:
            producer_start(tl.program_id(0))
        # One program per (KV head, grouped query row): combine the S split partials.
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
        out = (tl.sum(weight[:, None] * acc, 0) / tl.sum(weight * l, 0)).to(tl.bfloat16)
        token = (row % W).to(tl.int64)
        q_head = head * G + row // W
        if GATED:
            gate = tl.load(GATE + token * g_token + q_head * g_head + dims).to(tl.float32)
            one = tl.full([D], 1.0, tl.float32)
            sigmoid = _ratio(one, libdevice.exp(-gate)).to(tl.bfloat16).to(tl.float32)
            out = _product(out.to(tl.float32), sigmoid).to(tl.bfloat16)
        tl.store(OUT + (token * (KH * G) + q_head) * D + dims, out)
        if TILE:
            if FP8STORE:
                store_activation_packed(out, Q, SF, token, q_head * D, KH * G * D, D, TILE)
            else:
                store_activation_row(out, Q, SF, token, q_head * D, KH * G * D, D, TILE)


def supported(query, k_cache, v_cache, q_heads, kv_heads, dim):
    """True when the split attention can serve this call (CUDA BF16 rows, dense paged caches)."""
    return (triton is not None and query.is_cuda and query.dtype == torch.bfloat16
            and dim in (64, 128, 256) and q_heads % kv_heads == 0 and 0 < query.shape[0] <= 64
            and k_cache.dtype == v_cache.dtype == torch.bfloat16 and k_cache.is_cuda
            and k_cache.is_contiguous() and v_cache.is_contiguous()
            and tuple(k_cache.shape[-2:]) == (kv_heads, dim) and v_cache.shape == k_cache.shape)


def _attention_scratch(device, kv_heads, rows_padded, dim, splits):
    key = (device, kv_heads, rows_padded, dim, splits)
    if key not in _scratch:
        _scratch[key] = (torch.empty(kv_heads * splits * rows_padded, dim, dtype=torch.float32, device=device),
                         torch.empty(kv_heads * splits * rows_padded, 2, dtype=torch.float32, device=device))
    return _scratch[key]


def split_attention(query, k_cache, v_cache, block_table, context_lens, scale, q_heads, kv_heads, dim,
                    gate=None, *, activation=None, pdl=False, fp8store=False):
    """Causal paged attention of one verify row; ``gate`` ([width, q_heads, dim] view) gates the output.

    ``query`` is [width, q_heads * dim] (any token stride, dense heads); its rows end at
    ``context_lens[0]``. This step's keys and values must already be in the cache.
    Returns [width, q_heads * dim] BF16.
    """
    width = query.shape[0]
    group = q_heads // kv_heads
    rows = group * width
    block = min(BLOCK_ROWS, max(16, triton.next_power_of_2(rows)))
    rows_padded = triton.cdiv(rows, block) * block
    if (query.dtype != torch.bfloat16 or query.dim() != 2 or query.stride(-1) != 1
            or query.shape[1] != q_heads * dim or block_table.dim() != 1 or context_lens.numel() != 1
            or q_heads % kv_heads or k_cache.shape[-2:] != (kv_heads, dim)):
        raise ValueError('verify_attn27b serves one BF16 verify row over a dense paged cache')
    if gate is not None and (gate.shape != (width, q_heads, dim) or gate.stride(-1) != 1
                             or gate.dtype != torch.bfloat16):
        raise ValueError('verify_attn27b gate must be a BF16 [width, heads, dim] view with dense dims')
    out = query.new_empty(width, q_heads * dim)
    acc, ml = _attention_scratch(query.device, kv_heads, rows_padded, dim, SPLITS)
    _split_attention[(kv_heads * (rows_padded // block), SPLITS)](
        query, k_cache, v_cache, block_table, context_lens, acc, ml,
        float(scale) * 1.4426950408889634, query.stride(0), W=width, G=group, KH=kv_heads, D=dim,
        PAGE=k_cache.shape[1], S=SPLITS, R=rows, RP=rows_padded, BR=block, BN=KEY_TILE,
        num_warps=WARPS, num_stages=STAGES)
    gated = gate is not None
    layout, q, sf = activation if activation is not None else (None, None, None)
    _merge_gate[(kv_heads, rows)](
        acc, ml, gate if gated else out, out, gate.stride(0) if gated else 0, gate.stride(1) if gated else 0,
        W=width, G=group, KH=kv_heads, D=dim, S=SPLITS, RP=rows_padded, GATED=gated,
        Q=q, SF=sf, TILE=0 if layout is None else layout.token_tile, PDL=pdl,
        FP8STORE=fp8store, num_warps=4)
    if activation is not None:
        out._verify_w1 = activation
    return out
