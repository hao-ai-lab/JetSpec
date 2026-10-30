"""Dense (8B) N=1 verify attention: split-KV launch whose first K/V tile does not wait for the glue.

verify_cut_persist8b=3 runs the paged causal attention of the verify row as a split-KV
launch and a merge launch (kernels/verify_layer8b.py). Each split's key range follows
kv_len, so a program reads the context length, then the page table, then K and V: three
dependent loads, all after the PDL wait on the glue launch that writes the queries.

Here split s owns the key tiles s, s + S, s + 2S, ... of KEY_TILE keys, so the first tile's
page and K/V addresses do not depend on kv_len. With PDL a program loads the context length,
the page and its first K/V tile BEFORE it waits on the glue launch. Only the keys of this
verify row (positions kv_len - width .. kv_len - 1), which the glue launch writes, are loaded
again after the wait. Out-of-range table entries and pages are clamped to valid memory and
masked out. The merge launch reads only the splits that hold keys.
The first tile is peeled from the online-softmax loop. Merge programs reduce four query
rows together. Width-one calls retain value '3', whose merge is cheaper for decode.
The softmax order differs from FlashInfer and from value '3' at some lengths, so results are
near the BF16 rounding error, not bitwise (class B).
"""
import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # The CPU suite runs without Triton.
    triton = None

SPLITS = 16  # KV splits per KV head; the grid is fixed at capture
BLOCK_ROWS = 64  # grouped query rows per split program
KEY_TILE = 64  # keys per tile; it divides the page size
WARPS = 4
MERGE_ROWS = 4  # grouped query rows reduced together per merge program
_scratch = {}


if triton is not None:
    from jetspec.engines.jetspec_plus_plus.kernels.qwen3_8b.verify_layer8b import gdc_launch, gdc_wait, store_attention_rows

    @triton.jit
    def _split_attention(Q, KC, VC, TABLE, LENS, ACC, ML, scale_log2, table_len, pages,
                         W: tl.constexpr, G: tl.constexpr, KH: tl.constexpr, D: tl.constexpr,
                         PAGE: tl.constexpr, S: tl.constexpr, R: tl.constexpr, RP: tl.constexpr,
                         BR: tl.constexpr, BN: tl.constexpr, PDL: tl.constexpr):
        # Program (KV head x row block, split). Row r of a KV head is query head head * G + r // W
        # at token r % W; RP pads the R = G * W rows to a multiple of BR.
        blocks: tl.constexpr = RP // BR
        head = tl.program_id(0) // blocks
        rows = (tl.program_id(0) % blocks) * BR + tl.arange(0, BR)
        split = tl.program_id(1)
        dims = tl.arange(0, D)
        live_row = rows < R
        token = rows % W
        q_head = head * G + rows // W
        # The first tile: its page and K/V do not depend on kv_len (clamped to valid memory).
        keys = split * BN + tl.arange(0, BN)
        page = tl.load(TABLE + tl.minimum(split * BN // PAGE, table_len - 1)).to(tl.int64)
        page = tl.minimum(tl.maximum(page, 0), pages - 1)
        offsets = ((page * PAGE + keys % PAGE) * KH + head)[:, None] * D + dims[None, :]
        k = tl.load(KC + offsets)
        v = tl.load(VC + offsets)
        kv_len = tl.load(LENS)
        if PDL:
            # The glue launch writes the queries and this row's keys and values.
            gdc_wait()
            gdc_launch()
            fresh = (keys >= kv_len - W) & (keys < kv_len)
            k = tl.where(fresh[:, None], tl.load(KC + offsets, fresh[:, None], 0.), k)
            v = tl.where(fresh[:, None], tl.load(VC + offsets, fresh[:, None], 0.), v)
        # Clamped memory past the context may hold any bits: zero those values (scores are masked).
        v = tl.where((keys < kv_len)[:, None], v, 0.)
        if split * BN < kv_len:
            q = tl.load(Q + (token * (KH * G) + q_head)[:, None] * D + dims[None, :],
                        live_row[:, None], 0.)
            position = kv_len - W + token
            # Most verify contexts need only this first tile per split. Peel it so
            # that path carries no online-softmax state through a loop backedge.
            scores = tl.dot(q, tl.trans(k)) * scale_log2
            visible = (keys[None, :] < kv_len) & (keys[None, :] <= position[:, None])
            scores = tl.where(visible, scores, float('-inf'))
            m = tl.max(scores, 1)
            m_safe = tl.where(m == float('-inf'), 0., m)
            p = tl.math.exp2(scores - m_safe[:, None])
            l = tl.sum(p, 1)
            acc = tl.dot(p.to(tl.bfloat16), v)
            for tile in range((split + S) * BN, kv_len, S * BN):
                keys = tile + tl.arange(0, BN)
                live_key = keys < kv_len
                page = tl.load(TABLE + tile // PAGE).to(tl.int64)
                offsets = ((page * PAGE + keys % PAGE) * KH + head)[:, None] * D + dims[None, :]
                k = tl.load(KC + offsets, live_key[:, None], 0.)
                v = tl.load(VC + offsets, live_key[:, None], 0.)
                scores = tl.dot(q, tl.trans(k)) * scale_log2
                visible = (keys[None, :] < kv_len) & (keys[None, :] <= position[:, None])
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
    def _merge_splits(ACC, ML, LENS, OUT, W: tl.constexpr, G: tl.constexpr, KH: tl.constexpr,
                      D: tl.constexpr, S: tl.constexpr, RP: tl.constexpr, BN: tl.constexpr,
                      MR: tl.constexpr, PDL: tl.constexpr,
                      Q, SF, TILE: tl.constexpr):
        if PDL and TILE:
            gdc_launch()
        # Reduce MR query rows together so neighbouring partials share one program.
        kv_len = tl.load(LENS)
        if PDL:
            gdc_wait()
        head = tl.program_id(0)
        splits = tl.arange(0, S)
        dims = tl.arange(0, D)
        active = splits < tl.cdiv(kv_len, BN)
        rows = tl.program_id(1) * MR + tl.arange(0, MR)
        at = (head * S + splits[None, :]) * RP + rows[:, None]
        m = tl.load(ML + at * 2, active[None, :], float('-inf'))
        l = tl.load(ML + at * 2 + 1, active[None, :], 0.)
        top = tl.max(m, 1)
        weight = tl.where(m == float('-inf'), 0., tl.math.exp2(m - top[:, None]))
        acc = tl.load(ACC + at[:, :, None] * D + dims[None, None, :], active[None, :, None], 0.)
        out = tl.sum(weight[:, :, None] * acc, 1) / tl.sum(weight * l, 1)[:, None]
        token = rows % W
        q_head = head * G + rows // W
        tl.store(OUT + (token * (KH * G) + q_head)[:, None] * D + dims[None, :], out.to(tl.bfloat16))
        if TILE:
            store_attention_rows(out.to(tl.bfloat16), Q, SF, head, tl.program_id(1) * MR, W, G, KH, D, MR, TILE)


def _attention_scratch(device, kv_heads, splits, rows_padded, dim):
    key = (device, kv_heads, splits, rows_padded, dim)
    if key not in _scratch:
        _scratch[key] = (torch.empty(kv_heads * splits * rows_padded, dim, dtype=torch.float32, device=device),
                         torch.empty(kv_heads * splits * rows_padded, 2, dtype=torch.float32, device=device))
    return _scratch[key]


def split_attention(query, k_cache, v_cache, block_table, context_lens, scale, q_heads, kv_heads, dim,
                    pdl=True, splits=None, block_rows=None, key_tile=None, warps=None, merge_rows=None,
                    activation=None, mergefp8=False):
    """Causal paged attention of one verify row [width, q_heads * dim] (queries end at context_lens[0]).

    ``splits``, ``block_rows``, ``key_tile``, ``warps`` and ``merge_rows`` override the module
    defaults (timing sweeps only).
    """
    from jetspec.engines.jetspec_plus_plus.kernels.qwen3_8b import verify_layer8b
    width = query.shape[0]
    if width == 1 and all(value is None for value in (splits, block_rows, key_tile, warps, merge_rows)):
        # Decode has too few query rows to amortize the grouped merge.
        tuned = {'mergefp8': True} if mergefp8 else {}
        return verify_layer8b.split_attention(query, k_cache, v_cache, block_table, context_lens,
                                               scale, q_heads, kv_heads, dim, pdl=pdl, activation=activation,
                                               **tuned)
    group = q_heads // kv_heads
    rows = group * width
    splits = splits or SPLITS
    tile = key_tile or KEY_TILE
    block = min(block_rows or BLOCK_ROWS, max(16, triton.next_power_of_2(rows)))
    rows_padded = triton.cdiv(rows, block) * block
    merge = merge_rows or (MERGE_ROWS if width > 1 else 1)
    page = k_cache.shape[1]
    if (query.dtype != torch.bfloat16 or not query.is_contiguous() or block_table.dim() != 1
            or context_lens.numel() != 1 or q_heads % kv_heads or k_cache.shape[-2:] != (kv_heads, dim)
            or splits & (splits - 1) or page % tile or rows % merge):
        raise ValueError('verify_attn8b serves one BF16 verify row over a dense paged cache')
    out = torch.empty_like(query)
    acc, ml = _attention_scratch(query.device, kv_heads, splits, rows_padded, dim)
    pdl = pdl and verify_layer8b.USE_PDL
    extra = dict(launch_pdl=True) if pdl else {}
    layout, q, sf = activation if activation is not None else (None, None, None)
    _split_attention[(kv_heads * (rows_padded // block), splits)](
        query, k_cache, v_cache, block_table, context_lens, acc, ml, float(scale) * 1.4426950408889634,
        block_table.shape[0], k_cache.shape[0], W=width, G=group, KH=kv_heads, D=dim, PAGE=page,
        S=splits, R=rows, RP=rows_padded, BR=block, BN=tile, PDL=pdl, num_warps=warps or WARPS,
        num_stages=1, **extra)
    merge_kernel = _merge_splits
    if mergefp8 and activation is not None and dim == 128:
        from jetspec.engines.jetspec_plus_plus.kernels.qwen3_8b.verify_mergefp8_8b import merge as merge_kernel
    merge_kernel[(kv_heads, rows // merge)](acc, ml, context_lens, out, W=width, G=group, KH=kv_heads,
                                             D=dim, S=splits, RP=rows_padded, BN=tile, MR=merge, PDL=pdl,
                                             Q=q, SF=sf, TILE=0 if layout is None else layout.token_tile,
                                             num_warps=4, **extra)
    if activation is not None:
        out._verify_w1prod8b = activation
    return out
