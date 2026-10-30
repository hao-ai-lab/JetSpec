"""Fixed-refresh GQA over device-visible prefix, tail and earlier row slots."""

import torch

from jetspec.engines.jetspec_plus_plus.jetspec.c1_fused import attention_output


_eval_workspaces = {}


def refresh_workspace(q, heads, rows, splits):
    """Retain eval scratch addresses for every graph, isolated by stream/shape."""
    from jetspec.engines.jetspec_plus_plus.jetspec.profile import enabled as profile_enabled

    def allocate():
        return (torch.empty((heads, rows, splits, 128), device=q.device, dtype=torch.float32),
                torch.empty((heads, rows, splits), device=q.device, dtype=torch.float32))

    if not profile_enabled():
        return allocate()
    stream = torch.cuda.current_stream(q.device).cuda_stream if q.is_cuda else None
    key = (q.device, stream, heads, rows, splits)
    if key not in _eval_workspaces:
        _eval_workspaces[key] = allocate()
    return _eval_workspaces[key]


def attend(q, k, v, scale, length, tail, block, starts, *, tune=False):
    """B=1 attention; starts and tail contain one device integer per query."""
    heads, rows, dim = q.shape[1:]
    kv_heads, keys = k.shape[1:3]
    if (q.shape[0] != 1 or dim != 128 or k.shape != v.shape or k.shape[-1] != dim
            or heads % kv_heads or tail.numel() != rows or starts.numel() != rows):
        raise ValueError("refresh attention requires B=1, D=128 and per-query metadata")
    groups = heads // kv_heads
    if groups not in (1, 2, 4, 8):
        raise ValueError("refresh attention requires 1, 2, 4 or 8 query heads per KV head")
    if q.device.type == "cpu":
        # Executable reference for the visibility contract, without a dense bias.
        output = attention_output(q)
        for r in range(rows):
            slots = torch.cat((torch.arange(int(length), device=q.device),
                               torch.arange(int(length), int(length) + int(tail[r]), device=q.device),
                               torch.arange(int(block) + int(starts[r]),
                                            max(int(block) + int(starts[r]), int(block) + r),
                                            device=q.device)))
            slots = slots[slots < keys]
            kr = k[0, :, slots].repeat_interleave(groups, 0).float()
            vr = v[0, :, slots].repeat_interleave(groups, 0).float()
            scores = torch.einsum("hd,hsd->hs", q[0, :, r].float(), kr) * scale
            output[0, :, r] = torch.einsum("hs,hsd->hd", scores.softmax(-1), vr)
        return output
    import triton

    global _partials, _merge
    if _partials is None:
        _partials, _merge = _kernels()
    splits = triton.cdiv(keys, 128)
    partial, lse = refresh_workspace(q, heads, rows, splits)
    output = attention_output(q)
    from jetspec.engines.jetspec_plus_plus.kernels.nvfp4.producers import attention_activation, _arguments
    produced = attention_activation(output)
    from jetspec.engines.jetspec_plus_plus.profiles import kernel_value
    extra = ({'BATCHED': True} if produced is not None and
             kernel_value('draft_fuse_am', '0') == '2' else {})
    row_tile = 16
    from jetspec.engines.jetspec_plus_plus.serving.compile_pin import tuning_choice
    choice = tuning_choice('refresh_attention', [groups, dim, str(q.dtype), bool(tune)],
        lambda: dict(row_tile=row_tile, tune=bool(tune)))
    row_tile, tune = choice['row_tile'], choice['tune']
    _partials[(kv_heads, triton.cdiv(rows, row_tile), splits)](
        q, k, v, length, tail, block, starts, partial, lse,
        *q.stride()[1:], *k.stride()[1:], *v.stride()[1:],
        rows, keys, splits, groups, scale,
        QUERY_TILE=triton.next_power_of_2(row_tile * groups), ROW_TILE=row_tile,
        TUNE=tune, num_warps=4)
    _merge[(heads, triton.cdiv(rows, 16))](
        partial, lse, output, *output.stride()[1:], rows, splits,
        Q=None if produced is None else produced.data,
        SF=() if produced is None else _arguments(produced)[1],
        TILES=() if produced is None else produced.tiles,
        STRIDES=() if produced is None else _arguments(produced)[3],
        WIDTH=heads * dim, MX8=produced is not None,
        SPLITS=triton.next_power_of_2(splits), num_warps=4, **extra)
    return output


def _kernels():
    import triton
    import triton.language as tl
    from jetspec.engines.jetspec_plus_plus.kernels.attn_sm80 import _kernels as sm80_kernels

    @triton.jit
    def partials(Q, K, V, Length, Tail, Block, Starts, P, L,
                 qh: tl.constexpr, qr: tl.constexpr, qd: tl.constexpr,
                 kh: tl.constexpr, kk: tl.constexpr, kd: tl.constexpr,
                 vh: tl.constexpr, vk: tl.constexpr, vd: tl.constexpr,
                 rows: tl.constexpr, keys: tl.constexpr, splits: tl.constexpr,
                 groups: tl.constexpr, scale: tl.constexpr, QUERY_TILE: tl.constexpr,
                 ROW_TILE: tl.constexpr = 16, TUNE: tl.constexpr = False,
                 KEY_TILE: tl.constexpr = 128):
        h, rb, split = tl.program_id(0), tl.program_id(1), tl.program_id(2)
        query_index = tl.arange(0, QUERY_TILE)
        r = rb * ROW_TILE + query_index // groups
        query_head = h * groups + query_index % groups
        s = split * KEY_TILE + tl.arange(0, KEY_TILE)
        d = tl.arange(0, 128)
        length = tl.load(Length)
        block = tl.load(Block)
        if TUNE:
            # Decide split usefulness on one scalar per query. Keeping the dense
            # visibility matrix out of the dot's live range lowers register use.
            low = split * KEY_TILE
            high = tl.minimum(low + KEY_TILE, keys)
            tail = tl.load(Tail + r, r < rows, other=0)
            start = tl.load(Starts + r, r < rows, other=rows)
            overlaps = ((low < length) |
                        ((high > length) & (low < length + tail)) |
                        ((high > block + start) & (low < block + r) & (start < r)))
            needed = tl.sum((overlaps & (r < rows)).to(tl.int32), 0) > 0
        else:
            tail = tl.load(Tail + r, r < rows, other=0)
            start = tl.load(Starts + r, r < rows, other=rows)
            visible = ((s[None, :] < length) |
                       ((s[None, :] >= length) & (s[None, :] < length + tail[:, None])) |
                       ((s[None, :] >= block + start[:, None]) &
                        (s[None, :] < block + r[:, None])))
            needed = tl.sum(tl.sum(visible.to(tl.int32), 1), 0) > 0
        query = tl.load(Q + query_head[:, None] * qh + r[:, None] * qr + d[None, :] * qd,
                        r[:, None] < rows, other=0)
        # Entire invisible splits avoid reading either cache tensor.
        if needed:
            key = tl.load(K + h * kh + s[:, None] * kk + d[None, :] * kd,
                          s[:, None] < keys, other=0)
            value = tl.load(V + h * vh + s[:, None] * vk + d[None, :] * vd,
                            s[:, None] < keys, other=0)
            scores = tl.dot(query, tl.trans(key)) * scale
            if TUNE:
                visible = ((s[None, :] < length) |
                           ((s[None, :] >= length) & (s[None, :] < length + tail[:, None])) |
                           ((s[None, :] >= block + start[:, None]) &
                            (s[None, :] < block + r[:, None])))
            scores = tl.where(visible & (s[None, :] < keys), scores, -float("inf"))
            peak = tl.max(scores, 1)
            weights = tl.exp(scores - tl.where(peak == -float("inf"), 0., peak)[:, None])
            denom = tl.sum(weights, 1)
            numerator = tl.dot(weights.to(value.dtype), value)
            result = numerator / tl.where(denom == 0, 1., denom)[:, None]
            logsum = tl.where(denom == 0, -float("inf"), peak + tl.log(denom))
        else:
            result = tl.full((QUERY_TILE, 128), 0., tl.float32)
            logsum = tl.full((QUERY_TILE,), -float("inf"), tl.float32)
        tl.store(P + (query_head[:, None] * rows + r[:, None]) * splits * 128 + split * 128 + d[None, :],
                 result, r[:, None] < rows)
        tl.store(L + (query_head * rows + r) * splits + split, logsum, r < rows)

    return partials, sm80_kernels()[1]


_partials = _merge = None
