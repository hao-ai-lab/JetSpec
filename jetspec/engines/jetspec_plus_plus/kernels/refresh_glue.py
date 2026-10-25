"""Refresh decoder prelude and GQA over separate cached-prefix and local K/V."""

from jetspec.engines.jetspec_plus_plus.profiles import kernel_enabled, kernel_value
import os

import torch
from jetspec.engines.jetspec_plus_plus.models.config import setting


def enabled(*, run_config=None):
    value = setting(run_config, 'engine', 'engine_settings.refresh_swap_glue', '0')
    if value not in ('0', '1'):
        raise ValueError('JETSPEC_REFRESH_SWAP_GLUE must be 0 or 1')
    if value == '1' and setting(run_config, 'engine', 'engine_settings.refresh_swap', '0') != '1':
        raise ValueError('JETSPEC_REFRESH_SWAP_GLUE=1 requires JETSPEC_REFRESH_SWAP=1')
    return value == '1'


def allocate(model, prefix_kv, bucket, rows, lanes, like):
    from jetspec.engines.jetspec_plus_plus.profiles import kernel_enabled
    slots = lanes - 1
    tokens = 4 + lanes + rows * slots
    dim = model.layers[0].self_attn.head_dim
    for layer in model.layers:
        key, value = prefix_kv[layer.self_attn.layer_idx]
        groups = layer.self_attn.num_key_value_groups
        if key.shape != value.shape or key.shape[0] != 1 or key.shape[-2] < bucket:
            raise ValueError('refresh glue requires B=1 prefix K/V covering the bucket')
        if like.is_cuda and (dim != 128 or groups not in (1, 2, 4, 8)):
            raise ValueError('refresh glue CUDA requires D=128 and GQA ratio 1, 2, 4 or 8')
    hidden = like.new_empty(1, tokens, model.hidden_size)
    positions = torch.arange(bucket + 32, device=like.device)[None]
    cos, sin = model.rotary(hidden, positions)
    state = dict(glue=True, prefix_kv=prefix_kv, bucket=bucket, rows=rows, lanes=lanes,
                 width=bucket + (tokens + 15) // 16 * 16,
                 slots=slots, tokens=tokens, hidden=hidden,
                 cos=like.new_empty(1, tokens, dim), sin=like.new_empty(1, tokens, dim),
                 rope_table=(cos[0].contiguous(), sin[0].contiguous()))
    if kernel_enabled('refresh_swap_lean', '1') and like.is_cuda and like.dtype == torch.bfloat16:
        from jetspec.engines.jetspec_plus_plus.jetspec.chain.refresh_swap import allocate_compact
        state['compact'] = allocate_compact(rows, slots, model.lm_head.weight.shape[0], like.device)
    return state


def prelude(state, length, true_taps, tail, tail_counts, block, slot_counts):
    hidden, cos, sin = state['hidden'], state['cos'], state['sin']
    lanes, slots = state['lanes'], state['slots']
    tokens, width = hidden.shape[1:]
    if not hidden.is_cuda:
        local = torch.arange(tokens, device=hidden.device)
        row = ((local - 4 - lanes).clamp_min(0) // slots)
        index = (local - 4 - lanes).clamp_min(0) % slots
        positions = torch.where(local < 4, length - 4 + local,
                                torch.where(local < 4 + lanes, length + local - 4,
                                            length + tail_counts[row] + index)).clamp_min(0)
        valid = torch.arange(slots)[None] < slot_counts[:, None]
        block_hidden = torch.where(valid[..., None], block[:, :slots, -width:], 0)
        hidden[0].copy_(torch.cat((true_taps, tail[:lanes], block_hidden.reshape(-1, width))))
        cos.copy_(state['rope_table'][0].index_select(0, positions)[None])
        sin.copy_(state['rope_table'][1].index_select(0, positions)[None])
        return hidden, (cos, sin)
    import triton
    global _stage, _partials, _merge
    if _stage is None:
        _stage, _partials, _merge = _kernels()
    dim = cos.shape[-1]
    hb = triton.cdiv(tokens * width, 256)
    rb = triton.cdiv(tokens * dim, 256)
    from jetspec.engines.jetspec_plus_plus.kernels.nvfp4.entry import stage_entry
    if stage_entry(hidden, block, true_taps, tail, _stage,
                   (length, true_taps, tail, tail_counts, block, slot_counts,
                    *state['rope_table'], hidden, cos, sin),
                   (*true_taps.stride(), *tail.stride(), *block.stride(),
                    lanes, slots, tokens, width, dim, state['rope_table'][0].shape[0], 0, 256), rb):
        return hidden, (cos, sin)
    _stage[(hb + rb,)](length, true_taps, tail, tail_counts, block, slot_counts,
                       *state['rope_table'], hidden, cos, sin,
                       *true_taps.stride(), *tail.stride(), *block.stride(),
                       lanes, slots, tokens, width, dim, state['rope_table'][0].shape[0], hb,
                       BLOCK=256)
    return hidden, (cos, sin)


def attend(q, pk, pv, k, v, length, tails, lanes, slots, scale, *, run_config=None):
    """Cached columns below length-4 plus the true four, causal tail and private row."""
    heads, tokens, dim = q.shape[1:]
    kv_heads, bucket = pk.shape[1:3]
    groups = heads // kv_heads
    if not q.is_cuda:
        output = torch.empty_like(q)
        prefix = max(0, int(length) - 4)
        for r in range(tokens):
            if r < 4:
                local = torch.arange(r + 1)
            elif r < 4 + lanes:
                local = torch.arange(r + 1)
            else:
                row, index = divmod(r - 4 - lanes, slots)
                local = torch.cat((torch.arange(4 + int(tails[row])),
                                   torch.arange(4 + lanes + row * slots, r + 1)))
            for h in range(kv_heads):
                key = torch.cat((pk[0, h, :prefix], k[0, h, local]))
                value = torch.cat((pv[0, h, :prefix], v[0, h, local]))
                output[0, h * groups:(h + 1) * groups, r:r + 1] = (
                    torch.nn.functional.scaled_dot_product_attention(
                        q[0, h * groups:(h + 1) * groups, r:r + 1],
                        key[None], value[None], scale=scale))
        return output
    import triton
    global _stage, _partials, _merge
    if _partials is None:
        _stage, _partials, _merge = _kernels()
    prefix_splits = triton.cdiv(bucket, 128)
    splits = prefix_splits + triton.cdiv(tokens, 128)
    partial = torch.empty((heads, tokens, splits, 128), device=q.device, dtype=torch.float32)
    lse = torch.empty((heads, tokens, splits), device=q.device, dtype=torch.float32)
    output = torch.empty_like(q)
    args = (q, pk, pv, k, v, length, tails, partial, lse,
        *q.stride()[1:], *pk.stride()[1:], *pv.stride()[1:],
        *k.stride()[1:], *v.stride()[1:],
        tokens, bucket, prefix_splits, splits, groups, lanes, slots, scale)
    merge_args = (partial, lse, output, *output.stride()[1:], tokens, splits)
    if setting(run_config, 'engine', 'engine_settings.refresh_decoder_loop', '0') == '1':
        from jetspec.engines.jetspec_plus_plus.jetspec.rows import fused_kernels
        launch_cached = fused_kernels().launch_cached
        # Raw compiled launchers require every kernel argument, including constexprs.
        launch_cached(_partials, (kv_heads, triton.cdiv(tokens, 16), splits),
                      (*args, triton.next_power_of_2(16 * groups)), num_warps=4)
        launch_cached(_merge, (heads, triton.cdiv(tokens, 16)),
                      (*merge_args, triton.next_power_of_2(splits)), num_warps=4)
    else:
        _partials[(kv_heads, triton.cdiv(tokens, 16), splits)](
            *args, QUERY_TILE=triton.next_power_of_2(16 * groups), num_warps=4)
        _merge[(heads, triton.cdiv(tokens, 16))](
            *merge_args, SPLITS=triton.next_power_of_2(splits), num_warps=4)
    return output


def layers(model, state, *, length, true_taps, tail, tail_counts, block, slot_counts, run_config=None):
    from jetspec.engines.jetspec_plus_plus.models.decoder_head import decoder_window_project
    from jetspec.engines.jetspec_plus_plus.models.decoder_head import decoder_window_finish
    from jetspec.engines.jetspec_plus_plus.models.decoder_head import decoder_lean_project
    from jetspec.engines.jetspec_plus_plus.models.decoder_head import decoder_lean_finish
    from jetspec.engines.jetspec_plus_plus.models.decoder_head import fused_kernels
    hidden, rope = prelude(state, length, true_taps, tail, tail_counts, block, slot_counts)
    current, pending = hidden, None
    fused = fused_kernels() if hidden.is_cuda else None
    for index, layer in enumerate(model.layers):
        if hidden.is_cuda:
            q, k, v, residual = decoder_lean_project(
                layer, fused, hidden if pending is None else None, pending, rope)
        else:
            q, k, v = decoder_window_project(layer, current, rope)
        pk, pv = state['prefix_kv'][layer.self_attn.layer_idx]
        attended = attend(q, pk[..., :state['bucket'], :], pv[..., :state['bucket'], :],
                          k, v, length, tail_counts, state['lanes'], state['slots'],
                          layer.self_attn.scaling, run_config=run_config).transpose(1, 2).contiguous()
        if kernel_enabled('draft_fuse_t12', '1') and index == len(model.layers) - 1:
            start = 4 + state['lanes']
            attended = attended[:, start:]
            if hidden.is_cuda:
                residual = residual[:, start:]
            else:
                current = current[:, start:]
        if hidden.is_cuda:
            pending = decoder_lean_finish(layer, fused, residual, attended)
        else:
            current = decoder_window_finish(layer, current, attended)
    if hidden.is_cuda:
        decoded, _ = fused.fast_add_rms_norm(*pending, model.norm.weight, model.norm.variance_epsilon)
    else:
        decoded = model.norm(current)
    return hidden, decoded


def attention_launches(model, state, length, tails, fused):
    """Reuse warmed SWAP_GLUE specializations; None preserves cold/hooked launches."""
    if _partials is None:
        return None
    key = (length.dtype, length.data_ptr() % 16 == 0, tails.dtype, tails.data_ptr() % 16 == 0)
    entries = state.get('loop_attention_entries', {}).get(key)
    if fused._has_launch_hooks(fused.triton.knobs.runtime):
        return None
    if entries is not None:
        return entries
    candidates = []
    tokens, bucket = state['tokens'], state['bucket']
    for layer in model.layers:
        attn = layer.self_attn
        heads = attn.q_proj.out_features // attn.head_dim
        kv_heads = attn.k_proj.out_features // attn.head_dim
        groups = attn.num_key_value_groups
        q = state['hidden'].new_empty(1, heads, tokens, attn.head_dim)
        k = q.new_empty(1, kv_heads, tokens, attn.head_dim)
        pk, pv = [t[..., :bucket, :] for t in state['prefix_kv'][attn.layer_idx]]
        prefix_splits = fused.triton.cdiv(bucket, 128)
        splits = prefix_splits + fused.triton.cdiv(tokens, 128)
        partial = torch.empty((heads, tokens, splits, 128), device=q.device, dtype=torch.float32)
        lse = torch.empty((heads, tokens, splits), device=q.device, dtype=torch.float32)
        args = (q, pk, pv, k, k, length, tails, partial, lse,
                *q.stride()[1:], *pk.stride()[1:], *pv.stride()[1:],
                *k.stride()[1:], *k.stride()[1:], tokens, bucket, prefix_splits,
                splits, groups, state['lanes'], state['slots'], attn.scaling,
                fused.triton.next_power_of_2(16 * groups))
        pair = [fused.compiled_launcher(_partials, q.device.index, args,
                                       num_warps=4),
                fused.compiled_launcher(_merge, q.device.index,
                                        (partial, lse, q, *q.stride()[1:], tokens, splits,
                                         fused.triton.next_power_of_2(splits)), num_warps=4)]
        if any(entry is None for entry in pair):
            return None
        candidates.append(pair)
    # w308_layer shares geometry/scaling and one compiled pair across the layers.
    if any(pair != candidates[0] for pair in candidates[1:]) or any(
           float(layer.self_attn.scaling) != float(model.layers[0].self_attn.scaling)
           for layer in model.layers):
        return None
    state.setdefault('loop_attention_entries', {})[key] = candidates[0]
    return candidates[0]


def _kernels():
    import triton
    import triton.language as tl
    from jetspec.engines.jetspec_plus_plus.kernels.refresh_attn import _kernels as refresh_kernels

    @triton.jit
    def stage(Length, TRUE, Tail, Tails, BlockRows, Counts, CT, ST, Hidden, Cos, Sin,
              t0: tl.constexpr, t1: tl.constexpr, a0: tl.constexpr, a1: tl.constexpr,
              b0: tl.constexpr, b1: tl.constexpr, b2: tl.constexpr,
              lanes: tl.constexpr, slots: tl.constexpr, tokens: tl.constexpr,
              width: tl.constexpr, dim: tl.constexpr, table_rows: tl.constexpr,
              hidden_blocks: tl.constexpr, BLOCK: tl.constexpr, OFFSET: tl.constexpr = 0):
        p = tl.program_id(0) - OFFSET
        x = tl.arange(0, BLOCK)
        if p < hidden_blocks:
            flat = p * BLOCK + x
            r, d = flat // width, flat % width
            row = tl.maximum(r - 4 - lanes, 0) // slots
            slot = tl.maximum(r - 4 - lanes, 0) % slots
            count = tl.load(Counts + row, r < tokens, other=0)
            true = tl.load(TRUE + r * t0 + d * t1, (r < 4) & (r < tokens), other=0)
            tail = tl.load(Tail + (r - 4) * a0 + d * a1,
                           (r >= 4) & (r < 4 + lanes), other=0)
            block = tl.load(BlockRows + row * b0 + slot * b1 + (b1 - width + d) * b2,
                            (r >= 4 + lanes) & (r < tokens) & (slot < count), other=0)
            tl.store(Hidden + flat, tl.where(r < 4, true, tl.where(r < 4 + lanes, tail, block)),
                     flat < tokens * width)
        else:
            flat = (p - hidden_blocks) * BLOCK + x
            r, d = flat // dim, flat % dim
            row = tl.maximum(r - 4 - lanes, 0) // slots
            slot = tl.maximum(r - 4 - lanes, 0) % slots
            tail_count = tl.load(Tails + row, r < tokens, other=0)
            length = tl.load(Length)
            pos = tl.maximum(0, tl.where(r < 4, length - 4 + r,
                                         tl.where(r < 4 + lanes, length + r - 4,
                                                  length + tail_count + slot)))
            valid = (r < tokens) & (pos < table_rows)
            c = tl.load(CT + pos * dim + d, valid, other=0)
            s = tl.load(ST + pos * dim + d, valid, other=0)
            tl.store(Cos + flat, c, r < tokens)
            tl.store(Sin + flat, s, r < tokens)

    @triton.jit
    def partials(Q, PK, PV, K, V, Length, Tails, P, L,
                 qh: tl.constexpr, qr: tl.constexpr, qd: tl.constexpr,
                 ph: tl.constexpr, pr: tl.constexpr, pd: tl.constexpr,
                 uh: tl.constexpr, ur: tl.constexpr, ud: tl.constexpr,
                 kh: tl.constexpr, kr: tl.constexpr, kd: tl.constexpr,
                 vh: tl.constexpr, vr: tl.constexpr, vd: tl.constexpr,
                 tokens: tl.constexpr, bucket: tl.constexpr, prefix_splits: tl.constexpr,
                 splits: tl.constexpr, groups: tl.constexpr, lanes: tl.constexpr,
                 slots: tl.constexpr, scale: tl.constexpr, QUERY_TILE: tl.constexpr):
        h, rb, split = tl.program_id(0), tl.program_id(1), tl.program_id(2)
        i = tl.arange(0, QUERY_TILE)
        r, head = rb * 16 + i // groups, h * groups + i % groups
        d = tl.arange(0, 128)
        s = tl.arange(0, 128)
        length = tl.load(Length)
        if split < prefix_splits:
            col = split * 128 + s
            visible = (r[:, None] < tokens) & (col[None, :] < tl.maximum(length - 4, 0))
            kp, vp = PK + h * ph + col[:, None] * pr + d[None, :] * pd, PV + h * uh + col[:, None] * ur + d[None, :] * ud
            bounds = col < bucket
        else:
            col = (split - prefix_splits) * 128 + s
            row = tl.maximum(r - 4 - lanes, 0) // slots
            tail = tl.load(Tails + row, r < tokens, other=0)
            start = 4 + lanes + row * slots
            visible = ((col[None, :] < 4) & ((r[:, None] >= 4) | (col[None, :] <= r[:, None])))
            visible = visible | ((col[None, :] >= 4) & (col[None, :] < 4 + lanes) &
                                 (((r[:, None] >= 4) & (r[:, None] < 4 + lanes) & (col[None, :] <= r[:, None])) |
                                  ((r[:, None] >= 4 + lanes) & (col[None, :] < 4 + tail[:, None]))))
            visible = visible | ((r[:, None] >= 4 + lanes) & (col[None, :] >= start[:, None]) & (col[None, :] <= r[:, None]))
            visible = visible & (r[:, None] < tokens) & (col[None, :] < tokens)
            kp, vp = K + h * kh + col[:, None] * kr + d[None, :] * kd, V + h * vh + col[:, None] * vr + d[None, :] * vd
            bounds = col < tokens
        needed = tl.sum(tl.sum(visible.to(tl.int32), 1), 0) > 0
        if needed:
            query = tl.load(Q + head[:, None] * qh + r[:, None] * qr + d[None, :] * qd,
                            r[:, None] < tokens, other=0)
            key, value = tl.load(kp, bounds[:, None], other=0), tl.load(vp, bounds[:, None], other=0)
            scores = tl.dot(query, tl.trans(key), input_precision='ieee') * scale
            scores = tl.where(visible, scores, -float('inf'))
            peak = tl.max(scores, 1)
            weights = tl.exp(scores - tl.where(peak == -float('inf'), 0., peak)[:, None])
            denom = tl.sum(weights, 1)
            result = tl.dot(weights.to(value.dtype), value, input_precision='ieee') / tl.where(denom == 0, 1., denom)[:, None]
            logsum = tl.where(denom == 0, -float('inf'), peak + tl.log(denom))
        else:
            result = tl.full((QUERY_TILE, 128), 0., tl.float32)
            logsum = tl.full((QUERY_TILE,), -float('inf'), tl.float32)
        tl.store(P + (head[:, None] * tokens + r[:, None]) * splits * 128 + split * 128 + d[None, :], result, r[:, None] < tokens)
        tl.store(L + (head * tokens + r) * splits + split, logsum, r < tokens)

    return stage, partials, refresh_kernels()[1]


_stage = _partials = _merge = None
