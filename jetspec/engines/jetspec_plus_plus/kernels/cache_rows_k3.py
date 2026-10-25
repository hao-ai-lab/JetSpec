"""B16 joint row rule for the static chain (two launches by default).

The first launch reduces decoder logits to the per-offset BLEND hazard.  The
second launch forms the quantized mass, scores all 16 x 32 offers, ranks by
(-score, k, pool rank), and writes the complete U2 output contract.
"""
from __future__ import annotations

from jetspec.engines.jetspec_plus_plus.profiles import kernel_enabled, kernel_value

import torch

from jetspec.engines.jetspec_plus_plus.wire.triton_offsets import (  # noqa: F401  (read inside the kernels)
    AFTER_MISS, C1_COUNT, C1_START, CORRECTION, DEC_LEN, J_ROUND, RNG_IDX, RP_L, RP_TRUE_K, SLOT,
    TAU_IDX)

try:
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice
except ImportError:  # the Mac CPU reference does not need Triton
    triton = None


if triton is not None:
    @triton.jit
    def _hazard(Logits, Draft, Ptop, P64, Pack, Hazard,
                V: tl.constexpr, STRIDE: tl.constexpr, LANES: tl.constexpr,
                BLEND_DEC: tl.constexpr,
                A: tl.constexpr, B: tl.constexpr, C: tl.constexpr,
                G: tl.constexpr, T: tl.constexpr, P: tl.constexpr,
                BLOCK_V: tl.constexpr):
        k = tl.program_id(0)
        col = tl.arange(0, BLOCK_V)
        maximum = -float('inf')
        argmax = V
        for part in range((V + BLOCK_V - 1) // BLOCK_V):
            offset = part * BLOCK_V + col
            value = tl.load(Logits + k * V + offset, offset < V,
                            other=-float('inf')).to(tl.float32)
            local_max = tl.max(value, 0)
            local_argmax = tl.min(tl.where(value == local_max, offset, V), 0)
            argmax = tl.where(local_max > maximum, local_argmax,
                              tl.where(local_max == maximum,
                                       tl.minimum(argmax, local_argmax), argmax))
            maximum = tl.maximum(maximum, local_max)
        # The decoder reference calls float32 softmax, then promotes the selected
        # probability to float64 for the BLEND logit.
        denominator = 0.
        for part in range((V + BLOCK_V - 1) // BLOCK_V):
            offset = part * BLOCK_V + col
            value = tl.load(Logits + k * V + offset, offset < V,
                            other=-float('inf')).to(tl.float32)
            denominator += tl.sum(tl.exp(value - maximum), 0)
        rejected = tl.load(Draft + k)
        selected = tl.load(Logits + k * V + tl.maximum(rejected, 0)).to(tl.float32)
        q = tl.exp(selected - maximum) / denominator
        drafter = tl.load(Ptop + k + 1).to(tl.float64)
        drafter = tl.minimum(tl.maximum(drafter, 1.e-6), 1. - 1.e-6)
        decoder = tl.minimum(tl.maximum(q.to(tl.float64), 1.e-6), 1. - 1.e-6)
        z = A * (libdevice.log(drafter) - libdevice.log1p(-drafter)) + B * (
            libdevice.log(decoder) - libdevice.log1p(-decoder)) + C
        if BLEND_DEC:
            size = tl.load(Pack + k)
            rank = tl.arange(0, 64)
            offer = tl.load(P64 + k * STRIDE + rank, rank < STRIDE, other=-float('inf'))
            top = tl.max(tl.where(rank < size, offer, -float('inf')), 0)
            top = tl.minimum(tl.maximum(top, 1.e-6), 1. - 1.e-6)
            z = z + G * (argmax == rejected).to(tl.float64) + T * (
                libdevice.log(top) - libdevice.log1p(-top)) + P * (
                    (k + 1).to(tl.float64) / 14.)
        hazard = 1. / (1. + libdevice.exp(-z))
        tl.store(Hazard + k, hazard.to(tl.int64, bitcast=True))


    @triton.jit
    def _hazard_max_tiles(Logits, TileMax, TileId,
                          V: tl.constexpr, PARTS: tl.constexpr, BLOCK_V: tl.constexpr):
        k = tl.program_id(0)
        part = tl.program_id(1)
        col = part * BLOCK_V + tl.arange(0, BLOCK_V)
        value = tl.load(Logits + k * V + col, col < V,
                        other=-float('inf')).to(tl.float32)
        maximum = tl.max(value, 0)
        winner = tl.min(tl.where((col < V) & (value == maximum), col, V), 0)
        tl.store(TileMax + k * PARTS + part, maximum)
        tl.store(TileId + k * PARTS + part, winner)


    @triton.jit
    def _hazard_sum_tiles(Logits, TileMax, TileSum,
                          V: tl.constexpr, PARTS: tl.constexpr,
                          BLOCK_PARTS: tl.constexpr, BLOCK_V: tl.constexpr):
        k = tl.program_id(0)
        part = tl.program_id(1)
        pieces = tl.arange(0, BLOCK_PARTS)
        maxima = tl.load(TileMax + k * PARTS + pieces, pieces < PARTS,
                         other=-float('inf'))
        maximum = tl.max(maxima, 0)
        col = part * BLOCK_V + tl.arange(0, BLOCK_V)
        value = tl.load(Logits + k * V + col, col < V,
                        other=-float('inf')).to(tl.float32)
        # This is the same 4096-wide reduction as the serial kernel.  The final
        # launch adds these partials in the original part order.
        subtotal = tl.sum(tl.exp(value - maximum), 0)
        tl.store(TileSum + k * PARTS + part, subtotal)


    @triton.jit
    def _hazard_finish_tiles(Logits, Draft, Ptop, P64, Pack, Hazard,
                             TileMax, TileId, TileSum,
                             V: tl.constexpr, STRIDE: tl.constexpr, LANES: tl.constexpr,
                             BLEND_DEC: tl.constexpr,
                             A: tl.constexpr, B: tl.constexpr, C: tl.constexpr,
                             G: tl.constexpr, T: tl.constexpr, P: tl.constexpr,
                             PARTS: tl.constexpr, BLOCK_PARTS: tl.constexpr):
        k = tl.program_id(0)
        part = tl.arange(0, BLOCK_PARTS)
        maxima = tl.load(TileMax + k * PARTS + part, part < PARTS,
                         other=-float('inf'))
        winners = tl.load(TileId + k * PARTS + part, part < PARTS, other=V)
        maximum = tl.max(maxima, 0)
        argmax = tl.min(tl.where(maxima == maximum, winners, V), 0)
        denominator = 0.
        for index in range(PARTS):
            denominator += tl.load(TileSum + k * PARTS + index)
        rejected = tl.load(Draft + k)
        selected = tl.load(Logits + k * V + tl.maximum(rejected, 0)).to(tl.float32)
        q = tl.exp(selected - maximum) / denominator
        drafter = tl.load(Ptop + k + 1).to(tl.float64)
        drafter = tl.minimum(tl.maximum(drafter, 1.e-6), 1. - 1.e-6)
        decoder = tl.minimum(tl.maximum(q.to(tl.float64), 1.e-6), 1. - 1.e-6)
        z = A * (libdevice.log(drafter) - libdevice.log1p(-drafter)) + B * (
            libdevice.log(decoder) - libdevice.log1p(-decoder)) + C
        if BLEND_DEC:
            size = tl.load(Pack + k)
            rank = tl.arange(0, 64)
            offer = tl.load(P64 + k * STRIDE + rank, rank < STRIDE, other=-float('inf'))
            top = tl.max(tl.where(rank < size, offer, -float('inf')), 0)
            top = tl.minimum(tl.maximum(top, 1.e-6), 1. - 1.e-6)
            z = z + G * (argmax == rejected).to(tl.float64) + T * (
                libdevice.log(top) - libdevice.log1p(-top)) + P * (
                    (k + 1).to(tl.float64) / 14.)
        hazard = 1. / (1. + libdevice.exp(-z))
        tl.store(Hazard + k, hazard.to(tl.int64, bitcast=True))


    @triton.jit
    def _rows(Hazard, Draft, IDs, P64, Pack, RP, Tau, Previous,
              Keys, Active, Requested, Home, Err, Scores,
              R: tl.constexpr, LANES: tl.constexpr, STRIDE: tl.constexpr,
              TEMPERATURE: tl.constexpr, HAS_PREVIOUS: tl.constexpr,
              STORE_SCORES: tl.constexpr, FLOOR: tl.constexpr,
              FILL_ZERO: tl.constexpr = False, INTAKE_ERR=None,
              FOLD_INTAKE: tl.constexpr = False):
        pair_count: tl.constexpr = LANES * 32
        order_bits: tl.constexpr = 9 if LANES == 16 else 10
        k = tl.arange(0, LANES)
        rank = tl.arange(0, 32)
        flat = tl.arange(0, pair_count)
        x = tl.load(Hazard + k, k < LANES - 1, other=0).to(tl.float64, bitcast=True)
        x = tl.where(k < LANES - 1, x, 1.)
        # All prefix products are float64, as in the torch reference.
        cumulative = tl.cumprod(x, 0)
        prefix = tl.where(k == 0, 1., tl.gather(cumulative, tl.maximum(k - 1, 0), 0))
        mass = tl.where(k == LANES - 1, prefix, prefix * (1. - x))
        mass = libdevice.nearbyint(mass * 4294967296.) / 4294967296.
        miss = tl.load(RP + AFTER_MISS) != 0
        scoring_mass = tl.where(miss, (k == 0).to(tl.float64), mass)
        probability = tl.load(P64 + k[:, None] * STRIDE + rank[None, :])
        # Triton has no mixed fp64/fp32 pow: make the constexpr exponent an fp64 scalar.
        scaled = libdevice.pow(tl.maximum(probability, 1.e-300),
                               tl.full((), 1. / TEMPERATURE, tl.float64))
        anchor = scaled / tl.maximum(tl.sum(scaled, 1)[:, None], 1.e-300)
        score = libdevice.nearbyint(scoring_mass[:, None] * anchor * 4294967296.) / 4294967296.
        score_flat = tl.reshape(score, (pair_count,))
        if STORE_SCORES:
            tl.store(Scores + flat, score_flat)
        size = tl.load(Pack + k)
        tau_idx = tl.load(RP + TAU_IDX)
        tau = tl.load(Tau + tau_idx)
        passing = (score >= tau) & (score > 0.) & (rank[None, :] < size[:, None]) & (
            ~miss | (k[:, None] == 0))
        passed = tl.reshape(passing, (pair_count,))
        requested = tl.sum(passed.to(tl.int32), 0)
        # Scores are on the 2^-32 grid; the lower order bits give the stable
        # row-major tie order used by torch.argsort(stable=True).
        score_word = (score_flat * 4294967296.).to(tl.int64)
        composite = (score_word << order_bits) | (pair_count - 1 - flat).to(tl.int64)
        ordered = tl.sort(composite, descending=True)
        index = pair_count - 1 - (ordered & (pair_count - 1)).to(tl.int32)
        chosen = tl.gather(passed, index, 0)
        if FLOOR > 0:
            valid_positive = ((score > 0.) | FILL_ZERO) & (rank[None, :] < size[:, None]) & (
                ~miss | (k[:, None] == 0))
            positive = tl.gather(tl.reshape(valid_positive, (pair_count,)), index, 0)
            chosen = chosen | (positive & (tl.cumsum(positive.to(tl.int32), 0) <= FLOOR))
            requested = tl.sum(chosen.to(tl.int32), 0)
        position = tl.cumsum(chosen.to(tl.int32), 0) - 1
        slot = tl.arange(0, R)
        selected = tl.min(tl.where(chosen[None, :] & (position[None, :] == slot[:, None]),
                                   index[None, :], pair_count), 1)
        found = selected < pair_count
        selected_k = selected // 32
        token = tl.load(IDs + selected_k * STRIDE + selected % 32, found, other=-1)
        key_k = tl.where(found, selected_k, -1)
        key_k = tl.where((slot == 0) & (requested == 0), 0, key_k)
        token = tl.where((slot == 0) & (requested == 0), tl.load(RP + CORRECTION), token)
        tl.store(Keys + slot * 2, key_k.to(tl.int64))
        tl.store(Keys + slot * 2 + 1, token)
        tl.store(Active + slot, slot < tl.minimum(tl.maximum(requested, 1), R))
        tl.store(Requested, requested)
        tl.store(Home, requested)
        tl.store(Home + 1 + slot * 2, key_k.to(tl.int64))
        tl.store(Home + 2 + slot * 2, token)
        pack_index = tl.arange(0, LANES * 8)
        pack_value = tl.load(Pack + pack_index, pack_index < 5 * LANES, other=0)
        tl.store(Home + 1 + 2 * R + pack_index, pack_value, pack_index < 5 * LANES)
        home_mass = tl.arange(0, LANES * 2)
        if HAS_PREVIOUS:
            prior = tl.load(Previous + home_mass, home_mass < LANES + 1, other=0.)
            value = tl.where(miss, prior, tl.where(home_mass == 0, 0., tl.gather(
                mass, tl.minimum(tl.maximum(home_mass - 1, 0), LANES - 1), 0)))
        else:
            value = tl.where(home_mass == 0, 0., tl.gather(
                mass, tl.minimum(tl.maximum(home_mass - 1, 0), LANES - 1), 0))
        tl.store(Home + 1 + 2 * R + 5 * LANES + home_mass,
                 value.to(tl.float64).to(tl.int64, bitcast=True), home_mass < LANES + 1)
        draft_missing = tl.sum((tl.load(Draft + k, k < LANES - 1, other=0) < 0).to(tl.int32), 0) > 0
        offer_missing = tl.sum((size < 1).to(tl.int32) * (k < LANES - 1).to(tl.int32), 0) > 0
        flag0 = tl.sum((tl.load(Pack + 3 * LANES + k) == 0).to(tl.int32), 0) > 0
        flag1 = tl.sum((tl.load(Pack + 4 * LANES + k) != 0).to(tl.int32), 0) > 0
        flag4 = tl.sum(((tl.abs(mass) == float('inf')) | (mass != mass)).to(tl.int32), 0) > 0
        scored = ~miss | (k == 0)
        flag5 = tl.sum(((size < 32) & scored).to(tl.int32), 0) > 0
        flag6 = tl.sum(tl.sum((((tl.abs(score) == float('inf')) | (score != score)) &
                               scored[:, None]).to(tl.int32), 1), 0) > 0
        error = (flag0.to(tl.int64) | (flag1.to(tl.int64) << 1) |
                 ((draft_missing & ~miss).to(tl.int64) << 2) |
                 ((offer_missing & ~miss).to(tl.int64) << 3) |
                 ((flag4 & ~miss).to(tl.int64) << 4) |
                 (flag5.to(tl.int64) << 5) | (flag6.to(tl.int64) << 6))
        if FOLD_INTAKE:
            tl.store(Err, error | tl.load(INTAKE_ERR))
        else:
            tl.store(Err, error)
        tl.store(Home + 1 + 2 * R + 5 * LANES + LANES + 1, error)


def rows_k3_body(rp, tokens, offers_ids, offers_p64, offers_pack, dec_logits,
                 p_top1_prev, R, *, thresholds, dataset, tau_table, temperature,
                 out, previous_mass16=None, k_score='blend_dec',
                 decoder_frame='wire', floor=0, fill_zero=False):
    """Adapt the chain packet to the registered cache-row override."""
    if triton is None or dec_logits.device.type != 'cuda':
        raise RuntimeError('JETSPEC_ROWS_K3 requires CUDA and Triton')
    from jetspec.engines.jetspec_plus_plus.draft.method.calibration import dataset_entry
    from jetspec.engines.jetspec_plus_plus.wire.contract import Offset
    import os
    coeff = dataset_entry(thresholds['e04_kscore_blend'], dataset)
    names = ('a', 'b', 'c') + (('argmax', 'top', 'pos') if k_score == 'blend_dec' else ())
    return _cache_rows_k3(
        tokens.pending_draft_ids, offers_ids, offers_p64, offers_pack, dec_logits,
        p_top1_prev, R, coefficients=tuple(float(coeff[name]['value']) for name in names),
        miss=rp[Offset.AFTER_MISS], tau_idx=rp[Offset.TAU_IDX], correction=rp[Offset.CORRECTION],
        tau_table=tau_table, temperature=temperature, out=out,
        previous_mass16=previous_mass16, floor=floor, fill_zero=fill_zero,
        control=rp, decoder_frame=decoder_frame)


def _cache_rows_k3(draft_ids, offers_ids, offers_p64, offers_pack, dec_logits,
                   p_top1_prev, R, *, coefficients, miss, tau_idx, correction,
                   tau_table, temperature, out, previous_mass16=None,
                   floor=0, fill_zero=False, control=None, decoder_frame='wire'):
    """K3 override of cache_rows_chain; unsupported inputs use the method oracle."""
    from jetspec.engines.jetspec_plus_plus.profiles import kernel_enabled
    from jetspec.engines.jetspec_plus_plus.draft.method.ops import reference
    import os
    if (triton is None or not dec_logits.is_cuda or control is None
            or not kernel_enabled('rows_k3', '1')):
        return reference('cache_rows_chain')(
            draft_ids, offers_ids, offers_p64, offers_pack, dec_logits, p_top1_prev, R,
            coefficients=coefficients, miss=miss, tau_idx=tau_idx, correction=correction,
            tau_table=tau_table, temperature=temperature, out=out,
            previous_mass16=previous_mass16, floor=floor, fill_zero=fill_zero, control=control)
    lanes = offers_ids.shape[0]
    if (draft_ids.numel() != lanes - 1 or lanes not in ((16, 32) if fill_zero else (16,))
            or R not in (4, 8, 16)):
        raise ValueError('K3 rows require b16 (b32 in eval) and R=4,8,16')
    if decoder_frame != 'wire':
        raise ValueError('K3 rows require the chain wire decoder frame')
    blend_dec = len(coefficients) == 6
    constants = list(coefficients) if blend_dec else list(coefficients) + [0., 0., 0.]
    # The first 15 words of row_home are scratch until the second launch writes
    # its final packed payload. No extra chain buffer or capture allocation.
    hazard = out['row_home']
    scores = out.get('k3_scores')
    if scores is not None and (scores.shape != (lanes, 32) or scores.dtype != torch.float64):
        raise ValueError('K3 score probe must be float64 [lanes,32]')
    stride = offers_p64.shape[1]
    stats = out.get('select_stats') if (kernel_value('draft_fuse_t4', '0') == '2'
                                      or kernel_enabled('draft_fuse_t5', '1')) else None
    if stats is not None:
        # Unit T4 level 2: the select kernel's unfiltered max, argmax and sum-exp replace
        # the two hazard scans; the finish reads them as one part per row.
        _hazard_finish_tiles[(lanes - 1,)](
            dec_logits, draft_ids, p_top1_prev, offers_p64, offers_pack,
            hazard, stats[0], out['select_argmax'], stats[1], dec_logits.shape[1], stride,
            lanes, blend_dec, *constants, 1, 2,
            num_warps=8, enable_fp_fusion=False)
    elif kernel_enabled('k3_hazard_split', '1'):
        vocab = dec_logits.shape[1]
        block_v = 4096
        parts = triton.cdiv(vocab, block_v)
        tile_max = torch.empty((lanes - 1, parts), device=dec_logits.device, dtype=torch.float32)
        tile_id = torch.empty((lanes - 1, parts), device=dec_logits.device, dtype=torch.int32)
        tile_sum = torch.empty_like(tile_max)
        _hazard_max_tiles[(lanes - 1, parts)](
            dec_logits, tile_max, tile_id, vocab, parts, block_v,
            num_warps=8, enable_fp_fusion=False)
        _hazard_sum_tiles[(lanes - 1, parts)](
            dec_logits, tile_max, tile_sum, vocab, parts,
            triton.next_power_of_2(parts), block_v,
            num_warps=8, enable_fp_fusion=False)
        _hazard_finish_tiles[(lanes - 1,)](
            dec_logits, draft_ids, p_top1_prev, offers_p64, offers_pack,
            hazard, tile_max, tile_id, tile_sum, vocab, stride, lanes, blend_dec,
            *constants, parts, triton.next_power_of_2(parts),
            num_warps=8, enable_fp_fusion=False)
    else:
        _hazard[(lanes - 1,)](
            dec_logits, draft_ids, p_top1_prev, offers_p64, offers_pack,
            hazard, dec_logits.shape[1], stride, lanes,
            blend_dec,
            *constants, 4096, num_warps=8, enable_fp_fusion=False)
    _rows[(1,)](
        hazard, draft_ids, offers_ids, offers_p64, offers_pack,
        control, tau_table, previous_mass16 if previous_mass16 is not None else hazard,
        out['keys'], out['active'], out['requested'], out['row_home'], out['err'],
        scores if scores is not None else hazard,
        R, lanes, stride, temperature, previous_mass16 is not None,
        scores is not None, floor, fill_zero,
        **(dict(INTAKE_ERR=out['_t14_intake_err'], FOLD_INTAKE=True)
           if '_t14_intake_err' in out else {}),
        num_warps=8, enable_fp_fusion=False)
    return out['keys'], out['active'], out['requested'], out['row_home'], out['err']


from jetspec.engines.jetspec_plus_plus.draft.method import cache_rows as _cache_rows
from jetspec.engines.jetspec_plus_plus.kernels import register_optimization as _register_optimization

_register_optimization('cache_rows_chain', _cache_rows_k3)
