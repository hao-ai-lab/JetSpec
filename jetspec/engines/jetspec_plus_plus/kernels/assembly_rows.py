"""Device kernels for draft/assembly_batch.py."""
from __future__ import annotations

try:
    import triton
    import triton.language as tl
except ImportError:  # Triton is Linux/CUDA-only in the pinned environment.
    triton = None
    tl = None

libdevice = None
if triton is not None:
    try:
        from triton.language.extra import libdevice
    except ImportError:
        try:
            from triton.language.extra.cuda import libdevice
        except ImportError:
            libdevice = None


if triton is not None:

    @triton.jit
    def _stage_rows_kernel(score_ptr, tau_ptr, sizes_ptr, miss_ptr, ids_ptr,
                           requested_ptr, keys_ptr, bad_ptr, sizes_stride, ids_request_stride,
                           ids_row_stride, ROWS: tl.constexpr, POOL: tl.constexpr,
                           CAP: tl.constexpr, CAPACITY: tl.constexpr, BLOCK_J: tl.constexpr,
                           BLOCK_I: tl.constexpr):
        # one program per request; pair j = row * POOL + offer (the flattened score order)
        request = tl.program_id(0).to(tl.int64)
        total = ROWS * POOL
        j = tl.arange(0, BLOCK_J)
        inside_j = j < total
        score_j = tl.load(score_ptr + request * total + j, mask=inside_j, other=0.0)
        row_j = j // POOL
        size_j = tl.load(sizes_ptr + request * sizes_stride + row_j, mask=inside_j, other=0)
        kept_rows = tl.load(miss_ptr + request).to(tl.int32) == 0  # a normal round scores every row
        tau = tl.load(tau_ptr + request)
        scored_j = ((row_j == 0) | kept_rows) & inside_j
        passing_j = (score_j >= tau) & (score_j > 0.0) & ((j % POOL) < size_j) & scored_j
        requested = tl.sum(passing_j.to(tl.int32), axis=0)
        finite_j = (score_j - score_j) == 0.0  # NaN and +-inf give NaN
        bad = tl.sum((scored_j & (~finite_j)).to(tl.int32), axis=0) > 0
        tl.store(requested_ptr + request, requested.to(tl.int64))
        tl.store(bad_ptr + request, bad.to(tl.int64))
        kept = tl.minimum(requested, CAP)
        slot = tl.arange(0, CAPACITY)
        empty = slot >= kept  # the slots no kept pair takes stay -1
        tl.store(keys_ptr + (request * CAPACITY + slot) * 2, tl.full([CAPACITY], -1, tl.int64), mask=empty)
        tl.store(keys_ptr + (request * CAPACITY + slot) * 2 + 1, tl.full([CAPACITY], -1, tl.int64),
                 mask=empty)
        for start in range(0, total, BLOCK_I):
            i = start + tl.arange(0, BLOCK_I)
            inside_i = i < total
            score_i = tl.load(score_ptr + request * total + i, mask=inside_i, other=0.0)
            row_i = i // POOL
            size_i = tl.load(sizes_ptr + request * sizes_stride + row_i, mask=inside_i, other=0)
            passing_i = ((score_i >= tau) & (score_i > 0.0) & ((i % POOL) < size_i)
                         & ((row_i == 0) | kept_rows) & inside_i)
            before = ((score_j[None, :] > score_i[:, None])
                      | ((score_j[None, :] == score_i[:, None]) & (j[None, :] < i[:, None])))
            rank = tl.sum((before & passing_j[None, :]).to(tl.int32), axis=1)
            keep = passing_i & (rank < CAP)
            anchor = tl.load(ids_ptr + request * ids_request_stride + row_i * ids_row_stride + i % POOL,
                             mask=keep, other=0)
            target = keys_ptr + (request * CAPACITY + rank.to(tl.int64)) * 2
            tl.store(target, row_i.to(tl.int64), mask=keep)
            tl.store(target + 1, anchor, mask=keep)

    @triton.jit
    def _row_inputs_kernel(requested_ptr, keys_ptr, guesses_ptr, source_ptr, samples_ptr,
                           keys_out_ptr, counts_ptr, geometry_ptr, active_ptr, preceding_ptr,
                           first_ptr, count, source_stride, samples_stride,
                           H: tl.constexpr, CAPACITY: tl.constexpr, CAP: tl.constexpr,
                           B: tl.constexpr, BLOCK_H: tl.constexpr, BLOCK_B: tl.constexpr):
        # one program per bucket request (the padded ones are all zero keys and no rows)
        request = tl.program_id(0).to(tl.int64)
        real = request < count
        row = tl.arange(0, BLOCK_H)
        inside = row < H
        requested = tl.load(requested_ptr + request, mask=real, other=0)
        guess = tl.load(guesses_ptr + request, mask=real, other=0)
        k = tl.load(keys_ptr + (request * CAPACITY + row) * 2, mask=inside & real, other=0)
        anchor = tl.load(keys_ptr + (request * CAPACITY + row) * 2 + 1, mask=inside & real, other=0)
        private = real & (requested == 0) & (row == 0)
        k = tl.where(private, 0, k)
        anchor = tl.where(private, guess, anchor)
        k = tl.maximum(k, 0)
        anchor = tl.maximum(anchor, 0)
        live = tl.where(real, tl.maximum(tl.minimum(requested, CAP), 1), 0)
        tl.store(counts_ptr + request, live)
        flat = request * H + row
        tl.store(keys_out_ptr + flat * 2, k, mask=inside)
        tl.store(keys_out_ptr + flat * 2 + 1, anchor, mask=inside)
        tl.store(geometry_ptr + flat, k + 1, mask=inside)
        tl.store(active_ptr + flat, row < live, mask=inside)
        preceding = tl.load(source_ptr + request * source_stride + k, mask=inside, other=0)
        tl.store(preceding_ptr + flat, preceding, mask=inside)
        slot = tl.arange(0, BLOCK_B)
        sample = tl.load(samples_ptr + request * samples_stride + slot, mask=slot < B, other=0)
        ids = tl.where(slot[None, :] == 0, anchor[:, None], sample[None, :])
        tl.store(first_ptr + flat[:, None] * B + slot[None, :], ids,
                 mask=inside[:, None] & (slot[None, :] < B))


if triton is not None and libdevice is not None:

    @triton.jit
    def _clamp_nan(value, low, high):
        # aten clamp: a NaN stays NaN, everything else min(max(value, low), high)
        return tl.where(value != value, value, tl.minimum(tl.maximum(value, low), high))

    @triton.jit
    def _blend_x_kernel(p_top1_ptr, q_ptr, rejected_ptr, argmax_ptr, sizes_ptr, p64_ptr,
                        position_ptr, const_ptr, x_ptr, checks_ptr, sizes_stride,
                        p64_request_stride, p64_row_stride, N: tl.constexpr, ROWS: tl.constexpr,
                        WIDTH: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_W: tl.constexpr,
                        FEATURES: tl.constexpr):
        # e04_kscore_blend_batch_device up to x = sigmoid(z), one program per request
        request = tl.program_id(0).to(tl.int64)
        k = tl.arange(0, BLOCK_N)
        inside = k < N
        eps = tl.load(const_ptr + 6)
        upper = tl.load(const_ptr + 7)
        drafter = _clamp_nan(tl.load(p_top1_ptr + request * ROWS + 1 + k, mask=inside, other=0.5),
                             eps, upper)
        decoder = _clamp_nan(tl.load(q_ptr + request * ROWS + k, mask=inside, other=0.5), eps, upper)
        z = (tl.load(const_ptr + 0) * (libdevice.log(drafter) - libdevice.log1p(-drafter))
             + tl.load(const_ptr + 1) * (libdevice.log(decoder) - libdevice.log1p(-decoder))
             ) + tl.load(const_ptr + 2)
        rejected = tl.load(rejected_ptr + request * ROWS + k, mask=inside, other=0)
        missing_draft = tl.sum((inside & (rejected < 0)).to(tl.int32), axis=0) > 0
        missing_offer = tl.sum(tl.zeros([BLOCK_N], tl.int32), axis=0) > 0
        if FEATURES:
            sizes = tl.load(sizes_ptr + request * sizes_stride + k, mask=inside, other=1)
            missing_offer = tl.sum((inside & (sizes < 1)).to(tl.int32), axis=0) > 0
            argmax = tl.load(argmax_ptr + request * ROWS + k, mask=inside, other=0)
            argmax_eq = (argmax == rejected).to(tl.float64)
            w = tl.arange(0, BLOCK_W)
            valid = (w[None, :] < sizes[:, None]) & (w[None, :] < WIDTH) & inside[:, None]
            offers = tl.load(p64_ptr + request * p64_request_stride + k[:, None] * p64_row_stride
                             + w[None, :], mask=valid, other=0.0)
            top = tl.max(tl.where(valid, offers, float('-inf')), axis=1)
            has_nan = tl.sum((valid & (offers != offers)).to(tl.int32), axis=1) > 0
            top = _clamp_nan(tl.where(has_nan, tl.load(const_ptr + 8), top), eps, upper)
            position = tl.load(position_ptr + k, mask=inside, other=0.0)
            z = ((z + tl.load(const_ptr + 3) * argmax_eq)
                 + tl.load(const_ptr + 4) * (libdevice.log(top) - libdevice.log1p(-top))
                 ) + tl.load(const_ptr + 5) * position
        x = 1.0 / (1.0 + libdevice.exp(-z))
        tl.store(x_ptr + request * N + k, x, mask=inside)
        tl.store(checks_ptr + request * 2, missing_draft.to(tl.int64))
        tl.store(checks_ptr + request * 2 + 1, missing_offer.to(tl.int64))

    @triton.jit
    def _blend_mass_kernel(x_ptr, cumulative_ptr, miss_ptr, saved_ptr, p64_ptr, const_ptr,
                           mass16_ptr, scoring_ptr, scaled_ptr, checks_ptr, p64_request_stride,
                           p64_row_stride, N: tl.constexpr, ROWS: tl.constexpr, POOL: tl.constexpr,
                           BLOCK_R: tl.constexpr, BLOCK_P: tl.constexpr):
        # the mass on the 2^-32 grid, the after-miss overrides and the pool's tempered offers
        request = tl.program_id(0).to(tl.int64)
        k = tl.arange(0, BLOCK_R)
        rows = k < ROWS
        x = tl.load(x_ptr + request * N + k, mask=k < N, other=0.0)
        survival = tl.load(cumulative_ptr + request * N + k - 1, mask=(k >= 1) & (k <= N), other=1.0)
        mass = tl.where(k < N, survival * (1.0 - x), survival)
        mass = libdevice.nearbyint(mass * tl.load(const_ptr + 10)) * tl.load(const_ptr + 11)
        miss = tl.load(miss_ptr + request).to(tl.int32) != 0
        saved = tl.load(saved_ptr + request * (ROWS + 1) + 1 + k, mask=rows, other=0.0)
        tl.store(mass16_ptr + request * (ROWS + 1) + 1 + k, tl.where(miss, saved, mass), mask=rows)
        tl.store(mass16_ptr + request * (ROWS + 1),
                 tl.where(miss, tl.load(saved_ptr + request * (ROWS + 1)), 0.0))
        tl.store(scoring_ptr + request * ROWS + k, tl.where(miss, tl.where(k == 0, 1.0, 0.0), mass),
                 mask=rows)
        pair = tl.arange(0, 2)
        checks = tl.load(checks_ptr + request * 2 + pair)
        tl.store(checks_ptr + request * 2 + pair, tl.where(miss, 0, checks))
        w = tl.arange(0, BLOCK_P)
        inside = rows[:, None] & (w[None, :] < POOL)
        offers = tl.load(p64_ptr + request * p64_request_stride + k[:, None] * p64_row_stride
                         + w[None, :], mask=inside, other=1.0)
        tiny = tl.load(const_ptr + 9)
        offers = tl.where(offers != offers, offers, tl.maximum(offers, tiny))  # clamp_min
        scaled = libdevice.pow(offers, tl.broadcast_to(tl.load(const_ptr + 12), [BLOCK_R, BLOCK_P]))
        tl.store(scaled_ptr + (request * ROWS + k[:, None]) * POOL + w[None, :], scaled, mask=inside)

    @triton.jit
    def _joint_score_kernel(scaled_ptr, total_ptr, scoring_ptr, const_ptr, score_ptr,
                            ROWS: tl.constexpr, POOL: tl.constexpr, BLOCK_R: tl.constexpr,
                            BLOCK_P: tl.constexpr):
        # score = round(P(k) * P_T(a | k) * 2^32) / 2^32, P_T the tempered offer over its pool sum
        request = tl.program_id(0).to(tl.int64)
        k = tl.arange(0, BLOCK_R)
        w = tl.arange(0, BLOCK_P)
        rows = k < ROWS
        inside = rows[:, None] & (w[None, :] < POOL)
        scaled = tl.load(scaled_ptr + (request * ROWS + k[:, None]) * POOL + w[None, :], mask=inside,
                         other=1.0)
        total = tl.load(total_ptr + request * ROWS + k, mask=rows, other=1.0)
        tiny = tl.load(const_ptr + 9)
        total = tl.where(total != total, total, tl.maximum(total, tiny))
        anchor = scaled / total[:, None]
        mass = tl.load(scoring_ptr + request * ROWS + k, mask=rows, other=0.0)
        score = (libdevice.nearbyint((mass[:, None] * anchor) * tl.load(const_ptr + 10))
                 * tl.load(const_ptr + 11))
        tl.store(score_ptr + (request * ROWS + k[:, None]) * POOL + w[None, :], score, mask=inside)

