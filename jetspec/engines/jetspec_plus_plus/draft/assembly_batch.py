"""JETSPEC_BATCH_ROWS_ASSEMBLY_GRAPH: the batched draft round's row assembly as one captured call.

Under JETSPEC_BATCH_DRAFT_GRAPH=1 a proposal round of C requests assembles its rows between the
anchor decoder and the Jacobi passes: the BLEND k score (float64), the joint row rule and its
ranking of the (k, anchor) pairs, the one copy home, and the row inputs of the row-cap bucket.
Run eagerly (KthRowsDrafter._assemble_rows_batch and the row inputs of
_flat_rows_and_passes_batch) that is ~170 small launches, and the device waits on the host
between them.

Here the decoder's compact q and top-1 fields, or the legacy logits' two vocabulary reductions,
are copied into static buffers (one copy
per input, one pinned upload for the host values), and the rest is one CUDA graph replay per
request count C:

* the BLEND arithmetic and the joint score: the eager path's torch operations, in the same order,
  on the same float64 values, so the same bits (a graph replays the kernels eager mode launches);
* the joint rule's selection (``stage_rows``): per request, a passing pair's slot is its rank among
  the passing pairs by (score descending, flat index ascending), the order BatchedRowStage's stable
  descending argsort gives them; comparisons and counts only, so exact;
* the row inputs (``row_inputs``): the keys with an empty round's private row, the drafted row
  counts, the geometry, the anchors and the first pass's ids; integer work, so exact.

On CUDA the last two are one Triton kernel each; elsewhere they are the eager path's own
operations (the references the GPU check compares the kernels with).
"""
from __future__ import annotations

from jetspec.engines.jetspec_plus_plus.profiles import kernel_enabled, kernel_value

import os

import torch
from jetspec.engines.jetspec_plus_plus.draft.method.cache_rows import blend_rows_batch

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

GRID = 2.0 ** 32               # live_rows.E04_KSCORE_GRID
EPS = 1e-6                     # live_rows.E04_KSCORE_EPS
POSITION_SCALE = 14.0          # live_rows.E04_KSCORE_POSITION_SCALE
PACK_ROWS = 5                  # the anchor offers' packed summary rows




def rows_assembly_fused():
    """JETSPEC_BATCH_ROWS_ASSEMBLY_FUSED (default 1; with JETSPEC_BATCH_ROWS_ASSEMBLY_GRAPH on CUDA):
    the BLEND float64 chain as three Triton kernels (``_blend_x_kernel``, ``_blend_mass_kernel``,
    ``_joint_score_kernel``) around the eager path's own cumprod and pool sum: every elementwise
    step is the eager op's own IEEE operation or libdevice function, one rounding each (no fused
    multiply-add), so the values are the eager ones; the GPU check compares them bitwise."""
    from jetspec.engines.jetspec_plus_plus.profiles import kernel_value
    value = kernel_value('batch_rows_assembly_fused', '1')
    if value not in ('0', '1'):
        raise ValueError('JETSPEC_BATCH_ROWS_ASSEMBLY_FUSED must be 0 or 1')
    return value == '1'


# the fused chain's float64 constants, loaded by the kernels (a Python float argument is fp32):
# 0-5 a, b, c, argmax, top, pos; 6 EPS; 7 1 - EPS; 8 NaN; 9 1e-300; 10 2^32; 11 2^-32; 12 1 / T
# pow's exponents with their own kernels in aten (sqrt, rsqrt, reciprocal, square, cube, 1/x^2, copy, fill)
_POW_SPECIAL = (0.5, -0.5, -1.0, 2.0, 3.0, -2.0, 1.0, 0.0)


def _batched(predictions):
    """The requests' decoder logits as one [C, rows, vocab] tensor: a view of the batch
    decoder's own output when every request's logits are its consecutive dense rows (no copy;
    read from the storage, since inference-mode views keep no base), else a stack."""
    first = predictions[0].logits
    storage = first.untyped_storage().data_ptr()
    if first.is_contiguous() and all(
            prediction.logits.is_contiguous() and prediction.logits.shape == first.shape
            and prediction.logits.dtype == first.dtype and prediction.logits.device == first.device
            and prediction.logits.untyped_storage().data_ptr() == storage
            and prediction.logits.storage_offset() == first.storage_offset() + index * first.numel()
            for index, prediction in enumerate(predictions)):
        return first.as_strided((len(predictions), *first.shape), (first.numel(), *first.stride()),
                                first.storage_offset())
    return torch.stack([prediction.logits for prediction in predictions])


from jetspec.engines.jetspec_plus_plus.draft.method.rows import stage_rows_reference


from jetspec.engines.jetspec_plus_plus.draft.method.rows import row_inputs_reference


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


def stage_rows(score, tau, sizes, miss, ids, *, cap, capacity):
    """``stage_rows_reference`` in one kernel on CUDA (exact: comparisons and counts)."""
    if triton is None or score.device.type != 'cuda':
        return stage_rows_reference(score, tau, sizes, miss, ids, cap=cap, capacity=capacity)
    count, rows, pool = score.shape
    if (not score.is_contiguous() or ids.stride(-1) != 1 or sizes.stride(-1) != 1
            or capacity & (capacity - 1)):
        raise ValueError('the stage kernel needs dense scores, offer ids and sizes, and a power-of-two capacity')
    requested = torch.empty(count, dtype=torch.long, device=score.device)
    keys = torch.empty(count, capacity, 2, dtype=torch.long, device=score.device)
    bad = torch.empty(count, dtype=torch.long, device=score.device)
    _stage_rows_kernel[(count,)](
        score, tau, sizes, miss, ids, requested, keys, bad, sizes.stride(0), ids.stride(0),
        ids.stride(1), ROWS=rows, POOL=pool, CAP=int(cap), CAPACITY=int(capacity),
        BLOCK_J=triton.next_power_of_2(rows * pool), BLOCK_I=32, num_warps=8)
    return requested, keys, bad


def row_inputs(requested, keys, guesses, source, samples, *, count, padded, H, cap):
    """``row_inputs_reference`` in one kernel on CUDA (exact: integer work)."""
    if triton is None or keys.device.type != 'cuda':
        return row_inputs_reference(requested, keys, guesses, source, samples, count=count,
                                    padded=padded, H=H, cap=cap)
    B = samples.shape[1]
    if not keys.is_contiguous() or source.stride(1) != 1 or samples.stride(1) != 1:
        raise ValueError('the row-input kernel needs dense keys, source and samples')
    device = keys.device
    keys_device = torch.empty(padded, H, 2, dtype=torch.long, device=device)
    live_counts = torch.empty(padded, dtype=torch.long, device=device)
    geometry = torch.empty(padded * H, dtype=torch.long, device=device)
    active_rows = torch.empty(padded * H, dtype=torch.bool, device=device)
    preceding = torch.empty(padded * H, 1, dtype=torch.long, device=device)
    first = torch.empty(padded * H, B, dtype=torch.long, device=device)
    _row_inputs_kernel[(padded,)](
        requested, keys, guesses, source, samples, keys_device, live_counts, geometry,
        active_rows, preceding, first, int(count), source.stride(0), samples.stride(0),
        H=int(H), CAPACITY=int(keys.shape[1]), CAP=int(cap), B=int(B),
        BLOCK_H=triton.next_power_of_2(int(H)), BLOCK_B=triton.next_power_of_2(int(B)),
        num_warps=1)
    return keys_device, live_counts, geometry, active_rows, preceding, first


class BatchRowAssembly:
    """Static inputs and the captured assembly of a C-request round in a ``padded`` bucket.

    ``load`` fills the inputs from the round, ``run`` replays (on CUDA; eager elsewhere) and
    queues the one copy home; the outputs below are overwritten by the next round's replay."""

    def __init__(self, *, count, padded, drafts, offer_width, pack_dtype, pool, H, cap, capacity,
                 block_size, coefficients, temperature, decoder_features, device, graphs):
        self.params = dict(drafts=drafts, offer_width=offer_width, pack_dtype=pack_dtype, pool=pool,
                           H=H, cap=cap, capacity=capacity, block_size=block_size,
                           coefficients=coefficients, temperature=temperature,
                           decoder_features=decoder_features, device=device, graphs=graphs)
        self.count, self.padded, self.n = int(count), int(padded), int(drafts)
        self.pool, self.H, self.cap, self.capacity = int(pool), int(H), int(cap), int(capacity)
        self.B, self.device = int(block_size), torch.device(device)
        self.coefficients = tuple(float(value) for value in coefficients)
        self.temperature, self.decoder_features = float(temperature), bool(decoder_features)
        self.graphs = bool(graphs) and self.device.type == 'cuda'
        C, rows = self.count, self.n + 1
        f64, i64 = torch.float64, torch.long

        def zeros(*shape, dtype=i64):
            return torch.zeros(*shape, dtype=dtype, device=self.device)

        self.q, self.p_top1 = zeros(C, rows, dtype=f64), zeros(C, rows, dtype=f64)
        self.saved = zeros(C, rows + 1, dtype=f64)  # an after-miss round's mass (the 16-slot shape + 1)
        self.argmax, self.rejected = zeros(C, rows), zeros(C, rows)
        self.ids, self.p64 = zeros(C, rows, offer_width), zeros(C, rows, offer_width, dtype=f64)
        self.pack = zeros(C, PACK_ROWS, rows, dtype=pack_dtype)
        self.drafts = zeros(C, self.n)
        self.host = zeros(3 * C + C * (self.B + 1))  # miss, guesses, tau bits, context gather indices
        self.gathered = zeros(C * (self.B + 1))
        # the eager path's constants, made once by the same operations
        self.ones = torch.ones(C, 1, dtype=f64, device=self.device)
        self.zeros1 = torch.zeros(C, 1, dtype=f64, device=self.device)
        self.no_offer = torch.zeros(C, dtype=torch.bool, device=self.device)
        self.after_miss_mass = torch.nn.functional.one_hot(
            torch.zeros(C, dtype=i64, device=self.device), rows).double()
        self.offer_index = torch.arange(offer_width, device=self.device)
        self.position = (torch.arange(1, self.n + 1, device=self.device, dtype=f64) / POSITION_SCALE)
        exponent = 1.0 / self.temperature
        self.fused = (rows_assembly_fused() and self.device.type == 'cuda' and libdevice is not None
                      and exponent not in _POW_SPECIAL)
        coefficients = self.coefficients + (0.0,) * (6 - len(self.coefficients))
        self.constants = torch.tensor(
            coefficients + (EPS, 1.0 - EPS, float('nan'), 1e-300, GRID, 1.0 / GRID, exponent),
            dtype=f64, device=self.device)
        self.graph = None
        self.outputs = None


    def load(self, predictions, blocks, misses, tau, sequences, *, seed, run_config=None):
        """Copy one round's inputs into the static buffers."""
        from jetspec.engines.jetspec_plus_plus.models.decoder_head import rejected_tokens
        C, n, B = self.count, self.n, self.B
        rejected = ([prediction.rejected for prediction in predictions]
                    if all(getattr(prediction, 'rejected', None) is not None for prediction in predictions)
                    else [rejected_tokens(block, after_miss=False, run_config=run_config if run_config is not None else getattr(self, 'run_config', None)) for block in blocks])
        self._stack(rejected, self.rejected)
        if all(getattr(prediction, 'compact_q', None) is not None for prediction in predictions):
            self._stack([prediction.compact_q for prediction in predictions], self.q)
            equals = torch.stack([prediction.compact_argmax_eq for prediction in predictions])
            self.argmax.copy_(torch.where(equals, self.rejected, -2))
        else:
            logits = _batched(predictions)
            # The argmax of bf16 logits is the upcast's, including first-index ties.
            self.q.copy_(logits.float().softmax(-1).gather(
                2, self.rejected.clamp_min(0)[..., None])[..., 0])
            torch.argmax(logits, -1, out=self.argmax)
        self._stack([prediction.offers.device[0] for prediction in predictions], self.ids)
        self._stack([prediction.offers.device[1] for prediction in predictions], self.p64)
        self._stack([prediction.offers.device[2] for prediction in predictions], self.pack)
        self._stack([block.e04_p_top1.reshape(-1) for block in blocks], self.p_top1)
        self._stack([sequence.pending_block.draft_ids.reshape(-1) for sequence in sequences],
                    self.drafts)
        for index, (block, miss) in enumerate(zip(blocks, misses)):
            if miss:
                if block.k_probabilities is None:
                    # s_miss: an unredrafted miss without a legacy k head has no saved mass;
                    # its home copy carries zeros, the eager stage's placeholder (the stage
                    # reports None), and the rule scores k = 0 alone either way
                    self.saved[index].zero_()
                else:
                    self.saved[index].copy_(block.k_probabilities.reshape(-1))
        # the random context of every request (live_rows.batched_kth_random_context's draws)
        # and its last committed id, gathered from one flat copy of the committed ids
        indices, tails, offset, contexts = [], [], 0, []
        for sequence in sequences:
            context = sequence.committed_ids.reshape(-1).to(self.device)
            length = context.numel()
            if length < 2:
                raise ValueError('kth mode requires a token before the correction')
            generator = torch.Generator(device='cpu')
            generator.manual_seed(int(seed) ^ int(sequence.prompt_index))
            draws = torch.randint(length, (B - 1,), generator=generator)
            indices.extend([offset + length - 2, *(offset + draws).tolist()])
            tails.append(offset + length - 1)
            contexts.append(context)
            offset += length
        host = torch.tensor([int(miss) for miss in misses]
                            + [int(sequence.pending_block.correction_guess) for sequence in sequences]
                            + torch.tensor(tau, dtype=torch.float64).view(torch.int64).tolist()
                            + indices + tails, dtype=torch.long)
        cuda = self.device.type == 'cuda'
        self.host.copy_(host.pin_memory() if cuda else host, non_blocking=cuda)
        torch.index_select(torch.cat(contexts), 0, self.host[3 * C:], out=self.gathered)

    def _stack(self, tensors, out):
        tensors = [tensor.to(self.device) if tensor.device != self.device else tensor for tensor in tensors]
        if all(tensor.dtype == out.dtype for tensor in tensors):
            torch.stack(tensors, out=out)
        else:
            out.copy_(torch.stack(tensors))

    def body(self):
        """The assembly after the vocabulary reductions; every value the eager path's."""
        C = self.count
        miss = self.host[:C] != 0
        guesses = self.host[C:2 * C]
        tau = self.host[2 * C:3 * C].view(torch.float64)
        sizes = self.pack[:, 0]
        mass16, checks, score = (self._blend_fused(miss, sizes) if self.fused
                                 else self._blend(miss, sizes))
        requested, keys, score_bad = stage_rows(score, tau, sizes, miss, self.ids,
                                                cap=self.cap, capacity=self.capacity)
        payload = torch.cat((requested, keys.reshape(-1), self.pack.reshape(-1).long(),
                             mass16.reshape(-1).view(torch.int64), checks.reshape(-1), score_bad))
        # the row inputs of the padded bucket
        samples = torch.cat((self.gathered[:C * self.B].view(C, self.B),
                             self.gathered.new_zeros(self.padded - C, self.B)))
        last = self.gathered[C * self.B:]
        source = torch.cat((last[:, None], self.drafts, self.drafts[:, -1:]), dim=1)
        source = torch.cat((source, source.new_zeros(self.padded - C, source.shape[1])))
        rows = row_inputs(requested, keys, guesses, source, samples, count=C, padded=self.padded,
                          H=self.H, cap=self.cap)
        return dict(requested=requested, keys=keys, payload=payload, row_inputs=rows)

    def _blend_fused(self, miss, sizes):
        """``_blend`` in three kernels around the eager cumprod and pool sum (rows_assembly_fused)."""
        if triton is None or libdevice is None or self.p64.device.type != 'cuda':
            from jetspec.engines.jetspec_plus_plus.draft.method.ops import reference
            return reference('blend_rows_batch')(self, miss, sizes)
        C, n, rows, pool = self.count, self.n, self.n + 1, self.pool
        f64 = torch.float64
        x = torch.empty(C, n, dtype=f64, device=self.device)
        checks = torch.empty(C, 2, dtype=torch.long, device=self.device)
        width = self.p64.shape[-1]
        _blend_x_kernel[(C,)](
            self.p_top1, self.q, self.rejected, self.argmax, sizes, self.p64, self.position,
            self.constants, x, checks, sizes.stride(0), self.p64.stride(0), self.p64.stride(1),
            N=n, ROWS=rows, WIDTH=width, BLOCK_N=triton.next_power_of_2(n),
            BLOCK_W=triton.next_power_of_2(width), FEATURES=self.decoder_features, num_warps=4,
            enable_fp_fusion=False)
        cumulative = x.cumprod(1)
        mass16 = torch.empty(C, rows + 1, dtype=f64, device=self.device)
        scoring = torch.empty(C, rows, dtype=f64, device=self.device)
        scaled = torch.empty(C, rows, pool, dtype=f64, device=self.device)
        blocks = dict(BLOCK_R=triton.next_power_of_2(rows), BLOCK_P=triton.next_power_of_2(pool))
        _blend_mass_kernel[(C,)](
            x, cumulative, miss, self.saved, self.p64, self.constants, mass16, scoring, scaled, checks,
            self.p64.stride(0), self.p64.stride(1), N=n, ROWS=rows, POOL=pool, num_warps=4,
            enable_fp_fusion=False, **blocks)
        total = scaled.sum(-1, keepdim=True)
        score = torch.empty(C, rows, pool, dtype=f64, device=self.device)
        _joint_score_kernel[(C,)](scaled, total, scoring, self.constants, score, ROWS=rows,
                                  POOL=pool, num_warps=4, enable_fp_fusion=False, **blocks)
        return mass16, checks, score

    def _blend(self, miss, sizes):
        from jetspec.engines.jetspec_plus_plus.draft.method.ops import reference
        return reference('blend_rows_batch')(self, miss, sizes)

    def run(self):
        """Replay the captured assembly (captured on first use) and queue its one copy home."""
        if not self.graphs:
            self.outputs = self.body()
        else:
            if self.graph is None:
                stream = torch.cuda.Stream(device=self.device)
                stream.wait_stream(torch.cuda.current_stream(self.device))
                with torch.cuda.stream(stream):
                    for _ in range(2):
                        self.body()
                torch.cuda.current_stream(self.device).wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    self.outputs = self.body()
                self.graph = graph
                print(f'BATCH_DRAFT_GRAPH_CAPTURE assembly count={self.count} bucket={self.padded}',
                      flush=True)
            self.graph.replay()
        payload = self.outputs['payload']
        host = torch.empty(payload.shape, dtype=payload.dtype, pin_memory=payload.is_cuda)
        host.copy_(payload, non_blocking=payload.is_cuda)
        event = None
        if payload.is_cuda:
            event = torch.cuda.Event()
            event.record()

        def read():
            if event is not None:
                event.synchronize()
            return self._unpack(host)
        self.home_read = read
        self.keys = self.outputs['keys']
        self.requested = self.outputs['requested']
        self.row_inputs = self.outputs['row_inputs']
        return read

    def _unpack(self, host):
        """BatchedRowStage._unpack's layout: requested, keys, pack, mass16 (float64 bits),
        checks, score_bad."""
        C, rows = self.count, self.n + 1
        requested = host[:C]
        offset = C + 2 * C * self.capacity
        keys = host[C:offset].view(C, self.capacity, 2)
        values = []
        for shape, is_float in (((C, PACK_ROWS, rows), False), ((C, rows + 1), True), ((C, 2), False),
                                ((C,), False)):
            size = 1
            for dim in shape:
                size *= dim
            part = host[offset:offset + size]
            values.append((part.view(torch.float64) if is_float else part).reshape(shape))
            offset += size
        return requested, keys, values


from jetspec.engines.jetspec_plus_plus.kernels import register_optimization as _register_optimization

_register_optimization('blend_rows_batch', BatchRowAssembly._blend_fused)
