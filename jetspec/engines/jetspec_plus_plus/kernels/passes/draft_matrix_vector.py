"""Default-off hybrid draft producers using the existing W4A8 activation ownership."""
import torch

from jetspec.engines.jetspec_plus_plus.engine import current_engine
from jetspec.engines.jetspec_plus_plus.profiles import kernel_enabled

try:
    import triton
    import triton.language as tl
    from triton.language.extra.cuda import libdevice
    from jetspec.engines.jetspec_plus_plus.kernels.nvfp4._w4a8_quant import _launch_dependents
    from jetspec.engines.jetspec_plus_plus.kernels.nvfp4.producers import store_mx8
except ImportError:
    triton = tl = None


def enabled(x):
    """Keep non-W4A8, CPU and unsupported row shapes on their original paths."""
    return (kernel_enabled('draft_fuse_w1h27', '1')
            and current_engine().precision_draft == 'mxfp4_w4a8'
            and x.is_cuda and x.dtype == torch.bfloat16
            and x.shape[-1] % 128 == 0
            and 1 <= x.numel() // x.shape[-1] <= 256)


if triton is not None:
    @triton.jit
    def _word_lane(word, SHIFT: tl.constexpr):
        return ((word >> SHIFT).to(tl.uint32) << 16).to(tl.float32, bitcast=True)

    @triton.jit
    def _packed_rms_scale(X, Update, Residual, row, WIDTH: tl.constexpr,
                           THREADS: tl.constexpr, EPS,
                           PENDING: tl.constexpr):
        # Four adjacent BF16 values belong to each reference lane. Load one
        # aligned word instead of four strided scalar loads, retaining all adds.
        lane = tl.arange(0, THREADS)
        a0 = tl.full((THREADS,), 0, tl.float32)
        a1 = tl.full((THREADS,), 0, tl.float32)
        a2 = tl.full((THREADS,), 0, tl.float32)
        a3 = tl.full((THREADS,), 0, tl.float32)
        for step in tl.static_range((WIDTH + THREADS * 4 - 1) // (THREADS * 4)):
            column = step * THREADS + lane
            index = row * (WIDTH // 4) + column
            if PENDING:
                left = tl.load(Update.to(tl.pointer_type(tl.uint64)) + index,
                               column < WIDTH // 4, 0)
                right = tl.load(Residual.to(tl.pointer_type(tl.uint64)) + index,
                                column < WIDTH // 4, 0)
                low = tl.inline_asm_elementwise("add.rn.bf16x2 $0, $1, $2;", "=r,r,r",
                    [left.to(tl.uint32), right.to(tl.uint32)], dtype=tl.uint32,
                    is_pure=True, pack=1)
                high = tl.inline_asm_elementwise("add.rn.bf16x2 $0, $1, $2;", "=r,r,r",
                    [(left >> 32).to(tl.uint32), (right >> 32).to(tl.uint32)],
                    dtype=tl.uint32, is_pure=True, pack=1)
                x0, x1, x2, x3 = (_word_lane(low, 0), _word_lane(low, 16),
                                  _word_lane(high, 0), _word_lane(high, 16))
            else:
                word = tl.load(X.to(tl.pointer_type(tl.uint64)) + index,
                               column < WIDTH // 4, 0)
                x0, x1, x2, x3 = (_word_lane(word, 0), _word_lane(word, 16),
                                  _word_lane(word, 32), _word_lane(word, 48))
            a0, a1, a2, a3 = a0 + x0 * x0, a1 + x1 * x1, a2 + x2 * x2, a3 + x3 * x3
        summed = ((a0 + a1) + a2) + a3
        for shift in tl.static_range(4):
            offset = 256 >> shift
            if THREADS > offset:
                summed = summed + tl.gather(summed, (lane + offset) % THREADS, 0)
        for shift in tl.static_range(5):
            other = tl.minimum(lane + (1 << shift), (lane // 32 + 1) * 32 - 1)
            summed = summed + tl.gather(summed, tl.minimum(other, THREADS - 1), 0)
        total = tl.sum(tl.where(lane == 0, summed, 0), 0)
        return libdevice.rsqrt(total * (1.0 / WIDTH) + EPS)

    @triton.jit
    def _offset_norm_tiles(X, Update, Residual, Window, Gamma, KVGamma,
                           Y, State, KV, Q, SF, KVQ, KVSF,
                           ROWS: tl.constexpr, WIDTH: tl.constexpr, KV_BLOCKS: tl.constexpr,
                           EPS: tl.constexpr, KVEPS: tl.constexpr,
                           PENDING: tl.constexpr, KV_FROM_STATE: tl.constexpr,
                           THREADS: tl.constexpr, TILES: tl.constexpr,
                           STRIDES: tl.constexpr, KVTILES: tl.constexpr,
                           KVSTRIDES: tl.constexpr, BLOCK: tl.constexpr):
        program = tl.program_id(0)
        chunk = tl.program_id(1)
        _launch_dependents(program)
        row, group = program % ROWS, program // ROWS
        column = chunk * BLOCK + tl.arange(0, BLOCK)
        index = row * WIDTH + column
        if KV_BLOCKS == 0 or not PENDING or KV_FROM_STATE:
            source = X if KV_BLOCKS == 0 or KV_FROM_STATE else tl.where(group == 0, X, Window)
            epsilon = tl.where(group == 0, EPS, KVEPS)
            scale = _packed_rms_scale(source, Update, Residual, row, WIDTH,
                                      THREADS, epsilon, PENDING)
        elif group == 0:
            scale = _packed_rms_scale(X, Update, Residual, row, WIDTH, THREADS, EPS, True)
        else:
            scale = _packed_rms_scale(Window, Update, Residual, row, WIDTH,
                                      THREADS, KVEPS, False)
        if KV_BLOCKS == 0 or group == 0:
            if PENDING:
                values = (tl.load(Update + index, column < WIDTH, 0) +
                          tl.load(Residual + index, column < WIDTH, 0)).to(tl.bfloat16)
                tl.store(State + index, values, column < WIDTH)
            else:
                values = tl.load(X + index, column < WIDTH, 0)
            weights = tl.load(Gamma + column, column < WIDTH, 0).to(tl.float32)
            result = (values.to(tl.float32) * scale) * (1.0 + weights)
            tl.store(Y + index, result, column < WIDTH)
            store_mx8(result, Q, SF, row, 0, chunk * BLOCK,
                      WIDTH, BLOCK, TILES, STRIDES)
        else:
            if PENDING and KV_FROM_STATE:
                values = (tl.load(Update + index, column < WIDTH, 0) +
                          tl.load(Residual + index, column < WIDTH, 0)).to(tl.bfloat16)
            elif KV_FROM_STATE:
                values = tl.load(X + index, column < WIDTH, 0)
            else:
                values = tl.load(Window + index, column < WIDTH, 0)
            weights = tl.load(KVGamma + (group - 1) * WIDTH + column,
                              column < WIDTH, 0).to(tl.float32)
            result = (values.to(tl.float32) * scale) * (1.0 + weights)
            tl.store(KV + (group - 1) * ROWS * WIDTH + index, result, column < WIDTH)
            if len(KVTILES):
                store_mx8(result, KVQ, KVSF, row, group - 1, chunk * BLOCK,
                          WIDTH, BLOCK, KVTILES, KVSTRIDES, ROWS)

    @triton.jit
    def _g27d_prefix_norm(X, W, Y, Q, SF, EPS: tl.constexpr,
                          ROWS: tl.constexpr, WIDTH: tl.constexpr, THREADS: tl.constexpr,
                          TILES: tl.constexpr, STRIDES: tl.constexpr):
        for block in tl.static_range(len(X)):
            if tl.program_id(0) // ROWS == block:
                row = tl.program_id(0) % ROWS
                chunk = tl.program_id(1)
                _launch_dependents(row)
                epsilon = tl.full((), EPS[block], tl.float32)
                scale = _packed_rms_scale(X[block], X[block], X[block], row,
                                          WIDTH, THREADS, epsilon, False)
                column = chunk * 4096 + tl.arange(0, 4096)
                values = tl.load(X[block] + row * WIDTH + column, column < WIDTH, 0).to(tl.float32)
                weights = tl.load(W[block] + column, column < WIDTH, 0).to(tl.float32)
                result = (values * scale) * (1.0 + weights)
                tl.store(Y[block] + row * WIDTH + column, result, column < WIDTH)
                store_mx8(result, Q[block], SF[block], row, 0, chunk * 4096,
                          WIDTH, 4096, TILES, STRIDES)


def prefix_norms(inputs, weights, eps):
    """Batch independent prefix norms while retaining each tensor's MX8 owner."""
    from jetspec.engines.jetspec_plus_plus.kernels.c1_fused import _module_reduce_threads
    from jetspec.engines.jetspec_plus_plus.kernels.nvfp4.producers import allocate_activation, _arguments
    outputs = tuple(torch.empty_like(x) for x in inputs)
    args = [_arguments(allocate_activation(x)) for x in outputs]
    q, sf, tiles, strides = zip(*args)
    rows, width = inputs[0].shape
    _g27d_prefix_norm[(len(inputs) * rows, triton.cdiv(width, 4096))](
        inputs, weights, outputs, q, sf, eps, rows, width,
        _module_reduce_threads(width, rows), tiles[0], strides[0],
        num_warps=4, enable_fp_fusion=False)
    return outputs


def offset_norm(x, window, weight, kv_weights, eps, kv_eps, *, pending=None,
                kv_from_state=False):
    """Distribute each exact offset norm and its MX8 stores over two or three CTAs."""
    from jetspec.engines.jetspec_plus_plus.kernels.c1_fused import _module_reduce_threads
    from jetspec.engines.jetspec_plus_plus.kernels.nvfp4.producers import allocate_activation, _arguments
    rows, width = x.shape
    output = torch.empty_like(x)
    state = x if pending is None else torch.empty_like(x)
    kv = x.new_empty((kv_weights.shape[0], rows, width))
    produced = allocate_activation(output)
    kv_produced = allocate_activation(kv, group_rows=rows) if kv_weights.shape[0] else None
    update, residual = (x, x) if pending is None else pending
    q, sf, tiles, strides = _arguments(produced)
    kq, ksf, ktiles, kstrides = (_arguments(kv_produced) if kv_produced is not None
                               else (None, (), (), ()))
    # Packed pending adds favor three smaller tiles; ordinary norms favor two.
    block = 4096 if pending is None else 2048
    _offset_norm_tiles[((kv_weights.shape[0] + 1) * rows, triton.cdiv(width, block))](
        x, update, residual, window, weight, kv_weights, output, state, kv,
        q, sf, kq, ksf, rows, width, kv_weights.shape[0], eps, kv_eps,
        pending is not None, kv_from_state,
        _module_reduce_threads(width, rows), tiles, strides, ktiles, kstrides, block,
        num_warps=4, enable_fp_fusion=False)
    return output, state, kv
