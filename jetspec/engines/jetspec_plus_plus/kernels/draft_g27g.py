"""Bitwise hybrid GDN SiLU, repeated heads and private input copies."""
import torch

try:
    import triton
    import triton.language as tl
    from triton.language.extra.cuda import libdevice
    from jetspec.engines.jetspec_plus_plus.kernels.verify_mlp_glue import _ratio
except ImportError:
    triton = tl = None


if triton is not None:
    @triton.jit
    def _g27g_finish(Streams, Residual, Update, Out, WIDTH: tl.constexpr,
                     ROWS: tl.constexpr, BLOCK: tl.constexpr):
        row, group = tl.program_id(0), tl.program_id(1)
        column = tl.program_id(2) * BLOCK + tl.arange(0, BLOCK)
        offset = row * WIDTH + column
        if group == len(Streams):
            left = tl.load(Residual + offset, column < WIDTH, 0).to(tl.float32)
            right = tl.load(Update + offset, column < WIDTH, 0).to(tl.float32)
            value = (left + right).to(tl.bfloat16)
            tl.store(Out + (row * (len(Streams) + 1) + group) * WIDTH + column,
                     value, column < WIDTH)
        else:
            for i in tl.static_range(len(Streams)):
                if group == i:
                    value = tl.load(Streams[i] + offset, column < WIDTH, 0)
                    tl.store(Out + (row * (len(Streams) + 1) + group) * WIDTH + column,
                             value, column < WIDTH)

    @triton.jit
    def _g27g_inputs(X, Q, K, V, State, Private,
                     TOKENS: tl.constexpr, HEADS: tl.constexpr, REPEAT: tl.constexpr,
                     DIM: tl.constexpr, VDIM: tl.constexpr, XT: tl.constexpr,
                     XC: tl.constexpr, QPROGRAMS: tl.constexpr, VPROGRAMS: tl.constexpr,
                     STATE_SIZE: tl.constexpr, BLOCK: tl.constexpr):
        program = tl.program_id(0)
        index = tl.arange(0, BLOCK)
        if program < QPROGRAMS:
            flat = program * BLOCK + index
            column = flat % DIM
            head = flat // DIM % (HEADS * REPEAT)
            token = flat // (DIM * HEADS * REPEAT)
            offset = token * XT + (head // REPEAT * DIM + column) * XC
            q = tl.load(X + offset, token < TOKENS, 0).to(tl.float32)
            k = tl.load(X + offset + HEADS * DIM * XC, token < TOKENS, 0).to(tl.float32)
            tl.store(Q + flat, _ratio(q, libdevice.exp(-q)), token < TOKENS)
            tl.store(K + flat, _ratio(k, libdevice.exp(-k)), token < TOKENS)
        elif program < QPROGRAMS + VPROGRAMS:
            flat = (program - QPROGRAMS) * BLOCK + index
            width: tl.constexpr = HEADS * REPEAT * VDIM
            token, column = flat // width, flat % width
            v = tl.load(X + token * XT + (2 * HEADS * DIM + column) * XC,
                        token < TOKENS, 0).to(tl.float32)
            tl.store(V + flat, _ratio(v, libdevice.exp(-v)), token < TOKENS)
        elif STATE_SIZE:
            flat = (program - QPROGRAMS - VPROGRAMS) * BLOCK + index
            value = tl.load(State + flat, flat < STATE_SIZE, 0)
            tl.store(Private + flat, value, flat < STATE_SIZE)


def gdn_inputs(convolved, heads, repeat, dim, value_dim, *, state=None):
    """Write exact SiLU Q/K/V in contiguous FLA layouts and optionally clone state."""
    tokens = convolved.shape[0]
    if triton is None or not convolved.is_cuda:
        activated = torch.nn.functional.silu(convolved)
        q, k, v = activated.split((heads * dim, heads * dim, heads * repeat * value_dim), -1)
        return (q.reshape(tokens, heads, dim).repeat_interleave(repeat, 1),
                k.reshape(tokens, heads, dim).repeat_interleave(repeat, 1),
                v.reshape(tokens, heads * repeat, value_dim).contiguous(),
                None if state is None else state.clone())
    q = convolved.new_empty(tokens, heads * repeat, dim)
    k = torch.empty_like(q)
    v = convolved.new_empty(tokens, heads * repeat, value_dim)
    private = None if state is None else torch.empty_like(state)
    qprograms, vprograms = triton.cdiv(q.numel(), 256), triton.cdiv(v.numel(), 256)
    size = 0 if state is None else state.numel()
    _g27g_inputs[(qprograms + vprograms + triton.cdiv(size, 256),)](
        convolved, q, k, v, state, private, tokens, heads, repeat, dim, value_dim,
        *convolved.stride(), qprograms, vprograms, size, 256,
        num_warps=4, enable_fp_fusion=False)
    return q, k, v, private


def finish_adapter(streams, residual, update):
    """Concatenate stream states and the final BF16 residual sum in one launch."""
    if triton is None or not residual.is_cuda:
        return torch.cat([*streams, residual + update], -1)
    rows, width = residual.shape
    output = residual.new_empty(rows, (len(streams) + 1) * width)
    _g27g_finish[(rows, len(streams) + 1, triton.cdiv(width, 256))](
        tuple(streams), residual, update, output, width, rows, 256,
        num_warps=4, enable_fp_fusion=False)
    return output
