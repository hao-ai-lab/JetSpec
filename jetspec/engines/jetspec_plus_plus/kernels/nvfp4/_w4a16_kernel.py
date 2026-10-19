"""CUDA-only Triton kernel for `w4a16.py`; imported lazily on the first CUDA call."""
import triton
import triton.language as tl


def _asm(decode: str) -> str:
    """Inline PTX: 4 kernel-order bytes + 4 equal scale bytes -> 4 even, 4 odd weights."""
    lines = ["{", ".reg .b8 b<4>;", ".reg .b16 sl, sh;", ".reg .b32 sc, h, t, u, c;",
             "mov.b32 {b0, b1, b2, b3}, $4;", "mov.b32 {sl, sh}, $5;",
             "cvt.rn.f16x2.e4m3x2 sc, sl;"]
    if decode == "bf16":
        # bf16 bits of scale*2**112: f16 exponent + 224, mantissa >> 3 (scales are nonzero).
        lines += ["shr.b32 sc, sc, 3;", "add.u32 sc, sc, 0x70007000;"]
    if decode == "int":
        lines += ["mov.b32 c, 0x74007400;"]
    for index in range(4):
        if decode == "int":
            lines += [f"bfe.u32 t, $4, {8 * index}, 4;", f"bfe.u32 u, $4, {8 * index + 4}, 4;",
                      "shl.b32 u, u, 16;", "or.b32 t, t, u;",
                      "and.b32 u, t, 0x00080008;", "shl.b32 u, u, 12;",
                      "and.b32 t, t, 0x00070007;", "shl.b32 t, t, 9;", "or.b32 h, t, u;",
                      "mul.rn.f16x2 h, h, c;"]
        else:
            lines += [f"cvt.rn.f16x2.e2m1x2 h, b{index};"]
        if decode == "bf16":
            # bf16 bits of value*2**-112: magnitude >> 3, sign kept; exact for E2M1.
            lines += ["and.b32 t, h, 0x7FFF7FFF;", "shr.b32 t, t, 3;",
                      "and.b32 u, h, 0x80008000;", "or.b32 h, t, u;",
                      f"mul.rn.bf16x2 ${index}, h, sc;"]
        else:
            lines += [f"mul.rn.f16x2 ${index}, h, sc;"]
    return "\n".join(lines + ["}"])


_ASM_F16 = tl.constexpr(_asm("f16"))
_ASM_BF16 = tl.constexpr(_asm("bf16"))
_ASM_INT = tl.constexpr(_asm("int"))


@triton.jit
def _decode(q, s, DECODE: tl.constexpr):
    if DECODE == 0:
        lo, hi = tl.inline_asm_elementwise(
            _ASM_F16, "=r,=r,=r,=r,r,r", [q, s], dtype=(tl.float16, tl.float16), is_pure=True, pack=4)
    elif DECODE == 1:
        lo, hi = tl.inline_asm_elementwise(
            _ASM_BF16, "=r,=r,=r,=r,r,r", [q, s], dtype=(tl.bfloat16, tl.bfloat16), is_pure=True, pack=4)
    else:
        lo, hi = tl.inline_asm_elementwise(
            _ASM_INT, "=r,=r,=r,=r,r,r", [q, s], dtype=(tl.float16, tl.float16), is_pure=True, pack=4)
    return lo, hi


@triton.jit
def _nibbles(pairs, scale, BLOCK_N: tl.constexpr, HALF: tl.constexpr):
    """Pure-Triton decode: (low, high) nibbles -> E2M1 * scale in BF16, via F16 bits."""
    codes = tl.reshape(tl.join(pairs & 15, pairs >> 4), (BLOCK_N, HALF))
    bits = ((codes & 7).to(tl.int16) << 9) | ((codes & 8).to(tl.int16) << 12)
    return (bits.to(tl.float16, bitcast=True).to(tl.float32) * scale).to(tl.bfloat16)


@triton.jit
def _load(q_ptr, s_ptr, x_ptr, tile, last, row_ok, token_ok, NB: tl.constexpr, HALF: tl.constexpr,
          BLOCK_K: tl.constexpr):
    live = tile < last
    q = tl.load(q_ptr + tile * HALF, mask=row_ok[:, None, None] & live, other=0)
    s = tl.load(s_ptr + tile * NB, mask=row_ok[:, None] & live, other=0)
    x = tl.load(x_ptr + tile * BLOCK_K, mask=token_ok[:, None] & live, other=0.)
    return q, s, x


@triton.jit
def _tile(q, s, x, acc, BLOCK_N: tl.constexpr, BLOCK_M: tl.constexpr, NB: tl.constexpr,
          HALF: tl.constexpr, DECODE: tl.constexpr, X_SCALE: tl.constexpr, ABLATE: tl.constexpr = 0):
    # ABLATE is a diagnostic: 1 skips the decode (bytes cast to bf16), 2 skips the MMA.
    if ABLATE == 1:
        lo = tl.reshape(q.to(tl.bfloat16), (BLOCK_N, HALF))
        hi = lo
    elif DECODE == 3:
        lo_bytes, hi_bytes = tl.split(tl.permute(tl.reshape(q, (BLOCK_N, NB, 2, 2, 2)), (0, 1, 2, 4, 3)))
        block_scale = s.to(tl.float8e4nv, bitcast=True).to(tl.float32) * 16384.
        block_scale = tl.reshape(tl.broadcast_to(block_scale[:, :, None], (BLOCK_N, NB, 8)), (BLOCK_N, HALF))
        lo = _nibbles(lo_bytes, block_scale, BLOCK_N, HALF)
        hi = _nibbles(hi_bytes, block_scale, BLOCK_N, HALF)
    else:
        lo, hi = _decode(q, tl.broadcast_to(s[:, :, None], (BLOCK_N, NB, 8)), DECODE)
        lo = tl.reshape(lo, (BLOCK_N, HALF))
        hi = tl.reshape(hi, (BLOCK_N, HALF))
    if DECODE % 2 == 0 and ABLATE != 1:
        x = (x.to(tl.float32) * X_SCALE).to(tl.float16)
    if ABLATE == 2:
        return acc + tl.sum(lo.to(tl.float32) + hi.to(tl.float32), axis=1)[:, None] + tl.sum(x.to(tl.float32))
    even, odd = tl.split(tl.reshape(x, (BLOCK_M, HALF, 2)))
    acc = tl.dot(lo, tl.trans(even), acc)
    return tl.dot(hi, tl.trans(odd), acc)


@triton.jit
def _w4a16_kernel(X, Q, S, G, Bias, Out, Workspace, Counters, XD, QD, SD, M, N, K, stride_xm, stride_om,
                  HAS_BIAS: tl.constexpr, SPLIT_K: tl.constexpr, BLOCK_M: tl.constexpr,
                  BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, DECODE: tl.constexpr,
                  X_SCALE: tl.constexpr, OUT_SHIFT: tl.constexpr, NUM_STAGES: tl.constexpr,
                  PREFETCH: tl.constexpr, USE_TMA: tl.constexpr, SCALE_TMA: tl.constexpr,
                  ABLATE: tl.constexpr = 0):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    NB: tl.constexpr = BLOCK_K // 16
    HALF: tl.constexpr = BLOCK_K // 2
    k_tiles = K // BLOCK_K
    first = pid_k * k_tiles // SPLIT_K
    last = (pid_k + 1) * k_tiles // SPLIT_K
    rows = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    row_ok = rows < N
    rows64 = rows.to(tl.int64)
    blocks = tl.arange(0, NB)
    q_ptr = (Q + rows64[:, None, None] * (K // 2) + blocks[None, :, None] * 8
             + tl.arange(0, 8)[None, None, :])
    s_ptr = S + rows64[:, None] * (K // 16) + blocks[None, :]
    tokens = tl.arange(0, BLOCK_M)
    token_ok = tokens < M
    x_ptr = X + tokens[:, None] * stride_xm + tl.arange(0, BLOCK_K)[None, :]
    acc = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
    if USE_TMA:
        # Descriptor (TMA) loads: one bulk copy per tile, no per-element addresses.
        for tile in tl.range(first, last, num_stages=NUM_STAGES):
            q = tl.reshape(QD.load([pid_n * BLOCK_N, tile * HALF]), (BLOCK_N, NB, 8))
            if SCALE_TMA:
                s = SD.load([pid_n * BLOCK_N, tile * NB])
            else:
                s = tl.load(s_ptr + tile * NB, mask=row_ok[:, None], other=0)
            x = XD.load([0, tile * BLOCK_K])
            acc = _tile(q, s, x, acc, BLOCK_N, BLOCK_M, NB, HALF, DECODE, X_SCALE, ABLATE)
    elif PREFETCH == 0:
        for tile in tl.range(first, last, num_stages=NUM_STAGES):
            q, s, x = _load(q_ptr, s_ptr, x_ptr, tile, last, row_ok, token_ok, NB, HALF, BLOCK_K)
            acc = _tile(q, s, x, acc, BLOCK_N, BLOCK_M, NB, HALF, DECODE, X_SCALE, ABLATE)
    else:
        # Manual register prefetch: the next tiles' loads are issued before this tile's math.
        qa, sa, xa = _load(q_ptr, s_ptr, x_ptr, first, last, row_ok, token_ok, NB, HALF, BLOCK_K)
        if PREFETCH == 2:
            qb, sb, xb = _load(q_ptr, s_ptr, x_ptr, first + 1, last, row_ok, token_ok, NB, HALF, BLOCK_K)
        for tile in tl.range(first, last, num_stages=1):
            qn, sn, xn = _load(q_ptr, s_ptr, x_ptr, tile + PREFETCH, last, row_ok, token_ok, NB, HALF, BLOCK_K)
            acc = _tile(qa, sa, xa, acc, BLOCK_N, BLOCK_M, NB, HALF, DECODE, X_SCALE, ABLATE)
            if PREFETCH == 2:
                qa, sa, xa = qb, sb, xb
                qb, sb, xb = qn, sn, xn
            else:
                qa, sa, xa = qn, sn, xn
    scale = tl.load(G)
    if DECODE % 2 == 0:
        scale = scale * OUT_SHIFT
    out_ptr = Out + tokens[None, :] * stride_om + rows[:, None]
    mask = row_ok[:, None] & token_ok[None, :]
    if SPLIT_K == 1:
        result = acc * scale
        if HAS_BIAS:
            result += tl.load(Bias + rows, mask=row_ok, other=0.).to(tl.float32)[:, None]
        tl.store(out_ptr, result.to(tl.bfloat16), mask=mask)
    else:
        part = tokens[None, :] * N + rows[:, None]
        tl.store(Workspace + pid_k * M * N + part, acc, mask=mask)
        tl.debug_barrier()
        arrived = tl.atomic_add(Counters + pid_n, 1, sem="acq_rel", scope="gpu")
        if arrived == SPLIT_K - 1:
            total = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
            for split in tl.static_range(SPLIT_K):
                total += tl.load(Workspace + split * M * N + part, mask=mask, other=0.,
                                 cache_modifier=".cg")
            result = total * scale
            if HAS_BIAS:
                result += tl.load(Bias + rows, mask=row_ok, other=0.).to(tl.float32)[:, None]
            tl.store(out_ptr, result.to(tl.bfloat16), mask=mask)
            tl.atomic_xchg(Counters + pid_n, 0, sem="relaxed", scope="gpu")
