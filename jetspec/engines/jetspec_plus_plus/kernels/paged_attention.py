"""Device kernels for models/triton_paged.py."""

# Adaptation: defer optional accelerator imports for CPU package inspection.
from jetspec.engines.jetspec_plus_plus._optional import triton
from jetspec.engines.jetspec_plus_plus._optional import tl


@triton.jit
def _partial(Q, K, V, T, C, P, L,
             q0: tl.constexpr, q1: tl.constexpr, q2: tl.constexpr,
             k0: tl.constexpr, k1: tl.constexpr, k2: tl.constexpr, k3: tl.constexpr,
             v0: tl.constexpr, v1: tl.constexpr, v2: tl.constexpr, v3: tl.constexpr,
             t0: tl.constexpr, t1: tl.constexpr,
             WIDTH: tl.constexpr, ROWS: tl.constexpr, HEADS: tl.constexpr, GROUP: tl.constexpr,
             PAGE: tl.constexpr, PAGES: tl.constexpr, SPLITS: tl.constexpr,
             SCALE: tl.constexpr):
    b = tl.program_id(0)
    row_tiles: tl.constexpr = tl.cdiv(WIDTH, ROWS)
    h = tl.program_id(1) // row_tiles
    split = tl.program_id(2)
    row = (tl.program_id(1) % row_tiles) * ROWS + tl.arange(0, ROWS)
    key = split * 64 + tl.arange(0, 64)
    dim = tl.arange(0, 256)
    length = tl.load(C + b)
    page = tl.load(T + b * t0 + (key // PAGE) * t1,
                   key < PAGES * PAGE, other=0)
    query = tl.load(Q + (b * WIDTH + row[:, None]) * q0 + h * q1 + dim[None, :] * q2,
                    row[:, None] < WIDTH, other=0)
    kv_mask = key[:, None] < length
    keys = tl.load(K + page[:, None] * k0 + (key % PAGE)[:, None] * k1
                   + (h // GROUP) * k2 + dim[None, :] * k3,
                   kv_mask, other=0)
    values = tl.load(V + page[:, None] * v0 + (key % PAGE)[:, None] * v1
                     + (h // GROUP) * v2 + dim[None, :] * v3,
                     kv_mask, other=0)
    if Q.dtype.element_ty == tl.float32:
        score = tl.dot(query, tl.trans(keys), input_precision="ieee") * SCALE
    else:
        score = tl.dot(query, tl.trans(keys)) * SCALE
    valid = ((row[:, None] < WIDTH) & (key[None, :] < length)
             & (key[None, :] <= length - WIDTH + row[:, None]))
    score = tl.where(valid, score, -float("inf"))
    peak = tl.max(score, 1)
    weight = tl.exp(score - tl.where(peak == -float("inf"), 0., peak)[:, None])
    denominator = tl.sum(weight, 1)
    if Q.dtype.element_ty == tl.float32:
        numerator = tl.dot(weight, values, input_precision="ieee")
    else:
        numerator = tl.dot(weight, values.to(tl.float32), input_precision="tf32x3")
    partial = numerator / tl.where(denominator == 0, 1., denominator)[:, None]
    p_base = ((b * HEADS + h) * SPLITS + split) * WIDTH
    tl.store(P + (p_base + row[:, None]) * 256 + dim[None, :],
             partial, row[:, None] < WIDTH)
    tl.store(L + p_base + row,
             tl.where(denominator == 0, -float("inf"), peak + tl.log(denominator)),
             row < WIDTH)


@triton.jit
def _merge(P, L, O, o0: tl.constexpr, o1: tl.constexpr, o2: tl.constexpr,
           WIDTH: tl.constexpr, HEADS: tl.constexpr, SPLITS: tl.constexpr,
           PAD_SPLITS: tl.constexpr):
    b = tl.program_id(0)
    h = tl.program_id(1)
    row = tl.program_id(2)
    split = tl.arange(0, PAD_SPLITS)
    dim = tl.arange(0, 256)
    offset = ((b * HEADS + h) * SPLITS + split) * WIDTH + row
    lse = tl.load(L + offset, split < SPLITS, other=-float("inf"))
    peak = tl.max(lse, 0)
    weight = tl.exp(lse - tl.where(peak == -float("inf"), 0., peak))
    weight = weight / tl.where(tl.sum(weight, 0) == 0, 1., tl.sum(weight, 0))
    partial = tl.load(P + offset[:, None] * 256 + dim[None, :],
                      split[:, None] < SPLITS, other=0)
    output = tl.sum(partial * weight[:, None], 0)
    tl.store(O + (b * WIDTH + row) * o0 + h * o1 + dim * o2, output)

