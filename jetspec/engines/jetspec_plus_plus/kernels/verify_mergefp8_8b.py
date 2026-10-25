"""8B merge output stores without gathering each row into a separate layout."""
# Adaptation: defer unavailable accelerator imports during CPU inspection.
from jetspec.engines.jetspec_plus_plus._optional import triton, tl
try:
    from .nvfp4._w4a8_quant import quantize_activation_block
    from .verify_layer8b import gdc_launch, gdc_wait
except ImportError:
    quantize_activation_block = gdc_launch = gdc_wait = None


@triton.jit
def store_rows(output, OUT, Q, SF, rows, head, W: tl.constexpr, G: tl.constexpr,
               KH: tl.constexpr, D: tl.constexpr, MR: tl.constexpr, TILE: tl.constexpr):
    """Each program owns whole 32-K blocks and stores all its rows together."""
    tl.static_assert(D == 128)
    token = rows % W
    q_head = head * G + rows // W
    dims = tl.arange(0, D)
    rounded = output.to(tl.bfloat16)
    address = (token * (KH * G) + q_head)[:, None] * D + dims[None, :]
    tl.store(OUT + address, rounded)
    q, byte = quantize_activation_block(tl.reshape(rounded, (MR * D,)), MR * D)
    tl.store(Q + address, tl.reshape(q, (MR, D)))
    block = tl.arange(0, D // 32)
    byte = tl.reshape(byte, (MR, D // 32))
    atom: tl.constexpr = min(TILE, 128)
    within = token % atom
    physical = ((token // atom) * (KH * G * D // 128) + q_head * (D // 128)) * 512
    physical += (within % 32) * 16 + (within // 32) * 4
    # Four UE8M0 bytes occupy one aligned word in the public scale atom.
    packed = tl.sum(byte.to(tl.uint32) << (block[None, :] * 8), 1)
    tl.store((SF + physical).to(tl.pointer_type(tl.uint32)), packed)


@triton.jit
def merge(ACC, ML, LENS, OUT, W: tl.constexpr, G: tl.constexpr, KH: tl.constexpr,
          D: tl.constexpr, S: tl.constexpr, RP: tl.constexpr, BN: tl.constexpr,
          MR: tl.constexpr, PDL: tl.constexpr, Q, SF, TILE: tl.constexpr):
    """Retain the grouped merge's FP32 expression and split-reduction order."""
    if PDL:
        gdc_launch()
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
    store_rows(out, OUT, Q, SF, rows, head, W, G, KH, D, MR, TILE)


@triton.jit
def merge_decode(ACC, ML, OUT, W: tl.constexpr, G: tl.constexpr, KH: tl.constexpr,
                 D: tl.constexpr, S: tl.constexpr, RP: tl.constexpr, PDL: tl.constexpr,
                 Q, SF, TILE: tl.constexpr):
    """Width one keeps the original non-grouped split arithmetic."""
    if PDL:
        gdc_launch()
        gdc_wait()
    head = tl.program_id(0)
    row = tl.program_id(1)
    splits = tl.arange(0, S)
    dims = tl.arange(0, D)
    at = (head * S + splits) * RP + row
    m = tl.load(ML + at * 2)
    l = tl.load(ML + at * 2 + 1)
    top = tl.max(m, 0)
    weight = tl.where(m == float('-inf'), 0., tl.math.exp2(m - top))
    acc = tl.load(ACC + at[:, None] * D + dims[None, :])
    out = tl.sum(weight[:, None] * acc, 0) / tl.sum(weight * l, 0)
    store_rows(tl.reshape(out, (1, D)), OUT, Q, SF, tl.full((1,), 0, tl.int32) + row,
               head, W, G, KH, D, 1, TILE)
