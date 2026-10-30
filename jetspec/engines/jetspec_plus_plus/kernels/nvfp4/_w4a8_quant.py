"""Triton helpers for `w4a8.py`: the MXFP8 activation quantizer and the bias split-K sum.

One program quantizes 32 blocks of 32 K elements of one row. It writes FP8 E4M3
values `[M,K]` and UE8M0 bytes into the 128x4 physical scale layout that the
K1 kernel reads for its B operand: one 128-row atom per token tile, or per 128
rows of a larger tile. The arithmetic matches `w4a8.quantize_activation_mx8`
bit for bit.
"""
from __future__ import annotations

import triton
import triton.language as tl

GROUPS = 32


@triton.jit
def quantize_activation_block(values, BLOCK: tl.constexpr):
    """Quantize one flat vector of complete 32-K blocks; return FP8 and UE8M0 bytes."""
    blocks = tl.reshape(values.to(tl.float32), (BLOCK // 32, 32))
    amax = tl.max(tl.abs(blocks), axis=1)
    bits = amax.to(tl.int32, bitcast=True)
    byte = ((bits >> 23) & 0xFF) - 8 + ((bits & 0x7FFFFF) > 0x600000).to(tl.int32)
    byte = tl.minimum(tl.maximum(byte, 1), 253)
    inverse = ((254 - byte) << 23).to(tl.float32, bitcast=True)
    q = tl.reshape((blocks * inverse[:, None]).to(tl.float8e4nv), (BLOCK,))
    return q, byte.to(tl.uint8)


@triton.jit
def store_activation_row(values, q_ptr, sf_ptr, row, column, K,
                         BLOCK: tl.constexpr, TILE: tl.constexpr):
    """Public producer helper: store one row's contiguous, 32-aligned K tile.

    values is a flat power-of-two vector, BLOCK is a multiple of 32 and
    column is the tile's starting K offset. Masked columns are zeroed here.
    Pass the BF16-rounded producer result for parity with launch_quant.
    TILE must be activation_layout(...).token_tile. Pointers address one
    complete activation matrix and its physical scale storage.
    """
    col = column + tl.arange(0, BLOCK)
    q, byte = quantize_activation_block(tl.where(col < K, values, 0.), BLOCK)
    tl.store(q_ptr + row * K + col, q, col < K)
    block = column // 32 + tl.arange(0, BLOCK // 32)
    atom: tl.constexpr = min(TILE, 128)
    within = row % atom
    physical = ((row // atom) * (K // 128) + block // 4) * 512
    physical += (within % 32) * 16 + (within // 32) * 4 + block % 4
    tl.store(sf_ptr + physical, byte, block < K // 32)


@triton.jit
def _launch_dependents(anchor):
    # Programmatic dependent launch: let the GEMM grid start its weight loads now.
    return tl.inline_asm_elementwise("griddepcontrol.launch_dependents;\n mov.u32 $0, $1;", "=r,r",
                                     [anchor], dtype=tl.int32, is_pure=False, pack=1)


@triton.jit
def _quant_kernel(x_ptr, q_ptr, sf_ptr, K, BLOCKS: tl.constexpr, TRIGGER: tl.constexpr,
                  ATOM_ROWS: tl.constexpr):
    row = tl.program_id(0)
    chunk = tl.program_id(1)
    if TRIGGER:
        _launch_dependents(row)
    block = chunk * BLOCKS + tl.arange(0, BLOCKS)
    column = block[:, None] * 32 + tl.arange(0, 32)[None, :]
    valid = block < K // 32
    x = tl.load(x_ptr + row * K + column, mask=valid[:, None], other=0.).to(tl.float32)
    q, byte = quantize_activation_block(tl.reshape(x, (BLOCKS * 32,)), BLOCKS * 32)
    tl.store(q_ptr + row * K + column, tl.reshape(q, (BLOCKS, 32)), mask=valid[:, None])
    within = row % ATOM_ROWS
    physical = ((row // ATOM_ROWS) * (K // 128) + block // 4) * 512 + (within % 32) * 16 + (within // 32) * 4 \
        + block % 4
    tl.store(sf_ptr + physical, byte, mask=valid)


def launch_quant(x, q, sf, *, trigger=True, tile=128):
    """Quantize `x`; token tile `tile` owns rows `[t*tile, (t+1)*tile)` and its own scale atom(s)."""
    m, k = x.shape
    _quant_kernel[(m, triton.cdiv(k // 32, GROUPS))](x, q, sf, k, BLOCKS=GROUPS, TRIGGER=trigger,
                                                     ATOM_ROWS=min(tile, 128), num_warps=4)


@triton.jit
def _reduce_bias_kernel(partials_ptr, bias_ptr, out_ptr, count, N, SPLITS: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = offsets < count
    total = tl.zeros((BLOCK,), dtype=tl.float32)
    for split in tl.static_range(SPLITS):
        total += tl.load(partials_ptr + split * count + offsets, mask=valid, other=0.)
    total += tl.load(bias_ptr + offsets % N, mask=valid, other=0.).to(tl.float32)
    tl.store(out_ptr + offsets, total.to(tl.bfloat16), mask=valid)


def reduce_with_bias(partials, bias, out):
    """Workspace split-K sum in fixed order plus the bias, rounded once to bfloat16."""
    splits, m, n = partials.shape
    _reduce_bias_kernel[(triton.cdiv(m * n, 1024),)](partials, bias, out, m * n, n, SPLITS=splits, BLOCK=1024,
                                                     num_warps=4)
