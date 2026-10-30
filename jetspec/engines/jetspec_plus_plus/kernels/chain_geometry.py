"""Device kernels for draft/rows.py (one-launch post-join chain geometry)."""
from __future__ import annotations

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None


if triton is not None:
    @triton.jit
    def _chain_geometry(Cos, Sin, Length, Keys, PrefixBank, TailBank, BlockBank,
                        CCos, CSin, QCos, QSin, Mask,
                        KEY_STRIDE: tl.constexpr, KEY_OFFSET: tl.constexpr,
                        TABLE_STRIDE: tl.constexpr, PREFIX: tl.constexpr,
                        TAIL: tl.constexpr, DIM: tl.constexpr,
                        MASK_BYTES: tl.constexpr, PREFIX_BYTES: tl.constexpr,
                        BLOCK_BYTES: tl.constexpr, MASK_TILES: tl.constexpr,
                        ROPE_ELEMENTS: tl.constexpr, TILE: tl.constexpr):
        # Row r's tail count is Keys[r * KEY_STRIDE] + KEY_OFFSET. Mask byte i
        # is prefix[length] | tail[k_r] | block, the bytes of packed_mask.
        # Rotary element e is one table row gathered at its position.
        tile = tl.program_id(0)
        lane = tl.arange(0, TILE)
        length = tl.load(Length)
        if tile < MASK_TILES:
            index = tile * TILE + lane
            valid = index < MASK_BYTES
            row = index // BLOCK_BYTES
            k = tl.load(Keys + row * KEY_STRIDE, valid, other=0) + KEY_OFFSET
            p = tl.load(PrefixBank + length * PREFIX_BYTES + index % PREFIX_BYTES,
                        valid, other=0)
            t = tl.load(TailBank + k * BLOCK_BYTES + index % BLOCK_BYTES, valid, other=0)
            b = tl.load(BlockBank + index, valid, other=0)
            tl.store(Mask + index, p | t | b, valid)
        else:
            index = (tile - MASK_TILES) * TILE + lane
            valid = index < ROPE_ELEMENTS
            token = index // DIM
            col = index % DIM
            is_tail = token < TAIL
            query = token - TAIL
            k = tl.load(Keys + (query // 16) * KEY_STRIDE, valid & ~is_tail,
                        other=0) + KEY_OFFSET
            position = length + tl.where(is_tail, token, k + query % 16)
            c = tl.load(Cos + position * TABLE_STRIDE + col, valid, other=0)
            s = tl.load(Sin + position * TABLE_STRIDE + col, valid, other=0)
            tl.store(CCos + (PREFIX + token) * DIM + col, c, valid & is_tail)
            tl.store(CSin + (PREFIX + token) * DIM + col, s, valid & is_tail)
            tl.store(QCos + query * DIM + col, c, valid & ~is_tail)
            tl.store(QSin + query * DIM + col, s, valid & ~is_tail)
