"""Gated-attention partial RoPE for query and key in one BF16 launch."""
import torch


_rotate = None


def gated_rope(query, key, cos_table, sin_table, positions, rotary_dim):
    """Keep the eager BF16 product and sum roundings; gather cos and sin by position."""
    import triton

    rows, q_heads, dim = query.shape
    kv_heads = key.shape[1]
    table = cos_table.shape[0]
    if (not query.is_cuda or query.dtype != torch.bfloat16 or key.dtype != torch.bfloat16
            or key.shape != (rows, kv_heads, dim) or positions.shape != (rows,)
            or cos_table.dtype != torch.bfloat16 or sin_table.dtype != torch.bfloat16
            or cos_table.shape != (table, rotary_dim) or sin_table.shape != cos_table.shape
            or not cos_table.is_contiguous() or not sin_table.is_contiguous()
            or query.stride(-1) != 1 or key.stride(-1) != 1
            or not 0 < rotary_dim <= dim or rotary_dim % 2):
        raise ValueError('gated RoPE requires BF16 CUDA heads, dense tables and one position per row')
    q_out, k_out = query.new_empty(rows, q_heads, dim), key.new_empty(rows, kv_heads, dim)
    if rows == 0:
        return q_out, k_out
    global _rotate
    if _rotate is None:
        _rotate = _kernel()
    _rotate[(rows, q_heads + kv_heads)](
        query, key, cos_table, sin_table, positions, q_out, k_out,
        query.stride(0), query.stride(1), key.stride(0), key.stride(1), table,
        QH=q_heads, KH=kv_heads, D=dim, R=rotary_dim,
        BLOCK=triton.next_power_of_2(dim), num_warps=4, enable_fp_fusion=False)
    return q_out, k_out


def _kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def rotate(Q, K, Cos, Sin, Pos, QO, KO, QS0, QS1, KS0, KS1, P,
               QH: tl.constexpr, KH: tl.constexpr, D: tl.constexpr, R: tl.constexpr,
               BLOCK: tl.constexpr):
        row, head = tl.program_id(0), tl.program_id(1)
        col = tl.arange(0, BLOCK)
        valid = col < D
        if head < QH:
            source = Q + row * QS0 + head * QS1
            target = QO + (row * QH + head) * D
        else:
            source = K + row * KS0 + (head - QH) * KS1
            target = KO + (row * KH + head - QH) * D
        value = tl.load(source + col, valid, other=0)
        rot = col < R
        partner = tl.where(col < R // 2, col + R // 2, col - R // 2)
        other = tl.load(source + partner, rot, other=0)
        shifted = tl.where(col < R // 2, -other, other)
        position = tl.load(Pos + row).to(tl.int64)
        # A padded row out of the table range reads zeros instead of faulting.
        live = rot & (position >= 0) & (position < P)
        cosine = tl.load(Cos + position * R + col, live, other=0)
        sine = tl.load(Sin + position * R + col, live, other=0)
        first = (value.to(tl.float32) * cosine.to(tl.float32)).to(tl.bfloat16)
        second = (shifted.to(tl.float32) * sine.to(tl.float32)).to(tl.bfloat16)
        changed = (first.to(tl.float32) + second.to(tl.float32)).to(tl.bfloat16)
        tl.store(target + col, tl.where(rot, changed, value), valid)

    return rotate
