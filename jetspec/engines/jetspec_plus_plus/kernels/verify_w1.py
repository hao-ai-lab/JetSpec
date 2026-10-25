"""Shared verify producer buffers and consumers, extracted from verify_layer8b W1.

The exact BF16 output owns its immutable FP8 side output. Copies and views take
the ordinary path unless the caller explicitly hands the activation through.
Warm each owning projection before capture; allocations then live in the graph pool.
"""
import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None


if triton is not None:
    from .nvfp4._w4a8_quant import quantize_activation_block

    @triton.jit
    def store_fp8_vector(pointer, values, live):
        """Store eight adjacent E4M3 bytes directly from their packed registers."""
        tl.inline_asm_elementwise(
            """{
            .reg .pred p;
            setp.ne.u32 p, $18, 0;
            @p st.global.v2.b32 [$8], {$16, $17};
            mov.u32 $0, 0; mov.u32 $1, 0; mov.u32 $2, 0; mov.u32 $3, 0;
            mov.u32 $4, 0; mov.u32 $5, 0; mov.u32 $6, 0; mov.u32 $7, 0;
            }""",
            '=r,=r,=r,=r,=r,=r,=r,=r,l,l,l,l,l,l,l,l,r,r,r,r,r,r,r,r,r,r',
            [pointer, values.to(tl.uint8, bitcast=True), live.to(tl.int32)],
            dtype=tl.int32, is_pure=False, pack=8)

    @triton.jit
    def store_activation_vector(values, q_ptr, sf_ptr, row, column, K: tl.constexpr,
                                BLOCK: tl.constexpr, TILE: tl.constexpr):
        """Public quantizer arithmetic with vector FP8 writes and one byte per scale group.

        The 27B widths and column tiles are multiples of 128, so every masked
        eight-byte vector is either completely live or completely outside K.
        The block amax retains the public quantizer's register/shuffle reduction.
        """
        col = column + tl.arange(0, BLOCK)
        q, byte = quantize_activation_block(tl.where(col < K, values, 0.), BLOCK)
        store_fp8_vector(q_ptr + row * K + col, q, col < K)
        block = column // 32 + tl.arange(0, BLOCK // 32)
        atom: tl.constexpr = min(TILE, 128)
        within = row % atom
        physical = ((row // atom) * (K // 128) + block // 4) * 512
        physical += (within % 32) * 16 + (within // 32) * 4 + block % 4
        tl.store(sf_ptr + physical, byte, block < K // 32)

    @triton.jit
    def store_activation_packed(values, q_ptr, sf_ptr, row, column, K: tl.constexpr,
                                BLOCK: tl.constexpr, TILE: tl.constexpr):
        """Pack adjacent bytes with register/shuffle integer reductions before word stores.

        Unlike opaque per-thread vector assembly, reshaping describes logical
        adjacency even when a small producer holds strided elements per thread.
        All supported K widths and producer tiles contain complete 128-K groups.
        """
        col = column + tl.arange(0, BLOCK)
        q, byte = quantize_activation_block(tl.where(col < K, values, 0.), BLOCK)
        shifts = tl.arange(0, 4) * 8
        packed_q = tl.sum(tl.reshape(q.to(tl.uint8, bitcast=True).to(tl.uint32), (BLOCK // 4, 4))
                          << shifts[None, :], 1)
        words = column // 4 + tl.arange(0, BLOCK // 4)
        tl.store(q_ptr.to(tl.pointer_type(tl.uint32)) + row * (K // 4) + words,
                 packed_q, words < K // 4)
        packed_sf = tl.sum(tl.reshape(byte.to(tl.uint32), (BLOCK // 128, 4)) << shifts[None, :], 1)
        group = column // 128 + tl.arange(0, BLOCK // 128)
        atom: tl.constexpr = min(TILE, 128)
        within = row % atom
        physical_word = ((row // atom) * (K // 128) + group) * 128
        physical_word += (within % 32) * 4 + within // 32
        tl.store(sf_ptr.to(tl.pointer_type(tl.uint32)) + physical_word,
                 packed_sf, group < K // 128)

    @triton.jit
    def store_bf16_vector(pointer, values, live):
        """Store eight BF16 values without redistributing the FP8 producer's registers.

        The ordinary BF16 store selects eight values per thread, whereas FP8
        selects sixteen. An opaque vector store lets each thread issue its two
        BF16 stores directly instead of exchanging the whole row through shared
        memory. Complete eight-element vectors are masked together.
        Triton packs the eight BF16 arguments into four 32-bit input registers.
        """
        tl.inline_asm_elementwise(
            """{
            .reg .pred p;
            setp.ne.u32 p, $20, 0;
            @p st.global.v4.b32 [$8], {$16, $17, $18, $19};
            mov.u32 $0, 0; mov.u32 $1, 0; mov.u32 $2, 0; mov.u32 $3, 0;
            mov.u32 $4, 0; mov.u32 $5, 0; mov.u32 $6, 0; mov.u32 $7, 0;
            }""",
            '=r,=r,=r,=r,=r,=r,=r,=r,l,l,l,l,l,l,l,l,r,r,r,r,r,r,r,r,r,r,r,r',
            [pointer, values.to(tl.bfloat16).to(tl.int16, bitcast=True), live.to(tl.int32)],
            dtype=tl.int32, is_pure=False, pack=8)


def fp8store27b_enabled(run_config):
    """Select the byte-identical 27B vector stores only with their W1 producers."""
    from jetspec.engines.jetspec_plus_plus.models.config import setting
    enabled = setting(run_config, 'engine', 'engine_settings.verify_cut_fp8store27b', '0') == '1'
    if enabled and setting(run_config, 'engine', 'engine_settings.verify_cut_w1prod27b', '0') != '1':
        raise ValueError('fp8store27b requires w1prod27b=1')
    return enabled


def activation_buffers(linear, rows):
    """Private verify producer buffers in the consumer's public W1 layout."""
    from .nvfp4.w4a8 import activation_layout, pack_w4a8
    pack = getattr(linear, '_verify_w1_pack', None)
    if pack is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError('warm the W1 verify projection before CUDA graph capture')
        pack = linear._verify_w1_pack = pack_w4a8(linear.weight.detach())
    # Preserve the landed 8B diagnostic attribute; both names borrow the same pack.
    linear._verify_w1prod8b_pack = pack
    layout = activation_layout(pack, rows)
    return (layout, torch.empty(layout.data_shape, dtype=layout.data_dtype, device=linear.weight.device),
            torch.empty(layout.scale_shape, dtype=layout.scale_dtype, device=linear.weight.device))


def w1_linear(linear, x, *, activation=None):
    """Consume one immutable producer result through the public W4A8 API."""
    if activation is None:
        activation = getattr(x, '_verify_w1', None)
        if activation is None:
            activation = getattr(x, '_verify_w1prod8b', None)
    if activation is None:
        return linear(x)
    from .nvfp4.w4a8 import w4a8_linear
    layout, q, sf = activation
    bias = linear.bias if linear.layout != 'row' or linear.tp_rank == 0 else None
    out = w4a8_linear(x.reshape(-1, x.shape[-1]), linear._verify_w1_pack, bias,
                      config=layout.config, x_fp8=q, x_scale=sf)
    if linear.layout == 'row' and linear.tp_size > 1:
        torch.distributed.all_reduce(out, group=linear.tp_group)
    return out.view(*x.shape[:-1], out.shape[-1])


def packed_activation(owner, rows):
    """Buffers for the enabled packed projection group, or the ordinary path."""
    if not owner.verify_cut_pack27b2:
        return None
    from jetspec.engines.jetspec_plus_plus.models.target_pack import module_pack
    pack = module_pack(owner, owner.pack27b2_members)
    return None if pack is None else activation_buffers(pack.linear, rows)


def w1_packed(pack, x, *, activation=None):
    """The packed GEMM and its member views, retaining an explicit side output."""
    return w1_linear(pack.linear, x, activation=activation).split(pack.sizes, dim=-1)[:len(pack.members)]
