
from typing import Optional, Tuple
import weakref
from functools import partial

import cuda.bindings.driver as cuda


import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, const_expr
from cutlass.cute.runtime import from_dlpack

import torch
from torch import Tensor

from optimus_cutedsl.utils import torch2cute_dtype_map, row_reduce, _convert_from_dlpack_cached
from optimus_cutedsl.reduction_base import ReductionBase
import optimus_cutedsl.utils as utils
from optimus_cutedsl.flash_attn.copy_utils import get_copy_atom
import optimus_cutedsl.flash_attn.utils as copy_utils
try:
    # Vendored into parallel-tree-decoding: the optimus_benchmarks harness is not
    # shipped (benchmark-only; not used by the tree-attn decode path). Keep the
    # import optional so this module loads without it. See ../NOTICE.
    from optimus_benchmarks.api import benchmark_case, Float16Tensor, IntTensor
except ImportError:  # pragma: no cover - benchmark harness not vendored
    benchmark_case = None
    Float16Tensor = None
    IntTensor = None


# TODO（yuanxiaolan）: 
# add swizzle to reduce bank conflict
class FusedQKNormRope(ReductionBase):
    def __init__(self, dtype: cutlass.Numeric, N: int, head_dim: int, num_q_head: int, num_kv_head: int, rotary_dim: int):
        super().__init__(dtype, N, stage=1)
        self.reload_from = None if N <= 16384 else "smem"
        self.delay_w_load = False
        self.head_dim = head_dim
        self.num_q_head = num_q_head
        self.num_kv_head = num_kv_head
        self.rotary_dim = rotary_dim

    def _calculate_threads_per_head(self):
        """Calculate the number of threads per row for the RMSNorm kernel."""
        N = self.head_dim
        if N <= 64:
            return 8
        elif N <= 128:
            return 16
        elif N <= 3072:
            return 32
        elif N <= 6144:
            return 64
        elif N <= 16384:
            return 128
        else:
            return 256
    def _get_num_threads(self):
        return 128 if self.N <= 16384 else 256

    def _get_tv_layout(self, num_copy_bits=128):
        vecsize = num_copy_bits // self.dtype.width #8
        assert self.head_dim % vecsize == 0, f"Input N {self.head_dim} is not divisible by vector size {vecsize}"
        num_threads = self._get_num_threads()
        assert num_threads % cute.arch.WARP_SIZE == 0

        threads_per_head = self._calculate_threads_per_head() # 16
        num_blocks_N = cute.ceil_div(self.head_dim // vecsize, threads_per_head * self.cluster_n) # 1
        cols_per_block = num_threads // threads_per_head # 8
        tiler_mn = (cols_per_block, vecsize * num_blocks_N * threads_per_head) # 8, 128
        tv_layout = cute.make_layout(
            ((threads_per_head, cols_per_block), (vecsize, num_blocks_N)),
            stride=(
                (vecsize * cols_per_block, 1),
                (cols_per_block, cols_per_block * vecsize * threads_per_head),
            ),
        )
        return tiler_mn, tv_layout

    def _set_cluster_n(self):
        """
        Set the number of clusters for the RMSNorm kernel.
        Stored in self.cluster_n.
        """
        N = self.head_dim

        # cluster_n = 4 is faster and cluster_n = 2 for N=64k for some reason
        # Similarly cluster_n = 8 is faster for N=128k
        if const_expr(self.dtype.width == 16):
            # 16-bit types (fp16, bf16)
            if N <= 16 * 1024:
                cluster_n = 1
            elif N <= 32 * 1024:
                cluster_n = 2
            elif N <= 64 * 1024:
                cluster_n = 4
            elif N <= 128 * 1024:
                cluster_n = 8
            else:
                cluster_n = 16
        else:
            # 32-bit types (fp32)
            if N <= 32 * 1024:
                cluster_n = 1
            elif N <= 64 * 1024:
                cluster_n = 2
            elif N <= 128 * 1024:
                cluster_n = 4
            elif N <= 256 * 1024:
                cluster_n = 8
            else:
                cluster_n = 16

        self.cluster_n = cluster_n

    def _smem_size_in_bytes(self, tiler_mn, num_warps):
        return (
            cute.size_in_bytes(self.dtype, cute.make_layout(tiler_mn))
            + self.stage * num_warps * self.cluster_n * (self.reduction_dtype.width // 8)
            + self.stage * (cutlass.Int64.width // 8)
        )

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,
        mQW: Optional[cute.Tensor],
        mKW: Optional[cute.Tensor],
        mCos: cute.Tensor,        
        mSin: cute.Tensor,
        mPosId: cute.Tensor,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        head_dim: Int32,
        num_q_head: Int32,
        num_kv_head: Int32,
        stream: cuda.CUstream,
        eps: Float32 = 1e-6,
        norm_weight_bias: Float32 = 1.0,
    ):
        semistatic_shape_X = (*mX.shape[:-1], self.num_q_head + self.num_kv_head * 2, self.head_dim)

        new_stride = lambda t: (
            cute.assume(t.stride[0], divby=self.head_dim),
            cute.assume(t.stride[0], divby=128 // t.element_type.width),
            t.stride[1],
        )

        mX = cute.make_tensor(mX.iterator, cute.make_layout(semistatic_shape_X, stride=new_stride(mX)))

        semistatic_shape_Q = (*mQ.shape[:-1], self.num_q_head, self.head_dim) 
        semistatic_shape_KV = (*mK.shape[:-1], self.num_kv_head, self.head_dim) 

        new_stride_QKV = lambda t: (
            cute.assume(t.stride[0], divby=self.head_dim),
            self.head_dim // (128 // t.element_type.width),
            t.stride[1],
        )
        mQ = cute.make_tensor(mQ.iterator, cute.make_layout(semistatic_shape_Q, stride=new_stride_QKV(mQ)))
        
        mK, mV = [
            cute.make_tensor(t.iterator, cute.make_layout(semistatic_shape_KV, stride=new_stride_QKV(t)))
            if const_expr(t is not None)
            else None
            for t in (mK, mV)
        ]

        semistatic_shape_Cos_Sin = (*mCos.shape[:-1], self.rotary_dim)
        new_stride_Cos_Sin = lambda t: (
            cute.assume(t.stride[0], divby=64 // t.element_type.width),
            t.stride[1],
        )
        mCos = cute.make_tensor(mCos.iterator, cute.make_layout(semistatic_shape_Cos_Sin, stride=new_stride_Cos_Sin(mCos)))
        mSin = cute.make_tensor(mSin.iterator, cute.make_layout(semistatic_shape_Cos_Sin, stride=new_stride_Cos_Sin(mSin)))

        self._set_cluster_n()
        largest_dtype_width = const_expr(
                mX.element_type.width,
        )
        tiler_mn, tv_layout = self._get_tv_layout(
            num_copy_bits=128 // largest_dtype_width * mX.element_type.width
        )

        num_threads = cute.size(tv_layout, mode=[0])
        num_warps = num_threads // cute.arch.WARP_SIZE
        if const_expr(mQW is not None):
            mQW_expanded_layout = cute.prepend(
                mQW.layout, cute.make_layout((tiler_mn[0],), stride=(0,))
            )
            mQW = cute.make_tensor(mQW.iterator, mQW_expanded_layout)
        if const_expr(mKW is not None):
            mKW_expanded_layout = cute.prepend(
                mKW.layout, cute.make_layout((tiler_mn[0],), stride=(0,))
            )
            mKW = cute.make_tensor(mKW.iterator, mKW_expanded_layout)

        self.kernel(
            mX, mQW, mKW, mCos, mSin, mPosId, mQ, mK, mV, num_q_head, num_kv_head, eps, norm_weight_bias, tv_layout, tiler_mn, self.reload_from
        ).launch(
            grid=[cute.ceil_div(mX.shape[0], tiler_mn[0]),  self.cluster_n, self.num_q_head + self.num_kv_head * 2],
            block=[num_threads, 1, 1],
            cluster=([1, self.cluster_n, 1] if const_expr(self.cluster_n > 1) else None),
            smem=self._smem_size_in_bytes(
                tiler_mn, num_warps, 
            ),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mX: cute.Tensor,
        mQW: Optional[cute.Tensor],
        mKW: Optional[cute.Tensor],
        mCos: cute.Tensor,
        mSin: cute.Tensor,
        mPosId: cute.Tensor,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        num_q_head: Int32,
        num_kv_head: Int32,
        eps: cute.Float32,
        norm_weight_bias: cute.Float32,
        tv_layout: cute.Layout,
        tiler_mn: cute.Shape,
        reload_from: cutlass.Constexpr = None,
        delay_w_load: cutlass.Constexpr = False,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, bidz = cute.arch.block_idx()

        VALS_PER_THREAD = 64 // mCos.element_type.width  # 4
        LANES_PER_HEAD = tv_layout.shape[0][0]
        num_threads = cute.size(tv_layout, mode=[0])
        NUM_LANES = num_threads // LANES_PER_HEAD
        rowid_in_block = tidx // LANES_PER_HEAD

        if const_expr(self.cluster_n > 1):
            cluster_y = cute.arch.block_idx()[1]
        else:
            cluster_y = const_expr(0)

        smem = cutlass.utils.SmemAllocator()
        sX = smem.allocate_tensor(
            mX.element_type,
            cute.make_ordered_layout(tiler_mn, order=(1, 0)),
            byte_alignment=16,
        )

        reduction_buffer, mbar_ptr = self._allocate_reduction_buffer_and_mbar(smem, tv_layout)

        shape_X = (mX.shape[0] * mX.shape[1], mX.shape[2])

        # slice for CTAs
        # We use domain_offset_i64 to deal with tensors larger than 2^31 elements
        num_qkv_heads = num_q_head + num_kv_head * 2
        row_base = bidx * tiler_mn[0] + rowid_in_block
        row = row_base * num_qkv_heads + bidz
        pred_row = row < shape_X[0]

        # 直接应用对齐，假设iterator支持align方法
        new_iterator = (mX.iterator + bidz * mX.shape[2]).align(16)

        mX = cute.make_tensor(
            new_iterator,
            cute.make_layout((mX.shape[0], mX.shape[2]), stride=(mX.stride[0], mX.stride[2]))
        )
        gX = cute.local_tile(mX, tiler_mn, (bidx, cluster_y)) #8*128

        # declare the atoms which will be used later for memory copy
        num_copy_elems_X = tv_layout.shape[1][0]

        copy_atom_load_X_async = get_copy_atom(
            mX.element_type, num_copy_elems_X, is_async=True
        )

        thr_copy_X = cute.make_tiled_copy(
            copy_atom_load_X_async, 
            tv_layout,  #((16,8),(8,1)):((64,1),(8,1024))
            tiler_mn #(8, 128)
            ).get_slice(
            tidx
        )

        tXgX = thr_copy_X.partition_S(gX)

        tXsX = thr_copy_X.partition_D(sX)


        token_idx = row_base
        head_idx = bidz


        new_iterator = (mQ.iterator + bidz * mQ.shape[2]).align(16)
        mQ = cute.make_tensor(
            new_iterator,
            cute.make_layout((mQ.shape[0], mQ.shape[2]), stride=(mQ.stride[0], mQ.stride[2]))
        )
        new_iterator = (mK.iterator + (bidz - self.num_q_head) * mK.shape[2]).align(16)
        mK = cute.make_tensor(
            new_iterator,
            cute.make_layout((mK.shape[0], mK.shape[2]), stride=(mK.stride[0], mK.stride[2]))
        )
        new_iterator = (mV.iterator + (bidz - self.num_q_head - self.num_kv_head) * mV.shape[2]).align(16)
        mV = cute.make_tensor(
            new_iterator,
            cute.make_layout((mV.shape[0], mV.shape[2]), stride=(mV.stride[0], mV.stride[2]))
        )
        gQ, gK, gV = [
            cute.local_tile(mT, tiler_mn, (bidx, cluster_y)) if mT is not None else None
            for mT in (mQ, mK, mV)
        ] #8*128
        gQW = cute.local_tile(mQW, tiler_mn, (0, cluster_y))
        tXgQW = thr_copy_X.partition_S(gQW)
        tXrQW = cute.make_fragment_like(tXgQW)
        tXrQW.fill(0.0)

        gKW = cute.local_tile(mKW, tiler_mn, (0, cluster_y))
        tXgKW = thr_copy_X.partition_S(gKW)
        tXrKW = cute.make_fragment_like(tXgKW)
        tXrKW.fill(0.0)

        tXrX = cute.make_fragment_like(tXgX)

        # tXgO = thr_copy_X.partition_D(gQ) if head_idx < self.num_q_head else thr_copy_X.partition_D(gK) if head_idx < self.num_q_head + self.num_kv_head else thr_copy_X.partition_D(gV)
        tXgQ = thr_copy_X.partition_D(gQ)
        tXrQ = cute.make_fragment_like(tXgQ)
        tXgK = thr_copy_X.partition_D(gK)
        tXrK = cute.make_fragment_like(tXgK)
        tXgV = thr_copy_X.partition_D(gV)
        tXrV = cute.make_fragment_like(tXgV)




        num_warps = cute.size(tv_layout, mode=[0]) // cute.arch.WARP_SIZE
        self._initialize_cluster(tidx, mbar_ptr, num_warps)

        is_even_N = cutlass.const_expr(shape_X[1] == tiler_mn[1] * self.cluster_n)

        tXpX = utils.predicate_k(tXgX, limit=shape_X[1]) if not is_even_N else None
        # Each copy will use the same number of elements as X and same predicate
        copy = partial(copy_utils.copy, pred=tXpX, num_copy_elems=num_copy_elems_X)


        if row < shape_X[0]:
            copy(tXgX, tXsX, is_async=True)
        cute.arch.cp_async_commit_group()

        if const_expr(not delay_w_load):
            if head_idx < num_q_head:
                copy(tXgQW, tXrQW)
            elif head_idx < num_q_head + num_kv_head:
                copy(tXgKW, tXrKW)

        cute.arch.cp_async_wait_group(0)
        cute.autovec_copy(tXsX, tXrX)
        y = tXrX.load().to(cute.Float32)

        if head_idx < num_q_head + num_kv_head:
            x = tXrX.load().to(cute.Float32)

            threads_per_row = tv_layout.shape[0][0]
            sum_sq_x = row_reduce(
                x * x,
                cute.ReductionOp.ADD,
                threads_per_row,
                reduction_buffer[None, None, 0],
                mbar_ptr,
                init_val=0.0,
                hook_fn=(cute.arch.cluster_wait if const_expr(self.cluster_n > 1) else None),
            )
            rstd = cute.math.rsqrt(sum_sq_x / shape_X[1] + eps, fastmath=True)

            if const_expr(delay_w_load):
                if head_idx < num_q_head:
                    copy(tXgQW, tXrQW)
                elif head_idx < num_q_head + num_kv_head:
                    copy(tXgKW, tXrKW)

            if const_expr(reload_from == "smem" or reload_from == "gmem"):
                if const_expr(reload_from == "smem"):
                    cute.autovec_copy(tXsX, tXrX)
                else:
                    copy(tXgX, tXrX)
                x = tXrX.load().to(cute.Float32)

            x_hat = x * rstd

            y = x_hat
            if head_idx < num_q_head:
                y *= tXrQW.load().to(cute.Float32) + norm_weight_bias
                tXrQ.store(y.to(tXrQ.element_type))
                copy(tXrQ, tXsX)
            elif head_idx < num_q_head + num_kv_head:
                y *= tXrKW.load().to(cute.Float32) + norm_weight_bias
                tXrK.store(y.to(tXrK.element_type))
                copy(tXrK, tXsX)
            cute.arch.sync_warp()
        # Apply RoPE to Q and K heads (not V)
        if const_expr(mCos is not None and mSin is not None and mPosId is not None):
            is_q_or_k = head_idx < (num_q_head + num_kv_head)
            if is_q_or_k:


                lane = tidx % LANES_PER_HEAD

                if lane < self.rotary_dim // VALS_PER_THREAD:


                    # Load position ID for this row.
                    # Ensure the `arith.select` branches have the same integer type in MLIR.
                    pos = (
                        cutlass.Int64(mPosId[token_idx])
                        if row < shape_X[0]
                        else cutlass.Int64(0)
                    )

                    mCos = utils.domain_offset_i64((pos, 0), mCos)
                    mSin = utils.domain_offset_i64((pos, 0), mSin)
                    gCos = cute.local_tile(mCos, (1, 64), (0, 0))
                    gSin = cute.local_tile(mSin, (1, 64), (0, 0))

                    copy_atom_b16 = cute.make_copy_atom(
                        cute.nvgpu.CopyUniversalOp(),
                        mCos.element_type,
                        num_bits_per_copy=64,
                    )

                    thr_layout_cs = cute.make_layout((1, 16), stride=(0, 1))  # 16 threads
                    val_layout_cs = cute.make_layout((1, VALS_PER_THREAD), stride=(0, 1))
                    tiled_cs = cute.make_tiled_copy_tv(copy_atom_b16, thr_layout_cs, val_layout_cs)
                    thr_cs = tiled_cs.get_slice(lane)

                    tCgC = thr_cs.partition_S(gCos)
                    tSgS = thr_cs.partition_S(gSin)

                    cos_frag = cute.make_fragment_like(tCgC)
                    sin_frag = cute.make_fragment_like(tSgS)
                    cute.copy(copy_atom_b16, tCgC, cos_frag)
                    cute.copy(copy_atom_b16, tSgS, sin_frag)


                    sX0 = cute.make_tensor(sX.iterator, cute.make_layout((8, 64), stride=(128, 1)))
                    sX1 = cute.make_tensor(sX.iterator + self.rotary_dim, cute.make_layout((8, 64), stride=(128, 1)))

                    thr_layout = cute.make_layout((8, 16), stride=(16, 1))

                    tiled_copy = cute.make_tiled_copy_tv(copy_atom_b16, thr_layout, val_layout_cs)
                    thr_copy = tiled_copy.get_slice(tidx)

                    # -------- load first half -> tXrX0 --------
                    tXsX0 = thr_copy.partition_S(sX0)          # source slice in shared
                    tXrX0 = cute.make_fragment_like(tXsX0)     # reg fragment (4 vals)
                    cute.copy(copy_atom_b16, tXsX0, tXrX0)

                    # -------- load second half -> tXrX1 --------
                    tXsX1 = thr_copy.partition_S(sX1)
                    tXrX1 = cute.make_fragment_like(tXsX1)
                    cute.copy(copy_atom_b16, tXsX1, tXrX1)  

                    x0 = tXrX0.load().to(cute.Float32)
                    x1 = tXrX1.load().to(cute.Float32)
                    cos = cos_frag.load().to(cute.Float32)
                    sin = sin_frag.load().to(cute.Float32)
                    y0 = x0 * cos - x1 * sin
                    y1 = x0 * sin + x1 * cos
                    tXsX0.store(y0.to(tXsX0.element_type))
                    tXsX1.store(y1.to(tXsX1.element_type))

                cute.arch.sync_warp()

                # # For each pair of elements (i, i+half_dim) - real and imaginary parts
                if head_idx < num_q_head:
                    if row < shape_X[0]:
                        copy(tXsX, tXgQ)
                elif head_idx < num_q_head + num_kv_head:
                    if row < shape_X[0]:
                        copy(tXsX, tXgK)
            else:
                tXrV.store(y.to(tXrV.element_type))
                if row < shape_X[0]:
                    copy(tXrV, tXgV)


def _qknorm_rope_impl(
    qkv: Tensor,
    qnorm_weight: Optional[Tensor],
    knorm_weight: Optional[Tensor],
    cos: Optional[Tensor],
    sin: Optional[Tensor],
    pos_id: Optional[Tensor],
    q: Tensor,
    k: Tensor,
    v: Tensor,
    head_dim: int,
    num_q_head: int,
    num_kv_head: int,
    rotary_dim: int,
    eps: float = 1e-6,    
    norm_weight_bias: float = 1.0,
) -> None:
    """RMSNorm forward pass.
    Args:
        x: Input tensor of shape (M, N)
        weight: Optional weight tensor of shape (N,)
        eps: Small value for numerical stability
    Returns:
        Normalized output tensor of same shape as x
    """
    qkv_dtype = qkv.dtype
    assert qkv_dtype in [torch.float16, torch.bfloat16, torch.float32], "Unsupported dtype"
    if qnorm_weight is not None and knorm_weight is not None:
        assert qnorm_weight.dtype in [
            torch.bfloat16,
            torch.float16,
        ], "qnorm_weight must be float32, float16 or bfloat16"
    assert rotary_dim % 4 == 0, "rotary_dim must be divisible by 4"

    # device = qkv.device
    dtype = torch2cute_dtype_map[qkv.dtype]
    convert_from_dlpack_ = lambda x: (
        from_dlpack(x, assumed_align=16, enable_tvm_ffi=True).mark_layout_dynamic(leading_dim=1)
    )

    qkv_tensor = convert_from_dlpack_(qkv)
    q_tensor = convert_from_dlpack_(q)
    k_tensor = convert_from_dlpack_(k)
    v_tensor = convert_from_dlpack_(v)

    cos_tensor =  _convert_from_dlpack_cached(cos, leading_dim=1)
    sin_tensor =  _convert_from_dlpack_cached(sin, leading_dim=1)

    qnorm_weight_tensor = _convert_from_dlpack_cached(qnorm_weight, leading_dim=0)
    knorm_weight_tensor = _convert_from_dlpack_cached(knorm_weight, leading_dim=0)
    pos_id_tensor = from_dlpack(pos_id, assumed_align=16, enable_tvm_ffi=True).mark_layout_dynamic(leading_dim=0)


    current_stream = cuda.CUstream(torch.cuda.current_stream(qkv.device).cuda_stream)
    # current_stream = cuda.CUstream(stream.cuda_stream) if stream is not None else None
    compile_key = (
        head_dim,
        num_q_head,
        num_kv_head,
        rotary_dim,
        norm_weight_bias,
        dtype,
        qnorm_weight_tensor.element_type if qnorm_weight_tensor is not None else None,

    )
    if compile_key not in _qknorm_rope_impl.compile_cache:
        rmsnorm_op = FusedQKNormRope(dtype, head_dim * (num_q_head + num_kv_head * 2), head_dim, num_q_head, num_kv_head, rotary_dim)
        compile_options = "--enable-tvm-ffi"
        _qknorm_rope_impl.compile_cache[compile_key] = cute.compile(
            rmsnorm_op,
            qkv_tensor,
            qnorm_weight_tensor,
            knorm_weight_tensor,
            cos_tensor,
            sin_tensor,
            pos_id_tensor,
            q_tensor,
            k_tensor,
            v_tensor,
            head_dim,
            num_q_head,
            num_kv_head,
            current_stream,
            eps,
            norm_weight_bias,
            options=compile_options,
        )

    _qknorm_rope_impl.compile_cache[compile_key](
        qkv_tensor,
        qnorm_weight_tensor,
        knorm_weight_tensor,
        cos_tensor,
        sin_tensor,
        pos_id_tensor,
        q_tensor,
        k_tensor,
        v_tensor,
        head_dim,
        num_q_head,
        num_kv_head,
        current_stream,
        eps,
        norm_weight_bias,
    )
    


_qknorm_rope_impl.compile_cache = {}

@torch.no_grad()
@benchmark_case(
    tag="small-fwd",
    description="256 tokens",
    axis_sizes={"tokens": 256, "seq": 512, "rotary_dim": 64, "head_dim": 128, "num_q_head": 64, "num_kv_head": 8, "hidden":10240},
    inputs={
        "eps": {"kind": "scalar", "value": 1e-5, "scalar_type": "float"},
        "head_dim": {"kind": "scalar", "value": 128, "scalar_type": "int"},
        "num_q_head": {"kind": "scalar", "value": 64, "scalar_type": "int"},
        "num_kv_head": {"kind": "scalar", "value": 8, "scalar_type": "int"},
        "rotary_dim": {"kind": "scalar", "value": 64, "scalar_type": "int"},
        "pos_id": {"generator": {"type": "randint", "low": 0, "high": 512}},
        "norm_weight_bias": {"kind": "scalar", "value": 1.0, "scalar_type": "float"},
    },
)
@benchmark_case(
    tag="large-fwd",
    description="8192 tokens",
    axis_sizes={"tokens": 8192, "seq": 16384, "rotary_dim": 64, "head_dim": 128, "num_q_head": 64, "num_kv_head": 8, "hidden":10240},
    inputs={
        "eps": {"kind": "scalar", "value": 1e-5, "scalar_type": "float"},
        "head_dim": {"kind": "scalar", "value": 128, "scalar_type": "int"},
        "num_q_head": {"kind": "scalar", "value": 64, "scalar_type": "int"},
        "num_kv_head": {"kind": "scalar", "value": 8, "scalar_type": "int"},
        "rotary_dim": {"kind": "scalar", "value": 64, "scalar_type": "int"},
        "pos_id": {"generator": {"type": "randint", "low": 0, "high": 8192}},
        "norm_weight_bias": {"kind": "scalar", "value": 1.0, "scalar_type": "float"},
    },
)
def fused_qknorm_rope_forward_impl(
    qkv: Float16Tensor[torch.Tensor, "tokens hidden"],
    qnorm_weight: Float16Tensor[torch.Tensor, "head_dim"],
    knorm_weight: Float16Tensor[torch.Tensor, "head_dim"],
    cos: Float16Tensor[torch.Tensor, "seq rotary_dim"],
    sin: Float16Tensor[torch.Tensor, "seq rotary_dim"],
    pos_id: IntTensor[torch.Tensor, "tokens"],
    head_dim: int,
    num_q_head: int,
    num_kv_head: int,
    rotary_dim: int,
    eps: float = 1e-5,
    norm_weight_bias: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    '''
    qkv: (tokens, (q_head+k_head+v_head)*head_dim)
    norm: (head_dim)
    cos: (seq, rotary)
    sin: (seq, rotary)
    pos_id: (tokens)
    eps: float
    norm_weight_bias: float
    return: (tokens, (q_head)*head_dim), (tokens, (k_head)*head_dim), (tokens, (v_head)*head_dim)
    '''
    assert qkv.is_cuda and qnorm_weight.is_cuda and knorm_weight.is_cuda and cos.is_cuda and sin.is_cuda and pos_id.is_cuda
    tokens = qkv.shape[0]
    device = qkv.device
    dtype = torch2cute_dtype_map[qkv.dtype]
    q = torch.empty((tokens, num_q_head * head_dim), device=device, dtype=qkv.dtype)
    k = torch.empty((tokens, num_kv_head * head_dim), device=device, dtype=qkv.dtype)
    v = torch.empty((tokens, num_kv_head * head_dim), device=device, dtype=qkv.dtype)

    _qknorm_rope_impl(
        qkv, 
        qnorm_weight, 
        knorm_weight, 
        cos, 
        sin, 
        pos_id, 
        q, 
        k, 
        v, 
        head_dim, 
        num_q_head, 
        num_kv_head,
        rotary_dim,
        eps,
        norm_weight_bias,
    )

    return q, k, v
