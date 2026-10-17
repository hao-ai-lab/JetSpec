# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# Fork: FlashInfer v0.7.2 (19faf79b70cdbe2127530784016a73c9e015e0f1),
# flashinfer/gemm/kernels/cute_dsl/low_latency_blockscaled_gemm.py.
# K1 widens the token tile, adds streamed-weight L2 policy and workspace split-K.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
CUTE DSL implementation of the Low-Latency Blackwell GEMM kernel.

This is a low-latency Blackwell GEMM kernel: C = A * B, with optional bias
- A (M, K, L) with K contiguous (UmmaMajor::K)
- B (N, K, L) with K contiguous
- C (M, N, L) with M contiguous

Features:
- TMA loads for A and B matrices (GMEM -> SMEM)
- Block-scaled matrix multiply-accumulate in TMEM
- Multi-stage pipeline for overlapping TMA and MMA
- Non-TMA-store epilogue (TMEM -> RMEM -> GMEM with direct store)
- Optional M-element bias, broadcast across N and L
- 1 SM mode (no 2CTA instructions)
- 1x1 M/N cluster with optional cluster split-K
- Programmatic Dependent Launch support

Blockscaled variant of the Low-Latency Blackwell GEMM kernel.
Default config: CTA_M=128, CTA_N=8, CTA_K=256, DMA_Stage=8
  TypeA=Float4E2M1FN, TypeB=Float4E2M1FN, TypeC=BFloat16, AccType=float
  SF=Float8E4M3FN, sf_vec_size=16
  UmmaMajorA=Major::K, UmmaMajorB=Major::K

Warp assignment (384 threads, 12 warps):
  Warp 0: DMA_A - loads A tiles and scale factors
  Warp 1: DMA_B - loads B tiles and scale factors
  Warp 2: MMA - performs the block-scaled matrix multiply
  Warps 3-6: SFB - stage B scale factors in TMEM for the register-mediated path
  Warp 7: unused
  Warps 8-11: EPILOG - copy accumulators to registers, reduce split-K, and store C
"""

from typing import NamedTuple, Optional, Tuple, Type

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import dsl_user_op
from cutlass.cute.nvgpu import cpasync, tcgen05
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
from cutlass.pipeline import (
    PipelineAsync,
    PipelineAsyncUmma,
    PipelineUmmaAsync,
    CooperativeGroup,
    Agent,
    pipeline_init_arrive,
    pipeline_init_wait,
    make_pipeline_state,
    PipelineUserType,
)



@dsl_user_op
def _l2_evict_first(*, loc=None, ip=None):
    return cutlass.Int64(llvm.inline_asm(
        cutlass.Int64.mlir_type, [],
        "createpolicy.fractional.L2::evict_first.b64 $0, 1.0;", "=l",
        has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    ))


class WorkTileInfo(NamedTuple):
    """Which output tile this CTA processes.
    For non-persistent static scheduler, CTA id is the work tile info."""

    M_idx: cutlass.Int32
    N_idx: cutlass.Int32
    L_idx: cutlass.Int32
    K_idx_start: cutlass.Int32  # kblock range [K_idx_start, K_idx_end)
    K_idx_end: cutlass.Int32


_MAX_AB_STAGES = 12
_SUPPORTED_SPLIT_K = (1, 2, 4, 8)
_SMEM_CAPACITY_BYTES = cutlass.utils.get_smem_capacity_in_bytes("sm_100")


def _align_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def _smem_bytes(
    a_dtype: Type[cutlass.Numeric],
    b_dtype: Type[cutlass.Numeric],
    sf_vec_size: int,
    cta_k: int,
    num_ab_stage: int,
    num_sfb_tmem_stage: int,
    split_k: int,
    cta_n: int = 8,
) -> int:
    """Mirror the kernel's shared-memory allocator closely enough for tactics."""
    mailbox_elems = max(split_k - 1, 0) * 128 * (cta_n // split_k)
    cursor = (
        16 * num_ab_stage
        + 16 * num_sfb_tmem_stage
        + 16  # MMA-to-epilogue barriers
        + 8  # TMEM allocation barrier
        + 8  # bias-load barrier
        + 4  # TMEM base pointer
    )
    cursor = _align_up(cursor, 8) + 8 + mailbox_elems * 4

    mixed_width = a_dtype.width != b_dtype.width
    a_smem_width = 8 if mixed_width and a_dtype.width < 8 else a_dtype.width
    b_smem_width = 8 if mixed_width and b_dtype.width < 8 else b_dtype.width
    sf_k = cta_k // sf_vec_size
    buffer_sizes = (
        128 * cta_k * a_smem_width // 8 * num_ab_stage,
        cta_n * cta_k * b_smem_width // 8 * num_ab_stage,
        128 * sf_k * num_ab_stage,
        cta_n * sf_k * num_ab_stage,
        128 * sf_k * num_ab_stage,
    )
    for size in buffer_sizes:
        cursor = _align_up(cursor, 128) + size
    return cursor


class K1Kernel:
    """Low-latency, warp-specialized Blackwell block-scaled GEMM.

    The 12 warps are assigned to two TMA load roles, one MMA role, four
    optional scale-factor producer roles, four epilogue roles, and one unused
    warp.
    """

    def __init__(
        self,
        acc_dtype: Type[cutlass.Numeric] = cutlass.Float32,
        mma_tiler_mnk: tuple[int, int, int] = (128, 8, 256),
        num_ab_stage: int = 8,
        num_sfb_tmem_stage: int = 4,
        sf_vec_size: int = 16,
        use_pdl: bool = False,
        pdl_count: int = -1,
        split_k: int = 1,
        sfb_tmem_store: bool = False,
        use_scale: bool = False,
        use_bias: bool = False,
        workspace_split: int = 1,
        k2_epilogue: int = 0,
        prologue_mode: int = 0,
        prologue_k: int = 0,
        prologue_overlap: bool = False,
        per_row_scale: bool = False,
    ):
        """Initialize the Low-Latency blockscaled GEMM kernel configuration.

        :param acc_dtype: Data type for the MMA accumulator (split-K requires Float32).
        :type acc_dtype: Type[cutlass.Numeric]
        :param mma_tiler_mnk: CTA tile shape (M, N, K).
        :type mma_tiler_mnk: tuple[int, int, int]
        :param num_ab_stage: Number of A/B SMEM pipeline stages.
        :type num_ab_stage: int
        :param num_sfb_tmem_stage: Number of staged SFB TMEM buffers.
        :type num_sfb_tmem_stage: int
        :param sf_vec_size: Scale factor vector size (16 for NVFP4, 32 for MX).
        :type sf_vec_size: int
        :param use_pdl: Enable Programmatic Dependent Launch.
        :type use_pdl: bool
        :param pdl_count: K-tile index to launch dependent grids at; -1 launches at the end.
        :type pdl_count: int
        :param split_k: Cluster split-K factor (1 = no split).
        :type split_k: int
        :param sfb_tmem_store: Use the register-mediated SFB staging path
            instead of the direct SMEM-to-TMEM copy path.
        :type sfb_tmem_store: bool
        :param use_scale: Apply one output scale in accumulator precision.
        :type use_scale: bool
        :param use_bias: Add an M-element bias in the epilogue, broadcast across N and L.
        :type use_bias: bool
        """
        if split_k > 1 and acc_dtype is not cutlass.Float32:
            raise ValueError(
                "block-scaled split-k requires acc_dtype=cutlass.Float32 because "
                "SM100 block-scaled MMA produces Float32 accumulators"
            )

        cta_m, cta_n, cta_k = mma_tiler_mnk
        self.acc_dtype = acc_dtype
        self.cta_m = cta_m
        self.cta_n = cta_n
        self.cta_k = cta_k
        self.num_ab_stage = num_ab_stage
        self.num_sfb_tmem_stage = num_sfb_tmem_stage
        self.sf_vec_size = sf_vec_size
        self.use_pdl = use_pdl
        self.pdl_count = pdl_count
        self.split_k = split_k
        self.workspace_split = workspace_split
        self.sfb_tmem_store = sfb_tmem_store
        self.use_scale = use_scale
        self.per_row_scale = per_row_scale
        self.use_bias = use_bias
        self.k2_epilogue = k2_epilogue
        self.prologue_mode = prologue_mode
        self.prologue_k = prologue_k
        self.prologue_overlap = prologue_overlap
        self.num_b_stage = prologue_k // cta_k if prologue_mode else num_ab_stage
        if k2_epilogue and (split_k != 1 or workspace_split != 1):
            raise ValueError("K2 fused epilogue requires an unsplit GEMM")

        # Size the distributed shared memory mailbox used by split-K
        self._mailbox_elems_per_thread = (cta_m * cta_n) // 128
        self._shard_elems_per_thread = self._mailbox_elems_per_thread // max(split_k, 1)
        if split_k > 1 and self._mailbox_elems_per_thread % split_k != 0:
            raise ValueError(
                f"elems_per_thread={self._mailbox_elems_per_thread} must be "
                f"divisible by split_k={split_k}"
            )
        self._mailbox_total_elems = (
            max(split_k - 1, 0) * 128 * self._shard_elems_per_thread
        )
        self._mailbox_tx_per_sender = 128 * self._shard_elems_per_thread * 4
        self._mailbox_tx_total = max(split_k - 1, 0) * self._mailbox_tx_per_sender

        # Fixed configuration
        self.threads_per_cta = 384  # 12 warps (3 active + 4 SFB + 1 unused + 4 epilog)
        self.use_2cta_instrs = False  # 1 SM mode
        self.cluster_shape_mn = (1, 1)  # No multicast, 1x1 cluster
        self.cta_group = tcgen05.CtaGroup.ONE

    def _setup_attributes(self):
        """Set up derived config."""
        mma_tiler_mn = (self.cta_m, self.cta_n)

        # Build the block-scaled MMA atom
        tiled_mma = sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype,
            self.b_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.sf_dtype,
            self.sf_vec_size,
            self.cta_group,
            mma_tiler_mn,
        )
        assert self.cta_group == tcgen05.CtaGroup.ONE

        # Number of MMA instructions along the CTA K tile
        mma_inst_shape_k = cute.size(tiled_mma.shape_mnk, mode=[2])
        self.mma_inst_tile_k = self.cta_k // mma_inst_shape_k

        # Shared SFB layout constants used by the SFB and MMA warps
        self._sf_atom_mn = 32
        self._sf_per_mma_k = mma_inst_shape_k // self.sf_vec_size
        self._num_n_atoms = 2  # two columns for register-mediated SFB staging
        self._sfb_lane_stride = 1 << 18
        self._num_sfa_tmem_cols = (
            self.cta_m // self._sf_atom_mn
        ) * self.mma_inst_tile_k
        if cutlass.const_expr(self.sfb_tmem_store):
            self._sfb_tmem_cols_per_stage = self.mma_inst_tile_k * self._num_n_atoms
        else:
            self._sfb_tmem_cols_per_stage = self.mma_inst_tile_k * 4  # standard padded

        # Keep the accumulator and staged scale factors within half of TMEM
        total_tmem_cols = (
            self.cta_n  # accumulator
            + self._num_sfa_tmem_cols  # SFA
            + self._sfb_tmem_cols_per_stage * self.num_sfb_tmem_stage  # SFB (staged)
        )
        assert total_tmem_cols <= 256, (
            f"TMEM column budget exceeded: {total_tmem_cols} > 256 "
            f"(acc={self.cta_n}, sfa={self._num_sfa_tmem_cols}, "
            f"sfb={self._sfb_tmem_cols_per_stage}*{self.num_sfb_tmem_stage}="
            f"{self._sfb_tmem_cols_per_stage * self.num_sfb_tmem_stage})"
        )

        # Map cluster ranks to (V, M, N, K); split-K varies the final coordinate
        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, self.split_k)),
            (tiled_mma.thr_id.shape,),
        )

        # Create SMEM layouts for A and B with swizzle
        # sA_layout: ((Mma_M, Mma_K), NumMma_M, NumMma_K, DMA_Stage)
        self.a_smem_layout_staged = sm100_utils.make_smem_layout_a(
            tiled_mma,
            (self.cta_m, self.cta_n, self.cta_k),
            self.smem_alloc_a_dtype,
            self.num_ab_stage,
        )
        # sB_layout: ((Mma_N, Mma_K), NumMma_N, NumMma_K, DMA_Stage)
        self.b_smem_layout_staged = sm100_utils.make_smem_layout_b(
            tiled_mma,
            (self.cta_m, self.cta_n, self.cta_k),
            self.smem_alloc_b_dtype,
            self.num_b_stage,
        )

        # SFA SMEM layout: ((Atom_M, Atom_K), MMA_M, MMA_K, STAGE)
        self.sfa_smem_layout_staged = blockscaled_utils.make_smem_layout_sfa(
            tiled_mma,
            (self.cta_m, self.cta_n, self.cta_k),
            self.sf_vec_size,
            self.num_ab_stage,
        )
        # Compact layout for register-mediated SFB staging
        sfb_n = self.cta_n
        sfb_k = self.cta_k // self.sf_vec_size
        self._sfb_smem_layout_flat = cute.make_layout(
            (sfb_n, sfb_k, self.num_ab_stage),
            stride=(sfb_k, 1, sfb_n * sfb_k),
        )
        # Padded layout for direct SFB staging
        self._mma_tiler_sfb = (self.cta_m, cute.round_up(self.cta_n, 128), self.cta_k)
        self._tiled_mma_sfb = sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype,
            self.b_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.sf_dtype,
            self.sf_vec_size,
            tcgen05.CtaGroup.ONE,
            (self.cta_m, cute.round_up(self.cta_n, 128)),
        )
        self._sfb_smem_layout_padded = blockscaled_utils.make_smem_layout_sfb(
            self._tiled_mma_sfb,
            self._mma_tiler_sfb,
            self.sf_vec_size,
            self.num_b_stage,
        )
        if cutlass.const_expr(self.sfb_tmem_store):
            self.sfb_smem_layout_staged = self._sfb_smem_layout_flat
        else:
            self.sfb_smem_layout_staged = self._sfb_smem_layout_padded

        # Accumulator shape for TMEM allocation size calculation
        acc_shape = tiled_mma.partition_shape_C((self.cta_m, self.cta_n))
        tCtAcc_fake = tiled_mma.make_fragment_C(acc_shape)
        self.num_tmem_alloc_cols = cutlass.utils.get_num_tmem_alloc_cols(tCtAcc_fake)

    @cute.jit
    def __call__(
        self,
        a_ptr: cute.Pointer,
        sfa_ptr: cute.Pointer,
        b_ptr: cute.Pointer,
        sfb_ptr: cute.Pointer,
        c_ptr: cute.Pointer,
        scale_ptr: Optional[cute.Pointer],
        bias_ptr: Optional[cute.Pointer],
        quant: Optional[Tuple[cute.Pointer, cute.Pointer, cute.Pointer, cute.Pointer]],
        prologue: Optional[Tuple[cute.Pointer, cute.Pointer, cute.Pointer, cute.Pointer, cute.Pointer]],
        problem_mnkl: Tuple,
        stream: cuda.CUstream,
    ):
        """Execute the GEMM kernel warp specialized for DMA_A, DMA_B, MMA, SFB, and EPILOG.
        - DMA_A: loads A tiles via TMA
        - DMA_B: loads B tiles via TMA
        - MMA: performs the block-scaled matrix multiply
        - SFB: stages SFB into TMEM through registers (sfb_tmem_store only)
        - EPILOG: TMEM->RMEM->GMEM store

        :param a_ptr: Pointer to the A operand.
        :type a_ptr: cute.Pointer
        :param sfa_ptr: Pointer to the SFA operand.
        :type sfa_ptr: cute.Pointer
        :param b_ptr: Pointer to the B operand.
        :type b_ptr: cute.Pointer
        :param sfb_ptr: Pointer to the SFB operand.
        :type sfb_ptr: cute.Pointer
        :param c_ptr: Pointer to the C operand.
        :type c_ptr: cute.Pointer
        :param scale_ptr: Pointer to the optional FP32 output scale.
        :type scale_ptr: Optional[cute.Pointer]
        :param bias_ptr: Pointer to the optional M-element bias, or None.
        :type bias_ptr: Optional[cute.Pointer]
        :param problem_mnkl: Problem shape (M, N, K, L).
        :type problem_mnkl: Tuple
        :param stream: CUDA stream for asynchronous execution.
        :type stream: cuda.CUstream
        """
        m, n, k, l = problem_mnkl
        a = cute.make_tensor(
            a_ptr, cute.make_ordered_layout((m, k, l), order=(1, 0, 2))
        )
        b = cute.make_tensor(
            b_ptr, cute.make_ordered_layout((n, k, l), order=(1, 0, 2))
        )
        c = cute.make_tensor(
            c_ptr, cute.make_ordered_layout((m, n, l), order=(0, 1, 2))
        )
        self.a_dtype = a_ptr.value_type
        self.b_dtype = b_ptr.value_type
        self.c_dtype = c_ptr.value_type
        if cutlass.const_expr(self.use_scale):
            assert scale_ptr is not None
            scale = cute.make_tensor(scale_ptr, cute.make_layout((1 + n if self.per_row_scale else 2,)))
        else:
            assert scale_ptr is None
            scale = None
        if cutlass.const_expr(self.use_bias):
            assert bias_ptr is not None
            bias = cute.make_tensor(
                bias_ptr,
                cute.make_layout((m, n, l), stride=(1, 0, 0)),
            )
            self.bias_dtype = bias_ptr.value_type
        else:
            assert bias_ptr is None
            bias = None
            self.bias_dtype = self.c_dtype
        self.sf_dtype = sfa_ptr.value_type
        self.mxf8f6f4 = self.needs_unpack_tma(self.a_dtype, self.b_dtype)
        self.smem_alloc_a_dtype = (
            cutlass.Int8 if self.mxf8f6f4 and self.a_dtype.width < 8 else self.a_dtype
        )
        self.smem_alloc_b_dtype = (
            cutlass.Int8 if self.mxf8f6f4 and self.b_dtype.width < 8 else self.b_dtype
        )
        self.a_major_mode = cutlass.utils.LayoutEnum.from_tensor(a).mma_major_mode()
        self.b_major_mode = cutlass.utils.LayoutEnum.from_tensor(b).mma_major_mode()
        self.c_layout = cutlass.utils.LayoutEnum.from_tensor(c)
        sfa = cute.make_tensor(
            sfa_ptr, blockscaled_utils.tile_atom_to_shape_SF(a.shape, self.sf_vec_size)
        )
        if cutlass.const_expr(self.sfb_tmem_store):
            sfb = cute.make_tensor(
                sfb_ptr,
                cute.make_ordered_layout(
                    (n, cute.assume(k // self.sf_vec_size, 16), l), order=(1, 0, 2)
                ),
            )
        else:
            # Give each CTA N tile a separate 128-row SFB atom
            n_padded_sfb = cute.ceil_div(n, self.cta_n) * cute.round_up(self.cta_n, 128)
            b_shape_for_sfb = cute.make_ordered_layout(
                (n_padded_sfb, k, l), order=(1, 0, 2)
            ).shape
            sfb = cute.make_tensor(
                sfb_ptr,
                blockscaled_utils.tile_atom_to_shape_SF(
                    b_shape_for_sfb, self.sf_vec_size
                ),
            )
        self._setup_attributes()
        tiled_mma = sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype,
            self.b_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.sf_dtype,
            self.sf_vec_size,
            self.cta_group,
            (self.cta_m, self.cta_n),
        )
        atom_thr_size = cute.size(tiled_mma.thr_id.shape)
        a_op = sm100_utils.cluster_shape_to_tma_atom_A(
            self.cluster_shape_mn, tiled_mma.thr_id
        )
        a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, None, 0))
        tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
            a_op,
            a,
            a_smem_layout,
            (self.cta_m, self.cta_n, self.cta_k),
            tiled_mma,
            self.cluster_layout_vmnk.shape,
            internal_type=self.smem_alloc_a_dtype
            if self.mxf8f6f4 and self.a_dtype.width < 8
            else None,
        )
        b_op = sm100_utils.cluster_shape_to_tma_atom_B(
            self.cluster_shape_mn, tiled_mma.thr_id
        )
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0))
        tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
            b_op,
            b,
            b_smem_layout,
            (self.cta_m, self.cta_n, self.cta_k),
            tiled_mma,
            self.cluster_layout_vmnk.shape,
            internal_type=self.smem_alloc_b_dtype
            if self.mxf8f6f4 and self.b_dtype.width < 8
            else None,
        )
        sfa_op = sm100_utils.cluster_shape_to_tma_atom_A(
            self.cluster_shape_mn, tiled_mma.thr_id
        )
        sfa_smem_layout = cute.slice_(
            self.sfa_smem_layout_staged, (None, None, None, 0)
        )
        tma_atom_sfa, tma_tensor_sfa = cute.nvgpu.make_tiled_tma_atom_A(
            sfa_op,
            sfa,
            sfa_smem_layout,
            (self.cta_m, self.cta_n, self.cta_k),
            tiled_mma,
            self.cluster_layout_vmnk.shape,
            internal_type=cutlass.Int16,
        )
        if cutlass.const_expr(self.sfb_tmem_store):
            sfb_smem_layout = cute.slice_(self._sfb_smem_layout_flat, (None, None, 0))
            sfb_k = self.cta_k // self.sf_vec_size
            sfb_tma_op = cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)
            tma_atom_sfb, tma_tensor_sfb = cpasync.make_tiled_tma_atom(
                sfb_tma_op,
                sfb,
                sfb_smem_layout,
                (self.cta_n, sfb_k),
            )
        else:
            sfb_op = sm100_utils.cluster_shape_to_tma_atom_B(
                self.cluster_shape_mn, self._tiled_mma_sfb.thr_id
            )
            sfb_smem_layout = cute.slice_(
                self._sfb_smem_layout_padded, (None, None, None, 0)
            )
            tma_atom_sfb, tma_tensor_sfb = cute.nvgpu.make_tiled_tma_atom_B(
                sfb_op,
                sfb,
                sfb_smem_layout,
                self._mma_tiler_sfb,
                self._tiled_mma_sfb,
                self.cluster_layout_vmnk.shape,
                internal_type=cutlass.Int16,
            )
        # Count input bytes before unpacking mixed-width operands
        a_copy_size = cute.size_in_bytes(self.a_dtype, a_smem_layout)
        b_copy_size = cute.size_in_bytes(self.b_dtype, b_smem_layout)
        sfa_copy_size = cute.size_in_bytes(self.sf_dtype, sfa_smem_layout)
        sfb_copy_size = cute.size_in_bytes(self.sf_dtype, sfb_smem_layout)
        self.tma_bytes_a = (a_copy_size + sfa_copy_size) * atom_thr_size
        self.tma_bytes_b = (b_copy_size + sfb_copy_size) * atom_thr_size
        grid = (
            cute.ceil_div(c.layout.shape[0], self.cta_m),
            cute.ceil_div(c.layout.shape[1], self.cta_n),
            c.layout.shape[2] * self.split_k * self.workspace_split,
        )
        self.kernel(
            tiled_mma,
            self._tiled_mma_sfb,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            tma_atom_sfa,
            tma_tensor_sfa,
            tma_atom_sfb,
            tma_tensor_sfb,
            c,
            scale,
            bias,
            quant,
            prologue,
            self.cluster_layout_vmnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.sfa_smem_layout_staged,
            self._sfb_smem_layout_flat,
            self._sfb_smem_layout_padded,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=(*self.cluster_shape_mn, self.split_k),
            stream=stream,
            use_pdl=self.use_pdl,
        )
        return

    @staticmethod
    def needs_unpack_tma(
        a_dtype: Type[cutlass.Numeric], b_dtype: Type[cutlass.Numeric]
    ) -> bool:
        """Return whether mixed-width operands use byte containers in shared memory.

        Mixed-width block-scaled MMA requires uniform one-byte containers. The
        load path therefore unpacks a sub-byte operand when A and B widths
        differ.
        """
        return a_dtype.width != b_dtype.width

    @staticmethod
    def can_implement(
        problem_sizes_mnkl: tuple[int, int, int, int],
        a_dtype: Type[cutlass.Numeric],
        b_dtype: Type[cutlass.Numeric],
        sf_dtype: Type[cutlass.Numeric],
        sf_vec_size: int,
        c_dtype: Type[cutlass.Numeric],
        mma_tiler_mnk: tuple[int, int, int | None] = (128, 8, None),
        num_ab_stage: int = 8,
        num_sfb_tmem_stage: int = 4,
        split_k: int = 1,
        sfb_tmem_store: bool = False,
    ) -> bool:
        """Return whether the kernel supports the problem and configuration."""
        operand_dtypes = {
            cutlass.Float4E2M1FN,
            cutlass.Float8E5M2,
            cutlass.Float8E4M3FN,
        }
        if a_dtype not in operand_dtypes or b_dtype not in operand_dtypes:
            return False
        if c_dtype not in {cutlass.Float32, cutlass.Float16, cutlass.BFloat16}:
            return False
        is_nvfp4 = sf_dtype is cutlass.Float8E4M3FN
        if is_nvfp4:
            if not (
                sf_vec_size == 16
                and a_dtype is cutlass.Float4E2M1FN
                and b_dtype is cutlass.Float4E2M1FN
            ):
                return False
        elif sf_dtype is cutlass.Float8E8M0FNU:
            if sf_vec_size != 32:
                return False
        else:
            return False

        if sfb_tmem_store and not is_nvfp4:
            return False

        m, n, k, l = problem_sizes_mnkl
        if min(m, n, k, l) <= 0:
            return False

        cta_m, cta_n, cta_k = mma_tiler_mnk
        mma_k = 64 if a_dtype.width == 4 and b_dtype.width == 4 else 32
        if cta_k is None:
            cta_k = 4 * mma_k
        if cta_m != 128 or cta_n not in (8, 16, 32, 64, 80) or cta_k <= 0 or k % cta_k != 0:
            return False
        if cta_k % mma_k != 0:
            return False
        # MX block-scaled TMA layouts are built from four 32-wide MMA-K atoms.
        # Smaller/non-128-multiple tiles fail the DSL's CTA V-map equivalence.
        if sf_vec_size == 32 and cta_k % 128 != 0:
            return False
        if (
            num_ab_stage <= 0
            or num_ab_stage > _MAX_AB_STAGES
            or num_sfb_tmem_stage <= 0
            or split_k <= 0
        ):
            return False

        output_values_per_thread = (cta_m * cta_n) // 128
        if cta_m * cta_n % 128 != 0:
            return False
        if output_values_per_thread % split_k != 0:
            return False

        mma_tiles_k = cta_k // mma_k
        sfa_columns = (cta_m // 32) * mma_tiles_k
        sfb_columns_per_stage = mma_tiles_k * (2 if sfb_tmem_store else 4)
        tmem_columns = cta_n + sfa_columns + sfb_columns_per_stage * num_sfb_tmem_stage
        if cta_m % 32 != 0 or tmem_columns > 256:
            return False
        return (
            _smem_bytes(
                a_dtype,
                b_dtype,
                sf_vec_size,
                cta_k,
                num_ab_stage,
                num_sfb_tmem_stage,
                split_k,
                cta_n,
            )
            <= _SMEM_CAPACITY_BYTES
        )

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tiled_mma_sfb: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,  # (Gemm_M, Gemm_K, Gemm_L) — TMA coordinate tensor for A
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,  # (Gemm_N, Gemm_K, Gemm_L) — TMA coordinate tensor for B
        tma_atom_sfa: cute.CopyAtom,
        sfa: cute.Tensor,  # SFA TMA coordinate tensor
        tma_atom_sfb: cute.CopyAtom,
        mSFB_nkl: cute.Tensor,  # SFB TMA coordinate tensor
        mC_mnl: cute.Tensor,  # (Gemm_M, Gemm_N, Gemm_L) — output tensor in GMEM
        mScale: Optional[cute.Tensor],  # single output scale
        mBias_mnl: Optional[cute.Tensor],  # (Gemm_M, Gemm_N, Gemm_L):(1,0,0)
        quant: Optional[Tuple[cute.Pointer, cute.Pointer, cute.Pointer, cute.Pointer]],
        prologue: Optional[Tuple[cute.Pointer, cute.Pointer, cute.Pointer, cute.Pointer, cute.Pointer]],
        cluster_layout_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,  # ((Mma_M, Mma_K), NumMma_M, NumMma_K, DMA_Stage)
        b_smem_layout_staged: cute.ComposedLayout,  # ((Mma_N, Mma_K), NumMma_N, NumMma_K, DMA_Stage)
        sfa_smem_layout_staged: cute.Layout,
        sfb_smem_layout_flat_staged: cute.Layout,  # compact 3D SFB layout
        sfb_smem_layout_padded_staged: cute.Layout,  # padded 4D SFB layout
    ):
        """GPU device kernel: SMEM alloc, barrier init, warp dispatch.

        384 threads, 12 warps: 0=DMA_A, 1=DMA_B, 2=MMA, 3-6=SFB
        (register-mediated path only), 7=unused, 8-11=EPILOG. Tensor
        partitioning is done inside the warp functions.
        """
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)
        tidx, _, _ = cute.arch.thread_idx()
        bidx, bidy, bidz = cute.arch.block_idx()

        # Prefetch TMA descriptors (warp 0 only)
        if warp_idx == 0:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)
            cpasync.prefetch_descriptor(tma_atom_sfa)
            cpasync.prefetch_descriptor(tma_atom_sfb)

        # Derive the split rank and batch index from the cluster topology
        _, _, split_rank = cute.arch.block_in_cluster_idx()
        _, _, l_idx = cute.arch.cluster_idx()

        # Assign this CTA its split-K range
        total_k_tiles = cute.ceil_div(cute.size(mA_mkl, mode=[1]), self.cta_k)
        mma_tile_coord_v = bidx % cute.size(tiled_mma.thr_id.shape)
        if cutlass.const_expr(self.workspace_split > 1):
            split_rank = bidz % self.workspace_split
            l_idx = bidz // self.workspace_split
            mC_mnl = cute.make_tensor(
                mC_mnl.iterator + split_rank * cute.size(mC_mnl), mC_mnl.layout
            )
        k_tiles_per_split = cute.ceil_div(total_k_tiles, self.split_k * self.workspace_split)
        k_start = split_rank * k_tiles_per_split
        if k_start > total_k_tiles:
            k_start = total_k_tiles
        k_end = k_start + k_tiles_per_split
        if k_end > total_k_tiles:
            k_end = total_k_tiles
        work_tile_info = WorkTileInfo(
            M_idx=bidx // cute.size(tiled_mma.thr_id.shape),
            N_idx=bidy,
            L_idx=l_idx,
            K_idx_start=k_start,
            K_idx_end=k_end,
        )
        k_tile_count = work_tile_info.K_idx_end - work_tile_info.K_idx_start

        cta_rank_in_cluster = cute.arch.make_warp_uniform(
            cute.arch.block_idx_in_cluster()
        )
        # tma_partition uses the M/N coordinates to build multicast masks
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(
            cta_rank_in_cluster
        )

        # SMEM allocation (SharedStorage struct for barriers + TMEM ptr)
        DMA_Stage = self.num_ab_stage
        SFB_Stage = self.num_sfb_tmem_stage

        # AB full barriers track both arrivals and TMA transaction bytes. The
        # remaining barriers use plain arrive/wait synchronization
        MailboxElems = self._mailbox_total_elems  # (split_k-1) * 128 * elems_per_thread

        @cute.struct
        class SharedStorage:
            # AB pipeline barriers: full[0..DMA_Stage-1] + empty[DMA_Stage..2*DMA_Stage-1]
            ab_pipeline_bars: cute.struct.MemRange[cutlass.Int64, DMA_Stage * 2]
            # SFB pipeline barriers (unused on the direct-copy path)
            sfb_pipeline_bars: cute.struct.MemRange[cutlass.Int64, SFB_Stage * 2]
            # MMA→epilog pipeline: full[0] + empty[1]
            mma_epilog_bars: cute.struct.MemRange[cutlass.Int64, 2]
            # Standalone barriers coordinate tensor memory and split-K reduction
            tmem_allocation_result_barrier: cutlass.Int64
            tma_epilog_full_barrier: cutlass.Int64
            tmem_base_ptr: cutlass.Int32
            dsmem_mailbox_barrier: cutlass.Int64
            dsmem_mailbox: cute.struct.MemRange[cutlass.Float32, MailboxElems]

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        sGateUp = None
        if cutlass.const_expr(self.k2_epilogue):
            sGateUp = smem.allocate_tensor(
                cutlass.BFloat16, cute.make_layout((self.cta_n, 128), stride=(128, 1)),
                byte_alignment=128)

        # Defer synchronization until every pipeline and standalone barrier is ready
        # Two TMA producer warps feed one MMA consumer warp
        ab_pipeline = PipelineAsync.create(
            barrier_storage=storage.ab_pipeline_bars.data_ptr(),
            num_stages=DMA_Stage,
            producer_group=CooperativeGroup(Agent.Thread, 1 if self.prologue_mode else 2),
            consumer_group=CooperativeGroup(Agent.Thread),
            defer_sync=True,
        )
        ab_full_bar = ab_pipeline.sync_object_full.barrier_storage
        ab_empty_bar = ab_pipeline.sync_object_empty.barrier_storage

        # One-stage MMA-to-epilogue completion pipeline
        mma_epilog_pipeline = PipelineUmmaAsync.create(
            barrier_storage=storage.mma_epilog_bars.data_ptr(),
            num_stages=1,
            producer_group=CooperativeGroup(Agent.Thread),
            consumer_group=CooperativeGroup(Agent.Thread, 128),
            defer_sync=True,
        )
        mma_epilog_full_bar = mma_epilog_pipeline.sync_object_full.barrier_storage

        # Raw barrier pointers
        tmem_alloc_result_bar = storage.tmem_allocation_result_barrier.ptr
        tma_epilog_full_bar = storage.tma_epilog_full_barrier.ptr
        tmem_base_smem_ptr = storage.tmem_base_ptr.ptr
        mailbox_mbar = storage.dsmem_mailbox_barrier.ptr
        dsmem_mailbox_ptr = storage.dsmem_mailbox.data_ptr()
        # The register-mediated path has a separate SFB pipeline
        if cutlass.const_expr(self.sfb_tmem_store):
            sfb_pipeline = PipelineAsyncUmma.create(
                barrier_storage=storage.sfb_pipeline_bars.data_ptr(),
                num_stages=SFB_Stage,
                producer_group=CooperativeGroup(Agent.Thread, 4),
                consumer_group=CooperativeGroup(Agent.Thread),
                defer_sync=True,
            )
            sfb_full_bar = sfb_pipeline.sync_object_full.barrier_storage
            sfb_empty_bar = sfb_pipeline.sync_object_empty.barrier_storage
        else:
            sfb_full_bar = storage.sfb_pipeline_bars.data_ptr()
            sfb_empty_bar = sfb_full_bar + SFB_Stage

        # Initialize standalone TMEM and split-K barriers
        if warp_idx == 0:
            with cute.arch.elect_one():
                # The TMEM barrier covers allocation and release across MMA,
                # epilogue, and the optional SFB warps
                if cutlass.const_expr(self.sfb_tmem_store):
                    cute.arch.mbarrier_init(tmem_alloc_result_bar, 32 + 128 + 128)
                else:
                    cute.arch.mbarrier_init(tmem_alloc_result_bar, 32 + 128)
                if cutlass.const_expr(self.use_bias):
                    # All 32 B-load threads arrive after issuing activation loads
                    cute.arch.mbarrier_init(tma_epilog_full_bar, 32)
                if self.split_k > 1:
                    cute.arch.mbarrier_init(mailbox_mbar, 1)

        # Publish barrier initialization to every warp
        cluster_layout = (
            cute.make_layout((*self.cluster_shape_mn, self.split_k))
            if self.split_k > 1
            else None
        )
        pipeline_init_arrive(cluster_shape_mn=cluster_layout, is_relaxed=True)
        pipeline_init_wait(cluster_shape_mn=cluster_layout)

        # SMEM Tensor Allocation
        # sA: ((Mma_M, Mma_K), NumMma_M, NumMma_K, DMA_Stage) with swizzle
        sA = smem.allocate_tensor(
            element_type=self.smem_alloc_a_dtype,
            layout=a_smem_layout_staged.outer,
            byte_alignment=128,
            swizzle=a_smem_layout_staged.inner,
        )
        # sB: ((Mma_N, Mma_K), NumMma_N, NumMma_K, DMA_Stage) with swizzle
        sB = smem.allocate_tensor(
            element_type=self.smem_alloc_b_dtype,
            layout=b_smem_layout_staged.outer,
            byte_alignment=128,
            swizzle=b_smem_layout_staged.inner,
        )
        # sSFA: scale factor SMEM (no swizzle)
        sSFA = smem.allocate_tensor(
            element_type=self.sf_dtype,
            layout=sfa_smem_layout_staged,
            byte_alignment=128,
        )
        # Allocate both SFB layouts to keep the configurations type-compatible
        sSFB_flat = smem.allocate_tensor(
            element_type=self.sf_dtype,
            layout=sfb_smem_layout_flat_staged,
            byte_alignment=128,
        )
        sSFB_padded = smem.allocate_tensor(
            element_type=self.sf_dtype,
            layout=sfb_smem_layout_padded_staged,
            byte_alignment=128,
        )
        if cutlass.const_expr(self.prologue_mode):
            row_rstd = smem.allocate_tensor(cutlass.Float32, cute.make_layout(16), byte_alignment=16)
            if cutlass.const_expr(self.prologue_overlap):
                # Warp zero can fill the weight ring while ten warps prepare
                # B. The MMA warp joins only their final publication barrier.
                if warp_idx != 0:
                    if warp_idx != 2:
                        if cutlass.const_expr(self.use_pdl):
                            cute.arch.griddepcontrol_wait()
                        prologue_tid = tidx - 32 if warp_idx == 1 else tidx - 64
                        self.activation_prologue(prologue, sB, sSFB_padded, b_smem_layout_staged.inner,
                                                 row_rstd, prologue_tid, bidx == 0 and split_rank == 0)
                    cute.arch.barrier(barrier_id=12, number_of_threads=352)
            else:
                if cutlass.const_expr(self.use_pdl):
                    cute.arch.griddepcontrol_wait()
                self.activation_prologue(prologue, sB, sSFB_padded, b_smem_layout_staged.inner, row_rstd, tidx,
                                         bidx == 0 and split_rank == 0)

        # Warp dispatch
        # Each warp specializes in a different role:
        #   Warp 0 (threads 0-31):      TMA load A
        #   Warp 1 (threads 32-63):     TMA load B
        #   Warp 2 (threads 64-95):     MMA   — performs block-scaled MMA
        #   Warps 3-6 (threads 96-223): SFB producers
        #   Warp 7 (threads 224-255):   unused
        #   Warps 8-11 (threads 256-383): epilogue
        if warp_idx == 0:
            self.dma_a_warp(
                ab_full_bar,
                ab_empty_bar,
                tma_atom_a,
                mA_mkl,
                sA,
                tma_atom_sfa,
                sfa,
                sSFA,
                tiled_mma,
                cluster_layout_vmnk,
                block_in_cluster_coord_vmnk,
                mma_tile_coord_v,
                work_tile_info,
                k_tile_count,
            )
        elif warp_idx == 1:
            if cutlass.const_expr(not self.prologue_mode):
                sSFB = sSFB_flat if cutlass.const_expr(self.sfb_tmem_store) else sSFB_padded
                self.dma_b_warp(
                    ab_full_bar,
                    ab_empty_bar,
                    tma_epilog_full_bar,
                    tma_atom_b,
                    mB_nkl,
                    sB,
                    tma_atom_sfb,
                    mSFB_nkl,
                    sSFB,
                    tiled_mma,
                    tiled_mma_sfb,
                    cluster_layout_vmnk,
                    block_in_cluster_coord_vmnk,
                    mma_tile_coord_v,
                    work_tile_info,
                    k_tile_count,
                )
        elif warp_idx == 2:
            self.mma_warp(
                ab_full_bar,
                ab_empty_bar,
                mma_epilog_full_bar,
                tmem_alloc_result_bar,
                sfb_full_bar,
                sfb_empty_bar,
                tiled_mma,
                sA,
                sB,
                sSFA,
                sSFB_padded,
                sfa_smem_layout_staged,
                sfb_smem_layout_padded_staged,
                tmem_base_smem_ptr,
                k_tile_count,
                work_tile_info.K_idx_start,
            )
        elif warp_idx >= 8:
            epi_tid = tidx - 256
            self.epilog_warp(
                tma_epilog_full_bar,
                mma_epilog_full_bar,
                tmem_alloc_result_bar,
                tiled_mma,
                mC_mnl,
                mScale,
                mBias_mnl,
                quant,
                sGateUp,
                tmem_base_smem_ptr,
                mailbox_mbar,
                dsmem_mailbox_ptr,
                epi_tid,
                mma_tile_coord_v,
                work_tile_info,
                split_rank,
                k_tile_count,
            )

        # Register-mediated path only: warps 3-6 stage SFB into TMEM (idle on direct copy)
        if cutlass.const_expr(self.sfb_tmem_store):
            if warp_idx >= 3 and warp_idx <= 6:
                self.sfb_warp(
                    ab_full_bar,
                    sfb_full_bar,
                    sfb_empty_bar,
                    tmem_alloc_result_bar,
                    tmem_base_smem_ptr,
                    sSFB_flat,
                    k_tile_count,
                )

        # Final sync
        cute.arch.barrier()
        return

    @cute.jit
    def dma_a_warp(
        self,
        ab_full_bar,
        ab_empty_bar,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,  # (Gemm_M, Gemm_K, Gemm_L) — TMA coordinate tensor
        sA: cute.Tensor,  # ((Mma_M, Mma_K), NumMma_M, NumMma_K, DMA_Stage)
        tma_atom_sfa: cute.CopyAtom,
        mSFA_mkl: cute.Tensor,
        sSFA: cute.Tensor,
        tiled_mma: cute.TiledMma,
        cluster_layout_vmnk: cute.Layout,
        block_in_cluster_coord_vmnk: Tuple,
        mma_tile_coord_v: cutlass.Int32,
        work_tile_info: WorkTileInfo,
        k_tile_count: cutlass.Int32,
    ):
        """DMA_A warp: loads A tiles via TMA."""
        DMA_Stage = self.num_ab_stage

        # Tile mA_mkl (Gemm_M, Gemm_K, Gemm_L) into CTA-level tiles
        # gA_mkl: (CTA_M, CTA_K, Tiles_M, Tiles_K, Gemm_L) — all tiles
        gA_mkl = cute.local_tile(mA_mkl, (self.cta_m, self.cta_k), (None, None, None))
        # In 1SM mode, tiled_mma has one partition
        thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
        # tCgA: ((Mma_M, Mma_K), NumMma_M, NumMma_K, Tiles_M, Tiles_K, Gemm_L)
        tCgA = thr_mma.partition_A(gA_mkl)

        # A tiles multicast along the cluster N dimension. group_modes folds the
        # MMA modes into one TMA transfer; a 1x1 cluster reduces to a local copy
        a_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape
        )
        tAsA, tAgA = cpasync.tma_partition(
            tma_atom_a,
            block_in_cluster_coord_vmnk[2],  # this CTA's N coord within cluster
            a_cta_layout,
            cute.group_modes(
                sA, 0, 3
            ),  # (((Mma_M, Mma_K), NumMma_M, NumMma_K), DMA_Stage)
            cute.group_modes(
                tCgA, 0, 3
            ),  # (((Mma_M, Mma_K), NumMma_M, NumMma_K), Tiles_M, Tiles_K, Gemm_L)
        )
        # tAsA: ((TMA, NumTma_K), DMA_Stage) — SMEM destination for each pipeline stage
        # tAgA: ((TMA, NumTma_K), Tiles_M, Tiles_K, Gemm_L) — GMEM source tiles

        # Slice to this CTA's M tile and batch index, keep K tiles and TMA modes free
        # tAgA after slice: ((TMA, NumTma_K), Tiles_K)
        tAgA = tAgA[(None, work_tile_info.M_idx, None, work_tile_info.L_idx)]

        # SFA partition (same cta_layout as A — SFA multicasts along N)
        gSFA_mkl = cute.local_tile(
            mSFA_mkl, (self.cta_m, self.cta_k), (None, None, None)
        )
        tCgSFA = thr_mma.partition_A(gSFA_mkl)
        tAsSFA, tAgSFA = cpasync.tma_partition(
            tma_atom_sfa,
            block_in_cluster_coord_vmnk[2],
            a_cta_layout,
            cute.group_modes(sSFA, 0, 3),
            cute.group_modes(tCgSFA, 0, 3),
        )
        tAsSFA = cute.filter_zeros(tAsSFA)
        tAgSFA = cute.filter_zeros(tAgSFA)
        tAgSFA = tAgSFA[(None, work_tile_info.M_idx, None, work_tile_info.L_idx)]

        # Empty barriers start at phase 0, so the producer starts at phase 1
        ab_state = make_pipeline_state(PipelineUserType.Producer, DMA_Stage)
        pdl_count = self.pdl_count
        weight_policy = _l2_evict_first()

        for k_tile in cutlass.range(k_tile_count, unroll=1):
            # wait for MMA to signal SMEM slot empty (auto phase tracking)
            cute.arch.mbarrier_wait(ab_empty_bar + ab_state.index, ab_state.phase)

            # set transaction bytes on full barrier
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(
                    ab_full_bar + ab_state.index, self.tma_bytes_a
                )

            # tma_partition elects the issuer; another election around cute.copy
            # can prevent participating threads from making progress
            k_tile_global = k_tile + work_tile_info.K_idx_start
            cute.copy(
                tma_atom_a,
                tAgA[(None, k_tile_global)],
                tAsA[(None, ab_state.index)],
                tma_bar_ptr=ab_full_bar + ab_state.index,
                cache_policy=weight_policy,
            )
            cute.copy(
                tma_atom_sfa,
                tAgSFA[(None, k_tile_global)],
                tAsSFA[(None, ab_state.index)],
                tma_bar_ptr=ab_full_bar + ab_state.index,
                cache_policy=weight_policy,
            )

            ab_state.advance()  # update phase and index/stage

            # PDL: launch dependents at the computed k_tile, or unconditionally at end
            if self.use_pdl:
                if k_tile == pdl_count:
                    cute.arch.griddepcontrol_launch_dependents()

        if self.use_pdl:
            cute.arch.griddepcontrol_launch_dependents()

        # Producer tail: keep shared memory alive until MMA releases every stage
        for _k_tile in cutlass.range(DMA_Stage, unroll=1):
            cute.arch.mbarrier_wait(ab_empty_bar + ab_state.index, ab_state.phase)
            ab_state.advance()

    @cute.jit
    def dma_b_warp(
        self,
        ab_full_bar,
        ab_empty_bar,
        tma_epilog_full_bar,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,  # (Gemm_N, Gemm_K, Gemm_L) — TMA coordinate tensor
        sB: cute.Tensor,  # ((Mma_N, Mma_K), NumMma_N, NumMma_K, DMA_Stage)
        tma_atom_sfb: cute.CopyAtom,
        mSFB_nkl: cute.Tensor,  # SFB TMA coordinate tensor
        sSFB: cute.Tensor,  # SFB SMEM tensor (flat or padded, matching the path)
        tiled_mma: cute.TiledMma,
        tiled_mma_sfb: cute.TiledMma,  # only used by the direct-copy (padded) SFB path
        cluster_layout_vmnk: cute.Layout,
        block_in_cluster_coord_vmnk: Tuple,
        mma_tile_coord_v: cutlass.Int32,
        work_tile_info: WorkTileInfo,
        k_tile_count: cutlass.Int32,
    ):
        """DMA_B warp: loads B tiles + SFB scale factors via TMA.

        Only the SFB TMA partition differs between the two SFB paths.
        """
        DMA_Stage = self.num_ab_stage

        # Tile mB_nkl (Gemm_N, Gemm_K, Gemm_L) into CTA-level tiles
        # gB_nkl: (CTA_N, CTA_K, Tiles_N, Tiles_K, Gemm_L)
        gB_nkl = cute.local_tile(mB_nkl, (self.cta_n, self.cta_k), (None, None, None))
        thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
        # tCgB: ((Mma_N, Mma_K), NumMma_N, NumMma_K, Tiles_N, Tiles_K, Gemm_L)
        tCgB = thr_mma.partition_B(gB_nkl)

        # B tiles multicast along the cluster M dimension
        b_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape
        )
        tBsB, tBgB = cpasync.tma_partition(
            tma_atom_b,
            block_in_cluster_coord_vmnk[1],  # this CTA's M coord within cluster
            b_cta_layout,
            cute.group_modes(sB, 0, 3),
            cute.group_modes(tCgB, 0, 3),
        )
        # tBsB: ((TMA, NumTma_K), DMA_Stage)
        # tBgB: ((TMA, NumTma_K), Tiles_N, Tiles_K, Gemm_L)

        # Slice to this CTA's N tile and batch index
        tBgB = tBgB[(None, work_tile_info.N_idx, None, work_tile_info.L_idx)]

        if cutlass.const_expr(self.sfb_tmem_store):
            # Register-mediated path: flat packed SFB layout, partitioned directly
            sfb_k = self.cta_k // self.sf_vec_size
            gSFB_nkl = cute.local_tile(
                mSFB_nkl, (self.cta_n, sfb_k), (None, None, None)
            )
            tBsSFB, tBgSFB = cpasync.tma_partition(
                tma_atom_sfb,
                0,
                cute.make_layout(1),
                cute.group_modes(sSFB, 0, 2),
                cute.group_modes(gSFB_nkl, 0, 2),
            )
        else:
            # Direct-copy path: padded SFB partitioned like B via tiled_mma_sfb
            n_padded = cute.round_up(self.cta_n, 128)
            gSFB_nkl = cute.local_tile(
                mSFB_nkl, (n_padded, self.cta_k), (None, None, None)
            )
            thr_mma_sfb = tiled_mma_sfb.get_slice(mma_tile_coord_v)
            tCgSFB = thr_mma_sfb.partition_B(gSFB_nkl)
            tBsSFB, tBgSFB = cpasync.tma_partition(
                tma_atom_sfb,
                block_in_cluster_coord_vmnk[1],
                b_cta_layout,
                cute.group_modes(sSFB, 0, 3),
                cute.group_modes(tCgSFB, 0, 3),
            )
            tBsSFB = cute.filter_zeros(tBsSFB)
            tBgSFB = cute.filter_zeros(tBgSFB)
        tBgSFB = tBgSFB[(None, work_tile_info.N_idx, None, work_tile_info.L_idx)]

        # PDL: wait on dependent grids only for B (the activation tensor)
        if self.use_pdl:
            cute.arch.griddepcontrol_wait()

        ab_state = make_pipeline_state(PipelineUserType.Producer, DMA_Stage)

        for k_tile in cutlass.range(k_tile_count, unroll=1):
            cute.arch.mbarrier_wait(ab_empty_bar + ab_state.index, ab_state.phase)
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(
                    ab_full_bar + ab_state.index, self.tma_bytes_b
                )
            k_tile_global = k_tile + work_tile_info.K_idx_start
            cute.copy(
                tma_atom_b,
                tBgB[(None, k_tile_global)],
                tBsB[(None, ab_state.index)],
                tma_bar_ptr=ab_full_bar + ab_state.index,
            )
            cute.copy(
                tma_atom_sfb,
                tBgSFB[(None, k_tile_global)],
                tBsSFB[(None, ab_state.index)],
                tma_bar_ptr=ab_full_bar + ab_state.index,
            )
            ab_state.advance()

        # Release bias loads after the B-load warp satisfies the PDL dependency
        if cutlass.const_expr(self.use_bias):
            cute.arch.mbarrier_arrive(tma_epilog_full_bar)

        # Producer tail: keep shared memory alive until MMA releases every stage
        for _k_tile in cutlass.range(DMA_Stage, unroll=1):
            cute.arch.mbarrier_wait(ab_empty_bar + ab_state.index, ab_state.phase)
            ab_state.advance()

    def mainloop_s2t_copy_and_partition(
        self,
        sSF: cute.Tensor,
        tSF: cute.Tensor,
    ) -> Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
        """Make tiled copy for SMEM→TMEM load of scale factors, then partition."""
        tCsSF_compact = cute.filter_zeros(sSF)
        tCtSF_compact = cute.filter_zeros(tSF)
        copy_atom_s2t = cute.make_copy_atom(
            tcgen05.Cp4x32x128bOp(self.cta_group),
            self.sf_dtype,
        )
        tiled_copy_s2t = tcgen05.make_s2t_copy(copy_atom_s2t, tCtSF_compact)
        thr_copy_s2t = tiled_copy_s2t.get_slice(0)
        tCsSF_compact_s2t_ = thr_copy_s2t.partition_S(tCsSF_compact)
        tCsSF_compact_s2t = tcgen05.get_s2t_smem_desc_tensor(
            tiled_copy_s2t, tCsSF_compact_s2t_
        )
        tCtSF_compact_s2t = thr_copy_s2t.partition_D(tCtSF_compact)
        return tiled_copy_s2t, tCsSF_compact_s2t, tCtSF_compact_s2t

    @cute.jit
    def sfb_warp(
        self,
        ab_full_bar,
        sfb_full_bar,
        sfb_empty_bar,
        tmem_alloc_result_bar,
        tmem_base_smem_ptr,
        sSFB: cute.Tensor,
        k_tile_count: cutlass.Int32,
    ):
        """Move one SFB subpartition through registers into tensor memory."""
        DMA_Stage = self.num_ab_stage
        SFB_Stage = self.num_sfb_tmem_stage
        # Join the TMEM allocation barrier before reading the base address
        cute.arch.mbarrier_arrive(tmem_alloc_result_bar)
        cute.arch.mbarrier_wait(tmem_alloc_result_bar, 0)

        # Offset this subpartition past the accumulator and SFA storage
        tmem_ptr = cute.arch.retrieve_tmem_ptr(
            self.acc_dtype,
            16,
            tmem_base_smem_ptr,
        )
        sfb_tmem_ptr = cute.recast_ptr(
            tmem_ptr + self._num_sfa_tmem_cols + self.cta_n, dtype=self.sf_dtype
        )

        # staged TMEM layout: add SFB_Stage as outermost mode
        sfb_kblock_stride = self._num_n_atoms * self._sf_per_mma_k
        sfb_stage_stride = self.mma_inst_tile_k * sfb_kblock_stride
        sp_tmem_staged = cute.make_tensor(
            sfb_tmem_ptr,
            cute.make_layout(
                (
                    (self._sf_atom_mn, (1, self._sf_per_mma_k)),
                    1,
                    self.mma_inst_tile_k,
                    SFB_Stage,
                ),
                stride=(
                    (self._sfb_lane_stride, (0, 1)),
                    0,
                    sfb_kblock_stride,
                    sfb_stage_stride,
                ),
            ),
        )

        # single-stage slice for copy plan (layout same across stages)
        sp_tmem_single = sp_tmem_staged[(None, None, None, 0)]
        sp_tmem_compact_single = cute.filter_zeros(sp_tmem_single)
        # staged compact for destination partitioning
        sp_tmem_compact_staged = cute.filter_zeros(sp_tmem_staged)

        tidx_in_warp = cute.arch.lane_idx()
        tmem_store_atom = cute.make_copy_atom(
            tcgen05.St32x32bOp(tcgen05.Repetition(1)),
            self.sf_dtype,
        )
        tmem_store_tiled = tcgen05.make_tmem_copy(
            tmem_store_atom, sp_tmem_compact_single
        )
        tmem_store_thr = tmem_store_tiled.get_slice(tidx_in_warp)
        tStDst_staged = tmem_store_thr.partition_D(sp_tmem_compact_staged)
        tStSrc = tmem_store_thr.partition_S(sp_tmem_compact_single)
        rSFB = cute.make_rmem_tensor(tStSrc.shape, self.sf_dtype)

        # Observe the AB pipeline and produce into the SFB pipeline
        ab_observer = make_pipeline_state(PipelineUserType.Consumer, DMA_Stage)
        sfb_state = make_pipeline_state(PipelineUserType.Producer, SFB_Stage)

        for _k_tile in cutlass.range(k_tile_count, unroll=1):
            # SFB warps observe AB completion but do not release the slot
            cute.arch.mbarrier_wait(ab_full_bar + ab_observer.index, ab_observer.phase)

            # Load SFB from shared memory into registers
            sSFB_stage = sSFB[(None, None, ab_observer.index)]
            smem_row = tidx_in_warp
            if smem_row >= self.cta_n:
                smem_row = cutlass.Int32(0)
            for i in cutlass.range(cute.size(rSFB)):
                rSFB[i] = sSFB_stage[smem_row + i * self.cta_n]

            # Wait until MMA releases this SFB stage
            cute.arch.mbarrier_wait(sfb_empty_bar + sfb_state.index, sfb_state.phase)

            # Store the register fragment in its staged tensor memory destination
            cute.copy(
                tmem_store_tiled,
                rSFB,
                tStDst_staged[(None, None, None, None, sfb_state.index)],
            )
            cute.arch.fence_view_async_tmem_store()

            # Publish the completed SFB stage to MMA
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive(sfb_full_bar + sfb_state.index)

            ab_observer.advance()
            sfb_state.advance()

        # Join the TMEM release phase
        cute.arch.mbarrier_arrive(tmem_alloc_result_bar)

    @cute.jit
    def mma_warp(
        self,
        ab_full_bar,
        ab_empty_bar,
        mma_epilog_full_bar,
        tmem_alloc_result_bar,
        sfb_full_bar,
        sfb_empty_bar,
        tiled_mma: cute.TiledMma,
        sA: cute.Tensor,  # ((Mma_M, Mma_K), NumMma_M, NumMma_K, DMA_Stage)
        sB: cute.Tensor,  # ((Mma_N, Mma_K), NumMma_N, NumMma_K, DMA_Stage)
        sSFA: cute.Tensor,
        sSFB: cute.Tensor,
        sfa_smem_layout_staged: cute.Layout,
        sfb_smem_layout_padded_staged: cute.Layout,
        tmem_base_smem_ptr,
        k_tile_count: cutlass.Int32,
        k_start: cutlass.Int32,
    ):
        """Perform block-scaled MMA with the staged scale factors."""
        DMA_Stage = self.num_ab_stage
        SFB_Stage = self.num_sfb_tmem_stage

        # A and B fragments are SMEM descriptors consumed by cute.gemm
        # tCrA: ((Mma_M, Mma_K), NumMma_M, NumMma_K, DMA_Stage) — SMEM descriptors for A
        tCrA = tiled_mma.make_fragment_A(sA)
        # tCrB: ((Mma_N, Mma_K), NumMma_N, NumMma_K, DMA_Stage) — SMEM descriptors for B
        tCrB = tiled_mma.make_fragment_B(sB)

        # Build the accumulator layout before attaching its TMEM base pointer
        # acc_shape / tCtAcc: ((Mma_M, Mma_N), NumMma_M, NumMma_N)
        acc_shape = tiled_mma.partition_shape_C((self.cta_m, self.cta_n))
        tCtAcc_fake = tiled_mma.make_fragment_C(acc_shape)

        # Reserve half of TMEM so the next CTA can overlap allocation
        num_tmem_cols = 256  # SM100_TMEM_CAPACITY_COLUMNS / 2
        cute.arch.alloc_tmem(num_tmem_cols, tmem_base_smem_ptr)

        # All 32 MMA threads publish the allocation without waiting
        cute.arch.mbarrier_arrive(tmem_alloc_result_bar)

        # Relinquish TMEM allocation lock early so that the next prefetch CTA can be launched
        cute.arch.relinquish_tmem_alloc_permit()

        # Retrieve TMEM ptr from SMEM and create the accumulator tensor view
        tmem_ptr = cute.arch.retrieve_tmem_ptr(
            self.acc_dtype,
            16,
            tmem_base_smem_ptr,
        )
        tCtAcc = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)

        # SFA/SFB TMEM tensors and S2T copy partition
        sfa_tmem_ptr = cute.recast_ptr(tmem_ptr + self.cta_n, dtype=self.sf_dtype)
        sfa_mem_single = cute.slice_(sfa_smem_layout_staged, (None, None, None, 0))
        tCtSFA_layout = blockscaled_utils.make_tmem_layout_sfa(
            tiled_mma, tiled_mma.shape_mnk, self.sf_vec_size, sfa_mem_single
        )
        tCtSFA = cute.make_tensor(sfa_tmem_ptr, tCtSFA_layout)

        sfb_tmem_ptr = cute.recast_ptr(
            tmem_ptr + self._num_sfa_tmem_cols + self.cta_n, dtype=self.sf_dtype
        )

        # SFA: SMEM-to-TMEM copy
        (
            tiled_copy_s2t_sfa,
            tCsSFA_compact_s2t,
            tCtSFA_compact_s2t,
        ) = self.mainloop_s2t_copy_and_partition(sSFA, tCtSFA)

        # MMA mainloop
        if cutlass.const_expr(self.sfb_tmem_store):
            # Register-mediated staging uses two N atoms per SFB subpartition
            sfb_kblock_stride = self._num_n_atoms * self._sf_per_mma_k
            sfb_stage_stride = self.mma_inst_tile_k * sfb_kblock_stride
            tCtSFB_layout_staged = cute.make_layout(
                (
                    (
                        (
                            (self._sf_atom_mn, self._num_n_atoms),
                            128 // self._sf_atom_mn,
                        ),
                        (self.sf_vec_size, self._sf_per_mma_k),
                    ),
                    1,
                    self.mma_inst_tile_k,
                    SFB_Stage,
                ),
                stride=(
                    (
                        (
                            (self._sfb_lane_stride, self._sf_per_mma_k),
                            self._sf_atom_mn * self._sfb_lane_stride,
                        ),
                        (0, 1),
                    ),
                    0,
                    sfb_kblock_stride,
                    sfb_stage_stride,
                ),
            )
            tCtSFB_staged = cute.make_tensor(sfb_tmem_ptr, tCtSFB_layout_staged)

            # Register-mediated SFB mainloop
            ab_state = make_pipeline_state(PipelineUserType.Consumer, DMA_Stage)
            sfb_state = make_pipeline_state(PipelineUserType.Consumer, SFB_Stage)
            smem_wait_done = cutlass.Boolean(False)
            sfb_wait_done = cutlass.Boolean(False)

            for k_tile in cutlass.range(k_tile_count):
                if ~smem_wait_done:
                    cute.arch.mbarrier_wait(
                        ab_full_bar + ab_state.index, ab_state.phase
                    )

                old_ab_index = ab_state.index
                ab_state.advance()

                old_sfb_index = sfb_state.index
                old_sfb_phase = sfb_state.phase
                sfb_state.advance()

                smem_wait_done = cutlass.Boolean(False)
                if k_tile < (k_tile_count - 1):
                    smem_wait_done = cute.arch.mbarrier_try_wait(
                        ab_full_bar + ab_state.index, ab_state.phase
                    )

                # Copy SFA from shared memory to tensor memory
                s2t_stage_coord = (None, None, None, None, old_ab_index)
                cute.copy(
                    tiled_copy_s2t_sfa,
                    tCsSFA_compact_s2t[s2t_stage_coord],
                    tCtSFA_compact_s2t,
                )

                # Wait for the SFB warps to finish this TMEM stage
                if ~sfb_wait_done:
                    cute.arch.mbarrier_wait(sfb_full_bar + old_sfb_index, old_sfb_phase)

                # Select the staged SFB tensor used by this GEMM
                tCtSFB_mma = tCtSFB_staged[(None, None, None, old_sfb_index)]

                # A*SFA * B*SFB -> Acc
                tiled_mma.set(tcgen05.Field.ACCUMULATE, k_tile != 0)
                tile_crd = (None, None, None, old_ab_index)
                cute.gemm(
                    tiled_mma,
                    tCtAcc,
                    [tCrA[tile_crd], tCtSFA],
                    [tCrB[tile_crd], tCtSFB_mma],
                    tCtAcc,
                )

                with cute.arch.elect_one():
                    # Release the SFB and operand pipeline stages
                    tcgen05.commit(sfb_empty_bar + old_sfb_index, None, self.cta_group)
                    tcgen05.commit(ab_empty_bar + old_ab_index, None, self.cta_group)

                sfb_wait_done = cutlass.Boolean(False)
                if k_tile < (k_tile_count - 1):
                    sfb_wait_done = cute.arch.mbarrier_try_wait(
                        sfb_full_bar + sfb_state.index, sfb_state.phase
                    )
        else:
            # Direct staging reuses one standard SFB tensor memory stage
            sfb_smem_single = cute.slice_(
                sfb_smem_layout_padded_staged, (None, None, None, 0)
            )
            tCtSFB_layout = blockscaled_utils.make_tmem_layout_sfb(
                tiled_mma, tiled_mma.shape_mnk, self.sf_vec_size, sfb_smem_single
            )
            tCtSFB = cute.make_tensor(sfb_tmem_ptr, tCtSFB_layout)

            # Configure the direct SFB copy from shared to tensor memory
            (
                tiled_copy_s2t_sfb,
                tCsSFB_compact_s2t,
                tCtSFB_compact_s2t,
            ) = self.mainloop_s2t_copy_and_partition(sSFB, tCtSFB)

            # Direct SFB staging mainloop
            ab_state = make_pipeline_state(PipelineUserType.Consumer, DMA_Stage)
            smem_wait_done = cutlass.Boolean(False)

            for k_tile in cutlass.range(k_tile_count):
                if ~smem_wait_done:
                    cute.arch.mbarrier_wait(
                        ab_full_bar + ab_state.index, ab_state.phase
                    )

                old_ab_index = ab_state.index
                ab_state.advance()

                smem_wait_done = cutlass.Boolean(False)
                if k_tile < (k_tile_count - 1):
                    smem_wait_done = cute.arch.mbarrier_try_wait(
                        ab_full_bar + ab_state.index, ab_state.phase
                    )

                # Copy SFA from shared memory to tensor memory
                s2t_stage_coord = (None, None, None, None, old_ab_index)
                cute.copy(
                    tiled_copy_s2t_sfa,
                    tCsSFA_compact_s2t[s2t_stage_coord],
                    tCtSFA_compact_s2t,
                )

                # Copy SFB directly from shared memory to tensor memory
                b_index = k_tile + k_start if cutlass.const_expr(self.prologue_mode) else old_ab_index
                cute.copy(
                    tiled_copy_s2t_sfb,
                    tCsSFB_compact_s2t[(None, None, None, None, b_index)],
                    tCtSFB_compact_s2t,
                )

                tiled_mma.set(tcgen05.Field.ACCUMULATE, k_tile != 0)
                tile_crd = (None, None, None, old_ab_index)
                cute.gemm(
                    tiled_mma,
                    tCtAcc,
                    [tCrA[tile_crd], tCtSFA],
                    [tCrB[(None, None, None, b_index)], tCtSFB],
                    tCtAcc,
                )

                with cute.arch.elect_one():
                    tcgen05.commit(ab_empty_bar + old_ab_index, None, self.cta_group)

        # Signal that the accumulator is ready for the epilogue
        with cute.arch.elect_one():
            tcgen05.commit(mma_epilog_full_bar, None, self.cta_group)

        # Release the tensor memory allocation
        cute.arch.mbarrier_arrive(tmem_alloc_result_bar)
        cute.arch.mbarrier_wait(tmem_alloc_result_bar, 1)
        cute.arch.dealloc_tmem(tmem_ptr, num_tmem_cols)

    @cute.jit
    def epilog_warp(
        self,
        tma_epilog_full_bar,
        mma_epilog_full_bar,
        tmem_alloc_result_bar,
        tiled_mma: cute.TiledMma,
        mC_mnl: cute.Tensor,  # (Gemm_M, Gemm_N, Gemm_L) — output tensor in GMEM
        mScale: Optional[cute.Tensor],  # single output scale
        mBias_mnl: Optional[cute.Tensor],  # (Gemm_M, Gemm_N, Gemm_L):(1,0,0)
        quant: Optional[Tuple[cute.Pointer, cute.Pointer, cute.Pointer, cute.Pointer]],
        sGateUp: Optional[cute.Tensor],
        tmem_base_smem_ptr,
        mailbox_mbar,  # local transaction barrier signaled by peer CTAs
        dsmem_mailbox_ptr,  # SMEM pointer for DSMEM stores and reads
        epi_tid: cutlass.Int32,  # thread id within epilog warps (0-127)
        mma_tile_coord_v: cutlass.Int32,
        work_tile_info: WorkTileInfo,
        split_rank: cutlass.Int32,
        k_tile_count: cutlass.Int32,
    ):
        """EPILOG warp: TMEM -> RMEM -> type convert -> GMEM."""

        # Get this CTA's output tile
        # gC_mnl: (CTA_M, CTA_N, Tiles_M, Tiles_N, Gemm_L)
        gC_mnl = cute.local_tile(mC_mnl, (self.cta_m, self.cta_n), (None, None, None))
        thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
        # tCgC: ((Mma_M, Mma_N), NumMma_M, NumMma_N, Tiles_M, Tiles_N, Gemm_L)
        tCgC = thr_mma.partition_C(gC_mnl)

        # Recreate the MMA accumulator layout; attach its TMEM base pointer below
        # acc_shape / tCtAcc_fake: ((Mma_M, Mma_N), NumMma_M, NumMma_N)
        acc_shape = tiled_mma.partition_shape_C((self.cta_m, self.cta_n))
        tCtAcc_fake = tiled_mma.make_fragment_C(acc_shape)

        # All 128 epilogue threads arrive, then wait for MMA to publish TMEM
        cute.arch.mbarrier_arrive(tmem_alloc_result_bar)
        cute.arch.mbarrier_wait(tmem_alloc_result_bar, 0)

        # Update TMEM base ptr of the accumulator tensor view
        tmem_ptr = cute.arch.retrieve_tmem_ptr(
            self.acc_dtype,
            16,
            tmem_base_smem_ptr,
        )
        # tCtAcc: ((Mma_M, Mma_N), NumMma_M, NumMma_N) — TMEM accumulator view
        tCtAcc = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)

        # Select the tensor-to-register copy atom for this CTA
        copy_atom_t2r = sm100_utils.get_tmem_load_op(
            (self.cta_m, self.cta_n, self.cta_k),
            self.c_layout,
            self.c_dtype,
            self.acc_dtype,
            (self.cta_m, self.cta_n),
            self.use_2cta_instrs,
        )

        tiled_copy_t2r = tcgen05.make_tmem_copy(
            copy_atom_t2r, tCtAcc[((None, None), 0, 0)]
        )
        # Epilogue tid is 0-127 (threads 256-383 offset by 256)
        thr_copy_t2r = tiled_copy_t2r.get_slice(epi_tid)

        # tD describes the per-thread tensor-to-register partition
        # tDtAcc: per-subpartition view of the accumulator tensor (TMEM source)
        # Shape: (CpyS, NumCpy_M, NumCpy_N)
        tDtAcc = thr_copy_t2r.partition_S(tCtAcc[((None, None), 0, 0)])

        # Partition C before slicing so its full shape can size the register fragment
        # Shape: (CpyD, NumCpy_M, NumCpy_N, Tiles_M, Tiles_N, Gemm_L)
        # DSL tCgC has extra grid dims; partition with them, create rmem before slicing
        tDgC = thr_copy_t2r.partition_D(tCgC[((None, None), 0, 0, None, None, None)])

        # Allocate register fragments before slicing; slicing changes the rank
        # Shape: (CpyD, NumCpy_M, NumCpy_N)
        epi_frag_shape = tDgC[(None, None, None, 0, 0, 0)].shape
        # tDrAcc: (CpyD, NumCpy_M, NumCpy_N) — per-thread accumulator in rmem (AccType)
        tDrAcc = cute.make_rmem_tensor(epi_frag_shape, self.acc_dtype)
        # tDrC: (CpyD, NumCpy_M, NumCpy_N) — per-thread converted output in rmem (TypeC)
        tDrC = cute.make_rmem_tensor(epi_frag_shape, self.c_dtype)

        # Slice gmem partition to this CTA's output tile
        # tDgC after slice: (CpyD, NumCpy_M, NumCpy_N) — per-thread GMEM store destinations
        tDgC = tDgC[
            (
                None,
                None,
                None,
                work_tile_info.M_idx,
                work_tile_info.N_idx,
                work_tile_info.L_idx,
            )
        ]

        # Direct stores need explicit bounds checks; TMA loads handle OOB coordinates
        coordC = cute.make_identity_tensor(mC_mnl.shape)  # (M, N, L) -> (m, n, l)
        # Create the local tile of coordC, same tiling as gC
        gCcoord = cute.local_tile(coordC, (self.cta_m, self.cta_n), (None, None, None))
        # tCcC: ((Mma_M, Mma_N), NumMma_M, NumMma_N, Tiles_M, Tiles_N, Gemm_L)
        tCcC = thr_mma.partition_C(gCcoord)
        # tDcC: (CpyD, NumCpy_M, NumCpy_N, Tiles_M, Tiles_N, Gemm_L)
        # — per-thread coordinate tensor, same shape as tDgC but payload is (m,n,l) coord
        tDcC = thr_copy_t2r.partition_D(tCcC[((None, None), 0, 0, None, None, None)])
        # Slice to this CTA's tile: (CpyD, NumCpy_M, NumCpy_N)
        tDcC = tDcC[
            (
                None,
                None,
                None,
                work_tile_info.M_idx,
                work_tile_info.N_idx,
                work_tile_info.L_idx,
            )
        ]
        # Construct predicate tensor: compare each coordinate with problem shape (M, N, L)
        # tDpredC(t) = true if coordinate is in-bounds, false if out-of-bounds
        tDpredC = cute.make_rmem_tensor(
            tDcC.shape, cutlass.Boolean
        )  # (CpyD, NumCpy_M, NumCpy_N)
        for i in range(cute.size(tDpredC)):
            tDpredC[i] = cute.elem_less(tDcC[i], mC_mnl.shape)

        if cutlass.const_expr(self.use_scale and not self.per_row_scale):
            scale_value = (mScale[0] * mScale[1]).to(self.acc_dtype)

        # Use C's register mapping for the M-broadcast bias
        if cutlass.const_expr(self.use_bias):
            gBias_mnl = cute.local_tile(
                mBias_mnl, (self.cta_m, self.cta_n), (None, None, None)
            )
            tCgBias = thr_mma.partition_C(gBias_mnl)
            tDgBias = thr_copy_t2r.partition_D(
                tCgBias[((None, None), 0, 0, None, None, None)]
            )
            tDgBias = tDgBias[
                (
                    None,
                    None,
                    None,
                    work_tile_info.M_idx,
                    work_tile_info.N_idx,
                    work_tile_info.L_idx,
                )
            ]
            tDrBias = cute.make_rmem_tensor(epi_frag_shape, self.bias_dtype)
            tDrBiasAcc = cute.make_rmem_tensor(epi_frag_shape, self.acc_dtype)

            # Wait for the B-load warp to satisfy the bias producer dependency
            cute.arch.mbarrier_wait(tma_epilog_full_bar, 0)
            cute.basic_copy_if(tDpredC, tDgBias, tDrBias)
            tDrBiasAcc.store(tDrBias.load().to(self.acc_dtype))

        # Wait for MMA to finish (PipelineUmmaAsync consumer)
        mma_epilog_state = make_pipeline_state(PipelineUserType.Consumer, 1)
        cute.arch.mbarrier_wait(
            mma_epilog_full_bar + mma_epilog_state.index, mma_epilog_state.phase
        )

        # Copy the accumulator to the 128 epilogue threads' registers
        cute.copy(tiled_copy_t2r, tDtAcc, tDrAcc)

        # Fence asynchronous TMEM reads before releasing the allocation
        cute.arch.fence_view_async_tmem_load()

        # Empty K partitions contribute zero to the split-K reduction
        if cutlass.const_expr(self.split_k > 1):
            if k_tile_count == 0:
                tDrAcc.fill(0.0)

        cute.arch.mbarrier_arrive(tmem_alloc_result_bar)

        # Type conversion and GMEM store (with optional DSMEM split-k reduction)
        elems_per_thread = self._mailbox_elems_per_thread
        if self.split_k == 1:
            # Apply the scale in accumulator precision before output conversion
            if cutlass.const_expr(self.use_scale):
                if cutlass.const_expr(self.per_row_scale):
                    for i in range(cute.size(tDrAcc)):
                        if tDpredC[i]:
                            tDrAcc[i] = tDrAcc[i] * (mScale[0] * mScale[1 + tDcC[i][1]])
                else:
                    tDrAcc.store(tDrAcc.load() * scale_value)
            # Add the bias in accumulator precision before output conversion
            if cutlass.const_expr(self.use_bias):
                tDrAcc.store(tDrAcc.load() + tDrBiasAcc.load())
            acc_vec = tDrAcc.load().to(self.c_dtype)
            tDrC.store(acc_vec)
            if cutlass.const_expr(self.k2_epilogue):
                for i in cutlass.range_constexpr(cute.size(tDrC)):
                    if tDpredC[i]:
                        sGateUp[tDcC[i][1] % self.cta_n, tDcC[i][0] % 128] = tDrC[i]
                cute.arch.barrier(barrier_id=14, number_of_threads=128)
                self.quantize_gate_up(sGateUp, quant, epi_tid, work_tile_info.M_idx,
                                      mC_mnl.shape[0] // 2, mC_mnl.shape[1])
            else:
                cute.basic_copy_if(tDpredC, tDrC, tDgC)
        else:
            # Each CTA reduces and writes one output shard
            shard_ept = self._shard_elems_per_thread

            # Arm the local mailbox before peer stores can arrive
            if epi_tid == 0:
                cute.arch.mbarrier_arrive_and_expect_tx(
                    mailbox_mbar, self._mailbox_tx_total
                )

            # Complete tensor memory loads before the distributed scatter
            cute.arch.barrier(barrier_id=15, number_of_threads=128)

            for peer in range(self.split_k):
                if peer != split_rank:
                    # Elements in peer's shard: [peer*shard_ept, (peer+1)*shard_ept)
                    shard_start = peer * shard_ept
                    # Skip the peer's own rank when selecting its sender slot
                    sender_idx = split_rank - (
                        1 if split_rank > cutlass.Int32(peer) else 0
                    )
                    mailbox_base = sender_idx * 128 * shard_ept
                    for i in range(shard_ept):
                        cute.arch.store_async_dsmem(
                            dsmem_mailbox_ptr + mailbox_base + i * 128 + epi_tid,
                            tDrAcc[shard_start + i].bitcast(cutlass.Int32),
                            mailbox_mbar,
                            cutlass.Int32(peer),
                        )

            cute.arch.mbarrier_wait(mailbox_mbar, 0)

            my_shard_start = split_rank * shard_ept
            for s in range(self.split_k - 1):
                mailbox_base = s * 128 * shard_ept
                for i in range(shard_ept):
                    tDrAcc[my_shard_start + i] = (
                        tDrAcc[my_shard_start + i]
                        + (dsmem_mailbox_ptr + mailbox_base + i * 128 + epi_tid).load()
                    )

            # Each output belongs to exactly one shard, so scale and bias apply once.
            if cutlass.const_expr(self.use_scale):
                if cutlass.const_expr(self.per_row_scale):
                    for i in range(cute.size(tDrAcc)):
                        if tDpredC[i]:
                            tDrAcc[i] = tDrAcc[i] * (mScale[0] * mScale[1 + tDcC[i][1]])
                else:
                    tDrAcc.store(tDrAcc.load() * scale_value)
            if cutlass.const_expr(self.use_bias):
                tDrAcc.store(tDrAcc.load() + tDrBiasAcc.load())

            for i in range(elems_per_thread):
                if i < my_shard_start or i >= my_shard_start + shard_ept:
                    tDpredC[i] = cutlass.Boolean(False)

            acc_vec = tDrAcc.load().to(self.c_dtype)
            tDrC.store(acc_vec)
            cute.basic_copy_if(tDpredC, tDrC, tDgC)



    @cute.jit
    def activation_prologue(self, inputs: Tuple[cute.Pointer, cute.Pointer, cute.Pointer, cute.Pointer, cute.Pointer],
                            sB: cute.Tensor, sSF: cute.Tensor, b_swizzle: cute.Swizzle, rstd: cute.Tensor,
                            tid: cutlass.Int32, write_residual: cutlass.Boolean):
        """Prepare CTA-local B; attempt 2 overlaps this with weight DMA."""
        from flashinfer.cute_dsl.fp4_common import (
            cvt_f32_to_e4m3, fp8_e4m3_to_f32_and_rcp, quantize_and_pack_16,
            rcp_approx_ftz, fmin_f32, fmax_f32, fabs_f32,
            get_smem_ptr_as_int32, st_shared_i32,
        )
        from flashinfer.quantization.quantization_cute_dsl_utils import _nvfp4_standard_quant_from_amax
        x, residual, gamma, updated, multiplier = inputs
        k = self.prologue_k
        threads = 320 if cutlass.const_expr(self.prologue_overlap) else 384
        barrier_id = 13 if cutlass.const_expr(self.prologue_overlap) else 0
        sf_raw = cute.recast_ptr(sSF.iterator, dtype=cutlass.Uint8)
        # Pointer.toint returns the unswizzled absolute SMEM byte address.
        # Match MMA/TMA's swizzle including the allocation's base phase.
        physical_address = cute.make_composed_layout(b_swizzle, 0, cute.make_layout(1 << 20))
        # Padding scales must be initialized for the 128-row TMEM scale atom.
        for i in cutlass.range(tid, cute.cosize(sSF.layout), threads):
            sf_raw[i] = cutlass.Uint8(0)
        cute.arch.barrier(barrier_id=barrier_id, number_of_threads=threads)
        if cutlass.const_expr(self.prologue_mode == 2):
            lane = tid % 32
            warp = tid // 32
            for row_step in cutlass.range_constexpr(2):
                norm_row = warp + row_step * (threads // 32)
                if norm_row < 16:
                    total = cutlass.Float32(0)
                    for j in cutlass.range_constexpr(k // 32):
                        norm_column = lane + j * 32
                        norm_h = x[norm_row * k + norm_column].to(cutlass.Float32) + residual[norm_row * k + norm_column].to(cutlass.Float32)
                        total += norm_h * norm_h
                    total = cute.arch.warp_reduction_sum(total)
                    if lane == 0:
                        rstd[norm_row] = cute.math.rsqrt(total / k + cutlass.Float32(1e-6), fastmath=True)
            cute.arch.barrier(barrier_id=barrier_id, number_of_threads=threads)
        atom_n = cute.size(sB, mode=[0, 0])
        atom_k = cute.size(sB, mode=[0, 1])
        for index in cutlass.range(tid, 16 * (k // 16), threads):
            row = index // (k // 16)
            block = index % (k // 16)
            values = cute.make_rmem_tensor((16,), cutlass.Float32)
            maximum = cutlass.Float32(0)
            for i in cutlass.range_constexpr(16):
                column = block * 16 + i
                value = x[row * k + column].to(cutlass.Float32)
                if cutlass.const_expr(self.prologue_mode == 2):
                    h = value + residual[row * k + column].to(cutlass.Float32)
                    rounded = h.to(cutlass.BFloat16)
                    if write_residual:
                        updated[row * k + column] = rounded
                    value = rounded.to(cutlass.Float32) * rstd[row] * gamma[column].to(cutlass.Float32)
                values[i] = value
                maximum = fmax_f32(maximum, fabs_f32(value))
            if cutlass.const_expr(self.prologue_mode == 2):
                raw_scale = fmin_f32(multiplier[0] * maximum * rcp_approx_ftz(cutlass.Float32(6)), cutlass.Float32(448))
                scale_bits = cvt_f32_to_e4m3(raw_scale)
                scale = cutlass.Uint8(scale_bits & cutlass.Uint32(255))
                inverse = fp8_e4m3_to_f32_and_rcp(scale_bits) * multiplier[0]
            else:
                scale, inverse = _nvfp4_standard_quant_from_amax(maximum, multiplier[0], False)
            packed = quantize_and_pack_16(values, inverse)
            stage = block // (self.cta_k // 16)
            column = (block * 16) % self.cta_k
            offset = sB.layout(((row % atom_n, column % atom_k), row // atom_n, column // atom_k, stage))
            address = physical_address(get_smem_ptr_as_int32(sB, offset))
            st_shared_i32(address, packed.to(cutlass.Int32))
            st_shared_i32(address + 4, (packed >> cutlass.Uint64(32)).to(cutlass.Int32))
            local_block = block % (self.cta_k // 16)
            sf_offset = stage * 128 * (self.cta_k // 16) + (local_block // 4) * 512 + row * 16 + local_block % 4
            sf_raw[sf_offset] = scale
        cute.arch.fence_proxy("async.shared", space="cta")
        cute.arch.barrier(barrier_id=barrier_id, number_of_threads=threads)

    @cute.jit
    def quantize_gate_up(self, shared: cute.Tensor, quant: Tuple[cute.Pointer, cute.Pointer, cute.Pointer, cute.Pointer],
                         tid: cutlass.Int32, tile: cutlass.Int32,
                         columns: cutlass.Int32, rows: cutlass.Int32):
        # Structural attempt 1: one thread owns each 16-value output block.
        # Use the pinned vendor's register conversion helpers so BF16 SwiGLU
        # rounding, E4M3 rounding and E2M1 ties match its unfused quantizer.
        from flashinfer.quantization.quantization_cute_dsl_utils import (
            _silu_and_mul_bfloat2, _quantize_nvfp4_from_h2x8_bfloat,
        )
        words = cute.recast_ptr(shared.iterator, dtype=cutlass.Uint32)
        for step in cutlass.range_constexpr(cute.ceil_div(self.cta_n * 4, 128)):
            index = tid + step * 128
            row = index // 4
            block = index % 4
            if row < rows:
                base = row * 64 + block * 8
                values = cute.make_rmem_tensor((8,), cutlass.Uint32)
                for i in cutlass.range_constexpr(8):
                    values[i] = _silu_and_mul_bfloat2(words[base + i], words[base + 32 + i])
                sf, packed = _quantize_nvfp4_from_h2x8_bfloat(
                    values[0], values[1], values[2], values[3],
                    values[4], values[5], values[6], values[7], quant[2][0], False)
                column_block = tile * 4 + block
                quant[0][row * (columns // 16) + column_block] = packed
                address = (column_block // 4) * 512 + (row % 32) * 16 + (row // 32) * 4 + column_block % 4
                quant[1][address] = sf
                quant[3][row * (columns // 16) + column_block] = sf


@cute.kernel
def _reduce_workspace_kernel(partials: cute.Pointer, output: cute.Pointer,
                             alpha: cute.Pointer, count: cutlass.Int32,
                             splits: cutlass.Constexpr, row_columns: cutlass.Constexpr):
    tid, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()
    cute.arch.griddepcontrol_wait()
    index = bid * 256 + tid
    if index < count:
        value = cutlass.Float32(0)
        for split in cutlass.range_constexpr(splits):
            value += partials[index + split * count]
        scale_index = 1
        if cutlass.const_expr(row_columns > 0):
            scale_index = 1 + index // row_columns
        output[index] = (value * (alpha[0] * alpha[scale_index])).to(cutlass.BFloat16)
    cute.arch.griddepcontrol_launch_dependents()


@cute.jit
def reduce_workspace(partials: cute.Pointer, output: cute.Pointer,
                     alpha: cute.Pointer, count: cutlass.Int32,
                     splits: cutlass.Constexpr, row_columns: cutlass.Constexpr, stream: cuda.CUstream):
    _reduce_workspace_kernel(partials, output, alpha, count, splits, row_columns).launch(
        grid=(cute.ceil_div(count, 256), 1, 1), block=(256, 1, 1),
        stream=stream, use_pdl=True,
    )
