# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

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



import argparse
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple, Type
import math
import cuda.bindings.driver as cuda

import torch

import cutlass
import cutlass.cute as cute
import cutlass.cute.testing as testing
import cutlass.utils as utils
import cutlass.pipeline as pipeline
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
import cutlass.torch as cutlass_torch
from cutlass.cute.runtime import from_dlpack
import cutlass.utils.hopper_helpers as sm90_utils

from cutlass.cute.typing import Int32
from cutlass.cute.typing import Pointer
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm

from optimus_cutedsl.gemm_ar.autotuner import autotune, AutotuneConfig
from optimus_cutedsl.gemm_ar.gemm_config import GemmConfig, get_all_configs
from optimus_cutedsl.gemm_ar.gemm_ar_persistent import _fused_gemm_ar_persistent_impl
from optimus_cutedsl.utils import _convert_from_dlpack_cached
from functools import partial, lru_cache
import math



"""
A high-performance dense GEMM (C = allreduce(A * B + D(optional)) + E(optional)) example for the NVIDIA Hopper architecture
using CuTe DSL.
- Matrix A is MxK, A can be row-major("K") or column-major("M")
- Matrix B is NxK, B can be row-major("N") or column-major("K")
- Matrix C is MxN, C can be row-major("N") or column-major("M")
- Matrix D is MxN, D can be row-major("N") or column-major("M")
- Matrix E is MxN, E can be row-major("N") or column-major("M")

This GEMM kernel supports the following features:
    - Utilizes Tensor Memory Access (TMA) for efficient memory operations
    - Utilizes Hopper's WGMMA for matrix multiply-accumulate (MMA) operations
    - Implements TMA multicast with cluster to reduce L2 memory traffic
    - Supports multi-stage pipeline to overlap computation and memory access

This GEMM works as follows:
1. Load A and B matrices from global memory (GMEM) to shared memory (SMEM) using TMA operations.
2. Perform matrix multiply-accumulate (MMA) operations using WGMMA instruction.
3. Store results from registers (RMEM) to shared memory (SMEM), then to global memory (GMEM) with TMA operations.

Hopper WGMMA instructions operate as follows:
- Read matrix A from SMEM
- Read matrix B from SMEM
- Perform MMA operation and store the result in Accumulator(register)

To run this example:

.. code-block:: bash

    python examples/hopper/dense_gemm.py                                   \
      --mnk 8192,8192,8192 --tile_shape_mn 128,256                         \
      --cluster_shape_mn 1,1 --a_dtype Float16 --b_dtype Float16           \
      --c_dtype Float16 --acc_dtype Float32                                \
      --a_major k --b_major k --c_major n

torchrun --nproc-per-node 8 /data/step4/optimus_jit/src/optimus_cutedsl/gemm_ar_new.py --mnk  8192,4096,128 --tile_shape_mn 128,256 --cluster_shape_mn 1,1 --a_dtype Float16 --b_dtype Float16   --c_dtype Float16 --acc_dtype Float32  --a_major k --b_major k --c_major n > debug.log


The above example command compute GEMM with M=8192, N=8192, K=8192.
The Hopper WGMMA tile shape is 128x256x64 and the cluster shape
is (1,1). The input, mma accumulator and output data type are set as fp16, fp32
and fp16, respectively.

To collect performance with NCU profiler:

.. code-block:: bash

    ncu python examples/hopper/dense_gemm.py                               \
      --mnk 8192,8192,8192 --tile_shape_mn 128,256                         \
      --cluster_shape_mn 1,1 --a_dtype Float16 --b_dtype Float16           \
      --c_dtype Float16 --acc_dtype Float32                                \
      --a_major k --b_major k --c_major n

Constraints:
* Supported input data types: fp16, fp8 (e4m3fn, e5m2), int8, uint8
* For fp16 types, A and B must have the same data type
* For fp8 types, A and B can have different types (e4m3fn or e5m2)
* For 8-bit integer types, A and B can have different types (int8 or uint8)
* 8-bit types (e4m3fn, e5m2, int8, uint8) only support k-major layout
* CTA tile shape M must be 64/128
* CTA tile shape N must be 64/128/256
* Cluster shape M/N must be positive and power of 2, total cluster size <= 4
* The contiguous dimension of A/B/C tensors must be at least 16 bytes aligned,
  i.e, number of elements is a multiple of 8, 16 for Float16, and Float8, respectively.
"""


class HopperWgmmaGemmKernel:
    """
    This class implements matrix multiplication (C = A x B) with support for various data types
    and architectural features specific to Hopper GPUs.

    :param acc_dtype: Data type for accumulation during computation
    :type acc_dtype: type[cutlass.Numeric]
    :param tile_shape_mn: Shape of the CTA tile (M,N)
    :type tile_shape_mn: Tuple[int, int]
    :param cluster_shape_mn: Cluster dimensions (M,N) for parallel processing
    :type cluster_shape_mn: Tuple[int, int]

    :note: Supported A/B data types:
        - Float16
          A and B must have the same data type
        - Float8E4M3FN/Float8E5M2
          A and B can have different types (Float8E4M3FN/Float8E5M2)
          only support k-major layout
        - Int8/Uint8
          A and B can have different types (Int8/Uint8)
          only support k-major layout

    :note: Supported accumulation types:
        - Float32/Float16 (for all floating point inputs)
        - Int32 (for Int8/Uint8 inputs)

    :note: Constraints:
        - CTA tile M must be 64/128
        - CTA tile N must be 64/128/256
        - CTA tile K must be 64
        - Cluster shape M/N must be positive and power of 2, total cluster size <= 4

    Example:
        >>> gemm = HopperWgmmaGemmKernel(
        ...     acc_dtype=cutlass.Float32,
        ...     tile_shape_mn=(128, 256),
        ...     cluster_shape_mn=(1, 1)
        ... )
        >>> gemm(a_tensor, b_tensor, c_tensor, stream)
    """

    def __init__(
        self,
        acc_dtype: type[cutlass.Numeric],
        tile_shape_mn: tuple[int, int],
        cluster_shape_mn: tuple[int, int],
        rank_id: int,
        world_size: int,
    ):
        """
        Initializes the configuration for a Hopper dense GEMM kernel.

        This configuration includes data types for operands, tile shape, cluster configuration,
        and thread layout.

        :param acc_dtype: Data type for accumulation during computation
        :type acc_dtype: type[cutlass.Numeric]
        :param tile_shape_mn: Shape of the CTA tile (M,N)
        :type tile_shape_mn: Tuple[int, int]
        :param cluster_shape_mn: Cluster dimensions (M,N) for parallel processing
        :type cluster_shape_mn: Tuple[int, int]
        """

        self.acc_dtype = acc_dtype

        self.cluster_shape_mn = cluster_shape_mn
        self.mma_inst_shape_mn = None
        # K dimension is deferred in _setup_attributes
        self.tile_shape_mnk = (*tile_shape_mn, 1)
        # For large tile size, using two warp groups is preferred because using only one warp
        # group may result in register spill
        self.atom_layout_mnk = (
            (2, 1, 1)
            if self.tile_shape_mnk[0] > 64 and self.tile_shape_mnk[1] > 128
            else (1, 1, 1)
        )
        self.num_mcast_ctas_a = None
        self.num_mcast_ctas_b = None
        self.is_a_mcast = False
        self.is_b_mcast = False
        self.tiled_mma = None

        self.occupancy = 1
        self.mma_warp_groups = math.prod(self.atom_layout_mnk)
        self.num_threads_per_warp_group = 128
        self.threads_per_cta = self.mma_warp_groups * self.num_threads_per_warp_group
        self.smem_capacity = utils.get_smem_capacity_in_bytes("sm_90")

        self.ab_stage = None
        self.epi_stage = None

        self.a_smem_layout_staged = None
        self.b_smem_layout_staged = None
        self.epi_smem_layout_staged = None
        self.epi_tile = None

        self.shared_storage = None
        self.buffer_align_bytes = 1024

        self.all_reduce_sync_bar_id = 1

        self.all_reduce_sync_barrier = pipeline.NamedBarrier(
            barrier_id=self.all_reduce_sync_bar_id, 
            num_threads=self.threads_per_cta)
        self.rank_id = rank_id
        self.num_ranks = world_size
        self.moe_ld_bits = 128

    def _setup_attributes(self, fuse_moe: bool):
        """Set up configurations that are dependent on GEMM inputs

        This method configures various attributes based on the input tensor properties
        (data types, leading dimensions) and kernel settings:
        - Configuring tiled MMA
        - Computing MMA/cluster/tile shapes
        - Computing cluster layout
        - Computing multicast CTAs for A/B
        - Computing epilogue subtile
        - Setting up A/B/C stage counts in shared memory
        - Computing A/B/C shared memory layout
        """

        # check the cta tile shape
        if self.tile_shape_mnk[0] not in [64, 128]:
            raise ValueError("CTA tile shape M must be 64/128")
        if self.tile_shape_mnk[1] not in [64, 128, 256]:
            raise ValueError("CTA tile shape N must be 64/128/256")

        self.tiled_mma = sm90_utils.make_trivial_tiled_mma(
            self.a_dtype,
            self.b_dtype,
            self.a_layout.sm90_mma_major_mode(),
            self.b_layout.sm90_mma_major_mode(),
            self.acc_dtype,
            self.atom_layout_mnk,
            tiler_mn=(64, self.tile_shape_mnk[1]),
        )
        mma_inst_shape_k = cute.size(self.tiled_mma.shape_mnk, mode=[2])
        mma_inst_tile_k = 4
        self.tile_shape_mnk = (
            self.tile_shape_mnk[0],
            self.tile_shape_mnk[1],
            mma_inst_shape_k * mma_inst_tile_k,
        )

        self.cta_layout_mnk = cute.make_layout((*self.cluster_shape_mn, 1))
        self.num_mcast_ctas_a = self.cluster_shape_mn[1]
        self.num_mcast_ctas_b = self.cluster_shape_mn[0]
        self.is_a_mcast = self.num_mcast_ctas_a > 1
        self.is_b_mcast = self.num_mcast_ctas_b > 1

        is_cooperative = self.atom_layout_mnk == (2, 1, 1)
        self.epi_tile = sm90_utils.compute_tile_shape_or_override(
            self.tile_shape_mnk, self.c_dtype, is_cooperative=is_cooperative
        )

        # Compute stage before compute smem layout
        self.ab_stage, self.epi_stage = self._compute_stages(
            self.tile_shape_mnk,
            self.a_dtype,
            self.b_dtype,
            self.c_dtype,
            self.epi_tile,
            smem_capacity=self.smem_capacity,
            occupancy=self.occupancy,
            fuse_moe=fuse_moe,
        )

        (
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.epi_smem_layout_staged,
        ) = self._make_smem_layouts(
            self.tile_shape_mnk,
            self.epi_tile,
            self.a_dtype,
            self.a_layout,
            self.b_dtype,
            self.b_layout,
            self.ab_stage,
            self.c_dtype,
            self.c_layout,
            self.epi_stage,
        )

    @staticmethod
    def is_valid_dtypes(
        a_dtype: Type[cutlass.Numeric],
        b_dtype: Type[cutlass.Numeric],
        acc_dtype: Type[cutlass.Numeric],
        c_dtype: Type[cutlass.Numeric],
        a_major: str,
        b_major: str,
    ) -> bool:
        """
        Check if the dtypes are valid

        :param a_dtype: The data type of tensor A
        :type a_dtype: Type[cutlass.Numeric]
        :param b_dtype: The data type of tensor B
        :type b_dtype: Type[cutlass.Numeric]
        :param acc_dtype: The data type of the accumulator
        :type acc_dtype: Type[cutlass.Numeric]
        :param c_dtype: The data type of the output tensor
        :type c_dtype: Type[cutlass.Numeric]
        :param a_major: major mode of tensor A
        :type a_major: str
        :param b_major: major mode of tensor B
        :type b_major: str

        :return: True if the dtypes are valid, False otherwise
        :rtype: bool
        """
        is_valid = True

        valid_ab_dtypes = {
            cutlass.Float16,
            cutlass.BFloat16,
            cutlass.Float8E4M3FN,
            cutlass.Float8E5M2,
            cutlass.Uint8,
            cutlass.Int8,
        }
        if a_dtype not in valid_ab_dtypes:
            is_valid = False
        if b_dtype not in valid_ab_dtypes:
            is_valid = False

        # make sure a_dtype == b_dtype for Float16 and BFloat16
        if a_dtype.width == 16 and a_dtype != b_dtype:
            is_valid = False
        if a_dtype.width != b_dtype.width:
            is_valid = False
        if not a_dtype.is_same_kind(b_dtype):
            is_valid = False

        # for 8-bit types, this implementation only supports k-major layout
        if (a_dtype.width == 8 and a_major != "k") or (
            b_dtype.width == 8 and b_major != "k"
        ):
            is_valid = False

        # Define compatibility mapping between accumulator type and AB type
        acc_ab_compatibility = {
            cutlass.Float32: {
                cutlass.Float16,
                cutlass.BFloat16,
                cutlass.Float8E4M3FN,
                cutlass.Float8E5M2,
            },
            cutlass.Float16: {
                cutlass.Float16,
                cutlass.Float8E4M3FN,
                cutlass.Float8E5M2,
            },
            cutlass.Int32: {cutlass.Uint8, cutlass.Int8},
        }
        # Check compatibility between accumulator type and A type
        if a_dtype not in acc_ab_compatibility[acc_dtype]:
            is_valid = False

        # Define compatibility mapping between accumulator type and C type
        acc_c_compatibility = {
            cutlass.Float32: {
                cutlass.Float32,
                cutlass.Float16,
                cutlass.BFloat16,
                cutlass.Float8E4M3FN,
                cutlass.Float8E5M2,
            },
            cutlass.Float16: {
                cutlass.Float32,
                cutlass.Float16,
                cutlass.Float8E4M3FN,
                cutlass.Float8E5M2,
            },
            cutlass.Int32: {
                cutlass.Float32,
                cutlass.Float16,
                cutlass.Int32,
                cutlass.Int8,
                cutlass.Uint8,
            },
        }
        # Check compatibility between accumulator type and C type
        if c_dtype not in acc_c_compatibility[acc_dtype]:
            is_valid = False

        return is_valid


    @staticmethod
    def is_valid_tensor_alignment(
        m: int,
        n: int,
        k: int,
        ab_dtype: Type[cutlass.Numeric],
        c_dtype: Type[cutlass.Numeric],
        a_major: str,
        b_major: str,
        c_major: str,
    ) -> bool:
        """
        Check if the tensor alignment is valid

        :param m: The number of rows in the A tensor
        :type m: int
        :param n: The number of columns in the B tensor
        :type n: int
        :param k: The number of columns in the A tensor
        :type k: int
        :param ab_dtype: The data type of the A and B operands
        :type ab_dtype: Type[cutlass.Numeric]
        :param c_dtype: The data type of the output tensor
        :type c_dtype: Type[cutlass.Numeric]
        :param a_major: The major axis of the A tensor
        :type a_major: str
        :param b_major: The major axis of the B tensor
        :type b_major: str
        :param c_major: The major axis of the C tensor
        :type c_major: str

        :return: True if the problem shape is valid, False otherwise
        :rtype: bool
        """
        is_valid = True

        def check_contigous_16B_alignment(dtype, is_mode0_major, tensor_shape):
            major_mode_idx = 0 if is_mode0_major else 1
            num_major_elements = tensor_shape[major_mode_idx]
            num_contiguous_elements = 16 * 8 // dtype.width
            return num_major_elements % num_contiguous_elements == 0

        if (
            not check_contigous_16B_alignment(ab_dtype, a_major == "m", (m, k))
            or not check_contigous_16B_alignment(ab_dtype, b_major == "n", (n, k))
            or not check_contigous_16B_alignment(c_dtype, c_major == "m", (m, n))
        ):
            is_valid = False
        return is_valid

        
    @cute.jit
    def __call__(
        self,
        a: cute.Tensor,
        b: cute.Tensor,
        output: cute.Tensor,
        output_mc: cute.Tensor,
        moe_out: Optional[cute.Tensor],
        barrier_flag: cute.Tensor,
        barrier_flag_mc: cute.Tensor,
        res: Optional[cute.Tensor],
        out: cute.Tensor,
        stream: cuda.CUstream,
    ):
        """Execute the GEMM operation in steps:
        - Setup static attributes
        - Setup TMA load/store atoms and tensors
        - Compute grid size
        - Define shared storage for kernel
        - Launch the kernel synchronously

        :param a: Input tensor A
        :type a: cute.Tensor
        :param b: Input tensor B
        :type b: cute.Tensor
        :param c: Output tensor C
        :type c: cute.Tensor
        :param stream: CUDA stream for asynchronous execution
        :type stream: cuda.CUstream
        """

        # setup static attributes before smem/grid/tma computation
        self.a_dtype = a.element_type
        self.b_dtype = b.element_type
        self.c_dtype = output.element_type
        self.a_layout = utils.LayoutEnum.from_tensor(a)
        self.b_layout = utils.LayoutEnum.from_tensor(b)
        self.c_layout = utils.LayoutEnum.from_tensor(output)

        if cutlass.const_expr(self.c_dtype != cutlass.Float16 and self.c_dtype != cutlass.BFloat16):
            raise TypeError(
                f"multimem all-reduce path currently supports Float16 or BFloat16 only (got {self.c_dtype})"
            )

        if cutlass.const_expr(
            self.a_dtype.width == 16 and self.a_dtype != self.b_dtype
        ):
            raise TypeError(f"Type mismatch: {self.a_dtype} != {self.b_dtype}")
        if cutlass.const_expr(self.a_dtype.width != self.b_dtype.width):
            raise TypeError(
                f"Type width mismatch: {self.a_dtype.width} != {self.b_dtype.width}"
            )
        if cutlass.const_expr(self.a_dtype.width != 16 and self.a_dtype.width != 8):
            raise TypeError("a_dtype should be float16 or float8")

        self._setup_attributes(fuse_moe=cutlass.const_expr(moe_out is not None))

        tma_atom_a, tma_tensor_a = self._make_tma_atoms_and_tensors(
            a,
            self.a_smem_layout_staged,
            (self.tile_shape_mnk[0], self.tile_shape_mnk[2]),
            self.cluster_shape_mn[1],
        )

        tma_atom_b, tma_tensor_b = self._make_tma_atoms_and_tensors(
            b,
            self.b_smem_layout_staged,
            (self.tile_shape_mnk[1], self.tile_shape_mnk[2]),
            self.cluster_shape_mn[0],
        )

        tma_atom_c, tma_tensor_c = self._make_tma_store_atoms_and_tensors(
            output,
            self.epi_smem_layout_staged,
            self.epi_tile,
        )

        # TMA load for per-rank MoE output (same layout/stride as output).
        # We stage moe_out tiles into SMEM to avoid non-contiguous per-thread GMEM
        # access patterns from thr_mma.partition_C(gMoe) that prevent vectorized loads.
        if cutlass.const_expr(moe_out is not None):
            tma_atom_moe, tma_tensor_moe = self._make_tma_atoms_and_tensors(
                moe_out,
                self.epi_smem_layout_staged,
                self.epi_tile,
                1,
            )
        else:
            tma_atom_moe = None
            tma_tensor_moe = None

        if cutlass.const_expr(res is not None):
            tma_atom_res, tma_tensor_res = self._make_tma_atoms_and_tensors(
                res,
                self.epi_smem_layout_staged,
                self.epi_tile,
                1,
            )
        else:
            tma_atom_res = None
            tma_tensor_res = None

        grid = self._compute_grid(output, self.tile_shape_mnk, self.cluster_shape_mn)

        @cute.struct
        class SharedStorage:
            mainloop_pipeline_array_ptr: cute.struct.MemRange[
                cutlass.Int64, self.ab_stage * 2
            ]
            moe_pipeline_array_ptr: cute.struct.MemRange[
                cutlass.Int64, self.epi_stage * 2
            ]
            sA: cute.struct.Align[
                cute.struct.MemRange[
                    self.a_dtype, cute.cosize(self.a_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]
            sB: cute.struct.Align[
                cute.struct.MemRange[
                    self.b_dtype, cute.cosize(self.b_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]
            sMoe: cute.struct.Align[
                cute.struct.MemRange[
                    self.c_dtype, cute.cosize(self.epi_smem_layout_staged) if cutlass.const_expr(moe_out is not None) else self.buffer_align_bytes
                ],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage

        # Launch the kernel synchronously
        self.kernel(
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            tma_atom_c,
            tma_tensor_c,
            tma_atom_moe,
            tma_tensor_moe,
            tma_atom_res,
            tma_tensor_res,
            output,
            output_mc,
            barrier_flag,
            barrier_flag_mc,
            out,
            self.tiled_mma,
            self.cta_layout_mnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.epi_smem_layout_staged,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=(*self.cluster_shape_mn, 1),
            stream=stream,
        )
        return
    #  GPU device kernel
    @cute.kernel
    def kernel(
        self,
        tma_atom_a: cute.CopyAtom,
        mA_mk: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nk: cute.Tensor,
        tma_atom_c: cute.CopyAtom,
        mC_mn: cute.Tensor,
        tma_atom_moe: Optional[cute.CopyAtom],
        moe_out: Optional[cute.Tensor],        
        tma_atom_res: Optional[cute.CopyAtom],
        mRes: Optional[cute.Tensor],
        output: cute.Tensor,
        output_mc: cute.Tensor,
        barrier_flag: cute.Tensor,
        barrier_flag_mc: cute.Tensor,
        out: cute.Tensor,
        tiled_mma: cute.TiledMma,
        cta_layout_mnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        epi_smem_layout_staged: cute.ComposedLayout,
    ):
        """
        GPU device kernel performing the GEMM computation.

        :param tma_atom_a: TMA copy atom for A tensor
        :type tma_atom_a: cute.CopyAtom
        :param mA_mk: Input tensor A
        :type mA_mk: cute.Tensor
        :param tma_atom_b: TMA copy atom for B tensor
        :type tma_atom_b: cute.CopyAtom
        :param mB_nk: Input tensor B
        :type mB_nk: cute.Tensor
        :param tma_atom_c: TMA copy atom for C tensor
        :type tma_atom_c: cute.CopyAtom
        :param mC_mn: Output tensor C
        :type mC_mn: cute.Tensor
        :param tiled_mma: Tiled MMA object
        :type tiled_mma: cute.TiledMma
        :param cta_layout_mnk: CTA layout
        :type cta_layout_mnk: cute.Layout
        :param a_smem_layout_staged: Shared memory layout for A
        :type a_smem_layout_staged: cute.ComposedLayout
        :param b_smem_layout_staged: Shared memory layout for B
        :type b_smem_layout_staged: cute.ComposedLayout
        :param epi_smem_layout_staged: Shared memory layout for epilogue
        :type epi_smem_layout_staged: cute.ComposedLayout
        """

        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)



        # /////////////////////////////////////////////////////////////////////////////
        #  Prefetch Tma desc
        # /////////////////////////////////////////////////////////////////////////////
        if warp_idx == 0:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_a)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_b)
            if cutlass.const_expr(tma_atom_moe is not None):
                cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_moe)

        # ///////////////////////////////////////////////////////////////////////////////
        #  Get cta/warp/thread idx
        # ///////////////////////////////////////////////////////////////////////////////
        bidx, bidy, _ = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()

        cidx, cidy, _ = cute.arch.cluster_idx()
        cdimx, cdimy, _ = cute.arch.cluster_dim()

        cluster_id = cidx + cdimx * cidy

        # CTA Swizzle to promote L2 data reuse
        group_size_m = 8
        s_shape = (
            (group_size_m, cdimx // group_size_m),
            cdimy,
        )
        s_stride = ((1, cdimy * group_size_m), group_size_m)
        s_layout = cute.make_layout(s_shape, stride=s_stride)
        num_reg_cids = cute.size(s_shape)
        cid_m, cid_n = s_layout.get_flat_coord(cluster_id % num_reg_cids)

        # Deal with the tail part
        if cluster_id >= num_reg_cids:
            tail_size_m = cdimx % group_size_m
            tail_layout = cute.make_layout(
                (tail_size_m, cdimy), stride=(1, tail_size_m)
            )
            tail_cid = cluster_id - num_reg_cids
            tail_cid_m, tail_cid_n = tail_layout.get_flat_coord(tail_cid)
            cid_m = cute.size(s_shape, mode=[0]) + tail_cid_m
            cid_n = tail_cid_n

        # Get the pid from cluster id
        bidx_in_cluster = cute.arch.block_in_cluster_idx()
        pid_m = cid_m * self.cluster_shape_mn[0] + bidx_in_cluster[0]
        pid_n = cid_n * self.cluster_shape_mn[1] + bidx_in_cluster[1]

        # Bound the all-reduce (and any manual pointer arithmetic) to the *valid*
        # portion of this CTA tile. For padded CTAs introduced by cluster-aligned
        # grid rounding, this evaluates to 0 so the reduction loop becomes a no-op.
        m_total = Int32(output.shape[0])
        n_total = Int32(output.shape[1])
        tile_m_i32 = Int32(self.tile_shape_mnk[0])
        tile_n_i32 = Int32(self.tile_shape_mnk[1])
        pid_m_i32 = Int32(pid_m)
        pid_n_i32 = Int32(pid_n)
        m_offset = pid_m_i32 * tile_m_i32
        n_offset = pid_n_i32 * tile_n_i32
        valid_tile_m = cutlass.max(Int32(0), cutlass.min(tile_m_i32, m_total - m_offset))
        valid_tile_n = cutlass.max(Int32(0), cutlass.min(tile_n_i32, n_total - n_offset))

        tile_coord_mn = (pid_m, pid_n)
        cta_rank_in_cluster = cute.arch.make_warp_uniform(
            cute.arch.block_idx_in_cluster()
        )  # 0
        cluster_coord_mnk = cta_layout_mnk.get_flat_coord(cta_rank_in_cluster)

        # ///////////////////////////////////////////////////////////////////////////////
        # Get mcast mask
        # ///////////////////////////////////////////////////////////////////////////////
        a_mcast_mask = cute.make_layout_image_mask(
            cta_layout_mnk, cluster_coord_mnk, mode=1
        )
        b_mcast_mask = cute.make_layout_image_mask(
            cta_layout_mnk, cluster_coord_mnk, mode=0
        )

        a_mcast_mask = a_mcast_mask if self.is_a_mcast else 0
        b_mcast_mask = b_mcast_mask if self.is_b_mcast else 0
        a_smem_layout = cute.slice_(a_smem_layout_staged, (None, None, 0))
        b_smem_layout = cute.slice_(b_smem_layout_staged, (None, None, 0))
        tma_copy_bytes = cute.size_in_bytes(
            self.a_dtype, a_smem_layout
        ) + cute.size_in_bytes(self.b_dtype, b_smem_layout)

        # /////////////////////////////////////////////////////////////////////////////
        #  Alloc and init AB full/empty + ACC full mbar (pipeline)
        # /////////////////////////////////////////////////////////////////////////////
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        # mbar arrays
        mainloop_pipeline_array_ptr = storage.mainloop_pipeline_array_ptr.data_ptr()
        if cutlass.const_expr(moe_out is not None):
            moe_pipeline_array_ptr = storage.moe_pipeline_array_ptr.data_ptr()
        else:
            moe_pipeline_array_ptr = None

        # Threads/warps participating in this pipeline
        mainloop_pipeline_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread
        )
        # Each warp will constribute to the arrive count with the number of mcast size
        mcast_size = self.num_mcast_ctas_a + self.num_mcast_ctas_b - 1
        num_warps = self.threads_per_cta // 32
        consumer_arrive_cnt = mcast_size * num_warps
        mainloop_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, consumer_arrive_cnt
        )

        cta_layout_vmnk = cute.make_layout((1, *cta_layout_mnk.shape))
        mainloop_pipeline = pipeline.PipelineTmaAsync.create(
            barrier_storage=mainloop_pipeline_array_ptr,
            num_stages=self.ab_stage,
            producer_group=mainloop_pipeline_producer_group,
            consumer_group=mainloop_pipeline_consumer_group,
            tx_count=tma_copy_bytes,
            cta_layout_vmnk=cta_layout_vmnk,
            defer_sync=True,
        )

        if cutlass.const_expr(moe_out is not None):
            # MoE TMA load pipeline (G2S) used in the epilogue to stage moe_out tiles into SMEM.
            moe_smem_layout = cute.slice_(epi_smem_layout_staged, (None, None, 0))
            moe_tma_copy_bytes = cute.size_in_bytes(self.c_dtype, moe_smem_layout)
            moe_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
            moe_pipeline_consumer_group = pipeline.CooperativeGroup(
                pipeline.Agent.Thread, num_warps
            )
            moe_pipeline = pipeline.PipelineTmaAsync.create(
                barrier_storage=moe_pipeline_array_ptr,
                num_stages=self.epi_stage,
                producer_group=moe_pipeline_producer_group,
                consumer_group=moe_pipeline_consumer_group,
                tx_count=moe_tma_copy_bytes,
                cta_layout_vmnk=cta_layout_vmnk,
                defer_sync=True,
            )

        #  Cluster arrive after barrier init
        pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mn, is_relaxed=True)

        # ///////////////////////////////////////////////////////////////////////////////
        #  Generate smem tensor A/B
        # ///////////////////////////////////////////////////////////////////////////////
        sA = storage.sA.get_tensor(
            a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner
        )
        sB = storage.sB.get_tensor(
            b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner
        )
        sMoe = storage.sMoe.get_tensor(
            epi_smem_layout_staged.outer, swizzle=epi_smem_layout_staged.inner
        )
        sC_ptr = cute.recast_ptr(
            sA.iterator, epi_smem_layout_staged.inner, dtype=self.c_dtype
        )
        sC = cute.make_tensor(sC_ptr, epi_smem_layout_staged.outer)

        # ///////////////////////////////////////////////////////////////////////////////
        #  Local_tile partition global tensors
        # ///////////////////////////////////////////////////////////////////////////////
        # (bM, bK, RestK)
        gA_mk = cute.local_tile(
            mA_mk,
            (self.tile_shape_mnk[0], self.tile_shape_mnk[2]),
            (pid_m, None),
        )
        # (bN, bK, RestK)
        gB_nk = cute.local_tile(
            mB_nk,
            (self.tile_shape_mnk[1], self.tile_shape_mnk[2]),
            (pid_n, None),
        )
        # (bM, bN)
        gC_mn = cute.local_tile(
            mC_mn,
            (self.tile_shape_mnk[0], self.tile_shape_mnk[1]),
            tile_coord_mn,
        )
        gOut = cute.local_tile(
            out,
            (self.tile_shape_mnk[0], self.tile_shape_mnk[1]),
            tile_coord_mn,
        )
        gOut_mc = cute.local_tile(
            output_mc,
            (self.tile_shape_mnk[0], self.tile_shape_mnk[1]),
            tile_coord_mn,
        )
        if cutlass.const_expr(moe_out is not None):
            gMoe = cute.local_tile(
                moe_out,
                (self.tile_shape_mnk[0], self.tile_shape_mnk[1]),
                tile_coord_mn,
            )
        else:
            gMoe = None
        if cutlass.const_expr(mRes is not None):
            gRes = cute.local_tile(
                mRes,
                (self.tile_shape_mnk[0], self.tile_shape_mnk[1]),
                tile_coord_mn,
            )
        else:
            gRes = None

        # //////////////////////////////////////////////////////////////////////////////
        #  Partition global tensor for TiledMMA_A/B/C
        # //////////////////////////////////////////////////////////////////////////////
        warp_group_idx = cute.arch.make_warp_uniform(
            tidx // self.num_threads_per_warp_group
        )
        warp_group_thread_layout = cute.make_layout(
            self.mma_warp_groups, stride=self.num_threads_per_warp_group
        )
        thr_mma = tiled_mma.get_slice(warp_group_thread_layout(warp_group_idx))

        tCgC = thr_mma.partition_C(gC_mn)

        # //////////////////////////////////////////////////////////////////////////////
        #  Partition shared tensor for TMA load A/B
        # //////////////////////////////////////////////////////////////////////////////
        #  TMA load A partition_S/D
        a_cta_layout = cute.make_layout(cute.slice_(cta_layout_mnk, (0, None, 0)).shape)
        a_cta_crd = cluster_coord_mnk[1]
        sA_for_tma_partition = cute.group_modes(sA, 0, 2)

        gA_for_tma_partition = cute.group_modes(gA_mk, 0, 2)
      
        tAsA, tAgA_mk = cute.nvgpu.cpasync.tma_partition(
            tma_atom_a,
            a_cta_crd,
            a_cta_layout,
            sA_for_tma_partition,
            gA_for_tma_partition,
        )

        # TMA load B partition_S/D
        b_cta_layout = cute.make_layout(cute.slice_(cta_layout_mnk, (None, 0, 0)).shape)
        b_cta_crd = cluster_coord_mnk[0]
        sB_for_tma_partition = cute.group_modes(sB, 0, 2)
        gB_for_tma_partition = cute.group_modes(gB_nk, 0, 2)
        tBsB, tBgB_nk = cute.nvgpu.cpasync.tma_partition(
            tma_atom_b,
            b_cta_crd,
            b_cta_layout,
            sB_for_tma_partition,
            gB_for_tma_partition,
        )

        # //////////////////////////////////////////////////////////////////////////////
        #  Make fragments
        # //////////////////////////////////////////////////////////////////////////////
        tCsA = thr_mma.partition_A(sA)
        tCsB = thr_mma.partition_B(sB)
        tCrA = tiled_mma.make_fragment_A(tCsA) #(1, 1, 4, (1, 4))
        tCrB = tiled_mma.make_fragment_B(tCsB) #(1, 1, 4, (1, 4))

        acc_shape = tCgC.shape #((2, 2, 32), 1, 1)
        accumulators = cute.make_rmem_tensor(acc_shape, self.acc_dtype)

        # ///////////////////////////////////////////////////////////////////////////////
        #  Cluster wait
        # ///////////////////////////////////////////////////////////////////////////////
        # cluster wait for barrier init
        pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)
        # /////////////////////////////////////////////////////////////////////////////
        #  Prefetch
        # /////////////////////////////////////////////////////////////////////////////
        k_tile_cnt = cute.size(gA_mk, mode=[2])
        prefetch_k_tile_cnt = cutlass.max(cutlass.min(self.ab_stage, k_tile_cnt), 0)

        mainloop_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.ab_stage
        )
        if warp_idx == 0:
            # /////////////////////////////////////////////////////////////////////////////
            # Prefetch TMA load
            # /////////////////////////////////////////////////////////////////////////////
            for prefetch_idx in cutlass.range(prefetch_k_tile_cnt, unroll=1):
                # /////////////////////////////////////////////////////////////////////////////
                #  Wait for A/B buffers to be empty before loading into them
                #  Also sets the transaction barrier for the A/B buffers
                # /////////////////////////////////////////////////////////////////////////////
                mainloop_pipeline.producer_acquire(mainloop_producer_state)
                # /////////////////////////////////////////////////////////////////////////////
                #  Slice to global/shared memref to current k_tile
                # /////////////////////////////////////////////////////////////////////////////
                tAgA_k = tAgA_mk[(None, mainloop_producer_state.count)]
                tAsA_pipe = tAsA[(None, mainloop_producer_state.index)]

                tBgB_k = tBgB_nk[(None, mainloop_producer_state.count)]
                tBsB_pipe = tBsB[(None, mainloop_producer_state.index)]

                # /////////////////////////////////////////////////////////////////////////////
                #  TMA load A/B
                # /////////////////////////////////////////////////////////////////////////////
                cute.copy(
                    tma_atom_a,
                    tAgA_k,
                    tAsA_pipe,
                    tma_bar_ptr=mainloop_pipeline.producer_get_barrier(
                        mainloop_producer_state
                    ),
                    mcast_mask=a_mcast_mask,
                )
                cute.copy(
                    tma_atom_b,
                    tBgB_k,
                    tBsB_pipe,
                    tma_bar_ptr=mainloop_pipeline.producer_get_barrier(
                        mainloop_producer_state
                    ),
                    mcast_mask=b_mcast_mask,
                )
                # Mainloop pipeline's producer commit is a NOP
                mainloop_pipeline.producer_commit(mainloop_producer_state)
                mainloop_producer_state.advance()

        # /////////////////////////////////////////////////////////////////////////////
        #  Prologue MMAs
        # /////////////////////////////////////////////////////////////////////////////
        k_pipe_mmas = 1

        mainloop_consumer_read_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.ab_stage
        )
        mainloop_consumer_release_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.ab_stage
        )

        peek_ab_full_status = cutlass.Boolean(1)
        if mainloop_consumer_read_state.count < k_tile_cnt:
            peek_ab_full_status = mainloop_pipeline.consumer_try_wait(
                mainloop_consumer_read_state
            )

        tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, False)
        num_k_blocks = cute.size(tCrA, mode=[2])
        for k_tile in cutlass.range_constexpr(k_pipe_mmas):
            # Wait for A/B buffer to be ready
            mainloop_pipeline.consumer_wait(
                mainloop_consumer_read_state, peek_ab_full_status
            )

            cute.nvgpu.warpgroup.fence()
            for k_block_idx in cutlass.range(num_k_blocks, unroll_full=True):
                k_block_coord = (
                    None,
                    None,
                    k_block_idx,
                    mainloop_consumer_read_state.index,
                )
                tCrA_1phase = tCrA[k_block_coord]
                tCrB_1phase = tCrB[k_block_coord]

                cute.gemm(
                    tiled_mma,
                    accumulators,
                    tCrA_1phase,
                    tCrB_1phase,
                    accumulators,
                )
                tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, True)

            cute.nvgpu.warpgroup.commit_group()
            mainloop_consumer_read_state.advance()
            peek_ab_full_status = cutlass.Boolean(1)
            if mainloop_consumer_read_state.count < k_tile_cnt:
                peek_ab_full_status = mainloop_pipeline.consumer_try_wait(
                    mainloop_consumer_read_state
                )

        # /////////////////////////////////////////////////////////////////////////////
        #  MAINLOOP
        # /////////////////////////////////////////////////////////////////////////////
        for k_tile in cutlass.range(k_pipe_mmas, k_tile_cnt, 1, unroll=1):
            # /////////////////////////////////////////////////////////////////////////////
            #  Wait for TMA copies to complete
            # /////////////////////////////////////////////////////////////////////////////
            mainloop_pipeline.consumer_wait(
                mainloop_consumer_read_state, peek_ab_full_status
            )
            # /////////////////////////////////////////////////////////////////////////////
            #  WGMMA
            # /////////////////////////////////////////////////////////////////////////////
            cute.nvgpu.warpgroup.fence()
            for k_block_idx in cutlass.range(num_k_blocks, unroll_full=True):
                k_block_coord = (
                    None,
                    None,
                    k_block_idx,
                    mainloop_consumer_read_state.index,
                )
                tCrA_1phase = tCrA[k_block_coord]
                tCrB_1phase = tCrB[k_block_coord]

                cute.gemm(
                    tiled_mma,
                    accumulators,
                    tCrA_1phase,
                    tCrB_1phase,
                    accumulators,
                )

            cute.nvgpu.warpgroup.commit_group()
            # Wait on the wgmma barrier for previous k_pipe_mmas wgmmas to complete
            cute.nvgpu.warpgroup.wait_group(k_pipe_mmas)

            mainloop_pipeline.consumer_release(mainloop_consumer_release_state)

            mainloop_consumer_read_state.advance()
            mainloop_consumer_release_state.advance()

            peek_ab_full_status = cutlass.Boolean(1)
            if mainloop_consumer_read_state.count < k_tile_cnt:
                peek_ab_full_status = mainloop_pipeline.consumer_try_wait(
                    mainloop_consumer_read_state
                )
            # /////////////////////////////////////////////////////////////////////////////
            #  TMA load
            # /////////////////////////////////////////////////////////////////////////////
            if warp_idx == 0 and mainloop_producer_state.count < k_tile_cnt:
                # /////////////////////////////////////////////////////////////////////////////
                #  Wait for A/B buffers to be empty before loading into them
                #  Also sets the transaction barrier for the A/B buffers
                # /////////////////////////////////////////////////////////////////////////////
                mainloop_pipeline.producer_acquire(mainloop_producer_state)

                # /////////////////////////////////////////////////////////////////////////////
                #  Slice to global/shared memref to current k_tile
                # /////////////////////////////////////////////////////////////////////////////
                tAgA_k = tAgA_mk[(None, mainloop_producer_state.count)]
                tAsA_pipe = tAsA[(None, mainloop_producer_state.index)]

                tBgB_k = tBgB_nk[(None, mainloop_producer_state.count)]
                tBsB_pipe = tBsB[(None, mainloop_producer_state.index)]

                # /////////////////////////////////////////////////////////////////////////////
                #  TMA load A/B
                # /////////////////////////////////////////////////////////////////////////////
                cute.copy(
                    tma_atom_a,
                    tAgA_k,
                    tAsA_pipe,
                    tma_bar_ptr=mainloop_pipeline.producer_get_barrier(
                        mainloop_producer_state
                    ),
                    mcast_mask=a_mcast_mask,
                )
                cute.copy(
                    tma_atom_b,
                    tBgB_k,
                    tBsB_pipe,
                    tma_bar_ptr=mainloop_pipeline.producer_get_barrier(
                        mainloop_producer_state
                    ),
                    mcast_mask=b_mcast_mask,
                )
                # Mainloop pipeline's producer commit is a NOP
                mainloop_pipeline.producer_commit(mainloop_producer_state)
                mainloop_producer_state.advance()

        # /////////////////////////////////////////////////////////////////////////////
        #  EPILOG
        # /////////////////////////////////////////////////////////////////////////////
        cute.nvgpu.warpgroup.wait_group(0)

        if cute.size(self.cluster_shape_mn) > 1:
            # Wait for all threads in the cluster to finish, avoid early release of smem
            cute.arch.cluster_arrive()
            cute.arch.cluster_wait()
        else:
            # For cluster that has a single thread block, it might have more than one warp groups.
            # Wait for all warp groups in the thread block to finish, because smem for tensor A in
            # the mainloop is reused in the epilogue.
            cute.arch.sync_threads()

        # NOTE: We intentionally do not load moe_out directly from GMEM via
        # thr_mma.partition_C(gMoe). That per-thread access pattern is often
        # non-contiguous (e.g. stride-8), which prevents vectorized GMEM loads
        # and can fail MLIR verification (e.g. "cannot vectorized copy to 8 elements").
        # Instead, the epilogue stages moe_out tiles into SMEM via TMA and fuses
        # the add from SMEM -> RMEM.

        copy_atom_r2s = sm90_utils.sm90_get_smem_store_op(
            self.c_layout,
            elem_ty_d=self.c_dtype,
            elem_ty_acc=self.acc_dtype,
        )

        copy_atom_C = cute.make_copy_atom(
            cute.nvgpu.warp.StMatrix8x8x16bOp(
                self.c_layout.is_m_major_c(),
                4,
            ),
            self.c_dtype,
        )

        tiled_copy_C_Atom = cute.make_tiled_copy_C_atom(copy_atom_C, tiled_mma)

        tiled_copy_r2s = cute.make_tiled_copy_S(
            copy_atom_r2s,
            tiled_copy_C_Atom,
        )

        # (R2S, R2S_M, R2S_N, PIPE_D)
        thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)
        tRS_sD = thr_copy_r2s.partition_D(sC) #tensor<ptr<f16, smem, align<16>, S<2,4,3>> o (((2,4),1),1,2,(1,4)):(((1,2),0),0,16,(0,4096))>
        # (R2S, R2S_M, R2S_N)
        tRS_rAcc = tiled_copy_r2s.retile(accumulators)#((8,16),1,1):((1,8),0,0)>    8: ldmatrixx4 每条指令可以load/store的数量

        # Allocate D registers.
        rD_shape = cute.shape(thr_copy_r2s.partition_S(sC)) #(((2, 2, 2), 1), 1, 2, (1, 4))
        tRS_rD_layout = cute.make_layout(rD_shape[:3]) #(((2,2,2),1),1,2):(((1,2,4),0),0,8)
        tRS_rD = cute.make_rmem_tensor_like(tRS_rD_layout, self.acc_dtype)
        size_tRS_rD = cute.size(tRS_rD) #16

        sepi_for_tma_partition = cute.group_modes(sC, 0, 2)
        tCgC_for_tma_partition = cute.zipped_divide(gC_mn, self.epi_tile)
        bSG_sD, bSG_gD = cute.nvgpu.cpasync.tma_partition(
            tma_atom_c,
            0,
            cute.make_layout(1),
            sepi_for_tma_partition,
            tCgC_for_tma_partition,
        )

        # Partition for MoE G2S TMA loads (same tiling as C epilogue tiles).
        if cutlass.const_expr(moe_out is not None): 
            sepi_moe_for_tma_partition = cute.group_modes(sMoe, 0, 2)
            tCgMoe_for_tma_partition = cute.zipped_divide(gMoe, self.epi_tile)
            bLG_sMoe, bLG_gMoe = cute.nvgpu.cpasync.tma_partition(
                tma_atom_moe,
                0,
                cute.make_layout(1),
                sepi_moe_for_tma_partition,
                tCgMoe_for_tma_partition,
            )
        else:
            bLG_sMoe = None
            bLG_gMoe = None

        epi_tile_num = cute.size(tCgC_for_tma_partition, mode=[1]) # 8
        epi_tile_shape = tCgC_for_tma_partition.shape[1] #(1, 8)
        epi_tile_layout = cute.make_layout(  # (1,8):(8,1)
            epi_tile_shape, stride=(epi_tile_shape[1], 1)
        )

        # Initialize tma store c_pipeline
        c_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, self.threads_per_cta
        )

        c_pipeline = pipeline.PipelineTmaStore.create(
            num_stages=self.epi_stage,
            producer_group=c_producer_group,
            # consumer_group=c_consumer_group,
        )

        if warp_idx == 0:
            c_pipeline.producer_acquire()  # 第一次先 acquire

        # Prefetch initial MoE epilogue tiles into SMEM.
        if cutlass.const_expr(moe_out is not None):
            moe_prefetch_cnt = cutlass.max(
                cutlass.min(self.epi_stage, epi_tile_num), 0
            )
            moe_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.epi_stage
            )
            moe_consumer_read_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.epi_stage
            )
            moe_consumer_release_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.epi_stage
            )
            if warp_idx == 0:
                for prefetch_idx in cutlass.range(moe_prefetch_cnt, unroll=1):
                    moe_pipeline.producer_acquire(moe_producer_state)
                    gmem_coord = epi_tile_layout.get_hier_coord(moe_producer_state.count)
                    cute.copy(
                        tma_atom_moe,
                        bLG_gMoe[(None, gmem_coord)],
                        bLG_sMoe[(None, moe_producer_state.index)],
                        tma_bar_ptr=moe_pipeline.producer_get_barrier(moe_producer_state),
                    )
                    moe_pipeline.producer_commit(moe_producer_state)
                    moe_producer_state.advance()

            peek_moe_full_status = cutlass.Boolean(1)
            if moe_consumer_read_state.count < epi_tile_num:
                peek_moe_full_status = moe_pipeline.consumer_try_wait(
                    moe_consumer_read_state
                )

            # SMEM -> RMEM loader for staged moe_out tiles.
            # SMEM -> RMEM load for staged moe_out tiles.
            #
            # We prefer warp-level `ldmatrix` loads when the user asks for 128b+ copies.
            # This avoids MLIR "cannot vectorized copy to 8 elements" failures caused by
            # insufficiently provable per-thread alignment on swizzled SMEM layouts, while
            # still giving a wide, high-throughput SMEM read path.
            if cutlass.const_expr(self.moe_ld_bits >= 128):
                copy_atom_moe_s2r = cute.make_copy_atom(
                    cute.nvgpu.warp.LdMatrix8x8x16bOp(self.c_layout.is_m_major_c(), 4),
                    self.c_dtype,
                )
                tiled_copy_moe_s2r = cute.make_tiled_copy_C(copy_atom_moe_s2r, tiled_mma)
                thr_copy_moe_s2r = tiled_copy_moe_s2r.get_slice(tidx)
                tSR_sMoe = thr_copy_moe_s2r.partition_S(sMoe)
                tRS_rMoe = cute.make_rmem_tensor_like(tRS_rD_layout, self.c_dtype)
                tRS_rMoe_view = thr_copy_moe_s2r.retile(tRS_rMoe)
            else:
                copy_atom_moe_s2r = cute.make_copy_atom(
                    cute.nvgpu.CopyUniversalOp(),
                    self.c_dtype,
                    num_bits_per_copy=self.moe_ld_bits,
                )
                # Partition the *source* (sMoe) with the same tiling used for C, then load
                # into a per-thread RMEM fragment.
                tiled_copy_moe_s2r = cute.make_tiled_copy_S(copy_atom_moe_s2r, tiled_copy_C_Atom)
                thr_copy_moe_s2r = tiled_copy_moe_s2r.get_slice(tidx)
                tSR_sMoe = thr_copy_moe_s2r.partition_S(sMoe)
                tRS_rMoe = cute.make_rmem_tensor_like(tRS_rD_layout, self.c_dtype)
                tRS_rMoe_view = tRS_rMoe

        for epi_idx in cutlass.range_constexpr(epi_tile_num): #8

            # Copy from accumulators to D registers
            for epi_v in cutlass.range_constexpr(size_tRS_rD): #16
                tRS_rD[epi_v] = tRS_rAcc[epi_idx * size_tRS_rD + epi_v]

            if cutlass.const_expr(moe_out is not None):

                # Wait for the corresponding MoE tile to be available in SMEM and load it
                # into registers, then fuse the add in registers before type conversion.
                moe_pipeline.consumer_wait(moe_consumer_read_state, peek_moe_full_status)
                cute.copy(
                    tiled_copy_moe_s2r,
                    tSR_sMoe[(None, None, None, moe_consumer_read_state.index)],
                    tRS_rMoe_view,
                )

                acc_vec = tRS_rD.load()
                moe_vec = tRS_rMoe.load()
                tRS_rD.store(acc_vec + moe_vec.to(self.acc_dtype))

            # Type conversion
            tRS_rD_out = cute.make_rmem_tensor_like(tRS_rD_layout, self.c_dtype)
            tRS_rD_out.store(tRS_rD.load().to(self.c_dtype))

            # Copy from D registers to shared memory
            epi_buffer = epi_idx % cute.size(tRS_sD, mode=[3])
            cute.copy(
                tiled_copy_r2s, tRS_rD_out, tRS_sD[(None, None, None, epi_buffer)]
            )

            cute.arch.fence_proxy(
                cute.arch.ProxyKind.async_shared,
                space=cute.arch.SharedSpace.shared_cta,
            )
            # barrier for sync
            pipeline.sync(barrier_id=1)

            if cutlass.const_expr(moe_out is not None):
                moe_pipeline.consumer_release(moe_consumer_release_state)
                moe_consumer_read_state.advance()
                moe_consumer_release_state.advance()
                peek_moe_full_status = cutlass.Boolean(1)
                if moe_consumer_read_state.count < epi_tile_num:
                    peek_moe_full_status = moe_pipeline.consumer_try_wait(
                        moe_consumer_read_state
                    )
                if warp_idx == 0 and moe_producer_state.count < epi_tile_num:
                    moe_pipeline.producer_acquire(moe_producer_state)
                    gmem_coord_next = epi_tile_layout.get_hier_coord(moe_producer_state.count)
                    cute.copy(
                        tma_atom_moe,
                        bLG_gMoe[(None, gmem_coord_next)],
                        bLG_sMoe[(None, moe_producer_state.index)],
                        tma_bar_ptr=moe_pipeline.producer_get_barrier(moe_producer_state),
                    )
                    moe_pipeline.producer_commit(moe_producer_state)
                    moe_producer_state.advance()

            gmem_coord = epi_tile_layout.get_hier_coord(epi_idx)

            # Copy from shared memory to global memory
            if warp_idx == 0:
                cute.copy(
                    tma_atom_c,
                    bSG_sD[(None, epi_buffer)],
                    bSG_gD[(None, gmem_coord)],
                )
                c_pipeline.producer_commit()
                c_pipeline.producer_acquire()

            pipeline.sync(barrier_id=1)


        if warp_idx == 0:
            c_pipeline.producer_tail()

        pipeline.sync(barrier_id=1)

        # Ensure all threads reach this point after store completion
        self.all_reduce_sync_barrier.arrive_and_wait()

        gridx, gridy, _ = cute.arch.grid_dim()
        tile_linear = Int32(bidx + gridx * bidy)

        cute.arch.sync_threads()

        if warp_idx == 0:
            with cute.arch.elect_one():
                # Ensure the (async) TMA stores to C are ordered before we publish
                # the "tile ready" flag to peer ranks.
                #
                # Note: TMA stores are issued via cp.async.bulk.*.write; completion is
                # drained by the epilogue TMA-store pipeline tail. This fence provides
                # a system-scope ordering point before the subsequent release atomic.
                cute.arch.fence_acq_rel_sys()

                # Notify all ranks that this CTA tile is ready.
                #
                # Use a multicast pointer so a single multimem.red updates the
                # symmetric barrier flag on every rank (instead of N separate
                # remote red.add operations).
                utils.distributed.multimem_red_add1(
                    lock_ptr=barrier_flag_mc.iterator + tile_linear,
                    scope="sys",
                    order="release",
                )

                # Wait for all other ranks to notify this CTA tile
                utils.distributed.spin_lock_atom_cas_relaxed_wait(
                    lock_ptr=barrier_flag.iterator + tile_linear,
                    expected_val=self.num_ranks,
                    reset_val=0,
                    scope="sys",
                )

                # The spin-wait uses relaxed atomics; add an explicit system-scope
                # fence so subsequent peer loads observe data published-before the
                # release increments.
                cute.arch.fence_acq_rel_sys()

        # Ensure the whole CTA starts the reduction together
        self.all_reduce_sync_barrier.arrive_and_wait()

        # Multimem all-reduce: use NVSHMEM multicast pointer and multimem.ld_reduce.
        #
        # Support fp16 and bf16 (16-bit element types).

        # gOut_mc is the per-CTA output tile, so it's 2D.
        tile_m, tile_n = gOut_mc.shape
        stride_m, stride_n = gOut_mc.stride

        elems_per_chunk = Int32(8)  # 8x16-bit = 16 bytes
        total_elems = valid_tile_m * valid_tile_n
        total_chunks = total_elems // elems_per_chunk

        # Partitioned all-reduce + multicast store:
        # Split the tile into `num_ranks` disjoint chunk ranges. Each rank only
        # reduces its own range (1/num_ranks of the work) and writes the reduced
        # 16B payload back via a multicast pointer so every rank's symmetric
        # allocation is updated.
        num_ranks_i32 = Int32(self.num_ranks)
        rank_id_i32 = Int32(self.rank_id)
        chunks_per_rank = (total_chunks + num_ranks_i32 - Int32(1)) // num_ranks_i32
        rank_chunk_begin = rank_id_i32 * chunks_per_rank
        rank_chunk_end = cutlass.min(total_chunks, rank_chunk_begin + chunks_per_rank)

        chunk = Int32(tidx) + rank_chunk_begin
        stride = Int32(self.threads_per_cta)
        supported_n_contig = stride_n == 1
        supported_m_contig = stride_m == 1
        while chunk < rank_chunk_end:
            elem = chunk * elems_per_chunk
            if supported_n_contig:
                # Row-major / N-major: contiguous along N.
                m = elem // Int32(valid_tile_n)
                n = elem - m * Int32(valid_tile_n)
                offset = m * Int32(stride_m) + n
                mc_ptr = gOut_mc.iterator + offset
                if cutlass.const_expr(self.c_dtype == cutlass.Float16):
                    x, y, z, w = utils.distributed.multimem_ld_reduce_8xf16(
                        mc_ptr)
                else:
                    x, y, z, w = utils.distributed.multimem_ld_reduce_8xbf16(
                        mc_ptr)
                utils.distributed.multimem_st_4xb32(mc_ptr, x, y, z, w)
            elif supported_m_contig:
                # Column-major / M-major: contiguous along M.
                m = elem - (elem // Int32(valid_tile_m)) * Int32(valid_tile_m)
                n = elem // Int32(valid_tile_m)
                offset = m + n * Int32(stride_n)
                mc_ptr = gOut_mc.iterator + offset
                if cutlass.const_expr(self.c_dtype == cutlass.Float16):
                    x, y, z, w = utils.distributed.multimem_ld_reduce_8xf16(
                        mc_ptr)
                else:
                    x, y, z, w = utils.distributed.multimem_ld_reduce_8xbf16(
                        mc_ptr)
                utils.distributed.multimem_st_4xb32(mc_ptr, x, y, z, w)

            chunk += stride

        # Ensure all threads in this CTA finished issuing multimem ops before the
        # cross-rank completion barrier.
        cute.arch.sync_threads()

        # # # Cross-rank completion barrier for the partitioned reduction:
        # # #
        # # # Each rank only writes 1/num_ranks of chunks. Chunks owned by other ranks
        # # # are produced via remote multimem stores into our local symmetric C buffer.
        # # # We must not read from `mC_mn` / `gC_mn` until every rank finished its
        # # # chunk-range stores.
        # # if warp_idx == 0:
        # #     with cute.arch.elect_one():
        # #         cute.arch.fence_acq_rel_sys()
        # #         utils.distributed.multimem_red_add1(
        # #             lock_ptr=barrier_flag_mc.iterator + tile_linear,
        # #             scope="sys",
        # #             order="release",
        # #         )
        # #         utils.distributed.spin_lock_atom_cas_relaxed_wait(
        # #             lock_ptr=barrier_flag.iterator + tile_linear,
        # #             expected_val=self.num_ranks,
        # #             reset_val=0,
        # #             scope="sys",
        # #         )
        # #         cute.arch.fence_acq_rel_sys()

        # # # Ensure the whole CTA observes the completed all-reduce before reading gC.
        # # self.all_reduce_sync_barrier.arrive_and_wait()
        # # cute.arch.sync_threads()

        # # # # Write final output tile: out = allreduce(output) + res(optional).
        # # # #
        # # # # Important: the partitioned all-reduce writes the reduced payload back into
        # # # # the symmetric multicast allocation `output_mc` (i.e. `gOut_mc`), where
        # # # # chunks owned by other ranks arrive via remote multimem stores. Some stride
        # # # # values on `gC_mn` / `gOut` / `gRes` can be symbolic (e.g. `1@1`) and cannot
        # # # # be converted to Int32; to avoid this, reuse the linear offset computed for
        # # # # `gOut_mc` and apply it to `out`/`res` under the contract that they have the
        # # # # same shape/stride as GEMM C.
        # # # supported_n_contig_out = stride_n == 1
        # # # supported_m_contig_out = stride_m == 1

        # # chunk = Int32(tidx)
        # # stride = Int32(self.threads_per_cta)
        # # while chunk < total_chunks:
        # #     elem = chunk * elems_per_chunk
        # #     offset = 0
        # #     if supported_n_contig:
        # #         m = elem // Int32(valid_tile_n)
        # #         n = elem - m * Int32(valid_tile_n)
        # #         offset = m * Int32(stride_m) + n
        # #     elif supported_m_contig:
        # #         m = elem - (elem // Int32(valid_tile_m)) * Int32(valid_tile_m)
        # #         n = elem // Int32(valid_tile_m)
        # #         offset = m + n * Int32(stride_n)

        # #     in_ptr = gOut_mc.iterator + offset
        # #     out_ptr = gOut.iterator + offset
        # #     x0, x1, x2, x3 = _ld_global_v4_b32(in_ptr)
        # #     if cutlass.const_expr(gRes is not None):
        # #         res_ptr = gRes.iterator + offset
        # #         r0, r1, r2, r3 = _ld_global_v4_b32(res_ptr)
        # #         if cutlass.const_expr(self.c_dtype == cutlass.Float16):
        # #             x0 = _add_rn_f16x2(x0, r0)
        # #             x1 = _add_rn_f16x2(x1, r1)
        # #             x2 = _add_rn_f16x2(x2, r2)
        # #             x3 = _add_rn_f16x2(x3, r3)
        # #         else:
        # #             x0 = _add_rn_bf16x2(x0, r0)
        # #             x1 = _add_rn_bf16x2(x1, r1)
        # #             x2 = _add_rn_bf16x2(x2, r2)
        # #             x3 = _add_rn_bf16x2(x3, r3)

        # #     utils.distributed.st_4xb32(out_ptr, x0, x1, x2, x3)
        # #     chunk += stride

        # cute.arch.sync_threads()
        return

    @staticmethod
    def _compute_stages(
        tile_shape_mnk: tuple[int, int, int],
        a_dtype: type[cutlass.Numeric],
        b_dtype: type[cutlass.Numeric],
        c_dtype: type[cutlass.Numeric],
        epi_tile: tuple[int, int],
        *,
        smem_capacity: int,
        occupancy: int,
        fuse_moe: bool,
    ) -> tuple[int, int]:
        """Computes the number of stages for A/B/C operands based on heuristics.

        :param tile_shape_mnk: The shape (M, N, K) of the CTA tile.
        :type tile_shape_mnk: tuple[int, int, int]
        :param a_dtype: Data type of operand A.
        :type a_dtype: type[cutlass.Numeric]
        :param b_dtype: Data type of operand B.
        :type b_dtype: type[cutlass.Numeric]
        :param smem_capacity: Total available shared memory capacity in bytes.
        :type smem_capacity: int
        :param occupancy: Target number of CTAs per SM (occupancy).
        :type occupancy: int

        :return: A tuple containing the computed number of stages for:
                 (A/B operand stages, epilogue stages)
        :rtype: tuple[int, int]
        """

        epi_stage = 4
        epi_bytes = cute.size(epi_tile) * epi_stage * c_dtype.width // 8 if fuse_moe else 0

        a_shape = cute.slice_(tile_shape_mnk, (None, 0, None))
        b_shape = cute.slice_(tile_shape_mnk, (0, None, None))
        ab_bytes_per_stage = (
            cute.size(a_shape) * a_dtype.width // 8
            + cute.size(b_shape) * b_dtype.width // 8
        )
        mbar_helpers_bytes = 1024

        ab_stage = (
            smem_capacity // occupancy - mbar_helpers_bytes - epi_bytes
        ) // ab_bytes_per_stage
        return ab_stage, epi_stage

    @staticmethod
    def _make_smem_layouts(
        tile_shape_mnk: tuple[int, int, int],
        epi_tile: tuple[int, int],
        a_dtype: type[cutlass.Numeric],
        a_layout: utils.LayoutEnum,
        b_dtype: type[cutlass.Numeric],
        b_layout: utils.LayoutEnum,
        ab_stage: int,
        c_dtype: type[cutlass.Numeric],
        c_layout: utils.LayoutEnum,
        epi_stage: int,
    ) -> tuple[cute.ComposedLayout, cute.ComposedLayout, cute.ComposedLayout]:
        """Create shared memory layouts for A, B, and C tensors.

        :param tile_shape_mnk: CTA tile shape (M,N,K)
        :type tile_shape_mnk: Tuple[int, int, int]
        :param epi_tile: Epilogue tile shape
        :type epi_tile: Tuple[int, int]
        :param a_dtype: Data type for matrix A
        :type a_dtype: type[cutlass.Numeric]
        :param a_layout: Layout enum for matrix A
        :type a_layout: utils.LayoutEnum
        :param b_dtype: Data type for matrix B
        :type b_dtype: type[cutlass.Numeric]
        :param b_layout: Layout enum for matrix B
        :type b_layout: utils.LayoutEnum
        :param ab_stage: Number of stages for A/B tensors
        :type ab_stage: int
        :param c_dtype: Data type for output matrix C
        :type c_dtype: type[cutlass.Numeric]
        :param c_layout: Layout enum for the output matrix C
        :type c_layout: utils.LayoutEnum
        :param epi_stage: Number of epilogue stages
        :type epi_stage: int

        :return: Tuple of shared memory layouts for A, B, and C
        :rtype: Tuple[cute.ComposedLayout, cute.ComposedLayout, cute.ComposedLayout]
        """
        a_smem_layout_staged = sm90_utils.make_smem_layout_a(
            a_layout,
            tile_shape_mnk,
            a_dtype,
            ab_stage,
        )

        b_smem_layout_staged = sm90_utils.make_smem_layout_b(
            b_layout,
            tile_shape_mnk,
            b_dtype,
            ab_stage,
        )

        epi_smem_layout_staged = sm90_utils.make_smem_layout_epi(
            c_dtype,
            c_layout,
            epi_tile,
            epi_stage,
        )

        return a_smem_layout_staged, b_smem_layout_staged, epi_smem_layout_staged

    @staticmethod
    def _compute_grid(
        c: cute.Tensor,
        tile_shape_mnk: tuple[int, int, int],
        cluster_shape_mn: tuple[int, int],
    ) -> tuple[int, int, int]:
        """Compute grid shape for the output tensor C.

        :param c: The output tensor C
        :type c: cute.Tensor
        :param tile_shape_mnk: The shape (M, N, K) of the CTA tile.
        :type tile_shape_mnk: tuple[int, int, int]
        :param cluster_shape_mn: Shape of each cluster in M, N dimensions.
        :type cluster_shape_mn: tuple[int, int]

        :return: Grid shape for kernel launch.
        :rtype: tuple[int, int, int]
        """

        c_shape = (tile_shape_mnk[0], tile_shape_mnk[1])
        gc = cute.zipped_divide(c, tiler=c_shape)
        clusters = cute.ceil_div(cute.get(gc.layout, mode=[1]).shape, cluster_shape_mn)
        grid_xy = tuple(x * y for x, y in zip(clusters, cluster_shape_mn))
        return (grid_xy[0], grid_xy[1], 1)

    @staticmethod
    def _make_tma_store_atoms_and_tensors(
        tensor_c: cute.Tensor,
        epi_smem_layout_staged: cute.ComposedLayout,
        epi_tile: tuple[int, int],
    ) -> tuple[cute.CopyAtom, cute.Tensor]:
        """Create TMA atoms and tensors for C tensor storage.

        :param tensor_c: Output tensor C
        :type tensor_c: cute.Tensor
        :param epi_smem_layout_staged: Shared memory layout for epilogue
        :type epi_smem_layout_staged: cute.ComposedLayout
        :param epi_tile: Epilogue tile shape
        :type epi_tile: Tuple[int, int]

        :return: TMA atom and tensor for C
        :rtype: Tuple[cute.CopyAtom, cute.Tensor]
        """
        epi_smem_layout = cute.slice_(epi_smem_layout_staged, (None, None, 0))
        tma_atom_c, tma_tensor_c = cute.nvgpu.cpasync.make_tiled_tma_atom(
            cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp(),
            tensor_c,
            epi_smem_layout, # ((8,16),(32,1)):((32,256),(1,0))
            epi_tile, #(128, 32)
        )

        return tma_atom_c, tma_tensor_c

    @staticmethod
    def _make_tma_atoms_and_tensors(
        tensor: cute.Tensor,
        smem_layout_staged: cute.ComposedLayout,
        smem_tile: tuple[int, int],
        mcast_dim: int,
    ) -> tuple[cute.CopyAtom, cute.Tensor]:
        """Create TMA atoms and tensors for input tensors.

        :param tensor: Input tensor (A or B)
        :type tensor: cute.Tensor
        :param smem_layout_staged: Shared memory layout for the tensor
        :type smem_layout_staged: cute.ComposedLayout
        :param smem_tile: Shared memory tile shape
        :type smem_tile: Tuple[int, int]
        :param mcast_dim: Multicast dimension
        :type mcast_dim: int

        :return: TMA atom and tensor
        :rtype: Tuple[cute.CopyAtom, cute.Tensor]
        """
        op = (
            cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp()
            if mcast_dim == 1
            else cute.nvgpu.cpasync.CopyBulkTensorTileG2SMulticastOp()
        )

        smem_layout = cute.slice_(smem_layout_staged, (None, None, 0))
        tma_atom, tma_tensor = cute.nvgpu.cpasync.make_tiled_tma_atom(
            op,
            tensor,
            smem_layout,
            smem_tile,
            num_multicast=mcast_dim,
        )
        return tma_atom, tma_tensor


def _fused_gemm_ar_impl(
    x,
    weight,
    moe,
    res,
    gemm_out,
    gemm_out_mc,
    barrier_flag,
    barrier_flag_mc,
    output,
    rank_id,
    tp_size,
    stream,
    config,
):
    acc_dtype = cutlass.Float32

    leading_dim = 1
    m, k = x.shape
    n = weight.shape[0]

    mX = from_dlpack(x, assumed_align=16, enable_tvm_ffi=True).mark_layout_dynamic(leading_dim=leading_dim)
    mWeight = _convert_from_dlpack_cached(weight, leading_dim=leading_dim)
    mGemmOut = from_dlpack(gemm_out, assumed_align=16, enable_tvm_ffi=True).mark_layout_dynamic(leading_dim=leading_dim)
    mGemmOutMc = from_dlpack(gemm_out_mc, assumed_align=16, enable_tvm_ffi=True).mark_layout_dynamic(leading_dim=leading_dim)
    mBarrierFlag = _convert_from_dlpack_cached(barrier_flag, leading_dim=0)
    mBarrierFlagMc = _convert_from_dlpack_cached(barrier_flag_mc, leading_dim=0)

    if moe is not None:
        mMOE = from_dlpack(moe, assumed_align=16, enable_tvm_ffi=True).mark_layout_dynamic(leading_dim=leading_dim)
    else:
        mMOE = None

    if res is not None:
        mRes = from_dlpack(res, assumed_align=16, enable_tvm_ffi=True).mark_layout_dynamic(leading_dim=leading_dim)
    else:
        mRes = None

    mOutput = from_dlpack(output, assumed_align=16, enable_tvm_ffi=True).mark_layout_dynamic(leading_dim=leading_dim)


    tile_shape = (config.tile_m, config.tile_n)
    cluster_shape = (config.cluster_m, config.cluster_n)

    compile_key = (mX.element_type, acc_dtype, config.tile_m, config.tile_n, config.cluster_m, config.cluster_n, rank_id, tp_size, n, k)
    if compile_key not in _fused_gemm_ar_impl.compile_cache:
        # print(f"compile_key not in compile_cache:{compile_key}")
        gemm = HopperWgmmaGemmKernel(
            acc_dtype,
            tile_shape,
            cluster_shape,
            rank_id,
            tp_size,
        )
        compile_options = "--enable-tvm-ffi"
        _fused_gemm_ar_impl.compile_cache[compile_key] = cute.compile(
            gemm,
            mX,
            mWeight,
            mGemmOut,
            mGemmOutMc,
            mMOE,
            mBarrierFlag,
            mBarrierFlagMc,
            mRes,
            mOutput,
            stream,
            options=compile_options,
        )
    else:
        gemm = _fused_gemm_ar_impl.compile_cache[compile_key]

    gemm(
        mX,
        mWeight,
        mGemmOut,
        mGemmOutMc,
        mMOE,
        mBarrierFlag,
        mBarrierFlagMc,
        mRes,
        mOutput,
        stream,
    )


_fused_gemm_ar_impl.compile_cache = {}


def default_config(input):
    token_num = input.shape[0]
    if token_num <= 1024:
        tile_m = 64
        tile_n = 64
    else :
        tile_m = 128
        tile_n = 128
    return GemmConfig(tile_m=tile_m, tile_n=tile_n, cluster_m=1, cluster_n=1, pingpong=True, persistent=True, num_ar_warps=3)

def get_autotune_key(x, weight, moe, res, inplace):
    M, K = x.shape
    N = weight.shape[0]
    has_moe = moe is not None
    has_res = res is not None
    dtype = x.dtype
    return (math.ceil(math.log2(M)), K, N, has_moe, has_res, dtype, inplace)

# @autotune(
#     configs=[AutotuneConfig(config=c) for c in get_all_configs()],
#     key=["autotune_key"],
# )
def fused_gemm_add_ar_add_tuned(
    x, 
    weight, 
    moe, 
    res, 
    gemm_out, 
    gemm_out_mc, 
    barrier_flag, 
    barrier_flag_mc, 
    output, 
    rank_id, 
    tp_size, 
    stream,
    autotune_key=None,    
    config: Optional[GemmConfig] = None,
):
    if config is None:
        config = default_config(x)
    if config.persistent:
        _fused_gemm_ar_persistent_impl(
            x, 
            weight, 
            moe, 
            res, 
            gemm_out, 
            gemm_out_mc, 
            barrier_flag, 
            barrier_flag_mc, 
            output, 
            rank_id, 
            tp_size, 
            stream,
            config
        )
    else:
        _fused_gemm_ar_impl(
            x, 
            weight, 
            moe, 
            res, 
            gemm_out, 
            gemm_out_mc, 
            barrier_flag, 
            barrier_flag_mc, 
            output, 
            rank_id, 
            tp_size, 
            stream,
            config
        )
    
@torch.no_grad()
def fused_gemm_add_ar_add_forward_impl(
    x,
    weight,
    moe,
    res,
    gemm_out,
    gemm_out_mc,
    barrier_flag,
    barrier_flag_mc,
    inplace,
    rank_id,
    tp_size,
    stream,
):
    # if not inplace:
    #     output = torch.empty_like(gemm_out)
    # else:
    #     if res is not None:
    #         output = gemm_out
    #     elif moe is not None:
    #         output = gemm_out
    #     else:
    #         output = gemm_out
    assert inplace == True
    assert res is None
    output = gemm_out

    # autotune_key = get_autotune_key(x, weight, moe, res, inplace)

    fused_gemm_add_ar_add_tuned(
        x, 
        weight, 
        moe, 
        res, 
        gemm_out, 
        gemm_out_mc, 
        barrier_flag, 
        barrier_flag_mc, 
        output, 
        rank_id, 
        tp_size, 
        stream,
        # autotune_key,
    )
    return output
