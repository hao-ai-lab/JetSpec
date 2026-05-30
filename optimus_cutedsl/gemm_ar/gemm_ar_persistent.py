import argparse
from typing import Optional, Tuple, Type
import math
import cuda.bindings.driver as cuda

import torch

import cutlass
import cutlass.cute as cute
import cutlass.cute.testing as testing
import cutlass.pipeline as pipeline
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
import cutlass.torch as cutlass_torch
import cutlass.utils as utils
import cutlass.utils.hopper_helpers as sm90_utils
from torch._subclasses.fake_utils import output_alias_each_other
from cutlass.cute.typing import Int32, Float16, BFloat16, Float32, Float8E5M2, Float8E4M3FN
from cutlass.cute.runtime import from_dlpack
from optimus_cutedsl.gemm_ar.autotuner import autotune, AutotuneConfig
from optimus_cutedsl.gemm_ar.gemm_config import GemmConfig, get_all_configs
from optimus_cutedsl.cute_dsl_utils import get_max_active_clusters
from optimus_cutedsl.utils import _convert_from_dlpack_cached

from functools import partial, lru_cache
from typing import Any

def _torch_dtype_to_cutlass(dtype: torch.dtype) -> Optional[Any]:
    if dtype == torch.float16:
        return cutlass.Float16
    if dtype == torch.bfloat16:
        return cutlass.BFloat16
    return None

def _to_cute(t: torch.Tensor, *, cutlass_dtype: Any, leading_dim: int,
             assumed_align: int = 16):
    ct = from_dlpack(t, assumed_align=assumed_align, enable_tvm_ffi=True)
    ct.element_type = cutlass_dtype
    return ct.mark_layout_dynamic(leading_dim=leading_dim)

class HopperWgmmaGemmPersistentKernel:
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
        >>> gemm = HopperWgmmaGemmPersistentKernel(
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
        swizzle_size: int,
        raster_along_m: bool,
        num_ar_warps: int,
        max_active_clusters: int,
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
        self.swizzle_size = swizzle_size
        self.raster_along_m = raster_along_m
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
        self.num_dma_warp_groups = 1
        self.num_mma_warp_groups = math.prod(self.atom_layout_mnk)
        self.num_warps_per_warp_group = 4
        self.num_threads_per_warp_group = self.num_warps_per_warp_group * 32

        # All-reduce (AR) placement:
        # - Default: dedicate warp-groups to AR (keeps AR independent but increases CTA size).
        # - Optional: reuse the (otherwise mostly idle) DMA warp-group's extra warps for AR:
        #   warp0 does TMA, warp1-3 do AR. This reduces CTA size and can improve occupancy
        #   when dedicated AR warp-groups make the kernel resource-limited.

        self.ar_base_warp = 1
        self.num_ar_warps = num_ar_warps
        self.max_active_clusters = max_active_clusters
        self.threads_per_cta = (
            self.num_dma_warp_groups + self.num_mma_warp_groups
        ) * self.num_threads_per_warp_group
        self.load_warp_id = 0
        self.epi_store_warp_id = (
            self.num_dma_warp_groups * self.num_warps_per_warp_group
        )
        # Warp ID used for flag polling and per-SM completion flag publication.
        self.all_reduce_warp_id = self.ar_base_warp
        self.load_register_requirement = 40
        self.mma_register_requirement = 232
        self.smem_capacity = utils.get_smem_capacity_in_bytes("sm_90")

        self.ab_stage = None
        self.epi_stage = None

        self.a_smem_layout_staged = None
        self.b_smem_layout_staged = None
        self.epi_smem_layout_staged = None
        self.epi_tile = None

        self.shared_storage = None
        self.buffer_align_bytes = 1024

        self.num_mma_threads = (
            self.num_mma_warp_groups * self.num_threads_per_warp_group
        )

        self.num_ar_threads = self.num_ar_warps * 32
        self.epilog_sync_barrier = pipeline.NamedBarrier(
            barrier_id=1, num_threads=self.num_mma_threads
        )
        self.ar_sync_barrier = pipeline.NamedBarrier(
            barrier_id=2, num_threads=self.num_ar_threads
        )

        self.rank_id = rank_id
        self.num_ranks = world_size

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
        self.epi_tile = self._sm90_compute_tile_shape_or_override(
            self.tile_shape_mnk, self.c_dtype, is_cooperative=is_cooperative
        )

        # Compute stage before compute smem layout
        self.ab_stage, self.epi_stage = self._compute_stages(
            self.tile_shape_mnk,
            self.a_dtype,
            self.b_dtype,
            self.epi_tile,
            self.c_dtype,
            self.smem_capacity,
            self.occupancy,
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
        :param max_active_clusters: Maximum number of active clusters
        :type max_active_clusters: cutlass.Constexpr
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

        if cutlass.const_expr(
            self.a_dtype.width == 16 and self.a_dtype != self.b_dtype
        ):
            raise TypeError(f"Type mismatch: {self.a_dtype} != {self.b_dtype}")
        if cutlass.const_expr(self.a_dtype.width != self.b_dtype.width):
            raise TypeError(
                f"Type width mismatch: {self.a_dtype.width} != {self.b_dtype.width}"
            )
        if cutlass.const_expr(self.a_dtype.width != 16 and self.a_dtype.width != 8):
            raise TypeError("a_dtype should be float16, float8, or int8 ")

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

        tile_sched_params, grid = self._compute_grid(
            output,
            self.tile_shape_mnk,
            self.cluster_shape_mn,
            self.swizzle_size,
            self.raster_along_m,
            self.max_active_clusters,
        )

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
            sC: cute.struct.Align[
                cute.struct.MemRange[
                    self.c_dtype,
                    cute.cosize(self.epi_smem_layout_staged),
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
            tile_sched_params,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=(*self.cluster_shape_mn, 1),
            min_blocks_per_mp=1,
            stream=stream,
        )
        return

    # GPU device kernel
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
        tile_sched_params: utils.PersistentTileSchedulerParams,
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
        :param tile_sched_params: Parameters for the persistent tile scheduler
        :type tile_sched_params: utils.PersistentTileSchedulerParams
        """

        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)


        # Prefetch Tma desc
        if warp_idx == 0:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_a)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_b)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_c)

        cta_rank_in_cluster = cute.arch.make_warp_uniform(
            cute.arch.block_idx_in_cluster()
        )
        cluster_coord_mnk = cta_layout_mnk.get_flat_coord(cta_rank_in_cluster)

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

        # Alloc and init AB full/empty + ACC full mbar (pipeline)
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
        consumer_arrive_cnt = (
            mcast_size * self.num_mma_warp_groups * self.num_warps_per_warp_group
        )
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
            moe_smem_layout = cute.slice_(epi_smem_layout_staged, (None, None, 0))
            moe_tma_copy_bytes = cute.size_in_bytes(self.c_dtype, moe_smem_layout)
            moe_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
            num_compute_warps = self.num_mma_threads // 32
            moe_pipeline_consumer_group = pipeline.CooperativeGroup(
                pipeline.Agent.Thread, num_compute_warps
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

        # Cluster arrive after barrier init
        pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mn, is_relaxed=True)

        # Generate smem tensor A/B
        sA = storage.sA.get_tensor(
            a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner
        )
        sB = storage.sB.get_tensor(
            b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner
        )
        sC = storage.sC.get_tensor(
            epi_smem_layout_staged.outer, swizzle=epi_smem_layout_staged.inner
        )
        if cutlass.const_expr(moe_out is not None):
            sMoe = storage.sMoe.get_tensor(
                epi_smem_layout_staged.outer, swizzle=epi_smem_layout_staged.inner
            )

        # Local_tile partition global tensors
        # (bM, bK, RestM, RestK)
        gA_mk = cute.local_tile(
            mA_mk,
            (self.tile_shape_mnk[0], self.tile_shape_mnk[2]),
            (None, None),
        )
        # (bN, bK, RestN, RestK)
        gB_nk = cute.local_tile(
            mB_nk,
            (self.tile_shape_mnk[1], self.tile_shape_mnk[2]),
            (None, None),
        )
        # (bM, bN, RestM, RestN)
        gC_mn = cute.local_tile(
            mC_mn,
            (self.tile_shape_mnk[0], self.tile_shape_mnk[1]),
            (None, None),
        )
        if cutlass.const_expr(moe_out is not None):
            gMoe_mn = cute.local_tile(
                moe_out,
                (self.tile_shape_mnk[0], self.tile_shape_mnk[1]),
                (None, None),
            )

        gOut_mc = cute.local_tile(
            output_mc, 
            (self.tile_shape_mnk[0], self.tile_shape_mnk[1]),
            (None, None),
            )

        # Partition shared tensor for TMA load A/B
        # TMA load A partition_S/D
        a_cta_layout = cute.make_layout(cute.slice_(cta_layout_mnk, (0, None, 0)).shape)
        a_cta_crd = cluster_coord_mnk[1]
        tAsA, tAgA = cute.nvgpu.cpasync.tma_partition(
            tma_atom_a,
            a_cta_crd,
            a_cta_layout,
            cute.group_modes(sA, 0, 2),
            cute.group_modes(gA_mk, 0, 2),
        )

        # TMA load B partition_S/D
        b_cta_layout = cute.make_layout(cute.slice_(cta_layout_mnk, (None, 0, 0)).shape)
        b_cta_crd = cluster_coord_mnk[0]
        tBsB, tBgB = cute.nvgpu.cpasync.tma_partition(
            tma_atom_b,
            b_cta_crd,
            b_cta_layout,
            cute.group_modes(sB, 0, 2),
            cute.group_modes(gB_nk, 0, 2),
        )

        # Partition global tensor for TiledMMA_A/B/C
        warp_group_idx = cute.arch.make_warp_uniform(
            tidx // self.num_threads_per_warp_group
        )
        mma_warp_group_thread_layout = cute.make_layout(
            self.num_mma_warp_groups, stride=self.num_threads_per_warp_group
        )
        thr_mma = tiled_mma.get_slice(
            mma_warp_group_thread_layout(warp_group_idx - self.num_dma_warp_groups)
        )

        # Make fragments
        tCsA = thr_mma.partition_A(sA)
        tCsB = thr_mma.partition_B(sB)
        tCrA = tiled_mma.make_fragment_A(tCsA)
        tCrB = tiled_mma.make_fragment_B(tCsB)

        tCgC = thr_mma.partition_C(gC_mn)
        acc_shape = tCgC.shape[:3]
        accumulators = cute.make_rmem_tensor(acc_shape, self.acc_dtype)

        k_tile_cnt = cute.size(gA_mk, mode=[3])

        # Cluster wait for barrier init
        pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)

        is_dma_warp_group = warp_group_idx < self.num_dma_warp_groups
        if is_dma_warp_group:
            cute.arch.warpgroup_reg_dealloc(self.load_register_requirement)
        is_mma_warp_group = warp_group_idx >= self.num_dma_warp_groups and warp_group_idx < self.num_dma_warp_groups + self.num_mma_warp_groups
        if is_mma_warp_group:
            cute.arch.warpgroup_reg_alloc(self.mma_register_requirement)
        # Dedicated AR warp-groups start after DMA+MMA; optional AR reuse runs in warp1-3 of DMA WG.
        is_ar_warp_group = warp_group_idx >= self.num_dma_warp_groups + self.num_mma_warp_groups
        # if is_ar_warp_group:
        #     cute.arch.warpgroup_reg_alloc(self.ar_register_requirement)

        if warp_idx == self.load_warp_id:
            tile_sched = utils.StaticPersistentTileScheduler.create(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            work_tile = tile_sched.initial_work_tile_info()

            mainloop_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.ab_stage
            )

            while work_tile.is_valid_tile:
                tile_coord_mn = work_tile.tile_idx
                tile_m = tile_coord_mn[0]
                tile_n = tile_coord_mn[1]
                tAgA_mk = tAgA[(None, tile_m, None)]
                tBgB_nk = tBgB[(None, tile_n, None)]

                mainloop_producer_state.reset_count()

                for k_tile in range(k_tile_cnt):
                    # Conditionally wait for AB buffer empty
                    mainloop_pipeline.producer_acquire(mainloop_producer_state)
                    # Slice to global/shared memref to current k_tile
                    tAgA_k = tAgA_mk[(None, mainloop_producer_state.count)]
                    tAsA_pipe = tAsA[(None, mainloop_producer_state.index)]

                    tBgB_k = tBgB_nk[(None, mainloop_producer_state.count)]
                    tBsB_pipe = tBsB[(None, mainloop_producer_state.index)]

                    # TMA load A/B
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

                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

            mainloop_pipeline.producer_tail(mainloop_producer_state)

        # MMA warp group
        if is_mma_warp_group:
            # cute.arch.warpgroup_reg_alloc(self.mma_register_requirement)
            tile_sched = utils.StaticPersistentTileScheduler.create(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            work_tile = tile_sched.initial_work_tile_info()

            mainloop_consumer_read_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.ab_stage
            )
            mainloop_consumer_release_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.ab_stage
            )

            num_k_blocks = cute.size(tCrA, mode=[2])

            # Partition for epilogue
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
            mma_thread_idx = tidx - self.num_dma_warp_groups * self.num_threads_per_warp_group
            thr_copy_r2s = tiled_copy_r2s.get_slice(mma_thread_idx)
            # (t)hread-partition for (r)egister to (s)mem copy (tRS_)
            tRS_sD = thr_copy_r2s.partition_D(sC)
            # (R2S, R2S_M, R2S_N)
            tRS_rAcc = tiled_copy_r2s.retile(accumulators)

            # Allocate D registers.
            rD_shape = cute.shape(thr_copy_r2s.partition_S(sC))
            tRS_rD_layout = cute.make_layout(rD_shape[:3])
            tRS_rD = cute.make_rmem_tensor(tRS_rD_layout.shape, self.acc_dtype)
            tRS_rD_out = cute.make_rmem_tensor(tRS_rD_layout.shape, self.c_dtype)
            size_tRS_rD = cute.size(tRS_rD)

            if cutlass.const_expr(moe_out is not None):
                # SMEM -> RMEM loader for staged moe_out tiles.
                #
                # Prefer warp-level `ldmatrix` loads over `CopyUniversalOp(num_bits_per_copy=128)`.
                # On swizzled SMEM layouts, Cute/MLIR often cannot prove per-thread 16B alignment,
                # causing "cannot vectorized copy to 8 elements" verification failures.
                copy_atom_moe_s2r = cute.make_copy_atom(
                    cute.nvgpu.warp.LdMatrix8x8x16bOp(self.c_layout.is_m_major_c(), 4),
                    self.c_dtype,
                )
                tiled_copy_moe_s2r = cute.make_tiled_copy_C(copy_atom_moe_s2r, tiled_mma)
                thr_copy_moe_s2r = tiled_copy_moe_s2r.get_slice(mma_thread_idx)
                tSR_sMoe = thr_copy_moe_s2r.partition_S(sMoe)
                tRS_rMoe = cute.make_rmem_tensor(tRS_rD_layout.shape, self.c_dtype)
                tRS_rMoe_view = thr_copy_moe_s2r.retile(tRS_rMoe)

            k_pipe_mmas = 1
            prologue_mma_cnt = min(k_pipe_mmas, k_tile_cnt)

            # Initialize tma store pipeline
            tma_store_producer_group = pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                self.num_mma_threads,
            )
            tma_store_pipeline = pipeline.PipelineTmaStore.create(
                num_stages=self.epi_stage,
                producer_group=tma_store_producer_group,
            )

            while work_tile.is_valid_tile:
                tile_coord_mn = work_tile.tile_idx
                tile_m = tile_coord_mn[0]
                tile_n = tile_coord_mn[1]
                gC_mn_slice = gC_mn[(None, None, tile_m, tile_n)]

                # Epilogue tiling / partitions (compute early so MoE prefetch can overlap mainloop).
                tCgC_for_tma_partition = cute.zipped_divide(gC_mn_slice, self.epi_tile)

                # thread(b)lock-partition for (s)mem to (g)mem copy (bSG_)
                bSG_sD, bSG_gD = cute.nvgpu.cpasync.tma_partition(
                    tma_atom_c,
                    0,
                    cute.make_layout(1),
                    cute.group_modes(sC, 0, 2),
                    tCgC_for_tma_partition,
                )

                epi_tile_num = cute.size(tCgC_for_tma_partition, mode=[1])
                epi_tile_shape = tCgC_for_tma_partition.shape[1]
                epi_tile_layout = cute.make_layout(
                    epi_tile_shape, stride=(epi_tile_shape[1], 1)
                )

                num_prev_epi_tiles = tile_sched.num_tiles_executed * epi_tile_num

                if cutlass.const_expr(moe_out is not None):
                    gMoe_mn_slice = gMoe_mn[(None, None, tile_m, tile_n)]
                    tCgMoe_for_tma_partition = cute.zipped_divide(
                        gMoe_mn_slice, self.epi_tile
                    )
                    bLG_sMoe, bLG_gMoe = cute.nvgpu.cpasync.tma_partition(
                        tma_atom_moe,
                        0,
                        cute.make_layout(1),
                        cute.group_modes(sMoe, 0, 2),
                        tCgMoe_for_tma_partition,
                    )

                    # Kick off initial MoE prefetch as early as possible to overlap WGMMA.
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

                    if warp_idx == self.epi_store_warp_id:
                        for _ in cutlass.range(moe_prefetch_cnt, unroll=1):
                            moe_pipeline.producer_acquire(moe_producer_state)
                            gmem_coord = epi_tile_layout.get_hier_coord(
                                moe_producer_state.count
                            )
                            cute.copy(
                                tma_atom_moe,
                                bLG_gMoe[(None, gmem_coord)],
                                bLG_sMoe[(None, moe_producer_state.index)],
                                tma_bar_ptr=moe_pipeline.producer_get_barrier(
                                    moe_producer_state
                                ),
                            )
                            moe_pipeline.producer_commit(moe_producer_state)
                            moe_producer_state.advance()

                    peek_moe_full_status = cutlass.Boolean(1)
                    if moe_consumer_read_state.count < epi_tile_num:
                        peek_moe_full_status = moe_pipeline.consumer_try_wait(
                            moe_consumer_read_state
                        )

                # MAINLOOP
                mainloop_consumer_read_state.reset_count()
                mainloop_consumer_release_state.reset_count()
                accumulators.fill(0.0)
                tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, True)
                cute.nvgpu.warpgroup.fence()

                for k_tile in range(prologue_mma_cnt):
                    # Wait for TMA copies to complete
                    mainloop_pipeline.consumer_wait(mainloop_consumer_read_state)
                    # WGMMA
                    for k_block_idx in cutlass.range_constexpr(num_k_blocks):
                        k_block_coord = (
                            None,
                            None,
                            k_block_idx,
                            mainloop_consumer_read_state.index,
                        )
                        cute.gemm(
                            tiled_mma,
                            accumulators,
                            tCrA[k_block_coord],
                            tCrB[k_block_coord],
                            accumulators,
                        )

                    cute.nvgpu.warpgroup.commit_group()
                    mainloop_consumer_read_state.advance()

                for k_tile in range(prologue_mma_cnt, k_tile_cnt):
                    # Wait for TMA copies to complete
                    mainloop_pipeline.consumer_wait(mainloop_consumer_read_state)
                    # WGMMA
                    for k_block_idx in cutlass.range_constexpr(num_k_blocks):
                        k_block_coord = (
                            None,
                            None,
                            k_block_idx,
                            mainloop_consumer_read_state.index,
                        )
                        cute.gemm(
                            tiled_mma,
                            accumulators,
                            tCrA[k_block_coord],
                            tCrB[k_block_coord],
                            accumulators,
                        )

                    cute.nvgpu.warpgroup.commit_group()
                    # Wait on the wgmma barrier for WGMMA to complete
                    cute.nvgpu.warpgroup.wait_group(k_pipe_mmas)

                    mainloop_pipeline.consumer_release(mainloop_consumer_release_state)
                    mainloop_consumer_release_state.advance()
                    mainloop_consumer_read_state.advance()

                cute.nvgpu.warpgroup.wait_group(0)
                for k_tile in range(prologue_mma_cnt):
                    mainloop_pipeline.consumer_release(mainloop_consumer_release_state)
                    mainloop_consumer_release_state.advance()

                # Epilogue
                for epi_idx in cutlass.range_constexpr(epi_tile_num):
                    # Copy from accumulators to D registers
                    for epi_v in cutlass.range_constexpr(size_tRS_rD):
                        tRS_rD[epi_v] = tRS_rAcc[epi_idx * size_tRS_rD + epi_v]

                    if cutlass.const_expr(moe_out is not None):
                        # Load the staged MoE tile once per epilogue tile, then add it
                        # elementwise to the accumulator registers. Doing this inside
                        # the per-element loop would repeatedly reload and re-add MoE,
                        # and also operate on partially-initialized registers.
                        moe_pipeline.consumer_wait(
                            moe_consumer_read_state, peek_moe_full_status
                        )
                        cute.copy(
                            tiled_copy_moe_s2r,
                            tSR_sMoe[(None, None, None, moe_consumer_read_state.index)],
                            tRS_rMoe_view,
                        )
                        acc_vec = tRS_rD.load()
                        moe_vec = tRS_rMoe.load()
                        tRS_rD.store(acc_vec + moe_vec.to(self.acc_dtype))

                    # Type conversion
                    acc_vec = tRS_rD.load()
                    tRS_rD_out.store(acc_vec.to(self.c_dtype))

                    # Copy from D registers to shared memory
                    epi_buffer = (num_prev_epi_tiles + epi_idx) % cute.size(
                        tRS_sD, mode=[3]
                    )
                    cute.copy(
                        tiled_copy_r2s,
                        tRS_rD_out,
                        tRS_sD[(None, None, None, epi_buffer)],
                    )

                    cute.arch.fence_proxy(
                        cute.arch.ProxyKind.async_shared,
                        space=cute.arch.SharedSpace.shared_cta,
                    )
                    self.epilog_sync_barrier.arrive_and_wait()

                    if cutlass.const_expr(moe_out is not None):
                        moe_pipeline.consumer_release(moe_consumer_release_state)
                        moe_consumer_read_state.advance()
                        moe_consumer_release_state.advance()
                        peek_moe_full_status = cutlass.Boolean(1)
                        if moe_consumer_read_state.count < epi_tile_num:
                            peek_moe_full_status = moe_pipeline.consumer_try_wait(
                                moe_consumer_read_state
                            )
                        if (
                            warp_idx == self.epi_store_warp_id
                            and moe_producer_state.count < epi_tile_num
                        ):
                            moe_pipeline.producer_acquire(moe_producer_state)
                            gmem_coord_next = epi_tile_layout.get_hier_coord(
                                moe_producer_state.count
                            )
                            cute.copy(
                                tma_atom_moe,
                                bLG_gMoe[(None, gmem_coord_next)],
                                bLG_sMoe[(None, moe_producer_state.index)],
                                tma_bar_ptr=moe_pipeline.producer_get_barrier(
                                    moe_producer_state
                                ),
                            )
                            moe_pipeline.producer_commit(moe_producer_state)
                            moe_producer_state.advance()

                    gmem_coord = epi_tile_layout.get_hier_coord(epi_idx)
                    # Copy from shared memory to global memory
                    if warp_idx == self.epi_store_warp_id:
                        cute.copy(
                            tma_atom_c,
                            bSG_sD[(None, epi_buffer)],
                            bSG_gD[(None, gmem_coord)],
                        )
                        tma_store_pipeline.producer_commit()
                        tma_store_pipeline.producer_acquire()

                    self.epilog_sync_barrier.arrive_and_wait()

                tile_id_linear = Int32(
                    tile_sched._current_work_linear_idx * cute.size(self.cluster_shape_mn)
                    + cute.arch.block_idx_in_cluster()
                )
                tma_store_pipeline.producer_tail()
                if warp_idx == self.epi_store_warp_id:
                    with cute.arch.elect_one():
                        # if self.rank_id == 0:
                        #     cute.printf("epi tile_id_linear: %d\n", tile_id_linear)                        
                        # Ensure the (async) TMA stores are ordered before publishing
                        # the "tile ready" flag for cross-rank consumers.
                        # cute.arch.fence_acq_rel_sys()
                        utils.distributed.multimem_red_add1(
                            lock_ptr=barrier_flag_mc.iterator + tile_id_linear,
                            scope="sys",
                            order="release",
                        )

                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()


        # AR participants:
        # - Dedicated mode: all warps in the dedicated AR warp-groups.
        # - Reuse mode: use warp1-3 in DMA warp-group (warp0 remains the TMA producer).
        is_ar_participant = warp_idx >= self.ar_base_warp and warp_idx < (
            self.ar_base_warp + self.num_ar_warps
        )

        if is_ar_participant:

            rank_id = self.rank_id
            num_ranks = Int32(self.num_ranks)
            num_allreduce_warps = self.num_ar_warps

            tile_sched = utils.StaticPersistentTileScheduler.create(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            # Interleave tiles across AR warps:
            # - AR warp `w` starts from the `w`-th tile in this CTA's work stream.
            # - After processing a tile, it advances by `num_allreduce_warps` tiles.
            ar_warp_linear = warp_idx - self.ar_base_warp
            work_tile = tile_sched.initial_work_tile_info()
            if ar_warp_linear >= 1:
                tile_sched.advance_to_next_work()
            if ar_warp_linear >= 2:
                tile_sched.advance_to_next_work()
            if ar_warp_linear >= 3:
                tile_sched.advance_to_next_work()
            work_tile = tile_sched.get_current_work()

            # we want 128bit ld/st for better performance
            atom_val = 128 // output_mc.element_type.width
            atom_thr_n = self.tile_shape_mnk[1] // atom_val
            # Thread layout for the AR warp.
            #
            # Each AR warp (32 threads) is responsible for one tile's all-reduce.
            # Use a ceil-div so every participating thread maps to a valid (m,n)
            # coordinate in the tiled-copy thread layout.
            num_ar_threads = 32
            atom_thr_m = (num_ar_threads + atom_thr_n - 1) // atom_thr_n
            thr_layout = cute.make_layout(
                (atom_thr_m, atom_thr_n), stride=(atom_thr_n, 1)
            )
            val_layout = cute.make_layout((1, atom_val), stride=(atom_val, 1))

            copy_atom_load = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(), output_mc.element_type
            )
            tiled_copy_fake = cute.make_tiled_copy_tv(
                copy_atom_load, thr_layout, val_layout
            )
            # Map the participating threads to a contiguous [0, 32) range per AR warp.
            ar_base_tidx = warp_idx * 32
            ar_tidx = tidx - ar_base_tidx
            thr_copy_fake = tiled_copy_fake.get_slice(ar_tidx)
            # predicate tensor
            idC = cute.make_identity_tensor(output_mc.shape)

            while work_tile.is_valid_tile:
                cur_tile_coord = work_tile.tile_idx
                # tile_id = Int32(
                #     tile_sched._current_work_linear_idx
                #     * cute.size(self.cluster_shape_mn)
                #     + cute.arch.block_idx_in_cluster()
                # )
                tile_id_linear = Int32(
                    tile_sched._current_work_linear_idx * cute.size(self.cluster_shape_mn)
                    + cute.arch.block_idx_in_cluster()
                )
                # Cross-rank synchronization: the producer publishes "tile ready"
                # using a system-scope release atomic (via `multimem_red_add1`).
                #
                # IMPORTANT:
                # - Use exactly one thread to poll the flag; polling uses atomic add(0) as an
                #   atomic load and should be side-effect free, but having 4 warps spin is noisy.
                # - We poll the *local* symmetric pointer (`barrier_flag`) since the polling
                #   primitive is implemented via a normal sys-scope atomic (add 0). Polling a
                #   multicast-view pointer here can be undefined / not observed.
                with cute.arch.elect_one():
                    # cute.arch.fence_acq_rel_sys()
                    utils.distributed.spin_lock_atom_cas_relaxed_wait(
                        lock_ptr=barrier_flag.iterator + tile_id_linear,
                        expected_val=num_ranks,
                        reset_val=0,
                        scope="sys",
                    )
                cute.arch.sync_warp()

                gC_mc = cute.local_tile(
                    output_mc,
                    (self.tile_shape_mnk[0], self.tile_shape_mnk[1]),
                    (None, None),
                )

                cC = cute.local_tile(
                    idC,
                    (self.tile_shape_mnk[0], self.tile_shape_mnk[1]),
                    (None, None),
                )

                # Slice the current CTA tile: (bM, bN)
                tile_m = cur_tile_coord[0]
                tile_n = cur_tile_coord[1]
                gC_mc_slice = gC_mc[(None, None, tile_m, tile_n)]
                cC_slice = cC[(None, None, tile_m, tile_n)]

                # Partition based on the number of GPUs
                m_local_rank = int(self.tile_shape_mnk[0] / self.num_ranks)
                tCgC_mc_slice_partitioned = cute.zipped_divide(
                    gC_mc_slice, (m_local_rank, self.tile_shape_mnk[1])
                )
                tCpC_slice_partitioned = cute.zipped_divide(
                    cC_slice, (m_local_rank, self.tile_shape_mnk[1])
                )
                tCgC_mc_local_rank = cute.slice_(
                    tCgC_mc_slice_partitioned, ((None, None), (rank_id, 0))
                )
                tCpC_local_rank = cute.slice_(
                    tCpC_slice_partitioned, ((None, None), (rank_id, 0))
                )

                # Partition at thread level
                frgC_mc = thr_copy_fake.partition_S(tCgC_mc_local_rank)
                frpC = thr_copy_fake.partition_S(tCpC_local_rank)
                atom, loop_m, loop_n = frgC_mc.shape

                for i in cutlass.range_constexpr(loop_m):
                    for j in cutlass.range_constexpr(loop_n):
                        if cute.elem_less(frpC[0, i, j], output_mc.shape):
                            mc_ptr = frgC_mc[None, i, j].iterator
                            x, y, z, w = 0, 0, 0, 0
                            if cutlass.const_expr(self.c_dtype == Float16):
                                x, y, z, w = utils.distributed.multimem_ld_reduce_8xf16(
                                    mc_ptr
                                )
                            elif cutlass.const_expr(self.c_dtype == Float32):
                                x, y, z, w = utils.distributed.multimem_ld_reduce_4xf32(
                                    mc_ptr
                                )
                            elif cutlass.const_expr(self.c_dtype == BFloat16):
                                x, y, z, w = utils.distributed.multimem_ld_reduce_8xbf16(
                                    mc_ptr
                                )
                            elif cutlass.const_expr(self.c_dtype == Float8E4M3FN):
                                x, y, z, w = utils.distributed.multimem_ld_reduce_16xe4m3(
                                    mc_ptr
                                )
                            elif cutlass.const_expr(self.c_dtype == Float8E5M2):
                                x, y, z, w = utils.distributed.multimem_ld_reduce_16xe5m2(
                                    mc_ptr
                                )
                            utils.distributed.multimem_st_4xb32(mc_ptr, x, y, z, w)

                # Advance to next tile
                for _ in cutlass.range_constexpr(num_allreduce_warps):
                    tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

            self.ar_sync_barrier.arrive_and_wait()

            #
            # Set Per SM Flag with Release
            #
            # This ensure
            # 1. no rank early exit while other ranks are still issuing multimem.ld_reduce
            # 2. each rank's prior multimem.st have become visiable to all other ranks in the system (w/ .SYS scope)
            if warp_idx == self.all_reduce_warp_id:
                with cute.arch.elect_one():
                    # Offset to last tile flag idx
                    last_tile_id_linear = cute.size(
                        tile_sched.params.problem_layout_ncluster_mnl
                    ) * cute.size(self.cluster_shape_mn)
                    # Linear id of current SM.
                    sm_id_linear = (
                        cute.arch.block_idx()[0]
                        + cute.arch.block_idx()[1] * cute.arch.grid_dim()[0]
                        + cute.arch.block_idx()[2]
                        * cute.arch.grid_dim()[0]
                        * cute.arch.grid_dim()[1]
                    )
                    # Release flag with sys scope
                    utils.distributed.multimem_red_add1(
                        lock_ptr=barrier_flag_mc.iterator
                        + last_tile_id_linear
                        + sm_id_linear,
                        scope="sys",
                        order="release",
                    )
                    # Relaxed spin-lock wait flag with sys scope
                    utils.distributed.spin_lock_atom_cas_relaxed_wait(
                        lock_ptr=barrier_flag.iterator
                        + last_tile_id_linear
                        + sm_id_linear,
                        expected_val=num_ranks,
                        reset_val=0,
                        scope="sys",
                    )
          

    @staticmethod
    def _compute_stages(
        tile_shape_mnk: tuple[int, int, int],
        a_dtype: type[cutlass.Numeric],
        b_dtype: type[cutlass.Numeric],
        epi_tile: tuple[int, int],
        c_dtype: type[cutlass.Numeric],
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
        :param epi_tile: Epilogue tile shape
        :type epi_tile: Tuple[int, int]
        :param c_dtype: The data type of the output tensor
        :type c_dtype: type[cutlass.Numeric]
        :param smem_capacity: Total available shared memory capacity in bytes.
        :type smem_capacity: int
        :param occupancy: Target number of CTAs per SM (occupancy).
        :type occupancy: int

        :return: A tuple containing the computed number of stages for:
                 (A/B operand stages, epilogue stages)
        :rtype: tuple[int, int]
        """

        a_shape = cute.slice_(tile_shape_mnk, (None, 0, None))
        b_shape = cute.slice_(tile_shape_mnk, (0, None, None))
        ab_bytes_per_stage = (
            cute.size(a_shape) * a_dtype.width // 8
            + cute.size(b_shape) * b_dtype.width // 8
        )
        c_bytes_per_stage = cute.size(epi_tile) * c_dtype.width // 8
        epi_stage = 4
        epi_bytes = c_bytes_per_stage * epi_stage

        mbar_helpers_bytes = 1024
        moe_bytes = 0
        if fuse_moe:
            moe_bytes_per_stage = cute.size(epi_tile) * c_dtype.width // 8
            moe_bytes = moe_bytes_per_stage * epi_stage

        ab_stage = (
            smem_capacity // occupancy - (mbar_helpers_bytes + epi_bytes + moe_bytes)
        ) // ab_bytes_per_stage
        return ab_stage, epi_stage

    @staticmethod
    def _sm90_compute_tile_shape_or_override(
        tile_shape_mnk: tuple[int, int, int],
        element_type: type[cutlass.Numeric],
        is_cooperative: bool = False,
        epi_tile_override: Optional[tuple[int, int]] = None,
    ) -> tuple[int, int]:
        """Compute the epilogue tile shape or use override if provided.

        :param tile_shape_mnk: CTA tile shape (M,N,K)
        :type tile_shape_mnk: Tuple[int, int, int]
        :param element_type: Data type of elements
        :type element_type: type[cutlass.Numeric]
        :param is_cooperative: Whether to use cooperative approach
        :type is_cooperative: bool
        :param epi_tile_override: Optional override for epilogue tile shape
        :type epi_tile_override: Tuple[int, int] or None

        :return: Computed epilogue tile shape
        :rtype: Tuple[int, int]
        """
        if epi_tile_override is not None:
            return epi_tile_override
        if is_cooperative:
            tile_m = min(128, cute.size(tile_shape_mnk, mode=[0]))
            tile_n = min(32, cute.size(tile_shape_mnk, mode=[1]))
            return (tile_m, tile_n)
        else:
            n_perf = 64 if element_type.width == 8 else 32
            tile_m = min(64, cute.size(tile_shape_mnk, mode=[0]))
            tile_n = min(n_perf, cute.size(tile_shape_mnk, mode=[1]))
            return (tile_m, tile_n)

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
        a_smem_shape = cute.slice_(tile_shape_mnk, (None, 0, None))

        a_is_k_major = (
            a_layout.sm90_mma_major_mode() == cute.nvgpu.warpgroup.OperandMajorMode.K
        )
        b_is_k_major = (
            b_layout.sm90_mma_major_mode() == cute.nvgpu.warpgroup.OperandMajorMode.K
        )
        a_major_mode_size = tile_shape_mnk[2 if a_is_k_major else 0]
        a_smem_layout_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
            sm90_utils.get_smem_layout_atom(
                a_layout,
                a_dtype,
                a_major_mode_size,
            ),
            a_dtype,
        )
        a_smem_layout_staged = cute.tile_to_shape(
            a_smem_layout_atom,
            cute.append(a_smem_shape, ab_stage),
            order=(0, 1, 2) if a_is_k_major else (1, 0, 2),
        )

        b_smem_shape = cute.slice_(tile_shape_mnk, (0, None, None))

        b_major_mode_size = tile_shape_mnk[2 if b_is_k_major else 1]
        b_smem_layout_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
            sm90_utils.get_smem_layout_atom(
                b_layout,
                b_dtype,
                b_major_mode_size,
            ),
            b_dtype,
        )
        b_smem_layout_staged = cute.tile_to_shape(
            b_smem_layout_atom,
            cute.append(b_smem_shape, ab_stage),
            order=(0, 1, 2) if b_is_k_major else (1, 0, 2),
        )

        c_smem_shape = epi_tile
        c_major_mode_size = epi_tile[1] if c_layout.is_n_major_c() else epi_tile[0]
        c_smem_layout_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
            sm90_utils.get_smem_layout_atom(
                c_layout,
                c_dtype,
                c_major_mode_size,
            ),
            c_dtype,
        )
        epi_smem_layout_staged = cute.tile_to_shape(
            c_smem_layout_atom,
            cute.append(c_smem_shape, epi_stage),
            order=(1, 0, 2) if c_layout.is_m_major_c() else (0, 1, 2),
        )

        return a_smem_layout_staged, b_smem_layout_staged, epi_smem_layout_staged

    @staticmethod
    def _compute_grid(
        c: cute.Tensor,
        tile_shape_mnk: tuple[int, int, int],
        cluster_shape_mn: tuple[int, int],
        swizzle_size: int,
        raster_along_m: bool,
        max_active_clusters: cutlass.Constexpr,
    ) -> tuple[int, int, int]:
        """Compute grid shape for the output tensor C.

        :param c: The output tensor C
        :type c: cute.Tensor
        :param tile_shape_mnk: The shape (M, N, K) of the CTA tile.
        :type tile_shape_mnk: tuple[int, int, int]
        :param cluster_shape_mn: Shape of each cluster in M, N dimensions.
        :type cluster_shape_mn: tuple[int, int]
        :param max_active_clusters: Maximum number of active clusters.
        :type max_active_clusters: cutlass.Constexpr

        :return: Grid shape for kernel launch.
        :rtype: tuple[int, int, int]
        """

        c_shape = (tile_shape_mnk[0], tile_shape_mnk[1])
        gc = cute.zipped_divide(c, tiler=c_shape)
        num_ctas_mn = gc[(0, (None, None))].shape
        num_ctas_mn1 = (*num_ctas_mn, 1)
        cluster_shape_mn1 = (*cluster_shape_mn, 1)

        tile_sched_params = utils.PersistentTileSchedulerParams(
            num_ctas_mn1,
            cluster_shape_mn1,
            swizzle_size,
            raster_along_m,
        )
        grid = utils.StaticPersistentTileScheduler.get_grid_shape(
            tile_sched_params, max_active_clusters
        )
        return tile_sched_params, grid

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
            epi_smem_layout,
            epi_tile,
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
            cutlass.Float8E4M3FN,
            cutlass.Float8E5M2,
            cutlass.Uint8,
            cutlass.Int8,
        }
        if a_dtype not in valid_ab_dtypes:
            is_valid = False
        if b_dtype not in valid_ab_dtypes:
            is_valid = False

        # make sure a_dtype == b_dtype for Float16
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


def _fused_gemm_ar_persistent_impl(
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
    # print("persistent kernel called")
    # cutlass_dtype = _torch_dtype_to_cutlass(x.dtype)
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
    max_active_clusters = Int32(get_max_active_clusters(config.cluster_m * config.cluster_n))
    # print("max_active_clusters: ", max_active_clusters)

    # NOTE: `moe` / `res` are Optional and are used in `cutlass.const_expr(...)` branches
    # inside the jitted callable. Some Cutlass-DSL builds specialize the runtime
    # signature based on whether these optionals are present, so the compile cache
    # key must include them to avoid signature mismatches at runtime.
    compile_key = (
        mX.element_type,
        acc_dtype,
        config.tile_m,
        config.tile_n,
        config.cluster_m,
        config.cluster_n,
        # tile_shape,
        # cluster_shape,
        rank_id,
        tp_size,
        config.num_ar_warps,
        max_active_clusters,
        mMOE is not None,
        mRes is not None,
        n,
        k
    )
    if compile_key not in _fused_gemm_ar_persistent_impl.compile_cache:
        # (avoid noisy compile-time logging)
        gemm = HopperWgmmaGemmPersistentKernel(
            acc_dtype,
            tile_shape,
            cluster_shape,
            rank_id,
            tp_size,
            swizzle_size=1,
            raster_along_m=True,
            num_ar_warps=config.num_ar_warps,
            max_active_clusters=max_active_clusters,
        )
        compile_options = "--enable-tvm-ffi"
        _fused_gemm_ar_persistent_impl.compile_cache[compile_key] = cute.compile(
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
        gemm = _fused_gemm_ar_persistent_impl.compile_cache[compile_key]

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



_fused_gemm_ar_persistent_impl.compile_cache = {}

@lru_cache
def get_device_capacity(device: torch.device = None) -> Tuple[int, int]:
    # Keep module import safe in CPU-only environments (e.g., CI) by avoiding
    # unconditional CUDA initialization at import time.
    try:
        if not torch.cuda.is_available():
            # Default to Hopper/Sm90 for config enumeration; callers on real GPUs
            # will hit the fast path below.
            return (9, 0)
        return torch.cuda.get_device_capability(device)
    except RuntimeError:
        # e.g., "Found no NVIDIA driver on your system"
        return (9, 0)


# default_device_capacity = get_device_capacity(torch.device("cuda"))

# def default_config(device):
#     if get_device_capacity(device)[0] != 10:
#         return GemmConfig(tile_m=128, tile_n=128, cluster_m=2, cluster_n=1, pingpong=True, num_ar_warps=3)
#     else:
#         return GemmConfig(tile_m=256, tile_n=256, cluster_m=2, cluster_n=1, pingpong=False, num_ar_warps=3)

# def get_autotune_key(x, weight, moe, res, inplace):
#     M, K = x.shape
#     N = weight.shape[0]
#     has_moe = moe is not None
#     has_res = res is not None
#     dtype = x.dtype
#     return (math.ceil(math.log2(M)), K, N, has_moe, has_res, dtype, inplace)

# @autotune(
#     configs=[AutotuneConfig(config=c) for c in get_all_configs(default_device_capacity[0])],
#     key=["autotune_key"],
# )
# def fused_gemm_add_ar_add_tuned(
#     x, 
#     weight, 
#     moe, 
#     res, 
#     gemm_out, 
#     gemm_out_mc, 
#     barrier_flag, 
#     barrier_flag_mc, 
#     output, 
#     rank_id, 
#     tp_size, 
#     stream,
#     autotune_key=None,    
#     config: Optional[GemmConfig] = None,
# ):
#     # (avoid noisy runtime logging)
#     if config is None:
#         config = default_config(x.device)
#     # print("autotune_config: ", config)
#     _fused_gemm_ar_persistent_impl(
#         x, 
#         weight, 
#         moe, 
#         res, 
#         gemm_out, 
#         gemm_out_mc, 
#         barrier_flag, 
#         barrier_flag_mc, 
#         output, 
#         rank_id, 
#         tp_size, 
#         stream,
#         tile_M=config.tile_m,
#         tile_N=config.tile_n,
#         cluster_M=config.cluster_m,
#         cluster_N=config.cluster_n,
#         num_ar_warps=config.num_ar_warps,
#     )
    

# def fused_gemm_add_ar_add_forward_persistent_impl(
#     x,
#     weight,
#     moe,
#     res,
#     gemm_out,
#     gemm_out_mc,
#     barrier_flag,
#     barrier_flag_mc,
#     inplace,
#     rank_id,
#     tp_size,
#     stream,
# ):
#     if not inplace:
#         output = torch.empty_like(gemm_out)
#     else:
#         if res is not None:
#             output = res
#         elif moe is not None:
#             output = gemm_out
#         else:
#             output = gemm_out

#     autotune_key = get_autotune_key(x, weight, moe, res, inplace)

#     fused_gemm_add_ar_add_tuned(
#         x, 
#         weight, 
#         moe, 
#         res, 
#         gemm_out, 
#         gemm_out_mc, 
#         barrier_flag, 
#         barrier_flag_mc, 
#         output, 
#         rank_id, 
#         tp_size, 
#         stream,
#         autotune_key,
#     )
#     return output
