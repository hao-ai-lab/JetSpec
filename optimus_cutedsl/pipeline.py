# Copyright (c) 2025, Tri Dao.

from typing import Optional
from dataclasses import dataclass

import cutlass.cute as cute
from cutlass.cutlass_dsl import Boolean, Int32, if_generate
from cutlass.pipeline import CooperativeGroup, PipelineOp, pipeline_init_wait
from cutlass.pipeline import PipelineAsync, PipelineTmaAsync, PipelineState, PipelineUserType

from cutlass.cutlass_dsl import dsl_user_op
from cutlass._mlir.dialects import nvvm


@dsl_user_op
def cp_async_mbarrier_arrive_shared(
    mbar_ptr: cute.Pointer, noinc: bool = False, *, loc=None, ip=None
) -> None:
    nvvm.cp_async_mbarrier_arrive_shared(
        mbar_ptr.llvm_ptr,
        noinc=noinc,
        loc=loc,
        ip=ip,
    )


class PipelineStateWAdvance(PipelineState):
    def advance_iters(self, num_iterations: Int32):
        self._count += Int32(num_iterations)
        new_index = self._index + Int32(num_iterations)
        # How many times did we cross the stages boundary
        num_crossings = new_index // self.stages
        self._phase ^= num_crossings
        self._index = new_index % self.stages

    # This can be overridden by derived classes
    def __new_from_mlir_values__(self, values):
        return PipelineStateWAdvance(
            self.stages, Int32(values[0]), Int32(values[1]), Int32(values[2])
        )


def make_pipeline_state(type: PipelineUserType, stages: int):
    """
    Creates a pipeline state. Producers are assumed to start with an empty buffer and have a flipped phase bit of 1.
    """
    if type is PipelineUserType.Producer:
        return PipelineStateWAdvance(
            stages,
            Int32(0),
            Int32(0),
            Int32(1),
        )
    elif type is PipelineUserType.Consumer:
        return PipelineStateWAdvance(
            stages,
            Int32(0),
            Int32(0),
            Int32(0),
        )
    else:
        assert False, "Error: invalid PipelineUserType specified for make_pipeline_state."


@dataclass(frozen=True)
class PipelineTmaCpAsync(PipelineTmaAsync):
    """
    PipelineTmaCpAsync is used for CpAync + TMA producers and AsyncThread consumers
    """

    @staticmethod
    def create(
        *,
        num_stages: int,
        producer_group: CooperativeGroup,
        consumer_group: CooperativeGroup,
        tx_count: int,
        barrier_storage: cute.Pointer = None,
        cta_layout_vmnk: Optional[cute.Layout] = None,
        tidx: Optional[Int32] = None,
    ):
        """
        This helper function computes any necessary attributes and returns an instance of PipelineTmaAsync.
        :param barrier_storage: Pointer to the smem address for this pipeline's mbarriers
        :type barrier_storage: cute.Pointer
        :param num_stages: Number of buffer stages for this pipeline
        :type num_stages: Int32
        :param producer_group: CooperativeGroup for the producer agent
        :type producer_group: CooperativeGroup
        :param consumer_group: CooperativeGroup for the consumer agent
        :type consumer_group: CooperativeGroup
        :param tx_count: Number of bytes expected to be written to the transaction barrier for one stage
        :type tx_count: int
        :param cta_layout_vmnk: Layout of the cluster shape
        :type cta_layout_vmnk: cute.Layout | None
        :param tidx: thread index to consumer async threads
        :type tidx: Int32 | None
        """
        if not isinstance(barrier_storage, cute.Pointer):
            raise ValueError(
                f"Expected barrier_storage to be a cute.Pointer, but got {type(barrier_storage)}"
            )

        producer_type = PipelineOp.TmaLoad
        consumer_type = PipelineOp.AsyncThread

        producer = (producer_type, producer_group)
        consumer = (consumer_type, consumer_group)

        sync_object_full = PipelineAsync._make_sync_object(
            barrier_storage.align(min_align=8), num_stages, producer, tx_count
        )
        sync_object_empty = PipelineAsync._make_sync_object(
            barrier_storage.align(min_align=8) + num_stages, num_stages, consumer
        )
        if tidx is None:
            tidx, _, _ = cute.arch.thread_idx()
        if cta_layout_vmnk is None:
            cta_layout_vmnk = cute.make_layout((1, 1, 1, 1))
        (
            dst_rank,
            is_signalling_thread,
        ) = PipelineTmaAsync.init_empty_barrier_arrive_signal(cta_layout_vmnk, tidx)
        if cta_layout_vmnk is None or cute.size(cta_layout_vmnk) == 1:
            dst_rank = None
        else:
            dst_rank = dst_rank

        producer_mask = None

        pipeline_init_wait(cta_layout_vmnk)

        return PipelineTmaCpAsync(
            sync_object_full,
            sync_object_empty,
            num_stages,
            producer_mask,
            dst_rank,
            is_signalling_thread,
        )

    def producer_acquire(
        self,
        state: PipelineState,
        try_acquire_token: Optional[Boolean] = None,
        is_tma_warp: Optional[Boolean] = True,
    ):
        """
        TMA producer commit conditionally waits on buffer empty and sets the transaction barrier.
        """
        if_generate(
            try_acquire_token is None or try_acquire_token == 0,
            lambda: self.sync_object_empty.wait(state.index, state.phase),
        )
        # This is the difference between this and PipelineTmaAsync: we could have multiple
        # warps calling this, but only 1 warp should do the arrive on the full barrier
        if_generate(
            is_tma_warp,
            lambda: self.sync_object_full.arrive(state.index, self.producer_mask),
        )

    def producer_commit(self, state: PipelineState):
        """
        We need the mbarrier to track the completion of cp.async
        """
        cp_async_mbarrier_arrive_shared(self.producer_get_barrier(state), noinc=True)
