import math
from functools import partial
from types import SimpleNamespace
from typing import Callable, Optional

import torch

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Uint8, const_expr
from cutlass.cute.nvgpu import cpasync
from cutlass.cute.runtime import from_dlpack

from optimus_cutedsl.flash_attn import hopper_helpers as sm90_utils
from optimus_cutedsl.flash_attn import pipeline
from optimus_cutedsl.flash_attn import utils
from optimus_cutedsl.flash_attn.block_info import BlockInfo
from optimus_cutedsl.flash_attn.block_sparsity import BlockSparseTensors
from optimus_cutedsl.flash_attn.flash_fwd_sm90_paged import FlashAttentionForwardPagedSM90
from optimus_cutedsl.flash_attn.interface import maybe_contiguous, torch2cute_dtype_map
from optimus_cutedsl.flash_attn.mask import AttentionMask
from optimus_cutedsl.flash_attn.named_barrier import NamedBarrierFwd
from optimus_cutedsl.flash_attn.block_sparse_utils import consume_block_sparse_loads
from optimus_cutedsl.flash_attn.pack_gqa import PackGQA
from optimus_cutedsl.flash_attn.seqlen_info import SeqlenInfoQK
from optimus_cutedsl.flash_attn.softmax import Softmax
from optimus_cutedsl.flash_attn.tile_scheduler import ParamsBase


def _round_up(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


class FlashAttentionForwardPagedTreeSM90(FlashAttentionForwardPagedSM90):
    def _get_shared_storage_cls(self):
        sQ_alignment = 128 if const_expr(self.use_tma_Q) else 1024
        sK_alignment = 128
        sV_alignment = 128
        sQ_struct, sK_struct, sV_struct = [
            cute.struct.Align[cute.struct.MemRange[self.dtype, cute.cosize(layout)], alignment]
            for layout, alignment in zip(
                (self.sQ_layout, self.sK_layout, self.sV_layout),
                (sQ_alignment, sK_alignment, sV_alignment),
            )
        ]
        cosize_sQV = max(cute.cosize(self.sQ_layout), cute.cosize(self.sV_layout))
        sQV_struct = cute.struct.Align[cute.struct.MemRange[self.dtype, cosize_sQV], 1024]
        cosize_sP = cute.cosize(self.sP_layout) if const_expr(self.sP_layout is not None) else 0
        sP_struct = cute.struct.Align[cute.struct.MemRange[self.dtype, cosize_sP], 1024]
        # Note(wangbojun/codex): tree masking only touches at most 1-2 tail blocks, so staging one
        # full tile in shared avoids scattered global loads without perturbing the dense mainloop.
        sTreeMask_struct = cute.struct.Align[
            cute.struct.MemRange[Uint8, self.tile_m * self.tile_n], 1024
        ]
        mbar_ptr_QO_struct = cute.struct.MemRange[cutlass.Int64, 2]
        mbar_ptr_K_struct = cute.struct.MemRange[cutlass.Int64, self.num_stages * 2]
        mbar_ptr_V_struct = cute.struct.MemRange[cutlass.Int64, self.num_stages * 2]

        @cute.struct
        class SharedStorageQKV:
            mbar_ptr: mbar_ptr_QO_struct
            mbar_ptr_K: mbar_ptr_K_struct
            mbar_ptr_V: mbar_ptr_V_struct
            sV: sV_struct
            sQ: sQ_struct
            sK: sK_struct
            sP: sP_struct
            sTreeMask: sTreeMask_struct

        @cute.struct
        class SharedStorageSharedQV:
            mbar_ptr: mbar_ptr_QO_struct
            mbar_ptr_K: mbar_ptr_K_struct
            mbar_ptr_V: mbar_ptr_V_struct
            sQ: sQV_struct
            sK: sK_struct
            sP: sP_struct
            sTreeMask: sTreeMask_struct

        return SharedStorageQKV if const_expr(not self.Q_in_regs) else SharedStorageSharedQV

    @cute.jit
    def load_tree_mask_tile(
        self,
        sTreeMask: cute.Tensor,
        batch_idx: Int32,
        m_block: Int32,
        n_block: Int32,
        aux_tensors: list,
        tidx: Int32,
    ) -> None:
        prefix_lens = aux_tensors[0]
        tree_mask = aux_tensors[1]
        prefix_len = Int32(prefix_lens[batch_idx])
        values_per_thread = cute.ceil_div(self.tile_m * self.tile_n, self.num_mma_threads)

        for i in cutlass.range(values_per_thread, unroll=1):
            linear_idx = tidx + i * self.num_mma_threads
            if linear_idx < self.tile_m * self.tile_n:
                row_idx = linear_idx // self.tile_n
                col_idx = linear_idx - row_idx * self.tile_n
                tree_col_idx = n_block * self.tile_n + col_idx - prefix_len
                mask_value = Uint8(1)
                if tree_col_idx >= 0:
                    tree_row_idx = row_idx + m_block * self.tile_m
                    if const_expr(self.pack_gqa):
                        tree_row_idx = tree_row_idx // self.qhead_per_kvhead
                    mask_value = Uint8(tree_mask[batch_idx, tree_row_idx, tree_col_idx])
                sTreeMask[row_idx, col_idx] = mask_value

        cute.arch.barrier(
            barrier_id=int(NamedBarrierFwd.PFull),
            number_of_threads=self.num_mma_threads,
        )

    @cute.jit
    def apply_tree_tail_mask(
        self,
        acc_S: cute.Tensor,
        sTreeMask: cute.Tensor,
        batch_idx: Int32,
        m_block: Int32,
        n_block: Int32,
        tidx: Int32,
        thr_mma: cute.TiledMma,
        aux_tensors: list,
    ) -> None:
        prefix_lens = aux_tensors[0]
        prefix_len = Int32(prefix_lens[batch_idx])
        self.load_tree_mask_tile(
            sTreeMask,
            batch_idx=batch_idx,
            m_block=m_block,
            n_block=n_block,
            aux_tensors=aux_tensors,
            tidx=tidx,
        )

        acc_S_mn = utils.make_acc_tensor_mn_view(acc_S)
        cS = cute.make_identity_tensor((self.tile_m, self.tile_n))
        tScS_mn = utils.make_acc_tensor_mn_view(thr_mma.partition_C(cS))
        t0ScS_mn = utils.make_acc_tensor_mn_view(thr_mma.get_slice(0).partition_C(cS))
        thr_col_offset = tScS_mn[0][1]
        tree_col_offset = prefix_len - n_block * self.tile_n - thr_col_offset

        for r in cutlass.range(cute.size(tScS_mn.shape[0]), unroll_full=True):
            row_idx = tScS_mn[r, 0][0]
            for c in cutlass.range(cute.size(tScS_mn.shape[1]), unroll_full=True):
                col_idx = t0ScS_mn[0, c][1]
                full_col_idx = thr_col_offset + col_idx
                if col_idx >= tree_col_offset and sTreeMask[row_idx, full_col_idx] == 0:
                    acc_S_mn[r, c] = -Float32.inf

        cute.arch.barrier(
            barrier_id=int(NamedBarrierFwd.PEmpty),
            number_of_threads=self.num_mma_threads,
        )

    @cute.jit
    def apply_causal_tree_tail_mask(
        self,
        acc_S: cute.Tensor,
        sTreeMask: cute.Tensor,
        n_block: Int32,
        base_mask_fn,
        batch_idx: Int32,
        m_block: Int32,
        tidx: Int32,
        thr_mma: cute.TiledMma,
        aux_tensors: list,
        tree_start_n_block: Int32,
        mask_seqlen: cutlass.Constexpr = False,
    ) -> None:
        base_mask_fn(acc_S=acc_S, n_block=n_block, mask_seqlen=mask_seqlen)
        if n_block >= tree_start_n_block:
            self.apply_tree_tail_mask(
                acc_S,
                sTreeMask=sTreeMask,
                batch_idx=batch_idx,
                m_block=m_block,
                n_block=n_block,
                tidx=tidx,
                thr_mma=thr_mma,
                aux_tensors=aux_tensors,
            )

    @cute.jit
    def mma(
        self,
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        tiled_mma_pv_rs: cute.TiledMma,
        mQ: cute.Tensor,
        mO: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        sQ: cute.Tensor,
        sK: cute.Tensor,
        sVt: cute.Tensor,
        sP: Optional[cute.Tensor],
        sTreeMask: cute.Tensor,
        mGate: Optional[cute.Tensor],
        sO: cute.Tensor,
        learnable_sink: Optional[cute.Tensor],
        pipeline_k: cutlass.pipeline.PipelineAsync,
        pipeline_v: cutlass.pipeline.PipelineAsync,
        mbar_ptr_Q: cutlass.Pointer,
        gmem_tiled_copy_Q: cute.TiledCopy,
        gmem_tiled_copy_O: cute.TiledCopy,
        tma_atom_O: Optional[cute.CopyAtom],
        tidx: Int32,
        softmax_scale_log2: Float32,
        softmax_scale: Optional[Float32],
        block_info,
        SeqlenInfoCls,
        AttentionMaskCls,
        TileSchedulerCls,
        blocksparse_tensors,
        aux_tensors,
        fastdiv_mods=None,
    ):
        del fastdiv_mods
        warp_group_idx = cute.arch.make_warp_uniform(tidx // self.num_threads_per_warp_group)
        warp_group_thread_layout = cute.make_layout(
            self.num_mma_warp_groups, stride=self.num_threads_per_warp_group
        )
        thr_mma_qk = tiled_mma_qk.get_slice(tidx)
        wg_mma_qk = tiled_mma_qk.get_slice(warp_group_thread_layout(warp_group_idx))
        wg_mma_pv = tiled_mma_pv.get_slice(warp_group_thread_layout(warp_group_idx))
        tSrQ = tiled_mma_qk.make_fragment_A(wg_mma_qk.partition_A(sQ))
        tSrK = tiled_mma_qk.make_fragment_B(wg_mma_qk.partition_B(sK))
        if const_expr(self.mma_pv_is_rs):
            acc_S_shape = tiled_mma_qk.partition_shape_C((self.tile_m, self.tile_n))
            tOrP = cute.make_fragment(
                utils.convert_layout_acc_frgA(cute.make_layout(acc_S_shape)), self.dtype
            )
        else:
            tOrP = tiled_mma_pv.make_fragment_A(wg_mma_pv.partition_A(sP))
        tOrVt = tiled_mma_pv.make_fragment_B(wg_mma_pv.partition_B(sVt))
        cS = cute.make_identity_tensor((self.tile_m, self.tile_n))
        tScS_mn = utils.make_acc_tensor_mn_view(thr_mma_qk.partition_C(cS))

        smem_copy_atom_P = utils.get_smem_store_atom(self.arch, self.dtype)
        smem_thr_copy_P = cute.make_tiled_copy_C(smem_copy_atom_P, tiled_mma_qk).get_slice(tidx)
        tPsP = smem_thr_copy_P.partition_D(sP) if const_expr(sP is not None) else None

        self.mma_init()

        acc_shape_O = tiled_mma_pv.partition_shape_C((self.tile_m, self.tile_hdimv))
        acc_O = cute.make_fragment(acc_shape_O, Float32)
        smem_copy_params = SimpleNamespace(smem_thr_copy_P=smem_thr_copy_P, tPsP=tPsP)

        mma_qk_fn = partial(
            sm90_utils.gemm_zero_init, tiled_mma_qk, (self.tile_m, self.tile_n), tSrQ, tSrK
        )
        mma_pv_fn = partial(sm90_utils.gemm_w_idx, tiled_mma_pv, acc_O, tOrP, tOrVt)

        mma_one_n_block_all = partial(
            self.mma_one_n_block_intrawg_overlap
            if const_expr(self.intra_wg_overlap)
            else self.mma_one_n_block,
            mma_qk_fn=mma_qk_fn,
            tiled_mma_pv_rs=tiled_mma_pv_rs,
            pipeline_k=pipeline_k,
            pipeline_v=pipeline_v,
            acc_O=acc_O,
            tOrP=tOrP,
            smem_copy_params=smem_copy_params,
            check_inf=True,
        )

        q_consumer_phase = Int32(0)
        kv_consumer_state = pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Consumer, self.num_stages
        )

        tile_scheduler = TileSchedulerCls()
        num_splits = tile_scheduler.params.num_splits
        work_tile = tile_scheduler.initial_work_tile_info()
        softmax = Softmax.create(
            softmax_scale_log2,
            num_rows=acc_O.shape[0][0] * acc_O.shape[1],
            softmax_scale=softmax_scale,
        )

        process_first_half_block = partial(
            self.first_half_block_overlap,
            mma_qk_fn=mma_qk_fn,
            pipeline_k=pipeline_k,
            tOrP=tOrP,
            smem_copy_params=smem_copy_params,
            softmax=softmax,
        )
        process_last_half_block = partial(
            self.last_half_block_overlap,
            pipeline_v=pipeline_v,
            mma_pv_fn=mma_pv_fn,
        )
        while work_tile.is_valid_tile:
            m_block, head_idx, batch_idx, split_idx = work_tile.tile_idx
            seqlen = SeqlenInfoCls(batch_idx)
            gate_tile = None
            if const_expr(self.has_attention_gate):
                gate_batch = seqlen.offset_batch_Q(mGate, batch_idx, dim=1)[None, None, head_idx]
                gate_tile = cute.local_tile(gate_batch, (self.tile_m, 1), (m_block, 0))
            mask = AttentionMaskCls(seqlen.seqlen_q, seqlen.seqlen_k)
            mask_fn = partial(
                mask.apply_mask,
                batch_idx=batch_idx,
                head_idx=head_idx,
                m_block=m_block,
                thr_mma=thr_mma_qk,
                mask_causal=self.is_causal,
                mask_local=self.is_local,
                aux_tensors=aux_tensors,
                fastdiv_mods=None,
            )
            tree_start_n_block = (seqlen.seqlen_k - seqlen.seqlen_q) // self.tile_n
            tree_tail_mask_fn = partial(
                self.apply_causal_tree_tail_mask,
                base_mask_fn=mask_fn,
                batch_idx=batch_idx,
                m_block=m_block,
                tidx=tidx,
                thr_mma=thr_mma_qk,
                sTreeMask=sTreeMask,
                aux_tensors=aux_tensors,
                tree_start_n_block=tree_start_n_block,
            )
            tree_only_mask_fn = partial(
                self.apply_tree_tail_mask,
                sTreeMask=sTreeMask,
                batch_idx=batch_idx,
                m_block=m_block,
                tidx=tidx,
                thr_mma=thr_mma_qk,
                aux_tensors=aux_tensors,
            )
            score_mod_fn = None
            if const_expr(self.score_mod is not None):
                score_mod_fn = partial(
                    self.apply_score_mod,
                    thr_mma_qk,
                    batch_idx,
                    head_idx,
                    m_block,
                    softmax_scale=softmax_scale,
                    aux_tensors=aux_tensors,
                    fastdiv_mods=None,
                )
            mma_one_n_block = partial(
                mma_one_n_block_all,
                softmax=softmax,
                score_mod_fn=score_mod_fn,
            )
            if const_expr(not self.use_tma_Q):
                pack_gqa = PackGQA(
                    self.tile_m, self.tile_hdim, self.check_hdim_oob, self.qhead_per_kvhead
                )
                mQ_cur = seqlen.offset_batch_Q(mQ, batch_idx, dim=1)[None, None, head_idx]
                pack_gqa.load_Q(mQ_cur, sQ, gmem_tiled_copy_Q, tidx, m_block, seqlen.seqlen_q)
                cute.arch.cp_async_mbarrier_arrive_noinc(mbar_ptr_Q)

            n_block_min, n_block_max = block_info.get_n_block_min_max(
                seqlen, m_block, split_idx, num_splits
            )
            if const_expr(not self.use_tma_Q):
                cute.arch.mbarrier_wait(mbar_ptr_Q, phase=q_consumer_phase)
            q_consumer_phase ^= 1
            O_should_accumulate = False

            if const_expr(not self.use_block_sparsity):
                if const_expr(not self.is_split_kv) or n_block_min < n_block_max:
                    if const_expr(self.intra_wg_overlap):
                        kv_consumer_state = process_first_half_block(
                            n_block=n_block_max - 1,
                            kv_consumer_state=kv_consumer_state,
                            mask_fn=tree_tail_mask_fn,
                            score_mod_fn=score_mod_fn,
                            is_first_block=True,
                        )
                    else:
                        self.warp_scheduler_barrier_sync()
                        kv_consumer_state = mma_one_n_block(
                            kv_consumer_state,
                            n_block=n_block_max - 1,
                            mma_pv_fn=partial(mma_pv_fn, zero_init=True),
                            is_first_n_block=True,
                            mask_fn=partial(tree_tail_mask_fn, mask_seqlen=True),
                        )
                        O_should_accumulate = True
                    n_block_max -= 1
                    if const_expr(self.is_causal or self.is_local):
                        n_block_min_causal_local_mask = block_info.get_n_block_min_causal_local_mask(
                            seqlen, m_block, n_block_min
                        )
                        for n_tile in cutlass.range(
                            n_block_max - n_block_min_causal_local_mask, unroll=1
                        ):
                            kv_consumer_state = mma_one_n_block(
                                kv_consumer_state,
                                n_block=n_block_max - 1 - n_tile,
                                mma_pv_fn=partial(mma_pv_fn, zero_init=not O_should_accumulate),
                                mask_fn=partial(tree_tail_mask_fn, mask_seqlen=False),
                            )
                            O_should_accumulate = True
                        n_block_max = cutlass.min(n_block_max, n_block_min_causal_local_mask)
                    n_block_min_before_local_mask = block_info.get_n_block_min_before_local_mask(
                        seqlen, m_block, n_block_min
                    )
                    tree_only_n_block_min = cutlass.max(
                        n_block_min_before_local_mask, tree_start_n_block
                    )
                    # Note(wangbojun/codex): prefix_len may cut into an otherwise dense block,
                    # so tree-only masking needs its own pass after the causal tail.
                    for n_tile in cutlass.range(n_block_max - tree_only_n_block_min, unroll=1):
                        kv_consumer_state = mma_one_n_block(
                            kv_consumer_state,
                            n_block=n_block_max - 1 - n_tile,
                            mma_pv_fn=partial(mma_pv_fn, zero_init=not O_should_accumulate),
                            mask_fn=tree_only_mask_fn,
                        )
                        O_should_accumulate = True
                    n_block_max = cutlass.min(n_block_max, tree_only_n_block_min)
                    for n_tile in cutlass.range(
                        n_block_max - n_block_min_before_local_mask, unroll=1
                    ):
                        kv_consumer_state = mma_one_n_block(
                            kv_consumer_state,
                            n_block=n_block_max - 1 - n_tile,
                            mma_pv_fn=partial(mma_pv_fn, zero_init=not O_should_accumulate),
                        )
                        O_should_accumulate = True
                    if const_expr(self.is_local and block_info.window_size_left is not None):
                        n_block_max = cutlass.min(n_block_max, n_block_min_before_local_mask)
                        for n_tile in cutlass.range(n_block_max - n_block_min, unroll=1):
                            kv_consumer_state = mma_one_n_block(
                                kv_consumer_state,
                                n_block=n_block_max - 1 - n_tile,
                                mma_pv_fn=partial(mma_pv_fn, zero_init=not O_should_accumulate),
                                mask_fn=partial(tree_tail_mask_fn, mask_seqlen=False),
                            )
                            O_should_accumulate = True

                    if const_expr(self.intra_wg_overlap):
                        kv_consumer_state = process_last_half_block(
                            kv_consumer_state=kv_consumer_state,
                            zero_init=not O_should_accumulate,
                        )
                        O_should_accumulate = True
                    else:
                        self.warp_scheduler_barrier_arrive()
                else:
                    softmax.reset()
                    acc_O.fill(0.0)
            else:
                kv_consumer_state, O_should_accumulate, processed_any = consume_block_sparse_loads(
                    blocksparse_tensors,
                    batch_idx,
                    head_idx,
                    m_block,
                    kv_consumer_state,
                    mma_pv_fn,
                    mma_one_n_block,
                    process_first_half_block,
                    process_last_half_block,
                    tree_tail_mask_fn,
                    score_mod_fn,
                    O_should_accumulate,
                    self.mask_mod,
                    None,
                    self.intra_wg_overlap,
                    self.warp_scheduler_barrier_sync,
                    self.warp_scheduler_barrier_arrive,
                )
                if not processed_any:
                    softmax.reset()
                    acc_O.fill(0.0)

            sink_val = None
            if const_expr(learnable_sink is not None):
                if const_expr(not self.pack_gqa):
                    sink_val = Float32(learnable_sink[head_idx])
                else:
                    sink_val = cute.make_fragment_like(softmax.row_max, Float32)
                    for r in cutlass.range(cute.size(sink_val), unroll_full=True):
                        row = m_block * self.tile_m + tScS_mn[r][0]
                        q_head_idx = row % self.qhead_per_kvhead + head_idx * self.qhead_per_kvhead
                        sink_val[r] = Float32(learnable_sink[q_head_idx])

            row_scale = softmax.finalize(sink_val=sink_val)
            if const_expr(self.has_attention_gate):
                row_scale = self.apply_attention_gate(
                    row_scale, gate_tile, tScS_mn, m_block, seqlen.seqlen_q
                )
            softmax.rescale_O(acc_O, row_scale)

            self.epilogue(
                acc_O,
                softmax.row_sum,
                mO,
                mLSE,
                sO,
                seqlen,
                gmem_tiled_copy_O,
                tma_atom_O,
                tiled_mma_pv,
                tidx,
                m_block,
                head_idx,
                batch_idx,
                split_idx,
            )

            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()

    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mO: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        mGate: Optional[cute.Tensor],
        mCuSeqlensQ: Optional[cute.Tensor],
        mCuSeqlensK: Optional[cute.Tensor],
        mSeqUsedQ: Optional[cute.Tensor],
        mSeqUsedK: Optional[cute.Tensor],
        mPageTable: cute.Tensor,
        tma_atom_Q: Optional[cute.CopyAtom],
        tma_atom_K: Optional[cute.CopyAtom],
        tma_atom_V: Optional[cute.CopyAtom],
        tma_atom_O: Optional[cute.CopyAtom],
        softmax_scale_log2: Float32,
        softmax_scale: Optional[Float32],
        window_size_left: Optional[Int32],
        window_size_right: Optional[Int32],
        learnable_sink: Optional[cute.Tensor],
        blocksparse_tensors: Optional[BlockSparseTensors],
        sQ_layout: cute.ComposedLayout,
        sK_layout: cute.ComposedLayout,
        sV_layout: cute.ComposedLayout,
        sO_layout: cute.ComposedLayout,
        sP_layout: cute.ComposedLayout | None,
        gmem_tiled_copy_Q: cute.TiledCopy,
        gmem_tiled_copy_K: cute.TiledCopy,
        gmem_tiled_copy_V: cute.TiledCopy,
        gmem_tiled_copy_O: cute.TiledCopy,
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        tiled_mma_pv_rs: cute.TiledMma,
        tile_sched_params: ParamsBase,
        TileScheduler: cutlass.Constexpr[Callable],
        SharedStorage: cutlass.Constexpr[Callable],
        aux_tensors=Optional[list[cute.Tensor]],
        fastdiv_mods=None,
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        if warp_idx == 0:
            for tma_atom in (tma_atom_Q, tma_atom_K, tma_atom_V, tma_atom_O):
                if const_expr(tma_atom is not None):
                    cpasync.prefetch_descriptor(tma_atom)

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)

        mbar_ptr_Q = storage.mbar_ptr.data_ptr()
        if warp_idx == 1:
            if const_expr(not self.use_tma_Q):
                cute.arch.mbarrier_init(mbar_ptr_Q, self.num_Q_load_threads)

        pipeline_kv_producer_group = cutlass.pipeline.CooperativeGroup(
            cutlass.pipeline.Agent.Thread
        )
        pipeline_kv_consumer_group = cutlass.pipeline.CooperativeGroup(
            cutlass.pipeline.Agent.Thread, self.num_mma_threads // self.num_threads_per_warp_group
        )
        pipeline_k = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.mbar_ptr_K.data_ptr(),
            num_stages=self.num_stages,
            producer_group=pipeline_kv_producer_group,
            consumer_group=pipeline_kv_consumer_group,
            tx_count=self.tma_copy_bytes["K"],
            init_wait=False,
        )
        pipeline_v = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.mbar_ptr_V.data_ptr(),
            num_stages=self.num_stages,
            producer_group=pipeline_kv_producer_group,
            consumer_group=pipeline_kv_consumer_group,
            tx_count=self.tma_copy_bytes["V"],
        )

        sQ = storage.sQ.get_tensor(sQ_layout.outer, swizzle=sQ_layout.inner)
        sK = storage.sK.get_tensor(sK_layout.outer, swizzle=sK_layout.inner)
        if const_expr(not self.Q_in_regs):
            sV = storage.sV.get_tensor(sV_layout.outer, swizzle=sV_layout.inner)
        else:
            sV = storage.sQ.get_tensor(
                sV_layout.outer, swizzle=sV_layout.inner, dtype=mV.element_type
            )
        sVt = utils.transpose_view(sV)
        sP = None
        if const_expr(sP_layout is not None):
            sP = storage.sP.get_tensor(sP_layout.outer, swizzle=sP_layout.inner)
        sO = storage.sQ.get_tensor(sO_layout.outer, swizzle=sO_layout.inner, dtype=self.dtype)
        sTreeMask = storage.sTreeMask.get_tensor(
            cute.make_layout((self.tile_m, self.tile_n), stride=(self.tile_n, 1))
        )

        block_info = BlockInfo(
            self.tile_m,
            self.tile_n,
            self.is_causal,
            self.is_local,
            self.is_split_kv,
            window_size_left,
            window_size_right,
            qhead_per_kvhead_packgqa=self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1,
        )
        SeqlenInfoCls = partial(
            SeqlenInfoQK.create,
            seqlen_q_static=mQ.shape[0] if const_expr(not self.pack_gqa) else mQ.shape[0][1],
            seqlen_k_static=mK.shape[0],
            mCuSeqlensQ=mCuSeqlensQ,
            mCuSeqlensK=mCuSeqlensK,
            mSeqUsedQ=mSeqUsedQ,
            mSeqUsedK=mSeqUsedK,
        )
        AttentionMaskCls = partial(
            AttentionMask,
            self.tile_m,
            self.tile_n,
            window_size_left=window_size_left,
            window_size_right=window_size_right,
            qhead_per_kvhead_packgqa=self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1,
        )
        TileSchedulerCls = partial(TileScheduler.create, tile_sched_params)
        if warp_idx < 4:
            cute.arch.warpgroup_reg_dealloc(self.num_producer_regs)
            self.load(
                mQ,
                mK,
                mV,
                sQ,
                sK,
                sV,
                tma_atom_Q,
                tma_atom_K,
                tma_atom_V,
                pipeline_k,
                pipeline_v,
                mbar_ptr_Q,
                blocksparse_tensors,
                mPageTable,
                block_info,
                SeqlenInfoCls,
                TileSchedulerCls,
            )
        else:
            cute.arch.warpgroup_reg_alloc(self.num_mma_regs)
            tidx, _, _ = cute.arch.thread_idx()
            tidx = tidx - 128
            self.mma(
                tiled_mma_qk,
                tiled_mma_pv,
                tiled_mma_pv_rs,
                mQ,
                mO,
                mLSE,
                sQ,
                sK,
                sVt,
                sP,
                sTreeMask,
                mGate,
                sO,
                learnable_sink,
                pipeline_k,
                pipeline_v,
                mbar_ptr_Q,
                gmem_tiled_copy_Q,
                gmem_tiled_copy_O,
                tma_atom_O,
                tidx,
                softmax_scale_log2,
                softmax_scale,
                block_info,
                SeqlenInfoCls,
                AttentionMaskCls,
                TileSchedulerCls,
                blocksparse_tensors,
                aux_tensors,
                fastdiv_mods,
            )


def _normalize_tree_aux_tensors(
    tree_mask: torch.Tensor,
    batch_size: int,
    prefix_lens: torch.Tensor,
    m_block_size: int,
    n_block_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if tree_mask.dim() == 2:
        tree_mask = tree_mask.unsqueeze(0)
    if tree_mask.dim() != 3:
        raise ValueError("tree_mask must have shape (N, N) or (B, N, N)")
    if tree_mask.shape[1] != tree_mask.shape[2]:
        raise ValueError("tree_mask must be square over the tree dimension")
    if tree_mask.shape[0] == 1 and batch_size != 1:
        tree_mask = tree_mask.expand(batch_size, -1, -1)
    if tree_mask.shape[0] != batch_size:
        raise ValueError("tree_mask batch dimension must be 1 or equal to batch size")
    tree_mask = tree_mask.to(dtype=torch.int32, device=prefix_lens.device)
    padded_rows = _round_up(tree_mask.shape[1], m_block_size)
    padded_cols = _round_up(tree_mask.shape[2], n_block_size)
    if padded_rows != tree_mask.shape[1] or padded_cols != tree_mask.shape[2]:
        tree_mask = torch.nn.functional.pad(
            tree_mask,
            (0, padded_cols - tree_mask.shape[2], 0, padded_rows - tree_mask.shape[1]),
        )
    tree_prefix_lens = prefix_lens.new_empty(batch_size)
    tree_prefix_lens.copy_(prefix_lens)
    return tree_prefix_lens, tree_mask.contiguous()


def flash_attn_varlen_tree_paged_sm90(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    tree_mask: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    seqused_k: torch.Tensor,
    page_table: torch.Tensor,
    *,
    softmax_scale: Optional[float] = None,
    pack_gqa: Optional[bool] = None,
    out: Optional[torch.Tensor] = None,
    lse: Optional[torch.Tensor] = None,
    return_lse: bool = False,
    m_block_size: int = 128,
    n_block_size: int = 128,
    num_threads: int = 384,
    _compute_capability: Optional[int] = None,
):
    q, k, v = [maybe_contiguous(t) for t in (q, k, v)]
    if q.ndim != 3:
        raise ValueError("q must have shape (total_q, num_heads, head_dim)")
    if k.ndim != 4 or v.ndim != 4:
        raise ValueError("k and v must have paged KV shape (num_pages, page_size, num_head_kv, head_dim)")
    if cu_seqlens_q.ndim != 1 or seqused_k.ndim != 1:
        raise ValueError("cu_seqlens_q and seqused_k must be 1-D int32 tensors")
    if page_table.ndim != 2:
        raise ValueError("page_table must have shape (batch, max_num_pages_per_seq)")

    device = q.device
    batch_size = cu_seqlens_q.shape[0] - 1
    total_q, num_head, head_dim = q.shape
    num_pages, page_size, num_head_kv, _ = k.shape
    head_dim_v = v.shape[-1]

    if batch_size <= 0:
        raise ValueError("cu_seqlens_q must contain at least one sequence")
    if page_size != n_block_size:
        raise ValueError("paged SM90 tree kernel requires page_size == n_block_size")
    if q.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("q, k, v must use fp16 or bf16")
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise ValueError("q, k, v must have the same dtype")
    if not (q.is_cuda and k.is_cuda and v.is_cuda):
        raise ValueError("tree paged SM90 path requires CUDA q/k/v tensors")
    if not (cu_seqlens_q.is_cuda and seqused_k.is_cuda and page_table.is_cuda):
        raise ValueError("tree paged SM90 path requires CUDA metadata tensors")
    if cu_seqlens_q.dtype != torch.int32 or seqused_k.dtype != torch.int32:
        raise ValueError("cu_seqlens_q and seqused_k must be int32")
    if page_table.dtype != torch.int32:
        raise ValueError("page_table must be int32")
    if num_head % num_head_kv != 0:
        raise ValueError("num_head must be divisible by num_head_kv")
    if page_table.shape[0] != batch_size or seqused_k.shape[0] != batch_size:
        raise ValueError("metadata batch size does not match cu_seqlens_q")

    compute_capability = (
        torch.cuda.get_device_capability()[0]
        if _compute_capability is None
        else _compute_capability
    )
    if compute_capability != 9:
        raise ValueError("flash_attn_varlen_tree_paged_sm90 only supports SM90")

    dtype = torch2cute_dtype_map[q.dtype]
    qhead_per_kvhead = num_head // num_head_kv
    if pack_gqa is None:
        pack_gqa = qhead_per_kvhead > 1
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)

    if out is None:
        out = torch.empty(total_q, num_head, head_dim_v, dtype=q.dtype, device=device)
    if return_lse and lse is None:
        lse = torch.empty(num_head, total_q, dtype=torch.float32, device=device)

    q_lens = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
    prefix_lens = seqused_k - q_lens
    prefix_lens, tree_mask = _normalize_tree_aux_tensors(
        tree_mask,
        batch_size=batch_size,
        prefix_lens=prefix_lens,
        m_block_size=m_block_size,
        n_block_size=n_block_size,
    )

    current_stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    q_tensor, k_tensor, v_tensor, o_tensor = [
        from_dlpack(t.detach(), assumed_align=16).mark_layout_dynamic(leading_dim=t.ndim - 1)
        for t in (q, k, v, out)
    ]
    lse_tensor = (
        from_dlpack(lse.detach(), assumed_align=4).mark_layout_dynamic(leading_dim=lse.ndim - 1)
        if lse is not None
        else None
    )
    cu_seqlens_q_tensor = from_dlpack(cu_seqlens_q.detach(), assumed_align=4).mark_layout_dynamic(
        leading_dim=0
    )
    seqused_k_tensor = from_dlpack(seqused_k.detach(), assumed_align=4).mark_layout_dynamic(
        leading_dim=0
    )
    page_table_tensor = from_dlpack(page_table.detach(), assumed_align=4).mark_layout_dynamic(
        leading_dim=1
    )
    prefix_lens_tensor = from_dlpack(prefix_lens.detach(), assumed_align=4).mark_layout_dynamic(
        leading_dim=0
    )
    tree_mask_tensor = from_dlpack(tree_mask.detach(), assumed_align=4).mark_layout_dynamic(
        leading_dim=2
    )
    cute_aux_tensors = [prefix_lens_tensor, tree_mask_tensor]

    compile_key = (
        dtype,
        head_dim,
        head_dim_v,
        qhead_per_kvhead,
        pack_gqa,
        lse is None,
        m_block_size,
        n_block_size,
        num_threads,
        "paged_tree_sm90_v7",
    )
    if compile_key not in flash_attn_varlen_tree_paged_sm90.compile_cache:
        fa_fwd = FlashAttentionForwardPagedTreeSM90(
            dtype,
            head_dim,
            head_dim_v,
            qhead_per_kvhead,
            is_causal=True,
            is_local=False,
            pack_gqa=pack_gqa,
            tile_m=m_block_size,
            tile_n=n_block_size,
            num_stages=2,
            num_threads=num_threads,
            Q_in_regs=False,
            intra_wg_overlap=True,
            mma_pv_is_rs=True,
            is_split_kv=False,
            has_aux_tensors=False,
        )
        flash_attn_varlen_tree_paged_sm90.compile_cache[compile_key] = cute.compile(
            fa_fwd,
            q_tensor,
            k_tensor,
            v_tensor,
            o_tensor,
            lse_tensor,
            softmax_scale,
            current_stream,
            cu_seqlens_q_tensor,
            None,
            None,
            seqused_k_tensor,
            page_table_tensor,
            None,
            None,
            None,
            None,
            cute_aux_tensors,
            None,
        )
    flash_attn_varlen_tree_paged_sm90.compile_cache[compile_key](
        q_tensor,
        k_tensor,
        v_tensor,
        o_tensor,
        lse_tensor,
        softmax_scale,
        current_stream,
        cu_seqlens_q_tensor,
        None,
        None,
        seqused_k_tensor,
        page_table_tensor,
        None,
        None,
        None,
        None,
        cute_aux_tensors,
        None,
    )
    if return_lse:
        return out, lse
    return out


flash_attn_varlen_tree_paged_sm90.compile_cache = {}
