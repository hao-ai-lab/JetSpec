# Copyright (c) 2025, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
# [2025-07-04] Version in Cute-DSL, for Hopper and Blackwell. You'll need install nvidia-cutlass-dsl==4.2.0.
# [2025-07-04] Version in Cute-DSL, for Hopper and Blackwell. You'll need install nvidia-cutlass-dsl==4.2.0.
# [2025-07-04] Version in Cute-DSL, for Hopper and Blackwell. You'll need install nvidia-cutlass-dsl==4.2.0.

# Supported features:
# - BF16 & FP16 dtype
# - noncausal & causal attention
# - MHA, GQA, MQA
# - hdim 64, 96, 128.
# - (hdim_qk, hdim_v) = (192, 128) for Blackwell (i.e. DeepSeek shape)
# - varlen
# - sliding window
# - bwd pass for Ampere (will also run on Hopper/Blackwell, but will be slow)

# Features not supported yet:
# - split (i.e. FlashDecoding)
# - tuned block sizes
# - paged KV
# - append KV to existing KV cache
# - FP8
# - bwd pass optimized for Hopper/Blackwell

import math
import struct
import os
from typing import Optional, Tuple, Callable

import torch

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

from optimus_cutedsl.flash_attn import utils
from optimus_cutedsl.flash_attn.flash_fwd import (
    FlashAttentionForwardSm80,
    FlashAttentionForwardSm90,
)
from optimus_cutedsl.flash_attn.flash_fwd_sm90_paged import FlashAttentionForwardPagedSM90
from optimus_cutedsl.flash_attn.flash_fwd_sm100 import FlashAttentionForwardSm100
from optimus_cutedsl.flash_attn.flash_bwd_preprocess import FlashAttentionBackwardPreprocess
from optimus_cutedsl.flash_attn.flash_bwd import FlashAttentionBackwardSm80
from optimus_cutedsl.flash_attn.flash_bwd_sm90 import FlashAttentionBackwardSm90
from optimus_cutedsl.flash_attn.flash_bwd_sm100 import FlashAttentionBackwardSm100
from optimus_cutedsl.flash_attn.flash_bwd_postprocess import FlashAttentionBackwardPostprocess
from optimus_cutedsl.flash_attn.flash_fwd_combine_varlen import FlashAttentionForwardCombineVarlen
# from optimus_cutedsl.flash_attn.flash_token_sparse_fwd import (
#     TokenSparseFlashAttentionSm90,
#     TokenSparseTensors,
# )

from optimus_cutedsl.flash_attn.block_sparsity import (
    BlockSparseTensorsTorch,
    to_cute_block_sparse_tensors,
    normalize_block_sparse_tensors,
)

def maybe_contiguous(x):
    return x.contiguous() if x is not None and x.stride(-1) != 1 else x


torch2cute_dtype_map = {
    torch.float16: cutlass.Float16,
    torch.bfloat16: cutlass.BFloat16,
    torch.float32: cutlass.Float32,
}


def _reset_split_workspace(out_partial: torch.Tensor, lse_partial: torch.Tensor, stream: cuda.CUstream) -> None:
    """Reset SplitKV workspaces asynchronously using CUDA driver memset APIs."""
    stream_handle = int(stream)
    if out_partial is not None and out_partial.numel() > 0:
        bytes_out = out_partial.numel() * out_partial.element_size()
        cuda.cuMemsetD8Async(
            out_partial.data_ptr(),
            0,
            bytes_out,
            stream_handle,
        )
    if lse_partial is not None and lse_partial.numel() > 0:
        pattern = struct.unpack("<I", struct.pack("<f", float("-inf")))[0]
        cuda.cuMemsetD32Async(
            lse_partial.data_ptr(),
            pattern,
            lse_partial.numel(),
            stream_handle,
        )


def num_splits_heuristic(total_mblocks, num_SMs, num_n_blocks, max_splits):
    # If num_n_blocks is too small, use 1 split. For example, we never split for hdim = 128 and seqlen_k = 512.
    if num_n_blocks <= 4:
        return 1

    # NOTE: We should revisit this heuristic after persistence is supported for split KV.
    # Sometimes, it's ideal to over-schedule splits for better efficiency.
    return min(num_SMs // total_mblocks, max_splits, num_n_blocks)


def _flash_attn_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
    page_table: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    softcap: Optional[float] = None,
    window_size_left: Optional[int] = None,
    window_size_right: Optional[int] = None,
    learnable_sink: Optional[torch.Tensor] = None,
    # m_block_size: int = 128,
    # n_block_size: int = 64,
    # num_threads: int = 128,
    m_block_size: int = 128,
    n_block_size: int = 128,
    num_threads: int = 384,
    num_splits: int = 1,
    pack_gqa: Optional[bool] = None,
    _compute_capability: Optional[int] = None,
    score_mod: Optional[Callable] = None,
    mask_mod: Optional[Callable] = None,
    block_sparse_tensors: Optional[BlockSparseTensorsTorch] = None,
    return_lse: bool = False,
    out: Optional[torch.Tensor] = None,
    lse: Optional[torch.Tensor] = None,
    aux_tensors: Optional[list[torch.Tensor]] = None,
    attention_gate: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Forward pass for FlashAttention.

    Args:
        ...
        score_mod: A callable that takes the attention scores and applies a modification.
        mask_mod: A callable that takes token position information and selectively masks
        block_sparse_tensors: A tuple of tensors used for block sparsity. 
        return_lse: Whether to return the log softmax of the attention scores. If set to True will always calculate
        out: Optional pre-allocated output tensor. If None, will be allocated internally.
        lse: Optional pre-allocated log-sum-exp tensor. If None, will be allocated when needed.
        aux_tensors: Some score_mods will want to read from global aux_tensors. This is how we thread them through to the inner kernel.
    """
    q, k, v = [maybe_contiguous(t) for t in (q, k, v)]
    num_head, head_dim = q.shape[-2:]
    if cu_seqlens_q is None:
        batch_size, seqlen_q = q.shape[:2]
        total_q = batch_size * seqlen_q
    else:
        batch_size = cu_seqlens_q.shape[0] - 1
        seqlen_q = None
        total_q = q.shape[0]
    if page_table is not None:
        assert cu_seqlens_k is None, "page_table is not supported with cu_seqlens_k"
        assert page_table.dtype == torch.int32, "page_table must be int32"
        assert page_table.stride(-1) == 1, "page_table must be contiguous in the last dimension"
        max_num_pages_per_seq = page_table.shape[1]
        assert page_table.shape == (batch_size, max_num_pages_per_seq)
        num_pages, page_size = k.shape[:2]
        seqlen_k = num_pages * page_size
        if cu_seqlens_q is None:
            raise ValueError("page_table requires cu_seqlens_q for varlen decode.")
    else:
        num_pages, page_size = None, None
        seqlen_k = k.shape[-3]
    num_head_kv = k.shape[-2]
    head_dim_v = v.shape[-1]
    if cu_seqlens_k is None:
        if page_table is None:
            assert k.shape == (batch_size, seqlen_k, num_head_kv, head_dim)
            assert v.shape == (batch_size, seqlen_k, num_head_kv, head_dim_v)
        else:
            assert k.shape == (num_pages, page_size, num_head_kv, head_dim)
            assert v.shape == (num_pages, page_size, num_head_kv, head_dim_v)
    else:
        assert k.shape == (seqlen_k, num_head_kv, head_dim)
        assert v.shape == (seqlen_k, num_head_kv, head_dim_v)
        assert cu_seqlens_k.shape == (batch_size + 1,), (
            "cu_seqlens_k must have shape (batch_size + 1,)"
        )

    if cu_seqlens_q is not None:
        assert cu_seqlens_q.shape == (batch_size + 1,), (
            "cu_seqlens_q must have shape (batch_size + 1,)"
        )
    assert seqused_q is None or seqused_q.shape == (batch_size,), (
        "seqused_q must have shape (batch_size,)"
    )
    assert seqused_k is None or seqused_k.shape == (batch_size,), (
        "seqused_k must have shape (batch_size,)"
    )
    assert q.dtype in [torch.float16, torch.bfloat16], "inputs must be float16 or bfloat16"
    assert q.dtype == k.dtype == v.dtype, "inputs must have the same dtype"
    for t in [cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k]:
        if t is not None:
            assert t.dtype == torch.int32, (
                "cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k must be int32"
            )
            assert t.stride(0) == 1, (
                "cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k must be contiguous"
            )
    if learnable_sink is not None:
        assert learnable_sink.shape == (num_head,)
        assert learnable_sink.dtype == torch.bfloat16, "learnable_sink must be bfloat16"
    if attention_gate is not None:
        assert attention_gate.dtype == torch.float32, "attention_gate must be float32"
        assert attention_gate.shape[-1] == num_head, "attention_gate last dim must equal num_head"
        if cu_seqlens_q is None:
            assert attention_gate.shape == (batch_size, seqlen_q, num_head), (
                f"attention_gate must have shape {(batch_size, seqlen_q, num_head)}"
            )
        else:
            assert attention_gate.shape == (total_q, num_head), (
                f"attention_gate must have shape {(total_q, num_head)} when using cu_seqlens_q"
            )
        attention_gate = attention_gate.unsqueeze(-1).contiguous()
    assert all(
        t is None or t.is_cuda
        for t in (
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            seqused_q,
            seqused_k,
            page_table,
            learnable_sink,
            attention_gate,
        )
    ), "inputs must be on CUDA device"
    assert num_head % num_head_kv == 0, "num_head must be divisible by num_head_kv"
    assert head_dim <= 256, "head_dim must be less than or equal to 256"
    alignment = 16 // q.element_size()
    assert head_dim % alignment == 0, f"head_dim must be divisible by {alignment}"
    assert head_dim_v % alignment == 0, f"head_dim_v must be divisible by {alignment}"
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)
    if softcap == 0.0:
        softcap = None
    qhead_per_kvhead = num_head // num_head_kv
    if pack_gqa is None:
        pack_gqa = qhead_per_kvhead > 1

    out_torch_dtype = q.dtype
    device = q.device
    q_batch_seqlen_shape = (batch_size, seqlen_q) if cu_seqlens_q is None else (total_q,)
    lse_shape = (batch_size, num_head, seqlen_q) if cu_seqlens_q is None else (num_head, total_q)
    requires_grad = q.requires_grad or k.requires_grad or v.requires_grad

    if out is None:
        out = torch.empty(
            *q_batch_seqlen_shape, num_head, head_dim_v, dtype=out_torch_dtype, device=device
        )
    else:
        expected_out_shape = (*q_batch_seqlen_shape, num_head, head_dim_v)
        assert out.shape == expected_out_shape, (
            f"out tensor shape {out.shape} does not match expected shape {expected_out_shape}"
        )
        assert out.dtype == out_torch_dtype, (
            f"out tensor dtype {out.dtype} does not match expected dtype {out_torch_dtype}"
        )
        assert out.device == device, (
            f"out tensor device {out.device} does not match input device {device}"
        )
        assert out.is_cuda, "out tensor must be on CUDA device"

    if lse is None:
        lse = (
            torch.empty(lse_shape, dtype=torch.float32, device=device)
            if requires_grad or return_lse
            else None
        )
    elif lse is not None:
        assert lse.shape == lse_shape, (
            f"lse tensor shape {lse.shape} does not match expected shape {lse_shape}"
        )
        assert lse.dtype == torch.float32, (
            f"lse tensor dtype {lse.dtype} does not match expected dtype torch.float32"
        )
        assert lse.device == device, (
            f"lse tensor device {lse.device} does not match input device {device}"
        )
        assert lse.is_cuda, "lse tensor must be on CUDA device"

    dtype = torch2cute_dtype_map[q.dtype]
    if page_table is not None and cu_seqlens_k is None and seqused_k is None:
        raise ValueError(
            "Paged KV requires per-sequence key lengths; provide cu_seqlens_k or seqused_k."
        )

    (
        cu_seqlens_q_tensor,
        cu_seqlens_k_tensor,
        seqused_q_tensor,
        seqused_k_tensor,
        learnable_sink_tensor,
    ) = [
        from_dlpack(t.detach(), assumed_align=4).mark_layout_dynamic(leading_dim=0)
        if t is not None
        else None
        for t in (cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k, learnable_sink)
    ]
    page_table_tensor = (
        from_dlpack(page_table.detach(), assumed_align=4).mark_layout_dynamic(leading_dim=1)
        if page_table is not None
        else None
    )
    compute_capability = (
        torch.cuda.get_device_capability()[0]
        if _compute_capability is None
        else _compute_capability
    )

    assert compute_capability in [9, 10], "Unsupported compute capability. Supported: 9.x, 10.x"
    if attention_gate is not None and compute_capability != 9:
        raise NotImplementedError("attention_gate is only supported on SM90 (compute capability 9.x)")


    sparse_tensors = None
    if block_sparse_tensors is not None:
        if seqlen_q is None:
            raise ValueError("Block sparsity requires fixed-length sequences (seqlen_q must be known).")
        m_block_size_block = m_block_size
        if compute_capability == 10:
            # TODO: This multiplier should really be q_stage, wire up in later PR
            # 1 cta handles 2*tile_m row
            m_block_size_block = 2 * m_block_size
        expected_m_blocks = (seqlen_q + m_block_size_block - 1) // m_block_size_block
        expected_n_blocks = (seqlen_k + n_block_size - 1) // n_block_size
        block_sparse_tensors = normalize_block_sparse_tensors(
            block_sparse_tensors,
            expected_count_shape=(batch_size, num_head, expected_m_blocks),
            expected_index_shape=(batch_size, num_head, expected_m_blocks, expected_n_blocks),
        )
        sparse_tensors = to_cute_block_sparse_tensors(block_sparse_tensors)

    use_block_sparsity = sparse_tensors is not None

    if mask_mod is None:
        if causal:
            window_size_right = 0
        local = window_size_left is not None or window_size_right is not None
        if window_size_left is not None or window_size_right is not None:
            if window_size_left is None and window_size_right == 0:
                causal, local = True, False
            else:
                causal, local = False, True
    else:
        causal, local = False, False

    current_stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    if compute_capability == 9:  # TODO: tune block size according to hdim.
        if head_dim == head_dim_v == 128 and not causal and not local and not use_block_sparsity:
            n_block_size = 192
    if compute_capability == 10:
        # TODO: fix the varlen case
        if (
            pack_gqa
            and (128 % qhead_per_kvhead != 0)
            or (cu_seqlens_q is not None or seqused_q is not None)
        ):
            pack_gqa = False
        # TODO: fix GQA + SplitKV + non-varlen
        if pack_gqa and num_splits != 1 and cu_seqlens_q is None:
            pack_gqa = False

    if attention_gate is not None and pack_gqa:
        raise NotImplementedError("attention_gate is not yet supported with pack_gqa=True")

    if (
        attention_gate is not None
        and pack_gqa
        and (m_block_size % qhead_per_kvhead != 0)
    ):
        raise NotImplementedError(
            "attention_gate requires use_tma_Q=True (disable pack_gqa or choose tile_m divisible by qhead_per_kvhead)"
        )

    if num_splits < 1:
        # TODO[wangbojun]: optimize num_splits for varlen case,
        # We need to update scheduler to support varlen case gpu loadding balance.
        # Removing cpu behavior to fully support cuda-graph
        max_seqlen_k = (
            seqlen_k
            if cu_seqlens_k is None
            else (k.shape[0] + batch_size - 1) // batch_size
        )
        max_seqlen_q = (
            seqlen_q
            if cu_seqlens_q is None
            else (total_q + batch_size - 1) // batch_size
        )
        seqlen_q_packgqa = max_seqlen_q * qhead_per_kvhead
        seqlen_k_loaded = max_seqlen_k if not local else max(0, min(max_seqlen_k, window_size_right + window_size_left + 1 + m_block_size))
        num_n_blocks = (seqlen_k_loaded + n_block_size - 1) // n_block_size
        num_m_blocks = (seqlen_q_packgqa + m_block_size - 1) // m_block_size
        total_mblocks = batch_size * num_head_kv * num_m_blocks
        num_splits = num_splits_heuristic(
            total_mblocks,
            torch.cuda.get_device_properties(device).multi_processor_count,
            num_n_blocks,
            128,
        )

    is_split_kv = num_splits > 1
    if os.getenv("FLASH_ATTN_DEBUG_SPLITS") == "1":
        print(
            f"[flash_attn] is_split_kv={is_split_kv} (requested num_splits={num_splits}) "
            f"batch={batch_size} total_q={total_q} seqlen_k={seqlen_k} m_block={m_block_size} n_block={n_block_size}"
        )
    if is_split_kv:
        partial_dtype = (
            out_torch_dtype if compute_capability == 9 and page_table is not None else torch.float32
        )
        out_partial = torch.empty(
            num_splits,
            *q_batch_seqlen_shape,
            num_head,
            head_dim_v,
            dtype=partial_dtype,
            device=device,
        )
        lse_partial = torch.empty((num_splits, *lse_shape), dtype=torch.float32, device=device)

    q_tensor, k_tensor, v_tensor, o_tensor = [
        from_dlpack(t.detach(), assumed_align=16).mark_layout_dynamic(leading_dim=t.ndim - 1)
        for t in (q, k, v, out if not is_split_kv else out_partial)
    ]
    gate_tensor = (
        from_dlpack(attention_gate.detach(), assumed_align=16).mark_layout_dynamic(
            leading_dim=attention_gate.ndim - 1
        )
        if attention_gate is not None
        else None
    )
    if is_split_kv:
        lse_tensor = from_dlpack(lse_partial.detach(), assumed_align=4).mark_layout_dynamic(leading_dim=lse_partial.ndim - 1)
    elif lse is not None:
        lse_tensor = from_dlpack(lse.detach(), assumed_align=4).mark_layout_dynamic(leading_dim=lse.ndim - 1)
    else:
        lse_tensor = None 

    # hash score and mask mods for compile cache
    score_mod_hash = utils.hash_callable(score_mod) if score_mod is not None else False
    mask_mod_hash = utils.hash_callable(mask_mod) if mask_mod is not None else False

    if softcap is not None:
        assert score_mod is None, "softcap and score_mod cannot be used together"
        score_mod = utils.create_softcap_scoremod(softcap)

    is_varlen = (
        cu_seqlens_q is not None
        or cu_seqlens_k is not None
        or seqused_q is not None
        or seqused_k is not None
    )
    if score_mod is not None:
        if is_varlen:
            raise NotImplementedError(
                "score_mod with aux_tensors is not yet supported for varlen sequences. This will be fixed in a future PR."
            )

    if mask_mod is not None:
        if not use_block_sparsity:
            raise NotImplementedError(
                "mask_mod requires the use of block sparsity. This will be fixed in a future PR."
            )
        if is_varlen:
            raise NotImplementedError(
                "mask_mod with aux_tensors is not yet supported for varlen sequences. This will be fixed in a future PR."
            )
        if pack_gqa:
            raise NotImplementedError(
                "mask_mod with aux_tensors is not yet supported with pack_gqa=True. This will be fixed in a future PR."
            )

    if use_block_sparsity:
        if is_varlen:
            raise NotImplementedError(
                "Block sparsity is not yet supported for varlen sequences. This will be fixed in a future PR."
            )
        if pack_gqa:
            raise NotImplementedError(
                "Block sparsity is not yet supported with pack_gqa=True. This will be fixed in a future PR."
            )
        if is_split_kv:
            raise NotImplementedError(
                "Block sparsity is not yet supported with SplitKV. TODO: partition sparse block lists per split."
            )
    if attention_gate is not None and use_block_sparsity:
        raise NotImplementedError("attention_gate is not supported together with block sparsity yet")

    cute_aux_tensors = None
    if aux_tensors is not None:
        cute_aux_tensors = [from_dlpack(buf).mark_layout_dynamic() for buf in aux_tensors]

    compile_key = (
        dtype,
        head_dim,
        head_dim_v,
        qhead_per_kvhead,
        causal,
        score_mod_hash,
        mask_mod_hash,
        use_block_sparsity,
        len(aux_tensors) if aux_tensors is not None else 0,
        lse is None,
        cu_seqlens_q is None,
        cu_seqlens_k is None,
        seqused_q is None,
        seqused_k is None,
        ("paged_sm90_v2" if page_table is not None else None),
        window_size_left is not None,
        window_size_right is not None,
        learnable_sink is not None,
        m_block_size,
        n_block_size,
        num_threads,
        is_split_kv,
        pack_gqa,
        compute_capability,
        page_size not in [None, 128],  # paged KV non-TMA
        attention_gate is not None,
    )
    if compile_key not in _flash_attn_fwd.compile_cache:
        if compute_capability == 9:
            if page_table is not None:
                assert page_size == n_block_size, "paged KV on SM 9.0 requires page_size == n_block_size"
                assert block_sparse_tensors is None, "paged KV on SM 9.0 does not support block sparsity"
                assert cu_seqlens_q is not None, "paged KV on SM 9.0 requires cu_seqlens_q (varlen)"
                # Split KV is only supported for paged varlen decode on SM90
                fa_fwd = FlashAttentionForwardPagedSM90(
                    dtype,
                    head_dim,
                    head_dim_v,
                    qhead_per_kvhead,
                    is_causal=causal,
                    is_local=local,
                    pack_gqa=pack_gqa,
                    tile_m=m_block_size,
                    tile_n=n_block_size,
                    num_stages=2,
                    num_threads=num_threads,
                    Q_in_regs=False,
                    intra_wg_overlap=True,
                    mma_pv_is_rs=True,
                    is_split_kv=is_split_kv,
                    mask_mod=mask_mod,
                    score_mod=score_mod,
                    has_aux_tensors=aux_tensors is not None,
                    has_attention_gate=attention_gate is not None,
                )
            else:
                assert not is_split_kv, "SplitKV not supported on SM 9.0 without paged KV"
                fa_fwd = FlashAttentionForwardSm90(
                    dtype,
                    head_dim,
                    head_dim_v,
                    qhead_per_kvhead,
                    is_causal=causal,
                    is_local=local,
                    pack_gqa=pack_gqa,
                    tile_m=m_block_size,
                    tile_n=n_block_size,
                    num_stages=2,
                    num_threads=num_threads,
                    Q_in_regs=False,
                    intra_wg_overlap=True,
                    mma_pv_is_rs=True,
                    mask_mod=mask_mod,
                    score_mod=score_mod,
                    has_aux_tensors=aux_tensors is not None,
                    has_attention_gate=attention_gate is not None,
                )
        elif compute_capability == 10:
            fa_fwd = FlashAttentionForwardSm100(
                head_dim,
                head_dim_v,
                qhead_per_kvhead=qhead_per_kvhead,
                is_causal=causal,
                is_local=local,
                is_split_kv=is_split_kv,
                pack_gqa=pack_gqa,
                m_block_size=m_block_size,
                n_block_size=n_block_size,
                is_persistent=not causal
                    and not local
                    and cu_seqlens_q is None
                    and seqused_q is None
                    and not is_split_kv,
                score_mod=score_mod,
                mask_mod=mask_mod,
                has_aux_tensors=aux_tensors is not None,
                paged_kv_non_tma=page_size not in [None, 128],
                is_varlen_q=cu_seqlens_q is not None
                    or seqused_q is not None,
            )
        else:
            raise ValueError(
                f"Unsupported compute capability: {compute_capability}. Supported: 9.x, 10.x"
            )
        # TODO: check @can_implement
        use_paged_sm90 = compute_capability == 9 and page_table is not None
        compile_args = [
            fa_fwd,
            q_tensor,
            k_tensor,
            v_tensor,
            o_tensor,
            lse_tensor,
            softmax_scale,
            current_stream,
            cu_seqlens_q_tensor,
            cu_seqlens_k_tensor,
            seqused_q_tensor,
            seqused_k_tensor,
            page_table_tensor,
            window_size_left,
            window_size_right,
            learnable_sink_tensor,
            sparse_tensors,
            cute_aux_tensors,
        ]
        if compute_capability == 9:
            compile_args.append(gate_tensor)
        _flash_attn_fwd.compile_cache[compile_key] = cute.compile(*compile_args)
    if is_split_kv:
        _reset_split_workspace(out_partial, lse_partial, current_stream)
    call_args = [
        q_tensor,
        k_tensor,
        v_tensor,
        o_tensor,
        lse_tensor,
        softmax_scale,
        current_stream,
        cu_seqlens_q_tensor,
        cu_seqlens_k_tensor,
        seqused_q_tensor,
        seqused_k_tensor,
        page_table_tensor,
        window_size_left,
        window_size_right,
        learnable_sink_tensor,
        sparse_tensors,
        cute_aux_tensors,
    ]
    if compute_capability == 9:
        call_args.append(gate_tensor)
    _flash_attn_fwd.compile_cache[compile_key](*call_args)
    if is_split_kv:
        if os.getenv("FLASH_ATTN_DEBUG_SPLITS") == "1":
            print(
                "[flash_attn] launching combine kernel"
                f" ({'varlen' if (cu_seqlens_q is not None or seqused_q is not None) else 'batched'})"
            )
            print(f"### fa combie with is varlen: {is_varlen}")

        combine_fn = (
            _flash_attn_fwd_combine_varlen
            if (cu_seqlens_q is not None or seqused_q is not None)
            else _flash_attn_fwd_combine
        )
        combine_fn(
            out_partial,
            lse_partial.transpose(-1, -2),
            out,
            lse.transpose(-1, -2) if lse is not None else None,
            cu_seqlens_q,
            seqused_q,
        )
    return out, lse


_flash_attn_fwd.compile_cache = {}


def _flash_attn_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    dout: torch.Tensor,
    lse: torch.Tensor,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    softcap: float = 0.0,
    m_block_size: int = 64,
    n_block_size: int = 128,
    num_threads: int = 256,
    pack_gqa: bool = False,
    num_stages_Q: int = 2,
    num_stages_dO: int = 2,
    SdP_swapAB: bool = False,
    dKV_swapAB: bool = False,
    dQ_swapAB: bool = False,
    AtomLayoutMSdP: int = 2,
    AtomLayoutNdKV: int = 2,
    AtomLayoutMdQ: int = 2,
    V_in_regs: bool = False,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    compute_capability = torch.cuda.get_device_capability()[0]
    assert compute_capability in [9, 10], "Unsupported compute capability. Supported: 9.x, 10.x"

    if compute_capability == 9:
        m_block_size = 80 if not causal else 64
        n_block_size = 128
        num_stages_Q = 2
        num_stages_dO = 2
        num_stages_PdS = 2
        SdP_swapAB = True
        dKV_swapAB = False
        dQ_swapAB = not causal
        AtomLayoutMSdP = 1
        AtomLayoutNdKV = 2
        AtomLayoutMdQ = 1
        cluster_size = 1
    else:
        m_block_size = 128
        n_block_size = 128
        dQ_swapAB = False
        dKV_swapAB = False
        AtomLayoutMdQ = 1
        AtomLayoutNdKV = 1
        # TODO: support cluster size 2
        cluster_size = 1
    q, k, v, out, dout, lse, cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k = [
        maybe_contiguous(t)
        for t in (q, k, v, out, dout, lse, cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k)
    ]
    num_head, head_dim = q.shape[-2:]
    if cu_seqlens_q is None:
        batch_size, seqlen_q = q.shape[:2]
        total_q = batch_size * seqlen_q
    else:
        batch_size = cu_seqlens_q.shape[0] - 1
        seqlen_q = None
        total_q = q.shape[0]

    if cu_seqlens_k is None:
        batch_size, seqlen_k = k.shape[:2]
        total_k = batch_size * seqlen_k
    else:
        batch_size = cu_seqlens_k.shape[0] - 1
        seqlen_k = None
        total_k = k.shape[0]

    num_head_kv = k.shape[-2]
    head_dim_v = v.shape[-1]

    if cu_seqlens_k is None:
        assert k.shape == (batch_size, seqlen_k, num_head_kv, head_dim)
        assert v.shape == (batch_size, seqlen_k, num_head_kv, head_dim_v)
    else:
        assert k.shape == (total_k, num_head_kv, head_dim)
        assert v.shape == (total_k, num_head_kv, head_dim_v)
        assert cu_seqlens_k.shape == (batch_size + 1,), (
            "cu_seqlens_k must have shape (batch_size + 1,)"
        )

    if cu_seqlens_q is not None:
        assert cu_seqlens_q.shape == (batch_size + 1,), (
            "cu_seqlens_q must have shape (batch_size + 1,)"
        )

        assert out.shape == (total_q, num_head, head_dim_v)
        assert dout.shape == (total_q, num_head, head_dim_v)
        assert lse.shape == (num_head, total_q), "lse must have shape (num_head, total_q)"
    else:
        assert out.shape == (batch_size, seqlen_q, num_head, head_dim_v)
        assert dout.shape == (batch_size, seqlen_q, num_head, head_dim_v)
        assert lse.shape == (batch_size, num_head, seqlen_q), (
            "lse must have shape (batch_size, num_head, seqlen_q)"
        )

    assert q.dtype in [torch.float16, torch.bfloat16], "inputs must be float16 or bfloat16"
    assert q.dtype == k.dtype == v.dtype == out.dtype == dout.dtype, (
        "inputs must have the same dtype"
    )
    for t in [cu_seqlens_q, cu_seqlens_k]:
        if t is not None:
            assert t.dtype == torch.int32, "cu_seqlens_q, cu_seqlens_k must be int32"
    assert lse.dtype == torch.float32, "lse must be float32"
    assert all(
        t is None or t.is_cuda for t in (q, k, v, out, dout, lse, cu_seqlens_q, cu_seqlens_k)
    ), "inputs must be on CUDA device"
    assert num_head % num_head_kv == 0, "num_head must be divisible by num_head_kv"
    assert head_dim <= 256, "head_dim must be less than or equal to 256"
    alignment = 16 // q.element_size()
    assert head_dim % alignment == 0, f"head_dim must be divisible by {alignment}"
    assert head_dim_v % alignment == 0, f"head_dim_v must be divisible by {alignment}"
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)
    qhead_per_kvhead = num_head // num_head_kv
    if pack_gqa is None:
        pack_gqa = qhead_per_kvhead > 1
    if compute_capability == 10:
        pack_gqa = False # override for now

    device = q.device
    # TODO: check if this is the right rounding
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)

    head_dim_rounded = (head_dim + 32 - 1) // 32 * 32

    if cu_seqlens_q is None:
        seqlen_q_rounded = (seqlen_q + m_block_size - 1) // m_block_size * m_block_size
        dq_accum = torch.empty(
            batch_size,
            num_head,
            seqlen_q_rounded * head_dim_rounded,
            dtype=torch.float32,
            device=device,
        )
        dpsum = torch.empty(
            batch_size, num_head, seqlen_q_rounded, dtype=torch.float32, device=device
        )
        lse_log2 = torch.empty(
            batch_size, num_head, seqlen_q_rounded, dtype=torch.float32, device=device
        )
    else:
        total_q_rounded_padded = (
            (total_q + cu_seqlens_q.shape[0] * m_block_size - 1) // m_block_size * m_block_size
        )
        dq_accum = torch.empty(
            num_head, total_q_rounded_padded * head_dim_rounded, dtype=torch.float32, device=device
        )
        dpsum = torch.empty(num_head, total_q_rounded_padded, dtype=torch.float32, device=device)
        lse_log2 = torch.empty(num_head, total_q_rounded_padded, dtype=torch.float32, device=device)

    if qhead_per_kvhead > 1:
        head_dim_v_rounded = (head_dim_v + 32 - 1) // 32 * 32
        if cu_seqlens_k is None:
            seqlen_k_rounded = (seqlen_k + n_block_size - 1) // n_block_size * n_block_size
            num_n_blocks = seqlen_k_rounded // n_block_size
            if cluster_size == 2 and num_n_blocks % cluster_size != 0:
                seqlen_k_rounded = seqlen_k_rounded + n_block_size
            dk_accum = torch.zeros(
                batch_size,
                num_head_kv,
                seqlen_k_rounded * head_dim_rounded,
                dtype=torch.float32,
                device=device,
            )
            dv_accum = torch.zeros(
                batch_size,
                num_head_kv,
                seqlen_k_rounded * head_dim_v_rounded,
                dtype=torch.float32,
                device=device,
            )
        else:
            total_k_rounded_padded = (
                (total_k + cu_seqlens_k.shape[0] * n_block_size - 1) // n_block_size * n_block_size
            )
            num_n_blocks = total_k_rounded_padded // n_block_size
            if cluster_size == 2 and num_n_blocks % cluster_size != 0:
                total_k_rounded_padded = total_k_rounded_padded + n_block_size
            dk_accum = torch.zeros(
                num_head_kv,
                total_k_rounded_padded * head_dim_rounded,
                dtype=torch.float32,
                device=device,
            )
            dv_accum = torch.zeros(
                num_head_kv,
                total_k_rounded_padded * head_dim_v_rounded,
                dtype=torch.float32,
                device=device,
            )

    dtype = torch2cute_dtype_map[q.dtype]
    q_tensor, k_tensor, v_tensor, o_tensor, do_tensor, dq_tensor, dk_tensor, dv_tensor = [
        from_dlpack(t.detach(), assumed_align=16).mark_layout_dynamic(leading_dim=t.ndim - 1)
        for t in (q, k, v, out, dout, dq, dk, dv)
    ]
    lse_tensor = from_dlpack(lse.detach(), assumed_align=4).mark_layout_dynamic(
        leading_dim=lse.ndim - 1
    )
    dq_accum_tensor, dpsum_tensor, lse_log2_tensor = [
        from_dlpack(t.detach(), assumed_align=16).mark_layout_dynamic(leading_dim=t.ndim - 1)
        for t in (dq_accum, dpsum, lse_log2)
    ]
    if qhead_per_kvhead > 1:
        dk_accum_tensor, dv_accum_tensor = [
            from_dlpack(t.detach(), assumed_align=16).mark_layout_dynamic(leading_dim=t.ndim - 1)
            for t in (dk_accum, dv_accum)
        ]
    cu_seqlens_q_tensor, cu_seqlens_k_tensor, seqused_q_tensor, seqused_k_tensor = [
        from_dlpack(t.detach(), assumed_align=4).mark_layout_dynamic(leading_dim=t.ndim - 1)
        if t is not None
        else None
        for t in (cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k)
    ]
    current_stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    # Preprocess kernel: compute (o * dout).sum(dim=-1), lse * log2_e, and zero out dq_accum.
    compile_key_pre = (compute_capability, dtype, head_dim_v, m_block_size, num_threads)
    if compile_key_pre not in _flash_attn_bwd.compile_cache_pre:
        fa_bwd_pre = FlashAttentionBackwardPreprocess(
            dtype,
            head_dim_v,
            m_block_size,
            num_threads=num_threads,
        )
        # TODO: check @can_implement
        _flash_attn_bwd.compile_cache_pre[compile_key_pre] = cute.compile(
            fa_bwd_pre,
            o_tensor,
            do_tensor,
            dpsum_tensor,
            lse_tensor,
            lse_log2_tensor,
            dq_accum_tensor,
            cu_seqlens_q_tensor,
            seqused_q_tensor,
            current_stream,
        )
    _flash_attn_bwd.compile_cache_pre[compile_key_pre](
        o_tensor,
        do_tensor,
        dpsum_tensor,
        lse_tensor,
        lse_log2_tensor,
        dq_accum_tensor,
        cu_seqlens_q_tensor,
        seqused_q_tensor,
        current_stream,
    )

    # Backward kernel: compute dk, dv, dq_accum.
    if compute_capability == 9:
        compile_key = (
            compute_capability,
            dtype,
            head_dim,
            head_dim_v,
            qhead_per_kvhead,
            causal,
            softcap != 0.0,
            m_block_size,
            n_block_size,
            num_threads,
            pack_gqa,
            num_stages_Q,
            num_stages_dO,
            SdP_swapAB,
            dKV_swapAB,
            dQ_swapAB,
            AtomLayoutMSdP,
            AtomLayoutNdKV,
            AtomLayoutMdQ,
            V_in_regs,
        )
    else:
        compile_key = (
            compute_capability,
            dtype,
            head_dim,
            head_dim_v,
            qhead_per_kvhead,
            causal,
            softcap != 0.0,
            m_block_size,
            n_block_size,
            num_threads,
            pack_gqa,
            cluster_size,
        )
    num_threads = 384
    if compile_key not in _flash_attn_bwd.compile_cache:
        fa_bwd_sm80 = FlashAttentionBackwardSm80(
            dtype,
            head_dim,
            head_dim_v,
            qhead_per_kvhead,
            m_block_size,
            n_block_size,
            num_stages_Q,
            num_stages_dO,
            num_threads,
            pack_gqa,
            causal,
            SdP_swapAB,
            dKV_swapAB,
            dQ_swapAB,
            AtomLayoutMSdP,
            AtomLayoutNdKV,
            AtomLayoutMdQ,
            V_in_regs=V_in_regs,
        )
        if compute_capability == 9:
            fa_bwd_obj = FlashAttentionBackwardSm90(
                dtype,
                head_dim,
                head_dim_v,
                qhead_per_kvhead,
                causal,
                m_block_size,
                n_block_size,
                num_stages_Q,
                num_stages_dO,
                num_stages_PdS,
                SdP_swapAB,
                dKV_swapAB,
                dQ_swapAB,
                AtomLayoutMSdP,
                AtomLayoutNdKV,
                AtomLayoutMdQ,
                num_threads,
                V_in_regs=V_in_regs,
            )
        else:
            fa_bwd_obj = FlashAttentionBackwardSm100(
                head_dim,
                head_dim_v,
                is_causal=causal,
                qhead_per_kvhead=qhead_per_kvhead,
                # tile_m=m_block_size,
                # tile_n=n_block_size,
                cluster_size=cluster_size,
                # cluster_size=1,
            )
        # TODO: check @can_implement
        _flash_attn_bwd.compile_cache[compile_key] = cute.compile(
            fa_bwd_obj,
            q_tensor,
            k_tensor,
            v_tensor,
            do_tensor,
            lse_log2_tensor,
            dpsum_tensor,
            dq_accum_tensor,
            dk_tensor if qhead_per_kvhead == 1 else dk_accum_tensor,
            dv_tensor if qhead_per_kvhead == 1 else dv_accum_tensor,
            softmax_scale,
            current_stream,
            cu_seqlens_q_tensor,
            cu_seqlens_k_tensor,
            seqused_q_tensor,
            seqused_k_tensor,
        )
    _flash_attn_bwd.compile_cache[compile_key](
        q_tensor,
        k_tensor,
        v_tensor,
        do_tensor,
        lse_log2_tensor,
        dpsum_tensor,
        dq_accum_tensor,
        dk_tensor if qhead_per_kvhead == 1 else dk_accum_tensor,
        dv_tensor if qhead_per_kvhead == 1 else dv_accum_tensor,
        softmax_scale,
        current_stream,
        cu_seqlens_q_tensor,
        cu_seqlens_k_tensor,
        seqused_q_tensor,
        seqused_k_tensor,
    )

    num_threads = 256 if compute_capability == 9 else 128
    # Postprocess kernel: convert dq_accum from float32 to dq in bf16/fp16
    compile_key_post = (dtype, head_dim, m_block_size, num_threads, AtomLayoutMdQ, dQ_swapAB)
    if compile_key_post not in _flash_attn_bwd.compile_cache_post:
        arch = compute_capability * 10
        fa_bwd_post = FlashAttentionBackwardPostprocess(
            dtype, head_dim, arch, m_block_size, num_threads, AtomLayoutMdQ, dQ_swapAB
        )
        # TODO: check @can_implement
        _flash_attn_bwd.compile_cache_post[compile_key_post] = cute.compile(
            fa_bwd_post,
            dq_accum_tensor,
            dq_tensor,
            softmax_scale,
            cu_seqlens_q_tensor,
            seqused_q_tensor,
            current_stream,
        )
    _flash_attn_bwd.compile_cache_post[compile_key_post](
        dq_accum_tensor,
        dq_tensor,
        softmax_scale,
        cu_seqlens_q_tensor,
        seqused_q_tensor,
        current_stream,
    )

    if qhead_per_kvhead > 1:
        # Postprocess kernel: convert dk_accum & dv_accum from float32 to bf16/fp16
        compile_key_post = (dtype, head_dim, n_block_size, num_threads, AtomLayoutNdKV, dKV_swapAB)
        if compile_key_post not in _flash_attn_bwd.compile_cache_post:
            fa_bwd_post = FlashAttentionBackwardPostprocess(
                dtype, head_dim, n_block_size, num_threads, AtomLayoutNdKV, dKV_swapAB
            )
            # TODO: check @can_implement
            _flash_attn_bwd.compile_cache_post[compile_key_post] = cute.compile(
                fa_bwd_post,
                dk_accum_tensor,
                dk_tensor,
                softmax_scale,
                cu_seqlens_k_tensor,
                seqused_k_tensor,
                current_stream,
            )
        _flash_attn_bwd.compile_cache_post[compile_key_post](
            dk_accum_tensor,
            dk_tensor,
            softmax_scale,
            cu_seqlens_k_tensor,
            seqused_k_tensor,
            current_stream,
        )
        compile_key_post = (
            dtype,
            head_dim_v,
            n_block_size,
            num_threads,
            AtomLayoutNdKV,
            dKV_swapAB,
        )
        if compile_key_post not in _flash_attn_bwd.compile_cache_post:
            fa_bwd_post = FlashAttentionBackwardPostprocess(
                dtype, head_dim_v, n_block_size, num_threads, AtomLayoutNdKV, dKV_swapAB
            )
            # TODO: check @can_implement
            _flash_attn_bwd.compile_cache_post[compile_key_post] = cute.compile(
                fa_bwd_post,
                dv_accum_tensor,
                dv_tensor,
                cutlass.Float32(1.0),
                cu_seqlens_k_tensor,
                seqused_k_tensor,
                current_stream,
            )
        _flash_attn_bwd.compile_cache_post[compile_key_post](
            dv_accum_tensor,
            dv_tensor,
            cutlass.Float32(1.0),
            cu_seqlens_k_tensor,
            seqused_k_tensor,
            current_stream,
        )

    return dq, dk, dv


_flash_attn_bwd.compile_cache_pre = {}
_flash_attn_bwd.compile_cache = {}
_flash_attn_bwd.compile_cache_post = {}


class FlashAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        softmax_scale: Optional[float] = None,
        causal: bool = False,
        window_size: Tuple[Optional[int], Optional[int]] = (None, None),
        learnable_sink: Optional[torch.Tensor] = None,
        softcap: float = 0.0,
        num_splits: int = 1,
        pack_gqa: Optional[bool] = None,
        mask_mod: Optional[Callable] = None,
        full_block_cnt: Optional[torch.Tensor] = None,
        full_block_idx: Optional[torch.Tensor] = None,
        mask_block_cnt: Optional[torch.Tensor] = None,
        mask_block_idx: Optional[torch.Tensor] = None,
        attention_gate: Optional[torch.Tensor] = None,
    ):
        # Only create block sparse tensors if at least one block sparse parameter is provided
        block_sparse_tensors = None
        if any(t is not None for t in [full_block_cnt, full_block_idx, mask_block_cnt, mask_block_idx]):
            block_sparse_tensors = BlockSparseTensorsTorch(
                full_block_cnt=full_block_cnt,
                full_block_idx=full_block_idx,
                mask_block_cnt=mask_block_cnt,
                mask_block_idx=mask_block_idx,
            )
        out, lse = _flash_attn_fwd(
            q,
            k,
            v,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size_left=window_size[0],
            window_size_right=window_size[1],
            learnable_sink=learnable_sink,
            softcap=softcap,
            num_splits=num_splits,
            pack_gqa=pack_gqa,
            mask_mod=mask_mod,
            block_sparse_tensors=block_sparse_tensors,
            attention_gate=attention_gate,
        )
        ctx.save_for_backward(q, k, v, out, lse)
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.window_size = window_size
        ctx.softcap = softcap
        return out, lse

    @staticmethod
    def backward(ctx, dout, *args):
        q, k, v, out, lse = ctx.saved_tensors
        dq, dk, dv = _flash_attn_bwd(
            q,
            k,
            v,
            out,
            dout,
            lse,
            ctx.softmax_scale,
            ctx.causal,
            ctx.softcap,
        )
        return dq, dk, dv, *((None,) * 21)  # Extra Nones is fine


class FlashAttnVarlenFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_q: Optional[torch.Tensor],
        cu_seqlens_k: Optional[torch.Tensor],
        seqused_q: Optional[torch.Tensor] = None,
        seqused_k: Optional[torch.Tensor] = None,
        page_table: Optional[torch.Tensor] = None,
        softmax_scale: Optional[float] = None,
        causal: bool = False,
        window_size: Tuple[Optional[int], Optional[int]] = (None, None),
        learnable_sink: Optional[torch.Tensor] = None,
        softcap: float = 0.0,
        num_splits: int = 1,
        pack_gqa: Optional[bool] = None,
        attention_gate: Optional[torch.Tensor] = None,
        m_block_size: Optional[int] = None,
        n_block_size: Optional[int] = None,
    ):
        out, lse = _flash_attn_fwd(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            seqused_q,
            seqused_k,
            page_table=page_table,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size_left=window_size[0],
            window_size_right=window_size[1],
            learnable_sink=learnable_sink,
            softcap=softcap,
            num_splits=num_splits,
            pack_gqa=pack_gqa,
            attention_gate=attention_gate,
            m_block_size=m_block_size if m_block_size is not None else 128,
            n_block_size=n_block_size if n_block_size is not None else 128,
        )
        ctx.save_for_backward(q, k, v, out, lse, cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k)
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.window_size = window_size
        ctx.softcap = softcap
        return out, lse

    @staticmethod
    def backward(ctx, dout, *args):
        q, k, v, out, lse, cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k = ctx.saved_tensors
        assert seqused_q == seqused_k == None
        assert ctx.softcap == 0.0
        dq, dk, dv = _flash_attn_bwd(
            q,
            k,
            v,
            out,
            dout,
            lse,
            ctx.softmax_scale,
            ctx.causal,
            ctx.softcap,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            seqused_q=seqused_q,
            seqused_k=seqused_k,
        )

        return dq, dk, dv, *((None,) * 23)


def flash_attn_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Tuple[Optional[int], Optional[int]] = (None, None),
    learnable_sink: Optional[torch.Tensor] = None,
    softcap: float = 0.0,
    num_splits: int = 1,
    pack_gqa: Optional[bool] = None,
    mask_mod: Optional[Callable] = None,
    full_block_cnt: Optional[torch.Tensor] = None,
    full_block_idx: Optional[torch.Tensor] = None,
    mask_block_cnt: Optional[torch.Tensor] = None,
    mask_block_idx: Optional[torch.Tensor] = None,
    attention_gate: Optional[torch.Tensor] = None,
):
    return FlashAttnFunc.apply(
        q,
        k,
        v,
        softmax_scale,
        causal,
        window_size,
        learnable_sink,
        softcap,
        num_splits,
        pack_gqa,
        mask_mod,
        full_block_cnt,
        full_block_idx,
        mask_block_cnt,
        mask_block_idx,
        attention_gate,
    )


def flash_attn_varlen_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
    page_table: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Tuple[Optional[int], Optional[int]] = (None, None),
    learnable_sink: Optional[torch.Tensor] = None,
    softcap: float = 0.0,
    num_splits: int = 1,
    pack_gqa: Optional[bool] = None,
    attention_gate: Optional[torch.Tensor] = None,
    m_block_size: Optional[int] = None,
    n_block_size: Optional[int] = None,
):
    return FlashAttnVarlenFunc.apply(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        seqused_q,
        seqused_k,
        page_table,
        softmax_scale,
        causal,
        window_size,
        learnable_sink,
        softcap,
        num_splits,
        pack_gqa,
        attention_gate,
        m_block_size,
        n_block_size,
    )


def _select_fwd_combine_launch_config(head_dim: int, num_splits: int) -> Tuple[int, int, int, int, int]:
    """Pick launch/compile config for forward combine kernels.

    Returns:
        (k_block_size, m_block_size, log_max_splits, num_threads, stages)
    """
    k_block_size = 64 if head_dim <= 64 else 128
    m_block_size = 8 if k_block_size % 128 == 0 else (16 if k_block_size % 64 == 0 else 32)

    # For head_dim=128 (m_block_size=8), NCU tuning favors 128 threads and
    # a deeper pipeline. Stage-5 is better for very small split counts, while
    # stage-6 is better once split count grows.
    if m_block_size == 8:
        num_threads = 128
        min_log_max_splits = 4  # max_splits = 16
        stages = 5 if num_splits <= 4 else 6
    else:
        num_threads = 256
        min_log_max_splits = 4
        if num_splits <= 8:
            stages = 2
        elif num_splits <= 32:
            stages = 3
        else:
            stages = 4

    log_max_splits = max(math.ceil(math.log2(num_splits)), min_log_max_splits)
    return k_block_size, m_block_size, log_max_splits, num_threads, stages


def _validate_fwd_combine_inputs(
    out_partial: torch.Tensor,
    lse_partial: torch.Tensor,
    out: torch.Tensor,
    lse: Optional[torch.Tensor],
    cu_seqlens: Optional[torch.Tensor],
    seqused: Optional[torch.Tensor],
    num_splits_dynamic_ptr: Optional[torch.Tensor],
) -> bool:
    assert out_partial.dim() in [4, 5], "out_partial must have 4 or 5 dimensions"
    assert lse_partial.dim() in [3, 4], "lse_partial must have 3 or 4 dimensions"
    assert out_partial.dtype in [torch.float16, torch.bfloat16, torch.float32], (
        "out_partial must be fp16, bf16, or fp32"
    )
    assert lse_partial.dtype == torch.float32, "lse_partial must be fp32"
    assert out_partial.is_cuda and lse_partial.is_cuda, "tensors must be on CUDA device"
    assert out_partial.stride(-1) == 1, "out_partial must be contiguous in the last dimension"
    assert lse_partial.stride(-2) == 1, "lse_partial must be contiguous in the seqlen dimension"
    assert lse_partial.shape == out_partial.shape[:-1]

    is_varlen = out_partial.dim() == 4

    assert out.shape == out_partial.shape[1:], "out shape mismatch"
    if lse is not None:
        assert lse.shape == lse_partial.shape[1:], "lse shape mismatch"
        assert lse.dtype == torch.float32, "lse must be fp32"

    for t, name in [
        (cu_seqlens, "cu_seqlens"),
        (seqused, "seqused"),
        (num_splits_dynamic_ptr, "num_splits_dynamic_ptr"),
    ]:
        if t is not None:
            assert t.dtype == torch.int32, f"{name} must be int32"
            assert t.is_cuda, f"{name} must be on CUDA device"
            assert t.is_contiguous(), f"{name} must be contiguous"

    return is_varlen


def _to_cute_optional_int32_tensor(t: Optional[torch.Tensor]):
    if t is None:
        return None
    return from_dlpack(t.detach(), assumed_align=4).mark_layout_dynamic(leading_dim=0)


def _flash_attn_fwd_combine_impl(
    out_partial: torch.Tensor,
    lse_partial: torch.Tensor,
    out: torch.Tensor,
    lse: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    seqused: Optional[torch.Tensor] = None,
    num_splits_dynamic_ptr: Optional[torch.Tensor] = None,
    semaphore_to_reset: Optional[torch.Tensor] = None,
    *,
    compile_cache: dict,
    enable_varlen_flatten: bool,
) -> None:
    """Shared implementation for forward combine kernels."""
    is_varlen = _validate_fwd_combine_inputs(
        out_partial,
        lse_partial,
        out,
        lse,
        cu_seqlens,
        seqused,
        num_splits_dynamic_ptr,
    )

    head_dim = out_partial.shape[-1]
    num_splits = out_partial.shape[0]
    assert num_splits <= 256
    k_block_size, m_block_size, log_max_splits, num_threads, stages = _select_fwd_combine_launch_config(
        head_dim,
        num_splits,
    )

    out_partial_tensor = from_dlpack(out_partial.detach(), assumed_align=16).mark_layout_dynamic(
        leading_dim=4 if not is_varlen else 3
    )
    lse_partial_tensor = from_dlpack(lse_partial.detach(), assumed_align=4).mark_layout_dynamic(
        leading_dim=lse_partial.ndim - 2
    )
    out_tensor = from_dlpack(out.detach(), assumed_align=16).mark_layout_dynamic(leading_dim=3 if not is_varlen else 2)
    lse_tensor = (
        from_dlpack(lse.detach(), assumed_align=4).mark_layout_dynamic(leading_dim=lse.ndim - 2)
        if lse is not None
        else None
    )

    cu_seqlens_tensor = _to_cute_optional_int32_tensor(cu_seqlens)
    seqused_tensor = _to_cute_optional_int32_tensor(seqused)
    num_splits_dynamic_tensor = _to_cute_optional_int32_tensor(num_splits_dynamic_ptr)
    semaphore_tensor = _to_cute_optional_int32_tensor(semaphore_to_reset)

    # For varlen with fixed split count, flatten all query tokens into one
    # logical batch to avoid launching duplicated combine work per batch item.
    has_cu_seqlens = cu_seqlens is not None
    has_seqused = seqused is not None
    if enable_varlen_flatten and is_varlen and num_splits_dynamic_ptr is None:
        total_q = int(out_partial.shape[1])
        flat_cu_seqlens = torch.tensor([0, total_q], dtype=torch.int32, device=out_partial.device)
        cu_seqlens_tensor = from_dlpack(flat_cu_seqlens.detach(), assumed_align=4).mark_layout_dynamic(
            leading_dim=0
        )
        seqused_tensor = None
        has_cu_seqlens = True
        has_seqused = False

    current_stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    dtype = torch2cute_dtype_map[out.dtype]
    dtype_partial = torch2cute_dtype_map[out_partial.dtype]

    compile_key = (
        dtype,
        dtype_partial,
        head_dim,
        m_block_size,
        k_block_size,
        log_max_splits,
        num_threads,
        stages,
        has_cu_seqlens,
        has_seqused,
        lse is not None,
    )

    if compile_key not in compile_cache:
        fa_combine = FlashAttentionForwardCombineVarlen(
            dtype=dtype,
            dtype_partial=dtype_partial,
            head_dim=head_dim,
            m_block_size=m_block_size,
            k_block_size=k_block_size,
            log_max_splits=log_max_splits,
            num_threads=num_threads,
            stages=stages,
        )

        if not fa_combine.can_implement(
            dtype,
            dtype_partial,
            head_dim,
            m_block_size,
            k_block_size,
            log_max_splits,
            num_threads=num_threads,
        ):
            raise RuntimeError(
                "FlashAttention combine kernel cannot be implemented with given parameters"
            )

        compile_cache[compile_key] = cute.compile(
            fa_combine,
            out_partial_tensor,
            lse_partial_tensor,
            out_tensor,
            lse_tensor,
            cu_seqlens_tensor,
            seqused_tensor,
            num_splits_dynamic_tensor,
            semaphore_tensor,
            current_stream,
        )

    compile_cache[compile_key](
        out_partial_tensor,
        lse_partial_tensor,
        out_tensor,
        lse_tensor,
        cu_seqlens_tensor,
        seqused_tensor,
        num_splits_dynamic_tensor,
        semaphore_tensor,
        current_stream,
    )


def _flash_attn_fwd_combine(
    out_partial: torch.Tensor,
    lse_partial: torch.Tensor,
    out: torch.Tensor,
    lse: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    seqused: Optional[torch.Tensor] = None,
    num_splits_dynamic_ptr: Optional[torch.Tensor] = None,
    semaphore_to_reset: Optional[torch.Tensor] = None,
) -> None:
    """Forward combine kernel for split attention computation."""
    _flash_attn_fwd_combine_impl(
        out_partial,
        lse_partial,
        out,
        lse,
        cu_seqlens,
        seqused,
        num_splits_dynamic_ptr,
        semaphore_to_reset,
        compile_cache=_flash_attn_fwd_combine.compile_cache,
        enable_varlen_flatten=False,
    )


_flash_attn_fwd_combine.compile_cache = {}



def flash_attn_combine(
    out_partial: torch.Tensor,
    lse_partial: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    seqused: Optional[torch.Tensor] = None,
    return_lse: bool = True,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Flash Attention combine function for split attention computation.

    Combines partial outputs and log-sum-exp values from multiple splits
    of attention computation into final outputs. This is the main user-facing
    interface for the combine kernel.

    Args:
        out_partial: Partial outputs tensor with shape:
            - (num_splits, batch_size, seqlen, num_heads, head_size) for regular batched input
            - (num_splits, total_q, num_heads, head_size) for variable length input
        lse_partial: Partial LSE tensor with shape:
            - (num_splits, batch_size, seqlen, num_heads) for regular batched input
            - (num_splits, total_q, num_heads) for variable length input
        out: Optional output tensor. If None, will be created automatically.
        out_dtype: Optional output dtype. If None, will use fp16/bf16 based on input.
        cu_seqlens: Cumulative sequence lengths for variable length sequences
        seqused: Used sequence lengths for each batch
        return_lse: Whether to return the combined LSE tensor. Default is True.

    Returns:
        Tuple of (out, lse) where:
        - out: Combined output tensor with shape (batch_size, seqlen, num_heads, head_size)
              or (total_q, num_heads, head_size) for varlen
        - lse: Combined log-sum-exp tensor with shape (batch_size, seqlen, num_heads)
              or (total_q, num_heads) for varlen. None if return_lse=False

    Note:
        This function expects the input tensors to be in the format produced by
        split attention computation, where the first dimension is num_splits.
        The permuting from user format to kernel format is now done inside the kernel.
    """
    # Input validation
    assert out_partial.dim() in [4, 5], "out_partial must have 4 or 5 dimensions"
    assert lse_partial.dim() in [3, 4], "lse_partial must have 3 or 4 dimensions"
    assert out_partial.dtype == torch.float32, "out_partial must be fp32 (from accumulation)"
    assert lse_partial.dtype == torch.float32, "lse_partial must be fp32"

    # Determine if this is variable length based on dimensions
    is_varlen = out_partial.dim() == 4

    if is_varlen:
        # Variable length: (num_splits, total_q, num_heads, head_size)
        num_splits, total_q, num_heads, head_size = out_partial.shape
        assert lse_partial.shape == (num_splits, total_q, num_heads), (
            "lse_partial shape mismatch for varlen"
        )
        batch_size = 1  # Treat as single batch for varlen
        seqlen = total_q
    else:
        # Regular batched: (num_splits, batch_size, seqlen, num_heads, head_size)
        num_splits, batch_size, seqlen, num_heads, head_size = out_partial.shape
        assert lse_partial.shape == (num_splits, batch_size, seqlen, num_heads), (
            "lse_partial shape mismatch"
        )

    # Determine output dtype
    if out_dtype is None:
        out_dtype = out_partial.dtype

    # Create output if not provided
    device = out_partial.device
    if out is None:
        if is_varlen:
            out = torch.empty(total_q, num_heads, head_size, dtype=out_dtype, device=device)
        else:
            out = torch.empty(
                batch_size, seqlen, num_heads, head_size, dtype=out_dtype, device=device
            )

    # Create lse output only if requested
    if return_lse:
        if is_varlen:
            lse = torch.empty(num_heads, total_q, dtype=torch.float32, device=device).transpose(
                0, 1
            )
        else:
            lse = torch.empty(
                batch_size, num_heads, seqlen, dtype=torch.float32, device=device
            ).transpose(1, 2)
    else:
        lse = None

    combine_fn = _flash_attn_fwd_combine_varlen if is_varlen else _flash_attn_fwd_combine

    combine_fn(
        out_partial,
        lse_partial,
        out,
        lse,
        cu_seqlens,
        seqused,
    )
    return out, lse


def token_sparse_flash_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_counts: torch.Tensor,
    block_indices: torch.Tensor,
    softmax_scale: Optional[float] = None,
    out: Optional[torch.Tensor] = None,
    *,
    _compute_capability: Optional[int] = None,
) -> torch.Tensor:
    """Run the token-level sparse FlashAttention kernel (forward only, SM90)."""

    q, k, v = [maybe_contiguous(t) for t in (q, k, v)]
    block_counts, block_indices = [maybe_contiguous(t) for t in (block_counts, block_indices)]
    assert q.dtype in torch2cute_dtype_map, "q/k/v must be float16 or bfloat16"
    assert q.dtype == k.dtype == v.dtype, "q/k/v must share the same dtype"
    assert q.ndim == 4 and k.ndim == 4 and v.ndim == 4, "q/k/v must be rank-4 tensors"
    batch_size, seqlen_q, num_head_q, head_dim = q.shape
    seqlen_k = k.shape[1]
    num_head_kv = k.shape[2]
    head_dim_v = v.shape[-1]
    assert k.shape == (batch_size, seqlen_k, num_head_kv, head_dim), "k shape mismatch"
    assert v.shape == (batch_size, seqlen_k, num_head_kv, head_dim_v), "v shape mismatch"
    assert num_head_q % num_head_kv == 0, "num_head_q must be divisible by num_head_kv"
    qhead_per_kvhead = num_head_q // num_head_kv
    assert qhead_per_kvhead == 8, "token-sparse kernel currently requires qhead_per_kvhead == 8"
    assert num_head_kv == 1, "token-sparse kernel currently assumes a single KV head"
    num_head_kv = num_head_q // qhead_per_kvhead
    assert block_counts.shape == (batch_size, num_head_kv, seqlen_q), (
        "block_counts must have shape (batch, num_kv_heads, seqlen_q)"
    )
    assert block_indices.shape[:3] == (batch_size, num_head_kv, seqlen_q), (
        "block_indices leading dims must match (batch, num_kv_heads, seqlen_q)"
    )
    assert block_counts.dtype == torch.int32 and block_indices.dtype == torch.int32, (
        "block metadata must be int32 tensors"
    )
    assert block_counts.is_cuda and block_indices.is_cuda, "block metadata must be CUDA tensors"
    assert (
        block_counts.device == q.device and block_indices.device == q.device
    ), "block metadata must live on the same device as q/k/v"
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)

    if out is None:
        out = torch.empty(
            batch_size, seqlen_q, num_head_q, head_dim_v, dtype=v.dtype, device=q.device
        )
    else:
        expected_shape = (batch_size, seqlen_q, num_head_q, head_dim_v)
        assert out.shape == expected_shape, f"out shape must be {expected_shape}"
        assert out.dtype == v.dtype, "out dtype must match v.dtype"
        assert out.device == q.device and out.is_cuda, "out must be on the same CUDA device as q"

    num_head_kv = num_head_q // qhead_per_kvhead
    dtype = torch2cute_dtype_map[q.dtype]
    compute_capability = (
        torch.cuda.get_device_capability()[0]
        if _compute_capability is None
        else _compute_capability
    )
    assert compute_capability == 9, "token-sparse kernel is available on SM90 GPUs only"

    kernel = TokenSparseFlashAttentionSm90(
        dtype=dtype,
        head_dim=head_dim,
        head_dim_v=head_dim_v,
        qhead_per_kvhead=qhead_per_kvhead,
    )
    token_sparse = TokenSparseTensors(block_counts, block_indices)
    current_stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    kernel(
        q,
        k,
        v,
        out,
        token_sparse,
        float(softmax_scale),
        current_stream,
        mLSE=None,
    )
    return out



def _flash_attn_fwd_combine_varlen(
    out_partial: torch.Tensor,
    lse_partial: torch.Tensor,
    out: torch.Tensor,
    lse: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    seqused: Optional[torch.Tensor] = None,
    num_splits_dynamic_ptr: Optional[torch.Tensor] = None,
    semaphore_to_reset: Optional[torch.Tensor] = None,
) -> None:
    """Forward combine kernel for varlen split attention computation."""
    _flash_attn_fwd_combine_impl(
        out_partial,
        lse_partial,
        out,
        lse,
        cu_seqlens,
        seqused,
        num_splits_dynamic_ptr,
        semaphore_to_reset,
        compile_cache=_flash_attn_fwd_combine_varlen.compile_cache,
        enable_varlen_flatten=True,
    )


_flash_attn_fwd_combine_varlen.compile_cache = {}
