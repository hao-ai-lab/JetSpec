"""Cute-DSL implementation of masked per-token FP8 group quantization."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Float8E4M3FN, Int32

from optimus_cutedsl.utils import (
    convert_from_dlpack,
    warp_reduce,
    cvt_fp32x2_to_e4m3x2,
    store_fp8x2,
)


THREADS_PER_CTA = 256
LANES_PER_WARP = cute.arch.WARP_SIZE
WARPS_PER_CTA = THREADS_PER_CTA // LANES_PER_WARP
QUANT_GROUP_SIZE = 128
ROWS_PER_CHUNK = 32

# Use a simple MemRange to stage rows in shared memory. Cute-DSL structs currently
# require MemRange/array elements directly inside the struct, so we define a flat
# buffer of size WARPS_PER_CTA * QUANT_GROUP_SIZE.
@cute.kernel
def _per_token_group_quant_fp8_masked_kernel(
    mX: cute.Tensor,
    mXq: cute.Tensor,
    mXs: cute.Tensor,
    mMask: cute.Tensor,
    rows: Int32,
    rows_per_expert: Int32,
    num_experts: Int32,
    groups_per_row: Int32,
    eps: Float32,
    fp8_min: Float32,
    fp8_max: Float32,
):

    cta_idx, _, _ = cute.arch.block_idx()
    warp_idx = cute.arch.warp_idx()
    lane_idx = cute.arch.lane_idx()

    group_linear = cta_idx * WARPS_PER_CTA + warp_idx
    total_group_tiles = num_experts * groups_per_row

    if group_linear < total_group_tiles:
        expert_id = group_linear // groups_per_row
        group_idx = group_linear - expert_id * groups_per_row
        mask_limit = mMask[expert_id]

        if mask_limit > 0:
            row_base = expert_id * rows_per_expert
            group_start = group_idx * QUANT_GROUP_SIZE
            num_pairs = QUANT_GROUP_SIZE // 2
            pair_iters = cute.ceil_div(num_pairs, LANES_PER_WARP)
            rows_per_chunk = ROWS_PER_CHUNK
            max_chunks = cute.ceil_div(rows_per_expert, rows_per_chunk)
            chunk_limit = cute.ceil_div(mask_limit, rows_per_chunk)
            row_cache = cute.make_rmem_tensor((pair_iters, 2), Float32)

            for chunk_itr in cutlass.range(max_chunks):
                if chunk_itr < chunk_limit:
                    chunk_row_start = chunk_itr * rows_per_chunk

                    for row_offset in cutlass.range(rows_per_chunk):
                        row_itr = chunk_row_start + row_offset
                        if row_itr < mask_limit and row_itr < rows_per_expert:
                            row = row_base + row_itr

                            local_absmax = Float32(0.0)
                            for pair_itr in cutlass.range(pair_iters):
                                idx = pair_itr * LANES_PER_WARP + lane_idx
                                col0 = idx * 2
                                col1 = col0 + 1
                                val0 = Float32(0.0)
                                val1 = Float32(0.0)
                                if col0 < QUANT_GROUP_SIZE:
                                    val0 = Float32(mX[row, group_start + col0])
                                    if col1 < QUANT_GROUP_SIZE:
                                        val1 = Float32(mX[row, group_start + col1])
                                row_cache[pair_itr, 0] = val0
                                row_cache[pair_itr, 1] = val1
                                abs_val0 = cute.arch.fmax(val0, -val0)
                                abs_val1 = cute.arch.fmax(val1, -val1)
                                local_absmax = cute.arch.fmax(local_absmax, abs_val0)
                                local_absmax = cute.arch.fmax(local_absmax, abs_val1)

                            max_abs = warp_reduce(local_absmax, cute.arch.fmax, width=LANES_PER_WARP)
                            max_abs = cute.arch.fmax(max_abs, eps)
                            scale = max_abs / fp8_max
                            scale_inv = fp8_max / max_abs

                            mXs[row, group_idx] = scale

                            for pair_itr in cutlass.range(pair_iters):
                                idx = pair_itr * LANES_PER_WARP + lane_idx
                                col0 = idx * 2
                                col1 = col0 + 1
                                if col0 < QUANT_GROUP_SIZE:
                                    val0 = Float32(row_cache[pair_itr, 0]) * scale_inv
                                    val0 = cute.arch.fmax(-fp8_max, -val0)
                                    val0 = -val0
                                    val0 = cute.arch.fmax(fp8_min, val0)
                                    val1 = Float32(row_cache[pair_itr, 1]) * scale_inv
                                    val1 = cute.arch.fmax(-fp8_max, -val1)
                                    val1 = -val1
                                    val1 = cute.arch.fmax(fp8_min, val1)
                                    packed = cvt_fp32x2_to_e4m3x2(val1, val0)
                                    store_fp8x2(mXq, (row, group_start + col0), packed)


@cute.jit
def _launch_group_quant_kernel(
    mX: cute.Tensor,
    mXq: cute.Tensor,
    mXs: cute.Tensor,
    mMask: cute.Tensor,
    rows: int,
    rows_per_expert: int,
    num_experts: int,
    groups_per_row: int,
    eps: float,
    fp8_min: float,
    fp8_max: float,
):
    _per_token_group_quant_fp8_masked_kernel(
        mX,
        mXq,
        mXs,
        mMask,
        Int32(rows),
        Int32(rows_per_expert),
        Int32(num_experts),
        Int32(groups_per_row),
        Float32(eps),
        Float32(fp8_min),
        Float32(fp8_max),
    ).launch(
        grid=[cute.ceil_div(num_experts * groups_per_row, WARPS_PER_CTA), 1, 1],
        block=[THREADS_PER_CTA, 1, 1],
    )


_COMPILE_CACHE: Dict[
    Tuple[int, int, int, bool, torch.dtype, int], cute.JitFunction
] = {}


def _get_compiled_kernel(
    shape: Tuple[int, int],
    num_experts: int,
    column_major_scales: bool,
    dtype: torch.dtype,
    device: torch.device,
) -> cute.JitFunction:
    key = (shape[0], shape[1], column_major_scales, dtype, num_experts)
    cached = _COMPILE_CACHE.get(key)
    if cached is not None:
        return cached

    rows, hidden = shape
    groups_per_row = hidden // QUANT_GROUP_SIZE

    dummy_x = torch.zeros(shape, device=device, dtype=dtype)
    dummy_q_bits = torch.zeros_like(dummy_x, dtype=torch.uint8)
    dummy_mask = torch.zeros((num_experts,), device=device, dtype=torch.int32)
    if column_major_scales:
        dummy_s = torch.empty(
            (groups_per_row, rows), device=device, dtype=torch.float32
        ).permute(-1, -2)
        scale_leading_dim = 0
    else:
        dummy_s = torch.empty(
            (rows, groups_per_row), device=device, dtype=torch.float32
        )
        scale_leading_dim = 1

    mX = convert_from_dlpack(dummy_x.detach(), leading_dim=1)
    mXq = convert_from_dlpack(dummy_q_bits.detach(), leading_dim=1)
    mXq.element_type = Float8E4M3FN
    mXs = convert_from_dlpack(dummy_s.detach(), leading_dim=scale_leading_dim)
    mMask = convert_from_dlpack(dummy_mask.detach(), leading_dim=0)

    compiled = cute.compile(
        _launch_group_quant_kernel,
        mX,
        mXq,
        mXs,
        mMask,
        rows,
        max(1, rows // max(num_experts, 1)),
        num_experts,
        max(1, groups_per_row),
        1e-10,
        torch.finfo(torch.float8_e4m3fn).min,
        torch.finfo(torch.float8_e4m3fn).max,
    )
    _COMPILE_CACHE[key] = compiled
    return compiled


def per_token_group_quant_fp8_masked(
    x: torch.Tensor,
    masked_m: torch.Tensor,
    group_size: int = QUANT_GROUP_SIZE,
    eps: float = 1e-10,
    dtype: Optional[torch.dtype] = None,
    column_major_scales: bool = False,
    out_q: Optional[torch.Tensor] = None,
    use_ue8m0: Optional[bool] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.ndim == 2, "`x` must be 2D after flattening expert tiles."
    assert masked_m.ndim == 1, "`masked_m` must be 1D."
    assert x.is_cuda and masked_m.is_cuda, "Inputs must be CUDA tensors."
    assert x.dtype in (torch.float16, torch.bfloat16), "Only FP16/BF16 inputs supported."
    assert masked_m.shape[0] > 0, "`masked_m` must contain expert counts."
    assert x.shape[0] % masked_m.shape[0] == 0, "`x` rows must be divisible by num experts."
    assert group_size > 0, "`group_size` must be positive."
    if use_ue8m0:
        raise NotImplementedError("use_ue8m0=True is not supported in the Cute-DSL path yet.")
    dtype = torch.float8_e4m3fn if dtype is None else dtype
    if dtype != torch.float8_e4m3fn:
        raise NotImplementedError("Only torch.float8_e4m3fn is supported for the Cute-DSL path.")

    rows, hidden = x.shape
    assert hidden % group_size == 0, "`hidden` dimension must be divisible by `group_size`."
    assert group_size == QUANT_GROUP_SIZE, (
        f"group_size={group_size} is not supported; only {QUANT_GROUP_SIZE} works in this kernel."
    )
    num_experts = masked_m.shape[0]
    rows_per_expert = rows // num_experts
    groups_per_row = hidden // group_size

    x_contig = x.contiguous()
    masked_contig = masked_m.contiguous()

    if out_q is None:
        x_q = torch.empty_like(x_contig, dtype=dtype)
    else:
        assert out_q.shape == x_contig.shape, "`out_q` must have the same shape as `x`."
        assert out_q.device == x_contig.device, "`out_q` must be on the same device as `x`."
        assert out_q.dtype == dtype, "`out_q` must match the requested dtype."
        assert out_q.is_contiguous(), "`out_q` must be contiguous."
        x_q = out_q
    x_q_bits = x_q.view(torch.uint8)
    if column_major_scales:
        x_s = torch.empty(
            (groups_per_row, rows), device=x.device, dtype=torch.float32
        ).permute(-1, -2)
        scale_leading_dim = 0
    else:
        x_s = torch.empty(
            (rows, groups_per_row), device=x.device, dtype=torch.float32
        )
        scale_leading_dim = 1

    compiled = _get_compiled_kernel(
        x_contig.shape,
        num_experts,
        column_major_scales,
        x_contig.dtype,
        x_contig.device,
    )

    mX = convert_from_dlpack(x_contig.detach(), leading_dim=1)
    mXq = convert_from_dlpack(x_q_bits.detach(), leading_dim=1)
    mXq.element_type = Float8E4M3FN
    mXs = convert_from_dlpack(x_s.detach(), leading_dim=scale_leading_dim)
    mMask = convert_from_dlpack(masked_contig.detach(), leading_dim=0)

    finfo = torch.finfo(dtype)
    compiled(
        mX,
        mXq,
        mXs,
        mMask,
        rows,
        rows_per_expert,
        num_experts,
        groups_per_row,
        eps,
        finfo.min,
        finfo.max,
    )

    return x_q, x_s
