"""Cute-DSL implementation of fused SiLU + FP8 group quantization with masked rows."""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
import cutlass.utils
from cutlass import Boolean, Float32, Float8E4M3FN, Int16, Int32, Int64
from cutlass._mlir.dialects import vector
from cutlass.cutlass_dsl import T, dsl_user_op

from optimus_cutedsl.utils import (
    convert_from_dlpack,
    cvt_fp32x2_to_e4m3x2,
    elem_pointer,
    silu,
)
from .group_quant_fp8_masked import QUANT_GROUP_SIZE, ROWS_PER_CHUNK


THREADS_PER_CTA = 512
LANES_PER_WARP = cute.arch.WARP_SIZE
WARPS_PER_CTA = THREADS_PER_CTA // LANES_PER_WARP
VALS_PER_THREAD = 8
HALF_WARP = LANES_PER_WARP // 2
ELEMENTS_PER_HALF_WARP = HALF_WARP * VALS_PER_THREAD
assert (
    ELEMENTS_PER_HALF_WARP == QUANT_GROUP_SIZE
), "Each half warp must cover exactly one quantization group."
GROUPS_PER_WARP = 2
GROUPS_PER_CTA = GROUPS_PER_WARP * WARPS_PER_CTA
assert GROUPS_PER_CTA * QUANT_GROUP_SIZE == 4096, "CTA must cover 4096 hidden."
HALF_WARP_MASK_AND_CLAMP = ((LANES_PER_WARP - HALF_WARP) << 8) | (HALF_WARP - 1)
ROWS_PER_TILE = 2
CP_ASYNC_STAGE_CANDIDATES = (2, 3, 4, 5)
DEFAULT_CP_ASYNC_STAGES = CP_ASYNC_STAGE_CANDIDATES[-1]
CP_ASYNC_STAGES = DEFAULT_CP_ASYNC_STAGES
CTA_COLS = GROUPS_PER_CTA * QUANT_GROUP_SIZE
assert ROWS_PER_CHUNK % ROWS_PER_TILE == 0, "Tile size must divide ROWS_PER_CHUNK."
TILES_PER_CHUNK = ROWS_PER_CHUNK // ROWS_PER_TILE


@dsl_user_op
def pack_int16x4_to_int64(
    v0: Int16, v1: Int16, v2: Int16, v3: Int16, *, loc=None, ip=None
) -> Int64:
    vec_i16 = vector.from_elements(
        T.vector(4, T.i16()),
        (
            Int16(v0).ir_value(loc=loc, ip=ip),
            Int16(v1).ir_value(loc=loc, ip=ip),
            Int16(v2).ir_value(loc=loc, ip=ip),
            Int16(v3).ir_value(loc=loc, ip=ip),
        ),
        loc=loc,
        ip=ip,
    )
    vec_i64 = vector.bitcast(T.vector(1, T.i64()), vec_i16)
    packed = vector.extract(vec_i64, dynamic_position=[], static_position=[0], loc=loc, ip=ip)
    return Int64(packed)


@dsl_user_op
def store_fp8x8(
    tensor: cute.Tensor, coord: cute.Coord, packed: Int64, *, loc=None, ip=None
) -> None:
    ptr = elem_pointer(tensor, coord, loc=loc, ip=ip).align(8)
    int_ptr = cute.make_ptr(
        Int64,
        ptr.toint(),
        tensor.memspace,
        assumed_align=min(ptr.max_alignment, 8),
    )
    dst = cute.make_tensor(int_ptr, cute.make_layout((1,), stride=(1,)))
    dst[0] = packed


@cute.jit
def _half_warp_reduce_max(val: Float32) -> Float32:
    mask_and_clamp = Int32(HALF_WARP_MASK_AND_CLAMP)
    full_mask = Int32(-1)
    for itr in cutlass.range_constexpr(int(math.log2(HALF_WARP))):
        offset = HALF_WARP >> (itr + 1)
        other = cute.arch.shuffle_sync_bfly(
            val,
            offset=Int32(offset),
            mask=full_mask,
            mask_and_clamp=mask_and_clamp,
        )
        val = cute.arch.fmax(val, other)
    return val


def _round_up(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def _device_cache_key(device: torch.device) -> Tuple[str, Optional[int]]:
    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        return (device.type, index)
    return (device.type, None)


def _benchmark_cp_async_stage(
    compiled: cute.JitFunction,
    mGate: cute.Tensor,
    mUp: cute.Tensor,
    mXq: cute.Tensor,
    mXs: cute.Tensor,
    mMask: cute.Tensor,
    rows: int,
    rows_per_expert: int,
    num_experts: int,
    groups_per_row: int,
    num_m_cta_per_expert: int,
    num_quant_cta_per_expert: int,
    eps: float,
    fp8_min: float,
    fp8_max: float,
    gate_limit: float,
    has_gate_limit: bool,
    device_index: Optional[int],
) -> float:
    def _get_stream(idx: Optional[int] = None) -> cuda.CUstream:
        if idx is None:
            torch_stream = torch.cuda.current_stream()
        else:
            torch_stream = torch.cuda.current_stream(device=idx)
        return cuda.CUstream(torch_stream.cuda_stream)

    if device_index is None:
        stream = _get_stream()
        compiled(
            mGate,
            mUp,
            mXq,
            mXs,
            mMask,
            rows,
            rows_per_expert,
            num_experts,
            groups_per_row,
            num_m_cta_per_expert,
            num_quant_cta_per_expert,
            eps,
            fp8_min,
            fp8_max,
            gate_limit,
            stream,
        )
        return 0.0

    with torch.cuda.device(device_index):
        torch.cuda.synchronize()
        warmup_start = torch.cuda.Event(enable_timing=True)
        warmup_end = torch.cuda.Event(enable_timing=True)
        warmup_start.record()
        stream = _get_stream(device_index)
        compiled(
            mGate,
            mUp,
            mXq,
            mXs,
            mMask,
            rows,
            rows_per_expert,
            num_experts,
            groups_per_row,
            num_m_cta_per_expert,
            num_quant_cta_per_expert,
            eps,
            fp8_min,
            fp8_max,
            gate_limit,
            stream,
        )
        warmup_end.record()
        warmup_end.synchronize()

        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        compiled(
            mGate,
            mUp,
            mXq,
            mXs,
            mMask,
            rows,
            rows_per_expert,
            num_experts,
            groups_per_row,
            num_m_cta_per_expert,
            num_quant_cta_per_expert,
            eps,
            fp8_min,
            fp8_max,
            gate_limit,
            stream,
        )
        end.record()
        end.synchronize()
        torch.cuda.synchronize()
        return float(start.elapsed_time(end))


_STAGE_SELECTION_CACHE: Dict[
    Tuple[int, int, int, bool, torch.dtype, Tuple[str, Optional[int]], int, int, bool], int
] = {}
_LAUNCH_PARAMS_CACHE: Dict[
    Tuple[int, int, int, Tuple[str, Optional[int]]], Tuple[int, int]
] = {}


def _compute_launch_params(
    rows_per_expert: int,
    groups_per_row: int,
    num_experts: int,
    device: torch.device,
) -> Tuple[int, int]:
    device_key = _device_cache_key(device)
    cache_key = (rows_per_expert, groups_per_row, num_experts, device_key)
    cached = _LAUNCH_PARAMS_CACHE.get(cache_key)
    if cached is not None:
        return cached

    num_quant_cta = max(1, (groups_per_row + GROUPS_PER_CTA - 1) // GROUPS_PER_CTA)
    sm_count = 1
    if device.type == "cuda":
        try:
            dev_idx = device_key[1]
            if dev_idx is None:
                dev_idx = torch.cuda.current_device()
            sm_count = torch.cuda.get_device_properties(dev_idx).multi_processor_count
        except Exception:
            sm_count = 1
    total_quant_cta = max(1, num_experts * num_quant_cta)
    max_m_tiles = max(1, math.ceil(rows_per_expert / ROWS_PER_CHUNK))
    max_total_cta = max(total_quant_cta, sm_count * 4)

    best_wave_frac = float("inf")
    best_wave_under = float("inf")
    best_total_cta = total_quant_cta
    num_m_cta = 1

    for candidate_m in range(1, max_m_tiles + 1):
        total_cta = total_quant_cta * candidate_m
        if total_cta > max_total_cta:
            total_cta = max_total_cta
            candidate_m = max(1, total_cta // total_quant_cta)
        wave = total_cta / max(1, sm_count)
        frac = abs(wave - round(wave))
        wave_under = wave - math.floor(wave)

        if frac < best_wave_frac:
            best_wave_frac = frac
            best_wave_under = wave_under
            best_total_cta = total_cta
            num_m_cta = candidate_m
        elif math.isclose(frac, best_wave_frac):
            if wave_under < best_wave_under:
                best_wave_under = wave_under
                best_total_cta = total_cta
                num_m_cta = candidate_m
            elif math.isclose(wave_under, best_wave_under) and total_cta < best_total_cta:
                best_total_cta = total_cta
                num_m_cta = candidate_m

    result = (num_m_cta, num_quant_cta)
    _LAUNCH_PARAMS_CACHE[cache_key] = result
    return result


def _ensure_innermost_contiguous(x: torch.Tensor) -> torch.Tensor:
    """Avoid unnecessary clones; only enforce contiguous innermost dimension."""
    return x if x.stride(-1) == 1 else x.contiguous()


def _select_cp_async_stage(
    shape: Tuple[int, int],
    num_experts: int,
    column_major_scales: bool,
    dtype: torch.dtype,
    device: torch.device,
    num_m_cta_per_expert: int,
    num_quant_cta_per_expert: int,
    rows: int,
    rows_per_expert: int,
    groups_per_row: int,
    mGate: cute.Tensor,
    mUp: cute.Tensor,
    mMask: cute.Tensor,
    eps: float,
    fp8_min: float,
    fp8_max: float,
    scale_leading_dim: int,
    x_q_bits: torch.Tensor,
    x_s: torch.Tensor,
    has_gate_limit: bool,
    gate_limit: float,
) -> int:
    device_key = _device_cache_key(device)
    cache_key = (
        shape[0],
        shape[1],
        num_experts,
        column_major_scales,
        dtype,
        device_key,
        num_m_cta_per_expert,
        num_quant_cta_per_expert,
        has_gate_limit,
    )
    cached = _STAGE_SELECTION_CACHE.get(cache_key)
    if cached is not None:
        return cached

    if device.type != "cuda" or not torch.cuda.is_available():
        _STAGE_SELECTION_CACHE[cache_key] = DEFAULT_CP_ASYNC_STAGES
        return DEFAULT_CP_ASYNC_STAGES

    torch_stream = torch.cuda.current_stream(device=device)
    benchmark_stream = cuda.CUstream(torch_stream.cuda_stream)

    scratch_q_bits = torch.empty_like(x_q_bits)
    scratch_s = torch.empty_like(x_s)
    scratch_mXq = convert_from_dlpack(scratch_q_bits.detach(), leading_dim=1)
    scratch_mXq.element_type = Float8E4M3FN
    scratch_mXs = convert_from_dlpack(scratch_s.detach(), leading_dim=scale_leading_dim)

    best_stage = DEFAULT_CP_ASYNC_STAGES
    best_time_ms = float("inf")
    for stage in CP_ASYNC_STAGE_CANDIDATES:
        try:
            compiled = _get_compiled_kernel(
                shape,
                num_experts,
                column_major_scales,
                dtype,
                device,
                stage,
                num_m_cta_per_expert,
                num_quant_cta_per_expert,
                has_gate_limit,
                benchmark_stream,
            )
        except Exception:
            continue

        try:
            time_ms = _benchmark_cp_async_stage(
                compiled,
                mGate,
                mUp,
                scratch_mXq,
                scratch_mXs,
                mMask,
                rows,
                rows_per_expert,
                num_experts,
                groups_per_row,
                num_m_cta_per_expert,
                num_quant_cta_per_expert,
                eps,
                fp8_min,
                fp8_max,
                gate_limit,
                has_gate_limit,
                device_key[1],
            )
        except Exception:
            continue

        if time_ms < best_time_ms:
            best_time_ms = time_ms
            best_stage = stage

    _STAGE_SELECTION_CACHE[cache_key] = best_stage
    return best_stage


@cute.kernel
def _silu_mul_group_quant_fp8_masked_kernel(
    mGate: cute.Tensor,
    mUp: cute.Tensor,
    mXq: cute.Tensor,
    mXs: cute.Tensor,
    mMask: cute.Tensor,
    rows: Int32,
    rows_per_expert: Int32,
    num_experts: Int32,
    groups_per_row: Int32,
    num_m_tiles: Int32,
    eps: Float32,
    fp8_min: Float32,
    fp8_max: Float32,
    gate_limit: Float32,
    has_gate_limit: cutlass.Constexpr = False,
):

    cta_quant, cta_m, expert_id = cute.arch.block_idx()
    warp_idx = cute.arch.warp_idx()
    lane_idx = cute.arch.lane_idx()
    copy_atom_fp32 = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(),
        Float32,
        num_bits_per_copy=32,
    )
    copy_atom_b16 = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(),
        mGate.element_type,
        num_bits_per_copy=128,
    )
    copy_atom_async = cute.make_copy_atom(
        cute.nvgpu.cpasync.CopyG2SOp(),
        mGate.element_type,
        num_bits_per_copy=128,
    )

    @cute.struct
    class SharedStorage:
        gate: cute.struct.Align[
            cute.struct.MemRange[mGate.element_type, CP_ASYNC_STAGES * ROWS_PER_TILE * CTA_COLS],
            128,
        ]
        up: cute.struct.Align[
            cute.struct.MemRange[mUp.element_type, CP_ASYNC_STAGES * ROWS_PER_TILE * CTA_COLS],
            128,
        ]

    smem = cutlass.utils.SmemAllocator()
    storage = smem.allocate(SharedStorage)
    gate_smem = storage.gate.get_tensor(
        cute.make_layout(
            (CP_ASYNC_STAGES, ROWS_PER_TILE, CTA_COLS),
            stride=(ROWS_PER_TILE * CTA_COLS, CTA_COLS, 1),
        )
    )
    up_smem = storage.up.get_tensor(
        cute.make_layout(
            (CP_ASYNC_STAGES, ROWS_PER_TILE, CTA_COLS),
            stride=(ROWS_PER_TILE * CTA_COLS, CTA_COLS, 1),
        )
    )

    if expert_id < num_experts:
        mask_limit = mMask[expert_id]

        if mask_limit > 0:
            row_base = expert_id * rows_per_expert
            group_tile_start = cta_quant * GROUPS_PER_CTA

            if group_tile_start < groups_per_row and cta_m < num_m_tiles:
                total_chunks = cute.ceil_div(rows_per_expert, ROWS_PER_CHUNK)
                max_chunk_iters = cute.ceil_div(total_chunks, num_m_tiles)
                lane_half = lane_idx // Int32(HALF_WARP)
                lane_in_half = lane_idx - lane_half * Int32(HALF_WARP)
                base_col = lane_in_half * Int32(VALS_PER_THREAD)
                lane_vals = cute.make_rmem_tensor((VALS_PER_THREAD,), Float32)
                gate_fragment = cute.make_rmem_tensor((VALS_PER_THREAD,), mGate.element_type)
                up_fragment = cute.make_rmem_tensor((VALS_PER_THREAD,), mUp.element_type)
                packed_pairs = cute.make_rmem_tensor((VALS_PER_THREAD // 2,), Int16)

                group_idx = group_tile_start + warp_idx * GROUPS_PER_WARP + lane_half
                if group_idx < groups_per_row:
                    group_start = group_idx * QUANT_GROUP_SIZE
                    for chunk_itr in cutlass.range(max_chunk_iters):
                        chunk_idx = chunk_itr * num_m_tiles + cta_m
                        if chunk_idx < total_chunks:
                            chunk_row_start = chunk_idx * ROWS_PER_CHUNK
                            chunk_start_i32 = Int32(chunk_row_start)
                            rows_active = mask_limit - chunk_start_i32
                            if rows_active > Int32(ROWS_PER_CHUNK):
                                rows_active = Int32(ROWS_PER_CHUNK)
                            if rows_active < Int32(0):
                                rows_active = Int32(0)
                            if chunk_row_start < rows_per_expert and rows_active > Int32(0):
                                tiles_in_chunk = (
                                    rows_active + Int32(ROWS_PER_TILE - 1)
                                ) // Int32(ROWS_PER_TILE)
                                prefetched = Int32(0)
                                for stage_prefetch in cutlass.range_constexpr(
                                    CP_ASYNC_STAGES - 1
                                ):
                                    if stage_prefetch < tiles_in_chunk:
                                        tile_prefetch = Int32(stage_prefetch)
                                        tile_row_start = (
                                            chunk_row_start
                                            + tile_prefetch * Int32(ROWS_PER_TILE)
                                        )
                                        tile_has_valid = Boolean(False)
                                        for row_local in cutlass.range_constexpr(ROWS_PER_TILE):
                                            row_itr = tile_row_start + Int32(row_local)
                                            row = row_base + row_itr
                                            if row_itr < rows_per_expert and row_itr < mask_limit:
                                                gate_src = cute.make_tensor(
                                                    elem_pointer(
                                                        mGate, (row, group_start + base_col)
                                                    ).align(16),
                                                    cute.make_layout((VALS_PER_THREAD,), stride=(1,)),
                                                )
                                                gate_dst = cute.make_tensor(
                                                    elem_pointer(
                                                        gate_smem,
                                                        (
                                                            tile_prefetch,
                                                            row_local,
                                                            group_start + base_col,
                                                        ),
                                                    ).align(16),
                                                    cute.make_layout((VALS_PER_THREAD,), stride=(1,)),
                                                )
                                                up_src = cute.make_tensor(
                                                    elem_pointer(
                                                        mUp, (row, group_start + base_col)
                                                    ).align(16),
                                                    cute.make_layout((VALS_PER_THREAD,), stride=(1,)),
                                                )
                                                up_dst = cute.make_tensor(
                                                    elem_pointer(
                                                        up_smem,
                                                        (
                                                            tile_prefetch,
                                                            row_local,
                                                            group_start + base_col,
                                                        ),
                                                    ).align(16),
                                                    cute.make_layout((VALS_PER_THREAD,), stride=(1,)),
                                                )
                                                cute.copy(copy_atom_async, gate_src, gate_dst)
                                                cute.copy(copy_atom_async, up_src, up_dst)
                                                tile_has_valid = Boolean(True)
                                        if tile_has_valid:
                                            cute.arch.cp_async_commit_group()
                                            prefetched += Int32(1)
                                if prefetched > Int32(0):
                                    stage_mod = Int32(CP_ASYNC_STAGES)
                                    stage_read = Int32(0)
                                    stage_write = prefetched
                                    if stage_write >= stage_mod:
                                        stage_write = stage_write - stage_mod
                                    next_tile = prefetched
                                    tiles_processed = Int32(0)
                                    has_full_pipeline = Boolean(
                                        prefetched >= Int32(CP_ASYNC_STAGES - 1)
                                    )
                                    cute.arch.cp_async_wait_group(0)
                                    cute.arch.sync_threads()
                                    tile_has_valid = Boolean(False)
                                    for tile_local in cutlass.range(tiles_in_chunk):
                                        tile_row_start = (
                                            chunk_row_start
                                            + Int32(tile_local * ROWS_PER_TILE)
                                        )
                                        for row_local in cutlass.range_constexpr(ROWS_PER_TILE):
                                            row_itr = tile_row_start + Int32(row_local)
                                            if row_itr < mask_limit and row_itr < rows_per_expert:
                                                row = row_base + row_itr
                                                local_absmax = Float32(0.0)
                                                gate_tile = cute.make_tensor(
                                                    elem_pointer(
                                                        gate_smem,
                                                        (stage_read, row_local, group_start + base_col),
                                                    ).align(16),
                                                    cute.make_layout((VALS_PER_THREAD,), stride=(1,)),
                                                )
                                                up_tile = cute.make_tensor(
                                                    elem_pointer(
                                                        up_smem,
                                                        (stage_read, row_local, group_start + base_col),
                                                    ).align(16),
                                                    cute.make_layout((VALS_PER_THREAD,), stride=(1,)),
                                                )
                                                cute.copy(copy_atom_b16, gate_tile, gate_fragment)
                                                cute.copy(copy_atom_b16, up_tile, up_fragment)
                                                for elem in cutlass.range_constexpr(VALS_PER_THREAD):
                                                    gate_val = Float32(gate_fragment[elem])
                                                    up_val = Float32(up_fragment[elem])
                                                    if cutlass.const_expr(has_gate_limit):
                                                        gate_val = cute.arch.fmax(gate_val, gate_limit)
                                                        up_val = cute.arch.fmax(up_val, gate_limit)
                                                        up_val = - up_val
                                                        up_val = cute.arch.fmax(up_val, gate_limit)
                                                        up_val = - up_val
                                                    out_val = silu(gate_val) * up_val
                                                    lane_vals[elem] = out_val
                                                    abs_val = cute.arch.fmax(out_val, -out_val)
                                                    local_absmax = cute.arch.fmax(
                                                        local_absmax, abs_val
                                                    )

                                                max_abs = _half_warp_reduce_max(local_absmax)
                                                max_abs = cute.arch.fmax(max_abs, eps)
                                                scale = max_abs / fp8_max
                                                scale_inv = fp8_max / max_abs

                                                if lane_in_half == 0:
                                                    scale_src = cute.make_fragment((1,), Float32)
                                                    scale_src[0] = scale
                                                    scale_dst = cute.make_tensor(
                                                        elem_pointer(mXs, (row, group_idx)),
                                                        cute.make_layout((1,), stride=(1,)),
                                                    )
                                                    cute.copy(copy_atom_fp32, scale_src, scale_dst)

                                                for pair in cutlass.range_constexpr(VALS_PER_THREAD // 2):
                                                    idx0 = pair * 2
                                                    idx1 = idx0 + 1
                                                    val0 = Float32(lane_vals[idx0]) * scale_inv
                                                    val0 = cute.arch.fmax(-fp8_max, -val0)
                                                    val0 = -val0
                                                    val0 = cute.arch.fmax(fp8_min, val0)
                                                    val1 = Float32(lane_vals[idx1]) * scale_inv
                                                    val1 = cute.arch.fmax(-fp8_max, -val1)
                                                    val1 = -val1
                                                    val1 = cute.arch.fmax(fp8_min, val1)
                                                    packed_pairs[pair] = cvt_fp32x2_to_e4m3x2(
                                                        val1, val0
                                                    )

                                                packed_vec = pack_int16x4_to_int64(
                                                    packed_pairs[0],
                                                    packed_pairs[1],
                                                    packed_pairs[2],
                                                    packed_pairs[3],
                                                )
                                                store_fp8x8(
                                                    mXq,
                                                    (
                                                        row,
                                                        group_idx * QUANT_GROUP_SIZE + base_col,
                                                    ),
                                                    packed_vec,
                                        )
                                        if next_tile < tiles_in_chunk:
                                            next_row_start = (
                                                chunk_row_start + next_tile * Int32(ROWS_PER_TILE)
                                            )
                                            tile_has_valid = Boolean(False)
                                            for row_local in cutlass.range_constexpr(ROWS_PER_TILE):
                                                row_itr = next_row_start + Int32(row_local)
                                                row = row_base + row_itr
                                                if row_itr < rows_per_expert and row_itr < mask_limit:
                                                    gate_src = cute.make_tensor(
                                                        elem_pointer(
                                                            mGate, (row, group_start + base_col)
                                                        ).align(16),
                                                        cute.make_layout(
                                                            (VALS_PER_THREAD,), stride=(1,)
                                                        ),
                                                    )
                                                    gate_dst = cute.make_tensor(
                                                        elem_pointer(
                                                            gate_smem,
                                                            (
                                                                stage_write,
                                                                row_local,
                                                                group_start + base_col,
                                                            ),
                                                        ).align(16),
                                                        cute.make_layout(
                                                            (VALS_PER_THREAD,), stride=(1,)
                                                        ),
                                                    )
                                                    up_src = cute.make_tensor(
                                                        elem_pointer(
                                                            mUp, (row, group_start + base_col)
                                                        ).align(16),
                                                        cute.make_layout(
                                                            (VALS_PER_THREAD,), stride=(1,)
                                                        ),
                                                    )
                                                    up_dst = cute.make_tensor(
                                                        elem_pointer(
                                                            up_smem,
                                                            (
                                                                stage_write,
                                                                row_local,
                                                                group_start + base_col,
                                                            ),
                                                        ).align(16),
                                                        cute.make_layout(
                                                            (VALS_PER_THREAD,), stride=(1,)
                                                        ),
                                                    )
                                                    cute.copy(copy_atom_async, gate_src, gate_dst)
                                                    cute.copy(copy_atom_async, up_src, up_dst)
                                                    tile_has_valid = Boolean(True)
                                        if tile_has_valid:
                                            cute.arch.cp_async_commit_group()
                                            next_tile += Int32(1)
                                            stage_write = stage_write + Int32(1)
                                            if stage_write >= stage_mod:
                                                stage_write = stage_write - stage_mod
                                        stage_read = stage_read + Int32(1)
                                        if stage_read >= stage_mod:
                                            stage_read = stage_read - stage_mod
                                        tiles_processed = tiles_processed + Int32(1)
                                        if tile_local + Int32(1) < tiles_in_chunk:
                                            tiles_remaining = tiles_in_chunk - tiles_processed
                                            if has_full_pipeline and tiles_remaining >= Int32(
                                                CP_ASYNC_STAGES - 1
                                            ):
                                                cute.arch.cp_async_wait_group(CP_ASYNC_STAGES - 2)
                                            else:
                                                cute.arch.cp_async_wait_group(0)
                                            cute.arch.sync_threads()
@cute.jit
def _launch_silu_mul_group_quant_kernel(
    mGate: cute.Tensor,
    mUp: cute.Tensor,
    mXq: cute.Tensor,
    mXs: cute.Tensor,
    mMask: cute.Tensor,
    rows: int,
    rows_per_expert: int,
    num_experts: int,
    groups_per_row: int,
    num_m_cta_per_expert: int,
    num_quant_cta_per_expert: int,
    eps: float,
    fp8_min: float,
    fp8_max: float,
    gate_limit: float,
    stream: cuda.CUstream,
    has_gate_limit: cutlass.Constexpr = False,
):
    _silu_mul_group_quant_fp8_masked_kernel(
        mGate,
        mUp,
        mXq,
        mXs,
        mMask,
        Int32(rows),
        Int32(rows_per_expert),
        Int32(num_experts),
        Int32(groups_per_row),
        Int32(num_m_cta_per_expert),
        Float32(eps),
        Float32(fp8_min),
        Float32(fp8_max),
        Float32(gate_limit),
        has_gate_limit=has_gate_limit,
    ).launch(
        grid=[num_quant_cta_per_expert, num_m_cta_per_expert, num_experts],
        block=[THREADS_PER_CTA, 1, 1],
        stream=stream,
    )


_COMPILE_CACHE: Dict[
    Tuple[int, int, bool, torch.dtype, int, Tuple[str, Optional[int]], int, int, int, bool],
    cute.JitFunction,
] = {}


def _get_compiled_kernel(
    shape: Tuple[int, int],
    num_experts: int,
    column_major_scales: bool,
    dtype: torch.dtype,
    device: torch.device,
    cp_async_stages: int,
    num_m_cta_per_expert: int,
    num_quant_cta_per_expert: int,
    has_gate_limit: bool,
    stream: cuda.CUstream,
) -> cute.JitFunction:
    device_key = _device_cache_key(device)
    key = (
        shape[0],
        shape[1],
        column_major_scales,
        dtype,
        num_experts,
        device_key,
        cp_async_stages,
        num_m_cta_per_expert,
        num_quant_cta_per_expert,
        has_gate_limit,
    )
    cached = _COMPILE_CACHE.get(key)
    if cached is not None:
        return cached

    rows, hidden = shape
    rows_per_expert = rows // max(num_experts, 1)
    groups_per_row = hidden // QUANT_GROUP_SIZE

    dummy_gate = torch.zeros(shape, device=device, dtype=dtype)
    dummy_up = torch.zeros_like(dummy_gate)
    dummy_q_bits = torch.zeros_like(dummy_gate, dtype=torch.uint8)
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

    mGate = convert_from_dlpack(dummy_gate.detach(), leading_dim=1)
    mUp = convert_from_dlpack(dummy_up.detach(), leading_dim=1)
    mXq = convert_from_dlpack(dummy_q_bits.detach(), leading_dim=1)
    mXq.element_type = Float8E4M3FN
    mXs = convert_from_dlpack(dummy_s.detach(), leading_dim=scale_leading_dim)
    mMask = convert_from_dlpack(dummy_mask.detach(), leading_dim=0)

    prev_stage = CP_ASYNC_STAGES
    try:
        globals()["CP_ASYNC_STAGES"] = cp_async_stages
        compiled = cute.compile(
            _launch_silu_mul_group_quant_kernel,
            mGate,
            mUp,
            mXq,
            mXs,
            mMask,
            rows,
            rows_per_expert,
            num_experts,
            max(1, groups_per_row),
            num_m_cta_per_expert,
            num_quant_cta_per_expert,
            1e-10,
            torch.finfo(torch.float8_e4m3fn).min,
            torch.finfo(torch.float8_e4m3fn).max,
            0.0,
            stream,
            cutlass.const_expr(has_gate_limit),
        )
    finally:
        globals()["CP_ASYNC_STAGES"] = prev_stage

    _COMPILE_CACHE[key] = compiled
    return compiled


def silu_mul_group_quant_fp8_masked(
    x_gate: torch.Tensor,
    x_up: torch.Tensor,
    masked_m: torch.Tensor,
    group_size: int = QUANT_GROUP_SIZE,
    eps: float = 1e-10,
    gate_limit: Optional[float] = None,
    dtype: Optional[torch.dtype] = None,
    column_major_scales: bool = False,
    out_q: Optional[torch.Tensor] = None,
    out_scales: Optional[torch.Tensor] = None,
    use_ue8m0: Optional[bool] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x_gate.ndim == 2, "`x_gate` must be 2D after flattening expert tiles."
    assert x_up.ndim == 2, "`x_up` must be 2D after flattening expert tiles."
    assert masked_m.ndim == 1, "`masked_m` must be 1D."
    assert x_gate.shape == x_up.shape, "`x_gate` and `x_up` must have the same shape."
    assert x_gate.is_cuda and x_up.is_cuda and masked_m.is_cuda, "Inputs must be CUDA tensors."
    assert x_gate.dtype in (torch.float16, torch.bfloat16), "Only FP16/BF16 inputs supported."
    assert x_gate.dtype == x_up.dtype, "`x_gate` and `x_up` must share the same dtype."
    assert masked_m.shape[0] > 0, "`masked_m` must contain expert counts."
    assert x_gate.shape[0] % masked_m.shape[0] == 0, "`x_gate` rows must be divisible by num experts."
    assert group_size > 0, "`group_size` must be positive."
    if use_ue8m0:
        raise NotImplementedError("use_ue8m0=True is not supported in the Cute-DSL path yet.")
    dtype = torch.float8_e4m3fn if dtype is None else dtype
    if dtype != torch.float8_e4m3fn:
        raise NotImplementedError("Only torch.float8_e4m3fn is supported for the Cute-DSL path.")

    rows, hidden = x_gate.shape
    assert hidden % group_size == 0, "`hidden` dimension must be divisible by `group_size`."
    assert group_size == QUANT_GROUP_SIZE, (
        f"group_size={group_size} is not supported; only {QUANT_GROUP_SIZE} works in this kernel."
    )
    num_experts = masked_m.shape[0]
    rows_per_expert = rows // num_experts
    groups_per_row = hidden // group_size
    assert rows % num_experts == 0, "`x_gate` rows must be divisible by num experts."

    x_gate_contig = _ensure_innermost_contiguous(x_gate)
    x_up_contig = _ensure_innermost_contiguous(x_up)
    masked_contig = masked_m.contiguous()

    if out_q is None:
        x_q = torch.empty_like(x_up_contig, dtype=dtype)
    else:
        assert out_q.shape == x_up_contig.shape, "`out_q` must have the same shape as `x_up`."
        assert out_q.device == x_up_contig.device, "`out_q` must be on the same device as `x_up`."
        assert out_q.dtype == dtype, "`out_q` must match the requested dtype."
        assert out_q.is_contiguous(), "`out_q` must be contiguous."
        x_q = out_q
    x_q_bits = x_q.view(torch.uint8)
    if out_scales is None:
        if column_major_scales:
            x_s = torch.empty(
                (groups_per_row, rows), device=x_gate.device, dtype=torch.float32
            ).permute(-1, -2)
            scale_leading_dim = 0
        else:
            x_s = torch.empty(
                (rows, groups_per_row), device=x_gate.device, dtype=torch.float32
            )
            scale_leading_dim = 1
    else:
        x_s = out_scales
        if column_major_scales:
            scale_leading_dim = 0
        else:
            scale_leading_dim = 1
    mGate = convert_from_dlpack(x_gate_contig.detach(), leading_dim=1)
    mUp = convert_from_dlpack(x_up_contig.detach(), leading_dim=1)
    mXq = convert_from_dlpack(x_q_bits.detach(), leading_dim=1)
    mXq.element_type = Float8E4M3FN
    mXs = convert_from_dlpack(x_s.detach(), leading_dim=scale_leading_dim)
    mMask = convert_from_dlpack(masked_contig.detach(), leading_dim=0)
    has_gate_limit = gate_limit is not None
    gate_limit_value = float(gate_limit) if gate_limit is not None else 0.0
    num_m_cta_per_expert, num_quant_cta_per_expert = _compute_launch_params(
        rows_per_expert, groups_per_row, num_experts, x_gate_contig.device
    )

    finfo = torch.finfo(dtype)
    # cp_async_stage = _select_cp_async_stage(
    #     x_gate_contig.shape,
    #     num_experts,
    #     column_major_scales,
    #     x_gate_contig.dtype,
    #     x_gate_contig.device,
    #     num_m_cta_per_expert,
    #     num_quant_cta_per_expert,
    #     rows,
    #     rows_per_expert,
    #     groups_per_row,
    #     mGate,
    #     mUp,
    #     mMask,
    #     eps,
    #     finfo.min,
    #     finfo.max,
    #     scale_leading_dim,
    #     x_q_bits,
    #     x_s,
    #     has_gate_limit,
    #     gate_limit_value,
    # )
    cp_async_stage = 2
    torch_stream = torch.cuda.current_stream(device=x_gate_contig.device)
    current_stream = cuda.CUstream(torch_stream.cuda_stream)
    compiled = _get_compiled_kernel(
        x_gate_contig.shape,
        num_experts,
        column_major_scales,
        x_gate_contig.dtype,
        x_gate_contig.device,
        cp_async_stage,
        num_m_cta_per_expert,
        num_quant_cta_per_expert,
        has_gate_limit,
        current_stream,
    )

    compiled(
        mGate,
        mUp,
        mXq,
        mXs,
        mMask,
        rows,
        rows_per_expert,
        num_experts,
        groups_per_row,
        num_m_cta_per_expert,
        num_quant_cta_per_expert,
        eps,
        finfo.min,
        finfo.max,
        gate_limit_value,
        current_stream,
    )

    return x_q, x_s


__all__ = ["silu_mul_group_quant_fp8_masked"]
