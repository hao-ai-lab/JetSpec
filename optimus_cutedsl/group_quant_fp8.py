"""Cute-DSL implementation of per-token FP8 group quantization."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Float8E4M3FN, Int16, Int32, Int64
from cutlass._mlir.dialects import vector
from cutlass.cutlass_dsl import T, dsl_user_op

from optimus_cutedsl.utils import (
    convert_from_dlpack,
    cvt_fp32x2_to_e4m3x2,
    elem_pointer,
)


LANES_PER_WARP = cute.arch.WARP_SIZE
QUANT_GROUP_SIZE = 128
VALS_PER_THREAD = 8
HALF_WARP = LANES_PER_WARP // 2
GROUPS_PER_WARP = 2
HALF_WARP_MASK_AND_CLAMP = ((LANES_PER_WARP - HALF_WARP) << 8) | (HALF_WARP - 1)
assert QUANT_GROUP_SIZE == HALF_WARP * VALS_PER_THREAD, (
    "Half warp must cover one quantization group."
)

DEFAULT_THREADS_PER_CTA = 256
DEFAULT_ROWS_PER_CHUNK = 32


@dataclass(frozen=True)
class _KernelConfig:
    threads_per_cta: int
    rows_per_chunk: int


_DEFAULT_CONFIG = _KernelConfig(
    threads_per_cta=DEFAULT_THREADS_PER_CTA,
    rows_per_chunk=DEFAULT_ROWS_PER_CHUNK,
)

_THREADS_PER_CTA_OPTIONS = (128, 256, 384, 512, 768, 1024)
_ROWS_PER_CHUNK_OPTIONS = (16, 32, 64, 128, 256)
_WAVE_TARGETS = (1.0, 2.0, 4.0)

_AUTOTUNE_CONFIGS = tuple(
    _KernelConfig(threads_per_cta, rows_per_chunk)
    for threads_per_cta in _THREADS_PER_CTA_OPTIONS
    for rows_per_chunk in _ROWS_PER_CHUNK_OPTIONS
)


def _valid_config(config: _KernelConfig) -> bool:
    return (
        config.threads_per_cta > 0
        and config.rows_per_chunk > 0
        and config.threads_per_cta % LANES_PER_WARP == 0
        and config.threads_per_cta <= 1024
    )


_LAUNCH_KERNEL_CACHE: Dict[Tuple[int, int], cute.JitFunction] = {}


@dsl_user_op
def _pack_int16x4_to_int64(
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
def _store_fp8x8(
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


def _get_launch_kernel(config: _KernelConfig) -> cute.JitFunction:
    cache_key = (config.threads_per_cta, config.rows_per_chunk)
    cached = _LAUNCH_KERNEL_CACHE.get(cache_key)
    if cached is not None:
        return cached

    if not _valid_config(config):
        raise ValueError(f"Invalid kernel config: {config}")

    threads_per_cta = config.threads_per_cta
    rows_per_chunk = config.rows_per_chunk
    warps_per_cta = threads_per_cta // LANES_PER_WARP
    groups_per_cta = warps_per_cta * GROUPS_PER_WARP

    @cute.kernel
    def _per_token_group_quant_fp8_kernel(
        mX: cute.Tensor,
        mXq: cute.Tensor,
        mXs: cute.Tensor,
        rows: Int32,
        groups_per_row: Int32,
        eps: Float32,
        fp8_min: Float32,
        fp8_max: Float32,
        grid_y: Int32,
    ):
        cta_x, cta_y, _ = cute.arch.block_idx()
        warp_idx = cute.arch.warp_idx()
        lane_idx = cute.arch.lane_idx()

        half_idx = lane_idx // Int32(HALF_WARP)
        lane_in_half = lane_idx - half_idx * Int32(HALF_WARP)
        group_linear = cta_x * Int32(groups_per_cta) + warp_idx * Int32(GROUPS_PER_WARP) + half_idx
        total_group_tiles = groups_per_row

        if group_linear < total_group_tiles:
            group_idx = group_linear
            group_start = group_idx * Int32(QUANT_GROUP_SIZE)
            max_chunks = cute.ceil_div(rows, rows_per_chunk)
            base_col = lane_in_half * Int32(VALS_PER_THREAD)
            lane_vals = cute.make_rmem_tensor((VALS_PER_THREAD,), mX.element_type)
            packed_pairs = cute.make_rmem_tensor((VALS_PER_THREAD // 2,), Int16)

            row_tile = cta_y
            while row_tile < max_chunks:
                chunk_row_start = row_tile * rows_per_chunk
                for row_offset in cutlass.range(rows_per_chunk):
                    row = chunk_row_start + row_offset
                    if row < rows:
                        local_absmax = Float32(0.0)
                        src = cute.make_tensor(
                            elem_pointer(mX, (row, group_start + base_col)).align(16),
                            cute.make_layout((VALS_PER_THREAD,), stride=(1,)),
                        )
                        cute.autovec_copy(src, lane_vals)
                        for elem in cutlass.range_constexpr(VALS_PER_THREAD):
                            val = Float32(lane_vals[elem])
                            abs_val = cute.arch.fmax(val, -val)
                            local_absmax = cute.arch.fmax(local_absmax, abs_val)

                        max_abs = _half_warp_reduce_max(local_absmax)
                        max_abs = cute.arch.fmax(max_abs, eps)
                        scale = max_abs / fp8_max
                        scale_inv = fp8_max / max_abs

                        if lane_in_half == 0:
                            mXs[row, group_idx] = scale

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
                            packed_pairs[pair] = cvt_fp32x2_to_e4m3x2(val1, val0)

                        packed_vec = _pack_int16x4_to_int64(
                            packed_pairs[0],
                            packed_pairs[1],
                            packed_pairs[2],
                            packed_pairs[3],
                        )
                        _store_fp8x8(
                            mXq,
                            (row, group_start + base_col),
                            packed_vec,
                        )
                row_tile += grid_y

    @cute.jit
    def _launch_group_quant_kernel(
        mX: cute.Tensor,
        mXq: cute.Tensor,
        mXs: cute.Tensor,
        rows: int,
        groups_per_row: int,
        eps: float,
        fp8_min: float,
        fp8_max: float,
        grid_y: int,
    ):
        _per_token_group_quant_fp8_kernel(
            mX,
            mXq,
            mXs,
            Int32(rows),
            Int32(groups_per_row),
            Float32(eps),
            Float32(fp8_min),
            Float32(fp8_max),
            Int32(grid_y),
        ).launch(
            grid=[cute.ceil_div(groups_per_row, groups_per_cta), grid_y, 1],
            block=[threads_per_cta, 1, 1],
        )

    _LAUNCH_KERNEL_CACHE[cache_key] = _launch_group_quant_kernel
    return _launch_group_quant_kernel


_COMPILE_CACHE: Dict[
    Tuple[int, bool, torch.dtype, float, int, int], cute.JitFunction
] = {}
_AUTOTUNE_CACHE: Dict[
    Tuple[int, int, bool, torch.dtype, int], _KernelConfig
] = {}


def _compute_grid_x(groups_per_row: int, config: _KernelConfig) -> int:
    warps_per_cta = config.threads_per_cta // LANES_PER_WARP
    groups_per_cta = warps_per_cta * GROUPS_PER_WARP
    return max(1, (groups_per_row + groups_per_cta - 1) // groups_per_cta)

def _select_wave_target(desired_wave: float) -> float:
    best = _WAVE_TARGETS[0]
    best_diff = abs(desired_wave - best)
    for candidate in _WAVE_TARGETS[1:]:
        diff = abs(desired_wave - candidate)
        if diff < best_diff:
            best = candidate
            best_diff = diff
    return best


def _compute_grid_y(
    rows: int,
    groups_per_row: int,
    device: torch.device,
    config: _KernelConfig,
) -> Tuple[int, int, int, int, float]:
    row_tiles = max(1, (rows + config.rows_per_chunk - 1) // config.rows_per_chunk)
    grid_x = _compute_grid_x(groups_per_row, config)
    try:
        sm_count = torch.cuda.get_device_properties(device).multi_processor_count
    except Exception:
        sm_count = 1
    desired_wave = (grid_x * row_tiles) / max(1, sm_count)
    wave_target = _select_wave_target(desired_wave)
    target_ctas = max(1, int(math.ceil(wave_target * sm_count)))
    grid_y = min(row_tiles, max(1, int(math.ceil(target_ctas / grid_x))))
    return grid_y, sm_count, grid_x, row_tiles, wave_target


def _benchmark_kernel(
    compiled: cute.JitFunction,
    mX: cute.Tensor,
    mXq: cute.Tensor,
    mXs: cute.Tensor,
    rows: int,
    groups_per_row: int,
    eps: float,
    finfo: torch.finfo,
    grid_y: int,
    device: torch.device,
    warmup: int = 5,
    rep: int = 10,
) -> float:
    with torch.no_grad():
        with torch.cuda.device(device):
            for _ in range(warmup):
                compiled(
                    mX,
                    mXq,
                    mXs,
                    rows,
                    groups_per_row,
                    eps,
                    finfo.min,
                    finfo.max,
                    grid_y,
                )
            torch.cuda.synchronize(device)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(rep):
                compiled(
                    mX,
                    mXq,
                    mXs,
                    rows,
                    groups_per_row,
                    eps,
                    finfo.min,
                    finfo.max,
                    grid_y,
                )
            end.record()
            torch.cuda.synchronize(device)
    return start.elapsed_time(end) / max(rep, 1)


def _select_kernel_config(
    x: torch.Tensor,
    column_major_scales: bool,
    dtype: torch.dtype,
    eps: float,
) -> _KernelConfig:
    device = x.device
    rows, hidden = x.shape
    cache_key = (rows, hidden, column_major_scales, dtype, device.index or 0)
    cached = _AUTOTUNE_CACHE.get(cache_key)
    if cached is not None:
        return cached

    groups_per_row = hidden // QUANT_GROUP_SIZE
    x_q = torch.empty_like(x, dtype=dtype)
    x_q_bits = x_q.view(torch.uint8)

    if column_major_scales:
        x_s = torch.empty(
            (groups_per_row, rows), device=device, dtype=torch.float32
        ).permute(-1, -2)
        scale_leading_dim = 0
    else:
        x_s = torch.empty(
            (rows, groups_per_row), device=device, dtype=torch.float32
        )
        scale_leading_dim = 1

    mX = convert_from_dlpack(x.detach(), leading_dim=1)
    mXq = convert_from_dlpack(x_q_bits.detach(), leading_dim=1)
    mXq.element_type = Float8E4M3FN
    mXs = convert_from_dlpack(x_s.detach(), leading_dim=scale_leading_dim)

    finfo = torch.finfo(dtype)
    best_config = _DEFAULT_CONFIG
    best_time = float("inf")

    for config in _AUTOTUNE_CONFIGS:
        if not _valid_config(config):
            continue
        try:
            compiled = _get_compiled_kernel(
                x.shape,
                column_major_scales,
                x.dtype,
                device,
                config,
            )
            grid_y, _, _, _, _ = _compute_grid_y(rows, groups_per_row, device, config)
            avg_ms = _benchmark_kernel(
                compiled,
                mX,
                mXq,
                mXs,
                rows,
                groups_per_row,
                eps,
                finfo,
                grid_y,
                device,
            )
            if avg_ms < best_time:
                best_time = avg_ms
                best_config = config
        except Exception:
            continue

    _AUTOTUNE_CACHE[cache_key] = best_config
    return best_config


def _get_compiled_kernel(
    shape: Tuple[int, int],
    column_major_scales: bool,
    dtype: torch.dtype,
    device: torch.device,
    config: _KernelConfig,
) -> cute.JitFunction:
    rows, hidden = shape
    groups_per_row = hidden // QUANT_GROUP_SIZE
    grid_y, sm_count, grid_x, _, wave_target = _compute_grid_y(
        rows, groups_per_row, device, config
    )
    wave_bucket = wave_target

    key = (
        hidden,
        column_major_scales,
        dtype,
        wave_bucket,
        config.threads_per_cta,
        config.rows_per_chunk,
    )
    cached = _COMPILE_CACHE.get(key)
    if cached is not None:
        return cached

    dummy_rows = config.rows_per_chunk

    dummy_x = torch.zeros((dummy_rows, hidden), device=device, dtype=dtype)
    dummy_q_bits = torch.zeros_like(dummy_x, dtype=torch.uint8)
    if column_major_scales:
        dummy_s = torch.empty(
            (groups_per_row, dummy_rows), device=device, dtype=torch.float32
        ).permute(-1, -2)
        scale_leading_dim = 0
    else:
        dummy_s = torch.empty(
            (dummy_rows, groups_per_row), device=device, dtype=torch.float32
        )
        scale_leading_dim = 1

    mX = convert_from_dlpack(dummy_x.detach(), leading_dim=1)
    mXq = convert_from_dlpack(dummy_q_bits.detach(), leading_dim=1)
    mXq.element_type = Float8E4M3FN
    mXs = convert_from_dlpack(dummy_s.detach(), leading_dim=scale_leading_dim)

    launch_kernel = _get_launch_kernel(config)
    compiled = cute.compile(
        launch_kernel,
        mX,
        mXq,
        mXs,
        dummy_rows,
        max(1, groups_per_row),
        1e-10,
        torch.finfo(torch.float8_e4m3fn).min,
        torch.finfo(torch.float8_e4m3fn).max,
        1,
    )
    _COMPILE_CACHE[key] = compiled
    return compiled


def per_token_group_quant_fp8(
    x: torch.Tensor,
    group_size: int = QUANT_GROUP_SIZE,
    eps: float = 1e-10,
    dtype: Optional[torch.dtype] = None,
    column_major_scales: bool = False,
    out_q: Optional[torch.Tensor] = None,
    use_ue8m0: Optional[bool] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.ndim == 2, "`x` must be 2D."
    assert x.is_cuda, "Input must be a CUDA tensor."
    assert x.dtype in (torch.float16, torch.bfloat16), "Only FP16/BF16 inputs supported."
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
    groups_per_row = hidden // group_size

    x_contig = x.contiguous()

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

    config = _select_kernel_config(
        x_contig,
        column_major_scales,
        dtype,
        eps,
    )

    compiled = _get_compiled_kernel(
        x_contig.shape,
        column_major_scales,
        x_contig.dtype,
        x_contig.device,
        config,
    )

    grid_y, _, _, _, _ = _compute_grid_y(rows, groups_per_row, x.device, config)

    mX = convert_from_dlpack(x_contig.detach(), leading_dim=1)
    mXq = convert_from_dlpack(x_q_bits.detach(), leading_dim=1)
    mXq.element_type = Float8E4M3FN
    mXs = convert_from_dlpack(x_s.detach(), leading_dim=scale_leading_dim)

    finfo = torch.finfo(dtype)
    compiled(
        mX,
        mXq,
        mXs,
        rows,
        groups_per_row,
        eps,
        finfo.min,
        finfo.max,
        grid_y,
    )

    return x_q, x_s


__all__ = ["per_token_group_quant_fp8"]
