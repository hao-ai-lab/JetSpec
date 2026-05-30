"""Cute-DSL v2 implementation of fused SiLU + FP8 group quantization with cumsum scheduling."""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass import Boolean, Float32, Float8E4M3FN, Int16, Int32, Uint8
from cutlass.cute.runtime import make_fake_tensor

from optimus_cutedsl.reduction_base import torch2cute_dtype_map
from optimus_cutedsl.utils import (
    convert_from_dlpack,
    cvt_fp32x2_to_e4m3x2,
    elem_pointer,
    silu,
)
from .group_quant_fp8_masked import QUANT_GROUP_SIZE
from .silu_mul_group_quant_fp8_masked import (
    _half_warp_reduce_max,
    pack_int16x4_to_int64,
    store_fp8x8,
)


DEFAULT_BLOCK_SIZE = 512
DEFAULT_GRID_SIZE = 256
LANES_PER_WARP = cute.arch.WARP_SIZE
VALS_PER_THREAD = 8
HALF_WARP = LANES_PER_WARP // 2
ELEMENTS_PER_HALF_WARP = HALF_WARP * VALS_PER_THREAD
assert (
    ELEMENTS_PER_HALF_WARP == QUANT_GROUP_SIZE
), "Each half warp must cover exactly one quantization group."
GROUPS_PER_WARP = 2
THREADS_PER_BLOCK = DEFAULT_BLOCK_SIZE
WARPS_PER_BLOCK = THREADS_PER_BLOCK // LANES_PER_WARP


def _set_block_size(block_size: int) -> None:
    global THREADS_PER_BLOCK, WARPS_PER_BLOCK
    THREADS_PER_BLOCK = block_size
    WARPS_PER_BLOCK = block_size // LANES_PER_WARP


def _make_fake_tensor_with_layout(
    dtype,
    shape: Tuple[int, ...],
    stride: Tuple[int, ...],
    leading_dim: int,
    divisibility: int = 1,
    assumed_align: int = 16,
):
    tensor = make_fake_tensor(dtype, shape, stride=stride, assumed_align=assumed_align)
    tensor = tensor.mark_layout_dynamic(leading_dim=leading_dim)
    return tensor.mark_compact_shape_dynamic(mode=leading_dim, divisibility=divisibility)


def _device_cache_key(device: torch.device) -> Tuple[str, Optional[int]]:
    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        return (device.type, index)
    return (device.type, None)


def _ensure_innermost_contiguous(x: torch.Tensor) -> torch.Tensor:
    return x if x.stride(-1) == 1 else x.contiguous()


@cute.kernel
def _silu_mul_group_quant_fp8_masked_v2_kernel(
    mGate: cute.Tensor,
    mUp: cute.Tensor,
    mXq: cute.Tensor,
    mXs: cute.Tensor,
    mCumsum: cute.Tensor,
    max_rows: Int32,
    rows_per_expert: Int32,
    num_experts: Int32,
    groups_per_row: Int32,
    pairs_per_row: Int32,
    warps_per_grid: Int32,
    quant_groups_per_warp: Int32,
    eps: Float32,
    fp8_min: Float32,
    fp8_max: Float32,
    gate_limit: Float32,
    has_gate_limit: cutlass.Constexpr = False,
):

    block_idx, _, _ = cute.arch.block_idx()
    warp_idx = cute.arch.warp_idx()
    lane_idx = cute.arch.lane_idx()
    warp_linear = block_idx * Int32(WARPS_PER_BLOCK) + warp_idx

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

    lane_vals = cute.make_rmem_tensor((VALS_PER_THREAD,), Float32)
    gate_fragment = cute.make_rmem_tensor((VALS_PER_THREAD,), mGate.element_type)
    up_fragment = cute.make_rmem_tensor((VALS_PER_THREAD,), mUp.element_type)
    packed_pairs = cute.make_rmem_tensor((VALS_PER_THREAD // 2,), Int16)

    lane_half = lane_idx // Int32(HALF_WARP)
    lane_in_half = lane_idx - lane_half * Int32(HALF_WARP)
    base_col = lane_in_half * Int32(VALS_PER_THREAD)

    active_rows = mCumsum[num_experts]
    if active_rows < Int32(0):
        active_rows = Int32(0)
    if active_rows > max_rows:
        active_rows = max_rows
    total_tasks = active_rows * pairs_per_row
    remaining = total_tasks - warp_linear
    if remaining < Int32(0):
        remaining = Int32(0)
    max_iters = (remaining + warps_per_grid - Int32(1)) // warps_per_grid

    for itr in cutlass.range(max_iters):
        task_idx = warp_linear + itr * warps_per_grid
        if task_idx < total_tasks:
            row_linear = task_idx // pairs_per_row
            pair_in_row = task_idx - row_linear * pairs_per_row
            group_pair_base = pair_in_row * quant_groups_per_warp
            group_idx = group_pair_base + lane_half

            if row_linear < active_rows and group_idx < groups_per_row:
                row_found = Boolean(False)
                expert_id = Int32(0)
                row_offset = Int32(0)
                for expert_itr in cutlass.range(num_experts):
                    if not row_found:
                        start = mCumsum[expert_itr]
                        end = mCumsum[expert_itr + Int32(1)]
                        if row_linear >= start and row_linear < end:
                            expert_id = Int32(expert_itr)
                            row_offset = row_linear - start
                            row_found = Boolean(True)

                if row_found and row_offset < rows_per_expert:
                    row = expert_id * rows_per_expert + row_offset
                    group_start = group_idx * QUANT_GROUP_SIZE
                    gate_src = cute.make_tensor(
                        elem_pointer(mGate, (row, group_start + base_col)).align(16),
                        cute.make_layout((VALS_PER_THREAD,), stride=(1,)),
                    )
                    up_src = cute.make_tensor(
                        elem_pointer(mUp, (row, group_start + base_col)).align(16),
                        cute.make_layout((VALS_PER_THREAD,), stride=(1,)),
                    )
                    cute.copy(copy_atom_b16, gate_src, gate_fragment)
                    cute.copy(copy_atom_b16, up_src, up_fragment)

                    local_absmax = Float32(0.0)
                    for elem in cutlass.range_constexpr(VALS_PER_THREAD):
                        gate_val = Float32(gate_fragment[elem])
                        up_val = Float32(up_fragment[elem])
                        if cutlass.const_expr(has_gate_limit):
                            gate_val = -gate_val
                            gate_val = cute.arch.fmax(gate_val, -gate_limit)
                            up_val = cute.arch.fmax(up_val, -gate_limit)
                            up_val = -up_val
                            up_val = cute.arch.fmax(up_val, -gate_limit)
                            gate_val = -gate_val
                            up_val = -up_val
                        out_val = silu(gate_val) * up_val
                        lane_vals[elem] = out_val
                        abs_val = cute.arch.fmax(out_val, -out_val)
                        local_absmax = cute.arch.fmax(local_absmax, abs_val)

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
                        packed_pairs[pair] = cvt_fp32x2_to_e4m3x2(val1, val0)

                    packed_vec = pack_int16x4_to_int64(
                        packed_pairs[0],
                        packed_pairs[1],
                        packed_pairs[2],
                        packed_pairs[3],
                    )
                    store_fp8x8(
                        mXq,
                        (row, group_start + base_col),
                        packed_vec,
                    )


@cute.jit
def _launch_silu_mul_group_quant_kernel_v2(
    mGate: cute.Tensor,
    mUp: cute.Tensor,
    mXq: cute.Tensor,
    mXs: cute.Tensor,
    mCumsum: cute.Tensor,
    total_rows: int,
    rows_per_expert: int,
    num_experts: int,
    groups_per_row: int,
    pairs_per_row: int,
    warps_per_grid: int,
    quant_groups_per_warp: int,
    grid_size: int,
    eps: float,
    fp8_min: float,
    fp8_max: float,
    gate_limit: float,
    stream: cuda.CUstream,
    has_gate_limit: cutlass.Constexpr = False,
):
    _silu_mul_group_quant_fp8_masked_v2_kernel(
        mGate,
        mUp,
        mXq,
        mXs,
        mCumsum,
        Int32(total_rows),
        Int32(rows_per_expert),
        Int32(num_experts),
        Int32(groups_per_row),
        Int32(pairs_per_row),
        Int32(warps_per_grid),
        Int32(quant_groups_per_warp),
        Float32(eps),
        Float32(fp8_min),
        Float32(fp8_max),
        Float32(gate_limit),
        has_gate_limit=has_gate_limit,
    ).launch(
        grid=[grid_size, 1, 1],
        block=[THREADS_PER_BLOCK, 1, 1],
        stream=stream,
    )


_COMPILE_CACHE_V2: Dict[
    Tuple[
        int,
        int,
        int,
        bool,
        torch.dtype,
        Tuple[str, Optional[int]],
        int,
        int,
        bool,
        bool,
    ],
    cute.JitFunction,
] = {}


def _get_compiled_kernel_v2(
    shape: Tuple[int, int],
    num_experts: int,
    column_major_scales: bool,
    dtype: torch.dtype,
    device: torch.device,
    block_size: int,
    grid_size: int,
    has_gate_limit: bool,
    stream: cuda.CUstream,
) -> cute.JitFunction:
    device_key = _device_cache_key(device)
    key = (
        shape[0],
        shape[1],
        num_experts,
        column_major_scales,
        dtype,
        device_key,
        block_size,
        grid_size,
        has_gate_limit,
    )
    cached = _COMPILE_CACHE_V2.get(key)
    if cached is not None:
        return cached

    rows, hidden = shape
    groups_per_row = hidden // QUANT_GROUP_SIZE
    rows_per_expert = rows // max(num_experts, 1)
    pairs_per_row = max(1, math.ceil(groups_per_row / GROUPS_PER_WARP))

    assert dtype in torch2cute_dtype_map, f"Unsupported dtype {dtype} for TVM FFI path."
    dtype_cute = torch2cute_dtype_map[dtype]
    gate_tensor = _make_fake_tensor_with_layout(dtype_cute, (rows, hidden), (hidden, 1), leading_dim=1)
    up_tensor = _make_fake_tensor_with_layout(dtype_cute, (rows, hidden), (hidden, 1), leading_dim=1)
    xq_tensor = _make_fake_tensor_with_layout(Uint8, (rows, hidden), (hidden, 1), leading_dim=1)
    if column_major_scales:
        scale_stride = (1, rows)
        scale_leading_dim = 0
    else:
        scale_stride = (groups_per_row, 1)
        scale_leading_dim = 1
    xs_tensor = _make_fake_tensor_with_layout(
        Float32,
        (rows, groups_per_row),
        scale_stride,
        leading_dim=scale_leading_dim,
    )
    cumsum_tensor = _make_fake_tensor_with_layout(
        Int32,
        (num_experts + 1,),
        (1,),
        leading_dim=0,
        assumed_align=4,
    )
    tensor_args = (gate_tensor, up_tensor, xq_tensor, xs_tensor, cumsum_tensor)
    compile_options = "--enable-tvm-ffi"

    prev_block = THREADS_PER_BLOCK
    try:
        if block_size != THREADS_PER_BLOCK:
            _set_block_size(block_size)
        compiled = cute.compile(
            _launch_silu_mul_group_quant_kernel_v2,
            *tensor_args,
            rows,
            rows_per_expert,
            num_experts,
            max(1, groups_per_row),
            pairs_per_row,
            max(1, WARPS_PER_BLOCK * grid_size),
            GROUPS_PER_WARP,
            grid_size,
            1e-10,
            torch.finfo(torch.float8_e4m3fn).min,
            torch.finfo(torch.float8_e4m3fn).max,
            0.0,
            stream,
            cutlass.const_expr(has_gate_limit),
            options=compile_options,
        )
    finally:
        if THREADS_PER_BLOCK != prev_block:
            _set_block_size(prev_block)

    _COMPILE_CACHE_V2[key] = compiled
    return compiled


def silu_mul_group_quant_fp8_masked_v2(
    x_gate: torch.Tensor,
    x_up: torch.Tensor,
    cumsum_m: torch.Tensor,
    group_size: int = QUANT_GROUP_SIZE,
    eps: float = 1e-10,
    gate_limit: Optional[float] = None,
    dtype: Optional[torch.dtype] = None,
    column_major_scales: bool = False,
    out_q: Optional[torch.Tensor] = None,
    out_scales: Optional[torch.Tensor] = None,
    grid_size: int = DEFAULT_GRID_SIZE,
    block_size: int = DEFAULT_BLOCK_SIZE,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x_gate.ndim == 2, "`x_gate` must be 2D after flattening expert tiles."
    assert x_up.ndim == 2, "`x_up` must be 2D after flattening expert tiles."
    assert cumsum_m.ndim == 1, "`cumsum_m` must be 1D."
    assert x_gate.shape == x_up.shape, "`x_gate` and `x_up` must have the same shape."
    assert x_gate.is_cuda and x_up.is_cuda and cumsum_m.is_cuda, "Inputs must be CUDA tensors."
    assert x_gate.dtype in (torch.float16, torch.bfloat16), "Only FP16/BF16 inputs supported."
    assert x_gate.dtype == x_up.dtype, "`x_gate` and `x_up` must share the same dtype."
    assert cumsum_m.shape[0] >= 2, "`cumsum_m` must have length >= num_experts + 1."
    assert group_size == QUANT_GROUP_SIZE, (
        f"group_size={group_size} is not supported; only {QUANT_GROUP_SIZE} works in this kernel."
    )
    assert grid_size > 0, "`grid_size` must be positive."
    assert block_size % LANES_PER_WARP == 0, "`block_size` must be a multiple of warp size."
    assert block_size > 0, "`block_size` must be positive."

    dtype = torch.float8_e4m3fn if dtype is None else dtype
    if dtype != torch.float8_e4m3fn:
        raise NotImplementedError("Only torch.float8_e4m3fn is supported for the Cute-DSL path.")

    rows, hidden = x_gate.shape
    num_experts = cumsum_m.shape[0] - 1
    assert rows % num_experts == 0, "`x_gate` rows must be divisible by num experts."
    rows_per_expert = rows // num_experts
    groups_per_row = hidden // group_size
    assert hidden % group_size == 0, "`hidden` dimension must be divisible by `group_size`."

    cumsum_contig = cumsum_m.to(dtype=torch.int32).contiguous()
    assert cumsum_contig.shape[0] == num_experts + 1, "`cumsum_m` must be length num_experts + 1."

    x_gate_contig = _ensure_innermost_contiguous(x_gate)
    x_up_contig = _ensure_innermost_contiguous(x_up)

    if out_q is None:
        x_q = torch.empty_like(x_up_contig, dtype=dtype)
    else:
        assert out_q.shape == x_up_contig.shape, "`out_q` must match `x_up` shape."
        assert out_q.device == x_up_contig.device, "`out_q` must be on the same device as `x_up`."
        assert out_q.dtype == dtype, "`out_q` must match the requested dtype."
        assert out_q.is_contiguous(), "`out_q` must be contiguous."
        x_q = out_q

    if out_scales is None:
        if column_major_scales:
            x_s = torch.empty((groups_per_row, rows), device=x_gate.device, dtype=torch.float32).permute(
                -1, -2
            )
            scale_leading_dim = 0
        else:
            x_s = torch.empty((rows, groups_per_row), device=x_gate.device, dtype=torch.float32)
            scale_leading_dim = 1
    else:
        x_s = out_scales
        scale_leading_dim = 0 if column_major_scales else 1
    assert x_s.shape == (rows, groups_per_row), "`out_scales` must match (rows, groups_per_row)."

    mGate = convert_from_dlpack(x_gate_contig.detach(), leading_dim=1)
    mUp = convert_from_dlpack(x_up_contig.detach(), leading_dim=1)
    x_q_bytes = x_q.view(torch.uint8)
    mXq = convert_from_dlpack(x_q_bytes.detach(), leading_dim=1)
    mXs = convert_from_dlpack(x_s.detach(), leading_dim=scale_leading_dim)
    mCumsum = convert_from_dlpack(cumsum_contig.detach(), leading_dim=0)

    gate_limit_value = 0.0
    has_gate_limit = False
    if gate_limit is not None:
        gate_limit_value = float(gate_limit)
        if gate_limit_value <= eps:
            gate_limit_value = 0.0
        else:
            has_gate_limit = True

    pairs_per_row = max(1, math.ceil(groups_per_row / GROUPS_PER_WARP))
    warps_per_block = block_size // LANES_PER_WARP
    warps_per_grid = warps_per_block * grid_size
    assert warps_per_grid > 0, "`grid_size * (block_size / warp)` must be positive."
    torch_stream = torch.cuda.current_stream(device=x_gate_contig.device)
    current_stream = cuda.CUstream(torch_stream.cuda_stream)

    compiled = _get_compiled_kernel_v2(
        x_gate_contig.shape,
        num_experts,
        column_major_scales,
        x_gate_contig.dtype,
        x_gate_contig.device,
        block_size,
        grid_size,
        has_gate_limit,
        current_stream,
    )

    compiled(
        mGate,
        mUp,
        mXq,
        mXs,
        mCumsum,
        rows,
        rows_per_expert,
        num_experts,
        groups_per_row,
        pairs_per_row,
        warps_per_grid,
        GROUPS_PER_WARP,
        grid_size,
        eps,
        torch.finfo(torch.float8_e4m3fn).min,
        torch.finfo(torch.float8_e4m3fn).max,
        gate_limit_value,
        current_stream,
    )

    return x_q, x_s


__all__ = ["silu_mul_group_quant_fp8_masked_v2"]
