"""Cute-DSL v2 implementation of fused SiLU for masked grouped rows (FP16/BF16)."""

from __future__ import annotations

import math
import os
import warnings
from typing import Dict, Optional, Tuple

import torch

import cutlass
import cutlass.cute as cute
from cutlass import Boolean, Float32, Int32
from cutlass.cute.runtime import make_fake_tensor
import cuda.bindings.driver as cuda

from optimus_cutedsl.reduction_base import torch2cute_dtype_map
from optimus_cutedsl.utils import convert_from_dlpack, elem_pointer, silu


QUANT_GROUP_SIZE = 128
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

_DISABLE_TVM_FFI_VALUES = {"0", "false", "off", "no"}
_TVM_FFI_ENV = os.getenv("OPTIMUS_ENABLE_TVM_FFI")
_DEFAULT_USE_TVM_FFI = (
    True if _TVM_FFI_ENV is None else _TVM_FFI_ENV.strip().lower() not in _DISABLE_TVM_FFI_VALUES
)
_TVM_FFI_STATUS: Optional[bool] = None


def _resolve_use_tvm_ffi(user_choice: Optional[bool]) -> Tuple[bool, bool]:
    """Return (use_tvm_ffi, allow_fallback_to_legacy)."""
    if user_choice is not None:
        return user_choice, False
    if _TVM_FFI_STATUS is not None:
        return _TVM_FFI_STATUS, False
    return _DEFAULT_USE_TVM_FFI, _DEFAULT_USE_TVM_FFI


def _is_tvm_ffi_unavailable_error(err: BaseException) -> bool:
    if isinstance(err, (ImportError, ModuleNotFoundError)):
        return True
    msg = str(err).lower()
    if "tvm" not in msg:
        return False
    keywords = ("ffi", "module", "libtvm", "enable_tvm_ffi", "runtime")
    return any(token in msg for token in keywords)


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
def _silu_mul_group_masked_v2_kernel(
    mGate: cute.Tensor,
    mUp: cute.Tensor,
    mY: cute.Tensor,
    mCumsum: cute.Tensor,
    max_rows: Int32,
    rows_per_expert: Int32,
    num_experts: Int32,
    groups_per_row: Int32,
    pairs_per_row: Int32,
    warps_per_grid: Int32,
    quant_groups_per_warp: Int32,
    grid_size: Int32,
    gate_limit: Float32,
    has_gate_limit: cutlass.Constexpr = False,
):

    block_idx, _, _ = cute.arch.block_idx()
    warp_idx = cute.arch.warp_idx()
    lane_idx = cute.arch.lane_idx()
    warp_linear = block_idx * Int32(WARPS_PER_BLOCK) + warp_idx

    copy_atom_b16 = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(),
        mGate.element_type,
        num_bits_per_copy=128,
    )
    copy_atom_out = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(),
        mY.element_type,
        num_bits_per_copy=128,
    )

    gate_fragment = cute.make_rmem_tensor((VALS_PER_THREAD,), mGate.element_type)
    up_fragment = cute.make_rmem_tensor((VALS_PER_THREAD,), mUp.element_type)
    out_fragment = cute.make_rmem_tensor((VALS_PER_THREAD,), mY.element_type)

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
                        out_fragment[elem] = mY.element_type(out_val)

                    dst = cute.make_tensor(
                        elem_pointer(mY, (row, group_start + base_col)).align(16),
                        cute.make_layout((VALS_PER_THREAD,), stride=(1,)),
                    )
                    cute.copy(copy_atom_out, out_fragment, dst)


@cute.jit
def _launch_silu_mul_group_masked_kernel_v2(
    mGate: cute.Tensor,
    mUp: cute.Tensor,
    mY: cute.Tensor,
    mCumsum: cute.Tensor,
    total_rows: int,
    rows_per_expert: int,
    num_experts: int,
    groups_per_row: int,
    pairs_per_row: int,
    warps_per_grid: int,
    quant_groups_per_warp: int,
    grid_size: int,
    gate_limit: float,
    stream: cuda.CUstream,
    has_gate_limit: cutlass.Constexpr = False,
):
    _silu_mul_group_masked_v2_kernel(
        mGate,
        mUp,
        mY,
        mCumsum,
        Int32(total_rows),
        Int32(rows_per_expert),
        Int32(num_experts),
        Int32(groups_per_row),
        Int32(pairs_per_row),
        Int32(warps_per_grid),
        Int32(quant_groups_per_warp),
        Int32(grid_size),
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
    dtype: torch.dtype,
    device: torch.device,
    block_size: int,
    grid_size: int,
    has_gate_limit: bool,
    stream: cuda.CUstream,
    use_tvm_ffi: bool,
    allow_tvm_fallback: bool = False,
) -> cute.JitFunction:
    global _TVM_FFI_STATUS
    device_key = _device_cache_key(device)
    key = (
        shape[0],
        shape[1],
        num_experts,
        dtype,
        device_key,
        block_size,
        grid_size,
        has_gate_limit,
        use_tvm_ffi,
    )
    cached = _COMPILE_CACHE_V2.get(key)
    if cached is not None:
        return cached

    rows, hidden = shape
    groups_per_row = hidden // QUANT_GROUP_SIZE
    rows_per_expert = rows // max(num_experts, 1)
    pairs_per_row = max(1, math.ceil(groups_per_row / GROUPS_PER_WARP))

    assert dtype in torch2cute_dtype_map, f"Unsupported dtype {dtype} for the Cute-DSL path."
    dtype_cute = torch2cute_dtype_map[dtype]
    gate_tensor = _make_fake_tensor_with_layout(dtype_cute, (rows, hidden), (hidden, 1), leading_dim=1)
    up_tensor = _make_fake_tensor_with_layout(dtype_cute, (rows, hidden), (hidden, 1), leading_dim=1)
    y_tensor = _make_fake_tensor_with_layout(dtype_cute, (rows, hidden), (hidden, 1), leading_dim=1)
    cumsum_tensor = _make_fake_tensor_with_layout(
        Int32,
        (num_experts + 1,),
        (1,),
        leading_dim=0,
        assumed_align=4,
    )
    tensor_args = (gate_tensor, up_tensor, y_tensor, cumsum_tensor)

    prev_block = THREADS_PER_BLOCK
    try:
        if block_size != THREADS_PER_BLOCK:
            _set_block_size(block_size)
        compile_kwargs = {}
        if use_tvm_ffi:
            compile_kwargs["options"] = "--enable-tvm-ffi"
        compiled = cute.compile(
            _launch_silu_mul_group_masked_kernel_v2,
            *tensor_args,
            rows,
            rows_per_expert,
            num_experts,
            max(1, groups_per_row),
            pairs_per_row,
            max(1, WARPS_PER_BLOCK * grid_size),
            GROUPS_PER_WARP,
            grid_size,
            0.0,
            stream,
            cutlass.const_expr(has_gate_limit),
            **compile_kwargs,
        )
    except Exception as err:
        if use_tvm_ffi and allow_tvm_fallback and _is_tvm_ffi_unavailable_error(err):
            warnings.warn(
                "TVM FFI is unavailable; falling back to the legacy launch path for "
                "silu_mul_group_masked_v2.",
                RuntimeWarning,
            )
            _TVM_FFI_STATUS = False
            return _get_compiled_kernel_v2(
                shape,
                num_experts,
                dtype,
                device,
                block_size,
                grid_size,
                has_gate_limit,
                stream,
                use_tvm_ffi=False,
                allow_tvm_fallback=False,
            )
        raise
    finally:
        if THREADS_PER_BLOCK != prev_block:
            _set_block_size(prev_block)

    if use_tvm_ffi:
        _TVM_FFI_STATUS = True

    _COMPILE_CACHE_V2[key] = compiled
    return compiled


def silu_mul_group_masked_v2(
    x_gate: torch.Tensor,
    x_up: torch.Tensor,
    cumsum_m: torch.Tensor,
    group_size: int = QUANT_GROUP_SIZE,
    eps: float = 1e-10,
    gate_limit: Optional[float] = None,
    dtype: Optional[torch.dtype] = None,
    out: Optional[torch.Tensor] = None,
    grid_size: int = DEFAULT_GRID_SIZE,
    block_size: int = DEFAULT_BLOCK_SIZE,
    use_tvm_ffi: Optional[bool] = None,
) -> torch.Tensor:
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
    use_tvm_ffi_choice, allow_tvm_fallback = _resolve_use_tvm_ffi(use_tvm_ffi)

    if dtype is None:
        out_dtype = x_gate.dtype
    else:
        assert dtype in (torch.float16, torch.bfloat16), "`dtype` must be FP16 or BF16."
        if dtype != x_gate.dtype:
            raise NotImplementedError("Requested dtype must match input dtype for this kernel.")
        out_dtype = dtype

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

    if out is None:
        y = torch.empty_like(x_up_contig, dtype=out_dtype)
    else:
        assert out.shape == x_up_contig.shape, "`out` must match `x_up` shape."
        assert out.device == x_up_contig.device, "`out` must be on the same device as inputs."
        assert out.dtype == out_dtype, "`out` must match the requested dtype."
        assert out.is_contiguous(), "`out` must be contiguous."
        y = out

    mGate = convert_from_dlpack(x_gate_contig.detach(), leading_dim=1)
    mUp = convert_from_dlpack(x_up_contig.detach(), leading_dim=1)
    mY = convert_from_dlpack(y.detach(), leading_dim=1)
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
        x_gate_contig.dtype,
        x_gate_contig.device,
        block_size,
        grid_size,
        has_gate_limit,
        current_stream,
        use_tvm_ffi_choice,
        allow_tvm_fallback=allow_tvm_fallback,
    )

    compiled(
        mGate,
        mUp,
        mY,
        mCumsum,
        rows,
        rows_per_expert,
        num_experts,
        groups_per_row,
        pairs_per_row,
        warps_per_grid,
        GROUPS_PER_WARP,
        grid_size,
        gate_limit_value,
        current_stream,
    )

    return y


__all__ = ["silu_mul_group_masked_v2"]
