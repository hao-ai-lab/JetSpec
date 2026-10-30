"""Dynamic W4A4 activation quantization with reusable output and reduction buffers."""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .format import (NVFP4Tensor, _check_matrix, _quantize_chunk, quantize_weight,
                     scale_shape, unpack_e2m1)


@dataclass
class ActivationWorkspace:
    value: NVFP4Tensor
    partial_amax: torch.Tensor

    @classmethod
    def allocate(cls, rows: int, k: int, device, *, per_row=False) -> ActivationWorkspace:
        if rows <= 0 or k <= 0 or k % 16:
            raise ValueError("activation workspace needs positive rows and K divisible by 16")
        value = NVFP4Tensor(
            torch.empty((rows, k // 2), dtype=torch.uint8, device=device),
            torch.empty((rows, k // 16), dtype=torch.float8_e4m3fn, device=device),
            torch.empty(scale_shape(rows, k // 16), dtype=torch.float8_e4m3fn, device=device),
            torch.empty((rows if per_row else 1,), dtype=torch.float32, device=device),
        )
        return cls(value, torch.empty(((rows * k + 1023) // 1024,), dtype=torch.float32, device=device))


@torch.no_grad()
def quantize_activation(x: torch.Tensor, *, workspace: ActivationWorkspace | None = None) -> NVFP4Tensor:
    """Quantize finite contiguous bf16 [M,K]; all reductions stay on device.

    Supply a workspace allocated at M-bucket warm-up for CUDA graph capture.
    The CPU branch is a format oracle, not a performance implementation.
    """
    _check_matrix(x)
    if not x.is_contiguous():
        raise ValueError("activation matrix must be contiguous")
    if workspace is None:
        workspace = ActivationWorkspace.allocate(*x.shape, x.device)
    if workspace.value.shape != tuple(x.shape) or workspace.value.data.device != x.device:
        raise ValueError("activation workspace shape/device mismatch")
    if x.is_cuda:
        from .cuda_quant import quantize_into
        quantize_into(x, workspace)
    else:
        reference = quantize_weight(x)
        for target, source in zip(
                (workspace.value.data, workspace.value.block_scales,
                 workspace.value.swizzled_scales, workspace.value.global_scale),
                (reference.data, reference.block_scales, reference.swizzled_scales, reference.global_scale)):
            target.copy_(source)
    return workspace.value


@torch.no_grad()
def quantize_activation_rows(x: torch.Tensor, *, workspace: ActivationWorkspace) -> NVFP4Tensor:
    """Pack the same independent row rounding as fake_quantize_activation."""
    _check_matrix(x)
    value = workspace.value
    if (not x.is_contiguous() or value.shape != tuple(x.shape)
            or value.data.device != x.device or value.global_scale.shape != (x.shape[0],)):
        raise ValueError('per-row activation workspace shape/device/layout mismatch')
    if x.is_cuda:
        from .cuda_quant import quantize_rows_into
        quantize_rows_into(x, workspace)
    else:
        from .format import swizzle_scales
        maximum = x.float().abs().amax(-1)
        value.global_scale.copy_(torch.where(maximum == 0, 1., maximum / (6 * 448)))
        data, scales = _quantize_chunk(x, value.global_scale[:, None])
        value.data.copy_(data)
        value.block_scales.copy_(scales)
        value.swizzled_scales.copy_(swizzle_scales(scales))
    return value


@torch.no_grad()
def fake_quantize_activation(x: torch.Tensor, *, out: torch.Tensor | None = None) -> torch.Tensor:
    """Round rows independently using FP32 global and E4M3 block scales.

    Unlike the calibration API above, no reduction crosses token boundaries.
    Supply reusable bf16 output at bucket warm-up for CUDA graph capture.
    """
    _check_matrix(x)
    if not x.is_contiguous():
        raise ValueError('activation matrix must be contiguous')
    if out is None:
        out = torch.empty_like(x)
    if out.shape != x.shape or out.dtype != x.dtype or out.device != x.device or not out.is_contiguous():
        raise ValueError('activation output shape/dtype/device/layout mismatch')
    if x.is_cuda:
        from .cuda_quant import fake_quantize_rows_into
        fake_quantize_rows_into(x, out)
    else:
        maximum = x.float().abs().amax(-1, keepdim=True)
        global_scale = torch.where(maximum == 0, 1., maximum / (6 * 448))
        packed, scales = _quantize_chunk(x, global_scale)
        values = unpack_e2m1(packed).reshape(x.shape[0], -1, 16)
        out.copy_((values * (scales.float() * global_scale).unsqueeze(-1)).flatten(1))
    return out
