"""NVFP4 reference format and load-time conversion, using only torch.

ModelOpt convention: x ~= E2M1 * E4M3 * global_scale, with
global_scale = amax(x)/(6*448). Low nibble holds the even K element.
The independent implementation uses round-to-nearest, ties-to-even.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


def pack_e2m1(values: torch.Tensor) -> torch.Tensor:
    """Saturating E2M1 conversion; accepts an even trailing dimension."""
    if values.shape[-1] % 2:
        raise ValueError("E2M1 packing needs an even trailing dimension")
    magnitude = values.float().abs()
    code = torch.zeros_like(magnitude, dtype=torch.uint8)
    for index, threshold in enumerate((.25, .75, 1.25, 1.75, 2.5, 3.5, 5.)):
        # At a midpoint select the code with an even least-significant bit.
        code += (magnitude >= threshold if index % 2 else magnitude > threshold).to(torch.uint8)
    code |= torch.signbit(values).to(torch.uint8) << 3
    return code[..., ::2] | (code[..., 1::2] << 4)


def unpack_e2m1(packed: torch.Tensor) -> torch.Tensor:
    """Decode packed bytes to FP32, including signed zero."""
    packed = packed.view(torch.uint8)
    codes = torch.stack((packed & 15, packed >> 4), dim=-1).flatten(-2).long()
    levels = torch.tensor((0., .5, 1., 1.5, 2., 3., 4., 6.), device=packed.device)
    return torch.where(codes < 8, levels[codes & 7], -levels[codes & 7])


def scale_shape(rows: int, blocks: int) -> tuple[int, int]:
    """cuBLASLt's physical scale storage, padded in both dimensions."""
    return ((rows + 127) // 128 * 128, (blocks + 3) // 4 * 4)


def swizzle_scales(scales: torch.Tensor) -> torch.Tensor:
    """Convert logical [rows, K/16] scales to the 128x4 byte permutation.

    Physical shape is a storage descriptor, not logical row-major scales.
    Permute bytes so this works with CPU float8 and preserves every bit.
    """
    rows, blocks = scales.shape
    padded_rows, padded_blocks = scale_shape(rows, blocks)
    padded = torch.zeros((padded_rows, padded_blocks), dtype=torch.uint8, device=scales.device)
    padded[:rows, :blocks].copy_(scales.view(torch.uint8))
    return (padded.reshape(padded_rows // 128, 4, 32, padded_blocks // 4, 4)
            .permute(0, 3, 2, 1, 4).contiguous().reshape(padded_rows, padded_blocks)
            .view(scales.dtype))


def unswizzle_scales(scales: torch.Tensor, rows: int, blocks: int) -> torch.Tensor:
    """Recover logical scales, discarding physical padding."""
    padded_rows, padded_blocks = scale_shape(rows, blocks)
    linear = (scales.view(torch.uint8).reshape(padded_rows // 128, padded_blocks // 4, 32, 4, 4)
              .permute(0, 3, 2, 1, 4).contiguous().reshape(padded_rows, padded_blocks))
    return linear[:rows, :blocks].contiguous().view(scales.dtype)


@dataclass
class NVFP4Tensor:
    """Canonical packed rows, logical and swizzled E4M3 scales, FP32 descale.

    The logical scales are retained for inspection and CPU dequantization.
    A serving integration can discard them after constructing its backend plan.
    """

    data: torch.Tensor
    block_scales: torch.Tensor
    swizzled_scales: torch.Tensor
    global_scale: torch.Tensor

    @property
    def shape(self) -> tuple[int, int]:
        return (self.data.shape[0], self.data.shape[1] * 2)


def _check_matrix(value: torch.Tensor) -> None:
    if value.ndim != 2 or min(value.shape) <= 0 or value.shape[1] % 16:
        raise ValueError("NVFP4 needs a nonempty [rows, K] matrix with K divisible by 16")
    if value.dtype != torch.bfloat16:
        raise TypeError("NVFP4 input must be bfloat16")


def _quantize_chunk(value: torch.Tensor, global_scale: torch.Tensor):
    blocks = value.float().reshape(value.shape[0], -1, 16)
    raw_scales = blocks.abs().amax(-1) / (6. * global_scale)
    raw_scales = torch.where(raw_scales == 0, 1., raw_scales)
    # Positive minimum prevents nonzero blocks from underflowing their scale.
    scales = raw_scales.clamp(2. ** -9, 448.).to(torch.float8_e4m3fn)
    normalized = blocks / (scales.float() * global_scale).unsqueeze(-1)
    return pack_e2m1(normalized.reshape(value.shape)), scales


@torch.no_grad()
def quantize_weight(weight: torch.Tensor, *, chunk_rows: int = 256) -> NVFP4Tensor:
    """Load-time conversion with ModelOpt's amax/(6*448) global descale.

    Chunking bounds FP32 temporaries for an lm_head. Zero tensors use a global
    scale of one. Inputs must be finite; this load-time API validates them.
    """
    _check_matrix(weight)
    if chunk_rows <= 0:
        raise ValueError("chunk_rows must be positive")
    amax = torch.zeros((1,), dtype=torch.float32, device=weight.device)
    for start in range(0, weight.shape[0], chunk_rows):
        values = weight[start:start + chunk_rows].float()
        if not bool(torch.isfinite(values).all()):
            raise ValueError("NVFP4 input must be finite")
        torch.maximum(amax, values.abs().amax(), out=amax)
    global_scale = torch.where(amax == 0, 1., amax / (6 * 448))
    data = torch.empty((weight.shape[0], weight.shape[1] // 2), dtype=torch.uint8, device=weight.device)
    scales = torch.empty((weight.shape[0], weight.shape[1] // 16),
                         dtype=torch.float8_e4m3fn, device=weight.device)
    for start in range(0, weight.shape[0], chunk_rows):
        packed, block_scales = _quantize_chunk(weight[start:start + chunk_rows], global_scale)
        data[start:start + chunk_rows].copy_(packed)
        scales[start:start + chunk_rows].copy_(block_scales)
    return NVFP4Tensor(data, scales, swizzle_scales(scales), global_scale)


@torch.no_grad()
def dequantize(value: NVFP4Tensor, *, chunk_rows: int = 256,
               dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    """Unpack into bf16 (or FP32 for a numerical reference)."""
    if chunk_rows <= 0:
        raise ValueError("chunk_rows must be positive")
    result = torch.empty(value.shape, device=value.data.device, dtype=dtype)
    for start in range(0, value.shape[0], chunk_rows):
        decoded = unpack_e2m1(value.data[start:start + chunk_rows]).reshape(-1, value.shape[1] // 16, 16)
        scales = value.block_scales[start:start + chunk_rows].float() * value.global_scale
        result[start:start + chunk_rows].copy_((decoded * scales.unsqueeze(-1)).flatten(1))
    return result


def fake_quantize(weight: torch.Tensor, *, chunk_rows: int = 256) -> torch.Tensor:
    """NVFP4 weight-only rounding, returned as bf16 for ordinary linear calls."""
    return dequantize(quantize_weight(weight, chunk_rows=chunk_rows), chunk_rows=chunk_rows)
