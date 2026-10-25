"""K1 gate/up GEMM with a SwiGLU-to-NVFP4 epilogue for down.

Calibrate the output descale before construction. Runtime quantization uses
only 16-value block maxima; it never computes a batch-wide activation maximum.
"""
from __future__ import annotations

import torch

from .activation import ActivationWorkspace
from .format import NVFP4Tensor, swizzle_scales
from .gemm import _validate_weight
from .k1_swapab import K1Plan


def interleave_gate_up(weight):
    """Place gate[64], up[64] together in every 128-row weight tile."""
    n, k = weight.shape
    if n % 128:
        raise ValueError("K2 gate/up rows must form complete 128-row tiles")
    def pair(value):
        return (value.view(torch.uint8).reshape(2, n // 128, 64, -1)
                .transpose(0, 1).contiguous().reshape(n, -1).view(value.dtype))
    scales = pair(weight.block_scales)
    return NVFP4Tensor(pair(weight.data), scales, swizzle_scales(scales), weight.global_scale)


class K2Plan(K1Plan):
    """Own one fused gate/up plan and its packed down-input buffers.

    ``gemm_from`` accepts the real RMSNorm producer buffers, like K1. Set the
    input descale in ``workspace.value.global_scale`` before graph capture.
    The BF16 ``out`` allocation inherited from K1 supplies coordinate metadata
    to the TMEM copy layout; this variant never writes that allocation.
    """

    def __init__(self, weight, m, output_descale, *, epilogue=1):
        _validate_weight(weight, m)
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("construct and warm the K2 plan before CUDA graph capture")
        if epilogue != 1 or m != 16:
            raise ValueError("K2 currently supports M=16 and epilogue=1 (serial blocks)")
        if output_descale.shape != (1,) or output_descale.dtype != torch.float32 or output_descale.device != weight.data.device:
            raise ValueError("K2 needs a device FP32 output descale of shape (1,)")
        self._k2_epilogue = epilogue
        self.result = ActivationWorkspace.allocate(m, weight.shape[0] // 2, weight.data.device).value
        self.result.swizzled_scales.zero_()
        self.result.global_scale.copy_(output_descale)
        self.output_multiplier = self.result.global_scale.reciprocal()
        super().__init__(interleave_gate_up(weight), m, split=1, tile_n=32, cta_k=128)

    def gemm_from(self, packed, scales):
        super().gemm_from(packed, scales)
        return self.result

    def gemm(self):
        super().gemm()
        return self.result

    def __call__(self, x):
        super().__call__(x)
        return self.result
