"""CTA-local BF16/RMSNorm input prologue for K1 and fused K2 gate/up."""
from __future__ import annotations

import torch

from .activation import ActivationWorkspace
from .gemm import _validate_weight
from .k1_swapab import K1Plan
from .k2_swiglu import interleave_gate_up


class K1PPlan(K1Plan):
    """Fuse the producer inside each GEMM CTA with calibrated input descale.

    Residual input is immutable. ``residual_out`` is a distinct BF16 output,
    written only by weight tile zero, split zero. This prevents inter-CTA
    races that an in-place add would cause. Inputs must outlive graph replay.
    """

    def __init__(self, weight, m, input_descale, *, producer="plain", output_descale=None, attempt=1):
        _validate_weight(weight, m)
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("construct and warm K1P before graph capture")
        if m != 16 or weight.shape[1] != 4096 or producer not in ("plain", "rmsnorm"):
            raise ValueError("K1P supports M=16, K=4096 and plain/rmsnorm producers")
        if attempt not in (1, 2):
            raise ValueError("K1P attempt must be 1 (upfront) or 2 (overlapped weight DMA)")
        if input_descale.shape != (1,) or input_descale.dtype != torch.float32 or input_descale.device != weight.data.device:
            raise ValueError("K1P needs a device FP32 input descale of shape (1,)")
        self.producer = producer
        self._prologue_mode = 2 if producer == "rmsnorm" else 1
        self._prologue_overlap = attempt == 2
        device = weight.data.device
        self.input = torch.empty((m, 4096), device=device, dtype=torch.bfloat16)
        self.residual = torch.empty_like(self.input)
        self.residual_out = torch.empty_like(self.input)
        self.gamma = torch.ones(4096, device=device, dtype=torch.bfloat16)
        self.input_multiplier = input_descale.reciprocal()
        if output_descale is not None:
            self._k2_epilogue = 1
            self.result = ActivationWorkspace.allocate(m, weight.shape[0] // 2, device).value
            self.result.swizzled_scales.zero_()
            self.result.global_scale.copy_(output_descale)
            self.output_multiplier = self.result.global_scale.reciprocal()
            weight = interleave_gate_up(weight)
        super().__init__(weight, m)
        self.workspace.value.global_scale.copy_(input_descale)

    def warmup(self, x, residual=None, gamma=None, *, iterations=3):
        if iterations <= 0 or torch.cuda.is_current_stream_capturing():
            raise RuntimeError("warm K1P before capture with positive iterations")
        self._warmed = True
        for _ in range(iterations):
            self(x, residual, gamma)
        return self

    def __call__(self, x, residual=None, gamma=None):
        self._check_input(x)
        if not self._warmed:
            raise RuntimeError("warm K1P before use")
        if self.producer == "rmsnorm":
            if residual is None or gamma is None:
                raise ValueError("RMSNorm prologue requires residual and gamma")
            self._check_input(residual)
            if gamma.shape != (4096,) or gamma.dtype != x.dtype or gamma.device != x.device or not gamma.is_contiguous():
                raise ValueError("K1P gamma shape/dtype/device mismatch")
            if residual.data_ptr() == self.residual_out.data_ptr():
                raise ValueError("K1P residual input/output must not alias")
        else:
            residual, gamma = self.residual, self.gamma
        import cutlass
        from cutlass.cute.runtime import make_ptr
        def ptr(value, dtype):
            return make_ptr(dtype, value.data_ptr(), cutlass.AddressSpace.gmem, assumed_align=16)
        prologue = (ptr(x, cutlass.BFloat16), ptr(residual, cutlass.BFloat16),
                    ptr(gamma, cutlass.BFloat16), ptr(self.residual_out, cutlass.BFloat16),
                    self._args[8][4])
        args = self._args[:8] + (prologue,) + self._args[9:]
        stream = self._cuda.CUstream(torch.cuda.current_stream(x.device).cuda_stream)
        self._compiled(*args, stream)
        return self.result if getattr(self, "_k2_epilogue", 0) else self.out
