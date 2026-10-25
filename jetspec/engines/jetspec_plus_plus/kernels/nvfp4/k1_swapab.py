"""M-bucket plans for the K1 swap-AB tcgen05 NVFP4 kernel.

The CuTe DSL implementation is imported only at CUDA plan construction.
Warm up each plan on its capture stream. Outputs and all scratch storage are
owned by the plan; a plan cannot be used concurrently from multiple streams.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .activation import ActivationWorkspace, quantize_activation, quantize_activation_rows
from .gemm import _validate_weight


@dataclass(frozen=True)
class K1Config:
    tile_n: int
    split: int
    stages: int
    cta_k: int
    reduction: str


# GB200 cold-L2 M=16 sweep: (token tile, split-K, CTA-K). Other buckets
# retain the initial shape policy. Callers can override all tuning knobs.
_M16_DEFAULTS = {
    (6144, 4096): (16, 1, 256),
    (4096, 4096): (16, 4, 256),
    (24576, 4096): (32, 1, 128),
    (4096, 12288): (16, 4, 256),
    (151936, 4096): (32, 1, 128),
}


def k1_config(n, k, m, split=None, *, stages=8, tile_n=None, cta_k=None,
              reduction="auto"):
    """Select a token tile and a shape-specific split before capture."""
    measured = _M16_DEFAULTS.get((n, k)) if m == 16 else None
    default_k = measured[2] if measured else (128 if m > 32 else 256)
    cta_k = default_k if cta_k is None else cta_k
    if not 1 <= m <= 80 or n <= 0 or n % 128 or k <= 0 or cta_k not in (64, 128, 256) or k % cta_k:
        raise ValueError("K1 requires M in [1,80], N divisible by 128 and aligned K")
    if tile_n is None:
        tile_n = measured[0] if measured else next(tile for tile in (8, 16, 32, 64, 80) if m <= tile)
    if tile_n not in (8, 16, 32, 64, 80) or tile_n < m:
        raise ValueError("K1 token tile must be 8, 16, 32, 64 or 80")
    if split is None:
        split = measured[1] if measured else {(6144, 4096): 3, (4096, 4096): 4, (4096, 12288): 4}.get((n, k), 1)
    if split not in (1, 2, 3, 4, 6, 8) or not 1 <= stages <= 12:
        raise ValueError("K1 split must be 1/2/3/4/6/8 and stages in [1,12]")
    if reduction == "auto":
        reduction = "dsmem" if tile_n % split == 0 else "workspace"
    if reduction not in ("dsmem", "workspace") or (reduction == "dsmem" and tile_n % split):
        raise ValueError("DSMEM reduction requires split to divide the token tile")
    return K1Config(tile_n, split, stages, cta_k, reduction)


class K1Plan:
    """Compile once for one weight and token bucket, then launch without allocation."""

    def __init__(self, weight, m, split=None, *, stages=8, tile_n=None, cta_k=None,
                 reduction="auto", enable_pdl=True, per_row_scale=False):
        _validate_weight(weight, m)
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("construct and warm the K1 plan before CUDA graph capture")
        from .gemm import backend_status
        status = backend_status("k1")
        if not status.available:
            raise RuntimeError(status.reason)
        if torch.cuda.get_device_capability(weight.data.device)[0] != 10:
            raise RuntimeError("K1 requires Blackwell sm100/sm103 NVFP4 hardware")
        self.config = k1_config(*weight.shape, m, split, stages=stages, tile_n=tile_n,
                                cta_k=cta_k, reduction=reduction)
        self.weight, self.m, self.backend = weight, m, "k1"
        self.per_row_scale = per_row_scale
        self.workspace = ActivationWorkspace.allocate(m, weight.shape[1], weight.data.device,
                                                      per_row=per_row_scale)
        self.out = torch.empty((m, weight.shape[0]), dtype=torch.bfloat16, device=weight.data.device)
        # The quantizer writes the activation descale beside the fixed weight
        # descale. Their FP32 product is consumed by the epilogue, avoiding a
        # separate scalar-multiply kernel at every GEMM boundary.
        self.alpha = torch.empty((1 + m if per_row_scale else 2,), dtype=torch.float32, device=weight.data.device)
        self.alpha[:1].copy_(weight.global_scale)
        self.workspace.value.global_scale = self.alpha[1:]
        self._warmed = False
        self._compile(enable_pdl)

    def _compile(self, enable_pdl):
        import cuda.bindings.driver as cuda
        import cutlass
        import cutlass.cute as cute
        from cutlass.cute.runtime import make_fake_stream, make_ptr
        from ._k1_kernel import K1Kernel, reduce_workspace

        self._cuda = cuda
        cfg = self.config
        use_workspace = cfg.reduction == "workspace" and cfg.split > 1
        self.partials = (torch.empty((cfg.split, self.m, self.weight.shape[0]),
                                    dtype=torch.float32, device=self.out.device)
                         if use_workspace else None)
        kernel = K1Kernel(mma_tiler_mnk=(128, cfg.tile_n, cfg.cta_k),
                          num_ab_stage=cfg.stages, num_sfb_tmem_stage=1,
                          split_k=1 if use_workspace else cfg.split,
                          workspace_split=cfg.split if use_workspace else 1,
                          use_scale=not use_workspace, use_pdl=enable_pdl,
                          per_row_scale=self.per_row_scale,
                          k2_epilogue=getattr(self, "_k2_epilogue", 0),
                          prologue_mode=getattr(self, "_prologue_mode", 0),
                          prologue_overlap=getattr(self, "_prologue_overlap", False),
                          prologue_k=self.weight.shape[1] if getattr(self, "_prologue_mode", 0) else 0)
        def ptr(tensor, dtype, align):
            return make_ptr(dtype, tensor.data_ptr(), cutlass.AddressSpace.gmem, assumed_align=align)
        value = self.workspace.value
        quant = None
        prologue = None
        if getattr(self, "_prologue_mode", 0):
            prologue = (ptr(self.input, cutlass.BFloat16, 16),
                        ptr(self.residual, cutlass.BFloat16, 16),
                        ptr(self.gamma, cutlass.BFloat16, 16),
                        ptr(self.residual_out, cutlass.BFloat16, 16),
                        ptr(self.input_multiplier, cutlass.Float32, 4))
        if getattr(self, "_k2_epilogue", 0):
            quant = (ptr(self.result.data, cutlass.Uint64, 8),
                     ptr(self.result.swizzled_scales, cutlass.Uint8, 16),
                     ptr(self.output_multiplier, cutlass.Float32, 4),
                     ptr(self.result.block_scales, cutlass.Uint8, 16))
        self._args = (ptr(self.weight.data, cutlass.Float4E2M1FN, 16),
                      ptr(self.weight.swizzled_scales, cutlass.Float8E4M3FN, 32),
                      ptr(value.data, cutlass.Float4E2M1FN, 16),
                      ptr(value.swizzled_scales, cutlass.Float8E4M3FN, 16),
                      ptr(self.partials if use_workspace else self.out,
                          cutlass.Float32 if use_workspace else cutlass.BFloat16, 16),
                      None if use_workspace else ptr(self.alpha, cutlass.Float32, 4), None, quant, prologue,
                      tuple(cutlass.Int32(v) for v in (*self.weight.shape[:1], self.m,
                                                     self.weight.shape[1], 1)))
        self._compiled = cute.compile(kernel, *self._args, make_fake_stream(),
                                      options="--generate-line-info")
        self._reduce = None
        if use_workspace:
            self._reduce_args = (ptr(self.partials, cutlass.Float32, 16),
                                 ptr(self.out, cutlass.BFloat16, 16),
                                 ptr(self.alpha, cutlass.Float32, 4),
                                 cutlass.Int32(self.m * self.weight.shape[0]))
            self._reduce = cute.compile(reduce_workspace, *self._reduce_args,
                                        cfg.split, self.weight.shape[0] if self.per_row_scale else 0,
                                        make_fake_stream())

    def _check_input(self, x):
        if (tuple(x.shape) != (self.m, self.weight.shape[1]) or x.dtype != torch.bfloat16
                or x.device != self.out.device or not x.is_contiguous()):
            raise ValueError("NVFP4 plan input shape/dtype/device/stride mismatch")

    def _gemm(self):
        stream = self._cuda.CUstream(torch.cuda.current_stream(self.out.device).cuda_stream)
        self._compiled(*self._args, stream)
        if self._reduce is not None:
            self._reduce(*self._reduce_args, stream)
        return self.out

    def warmup(self, x, *, iterations=5):
        self._check_input(x)
        if iterations <= 0:
            raise ValueError("warm-up iterations must be positive")
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("warm the M-bucket plan before CUDA graph capture")
        for _ in range(iterations):
            self._quantize(x)
            self._gemm()
        self._warmed = True
        return self

    def _quantize(self, x):
        quantizer = quantize_activation_rows if self.per_row_scale else quantize_activation
        return quantizer(x, workspace=self.workspace)

    def gemm(self):
        """Run from FP4 activations already produced into this plan's workspace."""
        if not self._warmed:
            raise RuntimeError("warm the M-bucket plan before use")
        return self._gemm()

    def gemm_from(self, packed, scales):
        """Launch directly from an external fused producer's 128x4 buffers.

        Before capture, set ``workspace.value.global_scale`` to the producer's
        static descale. The caller owns the external buffers and their lifetime.
        This method performs no device copy, allocation or host synchronization.
        """
        if not self._warmed:
            raise RuntimeError("warm the M-bucket plan before use")
        if (tuple(packed.shape) != (self.m, self.weight.shape[1] // 2)
                or packed.device != self.out.device or scales.device != self.out.device
                or not packed.is_contiguous() or not scales.is_contiguous()
                or scales.numel() < self.workspace.value.swizzled_scales.numel()):
            raise ValueError("external NVFP4 activation shape/device/stride mismatch")
        import cutlass
        from cutlass.cute.runtime import make_ptr
        a = make_ptr(cutlass.Float4E2M1FN, packed.data_ptr(), cutlass.AddressSpace.gmem, assumed_align=16)
        sf = make_ptr(cutlass.Float8E4M3FN, scales.data_ptr(), cutlass.AddressSpace.gmem, assumed_align=16)
        args = self._args[:2] + (a, sf) + self._args[4:]
        stream = self._cuda.CUstream(torch.cuda.current_stream(self.out.device).cuda_stream)
        self._compiled(*args, stream)
        if self._reduce is not None:
            self._reduce(*self._reduce_args, stream)
        return self.out

    def __call__(self, x):
        self._check_input(x)
        if not self._warmed:
            raise RuntimeError("warm the M-bucket plan before use")
        self._quantize(x)
        return self._gemm()
