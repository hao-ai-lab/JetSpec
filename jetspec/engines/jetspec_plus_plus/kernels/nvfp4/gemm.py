"""Thin, explicit M-bucket plans over torch and optional FlashInfer APIs.

    plan = W4A4Plan(weight, m, backend="torch")
    plan.warmup(x)  # on the stream used for capture, before capture
    output = plan(x)

Outputs/workspaces are owned by a plan and overwritten by the next call.
Plans are not concurrent-stream safe. Internal backend allocator calls are
allowed: graph replay memory stability is measured by nvfp4_bench/bench.py.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .activation import ActivationWorkspace, quantize_activation
from .format import NVFP4Tensor


@dataclass(frozen=True)
class BackendStatus:
    available: bool
    reason: str = ""


def _flashinfer_api(name):
    # Import lazily: format/fake-quant must work without FlashInfer or CUDA.
    import flashinfer
    return getattr(flashinfer, name)


def backend_status(backend: str) -> BackendStatus:
    """API availability only; hardware/backend support is checked at warm-up."""
    if backend in FI072_BACKENDS:
        try:
            import flashinfer
            if tuple(int(x) for x in flashinfer.__version__.split('.')[:3]) < (0, 7, 2):
                raise ImportError(f"installed version is {flashinfer.__version__}")
            _flashinfer_api("mm_bf16" if backend.startswith("bf16-") else "mm_fp4")
            _flashinfer_api("autotune")
            if backend.startswith("flashinfer072-"):
                for name in ("nvfp4_quantize", "fp4_quantize", "add_rmsnorm_fp4quant",
                             "silu_and_mul_nvfp4_quantize"):
                    _flashinfer_api(name)
        except (ImportError, AttributeError, OSError, ValueError) as error:
            return BackendStatus(False, f"{backend} unavailable: requires FlashInfer >= 0.7.2: {error}")
        return BackendStatus(True)
    if backend == "torch":
        available = hasattr(torch, "float4_e2m1fn_x2") and hasattr(torch.ops.aten._scaled_mm, "out")
        return BackendStatus(available, "" if available else "requires torch 2.8 FP4 _scaled_mm.out")
    if backend not in ("flashinfer-cutlass", "flashinfer-cudnn", "flashinfer-trtllm",
                       "w4a16-cudnn", "w4a16-cute-dsl"):
        raise ValueError(f"unknown NVFP4 backend: {backend}")
    try:
        if backend.startswith("w4a16"):
            _flashinfer_api("mm_bf16_fp4")
            _flashinfer_api("prepare_bf16_fp4_weights")
        else:
            _flashinfer_api("mm_fp4")
    except (ImportError, AttributeError, OSError) as error:
        requirement = "FlashInfer >= 0.6.18 mm_bf16_fp4" if backend.startswith("w4a16") else "FlashInfer 0.5.2 mm_fp4"
        return BackendStatus(False, f"{backend} unavailable: requires {requirement}: {error}")
    return BackendStatus(True)


def _validate_weight(weight, m):
    if not weight.data.is_cuda:
        raise RuntimeError("NVFP4 GEMM plans require CUDA; use fake_quantize for CPU")
    n, k = weight.shape
    if m <= 0 or n % 16 or k % 32:
        raise ValueError("GEMM requires M > 0, N divisible by 16, K divisible by 32")
    if not weight.data.is_contiguous():
        raise ValueError("packed weight must be contiguous")


class W4A4Plan:
    """Dynamic activation quantize + NVFP4 GEMM, one plan per exact M bucket.

    A fused producer may write ``workspace.value`` and call ``gemm()`` to
    bypass the standalone activation quantizer. Global scales are descales;
    FlashInfer alpha is their product. Torch applies this product after GEMM.
    """

    def __init__(self, weight: NVFP4Tensor, m: int, *, backend: str = "torch"):
        _validate_weight(weight, m)
        if backend.startswith("w4a16"):
            raise ValueError("use W4A16Plan for bf16 activations")
        status = backend_status(backend)
        if not status.available:
            raise RuntimeError(status.reason)
        self.weight, self.m, self.backend = weight, m, backend
        self.workspace = ActivationWorkspace.allocate(m, weight.shape[1], weight.data.device)
        self.out = torch.empty((m, weight.shape[0]), dtype=torch.bfloat16, device=weight.data.device)
        self.alpha = torch.empty((1,), dtype=torch.float32, device=weight.data.device)
        self._warmed = False
        if backend == "torch":
            self.a = self.workspace.value.data.view(torch.float4_e2m1fn_x2)
            self.b = weight.data.view(torch.float4_e2m1fn_x2).T
            self.a_scale = self.workspace.value.swizzled_scales
            self.b_scale = weight.swizzled_scales
        else:
            self._mm = _flashinfer_api("mm_fp4")
            self.a = self.workspace.value.data
            self.a_scale = self.workspace.value.swizzled_scales
            if backend == "flashinfer-trtllm":
                self.b = _flashinfer_api("shuffle_matrix_a")(weight.data, 128).T
                self.b_scale = _flashinfer_api("shuffle_matrix_sf_a")(
                    weight.block_scales.view(torch.uint8), 128).reshape(weight.swizzled_scales.shape).T
            else:
                self.b = weight.data.T
                self.b_scale = weight.swizzled_scales.T

    def _check_input(self, x):
        if (tuple(x.shape) != (self.m, self.weight.shape[1]) or x.dtype != torch.bfloat16
                or x.device != self.out.device or not x.is_contiguous()):
            raise ValueError("NVFP4 plan input shape/dtype/device/stride mismatch")

    def warmup(self, x: torch.Tensor, *, iterations: int = 5):
        self._check_input(x)
        if iterations <= 0:
            raise ValueError("warm-up iterations must be positive")
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("warm the M-bucket plan before CUDA graph capture")
        for _ in range(iterations):
            quantize_activation(x, workspace=self.workspace)
            self._gemm()
        self._warmed = True
        return self

    def _gemm(self):
        torch.mul(self.workspace.value.global_scale, self.weight.global_scale, out=self.alpha)
        if self.backend == "torch":
            torch._scaled_mm(self.a, self.b, self.a_scale, self.b_scale,
                             out_dtype=torch.bfloat16, out=self.out)
            self.out.mul_(self.alpha)
        else:
            self._mm(self.a, self.b, self.a_scale, self.b_scale, self.alpha,
                     out_dtype=torch.bfloat16, out=self.out, block_size=16,
                     use_8x4_sf_layout=False, backend=self.backend.removeprefix("flashinfer-"))
        return self.out

    def gemm(self):
        """Run from already-produced workspace FP4/scales (no quantize launch)."""
        if not self._warmed:
            raise RuntimeError("warm the M-bucket plan before use")
        return self._gemm()

    def __call__(self, x: torch.Tensor):
        self._check_input(x)
        if not self._warmed:
            raise RuntimeError("warm the M-bucket plan before use")
        quantize_activation(x, workspace=self.workspace)
        return self._gemm()


class W4A16Plan:
    """Optional >=0.6.18 FlashInfer bf16 x FP4; absent under the engine pin."""

    def __init__(self, weight: NVFP4Tensor, m: int, *, backend: str = "w4a16-cudnn"):
        _validate_weight(weight, m)
        if backend not in ("w4a16-cudnn", "w4a16-cute-dsl"):
            raise ValueError("W4A16 backend must be w4a16-cudnn or w4a16-cute-dsl")
        status = backend_status(backend)
        if not status.available:
            raise RuntimeError(status.reason)
        self.weight, self.m = weight, m
        self.backend = backend.removeprefix("w4a16-")
        self._mm = _flashinfer_api("mm_bf16_fp4")
        self.b, self.b_scale, self.alpha = _flashinfer_api("prepare_bf16_fp4_weights")(
            weight.data, weight.swizzled_scales, weight.global_scale, backend=self.backend)
        self.out = torch.empty((m, weight.shape[0]), device=weight.data.device, dtype=torch.bfloat16)
        self._warmed = False

    def _check_input(self, x):
        if (tuple(x.shape) != (self.m, self.weight.shape[1]) or x.dtype != torch.bfloat16
                or x.device != self.out.device or not x.is_contiguous()):
            raise ValueError("NVFP4 plan input shape/dtype/device/stride mismatch")

    def _gemm(self, x):
        return self._mm(x, self.b, self.b_scale, self.alpha, backend=self.backend,
                        out_dtype=torch.bfloat16, out=self.out)

    def warmup(self, x: torch.Tensor, *, iterations: int = 5):
        self._check_input(x)
        if iterations <= 0:
            raise ValueError("warm-up iterations must be positive")
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("warm the M-bucket plan before CUDA graph capture")
        for _ in range(iterations):
            self._gemm(x)
        self._warmed = True
        return self

    def __call__(self, x: torch.Tensor):
        self._check_input(x)
        if not self._warmed:
            raise RuntimeError("warm the M-bucket plan before use")
        return self._gemm(x)


FI072_BACKENDS = {
    "flashinfer072-cutedsl": "cute-dsl",
    "flashinfer072-trtllm": "trtllm",
    "flashinfer072-lowlat": "cutedsl_low_latency",
    "bf16-fi-cutedsl": "cute-dsl",
    "bf16-fi-cublaslt": "cublaslt",
}


def rmsnorm_input(x, residual, weight, eps=1e-6):
    """Plain torch control for the fused residual-add/RMSNorm producer."""
    summed = (x + residual).to(x.dtype)
    values = summed.float()
    return (values * torch.rsqrt(values.square().mean(-1, keepdim=True) + eps)
            * weight.float()).to(x.dtype)


class FlashInfer072Plan:
    """Opt-in 0.7.2 plans, leaving the pinned engine's older plans unchanged.

    BF16 weights are quantized once by FlashInfer. Activation scales are
    calibrated once in warmup, outside graph capture. All three producers
    take multipliers (including add_rmsnorm_fp4quant in the 0.7.2 kernel,
    despite its docstring describing a descale).
    ``producer`` is plain, rmsnorm, or silu. All producer work is also paid
    by the bf16 controls. Plans own output storage and are single-stream.
    """

    def __init__(self, weight, m, *, backend, producer="plain", force_split_k=0):
        if backend not in FI072_BACKENDS:
            raise ValueError(f"unknown FlashInfer 0.7.2 backend: {backend}")
        if backend == "flashinfer072-lowlat" and m > 8:
            raise ValueError("flashinfer072-lowlat requires M <= 8")
        if producer not in ("plain", "rmsnorm", "silu"):
            raise ValueError("producer must be plain, rmsnorm, or silu")
        if force_split_k and (backend != "flashinfer072-cutedsl" or m > 32
                              or force_split_k not in (2, 4)):
            raise ValueError("forced split-K requires cute-dsl, M <= 32, and 2 or 4 slices")
        if not weight.is_cuda:
            raise RuntimeError("FlashInfer 0.7.2 plans require CUDA")
        if weight.ndim != 2 or weight.dtype != torch.bfloat16 or not weight.is_contiguous():
            raise ValueError("weight must be a contiguous bf16 matrix")
        status = backend_status(backend)
        if not status.available:
            raise RuntimeError(status.reason)
        import flashinfer as fi
        self.fi, self.backend, self.m = fi, backend, m
        self.producer, self.force_split_k = producer, force_split_k
        self.n, self.k = weight.shape
        self.fp4 = backend.startswith("flashinfer072-")
        self.use_8x4 = backend == "flashinfer072-trtllm" and m <= 32
        self.out = torch.empty((m, self.n), device=weight.device, dtype=torch.bfloat16)
        self._warmed = False
        self.tactic = None
        if self.fp4:
            self.weight_scale = weight.float().abs().max().clamp_min(1e-12).reshape(1) / 2688
            b, sf = fi.nvfp4_quantize(weight, self.weight_scale.reciprocal(),
                                    sfLayout=fi.SfLayout.layout_128x4,
                                    do_shuffle=backend == "flashinfer072-trtllm", enable_pdl=True)
            self.b, self.b_scale = b.T, sf.T
        else:
            self.b = weight.T

    def _input(self, x):
        if self.producer == "rmsnorm":
            return rmsnorm_input(self.zero_increment, self.residual, self.norm_weight)
        if self.producer == "silu":
            return torch.nn.functional.silu(x[:, :self.k]) * x[:, self.k:]
        return x

    def _produce(self, x):
        if not self.fp4:
            return self._input(x)
        if self.producer == "rmsnorm":
            # Zero increment keeps synthetic inputs identical across replays.
            # The fused kernel still executes the residual add and RMSNorm.
            a, sf = self.fi.add_rmsnorm_fp4quant(
                self.zero_increment, self.residual, self.norm_weight,
                global_scale=self.activation_multiplier, is_sf_swizzled_layout=True,
                enable_pdl=True)
            a = a.view(torch.uint8)
            sf = sf.reshape(128 * ((self.m + 127) // 128), -1)
            if self.use_8x4:
                sf = sf.view(torch.uint8).flatten()[self.scale_8x4_indices].reshape(
                    (self.m + 7) // 8 * 8, self.k // 16)
        elif self.producer == "silu":
            a, sf = self.fi.silu_and_mul_nvfp4_quantize(
                x, self.activation_multiplier, is_sf_8x4_layout=self.use_8x4, enable_pdl=True)
        else:
            a, sf = self.fi.fp4_quantize(x, self.activation_multiplier,
                                       is_sf_8x4_layout=self.use_8x4, enable_pdl=True)
        self.a, self.a_scale = a, sf
        return a

    def _gemm(self, activation):
        if not self.fp4:
            return self.fi.mm_bf16(activation, self.b, out=self.out,
                                  backend=FI072_BACKENDS[self.backend],
                                  pdl=self.backend == 'bf16-fi-cutedsl')
        if self.force_split_k:
            self.runner(inputs=[self.a, self.b, self.a_scale, self.b_scale, self.alpha,
                                torch.bfloat16, self.out, 16, True, self.scratch],
                        tactic=self.tactic)
            return self.out
        return self.fi.mm_fp4(self.a, self.b, self.a_scale, self.b_scale, self.alpha,
                              out=self.out, backend=FI072_BACKENDS[self.backend],
                              use_8x4_sf_layout=self.use_8x4, enable_pdl=True)

    def warmup(self, x, *, iterations=5, tune=True, cache=None):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("warm the M-bucket plan before CUDA graph capture")
        expected_k = self.k * (2 if self.producer == "silu" else 1)
        if (tuple(x.shape) != (self.m, expected_k) or x.dtype != torch.bfloat16
                or x.device != self.out.device or not x.is_contiguous()):
            raise ValueError("FlashInfer plan input shape/dtype/device/stride mismatch")
        if iterations <= 0:
            raise ValueError("warm-up iterations must be positive")
        if self.producer == "rmsnorm" and not hasattr(self, "residual"):
            self.zero_increment = torch.zeros_like(x)
            self.residual = x.clone()
            self.norm_weight = torch.ones(self.k, dtype=x.dtype, device=x.device)
        if self.fp4 and not hasattr(self, "alpha"):
            self.activation_descale = self._input(x).float().abs().max().clamp_min(1e-12).reshape(1) / 2688
            self.activation_multiplier = self.activation_descale.reciprocal()
            self.alpha = self.activation_descale * self.weight_scale
            if self.use_8x4 and self.producer == "rmsnorm":
                # Gather directly from 128x4 to 8x4. Conversion is timed.
                r = torch.arange((self.m + 7) // 8 * 8, device=x.device)[:, None]
                c = torch.arange(self.k // 16, device=x.device)[None, :]
                source = ((r // 128) * (self.k // 64) + c // 4) * 512
                source = source + (r % 32) * 16 + ((r % 128) // 32) * 4 + c % 4
                self.scale_8x4_indices = source.reshape(-1, 8, self.k // 64, 4).permute(
                    0, 2, 1, 3).contiguous().flatten()
            if self.force_split_k:
                from flashinfer.gemm import gemm_base as gb
                major, minor = torch.cuda.get_device_capability(x.device)
                self.runner = gb._cute_dsl_gemm_fp4_runner(major, minor, True, torch.bfloat16, True)
                self.tactic = ((128, 8 if self.m <= 8 else 16 if self.m <= 16 else 32),
                               (1, 1), True, False, gb._SM100_SPLITK_KERNEL_TYPE, self.force_split_k)
                self.scratch = torch.empty(1, device=x.device, dtype=torch.uint8)
        activation = self._produce(x)
        if tune and not self.force_split_k:
            with self.fi.autotune(True, cache=cache, tuning_buckets=(self.m,),
                                 cuda_graph_profile_replays=3):
                self._gemm(activation)
        for _ in range(iterations):
            self._gemm(self._produce(x))
        self._warmed = True
        return self

    def gemm(self):
        if not self._warmed or not self.fp4:
            raise RuntimeError("gemm() requires a warmed FP4 plan")
        return self._gemm(self.a)

    def __call__(self, x):
        if not self._warmed:
            raise RuntimeError("warm the M-bucket plan before use")
        return self._gemm(self._produce(x))
