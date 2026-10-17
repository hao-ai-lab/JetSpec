"""CUDA kernels for the opt-in C1 serving path.

The CPU implementations intentionally spell the reference arithmetic out so the
portable tests exercise the same public path without requiring Triton.
"""

from __future__ import annotations

from jetspec.engines.jetspec_plus_plus.profiles import kernel_enabled, kernel_value

import os

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    from triton.language.extra.cuda import libdevice
except ImportError:  # Triton is Linux/CUDA-only in the pinned environment.
    triton = None
    tl = None


_NORM_EXTENSION = None


def resolve_c1_qk_warps(tokens, *, run_config=None):
    """Retain the sixteen-token launch's tuning contract without a native env read."""
    if tokens != 16:
        return 16
    from jetspec.engines.jetspec_plus_plus.jetspec.config_values import config_values
    value = config_values(run_config).get('engine.engine_settings.c1_qk_warps', '16')
    if value not in ('4', '8', '16'):
        raise RuntimeError('C1 QK warps must be 4, 8 or 16')
    return int(value)


def query_output(source, heads, tokens, dim, *, batch=1):
    """Uninitialized query storage for a QK writer that honors output strides."""
    from jetspec.engines.jetspec_plus_plus.profiles import kernel_enabled
    if (kernel_enabled('adapter_query_layout_fused', '1')
            and source.dtype == torch.bfloat16 and batch == 1):
        return source.new_empty((batch, tokens, heads, dim)).transpose(1, 2)
    return source.new_empty((batch, heads, tokens, dim))


def attention_output(q):
    """Let the adapter projection consume attention output as a flat view."""
    return torch.empty_like(q)

_NORM_CPP = r"""
#include <torch/extension.h>

torch::Tensor c1_rms_norm_cuda(torch::Tensor input, torch::Tensor weight, double eps);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("rms_norm", &c1_rms_norm_cuda, "C1 BF16 RMSNorm");
}
"""

_NORM_CUDA = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>

__global__ void c1_rms_norm_kernel(
    const __nv_bfloat16* input,
    const __nv_bfloat16* weight,
    __nv_bfloat16* output,
    int rows,
    int width,
    int row_stride,
    float eps) {
  int row = blockIdx.x * blockDim.y + threadIdx.y;
  if (row >= rows) return;
  int thread = threadIdx.x;
  float partial[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  if (width == 128) {
#pragma unroll
    for (int item = 0; item < 4; ++item) {
      float value = __bfloat162float(
          input[row * row_stride + thread + item * 32]);
      partial[item] = __fmul_rn(value, value);
    }
  } else {
    for (int base = thread * 4; base < width; base += 128) {
#pragma unroll
      for (int item = 0; item < 4; ++item) {
        float value = __bfloat162float(input[row * row_stride + base + item]);
        partial[item] = __fadd_rn(partial[item], __fmul_rn(value, value));
      }
    }
  }
  float sum_sq = __fadd_rn(__fadd_rn(partial[0], partial[1]), partial[2]);
  sum_sq = __fadd_rn(sum_sq, partial[3]);
#pragma unroll
  for (int offset = 1; offset < 32; offset <<= 1) {
    sum_sq = __fadd_rn(
        sum_sq, __shfl_down_sync(0xffffffff, sum_sq, offset));
  }
  float reciprocal_rms = 0.0f;
  if (thread == 0) {
    float mean = __fmul_rn(sum_sq, 1.0f / float(width));
    reciprocal_rms = rsqrtf(__fadd_rn(mean, eps));
  }
  reciprocal_rms = __shfl_sync(0xffffffff, reciprocal_rms, 0);
  for (int column = thread; column < width; column += 32) {
      float value = __bfloat162float(input[row * row_stride + column]);
      __nv_bfloat16 normalized = __float2bfloat16_rn(value * reciprocal_rms);
      float scaled = __bfloat162float(normalized) * __bfloat162float(weight[column]);
      output[row * width + column] = __float2bfloat16_rn(scaled);
  }
}

torch::Tensor c1_rms_norm_cuda(torch::Tensor input, torch::Tensor weight, double eps) {
  TORCH_CHECK(input.is_cuda() && weight.is_cuda(), "C1 RMSNorm requires CUDA tensors");
  TORCH_CHECK(input.scalar_type() == torch::kBFloat16, "C1 RMSNorm requires BF16 input");
  TORCH_CHECK(weight.scalar_type() == torch::kBFloat16, "C1 RMSNorm requires BF16 weight");
  TORCH_CHECK(input.stride(-1) == 1 && weight.is_contiguous(),
              "C1 RMSNorm requires a dense last dimension");
  int width = input.size(-1);
  TORCH_CHECK(width == 128 || (width >= 128 && width % 4 == 0),
              "C1 RMSNorm only supports the pinned head widths");
  int rows = input.numel() / width;
  int row_stride = input.is_contiguous() ? width : input.stride(-2);
  TORCH_CHECK(input.is_contiguous() || (input.dim() == 3 && input.size(0) == 1),
              "C1 RMSNorm only permits a B1 strided tap view");
  auto output = torch::empty(input.sizes(), input.options());
  c10::cuda::CUDAGuard guard(input.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  dim3 threads(32, 16);
  int blocks = (rows + 15) / 16;
  c1_rms_norm_kernel<<<blocks, threads, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(weight.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
      rows, width, row_stride, float(eps));
  TORCH_CHECK(cudaGetLastError() == cudaSuccess, "C1 RMSNorm kernel launch failed");
  return output;
}
"""


def _norm_extension():
    global _NORM_EXTENSION
    if _NORM_EXTENSION is None:
        from torch.utils.cpp_extension import load_inline
        _NORM_EXTENSION = load_inline(
            name="jetspec_c1_norm_ext_v4",
            cpp_sources=_NORM_CPP,
            cuda_sources=_NORM_CUDA,
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3", "--fmad=false"],
            with_cuda=True,
            verbose=False,
        )
    return _NORM_EXTENSION


_WIDE_NORM_EXTENSION = None

# JETSPEC_C1_NORM=wide: the same RMSNorm arithmetic as c1_rms_norm_kernel, bit for bit
# (the same per-thread element order, the same fadd/fmul chain, the same shuffle
# tree and rsqrtf, the same two bf16 roundings), with the 4096-wide rows read once
# into registers by vectorised loads and written from them.  The certified kernel
# walks each 4096 row twice with scalar loads in a runtime-bounded loop, so one warp
# pays the memory latency per iteration: ~60 us for 15 rows.
_WIDE_NORM_CUDA = _NORM_CUDA.replace(
    "torch::Tensor c1_rms_norm_cuda(torch::Tensor input, torch::Tensor weight, double eps) {",
    r"""
__device__ __forceinline__ float c1_bf16_lane(unsigned int word, int high) {
  return __bfloat162float(__ushort_as_bfloat16(
      (unsigned short)(high ? (word >> 16) : (word & 0xFFFFu))));
}

__device__ __forceinline__ unsigned int c1_bf16_pack(__nv_bfloat16 low, __nv_bfloat16 high) {
  return (unsigned int)__bfloat16_as_ushort(low)
      | ((unsigned int)__bfloat16_as_ushort(high) << 16);
}

__global__ void c1_rms_norm_wide_kernel(
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16* __restrict__ output,
    int rows,
    int row_stride,
    float eps) {
  constexpr int WIDTH = 4096;
  constexpr int STEPS = WIDTH / 128;
  int row = blockIdx.x * blockDim.y + threadIdx.y;
  if (row >= rows) return;
  int thread = threadIdx.x;
  const __nv_bfloat16* source = input + (long long)row * row_stride + thread * 4;
  uint2 raw[STEPS];
#pragma unroll
  for (int step = 0; step < STEPS; ++step) {
    raw[step] = *reinterpret_cast<const uint2*>(source + step * 128);
  }
  float partial[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
  for (int step = 0; step < STEPS; ++step) {
#pragma unroll
    for (int item = 0; item < 4; ++item) {
      float value = c1_bf16_lane(item < 2 ? raw[step].x : raw[step].y, item & 1);
      partial[item] = __fadd_rn(partial[item], __fmul_rn(value, value));
    }
  }
  float sum_sq = __fadd_rn(__fadd_rn(partial[0], partial[1]), partial[2]);
  sum_sq = __fadd_rn(sum_sq, partial[3]);
#pragma unroll
  for (int offset = 1; offset < 32; offset <<= 1) {
    sum_sq = __fadd_rn(
        sum_sq, __shfl_down_sync(0xffffffff, sum_sq, offset));
  }
  float reciprocal_rms = 0.0f;
  if (thread == 0) {
    float mean = __fmul_rn(sum_sq, 1.0f / float(WIDTH));
    reciprocal_rms = rsqrtf(__fadd_rn(mean, eps));
  }
  reciprocal_rms = __shfl_sync(0xffffffff, reciprocal_rms, 0);
  const __nv_bfloat16* scale = weight + thread * 4;
  uint2 scale_all[STEPS];
#pragma unroll
  for (int step = 0; step < STEPS; ++step) {
    scale_all[step] = *reinterpret_cast<const uint2*>(scale + step * 128);
  }
  __nv_bfloat16* target = output + (long long)row * WIDTH + thread * 4;
#pragma unroll
  for (int step = 0; step < STEPS; ++step) {
    uint2 scale_raw = scale_all[step];
    __nv_bfloat16 results[4];
#pragma unroll
    for (int item = 0; item < 4; ++item) {
      float value = c1_bf16_lane(item < 2 ? raw[step].x : raw[step].y, item & 1);
      __nv_bfloat16 normalized = __float2bfloat16_rn(value * reciprocal_rms);
      float scaled = __bfloat162float(normalized)
          * c1_bf16_lane(item < 2 ? scale_raw.x : scale_raw.y, item & 1);
      results[item] = __float2bfloat16_rn(scaled);
    }
    uint2 result = make_uint2(c1_bf16_pack(results[0], results[1]),
                              c1_bf16_pack(results[2], results[3]));
    *reinterpret_cast<uint2*>(target + step * 128) = result;
  }
}

template<int STEPS>
__global__ void c1_rms_norm_wide_kernel_t(
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16* __restrict__ output,
    int rows, int row_stride, float eps) {
  constexpr int WIDTH = STEPS * 128;
  int row = blockIdx.x * blockDim.y + threadIdx.y;
  if (row >= rows) return;
  int thread = threadIdx.x;
  const __nv_bfloat16* source = input + (long long)row * row_stride + thread * 4;
  float partial[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
  for (int step = 0; step < STEPS; ++step) {
    uint2 raw = *reinterpret_cast<const uint2*>(source + step * 128);
#pragma unroll
    for (int item = 0; item < 4; ++item) {
      float value = c1_bf16_lane(item < 2 ? raw.x : raw.y, item & 1);
      partial[item] = __fadd_rn(partial[item], __fmul_rn(value, value));
    }
  }
  float sum_sq = __fadd_rn(__fadd_rn(partial[0], partial[1]), partial[2]);
  sum_sq = __fadd_rn(sum_sq, partial[3]);
#pragma unroll
  for (int offset = 1; offset < 32; offset <<= 1) {
    sum_sq = __fadd_rn(sum_sq, __shfl_down_sync(0xffffffff, sum_sq, offset));
  }
  float reciprocal_rms = 0.0f;
  if (thread == 0) {
    float mean = __fmul_rn(sum_sq, 1.0f / float(WIDTH));
    reciprocal_rms = rsqrtf(__fadd_rn(mean, eps));
  }
  reciprocal_rms = __shfl_sync(0xffffffff, reciprocal_rms, 0);
  const __nv_bfloat16* scale = weight + thread * 4;
  __nv_bfloat16* target = output + (long long)row * WIDTH + thread * 4;
#pragma unroll
  for (int step = 0; step < STEPS; ++step) {
    uint2 raw = *reinterpret_cast<const uint2*>(source + step * 128);
    uint2 scale_raw = *reinterpret_cast<const uint2*>(scale + step * 128);
    __nv_bfloat16 results[4];
#pragma unroll
    for (int item = 0; item < 4; ++item) {
      float value = c1_bf16_lane(item < 2 ? raw.x : raw.y, item & 1);
      __nv_bfloat16 normalized = __float2bfloat16_rn(value * reciprocal_rms);
      float scaled = __bfloat162float(normalized)
          * c1_bf16_lane(item < 2 ? scale_raw.x : scale_raw.y, item & 1);
      results[item] = __float2bfloat16_rn(scaled);
    }
    uint2 result = make_uint2(c1_bf16_pack(results[0], results[1]),
                              c1_bf16_pack(results[2], results[3]));
    *reinterpret_cast<uint2*>(target + step * 128) = result;
  }
}

torch::Tensor c1_rms_norm_cuda(torch::Tensor input, torch::Tensor weight, double eps) {""").replace(
    """  dim3 threads(32, 16);
  int blocks = (rows + 15) / 16;
  c1_rms_norm_kernel<<<blocks, threads, 0, stream>>>(""",
    """  if (width == 4096 && row_stride % 4 == 0
      && reinterpret_cast<uintptr_t>(input.data_ptr()) % 8 == 0
      && reinterpret_cast<uintptr_t>(weight.data_ptr()) % 8 == 0
      && reinterpret_cast<uintptr_t>(output.data_ptr()) % 8 == 0) {
    dim3 wide_threads(32, 4);
    int wide_blocks = (rows + 3) / 4;
    c1_rms_norm_wide_kernel<<<wide_blocks, wide_threads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(weight.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
        rows, row_stride, float(eps));
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "C1 wide RMSNorm kernel launch failed");
    return output;
  }
  if (width == 5120 && row_stride % 4 == 0
      && reinterpret_cast<uintptr_t>(input.data_ptr()) % 8 == 0
      && reinterpret_cast<uintptr_t>(weight.data_ptr()) % 8 == 0
      && reinterpret_cast<uintptr_t>(output.data_ptr()) % 8 == 0) {
    dim3 wide_threads(32, 4);
    int wide_blocks = (rows + 3) / 4;
    c1_rms_norm_wide_kernel_t<40><<<wide_blocks, wide_threads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(weight.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
        rows, row_stride, float(eps));
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "C1 wide RMSNorm kernel launch failed");
    return output;
  }
  dim3 threads(32, 16);
  int blocks = (rows + 15) / 16;
  c1_rms_norm_kernel<<<blocks, threads, 0, stream>>>(""")


def _wide_norm_extension():
    global _WIDE_NORM_EXTENSION
    if _WIDE_NORM_EXTENSION is None:
        from torch.utils.cpp_extension import load_inline
        _WIDE_NORM_EXTENSION = load_inline(
            name="jetspec_c1_norm_ext_wide_v4",
            cpp_sources=_NORM_CPP,
            cuda_sources=_WIDE_NORM_CUDA,
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3", "--fmad=false"],
            with_cuda=True,
            verbose=False,
        )
    return _WIDE_NORM_EXTENSION


if triton is not None:

    @triton.jit
    def _module_rms_scale(values, WIDTH: tl.constexpr, THREADS: tl.constexpr,
                          EPS: tl.constexpr, BLOCK: tl.constexpr):
        # Reduce.cuh's contiguous float mean: four accumulators per lane,
        # vectorized input above D128, then shared halves and ascending shuffles.
        lane = tl.arange(0, THREADS)
        a0 = tl.full((THREADS,), 0, tl.float32)
        a1 = tl.full((THREADS,), 0, tl.float32)
        a2 = tl.full((THREADS,), 0, tl.float32)
        a3 = tl.full((THREADS,), 0, tl.float32)
        STEPS: tl.constexpr = (WIDTH + THREADS * 4 - 1) // (THREADS * 4)
        for base in tl.static_range(STEPS):
            if WIDTH > 128:
                index = base * THREADS * 4 + lane * 4
                i0, i1, i2, i3 = index, index + 1, index + 2, index + 3
            else:
                index = base * THREADS * 4 + lane
                i0, i1, i2, i3 = index, index + THREADS, index + THREADS * 2, index + THREADS * 3
            x0 = tl.where(i0 < WIDTH, tl.gather(values, tl.minimum(i0, BLOCK - 1), 0), 0)
            x1 = tl.where(i1 < WIDTH, tl.gather(values, tl.minimum(i1, BLOCK - 1), 0), 0)
            x2 = tl.where(i2 < WIDTH, tl.gather(values, tl.minimum(i2, BLOCK - 1), 0), 0)
            x3 = tl.where(i3 < WIDTH, tl.gather(values, tl.minimum(i3, BLOCK - 1), 0), 0)
            a0, a1, a2, a3 = a0 + x0 * x0, a1 + x1 * x1, a2 + x2 * x2, a3 + x3 * x3
        summed = ((a0 + a1) + a2) + a3
        if THREADS > 256:
            summed = summed + tl.gather(summed, (lane + 256) % THREADS, 0)
        if THREADS > 128:
            summed = summed + tl.gather(summed, (lane + 128) % THREADS, 0)
        if THREADS > 64:
            summed = summed + tl.gather(summed, (lane + 64) % THREADS, 0)
        if THREADS > 32:
            summed = summed + tl.gather(summed, (lane + 32) % THREADS, 0)
        for shift in tl.static_range(5):
            if (1 << shift) < THREADS:
                other = tl.minimum(lane + (1 << shift), (lane // 32 + 1) * 32 - 1)
                other = tl.minimum(other, THREADS - 1)
                summed = summed + tl.gather(summed, other, 0)
        total = tl.sum(tl.where(lane == 0, summed, 0), 0)
        return libdevice.rsqrt(total * (1.0 / WIDTH) + EPS)

    @triton.jit
    def _pointer_module_rms_scale(SOURCE, UPDATE, RESIDUAL, row,
                                  WIDTH: tl.constexpr, THREADS: tl.constexpr,
                                  EPS: tl.constexpr, PENDING: tl.constexpr):
        # Load each lane's four accumulators directly. Gathering repeatedly from
        # an 8192-wide register vector creates expensive cross-warp traffic.
        lane = tl.arange(0, THREADS)
        a0 = tl.full((THREADS,), 0, tl.float32)
        a1 = tl.full((THREADS,), 0, tl.float32)
        a2 = tl.full((THREADS,), 0, tl.float32)
        a3 = tl.full((THREADS,), 0, tl.float32)
        for step in tl.static_range((WIDTH + THREADS * 4 - 1) // (THREADS * 4)):
            index = row * WIDTH + step * THREADS * 4 + lane * 4
            valid = step * THREADS * 4 + lane * 4 < WIDTH
            if PENDING:
                x0 = (tl.load(UPDATE + index, valid, 0) + tl.load(RESIDUAL + index, valid, 0)).to(tl.bfloat16).to(tl.float32)
                x1 = (tl.load(UPDATE + index + 1, valid, 0) + tl.load(RESIDUAL + index + 1, valid, 0)).to(tl.bfloat16).to(tl.float32)
                x2 = (tl.load(UPDATE + index + 2, valid, 0) + tl.load(RESIDUAL + index + 2, valid, 0)).to(tl.bfloat16).to(tl.float32)
                x3 = (tl.load(UPDATE + index + 3, valid, 0) + tl.load(RESIDUAL + index + 3, valid, 0)).to(tl.bfloat16).to(tl.float32)
            else:
                x0 = tl.load(SOURCE + index, valid, 0).to(tl.float32)
                x1 = tl.load(SOURCE + index + 1, valid, 0).to(tl.float32)
                x2 = tl.load(SOURCE + index + 2, valid, 0).to(tl.float32)
                x3 = tl.load(SOURCE + index + 3, valid, 0).to(tl.float32)
            a0, a1, a2, a3 = a0 + x0 * x0, a1 + x1 * x1, a2 + x2 * x2, a3 + x3 * x3
        summed = ((a0 + a1) + a2) + a3
        for shift in tl.static_range(4):
            offset = 256 >> shift
            if THREADS > offset:
                summed = summed + tl.gather(summed, (lane + offset) % THREADS, 0)
        for shift in tl.static_range(5):
            other = tl.minimum(lane + (1 << shift), (lane // 32 + 1) * 32 - 1)
            summed = summed + tl.gather(summed, tl.minimum(other, THREADS - 1), 0)
        total = tl.sum(tl.where(lane == 0, summed, 0), 0)
        return libdevice.rsqrt(total * (1.0 / WIDTH) + EPS)

    @triton.jit
    def _finish_streams_kernel(s0, s1, s2, s3, residual, update, out,
                               rows: tl.constexpr, width: tl.constexpr,
                               BLOCK: tl.constexpr):
        row = tl.program_id(0)
        stream = tl.program_id(1)
        col = tl.arange(0, BLOCK)
        offset = row * width + col
        if stream == 0:
            value = tl.load(s0 + offset, mask=col < width, other=0)
        elif stream == 1:
            value = tl.load(s1 + offset, mask=col < width, other=0)
        elif stream == 2:
            value = tl.load(s2 + offset, mask=col < width, other=0)
        elif stream == 3:
            value = tl.load(s3 + offset, mask=col < width, other=0)
        else:
            base = tl.load(residual + offset, mask=col < width, other=0)
            delta = tl.load(update + offset, mask=col < width, other=0)
            value = (base + delta).to(tl.bfloat16)
        tl.store(out + row * (5 * width) + stream * width + col,
                 value, mask=col < width)

    @triton.jit(do_not_specialize=["rows"])
    def _fast_rms_norm_kernel(input_ptr, weight_ptr, output_ptr,
                              rows, width: tl.constexpr,
                              eps: tl.constexpr, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        offsets = tl.arange(0, BLOCK)
        mask = offsets < width
        values = tl.load(
            input_ptr + row * width + offsets, mask=mask, other=0.0
        ).to(tl.float32)
        reciprocal_rms = tl.rsqrt(tl.sum(values * values, axis=0) / width + eps)
        normalized = (values * reciprocal_rms).to(tl.bfloat16)
        weights = tl.load(weight_ptr + offsets, mask=mask, other=0.0)
        tl.store(
            output_ptr + row * width + offsets,
            (normalized * weights).to(tl.bfloat16),
            mask=mask,
        )


    @triton.jit(do_not_specialize=["rows"])
    def _fast_add_rms_norm_kernel(update_ptr, residual_ptr, weight_ptr,
                                  normalized_ptr, state_ptr,
                                  rows, width: tl.constexpr,
                                  eps: tl.constexpr, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        offsets = tl.arange(0, BLOCK)
        mask = offsets < width
        base = row * width + offsets
        update = tl.load(update_ptr + base, mask=mask, other=0.0)
        residual = tl.load(residual_ptr + base, mask=mask, other=0.0)
        state = (update + residual).to(tl.bfloat16)
        tl.store(state_ptr + base, state, mask=mask)
        values = state.to(tl.float32)
        reciprocal_rms = tl.rsqrt(tl.sum(values * values, axis=0) / width + eps)
        normalized = (values * reciprocal_rms).to(tl.bfloat16)
        weights = tl.load(weight_ptr + offsets, mask=mask, other=0.0)
        tl.store(
            normalized_ptr + base,
            (normalized * weights).to(tl.bfloat16),
            mask=mask,
        )

    @triton.jit
    def _ew_pre_norm_kernel(input_ptr, update_ptr, residual_ptr, window_ptr,
                            input_weight_ptr, kv_weight_ptr, normalized_ptr,
                            state_ptr, kv_ptr, append_ptr, stream_ptr,
                            append_first: tl.constexpr, streams: tl.constexpr,
                            kv_blocks: tl.constexpr, FOLD: tl.constexpr,
                            rows: tl.constexpr,
                            width: tl.constexpr, input_eps: tl.constexpr,
                            kv_eps: tl.constexpr, PENDING: tl.constexpr,
                            KV_FROM_STATE: tl.constexpr, BLOCK: tl.constexpr,
                            NORM_OFFSET: tl.constexpr = 0, REDUCE_THREADS: tl.constexpr = 32,
                            ULP_STATS: tl.constexpr = False):
        program = tl.program_id(0)
        row = program % rows
        group = program // rows
        offsets = tl.arange(0, BLOCK)
        mask = offsets < width
        base = row * width + offsets
        if group == 0:
            if PENDING:
                update = tl.load(update_ptr + base, mask=mask, other=0.0)
                residual = tl.load(residual_ptr + base, mask=mask, other=0.0)
                values = (update + residual).to(tl.bfloat16)
                tl.store(state_ptr + base, values, mask=mask)
            else:
                values = tl.load(input_ptr + base, mask=mask, other=0.0)
            floats = values.to(tl.float32)
            weights = tl.load(input_weight_ptr + offsets, mask=mask, other=0.0)
            if NORM_OFFSET:
                if ULP_STATS and width == 5120:
                    scale = _pointer_module_rms_scale(input_ptr, update_ptr, residual_ptr,
                        row, width, REDUCE_THREADS, input_eps, PENDING)
                else:
                    scale = _module_rms_scale(floats, width, REDUCE_THREADS, input_eps, BLOCK)
                result = (floats * scale) * (NORM_OFFSET + weights.to(tl.float32))
            else:
                scale = tl.rsqrt(tl.sum(floats * floats, axis=0) / width + input_eps)
                normalized = (floats * scale).to(tl.bfloat16)
                result = (normalized * weights).to(tl.bfloat16)
            tl.store(normalized_ptr + base, result, mask=mask)
        elif not FOLD or group <= kv_blocks:
            if PENDING and KV_FROM_STATE:
                update = tl.load(update_ptr + base, mask=mask, other=0.0)
                residual = tl.load(residual_ptr + base, mask=mask, other=0.0)
                values = (update + residual).to(tl.bfloat16)
            elif KV_FROM_STATE:
                values = tl.load(input_ptr + base, mask=mask, other=0.0)
            else:
                values = tl.load(window_ptr + base, mask=mask, other=0.0)
            floats = values.to(tl.float32)
            weights = tl.load(kv_weight_ptr + (group - 1) * width + offsets,
                              mask=mask, other=0.0)
            if NORM_OFFSET:
                if ULP_STATS and width == 5120:
                    source = input_ptr if KV_FROM_STATE else window_ptr
                    scale = _pointer_module_rms_scale(source, update_ptr, residual_ptr,
                        row, width, REDUCE_THREADS, kv_eps, PENDING and KV_FROM_STATE)
                else:
                    scale = _module_rms_scale(floats, width, REDUCE_THREADS, kv_eps, BLOCK)
                result = (floats * scale) * (NORM_OFFSET + weights.to(tl.float32))
            else:
                scale = tl.rsqrt(tl.sum(floats * floats, axis=0) / width + kv_eps)
                normalized = (floats * scale).to(tl.bfloat16)
                result = (normalized * weights).to(tl.bfloat16)
            output_base = ((group - 1) * (2 if FOLD else 1) * rows
                           + (rows if FOLD else 0)) * width + base
            tl.store(kv_ptr + output_base,
                     result, mask=mask)
        else:
            block = group - kv_blocks - 1
            stream = tl.load(stream_ptr + append_first + block)
            append_values = tl.load(append_ptr + row * streams * width + stream * width + offsets,
                                    mask=mask, other=0.0).to(tl.float32)
            append_scale = tl.rsqrt(tl.sum(append_values * append_values, axis=0) / width + kv_eps)
            append_normalized = (append_values * append_scale).to(tl.bfloat16)
            append_weights = tl.load(kv_weight_ptr + block * width + offsets,
                                     mask=mask, other=0.0)
            tl.store(kv_ptr + (block * 2 * rows + row) * width + offsets,
                     (append_normalized * append_weights).to(tl.bfloat16), mask=mask)

    @triton.jit(do_not_specialize=["tokens"])
    def _q_rotary_kernel(normalized_ptr, cos_ptr, sin_ptr, output_ptr,
                         tokens, heads: tl.constexpr,
                         head_dim: tl.constexpr, BLOCK: tl.constexpr):
        program = tl.program_id(0)
        token = program % tokens
        head = (program // tokens) % heads
        batch = program // (tokens * heads)
        offsets = tl.arange(0, BLOCK)
        mask = offsets < head_dim
        input_base = ((batch * tokens + token) * heads + head) * head_dim
        normalized = tl.load(normalized_ptr + input_base + offsets,
                             mask=mask, other=0.0)
        half = head_dim // 2
        rotated_offsets = tl.where(offsets < half, offsets + half, offsets - half)
        rotated = tl.load(normalized_ptr + input_base + rotated_offsets,
                          mask=mask, other=0.0)
        rotated = tl.where(offsets < half, -rotated, rotated)
        rope_base = (batch * tokens + token) * head_dim
        cos = tl.load(cos_ptr + rope_base + offsets, mask=mask, other=0.0)
        sin = tl.load(sin_ptr + rope_base + offsets, mask=mask, other=0.0)
        output_base = ((batch * heads + head) * tokens + token) * head_dim
        first = (normalized * cos).to(tl.bfloat16)
        second = (rotated * sin).to(tl.bfloat16)
        tl.store(output_ptr + output_base + offsets,
                 (first + second).to(tl.bfloat16), mask=mask)


    @triton.jit(do_not_specialize=["tokens"])
    def _kv_rotary_repeat_kernel(normalized_key_ptr, projected_value_ptr,
                                 cos_ptr, sin_ptr, key_ptr, value_ptr,
                                 tokens, kv_heads: tl.constexpr,
                                 groups: tl.constexpr, head_dim: tl.constexpr,
                                 BLOCK: tl.constexpr):
        program = tl.program_id(0)
        token = program % tokens
        output_head = (program // tokens) % (kv_heads * groups)
        batch = program // (tokens * kv_heads * groups)
        input_head = output_head // groups
        offsets = tl.arange(0, BLOCK)
        mask = offsets < head_dim
        row_base = (batch * tokens + token) * (kv_heads * head_dim)
        key_base = row_base + input_head * head_dim
        normalized = tl.load(normalized_key_ptr + key_base + offsets,
                             mask=mask, other=0.0)
        half = head_dim // 2
        rotated_offsets = tl.where(offsets < half, offsets + half, offsets - half)
        rotated = tl.load(normalized_key_ptr + key_base + rotated_offsets,
                          mask=mask, other=0.0)
        rotated = tl.where(offsets < half, -rotated, rotated)
        rope_base = (batch * tokens + token) * head_dim
        cos = tl.load(cos_ptr + rope_base + offsets, mask=mask, other=0.0)
        sin = tl.load(sin_ptr + rope_base + offsets, mask=mask, other=0.0)
        output_base = ((batch * kv_heads * groups + output_head) * tokens + token) * head_dim
        first = (normalized * cos).to(tl.bfloat16)
        second = (rotated * sin).to(tl.bfloat16)
        tl.store(key_ptr + output_base + offsets,
                 (first + second).to(tl.bfloat16), mask=mask)
        value_base = row_base + input_head * head_dim
        value = tl.load(projected_value_ptr + value_base + offsets,
                        mask=mask, other=0.0)
        tl.store(value_ptr + output_base + offsets, value, mask=mask)

_LEAN_OPS_EXTENSION = None
_SILU_MUL_EXTENSION = None

_LEAN_OPS_CPP = r"""
#include <torch/extension.h>

std::vector<torch::Tensor> c1_add_rms_norm_cuda(torch::Tensor update, torch::Tensor residual,
                                                torch::Tensor weight, double eps);
std::vector<torch::Tensor> c1_ew_offset_norm_cuda(torch::Tensor input, torch::Tensor update,
    torch::Tensor residual, torch::Tensor window, torch::Tensor weight, torch::Tensor kv_weight,
    double eps, double kv_eps, bool pending, bool kv_from_state, int reduction_threads);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("add_rms_norm", &c1_add_rms_norm_cuda, "C1 BF16 residual add + RMSNorm");
  module.def("ew_offset_norm", &c1_ew_offset_norm_cuda, "C1 offset norms and residual");
}
"""

# JETSPEC_C1_LEAN: the bf16 residual add (torch's: the fp32 sum of the two bf16 values,
# rounded to bf16) and the certified RMSNorm of that sum (c1_rms_norm_wide_kernel's
# arithmetic, line for line) in one launch, returning (normalized, sum).
_LEAN_OPS_CUDA = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>

__device__ __forceinline__ float c1_bf16_lane(unsigned int word, int high) {
  return __bfloat162float(__ushort_as_bfloat16(
      (unsigned short)(high ? (word >> 16) : (word & 0xFFFFu))));
}

__device__ __forceinline__ unsigned int c1_bf16_pack(__nv_bfloat16 low, __nv_bfloat16 high) {
  return (unsigned int)__bfloat16_as_ushort(low)
      | ((unsigned int)__bfloat16_as_ushort(high) << 16);
}

__global__ void c1_add_rms_norm_wide_kernel(
    const __nv_bfloat16* __restrict__ update,
    const __nv_bfloat16* __restrict__ residual,
    const __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16* __restrict__ state,
    __nv_bfloat16* __restrict__ output,
    int rows,
    float eps) {
  constexpr int WIDTH = 4096;
  constexpr int STEPS = WIDTH / 128;
  int row = blockIdx.x * blockDim.y + threadIdx.y;
  if (row >= rows) return;
  int thread = threadIdx.x;
  long long base = (long long)row * WIDTH + thread * 4;
  uint2 raw[STEPS];
  uint2 update_all[STEPS];
#pragma unroll
  for (int step = 0; step < STEPS; ++step) {
    update_all[step] = *reinterpret_cast<const uint2*>(update + base + step * 128);
    raw[step] = *reinterpret_cast<const uint2*>(residual + base + step * 128);
  }
#pragma unroll
  for (int step = 0; step < STEPS; ++step) {
    uint2 a = update_all[step];
    uint2 b = raw[step];
    __nv_bfloat16 sums[4];
#pragma unroll
    for (int item = 0; item < 4; ++item) {
      sums[item] = __float2bfloat16_rn(__fadd_rn(
          c1_bf16_lane(item < 2 ? b.x : b.y, item & 1),
          c1_bf16_lane(item < 2 ? a.x : a.y, item & 1)));
    }
    raw[step] = make_uint2(c1_bf16_pack(sums[0], sums[1]), c1_bf16_pack(sums[2], sums[3]));
    *reinterpret_cast<uint2*>(state + base + step * 128) = raw[step];
  }
  float partial[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
  for (int step = 0; step < STEPS; ++step) {
#pragma unroll
    for (int item = 0; item < 4; ++item) {
      float value = c1_bf16_lane(item < 2 ? raw[step].x : raw[step].y, item & 1);
      partial[item] = __fadd_rn(partial[item], __fmul_rn(value, value));
    }
  }
  float sum_sq = __fadd_rn(__fadd_rn(partial[0], partial[1]), partial[2]);
  sum_sq = __fadd_rn(sum_sq, partial[3]);
#pragma unroll
  for (int offset = 1; offset < 32; offset <<= 1) {
    sum_sq = __fadd_rn(
        sum_sq, __shfl_down_sync(0xffffffff, sum_sq, offset));
  }
  float reciprocal_rms = 0.0f;
  if (thread == 0) {
    float mean = __fmul_rn(sum_sq, 1.0f / float(WIDTH));
    reciprocal_rms = rsqrtf(__fadd_rn(mean, eps));
  }
  reciprocal_rms = __shfl_sync(0xffffffff, reciprocal_rms, 0);
  const __nv_bfloat16* scale = weight + thread * 4;
  uint2 scale_all[STEPS];
#pragma unroll
  for (int step = 0; step < STEPS; ++step) {
    scale_all[step] = *reinterpret_cast<const uint2*>(scale + step * 128);
  }
#pragma unroll
  for (int step = 0; step < STEPS; ++step) {
    uint2 scale_raw = scale_all[step];
    __nv_bfloat16 results[4];
#pragma unroll
    for (int item = 0; item < 4; ++item) {
      float value = c1_bf16_lane(item < 2 ? raw[step].x : raw[step].y, item & 1);
      __nv_bfloat16 normalized = __float2bfloat16_rn(value * reciprocal_rms);
      float scaled = __bfloat162float(normalized)
          * c1_bf16_lane(item < 2 ? scale_raw.x : scale_raw.y, item & 1);
      results[item] = __float2bfloat16_rn(scaled);
    }
    uint2 result = make_uint2(c1_bf16_pack(results[0], results[1]),
                              c1_bf16_pack(results[2], results[3]));
    *reinterpret_cast<uint2*>(output + base + step * 128) = result;
  }
}

template<int STEPS>
__global__ void c1_add_rms_norm_wide_kernel_t(
    const __nv_bfloat16* __restrict__ update,
    const __nv_bfloat16* __restrict__ residual,
    const __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16* __restrict__ state,
    __nv_bfloat16* __restrict__ output,
    int rows, float eps) {
  constexpr int WIDTH = STEPS * 128;
  int row = blockIdx.x * blockDim.y + threadIdx.y;
  if (row >= rows) return;
  int thread = threadIdx.x;
  long long base = (long long)row * WIDTH + thread * 4;
  float partial[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
  for (int step = 0; step < STEPS; ++step) {
    uint2 a = *reinterpret_cast<const uint2*>(update + base + step * 128);
    uint2 b = *reinterpret_cast<const uint2*>(residual + base + step * 128);
    __nv_bfloat16 sums[4];
#pragma unroll
    for (int item = 0; item < 4; ++item) {
      sums[item] = __float2bfloat16_rn(__fadd_rn(
          c1_bf16_lane(item < 2 ? b.x : b.y, item & 1),
          c1_bf16_lane(item < 2 ? a.x : a.y, item & 1)));
      float value = __bfloat162float(sums[item]);
      partial[item] = __fadd_rn(partial[item], __fmul_rn(value, value));
    }
    uint2 raw = make_uint2(c1_bf16_pack(sums[0], sums[1]),
                           c1_bf16_pack(sums[2], sums[3]));
    *reinterpret_cast<uint2*>(state + base + step * 128) = raw;
  }
  float sum_sq = __fadd_rn(__fadd_rn(partial[0], partial[1]), partial[2]);
  sum_sq = __fadd_rn(sum_sq, partial[3]);
#pragma unroll
  for (int offset = 1; offset < 32; offset <<= 1) {
    sum_sq = __fadd_rn(sum_sq, __shfl_down_sync(0xffffffff, sum_sq, offset));
  }
  float reciprocal_rms = 0.0f;
  if (thread == 0) {
    float mean = __fmul_rn(sum_sq, 1.0f / float(WIDTH));
    reciprocal_rms = rsqrtf(__fadd_rn(mean, eps));
  }
  reciprocal_rms = __shfl_sync(0xffffffff, reciprocal_rms, 0);
  const __nv_bfloat16* scale = weight + thread * 4;
#pragma unroll
  for (int step = 0; step < STEPS; ++step) {
    uint2 raw = *reinterpret_cast<const uint2*>(state + base + step * 128);
    uint2 scale_raw = *reinterpret_cast<const uint2*>(scale + step * 128);
    __nv_bfloat16 results[4];
#pragma unroll
    for (int item = 0; item < 4; ++item) {
      float value = c1_bf16_lane(item < 2 ? raw.x : raw.y, item & 1);
      __nv_bfloat16 normalized = __float2bfloat16_rn(value * reciprocal_rms);
      float scaled = __bfloat162float(normalized)
          * c1_bf16_lane(item < 2 ? scale_raw.x : scale_raw.y, item & 1);
      results[item] = __float2bfloat16_rn(scaled);
    }
    uint2 result = make_uint2(c1_bf16_pack(results[0], results[1]),
                              c1_bf16_pack(results[2], results[3]));
    *reinterpret_cast<uint2*>(output + base + step * 128) = result;
  }
}

std::vector<torch::Tensor> c1_add_rms_norm_cuda(torch::Tensor update, torch::Tensor residual,
                                                torch::Tensor weight, double eps) {
  TORCH_CHECK(update.is_cuda() && residual.is_cuda() && weight.is_cuda(), "CUDA tensors");
  TORCH_CHECK(update.scalar_type() == torch::kBFloat16 && residual.scalar_type() == torch::kBFloat16
              && weight.scalar_type() == torch::kBFloat16, "BF16 tensors");
  TORCH_CHECK(update.is_contiguous() && residual.is_contiguous() && weight.is_contiguous(),
              "contiguous tensors");
  int width = update.size(-1);
  TORCH_CHECK(update.sizes() == residual.sizes() && weight.numel() == width,
              "matching update, residual and weight widths");
  TORCH_CHECK(width == 4096 || width == 5120, "C1 add RMSNorm supports widths 4096 and 5120");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(update.data_ptr()) % 8 == 0
              && reinterpret_cast<uintptr_t>(residual.data_ptr()) % 8 == 0
              && reinterpret_cast<uintptr_t>(weight.data_ptr()) % 8 == 0, "8-byte aligned");
  int rows = update.numel() / width;
  auto state = torch::empty(update.sizes(), update.options());
  auto output = torch::empty(update.sizes(), update.options());
  c10::cuda::CUDAGuard guard(update.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  dim3 threads(32, 4);
  int blocks = (rows + 3) / 4;
  const auto* update_ptr = reinterpret_cast<const __nv_bfloat16*>(update.data_ptr());
  const auto* residual_ptr = reinterpret_cast<const __nv_bfloat16*>(residual.data_ptr());
  const auto* weight_ptr = reinterpret_cast<const __nv_bfloat16*>(weight.data_ptr());
  auto* state_ptr = reinterpret_cast<__nv_bfloat16*>(state.data_ptr());
  auto* output_ptr = reinterpret_cast<__nv_bfloat16*>(output.data_ptr());
  if (width == 4096) {
    c1_add_rms_norm_wide_kernel<<<blocks, threads, 0, stream>>>(
        update_ptr, residual_ptr, weight_ptr, state_ptr, output_ptr, rows, float(eps));
  } else {
    c1_add_rms_norm_wide_kernel_t<40><<<blocks, threads, 0, stream>>>(
        update_ptr, residual_ptr, weight_ptr, state_ptr, output_ptr, rows, float(eps));
  }
  TORCH_CHECK(cudaGetLastError() == cudaSuccess, "C1 add RMSNorm kernel launch failed");
  return {output, state};
}

// Keep the same four accumulators and ascending shuffle order as the module's
// contiguous float mean, with the packed BF16 row held in registers.
template<int THREADS>
__global__ void c1_ew_offset_norm_kernel(
    const __nv_bfloat16* input, const __nv_bfloat16* update,
    const __nv_bfloat16* residual, const __nv_bfloat16* window,
    const __nv_bfloat16* weight, const __nv_bfloat16* kv_weight,
    __nv_bfloat16* output, __nv_bfloat16* state, __nv_bfloat16* kv,
    int rows, float eps, float kv_eps, bool pending, bool kv_from_state) {
  constexpr int WIDTH = 5120, STEPS = WIDTH / (THREADS * 4);
  int row = blockIdx.x * 4 + threadIdx.y, group = blockIdx.y;
  int thread = threadIdx.x;
  long long base = (long long)row * WIDTH + thread * 4;
  bool fold = pending && (group == 0 || kv_from_state);
  const auto* source = group == 0 || kv_from_state ? input : window;
  const auto* weights = group == 0 ? weight : kv_weight + (group - 1) * WIDTH;
  uint2 raw[STEPS];
  float partial[4] = {0, 0, 0, 0};
#pragma unroll
  for (int step = 0; step < STEPS; ++step) {
    long long index = base + step * THREADS * 4;
    uint2 value = make_uint2(0, 0);
    if (row < rows) {
      value = *reinterpret_cast<const uint2*>((fold ? residual : source) + index);
      if (fold) {
        uint2 delta = *reinterpret_cast<const uint2*>(update + index);
        __nv_bfloat16 sums[4];
#pragma unroll
        for (int item = 0; item < 4; ++item) {
          sums[item] = __float2bfloat16_rn(__fadd_rn(
              c1_bf16_lane(item < 2 ? value.x : value.y, item & 1),
              c1_bf16_lane(item < 2 ? delta.x : delta.y, item & 1)));
        }
        value = make_uint2(c1_bf16_pack(sums[0], sums[1]), c1_bf16_pack(sums[2], sums[3]));
        if (group == 0) *reinterpret_cast<uint2*>(state + index) = value;
      }
    }
    raw[step] = value;
#pragma unroll
    for (int item = 0; item < 4; ++item) {
      float x = c1_bf16_lane(item < 2 ? value.x : value.y, item & 1);
      partial[item] = __fadd_rn(partial[item], __fmul_rn(x, x));
    }
  }
  float sum = __fadd_rn(__fadd_rn(__fadd_rn(partial[0], partial[1]), partial[2]), partial[3]);
  __shared__ float upper[4][32], scales[4];
  if constexpr (THREADS == 64) {
    if (thread >= 32) upper[threadIdx.y][thread - 32] = sum;
    __syncthreads();
    if (thread < 32) sum = __fadd_rn(sum, upper[threadIdx.y][thread]);
  }
#pragma unroll
  for (int offset = 1; offset < 32; offset <<= 1)
    sum = __fadd_rn(sum, __shfl_down_sync(0xffffffff, sum, offset));
  float scale = 0;
  if (thread == 0) scale = rsqrtf(__fadd_rn(__fmul_rn(sum, 1.0f / WIDTH), group == 0 ? eps : kv_eps));
  if constexpr (THREADS == 64) {
    if (thread == 0) scales[threadIdx.y] = scale;
    __syncthreads();
    scale = scales[threadIdx.y];
  } else {
    scale = __shfl_sync(0xffffffff, scale, 0);
  }
  if (row >= rows) return;
  auto* target = group == 0 ? output : kv + (long long)(group - 1) * rows * WIDTH;
#pragma unroll
  for (int step = 0; step < STEPS; ++step) {
    uint2 w = *reinterpret_cast<const uint2*>(weights + thread * 4 + step * THREADS * 4);
    __nv_bfloat16 result[4];
#pragma unroll
    for (int item = 0; item < 4; ++item) {
      float x = c1_bf16_lane(item < 2 ? raw[step].x : raw[step].y, item & 1);
      float offset_weight = __fadd_rn(1.0f, c1_bf16_lane(item < 2 ? w.x : w.y, item & 1));
      result[item] = __float2bfloat16_rn(__fmul_rn(__fmul_rn(x, scale), offset_weight));
    }
    *reinterpret_cast<uint2*>(target + base + step * THREADS * 4) =
        make_uint2(c1_bf16_pack(result[0], result[1]), c1_bf16_pack(result[2], result[3]));
  }
}

std::vector<torch::Tensor> c1_ew_offset_norm_cuda(torch::Tensor input, torch::Tensor update,
    torch::Tensor residual, torch::Tensor window, torch::Tensor weight, torch::Tensor kv_weight,
    double eps, double kv_eps, bool pending, bool kv_from_state, int reduction_threads) {
  for (const auto& tensor : {input, update, residual, window, weight, kv_weight}) {
    TORCH_CHECK(tensor.is_cuda() && tensor.scalar_type() == torch::kBFloat16
        && tensor.is_contiguous() && tensor.device() == input.device(), "contiguous BF16 CUDA tensors");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(tensor.data_ptr()) % 8 == 0, "8-byte aligned");
  }
  TORCH_CHECK(input.dim() == 2 && input.size(1) == 5120 && input.sizes() == update.sizes()
      && input.sizes() == residual.sizes() && input.sizes() == window.sizes()
      && weight.numel() == 5120 && kv_weight.dim() == 2 && kv_weight.size(1) == 5120,
      "served C1 offset norm shapes");
  TORCH_CHECK(reduction_threads == 32 || reduction_threads == 64, "served mean reduction threads");
  int rows = input.size(0);
  auto output = torch::empty_like(input);
  auto state = pending ? torch::empty_like(input) : input;
  auto kv = torch::empty({kv_weight.size(0), rows, 5120}, input.options());
  c10::cuda::CUDAGuard guard(input.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  dim3 grid((rows + 3) / 4, kv_weight.size(0) + 1), threads(reduction_threads, 4);
#define C1_EW_ARGS reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()), \
    reinterpret_cast<const __nv_bfloat16*>(update.data_ptr()), \
    reinterpret_cast<const __nv_bfloat16*>(residual.data_ptr()), \
    reinterpret_cast<const __nv_bfloat16*>(window.data_ptr()), \
    reinterpret_cast<const __nv_bfloat16*>(weight.data_ptr()), \
    reinterpret_cast<const __nv_bfloat16*>(kv_weight.data_ptr()), \
    reinterpret_cast<__nv_bfloat16*>(output.data_ptr()), \
    reinterpret_cast<__nv_bfloat16*>(state.data_ptr()), \
    reinterpret_cast<__nv_bfloat16*>(kv.data_ptr()), rows, float(eps), float(kv_eps), pending, kv_from_state
  if (reduction_threads == 64) c1_ew_offset_norm_kernel<64><<<grid, threads, 0, stream>>>(C1_EW_ARGS);
  else c1_ew_offset_norm_kernel<32><<<grid, threads, 0, stream>>>(C1_EW_ARGS);
#undef C1_EW_ARGS
  TORCH_CHECK(cudaGetLastError() == cudaSuccess, "C1 offset norm kernel launch failed");
  return {output, state, kv};
}
"""

_SILU_MUL_CPP = r"""
#include <torch/extension.h>

torch::Tensor c1_silu_mul_cuda(torch::Tensor gate, torch::Tensor up);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("silu_mul", &c1_silu_mul_cuda, "silu(gate) * up, torch's two roundings");
}
"""

# JETSPEC_C1_LEAN: F.silu(gate) * up in one launch with torch's arithmetic: silu is
# x / (1 + exp(-x)) in fp32 rounded to bf16 (ActivationSiluKernel), then the fp32
# product of the two bf16 values rounded to bf16.  Built with nvcc's default flags,
# as torch's own kernels are.
_SILU_MUL_CUDA = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>

__global__ void c1_silu_mul_kernel(const __nv_bfloat16* gate, const __nv_bfloat16* up,
                                   __nv_bfloat16* output, long long count) {
  long long index = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count) return;
  float x = __bfloat162float(gate[index]);
  __nv_bfloat16 activated = __float2bfloat16(x / (1.0f + ::expf(-x)));
  output[index] = __float2bfloat16(__bfloat162float(activated) * __bfloat162float(up[index]));
}

__global__ void c1_silu_mul_strided_kernel(const __nv_bfloat16* gate, const __nv_bfloat16* up,
                                           __nv_bfloat16* output, long long count,
                                           long long width, long long gate_row_stride,
                                           long long up_row_stride) {
  long long index = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count) return;
  long long row = index / width;
  long long col = index % width;
  float x = __bfloat162float(gate[row * gate_row_stride + col]);
  __nv_bfloat16 activated = __float2bfloat16(x / (1.0f + ::expf(-x)));
  output[index] = __float2bfloat16(__bfloat162float(activated) *
                                   __bfloat162float(up[row * up_row_stride + col]));
}

torch::Tensor c1_silu_mul_cuda(torch::Tensor gate, torch::Tensor up) {
  TORCH_CHECK(gate.is_cuda() && up.is_cuda() && gate.scalar_type() == torch::kBFloat16
              && up.scalar_type() == torch::kBFloat16 && gate.sizes() == up.sizes()
              && gate.device() == up.device(), "matching BF16 CUDA tensors");
  bool contiguous = gate.is_contiguous() && up.is_contiguous();
  if (!contiguous) {
    TORCH_CHECK(gate.dim() >= 2 && up.dim() == gate.dim()
                && gate.stride(-1) == 1 && up.stride(-1) == 1,
                "silu mul requires contiguous rows");
    for (int64_t dim = 0; dim < gate.dim() - 2; ++dim) {
      TORCH_CHECK(gate.stride(dim) == gate.size(dim + 1) * gate.stride(dim + 1)
                  && up.stride(dim) == up.size(dim + 1) * up.stride(dim + 1),
                  "silu mul requires contiguous rows");
    }
  }
  auto output = torch::empty(gate.sizes(), gate.options());
  long long count = gate.numel();
  c10::cuda::CUDAGuard guard(gate.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  int threads = 256;
  long long blocks = (count + threads - 1) / threads;
  if (contiguous) {
    c1_silu_mul_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(gate.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(up.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr()), count);
  } else {
    c1_silu_mul_strided_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(gate.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(up.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr()), count,
        gate.size(-1), gate.stride(-2), up.stride(-2));
  }
  TORCH_CHECK(cudaGetLastError() == cudaSuccess, "silu mul kernel launch failed");
  return output;
}
"""


def _lean_ops_extension():
    global _LEAN_OPS_EXTENSION
    if _LEAN_OPS_EXTENSION is None:
        from torch.utils.cpp_extension import load_inline
        _LEAN_OPS_EXTENSION = load_inline(
            name="jetspec_c1_lean_ops_v5",
            cpp_sources=_LEAN_OPS_CPP,
            cuda_sources=_LEAN_OPS_CUDA,
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3", "--fmad=false"],
            with_cuda=True,
            verbose=False,
        )
    return _LEAN_OPS_EXTENSION


def _silu_mul_extension():
    global _SILU_MUL_EXTENSION
    if _SILU_MUL_EXTENSION is None:
        from torch.utils.cpp_extension import load_inline
        _SILU_MUL_EXTENSION = load_inline(
            name="jetspec_c1_silu_mul_v2",
            cpp_sources=_SILU_MUL_CPP,
            cuda_sources=_SILU_MUL_CUDA,
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3"],
            with_cuda=True,
            verbose=False,
        )
    return _SILU_MUL_EXTENSION


_COMPILED_LAUNCHES = {}


def _has_launch_hooks(runtime):
    return any(hook is not None and bool(getattr(hook, "calls", (hook,)))
               for hook in (runtime.launch_enter_hook, runtime.launch_exit_hook))


def _launch_key(kernel, device, args, options):
    runtime_indices = _RUNTIME_COUNT_INDEX.get(
        kernel, getattr(kernel, "_jetspec_runtime_count_indices", ()))
    return (kernel, device, tuple(sorted(options.items())), tuple(
        (arg.dtype, arg.data_ptr() % 16 == 0) if isinstance(arg, torch.Tensor)
        else None if index in runtime_indices else arg
        for index, arg in enumerate(args)))


def launch_cached(kernel, grid, args, **options):
    """``kernel[grid](*args, **options)``, through the compiled kernel's own launcher
    once this specialization has been compiled: the same binary on the same
    arguments, without Triton's per-call argument binding.  The kernels this is
    used for take pointers, constexprs and one unspecialized row count. The
    key retains tensor dtype/alignment and constexprs, but omits that count.
    """
    device = args[0].device.index
    key = _launch_key(kernel, device, args, options)
    entry = _COMPILED_LAUNCHES.get(key)
    if entry is None:
        compiled = kernel[grid](*args, **options)
        compiled._init_handles()
        # the launcher, function and metadata CompiledKernel.__getitem__'s runner uses
        _COMPILED_LAUNCHES[key] = (compiled, compiled.run, compiled.function,
                                   compiled.packed_metadata)
        return
    compiled, run, function, packed = entry
    grid = tuple(grid) + (1,) * (3 - len(grid))
    runtime = triton.knobs.runtime
    if _has_launch_hooks(runtime):
        compiled[grid](*args)
        return
    # the runner's call with no launch hooks installed (launch_metadata is then None)
    run(grid[0], grid[1], grid[2], torch._C._cuda_getCurrentRawStream(device), function,
        packed, None, None, None, *args)


def _lean():
    from jetspec.engines.jetspec_plus_plus.profiles import kernel_enabled
    return kernel_enabled('c1_lean', '1')


def efficient_attention(query, key, value, mask, scale):
    """``F.scaled_dot_product_attention(query, key, value, attn_mask=mask, scale=scale)``
    for a float mask on CUDA, where SDPA selects the memory-efficient kernel: the same
    kernel on the same operands, with the mask laid out in 16-aligned rows (as SDPA's
    preprocessing pads it) and broadcast over the heads, without the per-call backend
    selection."""
    rows, keys = mask.shape[-2], mask.shape[-1]
    if mask.stride(-1) != 1 or any(stride % 16 for stride in mask.stride()[:-1]):
        aligned = mask.new_zeros((*mask.shape[:-1], (keys + 15) // 16 * 16))
        aligned[..., :keys].copy_(mask)
        mask = aligned[..., :keys]
    bias = mask.expand(query.shape[0], query.shape[1], rows, keys)
    return torch.ops.aten._scaled_dot_product_efficient_attention(
        query, key, value, bias, False, 0.0, False, scale=scale)[0]


def lean_q_postprocess(projected, weight, cos, sin, eps, num_heads, head_dim, norm):
    """``fused_q_postprocess`` on its CUDA path for contiguous ``cos``/``sin`` and a
    norm extension whose output is fresh and contiguous; the rotary goes through
    ``launch_cached``."""
    batch, tokens, _ = projected.shape
    normalized = norm(projected.view(batch, tokens, num_heads, head_dim), weight, eps)
    output = projected.new_empty(batch, num_heads, tokens, head_dim)
    launch_cached(_q_rotary_kernel, (batch * tokens * num_heads,),
                  (normalized, cos, sin, output, tokens, num_heads, head_dim,
                   triton.next_power_of_2(head_dim)),
                  num_warps=4, enable_fp_fusion=False)
    return output


def _reference_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float,
                        norm_offset=0, norm_round=None) -> torch.Tensor:
    input_dtype = x.dtype
    values = x.to(torch.float32)
    values = values * torch.rsqrt(values.pow(2).mean(-1, keepdim=True) + eps)
    rounding = norm_round or ("after_weight" if norm_offset else "before_weight")
    weights = weight if norm_offset == 0 else norm_offset + weight.float()
    if rounding == "after_weight":
        return (values * weights.float()).to(input_dtype)
    return (weights if norm_offset == 0 else weights.to(input_dtype)) * values.to(input_dtype)


def finish_streams(streams, residual: torch.Tensor, update: torch.Tensor,
                   real: int | None = None) -> torch.Tensor:
    """Write the five C1 streams with the final bf16 residual add in one launch."""
    width = residual.shape[-1]
    rows = residual.numel() // width
    if real is not None and residual.ndim != 2:
        raise ValueError("C1 refresh row trimming requires flattened token rows")
    write_rows = rows if real is None else real
    shape = (*residual.shape[:-1], 5 * width) if real is None else (real, 5 * width)
    stream_shape = residual.shape if real is None else (real, width)
    if (triton is None or residual.device.type != "cuda" or
            residual.dtype != torch.bfloat16 or len(streams) != 4 or
            not all(stream.shape == stream_shape and
                    (stream.is_contiguous() if real is None
                     else stream.stride() == (width, 1))
                    for stream in streams) or
            not residual.is_contiguous() or not update.is_contiguous()):
        if real is None:
            return torch.cat([*streams, residual + update], dim=-1)
        return torch.cat([*(stream[:real] for stream in streams),
                          (residual + update)[:real]], dim=-1)
    out = residual.new_empty(shape)
    launch_cached(_finish_streams_kernel, (write_rows, 5),
                  (*streams, residual, update, out, write_rows, width,
                   triton.next_power_of_2(width)), num_warps=8)
    return out


def ew_pre_norm(u: torch.Tensor, window: torch.Tensor, input_weight: torch.Tensor,
                kv_weights: torch.Tensor, input_eps: float, kv_eps: float,
                pending=None, kv_from_state=False, append=None, norm_offset=0, ulp=False,
                norm_round=None):
    """One launch for the next input norm and independent K/V source norms."""
    rows, width = u.shape
    rounding = norm_round or ("after_weight" if norm_offset else "before_weight")
    specialized = rounding == ("after_weight" if norm_offset else "before_weight")
    if u.device.type != "cuda" or u.dtype != torch.bfloat16 or triton is None or not specialized:
        state = u if pending is None else pending[1] + pending[0]
        kv_input = state if kv_from_state else window
        sources = (torch.stack([_reference_rms_norm(kv_input, weight, kv_eps, norm_offset, rounding)
                                for weight in kv_weights]) if len(kv_weights) else
                   u.new_empty((0, rows, width)))
        if append is not None:
            dirty, streams, first = append
            taps = dirty.reshape(rows, -1, width)
            appended = torch.stack([
                _reference_rms_norm(taps[:, int(streams[first + block])], weight, kv_eps,
                                    norm_offset, rounding)
                for block, weight in enumerate(kv_weights)])
            sources = torch.cat((appended, sources), dim=1)
        return _reference_rms_norm(state, input_weight, input_eps, norm_offset, rounding), state, sources
    if norm_offset == 1 and width == 5120 and append is None and not ulp:
        update, residual = (u, u) if pending is None else pending
        return _lean_ops_extension().ew_offset_norm(
            u, update, residual, window, input_weight, kv_weights, input_eps, kv_eps,
            pending is not None, kv_from_state, _module_reduce_threads(width, rows))
    state = u if pending is None else torch.empty_like(u)
    normalized = torch.empty_like(u)
    sources = u.new_empty((kv_weights.shape[0], rows * (2 if append is not None else 1), width))
    update, residual = (u, u) if pending is None else pending
    dirty, streams, first = append if append is not None else (u, kv_weights, 0)
    launch_cached(_ew_pre_norm_kernel, ((kv_weights.shape[0] *
                                        (2 if append is not None else 1) + 1) * rows,),
                  (u, update, residual, window, input_weight, kv_weights,
                   normalized, state, sources, dirty, streams, first,
                   dirty.shape[-1] // width, kv_weights.shape[0], append is not None,
                   rows, width, input_eps, kv_eps,
                   pending is not None, kv_from_state, triton.next_power_of_2(width), norm_offset,
                   _module_reduce_threads(width, rows), ulp),
                  num_warps=8, enable_fp_fusion=False)
    return normalized, state, sources


def _module_reduce_threads(width, rows):
    """PyTorch float ReduceConfig for the served contiguous last-dimension means."""
    dim0 = width // 4 if width > 128 else width
    x = min(1 << (dim0.bit_length() - 1), 512, 32)
    y = min(1 << (rows.bit_length() - 1), 512 // x)
    return min(1 << (dim0.bit_length() - 1), 512 // y)


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    from jetspec.engines.jetspec_plus_plus.profiles import kernel_enabled
    if (x.device.type != "cuda" or x.dtype == torch.float32
            or not (x.shape[-1] == 128 or x.shape[-1] % 128 == 0)):
        return _reference_rms_norm(x, weight, eps)
    if kernel_enabled('c1_norm', 'wide'):
        return _wide_norm_extension().rms_norm(x, weight, eps)
    return _norm_extension().rms_norm(x, weight, eps)


def _reference_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
    rotary = cos.shape[-1]
    half = rotary // 2
    part = x[..., :rotary]
    rotated = torch.cat((-part[..., half:], part[..., :half]), dim=-1)
    result = part * cos + rotated * sin
    return result if rotary == x.shape[-1] else torch.cat((result, x[..., rotary:]), dim=-1)


def fused_q_postprocess(projected: torch.Tensor, weight: torch.Tensor,
                        cos: torch.Tensor, sin: torch.Tensor, eps: float,
                        num_heads: int, head_dim: int) -> torch.Tensor:
    batch, tokens, _ = projected.shape
    if projected.device.type != "cuda" or projected.dtype == torch.float32 or triton is None:
        q = projected.view(batch, tokens, num_heads, head_dim)
        q = _reference_rms_norm(q, weight, eps).transpose(1, 2)
        return _reference_rotary(q, cos, sin)
    normalized = rms_norm(
        projected.view(batch, tokens, num_heads, head_dim), weight, eps
    ).contiguous()
    cos, sin = cos.contiguous(), sin.contiguous()
    output = projected.new_empty(batch, num_heads, tokens, head_dim)
    block = triton.next_power_of_2(head_dim)
    _q_rotary_kernel[(batch * tokens * num_heads,)](
        normalized, cos, sin, output, tokens=tokens, heads=num_heads,
        head_dim=head_dim, BLOCK=block, num_warps=4, enable_fp_fusion=False,
    )
    return output


def fused_kv_postprocess(projected_key: torch.Tensor, projected_value: torch.Tensor,
                         weight: torch.Tensor,
                         cos: torch.Tensor, sin: torch.Tensor, eps: float,
                         kv_heads: int, groups: int, head_dim: int):
    batch, tokens, _ = projected_key.shape
    if (projected_key.device.type != "cuda" or projected_key.dtype == torch.float32
            or triton is None):
        k = _reference_rms_norm(
            projected_key.view(batch, tokens, kv_heads, head_dim), weight, eps
        ).transpose(1, 2)
        v = projected_value.view(batch, tokens, kv_heads, head_dim).transpose(1, 2)
        k = _reference_rotary(k, cos, sin)
        if groups != 1:
            k = k.repeat_interleave(groups, dim=1)
            v = v.repeat_interleave(groups, dim=1)
        return k, v
    normalized_key = rms_norm(
        projected_key.view(batch, tokens, kv_heads, head_dim), weight, eps
    ).contiguous()
    projected_value = projected_value.contiguous()
    cos, sin = cos.contiguous(), sin.contiguous()
    shape = (batch, kv_heads * groups, tokens, head_dim)
    key = projected_key.new_empty(shape)
    value = projected_value.new_empty(shape)
    block = triton.next_power_of_2(head_dim)
    _kv_rotary_repeat_kernel[(batch * tokens * kv_heads * groups,)](
        normalized_key, projected_value, cos, sin, key, value, tokens=tokens,
        kv_heads=kv_heads, groups=groups, head_dim=head_dim, BLOCK=block,
        num_warps=4, enable_fp_fusion=False,
    )
    return key, value


def fast_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    if x.device.type != "cuda" or x.dtype == torch.float32 or triton is None:
        return _reference_rms_norm(x, weight, eps)
    shape = x.shape
    source = x.contiguous().reshape(-1, shape[-1])
    output = torch.empty_like(source)
    block = triton.next_power_of_2(shape[-1])
    if _lean():
        launch_cached(_fast_rms_norm_kernel, (source.shape[0],),
                      (source, weight, output, source.shape[0], shape[-1], eps, block),
                      num_warps=8 if block >= 4096 else 4)
    else:
        _fast_rms_norm_kernel[(source.shape[0],)](
            source, weight, output, rows=source.shape[0], width=shape[-1], eps=eps,
            BLOCK=block, num_warps=8 if block >= 4096 else 4,
        )
    return output.reshape(shape)


if triton is not None:

    @triton.jit
    def _jpass_q_norm_rotary_kernel(source, weight, cos, sin, output,
                                    heads: tl.constexpr, head_dim: tl.constexpr,
                                    source_row_stride: tl.constexpr,
                                    cos_row_stride: tl.constexpr,
                                    eps: tl.constexpr, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        head = tl.program_id(1)
        col = tl.arange(0, BLOCK)
        mask = col < head_dim
        value = tl.load(source + row * source_row_stride + head * head_dim + col,
                        mask=mask, other=0.0).to(tl.float32)
        reciprocal_rms = tl.rsqrt(tl.sum(value * value, axis=0) / head_dim + eps)
        normalized = (value * reciprocal_rms).to(tl.bfloat16)
        weights = tl.load(weight + col, mask=mask, other=0.0)
        weighted = (normalized * weights).to(tl.bfloat16)
        half = head_dim // 2
        partner = tl.where(col < half, col + half, col - half)
        rotated = tl.gather(weighted, partner, 0)
        rotated = tl.where(col < half, -rotated, rotated)
        rope = row * cos_row_stride + col
        cosine = tl.load(cos + rope, mask=mask, other=0.0)
        sine = tl.load(sin + rope, mask=mask, other=0.0)
        first = (weighted * cosine).to(tl.bfloat16)
        second = (rotated * sine).to(tl.bfloat16)
        tl.store(output + (row * heads + head) * head_dim + col,
                 (first + second).to(tl.bfloat16), mask=mask)

    @triton.jit
    def _jpass_kv_norm_rotary_cache_kernel(key, value, weight, cos, sin,
                                           key_out, value_out,
                                           heads: tl.constexpr, head_dim: tl.constexpr,
                                           key_row_stride: tl.constexpr,
                                           value_row_stride: tl.constexpr,
                                           cos_row_stride: tl.constexpr,
                                           out_row_stride: tl.constexpr,
                                           out_head_stride,
                                           eps: tl.constexpr, BLOCK: tl.constexpr):
        # out_head_stride is a runtime value: the decoder prefix writes a (tokens, heads, dim)
        # view whose head stride is tokens * dim, and a constexpr compiled one kernel per prompt
        # length inside the measured read (14 Oct, B200 fast6 and the A100 default recipe).
        row = tl.program_id(0)
        head = tl.program_id(1)
        col = tl.arange(0, BLOCK)
        mask = col < head_dim
        key_offset = row * key_row_stride + head * head_dim + col
        source = tl.load(key + key_offset, mask=mask, other=0.0).to(tl.float32)
        reciprocal_rms = tl.rsqrt(tl.sum(source * source, axis=0) / head_dim + eps)
        normalized = (source * reciprocal_rms).to(tl.bfloat16)
        weights = tl.load(weight + col, mask=mask, other=0.0)
        weighted = (normalized * weights).to(tl.bfloat16)
        half = head_dim // 2
        partner = tl.where(col < half, col + half, col - half)
        rotated = tl.gather(weighted, partner, 0)
        rotated = tl.where(col < half, -rotated, rotated)
        rope = row * cos_row_stride + col
        cosine = tl.load(cos + rope, mask=mask, other=0.0)
        sine = tl.load(sin + rope, mask=mask, other=0.0)
        first = (weighted * cosine).to(tl.bfloat16)
        second = (rotated * sine).to(tl.bfloat16)
        out_offset = row * out_row_stride + head * out_head_stride + col
        tl.store(key_out + out_offset, (first + second).to(tl.bfloat16), mask=mask)
        copied = tl.load(value + row * value_row_stride + head * head_dim + col,
                         mask=mask, other=0.0)
        tl.store(value_out + out_offset, copied, mask=mask)


def jpass_q_norm_rotary(projected, weight, cos, sin, eps, heads, head_dim):
    rows = projected.shape[0]
    output = torch.empty((rows, heads, head_dim), device=projected.device,
                         dtype=projected.dtype)
    _jpass_q_norm_rotary_kernel[(rows, heads)](
        projected, weight, cos, sin, output, heads, head_dim,
        projected.stride(0), cos.stride(0), eps, triton.next_power_of_2(head_dim),
        num_warps=4, enable_fp_fusion=False)
    return output


def jpass_kv_norm_rotary_cache(projected_key, projected_value, weight, cos, sin,
                                key_out, value_out, eps, heads, head_dim):
    rows = projected_key.shape[0]
    _jpass_kv_norm_rotary_cache_kernel[(rows, heads)](
        projected_key, projected_value, weight, cos, sin, key_out, value_out,
        heads, head_dim, projected_key.stride(0), projected_value.stride(0),
        cos.stride(0), key_out.stride(0), key_out.stride(1), eps,
        triton.next_power_of_2(head_dim), num_warps=4, enable_fp_fusion=False)


def fast_q_postprocess(projected: torch.Tensor, weight: torch.Tensor,
                       cos: torch.Tensor, sin: torch.Tensor, eps: float,
                       num_heads: int, head_dim: int) -> torch.Tensor:
    batch, tokens, _ = projected.shape
    if projected.device.type != "cuda" or projected.dtype == torch.float32 or triton is None:
        q = projected.view(batch, tokens, num_heads, head_dim)
        q = _reference_rms_norm(q, weight, eps).transpose(1, 2)
        return _reference_rotary(q, cos, sin)
    normalized = fast_rms_norm(
        projected.view(batch, tokens, num_heads, head_dim), weight, eps
    ).contiguous()
    cos, sin = cos.contiguous(), sin.contiguous()
    output = projected.new_empty(batch, num_heads, tokens, head_dim)
    block = triton.next_power_of_2(head_dim)
    if _lean():
        launch_cached(_q_rotary_kernel, (batch * tokens * num_heads,),
                      (normalized, cos, sin, output, tokens, num_heads, head_dim, block),
                      num_warps=4, enable_fp_fusion=False)
    else:
        _q_rotary_kernel[(batch * tokens * num_heads,)](
            normalized, cos, sin, output, tokens=tokens, heads=num_heads,
            head_dim=head_dim, BLOCK=block, num_warps=4, enable_fp_fusion=False,
        )
    return output


def fast_kv_postprocess(projected_key: torch.Tensor, projected_value: torch.Tensor,
                        weight: torch.Tensor,
                        cos: torch.Tensor, sin: torch.Tensor, eps: float,
                        kv_heads: int, groups: int, head_dim: int):
    batch, tokens, _ = projected_key.shape
    if (projected_key.device.type != "cuda" or projected_key.dtype == torch.float32
            or triton is None):
        k = _reference_rms_norm(
            projected_key.view(batch, tokens, kv_heads, head_dim), weight, eps
        ).transpose(1, 2)
        v = projected_value.view(batch, tokens, kv_heads, head_dim).transpose(1, 2)
        k = _reference_rotary(k, cos, sin)
        if groups != 1:
            k = k.repeat_interleave(groups, dim=1)
            v = v.repeat_interleave(groups, dim=1)
        return k, v
    normalized_key = fast_rms_norm(
        projected_key.view(batch, tokens, kv_heads, head_dim), weight, eps
    ).contiguous()
    projected_value = projected_value.contiguous()
    cos, sin = cos.contiguous(), sin.contiguous()
    shape = (batch, kv_heads * groups, tokens, head_dim)
    key = projected_key.new_empty(shape)
    value = projected_value.new_empty(shape)
    block = triton.next_power_of_2(head_dim)
    if _lean():
        launch_cached(_kv_rotary_repeat_kernel, (batch * tokens * kv_heads * groups,),
                      (normalized_key, projected_value, cos, sin, key, value, tokens,
                       kv_heads, groups, head_dim, block),
                      num_warps=4, enable_fp_fusion=False)
    else:
        _kv_rotary_repeat_kernel[(batch * tokens * kv_heads * groups,)](
            normalized_key, projected_value, cos, sin, key, value, tokens=tokens,
            kv_heads=kv_heads, groups=groups, head_dim=head_dim, BLOCK=block,
            num_warps=4, enable_fp_fusion=False,
        )
    return key, value


def fast_add_rms_norm(update: torch.Tensor, residual: torch.Tensor,
                      weight: torch.Tensor, eps: float):
    if (update.device.type != "cuda" or update.dtype == torch.float32
            or triton is None):
        state = residual + update
        return _reference_rms_norm(state, weight, eps), state
    shape = update.shape
    update_flat = update.contiguous().reshape(-1, shape[-1])
    residual_flat = residual.contiguous().reshape(-1, shape[-1])
    normalized = torch.empty_like(update_flat)
    state = torch.empty_like(residual_flat)
    block = triton.next_power_of_2(shape[-1])
    if _lean():
        launch_cached(_fast_add_rms_norm_kernel, (update_flat.shape[0],),
                      (update_flat, residual_flat, weight, normalized, state,
                       update_flat.shape[0], shape[-1], eps, block),
                      num_warps=8 if block >= 4096 else 4)
    else:
        _fast_add_rms_norm_kernel[(update_flat.shape[0],)](
            update_flat, residual_flat, weight, normalized, state,
            rows=update_flat.shape[0], width=shape[-1], eps=eps, BLOCK=block,
            num_warps=8 if block >= 4096 else 4,
        )
    return normalized.reshape(shape), state.reshape(shape)


if triton is not None:

    @triton.jit(do_not_specialize=["tokens"])
    def _grouped_rms_norm_kernel(input_ptr, weight_ptr, stream_ptr, output_ptr,
                                 tokens, streams: tl.constexpr,
                                 width: tl.constexpr, eps: tl.constexpr,
                                 BLOCK: tl.constexpr):
        program = tl.program_id(0)
        block = program // tokens
        token = program % tokens
        stream = tl.load(stream_ptr + block)
        offsets = tl.arange(0, BLOCK)
        mask = offsets < width
        input_base = token * streams * width + stream * width
        values = tl.load(input_ptr + input_base + offsets, mask=mask, other=0.0).to(
            tl.float32
        )
        reciprocal_rms = tl.rsqrt(tl.sum(values * values, axis=0) / width + eps)
        normalized = (values * reciprocal_rms).to(tl.bfloat16)
        weights = tl.load(weight_ptr + block * width + offsets, mask=mask, other=0.0)
        output_base = (block * tokens + token) * width
        tl.store(output_ptr + output_base + offsets, normalized * weights, mask=mask)


    @triton.jit(do_not_specialize=["tokens", "block_stride"])
    def _batched_kv_postprocess_kernel(key_ptr, value_ptr, weight_ptr,
                                       cos_ptr, sin_ptr, key_output_ptr,
                                       value_output_ptr,
                                       tokens,
                                       kv_heads: tl.constexpr,
                                       groups: tl.constexpr,
                                       head_dim: tl.constexpr,
                                       block_stride,
                                       token_stride: tl.constexpr,
                                       eps: tl.constexpr,
                                       BLOCK: tl.constexpr):
        program = tl.program_id(0)
        output_heads = kv_heads * groups
        token = program % tokens
        output_head = (program // tokens) % output_heads
        block = program // (tokens * output_heads)
        input_head = output_head // groups
        offsets = tl.arange(0, BLOCK)
        mask = offsets < head_dim
        input_base = (
            block * block_stride + token * token_stride + input_head * head_dim
        )
        values = tl.load(key_ptr + input_base + offsets, mask=mask, other=0.0).to(
            tl.float32
        )
        reciprocal_rms = tl.rsqrt(
            tl.sum(values * values, axis=0) / head_dim + eps
        )
        weights = tl.load(
            weight_ptr + block * head_dim + offsets, mask=mask, other=0.0
        )
        normalized = (values * reciprocal_rms).to(tl.bfloat16)
        normalized = (normalized * weights).to(tl.bfloat16)
        half = head_dim // 2
        rotated_offsets = tl.where(offsets < half, offsets + half, offsets - half)
        rotated_values = tl.load(
            key_ptr + input_base + rotated_offsets, mask=mask, other=0.0
        ).to(tl.float32)
        rotated_weights = tl.load(
            weight_ptr + block * head_dim + rotated_offsets, mask=mask, other=0.0
        )
        rotated = (rotated_values * reciprocal_rms).to(tl.bfloat16)
        rotated = (rotated * rotated_weights).to(tl.bfloat16)
        rotated = tl.where(offsets < half, -rotated, rotated)
        rope_base = token * head_dim
        cos = tl.load(cos_ptr + rope_base + offsets, mask=mask, other=0.0)
        sin = tl.load(sin_ptr + rope_base + offsets, mask=mask, other=0.0)
        output_base = (
            ((block * output_heads + output_head) * tokens + token) * head_dim
        )
        first = (normalized * cos).to(tl.bfloat16)
        second = (rotated * sin).to(tl.bfloat16)
        tl.store(
            key_output_ptr + output_base + offsets,
            (first + second).to(tl.bfloat16),
            mask=mask,
        )
        projected_value = tl.load(
            value_ptr + input_base + offsets, mask=mask, other=0.0
        )
        tl.store(
            value_output_ptr + output_base + offsets, projected_value, mask=mask
        )


    @triton.jit
    def _folded_append_cache_kernel(key_ptr, value_ptr, weight_ptr,
                                    cos_ptr, sin_ptr, slots_ptr,
                                    key_cache_ptr, value_cache_ptr,
                                    tokens: tl.constexpr, kv_heads: tl.constexpr,
                                    groups: tl.constexpr, head_dim: tl.constexpr,
                                    input_block_stride: tl.constexpr,
                                    input_token_stride: tl.constexpr,
                                    cache_block_stride: tl.constexpr,
                                    cache_head_stride: tl.constexpr,
                                    cache_token_stride: tl.constexpr,
                                    eps: tl.constexpr, BLOCK: tl.constexpr):
        program = tl.program_id(0)
        output_heads = kv_heads * groups
        token = program % tokens
        output_head = (program // tokens) % output_heads
        block = program // (tokens * output_heads)
        input_head = output_head // groups
        offsets = tl.arange(0, BLOCK)
        mask = offsets < head_dim
        input_base = (block * input_block_stride + token * input_token_stride
                      + input_head * head_dim)
        values = tl.load(key_ptr + input_base + offsets, mask=mask, other=0.0).to(tl.float32)
        reciprocal_rms = tl.rsqrt(tl.sum(values * values, axis=0) / head_dim + eps)
        weights = tl.load(weight_ptr + block * head_dim + offsets, mask=mask, other=0.0)
        normalized = (values * reciprocal_rms).to(tl.bfloat16)
        normalized = (normalized * weights).to(tl.bfloat16)
        half = head_dim // 2
        rotated_offsets = tl.where(offsets < half, offsets + half, offsets - half)
        rotated_values = tl.load(key_ptr + input_base + rotated_offsets,
                                 mask=mask, other=0.0).to(tl.float32)
        rotated_weights = tl.load(weight_ptr + block * head_dim + rotated_offsets,
                                  mask=mask, other=0.0)
        rotated = (rotated_values * reciprocal_rms).to(tl.bfloat16)
        rotated = (rotated * rotated_weights).to(tl.bfloat16)
        rotated = tl.where(offsets < half, -rotated, rotated)
        rope_base = token * head_dim
        cos = tl.load(cos_ptr + rope_base + offsets, mask=mask, other=0.0)
        sin = tl.load(sin_ptr + rope_base + offsets, mask=mask, other=0.0)
        slot = tl.load(slots_ptr + token)
        output_base = (block * cache_block_stride + output_head * cache_head_stride
                       + slot * cache_token_stride)
        first = (normalized * cos).to(tl.bfloat16)
        second = (rotated * sin).to(tl.bfloat16)
        tl.store(key_cache_ptr + output_base + offsets,
                 (first + second).to(tl.bfloat16), mask=mask)
        projected_value = tl.load(value_ptr + input_base + offsets, mask=mask, other=0.0)
        tl.store(value_cache_ptr + output_base + offsets, projected_value, mask=mask)


_RUNTIME_COUNT_INDEX = ({
    _fast_rms_norm_kernel: (3,),
    _fast_add_rms_norm_kernel: (5,),
    _q_rotary_kernel: (4,),
    _kv_rotary_repeat_kernel: (6,),
    _grouped_rms_norm_kernel: (4,),
    _batched_kv_postprocess_kernel: (7, 11),
} if triton is not None else {})


def fast_grouped_rms_norm(prefix: torch.Tensor, weights: torch.Tensor,
                          stream_indices: torch.Tensor, eps: float) -> torch.Tensor:
    blocks, width = weights.shape
    tokens = prefix.shape[1]
    streams = prefix.shape[-1] // width
    if prefix.device.type != "cuda" or prefix.dtype == torch.float32 or triton is None:
        shaped = prefix.reshape(tokens, streams, width)
        return torch.stack([
            _reference_rms_norm(shaped[:, int(stream)], weights[block], eps)
            for block, stream in enumerate(stream_indices)
        ])
    output = prefix.new_empty(blocks, tokens, width)
    block_size = triton.next_power_of_2(width)
    _grouped_rms_norm_kernel[(blocks * tokens,)](
        prefix, weights, stream_indices, output,
        tokens=tokens, streams=streams, width=width, eps=eps,
        BLOCK=block_size, num_warps=8,
    )
    return output


def fast_batched_kv_postprocess(projected_key: torch.Tensor,
                                projected_value: torch.Tensor,
                                weights: torch.Tensor,
                                cos: torch.Tensor, sin: torch.Tensor, eps: float,
                                kv_heads: int, groups: int, head_dim: int):
    blocks, tokens, _ = projected_key.shape
    if (projected_key.device.type != "cuda" or projected_key.dtype == torch.float32
            or triton is None):
        keys, values = [], []
        for block in range(blocks):
            key, value = fast_kv_postprocess(
                projected_key[block].unsqueeze(0),
                projected_value[block].unsqueeze(0),
                weights[block], cos, sin, eps, kv_heads, groups, head_dim,
            )
            keys.append(key)
            values.append(value)
        return torch.stack(keys), torch.stack(values)
    cos, sin = cos.contiguous(), sin.contiguous()
    shape = (blocks, 1, kv_heads * groups, tokens, head_dim)
    key = projected_key.new_empty(shape)
    value = projected_value.new_empty(shape)
    block_size = triton.next_power_of_2(head_dim)
    _batched_kv_postprocess_kernel[(blocks * tokens * kv_heads * groups,)](
        projected_key, projected_value, weights, cos, sin, key, value,
        tokens=tokens, kv_heads=kv_heads, groups=groups, head_dim=head_dim,
        block_stride=projected_key.stride(0),
        token_stride=projected_key.stride(1),
        eps=eps, BLOCK=block_size, num_warps=4,
    )
    return key, value


def folded_append_kv_into_cache(projected_key, projected_value, weights,
                                cos, sin, slots, key_cache, value_cache,
                                eps, kv_heads, groups, head_dim):
    """Postprocess a folded span's append rows directly into their cache slots."""
    blocks, tokens, _ = projected_key.shape
    if (projected_key.device.type != "cuda" or projected_key.dtype == torch.float32
            or triton is None):
        key, value = fast_batched_kv_postprocess(projected_key, projected_value, weights,
                                                  cos, sin, eps, kv_heads, groups, head_dim)
        key_cache.index_copy_(3, slots, key)
        value_cache.index_copy_(3, slots, value)
        return
    cos, sin = cos.contiguous(), sin.contiguous()
    launch_cached(_folded_append_cache_kernel, (blocks * tokens * kv_heads * groups,),
                  (projected_key, projected_value, weights, cos, sin, slots,
                   key_cache, value_cache, tokens, kv_heads, groups, head_dim,
                   projected_key.stride(0), projected_key.stride(1),
                   key_cache.stride(0), key_cache.stride(2), key_cache.stride(3),
                   eps, triton.next_power_of_2(head_dim)), num_warps=4)


def lean_grouped_rms_norm(prefix, weights, stream_indices, eps):
    """``fast_grouped_rms_norm``'s CUDA branch through ``launch_cached`` (JETSPEC_C1_PRELOOP)."""
    blocks, width = weights.shape
    tokens = prefix.shape[1]
    output = prefix.new_empty(blocks, tokens, width)
    launch_cached(_grouped_rms_norm_kernel, (blocks * tokens,),
                  (prefix, weights, stream_indices, output, tokens, prefix.shape[-1] // width,
                   width, eps, triton.next_power_of_2(width)), num_warps=8)
    return output


def lean_batched_kv_postprocess(projected_key, projected_value, weights, cos, sin, eps,
                                kv_heads, groups, head_dim):
    """``fast_batched_kv_postprocess``'s CUDA branch through ``launch_cached``
    (JETSPEC_C1_PRELOOP); ``cos`` and ``sin`` must already be contiguous."""
    blocks, tokens, _ = projected_key.shape
    shape = (blocks, 1, kv_heads * groups, tokens, head_dim)
    key = projected_key.new_empty(shape)
    value = projected_value.new_empty(shape)
    launch_cached(_batched_kv_postprocess_kernel, (blocks * tokens * kv_heads * groups,),
                  (projected_key, projected_value, weights, cos, sin, key, value, tokens,
                   kv_heads, groups, head_dim, projected_key.stride(0), projected_key.stride(1),
                   eps, triton.next_power_of_2(head_dim)), num_warps=4)
    return key, value


# JETSPEC_C1_LOOP=1: the eager D1 lean query's 15-block loop in C++ (serving).  The same
# ATen ops (at::linear, the memory-efficient attention) and the same kernels (the
# certified norm / add+norm / silu*mul extensions, the compiled Triton rotary
# launchers) on the same operands, in the same order, so the output is bitwise the
# Python loop's; what goes is the Python interpreter and dispatcher work per op.
#
# JETSPEC_C1_GEMM=direct (serving): each projection calls cublasGemmEx itself with the
# arguments at::linear hands it (at::cuda::blas::gemm<BFloat16>: 't','n', m=N, n=M,
# k=K, lda=ldb=K, ldc=N, fp32 compute, CUBLAS_GEMM_DEFAULT_TENSOR_OP, the math mode
# from allowBF16ReductionCuBLAS, on the current handle and stream), so cuBLAS selects
# the same kernel and the output is bitwise at::linear's; what goes is ATen's
# linear -> matmul -> mm -> addmm dispatch, ~4 us of host per projection.
#
# JETSPEC_C1_NATIVE=1 (serving): the loop calls the certified norm, add+norm and silu*mul
# host functions through their C++ entry points in the already-built extensions (the
# same .so, so the same kernels on the same operands) instead of back through Python;
# what goes is the pybind11 round trip, 90 per round.
#
# JETSPEC_C1_PRELOOP=1 (serving): the host work before the loop's first kernel is device
# idle one for one (the device has only the append's ~0.2 ms to run meanwhile), so it
# is cut down: the D1 window mask is written straight into the loop's 16-aligned
# layout by one kernel (the prefix mask copied, 0 / bf16 lowest by the same position
# comparison, zero padding: the bits of torch.cat + the aligned copy), the loop's two
# compiled Triton entries are looked up once per cache, and the prefix append's two
# Triton kernels go through launch_cached.
#
# JETSPEC_C1_ROPE_NORM=1 (serving): the loop's q and k head norms and the rotary that reads
# them run as one kernel each (30 launches fewer per round): c1_rms_norm_kernel's
# width-128 arithmetic line for line, built with its own -O3 --fmad=false, then the
# rotary exactly as Triton lowers _q_rotary_kernel / _kv_rotary_into_cache_kernel on
# sm80 (read from their PTX: -x as fma.rn.bf16(x, -1, +0), each product as
# fma.rn.bf16(a, b, -0), the sum as fma.rn.bf16(first, 1, second)).
#
# JETSPEC_C1_APPEND=1 (serving): the steady-state round's work before the loop -- the
# query and prefix rotary rows from the ROPE_TABLE table, the prefix append (the zero
# pad to 16 rows, the grouped kv norm, the two stacked bmm, the batched k/v
# postprocess, the two cache copies) and the PRELOOP window mask -- as one C++ call:
# the same ATen ops and compiled Triton entries on the same operands, in the same
# order, without the Python between them (that host time is device idle one for one).
#
# JETSPEC_C1_KVNORM_BATCH=1 (serving): the three blocks of a stream normalise the same
# window rows with their own kv_norm weights; the loop runs those three certified wide
# norms as one launch (c1_rms_norm_wide_kernel's body line for line, the weight row
# picked per output row), 15 launches -> 5 per round.
_LOOP_EXTENSION = None

_LOOP_CUDA = r"""
#include <cstdlib>
#include <cstring>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>

__global__ void c1_d1_window_mask_kernel(
    const __nv_bfloat16* __restrict__ prefix, int64_t prefix_row_stride,
    const int64_t* __restrict__ positions, __nv_bfloat16* __restrict__ out,
    int64_t length, int64_t window, int64_t padded) {
  const int64_t row = blockIdx.x;
  const __nv_bfloat16 zero = __float2bfloat16(0.0f);
  const __nv_bfloat16 lowest = __ushort_as_bfloat16(static_cast<unsigned short>(0xFF7F));
  const int64_t mine = positions[row];
  for (int64_t col = threadIdx.x; col < padded; col += blockDim.x) {
    __nv_bfloat16 value = zero;
    if (col < length) {
      value = prefix[row * prefix_row_stride + col];
    } else if (col < length + window) {
      value = mine > positions[col - length] ? zero : lowest;
    }
    out[row * padded + col] = value;
  }
}

// c1_rms_norm_kernel's width-128 branch for one row held by one warp: each lane
// keeps columns lane + 32 * item, item 0..3.
__device__ __forceinline__ void c1_head_norm128(
    const __nv_bfloat16* __restrict__ input, const __nv_bfloat16* __restrict__ weight,
    float eps, int thread, __nv_bfloat16 out[4]) {
  float partial[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
  for (int item = 0; item < 4; ++item) {
    float value = __bfloat162float(input[thread + item * 32]);
    partial[item] = __fmul_rn(value, value);
  }
  float sum_sq = __fadd_rn(__fadd_rn(partial[0], partial[1]), partial[2]);
  sum_sq = __fadd_rn(sum_sq, partial[3]);
#pragma unroll
  for (int offset = 1; offset < 32; offset <<= 1) {
    sum_sq = __fadd_rn(sum_sq, __shfl_down_sync(0xffffffff, sum_sq, offset));
  }
  float reciprocal_rms = 0.0f;
  if (thread == 0) {
    float mean = __fmul_rn(sum_sq, 1.0f / float(128));
    reciprocal_rms = rsqrtf(__fadd_rn(mean, eps));
  }
  reciprocal_rms = __shfl_sync(0xffffffff, reciprocal_rms, 0);
#pragma unroll
  for (int item = 0; item < 4; ++item) {
    const int column = thread + item * 32;
    float value = __bfloat162float(input[column]);
    __nv_bfloat16 normalized = __float2bfloat16_rn(value * reciprocal_rms);
    float scaled = __bfloat162float(normalized) * __bfloat162float(weight[column]);
    out[item] = __float2bfloat16_rn(scaled);
  }
}

// The rotary of a 128-wide row (half 64: column c < 64 pairs with -row[c + 64], the rest
// with row[c - 64], so a lane's item pairs with its own item ^ 2).
__device__ __forceinline__ void c1_rotary128(
    const __nv_bfloat16 row[4], const __nv_bfloat16* __restrict__ cos,
    const __nv_bfloat16* __restrict__ sin, int thread, __nv_bfloat16 out[4]) {
  const __nv_bfloat16 minus_one = __ushort_as_bfloat16(static_cast<unsigned short>(0xBF80));
  const __nv_bfloat16 one = __ushort_as_bfloat16(static_cast<unsigned short>(0x3F80));
  const __nv_bfloat16 plus_zero = __ushort_as_bfloat16(static_cast<unsigned short>(0x0000));
  const __nv_bfloat16 minus_zero = __ushort_as_bfloat16(static_cast<unsigned short>(0x8000));
#pragma unroll
  for (int item = 0; item < 4; ++item) {
    const int column = thread + item * 32;
    __nv_bfloat16 rotated = row[item ^ 2];
    if (item < 2) rotated = __hfma(rotated, minus_one, plus_zero);
    const __nv_bfloat16 first = __hfma(row[item], cos[column], minus_zero);
    const __nv_bfloat16 second = __hfma(rotated, sin[column], minus_zero);
    out[item] = __hfma(first, one, second);
  }
}

__device__ __forceinline__ float c1_lane_bf16(unsigned int word, int high) {
  return __bfloat162float(__ushort_as_bfloat16(
      (unsigned short)(high ? (word >> 16) : (word & 0xFFFFu))));
}

__device__ __forceinline__ unsigned int c1_pack_bf16(__nv_bfloat16 low, __nv_bfloat16 high) {
  return (unsigned int)__bfloat16_as_ushort(low)
      | ((unsigned int)__bfloat16_as_ushort(high) << 16);
}

// c1_rms_norm_wide_kernel (the certified 4096-wide norm) for `which` weight rows over the
// same input rows: output row which * rows + row = norm(input row, weights[which]).
__global__ void c1_rms_norm_wide_rows_kernel(
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ weights,
    __nv_bfloat16* __restrict__ output,
    int rows,
    int total,
    int row_stride,
    float eps) {
  constexpr int WIDTH = 4096;
  constexpr int STEPS = WIDTH / 128;
  int flat = blockIdx.x * blockDim.y + threadIdx.y;
  if (flat >= total) return;
  int which = flat / rows, row = flat % rows;
  int thread = threadIdx.x;
  const __nv_bfloat16* source = input + (long long)row * row_stride + thread * 4;
  uint2 raw[STEPS];
#pragma unroll
  for (int step = 0; step < STEPS; ++step) {
    raw[step] = *reinterpret_cast<const uint2*>(source + step * 128);
  }
  float partial[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
  for (int step = 0; step < STEPS; ++step) {
#pragma unroll
    for (int item = 0; item < 4; ++item) {
      float value = c1_lane_bf16(item < 2 ? raw[step].x : raw[step].y, item & 1);
      partial[item] = __fadd_rn(partial[item], __fmul_rn(value, value));
    }
  }
  float sum_sq = __fadd_rn(__fadd_rn(partial[0], partial[1]), partial[2]);
  sum_sq = __fadd_rn(sum_sq, partial[3]);
#pragma unroll
  for (int offset = 1; offset < 32; offset <<= 1) {
    sum_sq = __fadd_rn(
        sum_sq, __shfl_down_sync(0xffffffff, sum_sq, offset));
  }
  float reciprocal_rms = 0.0f;
  if (thread == 0) {
    float mean = __fmul_rn(sum_sq, 1.0f / float(WIDTH));
    reciprocal_rms = rsqrtf(__fadd_rn(mean, eps));
  }
  reciprocal_rms = __shfl_sync(0xffffffff, reciprocal_rms, 0);
  const __nv_bfloat16* scale = weights + (long long)which * WIDTH + thread * 4;
  uint2 scale_all[STEPS];
#pragma unroll
  for (int step = 0; step < STEPS; ++step) {
    scale_all[step] = *reinterpret_cast<const uint2*>(scale + step * 128);
  }
  __nv_bfloat16* target = output + (long long)flat * WIDTH + thread * 4;
#pragma unroll
  for (int step = 0; step < STEPS; ++step) {
    uint2 scale_raw = scale_all[step];
    __nv_bfloat16 results[4];
#pragma unroll
    for (int item = 0; item < 4; ++item) {
      float value = c1_lane_bf16(item < 2 ? raw[step].x : raw[step].y, item & 1);
      __nv_bfloat16 normalized = __float2bfloat16_rn(value * reciprocal_rms);
      float scaled = __bfloat162float(normalized)
          * c1_lane_bf16(item < 2 ? scale_raw.x : scale_raw.y, item & 1);
      results[item] = __float2bfloat16_rn(scaled);
    }
    uint2 result = make_uint2(c1_pack_bf16(results[0], results[1]),
                              c1_pack_bf16(results[2], results[3]));
    *reinterpret_cast<uint2*>(target + step * 128) = result;
  }
}

template<int STEPS>
__global__ void c1_rms_norm_wide_rows_kernel_t(
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ weights,
    __nv_bfloat16* __restrict__ output,
    int rows, int total, int row_stride, float eps) {
  constexpr int WIDTH = STEPS * 128;
  int flat = blockIdx.x * blockDim.y + threadIdx.y;
  if (flat >= total) return;
  int which = flat / rows, row = flat % rows;
  int thread = threadIdx.x;
  const __nv_bfloat16* source = input + (long long)row * row_stride + thread * 4;
  float partial[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
  for (int step = 0; step < STEPS; ++step) {
    uint2 raw = *reinterpret_cast<const uint2*>(source + step * 128);
#pragma unroll
    for (int item = 0; item < 4; ++item) {
      float value = c1_lane_bf16(item < 2 ? raw.x : raw.y, item & 1);
      partial[item] = __fadd_rn(partial[item], __fmul_rn(value, value));
    }
  }
  float sum_sq = __fadd_rn(__fadd_rn(partial[0], partial[1]), partial[2]);
  sum_sq = __fadd_rn(sum_sq, partial[3]);
#pragma unroll
  for (int offset = 1; offset < 32; offset <<= 1) {
    sum_sq = __fadd_rn(sum_sq, __shfl_down_sync(0xffffffff, sum_sq, offset));
  }
  float reciprocal_rms = 0.0f;
  if (thread == 0) {
    float mean = __fmul_rn(sum_sq, 1.0f / float(WIDTH));
    reciprocal_rms = rsqrtf(__fadd_rn(mean, eps));
  }
  reciprocal_rms = __shfl_sync(0xffffffff, reciprocal_rms, 0);
  const __nv_bfloat16* scale = weights + (long long)which * WIDTH + thread * 4;
  __nv_bfloat16* target = output + (long long)flat * WIDTH + thread * 4;
#pragma unroll
  for (int step = 0; step < STEPS; ++step) {
    uint2 raw = *reinterpret_cast<const uint2*>(source + step * 128);
    uint2 scale_raw = *reinterpret_cast<const uint2*>(scale + step * 128);
    __nv_bfloat16 results[4];
#pragma unroll
    for (int item = 0; item < 4; ++item) {
      float value = c1_lane_bf16(item < 2 ? raw.x : raw.y, item & 1);
      __nv_bfloat16 normalized = __float2bfloat16_rn(value * reciprocal_rms);
      float scaled = __bfloat162float(normalized)
          * c1_lane_bf16(item < 2 ? scale_raw.x : scale_raw.y, item & 1);
      results[item] = __float2bfloat16_rn(scaled);
    }
    uint2 result = make_uint2(c1_pack_bf16(results[0], results[1]),
                              c1_pack_bf16(results[2], results[3]));
    *reinterpret_cast<uint2*>(target + step * 128) = result;
  }
}

// [rows, 4096] input and [n, 4096] weights -> [n, rows, 4096]; an empty tensor when the
// operands are not what the wide kernel reads (the caller then norms block by block).
torch::Tensor c1_rms_norm_wide_rows(torch::Tensor input, torch::Tensor weights, double eps) {
  if (input.dim() != 2 || weights.dim() != 2
      || input.size(1) != weights.size(1)
      || (weights.size(1) != 4096 && weights.size(1) != 5120)
      || input.stride(1) != 1 || input.stride(0) % 4 != 0
      || !weights.is_contiguous() || input.scalar_type() != at::kBFloat16
      || weights.scalar_type() != at::kBFloat16
      || reinterpret_cast<uintptr_t>(input.data_ptr()) % 8 != 0
      || reinterpret_cast<uintptr_t>(weights.data_ptr()) % 8 != 0) {
    return torch::Tensor();
  }
  const int rows = static_cast<int>(input.size(0));
  const int total = static_cast<int>(weights.size(0)) * rows;
  const int width = static_cast<int>(weights.size(1));
  auto output = at::empty({weights.size(0), input.size(0), width}, input.options());
  const auto stream = at::cuda::getCurrentCUDAStream(input.device().index());
  const auto* input_ptr = reinterpret_cast<const __nv_bfloat16*>(input.data_ptr());
  const auto* weights_ptr = reinterpret_cast<const __nv_bfloat16*>(weights.data_ptr());
  auto* output_ptr = reinterpret_cast<__nv_bfloat16*>(output.data_ptr());
  if (width == 4096) {
    c1_rms_norm_wide_rows_kernel<<<(total + 3) / 4, dim3(32, 4), 0, stream>>>(
        input_ptr, weights_ptr, output_ptr, rows, total,
        static_cast<int>(input.stride(0)), static_cast<float>(eps));
  } else {
    c1_rms_norm_wide_rows_kernel_t<40><<<(total + 3) / 4, dim3(32, 4), 0, stream>>>(
        input_ptr, weights_ptr, output_ptr, rows, total,
        static_cast<int>(input.stride(0)), static_cast<float>(eps));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

// q: [tokens, heads, 128] projection rows -> [1, heads, tokens, 128] (lean_q_postprocess).
__global__ void c1_q_norm_rotary_kernel(
    const __nv_bfloat16* __restrict__ projected, const __nv_bfloat16* __restrict__ weight,
    const __nv_bfloat16* __restrict__ cos, const __nv_bfloat16* __restrict__ sin,
    __nv_bfloat16* __restrict__ out, int tokens, int heads, float eps) {
  const int row = blockIdx.x * blockDim.y + threadIdx.y;
  if (row >= tokens * heads) return;
  const int token = row / heads, head = row % heads, thread = threadIdx.x;
  __nv_bfloat16 normalized[4], rotated[4];
  c1_head_norm128(projected + static_cast<int64_t>(row) * 128, weight, eps, thread, normalized);
  c1_rotary128(normalized, cos + token * 128, sin + token * 128, thread, rotated);
  __nv_bfloat16* target = out + (static_cast<int64_t>(head) * tokens + token) * 128;
#pragma unroll
  for (int item = 0; item < 4; ++item) target[thread + item * 32] = rotated[item];
}

// k: the k norm of [tokens, kv_heads * 128] projection rows, the rotary, and the key and
// value rows written into the cache slice for each repeated head
// (_kv_rotary_into_cache_kernel's program = token + tokens * output head).
__global__ void c1_k_norm_rotary_cache_kernel(
    const __nv_bfloat16* __restrict__ key, const __nv_bfloat16* __restrict__ value,
    const __nv_bfloat16* __restrict__ weight, const __nv_bfloat16* __restrict__ cos,
    const __nv_bfloat16* __restrict__ sin, __nv_bfloat16* __restrict__ key_out,
    __nv_bfloat16* __restrict__ value_out, const int64_t* __restrict__ start_slot,
    int tokens, int kv_heads, int groups,
    int64_t in_token_stride, int64_t out_token_stride, int64_t out_head_stride, float eps) {
  const int program = blockIdx.x * blockDim.y + threadIdx.y;
  if (program >= tokens * kv_heads * groups) return;
  const int token = program % tokens, output_head = program / tokens;
  const int input_head = output_head / groups, thread = threadIdx.x;
  const int64_t base = token * in_token_stride + input_head * 128;
  __nv_bfloat16 normalized[4], rotated[4];
  c1_head_norm128(key + base, weight, eps, thread, normalized);
  c1_rotary128(normalized, cos + token * 128, sin + token * 128, thread, rotated);
  // the graphed live layout's rows start at the slot held in device memory
  // (_kv_rotary_into_cache_at_kernel); eager passes no slot and a pre-offset slice
  const int64_t slot = (start_slot != nullptr ? start_slot[0] : 0) + token;
  const int64_t target = slot * out_token_stride + output_head * out_head_stride;
#pragma unroll
  for (int item = 0; item < 4; ++item) {
    key_out[target + thread + item * 32] = rotated[item];
    value_out[target + thread + item * 32] = value[base + thread + item * 32];
  }
}

void c1_q_norm_rotary(torch::Tensor projected, torch::Tensor weight, torch::Tensor cos,
                      torch::Tensor sin, torch::Tensor out, int64_t tokens, int64_t heads,
                      double eps) {
  const int rows = static_cast<int>(tokens * heads);
  const auto stream = at::cuda::getCurrentCUDAStream(projected.device().index());
  c1_q_norm_rotary_kernel<<<(rows + 15) / 16, dim3(32, 16), 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(projected.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(weight.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(cos.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(sin.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), static_cast<int>(tokens),
      static_cast<int>(heads), static_cast<float>(eps));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void c1_k_norm_rotary_cache(torch::Tensor key, torch::Tensor value, torch::Tensor weight,
                            torch::Tensor cos, torch::Tensor sin, torch::Tensor key_out,
                            torch::Tensor value_out, int64_t tokens, int64_t kv_heads,
                            int64_t groups, double eps, std::optional<torch::Tensor> start) {
  const int programs = static_cast<int>(tokens * kv_heads * groups);
  const int64_t* start_slot = nullptr;
  if (start.has_value()) {
    TORCH_CHECK(start->scalar_type() == at::kLong && start->is_cuda(), "C1 ROPE_NORM start slot");
    start_slot = start->data_ptr<int64_t>();
  }
  const auto stream = at::cuda::getCurrentCUDAStream(key.device().index());
  c1_k_norm_rotary_cache_kernel<<<(programs + 15) / 16, dim3(32, 16), 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(key.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(value.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(weight.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(cos.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(sin.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(key_out.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(value_out.data_ptr()), start_slot, static_cast<int>(tokens),
      static_cast<int>(kv_heads), static_cast<int>(groups), key.stride(0),
      key_out.stride(0), key_out.stride(1), static_cast<float>(eps));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// The refresh projections are independent: process their Q and K rows in one
// launch, using exactly the two existing kernels' per-warp arithmetic helpers.
__global__ void c1_qk_norm_rotary_cache_kernel(
    const __nv_bfloat16* __restrict__ query, const __nv_bfloat16* __restrict__ key,
    const __nv_bfloat16* __restrict__ value, const __nv_bfloat16* __restrict__ q_weight,
    const __nv_bfloat16* __restrict__ k_weight, const __nv_bfloat16* __restrict__ cos,
    const __nv_bfloat16* __restrict__ sin, __nv_bfloat16* __restrict__ q_out,
    __nv_bfloat16* __restrict__ key_out, __nv_bfloat16* __restrict__ value_out,
    const int64_t* __restrict__ start_slot, int tokens, int heads, int kv_heads,
    int groups, int64_t in_token_stride, int64_t out_token_stride,
    int64_t out_head_stride, int64_t query_token_stride, int64_t query_head_stride,
    float q_eps, float k_eps) {
  const int program = blockIdx.x * blockDim.y + threadIdx.y;
  const int query_rows = tokens * heads;
  if (program >= query_rows + tokens * kv_heads) return;
  const bool is_query = program < query_rows;
  const int flat = is_query ? program : program - query_rows;
  const int token = is_query ? flat / heads : flat % tokens;
  const int output_head = is_query ? flat % heads : (flat / tokens) * groups;
  const int input_head = is_query ? output_head : output_head / groups;
  const int thread = threadIdx.x;
  const int64_t base = is_query ? static_cast<int64_t>(flat) * 128
                               : token * in_token_stride + input_head * 128;
  __nv_bfloat16 normalized[4], rotated[4];
  c1_head_norm128((is_query ? query : key) + base,
                  is_query ? q_weight : k_weight, is_query ? q_eps : k_eps,
                  thread, normalized);
  c1_rotary128(normalized, cos + token * 128, sin + token * 128, thread, rotated);
  if (is_query) {
    const int64_t target = token * query_token_stride + output_head * query_head_stride;
#pragma unroll
    for (int item = 0; item < 4; ++item) q_out[target + thread + item * 32] = rotated[item];
  } else {
    const int64_t slot = (start_slot != nullptr ? start_slot[0] : 0) + token;
    const int64_t target = slot * out_token_stride + output_head * out_head_stride;
#pragma unroll
    for (int item = 0; item < 4; ++item) {
      const __nv_bfloat16 v = value[base + thread + item * 32];
      // Repeated heads share all operands. Normalize and rotate once, then
      // write the same bits to each repeated head's existing cache layout.
      for (int repeat = 0; repeat < groups; ++repeat) {
        const int64_t repeated = target + repeat * out_head_stride + thread + item * 32;
        key_out[repeated] = rotated[item];
        value_out[repeated] = v;
      }
    }
  }
}

void c1_qk_norm_rotary_cache(torch::Tensor query, torch::Tensor key, torch::Tensor value,
                             torch::Tensor q_weight, torch::Tensor k_weight,
                             torch::Tensor cos, torch::Tensor sin, torch::Tensor q_out,
                             torch::Tensor key_out, torch::Tensor value_out,
                             int64_t tokens, int64_t heads, int64_t kv_heads,
                             int64_t groups, double q_eps, double k_eps,
                             std::optional<torch::Tensor> start, int64_t requested_warps) {
  const int64_t* start_slot = nullptr;
  if (start.has_value()) {
    TORCH_CHECK(start->scalar_type() == at::kLong && start->is_cuda(), "C1 QK start slot");
    start_slot = start->data_ptr<int64_t>();
  }
  const int programs = static_cast<int>(tokens * (heads + kv_heads));
  const auto stream = at::cuda::getCurrentCUDAStream(query.device().index());
  int warps = 16;
  if (tokens == 16) {
    TORCH_CHECK(requested_warps == 4 || requested_warps == 8 || requested_warps == 16,
                "C1 QK warps must be 4, 8 or 16");
    warps = static_cast<int>(requested_warps);
  }
  c1_qk_norm_rotary_cache_kernel<<<(programs + warps - 1) / warps, dim3(32, warps), 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(query.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(key.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(value.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(q_weight.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(k_weight.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(cos.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(sin.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(q_out.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(key_out.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(value_out.data_ptr()), start_slot,
      static_cast<int>(tokens), static_cast<int>(heads), static_cast<int>(kv_heads),
      static_cast<int>(groups), key.stride(0), key_out.stride(0), key_out.stride(1),
      q_out.stride(2), q_out.stride(1),
      static_cast<float>(q_eps), static_cast<float>(k_eps));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

torch::Tensor c1_d1_window_mask(torch::Tensor prefix, torch::Tensor positions, int64_t length) {
  // prefix: [1, 1, W, >= length] bf16, last dim contiguous; positions: [1, W] int64.
  TORCH_CHECK(prefix.scalar_type() == at::kBFloat16 && prefix.dim() == 4
              && prefix.stride(3) == 1 && prefix.size(3) >= length, "C1 MASK prefix");
  TORCH_CHECK(positions.scalar_type() == at::kLong && positions.is_contiguous()
              && positions.numel() == prefix.size(2), "C1 MASK positions");
  const int64_t window = prefix.size(2);
  const int64_t padded = (length + window + 15) / 16 * 16;
  auto out = at::empty({1, 1, window, padded}, prefix.options());
  const auto stream = at::cuda::getCurrentCUDAStream(prefix.device().index());
  c1_d1_window_mask_kernel<<<window, 128, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(prefix.data_ptr()), prefix.stride(2),
      positions.data_ptr<int64_t>(), reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
      length, window, padded);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}
"""

_LOOP_CPP = r"""
#include <torch/extension.h>
#include <ATen/Context.h>
#include <ATen/cuda/CUDAContext.h>
#include <cublas_v2.h>
#include <tuple>

namespace py = pybind11;

static void c1_check_aligned(const torch::Tensor& tensor) {
  TORCH_CHECK(reinterpret_cast<uintptr_t>(tensor.data_ptr()) % 16 == 0,
              "C1 LOOP: a Triton operand is not 16-byte aligned");
}

// at::linear(x, w) for a bias-free bf16 projection: the cublasGemmEx call ATen makes,
// made directly (see JETSPEC_C1_GEMM above).  Anything but dense bf16 rows goes to
// at::linear itself.
static torch::Tensor c1_linear(const torch::Tensor& x, const torch::Tensor& w, bool direct) {
  if (!direct || x.dim() != 2 || w.dim() != 2 || !x.is_contiguous() || !w.is_contiguous()
      || x.scalar_type() != at::kBFloat16 || w.scalar_type() != at::kBFloat16
      || x.size(1) != w.size(1) || x.size(0) < 2 || w.size(0) < 2) {
    return at::linear(x, w);
  }
  const int64_t M = x.size(0), K = x.size(1), N = w.size(0);
  auto out = at::empty({M, N}, x.options());
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  cublasMath_t flags = CUBLAS_DEFAULT_MATH;
  if (!at::globalContext().allowBF16ReductionCuBLAS()) {
    flags = static_cast<cublasMath_t>(flags | CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION);
  }
  TORCH_CHECK(cublasSetMathMode(handle, flags) == CUBLAS_STATUS_SUCCESS, "C1 GEMM math mode");
  const float alpha = 1.0f, beta = 0.0f;
  const auto status = cublasGemmEx(
      handle, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha, w.data_ptr(), CUDA_R_16BF, K,
      x.data_ptr(), CUDA_R_16BF, K, &beta, out.data_ptr(), CUDA_R_16BF, N,
      CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
  TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS, "C1 GEMM cublasGemmEx ", static_cast<int>(status));
  TORCH_CHECK(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH) == CUBLAS_STATUS_SUCCESS,
              "C1 GEMM math mode reset");
  return out;
}

// JETSPEC_C1_ROPE_NORM's kernels (this extension's CUDA source).
void c1_q_norm_rotary(torch::Tensor projected, torch::Tensor weight, torch::Tensor cos,
                      torch::Tensor sin, torch::Tensor out, int64_t tokens, int64_t heads,
                      double eps);
void c1_k_norm_rotary_cache(torch::Tensor key, torch::Tensor value, torch::Tensor weight,
                            torch::Tensor cos, torch::Tensor sin, torch::Tensor key_out,
                            torch::Tensor value_out, int64_t tokens, int64_t kv_heads,
                            int64_t groups, double eps, std::optional<torch::Tensor> start);
void c1_qk_norm_rotary_cache(torch::Tensor query, torch::Tensor key, torch::Tensor value,
                             torch::Tensor q_weight, torch::Tensor k_weight,
                             torch::Tensor cos, torch::Tensor sin, torch::Tensor q_out,
                             torch::Tensor key_out, torch::Tensor value_out,
                             int64_t tokens, int64_t heads, int64_t kv_heads,
                             int64_t groups, double q_eps, double k_eps,
                             std::optional<torch::Tensor> start, int64_t requested_warps);

torch::Tensor c1_rms_norm_wide_rows(torch::Tensor input, torch::Tensor weights, double eps);

// The certified extensions' host functions (JETSPEC_C1_NATIVE above); 0 = through Python.
using c1_norm_fn = torch::Tensor (*)(torch::Tensor, torch::Tensor, double);
using c1_add_norm_fn = std::vector<torch::Tensor> (*)(torch::Tensor, torch::Tensor,
                                                      torch::Tensor, double);
using c1_silu_mul_fn = torch::Tensor (*)(torch::Tensor, torch::Tensor);

torch::Tensor c1_d1_lean_loop(
    torch::Tensor u, torch::Tensor cos, torch::Tensor sin, torch::Tensor cos0,
    torch::Tensor sin0, torch::Tensor bias, int64_t length, int64_t keys_total,
    std::vector<torch::Tensor> weights, std::vector<double> scalars,
    std::vector<int64_t> ints, std::vector<torch::Tensor> caches,
    py::object norm, py::object add_norm, py::object silu_mul,
    py::tuple q_launch, py::tuple kv_launch, int64_t block_width, bool direct_gemm,
    int64_t native_norm, int64_t native_add_norm, int64_t native_silu_mul, bool rope_norm,
    std::optional<torch::Tensor> kv_norm_batch, int64_t rows) {
  TORCH_CHECK((rows == 15 || rows == 16) && u.size(0) == rows, "C1 LOOP rows must be 15 or 16");
  // JETSPEC_C1_KVNORM_BATCH: the stacked kv_norm weights [blocks, 4096] turn it on
  const torch::Tensor kv_norm_stack = kv_norm_batch.has_value() ? *kv_norm_batch : torch::Tensor();
  const int64_t blocks = static_cast<int64_t>(ints.size() / 5);
  TORCH_CHECK(weights.size() == static_cast<size_t>(blocks * 12)
              && scalars.size() == static_cast<size_t>(blocks * 6)
              && caches.size() == static_cast<size_t>(blocks * 2), "C1 LOOP: plan sizes");
  py::object none = py::none();
  const auto norm_c = reinterpret_cast<c1_norm_fn>(native_norm);
  const auto add_norm_c = reinterpret_cast<c1_add_norm_fn>(native_add_norm);
  const auto silu_mul_c = reinterpret_cast<c1_silu_mul_fn>(native_silu_mul);
  auto do_norm = [&](const torch::Tensor& x, const torch::Tensor& weight, double eps) {
    return norm_c ? norm_c(x, weight, eps) : norm(x, weight, eps).cast<torch::Tensor>();
  };
  auto do_add_norm = [&](const torch::Tensor& update, const torch::Tensor& residual,
                         const torch::Tensor& weight, double eps) {
    return add_norm_c ? add_norm_c(update, residual, weight, eps)
                      : add_norm(update, residual, weight, eps).cast<std::vector<torch::Tensor>>();
  };
  auto do_silu_mul = [&](const torch::Tensor& gate, const torch::Tensor& up) {
    return silu_mul_c ? silu_mul_c(gate, up) : silu_mul(gate, up).cast<torch::Tensor>();
  };
  const int64_t stream = reinterpret_cast<int64_t>(
      at::cuda::getCurrentCUDAStream(u.device().index()).stream());
  std::vector<torch::Tensor> outputs;
  torch::Tensor window = u, update, residual;
  int64_t current = 0;
  bool pending = false;
  torch::Tensor kv_sources;
  int64_t kv_first = -1;
  for (int64_t block = 0; block < blocks; ++block) {
    const torch::Tensor* w = &weights[block * 12];
    const double* s = &scalars[block * 6];
    const int64_t* n = &ints[block * 5];
    const int64_t index = n[0], heads = n[1], kv_heads = n[2], groups = n[3], head_dim = n[4];
    // w: in, q, q_norm, kv_norm, k, k_norm, v, o, post, gate, up, down
    // s: in_eps, q_eps, kv_eps, k_eps, post_eps, scaling
    torch::Tensor normed;
    if (!pending) {
      normed = do_norm(u, w[0], s[0]);
    } else {
      auto pair = do_add_norm(update, residual, w[0], s[0]);
      normed = pair[0];
      u = pair[1];
    }
    if (index != current) {
      outputs.push_back(u);
      window = u;
      current = index;
    }
    auto projected = c1_linear(normed, w[1], direct_gemm).view({1, rows, -1});
    auto query = projected.new_empty({1, heads, rows, head_dim});
    const bool fused_rope = rope_norm && head_dim == 128 && projected.is_contiguous()
        && cos.is_contiguous() && sin.is_contiguous();
    if (fused_rope) {
      c1_q_norm_rotary(projected, w[2], cos, sin, query, rows, heads, s[1]);
    } else {
      TORCH_CHECK(q_launch.size() == 3, "C1 LOOP needs the Triton rotary entries");
      auto normalized = do_norm(projected.view({1, rows, heads, head_dim}), w[2], s[1]);
      for (const auto& operand : {normalized, cos, sin, query}) c1_check_aligned(operand);
      q_launch[0](rows * heads, 1, 1, stream, q_launch[1], q_launch[2], none, none, none,
                  normalized, cos, sin, query, rows, heads, head_dim, block_width);
    }
    torch::Tensor source;
    if (kv_norm_stack.defined()) {
      if (kv_first < 0 || ints[kv_first * 5] != index) {
        // this stream's blocks: consecutive plan entries with its index, one kv eps
        int64_t span = 1;
        while (block + span < blocks && ints[(block + span) * 5] == index
               && scalars[(block + span) * 6 + 2] == s[2]) ++span;
        kv_sources = c1_rms_norm_wide_rows(window, kv_norm_stack.narrow(0, block, span), s[2]);
        kv_first = block;
        if (!kv_sources.defined()) kv_first = -1;
      }
      if (kv_first >= 0 && block - kv_first < kv_sources.size(0)) {
        source = kv_sources.select(0, block - kv_first);
      } else {
        source = do_norm(window, w[3], s[2]);
      }
    } else {
      source = do_norm(window, w[3], s[2]);
    }
    auto k = caches[block * 2], v = caches[block * 2 + 1];
    const int64_t width = kv_heads * head_dim;
    auto key_out = k.select(0, 0).slice(1, length, keys_total).transpose(0, 1);
    auto value_out = v.select(0, 0).slice(1, length, keys_total).transpose(0, 1);
    auto projected_value = c1_linear(source, w[6], direct_gemm);
    auto projected_key = c1_linear(source, w[4], direct_gemm);
    if (fused_rope && projected_key.is_contiguous() && projected_value.is_contiguous()
        && cos0.is_contiguous() && sin0.is_contiguous()) {
      TORCH_CHECK(projected_value.strides() == projected_key.strides()
                  && key_out.strides() == value_out.strides() && key_out.stride(2) == 1,
                  "C1 LEAN cache write requires matching dense operands");
      c1_k_norm_rotary_cache(projected_key, projected_value, w[5], cos0, sin0, key_out,
                             value_out, rows, kv_heads, groups, s[3], std::nullopt);
    } else {
      TORCH_CHECK(kv_launch.size() == 3, "C1 LOOP needs the Triton rotary entries");
      auto normalized_key = do_norm(projected_key.view({rows, kv_heads, head_dim}), w[5], s[3])
          .view({rows, width});
      TORCH_CHECK(projected_value.strides() == normalized_key.strides()
                  && key_out.strides() == value_out.strides() && key_out.stride(2) == 1,
                  "C1 LEAN cache write requires matching dense operands");
      for (const auto& operand : {normalized_key, projected_value, cos0, sin0, key_out, value_out})
        c1_check_aligned(operand);
      kv_launch[0](rows * heads, 1, 1, stream, kv_launch[1], kv_launch[2], none, none, none,
                   normalized_key, projected_value, cos0, sin0, key_out, value_out, rows, kv_heads,
                   groups, head_dim, normalized_key.stride(0), key_out.stride(0),
                   key_out.stride(1), block_width);
    }
    auto attended = std::get<0>(at::_scaled_dot_product_efficient_attention(
        query, k.slice(2, 0, keys_total), v.slice(2, 0, keys_total),
        bias.expand({1, heads, rows, keys_total}), false, 0.0, false, s[5]));
    auto projected_o = c1_linear(attended.transpose(1, 2).reshape({rows, -1}), w[7], direct_gemm);
    auto post = do_add_norm(projected_o, u, w[8], s[4]);
    auto hidden = do_silu_mul(c1_linear(post[0], w[9], direct_gemm),
                              c1_linear(post[0], w[10], direct_gemm));
    update = c1_linear(hidden, w[11], direct_gemm);
    residual = post[1];
    pending = true;
  }
  if (pending) u = at::add(residual, update);
  outputs.push_back(u);
  return at::cat(outputs, -1).view({1, rows, -1});
}

torch::Tensor c1_d1_window_mask(torch::Tensor prefix, torch::Tensor positions, int64_t length);

// JETSPEC_C1_APPEND: _forward_cached's steady-state work before the loop (see above).
// Returns {cos_q, sin_q, aligned mask}.
std::vector<torch::Tensor> c1_d1_append(
    torch::Tensor prefix, torch::Tensor positions, torch::Tensor query_positions,
    torch::Tensor cos_table, torch::Tensor sin_table, torch::Tensor kv_norm, torch::Tensor streams,
    torch::Tensor key_weights, torch::Tensor value_weights, torch::Tensor k_norm,
    torch::Tensor key_cache, torch::Tensor value_cache, torch::Tensor attention_mask,
    int64_t start, int64_t length, double kv_eps, double k_eps, int64_t kv_heads, int64_t groups,
    int64_t head_dim, py::tuple norm_launch, py::tuple post_launch) {
  const int64_t count = prefix.size(1);
  const int64_t blocks = kv_norm.size(0), width = kv_norm.size(1);
  const int64_t heads = kv_heads * groups;
  TORCH_CHECK(count >= 1 && count <= 16 && length - start == count, "C1 APPEND count");
  for (const auto& operand : {prefix, kv_norm, streams, k_norm}) c1_check_aligned(operand);
  py::object none = py::none();
  const int64_t stream = reinterpret_cast<int64_t>(
      at::cuda::getCurrentCUDAStream(prefix.device().index()).stream());
  // _c1_rope(token_embeddings, query_position_ids) with the table
  auto flat_q = query_positions.reshape({-1});
  auto cos_q = cos_table.index_select(0, flat_q).unsqueeze(0);
  auto sin_q = sin_table.index_select(0, flat_q).unsqueeze(0);
  // _eager_extend: _c1_rope(prefix, positions), the pad to 16 rows (F.pad)
  auto flat = positions.reshape({-1});
  auto cos = cos_table.index_select(0, flat).unsqueeze(0);
  auto sin = sin_table.index_select(0, flat).unsqueeze(0);
  torch::Tensor rows = prefix;
  if (count < 16) {
    rows = at::constant_pad_nd(prefix, {0, 0, 0, 16 - count}, 0);
    cos = at::constant_pad_nd(cos, {0, 0, 0, 16 - count}, 0);
    sin = at::constant_pad_nd(sin, {0, 0, 0, 16 - count}, 0);
  }
  // _d1_cache_project_batched: lean_grouped_rms_norm, the two bmm, lean_batched_kv_postprocess
  auto source = at::empty({blocks, 16, width}, rows.options());
  TORCH_CHECK(rows.is_contiguous(), "C1 APPEND requires dense prefix rows");
  c1_check_aligned(rows);
  c1_check_aligned(source);
  norm_launch[0](blocks * 16, 1, 1, stream, norm_launch[1], norm_launch[2], none, none, none,
                 rows, kv_norm, streams, source, 16, rows.size(-1) / width, width, kv_eps, width);
  auto key = at::bmm(source, key_weights.transpose(1, 2));
  auto value = at::bmm(source, value_weights.transpose(1, 2));
  cos = cos.contiguous();
  sin = sin.contiguous();
  auto key_out = at::empty({blocks, 1, heads, 16, head_dim}, key.options());
  auto value_out = at::empty({blocks, 1, heads, 16, head_dim}, value.options());
  // the entries' constexprs assume dense rows (block and token strides of a fresh bmm)
  TORCH_CHECK(rows.is_contiguous() && key.is_contiguous() && value.is_contiguous(),
              "C1 APPEND requires dense operands");
  for (const auto& operand : {key, value, cos, sin, key_out, value_out}) c1_check_aligned(operand);
  post_launch[0](blocks * 16 * heads, 1, 1, stream, post_launch[1], post_launch[2], none, none,
                 none, key, value, k_norm, cos, sin, key_out, value_out, 16, kv_heads, groups,
                 head_dim, key.stride(0), key.stride(1), k_eps, head_dim);
  key_cache.slice(3, start, length).copy_(key_out.slice(3, 0, count));
  value_cache.slice(3, start, length).copy_(value_out.slice(3, 0, count));
  auto aligned = c1_d1_window_mask(attention_mask.slice(3, 0, length), query_positions, length);
  return {cos_q, sin_q, aligned};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("d1_lean_loop", &c1_d1_lean_loop, "the D1 lean query's block loop");
  module.def("d1_window_mask", &c1_d1_window_mask, "the D1 query mask in the loop's layout");
  module.def("d1_append", &c1_d1_append, "the D1 round's steady-state work before the loop");
  module.def("rms_norm_wide_rows", &c1_rms_norm_wide_rows,
             "the certified wide norm of the same rows under several weights");
  module.def("q_norm_rotary", &c1_q_norm_rotary, "q head norm + rotary (JETSPEC_C1_ROPE_NORM)");
  module.def("k_norm_rotary_cache", &c1_k_norm_rotary_cache,
             "k head norm + rotary + cache write (JETSPEC_C1_ROPE_NORM)");
  module.def("qk_norm_rotary_cache", &c1_qk_norm_rotary_cache,
             "refresh Q/K head norm + rotary + K/V cache write in one launch",
             py::arg("query"), py::arg("key"), py::arg("value"), py::arg("q_weight"),
             py::arg("k_weight"), py::arg("cos"), py::arg("sin"), py::arg("q_out"),
             py::arg("key_out"), py::arg("value_out"), py::arg("tokens"), py::arg("heads"),
             py::arg("kv_heads"), py::arg("groups"), py::arg("q_eps"), py::arg("k_eps"),
             py::arg("start"), py::arg("requested_warps") = 16);
}
"""


_REFRESH_LOOP_CPP = r"""
// _refresh_cached_lean, with the same projection order, stream boundaries and
// padded-row trimming. Optional backends remain bound to their existing kernels.
torch::Tensor c1_refresh_loop(
    torch::Tensor u, torch::Tensor cos, torch::Tensor sin,
    std::optional<torch::Tensor> bias, int64_t start, int64_t keys_total,
    int64_t tokens, int64_t real, std::optional<torch::Tensor> start_slot,
    std::vector<torch::Tensor> weights, std::vector<double> scalars,
    std::vector<int64_t> ints, std::vector<torch::Tensor> caches,
    std::vector<torch::Tensor> packed_kv, py::dict ops, bool ew, bool finish) {
  const int64_t blocks = ints.size() / 5;
  TORCH_CHECK(blocks > 0 && weights.size() == blocks * 12
      && scalars.size() == blocks * 6 && caches.size() == blocks * 2
      && (packed_kv.empty() || packed_kv.size() == blocks)
      && tokens == u.size(0) && real >= 0 && real <= tokens, "refresh loop plan sizes");
  auto norm = ops["norm"].cast<py::object>();
  auto add_norm = ops["add_norm"].cast<py::object>();
  auto silu_mul = ops["silu_mul"].cast<py::object>();
  using norm_fn = torch::Tensor (*)(torch::Tensor, torch::Tensor, double);
  using add_fn = std::vector<torch::Tensor> (*)(torch::Tensor, torch::Tensor, torch::Tensor, double);
  using silu_fn = torch::Tensor (*)(torch::Tensor, torch::Tensor);
  const auto norm_c = reinterpret_cast<norm_fn>(ops.contains("native_norm")
      ? ops["native_norm"].cast<int64_t>() : 0);
  const auto add_c = reinterpret_cast<add_fn>(ops.contains("native_add")
      ? ops["native_add"].cast<int64_t>() : 0);
  const auto silu_c = reinterpret_cast<silu_fn>(ops.contains("native_silu")
      ? ops["native_silu"].cast<int64_t>() : 0);
  auto do_norm = [&](const torch::Tensor& x, const torch::Tensor& w, double eps) {
    return norm_c ? norm_c(x, w, eps) : norm(x, w, eps).cast<torch::Tensor>();
  };
  auto do_add = [&](const torch::Tensor& x, const torch::Tensor& y, const torch::Tensor& w, double eps) {
    return add_c ? add_c(x, y, w, eps) : add_norm(x, y, w, eps).cast<std::vector<torch::Tensor>>();
  };
  auto linear = [&](const torch::Tensor& x, const torch::Tensor& w) {
    return ops.contains("linear") ? ops["linear"](x, w).cast<torch::Tensor>()
                                  : at::linear(x, w);
  };
  std::vector<torch::Tensor> outputs;
  torch::Tensor window = u, update, residual;
  auto cos0 = cos.select(0, 0), sin0 = sin.select(0, 0);
  int64_t current = 0;
  for (int64_t block = 0; block < blocks; ++block) {
    const auto* w = &weights[block * 12];
    const auto* s = &scalars[block * 6];
    const auto* n = &ints[block * 5];
    const int64_t index = n[0], heads = n[1], kv_heads = n[2], groups = n[3], dim = n[4];
    TORCH_CHECK(dim == 128, "JETSPEC_REFRESH_LEAN requires head_dim 128");
    torch::Tensor normalized, source;
    if (ew) {
      py::object pending = py::none();
      if (block) pending = py::make_tuple(update, residual);
      auto out = ops["ew_pre_norm"](u, window, w[0], w[3].unsqueeze(0), s[0], s[2],
                                    pending, index != current).cast<std::vector<torch::Tensor>>();
      normalized = out[0]; u = out[1]; source = out[2].select(0, 0);
    } else if (block == 0) {
      normalized = do_norm(u, w[0], s[0]);
    } else {
      auto out = do_add(update, residual, w[0], s[0]);
      normalized = out[0]; u = out[1];
    }
    if (index != current) {
      outputs.push_back(u.slice(0, 0, real)); window = u; current = index;
    }
    auto query = u.new_empty({1, heads, tokens, dim});
#ifdef JETSPEC_REFRESH_CUDA
    c1_q_norm_rotary(linear(normalized, w[1]), w[2], cos, sin, query, tokens, heads, s[1]);
#else
    ops["q_norm_rotary"](linear(normalized, w[1]), w[2], cos, sin, query, tokens, heads, s[1]);
#endif
    if (!ew) source = do_norm(window, w[3], s[2]);
    auto k = caches[block * 2], v = caches[block * 2 + 1];
    const int64_t first = start_slot.has_value() ? 0 : start;
    auto key_out = k.select(0, 0).slice(1, first, first + tokens).transpose(0, 1);
    auto value_out = v.select(0, 0).slice(1, first, first + tokens).transpose(0, 1);
    torch::Tensor projected_key, projected_value;
    if (!packed_kv.empty()) {
      auto parts = linear(source, packed_kv[block]).split_with_sizes({w[4].size(0), w[6].size(0)}, -1);
      projected_key = parts[0]; projected_value = parts[1];
    } else {
      projected_value = linear(source, w[6]);
      projected_key = linear(source, w[4]);
    }
#ifdef JETSPEC_REFRESH_CUDA
    c1_k_norm_rotary_cache(projected_key, projected_value, w[5], cos0, sin0,
        key_out, value_out, tokens, kv_heads, groups, s[3], start_slot);
#else
    ops["k_norm_rotary_cache"](projected_key, projected_value, w[5], cos0, sin0,
        key_out, value_out, tokens, kv_heads, groups, s[3], start_slot);
#endif
    torch::Tensor attended;
    if (ops.contains("attention")) {
      attended = ops["attention"](query, k, v, s[5], groups).cast<torch::Tensor>();
    } else {
      TORCH_CHECK(bias.has_value(), "refresh loop requires an attention mask");
      attended = std::get<0>(at::_scaled_dot_product_efficient_attention(
          query, k.slice(2, 0, keys_total), v.slice(2, 0, keys_total),
          bias->slice(-1, 0, keys_total).expand({1, heads, tokens, keys_total}),
          false, 0.0, false, s[5]));
    }
    auto projected = linear(attended.transpose(1, 2).reshape({tokens, -1}), w[7]);
    auto post = do_add(projected, u, w[8], s[4]);
    // Separate statements guarantee gate then up (C++ argument order is unspecified).
    auto gate = linear(post[0], w[9]);
    auto up = linear(post[0], w[10]);
    auto activated = silu_c ? silu_c(gate, up) : silu_mul(gate, up).cast<torch::Tensor>();
    update = linear(activated, w[11]); residual = post[1];
  }
  if (finish) return ops["finish_streams"](outputs, residual, update, real).cast<torch::Tensor>();
  u = at::add(residual, update);
  outputs.push_back(u.slice(0, 0, real));
  return at::cat(outputs, -1);
}
"""

_LOOP_CPP = _LOOP_CPP.replace(
    'PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {',
    '#define JETSPEC_REFRESH_CUDA\n' + _REFRESH_LOOP_CPP + '\nPYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {\n'
    '  module.def("refresh_loop", &c1_refresh_loop, "the lean adapter refresh layer loop");')


def _torch_cublas_ldflags():
    """Link the libcublas torch loaded (the pip wheel's), never another CUDA's: a second
    libcublas would be handed torch's cuBLAS handle."""
    try:
        import nvidia.cublas.lib
    except ImportError:  # a torch built against the system CUDA links it itself
        return ["-lcublas"]
    directory = os.path.dirname(nvidia.cublas.lib.__file__)
    return [f"-L{directory}", "-l:libcublas.so.12", f"-Wl,-rpath,{directory}"]


def _loop_extension():
    global _LOOP_EXTENSION
    if _LOOP_EXTENSION is None:
        from torch.utils.cpp_extension import load_inline
        _LOOP_EXTENSION = load_inline(
            name="jetspec_c1_d1_loop_v13",
            cpp_sources=_LOOP_CPP,
            cuda_sources=_LOOP_CUDA,
            extra_cflags=["-O3"],
            # the certified norm's flags (JETSPEC_C1_ROPE_NORM replays its arithmetic)
            extra_cuda_cflags=["-O3", "--fmad=false"],
            extra_ldflags=_torch_cublas_ldflags(),
            with_cuda=True,
            verbose=False,
        )
    return _LOOP_EXTENSION


_NATIVE_ENTRIES = {}

# Itanium-mangled names of the certified extensions' host functions (the declarations
# in _NORM_CPP, _LEAN_OPS_CPP and _SILU_MUL_CPP).
_NATIVE_SYMBOLS = {
    "rms_norm": "_Z16c1_rms_norm_cudaN2at6TensorES0_d",
    "add_rms_norm": "_Z20c1_add_rms_norm_cudaN2at6TensorES0_S0_d",
    "silu_mul": "_Z16c1_silu_mul_cudaN2at6TensorES0_",
}


def native_entry(extension, name):
    """The address of ``extension``'s C++ host function behind its ``name`` binding (the
    loaded module's own .so, so the same code the binding runs), or 0 if the symbol is
    not exported; JETSPEC_C1_NATIVE's loop then calls that binding through Python."""
    key = (extension.__file__, name)
    address = _NATIVE_ENTRIES.get(key)
    if address is None:
        import ctypes
        try:
            library = ctypes.CDLL(extension.__file__, mode=os.RTLD_NOW | os.RTLD_NOLOAD)
            address = ctypes.cast(getattr(library, _NATIVE_SYMBOLS[name]), ctypes.c_void_p).value
        except (OSError, AttributeError):
            address = 0
        _NATIVE_ENTRIES[key] = address or 0
    return address or 0


def compiled_launcher(kernel, device, args, **options):
    """``launch_cached``'s compiled entry (launcher, function, packed metadata) for these
    arguments, or None when that specialization has not been compiled yet or launch
    hooks are installed (the caller then takes the Python path, which compiles it)."""
    runtime = triton.knobs.runtime
    if _has_launch_hooks(runtime):
        return None
    key = _launch_key(kernel, device, args, options)
    entry = _COMPILED_LAUNCHES.get(key)
    return None if entry is None else entry[1:]

_block_checks = []


def block_check_errors(count):
    """Read per-block differences after replay of a diagnostic graph."""
    return [float(error) for error in _block_checks[-count:]]


if triton is not None:

    @triton.jit
    def _stacked_kv_norm(TILE, WINDOW, WEIGHT, OUT,
                         TILE_ROWS: tl.constexpr, WINDOW_TOKENS: tl.constexpr,
                         WIDTH: tl.constexpr, EPS: tl.constexpr,
                         TILE_ROW: tl.constexpr, TILE_COL: tl.constexpr,
                         WIN_BATCH: tl.constexpr, WIN_TOKEN: tl.constexpr,
                         WIN_COL: tl.constexpr, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        col = tl.arange(0, BLOCK)
        mask = col < WIDTH
        if row < TILE_ROWS:
            value = tl.load(TILE + row * TILE_ROW + col * TILE_COL, mask, 0)
        else:
            window_row = row - TILE_ROWS
            batch = window_row // WINDOW_TOKENS
            token = window_row % WINDOW_TOKENS
            value = tl.load(WINDOW + batch * WIN_BATCH + token * WIN_TOKEN
                            + col * WIN_COL, mask, 0)
        value = value.to(tl.float32)
        rms = tl.rsqrt(tl.sum(value * value, 0) / WIDTH + EPS)
        normalized = (value * rms).to(tl.bfloat16)
        weight = tl.load(WEIGHT + col, mask, 0)
        tl.store(OUT + row * WIDTH + col, (normalized * weight).to(tl.bfloat16), mask)

    @triton.jit
    def _q_norm_rotary(PROJ, WEIGHT, COS, SIN, OUT,
                       TOKENS: tl.constexpr, HEADS: tl.constexpr, DIM: tl.constexpr,
                       EPS: tl.constexpr, BLOCK: tl.constexpr,
                       NORM_OFFSET: tl.constexpr = 0, ROTARY: tl.constexpr = 0,
                       GATED: tl.constexpr = False, ROW_STRIDE: tl.constexpr = 0,
                       VALUE=None, CACHE_V=None, SLOTS=None,
                       CACHE: tl.constexpr = False, GROUPS: tl.constexpr = 1,
                       CAPACITY: tl.constexpr = 0, VALUE_STRIDE: tl.constexpr = 0,
                       REDUCE_THREADS: tl.constexpr = 32, ULP_STATS: tl.constexpr = False):
        head = tl.program_id(0) % HEADS
        token = tl.program_id(0) // HEADS % TOKENS
        batch = tl.program_id(0) // (HEADS * TOKENS)
        col = tl.arange(0, BLOCK)
        mask = col < DIM
        stride: tl.constexpr = ROW_STRIDE if ROW_STRIDE else HEADS * DIM * (2 if GATED else 1)
        rotary: tl.constexpr = ROTARY if ROTARY else DIM
        row = (batch * TOKENS + token) * stride + head * DIM * (2 if GATED else 1)
        value = tl.load(PROJ + row + col, mask, 0).to(tl.float32)
        weight = tl.load(WEIGHT + col, mask, 0)
        if NORM_OFFSET:
            if ULP_STATS:
                rms = libdevice.rsqrt(tl.sum(value * value, 0) / DIM + EPS)
            else:
                rms = _module_rms_scale(value, DIM, REDUCE_THREADS, EPS, BLOCK)
            norm = ((value * rms) * (NORM_OFFSET + weight.to(tl.float32))).to(tl.bfloat16)
        else:
            rms = tl.rsqrt(tl.sum(value * value, 0) / DIM + EPS)
            norm = (value * rms).to(tl.bfloat16)
            norm = (norm * weight).to(tl.bfloat16)
        other_col = tl.where(col < rotary // 2, col + rotary // 2, col - rotary // 2)
        other = tl.load(PROJ + row + other_col, col < rotary, 0).to(tl.float32)
        # Normalize the paired half with the same per-head RMS.
        other_weight = tl.load(WEIGHT + other_col, col < rotary, 0)
        if NORM_OFFSET:
            other = ((other * rms) * (NORM_OFFSET + other_weight.to(tl.float32))).to(tl.bfloat16)
        else:
            other = ((other * rms).to(tl.bfloat16) * other_weight).to(tl.bfloat16)
        other = tl.where(col < rotary // 2, -other, other)
        angle = (batch * TOKENS + token) * rotary + col
        cos = tl.load(COS + angle, col < rotary, 0)
        sin = tl.load(SIN + angle, col < rotary, 0)
        first = (norm * cos).to(tl.bfloat16)
        second = (other * sin).to(tl.bfloat16)
        result = tl.where(col < rotary, (first + second).to(tl.bfloat16), norm)
        if CACHE:
            slot = tl.load(SLOTS + token)
            v = tl.load(VALUE + token * VALUE_STRIDE + head * DIM + col, mask, 0)
            for group in range(GROUPS):
                output = ((head * GROUPS + group) * CAPACITY + slot) * DIM + col
                tl.store(OUT + output, result, mask)
                tl.store(CACHE_V + output, v, mask)
        else:
            output = ((batch * HEADS + head) * TOKENS + token) * DIM + col
            tl.store(OUT + output, result, mask)


    @triton.jit
    def _kv_norm_rotary(PROJ, WEIGHT, COS, SIN, KEY, VALUE,
                        CACHE_K, CACHE_V, SLOTS, POSITIONS,
                        TOKENS: tl.constexpr, HEADS: tl.constexpr, DIM: tl.constexpr,
                        EPS: tl.constexpr, BLOCK: tl.constexpr,
                        DIRECT_CACHE: tl.constexpr, TILE_ROWS: tl.constexpr,
                        WINDOW_TOKENS: tl.constexpr,
                        CK_SLOT: tl.constexpr, CK_HEAD: tl.constexpr, CK_TOKEN: tl.constexpr,
                        CV_SLOT: tl.constexpr, CV_HEAD: tl.constexpr, CV_TOKEN: tl.constexpr):
        head = tl.program_id(0) % HEADS
        token = tl.program_id(0) // HEADS
        col = tl.arange(0, BLOCK)
        mask = col < DIM
        width = HEADS * DIM
        row = token * (2 * width) + head * DIM
        key = tl.load(PROJ + row + col, mask, 0).to(tl.float32)
        rms = tl.rsqrt(tl.sum(key * key, 0) / DIM + EPS)
        norm = (key * rms).to(tl.bfloat16)
        weight = tl.load(WEIGHT + col, mask, 0)
        norm = (norm * weight).to(tl.bfloat16)
        other_col = tl.where(col < DIM // 2, col + DIM // 2, col - DIM // 2)
        other = tl.load(PROJ + row + other_col, mask, 0).to(tl.float32)
        other = ((other * rms).to(tl.bfloat16) * tl.load(WEIGHT + other_col, mask, 0)).to(tl.bfloat16)
        other = tl.where(col < DIM // 2, -other, other)
        angle = token * DIM + col
        cos = tl.load(COS + angle, mask, 0)
        sin = tl.load(SIN + angle, mask, 0)
        first = (norm * cos).to(tl.bfloat16)
        second = (other * sin).to(tl.bfloat16)
        value = tl.load(PROJ + row + width + col, mask, 0)
        if DIRECT_CACHE:
            if token < TILE_ROWS:
                slot = tl.load(SLOTS + token)
                position = tl.load(POSITIONS + token)
                key_offset = slot * CK_SLOT + head * CK_HEAD + position * CK_TOKEN + col
                value_offset = slot * CV_SLOT + head * CV_HEAD + position * CV_TOKEN + col
                tl.store(CACHE_K + key_offset, (first + second).to(tl.bfloat16), mask)
                tl.store(CACHE_V + value_offset, value, mask)
            else:
                window_row = token - TILE_ROWS
                batch = window_row // WINDOW_TOKENS
                query = window_row % WINDOW_TOKENS
                output = ((batch * HEADS + head) * WINDOW_TOKENS + query) * DIM + col
                tl.store(KEY + output, (first + second).to(tl.bfloat16), mask)
                tl.store(VALUE + output, value, mask)
        else:
            output = (head * TOKENS + token) * DIM + col
            tl.store(KEY + output, (first + second).to(tl.bfloat16), mask)
            tl.store(VALUE + output, value, mask)


def q_norm_rotary(fused, projected, weight, cos, sin, eps, heads, dim,
                  norm_offset=0, rotary_dim=None, gated=False, ulp=False, norm_round=None):
    """Head norm and RoPE in one BF16 launch for the batched adapter."""
    from jetspec.engines.jetspec_plus_plus.profiles import kernel_enabled
    rounding = norm_round or ("after_weight" if norm_offset else "before_weight")
    specialized = rounding == ("after_weight" if norm_offset else "before_weight")
    if specialized and norm_offset == 1 and rotary_dim == 64 and dim == 256 and kernel_enabled('hybrid_c1_gluon', '1'):
        from types import SimpleNamespace
        from jetspec.engines.jetspec_plus_plus.kernels import gluon_block
        query = projected.view(*projected.shape[:2], heads, dim * (2 if gated else 1))[..., :dim]
        if gluon_block.available(query):
            return gluon_block.norm_rotary(
                query, SimpleNamespace(weight=weight, variance_epsilon=eps), cos, sin)
    if projected.device.type != "cuda" or projected.dtype != torch.bfloat16 or triton is None or not specialized:
        if norm_offset or gated or not specialized or (rotary_dim is not None and rotary_dim != dim):
            q = projected.view(*projected.shape[:2], heads, dim * (2 if gated else 1))[..., :dim]
            return _reference_rotary(_reference_rms_norm(q, weight, eps, norm_offset, rounding).transpose(1, 2), cos, sin)
        return fused.fast_q_postprocess(projected, weight, cos, sin, eps, heads, dim)
    batch, tokens = projected.shape[:2]
    output = projected.new_empty(batch, heads, tokens, dim)
    _q_norm_rotary[(batch * tokens * heads,)](
        projected.contiguous(), weight, cos.contiguous(), sin.contiguous(), output,
        tokens, heads, dim, eps, triton.next_power_of_2(dim),
        norm_offset, rotary_dim or dim, gated,
        REDUCE_THREADS=_module_reduce_threads(dim, batch * tokens * heads),
        ULP_STATS=ulp,
        num_warps=4, enable_fp_fusion=False)
    return output


def kv_norm_rotary_cache(key, value, weight, cos, sin, eps, heads, groups, dim,
                         cache, slots, norm_offset=0, rotary_dim=None, ulp=False, norm_round=None):
    """Head norm, partial/full rotary and broadcast cache writes in one launch."""
    from jetspec.engines.jetspec_plus_plus.profiles import kernel_enabled
    rounding = norm_round or ("after_weight" if norm_offset else "before_weight")
    specialized = rounding == ("after_weight" if norm_offset else "before_weight")
    if specialized and norm_offset == 1 and rotary_dim == 64 and dim == 256 and kernel_enabled('hybrid_c1_gluon', '1'):
        from types import SimpleNamespace
        from jetspec.engines.jetspec_plus_plus.kernels import gluon_block
        shaped_key = key.view(*key.shape[:2], heads, dim)
        if gluon_block.available(shaped_key):
            return gluon_block.write_kv(
                shaped_key, value,
                SimpleNamespace(weight=weight, variance_epsilon=eps), cos, sin,
                cache, slots, groups)
    k, v = cache
    tokens = key.shape[1]
    if (key.device.type != "cuda" or key.dtype != torch.bfloat16 or triton is None
            or not specialized or key.shape[0] != 1):
        batch = key.shape[0]
        normalized = _reference_rms_norm(key.view(batch, tokens, heads, dim), weight, eps, norm_offset, rounding)
        rotated = _reference_rotary(normalized.transpose(1, 2), cos, sin)
        k.index_copy_(2, slots, rotated.repeat_interleave(groups, dim=1))
        v.index_copy_(2, slots, value.view(batch, tokens, heads, dim).transpose(1, 2).repeat_interleave(groups, dim=1))
        return
    launch_cached(_q_norm_rotary, (tokens * heads,),
                  (key, weight, cos.contiguous(), sin.contiguous(), k,
                   tokens, heads, dim, eps, triton.next_power_of_2(dim),
                   norm_offset, rotary_dim or dim, False, key.stride(-2),
                   value, v, slots, True, groups, k.shape[2], value.stride(-2),
                   _module_reduce_threads(dim, tokens * heads), ulp),
                  num_warps=4, enable_fp_fusion=False)


def mlp_epilogue(fused, mlp, rows, packed=None):
    if packed is None:
        gate, up = mlp.gate_proj(rows), mlp.up_proj(rows)
    else:
        gate, up = F.linear(rows, *packed).chunk(2, dim=-1)
    if (rows.device.type == "cuda" and rows.dtype == torch.bfloat16
            and gate.dtype == torch.bfloat16 and up.dtype == torch.bfloat16):
        activated = fused._silu_mul_extension().silu_mul(gate, up)
    else:
        activated = mlp.act_fn(gate) * up
    return mlp.down_proj(activated)


def adapter_query(head, embeddings, cos, sin, mask, length, window_slots=None,
                        stop_before_attention=False, resume=None):
    """Execute the dense bound-weight plan with per-block norm/rotary/gate geometry.

    Query and refresh use the same executor. The legacy dense specializations
    retain their default dispatch; this executor also supports their parameters.
    """
    if embeddings.shape[0] != 1:
        return head._adapter_cache_query(embeddings, cos, sin, mask, length, window_slots)
    plan = getattr(head, "_d1_variant_plan", None)
    if plan is None:
        layers = [layer for index, stream in enumerate(head.streams)
                  for layer in (stream, *head.extra_stream_layers[index])]
        plan = []
        for bound, layer in zip(head._d1_lean_plan(), layers):
            offset = layer.attention_spec.norm_offset
            rotary = int(layer.rotary_emb.inv_freq.numel() * 2)
            kv = getattr(layer, "_fused_kv", None)
            if kv is None:
                kv = (torch.cat((bound[8], bound[11])), None,
                      (bound[8].shape[0], bound[11].shape[0]))
            plan.append(dict(bound=bound, offset=offset, rotary=rotary,
                             norm_round=layer.attention_spec.norm_round,
                             gated=layer.attn_output_gate, kv=kv,
                             gate_up=torch.cat((bound[15], bound[16]))))
        head._d1_variant_plan = plan
    count = embeddings.shape[1]
    causal = head.resolved_config.get("window_causal_over_tokens", False)
    total = mask.shape[-1] if window_slots is not None else length + (count if causal else 0)
    slots = window_slots
    if slots is None and causal:
        slots = torch.arange(length, total, device=embeddings.device, dtype=torch.long)
    u = embeddings[0]
    window = u
    outputs = []
    pending = None
    current = plan[0]["bound"][0]
    first = 0
    empty = u.new_empty((0, u.shape[-1]))
    kv_sources = None
    kv_first = 0
    for block in range(first, len(plan)):
        item = plan[block]
        (index, in_w, in_eps, q_w, qn_w, qn_eps, kvn_w, kvn_eps, k_w, kn_w, kn_eps,
         v_w, o_w, post_w, post_eps, gate_w, up_w, down_w, heads, kv_heads,
         groups, dim, scaling) = item["bound"]
        offset, rotary, gated = item["offset"], item["rotary"], item["gated"]
        new_group = kv_sources is None or index != current
        weights = empty
        if new_group and causal:
            span = 1
            while block + span < len(plan) and plan[block + span]["bound"][0] == index:
                span += 1
            # Cache these immutable stacked weights before graph replay.
            weights = item.get("kv_norms")
            if weights is None:
                weights = torch.stack([p["bound"][6] for p in plan[block:block + span]])
                item["kv_norms"] = weights
            kv_first = block
        normed, u, sources = ew_pre_norm(u, window, in_w, weights, in_eps, kvn_eps,
            pending, kv_from_state=index != current, norm_offset=offset, ulp=bool(offset),
            norm_round=item["norm_round"])
        if new_group:
            kv_sources = sources
        if index != current:
            outputs.append(u)
            window, current = u, index
        projected_q = F.linear(normed, q_w)[None]
        gate = (projected_q.view(1, count, heads, dim * 2)[..., dim:] if gated else None)
        q = q_norm_rotary(head._c1_fused_module(), projected_q, qn_w, cos, sin,
                          qn_eps, heads, dim, offset, rotary, gated, norm_round=item["norm_round"])
        if causal:
            key, value = F.linear(kv_sources[block - kv_first], *item["kv"][:2]).split(
                item["kv"][2], dim=-1)
            kv_norm_rotary_cache(key[None], value[None], kn_w, cos, sin, kn_eps,
                                 kv_heads, groups, dim, head._c1_kv[block], slots,
                                 offset, rotary, norm_round=item["norm_round"])
        k, v = head._c1_kv[block]
        attended = F.scaled_dot_product_attention(q, k[:, :, :total], v[:, :, :total],
                                                  attn_mask=mask, scale=scaling)
        attended = attended.transpose(1, 2).reshape(1, count, heads * dim)
        if gate is not None:
            from jetspec.engines.jetspec_plus_plus.kernels.hybrid_c1_lean import gate_mul
            attended = gate_mul(attended, gate, heads, dim)
        projected = F.linear(attended[0], o_w)
        post, state, _ = ew_pre_norm(u, window, post_w, empty, post_eps, kvn_eps,
                                     pending=(projected, u), norm_offset=offset, ulp=bool(offset),
                                     norm_round=item["norm_round"])
        gate_rows, up_rows = F.linear(post, item["gate_up"]).chunk(2, dim=-1)
        if u.device.type == "cuda" and u.dtype == torch.bfloat16:
            activated = (ulp_silu_mul(gate_rows, up_rows) if offset else
                         _silu_mul_extension().silu_mul(gate_rows, up_rows))
        else:
            activated = F.silu(gate_rows) * up_rows
        pending = (F.linear(activated, down_w), state)
    outputs.append(pending[1] + pending[0])
    return torch.cat(outputs, dim=-1)[None]


def adapter_cache_extend(head, dirty, cos, sin, slots):
    """Fuse hybrid prefix norms, rotary and repeated-head writes into live slots."""
    tokens = dirty.shape[1]
    empty = dirty.new_empty((0, head.hidden_size))
    for block, (index, layer) in enumerate(head._c1_blocks):
        rows = dirty[0, :, index * head.hidden_size:(index + 1) * head.hidden_size].contiguous()
        normalized, _, _ = ew_pre_norm(
            rows, rows, layer.kv_norm.weight, empty,
            layer.kv_norm.variance_epsilon, layer.kv_norm.variance_epsilon,
            norm_offset=layer.attention_spec.norm_offset, ulp=True,
            norm_round=layer.attention_spec.norm_round)
        key, value = layer.project_kv(normalized[None])
        kv_norm_rotary_cache(
            key, value, layer.k_norm.weight, cos, sin,
            layer.k_norm.variance_epsilon, layer.num_key_value_heads,
            layer.num_key_value_groups, layer.head_dim, head._c1_kv[block],
            slots, norm_offset=layer.attention_spec.norm_offset, rotary_dim=cos.shape[-1],
            norm_round=layer.attention_spec.norm_round)


# Compatibility imports for existing qualification tools.
lean_variant_query = adapter_query
hybrid_cache_extend = adapter_cache_extend


if triton is not None:
    @triton.jit
    def _ulp_silu_up(GATE, UP, OUT, N: tl.constexpr, WIDTH: tl.constexpr,
                     GS: tl.constexpr, US: tl.constexpr, BLOCK: tl.constexpr):
        index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        row, column = index // WIDTH, index % WIDTH
        gate = tl.load(GATE + row * GS + column, index < N, 0).to(tl.float32)
        up = tl.load(UP + row * US + column, index < N, 0).to(tl.float32)
        activated = (gate / (1.0 + libdevice.exp(-gate))).to(tl.bfloat16).to(tl.float32)
        tl.store(OUT + index, activated * up, index < N)


def ulp_silu_mul(gate, up):
    """Hybrid-only Triton SiLU/up epilogue, preserving the two BF16 roundings."""
    if triton is None or gate.device.type != 'cuda' or gate.dtype != torch.bfloat16:
        return F.silu(gate) * up
    if gate.shape != up.shape or gate.stride(-1) != 1 or up.stride(-1) != 1:
        raise ValueError('ULP SiLU requires matching tensors with contiguous rows')
    out = torch.empty_like(gate, memory_format=torch.contiguous_format)
    _ulp_silu_up[(triton.cdiv(gate.numel(), 256),)](gate, up, out, gate.numel(),
        gate.shape[-1], gate.stride(-2), up.stride(-2), 256, enable_fp_fusion=False)
    return out


def stacked_kv(fused, tile, window, weight, bias, kv_weight, kv_eps,
               key_weight, key_eps, cos, sin, kv_heads, head_dim,
               cache=None, write_slots=None, tile_positions=None):
    """Project both row sets through one K/V GEMM, then restore their layouts."""
    batch, tokens, hidden = window.shape
    tile_count = tile.shape[0]
    if tile.device.type == "cuda" and tile.dtype == torch.bfloat16 and triton is not None:
        source = tile.new_empty(tile_count + batch * tokens, hidden)
        _stacked_kv_norm[(tile_count + batch * tokens,)](
            tile, window, kv_weight, source, tile_count, tokens, hidden, kv_eps,
            *tile.stride(), *window.stride(), triton.next_power_of_2(hidden),
            num_warps=8 if hidden >= 4096 else 4)
    else:
        rows = torch.cat((tile, window.reshape(batch * tokens, hidden)), dim=0)
        source = fused.fast_rms_norm(rows, kv_weight, kv_eps)
    projected = F.linear(source, weight, bias)[None]
    if projected.device.type == "cuda" and projected.dtype == torch.bfloat16 and triton is not None:
        direct_cache = cache is not None
        key = projected.new_empty(batch, kv_heads, tokens, head_dim) if direct_cache else \
            projected.new_empty(1, kv_heads, tile_count + batch * tokens, head_dim)
        value = torch.empty_like(key)
        cache_k, cache_v = cache if direct_cache else (key, value)
        _kv_norm_rotary[((tile_count + batch * tokens) * kv_heads,)](
            projected, key_weight, cos.contiguous(), sin.contiguous(), key, value,
            cache_k, cache_v, write_slots if direct_cache else projected,
            tile_positions if direct_cache else projected,
            tile_count + batch * tokens, kv_heads, head_dim, key_eps,
            triton.next_power_of_2(head_dim), direct_cache, tile_count, tokens,
            *cache_k.stride()[:3], *cache_v.stride()[:3],
            num_warps=4, enable_fp_fusion=False)
        if direct_cache:
            return None, None, key, value
    else:
        key_part, value_part = projected.split(kv_heads * head_dim, dim=-1)
        key, value = fused.fast_kv_postprocess(
            key_part, value_part, key_weight, cos, sin, key_eps, kv_heads, 1, head_dim)
    tile_key, tile_value = key[:, :, :tile_count], value[:, :, :tile_count]
    window_key = key[0, :, tile_count:].reshape(kv_heads, batch, tokens, head_dim)
    window_value = value[0, :, tile_count:].reshape(kv_heads, batch, tokens, head_dim)
    return (tile_key, tile_value, window_key.permute(1, 0, 2, 3),
            window_value.permute(1, 0, 2, 3))
