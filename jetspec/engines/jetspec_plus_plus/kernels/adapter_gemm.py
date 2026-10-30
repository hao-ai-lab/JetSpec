"""Graph-tuned cuBLASLt plans for small-row SM100 adapter projections."""
from functools import lru_cache

import torch
from torch.nn import functional as F


_CPP = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cublasLt.h>
#include <vector>
#include <cstring>

static void check(cublasStatus_t rc) {
  TORCH_CHECK(rc == CUBLAS_STATUS_SUCCESS, "adapter cuBLASLt status ", int(rc));
}

struct Plan {
  cublasLtMatmulDesc_t op;
  cublasLtMatrixLayout_t a, b, c;
  cublasLtMatmulAlgo_t algo;
  torch::Tensor workspace;
  int64_t m, n, k;
  int device;
  int algorithm_id = -1;
  float selected_ms = 0;
  float baseline_ms = 0;
  bool use_linear = true;

  Plan(torch::Tensor x, torch::Tensor w, std::vector<uint8_t> pinned = {}, int replay = -1):
                                        m(x.size(0)), n(w.size(0)), k(w.size(1)),
                                        device(x.get_device()) {
    c10::cuda::CUDAGuard guard(x.device());
    auto caller = at::cuda::getCurrentCUDAStream();
    TORCH_CHECK(cudaStreamSynchronize(caller) == cudaSuccess, "adapter warmup sync");
    auto stream = at::cuda::getStreamFromPool(false, device);
    c10::cuda::CUDAStreamGuard stream_guard(stream);
    auto handle = at::cuda::getCurrentCUDABlasLtHandle();
    check(cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    cublasOperation_t trans = CUBLAS_OP_T;
    check(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &trans, sizeof(trans)));
    check(cublasLtMatrixLayoutCreate(&a, CUDA_R_16BF, k, n, k));
    check(cublasLtMatrixLayoutCreate(&b, CUDA_R_16BF, k, m, k));
    check(cublasLtMatrixLayoutCreate(&c, CUDA_R_16BF, n, m, n));
    constexpr size_t workspace_bytes = 32 * 1024 * 1024;
    workspace = at::empty({int64_t(workspace_bytes)}, x.options().dtype(at::kByte));
    if (replay >= 0) {
      use_linear = replay == 1;
      if (!use_linear) {
        TORCH_CHECK(pinned.size() == sizeof(algo), "adapter pinned algorithm size mismatch");
        std::memcpy(&algo, pinned.data(), sizeof(algo));
        cublasLtMatmulHeuristicResult_t valid;
        check(cublasLtMatmulAlgoCheck(handle, op, a, b, c, c, &algo, &valid));
        TORCH_CHECK(valid.state == CUBLAS_STATUS_SUCCESS && valid.workspaceSize <= workspace_bytes,
                    "adapter pinned algorithm unsupported");
        size_t written = 0;
        check(cublasLtMatmulAlgoConfigGetAttribute(&algo, CUBLASLT_ALGO_CONFIG_ID,
                                                 &algorithm_id, sizeof(algorithm_id), &written));
      }
      return;
    }
    cublasLtMatmulPreference_t pref;
    check(cublasLtMatmulPreferenceCreate(&pref));
    check(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                             &workspace_bytes, sizeof(workspace_bytes)));
    std::vector<cublasLtMatmulAlgo_t> candidates;
    // Narrow-row heuristics can expose a useful tile omitted at the actual M.
    // Recheck every returned plan with the actual operand layouts and workspace.
    for (int rows : {4, 16, 32, int(m), 64, 80}) {
      cublasLtMatrixLayout_t small_b, small_c;
      check(cublasLtMatrixLayoutCreate(&small_b, CUDA_R_16BF, k, rows, k));
      check(cublasLtMatrixLayoutCreate(&small_c, CUDA_R_16BF, n, rows, n));
      cublasLtMatmulHeuristicResult_t results[128];
      int count = 0;
      check(cublasLtMatmulAlgoGetHeuristic(handle, op, a, small_b, small_c, small_c,
                                         pref, 128, results, &count));
      for (int i = 0; i < count; ++i) {
        auto candidate = results[i].algo;
        cublasLtMatmulHeuristicResult_t valid;
        if (cublasLtMatmulAlgoCheck(handle, op, a, b, c, c, &candidate, &valid)
              == CUBLAS_STATUS_SUCCESS && valid.state == CUBLAS_STATUS_SUCCESS
              && valid.workspaceSize <= workspace_bytes) candidates.push_back(candidate);
      }
      cublasLtMatrixLayoutDestroy(small_b);
      cublasLtMatrixLayoutDestroy(small_c);
    }
    cublasLtMatmulPreferenceDestroy(pref);
    auto out = at::empty({m, n}, x.options());
    const float alpha = 1, beta = 0;
    cudaEvent_t begin, end;
    TORCH_CHECK(cudaEventCreate(&begin) == cudaSuccess && cudaEventCreate(&end) == cudaSuccess,
                "adapter GEMM events");
    auto measure = [&](auto launch) {
      cudaGraph_t graph;
      cudaGraphExec_t replay;
      TORCH_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal) == cudaSuccess,
                  "adapter tune capture");
      for (int repeat = 0; repeat < 10; ++repeat) launch();
      TORCH_CHECK(cudaStreamEndCapture(stream, &graph) == cudaSuccess, "adapter tune end");
      TORCH_CHECK(cudaGraphInstantiate(&replay, graph, 0) == cudaSuccess, "adapter instantiate");
      for (int repeat = 0; repeat < 3; ++repeat) cudaGraphLaunch(replay, stream);
      cudaEventRecord(begin, stream);
      for (int repeat = 0; repeat < 10; ++repeat) cudaGraphLaunch(replay, stream);
      cudaEventRecord(end, stream);
      TORCH_CHECK(cudaEventSynchronize(end) == cudaSuccess, "adapter tune sync");
      float ms = 0;
      cudaEventElapsedTime(&ms, begin, end);
      cudaGraphExecDestroy(replay); cudaGraphDestroy(graph);
      return ms / 100;
    };
    float best = 1e30f;
    for (auto candidate : candidates) {
      auto launch = [&]() { return cublasLtMatmul(handle, op, &alpha, w.data_ptr(), a,
          x.data_ptr(), b, &beta, out.data_ptr(), c, out.data_ptr(), c,
          &candidate, workspace.data_ptr(), workspace_bytes, stream); };
      if (launch() != CUBLAS_STATUS_SUCCESS) continue;
      float ms = measure([&]() { check(launch()); });
      if (ms < best) { best = ms; algo = candidate; }
    }
    selected_ms = best;
    auto wt = w.t();
    for (int repeat = 0; repeat < 3; ++repeat) at::mm_out(out, x, wt);
    baseline_ms = measure([&]() { at::mm_out(out, x, wt); });
    cudaEventDestroy(begin); cudaEventDestroy(end);
    use_linear = best >= baseline_ms * .98f;
    if (!use_linear) {
      size_t written = 0;
      check(cublasLtMatmulAlgoConfigGetAttribute(&algo, CUBLASLT_ALGO_CONFIG_ID,
                                               &algorithm_id, sizeof(algorithm_id), &written));
    }
  }

  ~Plan() {
    cublasLtMatrixLayoutDestroy(a); cublasLtMatrixLayoutDestroy(b);
    cublasLtMatrixLayoutDestroy(c); cublasLtMatmulDescDestroy(op);
  }

  std::vector<uint8_t> algorithm() const {
    if (use_linear) return {};
    const auto *bytes = reinterpret_cast<const uint8_t *>(&algo);
    return std::vector<uint8_t>(bytes, bytes + sizeof(algo));
  }

  torch::Tensor linear(torch::Tensor x, torch::Tensor w) {
    TORCH_CHECK(x.is_cuda() && w.is_cuda() && x.get_device() == device && w.get_device() == device
        && x.scalar_type() == at::kBFloat16 && w.scalar_type() == at::kBFloat16
        && x.dim() == 2 && w.dim() == 2 && x.is_contiguous() && w.is_contiguous()
        && x.size(0) == m && x.size(1) == k && w.size(0) == n && w.size(1) == k,
        "adapter plan operand mismatch");
    c10::cuda::CUDAGuard guard(x.device());
    if (use_linear) return at::linear(x, w);
    auto out = at::empty({m, n}, x.options());
    const float alpha = 1, beta = 0;
    check(cublasLtMatmul(at::cuda::getCurrentCUDABlasLtHandle(), op, &alpha, w.data_ptr(), a,
        x.data_ptr(), b, &beta, out.data_ptr(), c, out.data_ptr(), c, &algo,
        workspace.data_ptr(), workspace.numel(), at::cuda::getCurrentCUDAStream()));
    return out;
  }
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  pybind11::class_<Plan>(module, "Plan")
    .def(pybind11::init<torch::Tensor, torch::Tensor>())
    .def(pybind11::init<torch::Tensor, torch::Tensor, std::vector<uint8_t>, int>())
    .def("algorithm", &Plan::algorithm)
    .def("linear", &Plan::linear)
    .def_readonly("algorithm_id", &Plan::algorithm_id)
    .def_readonly("use_linear", &Plan::use_linear)
    .def_readonly("selected_ms", &Plan::selected_ms)
    .def_readonly("baseline_ms", &Plan::baseline_ms);
}
'''


@lru_cache(None)
def _extension():
    from torch.utils.cpp_extension import load_inline
    from jetspec.engines.jetspec_plus_plus.kernels.c1_fused import _torch_cublas_ldflags

    return load_inline(name="jetspec_adapter_gemm_w602", cpp_sources=_CPP,
                       with_cuda=True, extra_cflags=["-O3"],
                       extra_ldflags=_torch_cublas_ldflags() + ["-l:libcublasLt.so.12"])


class AdapterGemm:
    """Warm a plan per dense shape before graph capture; FP32 keeps ATen."""

    def __init__(self, device):
        self.device = torch.device(device)
        self.enabled = (self.device.type == "cuda"
                        and torch.cuda.get_device_capability(self.device) == (10, 0))
        self.plans = {}

    def linear(self, x, weight, bias=None):
        if (not self.enabled or x.dtype != torch.bfloat16 or weight.dtype != x.dtype
                or x.device != self.device or weight.device != self.device
                or bias is not None or x.ndim != 2 or weight.ndim != 2
                or x.shape[0] != 64 or tuple(weight.shape) != (24576, 4096)
                or x.shape[1] != weight.shape[1]
                or not x.is_contiguous() or not weight.is_contiguous()):
            return F.linear(x, weight, bias)
        key = (x.shape[0], weight.shape[0], weight.shape[1])
        plan = self.plans.get(key)
        if plan is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("adapter GEMM must warm its shape before capture")
            from jetspec.engines.jetspec_plus_plus.serving.compile_pin import tuning_choice
            recorded = []
            def choose():
                candidate = _extension().Plan(x, weight)
                recorded.append(candidate)
                return dict(algorithm_id=candidate.algorithm_id, use_linear=candidate.use_linear,
                            algorithm=candidate.algorithm())
            choice = tuning_choice('adapter_gemm', [*key, str(x.dtype),
                list(torch.cuda.get_device_capability(x.device))], choose)
            plan = recorded[0] if recorded else _extension().Plan(
                x, weight, choice['algorithm'], int(choice['use_linear']))
            if plan.algorithm_id != choice['algorithm_id'] or plan.use_linear != choice['use_linear']:
                raise RuntimeError('C1 COMPILE PIN adapter algorithm mismatch')
            self.plans[key] = plan
        return plan.linear(x, weight)


def _fixture(shape, dtype, seed):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn((shape["M"], shape["K"]), device="cuda", dtype=dtype, generator=generator) * .2
    weight = torch.randn((shape["N"], shape["K"]), device="cuda", dtype=dtype, generator=generator) * .02
    return x, weight


def reference(shape, dtype, seed):
    x, weight = _fixture(shape, dtype, seed)
    return lambda: {"output": F.linear(x, weight)}


def candidate(shape, dtype, seed):
    x, weight = _fixture(shape, dtype, seed)
    gemm = AdapterGemm(x.device)
    gemm.linear(x, weight)
    for key, plan in gemm.plans.items():
        print("W602_PLAN", key, "algo", plan.algorithm_id, "aten", plan.use_linear,
              "selected_ms", plan.selected_ms, "baseline_ms", plan.baseline_ms, flush=True)
    return lambda: {"output": gemm.linear(x, weight)}
