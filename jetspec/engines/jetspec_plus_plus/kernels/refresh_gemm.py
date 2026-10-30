"""SM80 refresh GEMMs: select a no-split-K cuBLASLt plan before capture."""
from functools import lru_cache

import torch
from torch.nn import functional as F


_CPP = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cublasLt.h>
#include <vector>

static void check(cublasStatus_t rc) {
  TORCH_CHECK(rc == CUBLAS_STATUS_SUCCESS, "refresh cuBLASLt status ", int(rc));
}

struct Plan {
  cublasLtMatmulDesc_t op;
  cublasLtMatrixLayout_t a, b, c;
  cublasLtMatmulAlgo_t algo;
  int64_t m, n, k;
  int device;
  int algorithm_id = -1;
  float selected_ms = 0;
  bool use_linear = false;

  Plan(torch::Tensor x, torch::Tensor w): m(x.size(0)), n(w.size(0)), k(w.size(1)),
                                        device(x.get_device()) {
    c10::cuda::CUDAGuard guard(x.device());
    auto handle = at::cuda::getCurrentCUDABlasLtHandle();
    TORCH_CHECK(cudaStreamSynchronize(at::cuda::getCurrentCUDAStream()) == cudaSuccess,
                "refresh GEMM warmup sync");
    auto stream = at::cuda::getStreamFromPool(false, device);
    c10::cuda::CUDAStreamGuard stream_guard(stream);
    check(cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    cublasOperation_t trans = CUBLAS_OP_T;
    check(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &trans, sizeof(trans)));
    check(cublasLtMatrixLayoutCreate(&a, CUDA_R_16BF, k, n, k));
    check(cublasLtMatrixLayoutCreate(&b, CUDA_R_16BF, k, m, k));
    check(cublasLtMatrixLayoutCreate(&c, CUDA_R_16BF, n, m, n));
    cublasLtMatmulPreference_t pref;
    check(cublasLtMatmulPreferenceCreate(&pref));
    size_t workspace = 0;
    check(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                             &workspace, sizeof(workspace)));
    std::vector<cublasLtMatmulAlgo_t> candidates;
    // Include Seg A's row count as well as refresh's; check every plan at the actual M.
    for (int rows : {16, int(m)}) {
      cublasLtMatrixLayout_t small_b, small_c;
      check(cublasLtMatrixLayoutCreate(&small_b, CUDA_R_16BF, k, rows, k));
      check(cublasLtMatrixLayoutCreate(&small_c, CUDA_R_16BF, n, rows, n));
      cublasLtMatmulHeuristicResult_t results[128];
      int count = 0;
      check(cublasLtMatmulAlgoGetHeuristic(handle, op, a, small_b, small_c, small_c,
                                         pref, 128, results, &count));
      for (int i = 0; i < count; ++i) {
        auto candidate = results[i].algo;
        uint32_t splits = 0;
        size_t written = 0;
        check(cublasLtMatmulAlgoConfigGetAttribute(&candidate, CUBLASLT_ALGO_CONFIG_SPLITK_NUM,
                                                 &splits, sizeof(splits), &written));
        if (splits > 1) continue;
        cublasLtMatmulHeuristicResult_t valid;
        if (cublasLtMatmulAlgoCheck(handle, op, a, b, c, c, &candidate, &valid)
              == CUBLAS_STATUS_SUCCESS && valid.state == CUBLAS_STATUS_SUCCESS
              && valid.workspaceSize == 0) candidates.push_back(candidate);
      }
      cublasLtMatrixLayoutDestroy(small_b);
      cublasLtMatrixLayoutDestroy(small_c);
    }
    cublasLtMatmulPreferenceDestroy(pref);
    // Heuristics can omit useful narrow tiles for refresh M. Enumerate supported
    // tiles, stages and custom sliced variants, checking the actual operands.
    int ids[512], id_count = 0;
    check(cublasLtMatmulAlgoGetIds(handle, CUBLAS_COMPUTE_32F, CUDA_R_32F,
        CUDA_R_16BF, CUDA_R_16BF, CUDA_R_16BF, CUDA_R_16BF, 512, ids, &id_count));
    for (int i = 0; i < id_count; ++i) {
      cublasLtMatmulAlgo_t base;
      if (cublasLtMatmulAlgoInit(handle, CUBLAS_COMPUTE_32F, CUDA_R_32F,
          CUDA_R_16BF, CUDA_R_16BF, CUDA_R_16BF, CUDA_R_16BF, ids[i], &base)
          != CUBLAS_STATUS_SUCCESS) continue;
      size_t bytes = 0;
      cublasLtMatmulAlgoCapGetAttribute(&base, CUBLASLT_ALGO_CAP_TILE_IDS, nullptr, 0, &bytes);
      std::vector<uint32_t> tiles(bytes / sizeof(uint32_t));
      if (bytes) cublasLtMatmulAlgoCapGetAttribute(&base, CUBLASLT_ALGO_CAP_TILE_IDS,
                                                 tiles.data(), bytes, &bytes);
      if (tiles.empty()) tiles.push_back(0);
      cublasLtMatmulAlgoCapGetAttribute(&base, CUBLASLT_ALGO_CAP_STAGES_IDS, nullptr, 0, &bytes);
      std::vector<uint32_t> stages(bytes / sizeof(uint32_t));
      if (bytes) cublasLtMatmulAlgoCapGetAttribute(&base, CUBLASLT_ALGO_CAP_STAGES_IDS,
                                                 stages.data(), bytes, &bytes);
      if (stages.empty()) stages.push_back(0);
      int custom_max = 0;
      cublasLtMatmulAlgoCapGetAttribute(&base, CUBLASLT_ALGO_CAP_CUSTOM_OPTION_MAX,
                                      &custom_max, sizeof(custom_max), &bytes);
      for (uint32_t tile : tiles) for (uint32_t stage : stages)
          for (int custom = 0; custom <= custom_max; ++custom) {
        auto candidate = base;
        uint32_t split = 0;
        if (cublasLtMatmulAlgoConfigSetAttribute(&candidate, CUBLASLT_ALGO_CONFIG_TILE_ID,
            &tile, sizeof(tile)) != CUBLAS_STATUS_SUCCESS) continue;
        cublasLtMatmulAlgoConfigSetAttribute(&candidate, CUBLASLT_ALGO_CONFIG_STAGES_ID,
                                            &stage, sizeof(stage));
        cublasLtMatmulAlgoConfigSetAttribute(&candidate, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION,
                                            &custom, sizeof(custom));
        cublasLtMatmulAlgoConfigSetAttribute(&candidate, CUBLASLT_ALGO_CONFIG_SPLITK_NUM,
                                            &split, sizeof(split));
        cublasLtMatmulHeuristicResult_t valid;
        if (cublasLtMatmulAlgoCheck(handle, op, a, b, c, c, &candidate, &valid)
              == CUBLAS_STATUS_SUCCESS && valid.state == CUBLAS_STATUS_SUCCESS
              && valid.workspaceSize == 0) candidates.push_back(candidate);
      }
    }
    TORCH_CHECK(!candidates.empty(), "refresh GEMM has no valid no-split-K plan");
    auto out = at::empty({m, n}, x.options());
    const float alpha = 1, beta = 0;
    cudaEvent_t begin, end;
    TORCH_CHECK(cudaEventCreate(&begin) == cudaSuccess && cudaEventCreate(&end) == cudaSuccess,
                "refresh GEMM events");
    // Compare captured GPU work; native dispatch gaps distort small GEMMs.
    auto measure = [&](auto launch) {
      cudaGraph_t graph;
      cudaGraphExec_t replay;
      TORCH_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal) == cudaSuccess,
                  "refresh GEMM tune capture");
      for (int repeat = 0; repeat < 10; ++repeat) launch();
      TORCH_CHECK(cudaStreamEndCapture(stream, &graph) == cudaSuccess, "refresh GEMM tune end");
      TORCH_CHECK(cudaGraphInstantiate(&replay, graph, 0) == cudaSuccess,
                  "refresh GEMM tune instantiate");
      cudaEventRecord(begin, stream);
      for (int repeat = 0; repeat < 10; ++repeat) cudaGraphLaunch(replay, stream);
      cudaEventRecord(end, stream);
      TORCH_CHECK(cudaEventSynchronize(end) == cudaSuccess, "refresh GEMM tune sync");
      float ms = 0;
      cudaEventElapsedTime(&ms, begin, end);
      cudaGraphExecDestroy(replay); cudaGraphDestroy(graph);
      return ms / 100;
    };
    float best = 1e30f;
    for (auto candidate : candidates) {
      auto launch = [&]() { return cublasLtMatmul(handle, op, &alpha, w.data_ptr(), a,
          x.data_ptr(), b, &beta, out.data_ptr(), c, out.data_ptr(), c,
          &candidate, nullptr, 0, stream); };
      if (launch() != CUBLAS_STATUS_SUCCESS) continue;
      float ms = measure([&]() { check(launch()); });
      if (ms < best) { best = ms; algo = candidate; }
    }
    TORCH_CHECK(best < 1e30f, "refresh GEMM tuning failed");
    selected_ms = best;
    // Retain the current projection if removing split-K costs more than it saves.
    auto wt = w.t();
    for (int repeat = 0; repeat < 3; ++repeat) at::mm_out(out, x, wt);
    float baseline = measure([&]() { at::mm_out(out, x, wt); });
    cudaEventDestroy(begin); cudaEventDestroy(end);
    use_linear = best >= baseline * .995f;
    size_t written = 0;
    check(cublasLtMatmulAlgoConfigGetAttribute(&algo, CUBLASLT_ALGO_CONFIG_ID,
                                             &algorithm_id, sizeof(algorithm_id), &written));
  }

  ~Plan() {
    cublasLtMatrixLayoutDestroy(a); cublasLtMatrixLayoutDestroy(b);
    cublasLtMatrixLayoutDestroy(c); cublasLtMatmulDescDestroy(op);
  }

  torch::Tensor linear(torch::Tensor x, torch::Tensor w) {
    TORCH_CHECK(x.is_cuda() && w.is_cuda() && x.get_device() == device && w.get_device() == device
        && x.scalar_type() == at::kBFloat16 && w.scalar_type() == at::kBFloat16
        && x.dim() == 2 && w.dim() == 2 && x.is_contiguous() && w.is_contiguous()
        && x.size(0) == m && x.size(1) == k && w.size(0) == n && w.size(1) == k,
        "refresh GEMM plan operand mismatch");
    c10::cuda::CUDAGuard guard(x.device());
    if (use_linear) return at::linear(x, w);
    auto out = at::empty({m, n}, x.options());
    const float alpha = 1, beta = 0;
    check(cublasLtMatmul(at::cuda::getCurrentCUDABlasLtHandle(), op, &alpha, w.data_ptr(), a,
        x.data_ptr(), b, &beta, out.data_ptr(), c, out.data_ptr(), c, &algo,
        nullptr, 0, at::cuda::getCurrentCUDAStream()));
    return out;
  }
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  pybind11::class_<Plan>(module, "Plan")
    .def(pybind11::init<torch::Tensor, torch::Tensor>())
    .def("linear", &Plan::linear)
    .def_readonly("algorithm_id", &Plan::algorithm_id)
    .def_readonly("use_linear", &Plan::use_linear)
    .def_readonly("selected_ms", &Plan::selected_ms);
}
'''


@lru_cache(None)
def _extension():
    from torch.utils.cpp_extension import load_inline
    from jetspec.engines.jetspec_plus_plus.kernels.c1_fused import _torch_cublas_ldflags

    return load_inline(name='jetspec_refresh_gemm_w560', cpp_sources=_CPP,
                       with_cuda=True, extra_cflags=['-O3'],
                       extra_ldflags=_torch_cublas_ldflags() + ['-l:libcublasLt.so.12'])


class RefreshGemm:
    """Bind once per refresh body; cold plan creation is forbidden during capture."""

    def __init__(self, device, *, max_num_seqs=1):
        if max_num_seqs != 1:
            raise ValueError('refresh GEMM tuning requires N=1')
        self.device = torch.device(device)
        self.enabled = (self.device.type == 'cuda'
                        and torch.cuda.get_device_capability(self.device) == (8, 0))
        self.plans = {}

    def linear(self, x, w):
        if (x.ndim != 2 or w.ndim != 2 or x.shape[1] != w.shape[1]
                or not x.is_contiguous() or not w.is_contiguous()):
            raise ValueError('refresh GEMM requires dense matching 2D operands')
        if x.dtype != w.dtype or x.device != w.device or x.device != self.device:
            raise ValueError('refresh GEMM operands require matching dtype and device')
        if not self.enabled or x.dtype != torch.bfloat16:
            return F.linear(x, w)
        key = (x.shape[0], w.shape[0], w.shape[1])
        plan = self.plans.get(key)
        if plan is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError('refresh GEMM must warm its shape before capture')
            plan = self.plans[key] = _extension().Plan(x, w)
        return plan.linear(x, w)
