"""Fixed-refresh integer layout and bitwise gathers, with no activation arithmetic."""

import torch
from dataclasses import replace
from unittest.mock import patch
from jetspec.engines.jetspec_plus_plus.draft.method.ops import reference
from jetspec.engines.jetspec_plus_plus.draft.method import refresh
from . import register_optimization

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None


if triton is not None:
    @triton.jit
    def _bookkeeping(CHAINED, ACTIVE, KEYS, LENGTH, KHAT, COUNTS, TAILS, START,
                     D: tl.constexpr, R: tl.constexpr, WINDOW: tl.constexpr,
                     THRESHOLD: tl.constexpr, CAP: tl.constexpr,
                     K0: tl.constexpr, K1: tl.constexpr, B: tl.constexpr):
        row = tl.program_id(0)
        col = tl.arange(0, B)
        value = tl.load(CHAINED + row * D + col, col < D, other=1)
        threshold = tl.full((), THRESHOLD, tl.float64)
        khat = tl.min(tl.where((col < D) & (value < threshold), col + 1, D), 0)
        if CAP >= 0:
            khat = tl.minimum(khat, CAP)
        active = tl.load(ACTIVE + row)
        khat = tl.where(active, khat, 0)
        tl.store(KHAT + row, khat)
        tl.store(COUNTS + row, tl.where(active, tl.minimum(khat, D - 1) + 1, 0))
        tl.store(TAILS + row, tl.load(KEYS + row * K0) + 1)
        if row == 0:
            tl.store(START, tl.load(LENGTH) + WINDOW)

    @triton.jit
    def _prepare(EMBED, LENGTH, TAILS, COUNTS, CT, ST, DATA, COS, SIN,
                 ROW, SLOT, VALID, KOF, POS, STARTS,
                 R: tl.constexpr, W: tl.constexpr, H: tl.constexpr, D: tl.constexpr,
                 T: tl.constexpr, WINDOW: tl.constexpr,
                 E0: tl.constexpr, E1: tl.constexpr, E2: tl.constexpr,
                 C0: tl.constexpr, C1: tl.constexpr, S0: tl.constexpr, S1: tl.constexpr,
                 RB: tl.constexpr, B: tl.constexpr,
                 IDS=None, GATHER: tl.constexpr = False,
                 I0: tl.constexpr = 0, I1: tl.constexpr = 0):
        token = tl.program_id(0)
        part = tl.program_id(1)
        r = tl.arange(0, RB)
        count = tl.load(COUNTS + r, r < R, other=0).to(tl.int64)
        ends = tl.cumsum(count)
        row = tl.sum(((token >= ends) & (r < R)).to(tl.int64), 0)
        valid = row < R
        start = tl.sum(tl.where(r < row, count, 0), 0)
        slot = tl.where(valid, token - start, W)
        tail = tl.load(TAILS + row, valid, other=0).to(tl.int64)
        tail = tl.minimum(tl.maximum(tail, 0), WINDOW)
        length = tl.load(LENGTH).to(tl.int64)
        position = tl.where(valid, length + tail + slot, length)
        col = part * B + tl.arange(0, B)
        if GATHER:
            embedding_id = tl.load(IDS + row * I0 + slot * I1, valid, other=0)
            value = tl.load(EMBED + embedding_id * E0 + col * E2,
                            valid & (col < H), other=0)
        else:
            value = tl.load(EMBED + row * E0 + slot * E1 + col * E2,
                            valid & (col < H), other=0)
        tl.store(DATA + token * H + col, value, col < H)
        if part == 0:
            d = tl.arange(0, B)
            cos = tl.load(CT + position * C0 + d * C1, d < D, other=0)
            sin = tl.load(ST + position * S0 + d * S1, d < D, other=0)
            tl.store(COS + token * D + d, cos, d < D)
            tl.store(SIN + token * D + d, sin, d < D)
            tl.store(ROW + token, row)
            tl.store(SLOT + token, slot)
            tl.store(VALID + token, valid)
            tl.store(KOF + token, tail)
            tl.store(POS + token, position)
            tl.store(STARTS + token, tl.where(valid, start, T))

    @triton.jit
    def _scatter(OUTPUT, COUNTS, PADDED, R: tl.constexpr, W: tl.constexpr,
                 H: tl.constexpr, T: tl.constexpr, RB: tl.constexpr, B: tl.constexpr):
        flat = tl.program_id(0) * B + tl.arange(0, B)
        row = flat // ((W + 1) * H)
        slot = flat // H % (W + 1)
        col = flat % H
        start = tl.full((B,), 0, tl.int64)
        # Inverse gather writes zeros and real outputs together; filler writes never race.
        for r in tl.static_range(R):
            start += tl.where(r < row, tl.load(COUNTS + r).to(tl.int64), 0)
        count = tl.load(COUNTS + row, row < R, other=0)
        valid = (row < R) & (slot < count) & (slot < W) & (start + slot < T)
        value = tl.load(OUTPUT + (start + slot) * H + col, valid, other=0)
        tl.store(PADDED + flat, value, flat < (R + 1) * (W + 1) * H)


def bookkeeping(chained, active, keys, length, window, threshold, cap):
    """Fuse selection after the unchanged torch double cumprod rounding points."""
    rows, drafts = chained.shape
    khat = torch.empty(rows, device=keys.device, dtype=torch.long)
    counts, tails = torch.empty_like(khat), torch.empty_like(khat)
    start = torch.empty_like(length)
    if chained.is_cuda:
        if triton is None:
            raise RuntimeError('refresh bookkeeping requires CUDA Triton')
        _bookkeeping[(rows,)](chained, active, keys, length, khat, counts, tails, start,
                              drafts, rows, window, threshold, -1 if cap is None else cap,
                              *keys.stride(), triton.next_power_of_2(drafts))
    else:
        return reference('refresh_bookkeeping')(chained, active, keys, length, window, threshold, cap)
    return khat, counts, tails, start


register_optimization('refresh_bookkeeping', bookkeeping)


def prepare(embeddings, length, tails, counts, tokens, cos_table, sin_table, window,
            *, embedding_ids=None):
    """Pack rows, positions, attention metadata and table gathers in one launch."""
    if embedding_ids is not None:
        rows, width = embedding_ids.shape
        hidden = embeddings.shape[1]
    else:
        rows, width, hidden = embeddings.shape
    dim = cos_table.shape[1]
    result = {name: torch.empty(tokens, device=embeddings.device,
                               dtype=torch.bool if name == 'valid' else torch.long)
              for name in ('row', 'slot', 'valid', 'k_of', 'positions', 'starts')}
    result['data'] = embeddings.new_empty(tokens, hidden)
    result['cos'] = cos_table.new_empty(1, tokens, dim)
    result['sin'] = sin_table.new_empty(1, tokens, dim)
    if embeddings.is_cuda:
        if triton is None:
            raise RuntimeError('refresh prep requires CUDA Triton')
        block = max(256, triton.next_power_of_2(dim))
        _prepare[(tokens, triton.cdiv(hidden, block))](
            embeddings, length, tails, counts, cos_table, sin_table,
            result['data'], result['cos'], result['sin'], result['row'], result['slot'],
            result['valid'], result['k_of'], result['positions'], result['starts'],
            rows, width, hidden, dim, tokens, window,
            *(embeddings.stride() if embedding_ids is None else
              (embeddings.stride(0), 0, embeddings.stride(1))),
            *cos_table.stride(), *sin_table.stride(), triton.next_power_of_2(rows), block,
            **(dict(IDS=embedding_ids, GATHER=True, I0=embedding_ids.stride(0),
                    I1=embedding_ids.stride(1)) if embedding_ids is not None else {}))
    else:
        offsets = torch.cat((counts.new_zeros(1), counts.cumsum(0)))
        index = torch.arange(tokens, device=counts.device)
        row = torch.searchsorted(offsets[1:], index, right=True)
        valid = row < rows
        slot = torch.where(valid, index - offsets[row], width)
        k_of = torch.cat((tails.long().clamp(0, window), tails.new_zeros(1)))[row]
        positions = torch.where(valid, length + k_of + slot, length)
        if embedding_ids is not None:
            embeddings = embeddings[embedding_ids]
        result.update(row=row, slot=slot, valid=valid, k_of=k_of, positions=positions,
                      starts=torch.where(valid, offsets[row], tokens),
                      data=torch.where(valid[:, None],
                          embeddings[row.clamp(max=rows-1), slot.clamp(max=width-1)], 0),
                      cos=cos_table[positions][None], sin=sin_table[positions][None])
    return result


def scatter(output, counts, width):
    """Same returned shape, strides and zero slots as the eager padded scatter."""
    rows, hidden = counts.numel(), output.shape[1]
    padded = output.new_empty(rows + 1, width + 1, hidden)
    if output.is_cuda:
        if triton is None:
            raise RuntimeError('refresh scatter requires CUDA Triton')
        _scatter[(triton.cdiv(padded.numel(), 256),)](
            output, counts, padded, rows, width, hidden, output.shape[0],
            triton.next_power_of_2(rows), 256)
    else:
        offsets = torch.cat((counts.new_zeros(1), counts.cumsum(0)))
        index = torch.arange(output.shape[0], device=counts.device)
        row = torch.searchsorted(offsets[1:], index, right=True)
        slot = torch.where(row < rows, index - offsets[row], width)
        padded.zero_()
        padded[row, slot] = output
    return padded[:rows, :width]


_BUCKET_EXTENSION = None
_BUCKET_CPP = r"""
#include <torch/extension.h>
void attach_refresh_bucket(torch::Tensor counts, uint64_t small, uint64_t large);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("attach", &attach_refresh_bucket);
}
"""
_BUCKET_CUDA = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>

__global__ void refresh_condition(cudaGraphConditionalHandle handle,
                                  const int64_t* counts, int rows) {
  int64_t live = 0;
  for (int row = 0; row < rows; ++row) live += counts[row];
  cudaGraphSetConditional(handle, live <= 32);
}

void attach_refresh_bucket(torch::Tensor counts, uint64_t small, uint64_t large) {
  TORCH_CHECK(counts.is_cuda() && counts.scalar_type() == at::kLong && counts.is_contiguous(),
              "refresh bucket needs contiguous CUDA int64 counts");
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  cudaStreamCaptureStatus status;
  cudaGraph_t graph;
  const cudaGraphNode_t* dependencies;
  size_t size;
  C10_CUDA_CHECK(cudaStreamGetCaptureInfo_v2(stream, &status, nullptr, &graph,
                                          &dependencies, &size));
  TORCH_CHECK(status == cudaStreamCaptureStatusActive, "refresh bucket needs capture");
  cudaGraphConditionalHandle handle;
  C10_CUDA_CHECK(cudaGraphConditionalHandleCreate(&handle, graph, 0, 0));
  refresh_condition<<<1, 1, 0, stream>>>(handle, counts.data_ptr<int64_t>(), counts.numel());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  C10_CUDA_CHECK(cudaStreamGetCaptureInfo_v2(stream, &status, nullptr, &graph,
                                          &dependencies, &size));
  cudaGraphNodeParams params = {};
  params.type = cudaGraphNodeTypeConditional;
  params.conditional.handle = handle;
  params.conditional.type = cudaGraphCondTypeIf;
  params.conditional.size = 2;
  cudaGraphNode_t branch;
  C10_CUDA_CHECK(cudaGraphAddNode(&branch, graph, dependencies, size, &params));
  cudaGraphNode_t child;
  C10_CUDA_CHECK(cudaGraphAddChildGraphNode(&child, params.conditional.phGraph_out[0],
      nullptr, 0, reinterpret_cast<cudaGraph_t>(small)));
  C10_CUDA_CHECK(cudaGraphAddChildGraphNode(&child, params.conditional.phGraph_out[1],
      nullptr, 0, reinterpret_cast<cudaGraph_t>(large)));
  C10_CUDA_CHECK(cudaStreamUpdateCaptureDependencies(
      stream, &branch, 1, cudaStreamSetCaptureDependencies));
}
"""


def bucket_extension():
    """Build the CUDA conditional-node setter before any parent capture begins."""
    global _BUCKET_EXTENSION
    if _BUCKET_EXTENSION is None:
        from torch.utils.cpp_extension import load_inline
        _BUCKET_EXTENSION = load_inline(
            name='jetspec_refresh_bucket_v1', cpp_sources=_BUCKET_CPP,
            cuda_sources=_BUCKET_CUDA, extra_cuda_cflags=['-O3'])
    return _BUCKET_EXTENSION


def bucketed_refresh(head, embeddings, length, tails, counts, start, key_width,
                     *, embedding_ids=None):
    """Attach mutually exclusive T32/T64 bodies while retaining both private pools."""
    from jetspec.engines.jetspec_plus_plus.kernels.nvfp4 import w4a8

    key = (id(head._c1_kv), *(value.data_ptr() for value in
           (embeddings, length, tails, counts, start)), key_width,
           None if embedding_ids is None else embedding_ids.data_ptr())
    entries = getattr(head, '_refresh_buckets', None)
    if entries is None:
        entries = head._refresh_buckets = {}
    entry = entries.get(key)
    capturing = torch.cuda.is_current_stream_capturing()
    if entry is None:
        if capturing:
            raise RuntimeError('refresh buckets must be warmed before parent capture')
        extension = bucket_extension()
        default_config = w4a8.default_config

        def pinned(n, k, m, sms=148):
            config = default_config(n, k, 64 if m in (32, 64) else m, sms)
            return replace(config, tile_n=32) if m == 32 else config

        graphs, output, deferred = {}, None, {}
        head._refresh_bucket_building = True
        try:
            with patch.object(w4a8, 'default_config', pinned):
                for capacity in (64, 32):
                    def body():
                        return head._refresh_cached_fixed_prep(
                            embeddings, length, tails, counts, capacity, start, key_width,
                            **({'embedding_ids': embedding_ids} if embedding_ids is not None else {}))
                    for _ in range(2):
                        value = body()
                    tail = getattr(value.untyped_storage(), '_w4a8_adapter_tail', None)
                    if output is None:
                        output = value if tail is not None else torch.empty_like(value)
                    graph = torch.cuda.CUDAGraph(keep_graph=True)
                    with torch.cuda.graph(graph):
                        value = body()
                        if tail is None:
                            output.copy_(value)
                    if tail is not None:
                        deferred[capacity] = value.untyped_storage()._w4a8_adapter_tail
                        if capacity == 64:
                            output = value
                    graphs[capacity] = graph
        finally:
            head._refresh_bucket_building = False
        if deferred:
            # A deferred tensor has no bytes to copy. Both consumers select the
            # same live operands as the conditional adapter body on replay.
            output.untyped_storage()._w4a8_adapter_tail = replace(
                deferred[64], small=deferred[32])
        # Keep the pools and every input/cache owner alive as long as the parent.
        entry = entries[key] = (graphs, output, extension,
                                (head._c1_kv, embeddings, length, tails, counts, start, embedding_ids))
    graphs, output, extension, _ = entry
    if capturing:
        extension.attach(counts, graphs[32].raw_cuda_graph(), graphs[64].raw_cuda_graph())
    else:
        # Setup uses the full-capacity body; only captured replay chooses by device count.
        graphs[64].replay()
    return output
