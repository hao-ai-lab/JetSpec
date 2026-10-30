"""Opt-in bitwise draft glue tuning, with unchanged per-chunk reductions."""
import torch

from jetspec.engines.jetspec_plus_plus.wire.contract import Offset

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None


class PassPartials(tuple):
    """The original three partial arrays plus graph-owned, self-resetting tickets."""

    def __new__(cls, partials, counter):
        result = super().__new__(cls, partials)
        result.counter = counter
        return result


if triton is not None:
    from jetspec.engines.jetspec_plus_plus.kernels.drafter_pass import _finish

    @triton.jit
    def _partials_finish(LOGITS, PMAX, PSUM, PIDX, COUNTER,
                         IDS, KEYS, ACTIVE, FINISHED, PASSES,
                         FINAL_TOKENS, FINAL_TOP1, PREV_TOP1, META,
                         COMMITTED, PENDING, RP, PIN2, PIN_SLOT, PIN_TOKEN,
                         V: tl.constexpr, L: tl.constexpr, C: tl.constexpr,
                         S0: tl.constexpr, S1: tl.constexpr, S2: tl.constexpr,
                         B: tl.constexpr, CHUNKS: tl.constexpr, PC: tl.constexpr,
                         PASS: tl.constexpr, SAVE_PREV: tl.constexpr,
                         HAS_PIN2: tl.constexpr, HAS_PIN: tl.constexpr,
                         DENSE_PIN: tl.constexpr, RP_L: tl.constexpr,
                         PUBLISH: tl.constexpr = False, Packet=None, Published=None,
                         Wire=None, Scores=None, WirePasses=None, RingRows=None,
                         RingKeys=None, RingTop1=None, RingPasses=None, RingRounds=None,
                         ROWS: tl.constexpr = 0, CAP: tl.constexpr = 0,
                         RING: tl.constexpr = 0, WORDS: tl.constexpr = 0,
                         BLOCK_OFF: tl.constexpr = 0, KEY_OFF: tl.constexpr = 0,
                         PAD_BLOCK: tl.constexpr = 1,
                         REFRESH: tl.constexpr = False, RefreshChained=None, RefreshKhat=None,
                         RefreshCounts=None, RefreshTails=None, RefreshStart=None,
                         REFRESH_WINDOW: tl.constexpr = 0, REFRESH_THRESHOLD: tl.constexpr = 0.):
        slot, group = tl.program_id(0), tl.program_id(1)
        row = slot // L
        for chunk in range(group * CHUNKS, tl.minimum((group + 1) * CHUNKS, C)):
            x = chunk * B + tl.arange(0, B)
            value = tl.load(LOGITS + row * S0 + (slot % L) * S1 + x * S2,
                            x < V, other=-float('inf')).to(tl.float32)
            maximum = tl.max(value, 0)
            safe_maximum = tl.where(maximum == -float('inf'), 0.0, maximum)
            total = tl.sum(tl.exp(value - safe_maximum), 0)
            index = tl.min(tl.where(value == maximum, x, 2147483647), 0)
            out = slot * C + chunk
            tl.store(PMAX + out, maximum)
            tl.store(PSUM + out, total)
            tl.store(PIDX + out, index)
        # Every producer publishes before the ticket; the last CTA acquires all
        # earlier publications and reads partials in the original reduction tree.
        tl.debug_barrier()
        ticket = tl.atomic_add(COUNTER + row, 1, sem='acq_rel', scope='gpu')
        if ticket == L * tl.cdiv(C, CHUNKS) - 1:
            _finish(PMAX, PSUM, PIDX, IDS, KEYS, ACTIVE, FINISHED, PASSES,
                    FINAL_TOKENS, FINAL_TOP1, PREV_TOP1, META,
                    COMMITTED, PENDING, RP, PIN2, PIN_SLOT, PIN_TOKEN,
                    L, C, PC, PASS, SAVE_PREV, HAS_PIN2, RP_L,
                    HAS_PIN, DENSE_PIN, PUBLISH=PUBLISH, Packet=Packet,
                    Published=Published, Wire=Wire, Scores=Scores,
                    WirePasses=WirePasses, RingRows=RingRows, RingKeys=RingKeys,
                    RingTop1=RingTop1, RingPasses=RingPasses, RingRounds=RingRounds,
                    ROWS=ROWS, CAP=CAP, RING=RING, WORDS=WORDS, BLOCK_OFF=BLOCK_OFF,
                    KEY_OFF=KEY_OFF, PAD_BLOCK=PAD_BLOCK, ROW=row, USE_ROW=True,
                    REFRESH=REFRESH, RefreshChained=RefreshChained, RefreshKhat=RefreshKhat,
                    RefreshCounts=RefreshCounts, RefreshTails=RefreshTails,
                    RefreshStart=RefreshStart, REFRESH_WINDOW=REFRESH_WINDOW,
                    REFRESH_THRESHOLD=REFRESH_THRESHOLD)
            tl.debug_barrier()
            tl.atomic_xchg(COUNTER + row, 0, sem='release', scope='gpu')


    @triton.jit
    def _qkv(Q, K, V, QW, KW, Cos, Sin, Out, Cache, Slots,
             QH: tl.constexpr, KH: tl.constexpr, D: tl.constexpr,
             QS: tl.constexpr, KS: tl.constexpr, VS: tl.constexpr,
             CS: tl.constexpr, SS: tl.constexpr,
             CACHE_ROW: tl.constexpr, CACHE_KV: tl.constexpr, CACHE_HEAD: tl.constexpr,
             START: tl.constexpr, SPAN: tl.constexpr,
             QE: tl.constexpr, KE: tl.constexpr, BLOCK: tl.constexpr,
             HEADS: tl.constexpr):
        row, group = tl.program_id(0), tl.program_id(1)
        head = tl.arange(0, HEADS)
        col = tl.arange(0, BLOCK)
        partner = tl.broadcast_to(tl.where(col < D // 2, col + D // 2,
                                           col - D // 2)[None, :], (HEADS, BLOCK))
        cosine = tl.load(Cos + row * CS + col, col < D, other=0)
        sine = tl.load(Sin + row * SS + col, col < D, other=0)
        if group < tl.cdiv(QH, HEADS):
            head = group * HEADS + head
            valid = (head[:, None] < QH) & (col[None, :] < D)
            source = tl.load(Q + row * QS + head[:, None] * D + col[None, :],
                             valid, other=0).to(tl.float32)
            rms = tl.rsqrt(tl.sum(source * source, 1) / D + QE)
            normed = (source * rms[:, None]).to(tl.bfloat16)
            weighted = (normed * tl.load(QW + col, col < D, other=0)[None, :]).to(tl.bfloat16)
            rotated = tl.gather(weighted, partner, 1)
            rotated = tl.where(col[None, :] < D // 2, -rotated, rotated)
            first = (weighted * cosine[None, :]).to(tl.bfloat16)
            second = (rotated * sine[None, :]).to(tl.bfloat16)
            tl.store(Out + (row * QH + head[:, None]) * D + col[None, :],
                     (first + second).to(tl.bfloat16), valid)
        else:
            head = (group - tl.cdiv(QH, HEADS)) * HEADS + head
            valid = (head[:, None] < KH) & (col[None, :] < D)
            source = tl.load(K + row * KS + head[:, None] * D + col[None, :],
                             valid, other=0).to(tl.float32)
            rms = tl.rsqrt(tl.sum(source * source, 1) / D + KE)
            normed = (source * rms[:, None]).to(tl.bfloat16)
            weighted = (normed * tl.load(KW + col, col < D, other=0)[None, :]).to(tl.bfloat16)
            rotated = tl.gather(weighted, partner, 1)
            rotated = tl.where(col[None, :] < D // 2, -rotated, rotated)
            first = (weighted * cosine[None, :]).to(tl.bfloat16)
            second = (rotated * sine[None, :]).to(tl.bfloat16)
            if SPAN:
                slot = START + row
            else:
                slot = tl.load(Slots + row)
            target = slot * CACHE_ROW + head[:, None] * CACHE_HEAD + col[None, :]
            tl.store(Cache + target, (first + second).to(tl.bfloat16), valid)
            copied = tl.load(V + row * VS + head[:, None] * D + col[None, :], valid, other=0)
            tl.store(Cache + target + CACHE_KV, copied, valid)


def chain_pass_into(logits, ids, keys, active, outputs, previous_top1, tokens,
                    rp, partials, index, save_previous, pinned_second, pins):
    """Publish a chain pass in the last vocabulary producer, without a spin wait."""
    rows, lanes, vocab = logits.shape
    chunks = partials[0].shape[-1]
    from jetspec.engines.jetspec_plus_plus.profiles import kernel_value
    group = 4
    if vocab == 248320 and lanes == 16 and kernel_value('draft_fuse_stats27', '0') == '1':
        group = 3 if rows in (1, 4) else 6 if rows == 16 else 4
    _partials_finish[(rows * lanes, triton.cdiv(chunks, group))](
        logits, *partials, partials.counter,
        ids, keys, active, outputs['finished'], outputs['passes_run'],
        outputs['final_tokens'], outputs['final_p_top1'], previous_top1,
        outputs['meta'], tokens.committed_ids, tokens.pending_draft_ids, rp,
        pinned_second if pinned_second is not None else keys,
        *(pins if pins is not None else (keys, keys)),
        vocab, lanes, chunks, *logits.stride(),
        triton.next_power_of_2(triton.cdiv(vocab, chunks)), group,
        triton.next_power_of_2(chunks), index, save_previous,
        pinned_second is not None, pins is not None,
        pins is not None and pins[0].ndim == 2, int(Offset.L),
        # T14's fused J1 publication runs in the same finish, now in the last producer.
        **(outputs.get('_t14_publication', {}) if index == 1 else {}),
        **(outputs.get('_t17_refresh', {}) if index == 0 else {}), num_warps=4)


def qkv_into(query, key, value, q_weight, k_weight, cos, sin, output,
             cache, slots, span, q_eps, k_eps, q_heads, kv_heads, dim):
    """Keep each head's reduction and BF16 casts while sharing a CTA."""
    # Keep the original four warps per head: packing heads into fewer warps
    # changes FP32 reduction order and can cross a BF16 rounding boundary.
    from jetspec.engines.jetspec_plus_plus.profiles import kernel_value
    heads = 4
    if (query.shape[0] == 64 and (q_heads, kv_heads, dim) == (32, 8, 128)
            and kernel_value('draft_fuse_geo8', '0') == '1'):
        heads = 2
    _qkv[(query.shape[0], triton.cdiv(q_heads, heads) + triton.cdiv(kv_heads, heads))](
        query, key, value, q_weight, k_weight, cos, sin, output, cache, slots,
        q_heads, kv_heads, dim, query.stride(0), key.stride(0), value.stride(0),
        cos.stride(0), sin.stride(0), cache.stride(0), cache.stride(1), cache.stride(3),
        span[0] if span is not None else 0, span is not None, q_eps, k_eps,
        BLOCK=triton.next_power_of_2(dim), HEADS=heads, num_warps=4 * heads, enable_fp_fusion=False)


_SILU_EXTENSION = None
_SILU_CUDA = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>

struct __align__(16) Bf16x8 { __nv_bfloat16 value[8]; };

template <bool VECTOR>
__global__ void silu_rows(const __nv_bfloat16* gate, const __nv_bfloat16* up,
                          __nv_bfloat16* output, long long width,
                          long long gs, long long us) {
  const long long row = blockIdx.y;
  const long long col = ((long long)blockIdx.x * blockDim.x + threadIdx.x) * 8;
  if (col >= width) return;
  Bf16x8 g, u, result;
  if (VECTOR) {
    g = *reinterpret_cast<const Bf16x8*>(gate + row * gs + col);
    u = *reinterpret_cast<const Bf16x8*>(up + row * us + col);
  }
#pragma unroll
  for (int item = 0; item < 8; ++item) {
    if (col + item < width) {
      float x = __bfloat162float(VECTOR ? g.value[item] : gate[row * gs + col + item]);
      __nv_bfloat16 activated = __float2bfloat16(x / (1.0f + ::expf(-x)));
      result.value[item] = __float2bfloat16(__bfloat162float(activated) *
          __bfloat162float(VECTOR ? u.value[item] : up[row * us + col + item]));
      if (!VECTOR) output[row * width + col + item] = result.value[item];
    }
  }
  if (VECTOR) *reinterpret_cast<Bf16x8*>(output + row * width + col) = result;
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
  if (gate.numel() == 0) return output;
  long long width = gate.dim() ? gate.size(-1) : 1;
  long long rows = gate.numel() / width;
  long long gs = gate.dim() >= 2 ? gate.stride(-2) : width;
  long long us = up.dim() >= 2 ? up.stride(-2) : width;
  c10::cuda::CUDAGuard guard(gate.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  dim3 grid((width + 2047) / 2048, rows);
  const auto* g = reinterpret_cast<const __nv_bfloat16*>(gate.data_ptr());
  const auto* u = reinterpret_cast<const __nv_bfloat16*>(up.data_ptr());
  auto* out = reinterpret_cast<__nv_bfloat16*>(output.data_ptr());
  if (width % 8 == 0 && gs % 8 == 0 && us % 8 == 0 &&
      reinterpret_cast<uintptr_t>(g) % 16 == 0 && reinterpret_cast<uintptr_t>(u) % 16 == 0)
    silu_rows<true><<<grid, 256, 0, stream>>>(g, u, out, width, gs, us);
  else
    silu_rows<false><<<grid, 256, 0, stream>>>(g, u, out, width, gs, us);
  TORCH_CHECK(cudaGetLastError() == cudaSuccess, "silu mul kernel launch failed");
  return output;
}
"""


def silu_extension(cpp):
    """Export the same host symbol so the native adapter loop uses vector SiLU."""
    global _SILU_EXTENSION
    if _SILU_EXTENSION is None:
        from torch.utils.cpp_extension import load_inline
        _SILU_EXTENSION = load_inline(
            name='jetspec_draft_fuse_t6_silu_v1', cpp_sources=cpp,
            cuda_sources=_SILU_CUDA, extra_cflags=['-O3'],
            extra_cuda_cflags=['-O3'], with_cuda=True, verbose=False)
    return _SILU_EXTENSION
