"""Opt-in Blackwell attention for the C1 query and lean adapter refresh."""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

try:
    from jetspec.engines.jetspec_plus_plus._optional import triton
    from jetspec.engines.jetspec_plus_plus._optional import tl
except ImportError:  # CPU-only test environments have no Triton package.
    triton = None

    class _TritonLanguage:
        constexpr = int

    tl = _TritonLanguage()


def _triton_jit(function):
    return function if triton is None else triton.jit(function)


_slot_checks = []


@_triton_jit
def _slot_attention(Q, CK, CV, WK, WV, SLOTS, LENGTHS, OUT,
                    q_b: tl.constexpr, q_h: tl.constexpr, q_m: tl.constexpr,
                    ck_s: tl.constexpr, ck_h: tl.constexpr, ck_t: tl.constexpr,
                    cv_s: tl.constexpr, cv_h: tl.constexpr, cv_t: tl.constexpr,
                    wk_b: tl.constexpr, wk_h: tl.constexpr, wk_t: tl.constexpr,
                    wv_b: tl.constexpr, wv_h: tl.constexpr, wv_t: tl.constexpr,
                    o_b: tl.constexpr, o_h: tl.constexpr, o_m: tl.constexpr,
                    KEY_COUNT: tl.constexpr, WINDOW: tl.constexpr,
                    ROWS: tl.constexpr, DIM: tl.constexpr, SCALE: tl.constexpr,
                    INCLUSIVE: tl.constexpr, FLAT_WINDOW: tl.constexpr,
                    BM: tl.constexpr, BN: tl.constexpr,
                    BD: tl.constexpr):
    row_block = tl.program_id(0)
    batch = tl.program_id(1)
    head = tl.program_id(2)
    rows = row_block * BM + tl.arange(0, BM)
    keys = tl.arange(0, BN)
    dims = tl.arange(0, BD)
    slot = tl.load(SLOTS + batch)
    length = tl.load(LENGTHS + batch)
    query = tl.load(Q + batch * q_b + head * q_h + rows[:, None] * q_m
                    + dims[None, :], (rows[:, None] < ROWS) & (dims[None, :] < DIM),
                    other=0)
    maximum = tl.full((BM,), -float('inf'), tl.float32)
    total = tl.full((BM,), 0.0, tl.float32)
    accumulator = tl.full((BM, BD), 0.0, tl.float32)
    for begin in range(tl.cdiv(KEY_COUNT, BN)):
        positions = begin * BN + keys
        key = tl.load(CK + slot * ck_s + head * ck_h + positions[:, None] * ck_t
                      + dims[None, :], (positions[:, None] < KEY_COUNT)
                      & (dims[None, :] < DIM), other=0)
        value = tl.load(CV + slot * cv_s + head * cv_h + positions[:, None] * cv_t
                        + dims[None, :], (positions[:, None] < KEY_COUNT)
                        & (dims[None, :] < DIM), other=0)
        scores = tl.dot(query, tl.trans(key), input_precision='ieee') * SCALE
        scores = tl.where(positions[None, :] < length, scores, -float('inf'))
        next_max = tl.maximum(maximum, tl.max(scores, 1))
        safe_max = tl.where(next_max == -float('inf'), 0.0, next_max)
        correction = tl.where(maximum == -float('inf'), 0.0, tl.exp(maximum - safe_max))
        weights = tl.exp(scores - safe_max[:, None])
        accumulator = accumulator * correction[:, None] + tl.dot(weights.to(value.dtype), value)
        total = total * correction + tl.sum(weights, 1)
        maximum = next_max
    for begin in range(tl.cdiv(WINDOW, BN)):
        positions = begin * BN + keys
        key = tl.load(WK + batch * wk_b + head * wk_h + positions[:, None] * wk_t
                      + dims[None, :], (positions[:, None] < WINDOW)
                      & (dims[None, :] < DIM), other=0)
        value = tl.load(WV + batch * wv_b + head * wv_h + positions[:, None] * wv_t
                        + dims[None, :], (positions[:, None] < WINDOW)
                        & (dims[None, :] < DIM), other=0)
        scores = tl.dot(query, tl.trans(key), input_precision='ieee') * SCALE
        if INCLUSIVE:
            visible = positions[None, :] <= rows[:, None] % WINDOW
        else:
            visible = positions[None, :] < rows[:, None] % WINDOW
        scores = tl.where((positions[None, :] < WINDOW) & visible, scores, -float('inf'))
        next_max = tl.maximum(maximum, tl.max(scores, 1))
        safe_max = tl.where(next_max == -float('inf'), 0.0, next_max)
        correction = tl.where(maximum == -float('inf'), 0.0, tl.exp(maximum - safe_max))
        weights = tl.exp(scores - safe_max[:, None])
        accumulator = accumulator * correction[:, None] + tl.dot(weights.to(value.dtype), value)
        total = total * correction + tl.sum(weights, 1)
        maximum = next_max
    result = accumulator / total[:, None]
    if FLAT_WINDOW:
        groups = ROWS // WINDOW
        offset = (batch * o_b + (rows[:, None] % WINDOW) * o_h
                  + (head * groups + rows[:, None] // WINDOW) * DIM + dims[None, :])
    else:
        offset = batch * o_b + head * o_h + rows[:, None] * o_m + dims[None, :]
    tl.store(OUT + offset, result, (rows[:, None] < ROWS) & (dims[None, :] < DIM))








def native_enabled(device: torch.device) -> bool:
    return (os.environ.get("JETSPEC_ATTN_SM100", "0") == "1"
            and device.type == "cuda" and torch.cuda.get_device_capability(device) == (10, 0))


def flashinfer_enabled(device: torch.device, dtype: torch.dtype, *, chain=False) -> bool:
    return ((os.environ.get("JETSPEC_ATTN_SM100", "0") == "flashinfer"
             or (chain and os.environ.get("JETSPEC_CHAIN_ATTN3", "0") == "1"))
            and device.type == "cuda" and dtype in (torch.float16, torch.bfloat16)
            and torch.cuda.get_device_capability(device) == (10, 0))






class PrefixPlan:
    """One fixed graph buffer set, replanned from pinned exact lengths each round."""

    def __init__(self, q_rows: int, cache: torch.Tensor, scale: float,
                 q_heads: int | None = None, causal: bool = False):
        from jetspec.engines.jetspec_plus_plus._optional import flashinfer

        _, kv_heads, capacity, dim = cache.shape
        q_heads = kv_heads if q_heads is None else q_heads
        device = cache.device
        workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
        self.wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
            workspace, "HND", use_cuda_graph=True,
            qo_indptr_buf=torch.empty(2, dtype=torch.int32, device=device),
            kv_indptr_buf=torch.empty(2, dtype=torch.int32, device=device),
        )
        self.workspace = workspace
        self.q_indptr = torch.tensor([0, q_rows], dtype=torch.int32, pin_memory=True)
        self.kv_indptr = torch.tensor([0, 0], dtype=torch.int32, pin_memory=True)
        self.q_heads, self.kv_heads = q_heads, kv_heads
        self.capacity, self.dim = capacity, dim
        self.dtype, self.scale = cache.dtype, scale
        self.causal = causal

    def plan(self, length: int) -> None:
        if not 0 < length <= self.capacity:
            raise ValueError(f"FlashInfer prefix length {length} outside cache {self.capacity}")
        self.kv_indptr[1] = length
        self.wrapper.plan(
            self.q_indptr, self.kv_indptr,
            self.q_heads, self.kv_heads, self.dim, causal=self.causal,
            q_data_type=self.dtype, kv_data_type=self.dtype, sm_scale=self.scale)

    def run(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        query = q[0].transpose(0, 1).contiguous()
        # The cache is HND already; the plan limits it to the exact L.
        keys = k[0]
        values = v[0]
        # The ragged wrapper takes k and v separately; only the paged one takes a tuple.
        return self.wrapper.run(query, keys, values, return_lse=True)


def prepare_prefix(owner, stage: str, q_rows: int, cache: torch.Tensor,
                   length: int, scale: float) -> PrefixPlan:
    plans = getattr(owner, "_flashinfer_prefix_plans", None)
    if plans is None:
        plans = owner._flashinfer_prefix_plans = {}
    q_heads = owner._d1_lean_plan()[0][-5]
    key = (stage, q_rows, q_heads, cache.shape, cache.dtype, cache.device, scale)
    plan = plans.get(key)
    if plan is None:
        plan = plans[key] = PrefixPlan(q_rows, cache, scale, q_heads,
                                      causal=stage == "c1")
    plan.plan(length + q_rows - 1 if stage == "c1" else length)
    return plan




def attend_c1(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
              scale: float, plan: PrefixPlan) -> torch.Tensor:
    if scale != plan.scale or q.shape[1] != getattr(plan, "q_heads", q.shape[1]):
        raise ValueError("FlashInfer prefix plan does not match C1 attention heads or scale")
    output, _ = plan.run(q, k, v)
    return output.transpose(0, 1).unsqueeze(0)


def attend_refresh(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                   scale: float, plan: PrefixPlan, start: torch.Tensor,
                   window: int, meta: tuple[torch.Tensor, torch.Tensor,
                                           torch.Tensor, torch.Tensor]) -> torch.Tensor:
    if scale != plan.scale or q.shape[1] != getattr(plan, "q_heads", q.shape[1]):
        raise ValueError("FlashInfer prefix plan does not match refresh attention heads or scale")
    from jetspec.engines.jetspec_plus_plus.kernels.attn_suffix import suffix_state

    return _merge(plan.run(q, k, v),
                  suffix_state(q, k, v, start, window, scale, meta))


def adapter_split_enabled(q: torch.Tensor, k: torch.Tensor) -> bool:
    """Limit the SM100 split candidate to the two profiled fast5 query shapes."""
    return (os.environ.get("JETSPEC_ADAPTER_ATTN_SPLIT", "0") == "1"
            and q.device.type == "cuda" and q.dtype == torch.bfloat16
            and tuple(q.shape) == (1, 32, 16, 128)
            and tuple(k.shape) in ((1, 32, 640, 128), (1, 32, 1408, 128))
            and torch.cuda.get_device_capability(q.device) == (10, 0))




def attend(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
           mask: torch.Tensor, scale: float) -> torch.Tensor:
    from jetspec.engines.jetspec_plus_plus.kernels.attn_sm80 import attend as attend_sm80; from jetspec.engines.jetspec_plus_plus.kernels.attn_sm80 import enabled as sm80_enabled

    if adapter_split_enabled(q, k):
        # Fast5 cache writes repeat each of the eight KV heads four times.
        return attend_grouped(q, k[:, ::4], v[:, ::4], mask, scale)

    # Short buckets keep the established draft decisions; split-KV helps once
    # the long-key SDPA kernel becomes occupancy-bound.
    if sm80_enabled(q.device, q.dtype) and k.shape[2] > 512:
        return attend_sm80(q, k, v, mask, scale)
    if not native_enabled(q.device):
        return torch.ops.aten._scaled_dot_product_efficient_attention(
            q, k, v, mask, False, 0.0, False, scale=scale)[0]

    # Force cuDNN: the existing float mask, cache slots, and scale are unchanged.
    # A backend rejection is explicit; the opt-in path never silently uses sm80.
    with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
        return F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False, scale=scale)
