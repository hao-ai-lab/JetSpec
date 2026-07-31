"""HF tree-masked attention backends for the reference engine.

This is the small, non-paged Triton tree-attention hook used by the HF
benchmark path. It temporarily replaces the global HF SDPA attention function
for one target forward, while normal target forwards can still use FA2/SDPA.

The kernel preserves hybrid-model semantics: sliding-attention layers apply
their logical-position window to the cached prefix and tree nodes, and GPT-OSS
attention sinks contribute to the softmax denominator.
"""
from __future__ import annotations

from contextlib import contextmanager
import logging

import torch
import triton
import triton.language as tl
from triton.runtime.errors import OutOfResources

_LOG = logging.getLogger(__name__)

_tree_attn_state: dict = {
    "ancestor": None,
    "depths": None,
    "prefix_len": 0,
}
_original_sdpa_fn = None
_triton_call_count = 0


def _get_config_owner(module: torch.nn.Module) -> torch.nn.Module | None:
    if hasattr(module, "config"):
        return module
    orig_mod = getattr(module, "_orig_mod", None)
    if orig_mod is not None and hasattr(orig_mod, "config"):
        return orig_mod
    return None


@contextmanager
def use_attention_implementation(module: torch.nn.Module, attn_implementation: str):
    """Temporarily switch the HF attention dispatch key for one model call."""
    config_owner = _get_config_owner(module)
    if config_owner is None:
        yield
        return

    config = config_owner.config
    previous = getattr(config, "_attn_implementation", None)
    config._attn_implementation = attn_implementation
    try:
        yield
    finally:
        config._attn_implementation = previous


def _set_tree_mask(
    ancestor: torch.Tensor,
    depths: torch.Tensor,
    prefix_len: int,
) -> None:
    stored = ancestor if ancestor.dtype == torch.uint8 else ancestor.to(torch.uint8)
    _tree_attn_state["ancestor"] = stored.contiguous()
    _tree_attn_state["depths"] = depths.to(
        device=ancestor.device, dtype=torch.int32
    ).contiguous()
    _tree_attn_state["prefix_len"] = int(prefix_len)


def _clear_tree_mask() -> None:
    _tree_attn_state["ancestor"] = None
    _tree_attn_state["depths"] = None
    _tree_attn_state["prefix_len"] = 0


@contextmanager
def use_tree_attention(
    ancestor: torch.Tensor,
    depths: torch.Tensor,
    prefix_len: int,
    attn_impl: str = "triton",
):
    """Temporarily route HF SDPA dispatch through tree attention."""
    if attn_impl == "sdpa":
        yield
        return
    if attn_impl != "triton":
        raise ValueError(f"unknown tree attention impl: {attn_impl!r}")

    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    global _original_sdpa_fn
    _original_sdpa_fn = ALL_ATTENTION_FUNCTIONS.get("sdpa")
    _set_tree_mask(ancestor, depths, prefix_len)
    ALL_ATTENTION_FUNCTIONS["sdpa"] = _triton_attention_forward
    try:
        yield
    finally:
        if _original_sdpa_fn is not None:
            ALL_ATTENTION_FUNCTIONS["sdpa"] = _original_sdpa_fn
        _original_sdpa_fn = None
        _clear_tree_mask()


@triton.jit
def _tree_attn_fwd(
    Q, K, V, Out,
    Ancestor,
    Depths,
    Sinks,
    sm_scale,
    prefix_len,
    window_size,
    N,
    KV_LEN,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    H,
    H_KV,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    IS_SLIDING: tl.constexpr,
    HAS_SINKS: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_zh = tl.program_id(1)
    off_z = off_zh // H
    off_h = off_zh % H
    off_hkv = off_h * H_KV // H

    Q += off_z * stride_qz + off_h * stride_qh
    K += off_z * stride_kz + off_hkv * stride_kh
    V += off_z * stride_vz + off_hkv * stride_vh
    Out += off_z * stride_oz + off_h * stride_oh

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    mask_m = offs_m < N

    q = tl.load(
        Q + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk,
        mask=mask_m[:, None] & (offs_d[None, :] < HEAD_DIM),
        other=0.0,
    )

    if HAS_SINKS:
        sink = tl.load(Sinks + off_h).to(tl.float32)
        m_i = tl.zeros([BLOCK_M], dtype=tl.float32) + sink
        l_i = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    else:
        m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    query_depth = tl.load(Depths + offs_m, mask=mask_m, other=0)
    query_position = prefix_len + query_depth

    for start_n in range(0, KV_LEN, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < KV_LEN

        k = tl.load(
            K + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk,
            mask=mask_n[:, None] & (offs_d[None, :] < HEAD_DIM),
            other=0.0,
        )

        qk = tl.dot(q, tl.trans(k)).to(tl.float32) * sm_scale

        is_prefix = offs_n[None, :] < prefix_len
        in_tree = offs_n[None, :] >= prefix_len
        tree_kv = tl.maximum(offs_n[None, :] - prefix_len, 0)
        anc = tl.load(
            Ancestor + offs_m[:, None] * N + tree_kv,
            mask=mask_m[:, None] & in_tree & mask_n[None, :],
            other=0,
        )
        attend = is_prefix | (in_tree & (anc != 0))
        if IS_SLIDING:
            tree_depth = tl.load(
                Depths + tree_kv,
                mask=in_tree & mask_n[None, :],
                other=0,
            )
            key_position = tl.where(
                is_prefix,
                offs_n[None, :],
                prefix_len + tree_depth,
            )
            attend = attend & (
                key_position > query_position[:, None] - window_size
            )
        qk = tl.where(attend & mask_n[None, :] & mask_m[:, None], qk, float("-inf"))

        m_ij = tl.max(qk, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new[:, None])
        l_i = alpha * l_i + tl.sum(p, axis=1)

        v = tl.load(
            V + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk,
            mask=mask_n[:, None] & (offs_d[None, :] < HEAD_DIM),
            other=0.0,
        )

        acc = alpha[:, None] * acc + tl.dot(p.to(v.dtype), v).to(tl.float32)
        m_i = m_new

    acc = acc / l_i[:, None]
    tl.store(
        Out + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok,
        acc.to(Out.dtype.element_ty),
        mask=mask_m[:, None] & (offs_d[None, :] < HEAD_DIM),
    )


def _tree_sdpa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    ancestor: torch.Tensor,
    depths: torch.Tensor,
    prefix_len: int,
    sm_scale: float | None = None,
    sliding_window: int | None = None,
    sinks: torch.Tensor | None = None,
) -> torch.Tensor:
    B, H, N, D = query.shape
    _, H_KV, KV_LEN, _ = key.shape
    if sm_scale is None:
        sm_scale = D ** -0.5

    out = torch.empty_like(query)
    block_d = triton.next_power_of_2(D)
    # Gemma 4 uses head_dim=256 for sliding layers and 512 for global layers.
    # Triton's default staging for a 64x64 tile can request 246,272 bytes, above
    # an H100's 232,448-byte limit.  Select explicit compiler options for every
    # shape so backend defaults or an old autotune choice cannot reintroduce it.
    wide_head = block_d >= 256
    initial_config = (32, 32, 8, 1) if wide_head else (64, 64, 4, 2)

    def _launch(block_m: int, block_n: int, num_warps: int, num_stages: int):
        grid = (triton.cdiv(N, block_m), B * H)
        _tree_attn_fwd[grid](
            query, key, value, out,
            ancestor,
            depths,
            sinks if sinks is not None else query,
            sm_scale,
            prefix_len,
            int(sliding_window or 0),
            N,
            KV_LEN,
            query.stride(0), query.stride(1), query.stride(2), query.stride(3),
            key.stride(0), key.stride(1), key.stride(2), key.stride(3),
            value.stride(0), value.stride(1), value.stride(2), value.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            H,
            H_KV,
            HEAD_DIM=D,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_D=block_d,
            IS_SLIDING=sliding_window is not None and int(sliding_window) > 0,
            HAS_SINKS=sinks is not None,
            num_warps=num_warps,
            num_stages=num_stages,
        )

    try:
        _launch(*initial_config)
    except OutOfResources:
        # Different GPU generations expose different per-block shared-memory
        # limits.  This conservative specialization is a correctness fallback
        # and compiles to well below the limit even for head_dim=512.
        fallback_config = (16, 16, 4, 1)
        if initial_config == fallback_config:
            raise
        _LOG.warning(
            "tree-attention tile %sx%s exceeded device resources for head_dim=%s; "
            "retrying with %sx%s",
            initial_config[0], initial_config[1], D,
            fallback_config[0], fallback_config[1],
        )
        _launch(*fallback_config)
    return out


def _triton_attention_forward(module, query, key, value, attention_mask, **kwargs):
    ancestor = _tree_attn_state["ancestor"]
    depths = _tree_attn_state["depths"]
    if ancestor is None:
        if _original_sdpa_fn is None:
            raise RuntimeError("tree attention state is unset and no SDPA fallback exists")
        return _original_sdpa_fn(module, query, key, value, attention_mask, **kwargs)

    if not torch.compiler.is_compiling():
        global _triton_call_count
        _triton_call_count += 1
        if _triton_call_count <= 3:
            _LOG.info(
                "triton tree attention: Q=%s K=%s ancestor=%s prefix_len=%s",
                tuple(query.shape), tuple(key.shape), tuple(ancestor.shape),
                _tree_attn_state["prefix_len"],
            )

    sliding_window = kwargs.get(
        "sliding_window", getattr(module, "sliding_window", None)
    )
    sinks = kwargs.get("s_aux")
    out = _tree_sdpa(
        query, key, value, ancestor, depths,
        _tree_attn_state["prefix_len"],
        sm_scale=kwargs.get("scaling"),
        sliding_window=sliding_window,
        sinks=sinks,
    )
    return out.transpose(1, 2).contiguous(), None
