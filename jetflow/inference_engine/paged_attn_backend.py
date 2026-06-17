"""HF attention interface that routes Qwen3Attention through the paged tree-attention
triton kernel (JetFlow N3, opt-in).

`Qwen3Attention.forward` applies RoPE + q/k-norm, calls `past_key_values.update(...)`,
then immediately hands the result to `ALL_ATTENTION_FUNCTIONS[config._attn_implementation]`
touching nothing in between (verified, transformers 4.57). In paged-handoff mode
`PagedKVCache.update` returns `PagedHandle`s instead of the dense KV view, so this
fn receives the handle as `key`, reads the block pool + per-seq metadata back out,
and calls the kernel — no dense KV reconstruction, no padding waste. The SDPA path
in `engine.py` stays the default + correctness oracle.

Because HF forwards a rectangular `(B, S)` batch, every seq contributes exactly S
query rows, so the ragged kernel inputs collapse to: `total_q = B*S`,
`cu_seqlens_q = arange(0, (B+1)*S, S)`, and `seq_lens_k[i] = get_seq_length(seq_i)`
(= `past_i + S`, since `update` already appended this step's S tokens). This is
exact with zero query padding for N0 (S=1,B=1), N1 (S=N,B=1), and N2a (S=1,B);
N2b (padded S=max_N) is a follow-on (see `generate_tree_batch`).
"""
import torch

from transformers.integrations.sdpa_attention import sdpa_attention_forward

from jetflow.inference_engine.paged_kv_cache import PagedHandle
from jetflow.inference_engine.paged_tree_attn_op import paged_tree_attn


def _jetflow_paged_tree_attn_forward(
    module,
    query,            # (B, Hq, S, D) post-RoPE/q-norm
    key,              # PagedHandle (k) — update's return; OR a dense tensor (fallback)
    value,            # PagedHandle (v) — unused; KV is read from the pool
    attention_mask,   # ignored on the kernel path: the kernel masks (prefix + ancestor)
    dropout=0.0,
    scaling=None,
    **kwargs,
):
    """Paged tree-attention interface; returns `((B, S, Hq, D), None)` for HF.

    The kernel path fires only when `update` ran in paged-handoff mode (so `key` is
    a `PagedHandle`). A model forward that does NOT use a handoff PagedKVCache — e.g.
    a test drafter's internal `self.model(...)` over a plain `DynamicCache`, which
    still dispatches here because `config._attn_implementation` is set globally —
    hands us dense K/V tensors; we fall back to standard SDPA so those forwards stay
    correct (sdpa derives `is_causal` from query length + a None mask)."""
    if not isinstance(key, PagedHandle):
        return sdpa_attention_forward(
            module, query, key, value, attention_mask,
            dropout=dropout, scaling=scaling, **kwargs,
        )
    handle = key                       # PagedHandle (update's return)
    cache, layer_idx = handle.cache, handle.layer_idx
    meta = cache._jetflow_attn_meta        # {"seq_ids": [...], "qq_bias": tensor | None}
    if meta is None:                   # handoff on but meta unset -> clear error, not a TypeError
        raise RuntimeError(
            "jetflow_paged_tree: paged-handoff active but _jetflow_attn_meta is unset "
            "(the engine seam must set cache._jetflow_attn_meta before the forward)"
        )
    seq_ids = meta["seq_ids"]
    B, Hq, S, D = query.shape
    q_flat = query.permute(0, 2, 1, 3).reshape(B * S, Hq, D)
    k_pool, v_pool = cache.pool(layer_idx)
    block_table = cache.kernel_block_table(seq_ids, layer_idx, device=query.device)
    seq_lens_k = cache.kernel_seq_lens(seq_ids, layer_idx, device=query.device)
    cu = torch.arange(0, (B + 1) * S, S, dtype=torch.int32, device=query.device)
    nqpkv = Hq // cache._num_heads
    qq_bias = meta["qq_bias"]
    if qq_bias is not None:            # kernel wants fp32, row-contiguous
        qq_bias = qq_bias.to(dtype=torch.float32).contiguous()
    out = paged_tree_attn(
        q_flat, k_pool, v_pool, block_table, cu, seq_lens_k,
        qq_bias, scaling, nqpkv, cache.block_size,
    )
    return out.reshape(B, S, Hq, D), None


_REGISTERED = False


def register_jetflow_paged_tree() -> None:
    """Register the paged tree-attention interface under `"jetflow_paged_tree"` (idempotent).

    Registers ONLY in `ALL_ATTENTION_FUNCTIONS` (not `ALL_MASK_ATTENTION_FUNCTIONS`),
    so HF's `create_causal_mask` early-exits with `attention_mask=None` for this
    implementation (verified, transformers 4.57) — exactly the no-mask path the
    kernel needs (it does all masking itself)."""
    global _REGISTERED
    if _REGISTERED:
        return
    from transformers import AttentionInterface

    AttentionInterface().register("jetflow_paged_tree", _jetflow_paged_tree_attn_forward)
    _REGISTERED = True
