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

`PTD_LINEARIZE_VERIFY` can override the rectangular `cu` with a PathPlan's ragged
path segments. Those path rows are still appended to one physical seq in A-form;
the kernel sees one logical seq per path via its existing logical-slot remap.
"""
import torch

from transformers.integrations.sdpa_attention import sdpa_attention_forward

from ptd.jetflow.paged_kv_cache import PagedHandle
from ptd.jetflow.paged_tree_attn_op import paged_tree_attn


def _validate_ragged_cu(cu: torch.Tensor, total_q: int, device: torch.device) -> torch.Tensor:
    if not isinstance(cu, torch.Tensor):
        raise TypeError("cu_seqlens must be a torch.Tensor")
    if cu.dtype != torch.int32:
        raise ValueError(f"cu_seqlens must be int32, got {cu.dtype}")
    if cu.device != device:
        raise ValueError(f"cu_seqlens must be on {device}, got {cu.device}")
    if cu.ndim != 1 or cu.numel() < 2:
        raise ValueError("cu_seqlens must be a 1D tensor with at least two entries")

    if cu.device.type == "cpu":
        if int(cu[0].item()) != 0:
            raise ValueError("cu_seqlens must start at 0")
        if int(cu[-1].item()) != total_q:
            raise ValueError(f"cu_seqlens[-1] must equal total_q={total_q}")
        if not bool((cu[1:] >= cu[:-1]).all().item()):
            raise ValueError("cu_seqlens must be monotone non-decreasing")
    else:
        torch._assert(cu[0] == 0, "cu_seqlens must start at 0")
        torch._assert(cu[-1] == total_q, "cu_seqlens[-1] must equal total_q")
        torch._assert((cu[1:] >= cu[:-1]).all(), "cu_seqlens must be monotone non-decreasing")
    return cu.contiguous()


def _linearized_kernel_metadata(cache, layer_idx: int, seq_ids: list, cu: torch.Tensor,
                                total_q: int, device: torch.device):
    if len(seq_ids) != 1:
        raise ValueError("ragged cu_seqlens requires exactly one physical seq_id")

    seq_id = seq_ids[0]
    past_len = cache.get_seq_length(layer_idx, seq_id=seq_id) - total_q
    if past_len < 0:
        raise ValueError(
            f"ragged cu_seqlens total_q={total_q} exceeds cached seq length "
            f"{cache.get_seq_length(layer_idx, seq_id=seq_id)}"
        )

    base_block_table = cache.kernel_block_table([seq_id], layer_idx, device=device)
    path_lens = (cu[1:] - cu[:-1]).contiguous()
    num_paths = int(path_lens.numel())

    positions = past_len + torch.arange(total_q, device=device, dtype=torch.long)
    base_blocks = base_block_table[0, positions // cache.block_size].to(torch.long)
    physical_slots = base_blocks * cache.block_size + (positions % cache.block_size)

    cols = torch.arange(total_q, device=device, dtype=torch.long).view(1, -1)
    starts = cu[:-1].to(torch.long).view(-1, 1)
    lens = path_lens.to(torch.long).view(-1, 1)
    src = (starts + cols).clamp(max=total_q - 1)
    logical_kv_slots = physical_slots[src]
    logical_kv_slots = torch.where(
        cols < lens,
        logical_kv_slots,
        torch.zeros((), dtype=torch.long, device=device),
    ).contiguous()
    logical_kv_starts = torch.full((num_paths,), past_len, dtype=torch.int32, device=device)
    logical_kv_lens = path_lens
    seq_lens_k = (logical_kv_starts + logical_kv_lens).contiguous()
    block_table = base_block_table.expand(num_paths, -1).contiguous()
    return block_table, seq_lens_k, logical_kv_slots, logical_kv_starts, logical_kv_lens


def _ptd_paged_tree_attn_forward(
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
    meta = cache._ptd_attn_meta        # {"seq_ids": [...], "qq_bias": tensor | None}
    if meta is None:                   # handoff on but meta unset -> clear error, not a TypeError
        raise RuntimeError(
            "ptd_paged_tree: paged-handoff active but _ptd_attn_meta is unset "
            "(the engine seam must set cache._ptd_attn_meta before the forward)"
        )
    seq_ids = meta["seq_ids"]
    B, Hq, S, D = query.shape
    q_flat = query.permute(0, 2, 1, 3).reshape(B * S, Hq, D)
    k_pool, v_pool = cache.pool(layer_idx)
    total_q = B * S
    ragged_cu = meta.get("cu_seqlens")
    if ragged_cu is None:
        block_table = cache.kernel_block_table(seq_ids, layer_idx, device=query.device)
        seq_lens_k = cache.kernel_seq_lens(seq_ids, layer_idx, device=query.device)
        cu = torch.arange(0, (B + 1) * S, S, dtype=torch.int32, device=query.device)
        logical_kv_slots = None
        logical_kv_starts = None
        logical_kv_lens = None
    else:
        if meta["qq_bias"] is not None:
            raise ValueError("ragged cu_seqlens requires qq_bias=None")
        cu = _validate_ragged_cu(ragged_cu, total_q, query.device)
        (
            block_table,
            seq_lens_k,
            logical_kv_slots,
            logical_kv_starts,
            logical_kv_lens,
        ) = _linearized_kernel_metadata(cache, layer_idx, seq_ids, cu, total_q, query.device)
    nqpkv = Hq // cache._num_heads
    qq_bias = meta["qq_bias"]
    if qq_bias is not None:            # kernel wants fp32, row-contiguous
        qq_bias = qq_bias.to(dtype=torch.float32).contiguous()
    out = paged_tree_attn(
        q_flat, k_pool, v_pool, block_table, cu, seq_lens_k,
        qq_bias, scaling, nqpkv, cache.block_size,
        logical_kv_slots, logical_kv_starts, logical_kv_lens,
    )
    return out.reshape(B, S, Hq, D), None


_REGISTERED = False


def register_ptd_paged_tree() -> None:
    """Register the paged tree-attention interface under `"ptd_paged_tree"` (idempotent).

    Registers ONLY in `ALL_ATTENTION_FUNCTIONS` (not `ALL_MASK_ATTENTION_FUNCTIONS`),
    so HF's `create_causal_mask` early-exits with `attention_mask=None` for this
    implementation (verified, transformers 4.57) — exactly the no-mask path the
    kernel needs (it does all masking itself)."""
    global _REGISTERED
    if _REGISTERED:
        return
    from transformers import AttentionInterface

    AttentionInterface().register("ptd_paged_tree", _ptd_paged_tree_attn_forward)
    _REGISTERED = True
