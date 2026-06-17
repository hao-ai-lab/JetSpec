"""``torch.library.custom_op`` wrapper for the paged tree-attention kernel.

The triton host wrapper in ``paged_tree_attn.py`` is correct and validated
(kernel == SDPA, e2e 13/13 on b200), but ``torch.compile(fullgraph=True)``
cannot trace *through* a ``@triton.jit`` launch — it would graph-break, which
defeats the whole point of compiling the read-only decoder stack (Inductor only
fuses the surrounding QKV/O/MLP/lm_head GEMMs if it can trace past attention).

Registering the kernel as a ``custom_op`` makes it an **opaque fusion boundary**:
``fullgraph=True`` sees a single typed op (real impl = the triton wrapper, fake
impl = a shape/dtype-only ``empty_like``) and fuses everything around it. The
fake (meta) kernel runs no triton, so compile / fake-tensor tracing works on any
host — including a CPU-only box without triton installed.

This module imports only ``torch`` at module scope; the triton host wrapper is
imported lazily inside the real implementation, so ``register_jetflow_paged_tree``
(and this module) stay importable on a CPU/no-triton host. Behavior on the real
CUDA path is unchanged — the op just forwards to the existing wrapper.

``qq_bias`` is optional (``None`` on the decode path); ``Tensor?`` in the
custom-op schema expresses this directly, so no bool flag or op split is needed.
"""
from typing import Optional

import torch


@torch.library.custom_op("jetflow::paged_tree_attn", mutates_args=())
def paged_tree_attn(
    q: torch.Tensor,            # (total_q, Hq, D)            post-RoPE queries, ragged-batched
    k_pool: torch.Tensor,       # (num_blocks, block_size, Hkv, D)  post-RoPE keys, paged
    v_pool: torch.Tensor,       # (num_blocks, block_size, Hkv, D)
    block_table: torch.Tensor,  # (num_seqs, max_blocks) int32
    cu_seqlens_q: torch.Tensor,  # (num_seqs+1,) int32
    seq_lens_k: torch.Tensor,   # (num_seqs,) int32           TOTAL key length per seq
    qq_bias: Optional[torch.Tensor],  # (total_q, total_q) fp32 additive (-inf/0); None for decode
    scale: float,               # head_dim ** -0.5
    num_queries_per_kv: int,    # Hq // Hkv
    block_size: int,
    logical_kv_slots: Optional[torch.Tensor] = None,   # (num_seqs, max_logical_slots) physical slot ids
    logical_kv_starts: Optional[torch.Tensor] = None,  # (num_seqs,) first logical key pos to remap
    logical_kv_lens: Optional[torch.Tensor] = None,    # (num_seqs,) number of logical key positions
) -> torch.Tensor:
    """Opaque ``custom_op`` over the triton paged tree-attention kernel.

    Real implementation: forwards verbatim to the triton host wrapper in
    ``paged_tree_attn.py`` (imported lazily so this module loads without triton).
    Returns ``(total_q, Hq, D)`` in ``q``'s dtype/device. See the wrapper's
    docstring for the per-seq attention contract."""
    from jetflow.inference_engine.paged_tree_attn import paged_tree_attn as _paged_tree_attn_impl

    return _paged_tree_attn_impl(
        q, k_pool, v_pool, block_table, cu_seqlens_q, seq_lens_k,
        qq_bias, scale, num_queries_per_kv, block_size,
        logical_kv_slots, logical_kv_starts, logical_kv_lens,
    )


@paged_tree_attn.register_fake
def _paged_tree_attn_fake(
    q,
    k_pool,
    v_pool,
    block_table,
    cu_seqlens_q,
    seq_lens_k,
    qq_bias,
    scale,
    num_queries_per_kv,
    block_size,
    logical_kv_slots=None,
    logical_kv_starts=None,
    logical_kv_lens=None,
):
    """Meta/fake kernel: output is ``(total_q, Hq, D)`` matching ``q`` exactly.

    Runs no triton — this is what compile / fake-tensor tracing calls, so the op
    is an opaque fusion boundary the surrounding GEMMs fuse around."""
    return torch.empty_like(q)
