"""Compiled read-only tree-VERIFY forward (nano_vllm N3, A3-INT).

The verify leg of `generate_tree` runs the full Qwen3 stack over the N tree nodes
against the cached prefix. Going through HF's `model.__call__` pays the per-layer
Python metadata (mask construction, cache plumbing, `position_embeddings` packing,
attention-interface dispatch) and leaves the QKV/O/MLP/lm_head GEMMs + RoPE-cat +
elementwise unfused — so a per-round verify forward costs far more than one AR
forward even though it processes a comparable token count.

`CompiledVerifyStack` is a `torch.compile(fullgraph=True)` read-only stack that
BYPASSES `model.__call__` and reproduces EXACTLY the Qwen3 per-layer compute
(matched against the installed `Qwen3Attention.forward`): embed, then per layer
input-LN -> q/k/v proj -> q/k-norm (over head_dim, pre-transpose) -> RoPE ->
in-graph node-KV scatter into the block pool -> paged tree-attn (the opaque
`torch.ops.ptd.paged_tree_attn` custom_op fusion boundary) -> o-proj + residual,
MLP + residual; then final norm + lm_head. With the per-layer Python gone and the
attention kernel a single typed op, Inductor fuses the surrounding GEMMs.

The node-KV scatter happens IN-GRAPH: it keeps k/v live (Inductor can't DCE the
k/v projections) AND lands this round's nodes in the pool so the kernel reads
them, exactly mirroring the validated `a0_compile_fusion` prototype. The engine
seam reserves the node slots (via `PagedKVCache.reserve_tree_slots`) BEFORE the
call, so after verify the accepted-path `gather` reads those slots unchanged.

This unit is logits-only (`need_hidden=False`). The tapped-hidden variant (the
DraftHead path with `block_size > 1`) is a later unit; the engine falls back to
the eager kernel path for it.

SDPA stays the default + the lossless oracle. This module imports only `torch` at
scope (the custom_op + `apply_rotary_pos_emb` are bound at construction time over
the real model handles), so it stays importable on a CPU/no-triton host — the
`torch.ops.ptd.paged_tree_attn` op is registered by importing
`paged_tree_attn_op`, whose triton wrapper is itself lazily imported.
"""
import torch

from ptd.nano_vllm.paged_tree_attn_op import paged_tree_attn  # noqa: F401  (registers ptd::paged_tree_attn)


class CompiledVerifyStack:
    """A compiled, read-only Qwen3 tree-verify forward bound once over the real
    model handles. `__call__` runs the compiled stack and returns `(1, N, V)`
    logits — token-identical to the SDPA/kernel verify forward, at fused cost.

    Bound handles (from the loaded model):
      - `embed_tokens`, `layers` (each layer's `self_attn` / LN / `mlp`), `norm`,
        `lm_head`, and `apply_rotary_pos_emb` from the installed Qwen3 module.
    Per-call tensors come from the engine seam (RoPE cos/sin, the block pool +
    block table + per-seq key lengths, the ancestor `qq_bias`, and the reserved
    node-KV scatter indices)."""

    def __init__(self, model, block_size: int) -> None:
        # apply_rotary_pos_emb is bound from the SAME module HF dispatches through,
        # so RoPE is bit-identical to the model's own forward.
        from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

        self._apply_rotary_pos_emb = apply_rotary_pos_emb
        self.model = model
        cfg = model.config
        self.embed_tokens = model.model.embed_tokens
        self.layers = model.model.layers
        self.norm = model.model.norm
        self.lm_head = model.lm_head
        self.rotary_emb = model.model.rotary_emb

        self.num_heads_q = cfg.num_attention_heads
        self.num_heads_kv = cfg.num_key_value_heads
        self.head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
        self.num_queries_per_kv = self.num_heads_q // self.num_heads_kv   # nqpkv
        self.scaling = self.head_dim ** -0.5                              # head_dim ** -0.5
        self.block_size = int(block_size)

        # fullgraph=True so a graph break (e.g. an unexpected Python op leaking into
        # the trace) fails loudly rather than silently falling back to eager; the
        # paged tree-attn custom_op is the only opaque boundary. dynamic=False:
        # specialize on the concrete (N, block_count) — recompiles per distinct
        # shape (bucketing is a later unit).
        self._compiled = torch.compile(self._stack, fullgraph=True, dynamic=False)

    def _stack(
        self,
        input_ids,        # (1, N) long           tree-node token ids
        cos,              # (1, N, D)              RoPE cos for the node positions
        sin,              # (1, N, D)              RoPE sin for the node positions
        k_pools,          # list[(num_blocks, block_size, Hkv, D)] per layer
        v_pools,          # list[(num_blocks, block_size, Hkv, D)] per layer
        block_tables,     # list[(1, max_blocks) int32]  per layer (DIFFERENT block ids)
        cu,               # (2,) int32            == [0, N]
        seq_lens_k,       # (1,) int32            == past_len + N
        qq_bias,          # (N, N) fp32 (-inf/0) or None
        node_blks,        # list[(N,) long]       per layer: reserved pool block id per node
        node_offs,        # list[(N,) long]       per layer: reserved pool offset per node
    ):
        """Read-only Qwen3 forward over the N tree nodes -> `(1, N, V)` logits.

        Reproduces `Qwen3Attention.forward` per layer EXACTLY: q/k-norm over the
        head_dim BEFORE transpose, `apply_rotary_pos_emb` on the (1, H, N, D)
        layout (cos/sin unsqueeze_dim=1), scaling = head_dim**-0.5, GQA via
        nqpkv = Hq // Hkv. This round's node K/V is scattered into the pool in
        graph (keeps k/v live; the kernel then reads `[0, seq_lens_k)` = cached
        prefix + these nodes), then attention reads via the paged tree-attn
        custom_op under the ancestor `qq_bias`.

        Block tables / scatter maps are PER-LAYER: the pool assigns each layer its
        own physical blocks, so layer i uses `block_tables[i]` / `node_blks[i]` /
        `node_offs[i]` (a single shared grid would read/write the wrong blocks for
        layers other than the first)."""
        N = input_ids.shape[1]
        Hq, Hkv, Dh = self.num_heads_q, self.num_heads_kv, self.head_dim
        hidden = self.embed_tokens(input_ids)                # (1, N, hidden)
        hshape = (1, N, -1, Dh)
        for i, layer in enumerate(self.layers):
            attn = layer.self_attn
            residual = hidden
            h = layer.input_layernorm(hidden)
            q = attn.q_norm(attn.q_proj(h).view(hshape)).transpose(1, 2)    # (1, Hq, N, D)
            k = attn.k_norm(attn.k_proj(h).view(hshape)).transpose(1, 2)    # (1, Hkv, N, D)
            v = attn.v_proj(h).view(hshape).transpose(1, 2)                 # (1, Hkv, N, D)
            q, k = self._apply_rotary_pos_emb(q, k, cos, sin)
            # Scatter this layer's node K/V into ITS reserved pool slots in graph.
            # Keeps k/v live (Inductor can't DCE the k/v projections) AND lands the
            # nodes where the kernel reads them. (num_blocks, block_size, Hkv, D)
            # indexed by (node_blk, node_off) -> (N, Hkv, D).
            k_pools[i][node_blks[i], node_offs[i]] = k.transpose(1, 2).reshape(N, Hkv, Dh)
            v_pools[i][node_blks[i], node_offs[i]] = v.transpose(1, 2).reshape(N, Hkv, Dh)
            q_flat = q.permute(0, 2, 1, 3).reshape(N, Hq, Dh)              # (N, Hq, D)
            out = torch.ops.ptd.paged_tree_attn(
                q_flat, k_pools[i], v_pools[i], block_tables[i], cu, seq_lens_k,
                qq_bias, self.scaling, self.num_queries_per_kv, self.block_size,
            )
            attn_out = attn.o_proj(out.reshape(1, N, Hq * Dh))
            hidden = residual + attn_out
            residual = hidden
            h = layer.post_attention_layernorm(hidden)
            hidden = residual + layer.mlp(h)
        return self.lm_head(self.norm(hidden))                            # (1, N, V)

    def __call__(
        self,
        input_ids,
        cos,
        sin,
        k_pools,
        v_pools,
        block_tables,
        cu,
        seq_lens_k,
        qq_bias,
        node_blks,
        node_offs,
    ):
        """Run the compiled verify stack. Args mirror `_stack`; returns `(1, N, V)`."""
        return self._compiled(
            input_ids, cos, sin, k_pools, v_pools, block_tables, cu, seq_lens_k,
            qq_bias, node_blks, node_offs,
        )
