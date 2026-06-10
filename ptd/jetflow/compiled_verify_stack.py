"""Compiled read-only tree-VERIFY forward (JetFlow N3, A3-INT).

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

Two construction-time variants, selected by the `need_hidden` Python-constant
flag (A3-HIDDEN):
  - `need_hidden=False` (A3-INT): logits-only. Inductor DCEs the residual stream
    once `lm_head(norm(...))` is taken, so untapped per-layer hidden never escapes.
  - `need_hidden=True` (A3-HIDDEN): the real DraftHead path. The stack ALSO returns
    the tapped target-layer hidden states the head conditions on. It captures the
    POST-layer residual at each `target_layer_id` (a Python-constant list baked at
    construction, so the tap set is a compile-time constant) and returns their
    `torch.cat` along the feature dim as a second output — matching
    `draft_head.extract_context_feature(out.hidden_states, target_layer_ids)`
    EXACTLY: that helper taps `out.hidden_states[L + 1]` (offset=1: index 0 is the
    embedding output), i.e. the OUTPUT of target layer `L` = the residual stream
    AFTER layer `L`. In `_stack`, that residual is the value of `hidden` once the
    loop body for layer index `L` completes. The two variants are SEPARATE compiled
    callables (one `torch.compile` per instance), so the False graph DCEs the taps
    the True graph keeps live.

A wrong tap does NOT break token-losslessness (each verify row is still
target-greedy, so the committed tokens match SDPA regardless), but it silently
feeds the DraftHead the wrong context and DROPS accept_len — so A3-HIDDEN is gated
on accept_len equality vs the eager kernel path, not just token equality.

SDPA stays the default + the lossless oracle. This module imports only `torch` at
scope (the custom_op + `apply_rotary_pos_emb` are bound at construction time over
the real model handles), so it stays importable on a CPU/no-triton host — the
`torch.ops.ptd.paged_tree_attn` op is registered by importing
`paged_tree_attn_op`, whose triton wrapper is itself lazily imported.
"""
import torch
import torch.nn.functional as F

from ptd.jetflow.paged_tree_attn_op import paged_tree_attn  # noqa: F401  (registers ptd::paged_tree_attn)


def _cat_linear_params(modules):
    """Build a fresh fused linear weight/bias tuple from separate projections."""
    weight = torch.cat([module.weight.detach() for module in modules], dim=0).contiguous()
    biases = [module.bias for module in modules]
    if all(bias is None for bias in biases):
        return weight, None
    bias_parts = []
    for module, bias in zip(modules, biases):
        if bias is None:
            bias_parts.append(torch.zeros(
                module.weight.shape[0],
                device=module.weight.device,
                dtype=module.weight.dtype,
            ))
        else:
            bias_parts.append(bias.detach())
    return weight, torch.cat(bias_parts, dim=0).contiguous()


class CompiledVerifyStack:
    """A compiled, read-only Qwen3 tree-verify forward bound once over the real
    model handles. `__call__` runs the compiled stack and returns either `(1, N, V)`
    logits (`need_hidden=False`) or `(logits, target_hidden)` where `target_hidden`
    is `(1, N, len(target_layer_ids)*H)` (`need_hidden=True`) — token-identical to
    the SDPA/kernel verify forward, at fused cost, with `target_hidden` byte-matching
    `extract_context_feature` over the same tapped layers.

    Bound handles (from the loaded model):
      - `embed_tokens`, `layers` (each layer's `self_attn` / LN / `mlp`), `norm`,
        `lm_head`, and `apply_rotary_pos_emb` from the installed Qwen3 module.
    Per-call tensors come from the engine seam (RoPE cos/sin, the block pool +
    block table + per-seq key lengths, the ancestor `qq_bias`, and the reserved
    node-KV scatter indices).

    `need_hidden` / `target_layer_ids` are construction-time Python constants (the
    tap set is baked into the traced graph). Build one instance per variant — the
    engine keeps a `need_hidden=False` stack for logits-only verifies and a
    `need_hidden=True` stack for the DraftHead path."""

    def __init__(self, model, block_size: int, need_hidden: bool = False,
                 target_layer_ids=None, fuse_gemms: bool = False) -> None:
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
        self.fuse_gemms = bool(fuse_gemms)

        # A3-HIDDEN: tapped-hidden variant constants. `need_hidden` and the tap set
        # are Python constants baked into the traced graph: `_stack` branches on
        # `self.need_hidden` and reads `self.target_layer_ids` at trace time, so the
        # False graph DCEs the residual stream (logits-only) and the True graph keeps
        # exactly the tapped post-layer residuals live and concatenated. Stored as a
        # tuple so it's hashable/immutable and the membership test below is a constant.
        self.need_hidden = bool(need_hidden)
        self.target_layer_ids = tuple(target_layer_ids) if target_layer_ids is not None else ()
        if self.need_hidden and not self.target_layer_ids:
            raise ValueError("need_hidden=True requires a non-empty target_layer_ids")
        # HF's `output_hidden_states` tuple stores the FINAL layer's entry POST the
        # model's final RMSNorm (lm_head(hidden_states[-1]) == logits exactly), while
        # every earlier entry is the raw pre-norm post-layer residual. So tapping the
        # last layer means tapping `norm(hidden)`, not the residual — match it, or the
        # last-layer tap silently mismatches `extract_context_feature` (accept_len
        # drop). Compile-time constant index.
        self._last_layer_idx = len(self.layers) - 1
        if self.fuse_gemms:
            # W13: construction-time fused projection tensors. The HF modules stay
            # intact for eager/SDPA fallback paths; the compiled stack reads only these
            # constants when fusion is enabled.
            qkv_weights, qkv_biases = [], []
            gate_up_weights, gate_up_biases = [], []
            for layer in self.layers:
                attn = layer.self_attn
                weight, bias = _cat_linear_params((attn.q_proj, attn.k_proj, attn.v_proj))
                qkv_weights.append(weight)
                qkv_biases.append(bias)
                mlp = layer.mlp
                weight, bias = _cat_linear_params((mlp.gate_proj, mlp.up_proj))
                gate_up_weights.append(weight)
                gate_up_biases.append(bias)
            self._fused_qkv_weights = tuple(qkv_weights)
            self._fused_qkv_biases = tuple(qkv_biases)
            self._fused_gate_up_weights = tuple(gate_up_weights)
            self._fused_gate_up_biases = tuple(gate_up_biases)
        else:
            self._fused_qkv_weights = ()
            self._fused_qkv_biases = ()
            self._fused_gate_up_weights = ()
            self._fused_gate_up_biases = ()

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
        logical_kv_slots=None,    # list[(1, max_slots) int64] per layer, or None
        logical_kv_starts=None,   # (1,) int32 shared, or None
        logical_kv_lens=None,     # (1,) int32 shared, or None
    ):
        """Read-only Qwen3 forward over the N tree nodes -> `(1, N, V)` logits, or
        `(logits, target_hidden)` when `need_hidden` (A3-HIDDEN).

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
        layers other than the first).

        When `need_hidden`, after each layer body completes `hidden` holds that
        layer's OUTPUT residual stream; we capture it for every layer index in the
        constant `target_layer_ids` set (matching `extract_context_feature`'s
        `hidden_states[L + 1]` tap) and `torch.cat` the captures along the feature
        dim in `target_layer_ids` order -> `(1, N, len(ids)*H)`."""
        N = input_ids.shape[1]
        Hq, Hkv, Dh = self.num_heads_q, self.num_heads_kv, self.head_dim
        hidden = self.embed_tokens(input_ids)                # (1, N, hidden)
        hshape = (1, N, -1, Dh)
        # A3-HIDDEN: collect the post-layer residual for each tapped layer. Keyed by
        # layer index so we can re-emit in `target_layer_ids` order (the head's fc
        # concatenates in that order); the membership test reads a constant set.
        tap_set = set(self.target_layer_ids)
        taps = {}
        for i, layer in enumerate(self.layers):
            attn = layer.self_attn
            residual = hidden
            h = layer.input_layernorm(hidden)
            if self.fuse_gemms:
                q_out = Hq * Dh
                kv_out = Hkv * Dh
                qkv = F.linear(h, self._fused_qkv_weights[i], self._fused_qkv_biases[i])
                q_raw, k_raw, v_raw = torch.split(qkv, (q_out, kv_out, kv_out), dim=-1)
                q = attn.q_norm(q_raw.view(1, N, Hq, Dh)).transpose(1, 2)     # (1, Hq, N, D)
                k = attn.k_norm(k_raw.view(1, N, Hkv, Dh)).transpose(1, 2)    # (1, Hkv, N, D)
                v = v_raw.view(1, N, Hkv, Dh).transpose(1, 2)                 # (1, Hkv, N, D)
            else:
                q = attn.q_norm(attn.q_proj(h).view(hshape)).transpose(1, 2)  # (1, Hq, N, D)
                k = attn.k_norm(attn.k_proj(h).view(hshape)).transpose(1, 2)  # (1, Hkv, N, D)
                v = attn.v_proj(h).view(hshape).transpose(1, 2)               # (1, Hkv, N, D)
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
                logical_kv_slots[i] if logical_kv_slots is not None else None,
                logical_kv_starts, logical_kv_lens,
            )
            attn_out = attn.o_proj(out.reshape(1, N, Hq * Dh))
            hidden = residual + attn_out
            residual = hidden
            h = layer.post_attention_layernorm(hidden)
            if self.fuse_gemms:
                mlp = layer.mlp
                gate_up = F.linear(
                    h,
                    self._fused_gate_up_weights[i],
                    self._fused_gate_up_biases[i],
                )
                gate, up = gate_up.chunk(2, dim=-1)
                hidden = residual + mlp.down_proj(mlp.act_fn(gate) * up)
            else:
                hidden = residual + layer.mlp(h)
            # Post-layer-`i` residual == extract_context_feature's hidden_states[i+1]
            # for every layer EXCEPT the last, whose HF entry is post-final-norm.
            if self.need_hidden and i in tap_set:
                taps[i] = self.norm(hidden) if i == self._last_layer_idx else hidden
        logits = self.lm_head(self.norm(hidden))                          # (1, N, V)
        if self.need_hidden:
            # Concatenate in target_layer_ids order (the head's fc expects that
            # layout); matches extract_context_feature(out.hidden_states, ids).
            target_hidden = torch.cat([taps[L] for L in self.target_layer_ids], dim=-1)
            return logits, target_hidden
        return logits

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
        logical_kv_slots=None,
        logical_kv_starts=None,
        logical_kv_lens=None,
    ):
        """Run the compiled verify stack. Args mirror `_stack`; returns `(1, N, V)`
        logits, or `(logits, target_hidden)` when this stack was built with
        `need_hidden=True`."""
        # The KV-block pool count (`k_pools[i].shape[0]`) is set by the engine's
        # per-prompt `reserve_capacity(prompt_len + max_new_tokens + budget)`, so it
        # differs across prompts (and between a short warmup and a long decode).
        # `torch.compile(dynamic=False)` would specialize on that block-count and
        # recompile the whole 36-layer stack for every distinct pool size — a recompile
        # storm that blows dynamo's `cache_size_limit` and SILENTLY falls back to eager
        # (the verify forward then runs unfused, collapsing decode_cuda_speedup to ~2×).
        # Mark the pool block-dim dynamic so the stack compiles ONCE with a symbolic
        # block-count and is reused for any prompt length; only dim 0 (num_blocks)
        # varies — block_size / Hkv / head_dim stay static, and the in-graph scatter +
        # paged_tree_attn op both tolerate a symbolic block-count. (`mark_dynamic` is
        # honored even under `dynamic=False`; it is a cheap idempotent flag-set, a no-op
        # once the symbolic graph exists.)
        for t in k_pools:
            torch._dynamo.mark_dynamic(t, 0)
        for t in v_pools:
            torch._dynamo.mark_dynamic(t, 0)
        # Same story for the per-layer block tables: their column count is the reserved
        # block-table width = ceil(reserve_capacity / block_size), which also tracks
        # prompt length, so it too varies prompt-to-prompt and would re-trigger the
        # specialize-recompile. Mark the width (dim 1) dynamic; the kernel indexes the
        # table by runtime `seq_lens_k`, so a symbolic column count is safe.
        for t in block_tables:
            torch._dynamo.mark_dynamic(t, 1)
        if logical_kv_slots is not None:
            for t in logical_kv_slots:
                torch._dynamo.mark_dynamic(t, 1)
        return self._compiled(
            input_ids, cos, sin, k_pools, v_pools, block_tables, cu, seq_lens_k,
            qq_bias, node_blks, node_offs,
            logical_kv_slots, logical_kv_starts, logical_kv_lens,
        )
