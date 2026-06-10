"""CUDA-graph capture+replay over the compiled tree-VERIFY stack (JetFlow N3, A3-GRAPH).

`CompiledVerifyStack` already removes the per-layer Python and fuses the GEMMs, but at
B=1 single-stream the residual cost is the per-KERNEL CPU launch/dispatch — each of the
~36 layers' QKV/O/MLP GEMMs + the paged tree-attn op + RoPE issues a separate launch the
CPU can't outrun the GPU on. `torch.compile` can't remove that (it still launches each
fused region eagerly); a captured CUDA graph collapses the whole forward into ONE
`cudaGraphLaunch`, so the launch storm disappears.

`GraphedVerify` wraps a built `CompiledVerifyStack` and, for each tree-N bucket in
`_TREE_BUCKETS`, captures one `torch.cuda.CUDAGraph` of the compiled forward under a
SINGLE shared graph pool (mirroring upstream nano-vllm-ref
`model_runner.capture_cudagraph`: a
per-bucket graph, persistent input/output buffers, copy-in then `graph.replay()`). A
captured graph reads FIXED device addresses, so every per-round input must live in a
pre-allocated persistent buffer the engine copies this round's values INTO before
replay — the engine's per-round `torch.tensor`/`torch.where`/`reserve_tree_slots`
allocations land at NEW addresses each round, which a captured graph would not see.

What is and isn't staged:
  - STAGED (persistent buffers, copied in each round): input_ids, cos, sin, qq_bias, cu,
    seq_lens_k, and the PER-LAYER node_blks / node_offs / block_tables. These are the
    tensors `reserve_tree_slots` + the RoPE/bias math freshly allocate per round.
  - REUSED IN PLACE (not staged): the k/v pools. After `reserve_capacity` the pool
    tensors are shape- AND address-stable (no `torch.cat` realloc), so the captured
    graph's reads/writes hit the live pool directly — exactly what we want, since the
    in-graph node-KV scatter must land this round's nodes where `gather` then reads them.

The in-graph node-KV scatter (`compiled_verify_stack._stack` line ~181:
`k_pools[i][node_blks[i], node_offs[i]] = ...`) CAPTURES correctly under manual
`torch.cuda.graph` — it is a device-side `index_put` into the stable pool with no host
sync. (The `torch.compile(mode="reduce-overhead")` AUTO-cudagraph skips this op as a
"mutated input", but that heuristic does not gate MANUAL capture; a capture-time spike
proved replay refills zeroed pool slots, i.e. the scatter re-runs on replay.) So
A3-KVOUT (moving the scatter out of the graph) is NOT needed.

Losslessness: replay recomputes the identical fp32 forward over the staged inputs +
live pool, so the logits/target_hidden are token-identical to the compiled-non-graph
path (which is itself the SDPA-oracle-equal verify). The compiled-non-graph,
eager-kernel, and SDPA paths are untouched and remain the oracles.

CPU/no-CUDA hosts: importing this module is safe (only `torch` at scope); constructing
`GraphedVerify` requires CUDA (it allocates device buffers and captures graphs).
"""
import torch


class GraphedVerify:
    """Per-bucket CUDA-graph capture+replay around a `CompiledVerifyStack`.

    Construct over a BUILT `CompiledVerifyStack` (logits-only or need_hidden) plus the
    live, post-`reserve_capacity` k/v pools and a representative set of per-round shapes
    (block-table width, RoPE head_dim, layer count). `capture(buckets)` traces+captures
    one graph per bucket; `replay(B, ...)` copies the round's inputs into the persistent
    buffers and launches the bucket-B graph, returning the persistent output buffer(s)
    sliced to the real node count.

    One instance per (need_hidden, target_layer_ids) compiled stack — mirroring the
    engine's one-stack-per-tap-set caching. The capture set is fixed: a full decode
    replays the pre-captured graphs and NEVER recaptures (capture count == #buckets).
    """

    def __init__(self, stack, k_pools, v_pools, block_table_width, head_dim,
                 hidden_size, device, dtype, buckets, logical_kv_bind=None):
        """Allocate persistent input-staging buffers sized to the largest bucket.

        `stack` is the built `CompiledVerifyStack` (its `__call__` runs the compiled
        forward). `k_pools` / `v_pools` are the live per-layer pools (reused in place,
        NOT staged — stable post-`reserve_capacity`). `block_table_width` is the fixed
        per-layer block-table column count (`cache.reserved_block_table_width`).
        `buckets` is the ordered tree-N bucket tuple (`_TREE_BUCKETS`).
        """
        self.stack = stack
        self.k_pools = k_pools
        self.v_pools = v_pools
        # L5 (no-gather): per-layer logical slot rows + starts/lens, or None. These
        # are REUSED IN PLACE like the pools — engine-owned, address-stable for the
        # decode, mutated by the engine before each replay — NOT staged/copied. A new
        # decode's fresh buffers change the engine-side pool_tag, forcing a rebuild.
        self.logical_kv_bind = logical_kv_bind
        self.nlayers = len(k_pools)
        self.block_table_width = int(block_table_width)
        self.device = device
        self.dtype = dtype
        self.buckets = tuple(int(b) for b in buckets)
        self.need_hidden = bool(getattr(stack, "need_hidden", False))
        Bmax = max(self.buckets)
        self.Bmax = Bmax

        # --- persistent input buffers (allocated ONCE at Bmax; sliced [:B] per bucket).
        # Every per-round input the engine freshly allocates lands here via copy_ before
        # replay; the captured graph reads these fixed addresses.
        self.g_input_ids = torch.zeros((1, Bmax), dtype=torch.long, device=device)
        self.g_cos = torch.zeros((1, Bmax, head_dim), dtype=dtype, device=device)
        self.g_sin = torch.zeros((1, Bmax, head_dim), dtype=dtype, device=device)
        # qq_bias is fp32 (-inf/0) — the compiled stack's bias dtype. Pre-fill to -inf so
        # any not-copied entry (e.g. the [B:Bmax] tail a smaller bucket leaves stale) is a
        # masked no-op; each round's real (B,B) block overwrites the [:B,:B] corner.
        self.g_qq_bias = torch.full((Bmax, Bmax), float("-inf"),
                                    dtype=torch.float32, device=device)
        self.g_cu = torch.zeros((2,), dtype=torch.int32, device=device)
        self.g_seq_lens_k = torch.zeros((1,), dtype=torch.int32, device=device)
        # Per-layer staged scatter maps + block tables (each layer has its OWN physical
        # block ids, so these are per-layer — mirroring reserve_tree_slots' return).
        self.g_node_blks = [torch.zeros((Bmax,), dtype=torch.long, device=device)
                            for _ in range(self.nlayers)]
        self.g_node_offs = [torch.zeros((Bmax,), dtype=torch.long, device=device)
                            for _ in range(self.nlayers)]
        self.g_block_tables = [torch.zeros((1, self.block_table_width),
                                           dtype=torch.int32, device=device)
                               for _ in range(self.nlayers)]

        self.graphs = {}          # B -> torch.cuda.CUDAGraph
        self.outputs = {}         # B -> logits buffer or (logits, target_hidden) tuple
        self._pool = None         # shared graph memory pool (set on first capture)
        # (id(live k-pool), block_table_width) tag — the engine sets it so it can detect a
        # new prompt's pool/width and rebuild rather than replay graphs bound to a freed
        # pool's addresses. Initialized here so the attribute always exists.
        self.pool_tag = None

    def _call_stack(self, B):
        """Run the wrapped compiled stack over the [:B] slices of the persistent buffers.

        Used for both the pre-capture warmup and the captured region. Slicing the
        persistent buffers yields views into their fixed storage, so the captured graph's
        reads land on the addresses the engine copies into each round (the reference
        `graph_vars['x'][:bs]` pattern). The k/v pools are passed whole (the stack indexes
        them by the staged node_blks/node_offs)."""
        lk = self.logical_kv_bind
        return self.stack(
            self.g_input_ids[:, :B],
            self.g_cos[:, :B],
            self.g_sin[:, :B],
            self.k_pools,
            self.v_pools,
            [bt for bt in self.g_block_tables],
            self.g_cu,
            self.g_seq_lens_k,
            self.g_qq_bias[:B, :B],
            [nb[:B] for nb in self.g_node_blks],
            [no[:B] for no in self.g_node_offs],
            logical_kv_slots=lk[0] if lk is not None else None,
            logical_kv_starts=lk[1] if lk is not None else None,
            logical_kv_lens=lk[2] if lk is not None else None,
        )

    @torch.inference_mode()
    def _capture_bucket(self, B):
        """Trace + capture the bucket-B graph, ASSUMING the persistent buffers already
        hold a valid round's inputs for size B (the caller copies them in first).

        Capturing against real inputs is load-bearing: the warmup runs (and the captured
        region itself) execute the in-graph node-KV scatter, which writes into the pool
        slots the staged `node_blks`/`node_offs` point at. Seeding the buffers with this
        round's real, freshly-reserved tree slots means the warmup scatter lands in those
        transient slots (overwritten by the very next replay, and freed by `gather`) — NOT
        into a stale block 0 it would corrupt if the indices were left zero. We warm twice
        (first call traces/compiles this bucket's `_stack` specialization; second runs the
        warm kernels) before `torch.cuda.graph`, so the captured region is pure launches.

        Captures reuse ONE shared graph pool (first capture seeds it). Under
        `inference_mode` for the same reason the reference capture is: the pool copy_ /
        in-graph scatter are inference-tensor writes that error outside it."""
        self._call_stack(B)
        self._call_stack(B)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, pool=self._pool):
            out = self._call_stack(B)
        if self._pool is None:
            self._pool = g.pool()
        self.graphs[B] = g
        self.outputs[B] = out
        torch.cuda.synchronize()

    @torch.inference_mode()
    def replay(self, B, input_ids, cos, sin, block_tables, cu, seq_lens_k,
               qq_bias, node_blks, node_offs, N):
        """Copy this round's inputs into the persistent buffers and replay graph[B].

        Arguments mirror the engine's per-round verify call (already padded to bucket B):
        `input_ids (1,B)`, `cos/sin (1,B,D)`, per-layer `block_tables[i] (1,W)`,
        `cu (2,)`, `seq_lens_k (1,)`, `qq_bias (B,B)`, per-layer `node_blks[i] (B,)` /
        `node_offs[i] (B,)`. `N` is the real (pre-pad) node count; the returned logits
        (and target_hidden when need_hidden) are sliced to `[:N]`, matching the
        compiled-non-graph path.

        On the FIRST replay for a given B (cold bucket) the graph is captured here —
        AFTER the inputs are copied in, so the warmup scatter uses this round's real
        reserved slots (see `_capture_bucket`). Every later replay for that B reuses the
        captured graph (capture count == #distinct buckets seen; no per-round recapture).
        All copies are non-blocking device-to-device into fixed addresses; `graph.replay()`
        then reruns the whole captured forward — including the in-graph node-KV scatter —
        over the live pool."""
        self.g_input_ids[:, :B].copy_(input_ids)
        self.g_cos[:, :B].copy_(cos)
        self.g_sin[:, :B].copy_(sin)
        self.g_qq_bias[:B, :B].copy_(qq_bias)
        self.g_cu.copy_(cu)
        self.g_seq_lens_k.copy_(seq_lens_k)
        for i in range(self.nlayers):
            self.g_block_tables[i].copy_(block_tables[i])
            self.g_node_blks[i][:B].copy_(node_blks[i])
            self.g_node_offs[i][:B].copy_(node_offs[i])
        if B not in self.graphs:
            self._capture_bucket(B)
        self.graphs[B].replay()
        out = self.outputs[B]
        if self.need_hidden:
            logits, target_hidden = out
            return logits[:, :N, :], target_hidden[:, :N, :]
        return out[:, :N, :]
