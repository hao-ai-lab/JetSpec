"""Block-paged KV cache (nano_vllm N0 -> N2a).

A drop-in for HF's `DynamicCache` that swaps contiguous per-layer `(B, H, S, D)`
storage for a fixed-block pool (`block_size` tokens per block) plus a per-layer
block table. Conforms to `transformers.cache_utils.Cache` so it plugs straight
into `model(..., past_key_values=...)`: the model only ever calls `update`,
`get_seq_length`, and `get_mask_sizes` on us (verified against transformers
4.57), and `update` returns the *logical* (B, H, S, D) view that SDPA attends
over — the paging is invisible to the model.

N0 scope: single stream (batch=1), one sequence. The interesting bit is
`gather`, which compacts a non-contiguous keep set (a tree-verify accepted path)
back into a linear prefix at block granularity — the paged analogue of
`ptd.engine.llm._select_kv_cache`. N1 builds the tree-spec loop on it.

N2a scope (this revision): multi-sequence pooling over the SAME shared pool, with
per-sequence block tables, block ref-counting, a fixed-size pool, and LRU
eviction. The pool is keyed by `seq_id`; every public method takes an optional
`seq_id` keyword that defaults to `_default_seq_id` (0) so the N0/N1 single-stream
paths are byte-identical (they never name a seq, so they get seq 0). With the
default `max_batch_size=1` the pool still grows on demand (N0 behaviour); set
`max_batch_size>1` to switch to the fixed pool + ref-counting + eviction (N2a).

Ref-counting lets sequences share blocks (prompt-prefix sharing, tree-node reuse)
and makes eviction safe: a block returns to the free pool only when the LAST
sequence referencing it is freed. See `_decref_block` / `free` and the N2a tests.

KV layout: HF hands `update` keys/values shaped `(batch=1, num_heads, seq, head_dim)`
(seq on dim -2). Pool blocks store `(num_blocks, block_size, num_heads, head_dim)`,
so packing is a slice-assign on the block/offset axes and unpacking concatenates
the used slots back to `(1, num_heads, used, head_dim)`. Keys and values share one
block table; a parallel value pool mirrors the key pool slot-for-slot.
"""
from typing import Any, Optional

import torch
from transformers.cache_utils import Cache


class EvictionRequired(Exception):
    """Raised by `allocate`/`append` when the fixed pool is exhausted and the
    caller (scheduler) must evict a sequence before retrying."""


class EvictionFailed(Exception):
    """Raised by `evict_sequence` when the named sequence is unkillable (e.g. it
    is the only running sequence, so evicting it cannot free anyone else's room)."""


class PagedHandle:
    """Opaque K/V handle returned by `update` in paged-handoff mode (N3 kernel path).

    The triton tree-attention kernel reads K/V straight from the block pool, so on
    the kernel path `update` does NOT reconstruct the dense `(1, H, S, D)` view —
    it appends the new KV and returns one of these per K/V. The handle carries
    `(cache, layer_idx, which)` so the registered attention fn can pull the pool +
    block tables back out (Qwen3Attention forwards `update`'s return straight into
    the attention interface, touching nothing in between — verified, transformers
    4.57). `shape`/`dtype`/`device` are cheap logical-view properties only as a
    safety net in case any HF code path inspects them; the kernel never uses them."""

    __slots__ = ("cache", "layer_idx", "which")

    def __init__(self, cache: "PagedKVCache", layer_idx: int, which: str) -> None:
        self.cache = cache
        self.layer_idx = layer_idx
        self.which = which            # "k" or "v"

    @property
    def shape(self) -> torch.Size:
        sid = self.cache._handoff_seq_ids[0]
        seq_len = self.cache.get_seq_length(self.layer_idx, seq_id=sid)
        return torch.Size((1, self.cache._num_heads, seq_len, self.cache._head_dim))

    @property
    def dtype(self) -> torch.dtype:
        return self.cache.dtype

    @property
    def device(self) -> torch.device:
        return self.cache.device


class PagedKVCache(Cache):
    """Block-paged KV store conforming to the HF `Cache` interface.

    Per layer we keep two pool tensors `(num_blocks, block_size, num_heads, head_dim)`
    (keys, values) SHARED across all sequences, and per-sequence block tables
    (block ids in sequence order). The last block of each (seq, layer) may be
    partially filled; `_seq_filled[seq_id][layer_idx]` tracks how many tokens it
    holds. With `max_batch_size=1` the pool grows on demand (N0); with
    `max_batch_size>1` it is fixed at `max_total_tokens // block_size` blocks and
    ref-counting + LRU eviction manage contention (N2a).
    """

    _default_seq_id: int = 0

    def __init__(
        self,
        block_size: int = 16,
        max_batch_size: int = 1,
        max_total_tokens: int = 262144,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ) -> None:
        # We never call Cache.__init__ (it wants a layer list / layer_class); our
        # storage is the pools below, and we override every method HF touches.
        # `max_batch_size` etc. are read-only properties on the base class derived
        # from `self.layers`, which we don't keep — store ours privately instead.
        self._block_size = int(block_size)
        self._max_batch_size = int(max_batch_size)
        self._max_total_tokens = int(max_total_tokens)
        # N0 (single-seq) grows the pool on demand; N2a (multi-seq) fixes it.
        self._is_single_seq = self._max_batch_size == 1
        self.device = torch.device(device)
        self.dtype = dtype
        # Lazily sized on the first append (we learn num_heads/head_dim from the
        # KV the model hands us, mirroring DynamicLayer's lazy_initialization).
        self._num_heads: Optional[int] = None
        self._head_dim: Optional[int] = None
        self._kpool: dict[int, torch.Tensor] = {}       # layer_idx -> (num_blocks, block_size, H, D)
        self._vpool: dict[int, torch.Tensor] = {}
        # Per-sequence block tables + partial-last-block fill counts.
        self._seq_block_tables: dict[int, dict[int, list[int]]] = {}  # seq_id -> (layer_idx -> [block_id, ...])
        self._seq_filled: dict[int, dict[int, int]] = {}             # seq_id -> (layer_idx -> tokens in last block)
        # Block ref-counting (shared ownership). A block in `_free_blocks` has
        # refcount 0 (and is absent from `_block_refcounts`); a referenced block
        # has refcount >= 1 (and is absent from `_free_blocks`).
        self._block_refcounts: dict[int, int] = {}      # block_id -> count
        # LRU clock: monotonically increasing tick; per-seq last-touch for eviction.
        self._seq_last_used: dict[int, int] = {}        # seq_id -> tick
        self._clock: int = 0
        # Per-sequence metadata (telemetry: prompt_len, computed_len, priority, …).
        self._seq_metadata: dict[int, dict] = {}
        # N3 paged-handoff mode (opt-in; default off keeps the dense SDPA path
        # byte-identical). When on, `update` appends the new KV (per-row routed by
        # `_handoff_seq_ids`) and returns `PagedHandle`s instead of the dense view;
        # the registered attention fn reads `_ptd_attn_meta` (seq order + qq_bias)
        # and pulls K/V straight from the pool. Set by the engine seam, never by HF.
        self._paged_handoff = False
        self._handoff_seq_ids: Optional[list[int]] = None
        self._ptd_attn_meta: Optional[dict] = None
        if self._is_single_seq:
            # N0/N1: pool grows on demand once dims are known.
            self._free_blocks: list[int] = []
            self._num_blocks = 0
        else:
            # N2a: fixed pool — pre-reserve every block id up front.
            self._num_blocks = self._max_total_tokens // self._block_size
            self._free_blocks = list(range(self._num_blocks))

    # --- internal pool management -------------------------------------------

    def _grow_pool(self, extra_blocks: int) -> None:
        """Append `extra_blocks` empty blocks to every layer's pools, freeing them.

        Only used in single-seq (N0) mode; the N2a fixed pool never grows."""
        old = self._num_blocks
        self._num_blocks = old + extra_blocks
        for layer_idx in self._kpool:
            shape = (extra_blocks, self._block_size, self._num_heads, self._head_dim)
            self._kpool[layer_idx] = torch.cat(
                [self._kpool[layer_idx], torch.zeros(shape, dtype=self.dtype, device=self.device)], dim=0)
            self._vpool[layer_idx] = torch.cat(
                [self._vpool[layer_idx], torch.zeros(shape, dtype=self.dtype, device=self.device)], dim=0)
        self._free_blocks.extend(range(old, self._num_blocks))

    def _ensure_layer(self, layer_idx: int, key_states: torch.Tensor) -> None:
        """Create this layer's pools on first touch, learning dims from KV."""
        if self._num_heads is None:
            self._num_heads = key_states.shape[1]
            self._head_dim = key_states.shape[-1]
        if layer_idx not in self._kpool:
            shape = (self._num_blocks, self._block_size, self._num_heads, self._head_dim)
            self._kpool[layer_idx] = torch.zeros(shape, dtype=self.dtype, device=self.device)
            self._vpool[layer_idx] = torch.zeros(shape, dtype=self.dtype, device=self.device)

    def _ensure_seq(self, seq_id: int) -> None:
        """Create this sequence's per-layer block table + fill maps on first touch."""
        if seq_id not in self._seq_block_tables:
            self._seq_block_tables[seq_id] = {}
            self._seq_filled[seq_id] = {}
            self._seq_metadata.setdefault(seq_id, {})

    def _touch(self, seq_id: int) -> None:
        """Bump the LRU clock for `seq_id` (most-recently-used last)."""
        self._clock += 1
        self._seq_last_used[seq_id] = self._clock

    def allocate(self, num_blocks: int) -> torch.Tensor:
        """Pre-allocate `num_blocks` blocks from the free pool.

        Returns the allocated block ids `(num_blocks,)` (ascending, for
        deterministic reuse). In single-seq (N0) mode the pool grows on demand;
        in N2a (fixed pool) it raises `EvictionRequired` when the pool is
        exhausted so the scheduler can evict and retry. Allocation does NOT
        touch refcounts — the seq claims the block (increfs) in append/gather."""
        if num_blocks <= 0:
            return torch.empty(0, dtype=torch.long, device=self.device)
        if len(self._free_blocks) < num_blocks:
            if self._is_single_seq:
                if self._num_heads is None:
                    raise RuntimeError("cannot allocate before the first append (dims unknown)")
                self._grow_pool(num_blocks - len(self._free_blocks))
            else:
                raise EvictionRequired(
                    f"need {num_blocks} blocks, have {len(self._free_blocks)}"
                )
        ids = [self._free_blocks.pop(0) for _ in range(num_blocks)]
        return torch.tensor(ids, dtype=torch.long, device=self.device)

    # --- ref-counting --------------------------------------------------------

    def _incref_block(self, block_id: int) -> None:
        """Increment a block's refcount (called when a seq claims it in append/gather).

        The block must already be out of the free pool (claimed via `allocate` or
        a caller reservation) — incrementing a free block is a refcount-corruption
        bug, so we assert it."""
        assert block_id not in self._free_blocks, \
            f"incref on free block {block_id} (refcount corruption)"
        self._block_refcounts[block_id] = self._block_refcounts.get(block_id, 0) + 1

    def _decref_block(self, block_id: int) -> None:
        """Decrement a block's refcount; return it to the free pool at 0.

        Precondition: `block_id` is referenced (in `_block_refcounts`, count >= 1).
        A block reaching 0 is the LAST owner releasing it, so it becomes free; a
        block staying > 0 is still owned by another sequence (shared prefix / tree
        reuse) and survives — this is what makes eviction safe."""
        if block_id not in self._block_refcounts:
            raise RuntimeError(f"decref on unreferenced block {block_id}")
        self._block_refcounts[block_id] -= 1
        if self._block_refcounts[block_id] == 0:
            del self._block_refcounts[block_id]
            self._free_blocks.append(block_id)
            self._free_blocks.sort()                      # sorted -> deterministic reuse
        elif self._block_refcounts[block_id] < 0:
            raise RuntimeError(f"double-decref on block {block_id}")

    # --- append / gather -----------------------------------------------------

    def append(
        self,
        key_states: torch.Tensor,      # (batch=1, num_heads, seq_new, head_dim)
        value_states: torch.Tensor,    # (batch=1, num_heads, seq_new, head_dim)
        layer_idx: int,
        block_ids: Optional[torch.Tensor] = None,
        seq_id: Optional[int] = None,
    ) -> int:
        """Pack `seq_new` tokens' KV into `seq_id`'s blocks for `layer_idx` (in place).

        Fills the current last block first, then allocates blocks for the
        remainder (`block_ids` overrides the allocation if pre-reserved), increffing
        every newly-claimed block for this seq. Returns the new sequence length so
        HF can track `get_seq_length`. KV is the HF layout
        `(1, num_heads, seq, head_dim)` (seq on dim -2). `seq_id` defaults to seq 0
        for the N0/N1 single-stream paths."""
        seq_id = self._default_seq_id if seq_id is None else seq_id
        self._ensure_layer(layer_idx, key_states)
        self._ensure_seq(seq_id)
        keys = key_states[0].transpose(0, 1)             # (seq_new, num_heads, head_dim)
        values = value_states[0].transpose(0, 1)
        seq_new = keys.shape[0]
        if seq_new == 0:
            return self.get_seq_length(layer_idx, seq_id=seq_id)
        table = self._seq_block_tables[seq_id].setdefault(layer_idx, [])
        filled = self._seq_filled[seq_id].setdefault(layer_idx, 0)
        start = self.get_seq_length(layer_idx, seq_id=seq_id)   # tokens before this append (pre-extend)

        # New blocks beyond the room left in the current last block.
        room = (self._block_size - filled) if table else 0
        need = max(0, seq_new - room)
        new_block_count = (need + self._block_size - 1) // self._block_size
        if block_ids is not None:
            if block_ids.numel() != new_block_count:
                raise ValueError(f"block_ids has {block_ids.numel()} blocks, need {new_block_count}")
            new_blocks = [int(b) for b in block_ids.tolist()]
            for b in new_blocks:                         # claim the caller's reservation
                self._free_blocks.remove(b)
                self._incref_block(b)
        else:
            new_blocks = [int(b) for b in self.allocate(new_block_count).tolist()]
            for b in new_blocks:
                self._incref_block(b)
        table.extend(new_blocks)

        # Write keys+values to the same slots: the last `seq_new` positions.
        # Vectorized scatter (ONE kernel per pool) — the old per-token loop issued a
        # single-element DtoD memcpy per (token, layer), which profiled as ~40% of
        # tree-verify GPU time (228k DtoD copies/run). blk/off are computed by a
        # tensor gather over the (small) block table; the writes are identical.
        kpool, vpool = self._kpool[layer_idx], self._vpool[layer_idx]
        pos_t = start + torch.arange(seq_new, device=self.device)
        table_t = torch.tensor(table, device=self.device, dtype=torch.long)
        blk = table_t[pos_t // self._block_size]
        off = pos_t % self._block_size
        kpool[blk, off] = keys
        vpool[blk, off] = values
        self._seq_filled[seq_id][layer_idx] = ((start + seq_new - 1) % self._block_size) + 1
        self._touch(seq_id)
        return self.get_seq_length(layer_idx, seq_id=seq_id)

    def gather(self, positions: torch.Tensor, seq_id: Optional[int] = None) -> None:
        """Compact `seq_id`'s non-contiguous keep set into a linear prefix (all layers).

        `positions` is a 1-D LongTensor of cache positions to keep (the tree-verify
        accepted path: `[past, past+1, …, past+acc]`, possibly scattered in
        tree-order before this call). After gather the kept KV occupies positions
        `[0, 1, …, len(positions)-1]` in fresh blocks; the old blocks are decreffed
        (freed only if no other seq still references them) — the paged analogue of
        `ptd.engine.llm._select_kv_cache`. `seq_id` defaults to seq 0."""
        seq_id = self._default_seq_id if seq_id is None else seq_id
        idx = positions.to(self.device).long()
        keep = idx.numel()
        new_block_count = (keep + self._block_size - 1) // self._block_size
        tables = self._seq_block_tables.get(seq_id, {})
        for layer_idx in list(tables.keys()):
            table = tables[layer_idx]
            old_blocks = list(table)
            src_blk = torch.tensor([table[int(p) // self._block_size] for p in idx], device=self.device)
            src_off = torch.tensor([int(p) % self._block_size for p in idx], device=self.device)
            gathered_k = self._kpool[layer_idx][src_blk, src_off]      # (keep, H, D)
            gathered_v = self._vpool[layer_idx][src_blk, src_off]
            # Decref old blocks BEFORE allocating new ones so freed slots are
            # reusable in-place (the fixed N2a pool may need them for the compaction).
            for b in old_blocks:
                self._decref_block(b)
            new_blocks = [int(b) for b in self.allocate(new_block_count).tolist()]
            for b in new_blocks:
                self._incref_block(b)
            tables[layer_idx] = list(new_blocks)
            for j, blk in enumerate(new_blocks):
                lo, hi = j * self._block_size, min((j + 1) * self._block_size, keep)
                self._kpool[layer_idx][blk, : hi - lo] = gathered_k[lo:hi]
                self._vpool[layer_idx][blk, : hi - lo] = gathered_v[lo:hi]
            self._seq_filled[seq_id][layer_idx] = (
                keep - (new_block_count - 1) * self._block_size if new_block_count else 0
            )
        if tables:
            self._touch(seq_id)

    # --- multi-seq free / status (N2a) --------------------------------------

    def free(self, arg=None, seq_id: Optional[int] = None) -> None:
        """Release blocks.

        Two roles, disambiguated by argument type (kept for backward compat):
        - `free(block_ids: torch.Tensor)`: legacy N0 helper — return raw block ids
          to the free pool unconditionally (used by `crop`). Block ids are
          decreffed if tracked, else pushed straight back.
        - `free(seq_id=<int>)` / `free(<int>)`: N2a — release every block owned by
          `seq_id`, decreffing each (a block stays allocated while another seq
          still references it), then drop the seq's tables/fill/metadata.

        Called on sequence finish or eviction (N2a) or block reclaim (N0)."""
        if isinstance(arg, torch.Tensor):
            # Legacy: return raw block ids to the pool (crop / N0 stale-block path).
            for b in arg.tolist():
                bi = int(b)
                if bi in self._block_refcounts:
                    self._decref_block(bi)
                elif bi not in self._free_blocks:
                    self._free_blocks.append(bi)
            self._free_blocks = sorted(set(self._free_blocks))
            return

        # N2a: free a whole sequence.
        sid = arg if isinstance(arg, int) else seq_id
        sid = self._default_seq_id if sid is None else sid
        for layer_idx, table in self._seq_block_tables.get(sid, {}).items():
            for block_id in table:
                self._decref_block(block_id)
        self._seq_block_tables.pop(sid, None)
        self._seq_filled.pop(sid, None)
        self._seq_last_used.pop(sid, None)
        self._seq_metadata.pop(sid, None)

    def get_seq_status(self, seq_id: Optional[int] = None) -> dict[str, Any]:
        """Telemetry snapshot for `seq_id`: lengths, block count, last-used tick."""
        seq_id = self._default_seq_id if seq_id is None else seq_id
        tables = self._seq_block_tables.get(seq_id, {})
        meta = self._seq_metadata.get(seq_id, {})
        layers = sorted(tables.keys())
        num_blocks = sum(len(t) for t in tables.values())
        return {
            "seq_id": seq_id,
            "computed_len": self.get_seq_length(layers[0], seq_id=seq_id) if layers else 0,
            "input_len": meta.get("prompt_len", 0),
            "num_blocks_allocated": num_blocks,
            "num_layers": len(layers),
            "last_used": self._seq_last_used.get(seq_id, 0),
        }

    # --- block pool management (N2a: fixed + LRU eviction) ------------------

    def _blocks_for(self, num_tokens: int) -> int:
        """Ceil-div tokens to blocks."""
        return (num_tokens + self._block_size - 1) // self._block_size

    def admit_sequence(self, seq_id: int, prompt_len: int) -> bool:
        """Can `seq_id` be admitted to the (fixed) pool for `prompt_len` tokens?

        True if enough free blocks exist now, or can be freed by evicting other
        sequences (LRU). Does not actually evict — the scheduler calls
        `allocate_slots` / `evict_sequence` to commit. Records `prompt_len` for
        telemetry. In single-seq mode the pool grows, so admission is always True."""
        self._seq_metadata.setdefault(seq_id, {})["prompt_len"] = prompt_len
        if self._is_single_seq:
            return True
        need_blocks = self._blocks_for(prompt_len)
        if len(self._free_blocks) >= need_blocks:
            return True
        # Count blocks reclaimable by evicting other (not seq_id) sequences. Only
        # blocks whose refcount would hit 0 (sole owner = the evicted seq) free up;
        # shared blocks survive. We approximate with the seq's owned unique blocks.
        reclaimable = 0
        for other in self._seq_block_tables:
            if other == seq_id:
                continue
            reclaimable += self._sole_owned_block_count(other)
        return len(self._free_blocks) + reclaimable >= need_blocks

    def _sole_owned_block_count(self, seq_id: int) -> int:
        """Number of blocks `seq_id` owns whose refcount is exactly 1 (would free)."""
        seen: set[int] = set()
        for table in self._seq_block_tables.get(seq_id, {}).values():
            for b in table:
                seen.add(b)
        return sum(1 for b in seen if self._block_refcounts.get(b, 0) == 1)

    def evict_sequence(self, seq_id: Optional[int] = None) -> int:
        """Evict a sequence (named, or the LRU one) and return blocks freed.

        With `seq_id=None`, picks the least-recently-used sequence. Refuses to
        evict the only running sequence (nothing would be gained) by raising
        `EvictionFailed`. Frees the seq via `free`, which decrefs — shared blocks
        survive for their other owners."""
        if not self._seq_block_tables:
            raise EvictionFailed("no sequences to evict")
        if seq_id is None:
            # LRU: lowest last-used tick (unseen seqs sort first via default 0).
            seq_id = min(self._seq_block_tables, key=lambda s: self._seq_last_used.get(s, 0))
        if len(self._seq_block_tables) == 1 and seq_id in self._seq_block_tables:
            raise EvictionFailed(f"cannot evict the only running sequence {seq_id}")
        free_before = len(self._free_blocks)
        self.free(seq_id=seq_id)
        return len(self._free_blocks) - free_before

    def allocate_slots(self, seq_id: int, num_tokens: int) -> bool:
        """Pre-reserve enough blocks for `num_tokens`, evicting (LRU) if needed.

        Called by the scheduler BEFORE a forward. Returns True on success; False if
        even after evicting every other sequence the pool can't fit `num_tokens`
        (the caller must reject / shrink the request). The reserved blocks are
        recorded under `seq_id`'s metadata for the next append's `block_ids`."""
        if self._is_single_seq:
            return True
        need = self._blocks_for(num_tokens)
        while len(self._free_blocks) < need:
            try:
                self.evict_sequence(seq_id=None if seq_id not in self._seq_block_tables
                                    else self._lru_other(seq_id))
            except EvictionFailed:
                return len(self._free_blocks) >= need
        return True

    def _lru_other(self, seq_id: int) -> Optional[int]:
        """LRU sequence that is not `seq_id` (for self-preserving eviction)."""
        others = [s for s in self._seq_block_tables if s != seq_id]
        if not others:
            return None
        return min(others, key=lambda s: self._seq_last_used.get(s, 0))

    # --- HF transformers.Cache interface ------------------------------------

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """HF forward callback: append the new KV, then return the *logical*
        contiguous `(1, num_heads, seq, head_dim)` view SDPA attends over.

        `cache_kwargs["seq_id"]` selects the sequence (N2a batched forward); absent
        (N0/N1), it defaults to seq 0. Sliding-window metadata etc. is ignored.

        In paged-handoff mode (N3 kernel path) the new KV is appended (per-row
        routed by `_handoff_seq_ids`) and `PagedHandle`s are returned instead of
        the dense view — the registered attention fn reads straight from the pool,
        so no `_logical_kv` reconstruction happens. The dense path below is the
        unchanged default."""
        if self._paged_handoff:
            seq_ids = self._handoff_seq_ids
            if seq_ids is not None and len(seq_ids) > 1:
                # Batched (N2a): route each batch row to its own seq_id.
                for i, sid in enumerate(seq_ids):
                    self.append(key_states[i:i + 1], value_states[i:i + 1], layer_idx, seq_id=sid)
            else:
                sid = seq_ids[0] if seq_ids else None
                self.append(key_states, value_states, layer_idx, seq_id=sid)
            return PagedHandle(self, layer_idx, "k"), PagedHandle(self, layer_idx, "v")
        seq_id = cache_kwargs.get("seq_id") if cache_kwargs else None
        self.append(key_states, value_states, layer_idx, seq_id=seq_id)
        return self._logical_kv(layer_idx, seq_id=seq_id)

    def _logical_kv(self, layer_idx: int, seq_id: Optional[int] = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Reconstruct the dense `(1, num_heads, seq, head_dim)` KV for `seq_id`
        from its blocks. `seq_id` defaults to seq 0."""
        seq_id = self._default_seq_id if seq_id is None else seq_id
        table = self._seq_block_tables.get(seq_id, {}).get(layer_idx)
        if not table:                                    # empty layer/seq
            empty = torch.zeros((1, self._num_heads, 0, self._head_dim),
                                dtype=self.dtype, device=self.device)
            return empty, empty.clone()
        kpool, vpool = self._kpool[layer_idx], self._vpool[layer_idx]
        filled = self._seq_filled[seq_id][layer_idx]
        kparts, vparts = [], []
        for i, blk in enumerate(table):
            used = self._block_size if i < len(table) - 1 else filled
            kparts.append(kpool[blk, :used])             # (used, H, D)
            vparts.append(vpool[blk, :used])
        keys = torch.cat(kparts, dim=0)                  # (seq, H, D)
        values = torch.cat(vparts, dim=0)
        return (keys.transpose(0, 1).unsqueeze(0).contiguous(),    # (1, H, seq, D)
                values.transpose(0, 1).unsqueeze(0).contiguous())

    def get_seq_length(self, layer_idx: int = 0, seq_id: Optional[int] = None) -> int:
        """Current cached length for `(seq_id, layer_idx)` (full blocks + partial last).

        `seq_id` defaults to seq 0 (N0/N1)."""
        seq_id = self._default_seq_id if seq_id is None else seq_id
        table = self._seq_block_tables.get(seq_id, {}).get(layer_idx)
        if not table:
            return 0
        return (len(table) - 1) * self._block_size + self._seq_filled[seq_id][layer_idx]

    def crop(self, max_length: int, seq_id: Optional[int] = None) -> None:
        """Trim every layer of `seq_id` to its first `max_length` tokens.

        `seq_id` defaults to seq 0 (N0/N1). Dropped blocks are decreffed back to
        the pool (freed if no other seq references them)."""
        seq_id = self._default_seq_id if seq_id is None else seq_id
        tables = self._seq_block_tables.get(seq_id, {})
        for layer_idx in list(tables.keys()):
            cur = self.get_seq_length(layer_idx, seq_id=seq_id)
            target = cur - abs(max_length) if max_length < 0 else max_length
            if cur <= target:
                continue
            keep_blocks = (target + self._block_size - 1) // self._block_size
            table = tables[layer_idx]
            stale = table[keep_blocks:]
            tables[layer_idx] = table[:keep_blocks]
            self._seq_filled[seq_id][layer_idx] = (
                (target - (keep_blocks - 1) * self._block_size) if keep_blocks else 0
            )
            for b in stale:
                self._decref_block(b)

    def reset(self) -> None:
        """Clear all sequence state, returning every block to the pool."""
        self._free_blocks = list(range(self._num_blocks))
        self._block_refcounts = {}
        self._seq_block_tables = {}
        self._seq_filled = {}
        self._seq_last_used = {}
        self._seq_metadata = {}

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        """Beam reorder — nano is greedy; raise to catch misuse."""
        raise NotImplementedError("PagedKVCache N2a does not support beam search")

    def batch_repeat_interleave(self, repeats: int) -> None:
        raise NotImplementedError("PagedKVCache N2a does not support speculative / repeat")

    def batch_select_indices(self, indices: torch.Tensor) -> None:
        raise NotImplementedError("PagedKVCache N2a does not support index-based reordering")

    def get_max_cache_shape(self, layer_idx: int = 0) -> int:
        """Max tokens the current pool can hold (grows on demand in N0; fixed in N2a)."""
        return self._num_blocks * self._block_size

    def get_mask_sizes(self, cache_position: torch.Tensor, layer_idx: int) -> tuple:
        """(kv_length, kv_offset) for the 4D attention mask — mirrors DynamicLayer:
        past length + this step's query length, no offset (seq 0 / N0 path)."""
        query_length = cache_position.shape[0]
        kv_length = self.get_seq_length(layer_idx) + query_length
        return kv_length, 0

    def early_initialization(
        self,
        batch_size: int,
        num_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        """Pre-learn dims (HF may call before the loop). Pools stay lazy in N0."""
        self._num_heads = num_heads
        self._head_dim = head_dim
        self.dtype = dtype
        self.device = torch.device(device)

    # --- introspection (not part of the HF contract) ------------------------

    @property
    def block_size(self) -> int:
        return self._block_size

    @property
    def block_table(self) -> dict[int, list[int]]:
        """Per-layer block allocation for the default seq (0): `[block_id, ...]`.

        Back-compat view for the N0/N1 single-stream tests; multi-seq callers read
        `block_tables(seq_id)` instead."""
        return self._seq_block_tables.get(self._default_seq_id, {})

    def block_tables(self, seq_id: int) -> dict[int, list[int]]:
        """Per-layer block allocation for `seq_id`: `block_table[layer_idx] = [block_id, ...]`."""
        return self._seq_block_tables.get(seq_id, {})

    # --- N3 kernel accessors (paged tree-attention) -------------------------

    def kernel_block_table(self, seq_ids, layer_idx: int, device=None) -> torch.Tensor:
        """`(num_seqs, max_blocks)` int32 block table for the kernel (PER-LAYER).

        Row s is `_seq_block_tables[seq_ids[s]][layer_idx]` in logical order,
        right-padded to the batch-max block count (pad 0 — the kernel never reads
        past `seq_lens_k[s]`). Block tables are per-layer, so this is built fresh
        per layer (unlike Unit-2's pure-compute builder, which packs one layer). The
        padded grid is filled on CPU and moved to `device` in ONE transfer (this is
        a per-layer per-step hot path; per-row device scatters dominated otherwise)."""
        dev = self.device if device is None else torch.device(device)
        rows = [self._seq_block_tables.get(s, {}).get(layer_idx, []) for s in seq_ids]
        max_blocks = max((len(r) for r in rows), default=0)
        table = torch.zeros((len(rows), max_blocks), dtype=torch.int32)   # CPU
        for s, r in enumerate(rows):
            if r:
                table[s, : len(r)] = torch.tensor(r, dtype=torch.int32)
        return table.to(dev)

    def kernel_seq_lens(self, seq_ids, layer_idx: int, device=None) -> torch.Tensor:
        """`(num_seqs,)` int32 total key length per seq = `get_seq_length` each."""
        dev = self.device if device is None else torch.device(device)
        return torch.tensor(
            [self.get_seq_length(layer_idx, seq_id=s) for s in seq_ids],
            dtype=torch.int32, device=dev,
        )

    def pool(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """`(k_pool, v_pool)` `(num_blocks, block_size, H, D)` for `layer_idx`."""
        return self._kpool[layer_idx], self._vpool[layer_idx]

    def refcount(self, block_id: int) -> int:
        """Current refcount of `block_id` (0 if free / untracked)."""
        return self._block_refcounts.get(block_id, 0)

    @property
    def num_free_blocks(self) -> int:
        return len(self._free_blocks)
