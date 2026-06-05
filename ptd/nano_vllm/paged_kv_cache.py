"""Block-paged KV cache (nano_vllm N0).

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
`ptd.engine.llm._select_kv_cache`. N1 builds the tree-spec loop on it; N2 adds
multi-sequence pooling (the no-op batch methods below raise until then).

KV layout: HF hands `update` keys/values shaped `(batch=1, num_heads, seq, head_dim)`
(seq on dim -2). Pool blocks store `(num_blocks, block_size, num_heads, head_dim)`,
so packing is a slice-assign on the block/offset axes and unpacking concatenates
the used slots back to `(1, num_heads, used, head_dim)`. Keys and values share one
block table; a parallel value pool mirrors the key pool slot-for-slot.
"""
from typing import Optional

import torch
from transformers.cache_utils import Cache


class PagedKVCache(Cache):
    """Block-paged KV store conforming to the HF `Cache` interface.

    Per layer we keep two pool tensors `(num_blocks, block_size, num_heads, head_dim)`
    (keys, values) and one block table (block ids in sequence order). The last
    block may be partially filled; `_filled[layer_idx]` tracks how many tokens it
    holds. The pool grows on demand in N0; N2 fixes its size and adds eviction.
    """

    def __init__(
        self,
        block_size: int = 16,
        max_batch_size: int = 1,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if max_batch_size != 1:
            raise NotImplementedError("PagedKVCache N0 is single-stream (max_batch_size=1)")
        # We never call Cache.__init__ (it wants a layer list / layer_class); our
        # storage is the pools below, and we override every method HF touches.
        # `max_batch_size` etc. are read-only properties on the base class derived
        # from `self.layers`, which we don't keep — store ours privately instead.
        self._block_size = int(block_size)
        self._max_batch_size = max_batch_size
        self.device = torch.device(device)
        self.dtype = dtype
        # Lazily sized on the first append (we learn num_heads/head_dim from the
        # KV the model hands us, mirroring DynamicLayer's lazy_initialization).
        self._num_heads: Optional[int] = None
        self._head_dim: Optional[int] = None
        self._kpool: dict[int, torch.Tensor] = {}       # layer_idx -> (num_blocks, block_size, H, D)
        self._vpool: dict[int, torch.Tensor] = {}
        self._block_table: dict[int, list[int]] = {}    # layer_idx -> [block_id, ...]
        self._filled: dict[int, int] = {}               # layer_idx -> tokens in the last block
        self._free_blocks: list[int] = []               # free pool indices (ascending)
        self._num_blocks = 0                            # pool capacity (grown on demand)

    # --- internal pool management -------------------------------------------

    def _grow_pool(self, extra_blocks: int) -> None:
        """Append `extra_blocks` empty blocks to every layer's pools, freeing them."""
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
        """Create this layer's pools/table on first touch, learning dims from KV."""
        if self._num_heads is None:
            self._num_heads = key_states.shape[1]
            self._head_dim = key_states.shape[-1]
        if layer_idx not in self._kpool:
            shape = (self._num_blocks, self._block_size, self._num_heads, self._head_dim)
            self._kpool[layer_idx] = torch.zeros(shape, dtype=self.dtype, device=self.device)
            self._vpool[layer_idx] = torch.zeros(shape, dtype=self.dtype, device=self.device)
            self._block_table[layer_idx] = []
            self._filled[layer_idx] = 0

    def allocate(self, num_blocks: int) -> torch.Tensor:
        """Pre-allocate `num_blocks` blocks from the pool, growing it if needed.

        Returns the allocated block ids `(num_blocks,)` (ascending). Raises if the
        pool cannot grow because dims aren't known yet (N0 grows freely once the
        first append has fixed num_heads/head_dim; N2 adds an eviction policy)."""
        if num_blocks <= 0:
            return torch.empty(0, dtype=torch.long, device=self.device)
        if len(self._free_blocks) < num_blocks:
            if self._num_heads is None:
                raise RuntimeError("cannot allocate before the first append (dims unknown)")
            self._grow_pool(num_blocks - len(self._free_blocks))
        ids = [self._free_blocks.pop(0) for _ in range(num_blocks)]
        return torch.tensor(ids, dtype=torch.long, device=self.device)

    def free(self, block_ids: torch.Tensor) -> None:
        """Return `block_ids` to the pool (kept sorted for deterministic reuse)."""
        self._free_blocks.extend(int(b) for b in block_ids.tolist())
        self._free_blocks = sorted(set(self._free_blocks))

    # --- append / gather -----------------------------------------------------

    def append(
        self,
        key_states: torch.Tensor,      # (batch=1, num_heads, seq_new, head_dim)
        value_states: torch.Tensor,    # (batch=1, num_heads, seq_new, head_dim)
        layer_idx: int,
        block_ids: Optional[torch.Tensor] = None,
    ) -> int:
        """Pack `seq_new` tokens' KV into this layer's blocks (in place).

        Fills the current last block first, then allocates blocks for the
        remainder (`block_ids` overrides the allocation if pre-reserved). Returns
        the new sequence length so HF can track `get_seq_length`. KV is the HF
        layout `(1, num_heads, seq, head_dim)` (seq on dim -2)."""
        self._ensure_layer(layer_idx, key_states)
        keys = key_states[0].transpose(0, 1)             # (seq_new, num_heads, head_dim)
        values = value_states[0].transpose(0, 1)
        seq_new = keys.shape[0]
        if seq_new == 0:
            return self.get_seq_length(layer_idx)
        table = self._block_table[layer_idx]
        filled = self._filled[layer_idx]
        start = self.get_seq_length(layer_idx)           # tokens before this append (pre-extend)

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
        else:
            new_blocks = [int(b) for b in self.allocate(new_block_count).tolist()]
        table.extend(new_blocks)

        # Write keys+values to the same slots: the last `seq_new` positions.
        kpool, vpool = self._kpool[layer_idx], self._vpool[layer_idx]
        for i in range(seq_new):
            pos = start + i
            blk = table[pos // self._block_size]
            off = pos % self._block_size
            kpool[blk, off] = keys[i]
            vpool[blk, off] = values[i]
        self._filled[layer_idx] = ((start + seq_new - 1) % self._block_size) + 1
        return self.get_seq_length(layer_idx)

    def gather(self, positions: torch.Tensor) -> None:
        """Compact a non-contiguous keep set into a linear prefix (all layers).

        `positions` is a 1-D LongTensor of cache positions to keep (the tree-verify
        accepted path: `[past, past+1, …, past+acc]`, possibly scattered in
        tree-order before this call). After gather the kept KV occupies positions
        `[0, 1, …, len(positions)-1]` in fresh blocks and the old blocks are freed —
        the paged analogue of `ptd.engine.llm._select_kv_cache`."""
        idx = positions.to(self.device).long()
        keep = idx.numel()
        new_block_count = (keep + self._block_size - 1) // self._block_size
        for layer_idx in list(self._kpool.keys()):
            table = self._block_table[layer_idx]
            old_blocks = list(table)
            src_blk = torch.tensor([table[int(p) // self._block_size] for p in idx], device=self.device)
            src_off = torch.tensor([int(p) % self._block_size for p in idx], device=self.device)
            gathered_k = self._kpool[layer_idx][src_blk, src_off]      # (keep, H, D)
            gathered_v = self._vpool[layer_idx][src_blk, src_off]
            new_blocks = [int(b) for b in self.allocate(new_block_count).tolist()]
            self._block_table[layer_idx] = list(new_blocks)
            for j, blk in enumerate(new_blocks):
                lo, hi = j * self._block_size, min((j + 1) * self._block_size, keep)
                self._kpool[layer_idx][blk, : hi - lo] = gathered_k[lo:hi]
                self._vpool[layer_idx][blk, : hi - lo] = gathered_v[lo:hi]
            self._filled[layer_idx] = keep - (new_block_count - 1) * self._block_size if new_block_count else 0
            stale = [b for b in old_blocks if b not in new_blocks]
            if stale:
                self.free(torch.tensor(stale, device=self.device))

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

        `cache_kwargs` (sliding-window metadata etc.) is ignored in N0."""
        self.append(key_states, value_states, layer_idx)
        return self._logical_kv(layer_idx)

    def _logical_kv(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Reconstruct the dense `(1, num_heads, seq, head_dim)` KV from the blocks."""
        table = self._block_table[layer_idx]
        if not table:                                    # empty layer
            empty = torch.zeros((1, self._num_heads, 0, self._head_dim),
                                dtype=self.dtype, device=self.device)
            return empty, empty.clone()
        kpool, vpool = self._kpool[layer_idx], self._vpool[layer_idx]
        kparts, vparts = [], []
        for i, blk in enumerate(table):
            used = self._block_size if i < len(table) - 1 else self._filled[layer_idx]
            kparts.append(kpool[blk, :used])             # (used, H, D)
            vparts.append(vpool[blk, :used])
        keys = torch.cat(kparts, dim=0)                  # (seq, H, D)
        values = torch.cat(vparts, dim=0)
        return (keys.transpose(0, 1).unsqueeze(0).contiguous(),    # (1, H, seq, D)
                values.transpose(0, 1).unsqueeze(0).contiguous())

    def get_seq_length(self, layer_idx: int = 0) -> int:
        """Current cached length for `layer_idx` (full blocks + the partial last)."""
        table = self._block_table.get(layer_idx)
        if not table:
            return 0
        return (len(table) - 1) * self._block_size + self._filled[layer_idx]

    def crop(self, max_length: int) -> None:
        """Trim every layer to its first `max_length` tokens (drops later blocks)."""
        for layer_idx in list(self._kpool.keys()):
            cur = self.get_seq_length(layer_idx)
            target = cur - abs(max_length) if max_length < 0 else max_length
            if cur <= target:
                continue
            keep_blocks = (target + self._block_size - 1) // self._block_size
            table = self._block_table[layer_idx]
            stale = table[keep_blocks:]
            self._block_table[layer_idx] = table[:keep_blocks]
            self._filled[layer_idx] = (target - (keep_blocks - 1) * self._block_size) if keep_blocks else 0
            if stale:
                self.free(torch.tensor(stale, device=self.device))

    def reset(self) -> None:
        """Clear all sequence state, returning every block to the pool."""
        self._free_blocks = list(range(self._num_blocks))
        for layer_idx in self._block_table:
            self._block_table[layer_idx] = []
            self._filled[layer_idx] = 0

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        """Beam reorder — N0 is greedy single-stream; raise to catch N1+ misuse."""
        raise NotImplementedError("PagedKVCache N0 is single-stream (no beam reorder)")

    def batch_repeat_interleave(self, repeats: int) -> None:
        raise NotImplementedError("PagedKVCache N0 is single-stream (N2 adds batching)")

    def batch_select_indices(self, indices: torch.Tensor) -> None:
        raise NotImplementedError("PagedKVCache N0 is single-stream (N2 adds batching)")

    def get_max_cache_shape(self, layer_idx: int = 0) -> int:
        """Max tokens the current pool can hold (it grows on demand in N0)."""
        return self._num_blocks * self._block_size

    def get_mask_sizes(self, cache_position: torch.Tensor, layer_idx: int) -> tuple:
        """(kv_length, kv_offset) for the 4D attention mask — mirrors DynamicLayer:
        past length + this step's query length, no offset."""
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
        """Per-layer block allocation: `block_table[layer_idx] = [block_id, ...]`."""
        return self._block_table
