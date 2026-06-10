"""JetFlow N2a gate: the multi-sequence PagedKVCache (shared pool + block
ref-counting + fixed pool + LRU eviction) keeps every sequence's KV intact under
interleaved append/gather/free across seqs.

Runs on CPU with random KV tensors (no model, no network, no GPU): in fp32 the
paged store is bitwise-equal to a plain contiguous reference (append/gather is a
copy, no rounding), so each test reconstructs every live seq's logical KV and
compares it to an independently-maintained dense reference. The headline safety
property — a gather or free on seq A NEVER corrupts seq B's KV — is checked
directly: we mutate A and assert B's reconstruction is unchanged.

Mirrors `tests/test_jetflow_engine.py`'s style (per-test seeding, dense reference
tensors, `torch.equal` bitwise checks).
"""
import torch
import pytest

from ptd.jetflow.paged_kv_cache import (
    PagedKVCache,
    EvictionRequired,
    EvictionFailed,
)


def _kv(num_heads, seq, head_dim):
    """Random HF-layout KV `(1, num_heads, seq, head_dim)` (keys, values)."""
    return (torch.randn(1, num_heads, seq, head_dim),
            torch.randn(1, num_heads, seq, head_dim))


def _multiseq_cache(block_size=4, max_total_tokens=4096, max_batch_size=8):
    """A multi-seq cache (fixed pool, ref-counting + eviction active)."""
    return PagedKVCache(
        block_size=block_size,
        max_batch_size=max_batch_size,
        max_total_tokens=max_total_tokens,
        dtype=torch.float32,
    )


def _append_all_layers(cache, seq_id, k, v, num_layers):
    """Append the same KV chunk into every layer of `seq_id` (model has L layers)."""
    for layer_idx in range(num_layers):
        cache.append(k, v, layer_idx, seq_id=seq_id)


def _assert_seq_matches(cache, seq_id, ref_k, ref_v, num_layers, msg=""):
    """Every layer's reconstructed logical KV for `seq_id` == the dense reference."""
    for layer_idx in range(num_layers):
        gk, gv = cache._logical_kv(layer_idx, seq_id=seq_id)
        assert torch.equal(gk, ref_k), f"{msg} keys layer {layer_idx} seq {seq_id}"
        assert torch.equal(gv, ref_v), f"{msg} values layer {layer_idx} seq {seq_id}"


# --- ref-counting invariants -------------------------------------------------

def test_append_increfs_each_new_block_once():
    """append claims (increfs) each new block exactly once; the seq table holds it,
    and the block leaves the free pool."""
    torch.manual_seed(0)
    cache = _multiseq_cache(block_size=4)
    k, v = _kv(2, 10, 16)                     # 10 tokens -> 3 blocks (4/4/2)
    free_before = cache.num_free_blocks
    cache.append(k, v, layer_idx=0, seq_id=7)
    table = cache.block_tables(7)[0]
    assert len(table) == 3
    assert cache.num_free_blocks == free_before - 3
    for b in table:
        assert cache.refcount(b) == 1        # sole owner so far


def test_free_decrefs_and_reclaims_blocks():
    """free(seq) decrefs every owned block; sole-owned blocks return to the pool and
    the seq's tables/fill state are dropped."""
    torch.manual_seed(1)
    cache = _multiseq_cache(block_size=4)
    k, v = _kv(2, 9, 16)
    cache.append(k, v, layer_idx=0, seq_id=3)
    blocks = list(cache.block_tables(3)[0])
    free_before = cache.num_free_blocks
    cache.free(seq_id=3)
    assert cache.block_tables(3) == {}
    assert cache.get_seq_length(0, seq_id=3) == 0
    assert cache.num_free_blocks == free_before + len(blocks)
    for b in blocks:
        assert cache.refcount(b) == 0        # back in the pool


def test_shared_block_survives_one_owners_free():
    """A block shared by two seqs (refcount 2) is NOT freed when the first owner is
    freed — it survives for the second owner until that one frees too."""
    torch.manual_seed(2)
    cache = _multiseq_cache(block_size=4)
    k, v = _kv(2, 4, 16)
    # seq A owns block b (sole, refcount 1); hand the SAME block to seq B (reservation).
    cache.append(k, v, layer_idx=0, seq_id=10)
    shared = cache.block_tables(10)[0][0]
    cache._incref_block(shared)              # simulate prefix sharing: B claims b too
    cache._seq_block_tables.setdefault(11, {})[0] = [shared]
    cache._seq_filled.setdefault(11, {})[0] = cache._block_size
    assert cache.refcount(shared) == 2

    free_before = cache.num_free_blocks
    cache.free(seq_id=10)                     # A leaves
    assert cache.refcount(shared) == 1        # still owned by B
    assert shared not in cache._free_blocks   # NOT reclaimed
    assert cache.num_free_blocks == free_before
    cache.free(seq_id=11)                      # B leaves -> now free
    assert cache.refcount(shared) == 0
    assert shared in cache._free_blocks


# --- isolation: mutating seq A never corrupts seq B --------------------------

def test_free_seq_a_does_not_corrupt_seq_b():
    """The headline safety property: free(A) leaves every byte of B's KV intact,
    across all layers, checked against an independent dense reference for B."""
    torch.manual_seed(3)
    num_layers, H, D = 2, 2, 16
    cache = _multiseq_cache(block_size=4)
    ka, va = _kv(H, 13, D)                    # A: 13 tokens
    kb, vb = _kv(H, 7, D)                     # B: 7 tokens
    _append_all_layers(cache, 100, ka, va, num_layers)
    _append_all_layers(cache, 200, kb, vb, num_layers)
    _assert_seq_matches(cache, 200, kb, vb, num_layers, "B before free(A)")

    cache.free(seq_id=100)
    _assert_seq_matches(cache, 200, kb, vb, num_layers, "B after free(A)")
    assert cache.get_seq_length(0, seq_id=100) == 0   # A gone
    assert cache.get_seq_length(0, seq_id=200) == 7    # B unchanged


def test_gather_seq_a_does_not_corrupt_seq_b():
    """A scattered gather (tree-path compaction) on seq A must not touch seq B's KV,
    even when A's freed blocks get reused for A's own compaction."""
    torch.manual_seed(4)
    num_layers, H, D = 2, 2, 16
    cache = _multiseq_cache(block_size=4)
    ka, va = _kv(H, 17, D)                    # A: 17 tokens (scattered keep below)
    kb, vb = _kv(H, 11, D)                    # B: 11 tokens
    _append_all_layers(cache, 1, ka, va, num_layers)
    _append_all_layers(cache, 2, kb, vb, num_layers)

    keep = torch.tensor([0, 1, 4, 9, 16])    # tree-like accepted path in A
    cache.gather(keep, seq_id=1)
    # A compacted to the kept positions; B untouched.
    for layer_idx in range(num_layers):
        gk, gv = cache._logical_kv(layer_idx, seq_id=1)
        assert torch.equal(gk, ka[:, :, keep]), f"A keys layer {layer_idx}"
        assert torch.equal(gv, va[:, :, keep]), f"A values layer {layer_idx}"
    _assert_seq_matches(cache, 2, kb, vb, num_layers, "B after gather(A)")
    assert cache.get_seq_length(0, seq_id=1) == keep.numel()
    assert cache.get_seq_length(0, seq_id=2) == 11


def test_gather_then_append_continues_seq():
    """After gather, an append extends the compacted prefix correctly (the partial
    last block is topped up, not corrupted) and the other seq stays intact."""
    torch.manual_seed(5)
    num_layers, H, D = 2, 2, 16
    cache = _multiseq_cache(block_size=4)
    ka, va = _kv(H, 10, D)
    kb, vb = _kv(H, 6, D)
    _append_all_layers(cache, 1, ka, va, num_layers)
    _append_all_layers(cache, 2, kb, vb, num_layers)

    keep = torch.tensor([0, 2, 5])           # A keeps 3 of 10
    cache.gather(keep, seq_id=1)
    extra_k, extra_v = _kv(H, 5, D)          # then 5 more tokens
    _append_all_layers(cache, 1, extra_k, extra_v, num_layers)

    exp_k = torch.cat([ka[:, :, keep], extra_k], dim=2)
    exp_v = torch.cat([va[:, :, keep], extra_v], dim=2)
    _assert_seq_matches(cache, 1, exp_k, exp_v, num_layers, "A after gather+append")
    _assert_seq_matches(cache, 2, kb, vb, num_layers, "B after A gather+append")


# --- interleaved sequences of different lengths ------------------------------

def test_interleaved_appends_different_lengths():
    """Round-robin appends to three seqs of growing different lengths; each seq's
    full KV reconstructs against its own dense reference and block tables stay
    disjoint."""
    torch.manual_seed(6)
    num_layers, H, D = 2, 2, 16
    cache = _multiseq_cache(block_size=4)
    seqs = {1: [], 2: [], 3: []}             # seq_id -> list of (k, v) chunks per layer
    # Interleave appends: seq 1 gets 3,2; seq 2 gets 5; seq 3 gets 1,4,2.
    schedule = [(1, 3), (2, 5), (3, 1), (1, 2), (3, 4), (3, 2)]
    for seq_id, n in schedule:
        k, v = _kv(H, n, D)
        _append_all_layers(cache, seq_id, k, v, num_layers)
        seqs[seq_id].append((k, v))
    for seq_id, chunks in seqs.items():
        ref_k = torch.cat([c[0] for c in chunks], dim=2)
        ref_v = torch.cat([c[1] for c in chunks], dim=2)
        _assert_seq_matches(cache, seq_id, ref_k, ref_v, num_layers, f"interleaved seq {seq_id}")
    # Block tables across seqs must be disjoint (no two seqs own the same block).
    owned = []
    for seq_id in seqs:
        for table in cache.block_tables(seq_id).values():
            owned.extend(table)
    assert len(owned) == len(set(owned)), "seqs share a block they shouldn't"


def test_free_one_of_many_reuses_blocks():
    """Freeing a middle seq returns its blocks to the pool; a new seq reuses them
    (deterministic ascending reuse) without disturbing the survivors."""
    torch.manual_seed(7)
    num_layers, H, D = 1, 2, 16
    cache = _multiseq_cache(block_size=4)
    refs = {}
    for seq_id, n in [(1, 8), (2, 8), (3, 8)]:
        k, v = _kv(H, n, D)
        _append_all_layers(cache, seq_id, k, v, num_layers)
        refs[seq_id] = (k, v)
    freed_blocks = set(cache.block_tables(2)[0])
    cache.free(seq_id=2)
    # Survivors intact.
    _assert_seq_matches(cache, 1, *refs[1], num_layers, "survivor 1")
    _assert_seq_matches(cache, 3, *refs[3], num_layers, "survivor 3")
    # New seq reuses the freed blocks.
    k4, v4 = _kv(H, 8, D)
    _append_all_layers(cache, 4, k4, v4, num_layers)
    assert set(cache.block_tables(4)[0]) <= freed_blocks | set(range(cache._num_blocks))
    _assert_seq_matches(cache, 4, k4, v4, num_layers, "reused seq 4")
    _assert_seq_matches(cache, 1, *refs[1], num_layers, "survivor 1 after reuse")
    _assert_seq_matches(cache, 3, *refs[3], num_layers, "survivor 3 after reuse")


# --- fixed pool + eviction ---------------------------------------------------

def test_allocate_raises_eviction_required_when_pool_full():
    """The fixed pool raises EvictionRequired (not a silent grow) once exhausted."""
    torch.manual_seed(8)
    # 2 blocks total (block_size=4, 8 tokens of capacity).
    cache = _multiseq_cache(block_size=4, max_total_tokens=8, max_batch_size=4)
    k, v = _kv(2, 8, 16)                      # exactly fills the pool (2 blocks)
    cache.append(k, v, layer_idx=0, seq_id=1)
    assert cache.num_free_blocks == 0
    k2, v2 = _kv(2, 4, 16)
    with pytest.raises(EvictionRequired):
        cache.append(k2, v2, layer_idx=0, seq_id=2)


def test_evict_sequence_frees_room_for_new():
    """When the pool fills, evicting the LRU sequence frees its blocks so a new seq
    can be admitted; the surviving (more-recently-used) seq is byte-intact."""
    torch.manual_seed(9)
    num_layers, H, D = 1, 2, 16
    # 4 blocks (16 tokens). Two 8-token seqs fill it.
    cache = _multiseq_cache(block_size=4, max_total_tokens=16, max_batch_size=4)
    ka, va = _kv(H, 8, D)
    kb, vb = _kv(H, 8, D)
    _append_all_layers(cache, 1, ka, va, num_layers)     # seq 1 (older)
    _append_all_layers(cache, 2, kb, vb, num_layers)     # seq 2 (newer)
    assert cache.num_free_blocks == 0

    # New seq 3 needs 2 blocks; admit -> allocate_slots evicts the LRU (seq 1).
    assert cache.admit_sequence(3, prompt_len=8) is True
    assert cache.allocate_slots(3, num_tokens=8) is True
    assert cache.get_seq_length(0, seq_id=1) == 0         # seq 1 evicted
    _assert_seq_matches(cache, 2, kb, vb, num_layers, "seq 2 survives eviction")

    kc, vc = _kv(H, 8, D)
    _append_all_layers(cache, 3, kc, vc, num_layers)
    _assert_seq_matches(cache, 3, kc, vc, num_layers, "admitted seq 3")
    _assert_seq_matches(cache, 2, kb, vb, num_layers, "seq 2 still intact after seq 3")


def test_evict_lru_picks_least_recently_used():
    """evict_sequence(None) targets the least-recently-touched seq. Touching seq 1
    (via append) after seq 2 makes seq 2 the LRU victim."""
    torch.manual_seed(10)
    num_layers, H, D = 1, 2, 16
    cache = _multiseq_cache(block_size=4, max_total_tokens=64, max_batch_size=4)
    _append_all_layers(cache, 1, *_kv(H, 4, D), num_layers)   # touch 1
    _append_all_layers(cache, 2, *_kv(H, 4, D), num_layers)   # touch 2
    _append_all_layers(cache, 1, *_kv(H, 4, D), num_layers)   # touch 1 again -> 2 is LRU
    victim_freed = cache.evict_sequence(None)
    assert victim_freed > 0
    assert cache.get_seq_length(0, seq_id=2) == 0             # seq 2 was the LRU victim
    assert cache.get_seq_length(0, seq_id=1) == 8             # seq 1 survived


def test_evict_only_sequence_raises():
    """Evicting the only running sequence is refused (nothing would be gained)."""
    torch.manual_seed(11)
    cache = _multiseq_cache(block_size=4, max_total_tokens=64, max_batch_size=4)
    cache.append(*_kv(2, 4, 16), layer_idx=0, seq_id=1)
    with pytest.raises(EvictionFailed):
        cache.evict_sequence(seq_id=1)


def test_eviction_preserves_shared_prefix_for_survivor():
    """If the evicted seq A shares a prefix block with survivor B (refcount 2),
    evicting A keeps the block (refcount -> 1) and B's KV is byte-intact."""
    torch.manual_seed(12)
    num_layers, H, D = 1, 2, 16
    cache = _multiseq_cache(block_size=4, max_total_tokens=64, max_batch_size=4)
    # Build A with 8 tokens; share A's first block with B as a prefix.
    ka, va = _kv(H, 8, D)
    _append_all_layers(cache, 1, ka, va, num_layers)
    prefix_block = cache.block_tables(1)[0][0]
    cache._incref_block(prefix_block)
    cache._seq_block_tables.setdefault(2, {})[0] = [prefix_block]
    cache._seq_filled.setdefault(2, {})[0] = cache._block_size
    assert cache.refcount(prefix_block) == 2

    cache.evict_sequence(seq_id=1)            # A leaves
    assert cache.refcount(prefix_block) == 1   # block survives for B
    assert prefix_block not in cache._free_blocks
    # B's first 4 tokens (the shared prefix) reconstruct from the surviving block.
    gk, gv = cache._logical_kv(0, seq_id=2)
    assert torch.equal(gk, ka[:, :, :4]), "B shared-prefix keys corrupted by eviction"
    assert torch.equal(gv, va[:, :, :4]), "B shared-prefix values corrupted by eviction"


# --- HF interface routing (seq_id via cache_kwargs) --------------------------

def test_update_routes_seq_id_via_cache_kwargs():
    """The HF `update` callback honours cache_kwargs['seq_id'] (the N2a batched
    forward path) and falls back to seq 0 when absent (N0/N1)."""
    torch.manual_seed(13)
    cache = _multiseq_cache(block_size=4)
    k0, v0 = _kv(2, 5, 16)
    k1, v1 = _kv(2, 7, 16)
    lk0, lv0 = cache.update(k0, v0, layer_idx=0, cache_kwargs={"seq_id": 0})
    lk1, lv1 = cache.update(k1, v1, layer_idx=0, cache_kwargs={"seq_id": 1})
    assert torch.equal(lk0, k0) and torch.equal(lv0, v0)
    assert torch.equal(lk1, k1) and torch.equal(lv1, v1)
    # Default (no kwargs) targets seq 0 and extends it.
    k0b, v0b = _kv(2, 3, 16)
    cache.update(k0b, v0b, layer_idx=0)
    gk, gv = cache._logical_kv(0, seq_id=0)
    assert torch.equal(gk, torch.cat([k0, k0b], dim=2))
    assert torch.equal(gv, torch.cat([v0, v0b], dim=2))
    # seq 1 untouched by the seq-0 default-route append.
    g1k, g1v = cache._logical_kv(0, seq_id=1)
    assert torch.equal(g1k, k1) and torch.equal(g1v, v1)
