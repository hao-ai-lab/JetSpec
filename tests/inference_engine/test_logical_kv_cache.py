import pytest
import torch

from jetspec.inference_engine.paged_kv_cache import PagedKVCache


def _seed_cache(num_layers=3, prompt_len=5, block_size=4):
    cache = PagedKVCache(block_size=block_size, dtype=torch.float32)
    for layer_idx in range(num_layers):
        keys = torch.full((1, 2, prompt_len, 8), float(layer_idx + 1))
        values = -keys
        cache.update(keys, values, layer_idx=layer_idx)
    return cache


def test_reserve_capacity_decouples_pool_blocks_from_block_table_width():
    cache = _seed_cache(num_layers=3, prompt_len=5, block_size=4)
    cache.reserve_capacity(total_tokens=37, block_table_tokens=9)

    assert cache._num_blocks == 3 * (cache._blocks_for(37) + 1)
    assert cache.reserved_block_table_width == cache._blocks_for(9) + 1

    cache.reserve_capacity(total_tokens=12, block_table_tokens=5)
    assert cache._num_blocks == 3 * (cache._blocks_for(37) + 1)
    assert cache.reserved_block_table_width == cache._blocks_for(9) + 1


def test_freeze_pool_turns_late_growth_into_a_hard_error():
    cache = _seed_cache(num_layers=1, prompt_len=4, block_size=4)
    cache.reserve_capacity(total_tokens=8)
    cache.freeze_pool()

    before = cache._num_blocks
    with pytest.raises(RuntimeError, match="frozen"):
        cache.allocate(cache.num_free_blocks + 1)
    assert cache._num_blocks == before


def test_reserve_logical_slots_is_block_aligned_and_invisible_to_seq_tables():
    cache = _seed_cache(num_layers=3, prompt_len=5, block_size=4)
    before_tables = {layer: list(table) for layer, table in cache.block_table.items()}
    before_filled = {
        layer: cache._seq_filled[cache._default_seq_id][layer]
        for layer in before_tables
    }
    before_refcounts = dict(cache._block_refcounts)

    node_blks, round_blocks = cache.reserve_logical_slots(6)

    # LAYER-SHARED contract: one (n_nodes,) row + one flat id list serve every
    # layer (each layer writes the same ids in its own pool tensor).
    assert node_blks.shape == (6,)
    assert node_blks.dtype == torch.int64
    assert len(round_blocks) == 2
    expected = torch.tensor(round_blocks, dtype=torch.int64).repeat_interleave(4)[:6]
    assert torch.equal(node_blks.cpu(), expected)
    assert torch.equal(node_blks[0:4], node_blks[0].expand(4))
    assert node_blks[4] == node_blks[5]

    assert cache.block_table == before_tables
    assert cache._seq_filled[cache._default_seq_id] == before_filled
    assert cache._block_refcounts == before_refcounts


def test_release_round_blocks_reuses_freed_logical_blocks():
    cache = _seed_cache(num_layers=2, prompt_len=4, block_size=4)
    _, round_blocks = cache.reserve_logical_slots(5)
    released = sorted(round_blocks)

    cache.release_round_blocks(round_blocks, freed_idx=[0, 1])
    reused = sorted(int(block) for block in cache.allocate(len(released)).tolist())

    assert reused == released
    assert all(cache.refcount(block) == 0 for block in released)


def test_prefix_block_tables_pads_without_mutating_tables():
    cache = _seed_cache(num_layers=2, prompt_len=5, block_size=4)
    before_tables = {layer: list(table) for layer, table in cache.block_table.items()}

    prefixed = cache.prefix_block_tables(width=5)

    assert len(prefixed) == 2
    for layer_idx, row in enumerate(prefixed):
        assert row.shape == (1, 5)
        assert row.dtype == torch.int32
        table = before_tables[layer_idx]
        assert row[0, : len(table)].tolist() == table
        assert row[0, len(table):].tolist() == [0] * (5 - len(table))
    assert cache.block_table == before_tables

    with pytest.raises(ValueError, match="requested fixed width"):
        cache.prefix_block_tables(width=1)


def test_reserved_capacity_covers_worst_case_logical_decode_without_growth():
    block_size = 4
    prompt_len = 5
    max_new_tokens = 6
    bmax = 8
    extra_shared = max_new_tokens + block_size + (bmax + block_size - 1) // block_size + 1
    cache = _seed_cache(num_layers=2, prompt_len=prompt_len, block_size=block_size)
    cache.reserve_capacity(total_tokens=prompt_len,
                           block_table_tokens=prompt_len + max_new_tokens + bmax,
                           extra_shared_blocks=extra_shared)
    cache.freeze_pool()

    # Worst-case retained-fragment pattern: each round keeps one SHARED block and
    # releases the rest. A frozen pool must have been sized so no late growth occurs.
    for _ in range(max_new_tokens + block_size):
        _, round_blocks = cache.reserve_logical_slots(bmax)
        cache.release_round_blocks(round_blocks, freed_idx=[1])

    assert cache._num_blocks == 2 * (cache._blocks_for(prompt_len) + 1) + extra_shared


def test_slot_commit_overlap_uses_copy_before_write_semantics():
    slots = torch.arange(2 * 8, dtype=torch.int64).reshape(2, 8)
    wlen = 2
    path_t = torch.tensor([4, 2, 5])

    kept = slots[:, wlen + path_t]
    slots[:, wlen:wlen + path_t.numel()] = kept

    assert torch.equal(slots[:, 2:5], torch.tensor([[6, 4, 7], [14, 12, 15]]))
