"""JetSpec N3 (Unit 2) gate: the paged tree-attention metadata builder packs the
kernel's input tensors so that (a) `block_table` indexes the same logical KV the
cache reconstructs, (b) `qq_bias` encodes the same allowed-set as the engine's 4D
masks, and (c) the cu/seq-len arithmetic is internally consistent across decode /
tree / mixed batches.

Pure CPU (no GPU, no model, no triton): builds a tiny `PagedKVCache` with random
KV and small `DraftTree`s constructed directly, and cross-checks against
`PagedKVCache._logical_kv` and `jetspec.tree.build_ancestor_matrix` — the same sources
of truth the kernel and engine use.
"""
import torch

from jetspec.inference_engine.attn_metadata import build_attn_metadata
from jetspec.inference_engine.paged_kv_cache import PagedKVCache
from jetspec.tree import DraftTree, build_ancestor_matrix


def _tree(parents):
    """A minimal DraftTree from a parent list (depths derived; tokens arbitrary)."""
    n = len(parents)
    depth = [0] * n
    for i in range(1, n):
        depth[i] = depth[parents[i]] + 1
    return DraftTree(
        token_ids=torch.arange(n, dtype=torch.long),
        parent_indices=torch.tensor(parents, dtype=torch.long),
        depth=torch.tensor(depth, dtype=torch.long),
        num_nodes=n,
    )


def _append_seq(cache, seq_id, length, *, num_layers, heads, dim, seed):
    """Append `length` tokens of known random KV for `seq_id` across all layers.

    Returns the per-layer (length, H, D) keys so a test can reconstruct the
    expected logical KV independently of the cache."""
    torch.manual_seed(seed)
    keys_by_layer = {}
    for layer in range(num_layers):
        k = torch.randn(1, heads, length, dim)
        v = torch.randn(1, heads, length, dim)
        cache.append(k, v, layer, seq_id=seq_id)
        keys_by_layer[layer] = k
    return keys_by_layer


def test_block_table_indexes_same_logical_kv():
    """block_table + the kernel's slot math reproduce `_logical_kv` exactly, for
    seqs spanning partial last blocks and multiple blocks."""
    bs, num_layers, heads, dim = 4, 2, 2, 3
    cache = PagedKVCache(block_size=bs, max_batch_size=1, dtype=torch.float32)
    # lengths chosen to exercise: 1 partial block, exactly-full block, multi-block
    # with a partial tail.
    lengths = {0: 3, 1: 4, 2: 9}
    for i, (sid, ln) in enumerate(lengths.items()):
        _append_seq(cache, sid, ln, num_layers=num_layers, heads=heads, dim=dim, seed=i)

    layer = 0
    seq_ids = list(lengths)
    block_tables = [cache._seq_block_tables[s][layer] for s in seq_ids]
    past_lens = [cache.get_seq_length(layer, seq_id=s) for s in seq_ids]
    meta = build_attn_metadata(seq_ids, block_tables, past_lens, [None] * len(seq_ids))

    kpool = cache._kpool[layer]
    for s, sid in enumerate(seq_ids):
        # Walk the cached prefix [0, past_len) — the positions _logical_kv
        # reconstructs — via the kernel's slot math on block_table[s].
        plen = past_lens[s]
        assert int(meta.seq_lens_k[s].item()) == plen + 1, "decode seq_lens_k = past+1"
        gathered = torch.stack([
            kpool[int(meta.block_table[s, pos // bs].item()), pos % bs]
            for pos in range(plen)
        ])                                                        # (plen, H, D)
        logical_k, _ = cache._logical_kv(layer, seq_id=sid)       # (1, H, plen, D)
        expected = logical_k[0].transpose(0, 1)                   # (plen, H, D)
        assert torch.equal(gathered, expected), f"seq {sid} block_table mismatch"


def test_qq_bias_matches_ancestor_mask():
    """Each seq's qq_bias block (0/-inf) encodes exactly build_ancestor_matrix; the
    cross-seq blocks are all -inf; the decode-only batch returns qq_bias=None."""
    trees = [
        _tree([-1, 0, 0, 1]),       # 4 nodes: 0->{1,2}, 1->3
        _tree([-1, 0, 1, 1, 0]),    # 5 nodes: 0->{1,4}, 1->{2,3}
    ]
    seq_ids = [0, 1]
    block_tables = [[0], [1]]
    past_lens = [5, 7]
    meta = build_attn_metadata(seq_ids, block_tables, past_lens, trees)
    assert meta.qq_bias is not None

    cu = meta.cu_seqlens_q
    for s, tree in enumerate(trees):
        lo, hi = int(cu[s].item()), int(cu[s + 1].item())
        block = meta.qq_bias[lo:hi, lo:hi]
        anc = build_ancestor_matrix(tree).bool()
        assert torch.equal(block == 0, anc), f"seq {s} qq_bias != ancestor mask"
        assert torch.isneginf(block[~anc]).all(), "disallowed entries must be -inf"

    # cross-seq blocks fully -inf (no attending across sequences).
    a0, a1 = int(cu[1].item()), int(cu[2].item())
    assert torch.isneginf(meta.qq_bias[:a0, a0:a1]).all()
    assert torch.isneginf(meta.qq_bias[a0:a1, :a0]).all()

    # decode-only batch -> no bias.
    decode = build_attn_metadata([0, 1], [[0], [1]], [4, 6], [None, None])
    assert decode.qq_bias is None


def test_qq_bias_matches_engine_4d_mask_subblock():
    """The qq_bias block reproduces the tree-node sub-block of the engine's 4D
    additive mask: allowed[:, past:] = build_ancestor_matrix(tree)."""
    tree = _tree([-1, 0, 0, 1, 2])
    past = 6
    N = tree.num_nodes
    # engine's allowed-set over [prefix | tree] cols (engine.py ~L177-179).
    allowed = torch.zeros(N, past + N, dtype=torch.bool)
    allowed[:, :past] = True
    allowed[:, past:] = build_ancestor_matrix(tree).bool()
    engine_tree_block = allowed[:, past:]                         # (N, N)

    meta = build_attn_metadata([0], [[0]], [past], [tree])
    assert torch.equal(meta.qq_bias == 0, engine_tree_block)


def test_cu_and_seqlens_arithmetic_mixed_batch():
    """cu is non-decreasing with cu[-1]==total_q, and seq_lens_k[s]==past+Nq for a
    mixed batch (decode + tree seqs)."""
    trees = [None, _tree([-1, 0, 0]), None, _tree([-1, 0, 1, 1])]
    past_lens = [10, 5, 12, 3]
    seq_ids = [0, 1, 2, 3]
    block_tables = [[0], [1], [2], [3]]
    meta = build_attn_metadata(seq_ids, block_tables, past_lens, trees)

    nq = [1, 3, 1, 4]
    cu = meta.cu_seqlens_q
    assert int(cu[0].item()) == 0
    assert int(cu[-1].item()) == sum(nq)
    assert (cu[1:] >= cu[:-1]).all(), "cu_seqlens_q must be non-decreasing"
    assert cu.dtype == torch.int32

    for s in range(len(seq_ids)):
        assert int(meta.seq_lens_k[s].item()) == past_lens[s] + nq[s]
    assert meta.seq_lens_k.dtype == torch.int32

    # qq_bias spans total_q and the decode rows are self-only (0 on the diagonal).
    assert meta.qq_bias is not None
    assert meta.qq_bias.shape == (sum(nq), sum(nq))
    for s, n in enumerate(nq):
        lo = int(cu[s].item())
        if trees[s] is None:
            assert meta.qq_bias[lo, lo].item() == 0.0


def test_block_table_padding_and_dtypes():
    """block_table right-pads to the batch-max block count with dtype int32, and
    num_query_nodes overrides the decode query count."""
    block_tables = [[7], [3, 4, 5], []]
    meta = build_attn_metadata(
        [0, 1, 2], block_tables, [2, 9, 0], [None, None, None],
        num_query_nodes=[1, 1, 2],
    )
    assert meta.block_table.shape == (3, 3)
    assert meta.block_table.dtype == torch.int32
    assert meta.block_table[0].tolist() == [7, 0, 0]
    assert meta.block_table[1].tolist() == [3, 4, 5]
    # override applied: seq 2 contributes 2 query rows -> total 1+1+2 = 4.
    assert int(meta.cu_seqlens_q[-1].item()) == 4
    assert meta.seq_lens_k[2].item() == 2          # past 0 + 2 override nodes
