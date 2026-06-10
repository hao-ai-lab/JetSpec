"""Standalone correctness gate for the paged tree-attention kernel (JetFlow N3).

The kernel reads the exact post-RoPE K/V bytes SDPA reads, so correctness reduces
to "kernel output == SDPA output on the same paged pool". We build a RANDOM block
pool + per-seq block tables (including a partial last block and a multi-block seq),
gather each seq's dense K/V for the SDPA reference, and compare:

- decode (Nq_s == 1, qq_bias=None): full visibility over prefix + self.
- tree (Nq_s in 2..15): the additive ancestor bias from a real ``DraftTree`` over
  the node columns, 0 over the prefix columns.

fp32 is strict (atol/rtol 1e-4); bf16 is a looser band (the online-softmax
reduction order differs from SDPA — mirrors the documented N0/N1 borderline).

Needs CUDA (triton); skipped on a CPU-only host.
"""
import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="triton kernel needs CUDA"
)

if torch.cuda.is_available():
    from ptd.jetflow.paged_tree_attn import paged_tree_attn
from ptd.tree import build_ancestor_matrix
from ptd.tree._core.base import DraftTree

DEVICE = "cuda"


def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """(H, S, D) kv-head tensor -> (H*n_rep, S, D), GQA broadcast (no copy semantics)."""
    H, S, D = x.shape
    return x[:, None, :, :].expand(H, n_rep, S, D).reshape(H * n_rep, S, D)


def _random_pool(num_blocks, block_size, Hkv, D, dtype):
    k = torch.randn(num_blocks, block_size, Hkv, D, device=DEVICE, dtype=dtype)
    v = torch.randn(num_blocks, block_size, Hkv, D, device=DEVICE, dtype=dtype)
    return k, v


def _alloc_blocks(seq_lens, block_size, num_blocks):
    """Random, non-overlapping physical block ids per seq. Returns (block_table,
    max_blocks) — unused block-table slots are filled with a valid in-range id
    (the kernel never reads past seq_len, so the fill is harmless)."""
    perm = torch.randperm(num_blocks).tolist()
    cur = 0
    tables = []
    max_blocks = max((sl + block_size - 1) // block_size for sl in seq_lens)
    for sl in seq_lens:
        nb = (sl + block_size - 1) // block_size
        ids = perm[cur:cur + nb]
        cur += nb
        ids = ids + [ids[-1]] * (max_blocks - nb)  # pad with a valid id
        tables.append(ids)
    return torch.tensor(tables, dtype=torch.int32, device=DEVICE), max_blocks


def _gather_kv(k_pool, v_pool, table_row, seq_len, block_size):
    """Dense (Hkv, seq_len, D) K/V for one seq from the pool via its block table."""
    Hkv, D = k_pool.shape[2], k_pool.shape[3]
    kk = torch.empty(seq_len, Hkv, D, device=DEVICE, dtype=k_pool.dtype)
    vv = torch.empty(seq_len, Hkv, D, device=DEVICE, dtype=v_pool.dtype)
    for pos in range(seq_len):
        blk = int(table_row[pos // block_size])
        off = pos % block_size
        kk[pos] = k_pool[blk, off]
        vv[pos] = v_pool[blk, off]
    return kk.transpose(0, 1).contiguous(), vv.transpose(0, 1).contiguous()


def _sdpa_ref(q_s, K, V, n_rep, scale, attn_mask):
    """SDPA oracle for one seq. q_s (Nq, Hq, D); K/V (Hkv, S, D)."""
    Kr = _repeat_kv(K, n_rep)                  # (Hq, S, D)
    Vr = _repeat_kv(V, n_rep)
    q = q_s.transpose(0, 1).unsqueeze(0)        # (1, Hq, Nq, D)
    k = Kr.unsqueeze(0)                         # (1, Hq, S, D)
    v = Vr.unsqueeze(0)
    # SDPA requires the additive bias to match the query dtype (bf16/fp32).
    mask = attn_mask.to(q.dtype).unsqueeze(0).unsqueeze(0) if attn_mask is not None else None
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=scale)
    return out.squeeze(0).transpose(0, 1)       # (Nq, Hq, D)


def _build_tree(num_nodes, seed):
    """A small valid DraftTree in parent-before-child order (parents[i] < i)."""
    g = torch.Generator().manual_seed(seed)
    parents = [-1]
    for i in range(1, num_nodes):
        parents.append(int(torch.randint(0, i, (1,), generator=g)))
    parent_t = torch.tensor(parents, dtype=torch.long)
    depth = torch.zeros(num_nodes, dtype=torch.long)
    for i in range(1, num_nodes):
        depth[i] = depth[parents[i]] + 1
    return DraftTree(
        token_ids=torch.zeros(num_nodes, dtype=torch.long),
        parent_indices=parent_t,
        depth=depth,
        num_nodes=num_nodes,
    )


def _run_decode(num_seqs, Hq, Hkv, D, block_size, dtype, atol, rtol, seed=0):
    torch.manual_seed(seed)
    n_rep = Hq // Hkv
    scale = D ** -0.5
    # variable prefixes: include a partial last block and a multi-block seq.
    base = [block_size + 3, 2 * block_size, 3 * block_size + 5, 7, block_size - 1,
            5 * block_size + 1, block_size, 2 * block_size + 9]
    seq_lens = base[:num_seqs]
    num_blocks = sum((sl + block_size - 1) // block_size for sl in seq_lens) + 4
    k_pool, v_pool = _random_pool(num_blocks, block_size, Hkv, D, dtype)
    block_table, _ = _alloc_blocks(seq_lens, block_size, num_blocks)

    total_q = num_seqs                          # Nq_s == 1 each
    q = torch.randn(total_q, Hq, D, device=DEVICE, dtype=dtype)
    cu = torch.tensor([0] + list(range(1, num_seqs + 1)), dtype=torch.int32, device=DEVICE)
    seq_lens_k = torch.tensor(seq_lens, dtype=torch.int32, device=DEVICE)

    out = paged_tree_attn(q, k_pool, v_pool, block_table, cu, seq_lens_k,
                          None, scale, n_rep, block_size)

    refs = []
    for s in range(num_seqs):
        K, V = _gather_kv(k_pool, v_pool, block_table[s], seq_lens[s], block_size)
        refs.append(_sdpa_ref(q[s:s + 1], K, V, n_rep, scale, attn_mask=None))
    ref = torch.cat(refs, dim=0)
    assert out.shape == ref.shape
    maxdiff = (out.float() - ref.float()).abs().max().item()
    assert torch.allclose(out.float(), ref.float(), atol=atol, rtol=rtol), (
        f"decode mismatch dtype={dtype} num_seqs={num_seqs} Hq={Hq} Hkv={Hkv} "
        f"bs={block_size} max|diff|={maxdiff:.3e}"
    )
    return maxdiff


def _run_tree(num_seqs, Hq, Hkv, D, block_size, dtype, atol, rtol, seed=0):
    torch.manual_seed(seed)
    n_rep = Hq // Hkv
    scale = D ** -0.5
    node_counts = [2, 9, 15][:num_seqs]
    if num_seqs > 3:
        node_counts = ([2, 9, 15] * num_seqs)[:num_seqs]
    prefixes = [block_size + 2, 3 * block_size, 2 * block_size + 7, block_size - 1,
                5 * block_size + 3][:num_seqs]
    seq_lens = [prefixes[s] + node_counts[s] for s in range(num_seqs)]
    num_blocks = sum((sl + block_size - 1) // block_size for sl in seq_lens) + 4
    k_pool, v_pool = _random_pool(num_blocks, block_size, Hkv, D, dtype)
    block_table, _ = _alloc_blocks(seq_lens, block_size, num_blocks)

    total_q = sum(node_counts)
    q = torch.randn(total_q, Hq, D, device=DEVICE, dtype=dtype)
    cu_list = [0]
    for nc in node_counts:
        cu_list.append(cu_list[-1] + nc)
    cu = torch.tensor(cu_list, dtype=torch.int32, device=DEVICE)
    seq_lens_k = torch.tensor(seq_lens, dtype=torch.int32, device=DEVICE)

    # block-diagonal qq_bias (total_q, total_q): 0 where node j is ancestor-or-self
    # of node i within the seq, -inf otherwise (off-diagonal seq blocks stay -inf
    # but are never read — the kernel only loads this seq's columns).
    qq_bias = torch.full((total_q, total_q), float("-inf"), device=DEVICE, dtype=torch.float32)
    masks = []
    for s in range(num_seqs):
        tree = _build_tree(node_counts[s], seed=seed * 31 + s)
        anc = build_ancestor_matrix(tree).to(DEVICE)        # (N, N) bool: anc[i,j] = j is anc-or-self of i
        lo, hi = cu_list[s], cu_list[s + 1]
        block = torch.where(anc, 0.0, float("-inf")).to(torch.float32)
        qq_bias[lo:hi, lo:hi] = block
        masks.append(anc)

    out = paged_tree_attn(q, k_pool, v_pool, block_table, cu, seq_lens_k,
                          qq_bias, scale, n_rep, block_size)

    refs = []
    for s in range(num_seqs):
        K, V = _gather_kv(k_pool, v_pool, block_table[s], seq_lens[s], block_size)
        nc = node_counts[s]
        ctx = seq_lens[s] - nc
        # oracle mask: 0 over prefix cols, ancestor (-inf/0) over node cols.
        node_mask = torch.where(masks[s], 0.0, float("-inf")).to(torch.float32)
        attn_mask = torch.zeros(nc, seq_lens[s], device=DEVICE, dtype=torch.float32)
        attn_mask[:, ctx:] = node_mask
        q_s = q[cu_list[s]:cu_list[s + 1]]
        refs.append(_sdpa_ref(q_s, K, V, n_rep, scale, attn_mask=attn_mask))
    ref = torch.cat(refs, dim=0)
    assert out.shape == ref.shape
    maxdiff = (out.float() - ref.float()).abs().max().item()
    assert torch.allclose(out.float(), ref.float(), atol=atol, rtol=rtol), (
        f"tree mismatch dtype={dtype} num_seqs={num_seqs} Hq={Hq} Hkv={Hkv} "
        f"bs={block_size} max|diff|={maxdiff:.3e}"
    )
    return maxdiff


# --- decode case (i) --------------------------------------------------------

@pytest.mark.parametrize("num_seqs", [1, 3, 8])
@pytest.mark.parametrize("Hq,Hkv", [(32, 8), (8, 8)])
@pytest.mark.parametrize("block_size", [16, 32])
def test_decode_fp32(num_seqs, Hq, Hkv, block_size):
    _run_decode(num_seqs, Hq, Hkv, 128, block_size, torch.float32, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("num_seqs", [1, 3, 8])
@pytest.mark.parametrize("Hq,Hkv", [(32, 8), (8, 8)])
def test_decode_bf16(num_seqs, Hq, Hkv):
    _run_decode(num_seqs, Hq, Hkv, 128, 16, torch.bfloat16, atol=2e-2, rtol=2e-2)


# --- tree case (ii) ---------------------------------------------------------

@pytest.mark.parametrize("num_seqs", [1, 3])
@pytest.mark.parametrize("Hq,Hkv", [(32, 8), (8, 8)])
@pytest.mark.parametrize("block_size", [16, 32])
def test_tree_fp32(num_seqs, Hq, Hkv, block_size):
    _run_tree(num_seqs, Hq, Hkv, 128, block_size, torch.float32, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("num_seqs", [1, 3])
@pytest.mark.parametrize("Hq,Hkv", [(32, 8), (8, 8)])
def test_tree_bf16(num_seqs, Hq, Hkv):
    _run_tree(num_seqs, Hq, Hkv, 128, 16, torch.bfloat16, atol=2e-2, rtol=2e-2)
