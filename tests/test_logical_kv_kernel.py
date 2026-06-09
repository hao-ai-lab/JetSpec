"""GPU gate for the paged tree-attention logical-KV indirection branch.

The default kernel path reads K/V by logical key position through ``block_table``.
The logical-KV branch instead maps a contiguous logical key range to absolute
physical pool slots. These tests pin the three acceptance gates for that branch:
identity maps are bitwise-identical to the default path, shuffled maps match an
SDPA oracle over the same remapped K/V sequence, and the default path still
matches the pre-change block-table reference.
"""
import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="logical-KV triton kernel needs CUDA"
)

if torch.cuda.is_available():
    from ptd.nano_vllm.paged_tree_attn import paged_tree_attn

DEVICE = "cuda"


def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """(Hkv, S, D) -> (Hq, S, D) GQA broadcast."""
    hkv, seq_len, head_dim = x.shape
    return x[:, None, :, :].expand(hkv, n_rep, seq_len, head_dim).reshape(
        hkv * n_rep, seq_len, head_dim
    )


def _make_case():
    torch.manual_seed(7)
    block_size = 16
    hq, hkv, head_dim = 4, 2, 64
    n_rep = hq // hkv
    scale = head_dim ** -0.5
    q_lens = [3, 2]
    seq_lens = [block_size + 5 + q_lens[0], 2 * block_size + 1 + q_lens[1]]
    total_q = sum(q_lens)
    num_blocks = 12

    q = torch.randn(total_q, hq, head_dim, device=DEVICE, dtype=torch.float32)
    k_pool = torch.randn(num_blocks, block_size, hkv, head_dim, device=DEVICE)
    v_pool = torch.randn(num_blocks, block_size, hkv, head_dim, device=DEVICE)
    block_table = torch.tensor(
        [
            [5, 1, 7],
            [3, 10, 2],
        ],
        dtype=torch.int32,
        device=DEVICE,
    )
    cu = torch.tensor([0, q_lens[0], total_q], dtype=torch.int32, device=DEVICE)
    seq_lens_k = torch.tensor(seq_lens, dtype=torch.int32, device=DEVICE)
    return q, k_pool, v_pool, block_table, cu, seq_lens_k, q_lens, seq_lens, scale, n_rep, block_size


def _identity_slots(block_table: torch.Tensor, seq_lens, block_size: int) -> torch.Tensor:
    slots = torch.zeros((len(seq_lens), max(seq_lens)), dtype=torch.int64, device=DEVICE)
    for seq_idx, seq_len in enumerate(seq_lens):
        for pos in range(seq_len):
            block = int(block_table[seq_idx, pos // block_size])
            slots[seq_idx, pos] = block * block_size + (pos % block_size)
    return slots


def _logical_slots(seq_lens, block_size, block_table, *, shuffled: bool):
    slots = _identity_slots(block_table, seq_lens, block_size)
    if shuffled:
        for seq_idx, seq_len in enumerate(seq_lens):
            perm = torch.randperm(seq_len, device=DEVICE)
            slots[seq_idx, :seq_len] = slots[seq_idx, perm]
    starts = torch.zeros(len(seq_lens), dtype=torch.int32, device=DEVICE)
    lens = torch.tensor(seq_lens, dtype=torch.int32, device=DEVICE)
    return slots, starts, lens


def _gather_by_slots(k_pool, v_pool, slots, seq_len, block_size):
    physical_slots = slots[:seq_len].to(torch.long)
    blocks = physical_slots // block_size
    offsets = physical_slots % block_size
    k = k_pool[blocks, offsets].transpose(0, 1).contiguous()
    v = v_pool[blocks, offsets].transpose(0, 1).contiguous()
    return k, v


def _sdpa_reference(q, k_pool, v_pool, slots, q_lens, seq_lens, scale, n_rep, block_size):
    refs = []
    q_start = 0
    for seq_idx, (q_len, seq_len) in enumerate(zip(q_lens, seq_lens)):
        q_s = q[q_start:q_start + q_len]
        q_start += q_len
        k, v = _gather_by_slots(k_pool, v_pool, slots[seq_idx], seq_len, block_size)
        query_pos = torch.arange(q_len, device=DEVICE)[:, None]
        key_pos = torch.arange(seq_len, device=DEVICE)[None, :]
        context_len = seq_len - q_len
        causal = key_pos <= context_len + query_pos
        attn_mask = torch.where(
            causal,
            torch.zeros((), device=DEVICE),
            torch.full((), float("-inf"), device=DEVICE),
        ).to(torch.float32)
        out = F.scaled_dot_product_attention(
            q_s.transpose(0, 1).unsqueeze(0),
            _repeat_kv(k, n_rep).unsqueeze(0),
            _repeat_kv(v, n_rep).unsqueeze(0),
            attn_mask=attn_mask.unsqueeze(0).unsqueeze(0),
            scale=scale,
        )
        refs.append(out.squeeze(0).transpose(0, 1))
    return torch.cat(refs, dim=0)


def test_identity_logical_slots_are_bitwise_identical_to_default_path():
    q, k_pool, v_pool, block_table, cu, seq_lens_k, q_lens, seq_lens, scale, n_rep, block_size = _make_case()
    logical_slots, logical_starts, logical_lens = _logical_slots(
        seq_lens, block_size, block_table, shuffled=False
    )

    out_default = paged_tree_attn(
        q, k_pool, v_pool, block_table, cu, seq_lens_k,
        None, scale, n_rep, block_size,
    )
    out_logical_identity = paged_tree_attn(
        q, k_pool, v_pool, block_table, cu, seq_lens_k,
        None, scale, n_rep, block_size,
        logical_slots, logical_starts, logical_lens,
    )

    assert torch.equal(out_logical_identity, out_default)


def test_shuffled_logical_slots_match_manual_sdpa_reference():
    q, k_pool, v_pool, block_table, cu, seq_lens_k, q_lens, seq_lens, scale, n_rep, block_size = _make_case()
    logical_slots, logical_starts, logical_lens = _logical_slots(
        seq_lens, block_size, block_table, shuffled=True
    )

    out = paged_tree_attn(
        q, k_pool, v_pool, block_table, cu, seq_lens_k,
        None, scale, n_rep, block_size,
        logical_slots, logical_starts, logical_lens,
    )
    ref = _sdpa_reference(
        q, k_pool, v_pool, logical_slots, q_lens, seq_lens, scale, n_rep, block_size
    )

    maxdiff = (out - ref).abs().max().item()
    assert torch.allclose(out, ref, atol=1e-4, rtol=1e-4), (
        f"shuffled logical-KV mismatch max|diff|={maxdiff:.3e}"
    )


def test_default_path_still_matches_block_table_sdpa_reference():
    q, k_pool, v_pool, block_table, cu, seq_lens_k, q_lens, seq_lens, scale, n_rep, block_size = _make_case()
    identity_slots, _, _ = _logical_slots(seq_lens, block_size, block_table, shuffled=False)

    out_omitted = paged_tree_attn(
        q, k_pool, v_pool, block_table, cu, seq_lens_k,
        None, scale, n_rep, block_size,
    )
    out_explicit_none = paged_tree_attn(
        q, k_pool, v_pool, block_table, cu, seq_lens_k,
        None, scale, n_rep, block_size,
        None, None, None,
    )
    ref = _sdpa_reference(
        q, k_pool, v_pool, identity_slots, q_lens, seq_lens, scale, n_rep, block_size
    )

    assert torch.equal(out_explicit_none, out_omitted)
    maxdiff = (out_omitted - ref).abs().max().item()
    assert torch.allclose(out_omitted, ref, atol=1e-4, rtol=1e-4), (
        f"default path regression max|diff|={maxdiff:.3e}"
    )
