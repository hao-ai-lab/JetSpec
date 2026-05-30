"""Tests for tree-masked sparse attention.

Run:
    PYTHONPATH=/data/tree_attention/optimus_jit/src \\
    pytest /data/tree_attention/optimus_jit/src/optimus_cutedsl/flash_attn/tail_tree_attn/reference_tests.py -v -s

Each test builds a tree + ancestor mask, runs the naive PyTorch reference,
and checks it against F.sdpa with a 4-D mask. The cuda_tree_attention() bridge
is where the optimized kernel is connected for correctness checks.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from optimus_cutedsl import flash_attn_varlen_tree_paged_sm90

try:
    from .reference_ops import (
        DraftTree,
        build_ancestor_matrix,
        build_tree_attention_mask,
        build_tree_from_topk,
        compute_tree_budget,
        reference_tree_attention,
    )
except ImportError:
    from reference_ops import (
        DraftTree,
        build_ancestor_matrix,
        build_tree_attention_mask,
        build_tree_from_topk,
        compute_tree_budget,
        reference_tree_attention,
    )


_PAGE_SIZE = 128


def _pack_kv_to_paged(
    key: torch.Tensor,
    value: torch.Tensor,
    page_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Note(wangbojun/codex): the SM90 tree kernel consumes paged KV cache. This
    # repack exists only because this reference bridge exposes contiguous KV
    # tensors for correctness checks; a real decode hotpath should pass paged KV
    # directly instead of rebuilding page tables on every call.
    B, H_KV, KV_LEN, D = key.shape
    num_pages_per_seq = (KV_LEN + page_size - 1) // page_size
    padded_kv_len = num_pages_per_seq * page_size
    if padded_kv_len != KV_LEN:
        pad = padded_kv_len - KV_LEN
        key = F.pad(key, (0, 0, 0, pad))
        value = F.pad(value, (0, 0, 0, pad))
    key_pages = key.permute(0, 2, 1, 3).contiguous().view(
        B, num_pages_per_seq, page_size, H_KV, D
    )
    value_pages = value.permute(0, 2, 1, 3).contiguous().view(
        B, num_pages_per_seq, page_size, H_KV, D
    )
    page_table = torch.arange(
        B * num_pages_per_seq, device=key.device, dtype=torch.int32
    ).view(B, num_pages_per_seq)
    return key_pages.view(B * num_pages_per_seq, page_size, H_KV, D), value_pages.view(
        B * num_pages_per_seq, page_size, H_KV, D
    ), page_table


def cuda_tree_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    ancestor: torch.Tensor,
    prefix_len: int,
    sm_scale: float,
) -> torch.Tensor:
    """
    Query i attends to prefix positions 0..prefix_len-1 (always),
    and to tree position j (KV index prefix_len+j) only if ancestor[i,j].
    GQA: map query head h to KV head h * H_KV // H.
    Output: (B, H, N, D), same dtype as query.
    """
    if not (query.is_cuda and key.is_cuda and value.is_cuda and ancestor.is_cuda):
        raise ValueError("cuda_tree_attention requires CUDA query/key/value/ancestor tensors")
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("query/key/value must be 4-D tensors")
    if ancestor.ndim != 2 or ancestor.shape[0] != ancestor.shape[1]:
        raise ValueError("ancestor must have shape (N, N)")

    B, H, N, D = query.shape
    B_kv, H_KV, KV_LEN, D_k = key.shape
    if value.shape != (B_kv, H_KV, KV_LEN, D_k):
        raise ValueError("value must have the same shape as key")
    if B_kv != B:
        raise ValueError("query and key batch dimensions must match")
    if D_k != D:
        raise ValueError("query/key/value head_dim must match")
    if H % H_KV != 0:
        raise ValueError("num query heads must be divisible by num kv heads")
    if ancestor.shape[0] != N:
        raise ValueError("ancestor size must match query seqlen")
    if KV_LEN != prefix_len + N:
        raise ValueError("key/value seqlen must equal prefix_len + tree size")
    if torch.cuda.get_device_capability(query.device)[0] != 9:
        raise NotImplementedError("cuda_tree_attention currently requires SM90")

    q_varlen = query.permute(0, 2, 1, 3).contiguous().view(B * N, H, D)
    k_paged, v_paged, page_table = _pack_kv_to_paged(key, value, _PAGE_SIZE)
    cu_seqlens_q = torch.arange(0, (B + 1) * N, N, device=query.device, dtype=torch.int32)
    context_lens = torch.full((B,), KV_LEN, device=query.device, dtype=torch.int32)

    out = flash_attn_varlen_tree_paged_sm90(
        q_varlen,
        k_paged,
        v_paged,
        ancestor,
        cu_seqlens_q,
        context_lens,
        page_table,
        softmax_scale=sm_scale,
        pack_gqa=H_KV != H,
        m_block_size=_PAGE_SIZE,
        n_block_size=_PAGE_SIZE,
    )
    return out.view(B, N, H, D).permute(0, 2, 1, 3).contiguous()


def _require_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")


def _has_sm90():
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 9


def _make_tree(num_nodes, parent_indices, device):
    depths = [0] * num_nodes
    for i in range(1, num_nodes):
        depths[i] = depths[parent_indices[i]] + 1
    return DraftTree(
        token_ids=torch.randint(0, 32000, (num_nodes,), device=device),
        parent_indices=torch.tensor(parent_indices, dtype=torch.long, device=device),
        depth=torch.tensor(depths, dtype=torch.long, device=device),
        num_nodes=num_nodes,
    )


def _random_qkv(B, H, H_KV, N, KV_LEN, D, dtype, device):
    q = torch.randn(B, H, N, D, dtype=dtype, device=device)
    k = torch.randn(B, H_KV, KV_LEN, D, dtype=dtype, device=device)
    v = torch.randn(B, H_KV, KV_LEN, D, dtype=dtype, device=device)
    return q, k, v


def _check_ref_vs_sdpa(tree, prefix_len, B, H, H_KV, D, dtype, device):
    """Run reference and F.sdpa, assert they match, return all intermediates."""
    N = tree.num_nodes
    KV_LEN = prefix_len + N
    sm_scale = D**-0.5

    ancestor = build_ancestor_matrix(tree)
    q, k, v = _random_qkv(B, H, H_KV, N, KV_LEN, D, dtype, device)

    out_ref = reference_tree_attention(q, k, v, ancestor, prefix_len, sm_scale)

    mask_4d = build_tree_attention_mask(tree, prefix_len, dtype, device)
    k_exp = k.repeat_interleave(H // H_KV, dim=1)
    v_exp = v.repeat_interleave(H // H_KV, dim=1)
    out_sdpa = F.scaled_dot_product_attention(
        q, k_exp, v_exp, attn_mask=mask_4d, scale=sm_scale
    )

    torch.testing.assert_close(out_ref, out_sdpa, atol=1e-2, rtol=1e-2)
    return q, k, v, ancestor, prefix_len, sm_scale, out_ref


def _print_sparsity(ancestor, prefix_len):
    N = ancestor.shape[0]
    KV_LEN = prefix_len + N
    total = N * KV_LEN
    prefix_entries = N * prefix_len
    tree_attended = ancestor.sum().item()
    attended = prefix_entries + tree_attended
    print(f"  {N} queries x {KV_LEN} keys = {total} entries")
    print(f"  prefix (dense): {prefix_entries} ({100 * prefix_entries / total:.1f}%)")
    print(
        f"  tree block: {N * N} entries, {tree_attended} attended "
        f"({100 * tree_attended / (N * N):.1f}%)"
    )
    print(f"  overall density: {100 * attended / total:.1f}%")


class TestBlk8W2:
    BLOCK_SIZE = 8
    TREE_WIDTH = 2
    H, H_KV, D = 32, 8, 128
    DTYPE = torch.bfloat16

    def _build_tree(self, device):
        budget = compute_tree_budget(self.BLOCK_SIZE, self.TREE_WIDTH)
        draft_depth = self.BLOCK_SIZE - 1
        topk_tokens = torch.randint(0, 32000, (draft_depth, self.TREE_WIDTH), device=device)
        topk_logprobs = torch.randn(draft_depth, self.TREE_WIDTH, device=device)
        return build_tree_from_topk(42, topk_tokens, topk_logprobs, budget, device)

    def test_short_prefix(self):
        _require_cuda()
        device = torch.device("cuda")
        tree = self._build_tree(device)
        print(f"\n[blk8w2] N={tree.num_nodes}, prefix=64")
        _check_ref_vs_sdpa(tree, 64, 1, self.H, self.H_KV, self.D, self.DTYPE, device)

    def test_medium_prefix(self):
        _require_cuda()
        device = torch.device("cuda")
        tree = self._build_tree(device)
        print(f"\n[blk8w2] N={tree.num_nodes}, prefix=512")
        _, _, _, ancestor, prefix_len, _, _ = _check_ref_vs_sdpa(
            tree, 512, 1, self.H, self.H_KV, self.D, self.DTYPE, device
        )
        _print_sparsity(ancestor, prefix_len)

    def test_long_prefix(self):
        _require_cuda()
        device = torch.device("cuda")
        tree = self._build_tree(device)
        print(f"\n[blk8w2] N={tree.num_nodes}, prefix=2048")
        _check_ref_vs_sdpa(tree, 2048, 1, self.H, self.H_KV, self.D, self.DTYPE, device)

    @pytest.mark.skipif(not _has_sm90(), reason="SM90 tree kernel required")
    def test_cuda_kernel(self):
        _require_cuda()
        device = torch.device("cuda")
        tree = self._build_tree(device)
        N = tree.num_nodes
        prefix_len = 512
        sm_scale = self.D**-0.5

        ancestor = build_ancestor_matrix(tree)
        q, k, v = _random_qkv(1, self.H, self.H_KV, N, prefix_len + N, self.D, self.DTYPE, device)

        out_ref = reference_tree_attention(q, k, v, ancestor, prefix_len, sm_scale)
        out_cuda = cuda_tree_attention(q, k, v, ancestor.to(torch.uint8), prefix_len, sm_scale)
        torch.testing.assert_close(out_cuda, out_ref, atol=1e-2, rtol=1e-2)


class TestSmallTrees:
    H, H_KV, D = 32, 8, 128
    DTYPE = torch.bfloat16

    def test_single_node(self):
        _require_cuda()
        device = torch.device("cuda")
        tree = _make_tree(1, [-1], device)
        _check_ref_vs_sdpa(tree, 32, 1, self.H, self.H_KV, self.D, self.DTYPE, device)

    def test_linear_chain(self):
        _require_cuda()
        device = torch.device("cuda")
        tree = _make_tree(8, [-1, 0, 1, 2, 3, 4, 5, 6], device)
        _check_ref_vs_sdpa(tree, 64, 1, self.H, self.H_KV, self.D, self.DTYPE, device)

    def test_binary_tree_depth3(self):
        """       0
              / \\
             1   2
            /\\ /\\
           3 4 5 6"""

        _require_cuda()
        device = torch.device("cuda")
        tree = _make_tree(7, [-1, 0, 0, 1, 1, 2, 2], device)
        _, _, _, ancestor, _, _, _ = _check_ref_vs_sdpa(
            tree, 64, 1, self.H, self.H_KV, self.D, self.DTYPE, device
        )
        print("\n[binary depth-3] ancestor matrix (7x7):")
        print(ancestor.int())

    def test_no_prefix(self):
        _require_cuda()
        device = torch.device("cuda")
        tree = _make_tree(5, [-1, 0, 0, 1, 2], device)
        _check_ref_vs_sdpa(tree, 0, 1, self.H, self.H_KV, self.D, self.DTYPE, device)

    def test_fp16(self):
        _require_cuda()
        device = torch.device("cuda")
        tree = _make_tree(7, [-1, 0, 0, 1, 1, 2, 2], device)
        _check_ref_vs_sdpa(tree, 64, 1, self.H, self.H_KV, 64, torch.float16, device)

    @pytest.mark.skipif(not _has_sm90(), reason="SM90 tree kernel required")
    def test_cuda_kernel(self):
        _require_cuda()
        device = torch.device("cuda")
        tree = _make_tree(7, [-1, 0, 0, 1, 1, 2, 2], device)
        N = tree.num_nodes
        prefix_len = 64
        sm_scale = self.D**-0.5

        ancestor = build_ancestor_matrix(tree)
        q, k, v = _random_qkv(1, self.H, self.H_KV, N, prefix_len + N, self.D, self.DTYPE, device)

        out_ref = reference_tree_attention(q, k, v, ancestor, prefix_len, sm_scale)
        out_cuda = cuda_tree_attention(q, k, v, ancestor.to(torch.uint8), prefix_len, sm_scale)
        torch.testing.assert_close(out_cuda, out_ref, atol=1e-2, rtol=1e-2)


def main():
    assert torch.cuda.is_available(), "CUDA required"
    device = torch.device("cuda")
    dtype = torch.bfloat16

    block_size, tree_width = 8, 2
    budget = compute_tree_budget(block_size, tree_width)

    topk_tokens = torch.randint(0, 32000, (block_size - 1, tree_width), device=device)
    topk_logprobs = torch.randn(block_size - 1, tree_width, device=device)
    tree = build_tree_from_topk(42, topk_tokens, topk_logprobs, budget, device)

    N = tree.num_nodes
    ancestor = build_ancestor_matrix(tree)

    B, H, H_KV, D = 1, 32, 8, 128
    prefix_len = 512
    KV_LEN = prefix_len + N
    sm_scale = D**-0.5

    q, k, v = _random_qkv(B, H, H_KV, N, KV_LEN, D, dtype, device)

    print("=" * 60)
    print("blk8w2 reference case")
    print("=" * 60)
    print(f"  block_size={block_size}, tree_width={tree_width}, N={N}")
    print(f"  Q ({B}, {H}, {N}, {D}), K/V ({B}, {H_KV}, {KV_LEN}, {D})")
    print(f"  GQA groups={H // H_KV}, dtype={dtype}, prefix={prefix_len}")
    print()
    _print_sparsity(ancestor, prefix_len)
    print()

    out_ref = reference_tree_attention(q, k, v, ancestor, prefix_len, sm_scale)

    mask_4d = build_tree_attention_mask(tree, prefix_len, dtype, device)
    k_exp = k.repeat_interleave(H // H_KV, dim=1)
    v_exp = v.repeat_interleave(H // H_KV, dim=1)
    out_sdpa = F.scaled_dot_product_attention(q, k_exp, v_exp, attn_mask=mask_4d, scale=sm_scale)
    torch.testing.assert_close(out_ref, out_sdpa, atol=1e-2, rtol=1e-2)
    print("  PASS: reference matches F.sdpa")

    try:
        out_cuda = cuda_tree_attention(q, k, v, ancestor.to(torch.uint8), prefix_len, sm_scale)
        torch.testing.assert_close(out_cuda, out_ref, atol=1e-2, rtol=1e-2)
        print("  PASS: CUDA kernel matches reference")
    except NotImplementedError:
        print("  SKIP: cuda_tree_attention() not yet implemented")

    print("\nReplace cuda_tree_attention() and re-run to verify.")


if __name__ == "__main__":
    main()

