import pytest
import torch

from ptd.jetflow import paged_attn_backend
from ptd.jetflow.paged_kv_cache import PagedHandle, PagedKVCache
from ptd.draft import TargetEchoTreeDrafter
from ptd.engine.llm import SamplingParams
from ptd.tree import DraftTree
from ptd.tree.linearize import expand_tree_to_paths
from tests.test_jetflow_tree import PROMPT, _tiny_jetflow, _tiny_model


def _append_seq(cache, seq_id, length, *, layer=0, heads=2, dim=8):
    values = torch.arange(heads * length * dim, dtype=torch.float32).reshape(
        1, heads, length, dim
    )
    values = values + (1000 * seq_id)
    cache.append(values, values + 0.5, layer, seq_id=seq_id)


def _cache_with_lengths(lengths, *, block_size=4, heads=2, dim=8):
    cache = PagedKVCache(block_size=block_size, dtype=torch.float32)
    for seq_id, length in lengths.items():
        _append_seq(cache, seq_id, length, heads=heads, dim=dim)
    return cache


def _capture_kernel(monkeypatch):
    captured = {}

    def fake_paged_tree_attn(
        q,
        k_pool,
        v_pool,
        block_table,
        cu_seqlens_q,
        seq_lens_k,
        qq_bias,
        scaling,
        num_queries_per_kv,
        block_size,
        logical_kv_slots=None,
        logical_kv_starts=None,
        logical_kv_lens=None,
    ):
        captured.update(
            q=q,
            k_pool=k_pool,
            v_pool=v_pool,
            block_table=block_table,
            cu_seqlens_q=cu_seqlens_q,
            seq_lens_k=seq_lens_k,
            qq_bias=qq_bias,
            scaling=scaling,
            num_queries_per_kv=num_queries_per_kv,
            block_size=block_size,
            logical_kv_slots=logical_kv_slots,
            logical_kv_starts=logical_kv_starts,
            logical_kv_lens=logical_kv_lens,
        )
        return torch.zeros_like(q)

    monkeypatch.setattr(paged_attn_backend, "paged_tree_attn", fake_paged_tree_attn)
    return captured


def _branching_plan():
    tree = DraftTree(
        token_ids=torch.tensor([10, 11, 12, 13], dtype=torch.long),
        parent_indices=torch.tensor([-1, 0, 0, 1], dtype=torch.long),
        depth=torch.tensor([0, 1, 1, 2], dtype=torch.long),
        num_nodes=4,
    )
    return expand_tree_to_paths(tree)


def test_linearized_meta_passes_ragged_cu_and_no_bias_to_kernel(monkeypatch):
    captured = _capture_kernel(monkeypatch)
    plan = _branching_plan()
    total_q = int(plan.token_ids.numel())
    past_len = 3
    cache = _cache_with_lengths({0: past_len + total_q}, block_size=4)
    cache._ptd_attn_meta = {
        "seq_ids": [0],
        "qq_bias": None,
        "cu_seqlens": plan.cu_seqlens,
    }
    query = torch.randn(1, 4, total_q, 8)
    handle = PagedHandle(cache, 0, "k")

    out, attn_weights = paged_attn_backend._ptd_paged_tree_attn_forward(
        None, query, handle, PagedHandle(cache, 0, "v"), None, scaling=0.125
    )

    assert attn_weights is None
    assert out.shape == (1, total_q, 4, 8)
    assert torch.equal(captured["cu_seqlens_q"], plan.cu_seqlens)
    assert captured["qq_bias"] is None

    path_lens = plan.cu_seqlens[1:] - plan.cu_seqlens[:-1]
    assert captured["seq_lens_k"].tolist() == (path_lens + past_len).tolist()
    assert captured["block_table"].shape[0] == path_lens.numel()
    assert torch.equal(captured["block_table"][0], captured["block_table"][1])
    assert captured["logical_kv_starts"].tolist() == [past_len, past_len]
    assert captured["logical_kv_lens"].tolist() == path_lens.tolist()

    base_table = cache.kernel_block_table([0], 0, device=query.device)[0]
    physical_positions = past_len + torch.arange(total_q)
    expected_slots = (
        base_table[physical_positions // cache.block_size].to(torch.long)
        * cache.block_size
        + (physical_positions % cache.block_size)
    )
    logical = captured["logical_kv_slots"]
    assert logical.shape == (path_lens.numel(), total_q)
    assert torch.equal(logical[0, : path_lens[0]], expected_slots[: path_lens[0]])
    assert torch.equal(logical[1, : path_lens[1]], expected_slots[path_lens[0] :])


def test_without_ragged_cu_keeps_rectangular_kernel_metadata(monkeypatch):
    captured = _capture_kernel(monkeypatch)
    cache = _cache_with_lengths({0: 5, 1: 7}, block_size=4)
    qq_bias = torch.ones(6, 6, dtype=torch.float16)
    cache._ptd_attn_meta = {"seq_ids": [0, 1], "qq_bias": qq_bias}
    query = torch.randn(2, 4, 3, 8)
    handle = PagedHandle(cache, 0, "k")

    out, _ = paged_attn_backend._ptd_paged_tree_attn_forward(
        None, query, handle, PagedHandle(cache, 0, "v"), None, scaling=0.125
    )

    assert out.shape == (2, 3, 4, 8)
    assert captured["cu_seqlens_q"].tolist() == [0, 3, 6]
    assert captured["seq_lens_k"].tolist() == [5, 7]
    assert captured["qq_bias"].dtype == torch.float32
    assert captured["qq_bias"].is_contiguous()
    assert captured["logical_kv_slots"] is None
    assert captured["logical_kv_starts"] is None
    assert captured["logical_kv_lens"] is None


def test_ragged_cu_must_be_int32_and_well_formed(monkeypatch):
    _capture_kernel(monkeypatch)
    plan = _branching_plan()
    total_q = int(plan.token_ids.numel())
    cache = _cache_with_lengths({0: 2 + total_q}, block_size=4)
    cache._ptd_attn_meta = {
        "seq_ids": [0],
        "qq_bias": None,
        "cu_seqlens": plan.cu_seqlens.to(torch.int64),
    }
    query = torch.randn(1, 4, total_q, 8)
    handle = PagedHandle(cache, 0, "k")

    with pytest.raises(ValueError, match="cu_seqlens.*int32"):
        paged_attn_backend._ptd_paged_tree_attn_forward(
            None, query, handle, PagedHandle(cache, 0, "v"), None, scaling=0.125
        )


def test_linearized_eager_kernel_engine_reaches_ragged_backend(monkeypatch):
    captured = _capture_kernel(monkeypatch)
    monkeypatch.setenv("PTD_LINEARIZE_VERIFY", "1")
    paged_attn_backend.register_ptd_paged_tree()
    model = _tiny_model(0)
    model.config._attn_implementation = "ptd_paged_tree"
    eng = _tiny_jetflow(model, block_size=4)
    eng.attn_backend = "triton_paged_tree"

    out = eng.generate_tree(
        PROMPT,
        TargetEchoTreeDrafter(model),
        block_size=4,
        tree_width=2,
        budget=6,
        sampling_params=SamplingParams(0.0, 4),
    )

    assert out["token_ids"]
    assert captured["cu_seqlens_q"].numel() > 2
    assert captured["qq_bias"] is None
    assert captured["logical_kv_slots"] is not None
