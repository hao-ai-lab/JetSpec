import torch
import torch.nn.functional as F
import pytest
from transformers import Qwen3Config, Qwen3ForCausalLM


def _tiny_cpu_model(seed: int = 0) -> Qwen3ForCausalLM:
    torch.manual_seed(seed)
    cfg = Qwen3Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
        tie_word_embeddings=False,
    )
    return Qwen3ForCausalLM(cfg).eval().to(dtype=torch.float32)


def _cpu_paged_tree_attn(
    q,
    k_pool,
    v_pool,
    block_table,
    cu_seqlens_q,
    seq_lens_k,
    qq_bias,
    scale,
    num_queries_per_kv,
    block_size,
    logical_kv_slots=None,
    logical_kv_starts=None,
    logical_kv_lens=None,
):
    outs = []
    for seq_idx in range(seq_lens_k.numel()):
        q_start = int(cu_seqlens_q[seq_idx])
        q_end = int(cu_seqlens_q[seq_idx + 1])
        q_seq = q[q_start:q_end]
        key_len = int(seq_lens_k[seq_idx])
        slots = torch.arange(key_len, device=q.device)
        block_ids = block_table[seq_idx, slots // block_size].long()
        offsets = slots % block_size
        k_seq = k_pool[block_ids, offsets]
        v_seq = v_pool[block_ids, offsets]
        k_seq = k_seq.repeat_interleave(num_queries_per_kv, dim=1)
        v_seq = v_seq.repeat_interleave(num_queries_per_kv, dim=1)
        scores = torch.einsum("qhd,khd->hqk", q_seq, k_seq) * scale
        if qq_bias is not None:
            scores = scores + qq_bias[q_start:q_end, :key_len].unsqueeze(0)
        probs = torch.softmax(scores, dim=-1)
        outs.append(torch.einsum("hqk,khd->qhd", probs, v_seq))
    return torch.cat(outs, dim=0)


def _stack_args(model, n: int, *, real_n=None):
    real_n = n if real_n is None else real_n
    cfg = model.config
    block_size = 4
    num_blocks = (n + block_size - 1) // block_size
    torch.manual_seed(1000 + n + real_n)
    input_ids = torch.randint(0, cfg.vocab_size, (1, n))
    if real_n < n:
        input_ids[:, real_n:] = 0
    positions = torch.arange(n).unsqueeze(0)
    dummy = torch.zeros(1, n, cfg.hidden_size)
    cos, sin = model.model.rotary_emb(dummy, positions)
    slots = torch.arange(n)
    node_blks = [slots // block_size for _ in range(cfg.num_hidden_layers)]
    node_offs = [slots % block_size for _ in range(cfg.num_hidden_layers)]
    k_pools = [
        torch.zeros(num_blocks, block_size, cfg.num_key_value_heads, cfg.head_dim)
        for _ in range(cfg.num_hidden_layers)
    ]
    v_pools = [
        torch.zeros(num_blocks, block_size, cfg.num_key_value_heads, cfg.head_dim)
        for _ in range(cfg.num_hidden_layers)
    ]
    block_tables = [
        torch.arange(num_blocks, dtype=torch.int32).view(1, -1).clone()
        for _ in range(cfg.num_hidden_layers)
    ]
    qq_bias = torch.zeros(n, n)
    if real_n < n:
        qq_bias[:real_n, real_n:] = float("-inf")
    return dict(
        input_ids=input_ids,
        cos=cos,
        sin=sin,
        k_pools=k_pools,
        v_pools=v_pools,
        block_tables=block_tables,
        cu=torch.tensor([0, n], dtype=torch.int32),
        seq_lens_k=torch.tensor([n], dtype=torch.int32),
        qq_bias=qq_bias,
        node_blks=node_blks,
        node_offs=node_offs,
    )


def _fresh_args(args):
    out = {}
    for key, value in args.items():
        if isinstance(value, list):
            out[key] = [item.clone() for item in value]
        elif torch.is_tensor(value):
            out[key] = value.clone()
        else:
            out[key] = value
    return out


def _chain_qq_bias(n: int) -> torch.Tensor:
    from jetspec.tree import DraftTree, build_ancestor_matrix

    parent_indices = torch.tensor([-1] + list(range(n - 1)), dtype=torch.long)
    tree = DraftTree(
        token_ids=torch.arange(n, dtype=torch.long),
        parent_indices=parent_indices,
        depth=torch.arange(n, dtype=torch.long),
        num_nodes=n,
    )
    ancestor = build_ancestor_matrix(tree)
    return torch.where(ancestor, torch.zeros(()), torch.full((), float("-inf")))


def _bucketed_stack_args(model, real_n: int, bucket_n: int, qq_bias: torch.Tensor):
    cfg = model.config
    block_size = 4
    num_blocks = (bucket_n + block_size - 1) // block_size
    torch.manual_seed(2000 + real_n)
    input_ids = torch.randint(0, cfg.vocab_size, (1, real_n))
    positions = torch.arange(real_n).unsqueeze(0)
    if bucket_n > real_n:
        pad = bucket_n - real_n
        input_ids = torch.cat([input_ids, torch.zeros(1, pad, dtype=input_ids.dtype)], dim=1)
        positions = torch.cat([positions, positions[:, -1:].expand(1, pad)], dim=1)
        padded_bias = torch.full((bucket_n, bucket_n), float("-inf"))
        padded_bias[:real_n, :real_n] = qq_bias
        pad_idx = torch.arange(real_n, bucket_n)
        padded_bias[pad_idx, pad_idx] = 0.0
        qq_bias = padded_bias
    dummy = torch.zeros(1, bucket_n, cfg.hidden_size)
    cos, sin = model.model.rotary_emb(dummy, positions)
    slots = torch.arange(bucket_n)
    node_blks = [slots // block_size for _ in range(cfg.num_hidden_layers)]
    node_offs = [slots % block_size for _ in range(cfg.num_hidden_layers)]
    k_pools = [
        torch.zeros(num_blocks, block_size, cfg.num_key_value_heads, cfg.head_dim)
        for _ in range(cfg.num_hidden_layers)
    ]
    v_pools = [
        torch.zeros(num_blocks, block_size, cfg.num_key_value_heads, cfg.head_dim)
        for _ in range(cfg.num_hidden_layers)
    ]
    block_tables = [
        torch.arange(num_blocks, dtype=torch.int32).view(1, -1).clone()
        for _ in range(cfg.num_hidden_layers)
    ]
    return dict(
        input_ids=input_ids,
        cos=cos,
        sin=sin,
        k_pools=k_pools,
        v_pools=v_pools,
        block_tables=block_tables,
        cu=torch.tensor([0, bucket_n], dtype=torch.int32),
        seq_lens_k=torch.tensor([bucket_n], dtype=torch.int32),
        qq_bias=qq_bias,
        node_blks=node_blks,
        node_offs=node_offs,
    )


def test_small_bucket_math_on_cpu():
    from jetspec.inference_engine.engine import _TREE_BUCKETS, _bucket_for_n

    assert _TREE_BUCKETS == (16, 32, 64, 128, 192, 256)
    assert [_bucket_for_n(n) for n in (1, 15, 16, 17, 31, 32, 33, 63)] == [
        16,
        16,
        16,
        32,
        32,
        32,
        64,
        64,
    ]
    assert [_bucket_for_n(n) for n in (64, 65, 255, 256, 257)] == [
        64,
        128,
        256,
        256,
        512,
    ]


@pytest.mark.parametrize("n,real_n", [(1, 1), (8, 8), (12, 8)])
@pytest.mark.parametrize("target_layer_ids", [None, (0,), (0, 1)])
@torch.inference_mode()
def test_fused_gemms_match_unfused_stack_on_cpu(monkeypatch, n, real_n, target_layer_ids):
    from jetspec.inference_engine.compiled_verify_stack import CompiledVerifyStack

    monkeypatch.setattr(torch.ops.jetspec, "paged_tree_attn", _cpu_paged_tree_attn)
    model = _tiny_cpu_model()
    need_hidden = target_layer_ids is not None
    args = _stack_args(model, n, real_n=real_n)
    unfused = CompiledVerifyStack(
        model,
        block_size=4,
        need_hidden=need_hidden,
        target_layer_ids=target_layer_ids,
        fuse_gemms=False,
    )
    fused = CompiledVerifyStack(
        model,
        block_size=4,
        need_hidden=need_hidden,
        target_layer_ids=target_layer_ids,
        fuse_gemms=True,
    )

    expected = unfused._stack(**_fresh_args(args))
    actual = fused._stack(**_fresh_args(args))

    if need_hidden:
        expected_logits, expected_hidden = expected
        actual_logits, actual_hidden = actual
        torch.testing.assert_close(
            actual_hidden[:, :real_n].float(),
            expected_hidden[:, :real_n].float(),
            rtol=1e-6,
            atol=1e-6,
        )
    else:
        expected_logits = expected
        actual_logits = actual
    torch.testing.assert_close(
        actual_logits[:, :real_n].float(),
        expected_logits[:, :real_n].float(),
        rtol=1e-6,
        atol=1e-6,
    )


@pytest.mark.parametrize("real_n,bucket_n", [(15, 16), (31, 32)])
@torch.inference_mode()
def test_small_bucket_padding_matches_unpadded_stack_on_cpu(monkeypatch, real_n, bucket_n):
    from jetspec.inference_engine.compiled_verify_stack import CompiledVerifyStack

    monkeypatch.setattr(torch.ops.jetspec, "paged_tree_attn", _cpu_paged_tree_attn)
    model = _tiny_cpu_model()
    qq_bias = torch.zeros(real_n, real_n)
    stack = CompiledVerifyStack(model, block_size=4, fuse_gemms=False)
    unpadded = stack._stack(**_bucketed_stack_args(model, real_n, real_n, qq_bias))
    padded = stack._stack(**_bucketed_stack_args(model, real_n, bucket_n, qq_bias))

    torch.testing.assert_close(
        padded[:, :real_n].float(),
        unpadded.float(),
        rtol=1e-6,
        atol=1e-6,
    )


@torch.inference_mode()
def test_bucket_16_full_depth_chain_mask_matches_sdpa_on_cpu(monkeypatch):
    from jetspec.inference_engine.compiled_verify_stack import CompiledVerifyStack
    from transformers import DynamicCache

    monkeypatch.setattr(torch.ops.jetspec, "paged_tree_attn", _cpu_paged_tree_attn)
    model = _tiny_cpu_model()
    qq_bias = _chain_qq_bias(16)
    stack = CompiledVerifyStack(model, block_size=4, fuse_gemms=False)
    args = _bucketed_stack_args(model, 16, 16, qq_bias)
    logits = stack._stack(**_fresh_args(args))
    ref = model(
        input_ids=args["input_ids"],
        position_ids=torch.arange(16).unsqueeze(0),
        attention_mask=qq_bias.view(1, 1, 16, 16),
        past_key_values=DynamicCache(),
        use_cache=True,
    ).logits

    assert qq_bias.shape == (16, 16)
    assert torch.equal(qq_bias == 0, torch.tril(torch.ones(16, 16, dtype=torch.bool)))
    torch.testing.assert_close(logits.float(), ref.float(), rtol=1e-5, atol=1e-5)


@torch.inference_mode()
def test_fused_gemms_reduce_linear_calls_per_layer_on_cpu(monkeypatch):
    from jetspec.inference_engine.compiled_verify_stack import CompiledVerifyStack

    monkeypatch.setattr(torch.ops.jetspec, "paged_tree_attn", _cpu_paged_tree_attn)
    model = _tiny_cpu_model()
    args = _stack_args(model, 8)
    unfused = CompiledVerifyStack(model, block_size=4, fuse_gemms=False)
    fused = CompiledVerifyStack(model, block_size=4, fuse_gemms=True)
    counts = {"linear": 0}
    original_linear = F.linear

    def counting_linear(*linear_args, **linear_kwargs):
        counts["linear"] += 1
        return original_linear(*linear_args, **linear_kwargs)

    monkeypatch.setattr(F, "linear", counting_linear)
    unfused._stack(**_fresh_args(args))
    unfused_calls = counts["linear"]
    counts["linear"] = 0
    fused._stack(**_fresh_args(args))
    fused_calls = counts["linear"]

    num_layers = model.config.num_hidden_layers
    assert unfused_calls == num_layers * 7 + 1
    assert fused_calls == num_layers * 4 + 1
