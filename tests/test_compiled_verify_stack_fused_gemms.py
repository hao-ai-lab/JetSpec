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


@pytest.mark.parametrize("n,real_n", [(1, 1), (8, 8), (12, 8)])
@pytest.mark.parametrize("target_layer_ids", [None, (0,), (0, 1)])
@torch.inference_mode()
def test_fused_gemms_match_unfused_stack_on_cpu(monkeypatch, n, real_n, target_layer_ids):
    from ptd.nano_vllm.compiled_verify_stack import CompiledVerifyStack

    monkeypatch.setattr(torch.ops.ptd, "paged_tree_attn", _cpu_paged_tree_attn)
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


@torch.inference_mode()
def test_fused_gemms_reduce_linear_calls_per_layer_on_cpu(monkeypatch):
    from ptd.nano_vllm.compiled_verify_stack import CompiledVerifyStack

    monkeypatch.setattr(torch.ops.ptd, "paged_tree_attn", _cpu_paged_tree_attn)
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
