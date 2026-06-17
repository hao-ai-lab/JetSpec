"""CPU gate for the paged tree-attention ``custom_op`` (JetFlow N3, foundation).

The CPU-runnable checks need NO triton/CUDA: the op is registered, the fake
(meta) kernel returns the right shape/dtype, and ``torch.compile(fullgraph=True)``
traces *past* the op as an opaque fusion boundary. Full ``opcheck`` is CUDA-only
(every opcheck utility executes the real op, which forwards to triton) and runs
as part of the b200 kernel gate alongside the kernel==SDPA oracle in
``test_jetflow_kernel.py``.

The op exists so a future compiled read-only decoder forward fuses the GEMMs
around attention without graph-breaking on the ``@triton.jit`` launch.
"""
import pytest
import torch

from jetflow.inference_engine.paged_tree_attn_op import paged_tree_attn


# Small, kernel-shaped inputs: total_q=20 ragged over 3 seqs, Hq=8/Hkv=2 (GQA),
# head_dim=128, block_size=16. fp32 so dtype assertions are unambiguous on CPU.
TOTAL_Q, HQ, HKV, D = 20, 8, 2, 128
BLOCK_SIZE, NUM_BLOCKS = 16, 4
SCALE = D ** -0.5
NQPKV = HQ // HKV


def _inputs(with_bias: bool):
    q = torch.randn(TOTAL_Q, HQ, D)
    k_pool = torch.randn(NUM_BLOCKS, BLOCK_SIZE, HKV, D)
    v_pool = torch.randn(NUM_BLOCKS, BLOCK_SIZE, HKV, D)
    block_table = torch.zeros(3, 2, dtype=torch.int32)
    cu_seqlens_q = torch.tensor([0, 5, 12, 20], dtype=torch.int32)
    seq_lens_k = torch.tensor([10, 11, 12], dtype=torch.int32)
    qq_bias = torch.zeros(TOTAL_Q, TOTAL_Q) if with_bias else None
    return q, k_pool, v_pool, block_table, cu_seqlens_q, seq_lens_k, qq_bias


def _logical_inputs():
    logical_kv_slots = torch.arange(12, dtype=torch.int64).repeat(3, 1)
    logical_kv_starts = torch.zeros(3, dtype=torch.int32)
    logical_kv_lens = torch.tensor([10, 11, 12], dtype=torch.int32)
    return logical_kv_slots, logical_kv_starts, logical_kv_lens


def test_op_registered():
    """The op is registered under ``jetflow::paged_tree_attn`` with the typed schema
    (``Tensor?`` expresses the decode/logical-slot None cases directly)."""
    assert hasattr(torch.ops.jetflow, "paged_tree_attn")
    schema = str(paged_tree_attn._opoverload._schema)
    assert "jetflow::paged_tree_attn" in schema
    assert "Tensor? qq_bias" in schema  # Optional bias, no bool flag / op split
    assert "Tensor? logical_kv_slots=None" in schema
    assert "Tensor? logical_kv_starts=None" in schema
    assert "Tensor? logical_kv_lens=None" in schema


@pytest.mark.parametrize("with_bias", [True, False])
def test_fake_kernel_shape_dtype(with_bias):
    """The fake (meta) kernel returns ``(total_q, Hq, D)`` matching q, with no
    triton — so fake-tensor tracing has a correct shape/dtype to propagate."""
    q, k_pool, v_pool, block_table, cu, seq_lens_k, qq_bias = _inputs(with_bias)
    qm = q.to("meta")
    bias_m = qq_bias.to("meta") if qq_bias is not None else None
    out = torch.ops.jetflow.paged_tree_attn(
        qm, k_pool.to("meta"), v_pool.to("meta"), block_table.to("meta"),
        cu.to("meta"), seq_lens_k.to("meta"), bias_m, SCALE, NQPKV, BLOCK_SIZE,
    )
    assert out.shape == (TOTAL_Q, HQ, D)
    assert out.dtype == q.dtype
    assert out.device.type == "meta"


@pytest.mark.parametrize("with_bias", [True, False])
def test_fake_kernel_accepts_explicit_logical_kv_args(with_bias):
    """The appended logical-KV args are optional tensor args: explicit tensors
    propagate through the fake impl, while legacy omitted args remain valid."""
    q, k_pool, v_pool, block_table, cu, seq_lens_k, qq_bias = _inputs(with_bias)
    logical_kv_slots = torch.arange(24, dtype=torch.int64).reshape(3, 8)
    logical_kv_starts = torch.tensor([1, 2, 3], dtype=torch.int32)
    logical_kv_lens = torch.tensor([4, 5, 6], dtype=torch.int32)
    bias_m = qq_bias.to("meta") if qq_bias is not None else None

    out = torch.ops.jetflow.paged_tree_attn(
        q.to("meta"), k_pool.to("meta"), v_pool.to("meta"), block_table.to("meta"),
        cu.to("meta"), seq_lens_k.to("meta"), bias_m, SCALE, NQPKV, BLOCK_SIZE,
        logical_kv_slots.to("meta"), logical_kv_starts.to("meta"),
        logical_kv_lens.to("meta"),
    )

    assert out.shape == (TOTAL_Q, HQ, D)
    assert out.dtype == q.dtype
    assert out.device.type == "meta"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="opcheck runs the real triton op")
@pytest.mark.parametrize("with_bias", [True, False])
@pytest.mark.parametrize("with_logical", [False, True])
def test_opcheck_full(with_bias, with_logical):
    """Full ``torch.library.opcheck`` (schema, fake-tensor, aot-dispatch) on CUDA.

    Every opcheck utility *executes* the real op, which forwards to triton, so
    this is CUDA-only — it is the b200 kernel gate's schema+fake check. On a
    CPU/no-triton host the fake-only checks above stand in (shape/dtype +
    fullgraph traceability), since the real op cannot run there at all."""
    on_cuda = [
        t.cuda() if isinstance(t, torch.Tensor) else t for t in _inputs(with_bias)
    ]
    args = (*on_cuda, SCALE, NQPKV, BLOCK_SIZE)
    if with_logical:
        logical_cuda = tuple(t.cuda() for t in _logical_inputs())
        args = (*args, *logical_cuda)
    torch.library.opcheck(paged_tree_attn, args)


@pytest.mark.parametrize("with_bias", [True, False])
def test_fullgraph_traces_past_op(with_bias):
    """``torch.compile(fullgraph=True)`` traces the op with NO graph break (it is
    an opaque fusion boundary). Run on meta tensors so no triton fires; a graph
    break under fullgraph would raise, so reaching the shape assert is the proof."""
    def fn(q, k_pool, v_pool, block_table, cu, seq_lens_k, qq_bias):
        out = torch.ops.jetflow.paged_tree_attn(
            q, k_pool, v_pool, block_table, cu, seq_lens_k,
            qq_bias, SCALE, NQPKV, BLOCK_SIZE,
        )
        return out * 2.0  # a fusible consumer around the opaque boundary

    compiled = torch.compile(fn, fullgraph=True)
    q, k_pool, v_pool, block_table, cu, seq_lens_k, qq_bias = _inputs(with_bias)
    bias_m = qq_bias.to("meta") if qq_bias is not None else None
    out = compiled(
        q.to("meta"), k_pool.to("meta"), v_pool.to("meta"), block_table.to("meta"),
        cu.to("meta"), seq_lens_k.to("meta"), bias_m,
    )
    assert out.shape == (TOTAL_Q, HQ, D)
    assert out.dtype == q.dtype


@pytest.mark.parametrize("with_bias", [True, False])
def test_fullgraph_traces_with_logical_kv_args(with_bias):
    """The expanded schema stays traceable when the logical-slot tensors are
    present, which is what the later engine/cudagraph integration will call."""
    def fn(
        q,
        k_pool,
        v_pool,
        block_table,
        cu,
        seq_lens_k,
        qq_bias,
        logical_kv_slots,
        logical_kv_starts,
        logical_kv_lens,
    ):
        out = torch.ops.jetflow.paged_tree_attn(
            q, k_pool, v_pool, block_table, cu, seq_lens_k,
            qq_bias, SCALE, NQPKV, BLOCK_SIZE,
            logical_kv_slots, logical_kv_starts, logical_kv_lens,
        )
        return out * 2.0

    compiled = torch.compile(fn, fullgraph=True)
    q, k_pool, v_pool, block_table, cu, seq_lens_k, qq_bias = _inputs(with_bias)
    logical_kv_slots = torch.arange(24, dtype=torch.int64).reshape(3, 8)
    logical_kv_starts = torch.tensor([1, 2, 3], dtype=torch.int32)
    logical_kv_lens = torch.tensor([4, 5, 6], dtype=torch.int32)
    bias_m = qq_bias.to("meta") if qq_bias is not None else None
    out = compiled(
        q.to("meta"), k_pool.to("meta"), v_pool.to("meta"), block_table.to("meta"),
        cu.to("meta"), seq_lens_k.to("meta"), bias_m,
        logical_kv_slots.to("meta"), logical_kv_starts.to("meta"),
        logical_kv_lens.to("meta"),
    )
    assert out.shape == (TOTAL_Q, HQ, D)
    assert out.dtype == q.dtype


def test_compiled_verify_stack_traces_legacy_and_logical_kv_kwargs_on_meta():
    """The stack-level call stays fullgraph-traceable with omitted logical args and
    with per-layer logical-slot rows threaded down to the custom op."""
    from transformers import Qwen3Config, Qwen3ForCausalLM

    from jetflow.inference_engine.compiled_verify_stack import CompiledVerifyStack

    torch._dynamo.reset()
    cfg = Qwen3Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=32,
        tie_word_embeddings=False,
    )
    model = Qwen3ForCausalLM(cfg).eval().to(device="meta", dtype=torch.float32)
    stack = CompiledVerifyStack(model, block_size=4)
    n = 3
    args = dict(
        input_ids=torch.empty((1, n), dtype=torch.long, device="meta"),
        cos=torch.empty((1, n, 8), dtype=torch.float32, device="meta"),
        sin=torch.empty((1, n, 8), dtype=torch.float32, device="meta"),
        k_pools=[torch.empty((6, 4, 1, 8), dtype=torch.float32, device="meta")],
        v_pools=[torch.empty((6, 4, 1, 8), dtype=torch.float32, device="meta")],
        block_tables=[torch.empty((1, 5), dtype=torch.int32, device="meta")],
        cu=torch.empty((2,), dtype=torch.int32, device="meta"),
        seq_lens_k=torch.empty((1,), dtype=torch.int32, device="meta"),
        qq_bias=None,
        node_blks=[torch.empty((n,), dtype=torch.long, device="meta")],
        node_offs=[torch.empty((n,), dtype=torch.long, device="meta")],
    )

    with torch.no_grad():
        legacy = stack(**args)
        logical = stack(
            **args,
            logical_kv_slots=[torch.empty((1, 8), dtype=torch.int64, device="meta")],
            logical_kv_starts=torch.empty((1,), dtype=torch.int32, device="meta"),
            logical_kv_lens=torch.empty((1,), dtype=torch.int32, device="meta"),
        )

    assert legacy.shape == (1, n, cfg.vocab_size)
    assert logical.shape == (1, n, cfg.vocab_size)
