"""CPU gate for the paged tree-attention ``custom_op`` (nano_vllm N3, foundation).

The CPU-runnable checks need NO triton/CUDA: the op is registered, the fake
(meta) kernel returns the right shape/dtype, and ``torch.compile(fullgraph=True)``
traces *past* the op as an opaque fusion boundary. Full ``opcheck`` is CUDA-only
(every opcheck utility executes the real op, which forwards to triton) and runs
as part of the b200 kernel gate alongside the kernel==SDPA oracle in
``test_nano_kernel.py``.

The op exists so a future compiled read-only decoder forward fuses the GEMMs
around attention without graph-breaking on the ``@triton.jit`` launch.
"""
import pytest
import torch

from ptd.nano_vllm.paged_tree_attn_op import paged_tree_attn


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


def test_op_registered():
    """The op is registered under ``ptd::paged_tree_attn`` with the typed schema
    (``Tensor? qq_bias`` expresses the decode/None case directly)."""
    assert hasattr(torch.ops.ptd, "paged_tree_attn")
    schema = str(paged_tree_attn._opoverload._schema)
    assert "ptd::paged_tree_attn" in schema
    assert "Tensor? qq_bias" in schema  # Optional bias, no bool flag / op split


@pytest.mark.parametrize("with_bias", [True, False])
def test_fake_kernel_shape_dtype(with_bias):
    """The fake (meta) kernel returns ``(total_q, Hq, D)`` matching q, with no
    triton — so fake-tensor tracing has a correct shape/dtype to propagate."""
    q, k_pool, v_pool, block_table, cu, seq_lens_k, qq_bias = _inputs(with_bias)
    qm = q.to("meta")
    bias_m = qq_bias.to("meta") if qq_bias is not None else None
    out = torch.ops.ptd.paged_tree_attn(
        qm, k_pool.to("meta"), v_pool.to("meta"), block_table.to("meta"),
        cu.to("meta"), seq_lens_k.to("meta"), bias_m, SCALE, NQPKV, BLOCK_SIZE,
    )
    assert out.shape == (TOTAL_Q, HQ, D)
    assert out.dtype == q.dtype
    assert out.device.type == "meta"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="opcheck runs the real triton op")
@pytest.mark.parametrize("with_bias", [True, False])
def test_opcheck_full(with_bias):
    """Full ``torch.library.opcheck`` (schema, fake-tensor, aot-dispatch) on CUDA.

    Every opcheck utility *executes* the real op, which forwards to triton, so
    this is CUDA-only — it is the b200 kernel gate's schema+fake check. On a
    CPU/no-triton host the fake-only checks above stand in (shape/dtype +
    fullgraph traceability), since the real op cannot run there at all."""
    on_cuda = [
        t.cuda() if isinstance(t, torch.Tensor) else t for t in _inputs(with_bias)
    ]
    args = (*on_cuda, SCALE, NQPKV, BLOCK_SIZE)
    torch.library.opcheck(paged_tree_attn, args)


@pytest.mark.parametrize("with_bias", [True, False])
def test_fullgraph_traces_past_op(with_bias):
    """``torch.compile(fullgraph=True)`` traces the op with NO graph break (it is
    an opaque fusion boundary). Run on meta tensors so no triton fires; a graph
    break under fullgraph would raise, so reaching the shape assert is the proof."""
    def fn(q, k_pool, v_pool, block_table, cu, seq_lens_k, qq_bias):
        out = torch.ops.ptd.paged_tree_attn(
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
