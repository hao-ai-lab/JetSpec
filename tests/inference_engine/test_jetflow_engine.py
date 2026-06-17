"""JetFlow N0 gate: a paged KV cache + single-stream AR engine that is
token-identical to `jetflow.core` on a tiny fp32 Qwen3.

Runs on CPU with a tiny randomly-initialized fp32 model (no network, no GPU): in
fp32 the paged store and HF's `DynamicCache` are bitwise-equal (the gather/append
is a plain copy, no rounding), so this gates the block arithmetic directly.
Mirrors `tests/core/test_tree_kv_cache.py`'s `_tiny_llm` harness. (On b200 in bf16 a
block forward vs a single-token forward can flip a borderline argmax after ~tens
of exact tokens — the same class as the existing bf16 borderline-argmax caveat; validated on b200.)
"""
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from jetflow.core.llm import LLM, SamplingParams
from jetflow.core.model_runner import ModelRunner
from jetflow.inference_engine.engine import JetFlowEngine
from jetflow.inference_engine.paged_kv_cache import PagedKVCache


class _StubTokenizer:
    """Only `.decode` is exercised when prompts are passed as input_ids tensors."""

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(str(int(t)) for t in ids)


def _tiny_model(seed: int = 0) -> Qwen3ForCausalLM:
    """A tiny fp32 Qwen3 (head_dim=16 == default block_size; no network)."""
    torch.manual_seed(seed)
    cfg = Qwen3Config(
        vocab_size=128, hidden_size=64, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        head_dim=16, max_position_embeddings=256, tie_word_embeddings=False,
    )
    return Qwen3ForCausalLM(cfg).eval().to(torch.float32)


def _tiny_llm(model) -> LLM:
    """Wire a model into an `LLM` without touching the network."""
    llm = object.__new__(LLM)
    llm.model = model
    llm.tokenizer = _StubTokenizer()
    llm.runner = ModelRunner(model)
    llm.device = "cpu"
    llm.eos_token_ids = set()            # no EOS -> deterministic length
    return llm


def _tiny_jetflow(model, block_size: int = 16) -> JetFlowEngine:
    """Wire the same model into a `JetFlowEngine` without touching the network."""
    eng = object.__new__(JetFlowEngine)
    eng.model = model
    eng.tokenizer = _StubTokenizer()
    eng.runner = ModelRunner(model)
    eng.device = "cpu"
    eng.dtype = torch.float32
    eng.block_size = block_size
    eng.eos_token_ids = set()
    return eng


PROMPT = torch.tensor([[3, 14, 15, 92, 65, 35, 89, 7]])   # arbitrary fixed input_ids
SP = SamplingParams(0.0, 24)


# --- PagedKVCache unit test -------------------------------------------------

def test_paged_cache_append_gather_matches_dense():
    """Append KV across block boundaries, unpack the logical view, and gather a
    scattered keep set — all bitwise-equal to a plain contiguous reference."""
    torch.manual_seed(0)
    B, H, S, D = 1, 2, 21, 16            # 21 tokens, block_size=8 -> 3 blocks (8/8/5)
    keys = torch.randn(B, H, S, D)
    values = torch.randn(B, H, S, D)
    cache = PagedKVCache(block_size=8, dtype=torch.float32)

    # Append in two chunks to exercise the partial-last-block top-up path.
    k_logical, v_logical = cache.update(keys[:, :, :13], values[:, :, :13], layer_idx=0)
    k_logical, v_logical = cache.update(keys[:, :, 13:], values[:, :, 13:], layer_idx=0)
    assert cache.get_seq_length(0) == S
    assert cache.block_table[0] == [0, 1, 2]
    assert torch.equal(k_logical, keys) and torch.equal(v_logical, values)

    # Scattered gather (tree-like accepted path) -> compacted linear prefix.
    keep = torch.tensor([0, 1, 8, 16, 17])
    cache.gather(keep)
    assert cache.get_seq_length(0) == keep.numel()
    gk, gv = cache._logical_kv(0)
    assert torch.equal(gk, keys[:, :, keep]) and torch.equal(gv, values[:, :, keep])


def test_paged_cache_multi_layer_isolation_and_free():
    """Two layers keep independent pools/tables; crop frees the dropped blocks."""
    torch.manual_seed(1)
    k0, v0 = torch.randn(1, 2, 10, 16), torch.randn(1, 2, 10, 16)
    k1, v1 = torch.randn(1, 2, 5, 16), torch.randn(1, 2, 5, 16)
    cache = PagedKVCache(block_size=4, dtype=torch.float32)
    cache.update(k0, v0, layer_idx=0)
    cache.update(k1, v1, layer_idx=1)
    assert cache.get_seq_length(0) == 10 and cache.get_seq_length(1) == 5
    assert cache.block_table[0] != cache.block_table[1]      # disjoint blocks

    free_before = len(cache._free_blocks)
    cache.crop(4)                                            # layer1 (5 tok) drops 1 block
    assert cache.get_seq_length(0) == 4 and cache.get_seq_length(1) == 4
    assert len(cache._free_blocks) > free_before             # blocks returned to pool
    gk0, _ = cache._logical_kv(0)
    assert torch.equal(gk0, k0[:, :, :4])                    # surviving KV intact


# --- JetFlowEngine lossless gate (token-identical to jetflow.core) ----------------

def test_jetflow_ar_matches_llm_greedy_fp32():
    """JetFlowEngine greedy == LLM greedy, token-for-token, across seeds and block
    sizes (fp32 bitwise-equal). Block sizes that don't divide head_dim exercise
    the cross-boundary append/unpack arithmetic."""
    for seed in (0, 1, 7):
        model = _tiny_model(seed)
        ref = _tiny_llm(model).generate(PROMPT, SP)["token_ids"]
        for block_size in (16, 4, 5):
            got = _tiny_jetflow(model, block_size).generate(PROMPT, SP)["token_ids"]
            assert got == ref, (
                f"JetFlow AR diverged from jetflow.core (seed={seed}, block_size={block_size})"
            )
        assert len(ref) == SP.max_new_tokens


def test_jetflow_ar_cache_reuse_grows_by_one():
    """Decode reuses the prefix: the paged cache length grows by exactly one token
    per step (no recompute)."""
    model = _tiny_model(0)
    eng = _tiny_jetflow(model)
    cache = PagedKVCache(block_size=eng.block_size, dtype=torch.float32)
    pos = torch.arange(PROMPT.shape[1]).unsqueeze(0)
    logits, cache, _ = eng.runner.forward(PROMPT, cache, pos)
    assert cache.get_seq_length(0) == PROMPT.shape[1]
    from jetflow.core.sampler import sample
    tok = sample(logits[:, -1:, :], 0.0)
    for step in range(5):
        cur = PROMPT.shape[1] + step
        logits, cache, _ = eng.runner.forward(tok, cache, torch.tensor([[cur]]))
        assert cache.get_seq_length(0) == cur + 1
        tok = sample(logits[:, -1:, :], 0.0)


def test_jetflow_ar_temperature_sampling_deterministic():
    """temperature>0 is deterministic under a fixed seed and matches LLM's
    sampler-driven path (same RNG draws via the shared `sample`)."""
    model = _tiny_model(2)
    sp = SamplingParams(0.7, 16)
    torch.manual_seed(123)
    ref = _tiny_llm(model).generate(PROMPT, sp)["token_ids"]
    torch.manual_seed(123)
    got = _tiny_jetflow(model).generate(PROMPT, sp)["token_ids"]
    assert got == ref
