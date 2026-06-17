"""JetFlow N2a gate: continuous-batched AR over the shared multi-seq paged
cache must be token-identical to running single-stream `JetFlowEngine.generate` on
each prompt alone.

Runs on CPU with a tiny randomly-initialized fp32 Qwen3 (no network, no GPU): in
fp32 the pooled batched forward and the single-stream forward are bitwise-equal
(append/gather is a plain copy, and the pad-to-max mask makes padded KV a no-op
for attention), so this gates the batched mask / position / per-seq KV-routing
arithmetic directly. The headline property — decoding N prompts of DIFFERENT
lengths together yields the SAME tokens as decoding each alone — is checked
per-sequence. Mirrors `tests/inference_engine/test_jetflow_engine.py`'s `_tiny_jetflow` harness.
(On b200 in bf16 a batched forward vs a single-token forward can flip a borderline
argmax after ~tens of exact tokens — the same class as the bf16 borderline-argmax caveat.)
"""
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from jetflow.core.llm import SamplingParams
from jetflow.core.model_runner import ModelRunner
from jetflow.inference_engine.engine import JetFlowEngine
from jetflow.inference_engine.scheduler import Scheduler, SequenceRequest
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


def _tiny_jetflow(model, block_size: int = 16) -> JetFlowEngine:
    """Wire the same model into a `JetFlowEngine` without touching the network."""
    eng = object.__new__(JetFlowEngine)
    eng.model = model
    eng.tokenizer = _StubTokenizer()
    eng.runner = ModelRunner(model)
    eng.device = "cpu"
    eng.dtype = torch.float32
    eng.block_size = block_size
    eng.eos_token_ids = set()            # no EOS -> deterministic length
    return eng


# Prompts of DIFFERENT lengths (8 / 5 / 12 / 3) — the batch must align positions,
# masks, and per-seq KV across the ragged set.
PROMPTS = [
    torch.tensor([[3, 14, 15, 92, 65, 35, 89, 7]]),
    torch.tensor([[10, 20, 30, 40, 50]]),
    torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]]),
    torch.tensor([[64, 32, 16]]),
]
SP = SamplingParams(0.0, 24)


# --- the N2a lossless gate ---------------------------------------------------

def test_batch_matches_single_stream_each_seq_fp32():
    """generate_batch over N ragged prompts == generate on each prompt alone,
    token-for-token, across seeds and block sizes (fp32 bitwise-equal). Block sizes
    that don't divide head_dim exercise the cross-boundary append/unpack arithmetic
    on the pooled per-seq KV."""
    for seed in (0, 1, 7):
        model = _tiny_model(seed)
        ref = [_tiny_jetflow(model).generate(p, SP)["token_ids"] for p in PROMPTS]
        for block_size in (16, 4, 5):
            batched = _tiny_jetflow(model, block_size).generate_batch(PROMPTS, SP)
            for i, p in enumerate(PROMPTS):
                assert batched[i]["token_ids"] == ref[i], (
                    f"batched seq {i} diverged from single-stream "
                    f"(seed={seed}, block_size={block_size})"
                )
        assert all(len(r) == SP.max_new_tokens for r in ref)


def test_batch_order_invariant():
    """The pool is keyed by seq_id, not batch position, so permuting the batch must
    not change any sequence's tokens (no cross-talk between sequences)."""
    model = _tiny_model(3)
    forward = _tiny_jetflow(model).generate_batch(PROMPTS, SP)
    reversed_prompts = list(reversed(PROMPTS))
    backward = _tiny_jetflow(model).generate_batch(reversed_prompts, SP)
    for i in range(len(PROMPTS)):
        assert backward[len(PROMPTS) - 1 - i]["token_ids"] == forward[i]["token_ids"], (
            f"seq {i} tokens changed when the batch order was reversed"
        )


def test_batch_singleton_matches_generate():
    """A batch of one is identical to plain single-stream generate (degenerate
    N2a == N0)."""
    model = _tiny_model(1)
    ref = _tiny_jetflow(model).generate(PROMPTS[0], SP)["token_ids"]
    got = _tiny_jetflow(model).generate_batch([PROMPTS[0]], SP)[0]["token_ids"]
    assert got == ref


def test_batch_ragged_finish_keeps_survivors_lossless():
    """When sequences finish at different steps (a short token budget vs a long
    one), the survivors keep decoding losslessly — dropping a finished seq from the
    batch must not perturb the rest. seq 1 finishes early; seq 0 keeps going and
    must still match its single-stream run."""
    model = _tiny_model(0)
    eng = _tiny_jetflow(model)
    eng.eos_token_ids = set()
    sp_long = SamplingParams(0.0, 20)
    ref0 = eng.generate(PROMPTS[0], sp_long)["token_ids"]

    # Give seq 1 a tiny budget by truncating its reference; the batch uses the
    # shared SP, so instead assert via the EOS path: inject an EOS so seq 1 stops.
    eos_tok = ref0[3] if len(ref0) > 3 else 0
    eng_eos = _tiny_jetflow(model)
    eng_eos.eos_token_ids = {eos_tok}
    ref0_eos = eng_eos.generate(PROMPTS[0], sp_long)["token_ids"]
    ref2_eos = eng_eos.generate(PROMPTS[2], sp_long)["token_ids"]

    batched = eng_eos.generate_batch([PROMPTS[0], PROMPTS[2]], sp_long)
    assert batched[0]["token_ids"] == ref0_eos, "seq 0 diverged under ragged EOS finish"
    assert batched[1]["token_ids"] == ref2_eos, "seq 1 diverged under ragged EOS finish"


# --- scheduler unit tests ----------------------------------------------------

def test_scheduler_admits_fcfs_into_pool():
    """admit_request queues FCFS; step() drains the queue into the pool and reports
    the running batch + telemetry."""
    cache = PagedKVCache(block_size=4, max_batch_size=8, max_total_tokens=4096,
                         dtype=torch.float32)
    sched = Scheduler(cache, max_batch_size=8)
    for sid, plen in [(0, 8), (1, 5), (2, 12)]:
        assert sched.admit_request(SequenceRequest(seq_id=sid, input_ids=list(range(plen))))
    out = sched.step()
    assert len(out.admitted) == 3
    assert out.batch_size == 3
    assert set(sched.running) == {0, 1, 2}
    assert out.prefill_tokens == 8 + 5 + 12
    assert out.admission_reason == "prefill"


def test_scheduler_mark_decode_and_finish():
    """mark_decode_step advances a request to decode mode; mark_finished moves it to
    finished and frees its blocks."""
    cache = PagedKVCache(block_size=4, max_batch_size=8, max_total_tokens=4096,
                         dtype=torch.float32)
    sched = Scheduler(cache, max_batch_size=8)
    sched.admit_request(SequenceRequest(seq_id=0, input_ids=[1, 2, 3, 4]))
    sched.step()
    sched.mark_decode_step(0, next_token=99)
    req = sched.running[0]
    assert req.is_prefill is False
    assert req.output_tokens == [99]
    assert req.input_ids == [99]
    sched.mark_finished(0)
    assert 0 not in sched.running and 0 in sched.finished


def test_scheduler_evicts_to_admit_when_pool_full():
    """When the fixed pool can't fit a new prefill, step() evicts the LRU running
    sequence (checkpointing it) to make room."""
    # 4 blocks (16 tokens). Two 8-token seqs fill it.
    cache = PagedKVCache(block_size=4, max_batch_size=8, max_total_tokens=16,
                         dtype=torch.float32)
    sched = Scheduler(cache, max_batch_size=8)
    k = torch.randn(1, 2, 8, 16)
    v = torch.randn(1, 2, 8, 16)
    for sid in (0, 1):
        sched.admit_request(SequenceRequest(seq_id=sid, input_ids=list(range(8))))
        sched.step()
        cache.append(k, v, layer_idx=0, seq_id=sid)   # simulate the prefill write
        sched.mark_decode_step(sid, next_token=7)
    assert cache.num_free_blocks == 0

    # A third 8-token prompt forces eviction of the LRU running seq (seq 0).
    sched.admit_request(SequenceRequest(seq_id=2, input_ids=list(range(8))))
    out = sched.step()
    assert 2 in [r.seq_id for r in out.admitted]
    assert out.evicted, "expected an eviction when the pool was full"
    assert out.evicted[0].seq_id in (0, 1)
    assert out.evicted[0].seq_id not in sched.running
    assert out.evicted[0].seq_id in sched.evicted
