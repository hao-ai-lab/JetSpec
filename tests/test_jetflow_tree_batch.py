"""JetFlow N2b gate: batched per-sequence TREE-spec decode over the shared
multi-seq paged cache must be token-identical to running single-stream
`JetFlowEngine.generate_tree` on each prompt alone.

N2b is the tree-spec analogue of N2a (`generate_batch`) and the batched analogue
of N1 (`generate_tree`): each round every live sequence builds its own draft tree,
the trees (possibly different node counts) are padded and verified in ONE batched
forward under a padded per-seq 4D ancestor mask, and each sequence's accepted path
is taken / gathered independently. The headline property — tree-decoding N prompts
of DIFFERENT lengths together yields the SAME tokens as tree-decoding each alone —
is checked per-sequence.

Runs on CPU with a tiny randomly-initialized fp32 Qwen3 (no network, no GPU): in
fp32 the pooled batched verify and the single-stream verify are bitwise-equal
(append/gather is a plain copy, and the padded per-seq mask makes other seqs' KV /
padding a no-op for attention), so this gates the batched mask / RoPE-position /
per-seq KV-routing arithmetic directly. Mirrors `tests/test_jetflow_tree.py`'s and
`tests/test_jetflow_batch.py`'s `_tiny_jetflow` harness. (On b200 in bf16 a batched
verify vs a single-stream verify can flip a borderline argmax after ~tens of exact
tokens — the same class as the bf16 borderline-argmax caveat.)
"""
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from ptd.engine.llm import SamplingParams
from ptd.engine.model_runner import ModelRunner
from ptd.draft import RandomTreeDrafter, TargetEchoTreeDrafter
from ptd.jetflow.engine import JetFlowEngine


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
# masks, and per-seq tree KV across the ragged set (different prefix lengths AND
# trees of possibly different node counts).
PROMPTS = [
    torch.tensor([[3, 14, 15, 92, 65, 35, 89, 7]]),
    torch.tensor([[10, 20, 30, 40, 50]]),
    torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]]),
    torch.tensor([[64, 32, 16]]),
]
SP = SamplingParams(0.0, 24)


def _single_tree(eng, prompt, drafter, *, seed=1, block_size=4):
    # seed before each call so the random drafter builds an identical tree to the
    # batched run for the SAME prompt (losslessness holds for any tree regardless;
    # we pin the seed so the single-stream and batched trees coincide).
    torch.manual_seed(seed)
    return eng.generate_tree(
        prompt, drafter, block_size=block_size, tree_width=2, budget=15,
        sampling_params=SP,
    )["token_ids"]


def _batch_tree(eng, prompts, drafter, *, seed=1, block_size=4):
    torch.manual_seed(seed)
    return eng.generate_tree_batch(
        prompts, drafter, block_size=block_size, tree_width=2, budget=15,
        sampling_params=SP,
    )


# --- the N2b lossless gate ---------------------------------------------------

def test_tree_batch_matches_single_stream_random():
    """generate_tree_batch over N ragged prompts == generate_tree on each prompt
    alone, token-for-token (RandomTreeDrafter: accepts ~0/round, exercising the
    keep-root-only per-seq append every round). The random drafter is seeded once
    per generate_tree(_batch) call, and the batched loop builds each seq's tree in
    active order on the SAME seeded RNG as the per-prompt single-stream run."""
    for seed in (0, 1, 7):
        model = _tiny_model(seed)
        drafter = RandomTreeDrafter(128)
        # Single-stream references: build each prompt's tree on its own seeded RNG.
        ref = [_single_tree(_tiny_jetflow(model), p, drafter, seed=100 + seed)
               for p in PROMPTS]
        batched = _batch_tree(_tiny_jetflow(model), PROMPTS, drafter, seed=100 + seed)
        for i in range(len(PROMPTS)):
            assert batched[i]["token_ids"] == ref[i], (
                f"batched tree seq {i} diverged from single-stream (seed={seed})"
            )


def test_tree_batch_matches_single_stream_echo():
    """Echo drafter: the crossproduct top-1 path IS the greedy chain, so every seq
    accepts the full depth each round (multi-token accept). The batched per-seq KV
    append must keep the deep accepted path, matching the single-stream verify
    exactly AND accepting multiple tokens/round."""
    model = _tiny_model(0)
    drafter = TargetEchoTreeDrafter(model)
    ref = [_single_tree(_tiny_jetflow(model), p, drafter) for p in PROMPTS]
    batched = _batch_tree(_tiny_jetflow(model), PROMPTS, drafter)
    for i in range(len(PROMPTS)):
        assert batched[i]["token_ids"] == ref[i], (
            f"batched tree (echo) seq {i} diverged from single-stream"
        )
        assert batched[i]["tpf"] >= 2.0, (
            f"echo seq {i} should accept multiple tokens/round, got tpf={batched[i]['tpf']:.2f}"
        )


def test_tree_batch_block_sizes_match_single_stream():
    """Lossless across tree block sizes (depth) and across model seeds — the padded
    per-seq mask / RoPE arithmetic must hold for varying tree depths."""
    for seed in (0, 1):
        model = _tiny_model(seed)
        for block_size in (2, 4, 5):
            drafter = RandomTreeDrafter(128)
            ref = [_single_tree(_tiny_jetflow(model), p, drafter, seed=7, block_size=block_size)
                   for p in PROMPTS]
            batched = _batch_tree(_tiny_jetflow(model), PROMPTS, drafter, seed=7,
                                  block_size=block_size)
            for i in range(len(PROMPTS)):
                assert batched[i]["token_ids"] == ref[i], (
                    f"batched tree seq {i} diverged (seed={seed}, block_size={block_size})"
                )


def test_tree_batch_cache_block_sizes_match_single_stream():
    """Lossless across cache block sizes that don't divide head_dim (cross-boundary
    per-seq append/unpack on the pooled tree KV)."""
    model = _tiny_model(0)
    drafter = RandomTreeDrafter(128)
    for cache_block in (16, 4, 5):
        ref = [_single_tree(_tiny_jetflow(model, cache_block), p, drafter, seed=3)
               for p in PROMPTS]
        torch.manual_seed(3)
        batched = _tiny_jetflow(model, cache_block).generate_tree_batch(
            PROMPTS, drafter, block_size=4, tree_width=2, budget=15, sampling_params=SP)
        for i in range(len(PROMPTS)):
            assert batched[i]["token_ids"] == ref[i], (
                f"batched tree seq {i} diverged (cache_block={cache_block})"
            )


def test_tree_batch_order_invariant():
    """The pool is keyed by seq_id, not batch position, so permuting the batch must
    not change any sequence's tokens (no cross-talk between trees). Echo drafter is
    deterministic (no RNG), so reversing the prompt order is a clean permutation."""
    model = _tiny_model(3)
    drafter = TargetEchoTreeDrafter(model)
    forward = _tiny_jetflow(model).generate_tree_batch(
        PROMPTS, drafter, block_size=4, tree_width=2, budget=15, sampling_params=SP)
    reversed_prompts = list(reversed(PROMPTS))
    backward = _tiny_jetflow(model).generate_tree_batch(
        reversed_prompts, drafter, block_size=4, tree_width=2, budget=15, sampling_params=SP)
    for i in range(len(PROMPTS)):
        assert backward[len(PROMPTS) - 1 - i]["token_ids"] == forward[i]["token_ids"], (
            f"tree seq {i} tokens changed when the batch order was reversed"
        )


def test_tree_batch_singleton_matches_generate_tree():
    """A batch of one is identical to single-stream generate_tree (degenerate
    N2b == N1)."""
    model = _tiny_model(1)
    drafter = TargetEchoTreeDrafter(model)
    ref = _single_tree(_tiny_jetflow(model), PROMPTS[0], drafter)
    got = _batch_tree(_tiny_jetflow(model), [PROMPTS[0]], drafter)[0]["token_ids"]
    assert got == ref


def test_tree_batch_ragged_finish_keeps_survivors_lossless():
    """When sequences finish at different rounds (an EOS injected into one seq's
    stream), the survivors keep tree-decoding losslessly — dropping a finished seq
    from the batch must not perturb the rest. Echo drafter so the streams are
    deterministic; EOS taken from seq 0's own stream so it actually fires."""
    model = _tiny_model(0)
    drafter = TargetEchoTreeDrafter(model)
    # Find a token that appears in seq 0's no-EOS stream to use as EOS.
    plain = _tiny_jetflow(model)
    ref0_plain = plain.generate_tree(
        PROMPTS[0], drafter, block_size=4, tree_width=2, budget=15, sampling_params=SP)["token_ids"]
    eos_tok = ref0_plain[5] if len(ref0_plain) > 5 else ref0_plain[-1]

    eng_eos = _tiny_jetflow(model)
    eng_eos.eos_token_ids = {eos_tok}
    ref0 = eng_eos.generate_tree(
        PROMPTS[0], drafter, block_size=4, tree_width=2, budget=15, sampling_params=SP)["token_ids"]
    ref2 = eng_eos.generate_tree(
        PROMPTS[2], drafter, block_size=4, tree_width=2, budget=15, sampling_params=SP)["token_ids"]

    batched = eng_eos.generate_tree_batch(
        [PROMPTS[0], PROMPTS[2]], drafter, block_size=4, tree_width=2, budget=15,
        sampling_params=SP)
    assert batched[0]["token_ids"] == ref0, "seq 0 diverged under ragged EOS finish"
    assert batched[1]["token_ids"] == ref2, "seq 1 diverged under ragged EOS finish"
