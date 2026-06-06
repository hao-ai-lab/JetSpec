"""nano_vllm N1 gate: single-stream TREE-spec decode over the paged KV cache must
be token-identical to (a) plain greedy AR and (b) the `DynamicCache` reference
tree verify (`LLM.generate_tree(kv_cache_verify=True)`) — losslessness is
preserved by the `PagedKVCache.gather` that keeps only the accepted root-to-leaf
path's KV (the paged analogue of `_select_kv_cache`).

Runs on CPU with a tiny randomly-initialized fp32 Qwen3 (no network, no GPU): in
fp32 the paged store and HF's `DynamicCache` are bitwise-equal (gather/append is a
plain copy, no rounding), so this gates the gather / mask / cache_position
arithmetic directly. Mirrors `tests/test_nano_engine.py`'s `_tiny_nano` and
`tests/test_tree_kv_cache.py`'s fixtures. (On b200 in bf16 a block forward vs the
recompute path can flip a borderline argmax after ~tens of exact tokens — the same
class as the bf16 borderline-argmax caveat; validated separately on b200.)
"""
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from ptd.engine.llm import LLM, SamplingParams
from ptd.engine.model_runner import ModelRunner
from ptd.draft import RandomTreeDrafter, TargetEchoTreeDrafter
from ptd.nano_vllm.engine import NanoEngine


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
    """Wire a model into an `LLM` without touching the network (DynamicCache ref)."""
    llm = object.__new__(LLM)
    llm.model = model
    llm.tokenizer = _StubTokenizer()
    llm.runner = ModelRunner(model)
    llm.device = "cpu"
    llm.eos_token_ids = set()            # no EOS -> deterministic length
    return llm


def _tiny_nano(model, block_size: int = 16) -> NanoEngine:
    """Wire the same model into a `NanoEngine` without touching the network."""
    eng = object.__new__(NanoEngine)
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


def _greedy(eng):
    return eng.generate(PROMPT, SP)["token_ids"]


def _nano_tree(eng, drafter, *, seed=1, return_stats=False):
    # seed before each call so the random drafter builds identical trees across
    # runs (losslessness holds for any tree regardless).
    torch.manual_seed(seed)
    return eng.generate_tree(
        PROMPT, drafter, block_size=4, tree_width=2, budget=15,
        sampling_params=SP, return_stats=return_stats,
    )


def _ref_tree(llm, drafter, *, seed=1):
    # the DynamicCache reference path (LLM.generate_tree(kv_cache_verify=True)).
    torch.manual_seed(seed)
    return llm.generate_tree(
        PROMPT, drafter, block_size=4, tree_width=2, budget=15,
        kv_cache_verify=True, sampling_params=SP,
    )


def test_nano_tree_lossless_random():
    """Random drafter (accepts ~0/round) -> paged-cache tree == DynamicCache ref ==
    greedy. Exercises the gather's keep-root-only case every round."""
    model = _tiny_model(0)
    greedy = _greedy(_tiny_nano(model))
    nano = _nano_tree(_tiny_nano(model), RandomTreeDrafter(128))["token_ids"]
    ref = _ref_tree(_tiny_llm(model), RandomTreeDrafter(128))["token_ids"]
    n = min(len(greedy), len(nano))
    assert ref[:n] == greedy[:n], "DynamicCache tree diverged from greedy"
    assert nano[:n] == greedy[:n], "paged-cache tree diverged from greedy (gather bug)"
    assert nano == ref, "paged-cache tree != DynamicCache tree (not a drop-in)"


def test_nano_tree_lossless_echo():
    """Echo tree's top-1 path is the greedy chain -> full-depth accept, exercising
    the gather's deep non-contiguous keep set (acc > 0). Paged-cache tree must match
    both greedy and the DynamicCache reference, and accept multiple tokens/round."""
    model = _tiny_model(0)
    greedy = _greedy(_tiny_nano(model))
    nano = _nano_tree(_tiny_nano(model), TargetEchoTreeDrafter(model))
    ref = _ref_tree(_tiny_llm(model), TargetEchoTreeDrafter(model))
    n = min(len(greedy), len(nano["token_ids"]))
    assert ref["token_ids"][:n] == greedy[:n], "DynamicCache tree (echo) diverged from greedy"
    assert nano["token_ids"][:n] == greedy[:n], "paged-cache tree (echo) diverged from greedy"
    assert nano["token_ids"] == ref["token_ids"], "paged-cache tree (echo) != DynamicCache ref"
    assert nano["tpf"] >= 2.0, f"echo should accept multiple tokens/round, got tpf={nano['tpf']:.2f}"


def test_nano_tree_block_sizes_match_ref():
    """The paged engine stays lossless across cache block sizes that don't divide
    head_dim (cross-boundary gather/append), and across model seeds."""
    for seed in (0, 1, 7):
        model = _tiny_model(seed)
        ref = _ref_tree(_tiny_llm(model), RandomTreeDrafter(128))["token_ids"]
        for block_size in (16, 4, 5):
            nano = _nano_tree(_tiny_nano(model, block_size), RandomTreeDrafter(128))["token_ids"]
            assert nano == ref, (
                f"paged tree diverged from DynamicCache ref (seed={seed}, block_size={block_size})"
            )


def test_nano_tree_stats_shape():
    """return_stats exposes per-round accept lengths / tree sizes on the paged path,
    and every committed token after the first is accounted for."""
    model = _tiny_model(0)
    full = _nano_tree(_tiny_nano(model), RandomTreeDrafter(128), return_stats=True)
    assert len(full["accept_lengths"]) == full["rounds"]
    assert len(full["tree_sizes"]) == full["rounds"]
    assert all(a >= 1 for a in full["accept_lengths"])   # each round commits >= the correction
