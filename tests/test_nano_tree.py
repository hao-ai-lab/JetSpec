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
from ptd.tree import DraftTree
from ptd.tree._core.base import TreeAlgorithm
from ptd.tree._core.registry import register_tree_algo


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


class _DeepEchoTreeDrafter:
    def __init__(self, model, *, depth: int):
        self.model = model
        self.depth = int(depth)
        self.last_tokens: list[int] = []

    @torch.inference_mode()
    def propose_logits(self, context_ids, depth, target_hidden=None, **kwargs):
        from transformers import DynamicCache

        cache = DynamicCache()
        pos = torch.arange(context_ids.shape[1], device=context_ids.device).unsqueeze(0)
        logits = self.model(
            input_ids=context_ids,
            position_ids=pos,
            past_key_values=cache,
            use_cache=True,
        ).logits
        nxt = logits[:, -1:, :].argmax(-1)
        tokens = [int(nxt.item())]
        cur = context_ids.shape[1]
        for _ in range(self.depth - 1):
            p = torch.tensor([[cur]], device=context_ids.device)
            logits = self.model(
                input_ids=nxt,
                position_ids=p,
                past_key_values=cache,
                use_cache=True,
            ).logits
            nxt = logits[:, -1:, :].argmax(-1)
            tokens.append(int(nxt.item()))
            cur += 1
        self.last_tokens = tokens
        return torch.zeros(
            1,
            depth,
            self.model.config.vocab_size,
            dtype=torch.float32,
            device=context_ids.device,
        )


@register_tree_algo("_test_deep_chain")
class _DeepChainAlgorithm(TreeAlgorithm):
    def __init__(self, drafter: _DeepEchoTreeDrafter, *, extra_branches: int = 0):
        self.drafter = drafter
        self.extra_branches = int(extra_branches)

    def build(self, root_token, draft_logits, block_size, tree_width, budget, device, **kwargs):
        chain = [int(root_token), *self.drafter.last_tokens]
        tokens = list(chain)
        parents = [-1] + list(range(len(chain) - 1))
        depths = list(range(len(chain)))
        vocab_size = int(draft_logits.shape[-1])

        for i in range(self.extra_branches):
            parent = i % (len(chain) - 1)
            greedy_child = chain[parent + 1]
            tokens.append((greedy_child + 1 + i) % vocab_size)
            parents.append(parent)
            depths.append(depths[parent] + 1)

        assert len(tokens) <= budget
        return DraftTree(
            token_ids=torch.tensor(tokens, dtype=torch.long, device=device),
            parent_indices=torch.tensor(parents, dtype=torch.long, device=device),
            depth=torch.tensor(depths, dtype=torch.long, device=device),
            num_nodes=len(tokens),
        )


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


def test_nano_tree_lossless_deep_tree_beyond_block_depth():
    """A spliced tree can be much deeper than the drafter horizon.

    The verifier/accept/commit path must size by configured max tree depth and
    accept by the tree's actual depth, while bucket sizing remains node-count
    based. This tree is depth 20 with 33 nodes, so depth and N fall in different
    compiled buckets.
    """
    from ptd.nano_vllm.engine import _bucket_for_n

    model = _tiny_model(0)
    depth = 20
    extra_branches = 12
    budget = depth + 1 + extra_branches
    sp = SamplingParams(0.0, 44)
    drafter = _DeepEchoTreeDrafter(model, depth=depth)

    greedy = _tiny_nano(model).generate(PROMPT, sp)["token_ids"]
    out = _tiny_nano(model, block_size=4).generate_tree(
        PROMPT,
        drafter,
        block_size=4,
        tree_width=2,
        budget=budget,
        algo="_test_deep_chain",
        algo_kwargs={"drafter": drafter, "extra_branches": extra_branches},
        max_tree_depth=depth,
        sampling_params=sp,
        return_stats=True,
        tree_diag=True,
    )

    assert out["token_ids"] == greedy
    assert out["accept_lengths"][0] == depth + 1
    assert out["tree_sizes"][0] == budget
    assert len(out["tree_nodes_per_depth"]) == depth + 1
    assert _bucket_for_n(out["tree_sizes"][0]) == 64
    assert _bucket_for_n(depth) == 32


def _long_echo_tree(engine, model, *, tree_block_size: int, return_stats: bool = False):
    return engine.generate_tree(
        PROMPT,
        TargetEchoTreeDrafter(model),
        block_size=tree_block_size,
        tree_width=1,
        budget=tree_block_size,
        target_layer_ids=[0],
        sampling_params=SamplingParams(0.0, 80),
        return_stats=return_stats,
    )


def _first_unique_midblock_token(tokens, tree_block_size: int):
    seen = set()
    for idx, token in enumerate(tokens):
        if token in seen:
            continue
        seen.add(token)
        if idx == 0:
            continue
        round_idx = ((idx - 1) // tree_block_size) + 1
        offset = (idx - 1) % tree_block_size
        if round_idx >= 3 and 0 < offset < tree_block_size - 1:
            return token
    raise AssertionError(f"no unique mid-block EOS candidate for block_size={tree_block_size}")


def test_nano_tree_long_decode_eos_midblock_matches_ref():
    """Persistent commit buffers must stay token-identical to the DynamicCache ref.

    TargetEchoTreeDrafter commits multi-token blocks, so choosing EOS from inside
    a later block covers the commit-slice path, early-EOS break, and the
    need_hidden target_hidden slice-write path in one CPU regression.
    """
    for tree_block_size in (4, 16):
        model = _tiny_model(0)
        nano_full = _long_echo_tree(
            _tiny_nano(model), model, tree_block_size=tree_block_size, return_stats=True
        )
        ref_full = _long_echo_tree(
            _tiny_llm(model), model, tree_block_size=tree_block_size, return_stats=True
        )
        assert nano_full["token_ids"] == ref_full["token_ids"]
        assert nano_full["rounds"] >= 3

        eos_token = _first_unique_midblock_token(nano_full["token_ids"], tree_block_size)
        nano = _tiny_nano(model)
        ref = _tiny_llm(model)
        nano.eos_token_ids = {eos_token}
        ref.eos_token_ids = {eos_token}

        nano_eos = _long_echo_tree(
            nano, model, tree_block_size=tree_block_size, return_stats=True
        )
        ref_eos = _long_echo_tree(
            ref, model, tree_block_size=tree_block_size, return_stats=True
        )

        assert nano_eos["token_ids"] == ref_eos["token_ids"]
        assert nano_eos["token_ids"][-1] == eos_token
        eos_idx = len(nano_eos["token_ids"]) - 1
        eos_round = ((eos_idx - 1) // tree_block_size) + 1
        eos_offset = (eos_idx - 1) % tree_block_size
        assert eos_round >= 3
        assert 0 < eos_offset < tree_block_size - 1
