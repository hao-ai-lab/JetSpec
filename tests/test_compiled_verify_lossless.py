"""nano_vllm A3-INT lossless gate: the compiled read-only tree-VERIFY stack, wired
into `NanoEngine` behind `attn_backend="triton_paged_tree_compiled"`, must produce
the SAME tokens as the default SDPA path for the N1 (`generate_tree`) cases the e2e
suite uses.

SDPA is the correctness oracle (same as `test_nano_kernel_e2e`): the compiled stack
reproduces the exact Qwen3 per-layer compute and reads the exact post-RoPE K/V the
oracle reads, so on a tiny fp32 Qwen3 on CUDA the two argmax streams are identical.
We reuse the e2e suite's tiny fp32 CUDA model + drafters and assert token-for-token
equality (fp32 must be EXACT) across the random- and echo-drafter N1 cases.

Needs CUDA (triton); skipped on a CPU-only host.

A3-HIDDEN: the tapped-hidden variant (`need_hidden=True`, the real DraftHead path)
is ALSO covered. With `target_layer_ids` set and `block_size > 1`, the compiled
stack returns `target_hidden` tapped from the post-layer residuals; the engine
feeds it to the drafter exactly like the eager kernel's `extract_context_feature`
output. Token-losslessness holds regardless of the tap (each verify row is
target-greedy), so this gate asserts token equality; the accept_len-preservation
gate (compiled vs eager kernel with the real DraftHead) lives on the b200 harness."""
import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="compiled verify stack needs CUDA (triton)"
)

# This suite builds many CompiledVerifyStack instances in ONE process (per backend,
# per tap set, per N), each a distinct `_stack` specialization. Production caches one
# stack per (need_hidden, target_layer_ids) and never trips this, but the test
# multiplicity exceeds dynamo's default recompile_limit (8); raise it so the suite
# compiles every variant instead of falling back. Test-only, no production effect.
torch._dynamo.config.recompile_limit = 64

from ptd.engine.llm import SamplingParams
from ptd.draft import RandomTreeDrafter, TargetEchoTreeDrafter
from tests.test_nano_kernel_e2e import _tiny_model, _tiny_nano, PROMPT, SP


def _add_compiled_backend():
    """Extend `_tiny_nano` to also wire the compiled backend (the e2e fixture only
    knows sdpa / triton_paged_tree). We mirror its bypass-`__init__` construction and
    bind a `CompiledVerifyStack` over the tiny model, then route via the same fixture
    by post-attaching `compiled_verify`. Returns a builder `(model) -> NanoEngine`."""
    from ptd.nano_vllm.compiled_verify_stack import CompiledVerifyStack

    def build(model):
        eng = _tiny_nano(model, "triton_paged_tree")     # registers interface + flips impl
        eng.attn_backend = "triton_paged_tree_compiled"
        eng.compiled_verify = CompiledVerifyStack(model, block_size=eng.block_size)
        eng.compiled_ar = CompiledVerifyStack(model, block_size=eng.block_size)
        eng._compiled_verify_hidden = {}
        return eng

    return build


# --- N0: generate (AR decode), compiled AR stack vs SDPA ---------------------

def test_n0_compiled_ar_matches_sdpa():
    """A3-HIDDEN routes the AR decode forward (N=1) through a compiled stack so
    decode_cuda_speedup is compiled-vs-compiled. That path must stay token-identical
    to plain SDPA AR (fp32 exact on the tiny model)."""
    model = _tiny_model(0)
    build_compiled = _add_compiled_backend()
    sdpa = _tiny_nano(model, "sdpa").generate(PROMPT, SP)["token_ids"]
    comp = build_compiled(model).generate(PROMPT, SP)["token_ids"]
    assert comp == sdpa, "compiled AR decode diverged from SDPA"


# --- N1: generate_tree (single-stream tree spec), compiled verify vs SDPA ----

@pytest.mark.parametrize("seed", [1, 7])
def test_n1_compiled_verify_matches_sdpa_random(seed):
    model = _tiny_model(0)
    drafter = RandomTreeDrafter(vocab_size=model.config.vocab_size)
    build_compiled = _add_compiled_backend()

    def run(e):
        torch.manual_seed(seed)            # identical trees across both runs
        return e.generate_tree(PROMPT, drafter, block_size=4, tree_width=2,
                               budget=15, sampling_params=SP)["token_ids"]

    sdpa = run(_tiny_nano(model, "sdpa"))
    comp = run(build_compiled(model))
    assert comp == sdpa, f"compiled verify diverged from SDPA (random drafter, seed={seed})"


def test_n1_compiled_verify_matches_sdpa_echo():
    """TargetEchoTreeDrafter -> full-depth accepts (the multi-node-accept path)."""
    model = _tiny_model(0)
    drafter = TargetEchoTreeDrafter(model)
    build_compiled = _add_compiled_backend()

    def run(e):
        torch.manual_seed(1)
        return e.generate_tree(PROMPT, drafter, block_size=4, tree_width=2,
                               budget=15, sampling_params=SP)["token_ids"]

    sdpa = run(_tiny_nano(model, "sdpa"))
    comp = run(build_compiled(model))
    assert comp == sdpa, "compiled verify diverged from SDPA (echo drafter)"


# --- A3-HIDDEN: need_hidden=True compiled verify (tapped-hidden DraftHead path) ---
# `target_layer_ids` set + block_size > 1 flips need_hidden on; the compiled stack
# returns target_hidden tapped from the post-layer residuals (matching
# extract_context_feature). The echo drafter ignores target_hidden, so trees stay
# deterministic and tokens must match SDPA regardless of the tap. The tiny model has
# 2 layers, so valid taps are layer 0 / 1 (hidden_states[L+1], L in {0,1}).

@pytest.mark.parametrize("target_layer_ids", [[0], [0, 1]])
def test_n1_compiled_verify_need_hidden_matches_sdpa(target_layer_ids):
    """need_hidden=True compiled verify (DraftHead/echo) token-identical to SDPA."""
    model = _tiny_model(0)
    drafter = TargetEchoTreeDrafter(model)
    build_compiled = _add_compiled_backend()

    def run(e):
        torch.manual_seed(1)
        return e.generate_tree(PROMPT, drafter, block_size=4, tree_width=2,
                               budget=15, target_layer_ids=target_layer_ids,
                               sampling_params=SP)["token_ids"]

    sdpa = run(_tiny_nano(model, "sdpa"))
    comp = run(build_compiled(model))
    assert comp == sdpa, (
        f"compiled need_hidden verify diverged from SDPA (target_layer_ids={target_layer_ids})"
    )


@pytest.mark.parametrize("target_layer_ids", [[0], [1], [0, 1]])
@torch.inference_mode()      # custom op has no autograd formula; mirror generate_tree
def test_compiled_verify_hidden_matches_eager_kernel(target_layer_ids):
    """The compiled need_hidden stack's target_hidden must match the EAGER KERNEL's
    `extract_context_feature(out.hidden_states, ids)` (the `new_hidden` the engine
    feeds the drafter on the fallback path) — same kernel substrate, so the only gap
    is fp32 fusion rounding. This is the unit-level form of the accept_len gate: a
    wrong tap (offset / last-layer-norm / concat order) is token-lossless but feeds
    the head the wrong context and drops accept_len. Includes the last-layer tap
    `[1]` (HF stores it post-final-norm) which a naive raw-residual tap gets wrong.

    We seed the pool by running the eager kernel verify first (its forward writes the
    prefix + node KV and returns `new_hidden`), then run the compiled stack over a
    FRESH pool seeded the same way, and compare the two taps over the same node rows.
    """
    from ptd.nano_vllm.compiled_verify_stack import CompiledVerifyStack
    from ptd.nano_vllm.paged_kv_cache import PagedKVCache

    model = _tiny_model(0)
    device = next(model.parameters()).device
    N, past_len = 5, 7
    torch.manual_seed(3)
    prefix = torch.randint(0, model.config.vocab_size, (1, past_len), device=device)
    seq_step = torch.randint(0, model.config.vocab_size, (1, N), device=device)
    posN = torch.arange(past_len, past_len + N, device=device).unsqueeze(0)
    nlayers = model.config.num_hidden_layers

    def _fresh_kernel_cache(eng):
        cache = PagedKVCache(block_size=eng.block_size, device=device, dtype=torch.float32)
        cache._paged_handoff = True
        cache._handoff_seq_ids = [0]
        cache._ptd_attn_meta = {"seq_ids": [0], "qq_bias": None}
        eng.runner.forward(prefix, cache, torch.arange(past_len, device=device).unsqueeze(0))
        return cache

    eng = _tiny_nano(model, "triton_paged_tree")

    # Eager-kernel oracle: the verify forward returns new_hidden (= the engine's tap).
    cache_e = _fresh_kernel_cache(eng)
    cache_e._handoff_seq_ids = [0]
    cache_e._ptd_attn_meta = {"seq_ids": [0], "qq_bias": None}
    _, _, eager_hidden = eng.runner.forward(
        seq_step, cache_e, posN, attention_mask=None,
        cache_position=torch.arange(past_len, past_len + N, device=device),
        output_hidden_states=True, target_layer_ids=target_layer_ids,
    )

    # Compiled stack over a fresh, identically-seeded pool.
    cache_c = _fresh_kernel_cache(eng)
    dummy = torch.zeros(1, N, model.config.hidden_size, device=device, dtype=torch.float32)
    cos, sin = model.model.rotary_emb(dummy, posN)
    cu = torch.tensor([0, N], device=device, dtype=torch.int32)
    bts, node_blks, node_offs, slk = cache_c.reserve_tree_slots(0, N, past_len)
    k_pools = [cache_c.pool(i)[0] for i in range(nlayers)]
    v_pools = [cache_c.pool(i)[1] for i in range(nlayers)]
    stack = CompiledVerifyStack(model, block_size=eng.block_size,
                                need_hidden=True, target_layer_ids=target_layer_ids)
    _, comp_hidden = stack(seq_step, cos, sin, k_pools, v_pools, bts, cu, slk,
                           None, node_blks, node_offs)

    assert comp_hidden.shape == eager_hidden.shape
    max_diff = (comp_hidden - eager_hidden).abs().max().item()
    # fp32 same-substrate: only fusion-order rounding (~1e-6), NOT the ~O(1) gap a
    # wrong tap produces (the last-layer raw-residual bug showed max|d| ~ 4).
    assert max_diff < 1e-4, (
        f"compiled target_hidden diverged from eager-kernel tap "
        f"(target_layer_ids={target_layer_ids}, max|diff|={max_diff}) — accept_len would drop"
    )
