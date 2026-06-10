"""JetFlow A3-INT lossless gate: the compiled read-only tree-VERIFY stack, wired
into `JetFlowEngine` behind `attn_backend="triton_paged_tree_compiled"`, must produce
the SAME tokens as the default SDPA path for the N1 (`generate_tree`) cases the e2e
suite uses.

SDPA is the correctness oracle (same as `test_jetflow_kernel_e2e`): the compiled stack
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
from tests.test_jetflow_kernel_e2e import _tiny_model, _tiny_jetflow, PROMPT, SP


def _add_compiled_backend(cudagraph: bool = False):
    """Extend `_tiny_jetflow` to also wire the compiled backend (the e2e fixture only
    knows sdpa / triton_paged_tree). We mirror its bypass-`__init__` construction and
    bind a `CompiledVerifyStack` over the tiny model, then route via the same fixture
    by post-attaching `compiled_verify`. Returns a builder `(model) -> JetFlowEngine`.

    `cudagraph=True` (A3-GRAPH) flips on the opt-in CUDA-graph layer: the same compiled
    stacks, but the per-round tree verify replays a per-bucket captured graph instead of
    calling the stack directly. The compiled-non-graph build (`cudagraph=False`) stays the
    untouched oracle the graph path is diffed against."""
    from ptd.jetflow.compiled_verify_stack import CompiledVerifyStack
    from ptd.jetflow.engine import _env_flag

    def build(model):
        eng = _tiny_jetflow(model, "triton_paged_tree")     # registers interface + flips impl
        eng.attn_backend = ("triton_paged_tree_cudagraph" if cudagraph
                            else "triton_paged_tree_compiled")
        eng.fuse_gemms = _env_flag("JETFLOW_FUSE_GEMMS")
        eng.compiled_verify = CompiledVerifyStack(
            model, block_size=eng.block_size, fuse_gemms=eng.fuse_gemms,
        )
        eng.compiled_ar = CompiledVerifyStack(
            model, block_size=eng.block_size, fuse_gemms=eng.fuse_gemms,
        )
        eng._compiled_verify_hidden = {}
        eng._use_cudagraph = cudagraph
        eng._graphed_verify = {}
        return eng

    return build


# --- N0: generate (AR decode), compiled AR stack vs SDPA ---------------------

def test_n0_compiled_ar_matches_sdpa():
    """A3-HIDDEN routes the AR decode forward (N=1) through a compiled stack so
    decode_cuda_speedup is compiled-vs-compiled. That path must stay token-identical
    to plain SDPA AR (fp32 exact on the tiny model)."""
    model = _tiny_model(0)
    build_compiled = _add_compiled_backend()
    sdpa = _tiny_jetflow(model, "sdpa").generate(PROMPT, SP)["token_ids"]
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

    sdpa = run(_tiny_jetflow(model, "sdpa"))
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

    sdpa = run(_tiny_jetflow(model, "sdpa"))
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

    sdpa = run(_tiny_jetflow(model, "sdpa"))
    comp = run(build_compiled(model))
    assert comp == sdpa, (
        f"compiled need_hidden verify diverged from SDPA (target_layer_ids={target_layer_ids})"
    )


# --- A3-BUCKET: tree-N bucketing must stay token-lossless ---------------------
# Bucketing pads the N real tree rows up to a fixed bucket B (pad rows get -inf
# qq_bias both ways, a dummy token, and are sliced off after). The committed tokens
# and the tapped hidden must be bit-identical to the unbucketed path, because
# tree_accept walks only real child indices 0..N-1 and real rows assign -inf score to
# pad keys. We FORCE a non-trivial pad on the tiny model (whose trees are tiny, N~15)
# by monkeypatching the bucket function, then assert the bucketed compiled decode is
# token-identical to SDPA (the lossless oracle). A broken pad (e.g. real rows attending
# pad keys, or a pad row leaking into accept) would flip tokens here.

import ptd.jetflow.engine as _eng_mod


def _force_bucket(monkeypatch, pad: int):
    """Make `_bucket_for_n(N)` return `N + pad` so every round pads by exactly `pad`
    rows — exercises the padding path on the tiny model's small trees."""
    monkeypatch.setattr(_eng_mod, "_bucket_for_n", lambda n: n + pad)


def test_bucket_for_n_math():
    """`_bucket_for_n` snaps UP to the next bucket; beyond the max it rounds up to a
    multiple of the largest bucket (bounded shape set, never per-N)."""
    assert _eng_mod._TREE_BUCKETS == (16, 32, 64, 128, 192, 256)
    assert _eng_mod._bucket_for_n(1) == 16
    assert _eng_mod._bucket_for_n(15) == 16
    assert _eng_mod._bucket_for_n(16) == 16
    assert _eng_mod._bucket_for_n(17) == 32
    assert _eng_mod._bucket_for_n(31) == 32
    assert _eng_mod._bucket_for_n(32) == 32
    assert _eng_mod._bucket_for_n(33) == 64
    assert _eng_mod._bucket_for_n(63) == 64
    assert _eng_mod._bucket_for_n(64) == 64
    assert _eng_mod._bucket_for_n(65) == 128
    assert _eng_mod._bucket_for_n(255) == 256
    assert _eng_mod._bucket_for_n(256) == 256
    assert _eng_mod._bucket_for_n(257) == 512


def test_pad_tree_to_bucket_structure():
    """`_pad_tree_to_bucket` keeps the real (N,N) qq_bias block intact and sets every
    real/pad interaction to -inf; pad rows have only a self edge to avoid all-masked
    softmax NaNs and are never accepted. B==N is a no-op."""
    eng = object.__new__(_eng_mod.JetFlowEngine)
    eng.device = "cpu"
    N, pad = 4, 3
    B = N + pad
    seq_step = torch.arange(1, N + 1).view(1, N)
    posN = torch.arange(10, 10 + N).view(1, N)
    qq = torch.where(torch.eye(N).bool(),
                     torch.zeros(()), torch.full((), float("-inf")))
    ss_b, pos_b, qq_b = eng._pad_tree_to_bucket(seq_step, posN, qq, N, B)
    assert ss_b.shape == (1, B) and pos_b.shape == (1, B) and qq_b.shape == (B, B)
    assert torch.equal(ss_b[:, :N], seq_step) and (ss_b[:, N:] == 0).all()
    assert torch.equal(qq_b[:N, :N], qq)                 # real block unchanged
    assert torch.isneginf(qq_b[:N, N:]).all()            # real rows never attend pad
    assert torch.isneginf(qq_b[N:, :N]).all()            # pad rows never attend real
    pad_block = qq_b[N:, N:]
    assert torch.isneginf(pad_block.masked_fill(torch.eye(pad).bool(), float("-inf"))).all()
    assert torch.equal(torch.diagonal(pad_block), torch.zeros(pad))
    # B == N: identity (no copy, returns inputs)
    assert eng._pad_tree_to_bucket(seq_step, posN, qq, N, N)[0] is seq_step


@pytest.mark.parametrize("pad", [1, 5])
def test_n1_compiled_verify_bucketed_matches_sdpa(monkeypatch, pad):
    """Bucketed (padded) compiled verify token-identical to SDPA (logits-only)."""
    model = _tiny_model(0)
    drafter = TargetEchoTreeDrafter(model)
    build_compiled = _add_compiled_backend()
    sdpa = (lambda e: (torch.manual_seed(1), e.generate_tree(
        PROMPT, drafter, block_size=4, tree_width=2, budget=15,
        sampling_params=SP)["token_ids"])[1])(_tiny_jetflow(model, "sdpa"))
    _force_bucket(monkeypatch, pad)
    torch.manual_seed(1)
    comp = build_compiled(model).generate_tree(
        PROMPT, drafter, block_size=4, tree_width=2, budget=15,
        sampling_params=SP)["token_ids"]
    assert comp == sdpa, f"bucketed compiled verify diverged from SDPA (pad={pad})"


@pytest.mark.parametrize("target_layer_ids", [[0], [0, 1]])
def test_n1_compiled_verify_bucketed_need_hidden_matches_sdpa(monkeypatch, target_layer_ids):
    """Bucketed need_hidden compiled verify (DraftHead tap) token-identical to SDPA —
    the pad rows must not perturb the tapped target_hidden fed to the drafter."""
    model = _tiny_model(0)
    drafter = TargetEchoTreeDrafter(model)
    build_compiled = _add_compiled_backend()

    def run(e):
        torch.manual_seed(1)
        return e.generate_tree(PROMPT, drafter, block_size=4, tree_width=2,
                               budget=15, target_layer_ids=target_layer_ids,
                               sampling_params=SP)["token_ids"]

    sdpa = run(_tiny_jetflow(model, "sdpa"))
    _force_bucket(monkeypatch, 4)
    comp = run(build_compiled(model))
    assert comp == sdpa, (
        f"bucketed need_hidden compiled verify diverged from SDPA "
        f"(target_layer_ids={target_layer_ids})"
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
    from ptd.jetflow.compiled_verify_stack import CompiledVerifyStack
    from ptd.jetflow.engine import _env_flag
    from ptd.jetflow.paged_kv_cache import PagedKVCache

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

    eng = _tiny_jetflow(model, "triton_paged_tree")

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
    stack = CompiledVerifyStack(
        model,
        block_size=eng.block_size,
        need_hidden=True,
        target_layer_ids=target_layer_ids,
        fuse_gemms=_env_flag("JETFLOW_FUSE_GEMMS"),
    )
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


# --- A3-GRAPH: CUDA-graph capture+replay must stay token-lossless -------------
# The opt-in "triton_paged_tree_cudagraph" backend captures one CUDA graph per tree-N
# bucket around the compiled verify stack and replays it per round. Replay reruns the
# IDENTICAL fp32 forward over staged inputs + the live pool (incl. the in-graph node-KV
# scatter), so the committed tokens must be bit-identical to BOTH the SDPA oracle AND the
# compiled-non-graph path (the latter is the graph's own pre-capture behavior). A capture
# bug (stale-address read, un-replayed scatter, pad leak) would flip tokens here.


@pytest.mark.parametrize("seed", [1, 7])
def test_n1_cudagraph_verify_matches_sdpa_and_compiled_random(seed):
    """Graph-replay verify (logits-only) token-identical to SDPA and to compiled."""
    model = _tiny_model(0)
    drafter = RandomTreeDrafter(vocab_size=model.config.vocab_size)
    build_compiled = _add_compiled_backend(cudagraph=False)
    build_graph = _add_compiled_backend(cudagraph=True)

    def run(e):
        torch.manual_seed(seed)            # identical trees across builds
        return e.generate_tree(PROMPT, drafter, block_size=4, tree_width=2,
                               budget=15, sampling_params=SP)["token_ids"]

    sdpa = run(_tiny_jetflow(model, "sdpa"))
    comp = run(build_compiled(model))
    graph = run(build_graph(model))
    assert comp == sdpa, f"compiled diverged from SDPA (seed={seed})"
    assert graph == sdpa, f"cudagraph verify diverged from SDPA (seed={seed})"
    assert graph == comp, f"cudagraph verify diverged from compiled-non-graph (seed={seed})"


def test_n1_cudagraph_verify_matches_sdpa_echo():
    """Echo drafter (multi-node accept) — graph replay token-identical to SDPA."""
    model = _tiny_model(0)
    drafter = TargetEchoTreeDrafter(model)
    build_graph = _add_compiled_backend(cudagraph=True)

    def run(e):
        torch.manual_seed(1)
        return e.generate_tree(PROMPT, drafter, block_size=4, tree_width=2,
                               budget=15, sampling_params=SP)["token_ids"]

    sdpa = run(_tiny_jetflow(model, "sdpa"))
    graph = run(build_graph(model))
    assert graph == sdpa, "cudagraph verify diverged from SDPA (echo drafter)"


@pytest.mark.parametrize("target_layer_ids", [[0], [0, 1]])
def test_n1_cudagraph_verify_need_hidden_matches_sdpa(target_layer_ids):
    """need_hidden=True (DraftHead tap) graph replay token-identical to SDPA — the
    captured tuple's second output (target_hidden) must replay correctly too."""
    model = _tiny_model(0)
    drafter = TargetEchoTreeDrafter(model)
    build_graph = _add_compiled_backend(cudagraph=True)

    def run(e):
        torch.manual_seed(1)
        return e.generate_tree(PROMPT, drafter, block_size=4, tree_width=2,
                               budget=15, target_layer_ids=target_layer_ids,
                               sampling_params=SP)["token_ids"]

    sdpa = run(_tiny_jetflow(model, "sdpa"))
    graph = run(build_graph(model))
    assert graph == sdpa, (
        f"cudagraph need_hidden verify diverged from SDPA "
        f"(target_layer_ids={target_layer_ids})"
    )


def test_cudagraph_captures_once_per_bucket_no_recapture():
    """A full decode captures each tree-N bucket ONCE and never recaptures (the A3-GRAPH
    gate). We monkeypatch `_bucket_for_n` to a constant so every round hits ONE bucket,
    count `_capture_bucket` calls over a multi-round decode, and assert exactly one."""
    import ptd.jetflow.graph_capture as _gc_mod

    model = _tiny_model(0)
    drafter = TargetEchoTreeDrafter(model)
    eng = _add_compiled_backend(cudagraph=True)(model)

    calls = {"n": 0}
    orig = _gc_mod.GraphedVerify._capture_bucket

    def counting(self, B):
        calls["n"] += 1
        return orig(self, B)

    _gc_mod.GraphedVerify._capture_bucket = counting
    try:
        torch.manual_seed(1)
        # SP runs enough rounds that a per-round recapture would show up as calls > 1.
        out = eng.generate_tree(PROMPT, drafter, block_size=4, tree_width=2,
                                budget=15, sampling_params=SP, return_stats=True)
    finally:
        _gc_mod.GraphedVerify._capture_bucket = orig

    assert out["rounds"] >= 2, "need a multi-round decode to detect recapture"
    assert calls["n"] == 1, (
        f"expected ONE capture for the single bucket, got {calls['n']} "
        f"(per-round recapture over {out['rounds']} rounds)"
    )
