"""L5 no-gather lossless gate: `generate_tree` on the logical-KV backends must be
token-identical AND accept-length-identical to the gather-path oracles.

The no-gather path is a MEMORY-LAYOUT choice, not a semantics change: a committed
token's K/V is written once by the verify forward, already RoPE-rotated at its
final absolute position; gather copies those bytes into block-table order while
logical-KV leaves them in place and hands the kernel a slot map. Both present the
kernel the same position->bytes function, so attention output — and therefore
logits, the greedy walk, tokens, and accept lengths — must match EXACTLY (fp32
tiny model: bitwise). Any divergence is a wiring bug (wrong starts/lens/slots,
bad slot-commit overlap, or a freed-block reuse clobbering live KV).

Mirrors `test_compiled_verify_lossless.py`: tiny fp32 CUDA model + the e2e
drafters; the gather-path `triton_paged_tree_compiled` build is the oracle.

Needs CUDA (triton); skipped on a CPU-only host."""
import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="logical-KV kernel path needs CUDA (triton)"
)

# Same rationale as the compiled-verify suite: many stack specializations in one
# process exceed dynamo's default recompile_limit; test-only, no production effect.
torch._dynamo.config.recompile_limit = 64

from jetspec.core.llm import SamplingParams
from jetspec.draft import RandomTreeDrafter, TargetEchoTreeDrafter
from tests.inference_engine.test_jetspec_kernel_e2e import _tiny_model, _tiny_jetspec, PROMPT, SP


def _add_backend(backend: str):
    """Builder for any compiled-family backend over the tiny model (mirrors
    `test_compiled_verify_lossless._add_compiled_backend`, parameterized by the
    backend string so the logical-KV no-gather variants ride the same fixture)."""
    from jetspec.inference_engine.compiled_verify_stack import CompiledVerifyStack
    from jetspec.inference_engine.engine import _CUDAGRAPH_BACKENDS, _env_flag

    def build(model):
        eng = _tiny_jetspec(model, "triton_paged_tree")
        eng.attn_backend = backend
        eng.fuse_gemms = _env_flag("JETSPEC_FUSE_GEMMS")
        eng.compiled_verify = CompiledVerifyStack(
            model, block_size=eng.block_size, fuse_gemms=eng.fuse_gemms,
        )
        eng.compiled_ar = CompiledVerifyStack(
            model, block_size=eng.block_size, fuse_gemms=eng.fuse_gemms,
        )
        eng._compiled_verify_hidden = {}
        eng._use_cudagraph = backend in _CUDAGRAPH_BACKENDS
        eng._graphed_verify = {}
        return eng

    return build


def _run(eng, drafter, seed=1, **kw):
    torch.manual_seed(seed)
    out = eng.generate_tree(PROMPT, drafter, block_size=4, tree_width=2,
                            budget=15, sampling_params=SP, return_stats=True, **kw)
    return out["token_ids"], out.get("accept_lengths")


# --- no-gather (direct compiled) vs the gather-path oracle -------------------

@pytest.mark.parametrize("seed", [1, 7])
def test_nogather_compiled_matches_gather_random(seed):
    model = _tiny_model(0)
    drafter = RandomTreeDrafter(vocab_size=model.config.vocab_size)
    tok_g, acc_g = _run(_add_backend("triton_paged_tree_compiled")(model), drafter, seed)
    tok_l, acc_l = _run(_add_backend("triton_paged_tree_compiled_nogather")(model), drafter, seed)
    assert tok_l == tok_g, f"no-gather tokens diverged from gather oracle (seed={seed})"
    assert acc_l == acc_g, f"no-gather accept_lengths diverged (seed={seed})"


def test_nogather_compiled_matches_gather_echo():
    """Echo drafter -> full-depth accepts: the deepest slot-commit/overlap path."""
    model = _tiny_model(0)
    drafter = TargetEchoTreeDrafter(model)
    tok_g, acc_g = _run(_add_backend("triton_paged_tree_compiled")(model), drafter)
    tok_l, acc_l = _run(_add_backend("triton_paged_tree_compiled_nogather")(model), drafter)
    assert tok_l == tok_g, "no-gather tokens diverged (echo drafter)"
    assert acc_l == acc_g, "no-gather accept_lengths diverged (echo drafter)"


def test_nogather_matches_sdpa_directly():
    """Transitivity check straight against the primary SDPA oracle."""
    model = _tiny_model(0)
    drafter = TargetEchoTreeDrafter(model)
    torch.manual_seed(1)
    sdpa = _tiny_jetspec(model, "sdpa").generate_tree(
        PROMPT, drafter, block_size=4, tree_width=2, budget=15,
        sampling_params=SP)["token_ids"]
    tok_l, _ = _run(_add_backend("triton_paged_tree_compiled_nogather")(model), drafter)
    assert tok_l == sdpa, "no-gather diverged from SDPA"


@pytest.mark.parametrize("target_layer_ids", [[0], [0, 1]])
def test_nogather_need_hidden_matches_gather(target_layer_ids):
    """The tapped-hidden (DraftHead) path over no-gather: taps come from the verify
    forward, never the cache, so they must be unaffected by the layout choice."""
    model = _tiny_model(0)
    drafter = TargetEchoTreeDrafter(model)
    tok_g, acc_g = _run(_add_backend("triton_paged_tree_compiled")(model), drafter,
                        target_layer_ids=target_layer_ids)
    tok_l, acc_l = _run(_add_backend("triton_paged_tree_compiled_nogather")(model),
                        drafter, target_layer_ids=target_layer_ids)
    assert tok_l == tok_g and acc_l == acc_g, "need_hidden no-gather diverged"


# --- cudagraph no-gather vs direct no-gather (capture correctness) -----------

def test_cudagraph_nogather_matches_compiled_nogather():
    model = _tiny_model(0)
    drafter = TargetEchoTreeDrafter(model)
    tok_d, acc_d = _run(_add_backend("triton_paged_tree_compiled_nogather")(model), drafter)
    tok_c, acc_c = _run(_add_backend("triton_paged_tree_cudagraph_nogather")(model), drafter)
    assert tok_c == tok_d, "cudagraph no-gather diverged from direct no-gather"
    assert acc_c == acc_d, "cudagraph no-gather accept_lengths diverged"


def test_cudagraph_nogather_multi_prompt_recapture():
    """Back-to-back decodes: each builds fresh slot buffers + pool, so the pool_tag
    must rebuild the graphs (stale addresses would read freed slot maps). Both
    decodes must match the direct no-gather path."""
    model = _tiny_model(0)
    drafter = TargetEchoTreeDrafter(model)
    eng_c = _add_backend("triton_paged_tree_cudagraph_nogather")(model)
    eng_d = _add_backend("triton_paged_tree_compiled_nogather")(model)
    for seed in (1, 7):
        tok_c, _ = _run(eng_c, drafter, seed)
        tok_d, _ = _run(eng_d, drafter, seed)
        assert tok_c == tok_d, f"multi-prompt recapture diverged (seed={seed})"
