"""nano_vllm A3-INT lossless gate: the compiled read-only tree-VERIFY stack, wired
into `NanoEngine` behind `attn_backend="triton_paged_tree_compiled"`, must produce
the SAME tokens as the default SDPA path for the N1 (`generate_tree`) cases the e2e
suite uses.

SDPA is the correctness oracle (same as `test_nano_kernel_e2e`): the compiled stack
reproduces the exact Qwen3 per-layer compute and reads the exact post-RoPE K/V the
oracle reads, so on a tiny fp32 Qwen3 on CUDA the two argmax streams are identical.
We reuse the e2e suite's tiny fp32 CUDA model + drafters and assert token-for-token
equality (fp32 must be EXACT) across the random- and echo-drafter N1 cases.

Needs CUDA (triton); skipped on a CPU-only host. The compiled verify is logits-only
(need_hidden=False); the DraftHead tapped-hidden path falls back to the eager kernel
and is not a target of this gate."""
import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="compiled verify stack needs CUDA (triton)"
)

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
        return eng

    return build


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
