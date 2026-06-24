"""Real DraftHead gate: a trained DFlash head wired into the engine must produce
near-greedy output (catching structural bugs) AND beat the stubs on
tokens-per-forward (tpf > 1.5; stubs sit at ~1.0-1.1). A wrong draft_shift slice
or layer-id offset stays "lossless" but collapses tpf, so the tpf floor is the
real catch.

On the reference: speculative decoding is lossless *in exact arithmetic*, but the
recompute-based verify here runs block-forwards while autoregressive greedy runs
step-forwards. In bf16, SDPA's reduction order differs between the two, so the
argmax flips at a handful of borderline tokens (typically 100+ exact tokens, then
a rare flip). Exact bitwise losslessness needs a KV-cache verify (the deferred
optimization — the reference engine gets exactness that way). So we compare
against a *recompute* greedy and assert high token agreement, not byte-identity.

Needs CUDA + a real Qwen3-8B target + a trained head; run on b200 (GPU 0-3, the
JetSpec lane):

    CUDA_VISIBLE_DEVICES=0 JETSPEC_TEST_MODEL=Qwen/Qwen3-8B \
      JETSPEC_DRAFT_HEAD="<insert-trained-dflash-head-checkpoint-path>" \
      pytest tests/core/test_draft_head_lossless.py -x -s
"""
import os

import pytest
import torch
from transformers import DynamicCache

from jetspec.core.llm import LLM, SamplingParams
from jetspec.models.draft_head import load_draft_head
from jetspec.draft_head_adapter import DraftHeadDrafter, DraftHeadTreeDrafter

MODEL = os.environ.get("JETSPEC_TEST_MODEL", "Qwen/Qwen3-8B")
DRAFT_HEAD = os.environ.get("JETSPEC_DRAFT_HEAD")
MAX_NEW = 128
# Exact-prefix floor: a working engine matches recompute-greedy for many tokens,
# then (in bf16) a single SDPA reduction-order flip cascades. A structural bug
# (bad verify / position ids) diverges within a few tokens. So gate on the length
# of the exact prefix, not overall agreement (one late flip tanks agreement).
EXACT_PREFIX_FLOOR = 16

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not DRAFT_HEAD,
    reason="needs CUDA + a real Qwen3-8B target + JETSPEC_DRAFT_HEAD checkpoint",
)

PROMPT = "What is 127 times 384? Reason step by step, then give the final number."


def _recompute_greedy(llm, n):
    """Greedy via full recompute (the same numerics the spec verify uses), so the
    only residual mismatch is the block-vs-step SDPA reduction-order flip."""
    ids = llm.tokenizer(PROMPT, return_tensors="pt").input_ids.to(llm.device)
    out, cur = [], ids
    for _ in range(n):
        pos = torch.arange(cur.shape[1], device=llm.device).unsqueeze(0)
        logits, _, _ = llm.runner.forward(cur, DynamicCache(), pos)
        t = int(logits[0, -1].argmax())
        out.append(t)
        if t in llm.eos_token_ids:
            break
        cur = torch.cat([cur, torch.tensor([[t]], device=llm.device)], dim=1)
    return out


def _exact_prefix(a, b):
    """Length of the leading run where a and b agree (the exact prefix)."""
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


@pytest.fixture(scope="module")
def llm():
    return LLM(MODEL)


@pytest.fixture(scope="module")
def ref(llm):
    return _recompute_greedy(llm, MAX_NEW)


@pytest.fixture(scope="module")
def head(llm):
    # The DFlash head is in-place (non-shift); draft_shift stays False.
    return load_draft_head(DRAFT_HEAD)


def test_draft_head_chain_near_greedy_and_fast(llm, head, ref):
    """Real head → chain output agrees with recompute-greedy AND tpf > 1.5."""
    sp = SamplingParams(0.0, MAX_NEW)
    tli = head.target_layer_ids
    drafter = DraftHeadDrafter(
        head, target=llm.model, block_size=head.block_size,
        target_layer_ids=tli, draft_shift=False,
    )
    out = llm.generate_chain(
        PROMPT, drafter, block_size=head.block_size,
        target_layer_ids=tli, sampling_params=sp,
    )
    exact = _exact_prefix(ref, out["token_ids"])
    print(f"\nchain: tpf={out['tpf']:.3f} exact_prefix={exact}/{len(ref)}")
    assert exact >= EXACT_PREFIX_FLOOR, f"chain diverged from greedy at token {exact} (structural bug?)"
    assert out["tpf"] > 1.5, f"draft head should beat stubs (tpf>1.5), got {out['tpf']:.3f}"


def test_draft_head_tree_near_greedy_and_fast(llm, head, ref):
    """Real head → tree output agrees with recompute-greedy AND tpf > 1.5."""
    sp = SamplingParams(0.0, MAX_NEW)
    tli = head.target_layer_ids
    tree_drafter = DraftHeadTreeDrafter(
        head, target=llm.model, block_size=head.block_size,
        target_layer_ids=tli, draft_shift=False,
    )
    out = llm.generate_tree(
        PROMPT, tree_drafter, block_size=head.block_size, tree_width=2, budget=24,
        target_layer_ids=tli, sampling_params=sp,
    )
    exact = _exact_prefix(ref, out["token_ids"])
    print(f"\ntree: tpf={out['tpf']:.3f} exact_prefix={exact}/{len(ref)}")
    assert exact >= EXACT_PREFIX_FLOOR, f"tree diverged from greedy at token {exact} (structural bug?)"
    assert out["tpf"] > 1.5, f"draft head tree should beat stubs (tpf>1.5), got {out['tpf']:.3f}"
