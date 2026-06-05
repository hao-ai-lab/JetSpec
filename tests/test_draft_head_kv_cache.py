"""ENGINE-OUTPUT-LOSSLESS gate for the DFlash head's opt-in INCREMENTAL context K/V
cache (`use_context_cache`).

The cache removes BOTH the per-round `torch.cat([k_ctx, k_noise])` (the measured #1
GPU self-time bottleneck, `CatArrayBatched`) AND the per-round FULL context
re-projection. The engine only ever APPENDS to `target_hidden`, so the head projects
+ RoPEs ONLY the NEW context rows each round and appends their post-RoPE K/V to a
persistent per-layer buffer (the cached prefix is reused unchanged); the transient
block K/V is recomputed and placed after.

This is NOT head-logit bit-identical: projecting only-new-rows reduces in a different
order than the full `k_proj(target_hidden)`, so the head's PROPOSED logits drift at
fp32 epsilon (~1e-7). That is EXPECTED and ACCEPTABLE — the draft head only proposes
the tree; the target verify is the source of truth, so engine OUTPUT tokens stay
lossless. So we gate on TOKEN-identity to greedy, NOT head-logit bit-identity:

  - `generate_tree(cache drafter).token_ids == generate()` (the lossless gate), AND
  - `generate_tree(cache drafter).token_ids == generate_tree(recompute drafter)`
    (both lossless -> both == greedy -> equal each other).

`accept_len`/`tpf` MAY differ slightly (the tree differs microscopically — that's the
point); only TOKENS must match. fp32 on CPU makes greedy exact.

These run on CPU in fp32 — no GPU/checkpoint needed. We build a tiny DFlash head +
tiny Qwen3 target (mirroring `tests/test_nano_kernel_e2e.py`'s tiny model). A separate
test asserts the head-logit drift across a growing `target_hidden` is small (≤1e-4) and
NON-zero (confirming the projection is genuinely incremental, not the old re-projection).
"""
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from ptd.models.draft_head import DFlashDraftModel, DFlashContextCache
from ptd.draft_head_drafter import DraftHeadTreeDrafter
from ptd.engine.model_runner import ModelRunner
from ptd.nano_vllm.engine import NanoEngine
from ptd.engine.llm import SamplingParams

DEVICE = "cpu"
BLOCK_SIZE = 4


class _StubTokenizer:
    def decode(self, ids, skip_special_tokens=True):
        return " ".join(str(int(t)) for t in ids)


def _tiny_target(seed: int = 0) -> Qwen3ForCausalLM:
    """Tiny fp32 Qwen3 target (owns embed_tokens + lm_head the head shares)."""
    torch.manual_seed(seed)
    cfg = Qwen3Config(
        vocab_size=128, hidden_size=64, intermediate_size=128,
        num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
        head_dim=16, max_position_embeddings=256, tie_word_embeddings=False,
    )
    model = Qwen3ForCausalLM(cfg).eval().to(torch.float32).to(DEVICE)
    model.config._attn_implementation = "sdpa"
    return model


def _tiny_head(seed: int = 1) -> DFlashDraftModel:
    """Tiny fp32 DFlash head (2 draft layers, causal, block_size=4)."""
    torch.manual_seed(seed)
    cfg = Qwen3Config(
        vocab_size=128, hidden_size=64, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        head_dim=16, max_position_embeddings=256, tie_word_embeddings=False,
    )
    cfg.num_target_layers = 4
    cfg.block_size = BLOCK_SIZE
    cfg.dflash_config = {"causal_head": True, "mask_token_id": 7}
    head = DFlashDraftModel(cfg).eval().to(torch.float32).to(DEVICE)
    head.config._attn_implementation = "sdpa"
    return head


def _drafters():
    """A recompute and a cache drafter sharing the same head + target weights."""
    target, head = _tiny_target(), _tiny_head()
    tli = head.target_layer_ids
    recompute = DraftHeadTreeDrafter(
        head, target, head.block_size, tli, draft_shift=False, use_context_cache=False
    )
    cache = DraftHeadTreeDrafter(
        head, target, head.block_size, tli, draft_shift=False, use_context_cache=True
    )
    return recompute, cache, head


def test_context_cache_head_logits_drift_is_small_and_nonzero():
    """The incremental cache drifts from the recompute head logits ONLY at fp32
    epsilon (the only-new-rows projection reduces in a different order than the full
    `k_proj(target_hidden)`). Across rounds of a growing target_hidden (rows appended
    each round, exactly as engine.generate_tree does) the per-round max|diff| must be:
      - SMALL (≤1e-4) — RoPE/position/buffer correctness; a real bug blows up here, and
      - NON-zero on at least one round — confirming the projection is genuinely
        incremental (the cached prefix is reused, not re-projected; a zero everywhere
        would mean we silently fell back to the old full re-projection no-op).
    The engine TOKEN-identity gate (below) is the lossless gate; this only bounds drift.
    """
    recompute, cache, head = _drafters()
    dim_concat = head.fc.in_features
    depth = head.block_size - 1

    torch.manual_seed(2)
    ctx = torch.randn(1, 5, dim_concat, device=DEVICE)
    ctx_ids = torch.randint(0, 128, (1, 6), device=DEVICE)

    recompute.reset_context_cache()
    cache.reset_context_cache()
    max_drift = 0.0
    for r in range(6):
        lr = recompute.propose_logits(ctx_ids, depth, target_hidden=ctx)
        lc = cache.propose_logits(ctx_ids, depth, target_hidden=ctx)
        drift = (lr - lc).abs().max().item()
        max_drift = max(max_drift, drift)
        assert drift <= 1e-4, (
            f"round {r} (ctx_len={ctx.shape[1]}): cache logits drifted {drift:.3e} from "
            f"recompute (> 1e-4) — RoPE/position/buffer bug, not fp32 epsilon"
        )
        # Append rows the way engine.generate_tree does (accepted path: root+nodes).
        ctx = torch.cat([ctx, torch.randn(1, 2, dim_concat, device=DEVICE)], dim=1)
        ctx_ids = torch.cat([ctx_ids, torch.randint(0, 128, (1, 2), device=DEVICE)], dim=1)
    assert max_drift > 0.0, (
        "cache logits never drifted from recompute across 6 growing-context rounds — "
        "the projection is NOT incremental (silent fallback to full re-projection?)"
    )


def test_context_cache_reset_independent_streams():
    """reset_context_cache() drops prior context: a fresh stream after reset is
    bit-identical to recompute (no stale rows leak across generations)."""
    recompute, cache, head = _drafters()
    dim_concat = head.fc.in_features
    depth = head.block_size - 1

    # Stream A: warm the cache with a long context, then discard it.
    torch.manual_seed(3)
    cache.reset_context_cache()
    ctx_a = torch.randn(1, 9, dim_concat, device=DEVICE)
    cache.propose_logits(torch.randint(0, 128, (1, 10), device=DEVICE), depth, target_hidden=ctx_a)

    # Stream B: reset, then a shorter context must match recompute exactly.
    torch.manual_seed(4)
    ctx_b = torch.randn(1, 4, dim_concat, device=DEVICE)
    ids_b = torch.randint(0, 128, (1, 5), device=DEVICE)
    cache.reset_context_cache()
    recompute.reset_context_cache()
    lc = cache.propose_logits(ids_b, depth, target_hidden=ctx_b)
    lr = recompute.propose_logits(ids_b, depth, target_hidden=ctx_b)
    assert torch.equal(lr, lc), "reset did not clear stale context K/V"


def _tiny_engine(target) -> NanoEngine:
    eng = object.__new__(NanoEngine)
    eng.model = target
    eng.tokenizer = _StubTokenizer()
    eng.runner = ModelRunner(target)
    eng.device = DEVICE
    eng.dtype = torch.float32
    eng.block_size = 16
    eng.eos_token_ids = set()
    eng.attn_backend = "sdpa"
    return eng


def test_context_cache_generate_tree_token_lossless():
    """The lossless gate. End-to-end, the cache drafter's generate_tree token stream
    must equal BOTH:
      - plain greedy generate() (the source-of-truth lossless gate — the target verify
        commits its own greedy along the accepted path), AND
      - the recompute drafter's generate_tree (both lossless -> both == greedy).
    fp32 on CPU makes greedy exact. tpf MAY differ between the two drafters (the
    proposed tree drifts at fp32 epsilon -> a microscopically different tree), so we
    do NOT assert tpf-identity; only TOKENS must match."""
    recompute, cache, head = _drafters()
    target = recompute.target
    tli = head.target_layer_ids
    prompt = torch.tensor([[3, 14, 15, 92, 65, 35, 89, 7]])
    sp = SamplingParams(0.0, 24)

    greedy = _tiny_engine(target).generate(prompt, sampling_params=sp)
    out_rec = _tiny_engine(target).generate_tree(
        prompt, recompute, block_size=BLOCK_SIZE, tree_width=2, budget=15,
        target_layer_ids=tli, sampling_params=sp,
    )
    out_cache = _tiny_engine(target).generate_tree(
        prompt, cache, block_size=BLOCK_SIZE, tree_width=2, budget=15,
        target_layer_ids=tli, sampling_params=sp,
    )
    assert out_rec["token_ids"] == greedy["token_ids"], "recompute drafter not lossless vs greedy"
    assert out_cache["token_ids"] == greedy["token_ids"], "cache-mode tokens diverged from greedy"
    assert out_cache["token_ids"] == out_rec["token_ids"], "cache-mode tokens diverged from recompute"


def test_context_cache_default_off():
    """The cache is opt-in: a default drafter has no live context cache, so the
    recompute (DynamicCache) path is taken untouched."""
    target, head = _tiny_target(), _tiny_head()
    drafter = DraftHeadTreeDrafter(head, target, head.block_size, head.target_layer_ids)
    assert drafter._fwd.use_context_cache is False
    assert drafter._fwd._context_cache is None
    # reset is a harmless no-op when the cache is off.
    drafter.reset_context_cache()


def test_context_cache_is_a_dflash_context_cache():
    """When enabled, the persistent cache is the dedicated class (scoped per stream)."""
    target, head = _tiny_target(), _tiny_head()
    drafter = DraftHeadTreeDrafter(
        head, target, head.block_size, head.target_layer_ids, use_context_cache=True
    )
    assert isinstance(drafter._fwd._context_cache, DFlashContextCache)
