# JetSpec++ correction layers

JetSpec++ adds a checkpoint-loaded **correction layer** to draft-head tree decoding.

## What it is for

A parallel drafter produces its whole block in one forward, so each depth's top-k are
**independent marginals**: a node at depth *d* is proposed without knowing which ancestor was
actually taken. The branches are individually plausible but mutually inconsistent.

A correction layer fixes exactly that. At each expanded tree node it re-runs **conditioned on
the realized ancestor chain** before the node's children are proposed. The gain shows up as
**acceptance** — more drafted tokens survive verification per round at the same node budget.

The public implementations are:

- **attention layer** — path-conditioned residual self-attention over a draft block, applied
  *before* the logits are formed;
- **Markov head** — an order-1 previous-token bias applied per expanded tree node, *after* the
  logits are formed. This is a logit-level bigram correction in the spirit of
  [DSpark](https://arxiv.org/abs/2606.19348);
- **none** — the same checkpoint with the correction disabled: the control arm.

## Losslessness — what is and is not claimed

**Lossless by construction.** The correction only shapes tree *construction*; the verification
and acceptance path is unmodified JetSpec. Acceptance is decided solely by the target's own
argmax on its own verify forward, so a correction layer changes *how many* drafted tokens are
accepted, never *which* tokens are committed. Adding a correction layer introduces **no** new
approximation.

**It is not byte-identical to single-token AR in bf16 — and neither is plain JetSpec.** A
tree/block verify forward (many query positions) and a single-token autoregressive forward (one
query) use different SDPA reduction orders. In bf16 that can flip a *borderline* argmax after
some tens of exact tokens. This is inherent to block verification at reduced precision, is
independent of the drafter and of the correction layer, and disappears in fp32.

So the meaningful test is **not** byte-identity against AR in bf16. It is fp32 exactness, plus a
bf16 comparison **against stock** rather than against an absolute threshold.

Both were run.

**fp32 — exact, all arms.** Every arm was compared token-by-token against `LLM.generate()` (plain
AR greedy) on the same prompts, in fp32 with TF32 explicitly disabled
(`torch.set_float32_matmul_precision("highest")`, `allow_tf32 = False` on both cuda and cudnn —
without that, fp32 matmuls silently route through TF32 and reintroduce the very reduction-order
difference under test).

| arm | fp32 vs AR greedy |
|---|---|
| stock | byte-identical, 8/8 prompts × 128 tokens |
| attention layer | byte-identical, 8/8 prompts × 128 tokens |
| Markov head | byte-identical, 8/8 prompts × 128 tokens |

24 of 24 comparisons had no first difference. **A correction layer introduces no approximation.**

**bf16 — the corrections diverge no earlier than stock.** Index of the first token differing from
bf16 AR greedy, over 20 MATH-500 prompts × 256 new tokens (13 observed, 7 censored at the horizon,
identically in every arm):

| arm | min | median | max |
|---|---:|---:|---:|
| stock | 5 | 68 | 239 |
| attention layer | 5 | **79** | 239 |
| Markov head | 5 | **81** | 208 |

The correction arms sit *later* than stock, not earlier. Since plain JetSpec carries the same
behaviour, the bf16 divergence is a property of block verification at reduced precision, not of
the correction.

> One shared early event is worth naming: on one prompt all three arms diverge at generated index
> 5, with the identical token transition. Being identical across stock and both corrections, it is
> an engine-level bf16 tree-verification signature rather than anything a correction layer
> introduces, and fp32 is exact on that path.

Do not quote "bitwise identical to AR" for a bf16 run of this or any other JetSpec arm; quote the
fp32 result instead.

## What it costs, and what it returns

The correction runs **once per expanded tree node**, so it buys acceptance by spending draft-time
compute. The per-node call is currently **eager and unbatched** — no batching across nodes, no
CUDA graph — so a round takes longer than stock. Reducing that is in progress; the current round
times are a property of this implementation, not of the method.

Measured through the shipped `bench/engine/tps_walltime.py` on B200 (`Qwen/Qwen3-8B`, MATH-500
×64, greedy, tree depth 15, budget 128, `--warm-all --session`, fused GEMMs, bf16, batch size 1),
all three arms on the eager drafter — internally comparable:

| arm | accept-length | vs stock |
|---|---:|---:|
| stock (no correction) | 9.302 | — |
| **attention layer** | **11.100** | **+1.80 (+19.3%)** |
| Markov head | 10.384 | +1.08 (+11.6%) |

Quality ranks attention > Markov > stock.

> **⚠ Never compare accept-length across drafters.** Measured, not assumed: the same stock
> configuration gives **9.302 eager** but **9.480 graphed**, with different generated-token totals.
> The graphed path pads `target_hidden` to coarse capacity buckets, so the two paths draft
> differently. Rows from different drafters are different measurement columns — compare
> eager-to-eager only.

**Independent corroboration.** These acceptance numbers were produced by the shipped benchmark
against the released checkpoints. The line's own earlier campaign — a different rail, different
harness — measured **11.075** for the attention layer and **10.183** for the Markov head at the
same draft depth. Agreement to 0.03 and 0.20, with the ordering and roughly the gap preserved.

## Drafter compatibility — read this before benchmarking

**Where a correction inserts decides whether it can use the graphed drafter.**

`GraphedDraftHead` / `CompiledDraftHead` (`--drafter graphed`, the fast path behind the engine's
headline throughput) returns **projected logits**, shape `(1, depth, vocab)`. That is all a
post-logits correction needs, but a pre-logits correction needs the **hidden block** it never
emits.

| correction | inserts | needs from the drafter | `--drafter graphed` |
|---|---|---|---|
| `markov_head` | after the logits | logits | ✅ works — measured at the full production config |
| `attention_layer` | before the logits | `(1, block_size, hidden)` + live absolute block positions | ❌ **fails loudly at startup** |

So the attention layer currently runs **eager only**. Benchmarking it against a graphed stock
baseline compares two different drafter paths and will understate it; either compare eager-to-eager,
or extend the graphed drafter to return the pre-projection hidden block.

## Checkpoints

| correction | checkpoint |
|---|---|
| attention layer | `JetSpec/jetspec-plus-plus-qwen3-8b-attention-layer` |
| Markov head | `JetSpec/jetspec-plus-plus-qwen3-8b-markov-head` |

Both target `Qwen/Qwen3-8B`, `block_size` 16 (draft depth 15), and are trained under one shared
recipe at a matched step — so they are directly comparable to each other.

> Both repositories now live under the `JetSpec` organization at exactly the ids above, and are
> currently **private** — whether they are published is the maintainers' call. While private,
> `--draft-head` needs either org access plus an authenticated `huggingface_hub` session, or a
> local checkpoint directory.

**A correction runs only on its own checkpoint.** The correction layer is trained weights, not a
decode-time switch — there is nothing to run if the checkpoint does not carry it. Mismatches
**fail closed at load** rather than silently degrading:

| `--correction` | attention-layer ckpt | Markov-head ckpt |
|---|---|---|
| `attention_layer` | ✅ runs | ❌ *"requires a correction_layer config"* |
| `markov_head` | ❌ refused — the checkpoint carries an enabled attention layer | ✅ runs |
| `none` | ✅ the control arm | ❌ — see below |

Two of these deserve a word:

- **`none` is the attention path with the layer switched off, not a general "no correction"
  switch.** It loads the attention-layer checkpoint and sets the layer inactive, so it needs that
  checkpoint. To measure a genuine no-correction baseline, run stock JetSpec (`accum_logp`) with
  the stock head — that, not `none`, is the right comparison denominator.
- The **`markov_head` × attention-checkpoint** refusal is deliberate: running the Markov path on a
  checkpoint that also carries an attention layer would produce an arm of ambiguous provenance.

> **Note — the bundle carries target weights.** Each checkpoint ships
> `target_release_weights.safetensors`, a copy of the target's `embed_tokens` and `lm_head` that
> the loader restores into `Qwen/Qwen3-8B` at load time. The correction layer was trained against
> those exact tensors; loading without them is not a supported configuration.

## Runner

Run the example with:

```bash
python examples/jetspec_plus_plus/run_correction.py \
  --draft-head /path/to/checkpoint \
  --correction attention_layer \
  --prompt-set gsm8k \
  --tree-width 7 \
  --budget 64
```

The public flags are:

- `--draft-head <checkpoint>` selects the local directory or Hub repository.
- `--correction attention_layer|markov_head|none` selects the correction.
- `--prompt-set`, `--tree-width`, and `--budget` retain the existing
  JetSpec benchmark meanings.

For the attention implementation, `JETSPEC_ATTENTION_LAYER_HEAD` can provide
the checkpoint when `--draft-head` is omitted.

## Checkpoint mapping

The primary config key and state-dict prefix are both `correction_layer`.
The two checkpoint adapters accept the pre-ratification key and prefix only as
read-only fallbacks when the public form is absent. The compatibility spelling
is intentionally confined to those adapters and is not a public API, flag, or
environment variable.

Checkpoint loading fails closed on the draft architecture, explicit causal-head
mode, mask token, target layer IDs, correction tensor shapes, and live nonzero
correction probes. The attention runner supplies `block_positions` as a
callable so absolute RoPE positions refresh every decode round.

## Engine boundary

**This example runs on the JetSpec `LLM.generate_tree` engine path. It does
not use the vLLM integration or the paged inference route.**
