# JetSpec++ on one TPU chip

A JAX port of JetSpec++ for Qwen3-8B on a single TPU chip (developed on a v6e-1, 32 GB HBM).
The output is the target's greedy decode. The drafter, adapter and anchor decoder follow the torch
reference in `jetspec/methods/jetspec_plus_plus/`.

```bash
pip install --no-deps -e .            # plus jax[tpu], safetensors, transformers, huggingface_hub
python -m jetspec.engines.jetspec_plus_plus_tpu "How many positive whole-number divisors does 196 have?" --check-ar
python -m jetspec.engines.jetspec_plus_plus_tpu.bench --dataset gsm8k --samples 32 --max-tokens 2048 \
    --schedules J1-R1,J1-R0,J0-R0 --output out/gsm8k.json
```

Both commands download the pinned `Qwen/Qwen3-8B` target and the `JetSpec/jetspec-plus-plus-qwen3-8b`
checkpoint, as the CUDA engine does. Compiled programs are kept in `~/.cache/jetspec-plus-plus-tpu/xla`
(`JETSPEC_TPU_XLA_CACHE`).

## Measured results (v6e-1)

Qwen3-8B, batch 1, temperature 0, 32 prompts per dataset, at most 2048 tokens, EOS stopping,
int8 draft unless noted. AR is the same JAX target code decoding one token per step.

| Dataset | AR tok/s | Schedule | τ | ms/round | tok/s | Speedup |
|---|---:|---|---:|---:|---:|---:|
| GSM8K | 68.9 | **J1-R1** | 8.44 | 20.17 | **418.6** | **6.08x** |
| GSM8K | | J1-R0 | 6.81 | 17.04 | 399.9 | 5.81x |
| GSM8K | | J0-R0 | 5.79 | 15.46 | 374.7 | 5.44x |
| GSM8K | | J1-R1, bf16 draft | 8.51 | 26.44 | 321.7 | 4.67x |
| MATH-500 | 71.0 | **J1-R1** | 10.45 | 20.22 | **517.0** | **7.29x** |
| MATH-500 | | J1-R0 | 8.97 | 17.13 | 523.9 | 7.38x |
| MATH-500 | | J0-R0 | 8.08 | 15.53 | 520.2 | 7.33x |
| MATH-500 | | J1-R1, bf16 draft | 10.47 | 26.41 | 396.4 | 5.58x |

Per round, the bf16 target verify of 16 tokens takes about 13.9 ms, which is the floor. A drafter pass
takes about 1.5 ms in int8 (2.85 ms in bf16), and a refresh (adapter and decoder) about 3 ms in int8 (6 ms in bf16).

Equality with AR: J1-R1 matched AR on 19/32 GSM8K and 14/32 MATH-500 outputs. All 31 divergences
were checked. In every one, the AR token and the JetSpec++ token are the target's top 2, with a logit
gap of 0.001 to 0.13, about one bf16 step at that logit size. The cause is bf16 rounding, not the draft.

## How one chip changes the method

The two-GPU engine hides drafting behind verification. While block t verifies, the draft GPU
pre-drafts block t+1 for up to sixteen (accepted length, anchor) outcomes. A single chip runs one
program at a time, so those rows have nowhere to hide. A hit row is also drafted from
adapter-predicted context, whereas the reference's miss path drafts from the exact verified context.
So each round here is one compiled program:

```
verify block t (target, 16 tokens) -> accept -> cache the accepted taps
  -> draft block t+1 from the exact context with the J/R schedule
```

The draft is `RoundDriver.run(admission=True)` under `JetSpecPPConfig.paper_eval`: J passes of the
drafter, then R refreshes. A refresh runs the adapter, then the anchor decoder (target layers 34-35, the
final norm and the LM head), then the margin-0 anchor swap with agreed pins. Frozen converged
passes are kept. On the same inputs, the TPU draft matches the torch reference token for token (J0, the swap and
the final block). The adapter-predicted taps agree to a cosine of at least 0.9999.

## What makes it fast

Batch-1 decoding is bound by HBM bandwidth. On this chip, XLA streams weights at about 1.1 TB/s,
and a plain reduction over the weights reaches the same rate. So the work is mostly about bytes:

- **One program per round.** Verify, acceptance, the cache writes and the next draft run with no host
  sync between them. The host reads round r only after it has dispatched round r+1.
- **Incremental context caches.** The reference re-projects the whole context on every pass. Here
  every projection that depends only on committed positions is cached once: the drafter's context
  K/V (5 layers), the adapter's prefix K/V (12 stream layers) and the decoder's layer-34/35 K/V. The
  last of these are the target's own KV cache rows, because the decoder is that slice of the target.
- **Head-major caches** (`[layer, kv_head, seq, dim]`). This is the layout the TPU attention uses, so
  XLA never copies a cache between layouts. That copy cost about 3.5 ms per round.
- **New K/V written after the layer scan.** The target attends to [cached prefix | new block], so
  the cache stays loop-invariant inside the scan.
- **Weight-only int8 on the draft side** (`--draft-dtype int8`, the default): the drafter, the adapter
  and draft copies of the decoder layers and the LM head. The target that verifies stays bf16, so the
  output does not change; only acceptance can move.
- **Span buckets** (512, 1024, 2048, …). All of them are compiled before timing starts.

## Notes

- Qwen3.8-27B does not fit: its bf16 weights alone need about 54 GB.
- Greedy JetSpec++ output can differ from greedy AR output in bf16. In every divergence we checked, the two
  tokens were the target's top 2, with a logit gap of 0.01 to 0.08. The 1-token AR forward and the
  16-token verify forward round differently. `AGENTS.md` reports the same effect for the CUDA engine.
- Throughput counts decode time only, without prefill or the first draft, as the paper does. The clock stops when
  the last needed round is read, for both AR and JetSpec++.
