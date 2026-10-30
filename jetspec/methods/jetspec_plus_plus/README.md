# JetSpec++ torch reference method

JetSpec++ refines a block of draft tokens with J passes and R refresh passes.
An adapter stages predicted target hidden states for the next round, while an
anchor decoder and refresh swap update the block; pins preserve agreed tokens
and keyed rows select proposals using BLEND k mass. The shared torch core
verifies each proposal against the target and emits the accepted tokens plus
the target correction, preserving greedy autoregressive output.

This package provides the readable torch method, local artifact loaders and an
acceptance-length reader for greedy decoding.

Install the method extra and run the commands from the repository root:

```bash
pip install -e '.[jetspec-plus-plus]'
```

Pass local paths for the target, drafter checkpoint, adapter artifact and heads
bundle. The adapter and heads must match the target and drafter provenance;
loaders validate checkpoint keys and target-file hashes. Use compatible local
artifacts:

```bash
export JETSPEC_MODEL=/path/to/Qwen3-8B
export JETSPEC_DRAFTER=/path/to/drafter
export JETSPEC_ADAPTER=/path/to/adapter
export JETSPEC_HEADS_BUNDLE=/path/to/heads-bundle
export JETSPEC_PROMPTS_FILE=/path/to/prompts.json
```

Prompt JSON is a list of strings or objects with `prompt` or `token_ids`, or
an object containing that list under `prompts`. For a short lossless check:

```bash
CUDA_VISIBLE_DEVICES=0 \
python bench/jetspec_plus_plus/verify.py \
  --model "$JETSPEC_MODEL" \
  --drafter "$JETSPEC_DRAFTER" \
  --adapter "$JETSPEC_ADAPTER" \
  --heads-bundle "$JETSPEC_HEADS_BUNDLE" \
  --prompts-file "$JETSPEC_PROMPTS_FILE" \
  --dataset gsm8k \
  --samples 4 \
  --max-tokens 256 \
  --schedule J1-R1 \
  --device cuda \
  --dtype fp32 \
  --output /path/to/verify.json
```

The command compares every output token with plain AR greedy generation and
exits nonzero on a mismatch. Its timing covers eager torch generation only.
To read pooled accepted length over a larger prompt set:

```bash
CUDA_VISIBLE_DEVICES=0 \
python bench/jetspec_plus_plus/tau.py \
  --prompts-file "$JETSPEC_PROMPTS_FILE" \
  --dataset gsm8k \
  --samples 64 \
  --max-tokens 2048 \
  --schedule J3-R3 \
  --device cuda \
  --dtype fp32 \
  --output /path/to/tau.json
```

Both commands accept the model and artifact flags shown above, or their
`JETSPEC_*` defaults. `--target` aliases `--model`; `--seed` controls draft
initialization. Precision options are documented by `--help`. Without a prompts
file, `--prompt-set` loads benchmark prompts (install the `bench` extra);
`--dataset` chooses the heads calibration entry and defaults to the prompt
set. Use `--device cpu` with full precision for local CPU checks.

For a prompt, `examples/jetspec_plus_plus/simple_generate.py` prints the
generated text, accepted-length mean and AR equality check:

```bash
python examples/jetspec_plus_plus/simple_generate.py \
  --prompt "The three primary colors are" \
  --dataset gsm8k \
  --max-tokens 64
```

For serving and speedup measurements, see the
[serving engine README](../../engines/jetspec_plus_plus/README.md).
