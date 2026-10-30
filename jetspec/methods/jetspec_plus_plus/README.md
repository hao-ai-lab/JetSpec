# JetSpec++ torch reference

This folder holds the JetSpec++ method in readable torch, with local artifact loaders and an acceptance-length reader. The shared torch core verifies each proposal against the target and emits the accepted tokens plus the target correction, which preserves greedy autoregressive output.

## One round

The shared torch core calls `JetSpecPPDrafter` in `drafter.py` once per round. It passes the verified context and the target's hidden states.

1. **Take the block.** On a hit, `_prepare` uses the block that was pre-drafted for this outcome. On a miss, or in the first round, `RoundDriver.run` in `round.py` drafts the block from the verified context.
2. **Score the outcomes.** An outcome is an accepted length and an anchor token. `refresh_decoder_logits` in `models/decoder.py` reads the adapter's predicted states, and `blend` in `method/kmass.py` gives each accepted length its probability.
3. **Select the rows.** A row is one pre-drafted block for one outcome. `joint_rule` and `select_rows` in `method/rows.py` keep the most likely outcomes.
4. **Draft the rows.** For each outcome, `build_protocol_hypothesis_contexts` in `context.py` builds the predicted context, and `RoundDriver.run` drafts one row. The rows wait in a cache.
5. **Verify.** The core verifies the block. `observe` looks up the accepted length and the target's next token in the cache. A match is a hit, and that row becomes the next block.

## One draft

`RoundDriver.run` follows the schedule, a list of `J` and `R` operations from `method/schedule.py`.

- **`J`** is one Jacobi pass. `draft_forward` runs the draft model, and `reference_pass` in `method/passes.py` takes the new tokens.
- **`R`** is a context refresh with an anchor replacement. `RoundDriver.predict` runs the adapter on the current draft, and the decoder reads the result. `RefreshSwap` in `method/swap.py` then writes the first token that the decoder disagrees with, and `inject` in `method/refresh.py` gives the next pass the refreshed states.
- **Pins** in `method/pins.py` hold the tokens that stay fixed.

## Code map

| Path | Holds |
| --- | --- |
| `drafter.py` | `JetSpecPPDrafter`, which the torch core calls |
| `round.py` | `RoundDriver`, one draft of a block |
| `context.py` | Block token ids and the hypothesis contexts |
| `method/` | Schedules, passes, refresh, swap, pins, row selection, acceptance |
| `models/` | Adapter, anchor decoder and draft model |
| `artifacts.py` | Artifact loaders and hash checks |
| `reader.py` | The acceptance-length reader |

## Run it

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
