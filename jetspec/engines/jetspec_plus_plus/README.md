# JetSpec++ serving engine

The engine runs JetSpec++ on two GPUs. One GPU verifies a block with the target model, and the other drafts the next block. It serves the dense Qwen3-8B and the hybrid Qwen3.8-27B.

## Where the logic lives

Three folders hold the serving logic:

| Path | Logic |
| --- | --- |
| `verify/` | Target rank: verification and commit |
| `draft/` | Draft rank: draft model, adapter, rows, chain graphs, round driver |
| `serving/` | `LLMEngine`, scheduler, KV cache, `ModelRunner` |

A round runs through them in this order:

1. `Scheduler` in `serving/scheduler.py` admits requests.
2. `SpecDecodeStep` in `verify/step.py` drives verification on the target GPU and sends the outcome through `wire/`.
3. `ChainRoundDriver` in `draft/driver.py` runs the draft passes on the draft GPU and sends the next rows back.

To read one round in plain torch, see the [torch reference](../../methods/jetspec_plus_plus/README.md).

The other paths support these three:

| Path | Holds |
| --- | --- |
| `models/` | Target and anchor decoder models |
| `wire/` | Packets and transport between the two GPUs |
| `kernels/` | Compiled tensor operations grouped by adapter, decoder, chain, passes, rows, refresh and verification, with target-specific helpers |
| `launch/` | The public command, recipe resolution, artifact checks, the child process |
| `config/`, `artifacts/` | Model, method, engine and hardware data, file hashes |
| `runtime/` | Run settings and DSpark serving integration |
| `launch/eos_stop.py`, `launch/run_manifest.py`, `runtime/round_timing.py` | EOS reduction, run records and round timing |
| `bench/jetspec_plus_plus/third_party/`, `bench/jetspec_plus_plus/baselines/` | Pinned DSpark code and comparison routes, loaded only when selected from a source checkout |

## From the command to the first forward pass

1. `__main__.py` calls `main` in `launch/cli.py`.
2. `launch/cli.py` maps the model, schedule and GPU family to an entry in `RECIPES`. `generate` also renders the prompt.
3. `launch/runner.py` checks the artifact hashes, builds the launch plan and starts a child process.
4. The child runs `launch/bench.py`, which constructs `LLMEngine` from `serving/llm_engine.py`.
5. `LLMEngine` constructs `TargetRunner` from `verify/forward.py` in the current process and spawns `DraftRunner` from `draft/rank.py`. Both inherit `ModelRunner` from `serving/model_runner.py`.
6. `ModelRunner` warms up, then allocates the KV cache. The first forward pass is the warmup prefill through `TargetForCausalLM` in `models/target.py`.

## Run the engine

Install the lock file for your platform as the [root README](../../../README.md) describes. Serving requires Linux, two CUDA GPUs with peer access, a CUDA compiler, and ninja.

Each model has one checkpoint repository. The drafter is at its root, the adapter is in `adapter/`, and the calibration (the heads bundle) is in `calibration/`:

```sh
hf download JetSpec/jetspec-plus-plus-qwen3-8b --local-dir jspp-8b
hf download JetSpec/jetspec-plus-plus-qwen3p8-27b --local-dir jspp-27b
```

To answer one prompt:

```sh
python -m jetspec.engines.jetspec_plus_plus generate --model qwen3-8b "PROMPT"
```

Benchmark prompt files live in `bench/jetspec_plus_plus/prompts/` and are located by
the bench dataloader. Serving configuration resolution defaults to that directory
relative to the source checkout; Table 3 passes the directory explicitly. The
`--prompts-json` option reads the caller's file.

The full command reads a prompts file. Its rows need `token_ids`, plus `text` for Qwen3.8-27B:

```sh
python -m jetspec.engines.jetspec_plus_plus \
  --model qwen3-8b \
  --target /path/to/Qwen3-8B --drafter jspp-8b \
  --adapter jspp-8b/adapter --heads-bundle jspp-8b/calibration \
  --prompts-json /path/to/prompts.json --count 8 --tokens 2048 \
  --devices 0,1 --output-json /path/to/read.json \
  --cache-root /path/to/compiler-cache
```

Choose the target with `--model` and the J/R schedule with `--schedule`. `fast` uses CUDA graphs and accepts on the GPU with fused kernels.

| `--model` | `--schedule` | `--engine` | Compute capability |
| --- | --- | --- | --- |
| `qwen3-8b` | `j1-r0` (default on 8.x) | `fast` | 8.x |
| `qwen3-8b` | `j1-r1` (default on 10.x) | `fast` | 8.x, 10.x, 10.x on Arm |
| `qwen3.8-27b` | `j1-r1` (default) | `fast` | 8.x, 10.x, 10.x on Arm |
| `qwen3.8-27b` | `j3-r3` (the paper's Table 3 configuration) | `fast` | 10.x on Arm |
| either | `ar` (graphed autoregressive baseline, one GPU) | `fast` | 8.x, 10.x, 10.x on Arm |

A combination outside this table stops with the list of supported ones.

- **Host file:** `--host-values` names a JSON file with schema `jetspec.serve_host_values/1`, `cuda_arch`, `rendezvous_mode`, `environment` and `artifacts`. Path flags override it. The environment equivalents are `JSPP_TARGET`, `JSPP_DRAFTER`, `JSPP_ADAPTER`, `JSPP_HEADS_BUNDLE`, `JSPP_TARGET_RELEASE` and `JSPP_HOST_VALUES`.
- **Compiler cache:** token parity between two reads requires one persistent cache. Set `--cache-root` or `JSPP_CACHE_ROOT`.
- **Stopping:** the entry defaults to the full token budget. Use `--stop-at-eos on` to stop at EOS.
- **Speedup:** `python bench/jetspec_plus_plus/speedup.py` takes the same flags and reports the ratio to the AR baseline.

The [root `AGENTS.md`](../../../AGENTS.md) lists the other options.

## Model names

The public name Qwen3.8-27B uses the checkpoint type `qwen3_5_text`, the hybrid layout. Its recurrent layer is the Gated DeltaNet (GDN) in `models/gdn.py`. `models/registry.py` selects `TargetForCausalLM` for both targets. `kernels/verify/` holds dense and hybrid verification kernels; `kernels/qwen3_8b/` and `kernels/qwen3p8_27b/` retain target-specific helpers.

## Add a model

1. Check that the target fits `ModelSpec` in `draft/method/model_spec.py`.
2. Add checkpoint parsing to `models/registry.py` and weight mapping to `models/target_loader.py`.
3. Add new layer or state arithmetic to `models/target_layers.py` and `models/state.py` only if the target needs it.
4. Train a drafter, an adapter and a calibration, then pin their hashes in `config/model.json` and `artifacts/manifest.json`.
5. Add the model, its recipes and its defaults to `launch/cli.py`. List each kernel family that you tested.
6. Extend the torch reference loader in `jetspec/methods/jetspec_plus_plus/drafter.py`.
7. Validate on GPUs: the eager target, reference rounds, graph replay, draft hits and misses, and EOS and token-cap stops.
