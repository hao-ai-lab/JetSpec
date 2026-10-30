# JetSpec++ serving engine

The serving engine packages the authors' model, draft, verification, scheduling,
and transport implementation for local inference. It serves the dense Qwen3-8B
and the hybrid Qwen3.8-27B targets.

## Code map

| Path | Holds |
| --- | --- |
| `entry.py`, `__main__.py` | The public command |
| `launch/` | Recipe resolution, artifact checks, the child process |
| `config/`, `artifacts/` | Model, method, engine and hardware data, file hashes |
| `serving/` | `LLMEngine`, scheduler, KV cache, `ModelRunner` |
| `draft/` | Draft rank, draft model, rows, chain graphs, round driver |
| `verify/` | Target rank, verification, commit |
| `models/` | Target, adapter and decoder models |
| `wire/` | Packets and transport between the two GPUs |
| `kernels/` | Compiled tensor operations, one folder per model |
| `runtime/` | Run settings, run records, DSpark serving |
| `analysis/`, `trace/` | Reports from finished reads |
| `third_party/` | Pinned DSpark code, checked by hash |

## From the command to the first forward pass

1. `__main__.py` calls `entry.main`.
2. `entry.py` maps the model, schedule and GPU family to an entry in `RECIPES`, then renders the prompt.
3. `launch/runner.py` checks the artifact hashes, builds the launch plan and starts a child process.
4. The child runs `launch/bench.py`, which constructs `LLMEngine` from `serving/llm_engine.py`.
5. `LLMEngine` constructs `TargetRunner` from `verify/forward.py` in the current process and spawns `DraftRunner` from `draft/rank.py`. Both inherit `ModelRunner` from `serving/model_runner.py`.
6. `ModelRunner` warms up, then allocates the KV cache. The first forward pass is the warmup prefill through `TargetForCausalLM` in `models/target.py`.

During serving, `Scheduler` in `serving/scheduler.py` admits requests, `SpecDecodeStep` in `verify/step.py` drives verification, and `ChainRoundDriver` in `draft/driver.py` runs the draft passes.

To read one round in plain torch, see the [torch reference](../../methods/jetspec_plus_plus/README.md).

## Model names

The public name Qwen3.8-27B uses the checkpoint type `qwen3_5_text`, the hybrid layout. Its recurrent layer is the Gated DeltaNet (GDN) in `models/gdn.py`. `models/registry.py` selects `TargetForCausalLM` for both targets. `kernels/qwen3_8b/` and `kernels/qwen3p8_27b/` hold the operations specific to each target.

## Add a model

1. Check that the target fits `ModelSpec` in `draft/method/model_spec.py`.
2. Add checkpoint parsing to `models/registry.py` and weight mapping to `models/target_loader.py`.
3. Add new layer or state arithmetic to `models/target_layers.py` and `models/state.py` only if the target needs it.
4. Train a drafter, an adapter and a calibration, then pin their hashes in `config/model.json` and `artifacts/manifest.json`.
5. Add the model, its recipes and its defaults to `entry.py`. List each kernel family that you tested.
6. Extend the torch reference loader in `jetspec/methods/jetspec_plus_plus/drafter.py`.
7. Validate on GPUs: the eager target, reference rounds, graph replay, draft hits and misses, and EOS and token-cap stops.

## Install and run

Install the exact measured environment, then this package, without dependency
resolution. A resolved install picks other versions and can change the output
tokens.

```sh
# Linux x86_64 (A100, B200)
python -m pip install --no-deps -r jetspec/engines/jetspec_plus_plus/requirements-x86_64.lock
# Linux aarch64 (GB200)
python -m pip install --no-deps -r jetspec/engines/jetspec_plus_plus/requirements-gb200-aarch64.lock \
  --extra-index-url https://download.pytorch.org/whl/cu128
python -m pip install --no-deps -e .
```

Serving requires Linux, two CUDA GPUs with peer access, a CUDA compiler, and
ninja. The lock files include the hybrid attention dependencies of the 27B target.
CPU imports defer accelerator dependencies; execution requires the real kernels.

Each model has one checkpoint repository. The drafter is at its root, the
adapter is in `adapter/`, and the calibration (the heads bundle) is in
`calibration/`:

```sh
hf download JetSpec/jetspec-plus-plus-qwen3-8b --local-dir jspp-8b
hf download JetSpec/jetspec-plus-plus-qwen3p8-27b --local-dir jspp-27b
```

To answer one prompt, `generate` downloads the pinned target and checkpoint
(unless `--target` and `--checkpoint` name local directories), renders the chat
template, and prints the answer with its throughput:

```sh
python -m jetspec.engines.jetspec_plus_plus generate --model qwen3-8b \
  "How many positive whole-number divisors does 196 have?"
```

The full command reads a prompts file whose rows carry `token_ids`:

```sh
python -m jetspec.engines.jetspec_plus_plus \
  --model qwen3-8b \
  --target /path/to/Qwen3-8B --drafter jspp-8b \
  --adapter jspp-8b/adapter --heads-bundle jspp-8b/calibration \
  --prompts-json /path/to/prompts.json --count 8 --tokens 2048 \
  --devices 0,1 --output-json /path/to/read.json \
  --cache-root /path/to/compiler-cache
```

Choose the target with `--model` (`qwen3-8b` or `qwen3.8-27b`), the
J/R schedule with `--schedule`, and the verification with `--engine` (`fast` uses
CUDA graphs, `reference` checks every round against plain verification). The
engine picks its kernels from the GPU compute capability: 8.x uses the sm80
kernels and 10.x the sm100 kernels; a 10.x GPU on an Arm host (GB200) uses its
own serving configuration.

| `--model` | `--schedule` | `--engine` | Compute capability |
| --- | --- | --- | --- |
| `qwen3-8b` | `j1-r0` (default on 8.x) | `fast` | 8.x |
| `qwen3-8b` | `j1-r1` (default on 10.x) | `fast` | 8.x, 10.x, 10.x on Arm |
| `qwen3-8b` | `j1-r1` | `reference` | 8.x, 10.x, 10.x on Arm |
| `qwen3.8-27b` | `j1-r1` (default) | `fast` | 8.x, 10.x, 10.x on Arm |
| either | `ar` (graphed autoregressive baseline, one GPU) | `fast` | 8.x, 10.x, 10.x on Arm |

A combination outside this table stops with the list of supported ones.
The engine takes the drafter's embedding and LM head from the target. Only the
`reference` engine still reads a separate release file through `--target-release`.
Artifact files must match the manifest's pinned hashes.

Prompts are exported JSON rows with `index` and `token_ids`. Temperature is zero,
the default dataset is `gsm8k`, and the default count is two. Outputs retain the
serving JSON schema, including `outputs[*].token_ids`.

Use `--host-values` for a JSON host file with schema
`jetspec.serve_host_values/1`, `cuda_arch`, `rendezvous_mode`, `environment`, and
`artifacts`. Path flags override that file. Environment equivalents are
`JSPP_TARGET`, `JSPP_DRAFTER`, `JSPP_ADAPTER`, `JSPP_HEADS_BUNDLE`,
`JSPP_TARGET_RELEASE`, and `JSPP_HOST_VALUES`.

Token parity between two reads requires one persistent compiler cache. Timing
selected Inductor and Triton configurations can otherwise change BF16 rounding.
Set `--cache-root` or `JSPP_CACHE_ROOT`; the flag takes precedence. The serving
entry defaults to the full token budget; use `--stop-at-eos on` to stop at EOS.

`--devices` names two GPUs as `nvidia-smi` numbers them; `CUDA_VISIBLE_DEVICES`
does not renumber them, and the engine refuses a GPU that another process uses.
The engine's generic draft slot reads the pinned `Qwen/Qwen3-0.6B` snapshot from
`SSD_HF_CACHE` (default: the Hugging Face hub cache), and the entry downloads it
when it is missing. `--heads-bundle` may name a checkpoint's `calibration/`
directory; the entry links it under the registration name the reader expects.

Measure speedup with `python bench/jetspec_plus_plus/speedup.py` and the same
artifact flags. The benchmark stops at EOS by default, runs AR on the first GPU,
and reports the ratio of measured decode throughput. It takes the same `--model`,
`--schedule` and `--engine` flags and uses the AR baseline of the same model.
