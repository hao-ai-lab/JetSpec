# JetSpec++ serving engine

The serving engine packages the authors' model, draft, verification, scheduling,
and transport implementation for local inference. It includes dense 8B targets
and a preview of hybrid 27B serving.

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
ninja. The 27B preview also requires the target's hybrid attention dependencies.
CPU imports defer accelerator dependencies; execution requires the real kernels.

Each model has one checkpoint repository. The drafter is at its root, the
adapter is in `adapter/`, and the calibration (the heads bundle) is in
`calibration/`:

```sh
hf download JetSpec/jetspec-plus-plus-qwen3-8b --local-dir jspp-8b
hf download JetSpec/jetspec-plus-plus-qwen3p8-27b --local-dir jspp-27b
```

```sh
python -m jetspec.engines.jetspec_plus_plus \
  --model qwen3-8b \
  --target /path/to/Qwen3-8B --drafter jspp-8b \
  --adapter jspp-8b/adapter --heads-bundle jspp-8b/calibration \
  --prompts-json /path/to/prompts.json --count 8 --tokens 2048 \
  --devices 0,1 --output-json /path/to/read.json \
  --cache-root /path/to/compiler-cache
```

Choose the target with `--model` (`qwen3-8b`, or `qwen3.8-27b` as a preview), the
J/R schedule with `--schedule`, and the verification with `--engine` (`fast` uses
CUDA graphs, `reference` checks every round against plain verification). The
engine picks its kernels from the GPU compute capability: 8.x uses the sm80
kernels and 10.x the sm100 kernels; a 10.x GPU on an Arm host (GB200) uses its
own serving configuration.

| `--model` | `--schedule` | `--engine` | Compute capability |
| --- | --- | --- | --- |
| `qwen3-8b` | `j1-r0` (default on 8.x) | `fast` | 8.x |
| `qwen3-8b` | `j1-r1` (default on 10.x) | `fast` | 10.x, 10.x on Arm |
| `qwen3-8b` | `j1-r1` | `reference` | 8.x, 10.x |
| `qwen3.8-27b` | `j1-r1` (default) | `fast` | 8.x, 10.x |
| either | `ar` (graphed autoregressive baseline, one GPU) | `fast` | 8.x, 10.x |

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

Measure speedup with `python bench/jetspec_plus_plus/speedup.py` and the same
artifact flags. The benchmark stops at EOS by default, runs AR on the first GPU,
and reports the ratio of measured decode throughput. It takes the same `--model`,
`--schedule` and `--engine` flags and uses the AR baseline of the same model.

Role folders retain the serving implementation's organization: `serving/`,
`models/`, `draft/`, `verify/`, `wire/`, `kernels/`, and `read/`. Configuration
data and artifact pins travel with the installed package.
