# JetSpec++ serving engine

The serving engine packages the authors' model, draft, verification, scheduling,
and transport implementation for local inference. It includes dense 8B targets
and a preview of hybrid 27B serving.

Install `.[jetspec-plus-plus-engine]` with a CUDA 12.8 PyTorch package index.
Serving requires Linux, two CUDA GPUs with peer access, a CUDA compiler, and
ninja. The 27B preview also requires the target's hybrid attention dependencies.
CPU imports defer accelerator dependencies; execution requires the real kernels.

```sh
python -m jetspec.engines.jetspec_plus_plus \
  --config reference-j1-r1 \
  --target /path/to/target --drafter /path/to/drafter \
  --adapter /path/to/adapter --heads-bundle /path/to/heads \
  --target-release /path/to/release \
  --prompts-json /path/to/prompts.json --count 8 --tokens 2048 \
  --devices 0,1 --output-json /path/to/read.json \
  --cache-root /path/to/compiler-cache \
  --knob SSD_SHARED_GPU_KV_UTILIZATION=0.1 \
  --knob JETSPEC_CHAIN_WARM_REACHABLE=1 \
  --knob JETSPEC_CHAIN_WARM_SINGLE_KEY=1
```

| Configuration | Target | Execution |
| --- | --- | --- |
| `a100-j1-r0` | 8B | J1-R0 paired execution on A100 |
| `reference-j1-r1` | 8B | J1-R1 reference verification, paired GPUs |
| `b200-j1-r1` | 8B | J1-R1 fast execution on B200 |
| `ar` | 8B | Graphed autoregressive baseline, one GPU |
| `a100-27b-j1-r1` | 27B preview | J1-R1 paired execution on A100 |
| `b200-27b-j1-r1` | 27B preview | J1-R1 paired execution on B200 |
| `ar-27b` | 27B preview | Graphed autoregressive baseline, one GPU |

Without a configuration flag, A100 selects `reference-j1-r1` and B200 selects
`b200-j1-r1`. The 27B AR configuration selects its kernel recipe by architecture.
The release directory supplies `target_release_weights.safetensors`; it may be
omitted when the drafter directory contains the release file. Artifact files
must match the manifest's pinned hashes.

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
and reports the ratio of measured decode throughput. Select the 27B configuration
to use its corresponding AR baseline.

Role folders retain the serving implementation's organization: `serving/`,
`models/`, `draft/`, `verify/`, `wire/`, `kernels/`, and `read/`. Configuration
data and artifact pins travel with the installed package.
