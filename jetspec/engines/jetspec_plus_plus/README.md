# JetSpec++ reference serving engine

This is the JetSpec++ reference serving engine from the authors' research code,
packaged for public use. It keeps the model, draft, verification, scheduling and
transport arithmetic of that engine unchanged. The package holds only the code
that the supported configurations below execute on A100 and B200, measured with
line coverage. Unused functions were removed; no remaining line was edited.

Install `.[jetspec-plus-plus-engine]` with a CUDA 12.8 PyTorch package index.
The dependency versions are pinned to the tested serving stack. Linux,
two CUDA GPUs with peer access, a CUDA compiler, and ninja are required for the
copied Triton and runtime-built CUDA extensions. CPU imports defer unavailable
accelerator dependencies; they do not supply replacement kernels.

```sh
python -m jetspec.engines.jetspec_plus_plus \
  --config a100-j1-r0 \
  --target /path/to/target --drafter /path/to/drafter \
  --adapter /path/to/adapter \
  --heads-bundle /path/to/served_heads_reused_e6v2_b16 \
  --target-release /path/to/target-release \
  --prompts-json /path/to/prompts.json --tokens 64 \
  --placement pair --devices 0,1 --output-json /path/to/read.json
```

The target-release directory supplies `target_release_weights.safetensors`
(the pinned target embedding and LM head). If it is included in the drafter
directory, `--target-release` can be omitted. Artifact files must match the
copied manifest pins. The reader fails closed on missing files or wrong hashes.
Prompts are exported JSON rows with `index` and `token_ids`;
the default dataset is `gsm8k`, count is two, and temperature is zero.
Output files retain the original serving JSON schema, including `outputs[*].token_ids`.

`a100-j1-r0` selects the unchanged lean sm80 J1-R0 recipe and e6v2 B16 bundle.
`b200-j1-r1` selects the unchanged graphed fast6 J1-R1 recipe and e6v3 B16
bundle, including its device verification path.
`reference-j1-r1` selects the unchanged Table 3 reference J1-R1 recipe and e6v3
B16 bundle. Without `--config`, sm100 devices (B200/GB200) select `b200-j1-r1`
and A100 devices select `a100-j1-r0`. These three 8B configurations are supported.

`configs/a100-host.example.json` documents the A100 host values with placeholders.
Supply your own file using `--host-values`, or set local artifact paths by flag.
Flags take precedence over that file. The corresponding environment names are
`JSPP_TARGET`, `JSPP_DRAFTER`, `JSPP_ADAPTER`, `JSPP_HEADS_BUNDLE`,
`JSPP_TARGET_RELEASE`, and `JSPP_HOST_VALUES`. On B200/GB200, set `cuda_arch`
to `10.0` in your host file. Compiler caches default beside the output and can
be placed explicitly with `--cache-root`, or configured for the deployment with
`JSPP_CACHE_ROOT`. The flag takes precedence over the environment variable.

Token equality between two runs needs the same persistent compiler cache. The
target's compiled layers use timing-selected Inductor and Triton configurations,
so independently autotuned caches can change bf16 results and generated tokens,
even with identical weights, prompts, recipe, seed and package versions. Set
`JSPP_CACHE_ROOT` to one shared cache directory to reproduce a read token for
token. The entry ignores EOS and runs to the requested token budget.

Role subpackages are `serving/` (entry read, target runtime and reader helpers),
`models/` (networks and bindings), `draft/` (lane and its copied reference ops),
`kernels/`, `verify/`, and `wire/`. The engine imports no other research package and does not modify the existing
public method or the `jetspec/engines/jetspec/` engine.
