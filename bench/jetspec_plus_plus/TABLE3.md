# Reproduce Table 3

Provision the paper artifacts locally, then run from the repository root on a
GB200 pair with the engine's GPU dependencies installed:

```sh
python bench/jetspec_plus_plus/table3.py --model qwen3-8b \
  --host-values host-8b.json --devices 0,1 --output-dir results/table3-8b
python bench/jetspec_plus_plus/table3.py --model qwen3.8-27b \
  --host-values host-27b.json --devices 0,1 --output-dir results/table3-27b
```

The host-values format is the serving engine's `jetspec.serve_host_values/1`:
set `cuda_arch` to `10.0`, `rendezvous_mode` to `host_port`, and
`environment.JETSPEC_TARGET_MODEL_PATH` to the local target directory. Map
artifact registration names to absolute directories under `artifacts`, and
name each directory after its registration (for example
`.../served_heads_reused_e6v3_b16`): the reader resolves a recipe's heads bundle
by that name beside the mapped path.
Alternatively, `--artifacts-dir` uses the manifest's destination directories.
No artifacts or datasets are downloaded. Prompts are bundled, including the
27B system-turn render.

| Model | Artifact registration names |
| --- | --- |
| 8B | `jetspecpp_e6_drafter`, `jetspecpp_e6_adapter`, `mask_original_drafter`, `served_heads_reused_e6v3_b16`, `blockhead800k_checkpoint`, `dflash_ours_drafter` |
| 27B | `qwen3p8_27b_drafter_main6ep_ep6`, `qwen3p8_27b_adapter_base_800k`, `jetspecpp_27b_placeholder_release`, `jetspecpp_27b_placeholder_heads`, `dspark_27b_800k_e6_checkpoint` |

The DSpark checkpoints download from Hugging Face: `JetSpec/dspark-qwen3-8b-800k` holds
`blockhead800k_checkpoint`, and `JetSpec/dspark-qwen3p8-27b-800k` holds
`dspark_27b_800k_e6_checkpoint`. Download each repository and map the
registration name to its directory in the host file.

SSD additionally needs the pinned `Qwen/Qwen3-0.6B` snapshot already present in
the local Hugging Face cache selected by `environment.SSD_HF_CACHE` (revision
`c1899de289a04d12100db370d81485cdf75e47ca`). The engine verifies the registered
artifact bytes. Availability of a registration does not imply its weights are
publicly downloadable.

`--resolve-only` prints all launch plans on CPU without weights. By default the
command runs AR, DSpark and JetSpec++. `--baselines all` adds DFlash and SSD to the
8B rows, which is the paper's population; `--baselines dflash,ssd` picks
baselines by name. `--arms` names protocol arms directly (8B
`AR,DS,DFlash,SSD,J1R1REF_F4`, 27B `AR,DS,J3R3DEV_F4`) and overrides `--baselines`.
`--benchmarks gsm8k,math500` selects the benchmarks. AR is always included as the
speedup denominator. Each arm reads offsets 0, 32, 64 and 96, at T=0, batch 1,
cap 2048 and EOS stopping.

`--assets release` runs the JetSpec++ arms on the published checkpoints
(`JetSpec/jetspec-plus-plus-qwen3-8b`, `JetSpec/jetspec-plus-plus-qwen3p8-27b`);
the baselines keep their checkpoints. Map the release registrations in the host
file: 8B `jetspecpp_e6_drafter_release`, `jetspecpp_e6_adapter_release` and
`jetspecpp_e6_calibration_release`; 27B `qwen3p8_27b_drafter_main6ep_ep6_release`,
`qwen3p8_27b_adapter_p2_97004` and `qwen3p8_27b_calibration_release`. The drafter
is the repository root, the adapter its `adapter/` directory and the calibration
its `calibration/` directory, named after the registration. `--offsets 0` reads
one group of 32 prompts per arm; fewer than four groups print per-group values
without pooling. `--resume` keeps the groups that already validate and reads the
rest; an invalid read moves to `invalid/<n>/` before its rerun.
`--reduce-only` validates and reduces existing outputs at the same output path.
Execution refuses to overwrite completed reads; choose a fresh output directory
for a new measurement.

The command writes `plans.json`, `table3.json`, and `table3.csv`, and prints the
paper's column order and rounding. Every group must validate before it enters
the table. Rule A cuts capped repetition loops only in analysis and retains the
closing round's complete time. Tau includes all retained rounds; throughput,
round, verify, hit and conditional tau omit each request's first decode round.
Exposed is round minus verify. Repeated runs of a group receive the weight of
one group. The paper's historical populations include repeated first groups;
fresh GPU timings and the 27B token streams can vary.

The JetSpec++ pair arms (8B `J1R1REF_F4` and 27B `J3R3DEV_F4`) pin each rank to
CPUs on its GPU's NUMA node. The engine's GB200 Table 3 hardware entry does the
same. Unpinned reads of the 8B arm ran 5 to 10 percent slower on GB200. All
groups of one command share the compiler cache in `<output-dir>/compiler-cache`,
so only the first group of each arm compiles. On a cold cache, a pinned pair's
warmup can exceed the 10-minute NCCL timeout of its first receive. The command
then reruns that group once with the warm cache and keeps the failed log as
`read.failed.log`.

The 27B JetSpec++ reads run the chain-graph error audit in
`jetspec/engines/jetspec_plus_plus/table3/audit/`, as the paper reads did. The
command puts it on `PYTHONPATH` and sets `TABLE3_CHAIN_AUDIT=1` for that arm only.
It prints the chain error word after each chain-graph run, and validation requires
a zero word from every run.

Allow roughly 2–4 hours per complete model command with fresh serving processes
and one shared compiler cache; this is an estimate, not a new GPU measurement. The
recorded group means contain about 27.5 minutes (8B) and 72 minutes (27B) of
steady decode alone, before model loading, graph capture, warmup, and prefill.

The CPU golden test uses a 737 KiB compressed fixture of the 96 recorded reads'
per-prompt sufficient statistics. Timing regions retain the first-round cost,
the sum and count of the rule-A-kept steady rounds, and the discarded suffix.
Capped token streams and round credits retain repetition and closing-round
semantics. It checks all 98 measured numeric cells and 16 derived speedups.
