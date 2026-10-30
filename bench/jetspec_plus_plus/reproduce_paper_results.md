# Reproduce Table 3

This guide reruns the paper's Table 3 on a GB200 pair. The default arms are AR, DSpark and JetSpec++.

## Run

Install the engine's GPU dependencies, then run from the repository root:

```sh
python bench/jetspec_plus_plus/reproduce_paper_results.py --model qwen3-8b --assets release \
  --host-values host-8b.json --devices 0,1 --output-dir results/table3-8b
python bench/jetspec_plus_plus/reproduce_paper_results.py --model qwen3.8-27b --assets release \
  --host-values host-27b.json --devices 0,1 --output-dir results/table3-27b
```

Allow 2 to 4 hours per model. The command writes `plans.json`, `table3.json` and `table3.csv`, and prints the table in the paper's column order and rounding. It refuses to overwrite completed reads, so choose a fresh output directory for a new measurement.

## Host file

The host file tells the command where the local files are:

```json
{
  "schema": "jetspec.serve_host_values/1",
  "cuda_arch": "10.0",
  "rendezvous_mode": "host_port",
  "environment": {"JETSPEC_TARGET_MODEL_PATH": "/models/Qwen3-8B"},
  "artifacts": {"jetspecpp_e6_drafter_release": "/models/jetspecpp_e6_drafter_release"}
}
```

Map each registration name below to an absolute directory, and name the directory after its registration. The reader finds a recipe's heads bundle by that name. The command downloads nothing, and the prompts are bundled.

| Arm | Repository | Path | 8B registration | 27B registration |
| --- | --- | --- | --- | --- |
| JetSpec++ | `JetSpec/jetspec-plus-plus-qwen3-8b`, `JetSpec/jetspec-plus-plus-qwen3p8-27b` | root | `jetspecpp_e6_drafter_release` | `qwen3p8_27b_drafter_main6ep_ep6_release` |
| JetSpec++ | same | `adapter/` | `jetspecpp_e6_adapter_release` | `qwen3p8_27b_adapter_p2_97004` |
| JetSpec++ | same | `calibration/` | `jetspecpp_e6_calibration_release` | `qwen3p8_27b_calibration_release` |
| DSpark | `JetSpec/dspark-qwen3-8b-800k`, `JetSpec/dspark-qwen3p8-27b-800k` | root | `blockhead800k_checkpoint` | `dspark_27b_800k_e6_checkpoint` |

`--assets paper`, the default, uses the paper's registrations in place of the JetSpec++ rows. Some of these weights are not public:

- 8B: `jetspecpp_e6_drafter`, `jetspecpp_e6_adapter`, `mask_original_drafter`, `served_heads_reused_e6v3_b16`, `dflash_ours_drafter`
- 27B: `qwen3p8_27b_drafter_main6ep_ep6`, `qwen3p8_27b_adapter_base_800k`, `jetspecpp_27b_placeholder_release`, `jetspecpp_27b_placeholder_heads`

`--artifacts-dir` uses the manifest's destination directories in place of the `artifacts` map.

## Options

- `--resolve-only` prints every launch plan on CPU, without weights.
- `--baselines all` adds DFlash and SSD to the 8B rows, which is the paper's population. `--baselines dflash,ssd` picks baselines by name.
- `--arms` names protocol arms directly and overrides `--baselines`: 8B `AR,DS,DFlash,SSD,J1R1REF_F4`, 27B `AR,DS,J3R3DEV_F4`.
- `--benchmarks gsm8k,math500` selects the benchmarks.
- `--offsets 0` reads one group of 32 prompts per arm. Fewer than four groups print per-group values without pooling.
- `--resume` keeps the groups that already validate and reads the rest. An invalid read moves to `invalid/<n>/` before its rerun.
- `--reduce-only` validates and reduces existing outputs at the same output path.

SSD also needs the pinned `Qwen/Qwen3-0.6B` snapshot (revision `c1899de289a04d12100db370d81485cdf75e47ca`) in the Hugging Face cache that `environment.SSD_HF_CACHE` selects.

## Protocol

- Each arm reads offsets 0, 32, 64 and 96, at temperature 0, batch size 1, a 2048-token cap and EOS stopping. AR is always included as the speedup denominator.
- Every group must validate before it enters the table. Repeated runs of a group receive the weight of one group.
- Rule A cuts capped repetition loops only in analysis, and it keeps the closing round's complete time.
- Tau includes all retained rounds. Throughput, round, verify, hit and conditional tau omit each request's first decode round. Exposed is round minus verify.

## GB200 notes

- The JetSpec++ pair arms pin each rank to CPUs on its GPU's NUMA node. Unpinned reads of the 8B arm ran 5 to 10 percent slower.
- All groups of one command share the compiler cache in `<output-dir>/compiler-cache`, so only the first group of each arm compiles.
- On a cold cache, a pinned pair's warmup can exceed the 10-minute NCCL timeout of its first receive. The command then reruns that group once with the warm cache and keeps the failed log as `read.failed.log`.
- The 27B JetSpec++ reads run the chain-graph error audit in `paper/audit/`, and validation requires a zero error word from every run.
