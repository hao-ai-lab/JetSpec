# Reproduce Table 3

This guide runs the Table 3 protocol on a GB200 pair with public checkpoints. The default arms are AR, stock DSpark and JetSpec++. JetSpec++ uses the current public fast
engine with J1-R1 (8B) or J3-R3 (27B), and a four-row floor. The README table records the
paper's measurements; release checkpoints and the current engine can give different numbers.
The DSpark recipes, prompt populations and analysis are the paper's.

## Run

Install the engine's GPU dependencies, then run from the repository root:

```sh
python bench/jetspec_plus_plus/reproduce_paper_results.py --model qwen3-8b --assets release \
  --host-values /path/to/host-8b.json --devices 0,1 --output-dir /path/to/table3-8b
python bench/jetspec_plus_plus/reproduce_paper_results.py --model qwen3.8-27b --assets release \
  --host-values /path/to/host-27b.json --devices 0,1 --output-dir /path/to/table3-27b
```

`--devices` counts from the first card that a scheduler grants through `CUDA_VISIBLE_DEVICES`, as the serving entry does. Without a scheduler, it names the cards as `nvidia-smi` numbers them.

Run the models in separate output directories. The command writes launch plans and tables in JSON and CSV, and prints the table in the paper's column order and rounding. It refuses to overwrite completed reads, so choose a fresh output directory for a new measurement.

## Host file

The host file tells the command where the local files are:

```json
{
  "schema": "jetspec.serve_host_values/1",
  "cuda_arch": "10.0",
  "rendezvous_mode": "host_port",
  "environment": {"JETSPEC_TARGET_MODEL_PATH": "/path/to/Qwen3-8B", "SSD_HF_CACHE": "/path/to/hub", "SSD_DATASET_DIR": "/path/to/datasets"},
  "artifacts": {"jetspecpp_e6_drafter_release": "/path/to/jetspecpp_e6_drafter_release"}
}
```

Map each registration name below to an absolute directory, and name the directory after its registration. The reader finds a recipe's heads bundle by that name. The command downloads nothing, and the prompts are bundled in `bench/jetspec_plus_plus/prompts/`, located by the bench dataloader.

| Arm | Repository | Path | 8B registration | 27B registration |
| --- | --- | --- | --- | --- |
| JetSpec++ | `JetSpec/jetspec-plus-plus-qwen3-8b`, `JetSpec/jetspec-plus-plus-qwen3p8-27b` | root | `jetspecpp_e6_drafter_release` | `qwen3p8_27b_drafter_main6ep_ep6_release` |
| JetSpec++ | same | `adapter/` | `jetspecpp_e6_adapter_release` | `qwen3p8_27b_adapter_p2_97004` |
| JetSpec++ | same | `calibration/` | `jetspecpp_e6_calibration_release` | `qwen3p8_27b_calibration_release` |
| DSpark | `JetSpec/dspark-qwen3-8b-800k`, `JetSpec/dspark-qwen3p8-27b-800k` | root | `blockhead800k_checkpoint` | `dspark_27b_800k_e6_checkpoint` |

`--artifacts-dir` uses the manifest's destination directories in place of the `artifacts` map.

## Options

- `--resolve-only` prints every launch plan on CPU, without weights.
- `--baselines dspark` is the default; AR and JetSpec++ always run.
- `--arms` selects `AR,DS,JetSpec++` directly. AR is always included.
- `--benchmarks gsm8k,math500` selects the benchmarks.
- `--offsets 0` reads one group of 32 prompts per arm. Fewer than four groups print per-group values without pooling.
- `--resume` keeps the groups that already validate and reads the rest. An invalid read moves to a numbered invalid subdirectory before its rerun.
- `--reduce-only` validates and reduces existing outputs at the same output path.

The reader also needs `Qwen/Qwen3-0.6B` at revision `c1899de289a04d12100db370d81485cdf75e47ca`
in the cache selected by `environment.SSD_HF_CACHE`, as the engine's generic draft slot.
The target is `Qwen/Qwen3-8B` or `Qwen/Qwen3.8-27B`; use the revision pinned in
`jetspec/engines/jetspec_plus_plus/config/model.json`. Download the models before running,
for example with `hf download REPOSITORY --revision REVISION --local-dir DIRECTORY`.

## Protocol

- Each arm reads offsets 0, 32, 64 and 96, at temperature 0, batch size 1, a 2048-token cap and EOS stopping. AR is always included as the speedup denominator.
- Every group must validate before it enters the table. Repeated runs of a group receive the weight of one group.
- Rule A cuts capped repetition loops only in analysis, and it keeps the closing round's complete time.
- Tau includes all retained rounds. Throughput, round, verify, hit and conditional tau omit each request's first decode round. Exposed is round minus verify.

## GB200 notes

- The JetSpec++ pair arms pin each rank to CPUs on its GPU's NUMA node. Unpinned reads of the 8B arm ran 5 to 10 percent slower.
- All groups of one command share the compiler cache in `<output-dir>/compiler-cache`, so only the first group of each arm compiles.
- On a cold cache, a pinned pair's warmup can exceed the 10-minute NCCL timeout of its first receive. The command then reruns that group once with the warm cache and keeps the failed log as `read.failed.log`.
