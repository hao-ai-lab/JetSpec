<p align="center">
  <img src="assets/jetflow_icon.png" alt="JetFlow" width="180" align="center">
</p>

<div align="center"><h1>JetFlow: Parallel Tree Drafting</h1></div>

<p align="center">
  <a href="https://huggingface.co/JetFlow">Hugging Face</a> ·
  <a href="https://hao-ai-lab.github.io/jetflow">Project Webpage</a>
</p>

JetFlow is an implementation of **parallel tree drafting** for fast LLM speculative decoding inference with up to 10x acceptance length, and 1000 TPS on coding and math tasks using B200 GPUs. A causal-parallel draft head proposes a token tree, and the frozen target model verifies the whole tree in one forward pass under a tree-causal attention mask. The accepted path is selected in accordance with the target's own logits, so decoding is lossless by construction.



## Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Model Weights](#model-weights)
- [Repo Overview](#repo-overview)
- [Usage](#usage)
  - [HF References](#hf-references)
  - [JetFlow Inference Engine](#jetflow-inference-engine)
  - [Benchmarks](#benchmarks)
- [Results](#results)
  - [Engine Results](#engine-results)
  - [Tree Algorithms](#tree-algorithms)
  - [Test Yourself](#test-yourself)
- [Citation](#citation)

## Introduction

Speculative decoding is fast when the target accepts many draft tokens and drafting remains cheap. Prior heads often trade off those two terms: autoregressive drafters condition on each path but pay a forward pass per depth, while block-diffusion drafters draft many positions in one pass but score branches independently.

JetFlow keeps the **one-pass drafting efficiency and restores causal branch conditioning**. The draft head reads fused hidden states from the frozen target and emits per-depth logits in a single parallel pass. Tree construction spends a draft budget over high-probability branches, and the target verifies every node in one batched/tree-masked forward.

On Qwen3-8B evaluations, JetFlow reaches up to **9.64x end-to-end speedup on MATH-500**, with strong gains across reasoning, code, and chat workloads: 7.82x on GSM8K, 8.78x on AIME25, 7.12x on HumanEval, 6.73x on MBPP, 7.67x on LCB, and 4.58x on MT-Bench.


<p align="center">
  <img src="assets/end_to_end_speedup_barplot_refined_v2_page.jpg" alt="End-to-end speedup over autoregressive decoding on Qwen3-8B across benchmarks" width="760">
</p>


## Installation

Create an environment and install the package:

```bash
cd /root/workspace/JetFlow
pip install -e '.[bench,kernel]'
```

For FlashAttention 2 benchmarks, install the extra after build dependencies are available:

```bash
pip install -e '.[bench,flash-attn]'
```

If you use `uv`, the project includes extra build dependency metadata for `flash-attn`.

## Model Weights

| Component | Default |
|---|---|
| Target model | `Qwen/Qwen3-8B` |
| Hugging Face org | [`JetFlow`](https://huggingface.co/JetFlow) |
| Draft head | `JetFlow/jetflow-qwen3-8b-fkl-epoch6-3e-4-no-gamma`(https://huggingface.co/JetFlow/jetflow-qwen3-8b-fkl-epoch6-3e-4-no-gamma) |

Most benchmark and diagnostic scripts read the draft head from `JETFLOW_DRAFT_HEAD`:

```bash
export JETFLOW_DRAFT_HEAD=JetFlow/<draft-head-repo>
```


## Repo Overview

The project has two execution paths:

- `jetflow/core/`: a lightweight HuggingFace-based reference implementation.
- `jetflow/inference_engine/`: an optimized serving engine with paged KV, custom Triton tree attention, and CUDA graphs for better wall-clock latency and throughput.

```text
jetflow/
  core/                 # HuggingFace reference core: LLM, ModelRunner, sampler, tree attention hook
  inference_engine/     # optimized JetFlow engine: paged KV, scheduler, kernels, CUDA graphs
  tree/                 # engine-agnostic tree construction and acceptance
  models/               # target/draft-head loading and model utilities
  draft.py              # simple drafters used in tests and correctness gates
  draft_head_adapter.py # trained draft-head adapter
```



## Usage

### HF References

```python
from jetflow import LLM, SamplingParams

llm = LLM("Qwen/Qwen3-8B", attn_implementation="flash_attention_2")
out = llm.generate(
    "The three primary colors are",
    SamplingParams(temperature=0.0, max_new_tokens=64),
)
print(out["text"])
```



### JetFlow Inference Engine

The optimized engine uses paged KV, a Triton tree-attention backend, compiled verification, CUDA graphs, and optional session reuse.

```python
from jetflow import load_draft_head, DraftHeadTreeDrafter
from jetflow.inference_engine import JetFlowEngine, SamplingParams

engine = JetFlowEngine(
    "Qwen/Qwen3-8B",
    attn_backend="triton_paged_tree_compiled",
)
head = load_draft_head("JetFlow/<draft-head-repo>")
drafter = DraftHeadTreeDrafter(
    head,
    target=engine.model,
    block_size=head.block_size,
    target_layer_ids=head.target_layer_ids,
)
out = engine.generate_tree(
    "The three primary colors are",
    drafter,
    block_size=head.block_size,
    tree_width=7,
    budget=63,
    target_layer_ids=head.target_layer_ids,
    sampling_params=SamplingParams(temperature=0.0, max_new_tokens=64),
)
print(out["text"])
print("tokens per forward:", out["tpf"])
```

### Example Benchmarking Script

HF reference benchmark:

```bash
CUDA_VISIBLE_DEVICES=0 \
JETFLOW_DRAFT_HEAD=JetFlow/<draft-head-repo> \
python bench/reference/benchmark.py \
  --model Qwen/Qwen3-8B \
  --attn-implementation flash_attention_2 \
  --tree-attn triton \
  --dataset gsm8k \
  --samples 64 \
  --algos accum_logp \
  --width 7 \
  --budget 255 \
  --max-new 256 \
  --warmup-samples-per-rank 1
```

Optimized JetFlow wall-clock benchmark:

```bash
CUDA_VISIBLE_DEVICES=0 \
JETFLOW_FUSE_GEMMS=1 \
JETFLOW_BACKEND=triton_paged_tree_cudagraph_nogather \
JETFLOW_DRAFT_HEAD=JetFlow/<draft-head-repo> \
python bench/engine/tps_walltime.py \
  --prompt-set gsm8k \
  --samples 64 \
  --max-tokens 2048 \
  --budget 127 \
  --session
```

[`vLLM fork for JetFlow support`](https://github.com/snyhlxde1/vllm-jetflow) MATH-500 reference test:

```bash
VLLM_FORK_DIR=/path/to/vllm-jetflow
TARGET_MODEL=/path/to/Qwen3-30B-A3B
DRAFT_MODEL=/path/to/jetflow-draft-head
PROFILER_DIR=/path/to/output/vllm-jetflow-math500
TP_SIZE=4
BATCH_SIZE=1
MAX_TREE_BUDGET=127
MAX_TOKENS=512
MAX_SAMPLES=16

cd "${VLLM_FORK_DIR}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}" \
bash examples/offline_inference/dflash_profiling_math500_causal_tree_budget_bsz_sweep_dgx_pod.sh \
  --model "${TARGET_MODEL}" \
  --draft-model "${DRAFT_MODEL}" \
  --profiler-dir "${PROFILER_DIR}" \
  --tree-attn-kernel triton \
  --enable-expert-parallel \
  --disable-cascade-attn \
  --cudagraph-mode default \
  --tp-size "${TP_SIZE}" \
  --batch-sizes "${BATCH_SIZE}" \
  --max-num-seqs "${BATCH_SIZE}" \
  --tree-budgets "${MAX_TREE_BUDGET}" \
  --max-tokens "${MAX_TOKENS}" \
  --max-samples "${MAX_SAMPLES}" \
  --num-warmup-runs 1 \
  --profiler none \
  --max-model-len 3072 \
  --max-num-batched-tokens 16384
```

The [`vLLM fork`](https://github.com/snyhlxde1/vllm-jetflow) supports MATH-500 testing through the command above and HumanEval testing through `examples/offline_inference/dflash_profiling_humaneval_causal_tree_unit_kvlayout_dgx_pod.sh`.



## Results

### Engine Results

The optimized engine runs single-stream Qwen3-8B tree-speculative decoding with paged KV and CUDA graph verification. Local B200 bf16 measurements, using the production configuration below, closely align with the [`vLLM fork for JetFlow support`](https://github.com/snyhlxde1/vllm-jetflow).

| dataset | JetFlow engine TPS | accept_len |
|---|---:|---:|
| MATH-500 | **910.3 tok/s** | 9.60 |
| GSM8K | **791.0 tok/s** | 7.72 |
| HumanEval| **738.6 tok/s** | 7.25 |



### Tree Algorithms

Common algorithms exposed by `bench/reference/benchmark.py`:

| Algorithm | Purpose |
|---|---|
| `accum_logp` | robust breadth-first cumulative-logprob tree; default baseline |
| `top2gap_fanout` | adaptive fanout using the per-depth top-2 logprob gap |
| `task_router` | prompt/task-aware routing over tree shapes |
| `reasoning_router` | reasoning-pattern-aware routing |
| `class_histogram` | class-conditioned profile-guided tree shaping |
| `depth_rank_histogram` | offline profile table over `(depth, rank)` acceptance |


## Citation

```bibtex
@inproceedings{jetflow2026,
  title = {JetFlow: Breaking the Scaling Ceiling of Speculative Decoding with Parallel Tree Drafting},
  author = {Hu, Lanxiang and Feng, Zhaoxiang and Wu, Yulun and Yuan, Haoran and Zhao, Yujie and Qian, Yu-Yang and Wang, Bojun and Jiang, Daxin and Zhu, Yibo and Rosing, Tajana and Zhang, Hao},
  year = {2026},
  note = {Preprint}
}
```
