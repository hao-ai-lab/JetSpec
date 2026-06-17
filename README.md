<p align="center">
  <img src="assets/jetflow_icon.png" alt="JetFlow" width="180" align="center">
</p>

<div align="center"><h1>JetFlow: Parallel Tree Drafting</h1></div>

<p align="center">
  <a href="https://github.com/snyhlxde1/parallel-tree-decoding">Code</a> ·
  <a href="https://huggingface.co/JetFlow">Hugging Face</a> ·
  <a href="https://github.com/aaronzhfeng/jetflow-web">Project Page</a> ·
  <a href="#citation">BibTeX</a>
</p>

JetFlow is a lightweight implementation of **parallel tree-speculative decoding** for LLM inference. A causal-parallel draft head proposes a scored token tree, and the frozen target model verifies the whole tree in one forward pass under a tree-causal attention mask. The accepted path is selected from the target's own logits, so decoding is lossless by construction.

This repository contains both a correctness-first HuggingFace reference core and an optimized JetFlow inference engine with paged KV, custom Triton tree attention, compiled verification, CUDA graphs, and benchmark scripts for Qwen3-8B.

## Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Model Weights](#model-weights)
- [Usage](#usage)
  - [HF Reference Core](#hf-reference-core)
  - [JetFlow Inference Engine](#jetflow-inference-engine)
  - [Benchmarks](#benchmarks)
- [Architecture](#architecture)
- [Results](#results)
- [Tree Algorithms](#tree-algorithms)
- [Testing](#testing)
- [Citation](#citation)

## Introduction

Speculative decoding is fast when the target accepts many draft tokens and drafting remains cheap. Prior heads often trade off those two terms: autoregressive drafters condition on each path but pay a forward pass per depth, while block-diffusion drafters draft many positions in one pass but score branches independently.

JetFlow keeps the one-pass drafting efficiency and restores causal branch conditioning. The draft head reads fused hidden states from the frozen target and emits per-depth logits in a single parallel pass. Tree construction spends a draft budget over high-probability branches, and the target verifies every node in one batched/tree-masked forward.

The project has two execution paths:

- `jetflow/core/`: a small HuggingFace-based reference core for correctness and benchmark alignment.
- `jetflow/inference_engine/`: the optimized owned engine used for wall-clock throughput.

## Installation

Create an environment and install the package:

```bash
cd /root/workspace/parallel-tree-decoding
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
| Draft head | `Snyhlxde/jetflow-qwen3-8b-distill-epoch6-3e-4-no-gamma` |

Most benchmark and diagnostic scripts read the draft head from `JETFLOW_DRAFT_HEAD`:

```bash
export JETFLOW_DRAFT_HEAD=Snyhlxde/jetflow-qwen3-8b-distill-epoch6-3e-4-no-gamma
```

You can also pass `--draft-head` directly to scripts that expose the flag. If you see `set --draft-head or JETFLOW_DRAFT_HEAD`, the environment variable is missing.

## Usage

### HF Reference Core

The reference core is intentionally small: HuggingFace model, `DynamicCache`, explicit positions, and tree verification.

```python
from jetflow import LLM, SamplingParams

llm = LLM("Qwen/Qwen3-8B", attn_implementation="flash_attention_2")
out = llm.generate(
    "The three primary colors are",
    SamplingParams(temperature=0.0, max_new_tokens=64),
)
print(out["text"])
```

Raw HF + FA2 single-token decode smoke test:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python bench/hf_fa2_kv_decode.py \
  --model Qwen/Qwen3-8B \
  --max-new 256 \
  --warmup 1
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
head = load_draft_head("Snyhlxde/jetflow-qwen3-8b-distill-epoch6-3e-4-no-gamma")
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

### Benchmarks

HF reference benchmark:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. \
JETFLOW_DRAFT_HEAD=Snyhlxde/jetflow-qwen3-8b-distill-epoch6-3e-4-no-gamma \
python bench/benchmark.py \
  --model Qwen/Qwen3-8B \
  --attn-implementation flash_attention_2 \
  --tree-attn triton \
  --dataset gsm8k \
  --samples 64 \
  --algos crossproduct \
  --width 7 \
  --budget 255 \
  --max-new 256 \
  --warmup-samples-per-rank 1
```

Optimized JetFlow wall-clock benchmark:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. \
JETFLOW_FUSE_GEMMS=1 \
JETFLOW_BACKEND=triton_paged_tree_cudagraph_nogather \
JETFLOW_DRAFT_HEAD=Snyhlxde/jetflow-qwen3-8b-distill-epoch6-3e-4-no-gamma \
python bench/tps_walltime.py \
  --prompt-set gsm8k \
  --samples 64 \
  --max-tokens 2048 \
  --budget 127 \
  --session
```

For 8 GPUs, use `torchrun` with the same command; benchmark scripts shard prompts across ranks.

## Architecture

```text
jetflow/
  core/                 # HuggingFace reference core: LLM, ModelRunner, sampler, tree attention hook
  inference_engine/     # optimized JetFlow engine: paged KV, scheduler, kernels, CUDA graphs
  tree/                 # engine-agnostic tree construction and acceptance
  models/               # target/draft-head loading and model utilities
  draft.py              # simple drafters used in tests and correctness gates
  draft_head_drafter.py # trained draft-head adapter
```

The dependency direction is strict: engines consume `jetflow.tree`, while tree algorithms never import either execution backend. Import tree behavior through the public `jetflow.tree` API rather than `jetflow.tree._core`.

## Results

The optimized engine runs single-stream Qwen3-8B tree-speculative decoding with paged KV and CUDA graph verification. Headline wall-clock measurements on B200, bf16, budget 127, width 7, trained epoch6 distill head, 2048-token generation window:

| dataset | JetFlow engine | reference fork | ratio | accept_len |
|---|---:|---:|---:|---:|
| HumanEval (164) | **738.6 tok/s** | 718.9 tok/s | **1.03x** | 7.25 |
| MATH-500 (100) | **910.3 tok/s** | 930.3 tok/s | 0.98x | 9.60 |
| GSM8K (64) | **791.0 tok/s** | 820.7 tok/s | 0.96x | 7.72 |

Production configuration:

```text
JETFLOW_FUSE_GEMMS=1
attn_backend="triton_paged_tree_cudagraph_nogather"
session=True
```

The public JetFlow project page reports paper-level speedups up to **9.64x** on MATH-500 for Qwen3-8B greedy decoding at budget 256, with accepted length `tau=10.76`. This repo reports measured implementation numbers and includes scripts to reproduce the local B200 runs.

Losslessness note: fp32 paths are token-identical to an SDPA oracle. In bf16, tree speculative decoding is lossless-by-construction because committed tokens come from the target verifier, but exact token equality against an AR KV-cache baseline can differ at borderline argmaxes due to kernel reduction order.

## Tree Algorithms

Common algorithms exposed by `bench/benchmark.py`:

| Algorithm | Purpose |
|---|---|
| `crossproduct` | robust breadth-first cumulative-logprob tree; default baseline |
| `top2gap_fanout` | adaptive fanout using the per-depth top-2 logprob gap |
| `task_router` | prompt/task-aware routing over tree shapes |
| `reasoning_router` | reasoning-pattern-aware routing |
| `class_histogram` | class-conditioned profile-guided tree shaping |
| `depth_rank_histogram` | offline profile table over `(depth, rank)` acceptance |

Same-day production picks:

| workload | pick | tok/s | accept_len |
|---|---|---:|---:|
| GSM8K | `top2gap_fanout`, budget 63 | **827.0** | 7.43 |
| MATH-500 | `top2gap_fanout`, budget 63 | **931.5** | 9.38 |
| HumanEval | `crossproduct`, budget 127 | **740.5** | 7.25 |

## Testing

Fast local validation:

```bash
PYTHONPATH=. pytest -q
```

Full real-model gates require CUDA and an explicit model:

```bash
CUDA_VISIBLE_DEVICES=0 \
JETFLOW_TEST_MODEL=Qwen/Qwen3-8B \
JETFLOW_DRAFT_HEAD=Snyhlxde/jetflow-qwen3-8b-distill-epoch6-3e-4-no-gamma \
PYTHONPATH=. pytest tests/
```

Recent remap validation:

```text
234 passed, 10 skipped
```

## Citation

```bibtex
@inproceedings{jetflow2026,
  title = {JetFlow: Breaking the Scaling Ceiling of Speculative Decoding with Parallel Tree Drafting},
  author = {Hu, Lanxiang and Feng, Zhaoxiang and Wu, Yulun and Yuan, Haoran and Zhao, Yujie and Qian, Yu-Yang and Wang, Bojun and Jiang, Daxin and Zhu, Yibo and Rosing, Tajana and Zhang, Hao},
  year = {2026},
  note = {Preprint}
}
```
