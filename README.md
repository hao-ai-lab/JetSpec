<p align="center">
  <img src="assets/jetspec_logo.png" alt="JetSpec" width="180">
</p>

<div align="center"><h1>JetSpec++: Foresight Parallel Drafting with Self Refinements</h1></div>

**JetSpec++** is a lossless speculative decoding method that hides drafting behind verification. The draft model runs on a second GPU, drafts the next block before the target finishes verifying the current one, and refines its draft while it waits.

<details open>
<summary><strong>JetSpec++</strong></summary>

[**Paper**](https://arxiv.org/abs/2610.19792) | [**Models**](https://huggingface.co/JetSpec)

<p align="center"><img src="assets/jetspec_plus_plus_pipeline.png" alt="JetSpec++ overlaps foresight drafting and refinement passes with verification" width="760"></p>

</details>

<details>
<summary><strong>JetSpec</strong></summary>

[**Paper**](https://arxiv.org/abs/2606.18394) | [**Blog**](https://haoailab.com/blogs/parallel-tree-decoding/) | [**Webpage**](https://jetspec-project.github.io/jetspec-web/) | [**Models**](https://huggingface.co/JetSpec)

Causal parallel tree drafting: a draft head proposes a token tree in one pass, and the target verifies the whole tree in one forward. Usage, engine results and tree algorithms are in the [JetSpec README](jetspec/methods/jetspec/README.md).

<p align="center"><img src="assets/end_to_end_speedup_barplot_refined_v2_page.png" alt="JetSpec end-to-end speedup over autoregressive decoding on Qwen3-8B" width="760"></p>

</details>

## Supported Models

| Target | Checkpoint | GPUs |
|---|---|---|
| Qwen3-8B | [`JetSpec/jetspec-plus-plus-qwen3-8b`](https://huggingface.co/JetSpec/jetspec-plus-plus-qwen3-8b) | A100, B200, GB200 |
| Qwen3.8-27B (preview) | [`JetSpec/jetspec-plus-plus-qwen3p8-27b`](https://huggingface.co/JetSpec/jetspec-plus-plus-qwen3p8-27b) | A100, B200 |

JetSpec draft heads for six targets are listed in the [JetSpec README](jetspec/methods/jetspec/README.md#model-weights).

## Installation

JetSpec++ runs on Linux with two GPUs that have peer access. Install the measured environment for your platform, then the package:

```bash
git clone https://github.com/hao-ai-lab/JetSpec.git && cd JetSpec
pip install --no-deps -r jetspec/engines/jetspec_plus_plus/requirements-x86_64.lock
pip install --no-deps -e .
```

On GB200, use `requirements-gb200-aarch64.lock` with `--extra-index-url https://download.pytorch.org/whl/cu128`.

## Quick Start

```bash
python -m jetspec.engines.jetspec_plus_plus generate --model qwen3-8b \
    "How many positive whole-number divisors does 196 have?"
```

The command downloads the target and the checkpoint, picks the schedule for your GPU, and prints the answer with its throughput.

## Evaluation

Speedup over autoregressive decoding on your GPU, on GSM8K, MATH-500, AIME25, HumanEval, MBPP, LiveCodeBench or MT-Bench:

```bash
python bench/jetspec_plus_plus/speedup.py --model qwen3-8b --dataset math500
```

The paper's Table 3 on a GB200 pair, against DSpark:

```bash
python bench/jetspec_plus_plus/table3.py --model qwen3-8b
python bench/jetspec_plus_plus/table3.py --model qwen3.8-27b
```

[`TABLE3.md`](bench/jetspec_plus_plus/TABLE3.md) describes the protocol and the other baselines.

## For Coding Agents

[`AGENTS.md`](AGENTS.md) describes every command, option and component in this repository.

## Citation

```bibtex
@article{feng2026jetspecpp,
  title   = {JetSpec++: Foresight Parallel Drafting with Self Refinements},
  author  = {Feng, Zhaoxiang and Hu, Lanxiang and Yuan, Haoran and Qian, Yu-Yang and Yang, Jihui and Zhao, Peng and Zhu, Yibo and Rosing, Tajana and Zhang, Hao},
  journal = {arXiv preprint arXiv:2610.19792},
  year    = {2026}
}

@article{hu2026jetspec,
  title   = {JetSpec: Breaking the Scaling Ceiling of Speculative Decoding with Parallel Tree Drafting},
  author  = {Hu, Lanxiang and Feng, Zhaoxiang and Wu, Yulun and Yuan, Haoran and Zhao, Yujie and Qian, Yu-Yang and Wang, Bojun and Zhao, Peng and Jiang, Daxin and Zhu, Yibo and Rosing, Tajana and Zhang, Hao},
  journal = {arXiv preprint arXiv:2606.18394},
  year    = {2026}
}
```
