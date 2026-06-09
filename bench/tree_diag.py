"""Tree-decode diagnostic fingerprint for nano_vllm on GSM8K.

Prints a fork-style ``metrics_report.txt`` key=value block so nano and the vLLM
DFlash fork can be diffed directly on acceptance shape and tree shape.
"""
import argparse
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ptd.engine.llm import SamplingParams
from ptd.nano_vllm.engine import NanoEngine
from ptd.models.draft_head import load_draft_head
from ptd.draft_head_drafter import DraftHeadTreeDrafter


GSM8K_PROMPT = (
    "{question}\n"
    "Please reason step by step, and put your final answer within \\boxed{{}}."
)


def summarize_tree_diag(
    *,
    accept_lengths: list[int],
    tree_nodes_per_depth: list[int],
    output_tokens: int,
    num_samples: int,
    block_size: int,
) -> dict:
    rounds = len(accept_lengths)
    hist_counts = [0] * block_size
    for accept_len in accept_lengths:
        accepted_draft_len = int(accept_len) - 1
        if accepted_draft_len < 0 or accepted_draft_len >= block_size:
            raise ValueError(
                f"accept length {accept_len} maps to draft length "
                f"{accepted_draft_len}, outside [0, {block_size - 1}]"
            )
        hist_counts[accepted_draft_len] += 1

    denom = rounds if rounds else 1
    nodes = [int(v) for v in tree_nodes_per_depth]
    if len(nodes) < block_size:
        nodes = nodes + [0] * (block_size - len(nodes))
    else:
        nodes = nodes[:block_size]

    return {
        "output_tokens": int(output_tokens),
        "num_drafts": rounds,
        "num_samples": int(num_samples),
        "tokens_per_sample": (float(output_tokens) / num_samples if num_samples else 0.0),
        "acceptance_length": (sum(accept_lengths) / rounds if rounds else 0.0),
        "acceptance_length_histogram": [c / denom for c in hist_counts],
        "per_depth_acceptance_rate": [
            sum(1 for accept_len in accept_lengths if int(accept_len) - 1 > depth) / denom
            for depth in range(block_size - 1)
        ],
        "avg_tree_nodes_per_depth": [
            nodes[depth] / denom for depth in range(1, block_size)
        ],
    }


def format_metrics_report(
    metrics: dict,
    *,
    attention_backend: str,
    block_size: int,
    tree_width: int,
    budget: int,
    algo: str,
    drafter: str,
) -> str:
    lines = [
        "mode=dflash",
        "engine=nano_vllm",
        "prompt_set=gsm8k",
        "prompt_format=chat_template",
        f"attention_backend={attention_backend}",
        "head_type=causal",
        f"block_size={block_size}",
        f"tree_width={tree_width}",
        f"max_tree_budget={budget}",
        f"tree_draft={algo}",
        f"drafter={drafter}",
        f"num_samples={metrics['num_samples']}",
        f"output_tokens={metrics['output_tokens']}",
        f"tokens_per_sample={metrics['tokens_per_sample']:.6f}",
        f"num_drafts={metrics['num_drafts']}",
        f"acceptance_length={metrics['acceptance_length']:.6f}",
        "per_depth_acceptance_rate="
        + ",".join(f"{v:.6f}" for v in metrics["per_depth_acceptance_rate"]),
        "acceptance_length_histogram="
        + ",".join(f"{v:.6f}" for v in metrics["acceptance_length_histogram"]),
        "avg_tree_nodes_per_depth="
        + ",".join(f"{v:.2f}" for v in metrics["avg_tree_nodes_per_depth"]),
    ]
    return "\n".join(lines) + "\n"


def build_prompts(tokenizer, samples: int) -> list[str]:
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split="test")
    prompts = []
    for i in range(min(samples, len(ds))):
        prompt = GSM8K_PROMPT.format(question=ds[i]["question"])
        prompts.append(tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        ))
    return prompts


def build_drafter(args, eng: NanoEngine):
    head = load_draft_head(os.environ["PTD_DRAFT_HEAD"])
    tli, bs = head.target_layer_ids, head.block_size
    if args.drafter == "compiled":
        from ptd.draft_head_drafter import CompiledDraftHead
        drafter = CompiledDraftHead(head, target=eng.model, block_size=bs,
                                    target_layer_ids=tli, draft_shift=False)
    elif args.drafter == "graphed":
        from ptd.draft_head_drafter import GraphedDraftHead
        drafter = GraphedDraftHead(head, target=eng.model, block_size=bs,
                                   target_layer_ids=tli, draft_shift=False)
    else:
        drafter = DraftHeadTreeDrafter(head, target=eng.model, block_size=bs,
                                       target_layer_ids=tli, draft_shift=False)
    return drafter, tli, bs


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=64)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--budget", type=int, default=127)
    ap.add_argument("--tree-width", type=int, default=7)
    ap.add_argument("--algo", type=str, default="crossproduct")
    ap.add_argument("--drafter", choices=("eager", "compiled", "graphed"), default="eager")
    return ap.parse_args()


@torch.inference_mode()
def main():
    args = parse_args()
    backend = os.environ.get("NANO_BACKEND", "triton_paged_tree_cudagraph")
    eng = NanoEngine(
        "Qwen/Qwen3-8B",
        device="cuda",
        dtype=torch.bfloat16,
        attn_backend=backend,
        block_size=16,
    )
    drafter, target_layer_ids, block_size = build_drafter(args, eng)
    prompts = build_prompts(eng.tokenizer, args.samples)
    sp = SamplingParams(0.0, args.max_tokens)
    tree_kwargs = dict(
        block_size=block_size,
        tree_width=args.tree_width,
        budget=args.budget,
        algo=args.algo,
        target_layer_ids=target_layer_ids,
        sampling_params=sp,
        return_stats=True,
        tree_diag=True,
    )

    eng.generate_tree(prompts[0], drafter, **tree_kwargs)
    eng.generate_tree(prompts[0], drafter, **tree_kwargs)
    torch.cuda.synchronize()

    all_accept_lengths: list[int] = []
    tree_nodes_per_depth = [0] * block_size
    output_tokens = 0
    for prompt in prompts:
        out = eng.generate_tree(prompt, drafter, **tree_kwargs)
        output_tokens += len(out["token_ids"])
        all_accept_lengths.extend(out["accept_lengths"])
        for depth, count in enumerate(out["tree_nodes_per_depth"]):
            if depth < block_size:
                tree_nodes_per_depth[depth] += int(count)
    torch.cuda.synchronize()

    metrics = summarize_tree_diag(
        accept_lengths=all_accept_lengths,
        tree_nodes_per_depth=tree_nodes_per_depth,
        output_tokens=output_tokens,
        num_samples=len(prompts),
        block_size=block_size,
    )
    print(format_metrics_report(
        metrics,
        attention_backend=backend,
        block_size=block_size,
        tree_width=args.tree_width,
        budget=args.budget,
        algo=args.algo,
        drafter=args.drafter,
    ), end="")


if __name__ == "__main__":
    main()
