"""Run JetSpec++ correction layers through the JetSpec tree engine."""
from __future__ import annotations

import argparse
import os

import torch

from jetspec.core.llm import LLM, SamplingParams
from jetspec.correction.attention_layer_adapter import (
    AttentionLayerDrafter,
    assert_live_attention_layer,
    load_attention_layer_head,
    load_attention_layer_target_weights,
)
from jetspec.correction.markov_head_adapter import (
    MarkovHeadDrafter,
    assert_live_markov_bias,
    load_markov_head,
    load_markov_head_target_weights,
)
from jetspec.models.draft_head import assert_matching_rotary


_MATH_FORMAT = (
    "{problem}\nPlease reason step by step, and put your final answer within "
    "\\boxed{{}}."
)


def _load_prompt_bank(prompt_set: str, samples: int) -> list[str]:
    from datasets import load_dataset

    if prompt_set == "gsm8k":
        dataset = load_dataset("openai/gsm8k", "main", split="test")
        template = (
            "{question}\nPlease reason step by step, and put your final answer "
            "within \\boxed{{}}."
        )
        return [
            template.format(question=dataset[index]["question"])
            for index in range(min(samples, len(dataset)))
        ]
    if prompt_set == "math500":
        dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
        return [
            _MATH_FORMAT.format(problem=dataset[index]["problem"])
            for index in range(min(samples, len(dataset)))
        ]
    if prompt_set == "aime":
        dataset = load_dataset("HuggingFaceH4/aime_2024", split="train")
        return [
            _MATH_FORMAT.format(problem=dataset[index]["problem"])
            for index in range(min(samples, len(dataset)))
        ]
    if prompt_set == "aime25":
        dataset = load_dataset("yentinglin/aime_2025", split="train")
        return [
            _MATH_FORMAT.format(problem=dataset[index]["problem"])
            for index in range(min(samples, len(dataset)))
        ]
    if prompt_set == "humaneval":
        dataset = load_dataset("openai/openai_humaneval", split="test")
        template = (
            "Write a solution to the following problem and make sure that it "
            "passes the tests:\n\u0060\u0060\u0060python\n{prompt}\n\u0060\u0060\u0060"
        )
        return [
            template.format(prompt=dataset[index]["prompt"])
            for index in range(min(samples, len(dataset)))
        ]
    if prompt_set == "mbpp":
        dataset = load_dataset(
            "google-research-datasets/mbpp",
            "full",
            split="test",
        )
        template = (
            "You are an expert Python programmer. Write a solution to the "
            "following task and make sure it passes the tests:\n{text}\n"
            "Your code should pass these tests:\n{tests}"
        )
        return [
            template.format(
                text=dataset[index]["text"],
                tests="\n".join(dataset[index]["test_list"]),
            )
            for index in range(min(samples, len(dataset)))
        ]
    if prompt_set == "livecodebench":
        dataset = load_dataset(
            "livecodebench/code_generation_lite",
            version_tag="release_v5",
            split="test",
            trust_remote_code=True,
        )
        template = (
            "Write a Python solution to the following competitive-programming "
            "problem:\n\u0060\u0060\u0060\n{question}\n\u0060\u0060\u0060"
        )
        return [
            template.format(question=dataset[index]["question_content"])
            for index in range(min(samples, len(dataset)))
        ]
    if prompt_set == "mt_bench":
        dataset = load_dataset("HuggingFaceH4/mt_bench_prompts", split="train")
        return [
            dataset[index]["prompt"][0]
            for index in range(min(samples, len(dataset)))
        ]
    raise ValueError(f"unknown prompt set: {prompt_set}")


def _load_correction(args, llm, device: str, attn_implementation: str):
    checkpoint = args.draft_head
    if checkpoint is None and args.correction in {"attention_layer", "none"}:
        checkpoint = os.environ.get("JETSPEC_ATTENTION_LAYER_HEAD")
    if checkpoint is None:
        raise SystemExit(
            "set --draft-head (or JETSPEC_ATTENTION_LAYER_HEAD for the "
            "attention-layer checkpoint)"
        )

    if args.correction in {"attention_layer", "none"}:
        head, layout = load_attention_layer_head(
            checkpoint,
            device=device,
            attn_implementation=attn_implementation,
        )
        load_attention_layer_target_weights(
            llm.model,
            checkpoint,
            arm=args.correction,
        )
        assert_matching_rotary(head, llm.model, arm=args.correction)
        assert_live_attention_layer(head.attention_layer, arm=args.correction)
        drafter = AttentionLayerDrafter(
            head,
            llm.model,
            layout.native_block,
            layout.target_layer_ids,
        )
        algo_kwargs = {
            "attention_layer": drafter.attention_layer,
            "embed_tokens": drafter.embed_tokens,
            "lm_head": drafter.lm_head,
            "block_positions": lambda: drafter.block_positions,
            "mask_token_id": drafter.mask_token_id,
            "attention_layer_off": args.correction == "none",
        }
        return (
            drafter,
            layout,
            "attention_layer_accum_logp",
            algo_kwargs,
        )

    head, layout = load_markov_head(
        checkpoint,
        device=device,
        attn_implementation=attn_implementation,
    )
    load_markov_head_target_weights(
        llm.model,
        checkpoint,
        arm=args.correction,
    )
    assert_matching_rotary(head, llm.model, arm=args.correction)
    assert_live_markov_bias(head.markov_head, arm=args.correction)
    drafter = MarkovHeadDrafter(head, llm.model, layout)
    return (
        drafter,
        layout,
        "markov_head_accum_logp",
        {"markov_head": drafter.markov_head},
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--draft-head", default=None)
    parser.add_argument(
        "--correction",
        choices=["attention_layer", "markov_head", "none"],
        default="attention_layer",
    )
    parser.add_argument(
        "--prompt-set",
        choices=[
            "gsm8k",
            "math500",
            "aime",
            "humaneval",
            "aime25",
            "mbpp",
            "livecodebench",
            "mt_bench",
        ],
        default="gsm8k",
    )
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--max-new", type=int, default=1024)
    parser.add_argument("--tree-width", type=int, default=7)
    parser.add_argument("--budget", type=int, default=64)
    parser.add_argument("--tree-attn", choices=["sdpa", "triton"], default="sdpa")
    args = parser.parse_args()

    torch.set_float32_matmul_precision("high")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    llm = LLM(args.model, device=device, attn_implementation="sdpa")
    resolved_attn = getattr(llm.model, "_jetspec_attn_implementation", "sdpa")
    drafter, layout, algo, algo_kwargs = _load_correction(
        args,
        llm,
        device,
        resolved_attn,
    )

    prompts = _load_prompt_bank(args.prompt_set, args.samples)
    sampling = SamplingParams(temperature=0.0, max_new_tokens=args.max_new)
    for index, raw_prompt in enumerate(prompts):
        prompt = llm.tokenizer.apply_chat_template(
            [{"role": "user", "content": raw_prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        drafter.reset_cache()
        output = llm.generate_tree(
            prompt,
            drafter,
            block_size=layout.engine_block,
            tree_width=args.tree_width,
            budget=args.budget,
            algo=algo,
            algo_kwargs=algo_kwargs,
            target_layer_ids=layout.target_layer_ids,
            sampling_params=sampling,
            tree_attn=args.tree_attn,
        )
        print(f"[{index}] {output['text']}")
        print(f"[{index}] tokens-per-forward={output['tpf']:.2f}")


if __name__ == "__main__":
    main()
