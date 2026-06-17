"""Tree-decode diagnostic fingerprint for JetFlow on GSM8K.

Prints a fork-style ``metrics_report.txt`` key=value block so JetFlow and the vLLM
DFlash fork can be diffed directly on acceptance shape and tree shape.
"""
import argparse
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from jetflow.core.llm import SamplingParams
from jetflow.inference_engine.engine import JetFlowEngine
from jetflow.models.draft_head import load_draft_head
from jetflow.draft_head_drafter import DraftHeadTreeDrafter


GSM8K_PROMPT = (
    "{question}\n"
    "Please reason step by step, and put your final answer within \\boxed{{}}."
)

MATH_PROMPT = (
    "{problem}\n"
    "Please reason step by step, and put your final answer within \\boxed{{}}."
)

HUMANEVAL_PROMPT = (
    "Write a solution to the following problem and make sure that it "
    "passes the tests:\n```python\n{prompt}\n```"
)


def summarize_tree_diag(
    *,
    accept_lengths: list[int],
    tree_nodes_per_depth: list[int],
    output_tokens: int,
    num_samples: int,
    block_size: int,
    max_depth: int = None,
) -> dict:
    rounds = len(accept_lengths)
    depth_slots = block_size if max_depth is None else int(max_depth)
    if depth_slots <= 0:
        raise ValueError(f"max_depth must be positive; got {depth_slots}")
    hist_counts = [0] * depth_slots
    for accept_len in accept_lengths:
        accepted_draft_len = int(accept_len) - 1
        if accepted_draft_len < 0 or accepted_draft_len >= depth_slots:
            raise ValueError(
                f"accept length {accept_len} maps to draft length "
                f"{accepted_draft_len}, outside [0, {depth_slots - 1}]"
            )
        hist_counts[accepted_draft_len] += 1

    denom = rounds if rounds else 1
    nodes = [int(v) for v in tree_nodes_per_depth]
    if len(nodes) < depth_slots:
        nodes = nodes + [0] * (depth_slots - len(nodes))
    else:
        nodes = nodes[:depth_slots]

    return {
        "output_tokens": int(output_tokens),
        "num_drafts": rounds,
        "num_samples": int(num_samples),
        "tokens_per_sample": (float(output_tokens) / num_samples if num_samples else 0.0),
        "acceptance_length": (sum(accept_lengths) / rounds if rounds else 0.0),
        "acceptance_length_histogram": [c / denom for c in hist_counts],
        "per_depth_acceptance_rate": [
            sum(1 for accept_len in accept_lengths if int(accept_len) - 1 > depth) / denom
            for depth in range(depth_slots - 1)
        ],
        "avg_tree_nodes_per_depth": [
            nodes[depth] / denom for depth in range(1, depth_slots)
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
        "engine=JetFlow",
        f"prompt_set={metrics.get('prompt_set', 'gsm8k')}",
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


class DraftRoundTopKRecorder:
    def __init__(self, drafter, max_rounds: int):
        self.drafter = drafter
        self.max_rounds = int(max_rounds)
        self.records: list[dict] = []

    def __getattr__(self, name):
        return getattr(self.drafter, name)

    def reset(self):
        self.records.clear()

    def propose_logits(
        self,
        context_ids: torch.Tensor,
        depth: int,
        target_hidden: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:
        logits = self.drafter.propose_logits(
            context_ids,
            depth,
            target_hidden=target_hidden,
            **kwargs,
        )
        if len(self.records) < self.max_rounds:
            # generate_tree passes the same committed root to the tree builder
            # immediately after this call, so the wrapper can record it exactly.
            root_token = int(context_ids[0, -1].detach().cpu().item())
            self.records.append({
                "root_token": root_token,
                "draft_logits": logits.detach().float().cpu().clone(),
            })
        return logits


def format_draft_round_dump(
    records: list[dict],
    accept_lengths: list[int],
    *,
    tree_width: int,
) -> str:
    lines = []
    for round_index, record in enumerate(records):
        accepted_len = int(accept_lengths[round_index])
        logits = record["draft_logits"]
        logprobs = torch.log_softmax(logits, dim=-1)
        topk_lp, topk_tok = torch.topk(logprobs, tree_width, dim=-1)
        lines.append(
            f"[ROUND {round_index}] root_token={record['root_token']} "
            f"accepted_len={accepted_len}"
        )
        for depth in range(logits.shape[1]):
            toks = ",".join(str(int(v)) for v in topk_tok[0, depth].tolist())
            lines.append(f"[ROUND {round_index}] topk_tok[{depth}]={toks}")
        for depth in range(logits.shape[1]):
            vals = ",".join(f"{float(v):.6f}" for v in topk_lp[0, depth].tolist())
            lines.append(f"[ROUND {round_index}] topk_lp[{depth}]={vals}")
    return ("\n".join(lines) + "\n") if lines else ""


def run_tree_diag_measurement(
    eng,
    prompts: list,
    drafter,
    tree_kwargs: dict,
    *,
    block_size: int,
    tree_width: int,
    dump_first_rounds: int = 0,
) -> tuple[dict, str]:
    all_accept_lengths: list[int] = []
    depth_slots = block_size
    if tree_kwargs.get("max_tree_depth") is not None:
        depth_slots = int(tree_kwargs["max_tree_depth"]) + 1
    tree_nodes_per_depth = [0] * depth_slots
    output_tokens = 0
    dump_text = ""
    recorder = (
        DraftRoundTopKRecorder(drafter, dump_first_rounds)
        if dump_first_rounds > 0 else None
    )

    for prompt_index, prompt in enumerate(prompts):
        prompt_drafter = drafter
        if prompt_index == 0 and recorder is not None:
            recorder.reset()
            prompt_drafter = recorder
        out = eng.generate_tree(prompt, prompt_drafter, **tree_kwargs)
        output_tokens += len(out["token_ids"])
        all_accept_lengths.extend(out["accept_lengths"])
        for depth, count in enumerate(out["tree_nodes_per_depth"]):
            if depth < depth_slots:
                tree_nodes_per_depth[depth] += int(count)
        if prompt_index == 0 and recorder is not None:
            dump_text = format_draft_round_dump(
                recorder.records,
                out["accept_lengths"],
                tree_width=tree_width,
            )

    metrics = summarize_tree_diag(
        accept_lengths=all_accept_lengths,
        tree_nodes_per_depth=tree_nodes_per_depth,
        output_tokens=output_tokens,
        num_samples=len(prompts),
        block_size=block_size,
        max_depth=depth_slots,
    )
    return metrics, dump_text


def build_prompts(tokenizer, samples: int, prompt_set: str = "gsm8k") -> list[str]:
    from datasets import load_dataset

    if prompt_set == "gsm8k":
        ds = load_dataset("openai/gsm8k", "main", split="test")
        raw = [GSM8K_PROMPT.format(question=row["question"]) for row in ds]
    elif prompt_set == "math500":
        ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
        raw = [MATH_PROMPT.format(problem=row["problem"]) for row in ds]
    elif prompt_set == "aime":
        ds = load_dataset("HuggingFaceH4/aime_2024", split="train")
        raw = [MATH_PROMPT.format(problem=row["problem"]) for row in ds]
    elif prompt_set == "humaneval":
        ds = load_dataset("openai/openai_humaneval", split="test")
        raw = [HUMANEVAL_PROMPT.format(prompt=row["prompt"]) for row in ds]
    else:
        raise ValueError(f"unknown prompt set: {prompt_set}")
    prompts = []
    for prompt in raw[:samples]:
        prompts.append(tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        ))
    return prompts


def build_drafter(args, eng: JetFlowEngine):
    head = load_draft_head(os.environ["JETFLOW_DRAFT_HEAD"])
    tli, bs = head.target_layer_ids, head.block_size
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
    ap.add_argument("--session", action="store_true",
                    help="reuse the tree session (pool + captured graphs) across prompts")
    ap.add_argument("--prompt-set", default="gsm8k",
                    choices=["gsm8k", "math500", "humaneval", "aime"])
    ap.add_argument("--dump-first-rounds", type=int, default=0,
                    help="dump drafter top-k tokens/logprobs for the first K rounds of the first prompt")
    ap.add_argument("--profile-json", default=None,
                    help="profile_table JSON for profile-guided algos (bench/build_depth_rank_profile.py output)")
    ap.add_argument("--tau", type=float, default=None,
                    help="acceptance threshold kwarg for depth_rank_histogram")
    ap.add_argument("--extend-budget", type=int, default=None,
                    help="P1 ceiling raise: extension chain length (enables extend_kwargs)")
    ap.add_argument("--extend-gap", type=float, default=1.0,
                    help="P1 gate: mean top-2 gap threshold along the rank-1 chain")
    return ap.parse_args()


@torch.inference_mode()
def main():
    args = parse_args()
    backend = os.environ.get("JETFLOW_BACKEND", "triton_paged_tree_cudagraph")
    eng = JetFlowEngine(
        "Qwen/Qwen3-8B",
        device="cuda",
        dtype=torch.bfloat16,
        attn_backend=backend,
        block_size=16,
    )
    drafter, target_layer_ids, block_size = build_drafter(args, eng)
    prompts = build_prompts(eng.tokenizer, args.samples, prompt_set=args.prompt_set)
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
    if args.session:
        tree_kwargs["session"] = True
        # capacity = the longest prompt in the set (session guard is loud, not growing)
        max_len = max(eng.tokenizer(p, return_tensors="pt").input_ids.shape[1] for p in prompts)
        tree_kwargs["session_prompt_capacity"] = ((max_len + 255) // 256) * 256
    if args.tau is not None:
        tree_kwargs["algo_kwargs"] = {"tau": args.tau}
    if args.extend_budget is not None:
        tree_kwargs["extend_kwargs"] = {
            "gap_threshold": args.extend_gap,
            "ext_budget": args.extend_budget,
            "mode": "chain",
        }
        tree_kwargs["max_tree_depth"] = (block_size - 1) + args.extend_budget
    if args.profile_json is not None:
        import json
        with open(args.profile_json) as f:
            tree_kwargs["profile_table"] = json.load(f)

    eng.generate_tree(prompts[0], drafter, **tree_kwargs)
    eng.generate_tree(prompts[0], drafter, **tree_kwargs)
    torch.cuda.synchronize()

    metrics, dump_text = run_tree_diag_measurement(
        eng,
        prompts,
        drafter,
        tree_kwargs,
        block_size=block_size,
        tree_width=args.tree_width,
        dump_first_rounds=args.dump_first_rounds,
    )
    torch.cuda.synchronize()
    metrics["prompt_set"] = args.prompt_set
    if dump_text:
        print(dump_text, end="")
    print(format_metrics_report(
        metrics,
        attention_backend=backend,
        block_size=block_size,
        tree_width=args.tree_width,
        budget=args.budget,
        algo=args.algo,
        drafter="eager",
    ), end="")


if __name__ == "__main__":
    main()
