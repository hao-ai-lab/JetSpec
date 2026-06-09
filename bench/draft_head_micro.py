"""Micro-benchmark for the bucketed DraftHead wrappers.

Run on the GPU box, for example:

    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python bench/draft_head_micro.py \
      --rounds 100 --ctx-lens 512,1024,2048

The benchmark times one `propose_logits` call for the eager DraftHead path, the
bucketed compiled wrapper, and the CUDA-graph wrapper when CUDA is available.
"""
import argparse
import os
import sys
import time

import torch
from transformers import DynamicCache

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ptd.draft_head_drafter import CompiledDraftHead, DraftHeadTreeDrafter, GraphedDraftHead
from ptd.engine.llm import LLM
from ptd.models.draft_head import load_draft_head


DEFAULT_HEAD = "Snyhlxde/ptd-qwen3-8b-distill-epoch6-3e-4-no-gamma"


def _dtype(name: str) -> torch.dtype:
    table = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    try:
        return table[name.lower()]
    except KeyError as exc:
        raise SystemExit(f"unknown dtype {name!r}; choose one of {sorted(table)}") from exc


def _sync(device: str) -> None:
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


@torch.inference_mode()
def _make_inputs(llm, ctx_len: int, target_layer_ids):
    vocab = int(llm.model.config.vocab_size)
    context_ids = torch.arange(1, ctx_len + 2, dtype=torch.long, device=llm.device)
    context_ids = context_ids.remainder(vocab).view(1, -1)
    prefix = context_ids[:, :ctx_len]
    pos = torch.arange(ctx_len, device=llm.device).unsqueeze(0)
    _, _, target_hidden = llm.runner.forward(
        prefix,
        DynamicCache(),
        pos,
        output_hidden_states=True,
        target_layer_ids=target_layer_ids,
    )
    return context_ids, target_hidden


@torch.inference_mode()
def _time_calls(fn, rounds: int, device: str):
    _sync(device)
    t0 = time.perf_counter()
    out = None
    for _ in range(rounds):
        out = fn()
    _sync(device)
    return out, (time.perf_counter() - t0) * 1000.0 / rounds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--draft-head", default=DEFAULT_HEAD)
    ap.add_argument("--rounds", type=int, default=100)
    ap.add_argument("--ctx-lens", default="512,1024,2048")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--rtol", type=float, default=1e-3)
    ap.add_argument("--atol", type=float, default=1e-3)
    args = ap.parse_args()

    dtype = _dtype(args.dtype)
    ctx_lens = tuple(int(x) for x in args.ctx_lens.split(",") if x.strip())
    if not ctx_lens:
        raise SystemExit("--ctx-lens must contain at least one length")

    llm = LLM(args.model, device=args.device, dtype=dtype)
    head = load_draft_head(args.draft_head, device=args.device, dtype=dtype)
    block_size = int(head.block_size)
    depth = block_size - 1
    target_layer_ids = head.target_layer_ids
    buckets = tuple(sorted(set(ctx_lens)))

    eager = DraftHeadTreeDrafter(
        head,
        target=llm.model,
        block_size=block_size,
        target_layer_ids=target_layer_ids,
        draft_shift=False,
    )
    compiled = CompiledDraftHead(
        head,
        target=llm.model,
        block_size=block_size,
        target_layer_ids=target_layer_ids,
        draft_shift=False,
        ctx_buckets=buckets,
    )
    graphed = None
    if str(args.device).startswith("cuda"):
        graphed = GraphedDraftHead(
            head,
            target=llm.model,
            block_size=block_size,
            target_layer_ids=target_layer_ids,
            draft_shift=False,
            ctx_buckets=buckets,
        )

    print(f"model={args.model}")
    print(f"draft_head={args.draft_head}")
    print(f"device={args.device} dtype={dtype} block_size={block_size} rounds={args.rounds}")
    print("ctx_len,bucket,eager_ms,compiled_ms,graph_ms,compiled_allclose,graph_allclose,max_abs_compiled,max_abs_graph")

    for ctx_len in ctx_lens:
        context_ids, target_hidden = _make_inputs(llm, ctx_len, target_layer_ids)

        eager_out = eager.propose_logits(context_ids, depth, target_hidden=target_hidden)
        compiled_out = compiled.propose_logits(context_ids, depth, target_hidden=target_hidden)
        graph_out = graphed.propose_logits(context_ids, depth, target_hidden=target_hidden) if graphed else None

        compiled_ok = torch.allclose(eager_out, compiled_out, rtol=args.rtol, atol=args.atol)
        max_abs_compiled = (eager_out - compiled_out).abs().max().item()
        graph_ok = ""
        max_abs_graph = ""
        if graph_out is not None:
            graph_ok = str(torch.allclose(eager_out, graph_out, rtol=args.rtol, atol=args.atol))
            max_abs_graph = f"{(eager_out - graph_out).abs().max().item():.6g}"

        _, eager_ms = _time_calls(
            lambda: eager.propose_logits(context_ids, depth, target_hidden=target_hidden),
            args.rounds,
            args.device,
        )
        _, compiled_ms = _time_calls(
            lambda: compiled.propose_logits(context_ids, depth, target_hidden=target_hidden),
            args.rounds,
            args.device,
        )
        graph_ms = ""
        if graphed is not None:
            _, gm = _time_calls(
                lambda: graphed.propose_logits(context_ids, depth, target_hidden=target_hidden),
                args.rounds,
                args.device,
            )
            graph_ms = f"{gm:.4f}"

        print(
            f"{ctx_len},{compiled.bucket_for_ctx_len(ctx_len)},"
            f"{eager_ms:.4f},{compiled_ms:.4f},{graph_ms},"
            f"{compiled_ok},{graph_ok},{max_abs_compiled:.6g},{max_abs_graph}"
        )


if __name__ == "__main__":
    main()
