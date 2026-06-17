"""Offline profiler for depth_rank_histogram (B2): per-(depth, rank) acceptance.

Runs crossproduct tree spec-decode over N calibration prompts and records, at
every expanded parent of depth d, the RANK (in the drafter's per-depth top-k) of
the token the target actually picks (its argmax). Aggregates

    depth_rank_accept[d][r] = P(target pick at a depth-d parent == drafter rank-r)

— the table `depth_rank_histogram` consumes to set its per-depth fanout cap. Pure
HF (no vLLM): one recompute verify forward per spec step, target_hidden threaded
like the engine so the DraftHead sees its real context.

    CUDA_VISIBLE_DEVICES=0 JETFLOW_DRAFT_HEAD=Snyhlxde/jetflow-qwen3-8b-distill-epoch6-3e-4-no-gamma \
      HF_HOME=/path/to/hf_cache HF_DATASETS_CACHE=/path/to/hf_cache/datasets \
      PYTHONPATH=. python bench/collect_profile.py --dataset gsm8k --samples 20 \
        --width 7 --budget 255 --out profiles/gsm8k_epoch6.json
"""
import argparse
import json
import os

import torch
from transformers import DynamicCache

from jetflow.core.llm import LLM, SamplingParams
from jetflow.models.draft_head import load_draft_head
from jetflow.draft_head_drafter import DraftHeadTreeDrafter
from jetflow.tree import get_algorithm, build_ancestor_matrix, tree_accept
from bench.benchmark import build_prompts


@torch.inference_mode()
def _profile_prompt(llm, drafter, prompt, block_size, tree_width, budget,
                    target_layer_ids, max_new, counts, totals):
    """Run crossproduct spec decode on one prompt, accumulating per-(depth, rank)
    acceptance into counts[d][r] / totals[d]. Mirrors generate_tree's recompute
    round (incl. target_hidden threading) so the drafter context is faithful."""
    D = block_size - 1
    K = tree_width
    dtype = llm.model.dtype
    neg = torch.finfo(dtype).min
    xprod = get_algorithm("crossproduct")
    committed = llm.tokenizer(prompt, return_tensors="pt").input_ids.to(llm.device)

    # prefill: seed target_hidden + the first committed token
    pos = torch.arange(committed.shape[1], device=llm.device).unsqueeze(0)
    logits, _, target_hidden = llm.runner.forward(
        committed, DynamicCache(), pos,
        output_hidden_states=True, target_layer_ids=target_layer_ids)
    first_tok = logits[:, -1:, :].argmax(-1)
    committed = torch.cat([committed, first_tok.view(1, 1)], dim=1)
    n_new = 1

    while n_new < max_new:
        draft_logits = drafter.propose_logits(committed, D, target_hidden=target_hidden).to(llm.device)
        # per-depth rank ordering (the ranks the histogram indexes)
        topk_tok = torch.topk(torch.log_softmax(draft_logits.squeeze(0), dim=-1), K, dim=-1).indices  # (D, K)
        rank_of = [{int(topk_tok[d, r]): r for r in range(K)} for d in range(D)]

        tree = xprod.build(int(committed[0, -1]), draft_logits, block_size, tree_width, budget, llm.device)
        N = tree.num_nodes
        prefix = committed[:, :-1]
        P = prefix.shape[1]
        seq = torch.cat([prefix, tree.token_ids.view(1, -1)], dim=1)
        depths = tree.depth.tolist()
        posv = torch.tensor([list(range(P)) + [P + d for d in depths]], device=llm.device)
        T = P + N
        allowed = torch.zeros(T, T, dtype=torch.bool, device=llm.device)
        if P > 0:
            allowed[:P, :P] = torch.tril(torch.ones(P, P, dtype=torch.bool, device=llm.device))
            allowed[P:, :P] = True
        allowed[P:, P:] = build_ancestor_matrix(tree).bool()
        mask = torch.where(allowed, torch.zeros((), dtype=dtype, device=llm.device),
                           torch.full((), neg, dtype=dtype, device=llm.device)).view(1, 1, T, T)
        logits, _, full_hidden = llm.runner.forward(
            seq, DynamicCache(), posv, attention_mask=mask,
            output_hidden_states=True, target_layer_ids=target_layer_ids)
        target_logits = logits[:, P:, :]                       # (1, N, V)
        posterior = target_logits.squeeze(0).argmax(-1).tolist()  # target pick at each node

        # record: at every depth-d parent (d < D), which rank did the target pick?
        for i in range(N):
            d = depths[i]
            if d >= D:
                continue
            totals[d] += 1
            r = rank_of[d].get(int(posterior[i]))
            if r is not None:
                counts[d][r] += 1

        accepted_path, acc, correction = tree_accept(tree, target_logits, 0.0)
        idx = list(range(P + 1)) + [P + j for j in accepted_path[1:]]
        target_hidden = full_hidden[:, idx, :]
        accepted = tree.token_ids[torch.tensor(accepted_path[1:], device=llm.device)] if acc > 0 \
            else torch.empty(0, dtype=tree.token_ids.dtype, device=llm.device)
        block = torch.cat([accepted, torch.tensor([correction], device=llm.device)])
        committed = torch.cat([committed, block.view(1, -1)], dim=1)
        n_new += int(block.numel())
        if int(block[-1]) in llm.eos_token_ids:
            break


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--draft-head", default=None)
    ap.add_argument("--dataset", default="gsm8k", choices=["gsm8k", "math500"])
    ap.add_argument("--samples", type=int, default=20)
    ap.add_argument("--width", type=int, default=7)
    ap.add_argument("--budget", type=int, default=255)
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--out", required=True, help="output JSON path for the profile table")
    args = ap.parse_args()
    head_path = args.draft_head or os.environ.get("JETFLOW_DRAFT_HEAD")
    if not head_path:
        raise SystemExit("set --draft-head or JETFLOW_DRAFT_HEAD")

    llm = LLM(args.model)
    head = load_draft_head(head_path)
    bs, tli = head.block_size, head.target_layer_ids
    drafter = DraftHeadTreeDrafter(head, target=llm.model, block_size=bs,
                                   target_layer_ids=tli, draft_shift=False)
    prompts = build_prompts(llm.tokenizer, args.dataset, args.samples)

    D, K = bs - 1, args.width
    counts = [[0] * K for _ in range(D)]
    totals = [0] * D
    for i, p in enumerate(prompts):
        _profile_prompt(llm, drafter, p, bs, args.width, args.budget, tli,
                        args.max_new, counts, totals)
        print(f"  profiled {i + 1}/{len(prompts)}")

    depth_rank_accept = [
        [(counts[d][r] / totals[d]) if totals[d] else 0.0 for r in range(K)]
        for d in range(D)
    ]
    table = {
        "depth_rank_accept": depth_rank_accept,
        "meta": {"model": args.model, "head": head_path, "dataset": args.dataset,
                 "samples": len(prompts), "width": K, "budget": args.budget,
                 "block_size": bs, "totals_per_depth": totals},
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(table, f, indent=2)
    print(f"\nwrote {args.out}")
    print("depth_rank_accept (rank-r acceptance per depth):")
    for d in range(D):
        row = " ".join(f"{depth_rank_accept[d][r]:.3f}" for r in range(K))
        print(f"  d{d:<2} (n={totals[d]:>5}): {row}")


if __name__ == "__main__":
    main()
