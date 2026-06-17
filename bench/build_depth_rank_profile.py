"""Collect B2 depth-rank acceptance profiles on JetFlow.

The output schema matches ``jetflow.tree.profile_guided.depth_rank_histogram``:

    {"depth_rank_accept": [[...]], "meta": {...}}

Each cell is:

    P(node accepted | a rank-r node existed at tree depth d+1)

where rank is the child index among a parent's children. Crossproduct appends
children in top-k order, so this is also the drafter's per-depth top-k rank.
The engine is left unchanged: this script records each round's root/logits,
rebuilds the deterministic crossproduct tree bench-side, and recovers the
accepted path from ``token_ids`` plus ``accept_lengths``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from jetflow.tree import get_algorithm
from jetflow.tree._core.base import DraftTree


@dataclass
class DepthRankProfileCounts:
    depths: int
    width: int
    presence: list[list[int]] = field(init=False)
    accepted: list[list[int]] = field(init=False)
    rounds_counted: int = 0
    skipped_rounds: int = 0

    def __post_init__(self) -> None:
        if self.depths <= 0:
            raise ValueError(f"depths must be positive; got {self.depths}")
        if self.width <= 0:
            raise ValueError(f"width must be positive; got {self.width}")
        self.presence = [[0 for _ in range(self.width)] for _ in range(self.depths)]
        self.accepted = [[0 for _ in range(self.width)] for _ in range(self.depths)]


class DraftRoundTreeRecorder:
    """Drafter wrapper that records enough to rebuild each round's tree."""

    def __init__(self, drafter: Any):
        self.drafter = drafter
        self.records: list[dict[str, Any]] = []

    def __getattr__(self, name: str) -> Any:
        return getattr(self.drafter, name)

    def reset(self) -> None:
        self.records.clear()

    def propose_logits(
        self,
        context_ids: torch.Tensor,
        depth: int,
        target_hidden: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        logits = self.drafter.propose_logits(
            context_ids,
            depth,
            target_hidden=target_hidden,
            **kwargs,
        )
        root_token = int(context_ids[0, -1].detach().cpu().item())
        self.records.append(
            {
                "root_token": root_token,
                "draft_logits": logits.detach().float().cpu().clone(),
            }
        )
        return logits


def _as_int_list(values: Sequence[int] | torch.Tensor) -> list[int]:
    if torch.is_tensor(values):
        return [int(v) for v in values.detach().cpu().tolist()]
    return [int(v) for v in values]


def _child_maps(tree: DraftTree) -> list[dict[int, int]]:
    if tree.child_maps is not None:
        return tree.child_maps
    token_ids = _as_int_list(tree.token_ids)
    parent_indices = _as_int_list(tree.parent_indices)
    maps: list[dict[int, int]] = [dict() for _ in range(tree.num_nodes)]
    for child_idx in range(1, tree.num_nodes):
        parent_idx = parent_indices[child_idx]
        if 0 <= parent_idx < tree.num_nodes:
            maps[parent_idx][token_ids[child_idx]] = child_idx
    tree.child_maps = maps
    return maps


def accepted_path_from_committed_tokens(
    tree: DraftTree,
    accepted_tokens: Iterable[int],
) -> list[int]:
    """Recover root-inclusive accepted node ids from committed draft tokens."""

    maps = _child_maps(tree)
    path = [0]
    current = 0
    for raw_token in accepted_tokens:
        token = int(raw_token)
        child_idx = maps[current].get(token)
        if child_idx is None:
            raise ValueError(
                f"token {token} is not a child of accepted path node {current}"
            )
        path.append(child_idx)
        current = child_idx
    return path


def _rank_by_node(tree: DraftTree) -> list[int | None]:
    parent_indices = _as_int_list(tree.parent_indices)
    ranks: list[int | None] = [None for _ in range(tree.num_nodes)]
    next_rank_by_parent: dict[int, int] = {}
    for node_idx in range(1, tree.num_nodes):
        parent_idx = parent_indices[node_idx]
        rank = next_rank_by_parent.get(parent_idx, 0)
        ranks[node_idx] = rank
        next_rank_by_parent[parent_idx] = rank + 1
    return ranks


def accumulate_round_profile(
    counts: DepthRankProfileCounts,
    tree: DraftTree,
    accepted_path: Sequence[int],
) -> None:
    """Count presence and accepted nodes for one verified tree round."""

    depths = _as_int_list(tree.depth)
    ranks = _rank_by_node(tree)
    accepted_nodes = {int(node_idx) for node_idx in accepted_path[1:]}
    for node_idx in range(1, tree.num_nodes):
        row = depths[node_idx] - 1
        if row < 0 or row >= counts.depths:
            continue
        rank = ranks[node_idx]
        if rank is None or rank < 0 or rank >= counts.width:
            continue
        counts.presence[row][rank] += 1
        if node_idx in accepted_nodes:
            counts.accepted[row][rank] += 1
    counts.rounds_counted += 1


def build_profile_table(
    counts: DepthRankProfileCounts,
    *,
    meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    depth_rank_accept = []
    for depth in range(counts.depths):
        row = []
        for rank in range(counts.width):
            denom = counts.presence[depth][rank]
            row.append((counts.accepted[depth][rank] / denom) if denom else 0.0)
        depth_rank_accept.append(row)

    out_meta = {
        "profile_definition": "P(node accepted | rank-r node existed at tree depth d+1)",
        "presence_counts": counts.presence,
        "accepted_counts": counts.accepted,
        "rounds_counted": counts.rounds_counted,
        "skipped_rounds": counts.skipped_rounds,
    }
    if meta:
        out_meta.update(meta)
    return {"depth_rank_accept": depth_rank_accept, "meta": out_meta}


def rebuild_recorded_tree(
    record: dict[str, Any],
    *,
    block_size: int,
    tree_width: int,
    budget: int,
    device: torch.device | str = "cpu",
) -> DraftTree:
    draft_logits = record["draft_logits"].to(device)
    return get_algorithm("crossproduct").build(
        int(record["root_token"]),
        draft_logits,
        block_size,
        tree_width,
        budget,
        torch.device(device),
    )


def accumulate_generation_profile(
    counts: DepthRankProfileCounts,
    *,
    records: Sequence[dict[str, Any]],
    token_ids: Sequence[int],
    accept_lengths: Sequence[int],
    block_size: int,
    tree_width: int,
    budget: int,
) -> None:
    """Post-process one JetFlow ``generate_tree`` output into profile counts."""

    cursor = 1  # generate_tree emits the first sampled token before tree rounds.
    if len(records) != len(accept_lengths):
        raise ValueError(
            f"recorded {len(records)} trees but got {len(accept_lengths)} accept lengths"
        )

    for round_index, (record, raw_accept_len) in enumerate(zip(records, accept_lengths)):
        accept_len = int(raw_accept_len)
        if accept_len < 1:
            raise ValueError(f"round {round_index} accept_len must be >= 1; got {accept_len}")
        accepted_token_count = accept_len - 1
        available = [int(v) for v in token_ids[cursor:]]
        if len(available) < accepted_token_count:
            counts.skipped_rounds += 1
            break

        tree = rebuild_recorded_tree(
            record,
            block_size=block_size,
            tree_width=tree_width,
            budget=budget,
            device="cpu",
        )
        try:
            path = accepted_path_from_committed_tokens(
                tree,
                available[:accepted_token_count],
            )
        except ValueError:
            counts.skipped_rounds += 1
            if len(available) < accept_len:
                break
            cursor += accept_len
            continue
        accumulate_round_profile(counts, tree, path)
        if len(available) < accept_len:
            break
        cursor += accept_len
        if cursor > len(token_ids):
            break


@torch.inference_mode()
def run_collection(
    eng: Any,
    prompts: Sequence[Any],
    drafter: Any,
    tree_kwargs: dict[str, Any],
    *,
    block_size: int,
    tree_width: int,
    budget: int,
) -> DepthRankProfileCounts:
    counts = DepthRankProfileCounts(depths=block_size - 1, width=tree_width)
    for prompt_index, prompt in enumerate(prompts):
        recorder = DraftRoundTreeRecorder(drafter)
        out = eng.generate_tree(prompt, recorder, **tree_kwargs)
        accumulate_generation_profile(
            counts,
            records=recorder.records,
            token_ids=out["token_ids"],
            accept_lengths=out["accept_lengths"],
            block_size=block_size,
            tree_width=tree_width,
            budget=budget,
        )
        print(
            f"profiled {prompt_index + 1}/{len(prompts)}: "
            f"rounds={len(recorder.records)} counted={counts.rounds_counted} "
            f"skipped={counts.skipped_rounds}"
        )
    return counts


def fail_if_skip_rate_too_high(counts: DepthRankProfileCounts) -> None:
    total_rounds = counts.rounds_counted + counts.skipped_rounds
    if total_rounds == 0:
        print("depth-rank profile skipped_rounds=0/0 (0.00%)")
        return

    skip_rate = counts.skipped_rounds / total_rounds
    print(
        f"depth-rank profile skipped_rounds={counts.skipped_rounds}/{total_rounds} "
        f"({skip_rate:.2%})"
    )
    if skip_rate > 0.05:
        raise SystemExit(
            f"depth-rank profile skipped {counts.skipped_rounds}/{total_rounds} "
            f"rounds ({skip_rate:.2%}) > 5%; aborting because path reconstruction "
            "is too noisy"
        )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--draft-head", default=None)
    ap.add_argument("--prompt-set", default="gsm8k", choices=["gsm8k", "math500", "humaneval", "aime"])
    ap.add_argument("--samples", type=int, default=64)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--tree-width", type=int, default=7)
    ap.add_argument("--budget", type=int, default=255)
    ap.add_argument("--session", action="store_true")
    ap.add_argument("--attention-backend", default=None)
    ap.add_argument("--out", required=True, help="output JSON path")
    return ap.parse_args()


@torch.inference_mode()
def main() -> None:
    from bench.tree_diag import build_drafter, build_prompts
    from jetflow.core.llm import SamplingParams
    from jetflow.inference_engine.engine import JetFlowEngine

    args = parse_args()
    backend = args.attention_backend or os.environ.get(
        "JETFLOW_BACKEND",
        "triton_paged_tree_cudagraph",
    )
    eng = JetFlowEngine(
        args.model,
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
        algo="crossproduct",
        target_layer_ids=target_layer_ids,
        sampling_params=sp,
        return_stats=True,
    )
    if args.session:
        tree_kwargs["session"] = True
        max_len = max(eng.tokenizer(p, return_tensors="pt").input_ids.shape[1] for p in prompts)
        tree_kwargs["session_prompt_capacity"] = ((max_len + 255) // 256) * 256

    eng.generate_tree(prompts[0], drafter, **tree_kwargs)
    eng.generate_tree(prompts[0], drafter, **tree_kwargs)
    torch.cuda.synchronize()

    counts = run_collection(
        eng,
        prompts,
        drafter,
        tree_kwargs,
        block_size=block_size,
        tree_width=args.tree_width,
        budget=args.budget,
    )
    fail_if_skip_rate_too_high(counts)
    torch.cuda.synchronize()

    table = build_profile_table(
        counts,
        meta={
            "engine": "JetFlow",
            "model": args.model,
            "head": args.draft_head or os.environ.get("JETFLOW_DRAFT_HEAD"),
            "prompt_set": args.prompt_set,
            "samples": len(prompts),
            "block_size": block_size,
            "tree_width": args.tree_width,
            "budget": args.budget,
            "max_tokens": args.max_tokens,
            "algo": "crossproduct",
            "attention_backend": backend,
            "drafter": "eager",
            "session": bool(args.session),
        },
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(table, indent=2) + "\n")
    print(f"wrote {out_path}")
    for depth, row in enumerate(table["depth_rank_accept"]):
        print(f"d{depth:02d}: " + " ".join(f"{value:.6f}" for value in row))


if __name__ == "__main__":
    main()
