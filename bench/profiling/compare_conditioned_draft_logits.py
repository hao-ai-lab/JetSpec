"""P3 decision probe: does path-conditioning predict the correction?

For recorded rounds of a real tree decode, reconstruct the target-verified
accepted path and correction token bench-side, then compare the marginal draft
distribution against one conditioned head forward that feeds only the accepted
path prefix. Positions after the accepted path stay masked; the correction is
the answer and is never fed to the draft head.

    JETFLOW_FUSE_GEMMS=1 JETFLOW_BACKEND=triton_paged_tree_cudagraph_nogather ... \
      python bench/profiling/compare_conditioned_draft_logits.py --rounds 50 --samples 4 --budget 127
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from dataclasses import dataclass
from typing import Any, Sequence

import torch
from transformers import DynamicCache

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from bench.profiling.depth_rank_profile import (
    accepted_path_from_committed_tokens,
    rebuild_recorded_tree,
)
from bench.profiling.collect_tree_diagnostics import build_prompts
from jetflow.draft_head_adapter import DraftHeadTreeDrafter
from jetflow.core.llm import SamplingParams
from jetflow.models.draft_head import load_draft_head
from jetflow.inference_engine.engine import JetFlowEngine


BUCKETS = ("shallow", "mid", "deep")


@dataclass(frozen=True)
class ReconstructedRound:
    round_index: int
    record: dict[str, Any]
    accepted_tokens: list[int]
    accepted_path: list[int]
    correction_token: int

    @property
    def accepted_length(self) -> int:
        return len(self.accepted_tokens)


@dataclass(frozen=True)
class RoundScore:
    accepted_length: int
    cond_hit: bool
    marg_hit: bool
    cond_rank: int
    marg_rank: int
    cond_topk_hit: bool
    marg_topk_hit: bool

    @property
    def bucket(self) -> str:
        return length_bucket(self.accepted_length)


class ReseedRoundRecorder:
    """Drafter wrapper that captures the marginal round state for P3 scoring."""

    def __init__(self, drafter: Any, *, limit: int | None = None):
        self.drafter = drafter
        self.limit = limit
        self.records: list[dict[str, Any]] = []

    def __getattr__(self, name: str) -> Any:
        return getattr(self.drafter, name)

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
        if self.limit is None or len(self.records) < self.limit:
            if target_hidden is None:
                raise ValueError("reseed probe requires target_hidden from generate_tree")
            self.records.append(
                {
                    "root_token": int(context_ids[0, -1].detach().cpu().item()),
                    "draft_logits": logits.detach().float().cpu().clone(),
                    "context_ids": context_ids.detach().clone(),
                    "target_hidden": target_hidden.detach().clone(),
                }
            )
        return logits


def _as_int_list(values: Sequence[int] | torch.Tensor) -> list[int]:
    if torch.is_tensor(values):
        return [int(v) for v in values.detach().cpu().tolist()]
    return [int(v) for v in values]


def length_bucket(accepted_length: int) -> str:
    if accepted_length <= 4:
        return "shallow"
    if accepted_length <= 9:
        return "mid"
    return "deep"


def _rank_of_token(logits: torch.Tensor, token_id: int) -> int:
    if token_id < 0 or token_id >= logits.numel():
        raise ValueError(f"token id {token_id} outside logits width {logits.numel()}")
    order = torch.argsort(logits, descending=True)
    match = (order == int(token_id)).nonzero(as_tuple=False)
    if match.numel() == 0:
        raise ValueError(f"token id {token_id} not present in ranked logits")
    return int(match[0, 0].item())


def score_round_logits(
    marginal_logits: torch.Tensor,
    conditioned_logits_: torch.Tensor,
    accepted_tokens: Sequence[int],
    correction_token: int,
    *,
    top_k: int = 7,
) -> RoundScore:
    """Score one reconstructed round at correction depth L."""

    accepted_length = len(accepted_tokens)
    if accepted_length >= marginal_logits.shape[0]:
        raise ValueError(
            f"accepted length L={accepted_length} is outside draft horizon "
            f"{marginal_logits.shape[0]}"
        )
    if conditioned_logits_.shape[0] <= accepted_length:
        raise ValueError(
            f"conditioned logits depth {conditioned_logits_.shape[0]} cannot score L={accepted_length}"
        )

    marg_row = marginal_logits[accepted_length]
    cond_row = conditioned_logits_[accepted_length]
    cond_rank = _rank_of_token(cond_row, correction_token)
    marg_rank = _rank_of_token(marg_row, correction_token)
    return RoundScore(
        accepted_length=accepted_length,
        cond_hit=cond_rank == 0,
        marg_hit=marg_rank == 0,
        cond_rank=cond_rank,
        marg_rank=marg_rank,
        cond_topk_hit=cond_rank < top_k,
        marg_topk_hit=marg_rank < top_k,
    )


def reconstruct_rounds(
    *,
    records: Sequence[dict[str, Any]],
    token_ids: Sequence[int],
    accept_lengths: Sequence[int],
    block_size: int,
    tree_width: int,
    budget: int,
    limit: int | None = None,
) -> tuple[list[ReconstructedRound], Counter[str]]:
    """Recover accepted path tokens plus the correction for recorded rounds."""

    cursor = 1  # generate_tree emits one sampled token before tree rounds.
    rounds: list[ReconstructedRound] = []
    skipped: Counter[str] = Counter()
    tokens = _as_int_list(token_ids)

    for round_index, (record, raw_accept_len) in enumerate(zip(records, accept_lengths)):
        accept_len = int(raw_accept_len)
        if accept_len < 1:
            raise ValueError(f"round {round_index} accept_len must be >= 1; got {accept_len}")
        accepted_token_count = accept_len - 1
        available = tokens[cursor:]
        if len(available) < accept_len:
            skipped["truncated"] += 1
            break

        tree = rebuild_recorded_tree(
            record,
            block_size=block_size,
            tree_width=tree_width,
            budget=budget,
            device="cpu",
        )
        accepted_tokens = available[:accepted_token_count]
        try:
            accepted_path = accepted_path_from_committed_tokens(tree, accepted_tokens)
        except ValueError:
            skipped["unreconstructable"] += 1
            cursor += accept_len
            continue

        rounds.append(
            ReconstructedRound(
                round_index=round_index,
                record=record,
                accepted_tokens=accepted_tokens,
                accepted_path=accepted_path,
                correction_token=int(available[accepted_token_count]),
            )
        )
        cursor += accept_len
        if limit is not None and len(rounds) >= limit:
            break

    return rounds, skipped


def _conditioned_block_output_ids(
    fwd: Any,
    context_ids: torch.Tensor,
    path_tokens: Sequence[int] | torch.Tensor,
) -> torch.Tensor:
    block_size = fwd.block_size
    anchor = context_ids[0, -1].view(1, 1).to(fwd.device)
    max_path = block_size - 1
    path = torch.as_tensor(path_tokens, dtype=anchor.dtype, device=fwd.device).reshape(1, -1)
    if path.shape[1] > max_path:
        path = path[:, :max_path]
    mask_len = max_path - path.shape[1]
    mask_fill = torch.full(
        (1, mask_len),
        int(fwd.mask_token_id),
        dtype=anchor.dtype,
        device=fwd.device,
    )
    return torch.cat([anchor, path, mask_fill], dim=1)


def conditioned_logits(
    fwd: Any,
    context_ids: torch.Tensor,
    path_tokens: Sequence[int],
    target_hidden: torch.Tensor,
) -> torch.Tensor:
    """One head forward with noise = [anchor, accepted path, masks...]."""

    block_size = fwd.block_size
    block_output_ids = _conditioned_block_output_ids(fwd, context_ids, path_tokens)
    noise_embedding = fwd.target.model.embed_tokens(block_output_ids)
    ctx_len = target_hidden.shape[1]
    position_ids = torch.arange(ctx_len + block_size, device=fwd.device).unsqueeze(0)
    hidden = fwd.head(
        target_hidden=target_hidden.to(device=fwd.device, dtype=fwd.dtype),
        noise_embedding=noise_embedding,
        position_ids=position_ids,
        past_key_values=DynamicCache(),
        use_cache=False,
        is_causal=fwd.head.resolve_causal_head("auto"),
    )
    draft_slice = slice(0, block_size - 1) if fwd.draft_shift else slice(-block_size + 1, None)
    return fwd.target.lm_head(hidden[:, draft_slice, :])


def summarise_scores(scores: Sequence[RoundScore], skipped: Counter[str]) -> dict[str, Any]:
    def rates(rows: Sequence[RoundScore]) -> dict[str, float | int]:
        n = len(rows)
        return {
            "n": n,
            "cond_top1": sum(s.cond_hit for s in rows) / n if n else 0.0,
            "marg_top1": sum(s.marg_hit for s in rows) / n if n else 0.0,
            "cond_top7": sum(s.cond_topk_hit for s in rows) / n if n else 0.0,
            "marg_top7": sum(s.marg_topk_hit for s in rows) / n if n else 0.0,
        }

    by_bucket = {
        bucket: rates([score for score in scores if score.bucket == bucket])
        for bucket in BUCKETS
    }
    overall = rates(scores)
    lift = overall["cond_top7"] - overall["marg_top7"]
    verdict = "P3_BUILD" if lift >= 0.15 else "P3_KILL"
    return {
        "overall": overall,
        "by_bucket": by_bucket,
        "skipped": dict(skipped),
        "top7_lift": lift,
        "verdict": verdict,
    }


def print_summary(summary: dict[str, Any]) -> None:
    overall = summary["overall"]
    print(f"rounds scored: {overall['n']}")
    print(f"skipped rounds: {summary['skipped']}")
    print(
        "overall: "
        f"cond_top1={overall['cond_top1']:.3f} "
        f"marg_top1={overall['marg_top1']:.3f} "
        f"cond_top7={overall['cond_top7']:.3f} "
        f"marg_top7={overall['marg_top7']:.3f}"
    )
    print("by accepted length bucket:")
    for bucket in BUCKETS:
        row = summary["by_bucket"][bucket]
        label = {
            "shallow": "L<=4",
            "mid": "5<=L<=9",
            "deep": "L>=10",
        }[bucket]
        print(
            f"  {bucket} ({label}): n={row['n']} "
            f"cond_top1={row['cond_top1']:.3f} "
            f"marg_top1={row['marg_top1']:.3f} "
            f"cond_top7={row['cond_top7']:.3f} "
            f"marg_top7={row['marg_top7']:.3f}"
        )
    print(
        f"P3_VERDICT: {summary['verdict']} "
        f"(cond_top7 - marg_top7 = {summary['top7_lift']:.3f})"
    )


@torch.inference_mode()
def collect_scores(
    eng: Any,
    prompts: Sequence[Any],
    drafter: Any,
    fwd: Any,
    tree_kwargs: dict[str, Any],
    *,
    rounds: int,
    block_size: int,
    tree_width: int,
    budget: int,
) -> tuple[list[RoundScore], Counter[str]]:
    scores: list[RoundScore] = []
    skipped: Counter[str] = Counter()

    for prompt_index, prompt in enumerate(prompts):
        remaining = rounds - len(scores)
        if remaining <= 0:
            break
        recorder = ReseedRoundRecorder(drafter, limit=remaining)
        out = eng.generate_tree(prompt, recorder, **tree_kwargs)
        reconstructed, prompt_skipped = reconstruct_rounds(
            records=recorder.records,
            token_ids=out["token_ids"],
            accept_lengths=out["accept_lengths"],
            block_size=block_size,
            tree_width=tree_width,
            budget=budget,
            limit=remaining,
        )
        skipped.update(prompt_skipped)
        for round_target in reconstructed:
            accepted_length = round_target.accepted_length
            marginal = round_target.record["draft_logits"][0]
            if accepted_length >= marginal.shape[0]:
                skipped["out_of_horizon"] += 1
                continue
            cond = conditioned_logits(
                fwd,
                round_target.record["context_ids"],
                round_target.accepted_tokens,
                round_target.record["target_hidden"],
            )[0].float().cpu()
            scores.append(
                score_round_logits(
                    marginal,
                    cond,
                    round_target.accepted_tokens,
                    round_target.correction_token,
                )
            )
            if len(scores) >= rounds:
                break
        print(
            f"prompt {prompt_index + 1}/{len(prompts)}: "
            f"records={len(recorder.records)} counted={len(scores)} skipped={dict(skipped)}"
        )

    return scores, skipped


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--draft-head", default=None)
    ap.add_argument("--rounds", type=int, default=50)
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--budget", type=int, default=127)
    ap.add_argument("--tree-width", type=int, default=7)
    ap.add_argument(
        "--prompt-set",
        default="gsm8k",
        choices=["gsm8k", "math500", "humaneval", "aime"],
    )
    ap.add_argument("--attention-backend", default=None)
    return ap.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if args.rounds <= 0:
        raise SystemExit("--rounds must be positive")
    if args.samples <= 0:
        raise SystemExit("--samples must be positive")

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
    head = load_draft_head(args.draft_head or os.environ["JETFLOW_DRAFT_HEAD"])
    target_layer_ids, block_size = head.target_layer_ids, head.block_size
    drafter = DraftHeadTreeDrafter(
        head,
        target=eng.model,
        block_size=block_size,
        target_layer_ids=target_layer_ids,
        draft_shift=False,
    )
    prompts = build_prompts(eng.tokenizer, args.samples, prompt_set=args.prompt_set)
    sampling_params = SamplingParams(0.0, args.max_tokens)
    tree_kwargs = dict(
        sampling_params=sampling_params,
        block_size=block_size,
        tree_width=args.tree_width,
        budget=args.budget,
        target_layer_ids=target_layer_ids,
        return_stats=True,
    )

    scores, skipped = collect_scores(
        eng,
        prompts,
        drafter,
        drafter._fwd,
        tree_kwargs,
        rounds=args.rounds,
        block_size=block_size,
        tree_width=args.tree_width,
        budget=args.budget,
    )
    if not scores:
        raise SystemExit(f"no reconstructable/scorable rounds; skipped={dict(skipped)}")
    print_summary(summarise_scores(scores, skipped))


if __name__ == "__main__":
    main()
