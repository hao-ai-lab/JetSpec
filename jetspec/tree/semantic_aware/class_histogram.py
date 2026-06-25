"""class_histogram — per-task fanout from a per-class acceptance histogram.

The data-driven sibling of `task_router`: classify the prompt into a task
class, then drive per-depth fanout from that class's acceptance histogram
rather than a hand-tuned template:

    task ← classify(prompt);   b_per_depth ← histogram[task]

When an offline-collected `profile_table["per_class_histograms"][task]` is
supplied, the cap at each depth is the largest rank whose acceptance rate is
positive (per-(depth, rank) structure a single global signal can't see). Its
niche is a multi-task serving mix (math + code + chat) where each class has a
measurably different acceptance profile.

Until a profile is supplied this ships a per-class fanout *template* stand-in
(the same defaults as `task_router`), so it is runnable now and byte-identical
to `task_router` in that mode — the divergence-from-router payoff lands once a
real per-class histogram is collected and passed via `profile_table`. Swapping
the template for real caps touches only `_select_b_per_depth`.

Routing signal: `prompt_info["task"]` if supplied; else the same
logit-fingerprint fallback as `task_router`.

Identity recovery: `force_baseline=True` → b_per_depth = [tree_width] * D
→ byte-identical to accum_logp.

Caveat: a real per-class histogram doubles the calibration cost (one histogram
per class) and rare classes leave noisy histograms.

Lineage: design id C6 (classifier_per_class_histogram).
"""
from __future__ import annotations

from typing import Any, Optional

import torch

from jetspec.tree._core.base import DraftTree, TreeAlgorithm
from jetspec.tree._core.fanout_cap_builder import build_with_per_depth_cap
from jetspec.tree._core.registry import register_tree_algo


# Default per-class fanout templates (stand-ins until a real per-(class, depth,
# rank) acceptance histogram is supplied via the `profile_table` kwarg).
_DEFAULT_TEMPLATES: dict[str, list[int]] = {
    "decisive_task": [1] * 15,
    "balanced_task": [3, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    "uncertain_task": [7, 7, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
}

# Discrete-label aliases (mirrors task_router; keeps the two compatible with the
# same caller-supplied task vocabulary).
_TASK_ALIASES: dict[str, str] = {
    "math": "decisive_task",
    "code": "decisive_task",
    "decisive_task": "decisive_task",
    "balanced": "balanced_task",
    "balanced_task": "balanced_task",
    "chat": "uncertain_task",
    "open_ended": "uncertain_task",
    "uncertain_task": "uncertain_task",
}


@register_tree_algo("class_histogram")
class ClassifierPerClassHistogram(TreeAlgorithm):
    """Per-class fanout cap (per-class acceptance histogram, template stand-in)."""

    def __init__(
        self,
        gap_high: float = 2.0,
        gap_low: float = 1.0,
        templates: Optional[dict[str, list[int]]] = None,
        force_baseline: bool = False,
    ):
        if gap_high < gap_low:
            raise ValueError(
                f"gap_high ({gap_high}) must be >= gap_low ({gap_low})"
            )
        self.gap_high = float(gap_high)
        self.gap_low = float(gap_low)
        self.templates: dict[str, list[int]] = {
            k: list(v) for k, v in (templates or _DEFAULT_TEMPLATES).items()
        }
        self.force_baseline = bool(force_baseline)

    def build(
        self,
        root_token: int,
        draft_logits: torch.Tensor,  # (1, D, V)
        block_size: int,
        tree_width: int,
        budget: int,
        device: torch.device,
        **kwargs,
    ) -> DraftTree:
        D_expected = block_size - 1
        if draft_logits.dim() != 3 or draft_logits.shape[0] != 1:
            raise ValueError(
                f"draft_logits must be (1, D, V); got {tuple(draft_logits.shape)}"
            )
        if draft_logits.shape[1] != D_expected:
            raise ValueError(
                f"draft_logits depth {draft_logits.shape[1]} != block_size-1 ({D_expected})"
            )

        prompt_info = kwargs.get("prompt_info", None)
        profile_table = kwargs.get("profile_table", None)

        log_probs = torch.log_softmax(draft_logits.squeeze(0), dim=-1)  # (D, V)
        k_eff = max(tree_width, 1)
        topk_lp_t, topk_tok_t = torch.topk(log_probs, k_eff, dim=-1)
        topk_tokens_cpu = topk_tok_t.tolist()
        topk_logprobs_cpu = topk_lp_t.tolist()

        if self.force_baseline:
            b_per_depth = [k_eff] * D_expected
            return build_with_per_depth_cap(
                root_token=int(root_token),
                topk_tokens_cpu=topk_tokens_cpu,
                topk_logprobs_cpu=topk_logprobs_cpu,
                b_per_depth=b_per_depth,
                budget=int(budget),
                device=device,
            )

        task = _resolve_task(prompt_info, topk_lp_t, self.gap_high, self.gap_low)
        b_per_depth = _select_b_per_depth(
            task=task,
            D=D_expected,
            k=k_eff,
            templates=self.templates,
            profile_table=profile_table,
        )

        return build_with_per_depth_cap(
            root_token=int(root_token),
            topk_tokens_cpu=topk_tokens_cpu,
            topk_logprobs_cpu=topk_logprobs_cpu,
            b_per_depth=b_per_depth,
            budget=int(budget),
            device=device,
        )


def _resolve_task(
    prompt_info: Optional[dict[str, Any]],
    topk_lp_t: torch.Tensor,  # (D, k)
    gap_high: float,
    gap_low: float,
) -> str:
    """Prefer the caller-supplied label; otherwise classify by top-2 gap."""
    if prompt_info is not None:
        label = prompt_info.get("task")
        if isinstance(label, str):
            return _TASK_ALIASES.get(label, "balanced_task")
    if topk_lp_t.shape[-1] < 2:
        return "balanced_task"
    gaps = topk_lp_t[:, 0] - topk_lp_t[:, 1]
    g_bar = float(gaps.mean().item())
    if g_bar > gap_high:
        return "decisive_task"
    if g_bar < gap_low:
        return "uncertain_task"
    return "balanced_task"


def _select_b_per_depth(
    task: str,
    D: int,
    k: int,
    templates: dict[str, list[int]],
    profile_table: Optional[dict[str, Any]],
) -> list[int]:
    """Pick per-depth fanout caps for the resolved task class.

    When `profile_table["per_class_histograms"][task]` is supplied, it is
    expected to be either:
      - a list[int] interpreted directly as per-depth caps, or
      - a 2D iterable shaped (D, k) of acceptance rates; b_d is set to the
        largest rank whose acceptance rate exceeds 0.0 (any positive support).
    Otherwise fall back to the hand-tuned per-class template.
    """
    if profile_table is not None:
        hists = profile_table.get("per_class_histograms")
        if isinstance(hists, dict):
            entry = hists.get(task)
            if entry is not None:
                derived = _per_depth_from_histogram(entry, D, k)
                if derived is not None:
                    return derived
    template = templates.get(task) or templates.get("balanced_task") or []
    return _expand_template(template, D, k)


def _per_depth_from_histogram(entry: Any, D: int, k: int) -> Optional[list[int]]:
    """Convert a histogram entry to per-depth caps; None if shape unrecognized."""
    # Case 1: already a flat per-depth list of ints.
    if isinstance(entry, (list, tuple)) and entry and not isinstance(entry[0], (list, tuple)):
        return _expand_template([int(x) for x in entry], D, k)
    # Case 2: 2D acceptance-rate matrix. Cap = largest rank with positive rate.
    if isinstance(entry, (list, tuple)) and entry and isinstance(entry[0], (list, tuple)):
        out: list[int] = []
        for d in range(D):
            row = entry[d] if d < len(entry) else []
            b = 1
            for r in range(min(len(row), k)):
                try:
                    if float(row[r]) > 0.0:
                        b = r + 1
                except (TypeError, ValueError):
                    break
            out.append(max(1, min(b, k)))
        return out
    return None


def _expand_template(template: list[int], D: int, k: int) -> list[int]:
    """Stretch/truncate a class template to length D, clamped to [1, k]."""
    out: list[int] = []
    for d in range(D):
        raw = template[d] if d < len(template) else 1
        out.append(max(1, min(int(raw), k)))
    return out
