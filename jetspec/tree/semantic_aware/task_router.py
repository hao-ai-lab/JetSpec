"""task_router — route each prompt to a per-task fanout template.

Classify the prompt into a coarse task class and apply that class's hand-tuned
per-depth fanout template, committing one fixed tree shape for the whole prompt:

    task ← classify(prompt);   b_per_depth ← TEMPLATES[task]

Routing signal: `prompt_info["task"]` if the caller supplies a label (aliased
onto the class names); otherwise a logit-fingerprint fallback over the
depth-averaged top-2 gap, so the algorithm runs with no routing input wired:

    g_bar = mean_d (log q_d^(1) - log q_d^(2))
    g_bar > gap_high → "decisive_task"  (math/code-like) → chain
    g_bar < gap_low  → "uncertain_task" (chat-like)      → wide-early
    else             → "balanced_task"                    → small fanout early

Per-class templates (D = block_size-1, clamped to tree_width):
- decisive_task  → [1, 1, …]            (chain)
- balanced_task  → [3, 2, 1, 1, …]      (small fanout early, chain after)
- uncertain_task → [7, 7, 1, 1, …]      (wide early, chain after)

The win is the cheapest possible low-budget gain (a chain is the right answer
for most reasoning prompts at small budget) at one drafter forward, with a
predictable per-prompt tree shape that's easy for a scheduler to reason about.
It is coarser than the per-depth caps (top2gap), so its niche is the
prompt-granularity rung, not peak accept-length.

Identity recovery: `force_baseline=True` → b_per_depth = [tree_width] * D
regardless of class → byte-identical to accum_logp.

Caveat: templates are hand-tuned and coarse; brittle to model swaps. A learned
prompt classifier can replace the gap fingerprint without touching build logic.

Lineage: design id C1 (prompt_classifier).
"""
from __future__ import annotations

from typing import Any, Optional

import torch

from jetspec.tree._core.base import DraftTree, TreeAlgorithm
from jetspec.tree._core.fanout_cap_builder import build_with_per_depth_cap
from jetspec.tree._core.registry import register_tree_algo


# Default per-class fanout templates. Each is interpreted left-to-right over
# depths; missing tail entries default to 1 (chain). Clamped to tree_width.
_DEFAULT_TEMPLATES: dict[str, list[int]] = {
    "decisive_task": [1] * 15,
    "balanced_task": [3, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    "uncertain_task": [7, 7, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
}

# Aliases mapping caller-supplied task labels onto the class names, so
# prompt_info["task"] = "math" resolves to the decisive template. Unknown
# labels fall through to "balanced_task" (safe middle ground).
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


@register_tree_algo("task_router")
class PromptTaskClassifier(TreeAlgorithm):
    """Per-class hand-tuned fanout template, selected by task label."""

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
        # Shallow-copy the defaults so per-instance overrides don't mutate
        # module-level state.
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

        # Per-depth top-k extraction (shared with accum_logp so the
        # identity-recovery path is byte-identical).
        log_probs = torch.log_softmax(draft_logits.squeeze(0), dim=-1)  # (D, V)
        k_eff = max(tree_width, 1)
        topk_lp_t, topk_tok_t = torch.topk(log_probs, k_eff, dim=-1)
        topk_tokens_cpu = topk_tok_t.tolist()
        topk_logprobs_cpu = topk_lp_t.tolist()

        if self.force_baseline:
            # Identity-recovery knob: ignore class entirely, emit the
            # full accum_logp cap so b_per_depth = [k_eff] * D.
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
        template = self.templates.get(task) or self.templates.get("balanced_task")
        b_per_depth = _expand_template(template or [], D_expected, k_eff)

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
    # Logit-fingerprint fallback: mean per-depth top-2 gap.
    if topk_lp_t.shape[-1] < 2:
        return "balanced_task"
    gaps = topk_lp_t[:, 0] - topk_lp_t[:, 1]
    g_bar = float(gaps.mean().item())
    if g_bar > gap_high:
        return "decisive_task"
    if g_bar < gap_low:
        return "uncertain_task"
    return "balanced_task"


def _expand_template(template: list[int], D: int, k: int) -> list[int]:
    """Stretch/truncate a class template to length D, clamped to [1, k]."""
    out: list[int] = []
    for d in range(D):
        raw = template[d] if d < len(template) else 1
        out.append(max(1, min(int(raw), k)))
    return out
