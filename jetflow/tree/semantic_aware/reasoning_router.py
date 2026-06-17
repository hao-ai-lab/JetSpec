"""reasoning_router — prompt-adaptive: chain for reasoning, fanout otherwise.

A semantic (not confidence-based) router: detect whether the prompt is a
step-by-step reasoning task and shape the whole tree accordingly:

    reasoning prompt → chain ([1] * D)          (each next token is "obvious")
    open-ended       → balanced fanout ([2,2,1,...] clamped to k)

Routing signal, in priority order:
1. `prompt_info["mode"]` / `prompt_info["task"]` if the caller supplies a label.
2. A regex over the decoded prompt text (markers: ``<think>``, ``"Step 1"``,
   ``"Let me think"``, ``"First,"`` …) when `prompt_info` carries
   ``token_ids`` + ``tokenizer`` (or ``text``). This is the real rule-based path.
3. Fallback (no `prompt_info`): a logit fingerprint — a confident drafter (large
   mean top-2 gap) reads as a "next-token-is-obvious" reasoning step. Keeps the
   algorithm runnable with no routing input wired.

Zero ML, microseconds of overhead — a free, interpretable "this is CoT → go
deep" rule for an engine that already has the prompt text.

Identity recovery: `force_baseline=True` → b_per_depth = [tree_width] * D
regardless of mode → byte-identical to accum_logp.

Caveat: brittle to model conventions; the regex can misfire on code with
similar tokens.

Lineage: design id C5 (reasoning_mode).
"""
from __future__ import annotations

import re
from typing import Any, Optional

import torch

from jetflow.tree._core.base import DraftTree, TreeAlgorithm
from jetflow.tree._core.fanout_cap_builder import build_with_per_depth_cap
from jetflow.tree._core.registry import register_tree_algo


# Reasoning-marker substrings checked against decoded prompt text when a
# tokenizer is supplied. Case-insensitive.
_REASONING_MARKERS: tuple[str, ...] = (
    "<think>",
    "step 1",
    "step 1:",
    "let me think",
    "let's think",
    "first,",
    "chain of thought",
)
_REASONING_REGEX = re.compile(
    "|".join(re.escape(m) for m in _REASONING_MARKERS), re.IGNORECASE
)

# Caller-supplied labels that map onto reasoning mode. Anything else falls
# through to open-ended.
_REASONING_LABELS: frozenset[str] = frozenset(
    {"reasoning", "chain_of_thought", "cot", "math", "code"}
)


@register_tree_algo("reasoning_router")
class ReasoningModeDetector(TreeAlgorithm):
    """Chain vs balanced fanout depending on detected prompt mode."""

    def __init__(
        self,
        gap_threshold: float = 1.5,
        open_ended_template: Optional[list[int]] = None,
        force_baseline: bool = False,
    ):
        self.gap_threshold = float(gap_threshold)
        # Open-ended template: small fanout for the first couple depths,
        # chain afterwards. Length is stretched/truncated at build time.
        self.open_ended_template: list[int] = list(
            open_ended_template
            if open_ended_template is not None
            else [2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        )
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

        is_reasoning = _detect_reasoning_mode(
            prompt_info, topk_lp_t, self.gap_threshold
        )
        if is_reasoning:
            # Chain: one child per depth, regardless of k.
            b_per_depth = [1] * D_expected
        else:
            b_per_depth = _expand_template(
                self.open_ended_template, D_expected, k_eff
            )

        return build_with_per_depth_cap(
            root_token=int(root_token),
            topk_tokens_cpu=topk_tokens_cpu,
            topk_logprobs_cpu=topk_logprobs_cpu,
            b_per_depth=b_per_depth,
            budget=int(budget),
            device=device,
        )


def _detect_reasoning_mode(
    prompt_info: Optional[dict[str, Any]],
    topk_lp_t: torch.Tensor,  # (D, k)
    gap_threshold: float,
) -> bool:
    """Return True iff the prompt looks like a step-by-step reasoning task."""
    if prompt_info is not None:
        # Explicit signals — checked in priority order.
        mode = prompt_info.get("mode")
        if isinstance(mode, str) and mode.lower() in {"reasoning", "cot"}:
            return True
        label = prompt_info.get("task")
        if isinstance(label, str) and label.lower() in _REASONING_LABELS:
            return True
        # Regex on decoded text, if available.
        token_ids = prompt_info.get("token_ids")
        tokenizer = prompt_info.get("tokenizer")
        if token_ids is not None and tokenizer is not None:
            try:
                text = tokenizer.decode(list(token_ids))
            except Exception:
                text = None
            if text and _REASONING_REGEX.search(text):
                return True
        # Already-decoded text (some callers pass raw strings).
        text = prompt_info.get("text")
        if isinstance(text, str) and _REASONING_REGEX.search(text):
            return True
    # Logit-fingerprint fallback: a confident drafter (large mean top-2 gap)
    # reads as a "next-token-is-obvious" reasoning step.
    if topk_lp_t.shape[-1] < 2:
        return False
    gaps = topk_lp_t[:, 0] - topk_lp_t[:, 1]
    return bool(float(gaps.mean().item()) > gap_threshold)


def _expand_template(template: list[int], D: int, k: int) -> list[int]:
    """Stretch/truncate a template to length D, clamped to [1, k]."""
    out: list[int] = []
    for d in range(D):
        raw = template[d] if d < len(template) else 1
        out.append(max(1, min(int(raw), k)))
    return out
