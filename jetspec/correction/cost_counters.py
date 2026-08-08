"""Opt-in operation counts and wall-time attribution for correction-layer trees.

The module is deliberately dormant unless ``JETSPEC_COST_COUNTERS=1`` and a
benchmark prompt has called :func:`start_prompt`.  This keeps unit gates and
unflagged generation on the original numerical path.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import json
import os
from statistics import median
import time
from typing import Iterator

import torch


_COUNT_NAMES = (
    "n_base_forwards",
    "n_attention_layer_calls",
    "n_lm_head_projections",
    "n_markov_applies",
)
_PHASE_NAMES = (
    "base_forward",
    "channel",
    "lm_head_reprojection",
    "tree_bookkeeping",
)


@dataclass
class _Round:
    number: int
    device: torch.device
    started_at: float
    counts: dict[str, int] = field(
        default_factory=lambda: {name: 0 for name in _COUNT_NAMES}
    )
    wall_seconds: dict[str, float] = field(
        default_factory=lambda: {name: 0.0 for name in _PHASE_NAMES}
    )


@dataclass
class _Prompt:
    arm: str
    rounds: list[dict] = field(default_factory=list)
    analytics: dict[str, int] = field(default_factory=dict)
    current: _Round | None = None


_active_prompt: _Prompt | None = None


def enabled() -> bool:
    """Return whether the exact opt-in environment flag is set."""
    return os.environ.get("JETSPEC_COST_COUNTERS") == "1"


def active(arm: str | None = None) -> bool:
    """Return whether a flagged prompt (optionally for ``arm``) is active."""
    if not enabled() or _active_prompt is None:
        return False
    return arm is None or _active_prompt.arm == arm


def _clock(device: torch.device) -> float:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter()


def start_prompt(arm: str):
    """Start collecting one prompt; returns ``None`` when counters are off."""
    global _active_prompt
    if not enabled():
        return None
    if _active_prompt is not None:
        raise RuntimeError("a JETSPEC cost-counter prompt is already active")
    _active_prompt = _Prompt(arm=str(arm))
    return True


def _start_round(arm: str, device: torch.device) -> None:
    prompt = _active_prompt
    if prompt is None or prompt.arm != arm:
        return
    if prompt.current is not None:
        _finish_round()
    dev = torch.device(device)
    prompt.current = _Round(
        number=len(prompt.rounds) + 1,
        device=dev,
        started_at=_clock(dev),
    )


def increment(name: str, amount: int = 1) -> None:
    prompt = _active_prompt
    if prompt is None or prompt.current is None:
        return
    if name not in prompt.current.counts:
        raise KeyError(f"unknown JETSPEC cost counter {name!r}")
    prompt.current.counts[name] += int(amount)


@contextmanager
def phase(name: str, device: torch.device) -> Iterator[None]:
    """Time one non-overlapping operation phase in the current round."""
    prompt = _active_prompt
    if (
        not enabled()
        or prompt is None
        or prompt.current is None
        or name not in _PHASE_NAMES
    ):
        yield
        return
    dev = torch.device(device)
    started = _clock(dev)
    try:
        yield
    finally:
        prompt.current.wall_seconds[name] += _clock(dev) - started


@contextmanager
def base_forward(arm: str, device: torch.device) -> Iterator[None]:
    """Begin a round and time its single full drafter forward."""
    if not active(arm):
        yield
        return
    _start_round(arm, torch.device(device))
    with phase("base_forward", torch.device(device)):
        yield
    increment("n_base_forwards")


@contextmanager
def tree_build(device: torch.device) -> Iterator[None]:
    """Attribute build time excluding channel and forced lm-head work."""
    prompt = _active_prompt
    if not enabled() or prompt is None or prompt.current is None:
        yield
        return
    current = prompt.current
    channel_before = current.wall_seconds["channel"]
    lm_head_before = current.wall_seconds["lm_head_reprojection"]
    dev = torch.device(device)
    started = _clock(dev)
    try:
        yield
    finally:
        elapsed = _clock(dev) - started
        nested = (
            current.wall_seconds["channel"]
            - channel_before
            + current.wall_seconds["lm_head_reprojection"]
            - lm_head_before
        )
        current.wall_seconds["tree_bookkeeping"] += max(0.0, elapsed - nested)


def set_analytic(name: str, value: int) -> None:
    """Record a device-independent MAC count shared by this prompt's rounds."""
    if _active_prompt is None:
        return
    _active_prompt.analytics[str(name)] = int(value)


def _finish_round() -> None:
    prompt = _active_prompt
    if prompt is None or prompt.current is None:
        return
    current = prompt.current
    total = _clock(current.device) - current.started_at
    explicit = sum(current.wall_seconds.values())
    if explicit > total:
        # perf_counter call overhead can leave a sub-microsecond attribution
        # overrun. Charge it back to bookkeeping so the split remains exact.
        overrun = explicit - total
        current.wall_seconds["tree_bookkeeping"] = max(
            0.0, current.wall_seconds["tree_bookkeeping"] - overrun
        )
        explicit = sum(current.wall_seconds.values())
    wall = dict(current.wall_seconds)
    wall["other"] = max(0.0, total - explicit)
    wall["total"] = sum(wall.values())
    prompt.rounds.append(
        {
            "round": current.number,
            "counts": dict(current.counts),
            "wall_seconds": wall,
        }
    )
    prompt.current = None


def finish_prompt() -> dict | None:
    """Finish the last round and return per-round, total, and median data."""
    global _active_prompt
    prompt = _active_prompt
    if prompt is None:
        return None
    _finish_round()
    rounds = list(prompt.rounds)
    total_counts = {
        name: sum(row["counts"][name] for row in rounds) for name in _COUNT_NAMES
    }
    wall_names = (*_PHASE_NAMES, "other", "total")
    total_wall = {
        name: sum(row["wall_seconds"][name] for row in rounds)
        for name in wall_names
    }
    median_wall = {
        name: median([row["wall_seconds"][name] for row in rounds])
        if rounds
        else 0.0
        for name in wall_names
    }
    report = {
        "arm": prompt.arm,
        "rounds": rounds,
        "totals": {"counts": total_counts, "wall_seconds": total_wall},
        "median_wall_seconds": median_wall,
        "analytics": dict(prompt.analytics),
    }
    _active_prompt = None
    return report


def emit_prompt_report(report: dict | None, *, prompt_index: int) -> None:
    """Print stable JSONL records for benchmark logs and Modal capture."""
    if report is None:
        return
    for row in report["rounds"]:
        payload = {
            "prompt_index": int(prompt_index),
            "arm": report["arm"],
            **row,
        }
        print("[JETSPEC_COST_ROUND] " + json.dumps(payload, sort_keys=True))
    summary = {
        "prompt_index": int(prompt_index),
        "arm": report["arm"],
        "totals": report["totals"],
        "median_wall_seconds": report["median_wall_seconds"],
        "analytics": report["analytics"],
    }
    print("[JETSPEC_COST_PROMPT] " + json.dumps(summary, sort_keys=True))


def _reset_for_tests() -> None:
    global _active_prompt
    _active_prompt = None


__all__ = [
    "active",
    "base_forward",
    "emit_prompt_report",
    "enabled",
    "finish_prompt",
    "increment",
    "phase",
    "set_analytic",
    "start_prompt",
    "tree_build",
]
