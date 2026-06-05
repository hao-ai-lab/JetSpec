"""PTD — parallel tree decoding.

A lightweight, single-stream, offline speculative-tree-decoding engine built on
top of HF `transformers` (the DFlash convention: the target is a standard HF
causal LM; the draft head subclasses the HF per-architecture model). The offline
autoregressive baseline ships today; the draft head + tree verify build on it.

Top-level names (LLM, drafters, load_draft_head) are re-exported LAZILY (PEP 562)
so that importing `ptd` — or just the engine-agnostic `ptd.tree` — does NOT eagerly
pull the engine (transformers + the draft-head model). External backends that
consume only `ptd.tree` (e.g. the vLLM integration) get the tree contract without
the engine's dependencies; the one-way tree<->engine separation holds at import
time too.
"""
import importlib

_LAZY = {
    "LLM": "ptd.engine.llm",
    "SamplingParams": "ptd.engine.llm",
    "Drafter": "ptd.draft",
    "RepeatDrafter": "ptd.draft",
    "TargetEchoDrafter": "ptd.draft",
    "TreeDrafter": "ptd.draft",
    "RandomTreeDrafter": "ptd.draft",
    "TargetEchoTreeDrafter": "ptd.draft",
    "DraftHeadDrafter": "ptd.draft_head_drafter",
    "DraftHeadTreeDrafter": "ptd.draft_head_drafter",
    "load_draft_head": "ptd.models.draft_head",
}


def __getattr__(name: str):
    if name in _LAZY:
        return getattr(importlib.import_module(_LAZY[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)


__all__ = [
    "LLM", "SamplingParams",
    "Drafter", "RepeatDrafter", "TargetEchoDrafter",
    "TreeDrafter", "RandomTreeDrafter", "TargetEchoTreeDrafter",
    "DraftHeadDrafter", "DraftHeadTreeDrafter", "load_draft_head",
]
