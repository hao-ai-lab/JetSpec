"""PTD — parallel tree decoding.

A lightweight, single-stream, offline speculative-tree-decoding engine built on
top of HF `transformers` (the DFlash convention: the target is a standard HF
causal LM; the draft head subclasses the HF per-architecture model). The offline
autoregressive baseline ships today; the draft head + tree verify build on it.
"""
from ptd.engine.llm import LLM, SamplingParams
from ptd.draft import (
    Drafter, RepeatDrafter, TargetEchoDrafter,
    TreeDrafter, RandomTreeDrafter, TargetEchoTreeDrafter,
)
from ptd.draft_head_drafter import DraftHeadDrafter, DraftHeadTreeDrafter
from ptd.models.draft_head import load_draft_head

__all__ = [
    "LLM", "SamplingParams",
    "Drafter", "RepeatDrafter", "TargetEchoDrafter",
    "TreeDrafter", "RandomTreeDrafter", "TargetEchoTreeDrafter",
    "DraftHeadDrafter", "DraftHeadTreeDrafter", "load_draft_head",
]
