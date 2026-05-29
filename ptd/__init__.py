"""PTD — parallel tree decoding.

A lightweight, single-stream, offline speculative-tree-decoding engine built on
top of HF `transformers` (the DFlash convention: the target is a standard HF
causal LM; the draft head subclasses the HF per-architecture model). M0 ships
the plain autoregressive baseline; the draft head + tree verify land in M1+.
"""
from ptd.engine.llm import LLM, SamplingParams
from ptd.draft import (
    Drafter, RepeatDrafter, TargetEchoDrafter,
    TreeDrafter, RandomTreeDrafter, TargetEchoTreeDrafter,
)

__all__ = [
    "LLM", "SamplingParams",
    "Drafter", "RepeatDrafter", "TargetEchoDrafter",
    "TreeDrafter", "RandomTreeDrafter", "TargetEchoTreeDrafter",
]
