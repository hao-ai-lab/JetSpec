"""Public correction-layer models, adapters, and liveness gates.

Imports are lazy so engine-agnostic tree users do not eagerly load model or
checkpoint dependencies.
"""
from __future__ import annotations

import importlib


_LAZY = {
    "AttentionLayer": "jetspec.correction.attention_layer",
    "Qwen3CorrectionAttention": "jetspec.correction.attention_layer",
    "VanillaMarkov": "jetspec.correction.markov_head",
    "GatedMarkovHead": "jetspec.correction.markov_head",
    "RNNHead": "jetspec.correction.markov_head",
    "AttentionLayerDrafter": "jetspec.correction.attention_layer_adapter",
    "AttentionLayerLayout": "jetspec.correction.attention_layer_adapter",
    "assert_live_attention_layer": "jetspec.correction.attention_layer_adapter",
    "load_attention_layer_head": "jetspec.correction.attention_layer_adapter",
    "load_attention_layer_target_weights": "jetspec.correction.attention_layer_adapter",
    "resolve_attention_layer_layout": "jetspec.correction.attention_layer_adapter",
    "MarkovHeadDrafter": "jetspec.correction.markov_head_adapter",
    "MarkovHeadLayout": "jetspec.correction.markov_head_adapter",
    "assert_live_markov_bias": "jetspec.correction.markov_head_adapter",
    "load_markov_head": "jetspec.correction.markov_head_adapter",
    "load_markov_head_target_weights": "jetspec.correction.markov_head_adapter",
    "resolve_markov_head_layout": "jetspec.correction.markov_head_adapter",
}


def __getattr__(name: str):
    if name in _LAZY:
        return getattr(importlib.import_module(_LAZY[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)


__all__ = sorted(_LAZY)
