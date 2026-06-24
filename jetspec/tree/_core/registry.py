"""Decorator-based registry for tree algorithms.

Usage:
    @register_tree_algo("V8_entropy_adjusted_score")
    class EntropyAdjustedScore(TreeAlgorithm):
        ...

    algo = get_algorithm("V8_entropy_adjusted_score", lambda_=0.3)
    tree = algo.build(root_token, draft_logits, ...)
"""
from __future__ import annotations

from .base import TreeAlgorithm

_REGISTRY: dict[str, type[TreeAlgorithm]] = {}


def register_tree_algo(name: str):
    """Decorator: register a TreeAlgorithm subclass under name."""
    def decorator(cls: type[TreeAlgorithm]) -> type[TreeAlgorithm]:
        if name in _REGISTRY:
            raise ValueError(f"tree algorithm {name!r} already registered")
        cls.name = name
        _REGISTRY[name] = cls
        return cls
    return decorator


def get_algorithm(name: str, **init_kwargs) -> TreeAlgorithm:
    """Instantiate a registered algorithm by name."""
    if name not in _REGISTRY:
        raise KeyError(
            f"unknown tree algorithm {name!r}. Available: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[name](**init_kwargs)


def list_algorithms() -> list[str]:
    return sorted(_REGISTRY.keys())
