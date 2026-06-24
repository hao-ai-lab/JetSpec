"""TreeAlgorithm ABC and DraftTree contract.

Single source of truth for the interface every tree-construction
algorithm in spec_infer must implement. The contract is engine-agnostic
(parent_indices + depths); engine adapters translate to vLLM's
DFlashRequestTreeSpec or SGLang's tree topology at the adapter boundary.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Optional

import torch


@dataclass
class DraftTree:
    """Flat representation of a draft token tree in BFS order.

    Mirrors causal_parallel_drafting.model.tree.DraftTree with one
    addition: ancestor_mask, an optional precomputed engine-side
    attention bias that an algorithm may produce in build() to skip
    verify-time reconstruction. None means the adapter recomputes.
    """
    token_ids: torch.Tensor          # (N,) int — token at each node
    parent_indices: torch.Tensor     # (N,) int — index of parent (-1 for root)
    depth: torch.Tensor              # (N,) int — depth of each node (root=0)
    num_nodes: int
    cum_logprob: torch.Tensor | None = None       # (N,) float — useful for heap-based algos
    child_maps: list[dict[int, int]] | None = None  # per-node child lookup (lazy)
    ancestor: torch.Tensor | None = None          # (N, N) bool — lazy
    ancestor_packed: torch.Tensor | None = None   # (N, N) uint8 — for Triton kernel
    ancestor_mask: torch.Tensor | None = None     # optional precomputed engine-side mask


class TreeAlgorithm(ABC):
    """All tree algorithms implement this contract.

    The name attribute is set by the @register_tree_algo decorator and
    must match the registry key. Subclasses generally do not set it
    themselves.
    """
    name: str

    @abstractmethod
    def build(
        self,
        root_token: int,
        draft_logits: torch.Tensor,    # (1, D, vocab_size) per-position drafter logits
        block_size: int,
        tree_width: int,
        budget: int,
        device: torch.device,
        refresh_drafter_fn: Optional[Callable[[list[int]], torch.Tensor]] = None,
        profile_table: Optional[dict[str, Any]] = None,
        prompt_info: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> DraftTree:
        """Construct the tree. Must produce a valid DraftTree within budget.

        D = block_size - 1 (per-depth logit positions).
        Algorithm-specific knobs flow through __init__; per-call overrides
        flow through **kwargs.

        Optional adapter-provided context (None when the harness doesn't
        wire it; algorithms must tolerate None gracefully):

        - **refresh_drafter_fn**: callable that takes a list of token IDs
          forming a prefix path from the root, runs the drafter
          conditioned on that prefix, and returns logits for the
          remaining depths. Shape: (remaining_D, vocab_size). Used by
          A-series (layer_conditional/) variants for path-conditional
          rerun. Each call costs roughly one drafter forward.

        - **profile_table**: dict carrying offline-precomputed acceptance
          statistics. Schema is up to each B-series (profile_guided/)
          variant; typical keys are per-(depth, rank) acceptance rates
          or per-prompt-template fanout templates. Loaded by the bench
          harness from a JSONL produced by a separate calibration run.

        - **prompt_info**: dict carrying prompt-derived context for
          C-series (semantic_aware/) variants. Typical keys: 'task'
          (discrete class label from a classifier), 'hidden_state'
          (prompt last-token hidden state from the target model), or
          'token_ids' (raw prompt tokens for heuristic classification).
          When the harness can't provide hidden states, C-series
          variants fall back to logit-fingerprint heuristics on
          draft_logits.
        """
        ...
