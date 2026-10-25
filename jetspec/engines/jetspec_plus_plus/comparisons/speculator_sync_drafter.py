"""Synchronous single-GPU speculator backed by a DFlash-class block drafter, fresh context."""

from __future__ import annotations

import torch

from jetspec.engines.jetspec_plus_plus.serving.sequence import Sequence
from jetspec.engines.jetspec_plus_plus.serving.speculate_types import SpeculateResult
from jetspec.engines.jetspec_plus_plus.serving.speculate_types import VerifyResult
from jetspec.engines.jetspec_plus_plus.comparisons.speculator_blockhead import SpeculatorBlockhead
from jetspec.engines.jetspec_plus_plus.comparisons.sync_drafter import load_sync_drafter


class SpeculatorSyncDrafter(SpeculatorBlockhead):
    """The block-head serial slot with a drafter loaded by ``JetSpecDrafter.load(fresh_only=True)``.

    The round loop, the round records and the verifier are the block-head mode's, unchanged; this
    class only supplies the drafter and tells it which prompt it is drafting for (the mask-free
    first block is seeded by the prompt index)."""

    def __init__(self, lookahead: int, device: torch.device, *, checkpoint_dir: str | None = None,
                 drafter=None, metrics: dict | None = None, max_context: int | None = None,
                 max_num_seqs: int = 1, run_config=None):
        if drafter is None:
            if checkpoint_dir is None:
                raise ValueError("sync drafter checkpoint is required")
            drafter = load_sync_drafter(checkpoint_dir, device=device, max_context=max_context,
                                        run_config=run_config)
        super().__init__(lookahead, device, drafter=drafter, metrics=metrics,
                         max_num_seqs=max_num_seqs, run_config=run_config)
        self.sync_drafter = True

    def _prepare_sequence(self, seq: Sequence) -> None:
        self.drafter.set_prompt_index(seq.jetspec_prompt_index)
