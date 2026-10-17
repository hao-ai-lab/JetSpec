"""Capture-safe N=1 keyed candidate wire publication for the draft chain."""
from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None

from jetspec.engines.jetspec_plus_plus.wire.contract import OutcomeOffset
from jetspec.engines.jetspec_plus_plus.wire.contract import CANDIDATE_TRAILER_WORDS
from jetspec.engines.jetspec_plus_plus.wire.triton_offsets import (  # noqa: F401  (read inside the kernels)
    A, AVAILABLE, BLOCKS, CANDIDATE_ROUND, CANDIDATE_SEQ_ID, CMD, EOS_FLAG, HASH16, K, L, RECOVERY,
    RESERVED, ROUND, SEQ_ID, TAP_COUNT, TRUE_K)


if triton is not None:
    from jetspec.engines.jetspec_plus_plus.kernels.chain_publish import (_publish_fused)


class ChainPublisher:
    """Keep the exact device wire, pinned host wire, and row-ring scores aligned."""

    def __init__(self, *, device, hypotheses: int, row_cap: int, rows: int,
                 lanes: int, run_config=None):
        from jetspec.engines.jetspec_plus_plus.jetspec.live_rows import candidate_wire_numel
        from jetspec.engines.jetspec_plus_plus.jetspec.live_rows import candidate_wire_views

        if not 0 < rows <= row_cap or lanes < 2:
            raise ValueError("chain rows must fit the candidate wire")
        self.rows = rows
        self.row_cap = row_cap
        words = candidate_wire_numel(1, hypotheses, 1, run_config=run_config if run_config is not None else getattr(self, 'run_config', None))
        self.wire = torch.empty(words, dtype=torch.int64, device=device)
        self.home = torch.empty(words, dtype=torch.int64,
                                pin_memory=torch.device(device).type == "cuda")
        self.views = candidate_wire_views(self.wire, 1, hypotheses, 1, run_config=run_config if run_config is not None else getattr(self, 'run_config', None))
        if len(self.views) != 6 or self.views[2].shape != (1, row_cap, lanes):
            raise ValueError("chain requires a keyed b16 candidate wire")
        self.scores = torch.zeros(row_cap, lanes, dtype=torch.float32, device=device)

    @property
    def wire_rows(self):
        return self.views[2][0]

    @property
    def wire_keys(self):
        return self.views[5][0]

    def body(self, packet: torch.Tensor, final_tokens: torch.Tensor,
             keys: torch.Tensor, final_p_top1: torch.Tensor,
             published: torch.Tensor) -> None:
        """Fill stable wire pointers and queue the single pinned D2H before E_pub."""
        if (final_tokens.shape != (self.rows, self.wire_rows.shape[1])
                or keys.shape != (self.rows, 2)
                or final_p_top1.shape != final_tokens.shape
                or published.shape != ()):
            raise ValueError("chain publish tensors do not match captured rows")
        available, control, blocks, timing, telemetry, row_keys = self.views
        self.wire.fill_(-1)
        visible = ~packet[OutcomeOffset.EOS_FLAG].bool()
        published.copy_(visible)
        available[0].copy_(visible.long())
        control[0, 0].copy_(packet[OutcomeOffset.SEQ_ID])
        control[0, 1].copy_(packet[OutcomeOffset.ROUND] + 1)
        timing.zero_()
        telemetry.zero_()
        blocks[0, :self.rows].copy_(final_tokens)
        row_keys[0, :self.rows].copy_(keys)
        self.scores.zero_()
        self.scores[:self.rows].copy_(final_p_top1)
        self.home.copy_(self.wire, non_blocking=True)

    def fused_body(self, packet, final_tokens, keys, final_p_top1, passes_run,
                   published, wire_passes_run, ring) -> None:
        """Publish the same wire and selected ring slot in one captured kernel."""
        if triton is None or not self.wire.is_cuda:
            raise RuntimeError("fused chain publish requires CUDA Triton")
        rows, lanes = final_tokens.shape
        ring_rows = ring.rows.shape[1]
        if (rows != self.rows or keys.shape != (rows, 2)
                or final_p_top1.shape != final_tokens.shape
                or passes_run.shape != (rows,)
                or wire_passes_run.shape != (ring_rows,)
                or ring_rows > self.row_cap):
            raise ValueError("fused publish tensors do not match captured rows")
        block_off = self.views[2].storage_offset()
        key_off = self.views[5].storage_offset()
        block = 256
        work = max(self.wire.numel(), self.row_cap * lanes, ring_rows * lanes)
        _publish_fused[(triton.cdiv(work, block),)](
            packet, final_tokens, keys, final_p_top1, passes_run, published,
            self.wire, self.scores, wire_passes_run, ring.rows, ring.keys,
            ring.p_top1, ring.passes_run, ring.rounds,
            rows, self.row_cap, ring_rows, lanes, self.wire.numel(),
            block_off, key_off, block, CANDIDATE_TRAILER_WORDS)
        self.home.copy_(self.wire, non_blocking=True)
