"""Capture-safe N=1 keyed candidate wire publication for the draft chain."""
from __future__ import annotations

import torch

try:
    from jetspec.engines.jetspec_plus_plus._optional import triton
    from jetspec.engines.jetspec_plus_plus._optional import tl
except ImportError:
    triton = None
    tl = None

from jetspec.engines.jetspec_plus_plus.wire.chain.contract import OutcomeOffset


if triton is not None:
    @triton.jit
    def _publish_fused(Packet, Final, Keys, Top1, Passes, Published,
                       Wire, Scores, WirePasses, RingRows, RingKeys,
                       RingTop1, RingPasses, RingRounds,
                       ROWS: tl.constexpr, CAP: tl.constexpr,
                       RING: tl.constexpr, LANES: tl.constexpr,
                       WORDS: tl.constexpr, BLOCK_OFF: tl.constexpr,
                       KEY_OFF: tl.constexpr, BLOCK: tl.constexpr):
        j = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        eos = tl.load(Packet + 6) != 0
        visible = ~eos
        round_id = tl.load(Packet + 4) + 1
        slot = round_id % 3
        row = j // LANES
        token = tl.load(Final + j, j < ROWS * LANES, other=-1)
        score = tl.load(Top1 + j, j < ROWS * LANES, other=0.)
        masked_token = tl.where(visible, token, -1)
        masked_score = tl.where(visible, score, 0.)
        tl.store(Final + j, masked_token, j < ROWS * LANES)
        tl.store(Top1 + j, masked_score, j < ROWS * LANES)
        tl.store(Scores + j, tl.where(row < ROWS, masked_score, 0.), j < CAP * LANES)
        tl.store(RingRows + slot * RING * LANES + j,
                 tl.where(visible, token, 0), j < RING * LANES)
        tl.store(RingTop1 + slot * RING * LANES + j,
                 masked_score, j < RING * LANES)

        key = tl.load(Keys + j, j < ROWS * 2, other=-1)
        masked_key = tl.where(visible, key, -1)
        tl.store(Keys + j, masked_key, j < ROWS * 2)
        tl.store(RingKeys + slot * RING * 2 + j, masked_key, j < RING * 2)
        passed = tl.load(Passes + j, j < ROWS, other=0)
        tl.store(WirePasses + j, tl.where(j < ROWS, passed, 0), j < RING)
        tl.store(RingPasses + slot * RING + j,
                 tl.where(visible, passed, 0), j < RING)

        wire_token = tl.load(Final + j - BLOCK_OFF,
                             (j >= BLOCK_OFF) & (j < BLOCK_OFF + ROWS * LANES),
                             other=-1)
        wire_key = tl.load(Keys + j - KEY_OFF,
                           (j >= KEY_OFF) & (j < KEY_OFF + ROWS * 2), other=-1)
        word = tl.full((BLOCK,), -1, tl.int64)
        word = tl.where((j >= BLOCK_OFF) & (j < BLOCK_OFF + CAP * LANES),
                        tl.where(visible, wire_token, -1), word)
        word = tl.where((j >= KEY_OFF) & (j < KEY_OFF + CAP * 2),
                        tl.where(visible, wire_key, -1), word)
        word = tl.where((j >= 3 + CAP * LANES) & (j < 6 + CAP * LANES), 0, word)
        word = tl.where(j == 0, visible.to(tl.int64), word)
        word = tl.where(j == 1, tl.load(Packet + 1), word)
        word = tl.where(j == 2, round_id, word)
        tl.store(Wire + j, word, j < WORDS)
        if tl.program_id(0) == 0:
            tl.store(Published, visible)
            tl.store(RingRounds + slot, round_id)


class ChainPublisher:
    """Keep the exact device wire, pinned host wire, and row-ring scores aligned."""

    def __init__(self, *, device, hypotheses: int, row_cap: int, rows: int,
                 lanes: int):
        from jetspec.engines.jetspec_plus_plus.draft.live_rows import candidate_wire_numel; from jetspec.engines.jetspec_plus_plus.draft.live_rows import candidate_wire_views

        if not 0 < rows <= row_cap or lanes < 2:
            raise ValueError("chain rows must fit the candidate wire")
        self.rows = rows
        self.row_cap = row_cap
        words = candidate_wire_numel(1, hypotheses, 1)
        self.wire = torch.empty(words, dtype=torch.int64, device=device)
        self.home = torch.empty(words, dtype=torch.int64,
                                pin_memory=torch.device(device).type == "cuda")
        self.views = candidate_wire_views(self.wire, 1, hypotheses, 1)
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
            block_off, key_off, block)
        self.home.copy_(self.wire, non_blocking=True)
