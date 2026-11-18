"""Fused order-1 (vanilla) Markov correction for the static DSpark block head.

Stock (`StaticBlockheadGraph._markov_step`) runs, for each of the block's positions in turn,
an embedding gather (W1[prev]), a 256 -> vocab linear (W2), a bf16 add onto the base logits and a
full-vocabulary argmax, about seven kernels per position. This module does one position in two
launches: a vocab-tiled kernel that gathers W1[prev], forms the W2 GEMV for its tile, adds the base
logit, writes the corrected row and a per-tile (max, index) partial, and a one-program finalize that
reduces the partials to the next token. Numerics follow stock: the GEMV is accumulated in fp32 and
rounded to bf16 (the bf16 nn.Linear output), the add is rounded to bf16, and ties take the lowest
index (torch.argmax).
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _markov_tile(LOGITS, W1, W2, PREV, OUT, PMAX, PIDX, V,
                 R: tl.constexpr, BV: tl.constexpr):
    tile = tl.program_id(0)
    rows = tile * BV + tl.arange(0, BV)
    live = rows < V
    rank = tl.arange(0, R)
    prev = tl.load(PREV)
    e = tl.load(W1 + prev * R + rank).to(tl.float32)
    w = tl.load(W2 + rows[:, None] * R + rank[None, :], mask=live[:, None], other=0.0)
    bias = tl.sum(w.to(tl.float32) * e[None, :], axis=1).to(tl.bfloat16)
    base = tl.load(LOGITS + rows, mask=live, other=0.0)
    step = (base.to(tl.float32) + bias.to(tl.float32)).to(tl.bfloat16)
    tl.store(OUT + rows, step, mask=live)
    value = tl.where(live, step.to(tl.float32), float('-inf'))
    best = tl.max(value, axis=0)
    index = tl.min(tl.where(value == best, rows, V), axis=0)
    tl.store(PMAX + tile, best)
    tl.store(PIDX + tile, index)


@triton.jit
def _markov_finalize(PMAX, PIDX, PREV, TOKENS, position, T, V, BT: tl.constexpr):
    lanes = tl.arange(0, BT)
    live = lanes < T
    value = tl.load(PMAX + lanes, mask=live, other=float('-inf'))
    index = tl.load(PIDX + lanes, mask=live, other=V)
    best = tl.max(value, axis=0)
    token = tl.min(tl.where(value == best, index, V), axis=0).to(tl.int64)
    tl.store(PREV, token)
    tl.store(TOKENS + position, token)


class FusedMarkov:
    """Preallocated, graph-capturable fused Markov correction for one sequence."""

    BV = 64  # rows per program: a [64, rank] bf16 W2 tile stays in registers

    def __init__(self, markov, vocab_size: int, device: torch.device):
        self.w1 = markov.markov_w1.weight.detach()
        self.w2 = markov.markov_w2.weight.detach()
        if self.w1.dtype != torch.bfloat16 or self.w2.dtype != torch.bfloat16:
            raise ValueError('fused Markov requires bf16 W1/W2')
        self.rank = int(self.w1.shape[1])
        if self.rank & (self.rank - 1) or tuple(self.w2.shape) != (vocab_size, self.rank):
            raise ValueError('fused Markov requires a power-of-two rank and W2 [vocab, rank]')
        self.vocab = int(vocab_size)
        self.tiles = triton.cdiv(self.vocab, self.BV)
        self.pmax = torch.empty(self.tiles, device=device, dtype=torch.float32)
        self.pidx = torch.empty(self.tiles, device=device, dtype=torch.int32)
        self.prev = torch.zeros(1, device=device, dtype=torch.int64)
        self.bt = triton.next_power_of_2(self.tiles)

    def __call__(self, logits: torch.Tensor, anchor: torch.Tensor, corrected: torch.Tensor,
                 tokens: torch.Tensor) -> None:
        """logits/corrected: [1, block, vocab] bf16; anchor: [1] long; tokens: [1, block] long."""
        if logits.dtype != torch.bfloat16 or corrected.dtype != torch.bfloat16:
            raise ValueError('fused Markov requires bf16 logits')
        block = int(logits.shape[1])
        self.prev.copy_(anchor.reshape(1))
        for position in range(block):
            _markov_tile[(self.tiles,)](
                logits[0, position], self.w1, self.w2, self.prev, corrected[0, position],
                self.pmax, self.pidx, self.vocab, R=self.rank, BV=self.BV, num_warps=4)
            _markov_finalize[(1,)](
                self.pmax, self.pidx, self.prev, tokens[0], position, self.tiles, self.vocab,
                BT=self.bt, num_warps=8)
