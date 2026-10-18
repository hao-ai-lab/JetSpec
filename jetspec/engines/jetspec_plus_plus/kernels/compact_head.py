"""Device kernels for draft/refresh_swap.py."""
from __future__ import annotations

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None


if triton is not None:
    @triton.jit
    def _compact_head(H, W, IDS, PMAX, PSUM, PTOP, PALT, PALTID, PCURRENT,
                      M: tl.constexpr, V: tl.constexpr, K: tl.constexpr,
                      WS0: tl.constexpr, WS1: tl.constexpr,
                      SLOTS: tl.constexpr, C: tl.constexpr,
                      BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                      PERSISTENT: tl.constexpr = False,
                      FROM_LOGITS: tl.constexpr = False):
        depth = tl.arange(0, BK)
        if PERSISTENT:
            first = tl.program_id(0)
            end = tl.cdiv(M, BM) * C
            step = tl.num_programs(0)
        else:
            first = tl.program_id(0) * C + tl.program_id(1)
            end = first + 1
            step = 1
        for tile in range(first, end, step):
            row = tile // C * BM + tl.arange(0, BM)
            col = tile % C * BN + tl.arange(0, BN)
            current = tl.load(IDS + row // SLOTS * (SLOTS + 1) + row % SLOTS + 1,
                              row < M, other=0)
            if FROM_LOGITS:
                value = tl.load(H + row[:, None] * V + col[None, :],
                                (row[:, None] < M) & (col[None, :] < V),
                                other=-float('inf')).to(tl.float32)
            else:
                acc = tl.full((BM, BN), 0, tl.float32)
                for block in range(tl.cdiv(K, BK)):
                    k = block * BK + depth
                    h = tl.load(H + row[:, None] * K + k[None, :],
                                (row[:, None] < M) & (k[None, :] < K), other=0)
                    w = tl.load(W + col[:, None] * WS0 + k[None, :] * WS1,
                                (col[:, None] < V) & (k[None, :] < K), other=0)
                    acc = tl.dot(h, tl.trans(w), acc)
                value = acc.to(tl.bfloat16).to(tl.float32)
            value = tl.where(col[None, :] < V, value, -float('inf'))
            maximum = tl.max(value, 1)
            top = tl.min(tl.where(value == maximum[:, None], col[None, :], 2147483647), 1)
            excluded = tl.where(col[None, :] == current[:, None], -float('inf'), value)
            alternative = tl.max(excluded, 1)
            alternate_id = tl.min(tl.where(excluded == alternative[:, None],
                                           col[None, :], 2147483647), 1)
            draft = tl.max(tl.where(col[None, :] == current[:, None], value, -float('inf')), 1)
            safe_maximum = tl.where(maximum == -float('inf'), 0.0, maximum)
            total = tl.sum(tl.exp(value - safe_maximum[:, None]), 1)
            out = row * C + tile % C
            tl.store(PMAX + out, maximum, row < M)
            tl.store(PSUM + out, total, row < M)
            tl.store(PTOP + out, top, row < M)
            tl.store(PALT + out, alternative, row < M)
            tl.store(PALTID + out, alternate_id, row < M)
            tl.store(PCURRENT + out, draft, row < M)

    @triton.jit
    def _compact_merge(PMAX, PSUM, PTOP, PALT, PALTID, PCURRENT, TOP, ALT, MARGIN,
                       C: tl.constexpr, PC: tl.constexpr):
        row = tl.program_id(0)
        chunk = tl.arange(0, PC)
        maximum_part = tl.load(PMAX + row * C + chunk, chunk < C, other=-float('inf'))
        total_part = tl.load(PSUM + row * C + chunk, chunk < C, other=0)
        top_part = tl.load(PTOP + row * C + chunk, chunk < C, other=2147483647)
        alternative_part = tl.load(PALT + row * C + chunk, chunk < C, other=-float('inf'))
        alternate_id = tl.load(PALTID + row * C + chunk, chunk < C, other=2147483647)
        draft_part = tl.load(PCURRENT + row * C + chunk, chunk < C, other=-float('inf'))
        maximum = tl.max(maximum_part, 0)
        top = tl.min(tl.where(maximum_part == maximum, top_part, 2147483647), 0)
        alternative = tl.max(alternative_part, 0)
        alternate = tl.min(tl.where(alternative_part == alternative, alternate_id, 2147483647), 0)
        draft = tl.max(draft_part, 0)
        denominator = tl.sum(total_part * tl.exp(maximum_part - maximum), 0)
        # Compare rounded probabilities, preserving margin-zero ties and underflow.
        p_new = tl.exp(alternative - maximum) / denominator
        p_draft = tl.exp(draft - maximum) / denominator
        tl.store(TOP + row, top)
        tl.store(ALT + row, alternate)
        tl.store(MARGIN + row, p_new >= p_draft)
