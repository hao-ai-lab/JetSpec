"""JETSPEC_ROWS_ASSEMBLY: a proposal round's keyed rows from one rule-agnostic device stage.

A round's rows are (k, anchor) pairs.  The stage takes, as device tensors,
  * ``mass``: P(k) for every scored decoder row (float64, one value per row);
  * the anchor offers of those rows: ids and float64 probabilities, padded to a common width,
    with a ``valid`` mask (the offers are a prefix of each row);
and a row rule, a small scoring function ``rule(mass, offers) -> (score, passing)`` over the
flattened pairs (row-major: scored row, then offer rank).  It then ranks the passing pairs by
(-score, flat index) exactly as ``torch.argsort(..., descending=True, stable=True)`` ranks them,
keeps the first ``cap`` and writes the round's padded key buffer, all on the device with no host
read.  One device-to-host copy (``RowStage.home``) brings back the requested count, the keys and
whatever small check values the caller appended; the device keys stay on the device for the pin.

Rules here: ``joint_rule`` (the e04 joint threshold, ``E04_ROWRULE=joint``) and ``budget_rule``
(``lac.keyed_rows``' joint probability, top ``cap``).  A new predictor plugs in by providing a
per-row P(k) and per-row anchor offers on the device, and a rule that maps them to a pair score
and a passing mask; the stage and the host trip do not change.
"""
from __future__ import annotations

from dataclasses import dataclass
import os

import torch

GRID = 2.0 ** 32  # live_rows.E04_KSCORE_GRID


from jetspec.engines.jetspec_plus_plus.draft.method.rows import DeviceOffers


from jetspec.engines.jetspec_plus_plus.draft.method.rows import floor_passing


def joint_rule(tau, temperature):
    from jetspec.engines.jetspec_plus_plus.draft.method.rows import joint_rule as method_rule
    from jetspec.engines.jetspec_plus_plus.serving.presets.budget import budget_type
    from jetspec.engines.jetspec_plus_plus.draft.serve_policy import rows_min_count
    return method_rule(tau, temperature, floor=rows_min_count(),
                       fill_zero=budget_type().uncut)


def chain_joint_rows(mass, offers, *, after_miss, tau_table, tau_idx, temperature, R):
    """Fixed-cap joint rows with the after-miss k=0 rule selected on device."""
    rows, width = offers.ids.shape
    row = torch.arange(rows, device=mass.device)
    scoring_mass = torch.where(after_miss.bool(), (row == 0).to(mass.dtype), mass)
    from jetspec.engines.jetspec_plus_plus.draft.method.rows import joint_scores
    score = joint_scores(scoring_mass, offers.probabilities, temperature).reshape(-1)
    tau = tau_table.index_select(0, tau_idx.reshape(1))[0]
    valid = offers.valid & (~after_miss.bool() | (row[:, None] == 0))
    passing = (score >= tau) & (score > 0.0) & valid.reshape(-1)
    from jetspec.engines.jetspec_plus_plus.draft.serve_policy import rows_min_count
    from jetspec.engines.jetspec_plus_plus.serving.presets.budget import budget_type
    passing = floor_passing(score, passing, valid, rows_min_count(), fill_zero=budget_type().uncut)
    stage = RowStage(score, passing, offers, cap=R, capacity=R, graph=False)
    active = torch.arange(R, device=mass.device) < stage.requested.clamp(min=1, max=R)
    return stage, active, score.reshape(rows, width)


from jetspec.engines.jetspec_plus_plus.draft.method.rows import budget_rule


class RowStage:
    """Select, rank and key the passing pairs in one device pass; ``home`` is the one trip."""

    _graph_cache = {}

    @classmethod
    def prewarm_graph(cls, device, *, rows, width, cap):
        """Capture the normal and after-miss selection shapes before live rounds."""
        capacity = 1 << (cap - 1).bit_length()
        for count in (rows, 1):
            score = torch.zeros(count * width, device=device, dtype=torch.float64)
            passing = torch.zeros(count * width, device=device, dtype=torch.bool)
            offers = DeviceOffers(
                ids=torch.zeros(count, width, device=device, dtype=torch.long),
                probabilities=torch.zeros(count, width, device=device, dtype=torch.float64),
                valid=torch.zeros(count, width, device=device, dtype=torch.bool),
                ks=torch.zeros(count, device=device, dtype=torch.long))
            cls._graph_select(score, passing, offers, cap, capacity)
        print(f'W470_ROW_STAGE_PREWARM scores={rows * width},{width} cap={cap}', flush=True)

    def __init__(self, score, passing, offers, *, cap, capacity=None, graph=None):
        cap = int(cap)
        if cap < 1:
            raise ValueError('the row cap must be positive')
        capacity = max(cap, int(capacity or cap))
        self.graph_replayed = False
        if os.environ.get('TORCHDYNAMO_DISABLE') == '1':
            graph = False
        if score.is_cuda and graph is not False and \
                (graph is True or os.environ.get('JETSPEC_ROW_STAGE_GRAPH', '0') == '1') and score.numel():
            self.requested, self.keys = self._graph_select(score, passing, offers, cap, capacity)
            self.graph_replayed = True
        else:
            self.requested, self.keys = self._select(score, passing, offers, cap, capacity)
        self.cap, self.capacity = cap, capacity

    from jetspec.engines.jetspec_plus_plus.draft.method.rows import select_rows as _select
    _select = staticmethod(_select)

    @classmethod
    def _graph_select(cls, score, passing, offers, cap, capacity):
        # The two live shapes (normal and after-miss) get separate buffers and graphs.
        # Clone the outputs because the next replay overwrites its static storage while
        # the current round's keys may still be queued for a draft pass or home copy.
        key = (score.device, score.shape, score.dtype, passing.dtype,
               offers.ids.shape, offers.ids.dtype, offers.ks.dtype, cap, capacity)
        entry = cls._graph_cache.get(key)
        if entry is None:
            static_score = torch.empty_like(score)
            static_passing = torch.empty_like(passing)
            static_ids = torch.empty_like(offers.ids)
            static_ks = torch.empty_like(offers.ks)
            static_offers = DeviceOffers(static_ids, offers.probabilities, offers.valid, static_ks)
            static_score.copy_(score)
            static_passing.copy_(passing)
            static_ids.copy_(offers.ids)
            static_ks.copy_(offers.ks)
            stream = torch.cuda.Stream(device=score.device)
            stream.wait_stream(torch.cuda.current_stream(score.device))
            with torch.cuda.stream(stream):
                cls._select(static_score, static_passing, static_offers, cap, capacity)
            torch.cuda.current_stream(score.device).wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                requested, keys = cls._select(static_score, static_passing, static_offers, cap, capacity)
            entry = (graph, static_score, static_passing, static_ids, static_ks, requested, keys)
            cls._graph_cache[key] = entry
        graph, static_score, static_passing, static_ids, static_ks, requested, keys = entry
        static_score.copy_(score)
        static_passing.copy_(passing)
        static_ids.copy_(offers.ids)
        static_ks.copy_(offers.ks)
        graph.replay()
        return requested.clone(), keys.clone()

    def home_async(self, *extra, host=None):
        """``home``'s one device-to-host copy issued into pinned memory without a wait;
        the returned callable waits for it and returns what ``home`` returns. ``host`` may be a
        reusable buffer when the caller has finished the previous read."""
        parts = [self.requested.reshape(1), self.keys.reshape(-1)]
        shapes = []
        for value in extra:
            if value.dtype == torch.float64:
                parts.append(value.reshape(-1).view(torch.int64)); shapes.append((value.shape, True))
            else:
                parts.append(value.reshape(-1).long()); shapes.append((value.shape, False))
        device = torch.cat(parts)
        if host is None:
            host = torch.empty(device.shape, dtype=device.dtype, pin_memory=device.is_cuda)
        elif (host.shape != device.shape or host.dtype != device.dtype or host.device.type != 'cpu'
              or (device.is_cuda and not host.is_pinned())):
            raise ValueError('row home buffer must match the packed int64 payload and be pinned on CUDA')
        host.copy_(device, non_blocking=True)
        event = torch.cuda.Event() if device.is_cuda else None
        if event is not None:
            event.record()

        def read():
            if event is not None:
                event.synchronize()
            return self._parse(host, shapes)
        return read




class BatchedRowStage:
    """The same stable, row-major selection as RowStage for C independent rounds."""

    def __init__(self, score, passing, ids, *, cap, capacity=None):
        self.requested, self.keys = select_rows_batch(score, passing, ids, cap=cap, capacity=capacity)
        self.cap, self.capacity = cap, self.keys.shape[1]

    def home_async(self, *extra):
        """``home``'s one packed copy issued into pinned memory without a wait (the batched
        JETSPEC_BATCH_DRAFT_GRAPH round queues its passes before parsing); the returned
        callable waits for that copy only and returns what ``home`` returns."""
        parts = [self.requested.reshape(-1), self.keys.reshape(-1)]
        shapes = []
        for value in extra:
            is_float = value.dtype == torch.float64
            parts.append(value.reshape(-1).view(torch.int64) if is_float else value.reshape(-1).long())
            shapes.append((value.shape, is_float))
        device = torch.cat(parts)
        host = torch.empty(device.shape, dtype=device.dtype, pin_memory=device.is_cuda)
        host.copy_(device, non_blocking=device.is_cuda)
        event = torch.cuda.Event() if device.is_cuda else None
        if event is not None:
            event.record()

        def read():
            if event is not None:
                event.synchronize()
            return self._unpack(host, shapes)
        return read


