"""Exact top-k anchor selection without the [rows, V] fp64 probability tensor.

Reference semantics (heads/decoder.py::anchor_offers with JETSPEC_HEADS_FAST=1):
  probabilities = softmax_fp32(logits with d_rej set to -inf).double()
  keep = probabilities >= 1 - q_hat, minus d_rej
  selected = the first min(count, cap) kept ids by (p descending, id ascending)
  lac_fallbacks = rows with count > cap and p[cap-1] == p[cap] (a tie across the cap)

Why the fp32 softmax stays: the reference probabilities are fp32 softmax outputs widened to
fp64, so the only way to match them to 1e-12 relative is to run the same fp32 softmax kernel on
the same masked logits. Recomputing exp(logit - logsumexp) in fp64 differs from the fp32 kernel
by about 1e-7 relative and orders fp32-tied probabilities by logit instead of by id (equiv.py
measures both on the recorded rounds). The fp32 softmax is ~0.06 ms; what this removes is the
full-vocab fp64 widening, keep mask, masked_fill and fp64 top-k, the per-row loop, and every host
sync before one transfer at the end.

Order and ties: an fp32 top-k of width cap + 1 + slack returns the right multiset of values but
an arbitrary member of a tied group. Inside that small set, key = (bits(p) << id_bits) |
(id_mask - id) is a strict total order equal to (p descending, id ascending), since the fp32 bit
pattern of p >= 0 read as int32 is monotone in p. d_rej is written as p = -1.0 (negative bits,
below every token). The first cap ids are exact unless the value at cap-1 is kept and still tied
at the last slot of the set (the tied group may extend past it); only those rows are recomputed
with a full-row int64 key top-k, after the transfer (no extra sync: the rows are known on host).

fast_offers_topk runs eagerly (any device); GraphedSelect replays the same device body as one
CUDA graph per (device, shape, cap, q pair), which removes ~40 kernel launches of host overhead
and runs the argmax on a side branch beside the softmax chain. (A chunk-maximum prefilter
before the top-k, and a Triton key top-64, were measured and did not beat torch.topk here: the
full-vocab top-k over 15 rows is latency-bound at ~0.12 ms either way.)
"""
from __future__ import annotations

from jetspec.engines.jetspec_plus_plus.profiles import kernel_enabled, kernel_value

import os
import torch

SLACK = 16

# Unit T4 (draft_fuse_t4): the chain's handoff buffers, set only while the chain body that
# owns them is captured. The select writes into them instead of fresh tensors.
_CHAIN_OUT = None


class chain_outputs:
    """Point GraphedSelect.chain_body at the chain buffers; ``written`` names what it filled."""

    def __init__(self, buffers):
        self.buffers = buffers
        self.written = set()

    def __enter__(self):
        global _CHAIN_OUT
        _CHAIN_OUT = self
        return self

    def __exit__(self, *_):
        global _CHAIN_OUT
        _CHAIN_OUT = None

    def target(self, name, shape, dtype, device):
        value = self.buffers.get(name)
        if (value is None or tuple(value.shape) != tuple(shape) or value.dtype != dtype
                or value.device != device or not value.is_contiguous()):
            return None
        return value



def _select_body(logits, rejected, q_a, q_b, cap, width, side=None, *, exact_merge=False,
                 bf16_logits=None, out=None):
    """Device-only body: no host syncs. q_a applies to rows whose top-1 is d_rej, q_b otherwise.
    With a side stream on the default path, argmax runs beside softmax."""
    from jetspec.engines.jetspec_plus_plus.profiles import kernel_enabled
    rows, vocab = logits.shape
    fused = (kernel_enabled('anchor_select_fused', '1')
             and logits.device.type == 'cuda' and logits.dtype == torch.float32
             and logits.is_contiguous() and rejected.is_contiguous() and cap <= vocab
             and (not isinstance(q_a, torch.Tensor) or q_a.dtype == torch.float64))
    if fused:
        from jetspec.engines.jetspec_plus_plus.jetspec.heads.anchor_select_fused import prepare_probabilities
        probabilities, before_top1 = prepare_probabilities(logits, rejected)
    else:
        if side is not None:
            main = torch.cuda.current_stream(logits.device)
            side.wait_stream(main)
            with torch.cuda.stream(side):
                before_top1 = logits.argmax(-1) == rejected
        else:
            before_top1 = logits.argmax(-1) == rejected
        valid = (rejected >= 0)[:, None]
        column = torch.where(valid, rejected[:, None], 0)
        filtered = logits.clone()
        filtered.scatter_(1, column, torch.where(valid, float('-inf'), filtered.gather(1, column)))
        probabilities = filtered.softmax(-1)
        probabilities.scatter_add_(1, column, -valid.to(probabilities.dtype))
    radix = (fused and bf16_logits is not None
             and bf16_logits.dtype == torch.bfloat16 and bf16_logits.is_contiguous()
             and width <= 64 and width < vocab and cap & (cap - 1) == 0
             and kernel_enabled('decoder_select_radix', '1'))
    if radix:
        from jetspec.engines.jetspec_plus_plus.kernels.decoder_select import radix_topk
        values, ids = radix_topk(bf16_logits, probabilities, rejected, width,
                                logit_pivot=True, bin_tile=16)
    else:
        values, ids = probabilities.topk(width, dim=-1, largest=True, sorted=False)
    if side is not None and not fused:
        main.wait_stream(side)
    if fused:
        from jetspec.engines.jetspec_plus_plus.jetspec.heads.anchor_select_fused import select_post
        if out is None:
            ids, p64, pack = select_post(values, ids, before_top1, q_a, q_b, cap, vocab)
            inexact = pack[4].bool()
        else:
            # T4: write into the chain buffers; the flag cast feeds only the fallback below.
            ids, p64, pack = select_post(values, ids, before_top1, q_a, q_b, cap, vocab,
                                         out=out)
            if exact_merge and cap & (cap - 1):
                inexact = pack[4].bool()
    else:
        # a non-finite softmax row is NaN throughout (NaN or +inf logits, or an all -inf row
        # make the row sum NaN), and top-k ranks NaN first.
        finite = torch.isfinite(values).all(-1)
        id_bits = max(1, (vocab - 1).bit_length())
        id_mask = (1 << id_bits) - 1
        keys = (values.view(torch.int32).long() << id_bits) | (id_mask - ids)
        # keys are distinct, so rank is a permutation: (p desc, id asc)
        rank = (keys[:, None, :] > keys[:, :, None]).sum(-1)
        keys = torch.empty_like(keys).scatter_(1, rank, keys)
        ids = id_mask - (keys & id_mask)
        values = (keys >> id_bits).int().view(torch.float32)
        p64 = values.double()
        if not isinstance(q_a, torch.Tensor):
            q_a, q_b = torch.full_like(p64[:, 0], q_a), torch.full_like(p64[:, 0], q_b)
        q = torch.where(before_top1, q_a, q_b)
        kept = p64 >= (1.0 - q)[:, None]
        sizes = kept[:, :cap].sum(-1)
        if width > cap:
            crosses = kept[:, cap] & (values[:, cap - 1] == values[:, cap])
        else:
            crosses = torch.zeros_like(before_top1)
        if width < vocab:
            inexact = kept[:, cap - 1] & (values[:, width - 1] == values[:, cap - 1])
        else:
            inexact = torch.zeros_like(before_top1)
        pack = torch.stack((sizes, before_top1.long(), crosses.long(), finite.long(), inexact.long()))
    if exact_merge:
        if fused and cap & (cap - 1) == 0:
            from jetspec.engines.jetspec_plus_plus.jetspec.heads.anchor_select_fused import repair_ties
            ids, p64 = repair_ties(probabilities, ids, p64, pack, cap)
        else:
            # A fixed-shape full-vocabulary selection keeps the tie fallback on device.
            # Only the first cap columns are observable; preserve the slack columns.
            exact_ids, exact_p = _exact_rows(
                probabilities, torch.arange(rows, device=logits.device), cap)
            ids = torch.cat((torch.where(inexact[:, None], exact_ids, ids[:, :cap]),
                             ids[:, cap:]), dim=1)
            p64 = torch.cat((torch.where(inexact[:, None], exact_p, p64[:, :cap]),
                             p64[:, cap:]), dim=1)
    return ids, p64, pack, probabilities


def _exact_rows(probabilities, rows, cap):
    """Full-row (p desc, id asc) top-cap for the few rows whose tied group reaches the slack."""
    part = probabilities.index_select(0, rows)
    vocab = part.shape[1]
    id_bits = max(1, (vocab - 1).bit_length())
    id_mask = (1 << id_bits) - 1
    ids = torch.arange(vocab, device=part.device, dtype=torch.long)
    keys = (part.view(torch.int32).long() << id_bits) | (id_mask - ids)
    keys = keys.topk(min(cap, vocab), dim=-1, largest=True, sorted=True).values
    top_ids = id_mask - (keys & id_mask)
    return top_ids, (keys >> id_bits).int().view(torch.float32).double()


def _finish(ids, p64, pack, probabilities, cap):
    host = pack.tolist()
    if not all(host[3]):
        raise ValueError('fast LAC probabilities and q values must be finite')
    inexact = [row for row, flag in enumerate(host[4]) if flag]
    if inexact:
        rows = torch.tensor(inexact, device=ids.device)
        exact_ids, exact_p = _exact_rows(probabilities, rows, cap)
        width = exact_ids.shape[1]
        ids[rows, :width] = exact_ids
        p64[rows, :width] = exact_p
    sizes = host[0]
    width = ids.shape[1]
    split = [part for size in sizes for part in (size, width - size)]
    return (ids.reshape(-1).split(split)[0::2], p64.reshape(-1).split(split)[0::2],
            sizes, [bool(flag) for flag in host[1]], sum(host[2]))


def _q_pair(q_hats):
    """A float (pooled) or a (drej-top1, not-drej-top1) pair (grouped)."""
    pair = (float(q_hats[0]), float(q_hats[1])) if isinstance(q_hats, tuple) else \
        (float(q_hats), float(q_hats))
    if not all(0.0 <= value <= 1.0 for value in pair):
        raise ValueError('fast LAC probabilities and q values must be finite')
    return pair


def _check(logits, rejected, cap):
    if logits.ndim != 2 or rejected.shape != logits.shape[:1]:
        raise ValueError('decoder logits and rejected ids must have shapes [15,V] and [15]')
    if int(cap) < 1:
        raise ValueError('fast LAC cap must be positive')


def fast_offers_topk(logits, rejected, q_hats, cap, *, slack=SLACK):
    """Eager form. Returns (ids, probabilities, set_sizes, rejected_was_top1, lac_fallbacks)
    with ids[k] long and probabilities[k] float64 on the logits device, identical to the
    reference fast path; q_hats is a float (pooled), a (drej-top1, not-drej-top1) pair, or a
    per-row sequence (tests)."""
    logits = torch.as_tensor(logits).float()
    rejected = torch.as_tensor(rejected, device=logits.device, dtype=torch.long)
    _check(logits, rejected, cap)
    cap = int(cap)
    width = min(logits.shape[1], cap + 1 + int(slack))
    if isinstance(q_hats, (tuple, int, float)):
        q_a, q_b = _q_pair(q_hats)
    else:
        q_rows = torch.as_tensor(q_hats, dtype=torch.float64).reshape(-1)
        if q_rows.shape != logits.shape[:1] or not bool(((q_rows >= 0) & (q_rows <= 1)).all()):
            raise ValueError('fast LAC probabilities and q values must be finite')
        q_a = q_b = q_rows.to(logits.device)
    ids, p64, pack, probabilities = _select_body(logits, rejected, q_a, q_b, cap, width)
    return _finish(ids, p64, pack.cpu(), probabilities, cap)


def _home(ids, p64, pack, probabilities, cap, width, *, combined=None):
    """The offers' trip home in ONE device-to-host copy (pack, ids and fp64 probabilities
    together).  Host (ids, probabilities) per row, set sizes, rejected-was-top1 flags and
    the LAC fallback count; a row needing the exact full-row recompute takes the two-copy
    path."""
    rows = ids.shape[0]
    host = (torch.cat((pack.reshape(-1), ids.reshape(-1),
                       p64.view(torch.int64).reshape(-1)))
            if combined is None else combined).cpu()
    pack_host = host[:pack.numel()].view(pack.shape)
    flags = pack_host.tolist()
    if not all(flags[3]):
        raise ValueError('fast LAC probabilities and q values must be finite')
    if any(flags[4]):
        selected, chosen, sizes, top1, fallbacks = _finish(
            ids.clone(), p64.clone(), pack_host, probabilities, cap)
        total = sum(sizes)
        both = torch.cat((torch.cat(selected).long(),
                          torch.cat(chosen).double().view(torch.int64))).cpu()
        return (both[:total].split(sizes), both[total:].view(torch.float64).split(sizes),
                sizes, top1, fallbacks)
    start = pack.numel()
    host_ids = host[start:start + ids.numel()].view(rows, width)
    host_p = host[start + ids.numel():].view(rows, width).view(torch.float64)
    sizes = flags[0]
    return ([host_ids[row, :size] for row, size in enumerate(sizes)],
            [host_p[row, :size] for row, size in enumerate(sizes)],
            sizes, [bool(flag) for flag in flags[1]], sum(flags[2]))


def fast_offers_topk_home(logits, rejected, q_hats, cap, *, slack=SLACK):
    """``fast_offers_topk`` with ``GraphedSelect.home``'s one-copy trip home, eagerly
    (JETSPEC_ANCHOR_SELECT_HOME=one on the eager path): host ids and probabilities."""
    logits = torch.as_tensor(logits).float()
    rejected = torch.as_tensor(rejected, device=logits.device, dtype=torch.long)
    _check(logits, rejected, cap)
    cap = int(cap)
    width = min(logits.shape[1], cap + 1 + int(slack))
    q_a, q_b = _q_pair(q_hats)
    ids, p64, pack, probabilities = _select_body(logits, rejected, q_a, q_b, cap, width)
    return _home(ids, p64, pack, probabilities, cap, width)




def _chain_body_t4(logits, rejected, q_a, q_b, cap, width, chain):
    """Unit T4 chain select: ids, p64 and pack land in the chain's handoff buffers."""
    rows = logits.shape[0]
    device = logits.device
    targets = (chain.target('offers_ids', (rows, width), torch.long, device),
               chain.target('offers_p64', (rows, width), torch.float64, device),
               chain.target('offers_pack', (5, rows), torch.long, device))
    fused = (kernel_enabled('anchor_select_fused', '1') and device.type == 'cuda'
             and rejected.is_contiguous() and cap <= logits.shape[1]
             and (not isinstance(q_a, torch.Tensor) or q_a.dtype == torch.float64))
    out = targets if fused and all(value is not None for value in targets) else None
    if out is not None and kernel_value('draft_fuse_t4', '0') == '2':
        from jetspec.engines.jetspec_plus_plus.kernels import select_stats
        if (select_stats.available(logits, rejected, width, cap)
                and kernel_enabled('decoder_select_radix', '1')):
            return _chain_body_stats(logits, rejected, q_a, q_b, cap, width, chain, out)
    ids, p64, pack, _ = _select_body(
        logits.float(), rejected, q_a, q_b, cap, width, exact_merge=True,
        bf16_logits=logits if kernel_enabled('decoder_select_radix', '1') else None,
        **({'out': out} if out is not None else {}))
    if out is not None:
        chain.written.update(name for name, value, target in zip(
            ('offers_ids', 'offers_p64', 'offers_pack'), (ids, p64, pack), out)
            if value is target)
    return ids, p64, pack


def _chain_body_stats(logits, rejected, q_a, q_b, cap, width, chain, out):
    """Unit T4 level 2: one BF16 logits read gives the select statistics, the radix pass 0,
    the K3 hazard's unfiltered statistics and the chain's dec_logits copy (tau gate)."""
    from jetspec.engines.jetspec_plus_plus.kernels import select_stats
    from jetspec.engines.jetspec_plus_plus.jetspec.heads.anchor_select_fused import select_post
    rows, vocab = logits.shape
    device = logits.device
    copy = chain.target('dec_logits', tuple(logits.shape), logits.dtype, device)
    stats = chain.target('select_stats', (4, rows), torch.float32, device)
    argmax = chain.target('select_argmax', (rows,), torch.int32, device)
    if stats is None or argmax is None:
        stats = argmax = None
    values, ids, before_top1, stats_out, argmax_out = select_stats.select_topk(
        logits, rejected, width, copy=copy, stats=stats, argmax=argmax)
    ids, p64, pack = select_post(values, ids, before_top1, q_a, q_b, cap, vocab, out=out)
    ids, p64 = select_stats.repair_ties(logits, rejected, stats_out, ids, p64, pack, cap)
    chain.written.update(('offers_ids', 'offers_p64', 'offers_pack'))
    if copy is not None:
        chain.written.add('dec_logits')
    if stats is not None:
        chain.written.update(('select_stats', 'select_argmax'))
    return ids, p64, pack


def _chain_body_t5(logits, rejected, q_a, q_b, cap, width, chain):
    """Opt-in T5 radix restructure; unsupported inputs keep the existing select."""
    from jetspec.engines.jetspec_plus_plus.kernels import select_stats, select_radix_fused
    if not (select_stats.available(logits, rejected, width, cap)
            and kernel_enabled('anchor_select_fused', '1')
            and kernel_enabled('decoder_select_radix', '1')
            and (not isinstance(q_a, torch.Tensor) or q_a.dtype == torch.float64)):
        return None
    rows = logits.shape[0]
    device = logits.device
    out = (chain.target('offers_ids', (rows, width), torch.long, device),
           chain.target('offers_p64', (rows, width), torch.float64, device),
           chain.target('offers_pack', (5, rows), torch.long, device))
    if any(value is None for value in out):
        return None
    copy = chain.target('dec_logits', tuple(logits.shape), logits.dtype, device)
    stats = chain.target('select_stats', (4, rows), torch.float32, device)
    argmax = chain.target('select_argmax', (rows,), torch.int32, device)
    if stats is None or argmax is None:
        stats = argmax = None
    selection = {}
    if kernel_value('draft_fuse_sel', '0') in ('2', '3'):
        totals = chain.target('select_totals', (rows, 256), torch.int32, device)
        if totals is None:
            return None
        selection['totals'] = totals
    ids, p64, pack, _, _ = select_radix_fused.select(
        logits, rejected, q_a, q_b, cap, width, copy=copy, stats=stats, argmax=argmax, out=out,
        **selection)
    chain.written.update(('offers_ids', 'offers_p64', 'offers_pack'))
    if copy is not None:
        chain.written.add('dec_logits')
    if stats is not None:
        chain.written.update(('select_stats', 'select_argmax'))
    return ids, p64, pack


class GraphedSelect:
    """One CUDA graph per (device, rows, vocab, cap, width, q pair); inputs copied in."""

    def __init__(self):
        self._graphs = {}

    @staticmethod
    def chain_body(logits, rejected, q_hats, cap, *, slack=SLACK):
        """Capture-safe selection with tie correction and no trip to the host."""
        from jetspec.engines.jetspec_plus_plus.profiles import kernel_enabled
        cap = int(cap)
        width = min(logits.shape[1], cap + 1 + int(slack))
        if isinstance(q_hats, torch.Tensor):
            if q_hats.shape != (2,):
                raise ValueError('chain thresholds must be a device pair')
            q_a, q_b = q_hats.unbind()
        else:
            q_a, q_b = _q_pair(q_hats)
        if kernel_enabled('draft_fuse_t5', '1') and _CHAIN_OUT is not None:
            outputs = _chain_body_t5(logits, rejected, q_a, q_b, cap, width, _CHAIN_OUT)
            if outputs is not None:
                return outputs
        level = kernel_value('draft_fuse_t4', '0')
        if level in ('1', '2') and _CHAIN_OUT is not None:
            return _chain_body_t4(logits, rejected, q_a, q_b, cap, width, _CHAIN_OUT)
        ids, p64, pack, _ = _select_body(
            logits.float(), rejected, q_a, q_b, cap, width, exact_merge=True,
            bf16_logits=logits if kernel_enabled('decoder_select_radix', '1') else None)
        return ids, p64, pack

    def __call__(self, logits, rejected, q_hats, cap, *, slack=SLACK):
        logits = torch.as_tensor(logits).float()
        rejected = torch.as_tensor(rejected, device=logits.device, dtype=torch.long)
        _check(logits, rejected, cap)
        if logits.device.type != 'cuda':
            return fast_offers_topk(logits, rejected, q_hats, cap, slack=slack)
        cap = int(cap)
        width = min(logits.shape[1], cap + 1 + int(slack))
        q_a, q_b = _q_pair(q_hats)
        key = (logits.device, tuple(logits.shape), cap, width, q_a, q_b)
        entry = self._graphs.get(key)
        if entry is None:
            entry = self._capture(logits, rejected, q_a, q_b, cap, width)
            self._graphs[key] = entry
        graph, static_logits, static_rejected, outputs = entry
        static_logits.copy_(logits)
        static_rejected.copy_(rejected)
        graph.replay()
        ids, p64, pack, probabilities = outputs
        ids, p64 = ids.clone(), p64.clone()
        return _finish(ids, p64, pack.cpu(), probabilities, cap)

    def home(self, logits, rejected, q_hats, cap, *, slack=SLACK):
        """``__call__`` plus the offers' trip home in ONE device-to-host copy (pack, ids
        and fp64 probabilities together), instead of the pack's copy and then a second
        copy of the selected ids and probabilities.  Returns host (ids, probabilities)
        per row, set sizes, rejected-was-top1 flags and the LAC fallback count; a row
        needing the exact full-row recompute takes the two-copy path."""
        logits = torch.as_tensor(logits).float()
        rejected = torch.as_tensor(rejected, device=logits.device, dtype=torch.long)
        _check(logits, rejected, cap)
        cap = int(cap)
        width = min(logits.shape[1], cap + 1 + int(slack))
        q_a, q_b = _q_pair(q_hats)
        key = (logits.device, tuple(logits.shape), cap, width, q_a, q_b)
        entry = self._graphs.get(key)
        if entry is None:
            entry = self._capture(logits, rejected, q_a, q_b, cap, width)
            self._graphs[key] = entry
        graph, static_logits, static_rejected, outputs = entry
        static_logits.copy_(logits)
        static_rejected.copy_(rejected)
        graph.replay()
        ids, p64, pack, probabilities = outputs
        return _home(ids, p64, pack, probabilities, cap, width)

    def device(self, logits, rejected, q_hats, cap, *, slack=SLACK):
        """serving: the graph's (ids, p64, pack, probabilities) left on the device (the first
        three cloned, so a later replay cannot change them), and the width."""
        logits = torch.as_tensor(logits).float()
        rejected = torch.as_tensor(rejected, device=logits.device, dtype=torch.long)
        _check(logits, rejected, cap)
        cap = int(cap)
        width = min(logits.shape[1], cap + 1 + int(slack))
        q_a, q_b = _q_pair(q_hats)
        key = (logits.device, tuple(logits.shape), cap, width, q_a, q_b)
        entry = self._graphs.get(key)
        if entry is None:
            entry = self._capture(logits, rejected, q_a, q_b, cap, width)
            self._graphs[key] = entry
        graph, static_logits, static_rejected, outputs = entry
        static_logits.copy_(logits)
        static_rejected.copy_(rejected)
        graph.replay()
        ids, p64, pack, probabilities = outputs
        return (ids.clone(), p64.clone(), pack.clone(), probabilities), cap, width

    @staticmethod
    def _capture(logits, rejected, q_a, q_b, cap, width):
        static_logits = logits.clone()
        static_rejected = rejected.clone()
        stream = torch.cuda.Stream(device=logits.device)
        stream.wait_stream(torch.cuda.current_stream(logits.device))
        side = torch.cuda.Stream(device=logits.device)
        with torch.cuda.stream(stream):
            for _ in range(2):
                _select_body(static_logits, static_rejected, q_a, q_b, cap, width, side)
        torch.cuda.current_stream(logits.device).wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, capture_error_mode='thread_local'):
            outputs = _select_body(static_logits, static_rejected, q_a, q_b, cap, width, side)
        return graph, static_logits, static_rejected, outputs


def lse_variant(logits, rejected, q_hats, cap):
    """The logsumexp route (fp64 exp(logit - lse)); kept only to measure why it cannot be exact."""
    logits = torch.as_tensor(logits).float()
    rejected = torch.as_tensor(rejected, device=logits.device, dtype=torch.long)
    q_a, q_b = _q_pair(q_hats)
    before_top1 = logits.argmax(-1) == rejected
    q = torch.where(before_top1, torch.full(before_top1.shape, q_a, dtype=torch.float64,
                                            device=logits.device), q_b)
    valid = (rejected >= 0)[:, None]
    column = torch.where(valid, rejected[:, None], 0)
    filtered = logits.clone()
    filtered.scatter_(1, column, torch.where(valid, float('-inf'), filtered.gather(1, column)))
    lse = filtered.double().logsumexp(-1)
    values, top_ids = filtered.topk(min(int(cap) + 1, logits.shape[1]), dim=-1, sorted=True)
    p = (values.double() - lse[:, None]).exp()
    kept = torch.isfinite(values) & (p >= (1.0 - q)[:, None])
    sizes = kept[:, :int(cap)].sum(-1).cpu().tolist()
    return ([top_ids[r, :s] for r, s in enumerate(sizes)],
            [p[r, :s] for r, s in enumerate(sizes)], sizes)
