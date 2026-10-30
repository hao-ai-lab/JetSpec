"""The C>1 batched JetSpec++ draft round: decoder rows, assembly, context and Jacobi passes.

Port of the N>1 branch of the retired flat-row path. The C=1 segmented chain is
unchanged. Lab, record, device split, device publish, refresh, deadline and the N=1 fallback
are not ported: a round they would serve raises ``BatchRoundError``.
"""
from __future__ import annotations

import gc
from collections import OrderedDict

import torch

from jetspec.engines.jetspec_plus_plus.runtime.config_values import config_values
from jetspec.engines.jetspec_plus_plus.draft.drafter import DraftForwardPass
from jetspec.engines.jetspec_plus_plus.draft.rows import device_tensor
from jetspec.engines.jetspec_plus_plus.draft.live_rows import (BucketPass, PackedBlocks, adapter_refresh_mode,
                                           batch_draft_bucket, batch_draft_front,
                                           batch_draft_graph, host_mark, row_layout_anchor0,
                                           rows_assembly, select_live_block, wide_rows)
from jetspec.engines.jetspec_plus_plus.draft.rows_live import (live_row_bucket, live_row_order, rows_live,
                                            rows_live_quantum)
from jetspec.engines.jetspec_plus_plus.draft.batch_assembly import assemble_rows_batch
from jetspec.engines.jetspec_plus_plus.draft.batch_context import batch_graph_context, live_row_pass, live_row_passes
from jetspec.engines.jetspec_plus_plus.draft.batch_refinement import packed_refinement

__all__ = ['BatchRoundError', 'batch_round_enabled', 'prefetch_batch', 'propose_batch',
           'flat_rows_and_passes_batch', 'prewarm_batch_graphs']

_FIELDS = ('predictor_hidden', 'predictor_inputs', 'predictor_outputs', 'predictor_p_top1',
           'posterior_probabilities')


class BatchRoundError(RuntimeError):
    """A C>1 round that the batched path does not serve; the retired N=1 fallback is gone."""


def _config(rows, run_config):
    return run_config if run_config is not None else getattr(rows, 'run_config', None)


def batch_round_enabled(rows, *, run_config=None):
    """True when the read serves C>1 rows through the batched draft graph. Every round of such
    a read takes it, a drained round of one sequence included (padded to the two-bucket)."""
    return batch_draft_graph(run_config=_config(rows, run_config))


def prefetch_batch(rows, sequences, *, run_config=None):
    """Append this round's committed rows to the anchor decoder's batch prefix while the
    adapter's device work runs (the N>1 branch of the retired ``prefetch_decoder_prefix``)."""
    run_config = _config(rows, run_config)
    values = config_values(run_config)
    if rows.decoder is None or values.get('engine.engine_settings.decoder_prefetch') != '1':
        return
    if len(sequences) == 1:
        rows.decoder.prefetch_prefix(sequences[0], run_config=run_config)
        return
    if batch_draft_graph(run_config=run_config):
        rows.decoder.retire_stale_batch_prefixes(sequences)
    if batch_draft_front(run_config=run_config) and rows.drafter.device.type == 'cuda':
        rows.decoder.prefetch_batch_graph(sequences, run_config=run_config)
    elif values.get('engine.engine_settings.decoder_batch_prefetch', '0') == '1':
        rows.decoder.prefetch_batch(sequences, run_config=run_config)
    else:
        for sequence in sequences:
            rows.decoder.prefetch_prefix(sequence, run_config=run_config)


@torch.inference_mode()
def propose_batch(rows, sequences, *, k_hypotheses, passes, abort_if=None, run_config=None):
    """Draft round+1 for the staged sequences of a C>1 read: one decoder batch, then the batched
    rows. A drained round of one sequence pads to the two-bucket."""
    run_config = _config(rows, run_config)
    values = config_values(run_config)
    if not sequences:
        raise BatchRoundError('propose_batch needs at least one sequence')
    if rows.decoder is None:
        raise BatchRoundError('C>1 rows require the anchor decoder')
    if (rows.graphs or not rows_assembly(run_config=rows.run_config)
            or not row_layout_anchor0(run_config=rows.run_config)
            or not batch_draft_graph(run_config=rows.run_config)):
        raise BatchRoundError('C>1 rows require rows_graphs=0, rows_assembly, the anchor0 '
                              'layout and batch_draft_graph=1')
    if values.get('engine.engine_settings.device_split', '0') == '1':
        raise BatchRoundError('device_split is not ported to the C>1 round')
    predictions = rows.decoder.predict_batch(
        sequences, [sequence.pending_block for sequence in sequences],
        dataset=values['benchmark.benchmark_settings.dataset'],
        coverage=values.get('method.method_settings.anchor_coverage', '0.99'),
        grouped=values.get('method.method_settings.anchor_drej', '0') == '1',
        cap=int(values.get('method.method_settings.anchor_cap', '16')),
        after_misses=[bool(getattr(sequence, '_jetspec_previous_selection_miss', False))
                      for sequence in sequences], run_config=run_config)
    return flat_rows_and_passes_batch(rows, sequences, predictions, passes=passes,
                                      abort_if=abort_if, run_config=run_config)


@torch.inference_mode()
def flat_rows_and_passes_batch(rows, sequences, predictions, *, passes, abort_if, run_config=None):
    """The joint-rule rows of C sequences: private row keys, one padded head and one J loop."""
    run_config = _config(rows, run_config)
    values = config_values(run_config)
    d = rows.drafter
    count = len(sequences)
    capacity = int(values.get('engine.engine_settings.rows_cap', '128'))
    if values.get('engine.engine_settings.device_publish', '0') == '1':
        raise BatchRoundError('device_publish is not ported to the C>1 round')
    if adapter_refresh_mode(run_config=rows.run_config) != 'off' and passes is not None and passes > 1:
        raise BatchRoundError('adapter refresh is not ported to the C>1 round')
    # The retired proposal-record writer (draft/wirecal.py) has no live home: a C>1 read leaves
    # its W280_RECORD directory empty, and tau and TPS come from the read output.
    if getattr(rows, 'deadline', None) is not None:
        raise BatchRoundError('the draft deadline serves N=1 only')
    stage_device, finish_stages = assemble_rows_batch(rows, sequences, predictions, defer=True,
                                                      run_config=run_config)
    selected_keys = stage_device.keys
    padded = batch_draft_bucket(count)
    H = int(selected_keys.shape[1])
    prefixes = [sequence.exact_hidden(device=d.device, dtype=d.dtype)[0] for sequence in sequences]
    prefix_bucket = ((max(prefix.shape[0] for prefix in prefixes) + 255) // 256) * 256
    wide = wide_rows(run_config=run_config)
    key = ('flat', padded, H, prefix_bucket, 'anchor0')
    # one captured bucket per (batch bucket, stage row capacity, prefix bucket); the padded
    # requests' rows are inactive from the start
    buckets = rows.__dict__.setdefault('_batch_graph_buckets', OrderedDict())
    built = key not in buckets
    if built:
        if len(buckets) >= rows.__dict__.get('_batch_graph_bucket_limit', 6):
            buckets.popitem(last=False)
        buckets[key] = BucketPass(d, N=padded, H=H, prefix_bucket=prefix_bucket,
                                  backend=rows.backend, host_fast=False, flat=True, graphs=True,
                                  anchor0=True, wide=wide, run_config=run_config)
    buckets.move_to_end(key)
    bucket = buckets[key]
    compact = H == capacity and rows_live(run_config=run_config)
    rows_pass = bucket
    if compact:
        live_passes = live_row_passes(rows, bucket, padded, H)
    B = d.block_size
    tails = torch.cat([sequence.staged.predicted_hidden[:, :B if wide else 15]
                       for sequence in sequences], dim=0)
    captured = getattr(stage_device, 'row_inputs', None)
    if captured is not None:
        keys_device, live_counts, geometry, active_rows, preceding, first = captured
        if padded > count:
            extra = padded - count
            tails = torch.cat((tails, tails.new_zeros(extra, *tails.shape[1:])))
            prefixes = prefixes + [prefixes[0].new_zeros(1, prefixes[0].shape[-1])] * extra
    else:
        # the row inputs as batched gathers: the private row of an empty round, the live counts
        # from the device counts, the request's sampled context and its anchor source
        from jetspec.engines.jetspec_plus_plus.draft.live_rows import batched_kth_random_context
        keys_device = selected_keys[:, :H].clone()
        requested = stage_device.requested
        guesses = device_tensor([int(sequence.pending_block.correction_guess)
                                 for sequence in sequences], d.device)
        private = torch.stack((torch.zeros_like(guesses), guesses), dim=1)
        keys_device[:, 0] = torch.where((requested == 0)[:, None], private, keys_device[:, 0])
        live_counts = requested.clamp(max=stage_device.cap).clamp(min=1)
        keys_device = keys_device.clamp_min(0)
        samples = batched_kth_random_context(sequences, block_size=B, seed=d.seed, device=d.device)
        source = torch.cat((
            torch.cat([sequence.committed_ids[:, -1:].to(d.device) for sequence in sequences]),
            torch.stack([sequence.pending_block.draft_ids.reshape(-1).to(d.device)
                         for sequence in sequences])), dim=1)
        source = torch.cat((source, source[:, -1:]), dim=1)
        if padded > count:
            extra = padded - count
            keys_device = torch.cat((keys_device, keys_device.new_zeros(extra, H, 2)))
            live_counts = torch.cat((live_counts, live_counts.new_zeros(extra)))
            tails = torch.cat((tails, tails.new_zeros(extra, *tails.shape[1:])))
            samples = torch.cat((samples, samples.new_zeros(extra, B)))
            source = torch.cat((source, source.new_zeros(extra, source.shape[1])))
            prefixes = prefixes + [prefixes[0].new_zeros(1, prefixes[0].shape[-1])] * extra
        geometry = (keys_device[:, :, 0] + 1).reshape(-1)
        active_rows = (torch.arange(H, device=d.device)[None, :] < live_counts[:, None]).reshape(-1)
        preceding = source.gather(1, keys_device[:, :, 0]).reshape(-1, 1)
        first = samples[:, None, :].expand(-1, H, -1).clone()
        first[:, :, 0] = keys_device[:, :, 1]
        first = first.reshape(padded * H, B)
    batch_graph_context(rows, bucket, sequences, prefixes[:count], tails, geometry)
    device_keys = keys_device.reshape(-1, 2)
    pass_rows, row_starts = padded * H, [n * H for n in range(count)]
    if compact:
        # the round's drafted row counts from the stage's copy home (queued behind the
        # context K/V, so the device keeps working while the host reads them)
        requested_home = stage_device.home_read()[0][:count].tolist()
        counts = [max(1, min(int(value), stage_device.cap)) for value in requested_home]
        counts += [0] * (padded - count)
        live_total = sum(counts)
        pass_rows = live_row_bucket(live_total, padded * H,
                                    rows_live_quantum(padded * H, run_config=run_config))
        rows_pass = live_passes.get(pass_rows)
        if rows_pass is None:
            rows_pass = live_passes[pass_rows] = live_row_pass(rows, bucket, pass_rows)
        row_starts = [sum(counts[:n]) for n in range(count)]
        row_order, (preceding, first, device_keys, active_rows) = bucket.live_shelf.write_order(
            live_row_order(counts, H), pass_rows, bucket.head,
            (preceding, first, device_keys, active_rows))
    mask_hits = []
    parsed = {}
    last_pass = (passes if passes is not None else d.convergence_limit) - 1

    def parse_stages():
        if 'stages' not in parsed:
            parsed['stages'] = finish_stages()

    def forward(*, fill_tokens, pass_index):
        if rows_pass.wide:
            ids = first if fill_tokens is None else fill_tokens
        else:
            ids = first if fill_tokens is None else torch.cat((fill_tokens, rows_pass.extra_tokens), dim=1)
        mask_hits.append((ids == int(d.head.mask_token_id)).any())
        logits = rows_pass(ids)
        if pass_index >= last_pass:
            parse_stages()  # the stage's host checks run while the last pass computes
        return DraftForwardPass(logits=logits, input_tokens=ids if rows_pass.wide else ids[:, :15],
                                hidden=rows_pass.hidden)

    combined = packed_refinement(
        forward, anchors=preceding, hypotheses=list(range(pass_rows)), sequences=1,
        passes=passes, convergence_limit=d.convergence_limit, abort_if=abort_if,
        output_indices=list(range(pass_rows)), pinned_tokens=device_keys[:, 1],
        fast_top1=True, active_rows=active_rows, predictor=rows.predictor,
        correction_head=d.correction_head, run_config=run_config)
    parse_stages()
    stages = parsed['stages']
    if stages is None:
        raise BatchRoundError('the deferred row stage declined the round')
    if mask_hits and bool(torch.stack(mask_hits).any()):
        raise ValueError('Jacobi input must be mask-free')
    all_rows = combined[0]
    kpred_calls, kpred_total_ms = ((0, 0.0) if rows.predictor is None else
                                  rows.predictor.consume_metrics())
    packed = None
    if H == capacity:
        # one copy of every request's rows (the padded rows of the fixed bucket are already
        # -1; an empty round's private row is hidden); the per-request tensors are its views
        kept = pass_rows if compact else count * H
        wires = all_rows.block_wire[:kept].clone()
        if compact:
            # the compact rows back at their bucket rows; every row the pass left out and
            # every inactive compact row is -1, as the fixed-cap pass leaves a dead row
            candidates = wires.new_full((padded * H, wires.shape[-1]), -1)
            candidates.index_copy_(0, row_order, wires)
            candidates = candidates[:count * H].view(count, H, -1)
        else:
            candidates = wires.view(count, H, -1).clone()
        empty = [n for n, stage in enumerate(stages) if not len(stage['keys'])]
        if empty:
            candidates[empty, 0] = -1
        packed = dict(wires=wires, candidates=candidates,
                      **{name: getattr(all_rows, name)[:kept].clone() for name in _FIELDS
                         if getattr(all_rows, name, None) is not None})
    heads_fast = values.get('engine.engine_settings.heads_fast') == '1'
    output = []
    for n, (sequence, prediction, stage) in enumerate(zip(sequences, predictions, stages)):
        row0 = row_starts[n]
        sl = slice(row0, row0 + H)
        live = len(stage['keys'])
        if packed is not None:
            wire = packed['wires'][sl][:live]
            candidate = packed['candidates'][n]
        else:
            wire = all_rows.block_wire[sl][:live].clone()
            candidate = wire.new_full((capacity, wire.shape[-1]), -1)
            candidate[:live] = wire
        row_keys = stage['keys'].new_full((capacity, 2), -1)
        row_keys[:live] = stage['keys']

        def kwargs(part):
            if packed is not None:
                return {name: packed[name][part] for name in _FIELDS if name in packed}
            return {name: getattr(all_rows, name)[part].clone()
                    for name in _FIELDS if getattr(all_rows, name, None) is not None}

        metadata = all_rows.metadata[sl][:live]
        result = PackedBlocks(wire, None if all_rows.logits is None else all_rows.logits[sl][:live],
                              metadata.clone(), stage['keys'][:, 0].tolist(),
                              candidate_wire=candidate, row_keys=row_keys,
                              predictor=rows.predictor, **kwargs(slice(row0, row0 + live)),
                              run_config=run_config)
        first_row = PackedBlocks(packed['wires'][row0:row0 + 1] if packed is not None
                                 else all_rows.block_wire[row0:row0 + 1].clone(),
                                 None if all_rows.logits is None else all_rows.logits[row0:row0 + 1],
                                 all_rows.metadata[row0:row0 + 1].clone(),
                                 [int(stage['keys'][0, 0]) if live else 0], predictor=rows.predictor,
                                 **kwargs(slice(row0, row0 + 1)), run_config=run_config)
        result.posterior = True
        if heads_fast:
            result._fallback_blocks = first_row
        else:
            result.fallback = first_row[0]
        result.head_metrics = dict(
            decoder_ready=True, decoder_ms=prediction.elapsed_ms,
            decoder_total_ms=prediction.total_ms, decoder_prefix_ms=prediction.prefix_ms,
            decoder_select_ms=prediction.select_ms,
            kpred_ms=float(sequence.pending_block.kpred_ms), kpred_calls=kpred_calls,
            kpred_total_ms=kpred_total_ms, lac_fallbacks=prediction.offers.lac_fallbacks,
            kpred_pass=sequence.pending_block.kpred_pass, rows_drafted=H,
            rows_published=live, rows_truncated=stage['truncated'],
            rows_requested=stage['requested'], k_set_size=len(stage['ks']),
            k_set=list(map(int, stage['ks'])), anchor_set_sizes=stage['sizes'],
            k15_dropped=0, kpred_frame_dropped0=0,
            ctx_update=bucket.head.last_ctx_update, ctx_rows=bucket.head.last_ctx_rows,
            bucket_built=int(built), row_layout='anchor0')
        output.append(result)
    host_mark('flat_ret', run_config=run_config)
    return output


@torch.inference_mode()
def prewarm_batch_graphs(rows, make_sequence, *, max_batch, max_model_len, initial_ks, round_ks,
                         passes, run_config=None):
    """Before any request is admitted, capture every C>1 graph a read can reach: each batch
    bucket up to ``max_batch`` (the drain buckets too) at each 128-row prefix bucket up to
    ``max_model_len``, with two synthetic rounds per cell (a cold round, then one that appends
    up to a window per request). Returns the number of (bucket, prefix) cells warmed."""
    run_config = _config(rows, run_config)
    from jetspec.engines.jetspec_plus_plus.wire.protocol import draft_width
    served_block = draft_width(run_config=run_config) + 1
    buckets = sorted({batch_draft_bucket(n) for n in range(2, int(max_batch) + 1)})
    if not buckets:
        return 0
    lengths = [128 * k - 20 for k in range(1, max_model_len // 128 + 1)
               if 128 * k - 20 + 16 <= max_model_len - 4]
    rows_prefixes = {((length + 16 + 255) // 256) * 256 for length in lengths}
    rows._batch_graph_bucket_limit = max(6, len(buckets) * (len(rows_prefixes) + 1) + 2)
    if not rows.__dict__.get('graphs', False):
        rows._batch_admission_graphs = True
    if rows.decoder is not None and hasattr(rows.decoder, 'presize_batch_pool'):
        rows.decoder.presize_batch_pool(max(32, 2 * buckets[-1] + 2))
    cells = 0
    # (1, shortest prefix) first: a read's first and last rounds can hold one live sequence, whose
    # decoder takes the one-request select (the batch epilogue serves two or more)
    plan = [(1, lengths[0])] + [(bucket, length) for bucket in buckets for length in lengths]
    # The adapter's batched graphs share one pool. Capture them longest first, so each shorter
    # capture reuses the longer one's freed blocks (ascending capture stranded 31 GiB at 27B C4).
    # Only the adapter stage runs here; the row and context graphs keep the ascending plan below.
    for warm, (bucket, length) in enumerate((bucket, length) for bucket in buckets
                                             for length in reversed(lengths)):
        sequences = [make_sequence(max(8, length - n)) for n in range(bucket)]
        for n, sequence in enumerate(sequences):
            sequence._jetspec_seq_id = -100000 - 64 * warm - n
            sequence._jetspec_candidate_round = 1
        chosen = [select_live_block(rows.propose([sequence], k_hypotheses=initial_ks, passes=passes,
                                                 initial=True, run_config=run_config)[0],
                                    true_k=initial_ks[0], correction=-1)
                  for sequence in sequences]
        rows.stage(sequences, chosen, run_config=run_config)
        del sequences, chosen
        gc.collect()
    for bucket, length in plan:
        sequences = [make_sequence(max(8, length - n)) for n in range(bucket)]
        for n, sequence in enumerate(sequences):
            # a warm-up id per sequence (never a live one) and one round per call keep
            # the recorders' (seq_id, round) keys unique
            sequence._jetspec_seq_id = -1000 - 64 * cells - n
            sequence._jetspec_candidate_round = 1
        chosen = []
        for sequence in sequences:
            blocks = rows.propose([sequence], k_hypotheses=initial_ks, passes=passes,
                                  initial=True, run_config=run_config)
            chosen.append(select_live_block(blocks[0], true_k=initial_ks[0], correction=-1))
        for round_index in range(2):
            for sequence in sequences:
                sequence._jetspec_candidate_round = 1 + round_index
            rows.stage(sequences, chosen, run_config=run_config)
            prefetch_batch(rows, sequences, run_config=run_config)
            outputs = propose_batch(rows, sequences, k_hypotheses=round_ks, passes=passes,
                                    run_config=run_config)
            chosen = [select_live_block(blocks, true_k=0, correction=-1) for blocks in outputs]
            for n, sequence in enumerate(sequences):
                grow = 1 + (5 * n + round_index) % served_block
                sequence.taps = torch.cat((sequence.taps, sequence.taps.new_zeros(
                    (grow, *sequence.taps.shape[1:]))))
                sequence.committed_ids = torch.cat((sequence.committed_ids,
                    sequence.committed_ids.new_zeros((1, grow))), dim=1)
        cells += 1
        del sequences, chosen, outputs
        gc.collect()
    # every request count's captured assembly, not only the bucket sizes the cells ran (a
    # draining batch passes through every count)
    assemblies = rows.__dict__.get('_batch_row_assemblies') or {}
    for key, template in list(assemblies.items()):
        for count in range(1, int(max_batch) + 1):
            if (count,) + key[1:] not in assemblies:
                assemblies[(count,) + key[1:]] = template.for_count(count, batch_draft_bucket(count))
                assemblies[(count,) + key[1:]].run()
    if rows.drafter.device.type == 'cuda':
        torch.cuda.synchronize(rows.drafter.device)
    print(f'BATCH_DRAFT_GRAPH_PREWARM buckets={buckets} prefix_lengths={lengths} cells={cells} '
          f'rows_bucket_limit={rows._batch_graph_bucket_limit} '
          f'assemblies={len(assemblies)}', flush=True)
    return cells
