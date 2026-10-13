"""Compose the real U1, U2, and U3 bodies for one N=1 chain key."""
from __future__ import annotations

import os
import time

import torch

from jetspec.engines.jetspec_plus_plus.wire.chain.contract import ChainKey; from jetspec.engines.jetspec_plus_plus.wire.chain.contract import Offset; from jetspec.engines.jetspec_plus_plus.wire.chain.contract import OUTCOME_WORDS; from jetspec.engines.jetspec_plus_plus.wire.chain.contract import allocate_buffers; from jetspec.engines.jetspec_plus_plus.wire.chain.contract import outcome_message_views; from jetspec.engines.jetspec_plus_plus.wire.chain.contract import rng_words
from jetspec.engines.jetspec_plus_plus.wire.chain.intake import ChainRngTable
from jetspec.engines.jetspec_plus_plus.wire.chain.publish import ChainPublisher
from jetspec.engines.jetspec_plus_plus.draft.env import context_append_enabled
from jetspec.engines.jetspec_plus_plus.draft.pass_labels import chain_labels


def seed_front_from_admission(front, sequence, decoder, thresholds, dataset):
    """Transfer the admission decoder frame and K/V into stable chain pointers."""
    from jetspec.engines.jetspec_plus_plus.models.heads.fast import dataset_entry

    staged = sequence.staged.predicted_hidden
    stale = sequence.decoder_hidden_tail
    if staged is None or staged.shape != front.staged_hidden.shape:
        raise ValueError("admission lacks a full static C1 prediction block")
    if stale is None or stale.shape != front.decoder_stale.shape:
        raise ValueError("admission lacks four decoder stale rows")
    prefix = decoder._prefixes.get(id(sequence))
    expected = sequence.taps.shape[0] - stale.shape[0]
    if prefix is None or prefix.length != expected:
        raise ValueError("admission decoder prefix has the wrong committed length")
    front.staged_hidden.copy_(staged)
    front.decoder_stale.copy_(stale)
    for layer, (key, value) in front.decoder_prefix_kv.items():
        if layer not in prefix.kv:
            raise ValueError("admission decoder prefix is missing a layer")
        source_key, source_value = prefix.kv[layer]
        key[..., :expected, :].copy_(source_key[..., :expected, :])
        value[..., :expected, :].copy_(source_value[..., :expected, :])
    if getattr(front, "hybrid_recurrent", None) is not None:
        front.hybrid_recurrent.copy_(sequence.hybrid_recurrent)
        front.hybrid_conv.copy_(sequence.hybrid_conv)
        if sequence.hybrid_prefix_key.shape[0] > front.hybrid_prefix_key.shape[0]:
            raise ValueError("hybrid admission prefix exceeds static decoder capacity")
        front.hybrid_prefix_key.zero_()
        front.hybrid_prefix_value.zero_()
        front.hybrid_prefix_key[:sequence.hybrid_prefix_key.shape[0]].copy_(
            sequence.hybrid_prefix_key)
        front.hybrid_prefix_value[:sequence.hybrid_prefix_value.shape[0]].copy_(
            sequence.hybrid_prefix_value)
    coverage = os.environ.get('JETSPEC_ANCHOR_COVERAGE', '0.99')
    anchor = thresholds['anchor_predictor']
    if os.environ.get('JETSPEC_ANCHOR_DREJ', '0') == '1':
        values = dataset_entry(anchor['drej_top1_grouped'], dataset)[coverage]
        cutoffs = (float(values['drej-top1']['value']),
                   float(values['not-drej-top1']['value']))
    else:
        value = float(dataset_entry(anchor['pooled'], dataset)[coverage]['value'])
        cutoffs = (value, value)
    front.selection_thresholds.copy_(front.selection_thresholds.new_tensor(cutoffs))


def seed_row_ring_from_admission(ring, candidate, blocks, pass_indices=None):
    """Snapshot exactly the round-zero rows sent to the verifier."""
    if candidate.round_id != 0:
        raise ValueError("admission row ring requires a round-zero candidate")
    if candidate.blocks.ndim != 2 or candidate.row_keys.shape != (candidate.blocks.shape[0], 2):
        raise ValueError("admission candidate requires keyed rows")
    rows, lanes = ring.rows.shape[1:]
    if candidate.blocks.shape[1] != lanes or candidate.blocks.shape[0] < rows:
        raise ValueError("admission candidate does not fit the chain row ring")
    scores = getattr(blocks, "predictor_p_top1", None)
    if scores is None or scores.ndim != 2 or scores.shape[1] != lanes:
        raise ValueError("admission lacks frozen row top-1 scores")
    staged = torch.zeros(rows, lanes, dtype=torch.float32, device=ring.rows.device)
    count = min(rows, scores.shape[0])
    fast = os.environ.get("JETSPEC_CHAIN_ADMIT_GRAPH", "0") == "1"

    def upload(tensor, dtype=None):
        # JETSPEC_CHAIN_ADMIT_GRAPH: a host tensor goes up through pinned memory, so the
        # snapshot does not wait for the admission work queued ahead of it.
        tensor = tensor if dtype is None else tensor.to(dtype=dtype)
        if fast and tensor.device.type == "cpu" and ring.rows.device.type == "cuda":
            return tensor.pin_memory().to(ring.rows.device, non_blocking=True)
        return tensor.to(ring.rows.device)

    if fast:
        staged[:count].copy_(upload(scores[:count], staged.dtype))
    else:
        staged[:count].copy_(scores[:count].to(staged.device, dtype=staged.dtype))
    passes = torch.zeros(rows, dtype=torch.int64, device=ring.rows.device)
    if pass_indices is not None:
        if pass_indices.ndim != 1 or pass_indices.numel() < rows:
            raise ValueError("admission pass labels must cover the row ring")
        passes.copy_(pass_indices[:rows].to(passes.device, dtype=passes.dtype) + 1)
    ring.publish(0, upload(candidate.blocks[:rows]),
                 upload(candidate.row_keys[:rows]), staged, passes)
    if hasattr(ring, 'draft_top16_ids'):
        top = blocks.eval_draft_top16
        staged_top = (torch.full_like(ring.draft_top16_ids[0], -1),
                      torch.zeros_like(ring.draft_top16_probabilities[0]),
                      torch.zeros_like(ring.draft_token_probabilities[0]),
                      torch.zeros_like(ring.draft_pinned[0]))
        for target, value in zip(staged_top, top+blocks.eval_draft_values):
            target[:count].copy_(upload(value[:count]))
        ring.publish_candidates(0, *staged_top)
        from jetspec.engines.jetspec_plus_plus.wire.chain.wire_records import candidate_snapshot
        from jetspec.engines.jetspec_plus_plus.models.heads.wirecal import append_proposal_record; from jetspec.engines.jetspec_plus_plus.models.heads.wirecal import proposal_record
        from jetspec.engines.jetspec_plus_plus.draft.pass_labels import completed_label
        record = proposal_record(seq_id=candidate.seq_id, candidate_round=0,
            pass_label=completed_label(passes[0]), predictor_probabilities=None, anchor_logits=None,
            rejected=None, after_miss=False, k_set=list(range(count)), rows_requested=count,
            rows_published=count, rows_truncated=0, fallback_anchor=int(candidate.row_keys[0, 1]),
            published_keys=candidate.row_keys[:count])
        record.update(schema='jetspec.eval_round_candidates/1', record_kind='admission',
                      p_top1=staged[0].cpu().tolist(), q_shadow=None, shadow_argmax=None,
                      published_blocks=candidate.blocks[:count].cpu().tolist(),
                      draft_top16=candidate_snapshot(staged_top[0][0], staged_top[1][0]),
                      published_draft_top16=candidate_snapshot(*staged_top[:2]),
                      draft_token_ids=candidate.blocks[0].cpu().tolist(),
                      draft_token_probabilities=staged_top[2][0].cpu().tolist(),
                      draft_pinned=staged_top[3][0].cpu().tolist(),
                      published_draft_token_probabilities=staged_top[2].cpu().tolist(),
                      published_draft_pinned=staged_top[3].cpu().tolist(),
                      published_pass_labels=[completed_label(value) for value in passes[:count]])
        append_proposal_record(os.environ['W280_RECORD'], record)


def reseed_served_graph(graph, *, initial_ids, initial_taps, candidate, blocks,
                        admission_hidden, admission_tau, rng_rows, fake_packet):
    """Reuse captured pointers for a new sequence without a request-time capture."""
    if not graph._prewarmed or not graph.entries:
        raise RuntimeError("chain graph must be fully captured before sequence reseed")
    shared = next(iter(graph.entries.values())).buffers
    tokens, tap_store, ring = (shared[name] for name in
                               ("tokens", "tap_store", "row_ring"))
    batch = os.environ.get("JETSPEC_CHAIN_ADMIT_BATCH", "0") == "1"
    timing = os.environ.get("JETSPEC_CHAIN_ADMIT_TIMING", "0") == "1"
    start = time.perf_counter() if timing else None
    if not batch:
        tokens.committed_ids.zero_()
    tokens.seed_committed(initial_ids)
    tokens.pending_draft_ids.zero_()
    tap_store.storage.zero_()
    tap_store.seed(initial_taps)
    if batch:
        ring.rows.zero_()
        ring.p_top1.zero_()
        ring.passes_run.zero_()
        ring.keys.fill_(-1)
        ring.rounds.copy_(torch.arange(3, device=ring.rounds.device))
    else:
        for old_round in range(3):
            ring.empty(old_round)
    seed_row_ring_from_admission(ring, candidate, blocks, admission_tau)
    if batch:
        targets = getattr(graph, "_admit_reseed_targets", None)
        if targets is None:
            targets = tuple(tuple(entry.buffers[name] if name != "rng_table"
                                  else entry.buffers[name].rows
                                  for name in ("admission_hidden", "admission_tau", "rng_table",
                                               "packet", "taps", "rp", "expected_round"))
                            for entry in graph.entries.values())
            graph._admit_reseed_targets = targets
        for column, source in enumerate((admission_hidden, admission_tau, rng_rows, fake_packet)):
            torch._foreach_copy_([target[column] for target in targets],
                                 [source] * len(targets))
        torch._foreach_zero_([target[column] for target in targets for column in (4, 5, 6)])
    else:
        for entry in graph.entries.values():
            buffers = entry.buffers
            buffers["admission_hidden"].copy_(admission_hidden)
            buffers["admission_tau"].copy_(admission_tau)
            buffers["rng_table"].rows.copy_(rng_rows)
            buffers["packet"].copy_(fake_packet)
            buffers["taps"].zero_()
            buffers["rp"].zero_()
            buffers["expected_round"].zero_()
    graph.last_ms.clear()
    graph.last_pre_a0_gap_ms.clear()
    graph.last_a0_to_a1_gap_ms.clear()
    graph.mix = {name: 0 for name in graph.mix}
    if hasattr(graph, "cap_mix"):
        graph.cap_mix = {name: 0 for name in graph.cap_mix}
    graph.fallback_counts = {name: 0 for name in graph.fallback_counts}
    graph.round_syncs = 0
    if timing:
        if initial_ids.is_cuda:
            torch.cuda.synchronize(initial_ids.device)
        print(f"A29_ADMIT2 reseed_ms={(time.perf_counter() - start) * 1000:.3f}", flush=True)


def stage_admission_rows(rows, state, blocks, candidate, *, row_cap):
    """Stage every row that prefill published before the first outcome exists."""
    from jetspec.engines.jetspec_plus_plus.models.heads.decoder import e04_seed_admission_tail

    if state.pending_block is not None or state.staged.predicted_hidden is not None:
        raise ValueError("chain admission expected an unstaged prefill sequence")
    if candidate.blocks.shape[0] != row_cap:
        raise ValueError("chain admission row bank must cover the full published wire")
    timing = os.environ.get("JETSPEC_CHAIN_ADMIT_TIMING", "0") == "1"
    if (os.environ.get("JETSPEC_CHAIN_ADMIT_GRAPH", "0") == "1"
            and os.environ.get("JETSPEC_CHAIN_ADMIT_BATCH", "0") == "1"):
        live = [(index, blocks[index]) for index, key in enumerate(candidate.row_keys.tolist())
                if key[0] >= 0]
        if len(live) > 1:
            return _stage_admission_rows_fast(rows, state, live, row_cap=row_cap,
                                              timing=timing)
    start = time.perf_counter() if timing else None
    seeded = e04_seed_admission_tail(rows.drafter, state,
                                     hidden_size=rows.decoder.hidden_size)
    if timing:
        if state.taps.is_cuda:
            torch.cuda.synchronize(state.taps.device)
        seed_ms = (time.perf_counter() - start) * 1000
    if not seeded:
        reasons = []
        if state.staged.start_position is not None:
            reasons.append("staged.start_position is already set")
        if state.decoder_hidden_tail is not None:
            reasons.append("decoder_hidden_tail already exists")
        if not hasattr(state, "_jetspec_seq_id"):
            reasons.append("_jetspec_seq_id is missing (synthetic warm-up state)")
        raise RuntimeError("chain admission tail seed refused: " +
                           (", ".join(reasons) if reasons else "unknown precondition"))
    if state.decoder_hidden_tail is None:
        raise RuntimeError("chain admission tail seed returned success without four rows")
    start = time.perf_counter() if timing else None
    rows.decoder._prefix(state, state.taps.shape[0] - 4)
    if timing:
        if state.taps.is_cuda:
            torch.cuda.synchronize(state.taps.device)
        prefix_ms = (time.perf_counter() - start) * 1000
    start = time.perf_counter() if timing else None
    bank = None
    tau = torch.zeros(row_cap, dtype=torch.int64, device=state.taps.device)
    tau_labels = chain_labels()
    live = [(index, blocks[index]) for index, key in enumerate(candidate.row_keys.tolist())
            if key[0] >= 0]
    if os.environ.get("JETSPEC_CHAIN_ADMIT_BATCH", "0") == "1" and len(live) > 1:
        predicted = rows.stage_admission_batch(state, [block for _, block in live]).detach()
        bank = predicted.new_zeros(row_cap, *predicted.shape[1:])
        bank.index_copy_(0, torch.tensor([index for index, _ in live], device=bank.device),
                         predicted)
        for index, block in live:
            if str(block.kpred_pass) not in tau_labels:
                raise ValueError("chain admission row has no calibrated pass label")
            tau[index] = tau_labels.index(str(block.kpred_pass))
        if timing:
            if bank.is_cuda:
                torch.cuda.synchronize(bank.device)
            print(f"A29_ADMIT2 seed_tail_ms={seed_ms:.3f} decoder_prefix_ms={prefix_ms:.3f} "
                  f"rows_stage_ms={(time.perf_counter() - start) * 1000:.3f}", flush=True)
        return bank, tau
    for index, block in live:
        rows.stage([state], [block])
        if str(block.kpred_pass) not in tau_labels:
            raise ValueError("chain admission row has no calibrated pass label")
        tau[index] = tau_labels.index(str(block.kpred_pass))
        predicted = state.staged.predicted_hidden[0].detach()
        if bank is None:
            bank = predicted.new_zeros(row_cap, *predicted.shape)
        bank[index].copy_(predicted)
    if bank is None:
        raise RuntimeError("chain admission published no live rows")
    if timing:
        if bank.is_cuda:
            torch.cuda.synchronize(bank.device)
        print(f"A29_ADMIT2 seed_tail_ms={seed_ms:.3f} decoder_prefix_ms={prefix_ms:.3f} "
              f"rows_stage_ms={(time.perf_counter() - start) * 1000:.3f}", flush=True)
    return bank, tau




def _stage_admission_rows_fast(rows, state, live, *, row_cap, timing):
    """JETSPEC_CHAIN_ADMIT_GRAPH: the batched admission (two or more live rows) without a host wait.

    The same three device computations as the batched path, each on its own state:
    the adapter tail seed, the C1 row bank, then the decoder prefix, whose host work
    runs while the device replays the bank. The row keys and pass labels are read
    before the first launch; the bank index and labels upload through pinned memory.
    """
    from jetspec.engines.jetspec_plus_plus.models.heads.decoder import e04_seed_admission_tail
    from jetspec.engines.jetspec_plus_plus.draft.rows import device_tensor

    tau_labels = chain_labels()
    labels = [0] * row_cap
    for index, block in live:
        if str(block.kpred_pass) not in tau_labels:
            raise ValueError("chain admission row has no calibrated pass label")
        labels[index] = tau_labels.index(str(block.kpred_pass))
    device = state.taps.device
    stamps = []

    def stamp():
        if timing:
            if state.taps.is_cuda:
                torch.cuda.synchronize(device)
            stamps.append(time.perf_counter())

    stamp()
    if not e04_seed_admission_tail(rows.drafter, state, hidden_size=rows.decoder.hidden_size):
        _refuse_unseeded_tail(state)
    if state.decoder_hidden_tail is None:
        raise RuntimeError("chain admission tail seed returned success without four rows")
    stamp()
    predicted = rows.stage_admission_batch(state, [block for _, block in live]).detach()
    stamp()
    rows.decoder._prefix(state, state.taps.shape[0] - 4)
    stamp()
    bank = predicted.new_zeros(row_cap, *predicted.shape[1:])
    bank.index_copy_(0, device_tensor([index for index, _ in live], bank.device), predicted)
    tau = device_tensor(labels, device)
    if timing:
        stamp()
        seed_ms, rows_ms, prefix_ms, bank_ms = ((after - before) * 1000 for before, after
                                                in zip(stamps, stamps[1:]))
        print(f"A29_ADMIT2 seed_tail_ms={seed_ms:.3f} decoder_prefix_ms={prefix_ms:.3f} "
              f"rows_stage_ms={rows_ms + bank_ms:.3f}", flush=True)
    return bank, tau


def copy_front_state(source, target, *, prefix_length: int) -> None:
    """Carry the decoder window and exact prefix across a chain-key boundary."""
    if prefix_length < 0:
        raise ValueError("decoder prefix length must be nonnegative")
    target.staged_hidden.copy_(source.staged_hidden)
    target.decoder_stale.copy_(source.decoder_stale)
    for layer, (target_key, target_value) in target.decoder_prefix_kv.items():
        if layer not in source.decoder_prefix_kv:
            raise ValueError("source decoder prefix is missing a layer")
        source_key, source_value = source.decoder_prefix_kv[layer]
        if prefix_length > min(target_key.shape[-2], source_key.shape[-2]):
            raise ValueError("decoder prefix exceeds a chain key's storage")
        target_key[..., :prefix_length, :].copy_(source_key[..., :prefix_length, :])
        target_value[..., :prefix_length, :].copy_(source_value[..., :prefix_length, :])


def shadow_state_tensors(buffers, rows):
    """Mutable state read by the fused eager bodies and the captured chain.

    A T0 round saves these tensors, executes the eager bodies, restores them,
    then replays the graph. The snapshot includes the shared C1/decoder cache
    and row bucket, not just the public output bundle.
    """
    values = []
    seen = set()


    for value in buffers.values():
        if isinstance(value, torch.Tensor) and value.device.type != "cpu":
            add(value)
    for name in ("tokens", "tap_store", "row_ring", "front", "publisher"):
        obj = buffers.get(name)
        if obj is not None:
            for value in vars(obj).values():
                add(value)
    captured_c1 = buffers.get("c1_capture_state")
    if captured_c1 is not None:
        for name in ("aligned", "u", "window", "q", "kv_sources", "outputs"):
            add(captured_c1.get(name))
    for bucket_name in ("bucket", "redraft_bucket"):
        bucket = buffers.get(bucket_name)
        if bucket is None:
            continue
        add(bucket.ids)
        add(getattr(bucket, "chain_lmhead_partials", None))
        for key, value in vars(bucket.head).items():
            if key != "head":  # trained weights are read-only
                add(value)
    head = rows.drafter.adapter.head
    for name in ("_c1_kv", "_c1_fast_key_cache", "_c1_fast_value_cache",
                 "_c1_dirty", "_c1_dirty_positions", "_c1_window_sets",
                 "_c1_qcos", "_c1_qsin", "_c1_mask", "_c1_output"):
        add(getattr(head, name, None))
    add(getattr(rows.decoder, "_window_graph_state", None))
    add(getattr(rows.decoder, "_window_graph_states", None))  # decoder key bucket: one state per bucket
    return tuple(values)






def build_served_entry(key: ChainKey, *, rows, tokens, tap_store, row_ring,
                       thresholds, dataset, tau_table, temperature,
                       hypotheses: int, wire_row_cap: int, admission=None,
                       admission_hidden=None, admission_tau=None):
    """Bind stable pointers once; returned bodies only read device round state.

    The caller copies the received envelope and uploaded RoundParams into the
    returned buffers before ChainGraph.run. A key carries its own pass bucket
    and wire; token/tap mirrors and the published-row ring belong to the sequence.
    """
    from jetspec.engines.jetspec_plus_plus.draft.drafter import bind_front_inputs
    from jetspec.engines.jetspec_plus_plus.draft.live_rows import BucketPass; from jetspec.engines.jetspec_plus_plus.draft.live_rows import passes_chain_segments; from jetspec.engines.jetspec_plus_plus.draft.live_rows import rows_chain_body; from jetspec.engines.jetspec_plus_plus.draft.live_rows import rows_chain_home_words

    if key.lanes not in (16, 32) or key.R not in (4, 8, 16):
        raise ValueError("served pair chain requires b16/b32 and r4/r8/r16")
    drafter = rows.drafter
    decoder = rows.decoder
    if decoder is None:
        raise ValueError("served chain requires the decoder anchor")
    front = bind_front_inputs(drafter, decoder, key.L_bucket)
    if key.decoder_key_bucket and front.decoder_bucket != key.decoder_key_bucket:
        raise ValueError("chain decoder key does not match its bound window graph")
    if admission is not None:
        seed_front_from_admission(front, admission, decoder, thresholds, dataset)
    prefix_bucket = ((key.L_bucket + 255) // 256) * 256
    bucket = BucketPass(drafter, N=1, H=key.R, prefix_bucket=prefix_bucket,
                        backend=rows.backend, row_capacity=key.R, flat=True,
                        graphs=False, refresh=True, anchor0=True, wide=True)
    hidden = front.staged_hidden.shape[-1]
    vocab = decoder.lm_head.weight.shape[0]
    buffers = allocate_buffers(device=drafter.device, R=key.R, lanes=key.lanes,
                               hidden=hidden, vocab=vocab,
                               row_home_words=rows_chain_home_words(key.R, key.lanes))
    tap_dtype = front.taps.dtype
    tap_bytes = front.taps.numel() * front.taps.element_size()
    if getattr(front, "hybrid_recurrent", None) is None:
        envelope = torch.empty(OUTCOME_WORDS * 8 + tap_bytes, dtype=torch.uint8,
                               device=drafter.device)
        packet, taps = outcome_message_views(
            envelope, lanes=key.lanes, hidden=front.taps.shape[-1], tap_dtype=tap_dtype)
        hybrid_views = None
    else:
        envelope = torch.empty(decoder.layout.outcome_bytes, dtype=torch.uint8,
                               device=drafter.device)
        packet, taps, *hybrid_views = hybrid_outcome_views(envelope, decoder.layout)
    publisher = ChainPublisher(device=drafter.device, hypotheses=hypotheses,
                               row_cap=wire_row_cap, rows=key.R, lanes=key.lanes)
    buffers.update(packet=packet, taps=taps, envelope=envelope, tokens=tokens,
                   tap_store=tap_store, row_ring=row_ring, front=front,
                   bucket=bucket, publisher=publisher,
                   staged_hidden=front.staged_hidden,
                   wire_rows=publisher.wire_rows, wire_keys=publisher.wire_keys,
                   wire_p_top1=publisher.scores, wire_home=publisher.home,
                   published=torch.empty((), dtype=torch.bool, device=drafter.device),
                   intake_err=torch.zeros(1, dtype=torch.int64, device=drafter.device))
    if hasattr(row_ring, 'draft_top16_ids'):
        buffers['prior_draft_top16_ids'] = torch.empty_like(row_ring.draft_top16_ids[0, 0])
        buffers['prior_draft_top16_probabilities'] = torch.empty_like(row_ring.draft_top16_probabilities[0, 0])
        buffers['wire_draft_top16_ids'] = torch.full_like(row_ring.draft_top16_ids[0], -1)
        buffers['wire_draft_top16_probabilities'] = torch.zeros_like(row_ring.draft_top16_probabilities[0])
        buffers['prior_draft_token_ids'] = torch.empty_like(row_ring.rows[0, 0])
        buffers['prior_draft_token_probabilities'] = torch.empty_like(row_ring.draft_token_probabilities[0, 0])
        buffers['prior_draft_pinned'] = torch.empty_like(row_ring.draft_pinned[0, 0])
        buffers['wire_draft_token_probabilities'] = torch.zeros_like(row_ring.draft_token_probabilities[0])
        buffers['wire_draft_pinned'] = torch.zeros_like(row_ring.draft_pinned[0])
    from jetspec.engines.jetspec_plus_plus.draft.eval_profile import capture_decoder_inputs
    if capture_decoder_inputs():
        # Eval-only: the decoder window input and its prefix extent, copied inside the graph.
        stale, staged = front.decoder_stale, front.staged_hidden
        buffers['capture_dec_window'] = torch.empty(
            (stale.shape[0] + staged.shape[1], stale.shape[-1]), dtype=stale.dtype, device=stale.device)
        buffers['capture_dec_lengths'] = torch.empty(2, dtype=torch.int64, device=stale.device)
    if hybrid_views is not None:
        buffers.update(zip(("hybrid_recurrent", "hybrid_conv",
                            "hybrid_keys", "hybrid_values"), hybrid_views))
    if admission_hidden is not None:
        if admission_hidden.shape != (row_ring.rows.shape[1], key.lanes, hidden):
            raise ValueError("admission C1 bank must match all published rows")
        buffers["admission_hidden"] = admission_hidden
        if admission_tau is None or admission_tau.shape != (row_ring.rows.shape[1],):
            raise ValueError("admission pass labels must cover all published rows")
        buffers["admission_tau"] = admission_tau
    if admission is not None:
        buffers["rng_table"] = ChainRngTable(
            capacity=tokens.committed_ids.numel(), lanes=key.lanes,
            seed=drafter.seed, prompt_index=admission.prompt_index,
            device=drafter.device)
    T_ref = ((key.R * rng_words(key.lanes) + 15) // 16) * 16
    swap = None
    fixed_config = os.environ.get("JETSPEC_CHAIN_FIXED_CONFIG")
    from jetspec.engines.jetspec_plus_plus.draft.eval_profile import FIXED_CONFIGS; from jetspec.engines.jetspec_plus_plus.draft.eval_profile import enabled as eval_enabled; from jetspec.engines.jetspec_plus_plus.draft.eval_profile import states_only
    from jetspec.engines.jetspec_plus_plus.draft.chain.refresh_swap import refresh_swap_enabled
    # A states-only R0 schedule has no refresh and needs no swap/pin state.
    if refresh_swap_enabled() and not (states_only() and fixed_config in FIXED_CONFIGS
                                       and fixed_config.endswith('-R0')):
        # JETSPEC_REFRESH_SWAP: the decoder's refresh swap inside B1r (default off)
        from jetspec.engines.jetspec_plus_plus.draft.chain.refresh_swap import RefreshSwap; from jetspec.engines.jetspec_plus_plus.draft.chain.refresh_swap import check_regime
        check_regime()
        swap = RefreshSwap(decoder, front, tap_store, rows=key.R, lanes=key.lanes)
        if getattr(swap, 'fill_mode', 'keep') == 'noise':
            from functools import partial
            buffers['after_swap_fill_prepare'] = partial(
                swap.prepare_noise, vocab=drafter.lm_head.weight.shape[0],
                mask_token=int(drafter.head.mask_token_id))
    from jetspec.engines.jetspec_plus_plus.draft.method.schedule import Schedule
    schedule = Schedule.parse(fixed_config) if fixed_config else Schedule(3, 1)
    # Historical R0 cells warm B1r; states-only R0 needs no swap/pin state.
    refresh_at = schedule.refresh_at if states_only() else schedule.refresh_at or (1,)
    trailing = tuple(index for index in schedule.refresh_at if index > schedule.j)
    passes = passes_chain_segments(
        buffers["rp"], tokens, buffers["keys"], buffers["active"],
        front.staged_hidden, bucket, J=max(3, schedule.j), refresh_at=refresh_at, T_ref=T_ref,
        outputs=buffers, **({"refresh_swap": swap} if swap is not None else {}),
        **({'standalone_refresh_at': trailing} if trailing else {}))
    if swap is not None:
        buffers.update(swap.tensors())

    attn3 = os.environ.get("JETSPEC_CHAIN_ATTN3", "0") == "1"
    head = drafter.adapter.head
    if attn3:
        from jetspec.engines.jetspec_plus_plus.kernels.attn_sm100 import flashinfer_enabled

        # Planning and JIT happen once per key before any segment is captured.
        cache = head._c1_kv[0][0]
        if not flashinfer_enabled(drafter.device, front.c1_query_embeddings.dtype,
                                  chain=True):
            raise ValueError("JETSPEC_CHAIN_ATTN3 requires sm100 bf16/fp16 FlashInfer")
        capture_L = (min(key.L_bucket, admission.taps.shape[0])
                     if admission is not None else key.L_bucket)
        plan_served_prefix(head, key, capture_L, refresh=True)
        cos = torch.empty(1, key.lanes, cache.shape[-1],
                          device=drafter.device, dtype=front.c1_query_embeddings.dtype)
        sin = torch.empty_like(cos)
        mask = torch.empty(1, 1, key.lanes, front.c1_bucket,
                           device=drafter.device, dtype=front.c1_query_embeddings.dtype)
        buffers.update(c1_cos=cos, c1_sin=sin, c1_mask=mask)


        def first_finish():
            head.c1_chain_finish(buffers["rp"], front, cos, sin, mask,
                                 buffers["c1_state"])
    else:
        def first_finish():
            head.c1_chain_body(buffers["rp"], tokens, front, bucket)

    redraft = os.environ.get("JETSPEC_CHAIN_MISS_REDRAFT", "0") == "1"
    if redraft:
        from jetspec.engines.jetspec_plus_plus.wire.chain.contract import OutcomeOffset

        buffers["control_round"] = torch.zeros((), dtype=torch.int64, device=drafter.device)
        buffers["control_L"] = torch.zeros((), dtype=torch.int64, device=drafter.device)
        redraft_bucket = BucketPass(drafter, N=1, H=1,
                                     prefix_bucket=prefix_bucket,
                                     backend=rows.backend, row_capacity=1,
                                     flat=True, graphs=False, refresh=True,
                                     anchor0=True, wide=True)
        redraft_keys = torch.empty(1, 2, dtype=torch.int64, device=drafter.device)
        # The redraft row is anchored at L itself: its context is the committed
        # taps below L and no staged C1 lanes, so its row width is 0.
        redraft_width = torch.zeros(1, dtype=torch.int64, device=drafter.device)
        redraft_active = torch.ones(1, dtype=torch.bool, device=drafter.device)
        from jetspec.engines.jetspec_plus_plus.draft.eval_profile import enabled as eval_enabled
        eval_j = int(fixed_config[1]) if eval_enabled() else 0
        eval_r = int(fixed_config[-1]) if eval_enabled() else 0
        redraft_swap = (RefreshSwap(decoder, front, tap_store, rows=1, lanes=key.lanes)
                        if eval_enabled() and swap is not None else None)
        redraft_pass = passes_chain_segments(
            buffers["rp"], tokens, redraft_keys, redraft_active,
            front.staged_hidden, redraft_bucket, J=eval_j,
            refresh_at=tuple(range(1, eval_r + 1)),
            T_ref=((key.lanes - 1 + 15) // 16) * 16 if eval_r else 0,
            refresh_swap=redraft_swap,
            **({'exact_prefix': True} if eval_enabled() else {}))
        redraft_out = redraft_bucket.chain_outputs
        if eval_enabled():
            buffers['eval_redraft_audit'] = redraft_out['eval_audit']
            if 'eval_refresh_effects' in redraft_out:
                buffers['eval_redraft_refresh_effects'] = redraft_out['eval_refresh_effects']
            if 'final_draft_top16_ids' in redraft_out:
                buffers['redraft_draft_top16_ids'] = redraft_out['final_draft_top16_ids']
                buffers['redraft_draft_top16_probabilities'] = redraft_out['final_draft_top16_probabilities']
                buffers['redraft_draft_token_probabilities'] = redraft_out['final_draft_token_probabilities']
                buffers['redraft_draft_pinned'] = redraft_out['final_draft_pinned']
        redraft_publisher = ChainPublisher(
            device=drafter.device, hypotheses=hypotheses,
            row_cap=wire_row_cap, rows=1, lanes=key.lanes)
        buffers.update(redraft_bucket=redraft_bucket,
                       redraft_wire=redraft_publisher.wire,
                       redraft_home=redraft_publisher.home,
                       redraft_packet=torch.empty_like(buffers["packet"]),
                       redraft_published=torch.empty((), dtype=torch.bool,
                                                       device=drafter.device),
                       redraft_rows=torch.empty_like(row_ring.rows[0]),
                       redraft_keys=torch.empty_like(row_ring.keys[0]),
                       redraft_scores=torch.empty_like(row_ring.p_top1[0]),
                       redraft_passes=torch.empty_like(row_ring.passes_run[0]),
                       redraft_err=torch.empty(1, dtype=torch.int64,
                                               device=drafter.device),
                       redraft_err_host=torch.empty(1, dtype=torch.int64,
                                                    pin_memory=drafter.device.type == "cuda"))

    context_stream = None
    if (os.environ.get("JETSPEC_DRAFT_STREAMS", "0") == "1"
            and front.staged_hidden.dtype == torch.bfloat16):
        if not front.staged_hidden.is_cuda:
            raise ValueError("JETSPEC_DRAFT_STREAMS requires CUDA BF16")
        from jetspec.engines.jetspec_plus_plus.draft.chain.draft_streams import DraftStreams
        context_stream = DraftStreams(drafter.device)
        buffers["draft_stream_keys"] = torch.zeros(key.R, dtype=torch.int64,
                                                   device=drafter.device)

    def first_rest(*, incremental=False):
        if context_stream is not None:
            # Context rotary positions depend on L, not the decoder's row keys.
            # Temporary query geometry is overwritten after the fork joins.
            context_stream.start(lambda: bucket.head.update_chain(
                tap_store.prefix_padded(prefix_bucket), front.staged_hidden,
                buffers["rp"][Offset.L], buffers["draft_stream_keys"], mark=True,
                **({"incremental": True} if incremental else {})))
        if "capture_dec_window" in buffers:
            hidden = buffers["capture_dec_window"].shape[-1]
            buffers["capture_dec_window"].copy_(torch.cat(
                (front.decoder_stale, front.staged_hidden[0, :, -hidden:]), dim=0))
            buffers["capture_dec_lengths"][0].copy_(buffers["rp"][Offset.DEC_LEN])
            buffers["capture_dec_lengths"][1].copy_(buffers["rp"][Offset.C1_COUNT])
        logits, ids, p64, pack = decoder.decoder_chain_body(
            buffers["rp"], tokens, front, bucket)
        for name, value in (("dec_logits", logits), ("offers_ids", ids),
                            ("offers_p64", p64), ("offers_pack", pack)):
            buffers[name].copy_(value)
        rows_chain_body(
            buffers["rp"], tokens, buffers["offers_ids"],
            buffers["offers_p64"], buffers["offers_pack"],
            buffers["dec_logits"], buffers["prior_p_top1"], key.R,
            thresholds=thresholds, dataset=dataset, tau_table=tau_table,
            temperature=temperature, out=buffers)
        if context_stream is None:
            bucket.head.update_chain(tap_store.prefix_padded(prefix_bucket),
                                     front.staged_hidden, buffers["rp"][Offset.L],
                                     buffers["keys"][:, 0] + 1, mark=True,
                                     **({"incremental": True} if incremental else {}))
        else:
            context_stream.finish()
            bucket.head._prepare_geometry(buffers["rp"][Offset.L].reshape(1),
                                           buffers["keys"][:, 0] + 1)
        passes["A"]()

    def first():
        first_finish()
        first_rest()

    def first_miss():
        rp = buffers["rp"]
        redraft_keys[:, 0].copy_(rp[Offset.TRUE_K])
        redraft_keys[:, 1].copy_(rp[Offset.CORRECTION])
        if eval_enabled():
            redraft_keys[:, 0].zero_()  # exact-prefix draft has no predicted context tail
        if attn3:
            first_finish()  # consumes A0's prepared C1 state
        redraft_bucket.head.update_chain(
            tap_store.prefix_padded(prefix_bucket), front.staged_hidden,
            rp[Offset.L], redraft_width, mark=True)
        redraft_pass["A"]()
        if eval_j:
            redraft_pass['B1r' if eval_r else 'B1n']()
            if eval_j >= 2:
                redraft_pass['B2']()
            if eval_j == 3:
                redraft_pass['C']()
        if eval_enabled():
            redraft_keys[:, 0].copy_(rp[Offset.TRUE_K])
        result = redraft_out["final_tokens"]
        score = redraft_out["final_p_top1"]
        buffers["redraft_rows"].fill_(-1)
        buffers["redraft_keys"].fill_(-1)
        buffers["redraft_scores"].zero_()
        buffers["redraft_passes"].zero_()
        buffers["redraft_rows"][0].copy_(result[0])
        buffers["redraft_keys"][0].copy_(redraft_keys[0])
        buffers["redraft_scores"][0].copy_(score[0])
        if eval_j:
            buffers["redraft_passes"][0].copy_(redraft_out['passes_run'][0])
        else:
            buffers["redraft_passes"][0].fill_(1)
        row_ring.publish(buffers["packet"][OutcomeOffset.ROUND],
                         buffers["redraft_rows"], buffers["redraft_keys"],
                         buffers["redraft_scores"], buffers["redraft_passes"])
        tokens.pending_draft_ids.copy_(result[0, 1:])
        buffers["prior_p_top1"].copy_(score[0])
        if 'redraft_draft_top16_ids' in buffers:
            staged_ids = torch.full_like(row_ring.draft_top16_ids[0], -1)
            staged_probabilities = torch.zeros_like(row_ring.draft_top16_probabilities[0])
            staged_ids[0].copy_(buffers['redraft_draft_top16_ids'][0])
            staged_probabilities[0].copy_(buffers['redraft_draft_top16_probabilities'][0])
            staged_token_probabilities = torch.zeros_like(row_ring.draft_token_probabilities[0])
            staged_pinned = torch.zeros_like(row_ring.draft_pinned[0])
            staged_token_probabilities[0].copy_(buffers['redraft_draft_token_probabilities'][0])
            staged_pinned[0].copy_(buffers['redraft_draft_pinned'][0])
            row_ring.publish_candidates(buffers['packet'][OutcomeOffset.ROUND], staged_ids, staged_probabilities,
                                        staged_token_probabilities, staged_pinned)
            buffers['prior_draft_top16_ids'].copy_(staged_ids[0])
            buffers['prior_draft_top16_probabilities'].copy_(staged_probabilities[0])
            buffers['prior_draft_token_ids'].copy_(result[0])
            buffers['prior_draft_token_probabilities'].copy_(staged_token_probabilities[0])
            buffers['prior_draft_pinned'].copy_(staged_pinned[0])
        rp[Offset.AFTER_MISS].zero_()
        rp[Offset.TAU_IDX].fill_(eval_j)
        buffers["redraft_packet"].copy_(buffers["packet"])
        buffers["redraft_packet"][OutcomeOffset.ROUND].sub_(1)
        redraft_publisher.body(buffers["redraft_packet"], result,
                               redraft_keys, score,
                               buffers["redraft_published"])
        torch.bitwise_or(buffers["intake_err"], redraft_out["err"],
                         out=buffers["redraft_err"])
        buffers["redraft_err_host"].copy_(buffers["redraft_err"], non_blocking=True)

    def publish():
        publisher.body(buffers["packet"], buffers["final_tokens"],
                       buffers["keys"], buffers["final_p_top1"],
                       buffers["published"])

    if attn3:
        bodies = {**passes, "A0": first_prepare,
                  "A1": lambda: (first_finish(), first_rest()), "P": publish}
        del bodies["A"]
        if context_append_enabled():
            bodies["A1_inc"] = lambda: (first_finish(), first_rest(incremental=True))
    else:
        bodies = {**passes, "A": lambda: (first_finish(), first_rest()), "P": publish}
        if context_append_enabled():
            bodies["A_inc"] = lambda: (first_finish(), first_rest(incremental=True))
    if redraft:
        if attn3:
            bodies.update(A1_miss=first_miss, A1_hyp=first_rest)
        else:
            # C1 runs after the redraft row is the pending row, so the
            # hypotheses see its drafts rather than the missed lookup's zeros.
            bodies.update(A0=lambda: buffers["redraft_packet"].copy_(buffers["packet"]),
                          A1_miss=first_miss, A1_hyp=first)
    return buffers, bodies


def retire_chain_runtime(holder):
    """Release the previous sequence while retaining its captured graph pools."""
    import gc

    previous = holder._chain_runtime
    if previous is None:
        return
    if not previous.receiver.can_retire(previous.next_round - 1):
        raise RuntimeError("previous chain sequence still has an outstanding envelope")
    holder._chain_runtime = None
    holder._chain_last_front = None
    holder._chain_last_key = None
    del previous
    if (os.environ.get("JETSPEC_CHAIN_ADMIT_GC", "1") == "0"
            and getattr(holder, "_chain_graph", None) is not None):
        # Capture-once keeps the graph set (and its pools) across requests; the retired
        # runtime holds only small host objects. A full collection here costs about
        # 160 ms per admission and frees nothing the next request would reuse.
        return
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
