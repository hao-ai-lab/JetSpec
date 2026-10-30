"""Batched draft context slots and compact passes, alongside the C=1 chain."""
from __future__ import annotations

import torch

from jetspec.engines.jetspec_plus_plus.draft.live_rows import BucketPass, KthRowsDrafter, _ContextSlots
from jetspec.engines.jetspec_plus_plus.draft.rows import device_tensor, fused_kernels
from jetspec.engines.jetspec_plus_plus.draft.rows_live import LiveRowsShelf, live_row_buckets, rows_live_quantum
from jetspec.engines.jetspec_plus_plus.runtime.config_values import config_values

__all__ = ['batch_graph_context', 'ctx_project', 'context_kv_projections',
           'live_row_pass', 'live_row_passes']


def live_row_pass(drafter, bucket, rows):
    """Build a compact pass over ``rows`` rows of the bucket's existing head."""
    shelf = bucket.live_shelf
    return BucketPass(shelf.drafter, N=rows, H=1, prefix_bucket=bucket.head.P,
                      backend=drafter.backend, host_fast=False, flat=True, graphs=bucket.graphs,
                      anchor0=bucket.anchor0, wide=bucket.wide, pool=shelf.pool,
                      live_of=bucket.head, live_shelf=shelf,
                      run_config=getattr(drafter, 'run_config', None))


@torch.inference_mode()
def live_row_passes(drafter, bucket, padded, H):
    """Build every compact pass a bucket can use, capturing largest first on CUDA."""
    passes = bucket.__dict__.get('live_passes')
    if passes is not None:
        return passes
    d = drafter.drafter
    shelves = drafter.__dict__.setdefault('_live_rows_shelves', {})
    key = (padded, H, bucket.B)
    if key not in shelves:
        shelves[key] = LiveRowsShelf(d, padded=padded, rows_per_request=H,
                                    block_size=bucket.B, graphs=bucket.graphs)
    bucket.live_shelf = shelves[key]
    for cache in bucket.head.kv:
        cache.zero_()  # omitted rows retain finite, masked K/V
    quantum = rows_live_quantum(padded * H, run_config=getattr(drafter, 'run_config', None))
    passes = {rows: live_row_pass(drafter, bucket, rows)
              for rows in live_row_buckets(padded, H, quantum)}
    bucket.live_passes = passes
    if bucket.graphs and d.device.type == 'cuda':
        for rows in sorted(passes, reverse=True):
            passes[rows](torch.zeros(rows, bucket.B, dtype=torch.long, device=d.device))
    return passes


def context_kv_projections(layers, source):
    """Project the shared context input through every layer's K/V weights."""
    packed = getattr(layers[0], '_ctx_kv_one_gemm', None)
    if packed is None:
        projections = [projection for layer in layers
                       for projection in (layer.self_attn.k_proj, layer.self_attn.v_proj)]
        widths = [projection.out_features for projection in projections]
        weight = torch.cat([projection.weight for projection in projections]).detach()
        biases = [projection.bias for projection in projections]
        bias = torch.cat(biases).detach() if biases[0] is not None else None
        packed = layers[0]._ctx_kv_one_gemm = (weight, bias, widths)
    weight, bias, widths = packed
    output = torch.nn.functional.linear(source, weight, bias)
    if len(set(widths)) == 1:
        outputs = output.view(source.shape[0], len(widths), widths[0]).transpose(0, 1).contiguous().unbind(0)
    else:
        outputs = output.split(widths, dim=-1)
    return list(zip(outputs[::2], outputs[1::2]))


def ctx_project(drafter, contexts, positions, packed=False):
    """Context rows and positions -> per-layer (K, V) [rows, kv_heads, head_dim]."""
    head, fused = drafter.drafter.head, fused_kernels()
    hidden = fused.fast_rms_norm(head.fc(contexts), head.hidden_norm.weight,
                                 head.hidden_norm.variance_epsilon)
    stub = hidden.new_empty(1, 1, head.config.hidden_size)
    cos, sin = head.rotary_emb(stub, positions[None])
    projected = context_kv_projections(head.layers, hidden) if packed else None
    result = []
    for index, layer in enumerate(head.layers):
        attn = layer.self_attn
        nkv, hd = attn.config.num_key_value_heads, attn.head_dim
        key, value = (projected[index] if projected is not None else
                      (attn.k_proj(hidden), attn.v_proj(hidden)))
        k, v = fused.fast_kv_postprocess(key[None], value[None], attn.k_norm.weight,
                                         cos, sin, attn.k_norm.variance_epsilon, nkv, 1, hd)
        result.append((k[0].transpose(0, 1), v[0].transpose(0, 1)))
    return result


@torch.inference_mode()
def batch_graph_context(drafter, bucket, sequences, prefixes, tails, geometry):
    """Fill the bucket's context from reusable per-sequence prefix K/V slots.

    Project each new prefix window and the tails; cold or stale slots refill eagerly.
    Different GEMM row counts mean this is not bitwise to a full bucket update.
    Rows past each prefix length may remain stale and are hidden by the row mask.
    """
    d, head = drafter.drafter, bucket.head
    count, N, P, T = len(sequences), head.N, head.P, head.T
    width, window = head.contexts.shape[-1], 16
    marker = d.head.adapted_position_embedding.weight
    values = config_values(getattr(drafter, 'run_config', None))
    capacity = max(P, N * window, int(values.get('engine.engine_settings.w259_c1_cache_capacity', '768')))
    slots_wanted = max(int(values.get('engine.engine_settings.c1_multi_slots', '32')), N)
    store = drafter.__dict__.get('_ctx_store')
    if store is None or store['capacity'] < capacity or len(store['lengths']) < slots_wanted:
        layer = d.head.layers[0].self_attn
        shape = (slots_wanted + 1, capacity, 2, 1,
                 layer.config.num_key_value_heads, layer.head_dim)
        store = drafter._ctx_store = _ContextSlots(
            capacity=capacity, lengths=[0] * slots_wanted, slots={},
            free=list(reversed(range(slots_wanted))),
            kv=[head.contexts.new_zeros(shape) for _ in d.head.layers])
    capacity = store['capacity']
    trash = len(store['lengths'])
    active = {id(sequence) for sequence in sequences}
    store['active'] = active
    slots = [KthRowsDrafter._ctx_slot(drafter, store, sequence, active) for sequence in sequences]
    lengths = [int(prefix.shape[0]) for prefix in prefixes]
    write = [trash * capacity + j for j in range(N * window)]
    context = head.layout.context_slots.view(N, P + T)
    rows, parts = [], []
    for n, (slot, prefix, length) in enumerate(zip(slots, prefixes, lengths)):
        start = store['lengths'][slot]
        if start > length or length - start > window:
            cold = ((length + 63) // 64) * 64
            projected = ctx_project(
                drafter, torch.nn.functional.pad(prefix, (0, 0, 0, cold - length)) + marker[0],
                torch.arange(cold, device=prefix.device))
            for cache, (k, v) in zip(store['kv'], projected):
                cache[slot, :length, 0, 0].copy_(k[:length])
                cache[slot, :length, 1, 0].copy_(v[:length])
            start = length
        if length > start:
            parts.append(prefix[start:length])
            rows.extend(range(n * window, n * window + length - start))
            for j in range(length - start):
                write[n * window + j] = slot * capacity + start + j
        store['lengths'][slot] = length
    packed_rows = N * window
    full = lengths + [1] * (N - count)
    positions = ([store_index % capacity for store_index in write]
                 + [length + j for length in full for j in range(T)])
    graph = getattr(head, '_ctx_graph', None)
    # A larger prefix bucket can replace the shared store. Each head must then
    # recapture against the new allocation when it is next used.
    if graph is None or graph['store'] is not store:
        buffers = dict(tile=head.contexts.new_zeros(packed_rows, width),
                       tails=head.contexts.new_zeros(N * T, width),
                       ints=torch.zeros(2 * packed_rows + N * T + N,
                                        dtype=torch.long, device=d.device))
        ints = buffers['ints']
        buffers['write'] = ints[:packed_rows]
        buffers['positions'] = ints[packed_rows:2 * packed_rows + N * T]
        buffers['slots'] = ints[2 * packed_rows + N * T:]
        buffers['prefix_slots'] = context[:, :P].reshape(-1)
        buffers['tail_slots'] = context[:, P:].reshape(-1)
        buffers['columns'] = torch.arange(P, device=d.device)
        graph = head._ctx_graph = dict(buffers=buffers, graph=None, store=store)
    buffers = graph['buffers']
    host = torch.tensor(write + positions + slots + [trash] * (N - count), dtype=torch.long)
    cuda = d.device.type == 'cuda'
    buffers['ints'].copy_(host.pin_memory() if cuda else host, non_blocking=cuda)
    if parts:
        buffers['tile'].index_copy_(0, device_tensor(rows, d.device),
                                    torch.cat(parts) if len(parts) > 1 else parts[0])
    buffers['tails'][:count * T].copy_(tails.reshape(-1, width)[:count * T])
    head._update_geometry(device_tensor(full, d.device), geometry)

    def body():
        contexts = torch.cat((buffers['tile'] + marker[0], buffers['tails'] + marker[1]))
        projected = ctx_project(drafter, contexts, buffers['positions'])
        source = (buffers['slots'][:, None] * capacity + buffers['columns'][None]).reshape(-1)
        for cache, kv, (k, v) in zip(head.kv, store['kv'], projected):
            flat = kv.view(-1, *kv.shape[2:])
            flat[buffers['write'], 0, 0] = k[:packed_rows]
            flat[buffers['write'], 1, 0] = v[:packed_rows]
            cache.index_copy_(0, buffers['prefix_slots'], flat.index_select(0, source))
            cache[buffers['tail_slots'], 0, 0] = k[packed_rows:]
            cache[buffers['tail_slots'], 1, 0] = v[packed_rows:]

    if not cuda:
        body()
    else:
        if graph['graph'] is None:
            stream = torch.cuda.Stream(device=d.device)
            stream.wait_stream(torch.cuda.current_stream(d.device))
            with torch.cuda.stream(stream):
                for _ in range(2):
                    body()
            torch.cuda.current_stream(d.device).wait_stream(stream)
            captured = torch.cuda.CUDAGraph()
            with torch.cuda.graph(captured):
                body()
            graph['graph'] = captured
            print(f'BATCH_DRAFT_GRAPH_CAPTURE context bucket={N} prefix={P}', flush=True)
        graph['graph'].replay()
    head.last_ctx_update, head.last_ctx_rows = 'slots', packed_rows + N * T
