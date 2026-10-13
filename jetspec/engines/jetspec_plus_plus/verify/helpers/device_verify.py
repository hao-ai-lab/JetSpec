"""Fixed-width N=1 JetSpec verify inputs built from device proposal tokens."""

import numpy as np
import torch




def stage_single_verify_inputs(buffers, proposals, cached_tokens, block_table, block_size, *, input_staged=False):
    """Build one verify row directly in the replay buffers."""
    width = proposals.shape[0]
    if not input_staged:
        buffers["input_ids"][:width].copy_(proposals)
    positions = buffers["positions"][:width]
    offsets = buffers.get("position_offsets")
    if offsets is None:
        offsets = torch.arange(width, dtype=torch.int64, device=proposals.device)
    positions.copy_(offsets[:width]).add_(cached_tokens)
    pages = positions.div(block_size, rounding_mode="floor")
    slots = block_table[pages].to(torch.int32)
    slots.mul_(block_size).add_(positions.remainder(block_size).to(torch.int32))
    buffers["slot_mapping"][:width].copy_(slots)
    # fill_ launches a kernel; a Python-scalar __setitem__ is a blocking pageable H2D copy.
    buffers["cu_seqlens_q"][:1].fill_(0)
    buffers["cu_seqlens_q"][1:2].fill_(width)
    buffers["context_lens"][:1].fill_(cached_tokens + width)


def stage_single_verify_inputs_host(buffers, proposal_host, cached_tokens,
                                    block_table, block_size, storage, *, query_width=None,
                                    input_staged=False):
    """Stage host-known N=1 graph inputs through one reusable pinned allocation."""
    width = proposal_host.shape[0] if proposal_host is not None else query_width
    positions = storage[:width].numpy()
    words = storage[width:].view(torch.int32).numpy()
    slots = words[:width]
    for index in range(width):
        position = cached_tokens + index
        positions[index] = position
        slots[index] = block_table[position // block_size] * block_size + position % block_size
    words[width:width + 2] = (0, width)
    words[width + 2] = cached_tokens + width
    async_copy = buffers["input_ids"].is_cuda
    if proposal_host is not None and not input_staged:
        buffers["input_ids"][:width].copy_(proposal_host, non_blocking=async_copy)
    buffers["positions"][:width].copy_(storage[:width], non_blocking=async_copy)
    buffers["slot_mapping"][:width].copy_(storage[width:].view(torch.int32)[:width],
                                           non_blocking=async_copy)
    buffers["cu_seqlens_q"][:2].copy_(storage[width:].view(torch.int32)[width:width + 2],
                                      non_blocking=async_copy)
    buffers["context_lens"][:1].copy_(storage[width:].view(torch.int32)[width + 2:width + 3],
                                      non_blocking=async_copy)



def verify_stage_blob_views(base, max_width):
    """Typed views of one aligned block used by N=1 graph inputs and the wrapper."""
    p = max_width * 8
    q = p + max_width * 4
    return dict(
        positions=base[:p].view(torch.int64),
        slot_mapping=base[p:q].view(torch.int32),
        cu_seqlens_q=base[q:q + 8].view(torch.int32),
        context_lens=base[q + 8:q + 12].view(torch.int32),
        last_page_len=base[q + 12:q + 16].view(torch.int32),
        kv_lens=base[q + 16:q + 20].view(torch.int32),
    )


def stage_single_verify_inputs_blob(buffers, proposal_host, cached_tokens,
                                    block_table, block_size, host_blob, *,
                                    query_width=None, input_staged=False):
    """Vectorize host preparation, then upload all N=1 metadata in one copy."""
    width = proposal_host.shape[0] if proposal_host is not None else query_width
    host = host_blob["views"]
    positions = host["positions"].numpy()
    positions[:width] = cached_tokens + np.arange(width, dtype=np.int64)
    page_ids = np.asarray(block_table, dtype=np.int64)
    slots = host["slot_mapping"].numpy()
    slots[:width] = (page_ids[positions[:width] // block_size] * block_size
                     + positions[:width] % block_size)
    host["cu_seqlens_q"].numpy()[:] = (0, width)
    host["context_lens"].numpy()[0] = cached_tokens + width
    host["last_page_len"].numpy()[0] = (cached_tokens + width - 1) % block_size + 1
    if host["kv_lens"].numel():
        host["kv_lens"].numpy()[0] = cached_tokens + width
    if proposal_host is not None and not input_staged:
        buffers["input_ids"][:width].copy_(proposal_host,
                                           non_blocking=buffers["input_ids"].is_cuda)
    host_blob["device"].copy_(host_blob["base"],
                              non_blocking=host_blob["device"].is_cuda)


def stage_single_verify_inputs_blob_fast(buffers, proposal_host, cached_tokens,
                                         block_table, block_size, host_blob, *,
                                         query_width=None, input_staged=False,
                                         check=False):
    """Fill the cached blob views from only the pages touched by this query."""
    width = proposal_host.shape[0] if proposal_host is not None else query_width
    host = host_blob["np"]
    positions = host["positions"]
    positions[:width] = cached_tokens + host_blob["offsets"][:width]
    page_numbers = positions[:width] // block_size
    first_page = int(page_numbers[0])
    page_ids = np.asarray(block_table[first_page:int(page_numbers[-1]) + 1],
                          dtype=np.int64)
    host["slot_mapping"][:width] = (page_ids[page_numbers - first_page] * block_size
                                     + positions[:width] % block_size)
    if host_blob.get("last_width") != width:
        host["cu_seqlens_q"][:] = (0, width)
        host_blob["last_width"] = width
    kv_len = cached_tokens + width
    host["context_lens"][0] = kv_len
    host["last_page_len"][0] = (kv_len - 1) % block_size + 1
    if host["kv_lens"].size:
        host["kv_lens"][0] = kv_len
    mismatches = 0
    if check:
        scratch = host_blob["check"]
        scratch["base"].copy_(host_blob["base"])
        stage_single_verify_inputs_blob(
            buffers, None, cached_tokens, block_table, block_size, scratch,
            query_width=width, input_staged=True)
        mismatches = int(not torch.equal(host_blob["base"], scratch["base"]))
    if proposal_host is not None and not input_staged:
        buffers["input_ids"][:width].copy_(proposal_host,
                                           non_blocking=buffers["input_ids"].is_cuda)
    host_blob["device"].copy_(host_blob["base"],
                              non_blocking=host_blob["device"].is_cuda)
    return mismatches


def stage_single_verify_pages(sequence, replay_table, pages):
    """Retain N=1 page IDs on device; upload only newly allocated or changed pages."""
    if pages <= 0 or pages > replay_table.numel() or len(sequence.block_table) < pages:
        raise ValueError("device mixed verify lacks a target KV page")
    storage = getattr(sequence, "_jetspec_verify_page_storage", None)
    if storage is None:
        storage = torch.empty_like(replay_table)
        sequence._jetspec_verify_page_storage = storage
        sequence._jetspec_verify_page_host = torch.empty(
            replay_table.numel(), dtype=torch.int32, device="cpu",
            pin_memory=replay_table.is_cuda)
        sequence._jetspec_verify_page_ids = ()
    previous = sequence._jetspec_verify_page_ids
    current = tuple(sequence.block_table[:pages])
    changed = next((i for i, (old, new) in enumerate(zip(previous, current))
                    if old != new), min(len(previous), pages))
    if changed < pages:
        host = sequence._jetspec_verify_page_host
        for index in range(changed, pages):
            host[index] = current[index]
        storage[changed:pages].copy_(host[changed:pages],
                                     non_blocking=replay_table.is_cuda)
    sequence._jetspec_verify_page_ids = current
    replay_table[:pages].copy_(storage[:pages])


def stage_committed_ids(sequence, proposals, capacity):
    """Keep the verify rank's committed prefix on device across N=1 rounds."""
    start = sequence.num_cached_tokens
    width = proposals.shape[0]
    # The last accepted proposal is tap-bearing; its recovery occupies one
    # further slot even when every proposal is accepted.
    if start + width + 1 > capacity:
        raise ValueError("committed ID capacity is too small for the recovery")
    storage = getattr(sequence, "_jetspec_token_storage", None)
    if storage is None:
        storage = torch.empty(capacity, dtype=torch.int64, device=proposals.device)
        prefix = torch.tensor(sequence.token_ids[:start], dtype=torch.int64,
                              pin_memory=proposals.is_cuda)
        storage[:start].copy_(prefix, non_blocking=proposals.is_cuda)
        sequence._jetspec_token_storage = storage
    elif getattr(sequence, "_jetspec_device_valid_length", None) != start:
        raise RuntimeError("verify device token length differs from host metadata")
    storage[start:start + width].copy_(proposals)
    sequence._jetspec_pending_tokens = (start, width)


def finish_committed_ids(sequence, committed_count):
    start, width = sequence._jetspec_pending_tokens
    if not 0 <= committed_count <= width:
        raise ValueError("committed ID count outside the staged verify block")
    sequence._jetspec_device_valid_length = start + committed_count
    del sequence._jetspec_pending_tokens


def stage_recovery_id(sequence, committed_length, recovery):
    """Extend the verifier's token mirror through the next recovery token.

    The target KV cache remains valid only through ``committed_length``; the
    next proposal overwrites this same slot with its anchor token.
    """
    storage = sequence._jetspec_token_storage
    # stage_committed_ids already checked this host-known upper bound. Avoid
    # testing the device count here, which would synchronize the verify stream.
    bound = (sum(sequence._jetspec_pending_tokens)
             if isinstance(committed_length, torch.Tensor) else committed_length)
    if bound >= storage.numel():
        raise ValueError("committed ID capacity has no recovery slot")
    if isinstance(committed_length, torch.Tensor):
        # Indexing with a tensor reads it back to the host (aten._local_scalar_dense), which stalls
        # the host until the verify finishes; write through a device index instead.
        storage.index_copy_(0, committed_length.reshape(1).to(torch.long),
                            recovery.reshape(1).to(storage.dtype))
    else:
        storage[committed_length].copy_(recovery)
