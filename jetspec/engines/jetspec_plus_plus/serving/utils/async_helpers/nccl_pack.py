import torch
import torch.distributed as dist


def stage_committed_taps(sequence, candidate_taps: torch.Tensor, capacity: int):
    """Stage a fixed verify block in append-only device storage before acceptance is read."""
    if candidate_taps.ndim != 3:
        raise ValueError("candidate_taps must be [tokens, taps, hidden]")
    previous = sequence.jetspec_taps
    base = 0 if previous is None else previous.shape[0]
    width = candidate_taps.shape[0]
    if base + width > capacity:
        raise ValueError("tap storage capacity is too small for the verify block")
    storage = getattr(sequence, "_jetspec_tap_storage", None)
    if storage is None:
        storage = candidate_taps.new_empty((capacity, *candidate_taps.shape[1:]))
        if previous is not None:
            storage[:base].copy_(previous)
        sequence._jetspec_tap_storage = storage
    storage[base:base + width].copy_(candidate_taps)
    sequence._jetspec_tap_pending = (base, width)
    sequence._jetspec_tap_candidate_block = storage[base:base + width]


def finish_staged_taps(sequence, committed_count: int):
    """Expose the accepted prefix after the round's single outcome materialization."""
    base, width = sequence._jetspec_tap_pending
    if not 0 <= committed_count <= width:
        raise ValueError("committed_count outside the staged verify block")
    sequence.jetspec_taps = sequence._jetspec_tap_storage[:base + committed_count]
    del sequence._jetspec_tap_pending


def append_committed_taps(sequence, candidate_taps: torch.Tensor, committed_count: int):
    """Append only the committed prefix of a target verify's tap candidates."""
    if candidate_taps.ndim != 3:
        raise ValueError(
            f"candidate_taps must be [tokens, taps, hidden], got {tuple(candidate_taps.shape)}"
        )
    if not 0 <= committed_count <= candidate_taps.shape[0]:
        raise ValueError(
            f"committed_count={committed_count} outside [0, {candidate_taps.shape[0]}]"
        )
    if committed_count == 0:
        return

    committed = candidate_taps[:committed_count]
    if sequence.jetspec_taps is None:
        sequence.jetspec_taps = committed.clone()
    else:
        sequence.jetspec_taps = torch.cat(
            [sequence.jetspec_taps, committed], dim=0
        )


def pack_jetspec_taps(
    sequences,
    device: torch.device,
    tap_shape: tuple[int, int],
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack each sequence's unsent tap suffix into one ragged payload."""
    pending = []
    lengths = []
    for sequence in sequences:
        taps = sequence.jetspec_taps
        sent = sequence.jetspec_taps_sent
        suffix = taps[sent:] if taps is not None else None
        length = 0 if suffix is None else suffix.shape[0]
        lengths.append(length)
        if length:
            if tuple(suffix.shape[1:]) != tuple(tap_shape):
                raise ValueError(
                    f"tap suffix must be [tokens, {tap_shape[0]}, {tap_shape[1]}], "
                    f"got {tuple(suffix.shape)}"
                )
            pending.append(suffix.to(device=device, dtype=dtype))

    lengths_tensor = torch.tensor(lengths, dtype=torch.int64, device=device)
    if pending:
        payload = torch.cat(pending, dim=0).contiguous()
    else:
        payload = torch.empty((0, *tap_shape), dtype=dtype, device=device)
    return lengths_tensor, payload




def append_jetspec_tap_mirrors(
    mirrors: dict[int, torch.Tensor],
    seq_ids: torch.Tensor,
    payload: torch.Tensor,
    lengths: torch.Tensor,
):
    """Append one unpacked round to the speculator's per-sequence mirrors."""
    chunks = unpack_jetspec_taps(payload, lengths)
    for seq_id, chunk in zip(seq_ids.tolist(), chunks):
        if chunk.shape[0] == 0:
            continue
        seq_id = int(seq_id)
        if seq_id in mirrors:
            mirrors[seq_id] = torch.cat([mirrors[seq_id], chunk], dim=0)
        else:
            mirrors[seq_id] = chunk.clone()


def concat_int64(*tensors: torch.Tensor) -> torch.Tensor:
    """Concatenate tensors into a single flat int64 payload."""
    parts = []
    for t in tensors:
        if t is None:
            continue
        if t.dtype != torch.int64:
            t = t.to(torch.int64)
        parts.append(t.reshape(-1))
    if not parts:
        return torch.empty(0, dtype=torch.int64)
    return torch.cat(parts, dim=0)


def send_int64(pg, dst: int, *tensors: torch.Tensor):
    """Send many int64-compatible tensors as one fused payload in a fixed order."""
    payload = concat_int64(*tensors)
    if payload.numel() == 0:
        return
    dist.send(payload, dst=dst, group=pg)


def recv_int64(pg, src: int, total_length: int, device: torch.device) -> torch.Tensor:
    """Receive a fused int64 payload of known total length."""
    t = torch.empty((total_length,), dtype=torch.int64, device=device)
    if total_length > 0:
        dist.recv(t, src=src, group=pg)
    return t
