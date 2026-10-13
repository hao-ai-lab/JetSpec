import numpy as np


def verify_plan_signature(context_lengths, block_tables, bucket, query_len):
    """Exact planner inputs, including native repeated ghost rows."""
    if context_lengths is None or block_tables is None:
        return None
    if not context_lengths or len(context_lengths) != len(block_tables) or bucket < len(context_lengths):
        raise ValueError("invalid verify planner metadata")
    pad = bucket - len(context_lengths)
    return (query_len, context_lengths + (context_lengths[-1],) * pad,
            block_tables + (block_tables[-1],) * pad)


def staged_verify_rows(seqs, block_size, query_len, bucket):
    """Candidate-independent verify metadata, including repeated ghost rows."""
    assert 1 < len(seqs) <= bucket
    positions, slots, lengths, tables = [], [], [], []
    table_width = max(len(seq.block_table) for seq in seqs)
    for seq in seqs:
        start = len(seq) - 1
        length = start + query_len
        assert seq.num_cached_tokens == start
        assert len(seq.block_table) >= (length + block_size - 1) // block_size
        for pos in range(start, length):
            positions.append(pos)
            slots.append(seq.block_table[pos // block_size] * block_size + pos % block_size)
        lengths.append(length)
        tables.append(seq.block_table + [-1] * (table_width - len(seq.block_table)))
    for _ in range(bucket - len(seqs)):
        positions.extend([0] * query_len)
        slots.extend([-1] * query_len)
        lengths.append(lengths[-1])
        tables.append(tables[len(seqs) - 1])
    return positions, slots, lengths, tables




def stage_mixed_verify_rows(seqs, block_size, width):
    """Store candidate-independent rows in original batch order."""
    positions, slots, _, tables = vectorized_staged_verify_rows(
        seqs, block_size, width, len(seqs))
    return {
        "keys": tuple((id(seq), seq.num_cached_tokens, tuple(seq.block_table)) for seq in seqs),
        "positions": np.asarray(positions, dtype=np.int64).reshape(len(seqs), width),
        "slots": np.asarray(slots, dtype=np.int32).reshape(len(seqs), width),
        "tables": np.asarray(tables, dtype=np.int32),
    }


def assemble_full_staged_verify_rows(staged, bucket, width):
    """Expand prestaged rows to the legacy full-verify ghost bucket."""
    count = len(staged["keys"])
    positions = np.zeros((bucket, width), dtype=np.int64)
    slots = np.full((bucket, width), -1, dtype=np.int32)
    lengths = np.empty(bucket, dtype=np.int32)
    tables = np.empty((bucket, staged["tables"].shape[1]), dtype=np.int32)
    positions[:count] = staged["positions"]
    slots[:count] = staged["slots"]
    lengths[:count] = [key[1] + width for key in staged["keys"]]
    lengths[count:] = lengths[count - 1]
    tables[:count] = staged["tables"]
    tables[count:] = tables[count - 1]
    return positions.ravel(), slots.ravel(), lengths, tables


def assemble_mixed_verify_rows(staged, seqs, width):
    """Match prepare_prefill's graph inputs after full rows and miss rows are ordered."""
    indices = {key[0]: row for row, key in enumerate(staged["keys"])}
    rows = []
    widths = []
    for seq in seqs:
        row = indices.get(id(seq))
        if row is None:
            return None
        _, start, blocks = staged["keys"][row]
        query_width = len(seq) - seq.num_cached_tokens
        if (seq.num_cached_tokens != start or tuple(seq.block_table) != blocks
                or query_width not in (1, width)):
            return None
        rows.append(row)
        widths.append(query_width)
    # The mixed graph is captured for full queries first, then one-token rows.
    if widths != sorted(widths, reverse=True):
        return None
    ids = np.fromiter((token for seq, count in zip(seqs, widths)
                       for token in seq.token_ids[-count:]), dtype=np.int64)
    positions = np.concatenate([staged["positions"][row, :count]
                                for row, count in zip(rows, widths)])
    slots = np.concatenate([staged["slots"][row, :count]
                            for row, count in zip(rows, widths)])
    lengths = np.fromiter((len(seq) for seq in seqs), dtype=np.int32)
    cu_q = np.empty(len(seqs) + 1, dtype=np.int32)
    cu_k = np.empty(len(seqs) + 1, dtype=np.int32)
    cu_q[0] = cu_k[0] = 0
    np.cumsum(widths, out=cu_q[1:])
    np.cumsum(lengths, out=cu_k[1:])
    return ids, positions, slots, lengths, staged["tables"][rows], cu_q, cu_k
