from __future__ import annotations

import torch
import torch.distributed as dist

from jetspec.engines.jetspec_plus_plus.serving.sequence import Sequence

def prepare_prefill_payload(
    input_id_list: list[list[int]],
    eagle_acts: torch.Tensor,
    device: torch.device,
    max_blocks: int,
    draft_block_tables: list[list[int]] | torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    input_ids_flat = []
    num_tokens = []
    for input_ids in input_id_list:
        input_ids_flat.extend(input_ids)
        num_tokens.append(len(input_ids))
    input_ids_flat = torch.tensor(input_ids_flat, dtype=torch.int64, device=device)
    num_tokens = torch.tensor(num_tokens, dtype=torch.int64, device=device)
    if isinstance(draft_block_tables, list):
        draft_block_table = torch.tensor(
            [dbt + [-1] * (max_blocks - len(dbt)) for dbt in draft_block_tables],
            dtype=torch.int32, device=device,
        )
    else:
        assert draft_block_tables.shape == (len(input_id_list), max_blocks), (
            f"draft_block_tables shape mismatch: expected ({len(input_id_list), max_blocks}), got {draft_block_tables.shape}"
        )
        draft_block_table = draft_block_tables

    # 3) send cmd=1
    cmd = torch.tensor([1], dtype=torch.int64, device=device)

    # 4) send metadata for tensor reconstruction
    metadata = torch.tensor([
        input_ids_flat.size(0),
        len(input_id_list),  # batch_size
        max_blocks,
        1 if eagle_acts is not None else 0,
        eagle_acts.shape[1] if eagle_acts is not None else 0,
    ], dtype=torch.int64, device=device)

    if eagle_acts is not None:
        assert eagle_acts.shape[0] == input_ids_flat.shape[0], (
            f"Eagle activations length {eagle_acts.shape[0]} != input_ids_flat length {input_ids_flat.shape[0]}"
        )

    return cmd, metadata, input_ids_flat, num_tokens, draft_block_table, eagle_acts

def prepare_decode_tensors_from_seqs(
    seqs: list[Sequence],
    block_size: int,
    is_draft: bool,
    verify: bool = False,
    k: int = -1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    input_ids = []
    positions = []
    slot_mapping = []
    context_lens = []

    if not verify:  # normal decoding or draft fwd in speculation
        assert k == -1, "k should be -1 for normal decoding or draft fwd in speculation"
        for seq in seqs:
            block_table = seq.draft_block_table if is_draft else seq.block_table
            assert len(seq) // block_size <= len(block_table), "in sync spec draft decode, not enough blocks allocated"
            num_cached_tokens = seq.num_cached_tokens if not is_draft else seq.num_draft_cached_tokens
            assert num_cached_tokens == len(seq) - 1, "num_cached_tokens should be equal to len(seq) - 1 in pure sq decode path"
            input_ids.append(seq.last_token)
            positions.append(len(seq) - 1)
            context_lens.append(len(seq))

            pos = seq.num_tokens - 1
            block_idx = pos // block_size
            pos_in_block = pos % block_size
            slot_mapping.append(block_table[block_idx] * block_size + pos_in_block)
    else:  # verify and glue decode prep both go here
        assert not is_draft, "verify path only supported for target model" # we prep tensors to send to draft for glue on the target 
        assert k > 0, "k should be > 0 for target fwd in verify"

        for seq_idx, seq in enumerate(seqs):
            # can hardcode block_table here for target since this is only target codepath 
            assert (seq.num_tokens - 1) // block_size <= len(seq.block_table), "in sync spec target verify, not enough blocks allocated"
            
            pos0 = seq.num_tokens - (k+1)
            input_ids.extend(seq[pos0:])
            positions.extend(list(range(pos0, pos0 + k + 1)))
            assert seq.num_cached_tokens == pos0, f"num_cached_tokens={seq.num_cached_tokens} != pos0={pos0} (num_tokens={seq.num_tokens}, k={k})"
            context_lens.append(len(seq))  

            for j in range(k + 1):
                pos = pos0 + j
                block_idx = pos // block_size
                block_id = seq.block_table[block_idx]
                pos_in_block = pos % block_size
                slot_mapping.append(
                    block_id * block_size + pos_in_block)

    input_ids = torch.tensor(
        input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
    positions = torch.tensor(
        positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
    slot_mapping = torch.tensor(
        slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
    context_lens = torch.tensor(
        context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)

    return input_ids, positions, slot_mapping, context_lens

def prepare_block_tables_from_seqs(
    seqs: list[Sequence],
    is_draft: bool = False
) -> torch.Tensor:
        if is_draft:
            max_len = max(len(seq.draft_block_table) for seq in seqs)
            block_tables = [seq.draft_block_table + [-1] * (max_len - len(seq.draft_block_table)) for seq in seqs]
        else:
            max_len = max(len(seq.block_table) for seq in seqs)
            block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

def prepare_prefill_tensors_from_seqs(
    seqs: list[Sequence],
    block_size: int,
    is_draft: bool = False,
    skip_first_token: int = 0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    assert skip_first_token in (0, 1)
    input_ids = []
    positions = []
    cu_seqlens_q = [0]
    cu_seqlens_k = [0]
    max_seqlen_q = 0
    max_seqlen_k = 0
    slot_mapping = []
    
    for seq in seqs:
        seqlen = len(seq)
        if is_draft:
            num_cached_tokens = seq.num_draft_cached_tokens
            block_table = seq.draft_block_table
        else:
            num_cached_tokens = seq.num_cached_tokens
            block_table = seq.block_table

        start = num_cached_tokens + (skip_first_token if is_draft else 0)
        input_ids.extend(seq[start:])
        pos_offset = -skip_first_token if is_draft else 0
        positions.extend(list(range(start + pos_offset, seqlen + pos_offset)))
        seqlen_q = seqlen - start
        seqlen_k = seqlen + pos_offset
        cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
        cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
        max_seqlen_q = max(seqlen_q, max_seqlen_q)
        max_seqlen_k = max(seqlen_k, max_seqlen_k)

        if not block_table:  # first prefill
            continue

        # new: emit exactly one slot for each *new* token
        #    map each token index -> (block_id * block_size + offset)
        for pos in range(start + pos_offset, seq.num_tokens + pos_offset):
            block_i = pos // block_size
            offset = pos % block_size
            slot = block_table[block_i] * block_size + offset
            slot_mapping.append(slot)

    input_ids = torch.tensor(
        input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
    positions = torch.tensor(
        positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
    cu_seqlens_q = torch.tensor(
        cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
    cu_seqlens_k = torch.tensor(
        cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
    slot_mapping = torch.tensor(
        slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
    
    return input_ids, positions, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping
