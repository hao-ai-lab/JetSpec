import torch

from jetflow.inference_engine.engine import _LogicalRoundBuffers


def _ptrs(round_bufs, staged):
    seq_step_b, pos_b, qq_bias_b, dummy, cu = staged
    return {
        "slots": round_bufs.slots.data_ptr(),
        "starts": round_bufs.starts.data_ptr(),
        "lens": round_bufs.lens.data_ptr(),
        "seq_lens_k": round_bufs.seq_lens_k.data_ptr(),
        "seq_step": seq_step_b.data_ptr(),
        "pos": pos_b.data_ptr(),
        "qq_bias": qq_bias_b.data_ptr(),
        "dummy": dummy.data_ptr(),
        "cu": cu.data_ptr(),
        "offsets": round_bufs.node_offsets(4).data_ptr(),
    }


def test_logical_round_buffers_reuse_bucket_storage_and_stage_values():
    round_bufs = _LogicalRoundBuffers(
        max_slots=16,
        prompt_len=5,
        nlayers=3,
        hidden_size=4,
        block_size=4,
        device="cpu",
        dtype=torch.float32,
    )

    seq_step = torch.tensor([[11, 12, 13]], dtype=torch.long)
    depth = torch.tensor([0, 2, 3], dtype=torch.long)
    ancestor = torch.tensor([
        [True, False, False],
        [True, True, False],
        [False, True, True],
    ])
    staged = round_bufs.stage_tree_inputs(seq_step, depth, ancestor, past_len=7, N=3, B=4)
    round_bufs.stage_slots(wlen=2, node_blks=torch.tensor([20, 20, 20, 20]), B=4)
    slk = round_bufs.fill_lengths(wlen=2, past_len=7, B=4)
    before = _ptrs(round_bufs, staged)

    seq_step_b, pos_b, qq_bias_b, dummy, cu = staged
    assert seq_step_b.tolist() == [[11, 12, 13, 0]]
    assert pos_b.tolist() == [[7, 9, 10, 10]]
    assert cu.tolist() == [0, 4]
    assert slk.tolist() == [11]
    assert round_bufs.starts.tolist() == [5]
    assert round_bufs.lens.tolist() == [6]
    assert torch.equal(dummy, torch.zeros_like(dummy))
    assert round_bufs.slots[0, 2:6].tolist() == [80, 81, 82, 83]
    assert qq_bias_b[0, 0].item() == 0.0
    assert qq_bias_b[1, 0].item() == 0.0
    assert qq_bias_b[1, 1].item() == 0.0
    assert qq_bias_b[0, 1].item() == float("-inf")
    assert qq_bias_b[3, 0].item() == float("-inf")
    assert qq_bias_b[0, 3].item() == float("-inf")
    rows, starts, lens = round_bufs.logical_bind()
    assert starts is round_bufs.starts
    assert lens is round_bufs.lens
    assert len(rows) == 3
    assert all(row.data_ptr() == round_bufs.slots.data_ptr() for row in rows)

    seq_step2 = torch.tensor([[21, 22]], dtype=torch.long)
    depth2 = torch.tensor([1, 4], dtype=torch.long)
    ancestor2 = torch.eye(2, dtype=torch.bool)
    staged2 = round_bufs.stage_tree_inputs(seq_step2, depth2, ancestor2, past_len=9, N=2, B=4)
    round_bufs.stage_slots(wlen=4, node_blks=torch.tensor([30, 30, 30, 30]), B=4)
    round_bufs.fill_lengths(wlen=4, past_len=9, B=4)
    after = _ptrs(round_bufs, staged2)

    assert after == before
    assert staged2[0].tolist() == [[21, 22, 0, 0]]
    assert staged2[1].tolist() == [[10, 13, 13, 13]]
    assert round_bufs.slots[0, 4:8].tolist() == [120, 121, 122, 123]
    assert round_bufs.lens.tolist() == [8]
    assert round_bufs.seq_lens_k.tolist() == [13]
    assert staged2[2][0, 0].item() == 0.0
    assert staged2[2][0, 1].item() == float("-inf")
    assert staged2[2][2, 2].item() == float("-inf")
