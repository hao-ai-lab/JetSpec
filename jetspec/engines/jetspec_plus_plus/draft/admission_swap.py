"""Eval admission swap staging shared with the segmented chain."""
from __future__ import annotations

from types import SimpleNamespace
import torch
from jetspec.engines.jetspec_plus_plus.draft.chain.refresh_swap import RefreshSwap


def prepare_eager_swap(decoder, sequence, bucket, refresh_at, run_config=None):
    """Reuse a bucket's decoder read buffers; stage its current prefix once per round."""
    length = sequence.taps.shape[0]
    prefix = decoder._prefix(sequence, max(0, length - 4))
    swap = getattr(bucket, 'eager_swap', None)
    if swap is None:
        kv = {}
        # Dense decoder keys are [N,H,P,D]; the hybrid decoder binds its
        # committed target state separately and does not consume this mapping.
        if not hasattr(decoder, 'bind_front_state'):
            for layer, pair in prefix.kv.items():
                kv[layer] = tuple(value.new_zeros((*value.shape[:2], bucket.head.P,
                                                  value.shape[-1])) for value in pair)
        front = SimpleNamespace(decoder_prefix_taps=sequence.taps[-4:, -1],
                                decoder_prefix_kv=kv, decoder_bucket=bucket.head.P,
                                staged_hidden=sequence.staged.predicted_hidden)
        swap = RefreshSwap(decoder, front, SimpleNamespace(storage=sequence.taps),
                           rows=bucket.ids.shape[0], lanes=bucket.ids.shape[1], run_config=run_config)
        swap.configure_pins(refresh_at, rows=bucket.ids.shape[0], lanes=bucket.ids.shape[1])
        bucket.eager_swap = swap
        swap.length = torch.zeros((), dtype=torch.long, device=bucket.ids.device)
        swap.length_host = torch.empty((), dtype=torch.long, device='cpu',
                                      pin_memory=bucket.ids.is_cuda)
    swap.length_host.fill_(length)
    swap.length.copy_(swap.length_host, non_blocking=bucket.ids.is_cuda)
    for layer, pair in swap.front.decoder_prefix_kv.items():
        for target, source in zip(pair, prefix.kv[layer]):
            target.zero_()
            target[..., :prefix.length, :].copy_(source[..., :prefix.length, :])
    if hasattr(decoder, 'bind_front_state'):
        state = swap.window['prefix']
        state['recurrent'].copy_(sequence.hybrid_recurrent)
        state['conv'].copy_(sequence.hybrid_conv)
        for name in ('key', 'value'):
            source = getattr(sequence, 'hybrid_prefix_' + name)
            state[name].zero_()
            state[name][:length].copy_(source[:length])
    swap.front.staged_hidden = sequence.staged.predicted_hidden
    swap.tap_store.storage = sequence.taps
    swap.reset()
    swap.fill_mask_token = int(bucket.drafter.head.mask_token_id)
    if swap.fill_mode == 'noise':
        swap.prepare_noise(seq_id=getattr(sequence, '_jetspec_seq_id', sequence.prompt_index),
                           candidate_round=getattr(sequence, '_jetspec_candidate_round', 0),
                           vocab=decoder.lm_head.weight.shape[0], mask_token=swap.fill_mask_token)
    return swap

