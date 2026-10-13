"""Reference host-accept publication, preserving the baseline chain wire."""
import os
import torch

from jetspec.engines.jetspec_plus_plus.wire.chain.intake import HASH_WEIGHTS

from jetspec.engines.jetspec_plus_plus.wire.chain.contract import OUTCOME_WORDS; from jetspec.engines.jetspec_plus_plus.wire.chain.contract import OutcomeOffset


def build_reference_outcome(sequence, suffix, recovery, taps, *, start,
                            max_model_len, eos_id, record):
    """Publish host acceptance using the unchanged b16 chain wire contract.

    ``start`` is the cached length before the verify forward increments it.
    All candidate taps travel on the wire, including the rejected lanes.
    """
    from jetspec.engines.jetspec_plus_plus.verify.helpers.outcome_packet import hash16_committed; from jetspec.engines.jetspec_plus_plus.verify.helpers.outcome_packet import make_hash16_weights
    if taps is None or taps.ndim != 3 or taps.shape[:2] != (16, 5):
        raise ValueError("reference verify requires the full 16-lane tap block")
    if not 1 <= len(suffix) <= 16 or len(sequence.token_ids) < start + 16:
        raise ValueError("reference verify requires a full b16 proposal")
    if list(suffix) != sequence.token_ids[start:start + len(suffix)]:
        raise ValueError("reference accepted prefix differs from its proposal")
    row_k, row_a = sequence._jetspec_verified_row_key
    if row_k < 0 or row_a < 0:
        raise ValueError("reference verify requires the verified row key")
    L = start + len(suffix)
    host_outcome = os.environ.get('JETSPEC_REFERENCE_OUTCOME_HOST', '0') == '1'
    if host_outcome:
        committed = sequence.token_ids[max(0, L + 1 - 16):L] + [int(recovery)]
        digest = (sum(value * weight for value, weight in zip(committed, HASH_WEIGHTS[-len(committed):]))
                  + sum(HASH_WEIGHTS) + (L + 1) * 1000003)
        digest = (digest + (1 << 63)) % (1 << 64) - (1 << 63)
    else:
        committed_ids = torch.tensor(sequence.token_ids[:start] + list(suffix) + [int(recovery)],
                                     dtype=torch.int64, device="cpu")
    # The upstream scheduler commits only suffix here. An EOS recovery is
    # still untapped and needs the following full-block reference round;
    # marking this command terminal would suppress that round's redraft.
    eos = eos_id is not None and not sequence.ignore_eos and eos_id in suffix
    limit = (L - sequence.num_prompt_tokens >= sequence.max_new_tokens
             or L >= max_model_len)
    if host_outcome:
        slots = getattr(sequence, '_jetspec_reference_header_slots', None)
        if slots is None:
            # Alternate pinned sources retain the immediate predecessor.
            slots = [torch.zeros(OUTCOME_WORDS, dtype=torch.int64, device='cpu',
                                 pin_memory=taps.is_cuda) for _ in range(2)]
            sequence._jetspec_reference_header_slots = slots
            sequence._jetspec_reference_header_words = [slot.numpy() for slot in slots]
            sequence._jetspec_reference_header_next = 0
        index = sequence._jetspec_reference_header_next
        sequence._jetspec_reference_header_next = 1 - index
        packet = slots[index]
        words = sequence._jetspec_reference_header_words[index]
        words.fill(0)
    else:
        packet = torch.zeros(OUTCOME_WORDS, dtype=torch.int64, device="cpu")
        words = packet
    words[OutcomeOffset.CMD] = 1
    words[OutcomeOffset.SEQ_ID] = sequence.seq_id
    words[OutcomeOffset.ROUND] = sequence.jetspec_round + 1
    words[OutcomeOffset.TRUE_K] = len(suffix) - 1
    words[OutcomeOffset.RECOVERY] = int(recovery)
    words[OutcomeOffset.TAP_COUNT] = len(suffix)
    words[OutcomeOffset.L] = L
    words[OutcomeOffset.HASH16] = (digest if host_outcome else hash16_committed(
        committed_ids, torch.tensor(L + 1, device="cpu"), make_hash16_weights("cpu")))
    words[OutcomeOffset.EOS_FLAG] = int(eos or limit)
    words[OutcomeOffset.K] = row_k
    words[OutcomeOffset.A] = row_a
    outcome = packet.to(taps.device, non_blocking=taps.is_cuda) if host_outcome else packet.to(taps.device)
    if host_outcome:
        sequence._jetspec_reference_packet_host = packet
    sequence._jetspec_device_outcome = outcome
    sequence._jetspec_tap_candidate_block = taps
    sequence._jetspec_device_tap_count = len(suffix)
    sequence._jetspec_last_packet_l = L
    sequence._jetspec_last_packet_eos = int(eos or limit)
    record["next_correction"] = int(recovery)
    return outcome
