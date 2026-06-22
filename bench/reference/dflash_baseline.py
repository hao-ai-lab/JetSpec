"""Linear DFlash block baseline for HF reference benchmarks.

This file intentionally contains no tree decoding. It uses the DFlash draft head
to propose a linear block, verifies that block with the target model, accepts the
longest greedy-agreeing prefix plus one correction, and repeats.
"""
from __future__ import annotations

import time
from types import SimpleNamespace

import torch

from jetflow.core.llm import make_cache
from jetflow.models.draft_head import extract_context_feature


def _cuda_time(device: torch.device | str) -> float:
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter()


@torch.inference_mode()
def dflash_generate(
    *,
    target,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    block_size: int,
    stop_token_ids: list[int] | None,
    temperature: float = 0.0,
    drafter,
    target_layer_ids,
) -> SimpleNamespace:
    if temperature != 0.0:
        raise ValueError("DFlash block baseline currently supports greedy temperature=0 only")
    if block_size <= 1:
        raise ValueError("DFlash block baseline requires block_size > 1")

    device = input_ids.device
    prompt_len = input_ids.shape[1]
    max_length = prompt_len + int(max_new_tokens)
    output_ids = torch.empty((1, max_length + block_size), dtype=torch.long, device=device)
    position_ids = torch.arange(output_ids.shape[1], device=device).unsqueeze(0)
    cache = make_cache(target)
    stop_token_set = set(int(t) for t in stop_token_ids) if stop_token_ids else set()
    reset = getattr(drafter, "reset_cache", None)
    if reset is not None:
        reset()

    prefill_start = _cuda_time(device)
    prefill_out = target(
        input_ids=input_ids,
        position_ids=position_ids[:, :prompt_len],
        past_key_values=cache,
        cache_position=position_ids[0, :prompt_len],
        use_cache=True,
        output_hidden_states=True,
        logits_to_keep=1,
    )
    output_ids[:, :prompt_len] = input_ids
    first_tok = prefill_out.logits[:, -1:, :].argmax(dim=-1)
    output_ids[:, prompt_len:prompt_len + 1] = first_tok
    target_hidden = extract_context_feature(prefill_out.hidden_states, target_layer_ids)
    time_to_first_token = _cuda_time(device) - prefill_start

    decode_start = _cuda_time(device)
    start = prompt_len
    stopped = bool(stop_token_set and int(first_tok.item()) in stop_token_set)
    acceptance_lengths: list[int] = []
    rounds = 0
    k = block_size - 1

    while start < max_length and not stopped:
        committed = output_ids[:, :start + 1]
        draft_logits = drafter.propose_logits(
            committed,
            k,
            target_hidden=target_hidden,
        )
        drafts = draft_logits[:, :k, :].argmax(dim=-1).reshape(-1)
        remaining = max_length - start - 1
        if remaining <= 0:
            break
        drafts = drafts[:remaining]

        step_ids = torch.cat([output_ids[:, start:start + 1], drafts.view(1, -1)], dim=1)
        step_len = step_ids.shape[1]
        step_pos = position_ids[:, start:start + step_len]
        out = target(
            input_ids=step_ids,
            position_ids=step_pos,
            past_key_values=cache,
            cache_position=step_pos.squeeze(0),
            use_cache=True,
            output_hidden_states=True,
        )
        greedy = out.logits[0].argmax(dim=-1)
        acc = 0
        for i in range(drafts.numel()):
            if int(drafts[i]) == int(greedy[i]):
                acc += 1
            else:
                break
        correction = greedy[acc]

        keep_len = start + 1 + acc
        crop = getattr(cache, "crop", None)
        if crop is not None:
            crop(keep_len)
        else:
            for layer in getattr(cache, "layers", []):
                keys = getattr(layer, "keys", None)
                values = getattr(layer, "values", None)
                if keys is not None and values is not None:
                    layer.keys = keys[..., :keep_len, :]
                    layer.values = values[..., :keep_len, :]

        selected_hidden = extract_context_feature(out.hidden_states, target_layer_ids)[:, :1 + acc, :]
        target_hidden = torch.cat([target_hidden, selected_hidden], dim=1)

        if acc:
            output_ids[0, start + 1:start + 1 + acc] = drafts[:acc]
        output_ids[0, start + 1 + acc] = correction
        new_tokens = drafts[:acc].tolist()
        new_tokens.append(int(correction))
        start += acc + 1
        rounds += 1
        acceptance_lengths.append(acc + 1)
        stopped = bool(stop_token_set and any(int(t) in stop_token_set for t in new_tokens))

    decode_time = _cuda_time(device) - decode_start
    output_ids = output_ids[:, : min(start + 1, max_length)]
    if stop_token_set:
        stop_tensor = torch.tensor(list(stop_token_set), device=device)
        stop_idx = torch.isin(output_ids[0, prompt_len:], stop_tensor).nonzero(as_tuple=True)[0]
        if stop_idx.numel() > 0:
            output_ids = output_ids[:, : prompt_len + int(stop_idx[0].item()) + 1]

    num_output_tokens = int(output_ids.shape[1] - prompt_len)
    return SimpleNamespace(
        output_ids=output_ids,
        num_input_tokens=prompt_len,
        num_output_tokens=num_output_tokens,
        time_to_first_token=time_to_first_token,
        time_per_output_token=(decode_time / num_output_tokens if num_output_tokens else 0.0),
        acceptance_lengths=acceptance_lengths,
        rounds=rounds,
        decode_time=decode_time,
    )
