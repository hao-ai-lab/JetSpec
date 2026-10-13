"""Minimal raw-HF Qwen3-8B + FlashAttention2 KV-cache decode smoke test.

This intentionally bypasses JetSpec generation wrappers. It mirrors the AR path in
`bench/reference/benchmark.py`: chat-template prompt formatting, prompt prefill
once, preallocated token/position tensors, direct target calls, and decode-only
timing for one-token-at-a-time greedy KV-cache decode.

Example:

    CUDA_VISIBLE_DEVICES=0 python bench/reference/hf_fa2_kv_decode.py \
      --model Qwen/Qwen3-8B --max-new 256 --warmup 1 --ignore-eos
"""
from __future__ import annotations

import argparse
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache


DEFAULT_PROMPT = (
    "Janet's ducks lay 16 eggs per day. She eats three for breakfast every "
    "morning and bakes muffins for her friends every day with four. She sells "
    "the remainder at the farmers' market daily for $2 per fresh duck egg. "
    "How much in dollars does she make every day?\n"
    "Please reason step by step, and put your final answer within \\boxed{}."
)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _format_prompt(tokenizer, prompt: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def _sample_greedy(logits: torch.Tensor) -> torch.Tensor:
    if logits.dim() == 2:
        logits = logits.unsqueeze(1)
    return logits[:, -1:, :].argmax(dim=-1)


@torch.inference_mode()
def generate_one(
    model,
    tokenizer,
    prompt: str,
    max_new: int,
    device: torch.device,
    *,
    ignore_eos: bool,
) -> dict:
    text = _format_prompt(tokenizer, prompt)
    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
    prompt_len = input_ids.shape[1]
    max_length = prompt_len + max_new
    if max_new <= 0:
        return {
            "prompt_tokens": prompt_len,
            "output_tokens": 0,
            "prefill_time": 0.0,
            "decode_time": 0.0,
            "text": "",
        }

    output_ids = torch.empty((1, max_length + 1), dtype=torch.long, device=device)
    position_ids = torch.arange(output_ids.shape[1], device=device).unsqueeze(0)
    cache = DynamicCache()

    _sync(device)
    prefill_start = time.perf_counter()
    out = model(
        input_ids=input_ids,
        position_ids=position_ids[:, :prompt_len],
        past_key_values=cache,
        cache_position=position_ids[0, :prompt_len],
        use_cache=True,
        output_hidden_states=False,
        logits_to_keep=1,
    )
    _sync(device)
    prefill_time = time.perf_counter() - prefill_start

    cache = out.past_key_values
    output_ids[:, :prompt_len] = input_ids
    next_tok = _sample_greedy(out.logits)
    output_ids[:, prompt_len:prompt_len + 1] = next_tok
    eos_ids = set() if ignore_eos else {int(tokenizer.eos_token_id)}
    out_ids = [] if ignore_eos else [int(next_tok.item())]

    _sync(device)
    decode_start = time.perf_counter()
    for step in range(1, max_new):
        if out_ids and out_ids[-1] in eos_ids:
            break
        start = prompt_len + step - 1
        step_ids = output_ids[:, start:start + 1]
        step_pos = position_ids[:, start:start + 1]
        out = model(
            input_ids=step_ids,
            position_ids=step_pos,
            past_key_values=cache,
            cache_position=step_pos.squeeze(0),
            use_cache=True,
            output_hidden_states=False,
            logits_to_keep=1,
        )
        cache = out.past_key_values
        next_tok = _sample_greedy(out.logits)
        output_ids[:, start + 1:start + 2] = next_tok
        if not ignore_eos:
            out_ids.append(int(next_tok.item()))
    _sync(device)
    decode_time = time.perf_counter() - decode_start

    if ignore_eos:
        tokens = output_ids[0, prompt_len:max_length].cpu().tolist()
    else:
        tokens = out_ids
    return {
        "prompt_tokens": prompt_len,
        "output_tokens": len(tokens),
        "prefill_time": prefill_time,
        "decode_time": decode_time,
        "text": tokenizer.decode(tokens, skip_special_tokens=True),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-new", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    parser.add_argument("--ignore-eos", action="store_true",
                        help="Generate exactly --max-new tokens and avoid per-token .item() sync")
    parser.add_argument("--torch-compile", action="store_true",
                        help="Experimental: torch.compile(dynamic=True) the full HF model")
    parser.add_argument("--compile-cache-limit", type=int, default=512,
                        help="Dynamo graph cache/recompile limit for --torch-compile")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this FA2 smoke test.")
    try:
        import flash_attn  # noqa: F401
    except ImportError as exc:
        raise SystemExit("flash_attn is not installed; cannot use flash_attention_2.") from exc

    device = torch.device("cuda")
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=dtype,
        attn_implementation="flash_attention_2",
    ).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if args.torch_compile:
        torch._dynamo.config.allow_unspec_int_on_nn_module = True
        limit = max(args.compile_cache_limit, int(getattr(model.config, "num_hidden_layers", 0)) * 4)
        torch._dynamo.config.recompile_limit = max(torch._dynamo.config.recompile_limit, limit)
        torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, limit)
        torch._dynamo.config.accumulated_recompile_limit = max(
            torch._dynamo.config.accumulated_recompile_limit,
            limit * 4,
        )
        model = torch.compile(model, dynamic=True)

    for _ in range(max(0, args.warmup)):
        generate_one(model, tokenizer, args.prompt, args.max_new, device, ignore_eos=args.ignore_eos)

    result = generate_one(model, tokenizer, args.prompt, args.max_new, device, ignore_eos=args.ignore_eos)
    decode_tps = (
        result["output_tokens"] / result["decode_time"]
        if result["decode_time"] > 0 else 0.0
    )
    print(f"model={args.model} attn_implementation=flash_attention_2 dtype={args.dtype} "
          f"torch_compile={args.torch_compile} ignore_eos={args.ignore_eos}")
    print(f"prompt_tokens={result['prompt_tokens']} output_tokens={result['output_tokens']}")
    print(f"prefill_ms={1000.0 * result['prefill_time']:.1f}")
    print(
        f"decode_ms={1000.0 * result['decode_time']:.1f} "
        f"decode_tps={decode_tps:.1f} tok/s"
    )
    print("\n--- decoded text ---")
    print(result["text"])


if __name__ == "__main__":
    main()
