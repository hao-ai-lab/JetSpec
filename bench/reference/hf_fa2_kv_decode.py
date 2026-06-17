"""Minimal raw-HF Qwen3-8B + FlashAttention2 KV-cache decode smoke test.

This intentionally bypasses JetFlow wrappers. It measures the same baseline class as
the reference benchmark: prompt prefill once, then one-token-at-a-time greedy
decode through an HF `DynamicCache`.

Example:

    CUDA_VISIBLE_DEVICES=0 python bench/reference/hf_fa2_kv_decode.py \
      --model Qwen/Qwen3-8B --max-new 256 --warmup 1
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
    "How much in dollars does she make every day?"
)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.inference_mode()
def generate_one(model, tokenizer, prompt: str, max_new: int, device: torch.device) -> dict:
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    prompt_len = input_ids.shape[1]
    cache = DynamicCache()

    pos = torch.arange(prompt_len, device=device).unsqueeze(0)
    _sync(device)
    prefill_start = time.perf_counter()
    out = model(
        input_ids=input_ids,
        position_ids=pos,
        past_key_values=cache,
        use_cache=True,
        logits_to_keep=1,
    )
    _sync(device)
    prefill_time = time.perf_counter() - prefill_start

    cache = out.past_key_values
    next_tok = out.logits[:, -1:, :].argmax(dim=-1)
    tokens = [int(next_tok.item())]

    cur = prompt_len
    _sync(device)
    decode_start = time.perf_counter()
    for _ in range(max(0, max_new - 1)):
        pos = torch.tensor([[cur]], device=device)
        out = model(
            input_ids=next_tok,
            position_ids=pos,
            past_key_values=cache,
            use_cache=True,
            logits_to_keep=1,
        )
        cache = out.past_key_values
        next_tok = out.logits[:, -1:, :].argmax(dim=-1)
        tokens.append(int(next_tok.item()))
        cur += 1
    _sync(device)
    decode_time = time.perf_counter() - decode_start

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

    for _ in range(max(0, args.warmup)):
        generate_one(model, tokenizer, args.prompt, args.max_new, device)

    result = generate_one(model, tokenizer, args.prompt, args.max_new, device)
    decode_tps = (
        result["output_tokens"] / result["decode_time"]
        if result["decode_time"] > 0 else 0.0
    )
    print(f"model={args.model} attn_implementation=flash_attention_2 dtype={args.dtype}")
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
