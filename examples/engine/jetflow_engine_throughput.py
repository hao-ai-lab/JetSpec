"""JetFlow N3 throughput A/B: continuous-batched AR (`generate_batch`) under the
SDPA path vs the paged tree-attention triton kernel, on a real model.

    JETFLOW_TEST_MODEL=Qwen/Qwen3-8B python examples/engine/jetflow_engine_throughput.py

Reports tok/s for each backend over a fixed batch (B=8, 64 new tokens, greedy). The
SDPA N2a baseline reconstructs every seq's dense KV + pads + masks each step; the
kernel reads K/V straight from the pool with no pad/mask/copy-back. Decode-only (AR,
no tree), so this isolates the per-step attention substrate. Not a gate — a number."""
import os
import time

import torch

from jetflow.core.llm import SamplingParams
from jetflow.inference_engine.engine import JetFlowEngine


def _run(engine, prompts, sp):
    """Time `generate_batch` and return (tok/s, total new tokens, wall seconds)."""
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = engine.generate_batch(prompts, sp)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    total = sum(len(o["token_ids"]) for o in out)
    return total / dt, total, dt


def main():
    model = os.environ.get("JETFLOW_TEST_MODEL", "Qwen/Qwen3-8B")
    dtype = torch.bfloat16
    sp = SamplingParams(temperature=0.0, max_new_tokens=64)
    # B=8 distinct prompts (ragged lengths exercise the per-seq pool).
    raw = [
        "The three primary colors are",
        "In a distant galaxy, a lone explorer discovered",
        "The key to a good algorithm is",
        "Once upon a time, in a small village near the mountains,",
        "Photosynthesis is the process by which",
        "To prove the theorem, we first assume that",
        "The history of computing began when",
        "A balanced diet consists of",
    ]

    results = {}
    for backend in ("sdpa", "triton_paged_tree"):
        engine = JetFlowEngine(model, device="cuda", dtype=dtype, attn_backend=backend)
        prompts = [
            engine.tokenizer(p, return_tensors="pt").input_ids.to("cuda") for p in raw
        ]
        _run(engine, prompts, SamplingParams(temperature=0.0, max_new_tokens=4))  # warmup
        tps, total, dt = _run(engine, prompts, sp)
        results[backend] = tps
        print(f"{backend:18s}: {tps:8.2f} tok/s  ({total} tok in {dt:.3f}s)")
        del engine
        torch.cuda.empty_cache()

    sdpa, kern = results["sdpa"], results["triton_paged_tree"]
    print(f"speedup (kernel / sdpa): {kern / sdpa:.3f}x")


if __name__ == "__main__":
    main()
