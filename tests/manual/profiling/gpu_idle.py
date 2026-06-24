"""E3 sizing: per-round GPU-busy vs wall → the recoverable GPU-idle (the inter-graph
bubble + host gaps). gpu-busy = sum of CUDA kernel device-time (kupti traces the
kernels even inside a replayed cudagraph); wall = un-profiled real round. idle =
wall - gpu-busy = the ceiling for capture/overlap (E3). SESSION/production config."""
import os
import time

import torch
from torch.profiler import profile, ProfilerActivity

from jetspec.core.llm import SamplingParams
from jetspec.inference_engine.engine import JetSpecEngine
from jetspec.models.draft_head import load_draft_head
from jetspec.draft_head_adapter import DraftHeadTreeDrafter

GSM8K_FMT = ("{question}\n"
             "Please reason step by step, and put your final answer within \\boxed{{}}.")


def dev_us(k):
    return getattr(k, "self_device_time_total", None) or getattr(k, "self_cuda_time_total", 0)


def main():
    backend = os.environ.get("JETSPEC_BACKEND", "triton_paged_tree_cudagraph_nogather")
    head_id = os.environ["JETSPEC_DRAFT_HEAD"]
    eng = JetSpecEngine("Qwen/Qwen3-8B", device="cuda", dtype=torch.bfloat16,
                        attn_backend=backend, block_size=16)
    head = load_draft_head(head_id)
    tli, bs = head.target_layer_ids, head.block_size
    drafter = DraftHeadTreeDrafter(head, target=eng.model, block_size=bs,
                                   target_layer_ids=tli, draft_shift=False)

    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="test")
    prompts = [eng.tokenizer.apply_chat_template(
        [{"role": "user", "content": GSM8K_FMT.format(question=ds[i]["question"])}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
        for i in range(8)]
    sp = SamplingParams(0.0, 210)
    max_len = max(len(eng.tokenizer(p)["input_ids"]) for p in prompts)
    tkw = dict(block_size=bs, tree_width=7, budget=127, algo="accum_logp",
               target_layer_ids=tli, return_stats=True, session=True,
               session_prompt_capacity=((max_len + 255) // 256) * 256)

    # warm (capture graphs + prime session) before any timing.
    eng.generate_tree(prompts[0], drafter, sampling_params=sp, **tkw)
    eng.generate_tree(prompts[0], drafter, sampling_params=sp, **tkw)
    torch.cuda.synchronize()

    # un-profiled wall (the real round).
    rounds = 0
    t0 = time.perf_counter()
    for p in prompts:
        rounds += eng.generate_tree(p, drafter, sampling_params=sp, **tkw)["rounds"]
    torch.cuda.synchronize()
    wall_ms = (time.perf_counter() - t0) * 1e3

    # profiled gpu-busy (kernel device-time).
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        pr = 0
        for p in prompts:
            pr += eng.generate_tree(p, drafter, sampling_params=sp, **tkw)["rounds"]
        torch.cuda.synchronize()
    gpu_ms = sum(dev_us(k) for k in prof.key_averages()) / 1e3

    wpr, gpr = wall_ms / rounds, gpu_ms / pr
    print(f"\nrounds: wall={rounds} prof={pr}")
    print(f"wall/round      = {wpr:.2f} ms")
    print(f"gpu-busy/round  = {gpr:.2f} ms")
    print(f"GPU-idle/round  = {wpr - gpr:.2f} ms   <- recoverable ceiling (E3 + overlap)")
    print(f"gpu_util        = {gpr / wpr:.2f}")
    prof.export_chrome_trace("/tmp/round_trace.json")
    print("chrome trace -> /tmp/round_trace.json (inspect drafter|verify gap)")


if __name__ == "__main__":
    main()
