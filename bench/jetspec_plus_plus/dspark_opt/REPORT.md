# A fair DSpark baseline for Table 3

*Qwen3-8B, NVIDIA B200 (NVLink), branch `ranonrkm/baseline-opt`.*

## The argument in one paragraph

At batch size 1, decoding is memory-bound. Each layer reads its full weight matrices whether it processes 16 tokens or 128, so the cost is set by weight bytes and kernel launches, not token count. For example, the gate/up matmul takes 39.6 µs at 128 rows vs 24 µs at 16. So a 16-token DSpark draft should cost no more than a 128-token JetSpec++ drafter pass. In the released code it costs three times as much: 2.45 ms vs ~0.8 ms. The difference is implementation. The DSpark baseline runs with the repo's own optimized kernels forced off, without the device-side verify the 27B baseline uses, and on half as many GPUs. With those three gaps closed, synchronous DSpark matches or beats JetSpec++.

## Setup

| | |
|---|---|
| Target | Qwen/Qwen3-8B, bf16 (no quantization or precision change anywhere) |
| Task | GSM8K: the 128 sealed Table 3 prompts, run as 4 groups of 32 |
| Decoding | batch size 1, temperature 0, 2048-token cap, EOS stop |
| Metric | Table 3 reducer: rule A, steady rounds (first decode round of each request excluded) |
| DSpark (DS) | released `dspark-qwen3-8b-800k`, block 15 (16-token verify), Table 3 recipe |
| JetSpec++ | released `jetspec-plus-plus-qwen3-8b`, j1-r1, 8 rows per round, target and draft on 2 GPUs |

## The three gaps

| | JetSpec++ | DS in Table 3 |
|---|---|---|
| Draft kernels | packed QKV and gate/up, fused norm/RoPE/KV-write, length-aware attention, vocab-tiled LM head | the repo's four equivalent DSpark kernels are **asserted off** (`protocol.py:248`), and `--knob` can't turn them on (the engine profile is frozen first) |
| Target verify | on device, plus 13 more `JETSPEC_VERIFY_*` settings (157 engine settings in total) | host-side for 8B (10 settings, none for speed), although the **27B DS arm** uses device-side verify |
| GPUs | 2 | 1. The engine's 2-GPU option (`tp2`) is slower than 1 GPU and crashes with device-side verify |

Where the stock DS draft spends its time (one round, Nsight Systems):

| What it does | Time | Cost |
|---|---|---|
| Copies the whole KV cache (`cat`, then `repeat_kv`) in every layer, every round | ~0.48 ms | pure overhead |
| Runs attention over the full static cache capacity instead of the live length | 0.60 ms | 32 CTAs on a 148-SM GPU |
| Runs ~15 tiny kernels per layer for RMSNorm and RoPE | ~0.6 ms | launch-bound |
| Runs the Markov head as 15 positions × 7 kernels | 0.36 ms | launch-bound |
| Launches outside the CUDA graphs | ~0.76 ms | host-bound |

## What the branch changes (all opt-in, default off)

1. **Turn on the existing DSpark kernels** (`JETSPEC_BLOCKHEAD_{ATTN_LEN,GLUE_LEAN,QKV_FUSED,LMHEAD_KN}`). These are the same kinds of kernels JetSpec++'s draft uses. The fix lets per-run settings reach them.
2. **Fused Markov step** (`JETSPEC_BLOCKHEAD_MARKOV_FUSED`). One kernel per position instead of ~7, with identical math (fp32 accumulation, bf16 rounding, lowest-index argmax). 0/3,000 tokens differ from stock; 0.36 → 0.24 ms.
3. **Device-side verify** (`SSD_VERIFY_DEVICE=1`). This is the existing setting the 27B DS arm already uses.
4. **Make `tp2` work**, so DS gets the same 2 GPUs as JetSpec++:
   - **One-shot NVLink all-reduce** (`JETSPEC_TP_CUSTOM_ALLREDUCE`). At batch size 1 an all-reduce is a 128 KB, latency-bound message. NCCL takes 11.7 µs; FlashInfer's one-shot kernel takes 4.4 µs and gives bit-identical sums, across 73 all-reduces per verify. sglang, vLLM and TRT-LLM all use a kernel like this.
   - **Fix the device-verify crash under TP**, and send tensor arguments by broadcast instead of pickling them through host memory.
   - **Verify pre-issue** (`JETSPEC_TP_VERIFY_PREISSUE`). Rank 1 gets its verify command before rank 0 drafts, so both ranks start verify together.

## Results (B200, GSM8K ×128, batch size 1)

| Arm | GPUs | tok/s | vs AR | tokens/round | round | verify | beyond verify |
|---|---|---|---|---|---|---|---|
| AR (`ar-graphed`, unchanged) | 1 | 193.1 | 1.00× | 1.00 | 5.18 ms | — | — |
| DS, stock (Table 3) | 1 | 823.3 | 4.26× | 7.37 | 9.01 ms | 5.43 ms | 3.58 ms |
| DS + 1, 2 (draft kernels) | 1 | 1034.9 | 5.36× | 7.33 | 7.12 ms | 5.48 ms | 1.64 ms |
| DS + 1–3 (+ device verify) | 1 | 1148.0 | 5.94× | 7.28 | 6.38 ms | 4.75 ms | 1.63 ms |
| DS + 1–4 (`tp2`) | 2 | **1239.4** | **6.42×** | 7.36 | 5.98 ms | 3.98 ms | 2.00 ms |
| JetSpec++ j1-r1, 8 rows | 2 | **1169.7** | 6.06× | 7.83 | 6.77 ms | 4.54 ms | 2.24 ms |

**JetSpec++ over DS:** 1.42× (stock) → 1.13× → 1.02× → **0.94×** (same GPU count). The README claims 1.54×.

**AR.** We didn't change it. One AR step (5.18 ms) costs more than a whole 16-token JetSpec++ verify (4.54 ms) on the same target, because the AR path makes 7 host syncs and 36 launches outside the CUDA graphs per token. sglang's AR on the same GPU takes 3.57 ms. Speedups "over AR" are therefore inflated for every method.

## Checks

- **Tokens per round** is unchanged within run-to-run noise (stock reruns: 7.28–7.37).
- **Token-level output parity isn't a usable test in this engine, even for stock.** Two runs of the identical stock DS recipe agree on only 13–18 of 32 outputs per group, and stock DS matches greedy AR on 9–19 of 32. The cause is bf16 near-ties between runs. The optimized arms fall in the same range (7–18 of 32).
- **Rerun from this branch's commit:**
  - DS `tp2` groups p000 and p032 gave 1235.7 and 1210.2 tok/s (round 6.07 and 5.98 ms, verify 4.00 ms). The earlier runs gave 1270.7 and 1220.7 (5.87 and 5.99 ms, 3.97 and 3.99 ms).
  - The logs confirm all five draft settings, the one-shot all-reduce, device verify and pre-issue were active.
  - JetSpec++ from the same branch: 1163.9 tok/s, vs 1169.7 before.
- **Fused Markov:** `check_markov.py` finds 0/3,000 differing tokens on the real weights.

## Headroom (sglang, same GPU, same DSpark checkpoint, block 15)

sglang with `trtllm_mha` attention for target and draft. Its DSpark default, FlashInfer, adds a host-side plan before every step, so it isn't used here. 32 GSM8K prompts, batch size 1, median per step (CUDA events):

| | sglang TP=1 | sglang TP=2 | TP=2 speedup | this branch, 1 GPU | this branch, `tp2` |
|---|---|---|---|---|---|
| Draft (sglang) / beyond verify (here) | 1.04 ms | 0.86 ms | 1.21× | 1.63 ms | 2.00 ms |
| Verify | 3.86 ms | 3.02 ms | 1.28× | 4.75 ms | 3.98 ms |
| Round | 5.02 ms | 4.00 ms | 1.25× | 6.38 ms | 5.98 ms |

- **There is still headroom on both sides.** At 1 GPU, ~0.6 ms of draft-side time and ~0.9 ms of verify. Closing it would favor DSpark further.
- **Sharding the draft at TP=2 helps sglang modestly** (1.04 → 0.86 ms). Only the weight-bound part halves. The replicated Markov head, the 10 all-reduces plus the logit all-gather, and the launch costs stay fixed.
- **This branch doesn't shard the DS draft, and JetSpec++'s draft isn't sharded either.** An attempt with rank 0 driving rank 1 per round gained nothing, because rank 1 started late. sglang avoids that by running every rank as an equal.
- **With FlashInfer attention**, sglang measured draft 1.71 / 1.69 ms and verify 4.24 / 3.39 ms (TP=1 / TP=2). The per-step overhead hides the TP=2 draft gain.

## Reproduce

```
H=host-8b.json; O=out
K="--knob JETSPEC_BLOCKHEAD_ATTN_LEN=1 --knob JETSPEC_BLOCKHEAD_GLUE_LEAN=1 --knob JETSPEC_BLOCKHEAD_QKV_FUSED=1 --knob JETSPEC_BLOCKHEAD_LMHEAD_KN=1 --knob JETSPEC_BLOCKHEAD_MARKOV_FUSED=1"
python bench/jetspec_plus_plus/reproduce_paper_results.py --model qwen3-8b --host-values $H --devices 0,1 --benchmarks gsm8k --arms AR,DS --output-dir $O/stock
python bench/jetspec_plus_plus/dspark_opt/run_ds.py --label ds_kernels_vd --device 0 --host-values $H --stock-dir $O/stock --output-dir $O $K --knob SSD_VERIFY_DEVICE=1
python bench/jetspec_plus_plus/dspark_opt/run_ds.py --label ds_tp2 --placement tp2 --device 0,1 --host-values $H --stock-dir $O/stock --output-dir $O $K \
    --knob SSD_VERIFY_DEVICE=1 --knob JETSPEC_TP_CUSTOM_ALLREDUCE=1 --knob JETSPEC_TP_VERIFY_PREISSUE=1
python bench/jetspec_plus_plus/dspark_opt/check_markov.py /path/to/dspark-qwen3-8b-800k     # fused Markov == stock
python bench/jetspec_plus_plus/dspark_opt/reduce_single.py <jetspec++ read.json>             # same reducer for a JetSpec++ read
```
