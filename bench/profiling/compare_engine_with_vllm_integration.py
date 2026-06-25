"""Identical-conditions JetSpec-vs-fork comparison for the verify-only decode speedup.

Measures JetSpec's `decode_cuda_speedup` (AR verify-only GPU time per token ÷ tree
verify-only GPU time per token) under conditions MATCHED to a reference vLLM fork
DFlash profile that reported comparable speedup (`see-reference-fork-benchmarks` /
`gains_report.txt`). The fork's verify-only `decode_cuda_s` wraps only the target
`execute_model` (the drafter `propose` runs in a separate, untimed RPC), so this
harness EXCLUDES JetSpec's drafter from both legs via a CUDA-event split, matching the
fork's accounting.

Matched conditions (fork -> here):
  prompt_set            gsm8k                      gsm8k (same dataset)
  prompt_format         chat_template              apply_chat_template(enable_thinking=False)
  prompt_fmt            "{question}\nPlease ...\\boxed{{}}."   identical (dflash_profiling.py:48-55)
  target model          Qwen/Qwen3-8B              Qwen/Qwen3-8B
  dtype                 bf16                       bf16
  block_size (depth)    16                         16
  tree_width            7                          7
  max_tree_budget       255                        255
  tree_draft            accum_logp  +              algo="accum_logp"  (both = cumulative-logprob
  tree_construction     breadth_first +              breadth-biased heap, per-depth top-k width,
  max_draft_passes      0                            budget-capped: byte-for-byte the same heap loop)
  head_type             causal                     epoch6 causal distill head (JETSPEC_DRAFT_HEAD)
  max_num_seqs / bs     1 / 1                      single-stream
  samples               4                          --samples (default 4)
  output tokens/sample  ~208                       --max-tokens (default 210)

The intrinsic difference being measured is the verify ENGINE: JetSpec's triton paged
tree-attn vs the fork's FLASH_ATTN flash-varlen. That is the comparison SUBJECT, not
a condition to match.

    JETSPEC_BACKEND=triton_paged_tree_compiled CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. \
    JETSPEC_DRAFT_HEAD=JetSpec/jetspec-qwen3-8b \
    python bench/profiling/compare_engine_with_vllm_integration.py --samples 4 --max-tokens 210
"""
import argparse
import os
import statistics

import torch
from torch.cuda import Event

from jetspec.core.llm import SamplingParams
from jetspec.inference_engine.engine import JetSpecEngine
from jetspec.models.draft_head import load_draft_head
from jetspec.draft_head_adapter import DraftHeadTreeDrafter

# Fork-exact gsm8k prompt format (dflash_profiling.py `load_dataset_prompt_bank`).
GSM8K_FMT = ("{question}\n"
             "Please reason step by step, and put your final answer within \\boxed{{}}.")

# Fork reference numbers (gains_report.txt / metrics_summary.txt, gsm8k, tp1 bs1).
FORK = dict(decode_cuda_speedup=7.546, accept_len=7.155, avg_tree_nodes=254,
            ar_decode_cuda_s=20.366, dflash_decode_cuda_s=2.699)


class _GpuTimer:
    """Accumulate GPU self-time (ms) of a wrapped callable via CUDA events."""

    def __init__(self):
        self.ms = 0.0
        self.n = 0

    def wrap(self, fn):
        def inner(*a, **k):
            # During CUDA-graph capture the timed callable may be re-entered (e.g. the
            # graph captures `CompiledVerifyStack.__call__`); event-record + synchronize
            # are illegal mid-capture and would invalidate it. Pass through untimed when
            # the current stream is capturing — the capture itself is timed at the
            # `_capture_bucket` boundary, not at the inner stack call.
            if torch.cuda.is_current_stream_capturing():
                return fn(*a, **k)
            s, e = Event(enable_timing=True), Event(enable_timing=True)
            s.record()
            r = fn(*a, **k)
            e.record()
            e.synchronize()
            self.ms += s.elapsed_time(e)
            self.n += 1
            return r
        return inner


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=210)
    ap.add_argument("--tree-width", type=int, default=7)
    ap.add_argument("--budget", type=int, default=255)
    ap.add_argument("--algo", type=str, default="accum_logp")
    args = ap.parse_args()

    backend = os.environ.get("JETSPEC_BACKEND", "triton_paged_tree_compiled")
    head_id = os.environ["JETSPEC_DRAFT_HEAD"]
    eng = JetSpecEngine("Qwen/Qwen3-8B", device="cuda", dtype=torch.bfloat16,
                     attn_backend=backend, block_size=16)
    head = load_draft_head(head_id)
    tli, bs = head.target_layer_ids, head.block_size
    drafter = DraftHeadTreeDrafter(head, target=eng.model, block_size=bs,
                                   target_layer_ids=tli, draft_shift=False)
    print(f"backend={backend}  head={head_id}")
    print(f"head: target_layer_ids={tli} block_size={bs}")

    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="test")
    prompts = [eng.tokenizer.apply_chat_template(
        [{"role": "user", "content": GSM8K_FMT.format(question=ds[i]["question"])}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
        for i in range(args.samples)]

    sp = SamplingParams(0.0, args.max_tokens)
    tkw = dict(block_size=bs, tree_width=args.tree_width, budget=args.budget,
               algo=args.algo, target_layer_ids=tli, return_stats=True)

    # --- verify-only event split. The verify leg is whatever runs the target forward:
    #   - tree/AR, eager or sdpa   -> runner.forward.
    #   - tree, compiled backend   -> the compiled verify stack (compiled_verify or the
    #     lazily-built need_hidden stack); AR compiled -> compiled_ar. All compiled
    #     stacks are CompiledVerifyStack instances invoked as `stack(...)`, i.e. via
    #     `CompiledVerifyStack.__call__` — so we time at the CLASS method (instance-attr
    #     wrapping misses the `__call__` dunder, which Python resolves on the type).
    # The drafter (propose_logits) is timed separately and EXCLUDED from the speedup,
    # matching the fork (its `decode_cuda_s` wraps only `execute_model`, not `propose`).
    verify = _GpuTimer()
    draft = _GpuTimer()
    prefill = _GpuTimer()
    capture = _GpuTimer()      # A3-GRAPH: one-time per-prompt graph capture (subtracted)
    drafter.propose_logits = draft.wrap(drafter.propose_logits)
    if backend in ("triton_paged_tree_compiled", "triton_paged_tree_cudagraph"):
        # On the compiled backends, `runner.forward` runs ONLY prefill (decode goes
        # through the compiled stacks / captured graphs), so route it to a separate
        # `prefill` timer and report DECODE-only speedup — matching the fork's
        # `decode_cuda_s` (which excludes `prefill_cuda_s`).
        #   - "..._compiled": the tree verify leg IS `CompiledVerifyStack.__call__`.
        #   - "..._cudagraph": the steady-state tree verify is `GraphedVerify.replay`
        #     (the captured graph; the stack `__call__` only runs at warmup/capture,
        #     which is excluded). Time the leg that actually runs per decode round.
        # The AR leg stays the compiled N=1 stack on both, so we keep timing
        # `CompiledVerifyStack.__call__` too — but on the cudagraph backend the tree
        # verify no longer routes through it, so wrapping replay is what captures the
        # tree leg. Wrapping both is safe: AR -> stack, tree -> replay, disjoint.
        eng.runner.forward = prefill.wrap(eng.runner.forward)
        from jetspec.inference_engine.compiled_verify_stack import CompiledVerifyStack
        CompiledVerifyStack.__call__ = verify.wrap(CompiledVerifyStack.__call__)
        if backend == "triton_paged_tree_cudagraph":
            # The tree verify leg is `GraphedVerify.replay` (copy-in + `g.replay()`). The
            # FIRST replay per (pool,width) ALSO captures the graph inside `replay` (a
            # one-time per-prompt setup: warm + trace + capture, ~seconds). Timing replay
            # naively folds that capture into the steady-state number, so we time replay
            # ourselves and SKIP the round in which a capture fired (detected by the
            # `graphs` dict growing) — recording it to a separate `capture` timer instead.
            # The reported verify/round is then pure per-round cost (copy-in + launch),
            # matching how the compiled leg times only its per-round forward.
            from jetspec.inference_engine.graph_capture import GraphedVerify
            _orig_replay = GraphedVerify.replay

            def _timed_replay(self, *a, **k):
                pre = len(self.graphs)
                s, e = Event(enable_timing=True), Event(enable_timing=True)
                s.record()
                r = _orig_replay(self, *a, **k)
                e.record()
                e.synchronize()
                dt = s.elapsed_time(e)
                if len(self.graphs) > pre:        # this round captured -> setup, not steady
                    capture.ms += dt
                    capture.n += 1
                else:
                    verify.ms += dt
                    verify.n += 1
                return r

            GraphedVerify.replay = _timed_replay
    else:
        # eager/sdpa: runner.forward is BOTH prefill and decode; the per-sample prefill
        # (1 call over the ~291-tok prompt) is a small, symmetric add to both legs.
        eng.runner.forward = verify.wrap(eng.runner.forward)

    # warmup (excluded): warm at the SAME `sp` as the timed legs, so the warmup's
    # `reserve_capacity(prompt_len + max_new_tokens + budget)` reserves the SAME pool
    # block-count the timed loop will use. A short warmup (max_new_tokens=8/16) reserves
    # a SMALLER pool, so the timed loop hit a fresh pool shape on its first round and paid
    # a one-time recompile + Triton autotune INSIDE the measured window — which, over a
    # short decode, dominated the per-round average and collapsed decode_cuda_speedup to
    # ~2×. Warm twice so autotune fully settles. (The compiled stack marks the pool
    # block-dim dynamic, so the single warmed graph is reused across prompt lengths and
    # the timed loop sees zero recompiles.)
    eng.generate(prompts[0], sp)
    eng.generate(prompts[0], sp)
    eng.generate_tree(prompts[0], drafter, sampling_params=sp, **tkw)
    eng.generate_tree(prompts[0], drafter, sampling_params=sp, **tkw)
    torch.cuda.synchronize()

    # ---- AR leg: verify-only GPU time per output token --------------------------
    verify.ms, verify.n = 0.0, 0
    ar_tok = 0
    for p in prompts:
        o = eng.generate(p, sp)
        ar_tok += len(o["token_ids"])
    torch.cuda.synchronize()
    ar_verify_ms = verify.ms
    ar_per_tok = ar_verify_ms / ar_tok                     # ms verify GPU / output tok

    # ---- tree leg: verify-only GPU time per output token ------------------------
    verify.ms, verify.n, draft.ms, draft.n = 0.0, 0, 0.0, 0
    capture.ms, capture.n = 0.0, 0
    tree_tok, rounds, acc_sum, N_all = 0, 0, 0, []
    for p in prompts:
        o = eng.generate_tree(p, drafter, sampling_params=sp, **tkw)
        tree_tok += len(o["token_ids"])
        rounds += o["rounds"]
        acc_sum += sum(o["accept_lengths"])
        N_all += o["tree_sizes"]
    torch.cuda.synchronize()
    # `verify.ms` already excludes capture rounds (the cudagraph `_timed_replay` routes a
    # capturing round's time to the `capture` timer instead; on other backends capture
    # is unused and `verify.ms` is the per-round forward time directly).
    tree_verify_ms = verify.ms
    accept_len = acc_sum / rounds
    # Per-ROUND verify GPU over the verify-TIMED rounds (verify.n), then per-tok via
    # accept_len. Do NOT divide tree_verify_ms by all `tree_tok`/`rounds`: the cudagraph
    # `_timed_replay` routes each cold-bucket round to the `capture` timer (one per prompt,
    # since GraphedVerify rebuilds per prompt pool), so `verify.ms` covers `verify.n`
    # rounds -- not all `rounds`. Dividing by all tokens undercounts per-tok and INFLATES
    # the speedup; (verify.n, accept_len) is the consistent pair, matching the clean
    # per-call event-split methodology.
    tree_per_round = tree_verify_ms / max(1, verify.n)
    tree_per_tok = tree_per_round / accept_len
    avgN = statistics.mean(N_all)

    speedup = ar_per_tok / tree_per_tok
    print("\n=== verify-only (drafter excluded), identical fork conditions ===")
    print(f"AR   : verify_gpu={ar_verify_ms/1e3:7.3f}s  ntok={ar_tok:4d}  "
          f"verify/tok={ar_per_tok:6.2f}ms")
    print(f"tree : verify_gpu={tree_verify_ms/1e3:7.3f}s  ntok={tree_tok:4d}  rounds={rounds:4d}  "
          f"verify_rounds={verify.n:4d}  verify/round={tree_per_round:6.2f}ms  verify/tok={tree_per_tok:6.2f}ms")
    print(f"       drafter_gpu={draft.ms/1e3:7.3f}s ({draft.ms/(draft.ms+tree_verify_ms)*100:.0f}% of draft+verify, EXCLUDED)")
    if capture.n:
        print(f"       graph_capture_gpu={capture.ms/1e3:7.3f}s over {capture.n} captures "
              f"(one-time per-prompt setup, EXCLUDED from verify/round)")
    print(f"       accept_len={accept_len:.3f}   tree-N: min/mean/max={min(N_all)}/{avgN:.1f}/{max(N_all)}")
    print(f"\nRESULT JetSpec verify-only decode_cuda_speedup = {speedup:.2f}x   "
          f"[fork={FORK['decode_cuda_speedup']:.2f}x]")
    print(f"       accept_len {accept_len:.3f} vs fork {FORK['accept_len']}   "
          f"avgN {avgN:.0f} vs fork {FORK['avg_tree_nodes']}")


if __name__ == "__main__":
    main()
