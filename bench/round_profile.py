"""Per-round host/GPU phase breakdown of nano_vllm generate_tree.

THROWAWAY scratch (path-to-1000 instrumentation). Ranks the host-overhead levers:
drafter forward / tree build / KV reserve / verify replay / tree_accept / KV gather
vs the total per-round wall. gpu_util ~0.21 means ~80% of each round is host work
sitting between GPU ops — this says WHICH host phase to attack first.

Each phase timer does sync->perf_counter->call->sync, so a phase's number is its
GPU+host wall INCLUDING the D2H-sync stall it forces (that stall IS the cost we
want to see). The total is the real (un-instrumented-inner) generate_tree wall.

    JETFLOW_BACKEND=triton_paged_tree_cudagraph \
      CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. \
      PTD_DRAFT_HEAD=Snyhlxde/ptd-qwen3-8b-distill-epoch6-3e-4-no-gamma \
      python bench/round_profile.py --samples 3 --budget 255
"""
import argparse
import os
import time

import torch

from ptd.engine.llm import SamplingParams
from ptd.jetflow.engine import JetFlowEngine
from ptd.models.draft_head import load_draft_head
from ptd.draft_head_drafter import DraftHeadTreeDrafter
import ptd.jetflow.engine as eng_mod

GSM8K_FMT = ("{question}\n"
             "Please reason step by step, and put your final answer within \\boxed{{}}.")


class Phase:
    def __init__(self, name):
        self.name = name
        self.ms = 0.0
        self.n = 0

    def wrap(self, fn):
        def inner(*a, **k):
            # sync/timing is illegal mid CUDA-graph capture (it invalidates the
            # capture) — pass through untimed when the stream is capturing. Capture
            # only fires during warmup, which is excluded from the timed totals anyway.
            if torch.cuda.is_current_stream_capturing():
                return fn(*a, **k)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            r = fn(*a, **k)
            torch.cuda.synchronize()
            self.ms += (time.perf_counter() - t0) * 1e3
            self.n += 1
            return r
        return inner

    def reset(self):
        self.ms = 0.0
        self.n = 0


def _try_patch_class(phase, obj, attr):
    """Patch obj.attr (class method or module fn) with the phase timer; return True if done."""
    if hasattr(obj, attr):
        setattr(obj, attr, phase.wrap(getattr(obj, attr)))
        return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=210)
    ap.add_argument("--budget", type=int, default=255)
    ap.add_argument("--tree-width", type=int, default=7)
    ap.add_argument("--algo", default="crossproduct")
    ap.add_argument("--cprofile", action="store_true",
                    help="also run ONE decode under cProfile and print top host-side "
                         "functions by tottime (exposes the un-patched OTHER python work)")
    ap.add_argument("--session", action="store_true",
                    help="reserve + freeze the pool up front (matches tps_walltime --session "
                         "/ the production config) so _grow_pool never fires in the hot loop")
    args = ap.parse_args()

    backend = os.environ.get("JETFLOW_BACKEND", "triton_paged_tree_cudagraph")
    head_id = os.environ["PTD_DRAFT_HEAD"]
    eng = JetFlowEngine("Qwen/Qwen3-8B", device="cuda", dtype=torch.bfloat16,
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
        for i in range(args.samples)]
    sp = SamplingParams(0.0, args.max_tokens)
    tkw = dict(block_size=bs, tree_width=args.tree_width, budget=args.budget,
               algo=args.algo, target_layer_ids=tli, return_stats=True)
    if args.session:
        tkw["session"] = True
        max_len = max(len(eng.tokenizer(p)["input_ids"]) for p in prompts)
        tkw["session_prompt_capacity"] = ((max_len + 255) // 256) * 256

    # ---- phase patches (class / module / instance level so internally-created objects hit them) ----
    p_draft = Phase("drafter.propose_logits")
    p_verify = Phase("verify (replay/stack)")
    p_accept = Phase("tree_accept")
    p_reserve = Phase("reserve_tree_slots")
    p_gather = Phase("kv gather")
    p_build = Phase("tree build")
    patched = []

    drafter.propose_logits = p_draft.wrap(drafter.propose_logits)
    patched.append("drafter.propose_logits")

    from ptd.jetflow.graph_capture import GraphedVerify
    from ptd.jetflow.compiled_verify_stack import CompiledVerifyStack
    if _try_patch_class(p_verify, GraphedVerify, "replay"):
        patched.append("GraphedVerify.replay")
    CompiledVerifyStack.__call__ = p_verify.wrap(CompiledVerifyStack.__call__)
    patched.append("CompiledVerifyStack.__call__")

    # tree_accept / ancestor build: generate_tree does `from ptd.tree import ...`
    # INSIDE the function body each call, so patch the ptd.tree namespace (the
    # old eng_mod patch never fired — engine has no module-level tree_accept).
    import ptd.tree as ptree
    if _try_patch_class(p_accept, ptree, "tree_accept"):
        patched.append("ptd.tree.tree_accept")
    # greedy path uses the L2 GPU accept (engine.py:797 imports it per round)
    import ptd.tree._core.accept as pacc
    if _try_patch_class(p_accept, pacc, "gpu_tree_accept"):
        patched.append("gpu_tree_accept")
    p_anc = Phase("build_ancestor_matrix")
    if _try_patch_class(p_anc, ptree, "build_ancestor_matrix"):
        patched.append("ptd.tree.build_ancestor_matrix")
    # staging (L3 round buffers: qq_bias fill + pads + cu/slk)
    p_stage = Phase("stage_tree_inputs")
    from ptd.jetflow.engine import _LogicalRoundBuffers
    if _try_patch_class(p_stage, _LogicalRoundBuffers, "stage_tree_inputs"):
        patched.append("_LogicalRoundBuffers.stage_tree_inputs")

    # KV reserve + gather on the cache class. The L5 logical path uses
    # reserve_logical_slots/release_round_blocks (reserve_tree_slots is the
    # gather-backend name and shows 0 calls on nogather).
    p_release = Phase("release_round_blocks")
    try:
        from ptd.jetflow.paged_kv_cache import PagedKVCache
        if _try_patch_class(p_reserve, PagedKVCache, "reserve_tree_slots"):
            patched.append("PagedKVCache.reserve_tree_slots")
        if _try_patch_class(p_reserve, PagedKVCache, "reserve_logical_slots"):
            patched.append("PagedKVCache.reserve_logical_slots")
        if _try_patch_class(p_release, PagedKVCache, "release_round_blocks"):
            patched.append("PagedKVCache.release_round_blocks")
        if _try_patch_class(p_gather, PagedKVCache, "gather"):
            patched.append("PagedKVCache.gather")
    except Exception as e:
        print(f"(cache patch skipped: {e})")

    # tree build: patch the algorithm class's build (engine constructs algo_obj
    # internally; the old fanout_cap_builder patch caught 0 calls — crossproduct's
    # build does not route through it on this path).
    try:
        from ptd.tree.baselines.crossproduct import CrossProduct
        if _try_patch_class(p_build, CrossProduct, "build"):
            patched.append("CrossProduct.build")
        import ptd.tree._core.fanout_cap_builder as fcb
        if _try_patch_class(p_build, fcb, "build_with_per_depth_cap"):
            patched.append("build_with_per_depth_cap")
    except Exception as e:
        print(f"(build patch skipped: {e})")

    print(f"backend={backend}  patched: {patched}\n")

    # warmup (compile + capture), then reset timers.
    eng.generate_tree(prompts[0], drafter, sampling_params=sp, **tkw)
    eng.generate_tree(prompts[0], drafter, sampling_params=sp, **tkw)
    torch.cuda.synchronize()
    for ph in (p_draft, p_verify, p_accept, p_reserve, p_release, p_gather, p_build, p_anc, p_stage):
        ph.reset()
    eng_mod._COMMIT_MS[0] = 0.0

    # timed: total wall + per-phase.
    rounds = 0
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for p in prompts:
        o = eng.generate_tree(p, drafter, sampling_params=sp, **tkw)
        rounds += o["rounds"]
    torch.cuda.synchronize()
    total_ms = (time.perf_counter() - t0) * 1e3

    phases = [p_draft, p_verify, p_accept, p_reserve, p_release, p_gather, p_build, p_anc, p_stage]
    measured = sum(ph.ms for ph in phases)
    other = total_ms - measured

    print(f"rounds={rounds}  total_wall={total_ms/1e3:.3f}s  per-round={total_ms/rounds:.2f}ms\n")
    print(f"{'phase':<28}{'ms/round':>10}{'% round':>9}{'calls/round':>13}")
    print("-" * 60)
    for ph in phases:
        mr = ph.ms / rounds
        print(f"{ph.name:<28}{mr:>10.2f}{100*ph.ms/total_ms:>8.1f}%{ph.n/rounds:>13.2f}")
    print(f"{'OTHER (build*/assemble/commit/EOS host)':<28}{other/rounds:>10.2f}"
          f"{100*other/total_ms:>8.1f}%")
    if eng_mod._COMMIT_MS[0]:
        print(f"  [PTD_TIME_COMMIT] commit-proper (logical-commit + EOS): "
              f"{eng_mod._COMMIT_MS[0]/rounds:.2f}ms/round  "
              f"(the rest of OTHER = bs=1 bubbles + syncs + orchestration)")
    print("\n(note: instrumented per-phase sync inflates total vs the un-instrumented "
          "236 tok/s number; use the % split to RANK levers, not absolute ms.)")

    if args.cprofile:
        # Host-side python attribution of the OTHER bucket: one decode under
        # cProfile, top functions by tottime. GPU waits show up inside
        # synchronize/item callers; ignore those rows and read the pure-python
        # ones (heap build, list/tensor assembly, bookkeeping).
        import cProfile
        import io
        import pstats
        pr = cProfile.Profile()
        pr.enable()
        eng.generate_tree(prompts[0], drafter, sampling_params=sp, **tkw)
        pr.disable()
        s = io.StringIO()
        pstats.Stats(pr, stream=s).sort_stats("tottime").print_stats(30)
        print("\n### cProfile (1 decode, top 30 by tottime) ###")
        print(s.getvalue())


if __name__ == "__main__":
    main()
