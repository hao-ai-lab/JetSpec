"""P3 early-stop probe: does path-conditioning actually change the drafter's topk?

THROWAWAY scratch. For the first K rounds of a real decode: take the marginal
draft logits (normal propose: noise = [anchor, mask x14]), pick the heap-best
path from the built tree, then re-run the SAME head forward with the noise
positions filled by the real path tokens (the fork's prune_and_regrow trick).
Compare per-depth top-2 sets, refreshed vs marginal. If they agree on >90% of
depths, P3's coherence premise is false — kill before building. Big
divergence at deep depths = build P3.

    NANO_FUSE_GEMMS=1 NANO_BACKEND=triton_paged_tree_cudagraph_nogather ... \
      python bench/reseed_probe.py --rounds 12 --budget 127
"""
import argparse
import os

import torch
from transformers import DynamicCache

from bench.tree_diag import build_prompts
from ptd.engine.llm import SamplingParams
from ptd.nano_vllm.engine import NanoEngine
from ptd.models.draft_head import load_draft_head
from ptd.draft_head_drafter import DraftHeadTreeDrafter


def conditioned_logits(fwd, context_ids, path_tokens, target_hidden):
    """One head forward with noise = [anchor, path tokens] instead of masks.

    Mirrors `_DraftHeadForward._forward_head` (draft_head_drafter.py:47-91)
    except the block fill: position d's logits then condition on the REAL
    path prefix instead of mask tokens (the fork's prune_and_regrow trick).
    """
    block_size = fwd.block_size
    anchor = context_ids[0, -1].view(1, 1).to(fwd.device)
    fill = path_tokens[: block_size - 1].view(1, -1).to(device=fwd.device, dtype=anchor.dtype)
    block_output_ids = torch.cat([anchor, fill], dim=1)          # (1, block_size)
    noise_embedding = fwd.target.model.embed_tokens(block_output_ids)
    ctx_len = target_hidden.shape[1]
    position_ids = torch.arange(ctx_len + block_size, device=fwd.device).unsqueeze(0)
    hidden = fwd.head(
        target_hidden=target_hidden.to(device=fwd.device, dtype=fwd.dtype),
        noise_embedding=noise_embedding,
        position_ids=position_ids,
        past_key_values=DynamicCache(),
        use_cache=False,
        is_causal=fwd.head.resolve_causal_head("auto"),
    )
    draft_slice = slice(0, block_size - 1) if fwd.draft_shift else slice(-block_size + 1, None)
    return fwd.target.lm_head(hidden[:, draft_slice, :])         # (1, block_size-1, V)


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=12)
    ap.add_argument("--budget", type=int, default=127)
    ap.add_argument("--tree-width", type=int, default=7)
    ap.add_argument("--prompt-set", default="gsm8k")
    args = ap.parse_args()

    backend = os.environ.get("NANO_BACKEND", "triton_paged_tree_cudagraph")
    eng = NanoEngine("Qwen/Qwen3-8B", device="cuda", dtype=torch.bfloat16,
                     attn_backend=backend, block_size=16)
    head = load_draft_head(os.environ["PTD_DRAFT_HEAD"])
    tli, bs = head.target_layer_ids, head.block_size
    drafter = DraftHeadTreeDrafter(head, target=eng.model, block_size=bs,
                                   target_layer_ids=tli, draft_shift=False)
    fwd = drafter._fwd

    records = []

    class Recorder:
        def __getattr__(self, name):
            return getattr(drafter, name)

        def propose_logits(self, context_ids, depth, target_hidden=None, **kw):
            logits = drafter.propose_logits(context_ids, depth,
                                            target_hidden=target_hidden, **kw)
            if len(records) < args.rounds:
                records.append({
                    "context_ids": context_ids.detach().clone(),
                    "target_hidden": target_hidden.detach().clone(),
                    "marginal": logits.detach().float().cpu(),
                })
            return logits

    prompts = build_prompts(eng.tokenizer, 2, prompt_set=args.prompt_set)
    sp = SamplingParams(0.0, 1024)
    eng.generate_tree(prompts[0], Recorder(), sampling_params=sp,
                      block_size=bs, tree_width=args.tree_width,
                      budget=args.budget, target_layer_ids=tli, return_stats=True)

    print(f"rounds recorded: {len(records)}")
    agree_counts = torch.zeros(bs - 1)
    echo_counts = torch.zeros(bs - 1)
    n = 0
    for rec in records:
        marg = rec["marginal"][0]                                 # (D, V)
        # heap-best chain proxy: greedy rank-1 token per depth = the best path
        # crossproduct/top2gap put first (exact for the probe's purpose).
        path = marg.argmax(-1)                                    # (D,)
        cond = conditioned_logits(fwd, rec["context_ids"], path,
                                  rec["target_hidden"])[0].float().cpu()
        n += 1
        for d in range(marg.shape[0]):
            m1 = int(marg[d].argmax())
            c1 = int(cond[d].argmax())
            fed = int(path[d])         # the token fed at this position (echo check)
            agree_counts[d] += (m1 == c1)
            echo_counts[d] += (c1 == fed)
    rates = (agree_counts / max(n, 1)).tolist()
    echo = (echo_counts / max(n, 1)).tolist()
    print("per-depth: top-1 agreement (marg vs cond) | echo rate (cond top-1 == fed token):")
    for d in range(len(rates)):
        print(f"  d{d}: agree={rates[d]:.3f}  echo={echo[d]:.3f}")
    overall = sum(rates) / len(rates)
    overall_echo = sum(echo) / len(echo)
    print(f"overall: agree={overall:.3f} echo={overall_echo:.3f}")
    # fed token at depth d IS the marginal argmax (the path), so echo==agree
    # unless the head revises. High echo+agree = fixed-point (uninformative);
    # low agree + low echo = genuine revision = the P3 coherence signal.
    print("P3_VERDICT:", "KILL (cond just echoes/agrees - no new information)" if overall > 0.9
          else "BUILD (head genuinely revises the marginal path given the real prefix)")


if __name__ == "__main__":
    main()
