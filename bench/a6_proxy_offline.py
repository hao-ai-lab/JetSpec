"""A6 go/no-go probe — path_conditional_refresh ∘ top2gap_fanout vs crossproduct.

THROWAWAY, STANDALONE, ALLOWED-TO-BE-SLOW. This script answers ONE question
before we invest in deployment-grade draft-side KV machinery:

    Does feeding PATH-CONDITIONAL draft logits into top2gap_fanout's per-depth
    gate beat (a) plain crossproduct and (b) MARGINAL top2gap_fanout in
    acceptance length?

It does NOT modify a single line of the shipped engine/tree code — losslessness
of the OSS release stays byte-intact. The two cheap arms (crossproduct, marginal
top2gap) call `llm.generate_tree(...)` directly (zero re-implementation risk).
The expensive arm hand-rolls the per-round loop, mirroring
`ptd/engine/llm.py:generate_tree` verbatim, but injects per-node path-conditional
logits into a LOCAL COPY of `build_with_per_depth_cap` (the integration point is
the line-53 `child_cum_lp` accumulation in
`ptd/tree/_core/fanout_cap_builder.py`).

CHEAP-MODE proxy (the research stand-in for the vLLM draft-side tree attention):
each round, (1) build a MARGINAL top2gap scaffold tree; (2) run the engine's
ancestor-mask verify forward over [prefix | all N nodes] with
output_hidden_states=True and extract per-node path-conditional target hidden via
`extract_context_feature` — faithful because the 4D ancestor mask makes node v
attend only to its ancestors; (3) for each scaffold node call the REAL
`drafter.propose_logits(path_tokens, depth=D, target_hidden=node_hidden)` to get
its path-conditional (1,D,V); (4) REBUILD the tree with a per-node conditional
build (caps gate + heap children both read the PARENT node's conditional top-K,
same source → no rank mismatch); (5) verify the rebuilt tree with the engine's
ancestor-mask forward + `tree_accept` (both unchanged). Two-pass so the ancestor
matrix always matches the committed topology.

IDENTITY CHECKS (asserted in-script, before the verdict — the correctness net):
  1. Arm (c) with refresh DISABLED (conditional := marginal logprobs) reproduces
     arm (b)'s accept_lengths element-for-element on the same prompt/seed.
  2. Arm (c) at the top2gap identity knob (g_0=1e9 → caps=K → full fanout)
     equals crossproduct accept_len.
  3. The rebuilt tree's parent_indices are internally consistent before
     tree_accept.

SELF-TEST (`_selftest()`, CPU, no model load): the LOCAL conditional
build_with_per_depth_cap, fed the marginal logprobs as the "conditional" source,
produces a tree IDENTICAL to the stock build_with_per_depth_cap.

USAGE (orchestrator runs on b200; CANNOT run on GPU here — CPU-sanity only):
    # self-test only, no model:
    python bench/a6_proxy_offline.py --selftest
    # smoke (5 prompts):
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. \
      PTD_DRAFT_HEAD=Snyhlxde/ptd-qwen3-8b-distill-epoch6-3e-4-no-gamma \
      python bench/a6_proxy_offline.py --samples 5
    # full probe (20 prompts):
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. \
      PTD_DRAFT_HEAD=Snyhlxde/ptd-qwen3-8b-distill-epoch6-3e-4-no-gamma \
      python bench/a6_proxy_offline.py --samples 20
"""
import argparse
import heapq
import math
import os
import statistics

import torch

# Engine + tree public surface (read-only use; nothing here is mutated).
from ptd.tree import build_ancestor_matrix, tree_accept
from ptd.tree._core.accept import _build_child_maps_cpu
from ptd.tree._core.ancestor import (
    _build_ancestor_matrix_np,
    _build_packed_ancestor_matrix_np,
)
from ptd.tree._core.base import DraftTree


# ===========================================================================
# LOCAL COPY of build_with_per_depth_cap — the integration point.
#
# Differs from ptd/tree/_core/fanout_cap_builder.py:build_with_per_depth_cap in
# ONE way: the per-depth top-K tokens/logprobs are indexed by the PARENT NODE's
# ROOT→NODE TOKEN PATH (path-conditional), not by depth (marginal). When
# `cond_by_path` is empty, every node falls back to the marginal row at its depth
# and this reduces EXACTLY to the stock builder (asserted by _selftest +
# identity check 1).
#
# `cond_by_path`: dict {path-tuple -> ([K ints], [K floats])} giving the
# conditional distribution for the children OF a node, keyed by the node's
# root→node token path EXCLUDING the root (i.e. `()` for the root node). The
# distribution is the drafter's depth-d prediction conditioned on that path,
# where d is the node's depth. Keying by PATH (not node index) is essential:
# this builder is called for a SCAFFOLD pass (empty cond) and a REBUILD pass
# (with cond), and the rebuild's heap assigns DIFFERENT node indices than the
# scaffold — but a node's children distribution depends only on its token path,
# so a path key is stable across the rebuild. A path with no entry (beyond
# refresh budget / depth >= D) falls back to `marginal_*[d]`.
# ===========================================================================
def build_with_per_node_cap(
    root_token: int,
    marginal_tokens_cpu: list[list[int]],     # (D, K) — fallback / scaffold source
    marginal_logprobs_cpu: list[list[float]],  # (D, K)
    cond_by_path: dict,                        # path-tuple -> ([K ints], [K floats])
    caps_fn,                                   # (topk_logprobs_rows, K) -> per-row cap (list[int])
    budget: int,
    device: torch.device,
) -> DraftTree:
    """Heap loop where each popped parent expands using ITS OWN conditional top-K.

    For each parent at depth `d`, identified by its root→node token path (excl.
    root), we read the conditional row `cond_by_path[path]` (its drafter
    prediction for depth d), fall back to the marginal row `marginal_*[d]` when
    absent, compute the cap from that SAME row via `caps_fn` (top-2 gap on the
    conditional dist), and add children from it. The line-53
    `child_cum_lp = -neg_cum_lp + row_logprobs[j]` accumulation reads the
    conditional logprobs — THE A6 integration point.
    """
    D = len(marginal_tokens_cpu)
    K = len(marginal_tokens_cpu[0]) if D > 0 else 0

    tokens_list: list[int] = [root_token]
    parents_list: list[int] = [-1]
    depths_list: list[int] = [0]
    cum_lp_list: list[float] = [0.0]
    # Per-node root→node token path (excl. root); root node 0 has path ().
    chains_list: list[tuple] = [()]
    num_nodes = 1

    counter = 0
    heap: list[tuple[float, int, int]] = [(0.0, counter, 0)]

    while heap and num_nodes < budget:
        neg_cum_lp, _, node_idx = heapq.heappop(heap)
        d = depths_list[node_idx]
        if d >= D:
            continue
        # Per-node conditional row keyed by this node's token path (fall back to
        # marginal at this depth). Path-keying survives the rebuild's reindexing.
        row_tokens, row_logprobs = cond_by_path.get(
            chains_list[node_idx], (marginal_tokens_cpu[d], marginal_logprobs_cpu[d])
        )
        # Cap from the SAME row the children come from (no rank mismatch). caps_fn
        # expects a list of per-depth rows; we hand it the single row and take [0].
        b_d = caps_fn([row_logprobs], K)[0]
        children_to_add = min(b_d, K, budget - num_nodes)
        for j in range(children_to_add):
            child_token = row_tokens[j]
            child_cum_lp = -neg_cum_lp + row_logprobs[j]   # line-53 accumulation
            tokens_list.append(child_token)
            parents_list.append(node_idx)
            depths_list.append(d + 1)
            cum_lp_list.append(child_cum_lp)
            chains_list.append(chains_list[node_idx] + (child_token,))
            counter += 1
            heapq.heappush(heap, (-child_cum_lp, counter, num_nodes))
            num_nodes += 1

    ancestor_np = _build_ancestor_matrix_np(parents_list, num_nodes)
    ancestor_packed_np = _build_packed_ancestor_matrix_np(parents_list, num_nodes)
    child_maps = _build_child_maps_cpu(tokens_list, parents_list, num_nodes)

    return DraftTree(
        token_ids=torch.tensor(tokens_list, dtype=torch.long, device=device),
        parent_indices=torch.tensor(parents_list, dtype=torch.long, device=device),
        depth=torch.tensor(depths_list, dtype=torch.long, device=device),
        num_nodes=num_nodes,
        cum_logprob=torch.tensor(cum_lp_list, dtype=torch.float32, device=device),
        child_maps=child_maps,
        ancestor=torch.from_numpy(ancestor_np).to(device),
        ancestor_packed=torch.from_numpy(ancestor_packed_np).to(device),
    )


# ---- top2gap cap logic, copied locally (matches ptd .../top2gap.py:_sigmoid_cap)
def _sigmoid_cap(g_d: float, K: int, beta: float, g_0: float) -> int:
    arg = -beta * (g_d - g_0)
    if arg >= 0:
        s = 1.0 / (1.0 + math.exp(-arg))
    else:
        e = math.exp(arg)
        s = e / (1.0 + e)
    return max(1, int(round(K * s)))


def _make_top2gap_caps_fn(beta: float, g_0: float):
    """Return a caps_fn(topk_logprobs_rows, K) matching Top2GapFanout.caps_from_topk."""
    def caps_fn(topk_logprobs_rows, tree_width, **kwargs) -> list[int]:
        K = len(topk_logprobs_rows[0]) if topk_logprobs_rows else 0
        if K < 2:
            return [1] * len(topk_logprobs_rows)
        gaps = [lp[0] - lp[1] for lp in topk_logprobs_rows]
        return [_sigmoid_cap(g_d, K, beta, g_0) for g_d in gaps]
    return caps_fn


def _make_crossproduct_caps_fn():
    """caps_fn matching CrossProduct.caps_from_topk: full fanout (K at every row)."""
    def caps_fn(topk_logprobs_rows, tree_width, **kwargs) -> list[int]:
        K = len(topk_logprobs_rows[0]) if topk_logprobs_rows else max(tree_width, 1)
        return [K] * len(topk_logprobs_rows)
    return caps_fn


# ===========================================================================
# SELF-TEST (CPU, no model). The conditional builder fed the marginal rows as
# the "conditional" source must equal the stock build_with_per_depth_cap.
# ===========================================================================
def _selftest() -> bool:
    """Return True iff PASS. Compares build_with_per_node_cap (marginal-as-cond)
    against the SHIPPED build_with_per_depth_cap on fake logits, for both the
    top2gap and crossproduct caps."""
    from ptd.tree._core.fanout_cap_builder import build_with_per_depth_cap

    torch.manual_seed(0)
    D, V, K = 3, 50, 7
    budget = 63
    root_token = 11
    fake_logits = torch.randn(1, D, V)  # (1, D, V) — a fake drafter output
    log_probs = torch.log_softmax(fake_logits.squeeze(0), dim=-1)  # (D, V)
    topk_lp_t, topk_tok_t = torch.topk(log_probs, K, dim=-1)
    marginal_tokens = topk_tok_t.tolist()
    marginal_logprobs = topk_lp_t.tolist()

    ok = True
    for label, caps_fn, beta, g_0 in (
        ("top2gap(beta=2,g0=1)", _make_top2gap_caps_fn(2.0, 1.0), 2.0, 1.0),
        ("crossproduct", _make_crossproduct_caps_fn(), None, None),
    ):
        # Stock per-depth caps from the marginal rows.
        b_per_depth = caps_fn(marginal_logprobs, K)
        stock = build_with_per_depth_cap(
            root_token=root_token,
            topk_tokens_cpu=marginal_tokens,
            topk_logprobs_cpu=marginal_logprobs,
            b_per_depth=b_per_depth,
            budget=budget,
            device=torch.device("cpu"),
        )
        # Local conditional build with EMPTY cond -> every node falls back to marginal.
        local = build_with_per_node_cap(
            root_token=root_token,
            marginal_tokens_cpu=marginal_tokens,
            marginal_logprobs_cpu=marginal_logprobs,
            cond_by_path={},
            caps_fn=caps_fn,
            budget=budget,
            device=torch.device("cpu"),
        )
        same = (
            stock.num_nodes == local.num_nodes
            and torch.equal(stock.token_ids, local.token_ids)
            and torch.equal(stock.parent_indices, local.parent_indices)
            and torch.equal(stock.depth, local.depth)
        )
        print(f"  [selftest] {label:<22} stock_N={stock.num_nodes:>3} "
              f"local_N={local.num_nodes:>3} identical={same}")
        ok = ok and same
    print(f"  [selftest] RESULT: {'PASS' if ok else 'FAIL'}")
    return ok


def _assert_tree_consistent(tree: DraftTree) -> None:
    """parent_indices internal consistency: root has -1, every other parent is a
    lower index at depth d-1, and child_maps are well-formed."""
    parents = tree.parent_indices.tolist()
    depths = tree.depth.tolist()
    assert parents[0] == -1, "root parent must be -1"
    assert depths[0] == 0, "root depth must be 0"
    for i in range(1, tree.num_nodes):
        p = parents[i]
        assert 0 <= p < i, f"node {i} parent {p} not a prior node"
        assert depths[i] == depths[p] + 1, (
            f"node {i} depth {depths[i]} != parent depth {depths[p]} + 1"
        )


# ===========================================================================
# Prompt building — copied from bench/benchmark.py (gsm8k, seed=0 shuffle,
# chat-template enable_thinking=False).
# ===========================================================================
PROMPT_FMT = {
    "gsm8k": ("openai/gsm8k", "main", "test", "question",
              "{q}\nPlease reason step by step, and put your final answer within \\boxed{{}}."),
}


def build_prompts(tokenizer, dataset, n):
    from datasets import load_dataset
    repo, cfg, split, field, fmt = PROMPT_FMT[dataset]
    ds = load_dataset(repo, cfg, split=split) if cfg else load_dataset(repo, split=split)
    if n < len(ds):
        ds = ds.shuffle(seed=0).select(range(n))   # MATCH benchmark.py exactly
    prompts = []
    for i in range(min(n, len(ds))):
        user = fmt.format(q=ds[i][field])
        prompts.append(tokenizer.apply_chat_template(
            [{"role": "user", "content": user}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False))
    return prompts


# ===========================================================================
# Arm (c): hand-rolled per-round loop mirroring llm.py:generate_tree, with the
# two-pass path-conditional refresh + per-node conditional rebuild.
# ===========================================================================
@torch.inference_mode()
def generate_tree_path_conditional(
    llm, drafter, prompt, *, block_size, tree_width, budget, target_layer_ids,
    sp, beta, g_0, identity_knob, refresh_enabled, refresh_cap,
):
    """Path-conditional A6 proxy. Returns {'accept_lengths', 'tree_sizes', 'rounds'}.

    Mirrors ptd/engine/llm.py:generate_tree (recompute / SDPA path) verbatim for
    prefill, the 4D ancestor mask, the verify forward, and tree_accept. The ONLY
    additions are the scaffold→refresh→rebuild two-pass and the per-node
    conditional build (build_with_per_node_cap).

    Knobs:
      beta, g_0          top2gap params for the conditional caps gate.
      identity_knob      if True, use crossproduct caps (full fanout) for BOTH
                         scaffold and rebuild -> identity check 2 vs crossproduct.
      refresh_enabled    if False, the "conditional" source IS the marginal (no
                         drafter rerun) -> identity check 1 vs marginal top2gap.
      refresh_cap        cap on how many scaffold nodes get a refresh forward
                         (budget guard; nodes beyond it fall back to marginal).
    """
    from transformers import DynamicCache
    from ptd.models.draft_head import extract_context_feature
    from ptd.engine.sampler import sample

    device = llm.device
    dtype = llm.model.dtype
    neg = torch.finfo(dtype).min
    D = max(1, block_size - 1)
    need_hidden = target_layer_ids is not None and block_size > 1
    tli = target_layer_ids

    if identity_knob:
        caps_fn = _make_crossproduct_caps_fn()
    else:
        caps_fn = _make_top2gap_caps_fn(beta, g_0)

    committed = llm.tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    new_ids, rounds = [], 0
    accept_lengths, tree_sizes = [], []
    target_hidden = None

    # --- prefill (verbatim from generate_tree) ---
    pos = torch.arange(committed.shape[1], device=device).unsqueeze(0)
    logits, _, full_hidden = llm.runner.forward(
        committed, DynamicCache(), pos,
        output_hidden_states=need_hidden, target_layer_ids=tli,
    )
    if full_hidden is not None:
        target_hidden = full_hidden
    first_tok = sample(logits[:, -1:, :], sp.temperature)
    new_ids.append(int(first_tok.item()))
    committed = torch.cat([committed, first_tok.view(1, 1)], dim=1)
    if int(first_tok.item()) in llm.eos_token_ids:
        return {"accept_lengths": accept_lengths, "tree_sizes": tree_sizes, "rounds": rounds}

    def _verify_forward(tree):
        """Engine's exact recompute ancestor-mask verify forward over
        [prefix | nodes]. Returns (target_logits (1,N,V), full_hidden_or_None)."""
        N = tree.num_nodes
        prefix = committed[:, :-1]
        P = prefix.shape[1]
        seq = torch.cat([prefix, tree.token_ids.view(1, -1)], dim=1)   # (1, P+N)
        depths = tree.depth.tolist()
        # Reuse the engine's exact RoPE position construction.
        posf = torch.tensor([list(range(P)) + [P + d for d in depths]], device=device)
        T = P + N
        allowed = torch.zeros(T, T, dtype=torch.bool, device=device)
        if P > 0:
            allowed[:P, :P] = torch.tril(torch.ones(P, P, dtype=torch.bool, device=device))
            allowed[P:, :P] = True
        allowed[P:, P:] = build_ancestor_matrix(tree).bool()
        mask = torch.where(allowed, torch.zeros((), dtype=dtype, device=device),
                           torch.full((), neg, dtype=dtype, device=device)).view(1, 1, T, T)
        f_logits, _, f_hidden = llm.runner.forward(
            seq, DynamicCache(), posf, attention_mask=mask,
            output_hidden_states=need_hidden, target_layer_ids=tli,
        )
        return f_logits[:, P:, :], f_hidden, P

    while len(new_ids) < sp.max_new_tokens:
        # ---- pass 1: marginal scaffold (same marginal source as the engine) ----
        draft_logits = drafter.propose_logits(committed, D, target_hidden=target_hidden).to(device)  # (1,D,V)
        log_probs = torch.log_softmax(draft_logits.squeeze(0), dim=-1)   # (D, V)
        K = tree_width
        topk_lp_t, topk_tok_t = torch.topk(log_probs, K, dim=-1)
        marginal_tokens = topk_tok_t.tolist()
        marginal_logprobs = topk_lp_t.tolist()

        scaffold = build_with_per_node_cap(
            root_token=int(committed[0, -1]),
            marginal_tokens_cpu=marginal_tokens,
            marginal_logprobs_cpu=marginal_logprobs,
            cond_by_path={},   # marginal scaffold
            caps_fn=caps_fn, budget=budget, device=device,
        )

        cond_by_path = {}
        if refresh_enabled:
            # ---- pass 1.5: faithful per-node path-conditional hidden via the
            # engine's ancestor-mask forward over the scaffold (node v attends only
            # to its ancestors), then per-node REAL propose_logits. ----
            _, scaf_hidden, P = _verify_forward(scaffold)   # scaf_hidden: (1, P+N, dim) or None
            if scaf_hidden is not None:
                s_parents = scaffold.parent_indices.tolist()
                s_depths = scaffold.depth.tolist()
                s_tokens = scaffold.token_ids.tolist()

                # Path tokens (committed-prefix + root→node) for each node, and a
                # per-path cache key so siblings sharing an ancestor prefix reuse
                # the head forward within this round.
                base_ctx = committed  # (1, T); last token == root == node 0
                cache = {}

                def _path_token_ids(node_idx):
                    # root→node token chain (excluding root, which == committed[-1]).
                    chain = []
                    cur = node_idx
                    while cur != 0:
                        chain.append(s_tokens[cur])
                        cur = s_parents[cur]
                    chain.reverse()
                    return chain

                def _path_index_ids(node_idx):
                    # root→node NODE-INDEX chain INCLUDING the root (node 0):
                    # [0, a1, ..., node_idx]. The index analogue of _path_token_ids.
                    idx_chain = []
                    cur = node_idx
                    while cur != 0:
                        idx_chain.append(cur)
                        cur = s_parents[cur]
                    idx_chain.append(0)
                    idx_chain.reverse()
                    return idx_chain

                # Refresh nodes whose depth < D (they have a depth-d prediction to
                # condition) up to refresh_cap, BFS order (node index order == BFS).
                refreshed = 0
                for v in range(scaffold.num_nodes):
                    if s_depths[v] >= D:
                        continue
                    if refreshed >= refresh_cap:
                        break
                    chain = _path_token_ids(v)
                    key = tuple(chain)
                    if key not in cache:
                        # context_ids = committed + path (its last token = the anchor
                        # the head conditions on). committed == [prefix | root], chain
                        # excludes root, so path_ctx == [prefix | root | tok(a1..v)].
                        if chain:
                            path_ctx = torch.cat(
                                [base_ctx, torch.tensor([chain], device=device)], dim=1)
                        else:
                            path_ctx = base_ctx
                        # FULL path-conditional target hidden aligned to path_ctx,
                        # reconstructed from the scaffold verify (no extra forward).
                        # index_chain = [0, a1, ..., v] (node indices root→v). Under the
                        # 4D ancestor mask, scaffold node i's hidden at position P+i is
                        # its path-conditional hidden, and prefix positions are standard
                        # causal — so [prefix positions | index_chain positions] is
                        # exactly the target hidden for the linear path_ctx.
                        index_chain = _path_index_ids(v)
                        hid_idx = list(range(P)) + [P + i for i in index_chain]
                        node_hidden = scaf_hidden[:, hid_idx, :]   # (1, P+len(index_chain), dim)
                        assert node_hidden.shape[1] == path_ctx.shape[1], (
                            f"target_hidden len {node_hidden.shape[1]} != "
                            f"path_ctx len {path_ctx.shape[1]}"
                        )
                        cond_logits = drafter.propose_logits(
                            path_ctx, D, target_hidden=node_hidden).to(device)  # (1,D,V) — REAL propose
                        cache[key] = cond_logits
                    cond_logits = cache[key]
                    # The node's children are its DEPTH-d prediction (first row): row
                    # index 0 of the path-conditional output is the prediction for the
                    # token immediately following the path (i.e. this node's child).
                    cond_lp = torch.log_softmax(cond_logits.squeeze(0)[0], dim=-1)  # (V,)
                    c_lp_t, c_tok_t = torch.topk(cond_lp, K, dim=-1)
                    # Key by PATH (not node index) so the rebuild — which assigns
                    # different node indices — can retrieve this dist by token path.
                    cond_by_path[tuple(chain)] = (c_tok_t.tolist(), c_lp_t.tolist())
                    refreshed += 1

        # ---- pass 2: rebuild with per-node conditional source, then verify ----
        tree = build_with_per_node_cap(
            root_token=int(committed[0, -1]),
            marginal_tokens_cpu=marginal_tokens,
            marginal_logprobs_cpu=marginal_logprobs,
            cond_by_path=cond_by_path,
            caps_fn=caps_fn, budget=budget, device=device,
        )
        _assert_tree_consistent(tree)   # identity check 3

        target_logits, full_hidden, P = _verify_forward(tree)
        accepted_path, acc, correction = tree_accept(tree, target_logits, sp.temperature)
        if full_hidden is not None:
            idx = list(range(P + 1)) + [P + j for j in accepted_path[1:]]
            target_hidden = full_hidden[:, idx, :]
        accepted = tree.token_ids[torch.tensor(accepted_path[1:], device=device)] if acc > 0 \
            else torch.empty(0, dtype=tree.token_ids.dtype, device=device)
        block = torch.cat([accepted, torch.tensor([correction], device=device)])
        committed = torch.cat([committed, block.view(1, -1)], dim=1)
        rounds += 1
        accept_lengths.append(int(block.numel()))
        tree_sizes.append(int(tree.num_nodes))
        for t in block.tolist():
            new_ids.append(int(t))
            if int(t) in llm.eos_token_ids:
                break
        if new_ids and new_ids[-1] in llm.eos_token_ids:
            break

    return {"accept_lengths": accept_lengths, "tree_sizes": tree_sizes, "rounds": rounds}


# ===========================================================================
# Stats helpers + reporting.
# ===========================================================================
def _tau(accept_lengths):
    return sum(accept_lengths) / len(accept_lengths) if accept_lengths else 0.0


def _tpf(accept_lengths, rounds):
    return (sum(accept_lengths) / rounds) if rounds else 0.0


def _paired_delta(per_prompt_x, per_prompt_y):
    """mean over prompts of (mean accept_len_x - mean accept_len_y) — paired."""
    deltas = []
    for ax, ay in zip(per_prompt_x, per_prompt_y):
        if ax and ay:
            deltas.append(_tau(ax) - _tau(ay))
    return statistics.mean(deltas) if deltas else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--draft-head", default=None)
    ap.add_argument("--dataset", default="gsm8k", choices=list(PROMPT_FMT))
    ap.add_argument("--samples", type=int, default=20)
    ap.add_argument("--width", type=int, default=7)
    ap.add_argument("--budgets", default="63,127,255")
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--beta", type=float, default=2.0)
    ap.add_argument("--g0", type=float, default=1.0)
    ap.add_argument("--refresh-cap", type=int, default=64,
                    help="max scaffold nodes refreshed per round (budget guard)")
    ap.add_argument("--pathcond-caps", default="top2gap", choices=["top2gap", "crossproduct"],
                    help="caps for the path-conditional arm: 'top2gap' (the gate) or "
                         "'crossproduct' (full fanout — isolates conditional LOGIT quality "
                         "from top2gap's gate-collapse)")
    ap.add_argument("--selftest", action="store_true",
                    help="run the CPU self-test (no model load) and exit")
    args = ap.parse_args()

    if args.selftest:
        ok = _selftest()
        raise SystemExit(0 if ok else 1)

    # CPU self-test always runs first as a correctness gate before any model use.
    print("== A6 proxy self-test (CPU) ==")
    if not _selftest():
        raise SystemExit("self-test FAILED — aborting before model load")

    from ptd.engine.llm import LLM, SamplingParams
    from ptd.models.draft_head import load_draft_head
    from ptd.draft_head_drafter import DraftHeadTreeDrafter

    head_path = args.draft_head or os.environ.get("PTD_DRAFT_HEAD")
    if not head_path:
        raise SystemExit("set --draft-head or PTD_DRAFT_HEAD")
    budgets = [int(b) for b in args.budgets.split(",")]

    llm = LLM(args.model)
    head = load_draft_head(head_path)
    tli = head.target_layer_ids
    bs = head.block_size
    D = bs - 1
    assert D == bs - 1, "depth invariant"   # D == block_size - 1
    drafter = DraftHeadTreeDrafter(head, target=llm.model, block_size=bs,
                                   target_layer_ids=tli, draft_shift=False)
    prompts = build_prompts(llm.tokenizer, args.dataset, args.samples)
    sp = SamplingParams(0.0, args.max_new)

    print(f"\nmodel={args.model} head={head_path}")
    print(f"dataset={args.dataset} samples={len(prompts)} block_size={bs} D={D} "
          f"width={args.width} budgets={budgets} max_new={args.max_new} "
          f"beta={args.beta} g_0={args.g0}")
    print("engine = reference SDPA recompute verify (no kernel)\n")

    # --------------------------------------------------------------------
    # IDENTITY CHECKS on the FIRST prompt at the largest budget (cheap-ish;
    # element-for-element). These gate the whole probe.
    # --------------------------------------------------------------------
    p0 = prompts[0]
    big_b = max(budgets)
    print("== identity checks (prompt 0, budget=%d) ==" % big_b)

    def _engine(algo, algo_kwargs, budget):
        out = llm.generate_tree(
            p0, drafter, block_size=bs, tree_width=args.width, budget=budget,
            algo=algo, algo_kwargs=algo_kwargs, target_layer_ids=tli,
            sampling_params=sp, return_stats=True)
        return out["accept_lengths"]

    # (1) refresh DISABLED at top2gap knob == marginal top2gap_fanout engine arm.
    marg_engine = _engine("top2gap_fanout", {"beta": args.beta, "g_0": args.g0}, big_b)
    c_no_refresh = generate_tree_path_conditional(
        llm, drafter, p0, block_size=bs, tree_width=args.width, budget=big_b,
        target_layer_ids=tli, sp=sp, beta=args.beta, g_0=args.g0,
        identity_knob=False, refresh_enabled=False, refresh_cap=args.refresh_cap,
    )["accept_lengths"]
    id1 = (marg_engine == c_no_refresh)
    print(f"  [identity 1] refresh-OFF == engine marginal-top2gap : {id1}")
    if not id1:
        print(f"    engine : {marg_engine}")
        print(f"    arm(c) : {c_no_refresh}")

    # (2) identity knob (caps=K, full fanout) == crossproduct engine arm.
    cross_engine = _engine("crossproduct", {}, big_b)
    c_identity = generate_tree_path_conditional(
        llm, drafter, p0, block_size=bs, tree_width=args.width, budget=big_b,
        target_layer_ids=tli, sp=sp, beta=args.beta, g_0=args.g0,
        identity_knob=True, refresh_enabled=False, refresh_cap=args.refresh_cap,
    )["accept_lengths"]
    id2 = (cross_engine == c_identity)
    print(f"  [identity 2] identity-knob == engine crossproduct    : {id2}")
    if not id2:
        print(f"    engine : {cross_engine}")
        print(f"    arm(c) : {c_identity}")

    if not (id1 and id2):
        raise SystemExit("IDENTITY CHECK FAILED — the proxy is not faithful; verdict suppressed")
    print("  identity checks PASS\n")

    # --------------------------------------------------------------------
    # MEASUREMENT — budget × arm grid.
    # --------------------------------------------------------------------
    # per_prompt[budget][arm] -> list of per-prompt accept_lengths lists
    arms = ["crossproduct", "top2gap_marginal", "top2gap_path_cond"]
    per_prompt = {b: {a: [] for a in arms} for b in budgets}

    for b in budgets:
        for p in prompts:
            cp = llm.generate_tree(
                p, drafter, block_size=bs, tree_width=args.width, budget=b,
                algo="crossproduct", algo_kwargs={}, target_layer_ids=tli,
                sampling_params=sp, return_stats=True)["accept_lengths"]
            tg = llm.generate_tree(
                p, drafter, block_size=bs, tree_width=args.width, budget=b,
                algo="top2gap_fanout", algo_kwargs={"beta": args.beta, "g_0": args.g0},
                target_layer_ids=tli, sampling_params=sp, return_stats=True)["accept_lengths"]
            pc = generate_tree_path_conditional(
                llm, drafter, p, block_size=bs, tree_width=args.width, budget=b,
                target_layer_ids=tli, sp=sp, beta=args.beta, g_0=args.g0,
                identity_knob=(args.pathcond_caps == "crossproduct"),
                refresh_enabled=True, refresh_cap=args.refresh_cap,
            )["accept_lengths"]
            per_prompt[b]["crossproduct"].append(cp)
            per_prompt[b]["top2gap_marginal"].append(tg)
            per_prompt[b]["top2gap_path_cond"].append(pc)

    # ---- report table ----
    print("== results: budget × arm ==")
    hdr = (f"{'budget':>7}{'arm':>22}{'tau(acc_len)':>14}{'tpf':>9}"
           f"{'per-prompt std':>16}")
    print(hdr); print("-" * len(hdr))
    pretty = {"crossproduct": "crossproduct",
              "top2gap_marginal": "top2gap-marginal",
              "top2gap_path_cond": "top2gap-path-cond"}
    for b in budgets:
        for a in arms:
            flat = [x for pp in per_prompt[b][a] for x in pp]
            tau = _tau(flat)
            rounds = sum(len(pp) for pp in per_prompt[b][a])
            tpf = _tpf(flat, rounds)
            per_prompt_taus = [_tau(pp) for pp in per_prompt[b][a] if pp]
            std = statistics.pstdev(per_prompt_taus) if len(per_prompt_taus) > 1 else 0.0
            print(f"{b:>7}{pretty[a]:>22}{tau:>14.3f}{tpf:>9.3f}{std:>16.3f}")
        print()

    # ---- paired deltas ----
    print("== paired (same-prompt) deltas ==")
    print(f"{'budget':>7}{'(c)-(a) cross':>16}{'(c)-(b) marg':>16}")
    print("-" * 39)
    deltas_vs_cross = {}
    deltas_vs_marg = {}
    for b in budgets:
        dca = _paired_delta(per_prompt[b]["top2gap_path_cond"], per_prompt[b]["crossproduct"])
        dcb = _paired_delta(per_prompt[b]["top2gap_path_cond"], per_prompt[b]["top2gap_marginal"])
        deltas_vs_cross[b] = dca
        deltas_vs_marg[b] = dcb
        print(f"{b:>7}{dca:>16.4f}{dcb:>16.4f}")
    print()

    # --------------------------------------------------------------------
    # VERDICT. GO iff at B=255 (N>=20) path-cond accept_len >= crossproduct +8%
    # AND >= marginal-top2gap +5% AND the win (c)-(a) grows with budget.
    # --------------------------------------------------------------------
    B = 255 if 255 in budgets else max(budgets)
    def _flat_tau(b, a):
        flat = [x for pp in per_prompt[b][a] for x in pp]
        return _tau(flat)
    pc_tau = _flat_tau(B, "top2gap_path_cond")
    cross_tau = _flat_tau(B, "crossproduct")
    marg_tau = _flat_tau(B, "top2gap_marginal")
    cond_vs_cross = (pc_tau / cross_tau - 1.0) if cross_tau else 0.0
    cond_vs_marg = (pc_tau / marg_tau - 1.0) if marg_tau else 0.0
    sorted_b = sorted(budgets)
    win_grows = all(
        deltas_vs_cross[sorted_b[i]] <= deltas_vs_cross[sorted_b[i + 1]] + 1e-9
        for i in range(len(sorted_b) - 1)
    ) and (deltas_vs_cross[sorted_b[-1]] > deltas_vs_cross[sorted_b[0]])
    enough_n = args.samples >= 20

    c_cross = cond_vs_cross >= 0.08
    c_marg = cond_vs_marg >= 0.05
    go = c_cross and c_marg and win_grows and enough_n

    print("== VERDICT ==")
    print(f"  B={B}  path-cond tau={pc_tau:.3f}  cross tau={cross_tau:.3f}  "
          f"marg tau={marg_tau:.3f}")
    print(f"  cond vs cross = {cond_vs_cross*100:+.2f}%  (need >= +8%)  -> {c_cross}")
    print(f"  cond vs marg  = {cond_vs_marg*100:+.2f}%  (need >= +5%)  -> {c_marg}")
    print(f"  win grows with budget (c)-(a) = {win_grows}  "
          f"[deltas {[round(deltas_vs_cross[b],4) for b in sorted_b]}]")
    print(f"  N >= 20 = {enough_n}  (samples={args.samples})")
    print(f"\n  >>> {'GO' if go else 'NO-GO'} <<<")
    if not enough_n:
        print("  (note: N<20 — smoke run; verdict is indicative only, rerun with --samples 20)")


if __name__ == "__main__":
    main()
