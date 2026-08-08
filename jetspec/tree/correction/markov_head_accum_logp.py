"""Parent-conditioned cumulative-logprob tree for an order-1 Markov head.

Each expanded node adds the previous-token bias to its shared per-depth base
logits. A zero bias recovers the ordinary cumulative-logprob algorithm. The
Markov head is passed through algo_kwargs, keeping the tree package independent
of checkpoint-model implementations.
"""
from __future__ import annotations

import heapq

import torch

from jetspec.correction import cost_counters
from jetspec.tree._core.ancestor import (
    _build_ancestor_matrix_np,
    _build_packed_ancestor_matrix_np,
)
from jetspec.tree._core.accept import _build_child_maps_cpu
from jetspec.tree._core.base import DraftTree, TreeAlgorithm
from jetspec.tree._core.registry import register_tree_algo


@register_tree_algo("markov_head_accum_logp")
class MarkovHeadAccumLogP(TreeAlgorithm):
    """Parent-conditioned cumulative-logprob tree for draft (order-1 Markov)."""

    def __init__(self, markov_head=None, **_):
        if markov_head is not None and not hasattr(
            markov_head,
            "compute_step_bias",
        ):
            raise TypeError("markov_head must expose compute_step_bias(token_ids, hidden).")
        head_type = getattr(markov_head, "markov_head_type", "vanilla")
        if head_type != "vanilla":
            raise NotImplementedError(
                f"markov_head_accum_logp V0 supports order-1 vanilla heads only, got "
                f"markov_head_type={head_type!r} (gated/rnn carry intra-block state)."
            )
        self.markov_head = markov_head

    def build(
        self,
        root_token: int,
        draft_logits: torch.Tensor,  # (1, depth, vocab_size) base logits
        block_size: int,
        tree_width: int,
        budget: int,
        device: torch.device,
        **kwargs,
    ) -> DraftTree:
        if self.markov_head is None:
            raise ValueError(
                "markov_head_accum_logp requires markov_head via algo_kwargs"
            )
        if cost_counters.active("markov_head"):
            markov_macs = (
                self.markov_head.markov_w2.in_features
                * self.markov_head.markov_w2.out_features
            )
            cost_counters.set_analytic("markov_macs_per_apply", markov_macs)
            cost_counters.set_analytic(
                "channel_macs_per_expanded_node", markov_macs
            )
            with cost_counters.tree_build(device):
                return self._build(
                    root_token,
                    draft_logits,
                    block_size,
                    tree_width,
                    budget,
                    device,
                    **kwargs,
                )
        return self._build(
            root_token,
            draft_logits,
            block_size,
            tree_width,
            budget,
            device,
            **kwargs,
        )

    def _build(
        self,
        root_token: int,
        draft_logits: torch.Tensor,
        block_size: int,
        tree_width: int,
        budget: int,
        device: torch.device,
        **kwargs,
    ) -> DraftTree:
        depth_limit = block_size - 1
        if draft_logits.dim() != 3 or draft_logits.shape[0] != 1:
            raise ValueError(
                f"draft_logits must be (1, depth, vocab); got {tuple(draft_logits.shape)}"
            )
        if draft_logits.shape[1] != depth_limit:
            raise ValueError(
                "draft_logits depth "
                f"{draft_logits.shape[1]} != block_size-1 ({depth_limit})"
            )

        base = draft_logits.squeeze(0)
        mh = self.markov_head
        k = max(1, int(tree_width))

        # Per-build cache of the order-1 bias for the unique parent tokens in this
        # tree (cleared every round). bias(t) = markov_w2(markov_w1(t)) -> (V,).
        bias_cache: dict[int, torch.Tensor] = {}

        def _bias(token: int) -> torch.Tensor:
            b = bias_cache.get(token)
            if b is None:
                prev = torch.tensor([int(token)], device=base.device, dtype=torch.long)
                b = mh.compute_step_bias(prev, None).to(base.dtype).reshape(-1)  # (V,)
                bias_cache[token] = b
            return b

        def _child_topk(d: int, parent_token: int, width: int):
            if cost_counters.active("markov_head"):
                with cost_counters.phase("channel", base.device):
                    logits = base[d] + _bias(parent_token)  # (V,)
                cost_counters.increment("n_markov_applies")
            else:
                logits = base[d] + _bias(parent_token)  # (V,)
            top_logits, top_tokens = torch.topk(logits, width)
            # log p = logit - logsumexp; avoids materializing a full log_softmax.
            log_z = torch.logsumexp(logits.float(), dim=-1)
            top_logp = top_logits.float() - log_z
            return top_tokens.tolist(), top_logp.tolist()

        tokens_list: list[int] = [int(root_token)]
        parents_list: list[int] = [-1]
        depths_list: list[int] = [0]
        cum_lp_list: list[float] = [0.0]
        num_nodes = 1

        # Max-heap on cumulative log-prob (negate for Python's min-heap).
        counter = 0
        heap: list[tuple[float, int, int]] = [(0.0, counter, 0)]

        while heap and num_nodes < budget:
            neg_cum_lp, _, node_idx = heapq.heappop(heap)
            d = depths_list[node_idx]
            if d >= depth_limit:
                continue
            width = min(k, budget - num_nodes)
            if width <= 0:
                break
            toks, lps = _child_topk(d, tokens_list[node_idx], width)
            for j in range(len(toks)):
                child_cum_lp = -neg_cum_lp + lps[j]
                tokens_list.append(int(toks[j]))
                parents_list.append(node_idx)
                depths_list.append(d + 1)
                cum_lp_list.append(child_cum_lp)
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
            ancestor=torch.from_numpy(ancestor_np).to(device, non_blocking=True),
            ancestor_packed=torch.from_numpy(ancestor_packed_np).to(device, non_blocking=True),
        )
