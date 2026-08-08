"""Path-conditioned cumulative-logprob tree for the attention layer.

The draft head supplies one path-independent hidden block. For each expanded
node, the attention layer consumes the realized ancestor-token embeddings and
produces corrected child logits. Setting attention_layer_off=True recovers the
uncorrected cumulative-logprob control on the same checkpoint.
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


@register_tree_algo("attention_layer_accum_logp")
class AttentionLayerAccumLogP(TreeAlgorithm):
    """Path-conditioned cumulative-logprob tree for the attention layer."""

    def __init__(
        self,
        attention_layer=None,
        embed_tokens=None,
        lm_head=None,
        block_positions=None,
        mask_token_id=None,
        attention_layer_off: bool = False,
        markov_head=None,
        markov_off: bool = False,
        residual_compose: bool = False,
        **_,
    ):
        if attention_layer is not None and not hasattr(attention_layer, "forward"):
            raise TypeError("attention_layer must be callable (an nn.Module).")
        fusion_mode = getattr(attention_layer, "fusion_mode", "concat")
        needs_prev_embedding = not attention_layer_off and fusion_mode != "h_only"
        self.attention_layer = attention_layer
        self.embed_tokens = embed_tokens
        self.lm_head = lm_head
        # block_positions: (1, block_size) absolute RoPE positions the backbone
        # forward used (position_ids[:, -block_size:]). A callable is resolved at
        # build time because the absolute position advances each decoding round.
        self._block_positions = block_positions
        self.mask_token_id = mask_token_id
        self.attention_layer_off = bool(attention_layer_off)
        self.markov_head = markov_head
        self.markov_off = bool(markov_off)
        self.residual_compose = bool(residual_compose)
        if (
            self.markov_head is not None
            and not self.markov_off
            and not hasattr(self.markov_head, "compute_step_bias")
        ):
            raise TypeError(
                "markov_head must expose compute_step_bias(token_ids, hidden)."
            )
        self.fusion_mode = fusion_mode
        self.needs_prev_embedding = needs_prev_embedding

    @torch.inference_mode()
    def build(
        self,
        root_token: int,
        draft_logits: torch.Tensor,  # (1, block_size, H) base_hidden (NOT logits)
        block_size: int,
        tree_width: int,
        budget: int,
        device: torch.device,
        **kwargs,
    ) -> DraftTree:
        if self.lm_head is None:
            raise ValueError(
                "attention_layer_accum_logp requires lm_head via algo_kwargs"
            )
        if not self.attention_layer_off and self.attention_layer is None:
            raise ValueError(
                "attention_layer_accum_logp requires attention_layer via "
                "algo_kwargs unless attention_layer_off=True"
            )
        if self.needs_prev_embedding and self.embed_tokens is None:
            raise ValueError(
                "attention_layer_accum_logp requires embed_tokens via algo_kwargs"
            )
        if cost_counters.active("attention_layer"):
            if not self.attention_layer_off:
                blocks = [self.attention_layer, *self.attention_layer.extra_layers]
                attention_layer_macs = 0
                for block in blocks:
                    attn = block.self_attn
                    B = int(block.block_size)
                    fusion = getattr(block, "fusion", None)
                    fusion_macs = (
                        fusion.in_features * fusion.out_features
                        if fusion is not None
                        else 0
                    )
                    attention_layer_macs += B * (
                        fusion_macs
                        + attn.q_proj.in_features * attn.q_proj.out_features
                        + attn.k_proj.in_features * attn.k_proj.out_features
                        + attn.v_proj.in_features * attn.v_proj.out_features
                        + attn.o_proj.in_features * attn.o_proj.out_features
                    )
                    # Full BxB eager QK and attention-times-V matmuls.
                    attention_layer_macs += 2 * B * B * attn.q_proj.out_features
                lm_head_macs = self.lm_head.in_features * self.lm_head.out_features
                cost_counters.set_analytic(
                    "attention_layer_macs_per_call",
                    attention_layer_macs,
                )
                cost_counters.set_analytic(
                    "lm_head_macs_per_projection", lm_head_macs
                )
                cost_counters.set_analytic(
                    "channel_macs_per_expanded_node",
                    attention_layer_macs + lm_head_macs,
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
                f"base_hidden must be (1, block_size, H); got {tuple(draft_logits.shape)}"
            )
        if draft_logits.shape[1] != block_size:
            raise ValueError(
                f"base_hidden slots {draft_logits.shape[1]} != block_size ({block_size}); "
                "the attention-layer drafter must return the full post-norm block hidden."
            )

        base_hidden = draft_logits  # (1, block_size, H) on the head's device
        base_logits = (
            self.lm_head(base_hidden[:, 1:, :]).squeeze(0)
            if self.residual_compose
            else None
        )
        if self.residual_compose and self.attention_layer_off:
            if self.markov_head is not None and not self.markov_off:
                from jetspec.tree.correction.markov_head_accum_logp import MarkovHeadAccumLogP

                return MarkovHeadAccumLogP(markov_head=self.markov_head).build(
                    root_token,
                    base_logits.unsqueeze(0),
                    block_size,
                    tree_width,
                    budget,
                    device,
                )
            from jetspec.tree.baselines.accum_logp import AccumLogP

            return AccumLogP().build(
                root_token,
                base_logits.unsqueeze(0),
                block_size,
                tree_width,
                budget,
                device,
            )
        hidden_device = base_hidden.device
        hidden_dtype = base_hidden.dtype
        k = max(1, int(tree_width))
        mask_id = self.mask_token_id
        if self.needs_prev_embedding and mask_id is None:
            raise ValueError(
                "attention_layer_accum_logp requires mask_token_id when "
                "attention_layer is on."
            )

        # Use the exact absolute RoPE positions from the backbone forward. Even a
        # mathematically equivalent shifted RoPE frame can round differently in
        # low precision, so the live attention-layer path refuses an approximation.
        block_positions = (
            self._block_positions()
            if callable(self._block_positions)
            else self._block_positions
        )
        if block_positions is None and not self.attention_layer_off:
            raise ValueError(
                "attention_layer_accum_logp requires the backbone's absolute block_positions "
                "when attention_layer is on."
            )
        if block_positions is None:
            block_positions = torch.arange(block_size, device=hidden_device).unsqueeze(0)
        else:
            block_positions = block_positions.to(device=hidden_device, dtype=torch.long)

        # Per-build cache of the per-node top-k, keyed by the full realized path;
        # the cache is cleared every verification round.
        path_cache: dict[tuple[int, ...], tuple[list[int], list[float]]] = {}

        def _child_topk(path: tuple[int, ...], width: int):
            """Top-k children of a node with the supplied realized ancestor chain.

            `path = (t0, t1, ..., td)` with t0 = root/anchor and td = the node's own
            token; the node is at tree-depth d = len(path) - 1, so its children read
            the attention-layer-corrected slot d+1.
            """
            cached = path_cache.get(path)
            if cached is not None:
                toks, lps = cached
                return toks[:width], lps[:width]

            d = len(path) - 1
            slot = d + 1  # backbone slot that predicts the child token at offset d+1

            if self.attention_layer_off:
                # Identity: the bare per-depth base logits (the uncorrected control).
                corrected_slot = base_hidden[:, slot, :]
            else:
                if self.fusion_mode == "h_only":
                    # Preserve the four-argument layer call without gathering or
                    # embedding path tokens; h_only ignores this placeholder.
                    prev_emb = base_hidden
                else:
                    # prev_ids[o] = path[clamp(o-1, 0)] for o<=d+1 else mask.
                    # Block-local causality makes slots > d+1 irrelevant here.
                    prev_ids = []
                    for o in range(block_size):
                        off = o - 1 if o >= 1 else 0
                        prev_ids.append(path[off] if off <= d else int(mask_id))
                    prev_id_t = torch.tensor(
                        [prev_ids], dtype=torch.long, device=hidden_device
                    )
                    prev_emb = self.embed_tokens(prev_id_t).detach().to(dtype=hidden_dtype)
                keep = torch.ones((1, 1), dtype=hidden_dtype, device=hidden_device)
                if cost_counters.active("attention_layer"):
                    with cost_counters.phase("channel", hidden_device):
                        corrected = self.attention_layer(
                            base_hidden,
                            prev_emb,
                            block_positions,
                            keep,
                        )  # (1, block_size, H)
                    cost_counters.increment("n_attention_layer_calls")
                else:
                    corrected = self.attention_layer(
                        base_hidden,
                        prev_emb,
                        block_positions,
                        keep,
                    )  # (1, block_size, H)
                corrected_slot = corrected[:, slot, :]

            if self.residual_compose:
                if self.attention_layer_off:
                    logits = base_logits[d]
                else:
                    delta_slot = corrected_slot - base_hidden[:, slot, :]
                    logits = base_logits[d] + self.lm_head(
                        delta_slot
                    ).reshape(-1)
            elif cost_counters.active("attention_layer") and not self.attention_layer_off:
                with cost_counters.phase("lm_head_reprojection", hidden_device):
                    logits = self.lm_head(corrected_slot).reshape(-1)  # (V,)
                cost_counters.increment("n_lm_head_projections")
            else:
                logits = self.lm_head(corrected_slot).reshape(-1)  # (V,)
            if self.markov_head is not None and not self.markov_off:
                parent = torch.tensor(
                    [path[-1]], dtype=torch.long, device=hidden_device
                )
                markov_bias = self.markov_head.compute_step_bias(
                    parent, None
                ).to(dtype=logits.dtype).reshape(-1)
                logits = logits + markov_bias
            full_k = max(k, width)
            top_logits, top_tokens = torch.topk(logits, full_k)
            # log p = logit - logsumexp (the existing cumulative-score accounting).
            log_z = torch.logsumexp(logits.float(), dim=-1)
            top_logp = top_logits.float() - log_z
            toks = top_tokens.tolist()
            lps = top_logp.tolist()
            path_cache[path] = (toks, lps)
            return toks[:width], lps[:width]

        tokens_list: list[int] = [int(root_token)]
        parents_list: list[int] = [-1]
        depths_list: list[int] = [0]
        cum_lp_list: list[float] = [0.0]
        # Realized ancestor chain per node (path[node_idx] = (t0, ..., t_depth)).
        paths_list: list[tuple[int, ...]] = [(int(root_token),)]
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
            toks, lps = _child_topk(paths_list[node_idx], width)
            for j in range(len(toks)):
                child_cum_lp = -neg_cum_lp + lps[j]
                child_token = int(toks[j])
                tokens_list.append(child_token)
                parents_list.append(node_idx)
                depths_list.append(d + 1)
                cum_lp_list.append(child_cum_lp)
                paths_list.append(paths_list[node_idx] + (child_token,))
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
