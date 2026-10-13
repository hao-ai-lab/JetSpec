"""Serve a DFlash-class block drafter synchronously on the verifier's GPU.

The baseline form of speculative decoding: verify, then ONE drafter call that reads the target's
true (fresh) hidden states for every committed token and proposes 15 drafts after the known anchor
(block 16), chain, greedy. No adapter, no heads, no second process. It fills the same serial slot
as the block-head mode (speculator_blockhead.py) and reuses its round records; the drafter itself
loads through ``JetSpecDrafter.load(fresh_only=True)``, the loader the asynchronous path uses.

Default off: nothing here is imported unless ``JETSPEC_SYNC_DRAFTER`` names a checkpoint.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from time import perf_counter

import torch

SYNC_DRAFTER_ENV = "JETSPEC_SYNC_DRAFTER"
SYNC_PASSES_ENV = "JETSPEC_SYNC_PASSES"
SYNC_GRAPH_ENV = "JETSPEC_SYNC_GRAPH"
SYNC_LOOKAHEAD = 15
SYNC_SEED = 9504
# One captured graph per context capacity; the engine serves max_model_len 768 on this path.
SYNC_CONTEXT_BUCKETS = (128, 192, 256, 320, 384, 448, 512, 640, 768, 1024, 2048, 4096)
INITS = ("mask", "random_context")




def sync_drafter_dir() -> str | None:
    return os.environ.get(SYNC_DRAFTER_ENV) or None








def sync_drafter_config_overrides(checkpoint_dir: str | None, config_kwargs: dict) -> dict:
    """Apply the default-off environment selection before ``Config`` is built."""
    if checkpoint_dir is None:
        return dict(config_kwargs)
    if os.environ.get("JETSPEC_BLOCKHEAD"):
        raise ValueError(f"{SYNC_DRAFTER_ENV} and JETSPEC_BLOCKHEAD both claim the sync slot")
    if int(config_kwargs.get("num_gpus", 1)) != 1:
        raise ValueError("sync drafter serving requires one GPU")
    if int(config_kwargs.get("max_num_seqs", 1)) > 1 and sync_graph_requested():
        raise ValueError("sync drafter batching requires eager mode")
    if config_kwargs.get("draft_async", False):
        raise ValueError("sync drafter serving uses the synchronous serial slot")
    if config_kwargs.get("jetspec", False) or config_kwargs.get("use_eagle", False):
        raise ValueError("sync drafter serving cannot be combined with another draft mode")
    config = json.loads((Path(checkpoint_dir) / "config.json").read_text(encoding="utf-8"))
    if int(config.get("block_size", 0)) != SYNC_LOOKAHEAD + 1:
        raise ValueError(f"sync drafter block_size must be {SYNC_LOOKAHEAD + 1}")
    resolve_init(config.get("dflash_config"), sync_passes())
    overrides = dict(config_kwargs)
    overrides.update(speculate=True, draft_async=False, jetspec_taps=True,
                     speculate_k=sync_block_size_read() - 1)
    target_layer_ids = (config.get("dflash_config") or {}).get("target_layer_ids")
    if target_layer_ids:
        overrides["jetspec_target_layer_ids"] = tuple(target_layer_ids)
    return overrides




class SyncChainDrafter:
    """One fresh-context drafter call per round: anchor known, 15 drafts, ``passes`` forwards."""

    def __init__(self, *, head, embed_tokens, lm_head, passes, init, graph=False,
                 seed=SYNC_SEED):
        if init not in INITS:
            raise ValueError(f"init must be one of {INITS}")
        if init == "mask" and passes != 1:
            raise ValueError("a mask-initialised drafter runs one pass")
        self.eager_head = head
        self.head = head
        self.embed_tokens = embed_tokens
        self.lm_head = lm_head
        self.passes = int(passes)
        self.init = init
        self.seed = int(seed)
        self.prompt_index = 0
        self.block_size = int(head.block_size)
        self.mask_token_id = int(head.mask_token_id)
        self.dtype = next(head.parameters()).dtype
        self.device = next(head.parameters()).device
        mark = getattr(head, "adapted_position_embedding", None)
        # Row 0 marks a true-tap context row. Every row here is one, so row 1 is never read.
        self.fresh_mark = None if mark is None else mark.weight[0].detach()
        self.graph = bool(graph)
        self.self_check = None
        self.passes_run = []
        if self.graph:
            from jetspec.engines.jetspec_plus_plus.draft.graph import GraphedDraftHead; from jetspec.engines.jetspec_plus_plus.draft.graph import GraphedLMHead
            graphed = GraphedDraftHead(head, lm_head, buckets=SYNC_CONTEXT_BUCKETS)
            self.head = graphed
            self.lm_head = GraphedLMHead(graphed)





    def context(self, target_hidden: torch.Tensor) -> torch.Tensor:
        exact = target_hidden.to(device=self.device, dtype=self.dtype)
        if self.fresh_mark is None:
            return exact
        return exact + self.fresh_mark.to(dtype=exact.dtype)




SYNC_STATIC_ENV = "JETSPEC_SYNC_STATIC"
STATIC_CAPACITY = 768
STATIC_LANES = 16










class StaticSyncDrafter(SyncChainDrafter):
    """The same drafter call over a static context K/V cache, extended by the committed rows.

    A context row's keys and values depend on that row alone (its projected hidden state and its
    position), so they are written once when the row is committed and the block forward reads them:
    the per-round cost follows the accepted length, not the context length. ``graph=True`` replays
    one extension graph (up to 16 rows) and one block graph; ``graph=False`` runs the same two steps
    eagerly. A prompt's first call extends its whole context in 16-row steps, inside round one."""

    def __init__(self, *, capacity=STATIC_CAPACITY, **kwargs):
        graph = bool(kwargs.pop("graph", False))
        super().__init__(graph=False, **kwargs)
        head = self.eager_head
        if not head.resolve_causal_head("auto"):
            raise ValueError("the static sync drafter serves the causal block head")
        if getattr(head, "token_top_layer", None) is not None:
            raise ValueError("the static sync drafter refuses a token-top layer")
        if getattr(head, "has_masked_inject_layers", False):
            raise ValueError("the static sync drafter requires context injection in every layer")
        if any(layer.self_attn.sliding_window is not None for layer in head.layers):
            raise ValueError("the static sync drafter refuses sliding-window attention")
        if self.device.type != "cuda" and graph:
            raise ValueError("the graphed static sync drafter requires CUDA")
        self.graph = graph
        self.capacity = int(capacity)
        self.lanes = STATIC_LANES
        self._cache_length = 0
        self._graphs = {}
        self._state = {"head_graph": graph, "captures": 0, "query_replays": 0,
                       "extension_replays": 0, "prefill_extensions": 0}
        self._allocate()
        if graph:
            self._capture("extension", self._extension_step)
            self._capture("query", self._query_step)
            self._self_check()
        self.reset_cache()















