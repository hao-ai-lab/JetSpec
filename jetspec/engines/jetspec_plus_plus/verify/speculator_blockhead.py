"""Synchronous single-GPU speculator backed by one eager linear block head."""

from __future__ import annotations

import os
import torch

from jetspec.engines.jetspec_plus_plus.serving.sequence import Sequence
from jetspec.engines.jetspec_plus_plus.serving.helpers.speculate_types import SpeculateResult; from jetspec.engines.jetspec_plus_plus.serving.helpers.speculate_types import VerifyResult; from jetspec.engines.jetspec_plus_plus.serving.helpers.speculate_types import SpeculatorBase
from jetspec.engines.jetspec_plus_plus.draft.blockhead_serve import blockhead_batch_requested; from jetspec.engines.jetspec_plus_plus.draft.blockhead_serve import blockhead_graph_requested; from jetspec.engines.jetspec_plus_plus.draft.blockhead_serve import configure_blockhead_tf32; from jetspec.engines.jetspec_plus_plus.draft.blockhead_serve import load_blockhead; from jetspec.engines.jetspec_plus_plus.draft.blockhead_serve import pack_target_taps; from jetspec.engines.jetspec_plus_plus.draft.blockhead_serve import record_blockhead_runtime


class SpeculatorBlockhead(SpeculatorBase):
    def __init__(
        self,
        lookahead: int,
        device: torch.device,
        *,
        checkpoint_dir: str | None = None,
        target=None,
        drafter=None,
        metrics: dict | None = None,
        max_context: int | None = None,
        max_num_seqs: int = 1,
    ):
        super().__init__(lookahead, device)
        graph_requested = blockhead_graph_requested()
        self.batched = blockhead_batch_requested() and max_num_seqs > 1
        self.check_equivalence = self.batched and os.environ.get("JETSPEC_BLOCKHEAD_EQUIV", "0") == "1"
        if graph_requested and max_num_seqs > 1 and not self.batched:
            raise ValueError("batched block-head graph requires JETSPEC_BLOCKHEAD_BATCH=1")
        if drafter is None:
            if checkpoint_dir is None or target is None:
                raise ValueError("block-head checkpoint and target are required")
            loaded = load_blockhead(
                checkpoint_dir,
                target=target,
                device=device,
                max_num_seqs=max_num_seqs,
            )
            self.tf32_state = configure_blockhead_tf32()
            record_blockhead_runtime(self.tf32_state, graph=graph_requested)
            if loaded.lookahead != lookahead:
                raise ValueError(
                    "block-head lookahead mismatch: "
                    f"engine={lookahead} checkpoint={loaded.lookahead}"
                )
            if graph_requested:
                from jetspec.engines.jetspec_plus_plus.draft.blockhead_graph import StaticBlockheadGraph

                capacity = capacity_for(max_context or 768)
                if self.batched:
                    from jetspec.engines.jetspec_plus_plus.draft.blockhead_batch_graph import BatchedBlockheadGraph
                    drafter = BatchedBlockheadGraph(
                        loaded.drafter, capacity=capacity, max_num_seqs=max_num_seqs)
                else:
                    drafter = StaticBlockheadGraph(loaded.drafter.drafter, capacity=capacity)
                _run_manifest().record_value("blockhead_graph_cache_capacity", capacity)
            else:
                drafter = loaded.drafter
            self.target_weights = loaded.target_weights
            self.target_weights_path = loaded.target_weights_path
        else:
            if graph_requested:
                raise ValueError(
                    "injected block-head drafters cannot enable CUDA graph mode"
                )
            self.tf32_state = {
                "matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
                "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
            }
            self.target_weights = None
            self.target_weights_path = None
        self.drafter = drafter
        self.max_num_seqs = max_num_seqs
        self._sequence_caches = {}
        self.metrics = metrics
        self.blockhead = True
        self.graph_self_check = getattr(self.drafter, "self_check", None)
        if self.metrics is not None:
            for key in (
                "blockhead_graph_captures",
                "blockhead_graph_query_replays",
                "blockhead_graph_extension_replays",
                "blockhead_graph_prefill_extensions",
            ):
                self.metrics.setdefault(key, 0)

    def prefill(
        self, seqs: list[Sequence], verify_result: VerifyResult
    ) -> SpeculateResult:
        if verify_result.eagle_acts is not None:
            raise ValueError("block-head speculation does not support Eagle")
        for seq in seqs:
            seq.jetspec_round = 0
            self._sequence_caches.pop(seq.seq_id, None)
        return SpeculateResult([], [])


