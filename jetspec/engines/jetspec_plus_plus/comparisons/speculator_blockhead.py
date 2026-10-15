"""Synchronous single-GPU speculator backed by one eager linear block head."""

from __future__ import annotations

import torch
from jetspec.engines.jetspec_plus_plus.run_config import RunConfig

from jetspec.engines.jetspec_plus_plus.serving.sequence import Sequence
from jetspec.engines.jetspec_plus_plus.serving.speculate_types import SpeculateResult
from jetspec.engines.jetspec_plus_plus.serving.speculate_types import VerifyResult
from jetspec.engines.jetspec_plus_plus.serving.speculate_types import SpeculatorBase
from jetspec.engines.jetspec_plus_plus.jetspec.blockhead_serve import (
    blockhead_batch_requested,
    blockhead_graph_requested,
    configure_blockhead_tf32,
    load_blockhead,
    pack_target_taps,
    record_blockhead_runtime,
)


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
        run_config=None,
    ):
        super().__init__(lookahead, device)
        self.run_config = run_config if run_config is not None else RunConfig.from_environment({})
        graph_requested = blockhead_graph_requested(run_config=self.run_config)
        self.batched = blockhead_batch_requested(run_config=self.run_config) and max_num_seqs > 1
        self.check_equivalence = self.batched and self.run_config.engine.engine_settings.blockhead_equiv == "1"
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
            run_config=run_config)
            self.tf32_state = configure_blockhead_tf32(run_config=run_config)
            record_blockhead_runtime(self.tf32_state, graph=graph_requested, run_config=run_config)
            if loaded.lookahead != lookahead:
                raise ValueError(
                    "block-head lookahead mismatch: "
                    f"engine={lookahead} checkpoint={loaded.lookahead}"
                )
            if graph_requested:
                from jetspec.engines.jetspec_plus_plus.comparisons.blockhead_graph import StaticBlockheadGraph
                from jetspec.engines.jetspec_plus_plus.serving.cache_capacity import capacity_for
                from jetspec.engines.jetspec_plus_plus.jetspec.blockhead_serve import _run_manifest

                capacity = capacity_for(max_context or 768)
                if self.batched:
                    from jetspec.engines.jetspec_plus_plus.comparisons.blockhead_batch_graph import BatchedBlockheadGraph
                    drafter = BatchedBlockheadGraph(
                        loaded.drafter, capacity=capacity, max_num_seqs=max_num_seqs)
                else:
                    drafter = StaticBlockheadGraph(loaded.drafter.drafter, capacity=capacity,
                                                   run_config=self.run_config)
                _run_manifest().record_value("blockhead_graph_cache_capacity", capacity, run_config=self.run_config)
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

    def _prepare_sequence(self, seq: Sequence) -> None:
        """Subclasses may bind request-specific proposal inputs."""

    def speculate(
        self, seqs: list[Sequence], verify_result: VerifyResult
    ) -> SpeculateResult:
        if verify_result.eagle_acts is not None:
            raise ValueError("block-head speculation does not support Eagle")
        if not seqs:
            raise ValueError("block-head serving needs an active sequence")
        if len(seqs) > self.max_num_seqs:
            raise ValueError("active block-head sequences exceed configured batch")
        speculations = []
        logits = []
        anatomies = []
        linear = getattr(self.drafter, "drafter", None)
        forward = getattr(linear, "_fwd", None)
        if forward is None:
            forward = getattr(getattr(linear, "drafter", None), "_fwd", None)
        batch_proposal = None
        if self.batched:
            if self.check_equivalence:
                self._sequence_caches = {seq.seq_id: self._sequence_caches[seq.seq_id]
                                         for seq in seqs if seq.seq_id in self._sequence_caches}
            contexts = []
            taps = []
            for seq in seqs:
                self._prepare_sequence(seq)
                if seq.recovery_token_id is None:
                    raise ValueError("block-head recovery token is missing")
                if seq.jetspec_taps is None:
                    raise ValueError("block-head target taps are missing")
                taps.append(pack_target_taps(seq.jetspec_taps, ctx_len=seq.num_tokens))
                contexts.append(torch.tensor(
                    [seq.token_ids + [int(seq.recovery_token_id)]],
                    dtype=torch.int64, device=self.device))
            batch_proposal = self.drafter.propose_batch(
                contexts, self.lookahead, target_hidden=taps)
            if batch_proposal.tokens.shape != (len(seqs), self.lookahead):
                raise ValueError("block-head batch proposal shape mismatch")
            batch_tokens = batch_proposal.tokens.tolist()  # one device-to-host sync per round
            serial_tokens = None
            if self.check_equivalence:
                serial_tokens = []
                for seq, context, target_hidden in zip(seqs, contexts, taps):
                    if seq.jetspec_round == 0:
                        self.drafter.reset_cache()
                    else:
                        forward.cache = self._sequence_caches[seq.seq_id]
                    control = self.drafter.propose(
                        context, self.lookahead, target_hidden=target_hidden)
                    self._sequence_caches[seq.seq_id] = forward.cache
                    serial_tokens.append(control.tokens.tolist())
        elif self.max_num_seqs > 1:
            self._sequence_caches = {seq.seq_id: self._sequence_caches[seq.seq_id]
                                     for seq in seqs if seq.seq_id in self._sequence_caches}
        for row, seq in enumerate(seqs):
            if batch_proposal is None:
                self._prepare_sequence(seq)
                if seq.jetspec_round == 0:
                    self.drafter.reset_cache()
                elif self.max_num_seqs > 1 and forward is not None:
                    forward.cache = self._sequence_caches[seq.seq_id]
                if seq.recovery_token_id is None:
                    raise ValueError("block-head recovery token is missing")
                if seq.jetspec_taps is None:
                    raise ValueError("block-head target taps are missing")
                ctx_len = seq.num_tokens
                target_hidden = pack_target_taps(seq.jetspec_taps, ctx_len=ctx_len)
                context_ids = torch.tensor(
                    [seq.token_ids + [int(seq.recovery_token_id)]],
                    dtype=torch.int64, device=self.device)
                proposal = self.drafter.propose(
                    context_ids, self.lookahead, target_hidden=target_hidden)
                if self.max_num_seqs > 1 and forward is not None:
                    self._sequence_caches[seq.seq_id] = forward.cache
                if proposal.tokens.numel() != self.lookahead:
                    raise ValueError(
                        "block-head proposal width mismatch: "
                        f"expected={self.lookahead} actual={proposal.tokens.numel()}")
                draft_tokens = proposal.tokens.tolist()
                draft_logits = proposal.logits
            else:
                proposal = batch_proposal
                draft_tokens = batch_tokens[row]
                draft_logits = proposal.logits[row : row + 1]

            completion_before = seq.num_completion_tokens
            tokens = [int(seq.recovery_token_id), *draft_tokens]
            speculations.append(torch.tensor([tokens], dtype=torch.int64, device=self.device))
            logits.append(draft_logits)
            for token in tokens:
                seq.append_token(token)
            seq.num_draft_cached_tokens += self.lookahead + 1
            seq.jetspec_round += 1

            anatomy = {
                "seq_id": seq.seq_id,
                "prompt_index": seq.jetspec_prompt_index,
                "round": seq.jetspec_round,
                "head_ms": proposal.head_ms,
                "markov_ms": proposal.markov_ms,
                "completion_before": completion_before,
                "max_new_tokens": seq.max_new_tokens,
            }
            domino_ms = getattr(proposal, "domino_ms", None)
            if domino_ms is not None:
                anatomy["domino_ms"] = domino_ms
            if batch_proposal is not None:
                anatomy["batched"] = True
                anatomy["batch_draft_ms"] = proposal.head_ms + proposal.markov_ms
                if serial_tokens is not None:
                    anatomy["draft_mismatch_count"] = sum(
                        a != b for a, b in zip(draft_tokens, serial_tokens[row]))
                    anatomy["serial_draft_tokens"] = serial_tokens[row]
                    anatomy["batch_draft_tokens"] = draft_tokens
            anatomies.append(anatomy)
            if self.metrics is not None:
                record = {
                    **anatomy, **self.tf32_state,
                    "blockhead_graph_state": self.drafter.graph_state(),
                    "blockhead_graph_self_check": self.graph_self_check,
                    "step_index": len(self.metrics["target_step_times"]),
                    "verify_index": len(self.metrics["target_verify_times"]),
                }
                if self.max_num_seqs > 1:
                    record["accepted_index"] = len(
                        self.metrics["accepted_suffix_lens_with_recovery"]) + row
                self.metrics["blockhead_round_records"].append(record)
        return SpeculateResult(
            speculations=speculations[0] if len(seqs) == 1 else torch.cat(speculations),
            logits_q=logits[0] if len(seqs) == 1 else torch.cat(logits),
            blockhead_anatomy=anatomies,
        )
