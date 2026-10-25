"""CUDA graph for the existing ragged, batched block-head proposal."""

from __future__ import annotations

from time import perf_counter

import torch

from jetspec.engines.jetspec_plus_plus.jetspec.blockhead_serve import BlockheadProposal
from jetspec.engines.jetspec_plus_plus.comparisons.blockhead_graph import validate_graph_context_length


BATCH_BUCKETS = (2, 4, 8, 16, 32)


def batch_bucket(count: int) -> int:
    for bucket in BATCH_BUCKETS:
        if count <= bucket:
            return bucket
    raise ValueError(f"block-head graph batch {count} exceeds {BATCH_BUCKETS[-1]}")


class _WidthGraph:
    """One captured batch and context-width shape."""

    def __init__(self, timed_drafter, *, capacity: int, max_num_seqs: int, pool=None):
        self.drafter = timed_drafter
        linear = timed_drafter.drafter
        self.fwd = linear._fwd
        self.head = linear.head
        self.target = linear.target
        self.capacity = int(capacity)
        self.batch = batch_bucket(max_num_seqs)
        self._pool_request = pool
        self.block_size = int(self.fwd.block_size)
        self.offset = int(getattr(linear, "draft_offset", 0))
        self.lookahead = int(linear.lookahead)
        self.device = self.fwd.device
        self.dtype = self.fwd.dtype
        if self.device.type != "cuda":
            raise ValueError("batched block-head graph requires CUDA")
        if self.head.resolve_causal_head("auto"):
            raise ValueError("batched block-head graph requires non-causal attention")
        markov = self.head.markov_head
        if markov is not None and getattr(markov, "markov_head_type", None) != "vanilla":
            raise ValueError("batched block-head graph requires order-1 Markov head")
        if self.block_size not in (4, 7, 8, 15, 16, 32):
            raise ValueError("unsupported batched block-head graph width")
        if self.offset + self.lookahead > self.block_size:
            raise ValueError("block-head graph proposal exceeds block width")

        hidden_size = int(self.head.fc.in_features)
        vocab_size = int(self.target.lm_head.weight.shape[0])
        self._hidden = torch.zeros(
            self.batch, self.capacity, hidden_size, device=self.device, dtype=self.dtype)
        self._lengths = torch.ones(self.batch, device=self.device, dtype=torch.long)
        self._anchors = torch.zeros(self.batch, device=self.device, dtype=torch.long)
        self._context_positions = torch.arange(self.capacity, device=self.device)
        self._block_positions = torch.arange(self.block_size, device=self.device)
        self._base_logits = torch.empty(
            self.batch, self.lookahead, vocab_size, device=self.device,
            dtype=self.target.lm_head.weight.dtype)
        self._tokens = torch.empty(
            self.batch, self.lookahead, device=self.device, dtype=torch.long)
        self._corrected = torch.empty_like(self._base_logits)
        self._state = {
            "head_graph": True, "captures": 0, "query_replays": 0,
            "extension_replays": 0, "prefill_extensions": 0,
        }
        self._capture()
        self._self_check()

    def _head_step(self) -> None:
        noise_ids = torch.cat((
            self._anchors[:, None],
            torch.full((self.batch, self.block_size - 1),
                       self.fwd.mask_token_id, device=self.device, dtype=torch.long),
        ), dim=1)
        noise_embedding = self.target.model.embed_tokens(noise_ids).to(dtype=self.dtype)
        valid = self._context_positions[None, :] < self._lengths[:, None]
        key_valid = torch.cat((valid, torch.ones(
            self.batch, self.block_size, device=self.device, dtype=torch.bool)), dim=1)
        attention_mask = key_valid[:, None, None, :].expand(
            -1, 1, self.block_size, -1)
        position_ids = torch.cat((
            self._context_positions[None, :].expand(self.batch, -1),
            self._lengths[:, None] + self._block_positions[None, :],
        ), dim=1)
        hidden = self.head(
            target_hidden=self._hidden,
            noise_embedding=noise_embedding,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=None,
            use_cache=False,
            is_causal=False,
        )
        proposal_hidden = hidden[:, self.offset:self.offset + self.lookahead, :]
        lm_head = self.target.lm_head
        lm_input = proposal_hidden.to(dtype=lm_head.weight.dtype)
        self._base_logits.copy_(lm_head(lm_input))

    def _sample_step(self) -> None:
        sampled, corrected = self.head.sample_draft_tokens(
            self._base_logits, first_prev_token_ids=self._anchors,
            temperature=0.0)
        self._tokens.copy_(sampled)
        self._corrected.copy_(corrected)

    @torch.inference_mode()
    def _capture(self) -> None:
        started = perf_counter()
        stream = torch.cuda.Stream(device=self.device)
        stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(stream):
            for _ in range(3):
                self._head_step()
                self._sample_step()
        torch.cuda.current_stream(self.device).wait_stream(stream)
        self._head_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._head_graph, pool=self._pool_request):
            self._head_step()
        self._sample_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._sample_graph, pool=self._head_graph.pool()):
            self._sample_step()
        self._state["captures"] = 2
        self.pool = self._head_graph.pool()
        torch.cuda.synchronize(self.device)
        print(f"BLOCKHEAD_BATCH_GRAPH_CAPTURE bucket={self.batch} "
              f"width={self.capacity} captures=2 "
              f"ms={(perf_counter() - started) * 1000:.3f}", flush=True)

    @torch.inference_mode()
    def _self_check(self) -> None:
        self._head_step()
        self._sample_step()
        expected_tokens = self._tokens.clone()
        expected_logits = self._corrected.clone()
        self._head_graph.replay()
        self._sample_graph.replay()
        if not torch.equal(expected_tokens, self._tokens):
            raise RuntimeError("batched block-head graph token self-check failed")
        if not torch.equal(expected_logits, self._corrected):
            raise RuntimeError("batched block-head graph logits self-check failed")
        self.self_check = {"query_bitwise": True, "markov_bitwise": True}

    def graph_state(self) -> dict:
        return dict(self._state)

    @torch.inference_mode()
    def propose_batch(
        self, context_ids: list[torch.Tensor], k: int, *,
        target_hidden: list[torch.Tensor],
    ) -> BlockheadProposal:
        count = len(context_ids)
        if not 1 <= count <= self.batch or count != len(target_hidden):
            raise ValueError("batched block-head graph inputs differ or exceed bucket")
        if k != self.lookahead:
            raise ValueError("batched block-head graph lookahead changed")
        widths = [int(row.shape[1]) for row in target_hidden]
        for context, width in zip(context_ids, widths):
            validate_graph_context_length(width, capacity=self.capacity)
            if context.shape[1] != width + 1:
                raise ValueError("block-head context and target taps differ")
        for index, (row, width) in enumerate(zip(target_hidden, widths)):
            self._hidden[index, :width].copy_(row[0].to(device=self.device, dtype=self.dtype))
        self._lengths.copy_(torch.tensor(
            widths + [1] * (self.batch - count), device=self.device))
        self._anchors[:count].copy_(torch.stack(
            [row[0, -1] for row in context_ids]).to(self.device))

        start = torch.cuda.Event(enable_timing=True)
        split = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        self._head_graph.replay()
        split.record()
        self._sample_graph.replay()
        end.record()
        end.synchronize()
        self._state["query_replays"] += 1
        tokens = self._tokens[:count].clone()
        logits = self._corrected[:count].clone()
        return BlockheadProposal(
            tokens, logits,
            float(start.elapsed_time(split)), float(split.elapsed_time(end)))


def width_buckets(capacity: int) -> tuple[int, ...]:
    if capacity < 1:
        raise ValueError("block-head graph capacity must be positive")
    buckets = list(range(128, capacity + 1, 128))
    if not buckets or buckets[-1] != capacity:
        buckets.append(capacity)
    return tuple(buckets)


class BatchedBlockheadGraph:
    """Prewarm every admitted width for one fixed padded batch bucket."""

    def __init__(self, timed_drafter, *, capacity: int, max_num_seqs: int):
        self.drafter = timed_drafter
        self.capacity = int(capacity)
        self.batch = batch_bucket(max_num_seqs)
        self.widths = width_buckets(self.capacity)
        self._graphs = {}
        pool = None
        for width in self.widths:
            graph = _WidthGraph(timed_drafter, capacity=width,
                                max_num_seqs=max_num_seqs, pool=pool)
            self._graphs[width] = graph
            pool = graph.pool
        self.self_check = {
            "width_buckets": list(self.widths),
            "query_bitwise": all(g.self_check["query_bitwise"] for g in self._graphs.values()),
            "markov_bitwise": all(g.self_check["markov_bitwise"] for g in self._graphs.values()),
        }
        self._warmup_captures = self.graph_state()["captures"]
        self._steady_printed = False
        print(f"BLOCKHEAD_BATCH_GRAPH_WARMUP batch={self.batch} "
              f"widths={self.widths} captures={self._warmup_captures}", flush=True)

    def graph_state(self) -> dict:
        states = [graph.graph_state() for graph in self._graphs.values()]
        return {
            "head_graph": True,
            "captures": sum(state["captures"] for state in states),
            "query_replays": sum(state["query_replays"] for state in states),
            "extension_replays": 0,
            "prefill_extensions": 0,
        }

    def reset_cache(self) -> None:
        self.drafter.reset_cache()

    def propose(self, context_ids, k, *, target_hidden):
        return self.drafter.propose(context_ids, k, target_hidden=target_hidden)

    def propose_batch(self, context_ids, k, *, target_hidden) -> BlockheadProposal:
        if not target_hidden:
            raise ValueError("batched block-head graph inputs are empty")
        live_width = max(int(row.shape[1]) for row in target_hidden)
        width = next((bucket for bucket in self.widths if live_width <= bucket), None)
        if width is None:
            raise ValueError(f"block-head graph context {live_width} exceeds {self.capacity}")
        proposal = self._graphs[width].propose_batch(
            context_ids, k, target_hidden=target_hidden)
        captures_after_warmup = self.graph_state()["captures"] - self._warmup_captures
        if captures_after_warmup:
            raise RuntimeError("block-head graph captured after warm-up")
        if not self._steady_printed:
            print(f"BLOCKHEAD_BATCH_GRAPH_STEADY batch={self.batch} "
                  f"width={width} captures_after_warmup={captures_after_warmup}", flush=True)
            self._steady_printed = True
        return proposal
