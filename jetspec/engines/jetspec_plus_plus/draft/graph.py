"""CUDA-graph replay for the frozen JetSpec block head."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


CONTEXT_BUCKETS = (512, 1024, 2048, 4096, 8192, 16384, 32768)


def graph_bucket_key(
    hypotheses: int,
    context_length: int,
    buckets=CONTEXT_BUCKETS,
) -> tuple[int, int]:
    """Resolve the rail-style ``(H, context capacity)`` graph key."""
    hypotheses = int(hypotheses)
    context_length = int(context_length)
    if hypotheses < 1 or context_length < 1:
        raise ValueError("hypotheses and context_length must be positive")
    for capacity in buckets:
        if context_length <= int(capacity):
            return hypotheses, int(capacity)
    raise ValueError(
        f"context length {context_length} exceeds largest bucket {max(buckets)}"
    )


@dataclass
class _GraphState:
    graph: torch.cuda.CUDAGraph
    output: torch.Tensor
    logits: torch.Tensor | None
    target_hidden: torch.Tensor
    noise_embedding: torch.Tensor
    position_ids: torch.Tensor
    attention_mask: torch.Tensor
    call_signature: tuple[tuple[str, str], ...]


class GraphedDraftHead(nn.Module):
    """Replay one fixed CUDA graph for each ``(H, context bucket)`` shape.

    Only the block head is captured. The H-to-H adapter intentionally remains
    eager because its query count follows the accepted prefix length.
    """

    def __init__(
        self,
        eager_head: nn.Module,
        projection: nn.Module | None = None,
        *,
        buckets=CONTEXT_BUCKETS,
    ) -> None:
        super().__init__()
        device = next(eager_head.parameters()).device
        if device.type != "cuda":
            raise RuntimeError("GraphedDraftHead requires a CUDA draft head")
        self.eager_head = eager_head
        self.projection = projection
        self.buckets = tuple(int(value) for value in buckets)
        self._states: dict[tuple[int, int], _GraphState] = {}
        self._pool = None
        self._current_logits: torch.Tensor | None = None

    @property
    def block_size(self) -> int:
        return int(self.eager_head.block_size)

    @property
    def mask_token_id(self) -> int:
        return int(self.eager_head.mask_token_id)

    @property
    def adapted_position_embedding(self):
        return self.eager_head.adapted_position_embedding


    def _stage_inputs(
        self,
        *,
        target_hidden: torch.Tensor,
        noise_embedding: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> tuple[tuple[int, int], dict[str, torch.Tensor]]:
        hypotheses, context_length, hidden_size = target_hidden.shape
        if noise_embedding.shape[0] != hypotheses:
            raise ValueError("target and noise batches differ")
        if noise_embedding.shape[1] != self.block_size:
            raise ValueError("noise embedding must contain one full block")
        key = graph_bucket_key(hypotheses, context_length, self.buckets)
        capacity = key[1]
        device = target_hidden.device
        if position_ids.shape != (hypotheses, context_length + self.block_size):
            raise ValueError("position ids do not match context plus block")
        if attention_mask is not None and (
            attention_mask.shape[0] not in {1, hypotheses}
            or attention_mask.shape[-1] != context_length + self.block_size
        ):
            raise ValueError("attention mask does not match context plus block")

        padded_target = target_hidden.new_zeros(
            (hypotheses, capacity, hidden_size)
        )
        padded_target[:, :context_length].copy_(target_hidden)
        padded_positions = torch.zeros(
            (hypotheses, capacity + self.block_size),
            dtype=position_ids.dtype,
            device=device,
        )
        padded_positions[:, :context_length].copy_(
            position_ids[:, :context_length]
        )
        padded_positions[:, capacity:].copy_(
            position_ids[:, context_length:]
        )
        if capacity > context_length:
            padded_positions[:, context_length:capacity] = position_ids[
                :, context_length - 1:context_length
            ]

        if attention_mask is None:
            source_mask = torch.ones(
                (hypotheses, 1, 1, context_length + self.block_size),
                dtype=torch.bool,
                device=device,
            )
        else:
            if attention_mask.dtype != torch.bool:
                raise ValueError("graphed draft attention mask must be boolean")
            source_mask = attention_mask.to(device=device)
            if source_mask.shape[0] == 1 and hypotheses != 1:
                source_mask = source_mask.expand(hypotheses, -1, -1, -1)
        padded_mask = torch.zeros(
            (*source_mask.shape[:-1], capacity + self.block_size),
            dtype=torch.bool,
            device=device,
        )
        padded_mask[..., :context_length].copy_(
            source_mask[..., :context_length]
        )
        padded_mask[..., capacity:].copy_(source_mask[..., context_length:])
        return key, {
            "target_hidden": padded_target,
            "noise_embedding": noise_embedding,
            "position_ids": padded_positions,
            "attention_mask": padded_mask,
        }

    @torch.inference_mode()
    def _capture(
        self,
        key: tuple[int, int],
        inputs: dict[str, torch.Tensor],
        call_kwargs: dict,
        signature: tuple[tuple[str, str], ...],
    ) -> _GraphState:
        static = {name: value.clone() for name, value in inputs.items()}

        def call():
            output = self.eager_head(**static, **call_kwargs)
            logits = None if self.projection is None else self.projection(output)
            return output, logits

        call()
        call()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=self._pool):
            output, logits = call()
        if self._pool is None:
            self._pool = graph.pool()
        torch.cuda.synchronize()
        state = _GraphState(
            graph=graph,
            output=output,
            logits=logits,
            target_hidden=static["target_hidden"],
            noise_embedding=static["noise_embedding"],
            position_ids=static["position_ids"],
            attention_mask=static["attention_mask"],
            call_signature=signature,
        )
        self._states[key] = state
        print(
            f"W220_DRAFT_GRAPH_CAPTURE H={key[0]} context_bucket={key[1]} "
            "block=16 adapter=eager",
            flush=True,
        )
        return state

    @torch.inference_mode()
    def forward(
        self,
        *,
        target_hidden: torch.Tensor,
        noise_embedding: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **call_kwargs,
    ) -> torch.Tensor:
        key, inputs = self._stage_inputs(
            target_hidden=target_hidden,
            noise_embedding=noise_embedding,
            position_ids=position_ids,
            attention_mask=attention_mask,
        )
        signature = tuple(
            sorted((name, repr(value)) for name, value in call_kwargs.items())
        )
        state = self._states.get(key)
        if state is None:
            state = self._capture(key, inputs, call_kwargs, signature)
        elif state.call_signature != signature:
            raise RuntimeError(
                f"draft graph {key} was captured with different call arguments"
            )
        state.target_hidden.copy_(inputs["target_hidden"])
        state.noise_embedding.copy_(inputs["noise_embedding"])
        state.position_ids.copy_(inputs["position_ids"])
        state.attention_mask.copy_(inputs["attention_mask"])
        state.graph.replay()
        self._current_logits = state.logits
        return state.output



class GraphedLMHead:
    """Projection facade backed by the enclosing block-head CUDA graph."""

    def __init__(self, graphed_head: GraphedDraftHead) -> None:
        if graphed_head.projection is None:
            raise ValueError("graphed LM head requires a captured projection")
        self.graphed_head = graphed_head

    def __call__(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.graphed_head.projected_logits(hidden_states)
