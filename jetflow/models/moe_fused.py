"""Optional fused MoE replacement for HF Qwen3-MoE sparse blocks.

The patch is intentionally isolated from JetFlow: dense Qwen3 models have no
matching blocks and return 0, while Qwen3-MoE targets can opt into grouped GEMM
expert execution for lower decode/tree-verify launch overhead.
"""
from __future__ import annotations

import types

import torch
import torch.nn.functional as F
from torch import nn


def _fused_forward(self, hidden_states: torch.Tensor):
    """Grouped-mm equivalent of HF ``Qwen3MoeSparseMoeBlock.forward``."""
    bsz, seq_len, hidden = hidden_states.shape
    x = hidden_states.view(-1, hidden)
    n_tokens = x.shape[0]
    top_k = self.top_k
    num_experts = self.num_experts

    router_logits = self.gate(x)
    routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
    routing_weights, expert_indices = torch.topk(routing_weights, top_k, dim=-1)
    if self.norm_topk_prob:
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
    routing_weights = routing_weights.to(x.dtype)

    expanded_x = x.repeat_interleave(top_k, dim=0)
    expanded_idx = expert_indices.flatten()
    expanded_w = routing_weights.flatten()

    sort_perm = expanded_idx.argsort()
    sorted_x = expanded_x[sort_perm]
    sorted_idx = expanded_idx[sort_perm]
    sorted_w = expanded_w[sort_perm]

    counts = torch.bincount(sorted_idx, minlength=num_experts)
    offsets = counts.cumsum(0).to(torch.int32)

    intermediate = torch._grouped_mm(sorted_x, self._w_gate_up, offs=offsets)
    gate, up = intermediate.chunk(2, dim=-1)
    intermediate = F.silu(gate) * up
    output = torch._grouped_mm(intermediate, self._w_down, offs=offsets)

    output = output * sorted_w.unsqueeze(-1)
    unsort_perm = torch.argsort(sort_perm)
    output = output[unsort_perm].view(n_tokens, top_k, hidden).sum(dim=1)
    return output.view(bsz, seq_len, hidden), router_logits


def patch_qwen3_moe_with_grouped_mm(
    model: nn.Module,
    *,
    free_original_experts: bool = True,
) -> int:
    """Patch compatible Qwen3-MoE sparse blocks in-place.

    Returns the number of patched blocks. Dense Qwen3 models return 0. If the
    model contains Qwen3-MoE blocks but the runtime lacks ``torch._grouped_mm``,
    a RuntimeError is raised so benchmark configuration errors are visible.
    """
    try:
        from transformers.models.qwen3_moe.modeling_qwen3_moe import (
            Qwen3MoeSparseMoeBlock,
        )
    except ImportError:
        return 0

    moe_blocks = [
        module for module in model.modules()
        if isinstance(module, Qwen3MoeSparseMoeBlock)
    ]
    if not moe_blocks:
        return 0
    if not hasattr(torch, "_grouped_mm"):
        raise RuntimeError(
            "torch._grouped_mm is unavailable; --fused-moe requires a PyTorch "
            "runtime with grouped GEMM support."
        )

    n_patched = 0
    for module in moe_blocks:
        if hasattr(module, "_w_gate_up") and hasattr(module, "_w_down"):
            continue

        gate_w = torch.stack(
            [expert.gate_proj.weight.t().contiguous() for expert in module.experts],
            dim=0,
        )
        up_w = torch.stack(
            [expert.up_proj.weight.t().contiguous() for expert in module.experts],
            dim=0,
        )
        down_w = torch.stack(
            [expert.down_proj.weight.t().contiguous() for expert in module.experts],
            dim=0,
        )

        module.register_buffer(
            "_w_gate_up", torch.cat([gate_w, up_w], dim=-1).contiguous(),
            persistent=False,
        )
        module.register_buffer("_w_down", down_w, persistent=False)
        module.num_experts = len(module.experts)

        if free_original_experts:
            del module.experts
        del gate_w, up_w, down_w

        module.forward = types.MethodType(_fused_forward, module)
        n_patched += 1

    return n_patched
