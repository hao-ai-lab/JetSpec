"""Tensor-parallel all-reduce for the target: a one-shot NVLink kernel for decode-sized tensors.

`dist.all_reduce` on NCCL costs ~12 us per call for a [16, 4096] bf16 tensor inside a CUDA graph on
two B200s; a verify issues 73 of them (embedding + o_proj and down_proj per layer). FlashInfer's TRT-LLM
one-shot all-reduce (Lamport flags in an IPC workspace, graph-capturable, PDL launch) does the same
reduction in ~4.4 us and returns bit-identical sums (fp32 accumulation, two ranks). Tensors larger than
the workspace (prefill) fall back to NCCL.

Enabled with JETSPEC_TP_CUSTOM_ALLREDUCE=1; `init` runs on every target TP rank after the TP group
exists and before CUDA graphs are captured.
"""
from __future__ import annotations

import torch
import torch.distributed as dist

_STATE = None
MAX_TOKENS = 128


def init(group, rank: int, world_size: int, hidden_size: int, *, max_tokens: int = MAX_TOKENS) -> None:
    """Create the IPC workspace on every rank of `group` (a collective)."""
    global _STATE
    import flashinfer.comm as fc
    created = fc.trtllm_create_ipc_workspace_for_all_reduce_fusion(
        rank, world_size, max_tokens, hidden_size, group=group)
    _STATE = dict(fc=fc, group=group, rank=rank, world=world_size, hidden=hidden_size,
                  max_tokens=max_tokens, workspace=created[1], handles=created[0])
    print(f"TP_CUSTOM_ALLREDUCE rank={rank} world={world_size} hidden={hidden_size} "
          f"max_tokens={max_tokens}", flush=True)


def enabled() -> bool:
    return _STATE is not None


def all_reduce(x: torch.Tensor, group) -> torch.Tensor:
    """Sum `x` over the TP group; returns the reduced tensor (may be a new tensor)."""
    state = _STATE
    tokens = x.numel() // x.shape[-1] if x.dim() else 0
    if (state is None or x.dtype != torch.bfloat16 or x.shape[-1] != state['hidden']
            or not 1 <= tokens <= state['max_tokens'] or dist.get_world_size(group) != state['world']):
        dist.all_reduce(x, group=group)
        return x
    flat = x.reshape(tokens, state['hidden'])
    if not flat.is_contiguous():
        flat = flat.contiguous()
    out = torch.empty_like(flat)
    fc = state['fc']
    fc.trtllm_allreduce_fusion(
        allreduce_in=flat, world_size=state['world'], world_rank=state['rank'], token_num=tokens,
        hidden_dim=state['hidden'], workspace_ptrs=state['workspace'], launch_with_pdl=True,
        trigger_completion_at_end=True, fp32_acc=True,
        pattern_code=fc.AllReduceFusionPattern.kAllReduce, use_oneshot=True,
        allreduce_out=out, residual_in=None, residual_out=None, norm_out=None, quant_out=None,
        scale_out=None, rms_gamma=None, rms_eps=None, scale_factor=None, layout_code=None)
    return out.view(x.shape)
