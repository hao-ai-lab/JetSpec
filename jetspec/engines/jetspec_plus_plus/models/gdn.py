"""Gated DeltaNet mixer, retaining the original recurrent arithmetic."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from jetspec.engines.jetspec_plus_plus.models.config import setting
from jetspec.engines.jetspec_plus_plus.models.forward_context import get_forward_context
from jetspec.engines.jetspec_plus_plus.models.target_layers import HybridRMSNorm
from jetspec.engines.jetspec_plus_plus.models.target_linear import TargetLinear


def hybrid_linear(config, input_size, output_size, *, bias=False, run_config=None):
    storage = (setting(run_config, 'engine', 'engine_settings.hybrid_fp32_linear_bf16_storage', '0') == '1'
               and getattr(config, 'torch_dtype', torch.get_default_dtype()) == torch.float32)
    return TargetLinear(input_size, output_size, bias=bias, fp32_bf16_storage=storage)


def _torch_delta_rule(q, k, v, g, beta, initial):
    """Small-shape CPU reference with the FLA recurrent state convention."""
    q = q.float()
    k = k.float()
    q = q * torch.rsqrt(q.square().sum(-1, keepdim=True) + 1e-6) * (q.shape[-1] ** -0.5)
    k = k * torch.rsqrt(k.square().sum(-1, keepdim=True) + 1e-6)
    v, g, beta = v.float(), g.float(), beta.float()
    state = initial.float()
    outputs = []
    for step in range(q.shape[1]):
        state = state * g[:, step].exp()[..., None, None]
        delta = (v[:, step] - (state * k[:, step, :, :, None]).sum(-2)) * beta[:, step, :, None]
        state = state + k[:, step, :, :, None] * delta[:, :, None, :]
        outputs.append((state * q[:, step, :, :, None]).sum(-2))
    return torch.stack(outputs, dim=1).to(v.dtype), state


class HybridGDN(nn.Module):
    def __init__(self, config, max_verify_width: int, *, run_config=None):
        super().__init__()
        self.run_config = run_config
        hidden = config.hidden_size
        self.key_heads = config.linear_num_key_heads
        self.value_heads = config.linear_num_value_heads
        self.key_dim = config.linear_key_head_dim
        self.value_dim = config.linear_value_head_dim
        self.conv_kernel = config.linear_conv_kernel_dim
        self.fast_snapshot = setting(self.run_config, 'engine', 'engine_settings.hybrid_fast_snapshot', '0') == "1"
        key_width = self.key_heads * self.key_dim
        value_width = self.value_heads * self.value_dim
        conv_width = 2 * key_width + value_width
        self.key_width = key_width
        self.value_width = value_width
        self.in_proj_qkv = hybrid_linear(config, hidden, conv_width, run_config=run_config)
        self.in_proj_z = hybrid_linear(config, hidden, value_width, run_config=run_config)
        self.in_proj_b = hybrid_linear(config, hidden, self.value_heads, run_config=run_config)
        self.in_proj_a = hybrid_linear(config, hidden, self.value_heads, run_config=run_config)
        self.conv1d = nn.Conv1d(conv_width, conv_width, self.conv_kernel,
                                groups=conv_width, padding=self.conv_kernel - 1, bias=False)
        self.dt_bias = nn.Parameter(torch.ones(self.value_heads))
        self.A_log = nn.Parameter(torch.zeros(self.value_heads))
        self.out_proj = hybrid_linear(config, value_width, hidden, run_config=run_config)
        self.eps = config.rms_norm_eps
        try:
            from fla.modules import FusedRMSNormGated
            from fla.ops.gated_delta_rule import (chunk_gated_delta_rule,
                                                   fused_recurrent_gated_delta_rule)
        except ImportError:
            self._chunk = self._recurrent = None
            self.norm = HybridRMSNorm(self.value_dim, self.eps)
            with torch.no_grad():
                self.norm.weight.fill_(1)
        else:
            self._chunk = chunk_gated_delta_rule
            self._recurrent = fused_recurrent_gated_delta_rule
            self.norm = FusedRMSNormGated(self.value_dim, eps=self.eps,
                                          activation=config.hidden_act)
        state_shape = (1, self.value_heads, self.key_dim, self.value_dim)
        conv_shape = (1, conv_width, self.conv_kernel)
        self.register_buffer("committed_recurrent", torch.zeros(state_shape, dtype=torch.float32), persistent=False)
        self.register_buffer("committed_conv", torch.zeros(conv_shape), persistent=False)
        self.register_buffer("scratch_recurrent", torch.zeros((max_verify_width + 1,) + state_shape,
                                                              dtype=torch.float32),
                             persistent=False)
        self.register_buffer("scratch_conv", torch.zeros((max_verify_width + 1,) + conv_shape),
                             persistent=False)

    def _apply(self, fn):
        super()._apply(fn)
        self.committed_recurrent = self.committed_recurrent.float()
        self.scratch_recurrent = self.scratch_recurrent.float()
        return self

    def reset_state(self):
        self.committed_recurrent.zero_()
        self.committed_conv.zero_()

    def alloc(self, n):
        if not hasattr(self, '_slots'):
            from jetspec.engines.jetspec_plus_plus.models.state import SlotAllocator
            self._slots = SlotAllocator(self.committed_recurrent.shape[0],
                                         device=self.committed_recurrent.device)
        return self._slots.alloc(n)

    def reset(self, slot_ids):
        if self.committed_recurrent.shape[0] == 1:
            self.reset_state()
        else:
            self.committed_recurrent.index_fill_(0, slot_ids, 0)
            self.committed_conv.index_fill_(0, slot_ids, 0)

    def prepare_verify(self, slot_ids, width):
        if not 0 < width < self.scratch_recurrent.shape[0]:
            raise ValueError('verify width exceeds the GDN scratch capacity')
        # _forward snapshots these exact buffers after projection, as before.

    def free(self, slot_ids):
        self.reset(slot_ids)
        self._slots.free(slot_ids)

    def _delta(self, q, k, v, g, beta, initial, *, chunk):
        if q.is_cuda:
            if self._recurrent is None:
                raise RuntimeError("hybrid GDN on CUDA requires fla gated delta rule kernels")
            kernel = self._chunk if chunk else self._recurrent
            return kernel(q, k, v, g=g, beta=beta, initial_state=initial,
                          output_final_state=True, use_qk_l2norm_in_kernel=True)
        return _torch_delta_rule(q, k, v, g, beta, initial)

    def configure_slots(self, count):
        """Size private state at startup; F3 owns allocation/lifetime policy."""
        if type(count) is not int or count < 1:
            raise ValueError('state slot count must be positive')
        if count == self.committed_recurrent.shape[0]:
            return
        for name in ('committed_recurrent', 'committed_conv'):
            value = getattr(self, name)
            setattr(self, name, value.new_zeros((count,) + value.shape[1:]))
        for name in ('scratch_recurrent', 'scratch_conv'):
            value = getattr(self, name)
            setattr(self, name, value.new_zeros((value.shape[0], count) + value.shape[2:]))

    def forward(self, hidden: torch.Tensor, *, prefill: bool, verify: bool) -> torch.Tensor:
        context = get_forward_context()
        slots = context.slot_ids
        if slots is None or (self.committed_recurrent.shape[0] == 1 and slots.numel() == 1):
            return self._forward(hidden, self.committed_recurrent, self.committed_conv,
                                 self.scratch_recurrent, self.scratch_conv,
                                 prefill=prefill, verify=verify)
        if slots.numel() != 1:
            from jetspec.engines.jetspec_plus_plus.models.forward_context import forward_context
            lengths = context.prefill_lengths
            lengths = (lengths.tolist() if isinstance(lengths, torch.Tensor) else lengths)
            if lengths is None:
                lengths = [hidden.shape[0] // slots.numel()] * slots.numel()
            if len(lengths) != slots.numel() or sum(lengths) != hidden.shape[0]:
                raise ValueError('one query length is required per sequence slot')
            outputs, offset = [], 0
            for row, length in enumerate(lengths):
                with forward_context(verify=verify, slot_ids=slots[row:row + 1]):
                    outputs.append(self.forward(hidden[offset:offset + length],
                                                prefill=prefill, verify=verify))
                offset += length
            return torch.cat(outputs)
        slots = slots.to(device=self.committed_recurrent.device, dtype=torch.long)
        if prefill and context.prefill_lengths is not None:
            lengths = context.prefill_lengths
            if lengths.numel() != 1:
                raise ValueError('one prefill length is required per sequence slot')
            torch._assert_async((lengths[0] == hidden.shape[0]), 'prefill length differs from sequence width')
        recurrent = self.committed_recurrent.index_select(0, slots)
        conv = self.committed_conv.index_select(0, slots)
        scratch_recurrent = self.scratch_recurrent.index_select(1, slots)
        scratch_conv = self.scratch_conv.index_select(1, slots)
        output = self._forward(hidden, recurrent, conv, scratch_recurrent, scratch_conv,
                               prefill=prefill, verify=verify)
        if verify:
            self.scratch_recurrent.index_copy_(1, slots, scratch_recurrent)
            self.scratch_conv.index_copy_(1, slots, scratch_conv)
        else:
            self.committed_recurrent.index_copy_(0, slots, recurrent)
            self.committed_conv.index_copy_(0, slots, conv)
        return output

    def _forward(self, hidden: torch.Tensor, committed_recurrent, committed_conv,
                 scratch_recurrent, scratch_conv, *, prefill: bool, verify: bool) -> torch.Tensor:
        width = hidden.shape[0]
        x = hidden.unsqueeze(0)
        raw = self.in_proj_qkv(x).transpose(1, 2)
        z = self.in_proj_z(x).reshape(-1, self.value_dim)
        beta = self.in_proj_b(x).sigmoid()
        a = self.in_proj_a(x)
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)

        conv_input = raw if prefill else torch.cat((committed_conv, raw), dim=-1)
        convolved = F.silu(self.conv1d(conv_input)[:, :, :conv_input.shape[-1]])
        convolved = convolved[:, :, -width:].transpose(1, 2)
        q, k, v = convolved.split((self.key_width, self.key_width, self.value_width), dim=-1)
        q = q.reshape(1, width, self.key_heads, self.key_dim)
        k = k.reshape(1, width, self.key_heads, self.key_dim)
        v = v.reshape(1, width, self.value_heads, self.value_dim)
        repeat = self.value_heads // self.key_heads
        if repeat > 1:
            q = q.repeat_interleave(repeat, dim=2)
            k = k.repeat_interleave(repeat, dim=2)
        initial = None if prefill else committed_recurrent
        if initial is None:
            initial = torch.zeros_like(committed_recurrent)
        output, final = self._delta(q, k, v, g, beta, initial, chunk=prefill)

        if verify:
            if width >= scratch_recurrent.shape[0]:
                raise ValueError("verify width exceeds the GDN scratch capacity")
            scratch_recurrent[0].copy_(committed_recurrent)
            scratch_conv[0].copy_(committed_conv)
            if self.fast_snapshot and hidden.is_cuda:
                from jetspec.engines.jetspec_plus_plus.models.hybrid_snapshot import snapshot_states
                snapshot_states(k, v, g, beta, committed_recurrent,
                                scratch_recurrent)
            else:
                state = committed_recurrent
                for step in range(width):
                    _, state = self._delta(q[:, step:step + 1], k[:, step:step + 1],
                                           v[:, step:step + 1], g[:, step:step + 1],
                                           beta[:, step:step + 1], state, chunk=False)
                    scratch_recurrent[step + 1].copy_(state)
                    state = scratch_recurrent[step + 1]
            for step in range(width):
                scratch_conv[step + 1].copy_(
                    torch.cat((committed_conv, raw[:, :, :step + 1]), dim=-1)
                    [:, :, -self.conv_kernel:])
        else:
            committed_recurrent.copy_(final)
            committed_conv.copy_(
                F.pad(conv_input, (max(0, self.conv_kernel - conv_input.shape[-1]), 0))
                [:, :, -self.conv_kernel:])

        output = output.reshape(-1, self.value_dim)
        if self._recurrent is None or not output.is_cuda:
            x_float = output.float()
            output = (x_float * torch.rsqrt(x_float.square().mean(-1, keepdim=True) + self.eps)
                      * self.norm.weight.float() * F.silu(z.float())).to(output.dtype)
        else:
            output = self.norm(output, z)
        return self.out_proj(output.reshape(width, self.value_width))

    def commit(self, slot_ids: torch.Tensor, accepted_counts=None):
        # Keep the model shell's compatibility hook until its later identity sweep.
        accepted = slot_ids if accepted_counts is None else accepted_counts
        slots = get_forward_context().slot_ids if accepted_counts is None else slot_ids
        if self.committed_recurrent.shape[0] == 1:
            slots = None
        if slots is not None:
            slots = slots.to(device=self.committed_recurrent.device, dtype=torch.long)
            counts = accepted.to(device=slots.device, dtype=torch.long).reshape(-1)
            if counts.numel() != slots.numel():
                raise ValueError('one accepted count is required per sequence slot')
            self.committed_recurrent.index_copy_(0, slots, self.scratch_recurrent[counts, slots])
            self.committed_conv.index_copy_(0, slots, self.scratch_conv[counts, slots])
            return
        index = accepted.to(torch.long).reshape(1)
        self.committed_recurrent.copy_(self.scratch_recurrent.index_select(0, index)[0])
        self.committed_conv.copy_(self.scratch_conv.index_select(0, index)[0])
