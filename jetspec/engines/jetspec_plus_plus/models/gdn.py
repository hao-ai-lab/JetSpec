"""Gated DeltaNet mixer, retaining the original recurrent arithmetic."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from jetspec.engines.jetspec_plus_plus.kernels.gdn_gate import gate_beta
from jetspec.engines.jetspec_plus_plus.models.config import setting
from jetspec.engines.jetspec_plus_plus.models.forward_context import get_forward_context
from jetspec.engines.jetspec_plus_plus.models.target_layers import HybridRMSNorm
from jetspec.engines.jetspec_plus_plus.models.target_linear import TargetLinear


def hybrid_linear(config, input_size, output_size, *, bias=False, run_config=None):
    storage = (setting(run_config, 'engine', 'engine_settings.hybrid_fp32_linear_bf16_storage', '0') == '1'
               and getattr(config, 'torch_dtype', torch.get_default_dtype()) == torch.float32)
    return TargetLinear(input_size, output_size, bias=bias, fp32_bf16_storage=storage)


def _versioned_loader(parameter, loaded):
    """Copy a checkpoint tensor in place so the parameter version marks the new value."""
    with torch.no_grad():
        parameter.copy_(loaded)


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


# verify_cut_gdnlazy: the fp32 verify inputs that the commit replays (keys per key head).
_LAZY_KEEP = ('lazy_key', 'lazy_value', 'lazy_gate', 'lazy_beta')


class HybridGDN(nn.Module):
    def __init__(self, config, max_verify_width: int, *, run_config=None):
        super().__init__()
        self.run_config = run_config
        # Row-indexed verify snapshots (C>1); a plain flag keeps C=1 off nn.Module attribute lookup.
        self._row_scratch = False
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
        self.verify_cut_gbeta = setting(run_config, 'engine', 'engine_settings.verify_cut_gbeta', '0') == '1'
        if self.verify_cut_gbeta:
            self.A_log.weight_loader = _versioned_loader
            self._decay = self._decay_key = None
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
        # verify_cut_gdncommit: start with no recurrent scratch slot; _size_recurrent_scratch sizes it.
        self.verify_cut_gdncommit = setting(self.run_config, 'engine', 'engine_settings.verify_cut_gdncommit',
                                            '0') == '1'
        scratch_state = (0,) + state_shape[1:] if self.verify_cut_gdncommit else state_shape
        self.register_buffer("scratch_recurrent", torch.zeros((max_verify_width + 1,) + scratch_state,
                                                              dtype=torch.float32),
                             persistent=False)
        self.register_buffer("scratch_conv", torch.zeros((max_verify_width + 1,) + conv_shape),
                             persistent=False)
        self.verify_cut_convsnap = setting(self.run_config, 'engine', 'engine_settings.verify_cut_convsnap',
                                           '0') == '1'
        # verify_cut_gdnconv: conv1d, SiLU, q/k/v split, lazy keep and conv snapshots in one launch.
        self.verify_cut_gdnconv = setting(self.run_config, 'engine', 'engine_settings.verify_cut_gdnconv',
                                          '0') == '1'
        self.verify_cut_conv27b = setting(run_config, 'engine', 'engine_settings.verify_cut_conv27b', '0') == '1'
        if self.verify_cut_conv27b and not self.verify_cut_gdnconv:
            raise ValueError('verify_cut_conv27b requires verify_cut_gdnconv')
        # verify_cut_gdnrec27b: value-tiled recurrence with warp-local key reductions.
        self.verify_cut_gdnrec27b = setting(self.run_config, 'engine', 'engine_settings.verify_cut_gdnrec27b',
                                            '0') == '1'
        self.verify_cut_gdnrec27b2 = setting(self.run_config, 'engine',
            'engine_settings.verify_cut_gdnrec27b2', '0') == '1'
        if self.verify_cut_gdnrec27b2 and not self.verify_cut_gdnrec27b:
            raise ValueError('verify_cut_gdnrec27b2 requires verify_cut_gdnrec27b')
        self.verify_cut_gdnrec27b3 = setting(run_config, 'engine',
            'engine_settings.verify_cut_gdnrec27b3', '0') == '1'
        if self.verify_cut_gdnrec27b3 and not self.verify_cut_gdnrec27b2:
            raise ValueError('verify_cut_gdnrec27b3 requires verify_cut_gdnrec27b2')
        # Bit mask: 1 keeps packed z strided, 2 folds gates into conv, 4 folds norm into recurrence.
        self.verify_cut_gdnfuse27b = int(setting(run_config, 'engine',
                                                'engine_settings.verify_cut_gdnfuse27b', '0'))
        if self.verify_cut_gdnfuse27b not in (0, 1):
            raise ValueError('verify_cut_gdnfuse27b supports only 0 and 1; conv gate bit 2 and recurrent norm bit 4 are retired')
        if self.verify_cut_gdnfuse27b and not (self.verify_cut_gdnconv and self.verify_cut_gdnrec27b):
            raise ValueError('verify_cut_gdnfuse27b requires verify_cut_gdnconv and verify_cut_gdnrec27b')
        self.verify_cut_gdnlazy = setting(self.run_config, 'engine', 'engine_settings.verify_cut_gdnlazy',
                                          '0') == '1'
        self.verify_cut_gdnrepeat = setting(self.run_config, 'engine', 'engine_settings.verify_cut_gdnrepeat',
                                            '0') == '1'
        # verify_cut_pack27b2: in_proj q|k|v, z, b and a as one GEMM (models/target_pack.py).
        self.verify_cut_pack27b2 = setting(self.run_config, 'engine', 'engine_settings.verify_cut_pack27b2',
                                           '0') == '1'
        self.verify_cut_bafold27b = setting(self.run_config, 'engine', 'engine_settings.verify_cut_bafold27b',
                                            '0') == '1'
        self.verify_cut_w1prod27b = setting(
            run_config, 'engine', 'engine_settings.verify_cut_w1prod27b', '0') == '1'
        from jetspec.engines.jetspec_plus_plus.kernels.verify_pdl27b import cut_mode
        self.verify_cut_pdl27b = cut_mode(run_config)
        from jetspec.engines.jetspec_plus_plus.kernels.verify_w1 import fp8store27b_enabled
        self.verify_cut_fp8store27b = fp8store27b_enabled(run_config)
        if self.verify_cut_bafold27b and not self.verify_cut_pack27b2:
            raise ValueError('verify_cut_bafold27b requires verify_cut_pack27b2')
        if self.verify_cut_gdnlazy:
            from jetspec.engines.jetspec_plus_plus.kernels.gdn_lazy import keep_buffers
            for name, value in zip(_LAZY_KEEP, keep_buffers(max_verify_width, self.key_heads, self.value_heads,
                                                            self.key_dim, self.value_dim)):
                self.register_buffer(name, value, persistent=False)
            self._lazy_warm = False
        if self.verify_cut_gdncommit:
            self._size_recurrent_scratch()
        # verify_cut_commit27b: models/state.py ModelState commits every lazy C=1 layer in one launch.
        self.verify_cut_commit27b = setting(self.run_config, 'engine', 'engine_settings.verify_cut_commit27b',
                                            '0') == '1'
        self.verify_cut_commitdelta27b = setting(run_config, 'engine',
            'engine_settings.verify_cut_commitdelta27b', '0') == '1'
        self.verify_cut_conv27b_rawkeep = setting(run_config, 'engine',
            'engine_settings.verify_cut_conv27b_rawkeep', '0') == '1'
        if self.verify_cut_conv27b_rawkeep and not (self.verify_cut_conv27b and self.verify_cut_commitdelta27b
                                                   and self.conv_kernel == 4):
            raise ValueError('conv27b_rawkeep requires conv27b, commitdelta27b and kernel size 4')
        if self.verify_cut_commitdelta27b:
            if not (self.verify_cut_gdnrec27b2 and self.verify_cut_commit27b and self.verify_cut_gdnconv
                    and self.verify_cut_gdnlazy and self.verify_cut_gdnrepeat and self.verify_cut_convsnap
                    and self.fast_snapshot):
                raise ValueError('commitdelta27b requires gdnrec27b2, commit27b and the C=1 lazy fused conv path')
            from jetspec.engines.jetspec_plus_plus.kernels.verify_commitdelta27b import keep_buffers
            for name, value in zip(('lazy_normal_key', 'lazy_delta', 'lazy_exp_gate'),
                    keep_buffers(max_verify_width, self.key_heads, self.value_heads, self.key_dim, self.value_dim)):
                self.register_buffer(name, value, persistent=False)
        self.verify_cut_pdlgdn27b = int(setting(run_config, 'engine',
            'engine_settings.verify_cut_pdlgdn27b', '0'))
        if self.verify_cut_pdlgdn27b not in (0, 1, 2, 3):
            raise ValueError('pdlgdn27b must be 0, 1, 2 or 3')
        if self.verify_cut_pdlgdn27b and not (self.verify_cut_commitdelta27b
                and self.verify_cut_gdnfuse27b == 1 and self.verify_cut_w1prod27b):
            raise ValueError('pdlgdn27b requires commitdelta27b, gdnfuse27b=1 and w1prod27b')

    @property
    def pack27b2_members(self):
        """The verify_cut_pack27b2 group: all four in_proj, or q|k|v and z under mxfp4_w4a8.

        serving serves N=48 b/a with padded W4A8 packs in its faster buckets. bafold27b folds them
        into the input pack, removing their separate launches; with it off their routing is unchanged.
        """
        if self.in_proj_b.precision.weight_format.mode == 'mxfp4_w4a8' and not self.verify_cut_bafold27b:
            return ('in_proj_qkv', 'in_proj_z')
        return ('in_proj_qkv', 'in_proj_z', 'in_proj_b', 'in_proj_a')

    def _apply(self, fn):
        super()._apply(fn)
        self.committed_recurrent = self.committed_recurrent.float()
        self.scratch_recurrent = self.scratch_recurrent.float()
        if getattr(self, 'verify_cut_gdnlazy', False):
            for name in _LAZY_KEEP:
                setattr(self, name, getattr(self, name).float())
        if getattr(self, 'verify_cut_commitdelta27b', False):
            for name in ('lazy_normal_key', 'lazy_delta', 'lazy_exp_gate'):
                setattr(self, name, getattr(self, name).float())
        if getattr(self, 'verify_cut_gdncommit', False):
            self._size_recurrent_scratch()
        return self

    def _size_recurrent_scratch(self):
        """verify_cut_gdncommit: no recurrent scratch where the lazy commit replaces it (C=1 CUDA).

        ``_lazy_state`` verify and commit never read ``scratch_recurrent``, so it keeps 0 slot rows
        there. Every other path gets the full scratch back, with the slot rows of the committed state.
        """
        value = self.scratch_recurrent
        rows = 0 if self._lazy_state() else self.committed_recurrent.shape[0]
        if value.shape[1] != rows and (rows == 0 or value.shape[1] == 0):
            self.scratch_recurrent = value.new_zeros((value.shape[0], rows) + value.shape[2:])

    def gate_decay(self):
        """Return ``-exp(A_log.float())``, computed once per loaded ``A_log`` value.

        The key is the storage, version, dtype and device of ``A_log``. A new value is computed
        eagerly and written into the same tensor, so a captured graph reads the current value.
        """
        a_log = self.A_log
        version = None if a_log.is_inference() else a_log._version
        key = (a_log.data_ptr(), version, a_log.dtype, a_log.device)
        if self._decay_key != key:
            if a_log.is_cuda and torch.cuda.is_current_stream_capturing():
                raise RuntimeError('GDN gate decay must be computed before CUDA graph capture')
            with torch.inference_mode(False), torch.no_grad():
                decay = -a_log.float().exp()
                if (self._decay is not None and self._decay.shape == decay.shape
                        and self._decay.device == decay.device):
                    self._decay.copy_(decay)
                else:
                    self._decay = decay
            self._decay_key = key
        return self._decay

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
        if self._row_scratch:
            if slot_ids.numel() > self._verify_slots.numel():
                raise ValueError(f'verify rows exceed the GDN scratch row capacity: {slot_ids.numel()} > {self._verify_slots.numel()}')
            self._verify_slots.fill_(-1)
            self._verify_slots[:slot_ids.numel()].copy_(slot_ids)
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

    def configure_slots(self, count, *, verify_rows=None):
        """Size private state; F3 supplies bucket rows for row-indexed snapshots.

        The low-level slot-only API retains its original snapshot indexing.
        """
        if type(count) is not int or count < 1:
            raise ValueError('state slot count must be positive')
        if count == self.committed_recurrent.shape[0] and verify_rows is None:
            return
        for name in ('committed_recurrent', 'committed_conv'):
            value = getattr(self, name)
            setattr(self, name, value.new_zeros((count,) + value.shape[1:]))
        for name in ('scratch_recurrent', 'scratch_conv'):
            value = getattr(self, name)
            rows = count if verify_rows is None else verify_rows
            setattr(self, name, value.new_zeros((value.shape[0], rows) + value.shape[2:]))
        if verify_rows is not None:
            self.register_buffer('_verify_slots', torch.arange(verify_rows,
                                 device=self.committed_recurrent.device), persistent=False)
            self._row_scratch = True
        if self.verify_cut_gdncommit:
            self._size_recurrent_scratch()

    def forward(self, hidden: torch.Tensor, *, prefill: bool, verify: bool) -> torch.Tensor:
        context = get_forward_context()
        slots = context.slot_ids
        if slots is None or (self.committed_recurrent.shape[0] == 1 and slots.numel() == 1):
            return self._forward(hidden, self.committed_recurrent, self.committed_conv,
                                 self.scratch_recurrent, self.scratch_conv,
                                 prefill=prefill, verify=verify)
        if self._row_scratch:
            lengths = context.prefill_lengths
            lengths = lengths.tolist() if isinstance(lengths, torch.Tensor) else lengths
            if lengths is None:
                lengths = [hidden.shape[0] // slots.numel()] * slots.numel()
            if len(lengths) != slots.numel() or sum(lengths) != hidden.shape[0]:
                raise ValueError('one query length is required per sequence slot')
            if verify:
                self.prepare_verify(slots, max(lengths))
            slots = slots.to(device=self.committed_recurrent.device, dtype=torch.long)
            outputs, offset = [], 0
            for row, length in enumerate(lengths):
                slot = slots[row:row + 1]
                recurrent = self.committed_recurrent.index_select(0, slot)
                conv = self.committed_conv.index_select(0, slot)
                # Non-verify forwards never access snapshots, including cold probes.
                scratch_row = row if verify else 0
                outputs.append(self._forward(hidden[offset:offset + length], recurrent, conv,
                    self.scratch_recurrent[:, scratch_row:scratch_row + 1],
                    self.scratch_conv[:, scratch_row:scratch_row + 1], prefill=prefill, verify=verify))
                if not verify:
                    self.committed_recurrent.index_copy_(0, slot, recurrent)
                    self.committed_conv.index_copy_(0, slot, conv)
                offset += length
            return torch.cat(outputs)
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
        pack = None
        if self.verify_cut_pack27b2:
            from jetspec.engines.jetspec_plus_plus.models.target_pack import module_pack
            pack = module_pack(self, self.pack27b2_members)
        if pack is not None:
            if getattr(hidden, '_verify_w1', None) is not None:
                from jetspec.engines.jetspec_plus_plus.kernels.verify_w1 import w1_packed
                outputs = w1_packed(pack, x, activation=hidden._verify_w1)
            else:
                outputs = pack(x)
            raw = outputs[0].transpose(1, 2)
            if (self.verify_cut_gdnfuse27b and verify
                    and self.key_dim in (64, 128) and self.value_dim == 128 and width <= 16
                    and self._gdnconv_ready(raw, committed_recurrent, scratch_conv)):
                from jetspec.engines.jetspec_plus_plus.kernels.verify_gdnfuse27b import verify as fused_verify
                b, a = outputs[2:] if len(outputs) == 4 else (self.in_proj_b(x), self.in_proj_a(x))
                return fused_verify(self, raw, outputs[1], a, b, committed_recurrent, committed_conv, scratch_conv)
            z = outputs[1].reshape(-1, self.value_dim)
            b, a = outputs[2:] if len(outputs) == 4 else (self.in_proj_b(x), self.in_proj_a(x))
            if self.verify_cut_gbeta:
                from jetspec.engines.jetspec_plus_plus.kernels.verify_pack_glue import gate_beta_rows
                g, beta = gate_beta_rows(a, b, self.dt_bias, self.gate_decay())
            else:
                beta = b.sigmoid()
                g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
        else:
            raw = self.in_proj_qkv(x).transpose(1, 2)
            if (self.verify_cut_gdnfuse27b and verify
                    and self.key_dim in (64, 128) and self.value_dim == 128 and width <= 16
                    and self._gdnconv_ready(raw, committed_recurrent, scratch_conv)):
                from jetspec.engines.jetspec_plus_plus.kernels.verify_gdnfuse27b import verify as fused_verify
                return fused_verify(self, raw, self.in_proj_z(x), self.in_proj_a(x), self.in_proj_b(x),
                                    committed_recurrent, committed_conv, scratch_conv)
            z = self.in_proj_z(x).reshape(-1, self.value_dim)
            if self.verify_cut_gbeta:
                b = self.in_proj_b(x)
                g, beta = gate_beta(self.in_proj_a(x), b, self.dt_bias, self.gate_decay())
            else:
                beta = self.in_proj_b(x).sigmoid()
                a = self.in_proj_a(x)
                g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)

        if self.verify_cut_gdnconv and verify and self._gdnconv_ready(raw, committed_recurrent, scratch_conv):
            return self._verify_gdnconv(raw, z, g, beta, committed_recurrent, committed_conv, scratch_conv)
        if self.verify_cut_commitdelta27b and verify and raw.is_cuda:
            raise ValueError('commitdelta27b requires the supported C=1 fused conv verify geometry')
        conv_input = raw if prefill else torch.cat((committed_conv, raw), dim=-1)
        convolved = F.silu(self.conv1d(conv_input)[:, :, :conv_input.shape[-1]])
        convolved = convolved[:, :, -width:].transpose(1, 2)
        q, k, v = convolved.split((self.key_width, self.key_width, self.value_width), dim=-1)
        q = q.reshape(1, width, self.key_heads, self.key_dim)
        k = k.reshape(1, width, self.key_heads, self.key_dim)
        v = v.reshape(1, width, self.value_heads, self.value_dim)
        repeat = self.value_heads // self.key_heads
        lazy = grouped = False
        if self.verify_cut_gdnlazy or self.verify_cut_gdnrepeat:
            q, k, v, lazy, grouped = self._cut_inputs(q, k, v, g, beta, committed_recurrent,
                                                      prefill=prefill, verify=verify)
        if repeat > 1 and not grouped:
            q = q.repeat_interleave(repeat, dim=2)
            k = k.repeat_interleave(repeat, dim=2)
        initial = None if prefill else committed_recurrent
        if initial is None:
            initial = torch.zeros_like(committed_recurrent)
        output, final = self._delta(q, k, v, g, beta, initial, chunk=prefill)

        if verify:
            if width >= scratch_recurrent.shape[0]:
                raise ValueError("verify width exceeds the GDN scratch capacity")
            if not lazy:
                scratch_recurrent[0].copy_(committed_recurrent)
            if not self.verify_cut_convsnap:
                scratch_conv[0].copy_(committed_conv)
            if lazy:
                pass  # commit replays the kept inputs for the accepted count
            elif self.fast_snapshot and hidden.is_cuda:
                from jetspec.engines.jetspec_plus_plus.models.hybrid_snapshot import snapshot_states
                snapshot_states(k.repeat_interleave(repeat, dim=2) if grouped else k, v, g, beta,
                                committed_recurrent, scratch_recurrent)
            else:
                state = committed_recurrent
                for step in range(width):
                    _, state = self._delta(q[:, step:step + 1], k[:, step:step + 1],
                                           v[:, step:step + 1], g[:, step:step + 1],
                                           beta[:, step:step + 1], state, chunk=False)
                    scratch_recurrent[step + 1].copy_(state)
                    state = scratch_recurrent[step + 1]
            if self.verify_cut_convsnap:
                self._conv_snapshots(conv_input if not prefill else torch.cat((committed_conv, raw), dim=-1),
                                     committed_conv, scratch_conv, width)
            else:
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
        if (self.verify_cut_w1prod27b and verify and output.is_cuda
                and output.dtype == torch.bfloat16 and self._recurrent is not None):
            from jetspec.engines.jetspec_plus_plus.kernels.verify_gdn_w1 import gated_norm
            from jetspec.engines.jetspec_plus_plus.kernels.verify_w1 import activation_buffers, w1_linear
            output = gated_norm(output, z, self.norm, self.value_heads,
                                activation=activation_buffers(self.out_proj, width),
                                pdl=bool(self.verify_cut_pdl27b & 1),
                                **({'fp8store': True} if self.verify_cut_fp8store27b else {}))
            return w1_linear(self.out_proj, output)
        if self._recurrent is None or not output.is_cuda:
            x_float = output.float()
            output = (x_float * torch.rsqrt(x_float.square().mean(-1, keepdim=True) + self.eps)
                      * self.norm.weight.float() * F.silu(z.float())).to(output.dtype)
        else:
            output = self.norm(output, z)
        return self.out_proj(output.reshape(width, self.value_width))

    def _conv_snapshots(self, source, committed_conv, scratch_conv, width):
        # Window j of cat(committed, raw) is the conv state after j tokens: the loop's slices, in one copy.
        # Window 0 rides along only when the cat kept committed_conv's dtype, so its bits are unchanged.
        first = 0 if source.dtype == committed_conv.dtype else 1
        if first:
            scratch_conv[0].copy_(committed_conv)
        windows = source.unfold(-1, self.conv_kernel, 1)[:, :, first:width + 1]
        scratch_conv[first:width + 1].copy_(windows.permute(2, 0, 1, 3))

    def _lazy_state(self):
        """verify_cut_gdnlazy serves the C=1 buffers on CUDA with the fast snapshot; all else keeps scratch."""
        return (self.verify_cut_gdnlazy and self.fast_snapshot and not self._row_scratch
                and self.committed_recurrent.shape[0] == 1 and self.committed_recurrent.is_cuda)

    def _lazy_keep(self):
        return tuple(getattr(self, name) for name in _LAZY_KEEP)

    def _delta_keep(self):
        return self.lazy_normal_key, self.lazy_delta, self.lazy_exp_gate

    def _cut_inputs(self, q, k, v, g, beta, committed_recurrent, *, prefill, verify):
        """Apply verify_cut_gdnlazy (keep the verify inputs) and verify_cut_gdnrepeat (fla GVA heads)."""
        lazy = verify and self.verify_cut_gdnlazy and self._lazy_state()
        if lazy and committed_recurrent is not self.committed_recurrent:
            raise RuntimeError('verify_cut_gdnlazy needs the C=1 committed state buffers')
        grouped = (self.verify_cut_gdnrepeat and self.value_heads > self.key_heads and not prefill
                   and q.is_cuda and self._recurrent is not None)
        if not (lazy or grouped):
            return q, k, v, False, False
        from jetspec.engines.jetspec_plus_plus.kernels.gdn_lazy import commit_state, verify_inputs
        keep = self._lazy_keep() if lazy else None
        split = verify_inputs(q, k, v, g, beta, keep=keep, split=grouped)
        if lazy and not self._lazy_warm and not torch.cuda.is_current_stream_capturing():
            # Compile the commit kernel before a commit graph captures it; it writes a scratch tensor.
            commit_state(keep, torch.zeros(1, dtype=torch.int64, device=committed_recurrent.device),
                         committed_recurrent, out=torch.empty_like(committed_recurrent))
            self._lazy_warm = True
        if grouped:
            q, k, v = split
        return q, k, v, lazy, grouped

    def _gdnconv_ready(self, raw, committed_recurrent, scratch_conv):
        """verify_cut_gdnconv serves the C=1 lazy GVA verify on CUDA (gdnlazy, gdnrepeat, convsnap on)."""
        if not (self.verify_cut_gdnlazy and self.verify_cut_gdnrepeat and self.verify_cut_convsnap
                and self.value_heads > self.key_heads and self._recurrent is not None and raw.is_cuda
                and self._lazy_state() and committed_recurrent is self.committed_recurrent):
            return False
        from jetspec.engines.jetspec_plus_plus.kernels.verify_gdnconv import supported
        return supported(raw, self.committed_conv, self.conv1d.weight, scratch_conv)

    def _verify_gdnconv(self, raw, z, g, beta, committed_recurrent, committed_conv, scratch_conv):
        """The lazy GVA verify mixer with the fused conv launch (kernels/verify_gdnconv.py)."""
        from jetspec.engines.jetspec_plus_plus.kernels.gdn_lazy import commit_state
        from jetspec.engines.jetspec_plus_plus.kernels.verify_gdnconv import conv_verify
        width = raw.shape[-1]
        if width >= scratch_conv.shape[0]:
            raise ValueError("verify width exceeds the GDN scratch capacity")
        keep = self._lazy_keep()
        q, k, v = conv_verify(raw, committed_conv, self.conv1d.weight, g, beta, keep, scratch_conv,
                              self.key_heads, self.value_heads, self.key_dim, self.value_dim,
                              **({'conv27b': True} if self.verify_cut_conv27b else {}),
                              **({'rawkeep': True} if self.verify_cut_conv27b_rawkeep else {}))
        if not self._lazy_warm and not torch.cuda.is_current_stream_capturing():
            # Compile the commit kernel before a commit graph captures it; it writes a scratch tensor.
            commit_state(keep, torch.zeros(1, dtype=torch.int64, device=committed_recurrent.device),
                         committed_recurrent, out=torch.empty_like(committed_recurrent))
            self._lazy_warm = True
        output = None
        if self.verify_cut_gdnrec27b:
            from jetspec.engines.jetspec_plus_plus.kernels import verify_gdnrec27b
            if verify_gdnrec27b.supported(q, k, v, g, beta, committed_recurrent):
                # The final state is not read: the lazy commit replays the kept inputs.
                if self.verify_cut_gdnrec27b2:
                    from jetspec.engines.jetspec_plus_plus.kernels import verify_gdnrec27b2
                    output = verify_gdnrec27b2.recurrent_verify(q, k, v, g, beta, committed_recurrent,
                        **({'staged': True} if self.verify_cut_gdnrec27b3 else {}),
                        **({'delta_keep': self._delta_keep()} if self.verify_cut_commitdelta27b else {}))[0]
                else:
                    output = verify_gdnrec27b.recurrent_verify(q, k, v, g, beta, committed_recurrent)[0]
        if output is None:
            output, _ = self._delta(q, k, v, g, beta, committed_recurrent, chunk=False)
        if self.verify_cut_w1prod27b and output.dtype == torch.bfloat16:
            from jetspec.engines.jetspec_plus_plus.kernels.verify_gdn_w1 import gated_norm
            from jetspec.engines.jetspec_plus_plus.kernels.verify_w1 import activation_buffers, w1_linear
            output = gated_norm(output.reshape(-1, self.value_dim), z, self.norm, self.value_heads,
                                activation=activation_buffers(self.out_proj, width),
                                pdl=bool(self.verify_cut_pdl27b & 1),
                                **({'fp8store': True} if self.verify_cut_fp8store27b else {}))
            return w1_linear(self.out_proj, output)
        output = self.norm(output.reshape(-1, self.value_dim), z)
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
            if self._row_scratch:
                # Tensor lookup stays dynamic in commit graphs captured before verify.
                matches = slots[:, None] == self._verify_slots[None, :]
                rows = matches.to(torch.long).argmax(dim=1)
                present = matches.any(dim=1)
                torch._assert_async((present | (counts == 0)).all(),
                                    'commit slot was not present in the GDN verify rows')
                for name in ('recurrent', 'conv'):
                    committed = getattr(self, 'committed_' + name)
                    snapshots = getattr(self, 'scratch_' + name)[counts, rows]
                    mask = present.reshape((-1,) + (1,) * (committed.ndim - 1))
                    committed.index_copy_(0, slots, torch.where(mask, snapshots,
                                                               committed.index_select(0, slots)))
                return
            self.committed_recurrent.index_copy_(0, slots, self.scratch_recurrent[counts, slots])
            self.committed_conv.index_copy_(0, slots, self.scratch_conv[counts, slots])
            return
        index = accepted.to(torch.long).reshape(1)
        if self.verify_cut_gdnlazy and self._lazy_state():
            if self.verify_cut_commitdelta27b:
                from jetspec.engines.jetspec_plus_plus.kernels.verify_commitdelta27b import commit_state
                commit_state(self._delta_keep(), index, self.committed_recurrent)
            else:
                from jetspec.engines.jetspec_plus_plus.kernels.gdn_lazy import commit_state
                commit_state(self._lazy_keep(), index, self.committed_recurrent)
        else:
            self.committed_recurrent.copy_(self.scratch_recurrent.index_select(0, index)[0])
        if self.verify_cut_conv27b_rawkeep and self._lazy_state():
            from jetspec.engines.jetspec_plus_plus.kernels.verify_commitdelta27b import commit_conv_raw
            commit_conv_raw(self.scratch_conv, index, self.committed_conv)
        elif (self.verify_cut_gdncommit and self.scratch_conv.shape[1] == 1
                and self.scratch_conv.dtype == self.committed_conv.dtype):
            # One gather launch writes the accepted window straight into the committed state.
            torch.index_select(self.scratch_conv.select(1, 0), 0, index, out=self.committed_conv)
        else:
            self.committed_conv.copy_(self.scratch_conv.index_select(0, index)[0])
