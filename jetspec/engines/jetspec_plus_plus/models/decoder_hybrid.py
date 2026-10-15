"""Qwen3.5 anchor tail over the target's committed GDN state."""

from jetspec.engines.jetspec_plus_plus.profiles import kernel_enabled, kernel_value

import json
import os
import time
from pathlib import Path
from types import SimpleNamespace
from functools import partial

import torch
from jetspec.engines.jetspec_plus_plus.models.config import setting
from torch import nn
from torch.nn import functional as F
from safetensors import safe_open
from jetspec.engines.jetspec_plus_plus.draft.method.decoder.anchor import AnchorDecoder as MethodAnchorDecoder, decode_hidden


def scan_gdn_window(gdn, hidden, recurrent, conv):
    """Run one decoder window without changing the target or copied state."""
    if recurrent.dtype != torch.float32:
        raise ValueError("decoder GDN recurrent state must stay fp32")
    if hidden.ndim != 2 or recurrent.shape != gdn.committed_recurrent.shape \
            or conv.shape != gdn.committed_conv.shape:
        raise ValueError("decoder GDN state or window shape changed")
    width = hidden.shape[0]
    x = hidden.unsqueeze(0)
    raw = gdn.in_proj_qkv(x).transpose(1, 2)
    z = gdn.in_proj_z(x).reshape(-1, gdn.value_dim)
    beta = gdn.in_proj_b(x).sigmoid()
    a = gdn.in_proj_a(x)
    g = -gdn.A_log.float().exp() * F.softplus(a.float() + gdn.dt_bias)
    conv_input = torch.cat((conv.to(raw.dtype), raw), dim=-1)
    convolved = F.silu(gdn.conv1d(conv_input)[:, :, :conv_input.shape[-1]])
    convolved = convolved[:, :, -width:].transpose(1, 2)
    q, k, v = convolved.split((gdn.key_width, gdn.key_width, gdn.value_width), dim=-1)
    q = q.reshape(1, width, gdn.key_heads, gdn.key_dim)
    k = k.reshape(1, width, gdn.key_heads, gdn.key_dim)
    v = v.reshape(1, width, gdn.value_heads, gdn.value_dim)
    repeat = gdn.value_heads // gdn.key_heads
    if repeat > 1:
        q = q.repeat_interleave(repeat, dim=2)
        k = k.repeat_interleave(repeat, dim=2)
    output, _ = gdn._delta(q, k, v, g, beta, recurrent.clone(), chunk=False)
    output = output.reshape(-1, gdn.value_dim)
    if gdn._recurrent is None:
        x_float = output.float()
        output = (x_float * torch.rsqrt(x_float.square().mean(-1, keepdim=True) + gdn.eps)
                  * gdn.norm.weight.float() * F.silu(z.float())).to(output.dtype)
    else:
        output = gdn.norm(output, z)
    return gdn.out_proj(output.reshape(width, gdn.value_width))


def hybrid_gdn_tail(gdn_layer, hidden, *, recurrent, conv, **state):
    """The target GDN layer reads a private copy of committed recurrence."""
    first = gdn_layer.input_layernorm(hidden)
    x = hidden + scan_gdn_window(gdn_layer.linear_attn, first, recurrent, conv)
    mlp = gdn_layer.mlp
    first = gdn_layer.post_attention_layernorm(x)
    x = x + mlp.down_proj(F.silu(mlp.gate_proj(first)) * mlp.up_proj(first))

    return x


def hybrid_attention_tail(attention_layer, hidden, *, prefix_key, prefix_value,
                          start, rope_theta, window_kv=None, **state):
    """The target attention layer appends only private window K/V."""
    if prefix_key.shape != prefix_value.shape or prefix_key.ndim != 3 \
            or (isinstance(start, int) and prefix_key.shape[0] != start):
        raise ValueError("hybrid decoder needs target layer-63 KV through n-1")
    width = hidden.shape[0]
    x = hidden
    source = attention_layer.input_layernorm(x)
    attn = attention_layer.self_attn
    q, gate = attn.q_proj(source).view(width, attn.heads, 2 * attn.dim).chunk(2, dim=-1)
    q = attn.q_norm(q)
    key = attn.k_norm(attn.k_proj(source).view(width, attn.kv_heads, attn.dim))
    value = attn.v_proj(source).view(width, attn.kv_heads, attn.dim)
    if prefix_key.shape[1:] != (attn.kv_heads, attn.dim):
        raise ValueError("hybrid decoder prefix KV has the wrong head geometry")
    frequency = 1.0 / (rope_theta ** (torch.arange(
        0, attn.rotary_dim, 2, device=hidden.device, dtype=torch.float32)
        / attn.rotary_dim))
    angles = (torch.arange(width, device=hidden.device,
                           dtype=torch.float32) + start)[:, None] * frequency[None]
    angles = torch.cat((angles, angles), dim=-1)
    cos, sin = angles.cos().to(q.dtype), angles.sin().to(q.dtype)

    def rotate(tensor):
        rotary = tensor[..., :attn.rotary_dim]
        half = attn.rotary_dim // 2
        shifted = torch.cat((-rotary[..., half:], rotary[..., :half]), dim=-1)
        return torch.cat((rotary * cos[:, None] + shifted * sin[:, None],
                          tensor[..., attn.rotary_dim:]), dim=-1)

    q, key = rotate(q), rotate(key)
    all_key = torch.cat((prefix_key, key), dim=0)
    all_value = torch.cat((prefix_value, value), dim=0)
    groups = attn.heads // attn.kv_heads
    keys = all_key.repeat_interleave(groups, dim=1).transpose(0, 1)[None]
    values = all_value.repeat_interleave(groups, dim=1).transpose(0, 1)[None]
    queries = q.transpose(0, 1)[None]
    prefix_allowed = (torch.arange(prefix_key.shape[0], device=hidden.device)
                      < start).expand(width, -1)
    window_allowed = (torch.arange(width, device=hidden.device)[None, :]
                      <= torch.arange(width, device=hidden.device)[:, None])
    allowed = torch.cat((prefix_allowed, window_allowed), dim=-1)
    attended = F.scaled_dot_product_attention(
        queries, keys, values, attn_mask=allowed[None, None],
        dropout_p=0.0, scale=attn.dim ** -0.5)[0].transpose(0, 1)
    x = x + attn.o_proj((attended * torch.sigmoid(gate)).reshape(width, -1))
    mlp = attention_layer.mlp
    source = attention_layer.post_attention_layernorm(x)
    x = x + mlp.down_proj(F.silu(mlp.gate_proj(source)) * mlp.up_proj(source))
    if window_kv is not None:
        window_kv['kv'] = (key, value)
    return x


def hybrid_tail(layers):
    """Layer callables supplied to the common method AnchorDecoder."""
    return (partial(hybrid_gdn_tail, layers[0]),
            partial(hybrid_attention_tail, layers[1]))


def hybrid_anchor_window(gdn_layer, attention_layer, norm, lm_head, hidden,
                         recurrent, conv, prefix_key, prefix_value, *, start,
                         rope_theta):
    """Decode tap-61 rows using committed GDN and target layer-63 KV."""
    window_kv = {}
    decoded = decode_hidden((gdn_layer, attention_layer), hidden,
        tail=hybrid_tail((gdn_layer, attention_layer)), recurrent=recurrent, conv=conv,
        prefix_key=prefix_key, prefix_value=prefix_value, start=start,
        rope_theta=rope_theta, window_kv=window_kv)
    return lm_head(norm(decoded)), window_kv['kv']


class HybridAnchorDecoder(MethodAnchorDecoder):
    """The pinned target tail with a sequence-owned, device-resident prefix."""

    def __init__(self, layers, norm, lm_head, *, root, rope_theta, capacity, run_config=None):
        self.run_config = run_config
        super().__init__(layers=layers, norm=norm, lm_head=lm_head,
                         root=root, hidden_size=norm.weight.numel(), tail=hybrid_tail(layers))
        self.rope_theta = float(rope_theta)
        self.hidden_size = norm.weight.numel()
        self.capacity = capacity
        self._prefixes = {}
        self._state = None

    @classmethod
    def load(cls, root, drafter, target_model_path, manifest, *, run_config=None):
        from jetspec.engines.jetspec_plus_plus.jetspec import run_manifest
        from jetspec.engines.jetspec_plus_plus.models.decoder_head import _sha256
        from jetspec.engines.jetspec_plus_plus.models.qwen3_5 import HybridDecoderLayer, HybridRMSNorm

        if manifest is None or manifest.get("source") != "target_slice":
            raise ValueError("hybrid decoder requires a pinned target_slice manifest")
        run_manifest.record_file("heads/decoder_manifest", Path(root) / "decoder/manifest.json", run_config=run_config)
        target = Path(target_model_path)
        raw = json.loads((target / "config.json").read_text()).get("text_config")
        if raw is None or raw.get("model_type") != "qwen3_5_text" \
                or len(raw.get("layer_types", ())) != 64 \
                or raw["layer_types"][62:64] != ["linear_attention", "full_attention"]:
            raise ValueError("hybrid decoder needs the Qwen3.5 62/63 tail")
        taps = drafter.adapter.head.target_layer_ids
        if not taps or taps[-1] != 61:
            raise ValueError("hybrid decoder requires the last tap at layer 61")
        files = manifest.get("target_files", {})
        index_name = "model.safetensors.index.json"
        if not isinstance(files, dict) or index_name not in files or "config.json" not in files:
            raise ValueError("hybrid decoder needs config and index SHA256 pins")
        def verify(name):
            if name not in files or Path(name).is_absolute() or ".." in Path(name).parts:
                raise ValueError(f"hybrid decoder lacks a safe pin for {name}")
            path = target / name
            if _sha256(path) != files[name]:
                raise ValueError(f"hybrid decoder SHA256 mismatch for {name}")
            run_manifest.record_file(f"heads/decoder_target/{name}", path, run_config=run_config)
        verify("config.json")
        verify(index_name)
        config = SimpleNamespace(**raw)
        config.rope_theta = raw.get("rope_theta", raw["rope_parameters"]["rope_theta"])
        config.torch_dtype = drafter.dtype
        from jetspec.engines.jetspec_plus_plus.wire.protocol import draft_width
        lanes = draft_width(run_config=run_config) + 1
        with torch.device("meta"):
            layers = nn.ModuleList([
                HybridDecoderLayer(config, kind, speculate=False, spec_k=lanes - 1,
                                   max_verify_width=lanes, run_config=run_config)
                for kind in raw["layer_types"][62:64]])
            norm = HybridRMSNorm(config.hidden_size, config.rms_norm_eps)
            lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        wanted = {}
        for local, absolute in enumerate((62, 63)):
            for name in layers[local].state_dict():
                wanted[f"model.language_model.layers.{absolute}.{name}"] = f"layers.{local}.{name}"
        wanted["model.language_model.norm.weight"] = "norm.weight"
        wanted["lm_head.weight"] = "lm_head.weight"
        index = json.loads((target / index_name).read_text())["weight_map"]
        missing = set(wanted) - set(index)
        if missing:
            raise ValueError(f"hybrid decoder target lacks {sorted(missing)[:3]}")
        shards = sorted({index[name] for name in wanted})
        state = {}
        for shard in shards:
            verify(shard)
            with safe_open(str(target / shard), framework="pt", device="cpu") as handle:
                for name in wanted:
                    if index[name] == shard:
                        state[wanted[name]] = handle.get_tensor(name)
        module = cls(layers, norm, lm_head, root=root,
                     rope_theta=config.rope_theta,
                     capacity=int(setting(run_config, 'engine', 'engine_settings.decoder_cache_capacity', '800')) + lanes, run_config=run_config)
        from jetspec.engines.jetspec_plus_plus.wire.hybrid import HybridPairLayout
        module.layout = HybridPairLayout.from_config(config, lanes=lanes)
        module.to_empty(device="cpu")
        module.load_state_dict(state, strict=True, assign=True)
        if any(value.dtype != drafter.dtype for value in module.parameters()):
            raise ValueError("hybrid decoder target dtype differs from draft dtype")
        return module.to(device=drafter.device).eval().requires_grad_(False)

    def bind_front_state(self, *, device, dtype, capacity):
        if capacity > self.capacity:
            raise ValueError("hybrid decoder key exceeds its prefix capacity")
        if self._state is None:
            gdn = self.layers[0].linear_attn
            attn = self.layers[1].self_attn
            self._state = dict(
                recurrent=torch.empty_like(gdn.committed_recurrent, device=device),
                conv=torch.empty(gdn.committed_conv.shape, device=device, dtype=dtype),
                key=torch.zeros(self.capacity, attn.kv_heads, attn.dim,
                                dtype=dtype, device=device),
                value=torch.zeros(self.capacity, attn.kv_heads, attn.dim,
                                  dtype=dtype, device=device))
        return self._state

    def refresh_window_state(self, prefix_kv, bucket, *, rows, lanes, like, glue=None):
        """Bind the target's committed state, and allocate one fixed refresh tree.

        Unlike the dense decoder, the hybrid prefix is current through length-1.
        There are no true-tap lanes: the window is [tail | rows x slots]. All
        projections and MLPs read this single token matrix once per refresh.
        """
        from jetspec.engines.jetspec_plus_plus.profiles import kernel_enabled
        del prefix_kv
        if setting(self.run_config, 'engine', 'engine_settings.refresh_swap_glue', '0') == '1' if glue is None else glue:
            raise ValueError('JETSPEC_REFRESH_SWAP_GLUE refuses the hybrid decoder (GDN, D256)')
        prefix = self.bind_front_state(device=like.device, dtype=like.dtype,
                                       capacity=self.capacity)
        gdn = self.layers[0].linear_attn
        slots = lanes - 1
        tokens = lanes + rows * slots
        lane = torch.arange(tokens, device=like.device)
        is_block = lane >= lanes
        row = ((lane - lanes).clamp_min(0) // slots).clamp(max=rows - 1)
        index = torch.where(is_block, (lane - lanes).clamp_min(0) % slots, lane)
        state = dict(prefix=prefix, bucket=bucket, rows=rows, lanes=lanes,
                     refresh_staging=setting(self.run_config, 'engine', 'engine_settings.hybrid_decoder_refresh_staging', '0') == '1',
                     refresh_gates=setting(self.run_config, 'engine', 'engine_settings.hybrid_decoder_refresh_gates', '0') == '1',
                     refresh_state_lean=kernel_enabled('hybrid_decoder_refresh_state_lean', '1'),
                     refresh_selected_states=kernel_enabled('hybrid_decoder_refresh_state_lean', '1'),
                     slots=slots, tokens=tokens, lane=lane, is_block=is_block,
                     row=row, index=index,
                     conv_offset=torch.arange(1 - gdn.conv_kernel, 1, device=like.device),
                     prefix_column=torch.arange(self.capacity, device=like.device),
                     recurrent=torch.empty((lanes + 1,) + gdn.committed_recurrent.shape,
                                           device=like.device, dtype=torch.float32))
        if state['refresh_staging']:
            state['conv_input'] = like.new_empty(tokens, gdn.conv1d.weight.shape[0], gdn.conv_kernel)
            state['hidden_input'] = like.new_empty(tokens, self.hidden_size)
        if kernel_enabled('refresh_swap_lean', '1') and like.is_cuda and like.dtype == torch.bfloat16:
            from jetspec.engines.jetspec_plus_plus.jetspec.chain.refresh_swap import allocate_compact
            state['compact'] = allocate_compact(rows, slots, self.lm_head.weight.shape[0], like.device)
        return state

    def refresh_window_compact(self, state, *, ids, **kwargs):
        return self.refresh_window_logits(state, _compact_ids=ids, **kwargs)

    def refresh_window_logits(self, state, *, length, true_taps, tail, tail_counts,
                              block, slot_counts, _compact_ids=None):
        """Return [rows, lanes-1, vocab] in the dense refresh-swap layout.

        Row r starts from the committed recurrent/conv state plus its own
        tail[:tail_counts[r]], then reads its causal slots. Target K/V below
        length is shared; neither attention nor recurrence crosses rows.
        true_taps is already inside the committed target state and is unused.
        No data-dependent shapes or device-to-host reads occur during capture.
        """
        del true_taps
        rows, lanes, slots = state['rows'], state['lanes'], state['slots']
        tokens, prefix = state['tokens'], state['prefix']
        gdn_layer, attention_layer = self.layers
        gdn = gdn_layer.linear_attn
        gate_glue = None
        if state.get('refresh_gates', False):
            from jetspec.engines.jetspec_plus_plus.kernels import decoder_refresh as gate_glue
        state_glue = None
        if state.get('refresh_state_lean', False):
            from jetspec.engines.jetspec_plus_plus.kernels import decoder_refresh as state_glue
        if prefix['recurrent'].dtype != torch.float32:
            raise ValueError('decoder GDN recurrent state must stay fp32')
        row, index, is_block = state['row'], state['index'], state['is_block']
        counts, tails = slot_counts.long(), tail_counts.long()
        if state.get('refresh_staging', False):
            from jetspec.engines.jetspec_plus_plus.kernels.decoder_refresh import stage_hidden
            hidden = stage_hidden(tail, block, counts, lanes=lanes, slots=slots,
                                  out=state['hidden_input'])
        else:
            valid = state['index'][lanes:].view(rows, slots) < counts[:, None]
            blocks = torch.where(valid[..., None], block[:, :slots, -self.hidden_size:], 0)
            hidden = torch.cat((tail[:lanes], blocks.reshape(rows * slots, self.hidden_size)))
        source = gdn_layer.input_layernorm(hidden)
        raw = gdn.in_proj_qkv(source)
        z = gdn.in_proj_z(source).reshape(-1, gdn.value_dim)
        beta = gdn.in_proj_b(source).sigmoid()
        a = gdn.in_proj_a(source)
        g = -gdn.A_log.float().exp() * F.softplus(a.float() + gdn.dt_bias)

        # For every token gather its last kernel inputs. Negative logical tail
        # positions address committed conv; positive slot positions address only
        # that row. This includes the carry when tail_counts < kernel width.
        if state.get('refresh_staging', False):
            from jetspec.engines.jetspec_plus_plus.kernels.decoder_refresh import stage_conv
            conv_input = stage_conv(raw, prefix['conv'], tails, lanes=lanes,
                                    slots=slots, out=state['conv_input'])
        else:
            previous = index[:, None] + state['conv_offset'][None]
            own_slot = is_block[:, None] & (previous >= 0)
            tail_position = previous + torch.where(is_block, tails[row], 0)[:, None]
            raw_index = torch.where(own_slot, lanes + row[:, None] * slots + previous,
                                    tail_position)
            committed = prefix['conv'][0].transpose(0, 1)
            all_raw = torch.cat((committed.to(raw.dtype), raw))
            gather = torch.where(~own_slot & (tail_position < 0),
                                 gdn.conv_kernel + tail_position,
                                 gdn.conv_kernel + raw_index)
            conv_input = all_raw[gather].transpose(1, 2).contiguous()
        convolved = F.silu(F.conv1d(conv_input, gdn.conv1d.weight,
                                   groups=conv_input.shape[1]))[:, :, 0]
        q, k, v = convolved.split((gdn.key_width, gdn.key_width, gdn.value_width), -1)
        q = q.reshape(tokens, gdn.key_heads, gdn.key_dim)
        k = k.reshape(tokens, gdn.key_heads, gdn.key_dim)
        v = v.reshape(tokens, gdn.value_heads, gdn.value_dim)
        repeat = gdn.value_heads // gdn.key_heads
        if repeat > 1:
            q, k = q.repeat_interleave(repeat, 1), k.repeat_interleave(repeat, 1)
        scratch = state['recurrent']
        if state_glue is None or not hidden.is_cuda:
            scratch[0].copy_(prefix['recurrent'])
        if hidden.is_cuda:
            from jetspec.engines.jetspec_plus_plus.models.hybrid_snapshot import snapshot_states
            if state_glue is not None:
                tail_output = state_glue.recurrent_output(gdn, q[:lanes][None], k[:lanes][None],
                    v[:lanes][None], g[:lanes][None], beta[:lanes][None], prefix['recurrent'])
            else:
                tail_output, _ = gdn._delta(q[:lanes][None], k[:lanes][None], v[:lanes][None],
                                          g[:lanes][None], beta[:lanes][None],
                                          prefix['recurrent'], chunk=False)
            if not state.get('refresh_selected_states', False):
                snapshot_states(k[:lanes][None], v[:lanes][None], g[:lanes][None],
                                beta[:lanes][None], prefix['recurrent'], scratch)
        else:
            outputs = []
            for step in range(lanes):
                output, final = gdn._delta(q[step:step+1][None], k[step:step+1][None],
                    v[step:step+1][None], g[step:step+1][None], beta[step:step+1][None],
                    scratch[step], chunk=False)
                scratch[step+1].copy_(final)
                outputs.append(output)
            tail_output = torch.cat(outputs, 1)
        if state.get('refresh_selected_states', False):
            from jetspec.engines.jetspec_plus_plus.kernels.decoder_refresh import selected_states
            initial = selected_states(k[:lanes][None], v[:lanes][None], g[:lanes][None],
                                      beta[:lanes][None], prefix['recurrent'], tails)
        else:
            initial = (state_glue.initial_states(scratch, prefix['recurrent'], tails)
                   if state_glue is not None else scratch.index_select(0, tails)[:, 0].contiguous())
        delta_args = (q[lanes:].reshape(rows, slots, gdn.value_heads, gdn.key_dim),
            k[lanes:].reshape(rows, slots, gdn.value_heads, gdn.key_dim),
            v[lanes:].reshape(rows, slots, gdn.value_heads, gdn.value_dim),
            g[lanes:].reshape(rows, slots, gdn.value_heads),
            beta[lanes:].reshape(rows, slots, gdn.value_heads), initial)
        if state_glue is not None:
            block_output = state_glue.recurrent_output(gdn, *delta_args)
        else:
            block_output, _ = gdn._delta(*delta_args, chunk=False)
        output = torch.cat((tail_output.reshape(lanes, -1), block_output.reshape(rows * slots, -1)))
        output = output.reshape(-1, gdn.value_dim)
        if gdn._recurrent is None:
            output = (output.float() * torch.rsqrt(output.float().square().mean(-1, keepdim=True)
                      + gdn.eps) * gdn.norm.weight.float() * F.silu(z.float())).to(hidden.dtype)
        else:
            output = gdn.norm(output, z)
        projected = gdn.out_proj(output.reshape(tokens, gdn.value_width).to(hidden.dtype))
        x = hidden + projected
        source = gdn_layer.post_attention_layernorm(x)
        mlp = gdn_layer.mlp
        gate, up = mlp.gate_proj(source), mlp.up_proj(source)
        product = (gate_glue.gate_mul(up, gate, activation='silu') if gate_glue is not None
                   else F.silu(gate) * up)
        projected = mlp.down_proj(product)
        x = x + projected
        source = attention_layer.input_layernorm(x)
        attn = attention_layer.self_attn
        q, gate = attn.q_proj(source).view(tokens, attn.heads, 2 * attn.dim).chunk(2, -1)
        q = attn.q_norm(q)
        key = attn.k_proj(source).view(tokens, attn.kv_heads, attn.dim)
        key = attn.k_norm(key)
        value = attn.v_proj(source).view(tokens, attn.kv_heads, attn.dim)
        positions = length + index + torch.where(is_block, tails[row], 0)
        frequency = 1.0 / (self.rope_theta ** (torch.arange(
            0, attn.rotary_dim, 2, device=hidden.device, dtype=torch.float32) / attn.rotary_dim))
        angles = positions.float()[:, None] * frequency[None]
        angles = torch.cat((angles, angles), -1)
        cos, sin = angles.cos().to(q.dtype)[:, None], angles.sin().to(q.dtype)[:, None]
        def rotate(tensor):
            rotary = tensor[..., :attn.rotary_dim]
            half = attn.rotary_dim // 2
            shifted = torch.cat((-rotary[..., half:], rotary[..., :half]), -1)
            return torch.cat((rotary * cos + shifted * sin, tensor[..., attn.rotary_dim:]), -1)
        q, key = rotate(q), rotate(key)
        all_key, all_value = torch.cat((prefix['key'], key)), torch.cat((prefix['value'], value))
        allowed = (~is_block[None]) & (torch.where(is_block, tails[row], index + 1)[:, None]
                                       > index[None])
        allowed |= is_block[:, None] & is_block[None] & (row[:, None] == row[None]) \
                   & (index[:, None] >= index[None])
        allowed = torch.cat(((state['prefix_column'] < length).expand(tokens, -1), allowed), -1)
        groups = attn.heads // attn.kv_heads
        attended = F.scaled_dot_product_attention(q.transpose(0, 1)[None],
            all_key.repeat_interleave(groups, 1).transpose(0, 1)[None],
            all_value.repeat_interleave(groups, 1).transpose(0, 1)[None],
            attn_mask=allowed[None, None], dropout_p=0, scale=attn.dim ** -0.5)[0].transpose(0, 1)
        product = gate_glue.gate_mul(attended, gate) if gate_glue is not None else attended * gate.sigmoid()
        projected = attn.o_proj(product.reshape(tokens, -1))
        x = x + projected
        source = attention_layer.post_attention_layernorm(x)
        mlp = attention_layer.mlp
        gate, up = mlp.gate_proj(source), mlp.up_proj(source)
        product = (gate_glue.gate_mul(up, gate, activation='silu') if gate_glue is not None
                   else F.silu(gate) * up)
        projected = mlp.down_proj(product)
        x = x + projected
        decoded = self.norm(x[lanes:])
        if _compact_ids is not None:
            from jetspec.engines.jetspec_plus_plus.jetspec.chain.refresh_swap import compact_head
            from jetspec.engines.jetspec_plus_plus.jetspec.chain.refresh_swap import compact_logits
            if 'compact' in state:
                return compact_head(decoded.contiguous(), self.lm_head.weight, _compact_ids, state['compact'],
                                    persistent=state.get('refresh_staging', False))
            return compact_logits(self.lm_head(decoded).reshape(rows, slots, -1), _compact_ids)
        return self.lm_head(decoded).reshape(rows, slots, -1)

    def _window_buckets(self):
        return (self.capacity + 20,)

    def _ensure_warmup_state(self, sequence):
        if hasattr(sequence, "hybrid_recurrent"):
            return
        if getattr(sequence, "_jetspec_seq_id", None) != -1:
            raise ValueError("live hybrid admission lacks target committed state")
        parameter = next(self.layers.parameters())
        layout = self.layout
        sequence.hybrid_recurrent = torch.zeros(
            layout.recurrent_shape, device=parameter.device, dtype=torch.float32)
        sequence.hybrid_conv = torch.zeros(
            layout.conv_shape, device=parameter.device, dtype=parameter.dtype)
        shape = (sequence.taps.shape[0], layout.kv_heads, layout.attention_dim)
        sequence.hybrid_prefix_key = torch.zeros(shape, device=parameter.device,
                                                 dtype=parameter.dtype)
        sequence.hybrid_prefix_value = torch.zeros_like(sequence.hybrid_prefix_key)

    def _prefix(self, sequence, desired_length):
        self._ensure_warmup_state(sequence)
        prefix = SimpleNamespace(length=desired_length,
                                 kv={63: (sequence.hybrid_prefix_key,
                                          sequence.hybrid_prefix_value)})
        self._prefixes[id(sequence)] = prefix
        return prefix

    def prefetch_prefix(self, sequence, *, run_config=None):
        return None

    def prefetch_batch(self, sequences, *, run_config=None):
        return None

    def prewarm_prefix_appends(self, make_sequence, counts):
        return []

    @torch.inference_mode()
    def predict(self, sequence, pending_block, *, dataset, coverage, grouped, cap,
                after_miss=False, run_config=None):
        from jetspec.engines.jetspec_plus_plus.models.decoder_head import DecoderPrediction
        from jetspec.engines.jetspec_plus_plus.models.decoder_head import anchor_offers
        from jetspec.engines.jetspec_plus_plus.models.decoder_head import rejected_tokens
        staged = sequence.staged.predicted_hidden
        if staged is None:
            return None
        self._ensure_warmup_state(sequence)
        parameter = next(self.layers.parameters())
        from jetspec.engines.jetspec_plus_plus.wire.protocol import draft_width
        hidden = staged[0, :draft_width(run_config=run_config if run_config is not None else getattr(self, 'run_config', None)) + 1, -self.hidden_size:].to(parameter)
        start = sequence.taps.shape[0]
        began = time.perf_counter()
        _, logits = self(hidden, recurrent=sequence.hybrid_recurrent,
                         conv=sequence.hybrid_conv,
                         prefix_key=sequence.hybrid_prefix_key[:start],
                         prefix_value=sequence.hybrid_prefix_value[:start],
                         start=start, rope_theta=self.rope_theta)
        rejected = rejected_tokens(pending_block, after_miss=after_miss, run_config=self.run_config).to(logits.device)
        offers = anchor_offers(logits, rejected, root=self.root, dataset=dataset,
                               coverage=coverage, grouped=grouped, cap=cap, run_config=self.run_config)
        elapsed = (time.perf_counter() - began) * 1000
        return DecoderPrediction(hidden=self.norm(hidden), logits=logits,
                                 offers=offers, elapsed_ms=elapsed,
                                 total_ms=elapsed)

    def predict_batch(self, sequences, pending_blocks, *, dataset, coverage, grouped,
                      cap, after_misses, run_config=None):
        if len(sequences) != 1 or len(pending_blocks) != 1:
            raise ValueError("hybrid decoder requires N=1")
        return [self.predict(sequences[0], pending_blocks[0], dataset=dataset,
                             coverage=coverage, grouped=grouped, cap=cap,
                             after_miss=after_misses[0], run_config=run_config if run_config is not None else getattr(self, 'run_config', None))]

    def decoder_chain_body(self, rp, tokens, front, bucket):
        from jetspec.engines.jetspec_plus_plus.wire.contract import Offset
        from jetspec.engines.jetspec_plus_plus.models.decoder_head import chain_rejected_tokens
        from jetspec.engines.jetspec_plus_plus.jetspec.heads.topk_select import GraphedSelect
        del bucket
        hidden = front.staged_hidden[0, :front.lanes, -self.hidden_size:]
        logits, _ = hybrid_anchor_window(
            self.layers[0], self.layers[1], self.norm, self.lm_head,
            hidden, front.hybrid_recurrent, front.hybrid_conv,
            front.hybrid_prefix_key, front.hybrid_prefix_value,
            start=rp[Offset.L], rope_theta=self.rope_theta)
        front.rejected_ids.copy_(chain_rejected_tokens(rp, tokens, run_config=self.run_config))
        ids, p64, pack = GraphedSelect.chain_body(
            logits, front.rejected_ids, front.selection_thresholds, front.select_cap)
        return logits, ids, p64, pack
