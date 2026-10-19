"""Ported from the JetSpec++ reference implementation."""
import torch
from ..ops import reference_op

@reference_op('window_mask')
def window_mask(anchors, context_length, dtype, size=15):
    """Causal private blocks following every verified true prefix row."""
    anchors = torch.as_tensor(anchors).flatten()
    device = anchors.device
    query = torch.arange(len(anchors) * size, device=device)
    key = torch.arange(int(context_length) + len(query), device=device)
    block = query // size
    prefix = key[None, :] < anchors[block, None]
    local = key[None, :] - int(context_length)
    private = (local >= 0) & (local // size == block[:, None]) & (local % size <= query[:, None] % size)
    allowed = prefix | private
    return torch.zeros((1, 1, len(query), len(key)), device=device, dtype=dtype).masked_fill(~allowed[None, None], torch.finfo(dtype).min)

class IncrementalPrefix:
    """Append-only layer-34/35 K/V owned by one served sequence."""

    def __init__(self):
        self.length = 0
        self.kv = {}

    def update(self, key, value, layer_idx, cache_kwargs=None):
        if layer_idx in self.kv:
            key = torch.cat((self.kv[layer_idx][0], key), dim=-2)
            value = torch.cat((self.kv[layer_idx][1], value), dim=-2)
        self.kv[layer_idx] = (key.detach(), value.detach())
        return (key, value)
