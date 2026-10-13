"""Torch anchor decoder over target-compatible layers, norms, and projection."""
from pathlib import Path

import torch
from torch import nn

from jetspec.engines.jetspec_plus_plus.draft.method.ops import reference_op


@reference_op("decoder_layer")
def decode_layer(layer, hidden, mask, positions, rope, prefix):
    return layer(hidden, attention_mask=mask, position_ids=positions,
                 position_embeddings=rope, past_key_values=prefix, use_cache=False)


def decode_hidden(layers, hidden, *, mask=None, positions=None, rope=None, prefix=None):
    for layer in layers:
        hidden = decode_layer(layer, hidden, mask, positions, rope, prefix)
    return hidden


class AnchorDecoder(nn.Module):
    """One decoder definition; engines supply optimized ops and model loading."""
    def __init__(self, *, layers, norm, lm_head, hidden_size, prefix_layers=None,
                 rotary=None, root=None):
        super().__init__()
        self.layers = layers
        self.prefix_layers = prefix_layers
        self.norm = norm
        self.lm_head = lm_head
        self.rotary = rotary
        self.root = Path(root) if root is not None else None
        self.hidden_size = int(hidden_size)

    def forward(self, hidden, *, mask=None, positions=None, rope=None, prefix=None):
        if positions is None:
            positions = torch.arange(hidden.shape[1], device=hidden.device)[None]
        if rope is None and self.rotary is not None:
            rope = self.rotary(hidden, positions)
        decoded = self.norm(decode_hidden(self.layers, hidden, mask=mask, positions=positions,
                                         rope=rope, prefix=prefix))
        return decoded, self.lm_head(decoded)
