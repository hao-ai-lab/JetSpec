"""RoPE selection preserving the full-rotary implementation and cast order."""
import torch
from torch import nn

from jetspec.engines.jetspec_plus_plus.models.rotary_embedding import get_rope as full_rope, apply_rotary_emb


class GatedRotaryEmbedding:
    """The hybrid rotary arithmetic; frequencies are owned by the model shell."""
    def __init__(self, head_size, rotary_dim):
        if not 0 < rotary_dim <= head_size or rotary_dim % 2:
            raise ValueError('rotary dimension must be positive, even and at most head size')
        self.head_size, self.rotary_dim = head_size, rotary_dim

    def __call__(self, query, key, cos, sin):
        def rotate(tensor):
            rot = tensor[..., :self.rotary_dim]
            half = self.rotary_dim // 2
            shifted = torch.cat((-rot[..., half:], rot[..., :half]), dim=-1)
            changed = rot * cos[:, None, :] + shifted * sin[:, None, :]
            return torch.cat((changed, tensor[..., self.rotary_dim:]), dim=-1)
        return rotate(query), rotate(key)


class PartialRotaryEmbedding(nn.Module):
    def __init__(self, head_size, rotary_dim, max_position, base):
        super().__init__()
        if not 0 < rotary_dim < head_size or rotary_dim % 2:
            raise ValueError('partial rotary dimension must be positive, even and below head size')
        self.head_size, self.rotary_dim = head_size, rotary_dim
        inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
        freqs = torch.einsum('i,j -> ij', torch.arange(max_position, dtype=torch.float), inv_freq)
        self.register_buffer('cos_sin_cache', torch.cat((freqs.cos(), freqs.sin()), dim=-1), persistent=False)

    @torch.compile
    def forward(self, positions, query, key):
        cos, sin = self.cos_sin_cache[positions].chunk(2, dim=-1)
        def rotate(tensor):
            shape = tensor.shape
            tensor = tensor.view(positions.size(0), -1, self.head_size)
            rotated = apply_rotary_emb(tensor[..., :self.rotary_dim], cos, sin)
            return torch.cat((rotated, tensor[..., self.rotary_dim:]), dim=-1).view(shape)
        return rotate(query), rotate(key)


def get_rope(head_size, rotary_dim, max_position, base, rope_scaling=None, *,
             norm_round='before_weight'):
    if norm_round == 'after_weight':
        return GatedRotaryEmbedding(head_size, rotary_dim)
    if rotary_dim == head_size:
        return full_rope(head_size, rotary_dim, max_position, base, rope_scaling)
    # Reuse the default-RoPE validation without altering the pinned full path.
    if rope_scaling is not None:
        validated = full_rope(rotary_dim, rotary_dim, max_position, base, rope_scaling)
        rope = PartialRotaryEmbedding(head_size, rotary_dim, max_position, base)
        rope.cos_sin_cache = validated.cos_sin_cache
        return rope
    return PartialRotaryEmbedding(head_size, rotary_dim, max_position, base)
