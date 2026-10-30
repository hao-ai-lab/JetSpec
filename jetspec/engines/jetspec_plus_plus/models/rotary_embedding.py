from functools import lru_cache
import math
import torch
from torch import nn


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    cos = cos.unsqueeze(-2)
    sin = sin.unsqueeze(-2)
    x1, x2 = torch.chunk(x.float(), 2, dim=-1)
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    return torch.cat((y1, y2), dim=-1).to(x.dtype)


class RotaryEmbedding(nn.Module):

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        assert rotary_dim == head_size
        # print(f"head_size: {head_size}, rotary_dim: {rotary_dim}, max_position_embeddings: {max_position_embeddings}, base: {base}")
        inv_freq = 1.0 / (base**(torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos, sin), dim=-1)
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    @torch.compile
    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_tokens = positions.size(0)
        cos_sin = self.cos_sin_cache[positions]
        # cos_sin = torch.ones_like(cos_sin)
        cos, sin = cos_sin.chunk(2, dim=-1)
        query_shape = query.shape

        num_q_heads = query.shape[-1] // self.head_size
        query = query.view(num_tokens, num_q_heads, self.head_size)
        query = apply_rotary_emb(query, cos, sin).view(query_shape)
        key_shape = key.shape
        num_k_heads = key.shape[-1] // self.head_size
        key = key.view(num_tokens, num_k_heads, self.head_size)
        key = apply_rotary_emb(key, cos, sin).view(key_shape)
        return query, key


def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    rope_scaling: dict | None = None,
):
    if isinstance(rope_scaling, dict):
        if (rope_scaling.get("rope_type") != "default"
                or set(rope_scaling) - {"rope_type", "rope_theta"}):
            raise ValueError("SSD supports default RoPE only")
        base = rope_scaling.get("rope_theta", base)
        if (not isinstance(base, (int, float)) or isinstance(base, bool)
                or not math.isfinite(base) or base <= 0):
            raise ValueError("default RoPE requires a positive finite theta")
        rope_scaling = None
    assert rope_scaling is None
    return _get_rope(head_size, rotary_dim, max_position, base)


@lru_cache(1)
def _get_rope(head_size, rotary_dim, max_position, base):
    rotary_emb = RotaryEmbedding(head_size, rotary_dim, max_position, base)
    return rotary_emb
