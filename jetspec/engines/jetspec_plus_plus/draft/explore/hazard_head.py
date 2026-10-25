"""Head architectures. P0 = Linear(D,512) -> SiLU -> Linear(512,1), last layer zero-init, fp32.

Arm variants are switched by config keys (pred/arms.py), never by editing this file:
  width      hidden width (WIDTH arm)
  n_out      2 for AUX (unit 0 = hazard, unit 1 = aux BCE; both zero-init)
  residual   'floor_logit' for FR: l = logit(clip(p_top1)) + MLP(x)
  xattn      True for B1: u' = u + W_o MHA(LN(u) (+P_slot for q,k)) over slots, W_o zero-init
Append-type arms build layer 1 from P0's layer-1 weight at the same seed plus a zero block for the
appended groups (`append_init`), so step 0 equals P0 at that seed.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossSlot(nn.Module):
    """B1: one shared LayerNorm(512) on u; q,k read LN(u)+P_slot, v reads LN(u); 8x64 heads."""

    def __init__(self, width=512, heads=8, n_slots=15, generator=None):
        super().__init__()
        self.heads = heads
        self.ln = nn.LayerNorm(width, elementwise_affine=True)
        self.q = nn.Linear(width, width, bias=False)
        self.k = nn.Linear(width, width, bias=False)
        self.v = nn.Linear(width, width, bias=False)
        self.o = nn.Linear(width, width, bias=False)
        self.p_slot = nn.Parameter(torch.zeros(n_slots, width))
        with torch.no_grad():
            for lin in (self.q, self.k, self.v):
                bound = 1.0 / width ** 0.5
                lin.weight.uniform_(-bound, bound, generator=generator)
            self.p_slot.normal_(0.0, 0.02, generator=generator)
            self.o.weight.zero_()

    def forward(self, u):
        B, S, W = u.shape
        h = self.ln(u)
        qk_in = h + self.p_slot[None]
        q = self.q(qk_in).view(B, S, self.heads, -1).transpose(1, 2)
        k = self.k(qk_in).view(B, S, self.heads, -1).transpose(1, 2)
        v = self.v(h).view(B, S, self.heads, -1).transpose(1, 2)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        return u + self.o(a.transpose(1, 2).reshape(B, S, W))


class Head(nn.Module):
    def __init__(self, d_in, width=512, n_out=1, residual=None, xattn=False, xattn_seed=None):
        super().__init__()
        self.l1 = nn.Linear(d_in, width)
        self.l2 = nn.Linear(width, n_out)
        nn.init.zeros_(self.l2.weight)
        nn.init.zeros_(self.l2.bias)
        self.residual = residual
        self.xattn = None
        if xattn:
            g = torch.Generator().manual_seed(int(xattn_seed))
            self.xattn = CrossSlot(width, generator=g)

    def forward(self, x, floor_logit=None):
        u = self.l1(x)
        if self.xattn is not None:
            u = self.xattn(u)
        out = self.l2(F.silu(u))
        haz = out[..., 0]
        if self.residual == 'floor_logit':
            haz = haz + floor_logit
        aux = out[..., 1] if out.shape[-1] > 1 else None
        return haz, aux




